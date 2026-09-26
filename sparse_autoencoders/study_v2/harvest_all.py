"""
Phase 1 — Harvest activations for the study_v2 factorial analysis.

For each dataset (new=constructor_stubs, old=contrastive_stubs):
  - One forward pass per prompt hooks all 8 layers x {mlp, resid}
  - Cache 16 tensors + aligned metadata
  - Final last-token activation only (same as existing pipeline)

Also writes splits.json with a stratified 50/50 discovery vs held-out
split on the new dataset, keyed by (ast_node, variant_type).

Outputs under data/study_v2/cache/:
  acts_new_L{0..7}_{mlp,resid}.pt   [14683, 2048]
  acts_old_L{0..7}_{mlp,resid}.pt   [ 8897, 2048]
  metadata_new.parquet              aligned to acts_new_*
  metadata_old.parquet              aligned to acts_old_*
  splits.json                       {"discovery": [ids...], "held_out": [ids...]}
"""
import os, sys, json, random
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import (
    load_config, resolve_path, data_path, get_device,
    patch_environment, get_layers,
)

N_LAYERS = 8
SITES = ["mlp", "resid"]
CACHE = data_path("study_v2/cache")
SEED = 20260413
DATASETS = {
    "new": data_path("constructor_stubs.parquet"),
    "old": data_path("contrastive_stubs.parquet"),
}
META_COLS = [
    "ast_node", "builtin_obj", "variant_type", "category",
    "note", "prompt_text", "ast_group", "builtin_group",
    "prompt_id", "variation_id",
]


def refuse_overwrite(path):
    if os.path.exists(path):
        raise SystemExit(f"Refusing to overwrite existing file: {path}")


def harvest(model, tokenizer, device, df, tag):
    """Harvest activations for all prompts, all layers, all sites."""
    layers_list = get_layers(model)
    caches = {(L, site): [] for L in range(N_LAYERS) for site in SITES}
    hooks = []

    def make_hook(L, site):
        def hk(m, i, o):
            x = o[0] if isinstance(o, tuple) else o
            caches[(L, site)].append(x[:, -1, :].detach().cpu())
        return hk

    for L in range(N_LAYERS):
        block = layers_list[L]
        hooks.append(block.mlp.register_forward_hook(make_hook(L, "mlp")))
        hooks.append(block.register_forward_hook(make_hook(L, "resid")))

    prompts = df["prompt_text"].tolist()
    with torch.no_grad():
        for p in tqdm(prompts, desc=f"Harvest [{tag}]"):
            model(input_ids=tokenizer(p, return_tensors="pt").input_ids.to(device))

    for h in hooks:
        h.remove()

    for (L, site), lst in caches.items():
        X = torch.cat(lst, dim=0)
        assert X.shape[0] == len(df), f"Row mismatch L{L}/{site}: {X.shape[0]} != {len(df)}"
        out = os.path.join(CACHE, f"acts_{tag}_L{L}_{site}.pt")
        torch.save(X, out)
        print(f"  saved {out}  {tuple(X.shape)}")


def stratified_split(meta, seed=SEED):
    """50/50 split stratified by (ast_node, variant_type). Returns two row-index lists."""
    random.seed(seed)
    disc, held = [], []
    key = list(zip(meta["ast_node"].values, meta["variant_type"].values))
    from collections import defaultdict
    by = defaultdict(list)
    for i, k in enumerate(key):
        by[k].append(i)
    for k, idxs in by.items():
        random.shuffle(idxs)
        half = len(idxs) // 2
        disc.extend(idxs[:half])
        held.extend(idxs[half:])
    return sorted(disc), sorted(held)


def main():
    cfg = load_config()
    device = get_device()
    os.makedirs(CACHE, exist_ok=True)

    # Refuse to overwrite
    for tag in DATASETS:
        for L in range(N_LAYERS):
            for site in SITES:
                refuse_overwrite(os.path.join(CACHE, f"acts_{tag}_L{L}_{site}.pt"))
        refuse_overwrite(os.path.join(CACHE, f"metadata_{tag}.parquet"))
    refuse_overwrite(os.path.join(CACHE, "splits.json"))

    # Load metadata first; record splits before model load
    meta_new = pd.read_parquet(resolve_path(DATASETS["new"]))[
        [c for c in META_COLS if c in pd.read_parquet(resolve_path(DATASETS["new"])).columns]
    ].reset_index(drop=True)
    meta_old = pd.read_parquet(resolve_path(DATASETS["old"]))[
        [c for c in META_COLS if c in pd.read_parquet(resolve_path(DATASETS["old"])).columns]
    ].reset_index(drop=True)

    meta_new.to_parquet(os.path.join(CACHE, "metadata_new.parquet"), index=False)
    meta_old.to_parquet(os.path.join(CACHE, "metadata_old.parquet"), index=False)
    disc, held = stratified_split(meta_new, seed=SEED)
    with open(os.path.join(CACHE, "splits.json"), "w") as f:
        json.dump({"seed": SEED, "discovery": disc, "held_out": held}, f)
    print(f"splits: discovery={len(disc)} held_out={len(held)}")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    patch_environment()
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(
        "openai/circuit-sparsity", trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        "openai/circuit-sparsity", trust_remote_code=True
    ).to(device).eval()

    # New first (primary); old second
    df_new = pd.read_parquet(resolve_path(DATASETS["new"]))
    harvest(model, tokenizer, device, df_new, "new")

    df_old = pd.read_parquet(resolve_path(DATASETS["old"]))
    harvest(model, tokenizer, device, df_old, "old")

    print("DONE.")


if __name__ == "__main__":
    main()
