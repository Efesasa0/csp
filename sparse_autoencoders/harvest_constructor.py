"""
Harvest L7 MLP + resid activations over the full 14,683-prompt
constructor_stubs dataset. Writes to a NEW cache dir so nothing existing
is overwritten.

Outputs (under data/cache_constructor/):
  acts_L7_mlp.pt     -- float tensor [N, d_model]
  acts_L7_resid.pt   -- float tensor [N, d_model]
  metadata.parquet   -- per-row labels aligned with tensor rows
                        (ast_node, builtin_obj, variant_type, category,
                         note, prompt_text)
"""
import os
import sys
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import (
    load_config, resolve_path, data_path, get_device,
    patch_environment, get_layers,
)

LAYER = 7
SITES = ["mlp", "resid"]
OUT_DIR = data_path("cache_constructor")
META_COLS = [
    "ast_node", "builtin_obj", "variant_type", "category",
    "note", "prompt_text", "ast_group", "builtin_group",
    "prompt_id", "variation_id",
]


def main():
    cfg = load_config()
    device = get_device()
    os.makedirs(OUT_DIR, exist_ok=True)

    mlp_path = os.path.join(OUT_DIR, f"acts_L{LAYER}_mlp.pt")
    resid_path = os.path.join(OUT_DIR, f"acts_L{LAYER}_resid.pt")
    meta_path = os.path.join(OUT_DIR, "metadata.parquet")
    for p in [mlp_path, resid_path, meta_path]:
        if os.path.exists(p):
            raise SystemExit(f"Refusing to overwrite existing file: {p}")

    df = pd.read_parquet(resolve_path(cfg["paths"]["dataset"]))
    meta = df[[c for c in META_COLS if c in df.columns]].reset_index(drop=True)
    print(f"Dataset: {len(df)} prompts  cols: {list(meta.columns)}")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    patch_environment()
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(
        "openai/circuit-sparsity", trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        "openai/circuit-sparsity", trust_remote_code=True
    ).to(device).eval()

    block = get_layers(model)[LAYER]
    mlp_acts, resid_acts = [], []

    mlp_hook = block.mlp.register_forward_hook(
        lambda m, i, o: mlp_acts.append(
            (o[0] if isinstance(o, tuple) else o)[:, -1, :].detach().cpu()
        )
    )
    resid_hook = block.register_forward_hook(
        lambda m, i, o: resid_acts.append(
            (o[0] if isinstance(o, tuple) else o)[:, -1, :].detach().cpu()
        )
    )

    prompts = df["prompt_text"].tolist()
    with torch.no_grad():
        for p in tqdm(prompts, desc=f"Harvest L{LAYER}"):
            model(
                input_ids=tokenizer(p, return_tensors="pt").input_ids.to(device)
            )

    mlp_hook.remove()
    resid_hook.remove()

    X_mlp = torch.cat(mlp_acts, dim=0)
    X_resid = torch.cat(resid_acts, dim=0)
    assert X_mlp.shape[0] == len(df) == X_resid.shape[0], (
        f"Row mismatch: mlp={X_mlp.shape[0]} resid={X_resid.shape[0]} df={len(df)}"
    )

    torch.save(X_mlp, mlp_path)
    torch.save(X_resid, resid_path)
    meta.to_parquet(meta_path, index=False)
    print(f"Saved:\n  {mlp_path}  {tuple(X_mlp.shape)}")
    print(f"  {resid_path}  {tuple(X_resid.shape)}")
    print(f"  {meta_path}  rows={len(meta)}")


if __name__ == "__main__":
    main()
