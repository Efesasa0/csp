#!/usr/bin/env python3
"""
X02 Phase A — Per-prompt activation extraction (online, GPU required)

Extracts per-prompt activations projected onto each pair's circuit mask,
saving to an intermediate HDF5 for Phase B analysis.

Usage:
    python experiments/scripts/x02_extract_prompt_level.py
    python experiments/scripts/x02_extract_prompt_level.py \
        --atlas data/small_dynamic_feature_atlas.h5 \
        --prompts data/small_40x50x50_validated_prompts.parquet \
        --out experiments/outputs/x02/prompt_level_activations.h5 \
        --pairs For__list If__int          # limit to specific pairs (testing)
        --batch-size 8

Requires: torch, transformers, huggingface_hub, pandas, h5py
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import inspect
import os
import sys
import types
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "src"))

from module2.extraction import ActivationExtractor
from module2.io_utils import load_atlas_hdf5

MODEL_ID = "openai/circuit-sparsity"
DOMAIN_NAMES = ["finance", "biology", "gaming", "physics", "ecommerce"]

# Highest signal count wins; ties broken by order (first wins)
_DOMAIN_SIGNALS: list[tuple[str, list[str]]] = [
    ("finance",   ["ledger_entries", "account_records", "audit_record", "portfolio", "ticker"]),
    ("biology",   ["dna_samples", "genome_annotations", "analyze_genome", "genome_sequence", "brca1"]),
    ("gaming",    ["player_scores", "character_stats", "update_leaderboard", "gameengine", "spawn_entity"]),
    ("physics",   ["particle_data", "compute_trajectory", "simulationrunner", "muon_decay", "measurements"]),
    ("ecommerce", ["shopping_cart", "product_catalog", "process_checkout", "orderprocessor", "apply_discount"]),
]

def infer_domain(prompt_text: str) -> str:
    text = prompt_text.lower()
    best, best_count = DOMAIN_NAMES[0], 0
    for domain, signals in _DOMAIN_SIGNALS:
        count = sum(s in text for s in signals)
        if count > best_count:
            best, best_count = domain, count
    return best


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--atlas", default="data/small_dynamic_feature_atlas.h5")
    p.add_argument("--prompts", default="data/small_40x50x50_validated_prompts.parquet")
    p.add_argument("--out", default="experiments/outputs/x02/prompt_level_activations.h5")
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--pairs", nargs="*", default=None,
                   help="Limit to specific pairs, e.g. For__list If__int")
    return p.parse_args()


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _patch_circuit_sparsity(model_id: str):
    """Load the custom CSP architecture from HuggingFace (mirrors chat_csp.py)."""
    from huggingface_hub import hf_hub_download

    tm_base = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules/openai")
    gpt_hits = glob.glob(os.path.join(tm_base, "*/gpt.py"))

    if gpt_hits:
        gpt_path = gpt_hits[0]
        hook_path = os.path.join(os.path.dirname(gpt_path), "hook_utils.py")
    else:
        gpt_path = hf_hub_download(model_id, "gpt.py")
        hook_path = hf_hub_download(model_id, "hook_utils.py")

    hf_dir = os.path.dirname(gpt_path)

    cs_pkg = types.ModuleType("circuit_sparsity")
    cs_pkg.__path__ = [hf_dir]
    cs_pkg.__package__ = "circuit_sparsity"
    sys.modules["circuit_sparsity"] = cs_pkg

    hook_spec = importlib.util.spec_from_file_location(
        "circuit_sparsity.hook_utils", hook_path)
    if hook_spec is None or hook_spec.loader is None:
        raise RuntimeError("Failed to load circuit_sparsity.hook_utils")
    hook_mod = importlib.util.module_from_spec(hook_spec)
    hook_mod.__package__ = "circuit_sparsity"
    sys.modules["circuit_sparsity.hook_utils"] = hook_mod
    hook_spec.loader.exec_module(hook_mod)
    setattr(cs_pkg, "hook_utils", hook_mod)

    gpt_spec = importlib.util.spec_from_file_location(
        "circuit_sparsity.gpt", gpt_path)
    if gpt_spec is None or gpt_spec.loader is None:
        raise RuntimeError("Failed to load circuit_sparsity.gpt")
    gpt_mod = importlib.util.module_from_spec(gpt_spec)
    gpt_mod.__package__ = "circuit_sparsity"
    sys.modules["circuit_sparsity.gpt"] = gpt_mod
    gpt_spec.loader.exec_module(gpt_mod)

    real_cfg = gpt_mod.GPTConfig
    valid_params = set(inspect.signature(real_cfg.__init__).parameters) - {"self"}

    class PatchedGPTConfig(real_cfg):
        def __init__(self, **kwargs):
            filtered = {k: v for k, v in kwargs.items() if k in valid_params}
            super().__init__(**filtered)

    gpt_mod.GPTConfig = PatchedGPTConfig
    cs_pkg.gpt = gpt_mod

    return hook_mod


def load_model(model_id: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    hook_mod = _patch_circuit_sparsity(model_id)
    # NOTE: hook_recorder doesn't work through AutoModelForCausalLM wrapper
    # (returns 0 keys), so we use manual register_forward_hook instead.
    _ = getattr(hook_mod, "hook_recorder", None)  # kept for reference

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    model.to(device).eval()

    # hook_recorder doesn't work through the AutoModelForCausalLM wrapper
    # (returns 0 keys), so always use manual hooks with the correct pattern.
    extractor = ActivationExtractor(
        model=model,
        tokenizer=tokenizer,
        device=device,
        n_layers=8,
        use_hook_recorder=False,
        hook_recorder_fn=None,
    )

    # Detect the MLP module pattern from the actual model architecture
    module_names = [name for name, _ in model.named_modules()]
    mlp_modules = [n for n in module_names if n.endswith(".mlp")]
    print(f"  MLP modules found: {mlp_modules[:4]}...")

    if mlp_modules:
        # Derive pattern from first MLP module, e.g.
        # "circuit_model.transformer.h.0.mlp" -> "circuit_model.transformer.h.{layer_id}.mlp"
        first = mlp_modules[0]
        # Replace the layer number with {layer_id}
        import re
        pattern = re.sub(r'\.\d+\.mlp$', '.{layer_id}.mlp', first)
        print(f"  Hook pattern: {pattern}")
    else:
        pattern = "circuit_model.transformer.h.{layer_id}.mlp"
        print(f"  Hook pattern (fallback): {pattern}")

    extractor.set_hook_pattern(pattern)
    extractor.register_hooks()
    print(f"  Registered {len(extractor.hooks)} MLP hooks")

    # Also register residual stream hooks on Block modules
    block_pattern = pattern.rsplit('.mlp', 1)[0]
    print(f"  Residual hook pattern: {block_pattern}")
    extractor.set_resid_hook_pattern(block_pattern)
    extractor.register_resid_hooks()
    print(f"  Registered {len(extractor.resid_hooks)} residual hooks")

    return extractor, device


def build_domain_vocab(prompts_df: pd.DataFrame) -> list[str]:
    if "domain" in prompts_df.columns:
        domains = sorted(prompts_df["domain"].dropna().unique().tolist())
    else:
        domains = DOMAIN_NAMES
    return domains


def main():
    args = parse_args()
    atlas_path = resolve(args.atlas)
    prompts_path = resolve(args.prompts)
    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading atlas: {atlas_path}")
    atlas = load_atlas_hdf5(str(atlas_path))
    try:
        pair_masks = atlas["pair_masks"]
        meta = atlas["metadata"]
        n_layers = int(meta.get("n_layers", 8))
        n_neurons = 2048

        print(f"Loading prompts: {prompts_path}")
        df = pd.read_parquet(prompts_path)
        print(f"  {len(df)} rows, columns: {list(df.columns)}")

        domain_vocab = build_domain_vocab(df)
        domain_to_id = {d: i for i, d in enumerate(domain_vocab)}
        print(f"  Domains: {domain_vocab}")

        # Determine which pairs to process
        if args.pairs:
            requested = set(args.pairs)
            pair_keys = [(a, b) for (a, b) in pair_masks if f"{a}__{b}" in requested]
        else:
            pair_keys = list(pair_masks.keys())
        print(f"  Processing {len(pair_keys)} pairs")

        print("Loading model...")
        extractor, device = load_model(args.model, args.device)

        with h5py.File(str(out_path), "w") as f:
            # Metadata
            md = f.create_group("metadata")
            md.attrs["model_id"] = args.model
            md.attrs["n_layers"] = n_layers
            md.attrs["n_neurons"] = n_neurons
            md.attrs["atlas_file"] = str(atlas_path)
            md.attrs["parquet_file"] = str(prompts_path)
            md.attrs["d_model"] = 1024

            # Domain vocab
            dv = f.create_group("domain_vocab")
            dv.create_dataset("names", data=np.array(domain_vocab, dtype=h5py.string_dtype()))

            pairs_group = f.create_group("pairs")

            for pair_idx, (ast_n, blt_o) in enumerate(pair_keys):
                pair_key = f"{ast_n}__{blt_o}"
                print(f"\n[{pair_idx+1}/{len(pair_keys)}] {pair_key}")

                # Filter prompts for this pair
                mask_df = (df["ast_node"] == ast_n) & (df["builtin_obj"] == blt_o)
                pair_df = df[mask_df].reset_index(drop=True)
                if len(pair_df) == 0:
                    print(f"  [skip] no prompts found")
                    continue

                prompts = pair_df["prompt_text"].tolist()
                variation_ids = pair_df["variation_id"].tolist() if "variation_id" in pair_df.columns \
                    else list(range(len(pair_df)))
                domain_col = pair_df["domain"].tolist() if "domain" in pair_df.columns \
                    else [infer_domain(t) for t in pair_df["prompt_text"].tolist()]
                domain_ids = np.array([domain_to_id.get(d, 0) for d in domain_col], dtype=np.uint8)

                # Run extraction in batches
                all_activations: list[dict] = []
                all_resid_activations: list[dict] = []
                for start in range(0, len(prompts), args.batch_size):
                    batch = prompts[start:start + args.batch_size]
                    batch_mlp, batch_resid = extractor.extract_batch_with_resid(batch, token_pos=-1)
                    all_activations.extend(batch_mlp)
                    all_resid_activations.extend(batch_resid)
                    print(f"  extracted {min(start + args.batch_size, len(prompts))}/{len(prompts)}")

                # Sanity check on first pair only
                if pair_idx == 0:
                    n_empty = sum(1 for d in all_activations if not d)
                    if n_empty == len(all_activations):
                        print("  ERROR: all activations are empty dicts! "
                              "Hooks are not capturing. Aborting.")
                        sys.exit(1)
                    sample = all_activations[0]
                    sample_norm = sum(v.norm().item() for v in sample.values())
                    print(f"  sanity: {len(sample)} layers captured, "
                          f"L2 norm sum = {sample_norm:.4f}")
                    resid_sample = all_resid_activations[0]
                    if resid_sample:
                        resid_dim = list(resid_sample.values())[0].shape[0]
                        print(f"  resid sanity: {len(resid_sample)} layers, d_model={resid_dim}")

                # Write pair group
                pg = pairs_group.create_group(pair_key)
                pm = pg.create_group("prompt_meta")
                pm.create_dataset("variation_id",
                                  data=np.array(variation_ids, dtype=np.int32))
                pm.create_dataset("domain_id", data=domain_ids)
                # Globally unique prompt IDs
                prompt_ids = [f"{pair_key}_{i}" for i in range(len(pair_df))]
                pm.create_dataset("prompt_id",
                                  data=np.array(prompt_ids, dtype=h5py.string_dtype()))
                # Domain name strings for convenience
                pm.create_dataset("domain_name",
                                  data=np.array(domain_col, dtype=h5py.string_dtype()))

                # Project onto circuit per layer
                for lid in range(n_layers):
                    layer_mask = pair_masks.get((ast_n, blt_o), {}).get(lid)
                    if layer_mask is None:
                        continue
                    mask_indices = np.where(layer_mask)[0].astype(np.uint16)
                    K = len(mask_indices)
                    if K == 0:
                        continue

                    # Collect projected activations: shape (N, K)
                    N = len(all_activations)
                    acts_proj = np.zeros((N, K), dtype=np.float16)
                    for i, act_dict in enumerate(all_activations):
                        act = act_dict.get(lid)
                        if act is not None:
                            acts_proj[i] = act.numpy()[mask_indices].astype(np.float16)

                    lg = pg.create_group(f"layer_{lid}")
                    lg.create_dataset("mask_indices", data=mask_indices, compression="gzip")
                    lg.create_dataset("acts_proj_f16", data=acts_proj, compression="gzip")

                    # Save full residual stream (d_model=1024) for baseline comparison
                    d_model = list(all_resid_activations[0].values())[0].shape[0] if all_resid_activations and all_resid_activations[0] else 1024
                    resid_arr = np.zeros((N, d_model), dtype=np.float16)
                    for i, resid_dict in enumerate(all_resid_activations):
                        rv = resid_dict.get(lid)
                        if rv is not None:
                            resid_arr[i] = rv.numpy().astype(np.float16)
                    lg.create_dataset("resid_f16", data=resid_arr, compression="gzip")

                print(f"  written {len(prompts)} prompts × {n_layers} layers")

        print(f"\nDone. Output: {out_path}")
        if not extractor.use_hook_recorder:
            extractor.remove_hooks()
    finally:
        atlas["handle"].close()


if __name__ == "__main__":
    main()
