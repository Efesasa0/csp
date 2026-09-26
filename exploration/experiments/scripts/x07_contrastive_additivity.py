#!/usr/bin/env python3
"""
X07 — Contrastive Additivity Analysis (GPU required)

Tests whether AST and builtin representations are additive (independent) or
entangled (interaction circuit) in the circuit_sparsity residual stream.

For each explicit (AST, builtin) stub:
    additivity = cos_sim(act_combined, act_ast_baseline + act_builtin_baseline)
    ~1.0 => independent/additive, <<1.0 => interaction circuit

Usage:
    python experiments/scripts/x07_contrastive_additivity.py \
        --stubs data/contrastive_stubs_v2.json \
        --out experiments/outputs/x07 \
        --device auto

Requires: torch, transformers, huggingface_hub, matplotlib, numpy
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

import numpy as np

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

MODEL_ID = "openai/circuit-sparsity"


def parse_args():
    p = argparse.ArgumentParser(
        description="X07: contrastive additivity analysis")
    p.add_argument("--stubs", default="data/contrastive_stubs_v2.json",
                   help="Path to contrastive_stubs_v2.json")
    p.add_argument("--out", default="experiments/outputs/x07",
                   help="Output directory for plots and CSVs")
    p.add_argument("--cache", default=None,
                   help="Path for residual cache .npz (default: <out>/residuals_cache.npz)")
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--device", default="auto")
    p.add_argument("--layers", default=None,
                   help="Comma-separated layer indices to test (default: all)")
    return p.parse_args()


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


# ── Model loading (same pattern as x06_extract_subspace.py) ──────────────

def _patch_circuit_sparsity(model_id: str):
    """Load the custom CSP architecture from HuggingFace."""
    from huggingface_hub import hf_hub_download

    tm_base = os.path.expanduser(
        "~/.cache/huggingface/modules/transformers_modules/openai")
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


def load_model(model_id: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    _patch_circuit_sparsity(model_id)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    model.to(device).eval()

    return model, tokenizer


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    import matplotlib
    matplotlib.use("Agg")

    args = parse_args()
    stubs_path = resolve(args.stubs)
    out_dir = resolve(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache) if args.cache else out_dir / "residuals_cache.npz"

    if not stubs_path.exists():
        print(f"ERROR: stubs not found: {stubs_path}")
        print("Run: python generate_contrastive_stubs_v2.py")
        sys.exit(1)

    print(f"Loading model: {args.model}")
    model, tokenizer = load_model(args.model, args.device)

    # Import ContrastiveAnalysis from repo root
    from contrastive_analysis import ContrastiveAnalysis

    print(f"Loading stubs: {stubs_path}")
    ca = ContrastiveAnalysis(model, tokenizer, stubs_path=stubs_path, layer=-1)

    print("Extracting residuals (this may take a while)...")
    ca.extract_all(cache_path=str(cache_path))

    # Determine layers
    if args.layers:
        test_layers = [int(x.strip()) for x in args.layers.split(",")]
    else:
        n_layers = next(iter(ca._cache.values())).shape[0]
        test_layers = list(range(n_layers))
    print(f"Testing layers: {test_layers}")

    # Run additivity test
    results = ca.run_additivity_test(layers=test_layers)

    final_layer = test_layers[-1]
    final_metric = f"additivity_L{final_layer}"

    # Print summary
    ca.summary(results, metric=final_metric)

    # Save results CSV
    import pandas as pd
    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / "additivity_results.csv", index=False)
    print(f"Saved results CSV: {out_dir / 'additivity_results.csv'}")

    # Save pair-level aggregation
    pair_scores = ca.aggregate_by_pair(results, final_metric)
    pair_rows = []
    for (ast_node, builtin), score in sorted(pair_scores.items()):
        pair_rows.append({
            "ast_node": ast_node,
            "builtin": builtin,
            "pair": f"{ast_node}__{builtin}",
            "additivity_score": score,
        })
    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(out_dir / "pair_additivity.csv", index=False)
    print(f"Saved pair CSV: {out_dir / 'pair_additivity.csv'}")

    # Save layer-level summary (mean additivity per layer)
    layer_rows = []
    for L in test_layers:
        key = f"additivity_L{L}"
        scores = [r[key] for r in results if key in r]
        if scores:
            layer_rows.append({
                "layer": L,
                "mean_additivity": np.mean(scores),
                "std_additivity": np.std(scores),
                "min_additivity": np.min(scores),
                "max_additivity": np.max(scores),
                "n_stubs": len(scores),
            })
    layer_df = pd.DataFrame(layer_rows)
    layer_df.to_csv(out_dir / "layer_additivity.csv", index=False)
    print(f"Saved layer CSV: {out_dir / 'layer_additivity.csv'}")

    # Generate plots
    print("Generating plots...")

    ca.plot_heatmap(
        results, metric=final_metric,
        save_path=str(out_dir / "additivity_heatmap.png"),
    )

    ca.plot_layer_traces(
        results, metric_prefix="additivity", top_k=10,
        save_path=str(out_dir / "layer_traces.png"),
    )

    ca.plot_baseline_cosine_matrices(
        layer=final_layer,
        save_path=str(out_dir / "baseline_cosine_matrices.png"),
    )

    ca.plot_explicit_vs_proxy(
        results, metric=final_metric,
        save_path=str(out_dir / "explicit_vs_proxy.png"),
    )

    # Top interactions and top additive pairs
    top_interact = ca.top_interactions(results, final_metric, n=20, lowest=True)
    top_additive = ca.top_interactions(results, final_metric, n=20, lowest=False)
    pd.DataFrame(top_interact).to_csv(out_dir / "top_interactions.csv", index=False)
    pd.DataFrame(top_additive).to_csv(out_dir / "top_additive.csv", index=False)

    print(f"\nDone. All outputs in: {out_dir}")


if __name__ == "__main__":
    main()
