#!/usr/bin/env python3
"""
X06 Phase A — Full activation extraction for subspace geometry (online, GPU required)

Extracts FULL (not mask-projected) per-prompt activation vectors for a target
AST-builtin pair and its contrast/control/neighbor groups, saving to an
intermediate HDF5 for Phase B geometry analysis.

Usage:
    python experiments/scripts/x06_extract_subspace.py \
        --prompts data/small_40x50x50_validated_prompts.parquet \
        --target-pair For__list \
        --n-contrast 5 --m-per-contrast 10 --n-control 5 \
        --neighbors For__tuple,While__list,ListComp__list \
        --out experiments/outputs/x06/For__list_activations.h5 \
        --batch-size 8 --device auto --seed 42

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

MODEL_ID = "openai/circuit-sparsity"
N_LAYERS = 8
N_NEURONS = 2048


def parse_args():
    p = argparse.ArgumentParser(
        description="X06-A: extract full activation vectors for subspace geometry")
    p.add_argument("--prompts",
                   default="data/small_40x50x50_validated_prompts.parquet")
    p.add_argument("--target-pair", required=True,
                   help="Target pair, e.g. For__list")
    p.add_argument("--n-contrast", type=int, default=5,
                   help="Number of contrast builtins/ASTs to sample")
    p.add_argument("--m-per-contrast", type=int, default=10,
                   help="Number of prompts per contrast pair")
    p.add_argument("--n-control", type=int, default=5,
                   help="Number of control pairs (both differ)")
    p.add_argument("--neighbors", type=str, default="",
                   help="Comma-separated neighbor pairs, e.g. For__tuple,While__list")
    p.add_argument("--out",
                   default="experiments/outputs/x06/activations.h5")
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


# ── Model loading (copied from x02_extract_prompt_level.py) ─────────────

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

    return hook_mod


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

    # hook_recorder doesn't work through the AutoModelForCausalLM wrapper
    # (returns 0 keys), so always use manual hooks with the correct pattern.
    extractor = ActivationExtractor(
        model=model,
        tokenizer=tokenizer,
        device=device,
        n_layers=N_LAYERS,
        use_hook_recorder=False,
        hook_recorder_fn=None,
    )

    # Detect the MLP module pattern from the actual model architecture
    import re
    module_names = [name for name, _ in model.named_modules()]
    mlp_modules = [n for n in module_names if n.endswith(".mlp")]
    print(f"  MLP modules found: {mlp_modules[:4]}...")

    if mlp_modules:
        pattern = re.sub(r'\.\d+\.mlp$', '.{layer_id}.mlp', mlp_modules[0])
    else:
        pattern = "circuit_model.transformer.h.{layer_id}.mlp"
    print(f"  Hook pattern: {pattern}")

    extractor.set_hook_pattern(pattern)
    extractor.register_hooks()
    print(f"  Registered {len(extractor.hooks)} hooks")

    return extractor, device


# ── Prompt grouping ──────────────────────────────────────────────────────

def build_prompt_groups(
    df: pd.DataFrame,
    target_ast: str,
    target_builtin: str,
    n_contrast: int,
    m_per_contrast: int,
    n_control: int,
    neighbor_pairs: list[tuple[str, str]],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Build prompt groups for subspace geometry extraction.

    Returns a DataFrame with columns:
        prompt_text, ast_node, builtin_obj, group, variation_id
    """
    rows: list[dict] = []

    # Helper: sample up to m prompts for a given (ast, builtin)
    def sample_pair(ast_n: str, blt_o: str, group: str, m: int):
        mask = (df["ast_node"] == ast_n) & (df["builtin_obj"] == blt_o)
        sub = df[mask]
        if len(sub) == 0:
            return
        chosen = sub.sample(n=min(m, len(sub)), random_state=int(rng.integers(2**31)))
        for i, (_, row) in enumerate(chosen.iterrows()):
            rows.append({
                "prompt_text": row["prompt_text"],
                "ast_node": ast_n,
                "builtin_obj": blt_o,
                "group": group,
                "variation_id": row.get("variation_id", i),
            })

    # Group 1: target — all available prompts
    target_mask = (df["ast_node"] == target_ast) & (df["builtin_obj"] == target_builtin)
    target_df = df[target_mask]
    for i, (_, row) in enumerate(target_df.iterrows()):
        rows.append({
            "prompt_text": row["prompt_text"],
            "ast_node": target_ast,
            "builtin_obj": target_builtin,
            "group": "target",
            "variation_id": row.get("variation_id", i),
        })
    print(f"  target ({target_ast}, {target_builtin}): {len(target_df)} prompts")

    # Group 2: same_ast — same AST, different builtins
    other_builtins = df[
        (df["ast_node"] == target_ast) & (df["builtin_obj"] != target_builtin)
    ]["builtin_obj"].unique()
    if len(other_builtins) > n_contrast:
        other_builtins = rng.choice(other_builtins, size=n_contrast, replace=False)
    for blt in other_builtins:
        sample_pair(target_ast, blt, "same_ast", m_per_contrast)
    print(f"  same_ast: {len(other_builtins)} builtins × ≤{m_per_contrast} prompts")

    # Group 3: same_builtin — same builtin, different ASTs
    other_asts = df[
        (df["builtin_obj"] == target_builtin) & (df["ast_node"] != target_ast)
    ]["ast_node"].unique()
    if len(other_asts) > n_contrast:
        other_asts = rng.choice(other_asts, size=n_contrast, replace=False)
    for ast_n in other_asts:
        sample_pair(ast_n, target_builtin, "same_builtin", m_per_contrast)
    print(f"  same_builtin: {len(other_asts)} ASTs × ≤{m_per_contrast} prompts")

    # Group 4: control — both differ
    control_candidates = df[
        (df["ast_node"] != target_ast) & (df["builtin_obj"] != target_builtin)
    ].groupby(["ast_node", "builtin_obj"]).size().reset_index(name="count")
    control_candidates = control_candidates[control_candidates["count"] >= 5]
    if len(control_candidates) > n_control:
        control_candidates = control_candidates.sample(
            n=n_control, random_state=int(rng.integers(2**31)))
    for _, crow in control_candidates.iterrows():
        sample_pair(crow["ast_node"], crow["builtin_obj"], "control", m_per_contrast)
    print(f"  control: {len(control_candidates)} pairs × ≤{m_per_contrast} prompts")

    # Group 5: neighbors — explicit pairs
    for ast_n, blt_o in neighbor_pairs:
        sample_pair(ast_n, blt_o, "neighbor", m_per_contrast)
    print(f"  neighbor: {len(neighbor_pairs)} pairs × ≤{m_per_contrast} prompts")

    result = pd.DataFrame(rows)
    print(f"  TOTAL: {len(result)} prompts across {result['group'].nunique()} groups")
    return result


def parse_pair(pair_str: str) -> tuple[str, str]:
    """Parse 'For__list' into ('For', 'list')."""
    parts = pair_str.split("__")
    if len(parts) != 2:
        raise ValueError(f"Invalid pair format: {pair_str!r} (expected AST__builtin)")
    return parts[0], parts[1]


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    prompts_path = resolve(args.prompts)
    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    target_ast, target_builtin = parse_pair(args.target_pair)

    # Parse neighbor pairs
    neighbor_pairs = []
    if args.neighbors:
        for nb in args.neighbors.split(","):
            nb = nb.strip()
            if nb:
                neighbor_pairs.append(parse_pair(nb))

    print(f"Loading prompts: {prompts_path}")
    df = pd.read_parquet(prompts_path)
    print(f"  {len(df)} rows, columns: {list(df.columns)}")

    print("Building prompt groups...")
    groups_df = build_prompt_groups(
        df, target_ast, target_builtin,
        n_contrast=args.n_contrast,
        m_per_contrast=args.m_per_contrast,
        n_control=args.n_control,
        neighbor_pairs=neighbor_pairs,
        rng=rng,
    )

    if len(groups_df) == 0:
        print("ERROR: no prompts selected. Check --target-pair and parquet content.")
        sys.exit(1)

    print("Loading model...")
    extractor, _device = load_model(args.model, args.device)

    # Extract activations
    prompts = groups_df["prompt_text"].tolist()
    N = len(prompts)
    print(f"Extracting activations for {N} prompts...")

    all_activations: list[dict] = []
    for start in range(0, N, args.batch_size):
        batch = prompts[start:start + args.batch_size]
        batch_acts = extractor.extract_batch(batch, token_pos=-1)
        all_activations.extend(batch_acts)
        print(f"  extracted {min(start + args.batch_size, N)}/{N}")

    # Write HDF5
    print(f"Writing HDF5: {out_path}")
    with h5py.File(str(out_path), "w") as f:
        # Metadata
        f.attrs["model_id"] = args.model
        f.attrs["n_layers"] = N_LAYERS
        f.attrs["n_neurons"] = N_NEURONS
        f.attrs["target_ast"] = target_ast
        f.attrs["target_builtin"] = target_builtin

        # Prompt metadata
        pg = f.create_group("prompts")
        pg.create_dataset(
            "text",
            data=np.array(groups_df["prompt_text"].tolist(),
                          dtype=h5py.string_dtype()))
        pg.create_dataset(
            "ast_node",
            data=np.array(groups_df["ast_node"].tolist(),
                          dtype=h5py.string_dtype()))
        pg.create_dataset(
            "builtin_obj",
            data=np.array(groups_df["builtin_obj"].tolist(),
                          dtype=h5py.string_dtype()))
        pg.create_dataset(
            "group",
            data=np.array(groups_df["group"].tolist(),
                          dtype=h5py.string_dtype()))
        pg.create_dataset(
            "variation_id",
            data=np.array(groups_df["variation_id"].tolist(), dtype=np.int32))

        # Full activation vectors per layer
        ag = f.create_group("activations")
        for lid in range(N_LAYERS):
            layer_acts = np.zeros((N, N_NEURONS), dtype=np.float16)
            for i, act_dict in enumerate(all_activations):
                act = act_dict.get(lid)
                if act is not None:
                    layer_acts[i] = act.numpy().astype(np.float16)
            ag.create_dataset(
                f"layer_{lid}", data=layer_acts, compression="gzip")

        print(f"  Written: {N} prompts × {N_LAYERS} layers × {N_NEURONS} neurons")

    print(f"\nDone. Output: {out_path}")
    if not extractor.use_hook_recorder:
        extractor.remove_hooks()


if __name__ == "__main__":
    main()
