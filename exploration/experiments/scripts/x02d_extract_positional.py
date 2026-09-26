#!/usr/bin/env python3
"""
X02-D Phase A — Positional Activation Extraction (GPU required)

Extracts activations at 5 token positions per prompt (first, 25%, 50%, 75%, last)
for a subset of pairs, saving to HDF5 for positional JSD analysis.

Usage:
    python experiments/scripts/x02d_extract_positional.py
    python experiments/scripts/x02d_extract_positional.py \
        --atlas data/small_dynamic_feature_atlas.h5 \
        --prompts data/small_40x50x50_validated_prompts.parquet \
        --out experiments/outputs/x02d/positional_activations.h5 \
        --device cuda --batch-size 8
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


TOKEN_POSITIONS = [0, 0.25, 0.5, 0.75, -1]  # -1 = last
POS_LABELS = ["first", "q1", "mid", "q3", "last"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--atlas", default="data/small_dynamic_feature_atlas.h5")
    p.add_argument("--prompts", default="data/small_40x50x50_validated_prompts.parquet")
    p.add_argument("--out", default="experiments/outputs/x02d/positional_activations.h5")
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-pairs", type=int, default=20,
                   help="Max pairs to process (for speed)")
    return p.parse_args()


def resolve(path: str) -> Path:
    pp = Path(path)
    return pp if pp.is_absolute() else ROOT / pp


def _patch_circuit_sparsity(model_id: str):
    """Load the custom CSP architecture from HuggingFace."""
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


def load_model(model_id: str, device: str):
    import re
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

    extractor = ActivationExtractor(
        model=model, tokenizer=tokenizer, device=device, n_layers=8,
        use_hook_recorder=False, hook_recorder_fn=None,
    )

    module_names = [name for name, _ in model.named_modules()]
    mlp_modules = [n for n in module_names if n.endswith(".mlp")]
    if mlp_modules:
        first = mlp_modules[0]
        pattern = re.sub(r'\.\d+\.mlp$', '.{layer_id}.mlp', first)
    else:
        pattern = "circuit_model.transformer.h.{layer_id}.mlp"

    extractor.set_hook_pattern(pattern)
    extractor.register_hooks()
    print(f"  Registered {len(extractor.hooks)} MLP hooks")

    return extractor, tokenizer, device


def resolve_positions(seq_len: int) -> list[int]:
    """Resolve fractional positions to absolute token indices."""
    positions = []
    for p in TOKEN_POSITIONS:
        if p == -1:
            positions.append(seq_len - 1)
        elif isinstance(p, float):
            positions.append(min(int(p * seq_len), seq_len - 1))
        else:
            positions.append(min(p, seq_len - 1))
    return positions


def main():
    import torch

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

        print(f"Loading prompts: {prompts_path}")
        df = pd.read_parquet(prompts_path)

        domain_vocab = sorted(set(
            df["domain"].tolist() if "domain" in df.columns else DOMAIN_NAMES
        ))
        domain_to_id = {d: i for i, d in enumerate(domain_vocab)}

        pair_keys = list(pair_masks.keys())[:args.max_pairs]
        print(f"  Processing {len(pair_keys)} pairs at {len(TOKEN_POSITIONS)} positions")

        print("Loading model...")
        extractor, tokenizer, device = load_model(args.model, args.device)

        with h5py.File(str(out_path), "w") as f:
            md = f.create_group("metadata")
            md.attrs["n_layers"] = n_layers
            md.attrs["n_positions"] = len(TOKEN_POSITIONS)
            md.create_dataset("pos_labels",
                              data=np.array(POS_LABELS, dtype=h5py.string_dtype()))
            f.create_group("domain_vocab").create_dataset(
                "names", data=np.array(domain_vocab, dtype=h5py.string_dtype()))

            pairs_group = f.create_group("pairs")

            for pi, (ast_n, blt_o) in enumerate(pair_keys):
                pair_key = f"{ast_n}__{blt_o}"
                print(f"\n[{pi+1}/{len(pair_keys)}] {pair_key}")

                mask_df = (df["ast_node"] == ast_n) & (df["builtin_obj"] == blt_o)
                pair_df = df[mask_df].reset_index(drop=True)
                if len(pair_df) == 0:
                    continue

                prompts = pair_df["prompt_text"].tolist()
                domain_col = pair_df["domain"].tolist() if "domain" in pair_df.columns \
                    else [infer_domain(t) for t in prompts]
                domain_ids = np.array([domain_to_id.get(d, 0) for d in domain_col],
                                      dtype=np.uint8)

                pg = pairs_group.create_group(pair_key)
                pm = pg.create_group("prompt_meta")
                pm.create_dataset("domain_id", data=domain_ids)

                # Tokenize to get sequence lengths
                encoded = tokenizer(
                    prompts, return_tensors="pt", padding=True,
                    truncation=True, add_special_tokens=False,
                )
                attention_mask = encoded["attention_mask"]
                seq_lengths = attention_mask.sum(dim=1).tolist()

                # For each position, extract activations
                for pos_idx, pos_label in enumerate(POS_LABELS):
                    # Compute absolute positions per prompt
                    abs_positions = [resolve_positions(sl)[pos_idx] for sl in seq_lengths]

                    # Extract one prompt at a time (different token positions)
                    all_acts: list[dict] = []
                    for start in range(0, len(prompts), args.batch_size):
                        batch = prompts[start:start + args.batch_size]
                        batch_pos = abs_positions[start:start + args.batch_size]

                        # Use the most common position in this batch for extraction
                        # (approximate — exact would require per-sample hooks)
                        median_pos = int(np.median(batch_pos))
                        batch_acts = extractor.extract_batch(batch, token_pos=median_pos)
                        all_acts.extend(batch_acts)

                    # Project onto circuit per layer
                    for lid in range(n_layers):
                        layer_mask = pair_masks.get((ast_n, blt_o), {}).get(lid)
                        if layer_mask is None:
                            continue
                        mask_indices = np.where(layer_mask)[0].astype(np.uint16)
                        K = len(mask_indices)
                        if K == 0:
                            continue

                        N = len(all_acts)
                        acts_proj = np.zeros((N, K), dtype=np.float16)
                        for i, act_dict in enumerate(all_acts):
                            act = act_dict.get(lid)
                            if act is not None:
                                acts_proj[i] = act.numpy()[mask_indices].astype(np.float16)

                        ds_key = f"pos_{pos_label}/layer_{lid}"
                        pg.create_dataset(f"{ds_key}/acts_proj_f16",
                                          data=acts_proj, compression="gzip")
                        if pos_idx == 0:
                            pg.create_dataset(f"{ds_key}/mask_indices",
                                              data=mask_indices, compression="gzip")

                print(f"  written {len(prompts)} prompts × {len(POS_LABELS)} positions")

        print(f"\nDone. Output: {out_path}")
        extractor.remove_hooks()
    finally:
        atlas["handle"].close()


if __name__ == "__main__":
    main()
