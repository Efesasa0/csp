#!/usr/bin/env python3
"""
X02-A — Causal Ablation

Zero-ablates top-10 syntax-dominant and top-10 meaning-sensitive circuits,
then measures KL divergence shift per domain.

Usage:
    python experiments/scripts/x02a_causal_ablation.py
    python experiments/scripts/x02a_causal_ablation.py \
        --csv experiments/outputs/x02/ranked_circuits.csv \
        --atlas data/small_dynamic_feature_atlas.h5 \
        --prompts data/small_40x50x50_validated_prompts.parquet \
        --out experiments/outputs/x02a \
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

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "src"))

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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="experiments/outputs/x02/ranked_circuits.csv")
    p.add_argument("--atlas", default="data/small_dynamic_feature_atlas.h5")
    p.add_argument("--prompts", default="data/small_40x50x50_validated_prompts.parquet")
    p.add_argument("--out", default="experiments/outputs/x02a")
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--n-prompts-per-domain", type=int, default=5,
                   help="Number of prompts per domain per pair for ablation")
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

    # Detect MLP module pattern
    module_names = [name for name, _ in model.named_modules()]
    mlp_modules = [n for n in module_names if n.endswith(".mlp")]
    if mlp_modules:
        first = mlp_modules[0]
        pattern = re.sub(r'\.\d+\.mlp$', '.{layer_id}.mlp', first)
    else:
        pattern = "circuit_model.transformer.h.{layer_id}.mlp"

    return model, tokenizer, device, pattern


def kl_divergence(logits_base: torch.Tensor, logits_ablated: torch.Tensor) -> float:
    """KL(base || ablated) on last-token logits."""
    p = torch.softmax(logits_base, dim=-1)
    log_p = torch.log_softmax(logits_base, dim=-1)
    log_q = torch.log_softmax(logits_ablated, dim=-1)
    kl = (p * (log_p - log_q)).sum(dim=-1)
    return float(kl.mean().item())


def main():
    args = parse_args()
    csv_path = resolve(args.csv)
    atlas_path = resolve(args.atlas)
    prompts_path = resolve(args.prompts)
    out_dir = resolve(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found — run X02 Phase B first.")
        sys.exit(1)

    ranked = pd.read_csv(csv_path)
    prompts_df = pd.read_parquet(prompts_path)

    # Select top-K syntax and meaning circuits
    syntax_circuits = ranked[ranked["interpretation"] == "syntax-dominant"].head(args.top_k)
    meaning_circuits = ranked[ranked["interpretation"] == "meaning-sensitive"].tail(args.top_k)
    target_circuits = pd.concat([syntax_circuits, meaning_circuits])
    print(f"Ablating {len(target_circuits)} circuits ({args.top_k} syntax + {args.top_k} meaning)")

    print("Loading atlas...")
    atlas = load_atlas_hdf5(str(atlas_path))
    pair_masks = atlas["pair_masks"]

    print("Loading model...")
    model, tokenizer, device, mlp_pattern = load_model(args.model, args.device)

    results = []
    for _, circuit in target_circuits.iterrows():
        pair_key = circuit["pair"]
        lid = int(circuit["layer"])
        interp = circuit["interpretation"]

        parts = pair_key.split("__", 1)
        if len(parts) != 2:
            continue
        ast_n, blt_o = parts

        mask = pair_masks.get((ast_n, blt_o), {}).get(lid)
        if mask is None:
            continue
        ablate_indices = np.where(mask)[0]

        # Get prompts for this pair
        pair_df = prompts_df[
            (prompts_df["ast_node"] == ast_n) & (prompts_df["builtin_obj"] == blt_o)
        ]
        if pair_df.empty:
            continue

        # Sample prompts per domain
        domain_col = pair_df["domain"].tolist() if "domain" in pair_df.columns \
            else [infer_domain(t) for t in pair_df["prompt_text"].tolist()]
        pair_df = pair_df.copy()
        pair_df["_domain"] = domain_col

        for domain in DOMAIN_NAMES:
            dom_df = pair_df[pair_df["_domain"] == domain]
            if dom_df.empty:
                continue
            sample = dom_df.head(args.n_prompts_per_domain)
            texts = sample["prompt_text"].tolist()

            encoded = tokenizer(
                texts, return_tensors="pt", padding=True,
                truncation=True, add_special_tokens=False,
            )
            input_ids = encoded["input_ids"].to(device)

            # Baseline forward pass — circuit_sparsity accepts only input_ids
            # and returns logits directly (not .logits attribute)
            with torch.no_grad():
                base_out = model(input_ids=input_ids)
                if hasattr(base_out, "logits"):
                    base_logits = base_out.logits[:, -1, :]
                elif isinstance(base_out, tuple):
                    base_logits = base_out[0][:, -1, :]
                else:
                    base_logits = base_out[:, -1, :]

            # Ablation: zero out the circuit neurons in the target MLP layer
            mlp_name = mlp_pattern.format(layer_id=lid)
            module_dict = dict(model.named_modules())
            target_module = module_dict.get(mlp_name)
            if target_module is None:
                continue

            def make_ablation_hook(indices):
                def hook_fn(module, inp, out):
                    if isinstance(out, tuple):
                        tensor = out[0]
                    else:
                        tensor = out
                    tensor[:, :, indices] = 0.0
                    return (tensor,) + out[1:] if isinstance(out, tuple) else tensor
                return hook_fn

            handle = target_module.register_forward_hook(
                make_ablation_hook(ablate_indices))
            with torch.no_grad():
                abl_out = model(input_ids=input_ids)
                if hasattr(abl_out, "logits"):
                    abl_logits = abl_out.logits[:, -1, :]
                elif isinstance(abl_out, tuple):
                    abl_logits = abl_out[0][:, -1, :]
                else:
                    abl_logits = abl_out[:, -1, :]
            handle.remove()

            kl = kl_divergence(base_logits, abl_logits)
            results.append({
                "pair": pair_key,
                "layer": lid,
                "interpretation": interp,
                "domain": domain,
                "kl_divergence": kl,
                "n_ablated_neurons": len(ablate_indices),
                "n_prompts": len(texts),
            })

        print(f"  {pair_key} L{lid} ({interp}): done")

    atlas["handle"].close()

    rdf = pd.DataFrame(results)
    csv_out = out_dir / "ablation_results.csv"
    rdf.to_csv(csv_out, index=False)
    print(f"\nSaved: {csv_out}  ({len(rdf)} rows)")

    if not rdf.empty:
        for interp in ["syntax-dominant", "meaning-sensitive"]:
            sub = rdf[rdf["interpretation"] == interp]
            if not sub.empty:
                print(f"  {interp}: mean KL = {sub['kl_divergence'].mean():.4f}")


if __name__ == "__main__":
    main()
