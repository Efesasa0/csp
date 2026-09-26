"""
Parameter sweep runner — optimized to cache expensive model inference.

Since sweep params (selectivity_threshold, specificity_ratio) are post-hoc
filters, we precompute all latents and ablation results once, then sweep
thresholds cheaply.
"""
import os
import json
import copy
import itertools
import torch
import pandas as pd
from tqdm import tqdm
from core import load_config, ensure_dirs, resolve_path, get_path
from discovery.engine import DiscoveryEngine
from discovery.attribution import PathAttributor


def _set_nested(d, dotted_key, value):
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def run_sweep():
    sweep_path = os.path.join(os.path.dirname(__file__), "sweep_config.json")
    with open(sweep_path) as f:
        sweep_cfg = json.load(f)

    base_cfg = load_config()
    ensure_dirs(base_cfg)
    out_dir = resolve_path(sweep_cfg["output_dir"])
    os.makedirs(out_dir, exist_ok=True)

    param_names = list(sweep_cfg["sweep_params"].keys())
    param_values = list(sweep_cfg["sweep_params"].values())
    combos = list(itertools.product(*param_values))
    print(f"Sweep: {len(combos)} combinations of {param_names}")

    # ------------------------------------------------------------------ #
    #  Phase 1: Load model ONCE, compute latent means ONCE                 #
    # ------------------------------------------------------------------ #
    print("\n[Phase 1] Loading model and computing latent means...")
    engine = DiscoveryEngine(base_cfg)
    shared = {
        "tokenizer": engine.tokenizer,
        "model": engine.model,
        "sae": engine.sae,
        "device": engine.device,
    }
    attributor = PathAttributor(base_cfg, shared=shared)
    df = pd.read_parquet(resolve_path(base_cfg["paths"]["dataset"]))
    engine.compute_all_means(df, n_samples=200)

    nodes = base_cfg["sae_params"]["target_nodes"]

    # ------------------------------------------------------------------ #
    #  Phase 2: Precompute per-node frequency vectors ONCE                 #
    # ------------------------------------------------------------------ #
    print("\n[Phase 2] Precomputing per-node latent frequencies...")
    node_freqs = {}  # node -> tensor of firing frequencies
    node_prompts_cache = {}  # node -> (target_prompts, other_prompts)

    for node in tqdm(nodes, desc="Frequency vectors"):
        others = [n for n in nodes if n != node]
        t_prompts = df[df["ast_node"] == node]["prompt_text"].tolist()[:50]
        o_prompts = df[df["ast_node"].isin(others)]["prompt_text"].tolist()[:50]

        if not t_prompts or not o_prompts:
            continue

        t_freq = torch.stack(
            [engine.get_latents(p) > 1e-5 for p in t_prompts]
        ).float().mean(0)
        o_freq = torch.stack(
            [engine.get_latents(p) > 1e-5 for p in o_prompts]
        ).float().mean(0)

        node_freqs[node] = {"t_freq": t_freq, "o_freq": o_freq, "spec": t_freq - o_freq}
        node_prompts_cache[node] = {
            "prompt": df[df["ast_node"] == node].iloc[0]["prompt_text"],
            "dest": df[df["ast_node"].isin(others)].iloc[0]["prompt_text"],
        }

    # ------------------------------------------------------------------ #
    #  Phase 3: Find ALL candidate features across ALL thresholds          #
    # ------------------------------------------------------------------ #
    print("\n[Phase 3] Finding candidate features across all thresholds...")
    all_thresholds = sweep_cfg["sweep_params"].get(
        "sae_params.feature_selectivity_threshold",
        [base_cfg["sae_params"]["feature_selectivity_threshold"]]
    )
    min_threshold = min(all_thresholds)

    # For each node, find winners at the LOWEST threshold — supersets all others
    node_all_winners = {}
    for node, freq_data in node_freqs.items():
        spec = freq_data["spec"]
        t_freq = freq_data["t_freq"]
        candidates = torch.where((spec >= min_threshold) & (t_freq > 0.15))[0].tolist()
        candidates = sorted(candidates, key=lambda x: spec[x].item(), reverse=True)
        node_all_winners[node] = candidates

    # ------------------------------------------------------------------ #
    #  Phase 4: Cache ablation results for all unique (node, feature) pairs #
    # ------------------------------------------------------------------ #
    print("\n[Phase 4] Caching ablation results for candidate features...")
    ablation_cache = {}  # (node, feature_id) -> {drop_zero, drop_mean, patch_gain, argmax}

    unique_pairs = set()
    for node, winners in node_all_winners.items():
        if not winners:
            continue
        # Cache ablation for top features that could be #1 at any threshold
        for f_id in winners[:20]:
            unique_pairs.add((node, f_id))

    for node, f_id in tqdm(unique_pairs, desc="Ablation cache"):
        prompt = node_prompts_cache[node]["prompt"]
        dest = node_prompts_cache[node]["dest"]

        metrics = engine.run_ablation(prompt, f_id)
        metrics["patch_gain"] = engine.run_patching(prompt, dest, f_id)
        argmax = engine.run_argmax_check(prompt, f_id)
        upstream = attributor.get_upstream_edges(prompt, f_id)

        ablation_cache[(node, f_id)] = {
            "metrics": metrics,
            "argmax": argmax,
            "upstream": upstream,
        }

    # ------------------------------------------------------------------ #
    #  Phase 5: Sweep thresholds (CHEAP — just filtering + lookup)         #
    # ------------------------------------------------------------------ #
    print(f"\n[Phase 5] Running {len(combos)} threshold combinations...")
    summary = []

    for i, combo in enumerate(combos):
        cfg = copy.deepcopy(base_cfg)
        tag_parts = []
        for name, val in zip(param_names, combo):
            _set_nested(cfg, name, val)
            short = name.split(".")[-1]
            tag_parts.append(f"{short}={val}")
        tag = "__".join(tag_parts)

        threshold = cfg["sae_params"]["feature_selectivity_threshold"]
        spec_ratio = cfg["sae_params"]["specificity_ratio"]

        report = {}
        for node, freq_data in node_freqs.items():
            spec = freq_data["spec"]
            t_freq = freq_data["t_freq"]
            winners = torch.where((spec >= threshold) & (t_freq > 0.15))[0].tolist()
            winners = sorted(winners, key=lambda x: spec[x].item(), reverse=True)

            if not winners:
                continue

            f_id = winners[0]
            cache_key = (node, f_id)

            if cache_key not in ablation_cache:
                # Shouldn't happen often, but handle gracefully
                prompt = node_prompts_cache[node]["prompt"]
                dest = node_prompts_cache[node]["dest"]
                metrics = engine.run_ablation(prompt, f_id)
                metrics["patch_gain"] = engine.run_patching(prompt, dest, f_id)
                ablation_cache[cache_key] = {
                    "metrics": metrics,
                    "argmax": engine.run_argmax_check(prompt, f_id),
                    "upstream": attributor.get_upstream_edges(prompt, f_id),
                }

            cached = ablation_cache[cache_key]
            report[node] = {
                "feature": int(f_id),
                "n_winners": len(winners),
                "metrics": cached["metrics"],
                "argmax": cached["argmax"],
                "upstream": cached["upstream"],
            }

        # Cross-ablation specificity (reuses ablation_cache)
        node_prompts_map = {
            n: node_prompts_cache[n]["prompt"] for n in report
        }
        cross_matrix = {}
        for src_node, data in report.items():
            f_id = data["feature"]
            drops = engine.run_cross_ablation(f_id, node_prompts_map)
            specific, own_drop, avg_other, ratio = engine.is_specific(drops, src_node)
            cross_matrix[src_node] = {
                "drops": drops,
                "specificity": {
                    "is_specific": ratio > spec_ratio,
                    "own_drop": own_drop,
                    "avg_other_drop": round(avg_other, 4),
                    "ratio": round(ratio, 4),
                },
            }

        report["_cross_ablation"] = cross_matrix
        n_specific = sum(
            1 for v in cross_matrix.values()
            if v["specificity"]["is_specific"]
        )

        run_summary = {
            "params": {n: v for n, v in zip(param_names, combo)},
            "tag": tag,
            "nodes_found": len([k for k in report if not k.startswith("_")]),
            "n_specific": n_specific,
            "avg_drop_zero": round(
                sum(r["metrics"]["drop_zero"] for k, r in report.items() if not k.startswith("_"))
                / max(len([k for k in report if not k.startswith("_")]), 1), 4
            ),
        }
        summary.append(run_summary)

        status = f"[{i+1}/{len(combos)}] {tag}: " \
                 f"{run_summary['nodes_found']} nodes, " \
                 f"{n_specific} specific, " \
                 f"avg_drop={run_summary['avg_drop_zero']:.4f}"
        print(status)

        run_path = os.path.join(out_dir, f"run_{tag}.json")
        with open(run_path, "w") as f:
            json.dump(report, f, indent=2)

    # ------------------------------------------------------------------ #
    #  Save summary                                                        #
    # ------------------------------------------------------------------ #
    summary_path = os.path.join(out_dir, "sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    print(f"\n{'='*70}")
    print(f"SWEEP COMPLETE — {len(combos)} runs")
    print(f"{'='*70}")
    print(f"{'Threshold':<12} {'SpecRatio':<12} {'Nodes':<8} {'Specific':<10} {'AvgDrop'}")
    print("-" * 60)
    for s in summary:
        t = s["params"].get("sae_params.feature_selectivity_threshold", "?")
        r = s["params"].get("sae_params.specificity_ratio", "?")
        print(f"{t:<12} {r:<12} {s['nodes_found']:<8} {s['n_specific']:<10} {s['avg_drop_zero']:.4f}")

    print(f"\nResults: {out_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    run_sweep()
