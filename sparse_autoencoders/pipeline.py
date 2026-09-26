import pandas as pd
import json
from core import (
    load_config, ensure_dirs, get_path, resolve_path,
    patch_environment, get_device, get_sae_path, load_sae_from_checkpoint, get_n_embd
)
from discovery.engine import DiscoveryEngine
from discovery.attribution import PathAttributor
from transformers import AutoTokenizer, AutoModelForCausalLM


def run_pipeline_for_site(cfg, df, site, layer, shared, report_suffix=""):
    """Run the full discovery pipeline for a given site and layer."""
    engine = DiscoveryEngine(cfg, site=site, layer=layer, shared={
        "tokenizer": shared["tokenizer"],
        "model": shared["model"],
        "device": shared["device"],
        "sae": None,  # engine will load the right SAE
    })

    # Only run path attribution for MLP (attention attribution is MLP-specific)
    if site == "mlp":
        attr_shared = {
            "tokenizer": engine.tokenizer,
            "model": engine.model,
            "sae": engine.sae,
            "device": engine.device,
        }
        attributor = PathAttributor(cfg, shared=attr_shared)

    engine.compute_all_means(df)

    nodes = cfg["sae_params"]["target_nodes"]
    master_report = {}

    label = f"L{layer}/{site}"
    print(f"\n{'='*60}")
    print(f"  PIPELINE: {label}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------ #
    #  Per-node discovery                                                  #
    # ------------------------------------------------------------------ #
    for node in nodes:
        others = [n for n in nodes if n != node]
        winners = engine.find_best_feature(df, node, others)
        if not winners:
            continue
        f_id = winners[0]

        t_rows = df[df["ast_node"] == node]
        d_rows = df[df["ast_node"].isin(others)]
        if t_rows.empty or d_rows.empty:
            continue

        prompt = t_rows.iloc[0]["prompt_text"]
        dest_prompt = d_rows.iloc[0]["prompt_text"]

        metrics = engine.run_ablation(prompt, f_id)
        metrics["patch_gain"] = engine.run_patching(prompt, dest_prompt, f_id)

        entry = {
            "feature": int(f_id),
            "all_winners": [int(w) for w in winners[:10]],
            "prompts": {"source": prompt, "dest": dest_prompt},
            "metrics": metrics,
            "argmax": engine.run_argmax_check(prompt, f_id),
        }

        if site == "mlp":
            entry["upstream"] = attributor.get_upstream_edges(prompt, f_id)

        master_report[node] = entry
        print(f"  {node}: feature={f_id}, drop_zero={metrics['drop_zero']}, "
              f"patch_gain={metrics['patch_gain']}")

    # ------------------------------------------------------------------ #
    #  Cross-ablation matrix                                               #
    # ------------------------------------------------------------------ #
    node_prompts = {n: master_report[n]["prompts"]["source"] for n in master_report}
    cross_matrix = {}

    for src_node, data in master_report.items():
        f_id = data["feature"]
        drops = engine.run_cross_ablation(f_id, node_prompts)
        specific, own_drop, avg_other, ratio = engine.is_specific(drops, src_node)
        cross_matrix[src_node] = {
            "drops": drops,
            "specificity": {
                "is_specific": specific,
                "own_drop": own_drop,
                "avg_other_drop": round(avg_other, 4),
                "ratio": ratio,
            },
        }
        tag = "SPECIFIC" if specific else "shared"
        print(f"  {src_node}: ratio={ratio} [{tag}]")

    master_report["_cross_ablation"] = cross_matrix
    master_report["_meta"] = {"site": site, "layer": layer}

    # Save
    report_name = f"circuit_report_L{layer}_{site}.json"
    out_path = get_path(cfg, "reports", report_name)
    with open(out_path, "w") as f:
        json.dump(master_report, f, indent=4)
    print(f"  Report saved to {out_path}")

    return master_report


def run_pipeline():
    cfg = load_config()
    ensure_dirs(cfg)

    # Load model ONCE
    patch_environment()
    device = get_device()
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("openai/circuit-sparsity", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "openai/circuit-sparsity", trust_remote_code=True
    ).to(device).eval()

    shared = {"tokenizer": tokenizer, "model": model, "device": device}
    df = pd.read_parquet(resolve_path(cfg["paths"]["dataset"]))

    layer = cfg["model"]["target_layer"]
    all_reports = {}

    # Run for both sites
    for site in ["mlp", "resid"]:
        sae_path = get_sae_path(cfg, layer, site)
        import os
        if not os.path.exists(sae_path):
            print(f"  Skipping {site}: no SAE at {sae_path}")
            continue
        report = run_pipeline_for_site(cfg, df, site, layer, shared)
        all_reports[f"L{layer}_{site}"] = report

    # Also save combined master report for backward compat
    if f"L{layer}_mlp" in all_reports:
        out_path = get_path(cfg, "reports", cfg["paths"]["master_report"])
        with open(out_path, "w") as f:
            json.dump(all_reports[f"L{layer}_mlp"], f, indent=4)

    # Save combined
    combined_path = get_path(cfg, "reports", "combined_circuit_report.json")
    with open(combined_path, "w") as f:
        json.dump(all_reports, f, indent=4)
    print(f"\nCombined report saved to {combined_path}")


if __name__ == "__main__":
    run_pipeline()
