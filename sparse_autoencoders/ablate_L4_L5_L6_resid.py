"""Experiment #13: Causal ablation at L4_resid, L5_resid, L6_resid.

Reuses pipeline.run_pipeline_for_site. Writes
  data/reports/circuit_report_L{4,5,6}_resid.json
Then aggregates into data/posthoc/ablation_staged.json.
"""
import os, sys, json, copy, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from core import load_config, ensure_dirs, get_path, resolve_path, data_path, patch_environment, get_device, get_sae_path
from transformers import AutoTokenizer, AutoModelForCausalLM
from pipeline import run_pipeline_for_site

def main():
    cfg = load_config()
    ensure_dirs(cfg)

    patch_environment()
    device = get_device()
    print("Loading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained("openai/circuit-sparsity", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "openai/circuit-sparsity", trust_remote_code=True
    ).to(device).eval()
    print(f"  loaded in {time.time()-t0:.1f}s on {device}")

    shared = {"tokenizer": tokenizer, "model": model, "device": device}
    df = pd.read_parquet(resolve_path(cfg["paths"]["dataset"]))

    results = {}
    for layer in [4, 5, 6]:
        sae_path = get_sae_path(cfg, layer, "resid")
        if not os.path.exists(sae_path):
            print(f"  skip L{layer} — no SAE at {sae_path}")
            continue
        lay_cfg = copy.deepcopy(cfg)
        lay_cfg["model"]["target_layer"] = layer
        t0 = time.time()
        rep = run_pipeline_for_site(lay_cfg, df, "resid", layer, shared)
        elapsed = time.time() - t0
        print(f"=== L{layer}_resid done in {elapsed:.0f}s ===")

        drops = {}
        for node, v in rep.items():
            if node.startswith("_"):
                continue
            dz = v.get("metrics", {}).get("drop_zero")
            if dz is not None:
                drops[node] = dz
        results[f"L{layer}_resid"] = drops

    # Load existing L7 resid
    l7 = json.load(open(data_path("reports/circuit_report_L7_resid.json")))
    results["L7_resid"] = {n: v["metrics"]["drop_zero"]
                           for n, v in l7.items()
                           if not n.startswith("_") and v.get("metrics", {}).get("drop_zero") is not None}

    os.makedirs(data_path("posthoc"), exist_ok=True)
    with open(data_path("posthoc/ablation_staged.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nWrote data/posthoc/ablation_staged.json")

if __name__ == "__main__":
    main()
