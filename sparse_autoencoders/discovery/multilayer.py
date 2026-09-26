"""
Multi-layer, multi-site discovery pipeline.
Runs feature selectivity analysis on all 8 layers × 2 sites (MLP + resid).
Caches latent vectors per (layer, site, node) to avoid recomputation.
"""
import os, sys, json, torch
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import (
    load_config, TopKSAE, get_device, get_layers,
    resolve_path, data_path, ensure_dirs, get_n_embd, patch_environment
)

LAYERS = list(range(8))
SITES = ["mlp", "resid"]
N_PROMPTS = 50  # per node for selectivity analysis


def load_sae_for_site(sae_dir, layer, site, device, d_model):
    """Load SAE weights and return (sae, meta)."""
    if site == "mlp":
        path = os.path.join(sae_dir, f"sae_layer_{layer}.pt")
    else:
        path = os.path.join(sae_dir, f"sae_layer_{layer}_{site}.pt")

    if not os.path.exists(path):
        return None, None

    state = torch.load(path, map_location=device, weights_only=False)
    meta = state.pop("__meta__", {})
    expansion = meta.get("expansion", 8)
    k = meta.get("k", 32)

    sae = TopKSAE(d_model, d_model * expansion, k=k).to(device)
    sae.load_state_dict(state)
    sae.eval()
    return sae, meta


def get_hook_target(model, layer, site):
    """Return the module to hook for a given site."""
    layers = get_layers(model)
    if site == "mlp":
        return layers[layer].mlp
    else:  # resid = block output
        return layers[layer]


def harvest_latents(model, tokenizer, sae, device, layer, site, prompts, cache_dir, cache_key):
    """Get SAE latent vectors for a list of prompts. Cached."""
    cache_path = os.path.join(cache_dir, f"latents_L{layer}_{site}_{cache_key}.pt")
    if os.path.exists(cache_path):
        return torch.load(cache_path, map_location=device, weights_only=False)

    target = get_hook_target(model, layer, site)
    latents = []

    for p in prompts:
        acts = []
        hook = target.register_forward_hook(
            lambda m, i, o: acts.append(
                (o[0] if isinstance(o, tuple) else o)[:, -1, :].detach()
            )
        )
        with torch.no_grad():
            model(input_ids=tokenizer(p, return_tensors="pt").input_ids.to(device))
        hook.remove()
        _, z = sae(acts[0][0])
        latents.append(z.cpu())

    result = torch.stack(latents)
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(result, cache_path)
    return result.to(device)


def analyze_selectivity(t_latents, o_latents, threshold=0.10):
    """Find selective features: fire more for target than others."""
    t_freq = (t_latents > 1e-5).float().mean(0)
    o_freq = (o_latents > 1e-5).float().mean(0)
    selectivity = t_freq - o_freq

    # Magnitude-weighted selectivity
    t_mag = t_latents.mean(0)
    o_mag = o_latents.mean(0)
    mag_contrast = t_mag / (o_mag + 1e-8)

    winners = torch.where((selectivity >= threshold) & (t_freq > 0.15))[0]

    # Get top features by selectivity
    top_by_sel = selectivity.topk(min(10, len(selectivity))).indices.tolist()
    top_by_mag = mag_contrast.topk(min(10, len(mag_contrast))).indices.tolist()

    return {
        "n_selective": len(winners),
        "winner_features": winners.tolist()[:20],
        "top_by_selectivity": [
            {"feat": int(f), "sel": selectivity[f].item(), "t_freq": t_freq[f].item(), "o_freq": o_freq[f].item()}
            for f in top_by_sel
        ],
        "top_by_magnitude": [
            {"feat": int(f), "mag_ratio": mag_contrast[f].item(), "t_mag": t_mag[f].item(), "o_mag": o_mag[f].item()}
            for f in top_by_mag
        ],
    }


def main():
    cfg = load_config()
    ensure_dirs(cfg)
    device = get_device()

    patch_environment()
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("openai/circuit-sparsity", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "openai/circuit-sparsity", trust_remote_code=True
    ).to(device).eval()

    df = pd.read_parquet(resolve_path(cfg["paths"]["dataset"]))
    nodes = cfg["sae_params"]["target_nodes"]
    threshold = cfg["sae_params"]["feature_selectivity_threshold"]
    sae_dir = resolve_path(cfg["paths"]["sae_dir"])
    d_model = get_n_embd(model.config)
    latent_cache_dir = data_path("cache/latents")

    # Prepare prompts per node
    node_prompts = {}
    for node in nodes:
        node_prompts[node] = df[df["ast_node"] == node]["prompt_text"].tolist()[:N_PROMPTS]
    all_other_prompts = {}
    for node in nodes:
        others = [n for n in nodes if n != node]
        all_other_prompts[node] = df[df["ast_node"].isin(others)]["prompt_text"].tolist()[:N_PROMPTS]

    results = {}

    for site in SITES:
        print(f"\n{'='*60}")
        print(f"  SITE: {site.upper()}")
        print(f"{'='*60}")

        for layer in LAYERS:
            sae, meta = load_sae_for_site(sae_dir, layer, site, device, d_model)
            if sae is None:
                print(f"  L{layer}/{site}: no SAE found, skipping")
                continue

            r2 = meta.get("r2", "?")
            exp = meta.get("expansion", "?")
            k = meta.get("k", "?")
            print(f"\n  L{layer}/{site} (R²={r2}, exp={exp}x, k={k})")

            layer_results = {}
            total_selective = 0

            for node in nodes:
                t_prompts = node_prompts[node]
                o_prompts = all_other_prompts[node]
                if not t_prompts or not o_prompts:
                    layer_results[node] = {"n_selective": 0, "skipped": True}
                    continue

                t_latents = harvest_latents(
                    model, tokenizer, sae, device, layer, site,
                    t_prompts, latent_cache_dir, f"{node}_target"
                )
                o_latents = harvest_latents(
                    model, tokenizer, sae, device, layer, site,
                    o_prompts, latent_cache_dir, f"{node}_other"
                )

                analysis = analyze_selectivity(t_latents, o_latents, threshold)
                layer_results[node] = analysis
                total_selective += analysis["n_selective"]

                if analysis["n_selective"] > 0:
                    top = analysis["top_by_selectivity"][0]
                    print(f"    {node:<20} {analysis['n_selective']:>4} selective  (best: feat {top['feat']}, sel={top['sel']:.3f})")

            results[f"L{layer}_{site}"] = {
                "meta": meta,
                "total_selective": total_selective,
                "nodes": layer_results,
            }
            print(f"  L{layer}/{site} total: {total_selective} selective features")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("  HOURGLASS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Layer':<8} {'MLP':<12} {'Resid':<12}")
    print("-" * 32)
    for layer in LAYERS:
        mlp_total = results.get(f"L{layer}_mlp", {}).get("total_selective", "-")
        resid_total = results.get(f"L{layer}_resid", {}).get("total_selective", "-")
        print(f"L{layer:<7} {str(mlp_total):<12} {str(resid_total):<12}")

    # Save
    out_path = data_path("results/multilayer_multisite_discovery.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Convert meta to serializable
    for key in results:
        if "meta" in results[key]:
            results[key]["meta"] = {k: str(v) for k, v in results[key]["meta"].items()}

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
