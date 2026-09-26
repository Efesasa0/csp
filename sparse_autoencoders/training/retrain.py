"""
Retrain SAEs across all 8 layers with improved settings:
- 8k harvest (up from 2k) — uses all available data
- Cosine LR schedule with warmup
- 100 epochs
- Trains on BOTH MLP outputs and residual stream (post-block)
- Caches activations per layer+site so harvesting only happens once
- Saves R² in checkpoint metadata
"""
import os, sys, json, time, math, torch, torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import (
    load_config, TopKSAE, get_device, get_layers,
    resolve_path, ensure_dirs, get_path, get_n_embd, patch_environment
)

# ── Config ──────────────────────────────────────────────────────────────
HARVEST_N = 8000       # up from 2000
LAYERS    = list(range(8))
SITES     = ["mlp", "resid"]  # train SAEs on both activation sites

# Site-specific hyperparams: resid needs more capacity (richer signal)
SITE_PARAMS = {
    "mlp": {
        "epochs": 100,
        "warmup": 10,
        "batch": 256,
        "lr": 3e-4,
        "expansion": 8,     # 8x = 16,384 hidden
        "k": 32,
    },
    "resid": {
        "epochs": 150,       # more epochs — harder signal
        "warmup": 15,
        "batch": 256,
        "lr": 3e-4,
        "expansion": 16,     # 16x = 32,768 hidden — more capacity for richer signal
        "k": 64,             # less sparse — residual stream is less sparse than MLP
    },
}


def harvest_all_sites(model, tokenizer, device, df, layer, cache_dir, n_harvest):
    """Harvest both MLP and residual stream activations in a SINGLE forward pass.
    Returns dict: {"mlp": tensor, "resid": tensor}. Each cached separately."""
    results = {}
    all_cached = True

    for site in SITES:
        cache_path = os.path.join(cache_dir, f"acts_layer_{layer}_{site}_8k.pt")
        # Also check legacy name (no site suffix) for MLP
        legacy_path = os.path.join(cache_dir, f"acts_layer_{layer}_8k.pt")
        if os.path.exists(cache_path):
            print(f"  [cache hit] L{layer}/{site} from {cache_path}")
            results[site] = torch.load(cache_path, map_location=device, weights_only=False)
        elif site == "mlp" and os.path.exists(legacy_path):
            print(f"  [cache hit] L{layer}/{site} from legacy {legacy_path}")
            results[site] = torch.load(legacy_path, map_location=device, weights_only=False)
        else:
            all_cached = False

    if all_cached:
        return results

    # Need to harvest — do both sites in one pass
    print(f"  [harvesting] L{layer} (mlp + resid): {n_harvest} prompts...")
    layers_list = get_layers(model)
    block = layers_list[layer]

    mlp_acts = []
    resid_acts = []

    # Hook MLP output (before residual add)
    mlp_hook = block.mlp.register_forward_hook(
        lambda m, i, o: mlp_acts.append(
            (o[0] if isinstance(o, tuple) else o)[:, -1, :].detach().cpu()
        )
    )

    # Hook block output (full residual stream after attention + MLP)
    resid_hook = block.register_forward_hook(
        lambda m, i, o: resid_acts.append(
            (o[0] if isinstance(o, tuple) else o)[:, -1, :].detach().cpu()
        )
    )

    prompts = df["prompt_text"].tolist()[:n_harvest]
    for p in tqdm(prompts, desc=f"  Harvest L{layer}", leave=False):
        with torch.no_grad():
            model(input_ids=tokenizer(p, return_tensors="pt").input_ids.to(device))

    mlp_hook.remove()
    resid_hook.remove()

    # Save both
    for site, acts_list in [("mlp", mlp_acts), ("resid", resid_acts)]:
        if site not in results:
            X = torch.cat(acts_list, dim=0)
            cache_path = os.path.join(cache_dir, f"acts_layer_{layer}_{site}_8k.pt")
            os.makedirs(cache_dir, exist_ok=True)
            torch.save(X, cache_path)
            print(f"  [saved] L{layer}/{site}: {X.shape[0]} x {X.shape[1]} -> {cache_path}")
            results[site] = X.to(device)

    return results


def train_sae(X_all, device, site, label):
    """Train SAE with cosine LR schedule + warmup. Uses site-specific hyperparams."""
    sp = SITE_PARAMS[site]
    n_input = X_all.shape[1]
    expansion = sp["expansion"]
    k = sp["k"]
    epochs = sp["epochs"]
    warmup = sp["warmup"]
    lr = sp["lr"]
    batch = sp["batch"]

    print(f"  Config: expansion={expansion}x, k={k}, epochs={epochs}, lr={lr}")

    sae = TopKSAE(n_input, n_input * expansion, k=k).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    # Cosine schedule with linear warmup
    def lr_lambda(epoch):
        if epoch < warmup:
            return epoch / warmup
        progress = (epoch - warmup) / (epochs - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    loader = DataLoader(
        TensorDataset(X_all), batch_size=batch, shuffle=True
    )

    best_r2 = -999
    best_state = None

    for epoch in range(epochs):
        sae.train()
        total_mse = 0
        for (x_batch,) in loader:
            optimizer.zero_grad()
            x_hat, _ = sae(x_batch)
            loss = F.mse_loss(x_hat, x_batch)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                sae.decoder.weight.data /= (
                    sae.decoder.weight.data.norm(dim=0, keepdim=True) + 1e-8
                )
            total_mse += loss.item()

        scheduler.step()

        # R² every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            sae.eval()
            with torch.no_grad():
                sample = X_all[:2000]
                x_hat_val, _ = sae(sample)
                var = (sample - sample.mean(0)).pow(2).sum()
                err = (sample - x_hat_val).pow(2).sum()
                r2 = (1 - err / var).item()

            lr_now = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch+1:3d}/{epochs} | MSE: {total_mse/len(loader):.6f} | R²: {r2:.4f} | LR: {lr_now:.2e}")

            if r2 > best_r2:
                best_r2 = r2
                best_state = {kk: v.cpu().clone() for kk, v in sae.state_dict().items()}

    return sae, best_r2, best_state


def main():
    cfg = load_config()
    ensure_dirs(cfg)
    device = get_device()

    # Load model ONCE
    from transformers import AutoTokenizer, AutoModelForCausalLM
    patch_environment()
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("openai/circuit-sparsity", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "openai/circuit-sparsity", trust_remote_code=True
    ).to(device).eval()

    df = pd.read_parquet(resolve_path(cfg["paths"]["dataset"]))
    print(f"Dataset: {len(df)} prompts, harvesting {HARVEST_N}")
    print(f"Sites: {SITES}")

    cache_dir = resolve_path(cfg["paths"]["dirs"]["cache"])

    # ── Phase 1: Harvest all activations (cached per layer+site) ──
    print("\n=== Phase 1: Harvest activations ===")
    all_acts = {}  # (layer, site) -> tensor
    for layer in LAYERS:
        site_acts = harvest_all_sites(
            model, tokenizer, device, df, layer, cache_dir, HARVEST_N
        )
        for site, X in site_acts.items():
            if X.device != device:
                X = X.to(device)
            all_acts[(layer, site)] = X

    # Free model memory after harvesting
    del model, tokenizer
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"Harvested {len(all_acts)} activation sets")

    # ── Phase 2: Train SAEs for each (layer, site) ──
    print("\n=== Phase 2: Train SAEs ===")
    results = {}
    sae_dir = resolve_path(cfg["paths"]["sae_dir"])

    for site in SITES:
        for layer in LAYERS:
            key = (layer, site)

            # Check if weights already exist with good R² (skip retraining)
            if site == "mlp":
                filename = cfg["paths"]["sae_file_template"].format(layer=layer)
            else:
                filename = f"sae_layer_{layer}_{site}.pt"

            sae_path = os.path.join(sae_dir, filename)

            if os.path.exists(sae_path):
                try:
                    existing = torch.load(sae_path, map_location="cpu", weights_only=False)
                    if "__meta__" in existing and existing["__meta__"].get("harvest_n", 0) >= HARVEST_N:
                        r2 = existing["__meta__"]["r2"]
                        results[f"L{layer}_{site}"] = {"r2": r2, "time_s": 0, "site": site, "layer": layer, "skipped": True}
                        print(f"\n--- Layer {layer} / {site} --- [SKIP] existing R²={r2:.4f} (already trained with {HARVEST_N}+ samples)")
                        # Free activations we won't use
                        all_acts[key] = None
                        del existing
                        continue
                    del existing
                except Exception:
                    pass

            t0 = time.time()
            print(f"\n--- Layer {layer} / {site} ---")

            X = all_acts[key]
            sp = SITE_PARAMS[site]
            sae, r2, best_state = train_sae(X, device, site, f"L{layer}/{site}")

            os.makedirs(sae_dir, exist_ok=True)

            save_dict = best_state.copy()
            save_dict["__meta__"] = {
                "r2": r2,
                "site": site,
                "layer": layer,
                "harvest_n": HARVEST_N,
                "epochs": sp["epochs"],
                "lr": sp["lr"],
                "warmup": sp["warmup"],
                "k": sp["k"],
                "expansion": sp["expansion"],
            }
            torch.save(save_dict, sae_path)

            elapsed = time.time() - t0
            results[f"L{layer}_{site}"] = {"r2": r2, "time_s": elapsed, "site": site, "layer": layer}
            print(f"  -> L{layer}/{site}: R²={r2:.4f} ({elapsed:.0f}s) -> {sae_path}")

            # Free activations after training
            del X
            all_acts[key] = None

    # ── Summary ──
    print("\n=== RESULTS ===")
    print(f"{'Layer':<8} {'Site':<8} {'R²':<10} {'Time'}")
    print("-" * 36)
    for site in SITES:
        r2_vals = []
        for layer in LAYERS:
            k = f"L{layer}_{site}"
            r = results[k]
            r2_vals.append(r["r2"])
            print(f"L{layer:<7} {site:<8} {r['r2']:<10.4f} {r['time_s']:.0f}s")
        print(f"{'Avg':<8} {site:<8} {sum(r2_vals)/len(r2_vals):<10.4f}")
        print("-" * 36)

    # Save summary
    summary_path = get_path(cfg, "results", "retrain_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
