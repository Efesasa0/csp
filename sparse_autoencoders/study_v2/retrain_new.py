"""
Phase 3 — Retrain SAEs on the NEW-DATASET activations (constructor_stubs).

Uses the same per-site architecture as the existing SAEs for a fair
comparison (mlp: exp=8, k=32; resid: exp=16, k=64). Trains on the full
14,683-row harvest at each (layer, site).

Outputs go to data/study_v2/saes/ — the existing data/saes/ files
are NEVER touched.

Files:
  data/study_v2/saes/sae_layer_{L}.pt         (mlp)
  data/study_v2/saes/sae_layer_{L}_resid.pt   (resid)
  data/study_v2/saes/train_summary.json
"""
import os, sys, time, math, json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import resolve_path, data_path, get_device, TopKSAE

CACHE = data_path("study_v2/cache")
SAE_DIR = data_path("study_v2/saes")

SITE_PARAMS = {
    "mlp":   {"epochs": 100, "warmup": 10, "batch": 256, "lr": 3e-4,
               "expansion": 8,  "k": 32},
    "resid": {"epochs": 150, "warmup": 15, "batch": 256, "lr": 3e-4,
               "expansion": 16, "k": 64},
}


def train(X_all, device, site):
    sp = SITE_PARAMS[site]
    d = X_all.shape[1]
    sae = TopKSAE(n_input=d, n_hidden=d * sp["expansion"], k=sp["k"]).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=sp["lr"])

    def lr_lambda(ep):
        if ep < sp["warmup"]:
            return ep / sp["warmup"]
        p = (ep - sp["warmup"]) / (sp["epochs"] - sp["warmup"])
        return 0.5 * (1 + math.cos(math.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    loader = DataLoader(TensorDataset(X_all), batch_size=sp["batch"], shuffle=True)

    best_r2, best_state = -1e9, None
    for ep in range(sp["epochs"]):
        sae.train()
        tot = 0.0
        for (xb,) in loader:
            opt.zero_grad()
            xhat, _ = sae(xb)
            loss = F.mse_loss(xhat, xb)
            loss.backward()
            opt.step()
            with torch.no_grad():
                sae.decoder.weight.data /= (
                    sae.decoder.weight.data.norm(dim=0, keepdim=True) + 1e-8
                )
            tot += loss.item()
        sched.step()

        if (ep + 1) % 10 == 0 or ep == sp["epochs"] - 1:
            sae.eval()
            with torch.no_grad():
                sample = X_all[:2000]
                xhat, _ = sae(sample)
                var = (sample - sample.mean(0)).pow(2).sum()
                err = (sample - xhat).pow(2).sum()
                r2 = (1 - err / var).item()
            lr_now = sched.get_last_lr()[0]
            print(f"  ep {ep+1:3d}/{sp['epochs']}  mse={tot/len(loader):.6f}  R²={r2:.4f}  lr={lr_now:.2e}")
            if r2 > best_r2:
                best_r2 = r2
                best_state = {kk: v.cpu().clone() for kk, v in sae.state_dict().items()}
    return best_r2, best_state, sp


def sae_path(layer, site):
    if site == "mlp":
        return os.path.join(SAE_DIR, f"sae_layer_{layer}.pt")
    return os.path.join(SAE_DIR, f"sae_layer_{layer}_{site}.pt")


def main():
    device = get_device()
    os.makedirs(SAE_DIR, exist_ok=True)
    results = {}

    for site in ["mlp", "resid"]:
        for layer in range(8):
            out = sae_path(layer, site)
            if os.path.exists(out):
                print(f"[skip] exists: {out}")
                continue

            cache_path = os.path.join(CACHE, f"acts_new_L{layer}_{site}.pt")
            X = torch.load(cache_path, map_location=device, weights_only=False).to(device)
            print(f"\n--- L{layer}/{site}  X={tuple(X.shape)} ---")
            t0 = time.time()
            r2, state, sp = train(X, device, site)
            elapsed = time.time() - t0

            state["__meta__"] = {
                "r2": r2, "site": site, "layer": layer,
                "dataset": "constructor_stubs",
                "harvest_n": X.shape[0],
                "epochs": sp["epochs"], "lr": sp["lr"], "warmup": sp["warmup"],
                "k": sp["k"], "expansion": sp["expansion"],
            }
            torch.save(state, out)
            results[f"L{layer}_{site}"] = {"r2": r2, "time_s": elapsed}
            print(f"  -> {out}  R²={r2:.4f}  ({elapsed:.0f}s)")

            del X
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()

    with open(os.path.join(SAE_DIR, "train_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nDONE.")


if __name__ == "__main__":
    main()
