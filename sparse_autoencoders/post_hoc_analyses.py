"""Post-hoc analyses 1, 2, 3, 5, 7 — reuse cached activations + SAE checkpoints.

Outputs written to data/posthoc/:
  - feature_census.json        (# dead/rare features per layer, site)
  - feature_density.json        (firing-rate histogram per layer, site)
  - polysemy.json               (AST classes per winning feature)
  - effective_rank.json         (participation ratio + rank stats)
  - pearson_r_site_layer.json   (correlation per site where we have data)
"""
import os, sys, json, torch, numpy as np, pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import TopKSAE, load_config, resolve_path, data_path

cfg = load_config()
CACHE_DIR = resolve_path(cfg["paths"]["dirs"]["cache"])
SAE_DIR   = resolve_path(cfg["paths"]["sae_dir"])
OUT_DIR   = os.path.join(data_path(), "posthoc")
os.makedirs(OUT_DIR, exist_ok=True)

LAYERS = list(range(8))
SITES  = ["mlp", "resid"]
DEVICE = torch.device("cpu")  # analysis only, no training

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def load_sae(layer, site):
    fn = f"sae_layer_{layer}.pt" if site == "mlp" else f"sae_layer_{layer}_{site}.pt"
    ckpt = torch.load(os.path.join(SAE_DIR, fn), map_location="cpu", weights_only=False)
    meta = ckpt.pop("__meta__", {})
    expansion = meta.get("expansion", 8 if site == "mlp" else 16)
    k = meta.get("k", 32 if site == "mlp" else 64)
    # infer d from state dict
    w_enc = ckpt["encoder.weight"]     # (n_hidden, d)
    n_hidden, d = w_enc.shape
    sae = TopKSAE(n_input=d, n_hidden=n_hidden, k=k)
    sae.load_state_dict(ckpt)
    sae.eval()
    return sae, meta, d, n_hidden, k

def load_acts(layer, site):
    if site == "mlp":
        p = os.path.join(CACHE_DIR, f"acts_layer_{layer}_8k.pt")
    else:
        p = os.path.join(CACHE_DIR, f"acts_layer_{layer}_{site}_8k.pt")
    return torch.load(p, map_location="cpu", weights_only=False)

def get_latents(sae, X, batch=512):
    """Apply SAE forward in batches; return (N, n_hidden) latents on CPU."""
    outs = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch):
            _, z = sae(X[i:i+batch])
            outs.append(z.cpu())
    return torch.cat(outs, dim=0)

# ──────────────────────────────────────────────────────────────
# Shared per-site precompute: latents + firing stats
# ──────────────────────────────────────────────────────────────
print("Loading stubs + labels for polysemy analysis...")
df = pd.read_parquet(resolve_path(cfg["paths"]["dataset"]))
# truncate to 8k to match cache
df = df.iloc[:8000].reset_index(drop=True)
print(f"  {len(df)} prompts, cols: {df.columns.tolist()[:8]}")
# Label column
label_col = None
for c in ["target_node", "ast_node", "node", "label", "class"]:
    if c in df.columns:
        label_col = c; break
print(f"  label col: {label_col}")
labels = df[label_col].tolist() if label_col else None
unique_labels = sorted(set(labels)) if labels else []
print(f"  {len(unique_labels)} unique AST labels" if labels else "  (no labels)")

census = {}
density = {}
polysemy = {}
effective_rank = {}
TAU = 1e-5

for layer in LAYERS:
    for site in SITES:
        tag = f"L{layer}_{site}"
        print(f"\n── {tag} ──")
        sae, meta, d, n_hidden, K = load_sae(layer, site)
        X = load_acts(layer, site)
        if X.shape[0] > 8000: X = X[:8000]
        print(f"  acts {tuple(X.shape)}  | sae n_hidden={n_hidden} K={K}")
        Z = get_latents(sae, X)  # (N, H)
        firing = (Z > TAU).float()  # (N, H) binary
        firing_rates = firing.mean(0).numpy()  # per-feature rate
        N, H = Z.shape

        # 1. Dead / rare feature census
        census[tag] = {
            "n_hidden": H,
            "dead":    int((firing_rates == 0).sum()),
            "below_1pct": int((firing_rates < 0.01).sum()),
            "below_5pct": int((firing_rates < 0.05).sum()),
            "above_50pct": int((firing_rates > 0.50).sum()),
            "mean_rate": float(firing_rates.mean()),
            "median_rate": float(np.median(firing_rates)),
            "K": K,
        }

        # 2. Density histogram (log-bins)
        bins = [0, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.25, 0.5, 1.0]
        hist, _ = np.histogram(firing_rates, bins=bins)
        density[tag] = {
            "bins": bins,
            "counts": hist.tolist(),
            "total": int(H),
        }

        # 3. Polysemy: only for features that fire on >= 15% of SOME class
        if labels is not None:
            # build per-class firing rate matrix (C, H)
            mat = np.zeros((len(unique_labels), H), dtype=np.float32)
            for ci, lbl in enumerate(unique_labels):
                mask = np.array([l == lbl for l in labels])
                if mask.sum() == 0: continue
                mat[ci] = firing[mask].numpy().mean(0)
            # For each feature, count classes where fires > 15%
            classes_over_15 = (mat > 0.15).sum(0)  # shape (H,)
            # Focus on "non-dead, meaningful" features — firing rate > 1%
            active_mask = firing_rates > 0.01
            active_classes = classes_over_15[active_mask]
            polysemy[tag] = {
                "n_active_features": int(active_mask.sum()),
                "mean_classes_over_15pct": float(active_classes.mean()) if active_mask.sum() else 0,
                "median_classes_over_15pct": float(np.median(active_classes)) if active_mask.sum() else 0,
                "frac_monosemantic":  float(np.mean(active_classes == 1)) if active_mask.sum() else 0,
                "frac_bisemantic":    float(np.mean(active_classes == 2)) if active_mask.sum() else 0,
                "frac_trisemantic":   float(np.mean(active_classes == 3)) if active_mask.sum() else 0,
                "frac_polysemantic":  float(np.mean(active_classes >= 4)) if active_mask.sum() else 0,
                "hist_classes": np.bincount(active_classes, minlength=len(unique_labels)+1).tolist(),
            }

        # 5. Effective rank / participation ratio of latent activations
        # PR = (sum σ_i)^2 / sum (σ_i^2)
        # On Z matrix columns (features over samples). If Z is very sparse, we
        # compute on the centred Z.
        Zc = Z - Z.mean(0, keepdim=True)
        # Covariance eigenvalues via SVD on Z (more stable than full cov)
        # subsample to 2000 for speed
        idx = torch.randperm(N)[:2000]
        Zs = Zc[idx].numpy().astype(np.float32)
        # singular values
        try:
            s = np.linalg.svd(Zs, compute_uv=False)
            ev = s**2
            pr = (ev.sum()**2) / (ev**2).sum()
        except np.linalg.LinAlgError:
            pr = float('nan')
        # Also on raw activations X for reference
        Xc = (X - X.mean(0)).numpy().astype(np.float32)
        idx2 = np.random.choice(X.shape[0], size=min(2000, X.shape[0]), replace=False)
        sx = np.linalg.svd(Xc[idx2], compute_uv=False)
        evx = sx**2
        prx = (evx.sum()**2) / (evx**2).sum()
        effective_rank[tag] = {
            "latent_participation_ratio": float(pr),
            "raw_participation_ratio": float(prx),
            "d_input": int(d),
            "n_hidden": int(H),
            "K": int(K),
        }

# ──────────────────────────────────────────────────────────────
# 7. Pearson r per (layer, site) where we have ablation data.
#    We have master_circuit_report.json for L7_resid and circuit_report_L7_mlp.json.
# ──────────────────────────────────────────────────────────────
def pearson(x, y):
    x = np.array(x); y = np.array(y)
    return float(np.corrcoef(x, y)[0, 1])

adv2 = json.load(open(data_path("results/advanced_analysis_2.json")))
pearson_r = {}

# Available ablation sources
abl_sources = {
    "L7_resid": data_path("reports", "circuit_report_L7_resid.json"),
    "L7_mlp":   data_path("reports", "circuit_report_L7_mlp.json"),
}
for tag, path in abl_sources.items():
    if not os.path.exists(path):
        continue
    d = json.load(open(path))
    # Normalise: drop non-class keys
    nodes_drops = {}
    for n, v in d.items():
        if n.startswith("_"): continue
        dz = v.get("metrics", {}).get("drop_zero") if isinstance(v, dict) else None
        if dz is not None:
            nodes_drops[n] = abs(dz)
    pc = adv2[f"linear_probe_{tag}"]["raw_activation_probe"]["per_class_accuracy"]
    nodes = [n for n in nodes_drops if n in pc]
    accs = [pc[n] for n in nodes]
    drops = [nodes_drops[n] for n in nodes]
    signed = [d[n]["metrics"]["drop_zero"] for n in nodes]
    pearson_r[tag] = {
        "N": len(nodes),
        "r_abs": pearson(accs, drops),
        "r_signed": pearson(accs, signed),
        "median_drop": float(np.median(drops)),
        "mean_drop": float(np.mean(drops)),
        "nodes": nodes,
    }

# ──────────────────────────────────────────────────────────────
# Write
# ──────────────────────────────────────────────────────────────
with open(f"{OUT_DIR}/feature_census.json", "w") as f:
    json.dump(census, f, indent=2)
with open(f"{OUT_DIR}/feature_density.json", "w") as f:
    json.dump(density, f, indent=2)
with open(f"{OUT_DIR}/polysemy.json", "w") as f:
    json.dump(polysemy, f, indent=2)
with open(f"{OUT_DIR}/effective_rank.json", "w") as f:
    json.dump(effective_rank, f, indent=2)
with open(f"{OUT_DIR}/pearson_r_site_layer.json", "w") as f:
    json.dump(pearson_r, f, indent=2)

print(f"\nWrote 5 JSON files to {OUT_DIR}")
