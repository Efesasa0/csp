"""
Phase 2 — Per-(layer, site) factorial specificity study.

Inputs: acts_{new,old}_L{L}_{site}.pt + metadata_*.parquet + splits.json
        SAE weights from data/saes/ (old-data-trained, unless --sae-dir
        overrides to study_v2/saes for the 'ideal' re-run).

For each (layer, site, target AST node):
  1. Discover top-K candidate features on the NEW-DISCOVERY split by
     target-vs-other explicit mean activation difference.
  2. Evaluate each candidate on NEW-HELD-OUT split:
       - mean on explicit target, other explicit, baseline_ast (same
         node, if any), baseline_builtin, proxy (same node, if any)
       - Mann-Whitney U p-values: target vs other-explicit; explicit vs
         baseline_builtin; proxy vs other-explicit.
       - Cohen's d with 1,000-bootstrap 95% CI for target vs other-explicit.
       - Classification via confound-controlled rules.
  3. Cross-dataset replication on OLD: same target-vs-other explicit
     mean comparison; records pass/fail with p-value.
  4. Null distribution: for N_NULL random SAE features, compute the
     same target-vs-other ratio on held-out explicit — used to compute
     p-value vs. random feature background.

Outputs:
  data/study_v2/reports/per_layer/L{L}_{site}.json
  data/study_v2/reports/summary.json     (aggregated across all layers)

CLI:
  python run_study.py                                # uses old-trained SAEs
  python run_study.py --sae-dir data/study_v2/saes   # uses new-trained SAEs
  python run_study.py --tag ideal                    # output tag suffix
"""
import os, sys, json, argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import load_config, resolve_path, data_path, get_device, TopKSAE

N_LAYERS = 8
SITES = ["mlp", "resid"]
CACHE = data_path("study_v2/cache")
REPORTS_DIR = data_path("study_v2/reports")
TARGET_NODES = [
    "For", "While", "If", "FunctionDef", "AsyncFunctionDef", "ClassDef",
    "With", "Try", "Lambda", "Return", "Yield", "ListComp", "Assert",
    "AnnAssign", "AugAssign", "BoolOp", "Break", "Compare", "Continue",
    "Delete", "DictComp", "GeneratorExp", "Global", "IfExp", "Import",
    "Raise", "SetComp", "Subscript", "YieldFrom",
]
TOP_K = 10
N_NULL = 500
BOOTSTRAP = 1000
FIRE_FRAC = 0.30
DROP_FRAC = 0.20
RNG = np.random.default_rng(20260413)


def load_sae_at(sae_dir, layer, site, d_model, device):
    if site == "mlp":
        path = os.path.join(sae_dir, f"sae_layer_{layer}.pt")
    else:
        path = os.path.join(sae_dir, f"sae_layer_{layer}_{site}.pt")
    state = torch.load(path, map_location=device, weights_only=False)
    meta = state.pop("__meta__", {})
    expansion = meta.get("expansion", 16 if site == "resid" else 8)
    k = meta.get("k", 64 if site == "resid" else 32)
    sae = TopKSAE(n_input=d_model, n_hidden=d_model * expansion, k=k).to(device)
    sae.load_state_dict(state)
    sae.eval()
    return sae, meta, path


def encode_batches(X, sae, device, batch=512):
    outs = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch):
            _, z = sae(X[i:i+batch].to(device))
            outs.append(z.cpu())
    return torch.cat(outs, dim=0)


def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or len(b) < 2:
        return None
    va, vb = a.var(ddof=1), b.var(ddof=1)
    sp = np.sqrt(((len(a)-1)*va + (len(b)-1)*vb) / (len(a)+len(b)-2) + 1e-12)
    return float((a.mean() - b.mean()) / (sp + 1e-12))


def bootstrap_d_ci(a, b, n=BOOTSTRAP, alpha=0.05):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or len(b) < 2:
        return None, None
    ds = np.empty(n)
    for i in range(n):
        ra = RNG.choice(a, size=len(a), replace=True)
        rb = RNG.choice(b, size=len(b), replace=True)
        ds[i] = cohens_d(ra, rb) or 0.0
    lo, hi = np.quantile(ds, [alpha/2, 1 - alpha/2])
    return float(lo), float(hi)


def mwu_p(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or len(b) < 2:
        return None
    try:
        return float(mannwhitneyu(a, b, alternative="greater").pvalue)
    except ValueError:
        return None


def classify(explicit_mean, bast, bbi, proxy):
    if explicit_mean <= 1e-6:
        return {"primary": "inactive", "surface": None}
    def r(v):
        return None if v is None else v / explicit_mean
    r_ast, r_bi, r_px = r(bast), r(bbi), r(proxy)
    if r_ast is not None and r_bi is not None:
        af, bf = r_ast >= FIRE_FRAC, r_bi >= FIRE_FRAC
        ad, bd = r_ast <= DROP_FRAC, r_bi <= DROP_FRAC
        if af and bd: primary = "AST-specific"
        elif bf and ad: primary = "Builtin-specific"
        elif af and bf: primary = "Both-respond"
        elif ad and bd: primary = "Joint-required"
        else: primary = "Ambiguous"
    elif r_ast is not None:
        primary = "AST-specific" if r_ast >= FIRE_FRAC else "Joint-required"
    elif r_bi is not None:
        primary = "Builtin-specific" if r_bi >= FIRE_FRAC else "Joint-required"
    else:
        primary = "Unknown"
    surface = None
    if r_px is not None:
        surface = "Concept" if r_px >= FIRE_FRAC else "Surface-token"
    return {"primary": primary, "surface": surface,
            "r_baseline_ast": r_ast, "r_baseline_builtin": r_bi, "r_proxy": r_px}


def run_one(layer, site, sae_dir, meta_new, meta_old, splits):
    device = get_device()
    Xn = torch.load(os.path.join(CACHE, f"acts_new_L{layer}_{site}.pt"),
                    map_location="cpu", weights_only=False)
    Xo = torch.load(os.path.join(CACHE, f"acts_old_L{layer}_{site}.pt"),
                    map_location="cpu", weights_only=False)

    sae, meta, path = load_sae_at(sae_dir, layer, site, d_model=Xn.shape[1], device=device)
    print(f"  [SAE] {path}  R²={meta.get('r2','?')}")
    Zn = encode_batches(Xn, sae, device).numpy()        # [N_new, H]
    Zo = encode_batches(Xo, sae, device).numpy()        # [N_old, H]
    H = Zn.shape[1]

    disc = np.array(splits["discovery"])
    held = np.array(splits["held_out"])

    vtype_n = meta_new["variant_type"].values
    astn_n  = meta_new["ast_node"].values
    astn_o  = meta_old["ast_node"].values

    disc_expl = disc[vtype_n[disc] == "explicit"]
    held_expl = held[vtype_n[held] == "explicit"]

    out = {}

    for node in TARGET_NODES:
        tgt_disc = disc_expl[astn_n[disc_expl] == node]
        oth_disc = disc_expl[astn_n[disc_expl] != node]
        if len(tgt_disc) < 3 or len(oth_disc) < 3:
            continue

        # Top-K candidates by mean-diff on discovery split
        mean_tgt_disc = Zn[tgt_disc].mean(axis=0)
        mean_oth_disc = Zn[oth_disc].mean(axis=0)
        diff = mean_tgt_disc - mean_oth_disc
        top_idx = np.argsort(diff)[-TOP_K:][::-1]

        tgt_held = held_expl[astn_n[held_expl] == node]
        oth_held = held_expl[astn_n[held_expl] != node]
        bast_held = held[(vtype_n[held] == "baseline_ast") & (astn_n[held] == node)]
        bbi_held = held[vtype_n[held] == "baseline_builtin"]
        proxy_held = held[(vtype_n[held] == "proxy") & (astn_n[held] == node)]
        old_tgt = np.where(astn_o == node)[0]
        old_oth = np.where((astn_o != node) & (astn_o != ""))[0]

        candidates = []
        for feat in top_idx:
            feat = int(feat)
            a = Zn[tgt_held, feat]
            b = Zn[oth_held, feat]
            cohen = cohens_d(a, b)
            lo, hi = bootstrap_d_ci(a, b)

            m_expl = float(a.mean()) if len(a) else 0.0
            m_oth = float(b.mean()) if len(b) else 0.0
            m_bast = float(Zn[bast_held, feat].mean()) if len(bast_held) else None
            m_bbi = float(Zn[bbi_held, feat].mean()) if len(bbi_held) else None
            m_proxy = float(Zn[proxy_held, feat].mean()) if len(proxy_held) else None

            p_vs_other = mwu_p(a, b)
            p_expl_vs_bbi = (mwu_p(a, Zn[bbi_held, feat]) if len(bbi_held) else None)

            # Cross-dataset replication on old
            a_old = Zo[old_tgt, feat]; b_old = Zo[old_oth, feat]
            m_old_tgt = float(a_old.mean()) if len(a_old) else None
            m_old_oth = float(b_old.mean()) if len(b_old) else None
            p_old = mwu_p(a_old, b_old) if len(a_old) and len(b_old) else None

            cls = classify(m_expl, m_bast, m_bbi, m_proxy)

            candidates.append({
                "feature": feat,
                "discovery_diff": float(diff[feat]),
                "held_out": {
                    "n_target": int(len(tgt_held)),
                    "n_other":  int(len(oth_held)),
                    "n_baseline_ast":     int(len(bast_held)),
                    "n_baseline_builtin": int(len(bbi_held)),
                    "n_proxy":            int(len(proxy_held)),
                    "mean_target_expl": m_expl,
                    "mean_other_expl":  m_oth,
                    "mean_baseline_ast":     m_bast,
                    "mean_baseline_builtin": m_bbi,
                    "mean_proxy":            m_proxy,
                    "cohens_d_vs_other": cohen,
                    "cohens_d_ci95":    [lo, hi],
                    "p_target_vs_other":   p_vs_other,
                    "p_expl_vs_baseline_builtin": p_expl_vs_bbi,
                },
                "old_dataset": {
                    "mean_target": m_old_tgt, "mean_other": m_old_oth,
                    "p_target_vs_other": p_old,
                    "replicates": (p_old is not None and p_old < 0.05
                                   and m_old_tgt is not None and m_old_oth is not None
                                   and m_old_tgt > m_old_oth),
                },
                "classification": cls,
            })

        # Null distribution: specificity score for N_NULL random features
        null_feats = RNG.choice(H, size=N_NULL, replace=False)
        null_scores = []
        for nf in null_feats:
            a = Zn[tgt_held, nf]; b = Zn[oth_held, nf]
            if len(a) >= 2 and len(b) >= 2 and b.mean() > 1e-8:
                null_scores.append(float(a.mean() / b.mean()))
        null_scores = np.array(null_scores) if null_scores else np.array([1.0])
        null_pct99 = float(np.percentile(null_scores, 99))

        top = candidates[0]
        m_expl, m_oth = top["held_out"]["mean_target_expl"], top["held_out"]["mean_other_expl"]
        ratio = (m_expl / m_oth) if m_oth > 1e-8 else float("inf")
        top_vs_null = float((null_scores < ratio).mean()) if len(null_scores) else None

        out[node] = {
            "candidates": candidates,
            "null": {
                "n_sampled": int(len(null_scores)),
                "pct99_ratio": null_pct99,
                "winner_ratio": ratio,
                "winner_percentile_vs_null": top_vs_null,
            },
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae-dir", default=data_path("saes"))
    ap.add_argument("--tag", default="old_saes",
                    help="label for output files, e.g. 'old_saes' or 'ideal'")
    args = ap.parse_args()

    meta_new = pd.read_parquet(os.path.join(CACHE, "metadata_new.parquet"))
    meta_old = pd.read_parquet(os.path.join(CACHE, "metadata_old.parquet"))
    with open(os.path.join(CACHE, "splits.json")) as f:
        splits = json.load(f)

    all_reports = {}
    os.makedirs(os.path.join(REPORTS_DIR, "per_layer"), exist_ok=True)
    for layer in range(N_LAYERS):
        for site in SITES:
            label = f"L{layer}_{site}"
            print(f"\n=== {label} [sae_dir={args.sae_dir}] ===")
            out = run_one(layer, site, args.sae_dir, meta_new, meta_old, splits)
            fn = os.path.join(REPORTS_DIR, "per_layer", f"{label}_{args.tag}.json")
            with open(fn, "w") as f:
                json.dump(out, f, indent=2)
            print(f"  wrote {fn}")
            all_reports[label] = out

    # Aggregate summary
    summary = {"_meta": {"tag": args.tag, "sae_dir": args.sae_dir,
                          "top_k": TOP_K, "n_null": N_NULL,
                          "bootstrap": BOOTSTRAP}}
    for label, rep in all_reports.items():
        rows = []
        for node, nd in rep.items():
            top = nd["candidates"][0]
            rows.append({
                "node": node,
                "feature": top["feature"],
                "cohens_d": top["held_out"]["cohens_d_vs_other"],
                "p_vs_other": top["held_out"]["p_target_vs_other"],
                "p_vs_bbi":   top["held_out"]["p_expl_vs_baseline_builtin"],
                "classification": top["classification"]["primary"],
                "surface": top["classification"]["surface"],
                "replicates_old": top["old_dataset"]["replicates"],
                "winner_pct_vs_null": nd["null"]["winner_percentile_vs_null"],
            })
        summary[label] = rows
    out_path = os.path.join(REPORTS_DIR, f"summary_{args.tag}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: {out_path}")


if __name__ == "__main__":
    main()
