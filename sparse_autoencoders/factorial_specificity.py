"""
Factorial specificity test for the 'winner' SAE features discovered by
pipeline.py against constructor_stubs.

For each (site, ast_node) winner feature, compare mean activation across
variant types:
  explicit        AST + builtin together (the standard prompt)
  baseline_ast    AST construct with no builtin token
  baseline_builtin  builtin in neutral Assign context (no AST construct)
  proxy           semantic-equivalent rewrite (e.g. len via .__len__())

Classification per feature:
  AST-specific   : fires on explicit & baseline_ast, NOT on baseline_builtin
  Builtin-specific: fires on explicit & baseline_builtin, NOT on baseline_ast
  Joint-required : fires on explicit, drops on both baselines
  Surface-token  : fires on explicit but drops on proxy (tracks token, not concept)
  Robust         : fires on explicit & proxy (concept-level)

Writes:
  data/reports/factorial_specificity.json  (new file)
"""
import os, sys, json
import torch
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import load_config, resolve_path, data_path, get_device, load_sae_from_checkpoint, get_sae_path

LAYER = 7
CACHE_DIR = data_path("cache_constructor")
OUT_PATH = data_path("reports/factorial_specificity.json")

# Thresholds (on top of explicit-mean normalization)
FIRE_FRAC = 0.30   # variant-mean >= 30% of explicit-mean => "fires"
DROP_FRAC = 0.20   # variant-mean <= 20% of explicit-mean => "drops"


def encode_all(X, sae, device, batch=512):
    """Run SAE.encode on X and return latent activations [N, H]."""
    out = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch):
            z = sae(X[i:i+batch].to(device))[1].cpu()
            out.append(z)
    return torch.cat(out, dim=0)


def classify(explicit, baseline_ast, baseline_builtin, proxy):
    """Return a string label from variant-mean ratios."""
    if explicit <= 1e-6:
        return "inactive"
    def ratio(v):
        return v / explicit if v is not None else None
    r_ast = ratio(baseline_ast)
    r_bi = ratio(baseline_builtin)
    r_px = ratio(proxy)

    # Primary AST vs builtin question
    if r_ast is not None and r_bi is not None:
        ast_fires = r_ast >= FIRE_FRAC
        bi_fires = r_bi >= FIRE_FRAC
        ast_drops = r_ast <= DROP_FRAC
        bi_drops = r_bi <= DROP_FRAC
        if ast_fires and bi_drops:
            primary = "AST-specific"
        elif bi_fires and ast_drops:
            primary = "Builtin-specific"
        elif ast_fires and bi_fires:
            primary = "Both-respond"
        elif ast_drops and bi_drops:
            primary = "Joint-required"
        else:
            primary = "Ambiguous"
    elif r_ast is not None:
        primary = "AST-specific" if r_ast >= FIRE_FRAC else "Joint-required"
    elif r_bi is not None:
        primary = "Builtin-specific" if r_bi >= FIRE_FRAC else "Joint-required"
    else:
        primary = "Unknown"

    # Surface-token vs concept
    if r_px is not None:
        surface = "Concept" if r_px >= FIRE_FRAC else "Surface-token"
    else:
        surface = None

    return {"primary": primary, "surface": surface,
            "r_baseline_ast": r_ast, "r_baseline_builtin": r_bi, "r_proxy": r_px}


def main():
    cfg = load_config()
    device = get_device()

    meta = pd.read_parquet(os.path.join(CACHE_DIR, "metadata.parquet"))
    print(f"Loaded metadata: {len(meta)} rows, columns: {list(meta.columns)}")
    print(f"Variant types: {dict(meta['variant_type'].value_counts())}")

    # Load prior pipeline winners so we evaluate the SAME features
    mlp_rep = json.load(open(data_path("reports/circuit_report_L7_mlp.json")))
    res_rep = json.load(open(data_path("reports/circuit_report_L7_resid.json")))

    report = {"_meta": {"layer": LAYER,
                         "fire_frac": FIRE_FRAC, "drop_frac": DROP_FRAC,
                         "dataset": cfg["paths"]["dataset"]}}

    for site, rep, cache_name in [
        ("mlp",   mlp_rep, "acts_L7_mlp.pt"),
        ("resid", res_rep, "acts_L7_resid.pt"),
    ]:
        print(f"\n=== {site} ===")
        X = torch.load(os.path.join(CACHE_DIR, cache_name),
                        map_location="cpu", weights_only=False)
        sae, _ = load_sae_from_checkpoint(
            get_sae_path(cfg, LAYER, site),
            d_model=X.shape[1], device=device)
        # Encode in batches (16384/32768 hidden dims)
        Z = encode_all(X, sae, device)
        print(f"  Z shape: {tuple(Z.shape)}")

        # Precompute per-variant row masks
        explicit_mask = meta["variant_type"].values == "explicit"
        bast_mask     = meta["variant_type"].values == "baseline_ast"
        bbi_mask      = meta["variant_type"].values == "baseline_builtin"
        proxy_mask    = meta["variant_type"].values == "proxy"
        ast_nodes     = meta["ast_node"].values

        site_out = {}
        for node, entry in rep.items():
            if node.startswith("_"):
                continue
            f = int(entry["feature"])

            # Explicit rows whose AST node matches
            expl_node = explicit_mask & (ast_nodes == node)
            if expl_node.sum() == 0:
                continue
            m_expl = float(Z[expl_node, f].mean().item())

            # baseline_ast rows for this node
            ba_node = bast_mask & (ast_nodes == node)
            m_bast = float(Z[ba_node, f].mean().item()) if ba_node.sum() else None

            # baseline_builtin rows: ast_node = '__assign__' (builtin-only, no AST)
            # These are matched by builtin, not AST node. For an AST-construct
            # feature, they should be ~silent regardless of which builtin.
            # Mean over all baseline_builtin rows is the right comparison.
            m_bbi = float(Z[bbi_mask, f].mean().item()) if bbi_mask.sum() else None

            # proxy rows for this node (if any)
            px_node = proxy_mask & (ast_nodes == node)
            m_proxy = float(Z[px_node, f].mean().item()) if px_node.sum() else None

            # Also mean activation on *other* explicit rows (distractors)
            # as a cross-node contrast.
            other_expl = explicit_mask & (ast_nodes != node)
            m_other = float(Z[other_expl, f].mean().item()) if other_expl.sum() else None

            cls = classify(m_expl, m_bast, m_bbi, m_proxy)

            site_out[node] = {
                "feature": f,
                "mean_explicit": m_expl,
                "mean_baseline_ast": m_bast,
                "mean_baseline_builtin": m_bbi,
                "mean_proxy": m_proxy,
                "mean_other_explicit": m_other,
                "n_baseline_ast": int(ba_node.sum()),
                "n_proxy": int(px_node.sum()),
                "classification": cls,
            }
            print(f"  {node:18s} f={f:5d}  expl={m_expl:.3f}  "
                  f"bAST={m_bast if m_bast is None else f'{m_bast:.3f}'}  "
                  f"bBI={m_bbi if m_bbi is None else f'{m_bbi:.3f}'}  "
                  f"proxy={m_proxy if m_proxy is None else f'{m_proxy:.3f}'}  "
                  f"-> {cls['primary']}"
                  + (f" / {cls['surface']}" if cls['surface'] else ""))

        report[site] = site_out

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
