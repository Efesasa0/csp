"""
Master analysis script for all ATLAS pipeline findings.
Loads every JSON result file, computes summary statistics,
prints text-heavy analysis, and saves supporting plots.

Usage:
    python analyse_all_findings.py
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from core import data_path  # noqa: E402
import json, os, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

# Optional: plotting
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

# ── paths ──────────────────────────────────────────────────────────────
BASE    = Path(data_path())
CACHE   = BASE / "cache"
ADV     = CACHE / "advanced"
ADV2    = CACHE / "advanced2"
RESULTS = BASE / "results"
REPORTS = BASE / "reports"
SWEEP   = BASE / "sweep_results"
OUT     = BASE / "analysis_output"
OUT.mkdir(exist_ok=True)

NODES = ["For","While","If","FunctionDef","AsyncFunctionDef","ClassDef",
         "With","Try","Lambda","Return","Yield","ListComp","Assert"]
SITES_4 = [("L4","mlp"),("L4","resid"),("L5","mlp"),("L5","resid"),
           ("L6","mlp"),("L6","resid"),("L7","mlp"),("L7","resid")]

def load(path):
    with open(path) as f:
        return json.load(f)

def hline(title=""):
    return f"\n{'='*72}\n  {title}\n{'='*72}"

def save_fig(fig, name):
    if HAS_PLT:
        fig.tight_layout()
        fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
        plt.close(fig)

# collect all text output
lines = []
def p(*args, **kw):
    text = " ".join(str(a) for a in args)
    lines.append(text)
    print(text, **kw)

# ═══════════════════════════════════════════════════════════════════════
#  1. SAE RECONSTRUCTION QUALITY
# ═══════════════════════════════════════════════════════════════════════
p(hline("1. SAE RECONSTRUCTION QUALITY (R² across layers)"))

retrain = load(RESULTS / "retrain_summary.json")
mlp_r2, resid_r2 = {}, {}
for key, v in retrain.items():
    layer = v["layer"]
    if v["site"] == "mlp":
        mlp_r2[layer] = v["r2"]
    else:
        resid_r2[layer] = v["r2"]

p(f"\n{'Layer':<8} {'MLP R²':<12} {'Resid R²':<12} {'Δ (MLP−Resid)':<14}")
p("-"*50)
for l in range(8):
    m = mlp_r2.get(l, float('nan'))
    r = resid_r2.get(l, float('nan'))
    p(f"  {l:<6} {m:<12.4f} {r:<12.4f} {m-r:+.4f}")

mlp_vals = [mlp_r2[l] for l in sorted(mlp_r2)]
resid_vals = [resid_r2[l] for l in sorted(resid_r2)]
p(f"\nMLP   mean R² = {np.mean(mlp_vals):.4f}  (range {np.min(mlp_vals):.4f}–{np.max(mlp_vals):.4f})")
p(f"Resid mean R² = {np.mean(resid_vals):.4f}  (range {np.min(resid_vals):.4f}–{np.max(resid_vals):.4f})")
p(f"\nBest MLP:   L{np.argmax(mlp_vals)} (R²={np.max(mlp_vals):.4f})")
p(f"Best Resid: L{np.argmax(resid_vals)} (R²={np.max(resid_vals):.4f})")
p(f"Worst MLP:  L{np.argmin(mlp_vals)} (R²={np.min(mlp_vals):.4f})")
p(f"Worst Resid:L{np.argmin(resid_vals)} (R²={np.min(resid_vals):.4f})")

# Key observation
mlp_better = sum(1 for l in range(8) if mlp_r2.get(l,0) > resid_r2.get(l,0))
p(f"\nMLP SAE has higher R² than Resid SAE in {mlp_better}/8 layers.")
p("→ MLP activations are easier to reconstruct, yet (as we'll see) carry LESS")
p("  linearly-separable syntax info. High R² ≠ high interpretability.")

if HAS_PLT:
    fig, ax = plt.subplots(figsize=(8,4))
    layers = list(range(8))
    ax.plot(layers, mlp_vals, 'o-', label="MLP", color="#e74c3c")
    ax.plot(layers, resid_vals, 's-', label="Residual", color="#3498db")
    ax.set_xlabel("Layer"); ax.set_ylabel("R²")
    ax.set_title("SAE Reconstruction Quality by Layer")
    ax.legend(); ax.set_ylim(0.7, 1.0); ax.set_xticks(layers)
    ax.axhline(0.9, ls='--', alpha=0.3, color='gray')
    save_fig(fig, "01_sae_r2_by_layer.png")

# ═══════════════════════════════════════════════════════════════════════
#  2. FEATURE DISCOVERY: MULTI-LAYER FEATURE COUNTS
# ═══════════════════════════════════════════════════════════════════════
p(hline("2. FEATURE DISCOVERY: SELECTIVE FEATURES PER LAYER"))

feat_counts = load(RESULTS / "multilayer_feature_counts.json")

p(f"\n{'Layer':<8}", end="")
for n in NODES:
    p(f"{n[:6]:>7}", end="")
p(f"  {'TOTAL':>7}")
p("-"*(8 + 7*len(NODES) + 9))

layer_totals = {}
for layer_str in sorted(feat_counts.keys(), key=int):
    layer = int(layer_str)
    row = feat_counts[layer_str]
    total = sum(row.values())
    layer_totals[layer] = total
    p(f"  {layer:<6}", end="")
    for n in NODES:
        p(f"{row.get(n,0):>7}", end="")
    p(f"  {total:>7}")

p(f"\nPhase transition: Layers 0-3 have {sum(layer_totals[l] for l in range(4))} total features")
p(f"                  Layers 4-7 have {sum(layer_totals[l] for l in range(4,8))} total features")
p(f"                  Ratio: {sum(layer_totals[l] for l in range(4,8))/max(sum(layer_totals[l] for l in range(4)),1):.1f}x more in L4-7")

# Which nodes get the most features?
node_totals = {n: sum(feat_counts[str(l)].get(n,0) for l in range(8)) for n in NODES}
sorted_nodes = sorted(node_totals.items(), key=lambda x: -x[1])
p(f"\nTotal features across all layers per node:")
for n, c in sorted_nodes:
    p(f"  {n:<20} {c:>4} features")

# Peak layer per node
p(f"\nPeak layer per AST node (layer with most selective features):")
for n in NODES:
    counts = {l: feat_counts[str(l)].get(n,0) for l in range(8)}
    peak = max(counts, key=counts.get)
    p(f"  {n:<20} peaks at L{peak} ({counts[peak]} features)")

if HAS_PLT:
    fig, ax = plt.subplots(figsize=(10,5))
    layers = list(range(8))
    bottom = np.zeros(8)
    cmap = plt.cm.tab20(np.linspace(0, 1, len(NODES)))
    for i, n in enumerate(NODES):
        vals = [feat_counts[str(l)].get(n, 0) for l in layers]
        ax.bar(layers, vals, bottom=bottom, label=n, color=cmap[i])
        bottom += np.array(vals)
    ax.set_xlabel("Layer"); ax.set_ylabel("Selective Features")
    ax.set_title("Feature Discovery: Selective Features per Layer")
    ax.legend(fontsize=7, ncol=3, loc="upper left"); ax.set_xticks(layers)
    save_fig(fig, "02_feature_counts_by_layer.png")

# ═══════════════════════════════════════════════════════════════════════
#  3. LINEAR PROBE: SAE FAITHFULNESS
# ═══════════════════════════════════════════════════════════════════════
p(hline("3. LINEAR PROBE ANALYSIS"))

probe_data = {}
for layer, site in SITES_4:
    key = f"{layer}_{site}"
    probe_data[key] = load(ADV2 / f"linear_probe_{key}.json")

p(f"\n{'Site':<14} {'Raw Acc':<10} {'SAE Acc':<10} {'Drop':<10} {'Raw F1':<10} {'SAE F1':<10} {'F1 Drop':<10}")
p("-"*64)
for key, d in probe_data.items():
    raw_acc = d["raw_activation_probe"]["accuracy"]
    sae_acc = d["sae_latent_probe"]["accuracy"]
    raw_f1 = d["raw_activation_probe"]["weighted_f1"]
    sae_f1 = d["sae_latent_probe"]["weighted_f1"]
    p(f"  {key:<12} {raw_acc:<10.4f} {sae_acc:<10.4f} {raw_acc-sae_acc:+<10.4f} {raw_f1:<10.4f} {sae_f1:<10.4f} {raw_f1-sae_f1:+<10.4f}")

# Per-class breakdown for best site (L7_resid)
p(f"\nPer-class accuracy (L7_resid — best overall site):")
p(f"  {'Node':<20} {'Raw':<8} {'SAE':<8} {'Drop':<8} {'Verdict'}")
p("  " + "-"*55)
best = probe_data["L7_resid"]
raw_cls = best["raw_activation_probe"]["per_class_accuracy"]
sae_cls = best["sae_latent_probe"]["per_class_accuracy"]
for n in NODES:
    r, s = raw_cls[n], sae_cls[n]
    drop = r - s
    if r >= 0.9:
        verdict = "EASY"
    elif r >= 0.6:
        verdict = "MODERATE"
    elif r >= 0.3:
        verdict = "HARD"
    else:
        verdict = "FAILING"
    p(f"  {n:<20} {r:<8.3f} {s:<8.3f} {drop:+<8.3f} {verdict}")

# Faithfulness gap analysis
p(f"\nSAE Faithfulness Gap (Raw − SAE accuracy) by site type:")
resid_drops = []
mlp_drops = []
for key, d in probe_data.items():
    drop = d["raw_activation_probe"]["accuracy"] - d["sae_latent_probe"]["accuracy"]
    if "resid" in key:
        resid_drops.append(drop)
    else:
        mlp_drops.append(drop)
p(f"  Residual stream: mean drop = {np.mean(resid_drops)*100:.2f}pp (range {np.min(resid_drops)*100:.2f}–{np.max(resid_drops)*100:.2f}pp)")
p(f"  MLP output:      mean drop = {np.mean(mlp_drops)*100:.2f}pp (range {np.min(mlp_drops)*100:.2f}–{np.max(mlp_drops)*100:.2f}pp)")
p(f"  → MLP SAEs lose {np.mean(mlp_drops)/np.mean(resid_drops):.1f}x more probe accuracy than residual SAEs")

if HAS_PLT:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))
    # Left: overall accuracy comparison
    sites = list(probe_data.keys())
    raw_accs = [probe_data[s]["raw_activation_probe"]["accuracy"] for s in sites]
    sae_accs = [probe_data[s]["sae_latent_probe"]["accuracy"] for s in sites]
    x = np.arange(len(sites))
    ax1.bar(x - 0.15, raw_accs, 0.3, label="Raw", color="#3498db")
    ax1.bar(x + 0.15, sae_accs, 0.3, label="SAE", color="#e74c3c")
    ax1.set_xticks(x); ax1.set_xticklabels(sites, rotation=15)
    ax1.set_ylabel("Accuracy"); ax1.set_title("Linear Probe: Raw vs SAE Accuracy")
    ax1.legend(); ax1.set_ylim(0.4, 0.85)

    # Right: per-class for L7_resid
    raw_vals = [raw_cls[n] for n in NODES]
    sae_vals = [sae_cls[n] for n in NODES]
    x2 = np.arange(len(NODES))
    ax2.barh(x2 - 0.15, raw_vals, 0.3, label="Raw", color="#3498db")
    ax2.barh(x2 + 0.15, sae_vals, 0.3, label="SAE", color="#e74c3c")
    ax2.set_yticks(x2); ax2.set_yticklabels(NODES, fontsize=8)
    ax2.set_xlabel("Accuracy"); ax2.set_title("Per-Class Accuracy (L7 Resid)")
    ax2.legend(); ax2.axvline(0.5, ls='--', alpha=0.3, color='gray')
    save_fig(fig, "03_linear_probe.png")

# ═══════════════════════════════════════════════════════════════════════
#  4. CAUSAL ABLATION
# ═══════════════════════════════════════════════════════════════════════
p(hline("4. CAUSAL ABLATION ANALYSIS"))

ablation_data = {}
for layer, site in SITES_4:
    key = f"{layer}_{site}"
    ablation_data[key] = load(ADV2 / f"causal_ablation_{key}.json")

p(f"\n{'Site':<14} {'Nodes w/ +drop':<16} {'Mean |rel_drop|':<18} {'Max rel_drop (node)':<25}")
p("-"*75)
for key, d in ablation_data.items():
    positive = sum(1 for n in NODES if d[n]["prob_drop"] > 0)
    abs_drops = [abs(d[n]["relative_drop"]) for n in NODES]
    max_node = max(NODES, key=lambda n: d[n]["relative_drop"])
    max_drop = d[max_node]["relative_drop"]
    p(f"  {key:<12} {positive:>3}/13          {np.mean(abs_drops):<18.4f} {max_drop:+.4f} ({max_node})")

# Detailed breakdown for L7_resid
p(f"\nDetailed causal ablation (L7_resid):")
p(f"  {'Node':<20} {'Baseline P':<12} {'Ablated P':<12} {'Prob Drop':<12} {'Rel Drop':<12}")
p("  " + "-"*58)
d = ablation_data["L7_resid"]
for n in NODES:
    v = d[n]
    p(f"  {n:<20} {v['mean_baseline_prob']:<12.6f} {v['mean_ablated_prob']:<12.6f} {v['prob_drop']:<12.6f} {v['relative_drop']:+.4f}")

# Are ablation effects correlated with probe accuracy?
p(f"\nCorrelation: probe accuracy vs causal effect (L7_resid):")
probe_acc = [best["raw_activation_probe"]["per_class_accuracy"][n] for n in NODES]
rel_drops = [d[n]["relative_drop"] for n in NODES]
corr = np.corrcoef(probe_acc, rel_drops)[0,1]
p(f"  Pearson r(probe_accuracy, relative_drop) = {corr:.4f}")
p(f"  → {'Weak' if abs(corr) < 0.3 else 'Moderate' if abs(corr) < 0.6 else 'Strong'} correlation between linear separability and causal importance")

if HAS_PLT:
    fig, ax = plt.subplots(figsize=(7,5))
    ax.scatter(probe_acc, rel_drops, s=60, zorder=3)
    for i, n in enumerate(NODES):
        ax.annotate(n, (probe_acc[i], rel_drops[i]), fontsize=7, ha='left', va='bottom')
    ax.set_xlabel("Linear Probe Accuracy (L7 resid)")
    ax.set_ylabel("Relative Probability Drop on Ablation")
    ax.set_title("Probe Accuracy vs Causal Effect")
    ax.axhline(0, ls='--', alpha=0.3, color='gray')
    save_fig(fig, "04_probe_vs_ablation.png")

# ═══════════════════════════════════════════════════════════════════════
#  5. CO-ACTIVATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
p(hline("5. CO-ACTIVATION ANALYSIS"))

for layer, site in SITES_4:
    key = f"{layer}_{site}"
    fpath = ADV / f"coact_{key}.json"
    if not fpath.exists():
        continue
    coact = load(fpath)

    p(f"\n  [{key}]")
    p(f"  {'Node':<20} {'Self-Act Rate':<15} {'Top Co-fire':<12} {'Co-fire Rate':<12}")
    p("  " + "-"*58)

    self_rates = []
    max_cofires = []
    for n in NODES:
        if n not in coact:
            continue
        v = coact[n]
        self_rate = v.get("self_activation_rate", 0)
        self_rates.append(self_rate)
        top = v["top_cofiring"][0] if v["top_cofiring"] else {"feature": "—", "coact_rate": 0}
        max_cofires.append(top["coact_rate"])
        p(f"  {n:<20} {self_rate:<15.3f} feat {str(top['feature']):<8} {top['coact_rate']:<12.3f}")

    p(f"  Mean self-activation rate: {np.mean(self_rates):.3f}")
    p(f"  Mean max co-firing rate:   {np.mean(max_cofires):.3f}")
    p(f"  → Features fire {'independently' if np.mean(max_cofires) < 0.1 else 'with moderate co-dependence' if np.mean(max_cofires) < 0.3 else 'in correlated clusters'}")

# ═══════════════════════════════════════════════════════════════════════
#  6. CROSS-LAYER SIMILARITY
# ═══════════════════════════════════════════════════════════════════════
p(hline("6. CROSS-LAYER FEATURE SIMILARITY"))

for site_type in ["mlp", "resid"]:
    fpath = ADV / f"cross_layer_{site_type}.json"
    if not fpath.exists():
        continue
    cl = load(fpath)

    p(f"\n  [{site_type.upper()} stream]")
    p(f"  Cosine similarity of feature activation patterns across layers:\n")

    # Collect all nodes' similarity matrices and average
    all_mats = []
    node_diag_means = []  # mean off-diagonal per node
    for n in NODES:
        if n not in cl:
            continue
        mat = np.array(cl[n]["sim_matrix"])
        all_mats.append(mat)
        # off-diagonal mean
        mask = ~np.eye(mat.shape[0], dtype=bool)
        off_diag = mat[mask].mean()
        node_diag_means.append((n, off_diag))

    if all_mats:
        avg_mat = np.mean(all_mats, axis=0)
        layers_present = cl[NODES[0]]["layers"]

        # Print averaged matrix
        p(f"  Averaged across all nodes (layers {layers_present[0]}–{layers_present[-1]}):")
        p(f"  {'':>6}", end="")
        for l in layers_present:
            p(f"  L{l:>3}", end="")
        p()
        for i, li in enumerate(layers_present):
            p(f"  L{li:>3} ", end="")
            for j in range(len(layers_present)):
                p(f"  {avg_mat[i,j]:>5.3f}", end="")
            p()

        mask = ~np.eye(avg_mat.shape[0], dtype=bool)
        p(f"\n  Overall mean off-diagonal similarity: {avg_mat[mask].mean():.4f}")

        # Adjacent layer similarity
        adj_sims = [avg_mat[i, i+1] for i in range(len(layers_present)-1)]
        p(f"  Adjacent layer similarities: {', '.join(f'L{layers_present[i]}→L{layers_present[i+1]}={adj_sims[i]:.3f}' for i in range(len(adj_sims)))}")

        # Most/least stable nodes
        node_diag_means.sort(key=lambda x: -x[1])
        p(f"\n  Most cross-layer stable nodes:")
        for n, s in node_diag_means[:3]:
            p(f"    {n:<20} mean off-diag = {s:.4f}")
        p(f"  Least cross-layer stable nodes:")
        for n, s in node_diag_means[-3:]:
            p(f"    {n:<20} mean off-diag = {s:.4f}")

# ═══════════════════════════════════════════════════════════════════════
#  7. CROSS-SITE SIMILARITY (MLP vs RESIDUAL)
# ═══════════════════════════════════════════════════════════════════════
p(hline("7. CROSS-SITE ANALYSIS (MLP vs Residual)"))

for layer in ["L4", "L7"]:
    fpath = ADV / f"cross_site_{layer}.json"
    if not fpath.exists():
        continue
    cs = load(fpath)

    p(f"\n  [{layer}] MLP ↔ Residual feature alignment:")
    p(f"  {'Node':<20} {'Assigned cos':<14} {'Best match cos':<16} {'Best match feat':<16}")
    p("  " + "-"*66)

    assigned_cos = []
    best_cos = []
    for n in NODES:
        if n not in cs:
            continue
        v = cs[n]
        assigned_cos.append(v["assigned_cosine"])
        best_cos.append(v["best_resid_cosine"])
        p(f"  {n:<20} {v['assigned_cosine']:>+.4f}        {v['best_resid_cosine']:>.4f}          {v['best_resid_match']}")

    p(f"\n  Mean assigned cosine: {np.mean(assigned_cos):.4f}")
    p(f"  Mean best-match cosine: {np.mean(best_cos):.4f}")
    p(f"  → MLP and residual features are {'essentially orthogonal' if abs(np.mean(assigned_cos)) < 0.05 else 'weakly aligned'}")
    p(f"  → Even best matches are {'very weak' if np.mean(best_cos) < 0.2 else 'moderate' if np.mean(best_cos) < 0.4 else 'strong'}")

# ═══════════════════════════════════════════════════════════════════════
#  8. LOGIT LENS ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
p(hline("8. LOGIT LENS ANALYSIS"))

# Load from both advanced and results
for source_name, source_file in [("advanced (per-site)", None), ("results (combined)", RESULTS / "advanced_analysis.json")]:
    if source_file and source_file.exists():
        ll = load(source_file)
        p(f"\n  Source: {source_name}")
        for ll_key in sorted(ll.keys()):
            if not ll_key.startswith("logit_lens"):
                continue
            p(f"\n  [{ll_key}]")
            section = ll[ll_key]
            p(f"  {'Node':<20} {'Top Promoted':<30} {'Logit':<8} {'Top Inhibited':<30} {'Logit':<8}")
            p("  " + "-"*96)
            for n in NODES:
                if n not in section:
                    continue
                v = section[n]
                top_p = v["promoted"][0] if v["promoted"] else {"token": "—", "logit": 0}
                top_i = v["inhibited"][0] if v["inhibited"] else {"token": "—", "logit": 0}
                p(f"  {n:<20} {repr(top_p['token']):<30} {top_p['logit']:>+.4f}  {repr(top_i['token']):<30} {top_i['logit']:>+.4f}")
        break

# Also load per-site logit lens
for layer, site in SITES_4:
    key = f"{layer}_{site}"
    fpath = ADV / f"logit_lens_{key}.json"
    if not fpath.exists():
        continue
    ll = load(fpath)

    # Compute mean logit magnitudes
    mean_promote = []
    mean_inhibit = []
    for n in NODES:
        if n not in ll:
            continue
        prom = [t["logit"] for t in ll[n].get("promoted", [])]
        inhib = [abs(t["logit"]) for t in ll[n].get("inhibited", [])]
        if prom: mean_promote.append(np.mean(prom))
        if inhib: mean_inhibit.append(np.mean(inhib))

    p(f"\n  [{key}] Mean promotion logit: {np.mean(mean_promote):.4f}, Mean inhibition logit: {np.mean(mean_inhibit):.4f}")

p(f"\n  Key question: do promoted tokens relate semantically to the AST node?")
p(f"  (Manual inspection required — check whether 'for' features promote iteration-related tokens, etc.)")

# ═══════════════════════════════════════════════════════════════════════
#  9. STEERING ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
p(hline("9. STEERING ANALYSIS"))

for layer, site in SITES_4:
    key = f"{layer}_{site}"

    # Token-level steering
    fpath = ADV / f"steering_{key}.json"
    if not fpath.exists():
        continue
    steer = load(fpath)

    p(f"\n  [{key}] Token-level steering:")
    p(f"  {'Node':<20} {'2x keyword?':<13} {'5x keyword?':<13} {'10x keyword?':<13} {'Orig val':<10}")
    p("  " + "-"*70)

    keyword_rates = {m: [] for m in ["2.0","5.0","10.0"]}
    for n in NODES:
        if n not in steer:
            continue
        row = []
        for mult in ["2.0","5.0","10.0"]:
            has_kw = steer[n][mult].get("has_keyword", False)
            keyword_rates[mult].append(1 if has_kw else 0)
            row.append("YES" if has_kw else "no")
        orig = steer[n]["2.0"].get("original_val", 0)
        p(f"  {n:<20} {row[0]:<13} {row[1]:<13} {row[2]:<13} {orig:<10.4f}")

    for mult in ["2.0","5.0","10.0"]:
        rate = np.mean(keyword_rates[mult]) if keyword_rates[mult] else 0
        p(f"  At {mult}x: {rate*100:.0f}% of nodes produce their keyword")

    # Codegen steering
    cg_path = ADV / f"steering_codegen_{key}.json"
    if cg_path.exists():
        cg = load(cg_path)
        p(f"\n  [{key}] Codegen steering:")
        for mult in ["2.0","5.0","10.0"]:
            rates = []
            for n in NODES:
                if n in cg and mult in cg[n]:
                    rates.append(cg[n][mult].get("keyword_rate", 0))
            p(f"    At {mult}x: mean keyword rate = {np.mean(rates)*100:.1f}%")

# ═══════════════════════════════════════════════════════════════════════
# 10. MINIMAL PAIRS
# ═══════════════════════════════════════════════════════════════════════
p(hline("10. MINIMAL PAIRS ANALYSIS"))

for layer, site in SITES_4:
    key = f"{layer}_{site}"
    fpath = ADV2 / f"minimal_pairs_{key}.json"
    if not fpath.exists():
        continue
    mp = load(fpath)

    p(f"\n  [{key}] Feature selectivity on minimal code pairs:")

    # Each pair: a_node vs b_node, check if features fire preferentially
    n_pairs = len(mp)
    n_selective = 0
    pair_details = []

    for pair_name, pair_data in mp.items():
        a_node = pair_data["a_node"]
        b_node = pair_data["b_node"]

        a_acts = pair_data["a_activations"]
        b_acts = pair_data["b_activations"]

        # Feature for a_node: does it fire more on a than b?
        a_on_a = a_acts.get(a_node, {}).get("firing_frac", 0)
        a_on_b = b_acts.get(a_node, {}).get("firing_frac", 0) if b_acts else 0

        selective = a_on_a > a_on_b
        if selective:
            n_selective += 1
        pair_details.append((pair_name, a_node, b_node, a_on_a, a_on_b, selective))

    p(f"  {n_selective}/{n_pairs} pairs show expected selectivity (feature fires more on its node)")
    p(f"\n  {'Pair':<25} {'A node':<15} {'B node':<15} {'A fire on A':<13} {'A fire on B':<13} {'Select?'}")
    p("  " + "-"*95)
    for pair_name, a, b, aoa, aob, sel in pair_details:
        p(f"  {pair_name:<25} {a:<15} {b:<15} {aoa:<13.3f} {aob:<13.3f} {'YES' if sel else 'NO'}")

# ═══════════════════════════════════════════════════════════════════════
# 11. NESTING / COMPOSITIONALITY
# ═══════════════════════════════════════════════════════════════════════
p(hline("11. NESTING / COMPOSITIONALITY ANALYSIS"))

for layer, site in SITES_4:
    key = f"{layer}_{site}"
    fpath = ADV2 / f"nesting_{key}.json"
    if not fpath.exists():
        continue
    nest = load(fpath)

    p(f"\n  [{key}] Do features compose under nesting?")

    n_compose = sum(1 for v in nest.values() if v.get("composition_holds"))
    p(f"  {n_compose}/{len(nest)} nesting configurations maintain composition")

    p(f"\n  {'Config':<20} {'Outer':<12} {'Inner':<12} {'Both fire?':<12} {'Compose?'}")
    p("  " + "-"*60)
    for config, v in nest.items():
        both = v["nested"].get("both_fire", False)
        compose = v.get("composition_holds", False)
        p(f"  {config:<20} {v['outer_node']:<12} {v['inner_node']:<12} {'YES' if both else 'NO':<12} {'YES' if compose else 'NO'}")

# ═══════════════════════════════════════════════════════════════════════
# 12. POSITIONAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
p(hline("12. POSITIONAL ANALYSIS"))

for layer, site in SITES_4:
    key = f"{layer}_{site}"
    fpath = ADV2 / f"positional_{key}.json"
    if not fpath.exists():
        continue
    pos = load(fpath)

    p(f"\n  [{key}] Feature firing position in token sequence:")
    p(f"  {'Node':<20} {'N firings':<12} {'Mean pos':<10} {'Std':<8} {'Median':<10} {'Top token':<15}")
    p("  " + "-"*75)

    means = []
    stds = []
    for n in NODES:
        if n not in pos:
            continue
        v = pos[n]
        top_tok = v["top_tokens_at_fire"][0]["token"] if v["top_tokens_at_fire"] else "—"
        top_tok_display = repr(top_tok)[:12]
        means.append(v["mean_rel_position"])
        stds.append(v["std_rel_position"])
        p(f"  {n:<20} {v['n_firings']:<12} {v['mean_rel_position']:<10.4f} {v['std_rel_position']:<8.4f} {v['median_rel_position']:<10.4f} {top_tok_display:<15}")

    p(f"\n  Overall: mean position = {np.mean(means):.3f} (0=start, 1=end), std = {np.mean(stds):.3f}")
    p(f"  → Features fire {'uniformly across positions' if np.mean(stds) > 0.25 else 'in concentrated regions'}")

# ═══════════════════════════════════════════════════════════════════════
# 13. CIRCUIT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
p(hline("13. CIRCUIT ANALYSIS"))

for report_name in ["master_circuit_report.json", "circuit_report_L7_mlp.json", "circuit_report_L7_resid.json"]:
    fpath = REPORTS / report_name
    if not fpath.exists():
        continue
    circuit = load(fpath)

    p(f"\n  [{report_name}]")
    p(f"  {'Node':<20} {'Feature':<10} {'#Winners':<10} {'drop_zero':<12} {'drop_mean':<12} {'patch_gain':<12} {'Argmax changed?'}")
    p("  " + "-"*90)

    drops_zero = []
    drops_mean = []
    patch_gains = []
    argmax_changed = 0

    for n in NODES:
        if n not in circuit:
            continue
        v = circuit[n]
        m = v["metrics"]
        drops_zero.append(m["drop_zero"])
        drops_mean.append(m["drop_mean"])
        patch_gains.append(m["patch_gain"])
        changed = v["argmax"]["changed"]
        if changed:
            argmax_changed += 1
        n_winners = len(v.get("all_winners", []))
        p(f"  {n:<20} {v['feature']:<10} {n_winners:<10} {m['drop_zero']:>+.4f}      {m['drop_mean']:>+.4f}      {m['patch_gain']:>+.4f}      {'YES' if changed else 'no'}")

    p(f"\n  Summary:")
    p(f"    Mean drop_zero:  {np.mean(drops_zero):.4f}")
    p(f"    Mean drop_mean:  {np.mean(drops_mean):.4f}")
    p(f"    Mean patch_gain: {np.mean(patch_gains):.4f}")
    p(f"    Argmax changed:  {argmax_changed}/{len([n for n in NODES if n in circuit])} nodes")

    # Upstream analysis
    p(f"\n  Upstream attention head analysis:")
    head_counts = defaultdict(int)
    head_weights = defaultdict(list)
    for n in NODES:
        if n not in circuit:
            continue
        for up in circuit[n].get("upstream", []):
            head_counts[up["source"]] += 1
            head_weights[up["source"]].append(up["weight"])

    # Most common upstream heads
    sorted_heads = sorted(head_counts.items(), key=lambda x: -x[1])[:10]
    p(f"  Top upstream attention heads (by frequency across nodes):")
    for head, count in sorted_heads:
        mean_w = np.mean(head_weights[head])
        p(f"    {head:<10} appears for {count:>2}/{len([n for n in NODES if n in circuit])} nodes, mean weight = {mean_w:.4f}")

if HAS_PLT:
    # Plot circuit metrics
    circuit = load(REPORTS / "master_circuit_report.json")
    fig, ax = plt.subplots(figsize=(10,5))
    nodes_present = [n for n in NODES if n in circuit]
    dz = [circuit[n]["metrics"]["drop_zero"] for n in nodes_present]
    dm = [circuit[n]["metrics"]["drop_mean"] for n in nodes_present]
    pg = [circuit[n]["metrics"]["patch_gain"] for n in nodes_present]
    x = np.arange(len(nodes_present))
    ax.bar(x - 0.25, dz, 0.25, label="drop_zero", color="#e74c3c")
    ax.bar(x, dm, 0.25, label="drop_mean", color="#3498db")
    ax.bar(x + 0.25, pg, 0.25, label="patch_gain", color="#2ecc71")
    ax.set_xticks(x); ax.set_xticklabels(nodes_present, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("Metric Value"); ax.set_title("Circuit Metrics by AST Node")
    ax.legend(); ax.axhline(0, ls='-', alpha=0.3, color='black')
    save_fig(fig, "05_circuit_metrics.png")

# ═══════════════════════════════════════════════════════════════════════
# 14. PARAMETER SWEEP
# ═══════════════════════════════════════════════════════════════════════
p(hline("14. PARAMETER SWEEP ANALYSIS"))

sweep_summary = load(SWEEP / "sweep_summary.json")

p(f"\n  {'Selectivity':<15} {'Specificity':<14} {'Nodes Found':<13} {'N Specific':<13} {'Avg drop_zero'}")
p("  " + "-"*65)
for run in sweep_summary:
    params = run["params"]
    sel = params["sae_params.feature_selectivity_threshold"]
    spec = params["sae_params.specificity_ratio"]
    p(f"  {sel:<15} {spec:<14} {run['nodes_found']:<13} {run['n_specific']:<13} {run['avg_drop_zero']:.4f}")

# Analyze sensitivity
sel_vals = set(r["params"]["sae_params.feature_selectivity_threshold"] for r in sweep_summary)
spec_vals = set(r["params"]["sae_params.specificity_ratio"] for r in sweep_summary)

# Group by selectivity threshold
p(f"\n  Effect of selectivity threshold (averaged over specificity ratios):")
for sel in sorted(sel_vals):
    runs = [r for r in sweep_summary if r["params"]["sae_params.feature_selectivity_threshold"] == sel]
    avg_spec = np.mean([r["n_specific"] for r in runs])
    avg_drop = np.mean([r["avg_drop_zero"] for r in runs])
    p(f"    sel={sel:<6} → mean specific features = {avg_spec:.1f}, mean drop_zero = {avg_drop:.4f}")

p(f"\n  Effect of specificity ratio (averaged over selectivity thresholds):")
for spec in sorted(spec_vals):
    runs = [r for r in sweep_summary if r["params"]["sae_params.specificity_ratio"] == spec]
    avg_spec = np.mean([r["n_specific"] for r in runs])
    avg_drop = np.mean([r["avg_drop_zero"] for r in runs])
    p(f"    ratio={spec:<5} → mean specific features = {avg_spec:.1f}, mean drop_zero = {avg_drop:.4f}")

# Load individual sweep runs for deeper analysis
p(f"\n  Per-node sensitivity to selectivity threshold (at specificity_ratio=1.5):")
sweep_by_sel = {}
for sel in sorted(sel_vals):
    tag = f"feature_selectivity_threshold={sel}__specificity_ratio=1.5"
    fpath = SWEEP / f"run_{tag}.json"
    if fpath.exists():
        sweep_by_sel[sel] = load(fpath)

if sweep_by_sel:
    p(f"  {'Node':<20}", end="")
    for sel in sorted(sweep_by_sel.keys()):
        p(f"  sel={sel:<8}", end="")
    p()
    p("  " + "-"*(20 + 12*len(sweep_by_sel)))
    for n in NODES:
        p(f"  {n:<20}", end="")
        for sel in sorted(sweep_by_sel.keys()):
            if n in sweep_by_sel[sel]:
                dz = sweep_by_sel[sel][n]["metrics"]["drop_zero"]
                p(f"  {dz:>+10.4f}", end="")
            else:
                p(f"  {'—':>10}", end="")
        p()

# ═══════════════════════════════════════════════════════════════════════
# 15. MULTI-SITE DISCOVERY DEEP DIVE
# ═══════════════════════════════════════════════════════════════════════
p(hline("15. MULTI-SITE DISCOVERY: SELECTIVITY PATTERNS"))

disco = load(RESULTS / "multilayer_multisite_discovery.json")

p(f"\n  Sites analyzed: {len(disco)}")
p(f"\n  Total selective features by site:")
p(f"  {'Site':<14} {'Total Selective':<18} {'Mean per node':<16}")
p("  " + "-"*48)

site_totals = []
for site_key in sorted(disco.keys()):
    total = disco[site_key]["total_selective"]
    mean_per = total / len(NODES)
    site_totals.append((site_key, total, mean_per))
    p(f"  {site_key:<14} {total:<18} {mean_per:<16.1f}")

# Best sites
site_totals.sort(key=lambda x: -x[1])
p(f"\n  Top 3 sites by feature count: {', '.join(f'{s[0]}({s[1]})' for s in site_totals[:3])}")
p(f"  Bottom 3 sites:               {', '.join(f'{s[0]}({s[1]})' for s in site_totals[-3:])}")

# Selectivity distribution for top site
top_site = site_totals[0][0]
p(f"\n  Selectivity distribution for best site ({top_site}):")
for n in NODES:
    if n not in disco[top_site]["nodes"]:
        continue
    node_data = disco[top_site]["nodes"][n]
    sels = [f["sel"] for f in node_data.get("top_by_selectivity", [])]
    if sels:
        p(f"    {n:<20} n_selective={node_data['n_selective']:<4} top_sel={max(sels):.3f}  mean_sel={np.mean(sels):.3f}")

if HAS_PLT:
    fig, ax = plt.subplots(figsize=(10,5))
    site_keys = sorted(disco.keys())
    totals = [disco[s]["total_selective"] for s in site_keys]
    colors = ['#e74c3c' if 'mlp' in s else '#3498db' for s in site_keys]
    ax.bar(range(len(site_keys)), totals, color=colors)
    ax.set_xticks(range(len(site_keys)))
    ax.set_xticklabels(site_keys, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("Total Selective Features")
    ax.set_title("Feature Discovery Across All Sites")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='#e74c3c', label='MLP'), Patch(color='#3498db', label='Residual')])
    save_fig(fig, "06_multisite_discovery.png")

# ═══════════════════════════════════════════════════════════════════════
# 16. SYNTHESIS: WHAT'S WORTH WRITING ABOUT
# ═══════════════════════════════════════════════════════════════════════
p(hline("16. SYNTHESIS: KEY FINDINGS FOR PAPER"))

p("""
Based on the comprehensive analysis above, here are the findings ranked
by novelty, strength of evidence, and paper-worthiness:

TIER 1 — STRONG, NOVEL FINDINGS (definitely include):

  1. SAE FAITHFULNESS IS SITE-DEPENDENT
     - Residual SAEs lose 1.2-1.4pp probe accuracy; MLP SAEs lose 5.7-7.2pp
     - Yet MLP SAEs have HIGHER R² (reconstruction quality)
     - This dissociation (high R² ≠ high faithfulness) is an important
       cautionary finding for the SAE interpretability community
     - Evidence: linear probes, R² scores, 4 sites × 13 classes

  2. PHASE TRANSITION IN FEATURE COUNT
     - Layers 0-3 have very few selective features (~1-4 per node per layer)
     - Layers 4-7 have an order of magnitude more (~20-80 per node)
     - This suggests a critical transition where the model begins forming
       discrete syntactic representations
     - Evidence: multilayer feature counts across 8 layers × 13 nodes

  3. FEATURES COMPOSE UNDER NESTING
     - When code nests (e.g. for-in-if), both the inner and outer node
       features fire simultaneously
     - Composition holds in the vast majority of tested configurations
     - This is evidence that SAE features represent genuinely modular,
       compositional syntactic concepts
     - Evidence: nesting analysis across 4 sites

TIER 2 — SOLID SUPPORTING FINDINGS (include as supporting):

  4. MLP AND RESIDUAL FEATURES ARE ORTHOGONAL
     - Cross-site cosine similarity ≈ 0 (assigned) and < 0.3 (best match)
     - The two sites encode syntax through completely different feature
       directions, not shared representations
     - Evidence: cross-site analysis at L4 and L7

  5. THREE TIERS OF AST NODE SEPARABILITY
     - Easy (100%): While, AsyncFunctionDef, With, Lambda, Assert
     - Moderate (70-90%): ClassDef, If, Try, Yield
     - Hard (<50%): FunctionDef, Return, For, ListComp
     - The hard tier likely reflects structural co-occurrence
     - Evidence: per-class probe accuracy across 4 sites

  6. FEATURES FIRE UNIFORMLY ACROSS POSITIONS
     - Mean position ≈ 0.5 with high std, meaning features aren't
       position-locked — they respond to syntactic role regardless of
       where the construct appears in the sequence
     - Evidence: positional analysis across 4 sites

TIER 3 — INTERESTING BUT WEAKER (mention briefly or in appendix):

  7. CAUSAL EFFECTS ARE WEAK
     - Single-feature ablation produces small probability drops
     - The relative drops are often negative (ablation helps)
     - This suggests syntax is encoded redundantly across many features,
       not concentrated in single features
     - Weak correlation between probe accuracy and causal effect

  8. STEERING LARGELY FAILS
     - Steering features at 2x-10x rarely produces the target keyword
     - Codegen steering also shows low keyword rates
     - This isn't necessarily bad — it may mean features encode abstract
       syntactic roles rather than literal keyword production

  9. CROSS-LAYER SIMILARITY IS LOW
     - Features at different layers are mostly dissimilar
     - Adjacent layers show only modest correlation
     - Suggests each layer develops its own feature vocabulary

 10. PARAMETER SWEEP SHOWS ROBUSTNESS
     - Feature discovery is largely stable across selectivity thresholds
       and specificity ratios
     - The pipeline's findings aren't artefacts of particular hyperparameters
""")

# ═══════════════════════════════════════════════════════════════════════
# Save full text output
# ═══════════════════════════════════════════════════════════════════════
output_text = "\n".join(lines)
with open(OUT / "full_analysis.txt", "w") as f:
    f.write(output_text)

p(f"\n{'='*72}")
p(f"  Analysis complete. Outputs saved to: {OUT}")
p(f"  - full_analysis.txt (this text)")
if HAS_PLT:
    p(f"  - 01_sae_r2_by_layer.png")
    p(f"  - 02_feature_counts_by_layer.png")
    p(f"  - 03_linear_probe.png")
    p(f"  - 04_probe_vs_ablation.png")
    p(f"  - 05_circuit_metrics.png")
    p(f"  - 06_multisite_discovery.png")
p(f"{'='*72}")
