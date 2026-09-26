"""Generate figures for the SAE results section of the paper."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from core import data_path  # noqa: E402
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT = data_path("figures")
os.makedirs(OUT, exist_ok=True)
DATA = data_path("results")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
})

# ── 1. Phase-transition bar chart ──────────────────────────────────
with open(f"{DATA}/multilayer_feature_counts.json") as f:
    feat_counts = json.load(f)

layers = sorted(feat_counts.keys(), key=int)
totals = [sum(feat_counts[l].values()) for l in layers]

fig, ax = plt.subplots(figsize=(4.0, 2.4))
colors = ["#bdc3c7"] * 4 + ["#2980b9"] * 4  # grey for 0-3, blue for 4-7
bars = ax.bar([f"L{l}" for l in layers], totals, color=colors, edgecolor="white", linewidth=0.5)
for bar, val in zip(bars, totals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
            str(val), ha="center", va="bottom", fontsize=7)
ax.set_ylabel("Selective features")
ax.set_xlabel("Layer")
ax.set_title("Selective Feature Count by Layer")
ax.axvline(3.5, color="red", ls="--", lw=0.8, alpha=0.7)
ax.annotate("phase\ntransition", xy=(3.5, max(totals)*0.55),
            xytext=(2.3, max(totals)*0.7), fontsize=7, color="red",
            ha="center", arrowprops=dict(arrowstyle="->", color="red", lw=0.6))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(f"{OUT}/sae_phase_transition.pdf", bbox_inches="tight")
print("✓ sae_phase_transition.pdf")
plt.close()

# ── 2. Faithfulness gap: grouped bar chart ─────────────────────────
# Real numbers from advanced_analysis_2.json
with open(f"{DATA}/advanced_analysis_2.json") as f:
    adv2 = json.load(f)
with open(f"{DATA}/retrain_summary.json") as f:
    r2_data = json.load(f)

labels = []
raw_acc = []
sae_acc = []
r2_vals = []
for layer in [4, 5, 6, 7]:
    for site in ["mlp", "resid"]:
        key = f"L{layer}_{site}"
        probe = adv2[f"linear_probe_{key}"]
        labels.append(f"L{layer} {site}")
        raw_acc.append(probe["raw_activation_probe"]["accuracy"])
        sae_acc.append(probe["sae_latent_probe"]["accuracy"])
        r2_vals.append(r2_data[key]["r2"])

x = np.arange(len(labels))
w = 0.25

fig, ax = plt.subplots(figsize=(5.5, 2.6))
ax.bar(x - w, raw_acc, w, label="Raw probe acc.", color="#2ecc71", edgecolor="white", linewidth=0.5)
ax.bar(x,     sae_acc, w, label="SAE probe acc.", color="#e74c3c", edgecolor="white", linewidth=0.5)
ax.bar(x + w, r2_vals, w, label="SAE $R^2$",      color="#3498db", edgecolor="white", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_ylabel("Score")
ax.set_title("Faithfulness Gap: $R^2$ vs. Probe Accuracy")
ax.set_ylim(0.45, 1.02)
ax.legend(loc="lower right", framealpha=0.9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(f"{OUT}/sae_faithfulness_gap.pdf", bbox_inches="tight")
print("✓ sae_faithfulness_gap.pdf")
plt.close()

# ── 3. Causal ablation effect sizes (L7 residual report) ──
with open(data_path("reports", "circuit_report_L7_resid.json")) as f:
    mcr = json.load(f)

nodes = []
drops = []
for node, v in mcr.items():
    if node.startswith("_") or not isinstance(v, dict):
        continue
    dz = v.get("metrics", {}).get("drop_zero")
    if dz is not None:
        nodes.append(node)
        drops.append(abs(dz))
# Sort by effect size descending
order = np.argsort(drops)[::-1]
nodes = [nodes[i] for i in order]
drops = [drops[i] for i in order]

fig, ax = plt.subplots(figsize=(5.5, 2.6))
x = np.arange(len(nodes))
colors = ["#c0392b" if d >= 0.4 else "#e67e22" if d >= 0.1 else "#bdc3c7" for d in drops]
ax.bar(x, drops, 0.7, color=colors, edgecolor="white", linewidth=0.5)
ax.axhline(0.5, color="gray", ls="--", lw=0.7, alpha=0.6)
ax.text(len(x)-0.3, 0.51, "50%", fontsize=7, color="gray", ha="right")
for i, d in enumerate(drops):
    ax.text(i, d + 0.015, f"{d*100:.0f}%", ha="center", fontsize=6.5)
ax.set_xticks(x)
ax.set_xticklabels(nodes, rotation=45, ha="right", fontsize=7)
ax.set_ylabel(r"$|\Delta\log p|$")
ax.set_title("Top-Feature Ablation Effect per AST Node (L7 resid)")
ax.set_ylim(0, max(drops) * 1.15)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(f"{OUT}/sae_ablation_effects.pdf", bbox_inches="tight")
print("✓ sae_ablation_effects.pdf")
plt.close()

# ── 4. Nesting co-firing heatmap ──────────────────────────────────
nesting_combos = ["for_in_if", "def_in_class", "if_in_for", "try_in_for",
                  "with_in_def", "lambda_in_for"]
site_keys = []
for layer in [4,5,6,7]:
    for site in ["mlp", "resid"]:
        site_keys.append(f"nesting_L{layer}_{site}")

# Build matrix: rows=combos, cols=sites, value=1 if composition_holds
combo_labels = []
site_labels = []
matrix = []
for combo in nesting_combos:
    row = []
    for sk in site_keys:
        nest_data = adv2.get(sk, {})
        entry = nest_data.get(combo, {})
        holds = entry.get("composition_holds", None)
        if holds is None:
            row.append(float("nan"))
        else:
            row.append(1.0 if holds else 0.0)
    # Only include combo if it has data in at least one site
    if any(not np.isnan(v) for v in row):
        matrix.append(row)
        combo_labels.append(combo.replace("_", " "))

site_labels = [sk.replace("nesting_", "").replace("_", "\n") for sk in site_keys]
matrix = np.array(matrix)

fig, ax = plt.subplots(figsize=(5.0, 2.2))
cmap = matplotlib.colors.ListedColormap(["#e74c3c", "#2ecc71"])
bounds = [-0.5, 0.5, 1.5]
norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
ax.set_xticks(range(len(site_labels)))
ax.set_xticklabels(site_labels, fontsize=6)
ax.set_yticks(range(len(combo_labels)))
ax.set_yticklabels(combo_labels, fontsize=7)
ax.set_title("Compositional Nesting: Both Features Fire?")
# Add text
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        if not np.isnan(matrix[i, j]):
            txt = "Y" if matrix[i,j] == 1 else "N"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, fontweight="bold",
                    color="white")
        else:
            ax.text(j, i, "–", ha="center", va="center", fontsize=7, color="gray")
legend_patches = [mpatches.Patch(color="#2ecc71", label="Both fire"),
                  mpatches.Patch(color="#e74c3c", label="Fail")]
ax.legend(handles=legend_patches, loc="upper right", fontsize=6, framealpha=0.9,
          bbox_to_anchor=(1.22, 1.0))
fig.tight_layout()
fig.savefig(f"{OUT}/sae_nesting_heatmap.pdf", bbox_inches="tight")
print("✓ sae_nesting_heatmap.pdf")
plt.close()

# ── 5. Minimal-pair selectivity ───────────────────────────────────
# For each pair, compute: does the target feature fire more on target code?
pairs_all = []
for key in sorted(adv2.keys()):
    if not key.startswith("minimal_pairs_"):
        continue
    site = key.replace("minimal_pairs_", "")
    for pair_name, pdata in adv2[key].items():
        a_node = pdata["a_node"]
        b_node = pdata["b_node"]
        # a_activations: firing of a_node's feature on code-A
        # b_activations: firing of b_node's feature on code-B
        a_on_a = pdata["a_activations"].get(a_node, {}).get("firing_frac", 0)
        a_on_b = pdata["b_activations"].get(a_node, {}).get("firing_frac", 0)
        b_on_b = pdata["b_activations"].get(b_node, {}).get("firing_frac", 0)
        b_on_a = pdata["a_activations"].get(b_node, {}).get("firing_frac", 0)
        # Selectivity: does feature fire more on matching code?
        a_selective = a_on_a > a_on_b
        b_selective = b_on_b > b_on_a
        pairs_all.append({
            "site": site, "pair": pair_name,
            "a_selective": a_selective, "b_selective": b_selective,
            "a_gap": a_on_a - a_on_b, "b_gap": b_on_b - b_on_a,
        })

# Aggregate by pair across sites
from collections import Counter
pair_names = sorted(set(p["pair"] for p in pairs_all))
pair_results = {}
for pn in pair_names:
    subset = [p for p in pairs_all if p["pair"] == pn]
    n_total = len(subset)
    n_a_sel = sum(1 for p in subset if p["a_selective"])
    n_b_sel = sum(1 for p in subset if p["b_selective"])
    pair_results[pn] = {
        "n_sites": n_total,
        "a_selective_frac": n_a_sel / n_total if n_total else 0,
        "b_selective_frac": n_b_sel / n_total if n_total else 0,
    }

fig, ax = plt.subplots(figsize=(5.5, 2.6))
pn_labels = [p.replace("_vs_", " vs\n") for p in pair_names]
a_fracs = [pair_results[p]["a_selective_frac"] for p in pair_names]
b_fracs = [pair_results[p]["b_selective_frac"] for p in pair_names]
avg_fracs = [(a+b)/2 for a, b in zip(a_fracs, b_fracs)]

x = np.arange(len(pair_names))
colors_mp = ["#2ecc71" if f >= 0.5 else "#e67e22" for f in avg_fracs]
ax.bar(x, avg_fracs, 0.55, color=colors_mp, edgecolor="white", linewidth=0.5)
ax.axhline(0.5, color="gray", ls="--", lw=0.7, alpha=0.6)
ax.text(len(x)-0.5, 0.52, "chance", fontsize=7, color="gray", ha="right")
for i, (xi, v) in enumerate(zip(x, avg_fracs)):
    ax.text(xi, v + 0.02, f"{v:.0%}", ha="center", fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels(pn_labels, fontsize=7)
ax.set_ylabel("Frac. sites where\nfeature prefers target")
ax.set_title("Minimal-Pair Selectivity")
ax.set_ylim(0, 1.1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(f"{OUT}/sae_minimal_pairs.pdf", bbox_inches="tight")
print("✓ sae_minimal_pairs.pdf")
plt.close()

print("\nAll figures saved to", OUT)
