import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from core import data_path  # noqa: E402
#!/usr/bin/env python3
"""Generate a PDF report of three key SAE findings with plots and data."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec

DATA_DIR = data_path("results")
OUT_PDF  = data_path("results", "sae_key_findings.pdf")

LAYERS = [4, 5, 6, 7]
SITES  = ["mlp", "resid"]
AST_NODES = [
    "For", "While", "If", "FunctionDef", "AsyncFunctionDef",
    "ClassDef", "With", "Try", "Lambda", "Return", "Yield",
    "ListComp", "Assert",
]

# ── load data ─────────────────────────────────────────────────────────
with open(f"{DATA_DIR}/advanced_analysis.json") as f:
    adv1 = json.load(f)
with open(f"{DATA_DIR}/advanced_analysis_2.json") as f:
    adv2 = json.load(f)

# ── colours ───────────────────────────────────────────────────────────
MLP_COL   = "#2563eb"
RESID_COL = "#dc2626"
BG_COL    = "#f8fafc"
GRID_COL  = "#e2e8f0"


def section_title_page(pdf, title, subtitle):
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.5, 0.6, title, ha="center", va="center",
            fontsize=26, fontweight="bold", color="#1e293b")
    ax.text(0.5, 0.3, subtitle, ha="center", va="center",
            fontsize=13, color="#64748b", style="italic")
    ax.axis("off")
    fig.patch.set_facecolor(BG_COL)
    pdf.savefig(fig); plt.close(fig)


# ======================================================================
# FINDING 1 — Logit Lens per Feature
# ======================================================================

def plot_logit_lens(pdf):
    section_title_page(
        pdf,
        "Finding 1: Logit Lens per Feature",
        "Do SAE features promote tokens semantically aligned with their AST node?"
    )

    # --- Page 1: heatmap of top-1 promoted logit by (layer×site, node) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor(BG_COL)
    fig.suptitle("Top-1 Promoted Token Logit per AST Node", fontsize=16, fontweight="bold", y=0.97)

    for si, site in enumerate(SITES):
        mat = np.zeros((len(LAYERS), len(AST_NODES)))
        annot = [[""]*len(AST_NODES) for _ in LAYERS]
        for li, layer in enumerate(LAYERS):
            key = f"logit_lens_L{layer}_{site}"
            for ni, node in enumerate(AST_NODES):
                info = adv1[key][node]
                top = info["promoted"][0]
                mat[li, ni] = top["logit"]
                tok = top["token"][:8]
                annot[li][ni] = tok

        ax = axes[si]
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=0.25)
        ax.set_xticks(range(len(AST_NODES)))
        ax.set_xticklabels(AST_NODES, rotation=55, ha="right", fontsize=8)
        ax.set_yticks(range(len(LAYERS)))
        ax.set_yticklabels([f"L{l}" for l in LAYERS])
        ax.set_title(f"{site.upper()} site", fontsize=13, fontweight="bold")

        for li in range(len(LAYERS)):
            for ni in range(len(AST_NODES)):
                ax.text(ni, li, annot[li][ni], ha="center", va="center",
                        fontsize=6, color="black" if mat[li,ni] < 0.12 else "white")

    fig.colorbar(im, ax=axes, shrink=0.6, label="Logit contribution")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig); plt.close(fig)

    # --- Page 2: highlight notable semantic alignments ---
    notable = []
    for layer in LAYERS:
        for site in SITES:
            key = f"logit_lens_L{layer}_{site}"
            for node in AST_NODES:
                info = adv1[key][node]
                feat = info["feature"]
                for tok_info in info["promoted"][:5]:
                    t = tok_info["token"].lower().strip()
                    node_l = node.lower()
                    # check semantic alignment
                    aligned = False
                    kw_map = {
                        "for": ["for", "range", "iter", "loop", "enumerate"],
                        "while": ["while", "loop", "condition"],
                        "if": ["if", "else", "elif", "not", "condition"],
                        "functiondef": ["def", "func", "function", "return", "param", "args"],
                        "asyncfunctiondef": ["async", "await", "def"],
                        "classdef": ["class", "self", "__init__", "method"],
                        "with": ["with", "open", "file", "read", "context"],
                        "try": ["try", "except", "error", "raise", "catch"],
                        "lambda": ["lambda", "param", "args", "input"],
                        "return": ["return", "result", "value"],
                        "yield": ["yield", "generator", "iter", "next"],
                        "listcomp": ["list", "comp", "for", "["],
                        "assert": ["assert", "test", "error", "true", "false"],
                    }
                    for kw in kw_map.get(node_l, []):
                        if kw in t:
                            aligned = True
                            break
                    if aligned:
                        notable.append((layer, site, node, feat, tok_info["token"],
                                        tok_info["logit"]))

    fig = plt.figure(figsize=(11, 8))
    fig.patch.set_facecolor(BG_COL)
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_title("Semantically Aligned Promotions (keyword match in top-5 promoted tokens)",
                 fontsize=14, fontweight="bold", pad=20)

    if notable:
        col_labels = ["Layer", "Site", "AST Node", "Feature", "Promoted Token", "Logit"]
        cell_text = [[str(n[0]), n[1], n[2], str(n[3]), n[4], f"{n[5]:.4f}"] for n in notable[:30]]
        table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center",
                         cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.3)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#334155")
                cell.set_text_props(color="white", fontweight="bold")
            else:
                cell.set_facecolor("#f1f5f9" if row % 2 == 0 else "white")

    fig.tight_layout()
    pdf.savefig(fig); plt.close(fig)

    # --- Page 3: bar chart of alignment rate per layer ---
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG_COL)

    alignment_rates = {}
    for layer in LAYERS:
        for site in SITES:
            key = f"logit_lens_L{layer}_{site}"
            total = 0; aligned_count = 0
            for node in AST_NODES:
                total += 1
                info = adv1[key][node]
                node_l = node.lower()
                kw_map = {
                    "for": ["for", "range", "iter", "loop", "enumerate"],
                    "while": ["while", "loop", "condition"],
                    "if": ["if", "else", "elif", "not", "condition"],
                    "functiondef": ["def", "func", "function", "return", "param", "args"],
                    "asyncfunctiondef": ["async", "await", "def"],
                    "classdef": ["class", "self", "__init__", "method"],
                    "with": ["with", "open", "file", "read", "context"],
                    "try": ["try", "except", "error", "raise", "catch"],
                    "lambda": ["lambda", "param", "args", "input"],
                    "return": ["return", "result", "value"],
                    "yield": ["yield", "generator", "iter", "next"],
                    "listcomp": ["list", "comp", "for", "["],
                    "assert": ["assert", "test", "error", "true", "false"],
                }
                for tok_info in info["promoted"][:5]:
                    t = tok_info["token"].lower().strip()
                    for kw in kw_map.get(node_l, []):
                        if kw in t:
                            aligned_count += 1
                            break
                    else:
                        continue
                    break
            alignment_rates[(layer, site)] = aligned_count / total

    x = np.arange(len(LAYERS))
    w = 0.35
    mlp_rates = [alignment_rates[(l, "mlp")] for l in LAYERS]
    res_rates = [alignment_rates[(l, "resid")] for l in LAYERS]
    ax.bar(x - w/2, mlp_rates, w, label="MLP", color=MLP_COL, alpha=0.85)
    ax.bar(x + w/2, res_rates, w, label="Residual", color=RESID_COL, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Layer {l}" for l in LAYERS])
    ax.set_ylabel("Fraction of nodes with keyword in top-5 promoted tokens")
    ax.set_title("Semantic Alignment Rate by Layer and Site", fontsize=14, fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis="y", color=GRID_COL)
    ax.set_facecolor(BG_COL)
    fig.tight_layout()
    pdf.savefig(fig); plt.close(fig)


# ======================================================================
# FINDING 2 — Causal Ablation Hierarchy
# ======================================================================

def plot_causal_ablation(pdf):
    section_title_page(
        pdf,
        "Finding 2: Causal Ablation Hierarchy",
        "How does the causal importance of SAE features vary across layers and sites?"
    )

    # Compute per-site, per-layer mean |relative drop|
    # (use absolute to capture magnitude regardless of direction)
    mean_abs_rel = {s: [] for s in SITES}
    median_abs_rel = {s: [] for s in SITES}
    per_node_data = {}  # (layer, site) -> {node: rel_drop}

    for layer in LAYERS:
        for site in SITES:
            key = f"causal_ablation_L{layer}_{site}"
            vals = adv2[key]
            drops = []
            per_node_data[(layer, site)] = {}
            for node, info in vals.items():
                rd = info["relative_drop"]
                drops.append(abs(rd))
                per_node_data[(layer, site)][node] = rd
            mean_abs_rel[site].append(np.mean(drops))
            median_abs_rel[site].append(np.median(drops))

    # --- Page 1: mean |relative drop| by layer ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(BG_COL)
    fig.suptitle("Causal Effect of Feature Ablation Across Layers",
                 fontsize=16, fontweight="bold", y=0.99)

    x = np.arange(len(LAYERS))
    w = 0.35
    ax1.bar(x - w/2, mean_abs_rel["mlp"], w, label="MLP", color=MLP_COL, alpha=0.85)
    ax1.bar(x + w/2, mean_abs_rel["resid"], w, label="Residual", color=RESID_COL, alpha=0.85)
    ax1.set_xticks(x); ax1.set_xticklabels([f"L{l}" for l in LAYERS])
    ax1.set_ylabel("Mean |relative probability drop|")
    ax1.set_title("Mean Absolute Relative Drop", fontweight="bold")
    ax1.legend(); ax1.grid(axis="y", color=GRID_COL); ax1.set_facecolor(BG_COL)

    ax2.bar(x - w/2, median_abs_rel["mlp"], w, label="MLP", color=MLP_COL, alpha=0.85)
    ax2.bar(x + w/2, median_abs_rel["resid"], w, label="Residual", color=RESID_COL, alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels([f"L{l}" for l in LAYERS])
    ax2.set_ylabel("Median |relative probability drop|")
    ax2.set_title("Median Absolute Relative Drop", fontweight="bold")
    ax2.legend(); ax2.grid(axis="y", color=GRID_COL); ax2.set_facecolor(BG_COL)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig); plt.close(fig)

    # --- Page 2: heatmap of relative drop per node ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor(BG_COL)
    fig.suptitle("Relative Probability Drop per AST Node (feature ablation)",
                 fontsize=15, fontweight="bold", y=0.97)

    for si, site in enumerate(SITES):
        mat = np.zeros((len(LAYERS), len(AST_NODES)))
        for li, layer in enumerate(LAYERS):
            for ni, node in enumerate(AST_NODES):
                mat[li, ni] = per_node_data[(layer, site)].get(node, 0)

        ax = axes[si]
        # clip for visualization (L4 resid has extreme outliers)
        mat_clip = np.clip(mat, -5, 5)
        im = ax.imshow(mat_clip, aspect="auto", cmap="RdBu", vmin=-2, vmax=2)
        ax.set_xticks(range(len(AST_NODES)))
        ax.set_xticklabels(AST_NODES, rotation=55, ha="right", fontsize=8)
        ax.set_yticks(range(len(LAYERS)))
        ax.set_yticklabels([f"L{l}" for l in LAYERS])
        ax.set_title(f"{site.upper()} site", fontsize=13, fontweight="bold")

        for li in range(len(LAYERS)):
            for ni in range(len(AST_NODES)):
                v = mat[li, ni]
                ax.text(ni, li, f"{v:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if abs(mat_clip[li,ni]) > 1 else "black")

    fig.colorbar(im, ax=axes, shrink=0.6,
                 label="Relative drop (positive = keyword prob decreases on ablation)")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig); plt.close(fig)

    # --- Page 3: top causal features table ---
    fig = plt.figure(figsize=(11, 8))
    fig.patch.set_facecolor(BG_COL)
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_title("Strongest Causal Effects (by |relative drop|, excluding L4-resid outliers)",
                 fontsize=14, fontweight="bold", pad=20)

    rows = []
    for layer in LAYERS:
        for site in SITES:
            key = f"causal_ablation_L{layer}_{site}"
            for node, info in adv2[key].items():
                rd = info["relative_drop"]
                if layer == 4 and site == "resid":
                    continue  # skip outlier site
                rows.append((layer, site, node, info["feature"],
                             info["mean_baseline_prob"], info["prob_drop"], rd))

    rows.sort(key=lambda r: abs(r[6]), reverse=True)
    col_labels = ["Layer", "Site", "Node", "Feature", "Baseline P", "Prob Drop", "Rel Drop"]
    cell_text = [[str(r[0]), r[1], r[2], str(r[3]),
                  f"{r[4]:.5f}", f"{r[5]:+.6f}", f"{r[6]:+.3f}"] for r in rows[:25]]
    table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.25)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#334155")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#f1f5f9" if row % 2 == 0 else "white")

    fig.tight_layout()
    pdf.savefig(fig); plt.close(fig)


# ======================================================================
# FINDING 3 — Linear Probe Baseline
# ======================================================================

def plot_linear_probe(pdf):
    section_title_page(
        pdf,
        "Finding 3: Linear Probe — Raw vs SAE Latents",
        "Does SAE decomposition improve linear separability of AST node representations?"
    )

    # --- Page 1: accuracy comparison ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor(BG_COL)
    fig.suptitle("Linear Probe Classification Accuracy: Raw Activations vs SAE Latents",
                 fontsize=15, fontweight="bold", y=0.99)

    for si, site in enumerate(SITES):
        ax = [ax1, ax2][si]
        raw_accs = []
        sae_accs = []
        for layer in LAYERS:
            key = f"linear_probe_L{layer}_{site}"
            v = adv2[key]
            raw_accs.append(v["raw_activation_probe"]["accuracy"])
            sae_accs.append(v["sae_latent_probe"]["accuracy"])

        x = np.arange(len(LAYERS))
        w = 0.35
        bars1 = ax.bar(x - w/2, raw_accs, w, label="Raw activations", color="#6366f1", alpha=0.85)
        bars2 = ax.bar(x + w/2, sae_accs, w, label="SAE latents", color="#f59e0b", alpha=0.85)

        # add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., h + 0.005,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([f"Layer {l}" for l in LAYERS])
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{site.upper()} site", fontsize=13, fontweight="bold")
        ax.set_ylim(0.3, 0.85)
        ax.legend(loc="lower right")
        ax.grid(axis="y", color=GRID_COL)
        ax.set_facecolor(BG_COL)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig); plt.close(fig)

    # --- Page 2: delta (SAE - raw) across all sites ---
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG_COL)

    labels = []
    deltas = []
    colors = []
    for layer in LAYERS:
        for site in SITES:
            key = f"linear_probe_L{layer}_{site}"
            v = adv2[key]
            d = v["sae_latent_probe"]["accuracy"] - v["raw_activation_probe"]["accuracy"]
            labels.append(f"L{layer}\n{site}")
            deltas.append(d)
            colors.append(MLP_COL if site == "mlp" else RESID_COL)

    bars = ax.bar(range(len(labels)), deltas, color=colors, alpha=0.85)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Accuracy Delta (SAE - Raw)")
    ax.set_title("SAE Latent Accuracy Deficit Relative to Raw Activations",
                 fontsize=14, fontweight="bold")
    ax.grid(axis="y", color=GRID_COL)
    ax.set_facecolor(BG_COL)

    for bar, d in zip(bars, deltas):
        ax.text(bar.get_x() + bar.get_width()/2.,
                d - 0.003 if d < 0 else d + 0.003,
                f"{d:+.3f}", ha="center", va="top" if d < 0 else "bottom", fontsize=9)

    fig.tight_layout()
    pdf.savefig(fig); plt.close(fig)

    # --- Page 3: per-class accuracy comparison for best layer (L7 resid) ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(BG_COL)
    fig.suptitle("Per-Class Accuracy: Raw vs SAE (selected layer-site combinations)",
                 fontsize=15, fontweight="bold", y=0.98)

    combos = [(7, "resid"), (7, "mlp"), (4, "resid"), (6, "mlp")]
    for idx, (layer, site) in enumerate(combos):
        ax = axes[idx // 2][idx % 2]
        key = f"linear_probe_L{layer}_{site}"
        v = adv2[key]
        raw_pc = v["raw_activation_probe"]["per_class_accuracy"]
        sae_pc = v["sae_latent_probe"]["per_class_accuracy"]

        nodes = sorted(raw_pc.keys())
        raw_vals = [raw_pc[n] for n in nodes]
        sae_vals = [sae_pc[n] for n in nodes]

        x = np.arange(len(nodes))
        w = 0.35
        ax.bar(x - w/2, raw_vals, w, label="Raw", color="#6366f1", alpha=0.85)
        ax.bar(x + w/2, sae_vals, w, label="SAE", color="#f59e0b", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(nodes, rotation=55, ha="right", fontsize=7)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"L{layer} {site.upper()}", fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=8)
        ax.grid(axis="y", color=GRID_COL)
        ax.set_facecolor(BG_COL)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig); plt.close(fig)

    # --- Page 4: F1 comparison ---
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG_COL)

    labels = []
    raw_f1s = []
    sae_f1s = []
    for layer in LAYERS:
        for site in SITES:
            key = f"linear_probe_L{layer}_{site}"
            v = adv2[key]
            labels.append(f"L{layer} {site}")
            raw_f1s.append(v["raw_activation_probe"]["weighted_f1"])
            sae_f1s.append(v["sae_latent_probe"]["weighted_f1"])

    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, raw_f1s, w, label="Raw activations", color="#6366f1", alpha=0.85)
    ax.bar(x + w/2, sae_f1s, w, label="SAE latents", color="#f59e0b", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Weighted F1")
    ax.set_title("Weighted F1 Score: Raw vs SAE Latents", fontsize=14, fontweight="bold")
    ax.set_ylim(0.3, 0.8)
    ax.legend()
    ax.grid(axis="y", color=GRID_COL)
    ax.set_facecolor(BG_COL)
    fig.tight_layout()
    pdf.savefig(fig); plt.close(fig)


# ======================================================================
# Summary page
# ======================================================================

def summary_page(pdf):
    fig = plt.figure(figsize=(11, 8))
    fig.patch.set_facecolor(BG_COL)
    ax = fig.add_subplot(111)
    ax.axis("off")

    text = (
        "Summary of Key Findings\n"
        "━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "1. Logit Lens per Feature\n"
        "   SAE features show partial semantic alignment: some features promote\n"
        "   tokens related to their AST node (e.g., For→'for', ClassDef→'self',\n"
        "   Return→'Return', If→'not'/'else'), but alignment is inconsistent across\n"
        "   layers and often noisy. Residual-stream features at L4–L5 show the\n"
        "   strongest semantic specificity.\n\n"
        "2. Causal Ablation Hierarchy\n"
        "   Feature ablation effects are generally small in absolute probability terms,\n"
        "   reflecting that individual SAE features contribute incrementally. L4 residual\n"
        "   stream shows anomalously large effects (esp. AsyncFunctionDef, ListComp),\n"
        "   suggesting these features carry outsized causal weight at early layers.\n"
        "   Upper layers (L5–L7) show more uniform, moderate causal contributions.\n\n"
        "3. Linear Probe: Raw vs SAE Latents\n"
        "   Raw activations consistently outperform SAE latents on linear probe\n"
        "   classification (mean deficit: −1.2pp residual, −6.3pp MLP). This suggests\n"
        "   SAE decomposition trades linear separability for interpretability — the\n"
        "   sparsity constraint discards information useful for classification but\n"
        "   produces more human-interpretable features. Residual-stream SAEs preserve\n"
        "   substantially more classification-relevant information than MLP SAEs."
    )

    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=11,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cbd5e1"))
    fig.tight_layout()
    pdf.savefig(fig); plt.close(fig)


# ======================================================================
# Main
# ======================================================================

def main():
    with PdfPages(OUT_PDF) as pdf:
        # Cover
        fig, ax = plt.subplots(figsize=(11, 8))
        fig.patch.set_facecolor("#1e293b")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.text(0.5, 0.6, "SAE Key Findings Report", ha="center", va="center",
                fontsize=32, fontweight="bold", color="white")
        ax.text(0.5, 0.45, "Sparse Autoencoder Interpretability Analysis\nfor Python AST Node Circuits",
                ha="center", va="center", fontsize=14, color="#94a3b8")
        ax.text(0.5, 0.25, "COMP0087 Coursework — ATLAS Pipeline",
                ha="center", va="center", fontsize=12, color="#64748b")
        ax.axis("off")
        pdf.savefig(fig); plt.close(fig)

        plot_logit_lens(pdf)
        plot_causal_ablation(pdf)
        plot_linear_probe(pdf)
        summary_page(pdf)

    print(f"PDF saved to {OUT_PDF}")


if __name__ == "__main__":
    main()
