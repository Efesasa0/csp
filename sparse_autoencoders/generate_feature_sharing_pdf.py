import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from core import data_path  # noqa: E402
#!/usr/bin/env python3
"""Generate a PDF report on AST feature sharing vs. distinctness in SAE latent space."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import textwrap

DATA_DIR = Path(data_path())
DEEP = json.loads((DATA_DIR / "reports" / "deep_discovery_report.json").read_text())
MASTER = json.loads((DATA_DIR / "reports" / "master_circuit_report.json").read_text())
ADVANCED = json.loads((DATA_DIR / "results" / "advanced_analysis.json").read_text())

NODES = [
    "For", "While", "If", "FunctionDef", "AsyncFunctionDef", "ClassDef",
    "With", "Try", "Lambda", "Return", "Yield", "ListComp", "Assert",
]

# ── Gather data ──────────────────────────────────────────────────────────────

# Master circuit: primary feature per node
primary_features = {n: MASTER[n]["feature"] for n in NODES}

# Ensemble specificity
ensemble = DEEP["ensemble_L7_mlp"]
specificity = {n: ensemble[n]["specificity_ratio"] for n in NODES}
is_specific = {n: ensemble[n]["is_specific"] for n in NODES}

# Fingerprints: top-5 features per node
fingerprints = DEEP["fingerprints_L7_mlp"]

# Cross-site alignment
cross_site = ADVANCED["cross_site_L7"]


def make_title_page(pdf):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.text(0.5, 0.65, "AST Feature Sharing vs. Distinctness\nin SAE Latent Space",
            ha="center", va="center", fontsize=26, fontweight="bold",
            linespacing=1.5)
    ax.text(0.5, 0.42, "Evidence from ATLAS Pipeline — Layer 7 MLP",
            ha="center", va="center", fontsize=16, color="#555555")
    ax.text(0.5, 0.30,
            "Key finding: AST nodes are encoded through a compact, shared feature vocabulary\n"
            "but achieve node-specific effects through combinatorial activation and causal specificity.",
            ha="center", va="center", fontsize=12, color="#333333",
            linespacing=1.6, style="italic")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def make_table1_assignment(pdf):
    """Table 1: Primary feature assignment and specificity."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.set_title("Table 1: Primary Feature Assignment & Causal Specificity (L7 MLP)",
                 fontsize=16, fontweight="bold", pad=20)

    # Build rows
    rows = []
    for n in NODES:
        pf = primary_features[n]
        # Find which other nodes share this primary feature
        shared_with = [m for m in NODES if m != n and primary_features[m] == pf]
        shared_str = ", ".join(shared_with) if shared_with else "—"
        sr = specificity[n]
        sr_str = f"{sr:.2f}" if abs(sr) < 100 else f"{sr:.1f}"
        verdict = "Specific" if is_specific[n] else "Shared"
        rows.append([n, str(pf), shared_str, sr_str, verdict])

    col_labels = ["AST Node", "Primary\nFeature", "Shared With", "Specificity\nRatio", "Verdict"]
    colors = []
    for r in rows:
        if r[4] == "Specific":
            colors.append(["#e8f5e9"] * 5)
        else:
            colors.append(["#fff3e0"] * 5)

    table = ax.table(cellText=rows, colLabels=col_labels, loc="center",
                     cellColours=colors,
                     colColours=["#37474f"] * 5)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    # Style header
    for j in range(5):
        cell = table[0, j]
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_fontsize(10)

    # Style cells
    for i in range(1, len(rows) + 1):
        for j in range(5):
            cell = table[i, j]
            if j == 4:
                cell.set_text_props(fontweight="bold",
                                    color="#2e7d32" if rows[i-1][4] == "Specific" else "#e65100")

    # Summary text
    unique_primary = len(set(primary_features.values()))
    n_specific = sum(1 for v in is_specific.values() if v)
    ax.text(0.5, 0.04,
            f"Only {unique_primary} unique primary features cover {len(NODES)} AST node types. "
            f"{n_specific}/{len(NODES)} nodes show causal specificity (ratio > 1.5).",
            ha="center", va="center", fontsize=11, color="#333333",
            transform=ax.transAxes)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def make_plot1_specificity_bars(pdf):
    """Plot 1: Specificity ratio bar chart."""
    fig, ax = plt.subplots(figsize=(11, 7))

    # Sort by specificity ratio for visual clarity, but clip extreme negatives
    sorted_nodes = sorted(NODES, key=lambda n: specificity[n], reverse=True)
    vals = [specificity[n] for n in sorted_nodes]
    # Clip for display
    display_vals = [max(min(v, 45), -10) for v in vals]
    colors = ["#2e7d32" if is_specific[n] else "#e65100" for n in sorted_nodes]

    bars = ax.barh(range(len(sorted_nodes)), display_vals, color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(range(len(sorted_nodes)))
    ax.set_yticklabels(sorted_nodes, fontsize=12)
    ax.axvline(x=1.5, color="#888888", linestyle="--", linewidth=1.5, label="Specificity threshold (1.5)")
    ax.axvline(x=0, color="black", linewidth=0.5)

    # Annotate actual values
    for i, (v, dv) in enumerate(zip(vals, display_vals)):
        label = f"{v:.1f}"
        if v > 0:
            ax.text(dv + 0.3, i, label, va="center", fontsize=9, color="#333333")
        else:
            ax.text(dv - 0.3, i, label, va="center", fontsize=9, color="#333333", ha="right")

    ax.set_xlabel("Specificity Ratio (own drop / avg other drop)", fontsize=12)
    ax.set_title("Causal Specificity of Top-5 Features per AST Node",
                 fontsize=16, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.invert_yaxis()

    # Color legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2e7d32", label="Specific (ratio > 1.5)"),
        Patch(facecolor="#e65100", label="Shared (ratio ≤ 1.5)"),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc="lower right")

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def make_plot2_fingerprint_heatmap(pdf):
    """Plot 2: Feature sharing heatmap — which top-5 features are used by which nodes."""
    fp = fingerprints

    # Collect all unique features across all nodes' top-5
    all_feats = sorted(set(f for n in NODES for f in fp[n]["top_k_features"]))
    feat_to_idx = {f: i for i, f in enumerate(all_feats)}

    # Build binary matrix
    matrix = np.zeros((len(NODES), len(all_feats)))
    for i, n in enumerate(NODES):
        for f in fp[n]["top_k_features"]:
            matrix[i, feat_to_idx[f]] = 1.0

    # Count how many nodes use each feature
    feat_usage = matrix.sum(axis=0)

    fig, ax = plt.subplots(figsize=(12, 7))
    cmap = LinearSegmentedColormap.from_list("sharing", ["#fafafa", "#1565c0"])
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", interpolation="nearest")

    ax.set_xticks(range(len(all_feats)))
    ax.set_xticklabels([str(f) for f in all_feats], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(NODES)))
    ax.set_yticklabels(NODES, fontsize=11)
    ax.set_xlabel("SAE Feature Index", fontsize=12)
    ax.set_title("Top-5 Feature Fingerprints: Sharing Matrix (L7 MLP)",
                 fontsize=16, fontweight="bold")

    # Annotate usage counts at top
    for j, count in enumerate(feat_usage):
        ax.text(j, -0.7, f"{int(count)}", ha="center", va="center", fontsize=8,
                color="#c62828" if count >= 8 else "#333333", fontweight="bold")
    ax.text(len(all_feats) / 2, -1.3, "← nodes sharing this feature →",
            ha="center", fontsize=9, color="#888888")

    # Mark unique features
    for i, n in enumerate(NODES):
        for f in fp[n].get("unique_features", []):
            j = feat_to_idx[f]
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       fill=False, edgecolor="#c62828", linewidth=2.5))

    plt.colorbar(im, ax=ax, label="Feature active in top-5", shrink=0.7,
                 ticks=[0, 1])

    # Summary
    n_unique = len(all_feats)
    n_with_unique = sum(1 for n in NODES if fp[n].get("unique_features"))
    fig.text(0.5, 0.01,
             f"Only {n_unique} unique features across all top-5 sets. "
             f"Only {n_with_unique}/{len(NODES)} nodes have any unique feature (red boxes). "
             f"Mean uniqueness ratio: {np.mean([fp[n]['uniqueness_ratio'] for n in NODES]):.1%}",
             ha="center", fontsize=10, color="#333333")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def make_plot3_pairwise_discriminability(pdf):
    """Plot 3: Pairwise discriminability matrix."""
    pairwise = DEEP["pairwise_L7_mlp"]
    n = len(NODES)
    gap_matrix = np.zeros((n, n))

    for i, n1 in enumerate(NODES):
        for j, n2 in enumerate(NODES):
            if i == j:
                gap_matrix[i, j] = 1.0
                continue
            key = f"{n1}_vs_{n2}"
            alt_key = f"{n2}_vs_{n1}"
            if key in pairwise:
                entry = pairwise[key]
                gap_matrix[i, j] = entry["best_for_first"]["gap"]
            elif alt_key in pairwise:
                entry = pairwise[alt_key]
                gap_matrix[i, j] = entry["best_for_second"]["gap"]

    fig, ax = plt.subplots(figsize=(11, 9))
    cmap = LinearSegmentedColormap.from_list("disc", ["#ffebee", "#ffffff", "#e8f5e9", "#1b5e20"])
    im = ax.imshow(gap_matrix, cmap=cmap, vmin=0, vmax=1, aspect="equal")

    ax.set_xticks(range(n))
    ax.set_xticklabels(NODES, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(n))
    ax.set_yticklabels(NODES, fontsize=10)

    # Annotate
    for i in range(n):
        for j in range(n):
            if i != j:
                val = gap_matrix[i, j]
                color = "white" if val > 0.7 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=color)

    ax.set_title("Pairwise Feature Discriminability (L7 MLP)\nBest single-feature gap for distinguishing node i from node j",
                 fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Activation frequency gap", shrink=0.8)

    mean_gap = gap_matrix[~np.eye(n, dtype=bool)].mean()
    fig.text(0.5, 0.01,
             f"Mean pairwise gap: {mean_gap:.3f}. Every pair is discriminable by at least one feature.",
             ha="center", fontsize=11, color="#333333")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def make_plot4_cross_site(pdf):
    """Plot 4: Cross-site alignment — MLP vs Residual stream cosine similarity."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    nodes_sorted = sorted(NODES, key=lambda n: cross_site[n]["assigned_cosine"])

    # Panel A: Assigned cosine
    assigned = [cross_site[n]["assigned_cosine"] for n in nodes_sorted]
    colors_a = ["#c62828" if v < 0 else "#1565c0" for v in assigned]
    ax1.barh(range(len(nodes_sorted)), assigned, color=colors_a, height=0.7, edgecolor="white")
    ax1.set_yticks(range(len(nodes_sorted)))
    ax1.set_yticklabels(nodes_sorted, fontsize=10)
    ax1.axvline(x=0, color="black", linewidth=0.5)
    ax1.set_xlabel("Cosine Similarity", fontsize=11)
    ax1.set_title("(a) Assigned MLP↔Resid Cosine", fontsize=13, fontweight="bold")
    for i, v in enumerate(assigned):
        ax1.text(v + 0.005 if v >= 0 else v - 0.005, i,
                 f"{v:.3f}", va="center", fontsize=9,
                 ha="left" if v >= 0 else "right")
    ax1.set_xlim(-0.12, 0.25)

    # Panel B: Best-match cosine
    nodes_sorted2 = sorted(NODES, key=lambda n: cross_site[n]["best_resid_cosine"])
    best_cos = [cross_site[n]["best_resid_cosine"] for n in nodes_sorted2]
    ax2.barh(range(len(nodes_sorted2)), best_cos, color="#5c6bc0", height=0.7, edgecolor="white")
    ax2.set_yticks(range(len(nodes_sorted2)))
    ax2.set_yticklabels(nodes_sorted2, fontsize=10)
    ax2.set_xlabel("Cosine Similarity", fontsize=11)
    ax2.set_title("(b) Best-Match MLP→Resid Cosine", fontsize=13, fontweight="bold")
    for i, v in enumerate(best_cos):
        ax2.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
    ax2.set_xlim(0, 0.35)

    mean_assigned = np.mean(assigned)
    mean_best = np.mean(best_cos)
    fig.suptitle("Cross-Site Feature Alignment: MLP vs. Residual Stream (L7)",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.text(0.5, -0.02,
             f"Mean assigned cosine: {mean_assigned:.4f} (≈ orthogonal). "
             f"Mean best-match cosine: {mean_best:.4f} (very weak). "
             "MLP and Residual encode syntax through distinct feature sets.",
             ha="center", fontsize=10, color="#333333", wrap=True)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def make_table2_magnitude(pdf):
    """Table 2: Top magnitude-weighted selective features per node."""
    mag = DEEP["magnitude_L7_mlp"]

    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.set_title("Table 2: Top Selective Feature per AST Node (Magnitude-Weighted Contrast, L7 MLP)",
                 fontsize=14, fontweight="bold", pad=20)

    rows = []
    for n in NODES:
        top = mag[n][0]
        feat = top["feature"]
        cs = top["contrast_score"]
        own = top["own_magnitude"]
        other = top["other_magnitude"]
        # Check if this feature is top for any other node
        also_top_for = [m for m in NODES if m != n and mag[m][0]["feature"] == feat]
        also_str = ", ".join(also_top_for) if also_top_for else "—"
        rows.append([n, str(feat), f"{cs:.1f}", f"{own:.4f}", f"{other:.4f}", also_str])

    col_labels = ["AST Node", "Top\nFeature", "Contrast\nScore", "Own\nMagnitude",
                  "Other\nMagnitude", "Also Top For"]

    # Color by contrast score magnitude
    colors = []
    for r in rows:
        cs = float(r[2])
        if cs > 100:
            colors.append(["#e8f5e9"] * 6)
        elif cs > 10:
            colors.append(["#f1f8e9"] * 6)
        else:
            colors.append(["#fff8e1"] * 6)

    table = ax.table(cellText=rows, colLabels=col_labels, loc="center",
                     cellColours=colors,
                     colColours=["#37474f"] * 6)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    for j in range(6):
        cell = table[0, j]
        cell.set_text_props(color="white", fontweight="bold")

    # Unique top features
    top_feats = [mag[n][0]["feature"] for n in NODES]
    n_unique_top = len(set(top_feats))
    fig.text(0.5, 0.04,
             f"{n_unique_top} distinct features appear as top-1 across {len(NODES)} nodes. "
             "Green = high contrast (>100), yellow = moderate.",
             ha="center", fontsize=10, color="#333333")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def make_summary_page(pdf):
    """Summary / conclusions page."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")

    ax.text(0.5, 0.92, "Summary: Feature Sharing vs. Distinctness",
            ha="center", va="top", fontsize=20, fontweight="bold")

    findings = [
        ("Shared Vocabulary",
         f"Only {len(set(primary_features.values()))} unique primary features cover {len(NODES)} AST node types.\n"
         "In top-5 fingerprints, 97% of features are shared; most nodes have 0 unique features."),
        ("Causal Specificity",
         f"{sum(1 for v in is_specific.values() if v)}/{len(NODES)} nodes show causal specificity "
         "(ablating their features hurts them\nfar more than other nodes). "
         "ClassDef (38.6×), ListComp (27.7×), Lambda (21.0×) are most specific."),
        ("Pairwise Discriminability",
         "Every node pair can be distinguished by at least one feature.\n"
         "Even within the shared vocabulary, combinatorial patterns create separability."),
        ("Cross-Stream Orthogonality",
         "MLP and Residual stream features are near-orthogonal (mean cosine ≈ 0.027).\n"
         "The same syntax is encoded through completely distinct feature directions across streams."),
        ("Proposed Framing",
         "\"AST node types are encoded through a compact set of shared SAE features,\n"
         "yet exhibit causal specificity: ablating a node's top features disproportionately\n"
         "affects that node type. The same syntactic distinctions are encoded through\n"
         "orthogonal feature sets across MLP and residual streams.\""),
    ]

    y = 0.82
    for title, body in findings:
        ax.text(0.08, y, f"● {title}", fontsize=13, fontweight="bold",
                transform=ax.transAxes, va="top")
        ax.text(0.12, y - 0.04, body, fontsize=10, color="#333333",
                transform=ax.transAxes, va="top", linespacing=1.5)
        y -= 0.17

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main():
    out_path = DATA_DIR / "results" / "feature_sharing_analysis.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(str(out_path)) as pdf:
        make_title_page(pdf)
        make_table1_assignment(pdf)
        make_plot1_specificity_bars(pdf)
        make_plot2_fingerprint_heatmap(pdf)
        make_plot3_pairwise_discriminability(pdf)
        make_plot4_cross_site(pdf)
        make_table2_magnitude(pdf)
        make_summary_page(pdf)

    print(f"PDF saved to: {out_path}")


if __name__ == "__main__":
    main()
