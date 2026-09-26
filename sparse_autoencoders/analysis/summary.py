"""
Generate summary visualizations from the master circuit report.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from core import load_config, get_path


def main():
    cfg = load_config()
    report_path = get_path(cfg, "reports", cfg["paths"]["master_report"])
    with open(report_path) as f:
        data = json.load(f)

    nodes = [k for k in data.keys() if not k.startswith("_")]
    cross = data.get("_cross_ablation", {})

    # --- Fig 1: Ablation Drop Comparison ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1a: Zero-ablation drops
    drops = [data[n]["metrics"]["drop_zero"] for n in nodes]
    colors = ["#e74c3c" if cross.get(n, {}).get("specificity", {}).get("is_specific", False)
              else "#3498db" for n in nodes]
    axes[0].barh(nodes, drops, color=colors)
    axes[0].set_xlabel("Logprob Drop (zero ablation)")
    axes[0].set_title("Causal Impact per AST Node")
    axes[0].axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    red_patch = mpatches.Patch(color="#e74c3c", label="Node-Specific")
    blue_patch = mpatches.Patch(color="#3498db", label="Shared")
    axes[0].legend(handles=[red_patch, blue_patch], loc="lower right")

    # 1b: Patch gains
    gains = [data[n]["metrics"]["patch_gain"] for n in nodes]
    axes[1].barh(nodes, gains, color="#2ecc71")
    axes[1].set_xlabel("Patch Gain (logprob change)")
    axes[1].set_title("Feature Transfer Effect")
    axes[1].axvline(x=0, color="gray", linestyle="--", alpha=0.5)

    # 1c: Specificity ratios
    ratios = [cross.get(n, {}).get("specificity", {}).get("ratio", 0) for n in nodes]
    axes[2].barh(nodes, ratios, color=colors)
    axes[2].set_xlabel("Specificity Ratio (own/other drop)")
    axes[2].set_title("Feature Specificity")
    axes[2].axvline(x=1.5, color="red", linestyle="--", alpha=0.5, label="Threshold")
    axes[2].legend()

    plt.tight_layout()
    out1 = get_path(cfg, "results", "ablation_summary.png")
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"Saved: {out1}")
    plt.close()

    # --- Fig 2: Feature sharing matrix ---
    features = {n: data[n]["feature"] for n in nodes}
    unique_features = sorted(set(features.values()))
    matrix = np.zeros((len(nodes), len(unique_features)))
    for i, n in enumerate(nodes):
        j = unique_features.index(features[n])
        matrix[i, j] = 1.0

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(unique_features)))
    ax.set_xticklabels([str(f) for f in unique_features], rotation=45, ha="right")
    ax.set_yticks(range(len(nodes)))
    ax.set_yticklabels(nodes)
    ax.set_xlabel("SAE Feature ID")
    ax.set_ylabel("AST Node")
    ax.set_title("Feature Assignment Matrix (Layer 7)")
    plt.colorbar(im, ax=ax, shrink=0.6)
    plt.tight_layout()
    out2 = get_path(cfg, "results", "feature_sharing_matrix.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved: {out2}")
    plt.close()

    # --- Fig 3: Argmax change analysis ---
    fig, ax = plt.subplots(figsize=(10, 5))
    changed = []
    for n in nodes:
        a = data[n]["argmax"]
        changed.append(1 if a["changed"] else 0)

    bar_colors = ["#e74c3c" if c else "#2ecc71" for c in changed]
    ax.barh(nodes, changed, color=bar_colors)
    ax.set_xlabel("Argmax Token Changed After Ablation")
    ax.set_title("Ablation Changes Model Prediction")
    ax.set_xlim(-0.1, 1.1)

    # Annotate with before/after tokens
    for i, n in enumerate(nodes):
        a = data[n]["argmax"]
        label = f"'{a['before']}' → '{a['after']}'" if a["changed"] else f"'{a['before']}' (unchanged)"
        ax.text(0.5, i, label, ha="center", va="center", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    out3 = get_path(cfg, "results", "argmax_changes.png")
    plt.savefig(out3, dpi=150, bbox_inches="tight")
    print(f"Saved: {out3}")
    plt.close()

    # --- Summary stats ---
    print("\n=== EXPERIMENT SUMMARY ===")
    print(f"Nodes analyzed: {len(nodes)}")
    print(f"Unique features found: {len(unique_features)}")
    specific_nodes = [n for n in nodes
                      if cross.get(n, {}).get("specificity", {}).get("is_specific", False)]
    print(f"Node-specific features: {len(specific_nodes)} ({', '.join(specific_nodes)})")
    print(f"Argmax changed: {sum(changed)}/{len(nodes)}")
    print(f"Mean zero-ablation drop: {np.mean(drops):.4f}")
    print(f"Max zero-ablation drop: {max(drops):.4f} ({nodes[np.argmax(drops)]})")


if __name__ == "__main__":
    main()
