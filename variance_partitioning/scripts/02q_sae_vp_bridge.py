"""
02q_sae_vp_bridge.py  (submission version)

SAE–VP corroboration figure.

For each headline VP finding, this script shows that the SAE independently
recovers the same structure — without having seen the VP labels.

VP findings cross-referenced:
  1. Assert (N2218, L5) — monosemantic MLP neuron
  2. {FunctionDef, AsyncFunctionDef} — definitions group (share winner both sites)
  3. {ListComp, GeneratorExp} — VP G2 iteration group (share winner both sites)
  4. {If, Try} — control flow group (share MLP winner; diverge at Resid)

INPUTS
  --multilayer_path   multilayer_multisite_discovery.json
                      Keys: L{layer}_{site}  (site ∈ {mlp, resid})
                      Used for Panel (a): Assert per-layer selectivity
  --dendro_path       dendrogram_corroboration.json  (study_v2/reports/)
                      Contains mlp_per_node / resid_per_node winner features
                      Used for Panel (b): three VP–SAE corroborating groups

OUTPUTS (under out_dir/figures/sae_vp_bridge/)
  sae_vp_bridge.pdf   — two-panel figure:
    (a) Assert SAE top selectivity across layers (MLP + Residual) vs mean of
        other nodes; VP landmark at L5 annotated
    (b) Dot plot: winner SAE feature alignment for 3 VP-corroborating groups.
        Same colour = same feature ID → shared winner at that site.
        Groups 1–2 share at both sites; group 3 shares only at MLP.

Usage
-----
  python 02q_sae_vp_bridge.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

LAYERS = list(range(8))
SITES  = ["mlp", "resid"]
SITE_LABELS  = {"mlp": "MLP", "resid": "Residual"}
SITE_STYLES  = {"mlp": "-o", "resid": "-s"}
SITE_COLOURS = {"mlp": "#377eb8", "resid": "#e41a1c"}

VP_ASSERT_LAYER = 5   # VP landmark: N2218 identified at L5

# Three VP–SAE corroborating groups shown in panel (b)
CORR_GROUPS = [
    {"label": "Definitions",    "vp": "VP: definitions",    "nodes": ["FunctionDef", "AsyncFunctionDef"]},
    {"label": "Iteration (G2)", "vp": "VP: G2",             "nodes": ["ListComp",    "GeneratorExp"]},
    {"label": "Control flow",   "vp": "VP: control flow",   "nodes": ["If",          "Try"]},
]
GROUP_COLOURS = ["#ff7f00", "#4daf4a", "#377eb8"]   # one colour per group label


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ── Panel (a) helpers ─────────────────────────────────────────────────────────

def node_top_sel(data: dict, node: str, layer: int, site: str) -> float:
    entry = data.get(f"L{layer}_{site}", {}).get("nodes", {}).get(node, {})
    top = entry.get("top_by_selectivity", [])
    return top[0]["sel"] if top else 0.0


def all_nodes(data: dict) -> list[str]:
    nodes: set[str] = set()
    for v in data.values():
        nodes |= set(v.get("nodes", {}).keys())
    return sorted(nodes)


# ── Panel (b) helpers ─────────────────────────────────────────────────────────

def extract_winner_features(dendro: dict) -> dict[str, dict[str, int | None]]:
    """
    Returns  features[site][node] = feature_id  for all nodes in CORR_GROUPS.
    site ∈ {'mlp', 'resid'}.
    """
    all_group_nodes = {n for g in CORR_GROUPS for n in g["nodes"]}
    features: dict[str, dict[str, int | None]] = {"mlp": {}, "resid": {}}
    for entry in dendro.get("mlp_per_node", []):
        if entry["node"] in all_group_nodes:
            features["mlp"][entry["node"]] = entry.get("feature")
    for entry in dendro.get("resid_per_node", []):
        if entry["node"] in all_group_nodes:
            features["resid"][entry["node"]] = entry.get("feature")
    return features


# ─────────────────────────────────────────────────────────────────────────────
# Console summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(ml_data: dict, dendro: dict) -> None:
    nodes = all_nodes(ml_data)
    print("\n=== SAE–VP corroboration summary ===")

    print(f"\nAssert top-selectivity at VP landmark L{VP_ASSERT_LAYER}:")
    for site in SITES:
        sel    = node_top_sel(ml_data, "Assert", VP_ASSERT_LAYER, site)
        others = [node_top_sel(ml_data, n, VP_ASSERT_LAYER, site)
                  for n in nodes if n != "Assert"]
        mean_o = np.mean(others) if others else 0.0
        ratio  = f"{sel/mean_o:.2f}x" if mean_o > 0 else "n/a"
        print(f"  {site:<6}: sel={sel:.3f}  mean_others={mean_o:.3f}  ratio={ratio}")

    features = extract_winner_features(dendro)
    print("\nWinner features per group:")
    for g in CORR_GROUPS:
        print(f"  [{g['label']}]")
        for node in g["nodes"]:
            mlp_f   = features["mlp"].get(node,   "—")
            resid_f = features["resid"].get(node,  "—")
            shared_mlp   = "✓" if len({features["mlp"].get(n)   for n in g["nodes"]}) == 1 else "✗"
            shared_resid = "✓" if len({features["resid"].get(n) for n in g["nodes"]}) == 1 else "✗"
            print(f"    {node:<22} MLP={mlp_f}  Resid={resid_f}")
        print(f"    → shared at MLP:{shared_mlp}  Resid:{shared_resid}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────

def plot(ml_data: dict, dendro: dict, fmt: str, dpi: int, fig_dir: Path) -> None:
    nodes    = all_nodes(ml_data)
    x        = np.array(LAYERS)
    features = extract_winner_features(dendro)

    # Stacked layout: panel (a) on top, panel (b) below
    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(8, 9),
        gridspec_kw={"height_ratios": [1, 1.6]},
    )

    # ── Panel (a): Assert selectivity across layers ───────────────────────────
    for site in SITES:
        assert_profile = np.array(
            [node_top_sel(ml_data, "Assert", l, site) for l in LAYERS])
        other_mean = np.array([
            np.mean([node_top_sel(ml_data, n, l, site)
                     for n in nodes if n != "Assert"])
            for l in LAYERS
        ])
        ax_a.plot(x, assert_profile, SITE_STYLES[site],
                  color=SITE_COLOURS[site],
                  label=f"Assert ({SITE_LABELS[site]})",
                  linewidth=1.8, markersize=4)
        ax_a.plot(x, other_mean, "--",
                  color=SITE_COLOURS[site], alpha=0.45, linewidth=1.1)

    ax_a.axvline(VP_ASSERT_LAYER, color="0.5", linestyle=":", linewidth=1.2)
    ax_a.text(VP_ASSERT_LAYER + 0.12, 0.97,
              f"VP: N2218\n(L{VP_ASSERT_LAYER})",
              fontsize=7.5, color="0.45", va="top",
              transform=ax_a.get_xaxis_transform())
    ax_a.plot([], [], "--", color="0.6", alpha=0.6, label="mean of other nodes")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([f"L{l}" for l in x], fontsize=8)
    ax_a.set_xlabel("Layer", fontsize=10)
    ax_a.set_ylabel("Top SAE feature selectivity", fontsize=9)
    ax_a.set_title("(a)  Assert SAE selectivity profile\n"
                   "(VP: monosemantic MLP neuron at L5)", fontsize=9.5)
    ax_a.legend(fontsize=8, frameon=False, loc="upper left")
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    # ── Panel (b): 3 VP–SAE corroborating groups ─────────────────────────────
    # Two x-columns: MLP (left) and Residual (right).
    # Each node is a row; groups are separated by shaded bands.
    # Coloured dot + connecting line = nodes share the same SAE winner feature.
    # Grey dot = node has a different winner (diverges from group).
    # Feature ID is printed next to the connecting line.

    col_x  = {"mlp": 0.28, "resid": 0.72}
    marker = {"mlp": "o", "resid": "s"}
    band_alpha = 0.07

    # Build Y positions with spacing between groups
    y_pos: dict[str, float] = {}
    gap    = 1.0
    cursor = 0.0
    group_spans: list[tuple[float, float]] = []   # (y_lo, y_hi) for band shading
    group_ymid:  list[float] = []

    for g in CORR_GROUPS:
        ys = []
        for node in reversed(g["nodes"]):
            y_pos[node] = cursor
            ys.append(cursor)
            cursor += 1.0
        group_spans.append((min(ys) - 0.4, max(ys) + 0.4))
        group_ymid.append(np.mean(ys))
        cursor += gap

    total_height = cursor - gap

    # Shaded bands per group
    for gi, (y_lo, y_hi) in enumerate(group_spans):
        ax_b.axhspan(y_lo, y_hi, color=GROUP_COLOURS[gi],
                     alpha=band_alpha, zorder=0)

    # Group label inside the band, left margin
    for gi, g in enumerate(CORR_GROUPS):
        ax_b.text(-0.02, group_ymid[gi],
                  g["label"],
                  fontsize=8.5, color=GROUP_COLOURS[gi],
                  fontweight="bold", va="center", ha="right",
                  transform=ax_b.get_yaxis_transform())

    # Dots and connecting lines
    for gi, g in enumerate(CORR_GROUPS):
        gc = GROUP_COLOURS[gi]

        for site in SITES:
            feat_vals = [features[site].get(n) for n in g["nodes"]]
            unique_feats = {f for f in feat_vals if f is not None}
            shared = len(unique_feats) == 1

            for node in g["nodes"]:
                fid = features[site].get(node)
                if fid is None:
                    continue
                dot_c = gc if shared else "#c0c0c0"
                ec    = gc if shared else "#888888"
                ax_b.scatter(col_x[site], y_pos[node],
                             c=dot_c, s=130, zorder=5,
                             marker=marker[site],
                             edgecolors=ec, linewidths=0.8)

            # Connecting line + feature-ID label if shared
            if shared:
                fid = feat_vals[0]
                ys  = [y_pos[n] for n in g["nodes"]]
                ax_b.vlines(col_x[site], min(ys), max(ys),
                            color=gc, linewidth=2.5, alpha=0.55, zorder=4)
                # Label: "feat XXXXX" beside the midpoint of the line
                label_x = col_x[site] + 0.04
                ax_b.text(label_x, np.mean(ys),
                          f"feat {fid}",
                          fontsize=7, color=gc, va="center", ha="left",
                          zorder=6)

    # Column headers (drawn above the top band)
    top_y = total_height + 0.2
    for site in SITES:
        ax_b.text(col_x[site], top_y, SITE_LABELS[site],
                  fontsize=10, ha="center", va="bottom", fontweight="bold",
                  color="0.3")

    # Axes decoration
    ax_b.set_yticks(list(y_pos.values()))
    ax_b.set_yticklabels(list(y_pos.keys()), fontsize=9)
    ax_b.set_xlim(-0.05, 1.05)
    ax_b.set_ylim(-0.6, top_y + 0.4)
    ax_b.set_xticks([])
    ax_b.set_title("(b)  SAE winner-feature alignment for VP-identified groups",
                   fontsize=9.5, pad=8)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.spines["bottom"].set_visible(False)
    ax_b.spines["left"].set_visible(False)
    ax_b.tick_params(left=False)

    # Legend — top right, away from dots
    legend_elements = [
        Line2D([0], [0], marker="o", color="0.35", linestyle="None",
               markersize=7, label="MLP winner feature"),
        Line2D([0], [0], marker="s", color="0.35", linestyle="None",
               markersize=7, label="Residual winner feature"),
        Line2D([0], [0], marker="o", markerfacecolor="#c0c0c0",
               markeredgecolor="#888888", linestyle="None",
               markersize=7, label="Different feature (diverges)"),
        Line2D([0], [0], color="0.5", linewidth=2,
               label="Shared feature (connected)"),
    ]
    ax_b.legend(handles=legend_elements, fontsize=7.5,
                frameon=True, framealpha=0.9, edgecolor="0.8",
                loc="upper left", bbox_to_anchor=(-0.10, 1.06))

    fig.suptitle("SAE independently recovers VP-identified structure",
                 fontsize=10, y=1.01)
    fig.tight_layout(h_pad=2.5)

    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / f"sae_vp_bridge.{fmt}"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

_SAE_DATA     = Path(__file__).resolve().parents[2] / "sparse_autoencoders" / "data"
_DEFAULT_ML   = _SAE_DATA / "results" / "multilayer_multisite_discovery.json"
_DEFAULT_DEND = _SAE_DATA / "study_v2" / "reports" / "dendrogram_corroboration.json"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--multilayer_path", default=_DEFAULT_ML,
                   help="Path to multilayer_multisite_discovery.json")
    p.add_argument("--dendro_path", default=_DEFAULT_DEND,
                   help="Path to dendrogram_corroboration.json")
    p.add_argument("--out_dir", default="../results/output")
    p.add_argument("--format",  default="pdf", choices=["pdf", "png", "svg"])
    p.add_argument("--dpi",     type=int, default=200)
    args = p.parse_args()

    ml_path   = Path(args.multilayer_path)
    dend_path = Path(args.dendro_path)
    fig_dir   = Path(args.out_dir) / "figures" / "sae_vp_bridge"

    print(f"Loading {ml_path.name} …")
    ml_data = load_json(ml_path)
    print(f"  {len(ml_data)} (layer, site) combinations")

    print(f"Loading {dend_path.name} …")
    dendro = load_json(dend_path)

    print_summary(ml_data, dendro)

    print("\nPlotting …")
    plot(ml_data, dendro, fmt=args.format, dpi=args.dpi, fig_dir=fig_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
