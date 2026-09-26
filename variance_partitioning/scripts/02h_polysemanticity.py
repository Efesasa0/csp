"""
02h_polysemanticity.py  (lives in results/)

Visualises polysemanticity / superposition across MLP neurons, using the
per-class selectivity arrays computed in 02g_class_neurons.py.

Four figures
────────────
  A  assert_spotlight.{fmt}
        Bar chart of per-class mean activation for MLP neuron #2218 at L5.
        The only strictly monosemantic neuron found in the model.

  B  selectivity_waves.{fmt}
        Neurons sorted by owner class then descending selectivity.
        Each class drawn as a filled area across all neuron positions.
        Shows bell-curve peaks at owned neurons with visible bleed-through
        into other classes' neurons — the geometric signature of superposition.

  C  polysemanticity_dist.{fmt}
        Histogram of polysemanticity index  p = top2_sel / top1_sel
        across all pure MLP neurons.  p≈0 → monosemantic, p≈1 → polysemantic.
        Annotates the Assert neuron at p=0 and shows the distribution
        is heavily right-skewed (median ≈ 0.90).

  D  class_similarity_3d.html
        Interactive 3-D Plotly graph.  Nodes = AST classes.
        Edges weighted by cosine similarity of class selectivity vectors
        (how much two classes share the same neurons).
        Same layout / encoding as 02f_ast_graph.py.

INPUTS  (relative to --in_dir)
──────
  02g_<stem>_selectivity_neurons.npz
  02_<stem>_purity_masks.npz
  01_<stem>_mlp_neurons.npy
  01_<stem>_meta.json

OUTPUTS  (relative to --out_dir / figures / polysemanticity /)
───────
  assert_spotlight.{fmt}
  selectivity_waves.{fmt}
  polysemanticity_dist.{fmt}
  class_similarity_3d.html

Usage
─────
  python 02h_polysemanticity.py
  python 02h_polysemanticity.py --stem contrastive_stubs --in_dir ../data_more
  python 02h_polysemanticity.py --format png --dpi 200
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines   import Line2D
from matplotlib.patches import Patch
import plotly.graph_objects as go
import plotly.io as pio
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse        import csr_matrix
from sklearn.manifold    import MDS

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

ASSERT_NEURON_IDX = 2218   # global MLP neuron index
ASSERT_LAYER      = 5

FS_TITLE    = 12
FS_SUPTITLE = 14
FS_TICK     = 9
FS_ANNOT    = 8

EDGE_SIM_THRESH = 0.30     # hide edges below this cosine similarity in 3-D graph

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Polysemanticity visualisations")
    p.add_argument("--stem",    default="contrastive_stubs")
    p.add_argument("--in_dir",  default="../data_more")
    p.add_argument("--out_dir", default="../results/output")
    p.add_argument("--format",  default="pdf")
    p.add_argument("--dpi",     type=int, default=200)
    return p.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Colour helper  (matches 02g)
# ─────────────────────────────────────────────────────────────────────────────

def _class_colours(n: int) -> list:
    c1 = plt.get_cmap("tab20")
    c2 = plt.get_cmap("tab20b")
    return [c1(i) if i < 20 else c2(i - 20) for i in range(n)]

# ─────────────────────────────────────────────────────────────────────────────
# A — Assert neuron spotlight
# ─────────────────────────────────────────────────────────────────────────────

def plot_assert_spotlight(mlp: np.ndarray,
                          meta: list[dict],
                          classes: list[str],
                          fmt: str, dpi: int, fig_dir: Path) -> None:
    """
    Bar chart: mean activation of neuron #2218 at layer 5 per AST class.
    """
    ast_idx = np.array([classes.index(m["ast_node"]) for m in meta])
    acts    = mlp[:, ASSERT_LAYER, ASSERT_NEURON_IDX]   # (N,)

    means = np.array([
        acts[ast_idx == c].mean() if (ast_idx == c).sum() > 0 else 0.0
        for c in range(len(classes))
    ])
    colours  = _class_colours(len(classes))
    bar_cols = [
        "#d73027" if cls == "Assert" else "#aaaaaa"
        for cls in classes
    ]

    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(classes))
    ax.bar(x, means, color=bar_cols, edgecolor="none", width=0.75)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=FS_TICK)
    ax.axhline(0, lw=0.8, color="black")
    ax.set_ylabel("Mean activation", fontsize=FS_TITLE)
    ax.set_title(
        f"MLP neuron #{ASSERT_NEURON_IDX} at layer {ASSERT_LAYER} — "
        f"the only strictly monosemantic neuron identified",
        fontsize=FS_SUPTITLE, pad=8,
    )

    # annotate Assert bar
    assert_c = classes.index("Assert")
    ax.annotate(
        f"Assert only\n(z = 2.84)",
        xy=(assert_c, means[assert_c]),
        xytext=(assert_c + 2.5, means[assert_c] * 0.85),
        fontsize=FS_ANNOT,
        arrowprops=dict(arrowstyle="->", color="#d73027", lw=1.2),
        color="#d73027",
    )

    handles = [
        Patch(facecolor="#d73027", label="Assert (monosemantic)"),
        Patch(facecolor="#aaaaaa", label="All other classes (silent)"),
    ]
    ax.legend(handles=handles, fontsize=FS_ANNOT, loc="upper right")
    fig.tight_layout()

    out = fig_dir / f"assert_spotlight.{fmt}"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")

# ─────────────────────────────────────────────────────────────────────────────
# B — Selectivity waves
# ─────────────────────────────────────────────────────────────────────────────

def plot_selectivity_waves(sel_pure: np.ndarray,
                           pure_idx: np.ndarray,
                           owner: np.ndarray,
                           classes: list[str],
                           fmt: str, dpi: int, fig_dir: Path) -> None:
    """
    Neurons sorted by (argmax owner class, then descending selectivity).
    Each class drawn as a filled area showing its selectivity at every
    neuron position.  Bell-curve peaks form at owned neurons; bleed-through
    at others' neurons reveals superposition.
    """
    C, n_pure = sel_pure.shape
    colours = _class_colours(C)

    # sort: primary = owner class, secondary = descending top-1 sel
    top1_per = sel_pure.max(axis=0)
    sort_order = sorted(range(n_pure),
                        key=lambda i: (int(owner[i]), -float(top1_per[i])))
    sel_s     = sel_pure[:, sort_order]   # (C, n_pure) — re-ordered
    owner_s   = owner[sort_order]

    # class boundary positions (first index where owner changes)
    boundaries = [0]
    for i in range(1, n_pure):
        if owner_s[i] != owner_s[i - 1]:
            boundaries.append(i)
    boundaries.append(n_pure)

    # midpoint of each class block → label x-position
    label_x = [
        (boundaries[k] + boundaries[k + 1]) / 2
        for k in range(len(boundaries) - 1)
    ]

    # ── figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        2, 1, figsize=(16, 7),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )
    ax, ax_rug = axes

    x = np.arange(n_pure)

    # filled areas per class (sorted by neuron count so small classes are on top)
    class_order = sorted(range(C), key=lambda c: -(owner == c).sum())
    for c in class_order:
        y = sel_s[c]
        rgba = colours[c]
        ax.fill_between(x, 0, y, alpha=0.22, color=rgba, linewidth=0)
        # thin top edge for readability
        ax.plot(x, y, lw=0.3, alpha=0.5, color=rgba)

    # class boundary lines
    for b in boundaries[1:-1]:
        ax.axvline(b, color="#dddddd", lw=0.6, zorder=0)

    # class label ticks
    for k, (xm, owner_class) in enumerate(zip(label_x, sorted(set(owner_s)))):
        ax.text(xm, -0.12, classes[owner_class],
                ha="center", va="top", fontsize=5.5, color=colours[owner_class],
                rotation=45, transform=ax.get_xaxis_transform())

    # highlight Assert neuron
    assert_local  = np.where(pure_idx == ASSERT_NEURON_IDX)[0]
    if len(assert_local):
        pos_in_sort = np.where(np.array(sort_order) == int(assert_local[0]))[0]
        if len(pos_in_sort):
            ax_pos = int(pos_in_sort[0])
            ax.axvline(ax_pos, color="#d73027", lw=1.5, zorder=5)
            ax.annotate(
                f"Neuron #{ASSERT_NEURON_IDX}\n(Assert, p=0)",
                xy=(ax_pos, sel_s[classes.index("Assert"), ax_pos]),
                xytext=(ax_pos + n_pure * 0.03,
                        sel_s[classes.index("Assert"), ax_pos] + 0.4),
                fontsize=FS_ANNOT, color="#d73027",
                arrowprops=dict(arrowstyle="->", color="#d73027", lw=1.0),
            )

    ax.set_xlim(0, n_pure - 1)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Selectivity  (z-score)", fontsize=FS_TITLE)
    ax.set_xticks([])
    ax.set_title(
        "Neuron selectivity profiles — neurons sorted by owner class\n"
        "Bleed-through of each class curve into neighbouring blocks reveals superposition",
        fontsize=FS_SUPTITLE, pad=6,
    )

    # ── rug: owner class colour strip ─────────────────────────────────────────
    rug_img = np.array([colours[o] for o in owner_s])[:, :3]   # (n_pure, 3)
    ax_rug.imshow(rug_img[np.newaxis, :, :], aspect="auto",
                  extent=[0, n_pure, 0, 1])
    ax_rug.set_xlim(0, n_pure - 1)
    ax_rug.set_yticks([])
    ax_rug.set_xticks([])
    ax_rug.set_ylabel("Owner\nclass", fontsize=7, rotation=0, labelpad=28, va="center")

    fig.savefig(fig_dir / f"selectivity_waves.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fig_dir / f'selectivity_waves.{fmt}'}")

# ─────────────────────────────────────────────────────────────────────────────
# C — Polysemanticity index distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_polysemanticity_dist(sel_pure: np.ndarray,
                               pure_idx: np.ndarray,
                               classes: list[str],
                               fmt: str, dpi: int, fig_dir: Path) -> None:
    """
    Three-panel figure justifying the polysemanticity claim:
      Left:   Histogram of p = top2/top1 with CDF overlay, key percentiles marked
      Centre: Scatter — top1 selectivity vs top2 selectivity (each point = one neuron)
      Right:  Per-class median p, sorted, with global median line
    """
    C = len(classes)
    top1  = sel_pure.max(axis=0)
    top2  = np.sort(sel_pure, axis=0)[-2]
    poly  = np.where(top1 > 0, top2 / (top1 + 1e-9), 1.0)
    owner = sel_pure.argmax(axis=0)
    colours = _class_colours(C)

    med   = float(np.median(poly))
    p10   = float(np.percentile(poly, 10))
    p90   = float(np.percentile(poly, 90))

    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 3, width_ratios=[1.4, 1.2, 1.0], wspace=0.38)

    # ── left: histogram + CDF ─────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    bins = np.linspace(0, 1, 51)
    counts, edges = np.histogram(poly, bins=bins)
    ax0.bar(edges[:-1], counts, width=np.diff(edges),
            color="#4393c3", alpha=0.75, edgecolor="none", align="edge")

    # CDF on twin axis
    ax0r = ax0.twinx()
    sorted_p = np.sort(poly)
    cdf = np.arange(1, len(sorted_p) + 1) / len(sorted_p)
    ax0r.plot(sorted_p, cdf, color="#333333", lw=1.5, ls="-")
    ax0r.set_ylabel("Cumulative fraction", fontsize=FS_ANNOT, color="#333333")
    ax0r.set_ylim(0, 1.05)
    ax0r.tick_params(labelsize=FS_ANNOT - 1)

    # percentile markers
    for val, label, col in [
        (p10, f"p10={p10:.2f}", "#888888"),
        (med, f"median={med:.2f}", "#d73027"),
        (p90, f"p90={p90:.2f}", "#888888"),
    ]:
        ax0.axvline(val, ls="--", lw=1.2, color=col)
        ax0.text(val + 0.01, counts.max() * 0.92, label,
                 fontsize=FS_ANNOT - 1, color=col, va="top")

    # Assert neuron
    assert_local = np.where(pure_idx == ASSERT_NEURON_IDX)[0]
    if len(assert_local):
        p_assert = float(poly[assert_local[0]])
        ax0.annotate(
            f"#{ASSERT_NEURON_IDX} Assert\np = {p_assert:.3f}",
            xy=(p_assert + 0.005, counts[0]),
            xytext=(0.13, counts.max() * 0.55),
            fontsize=FS_ANNOT - 1, color="#d73027",
            arrowprops=dict(arrowstyle="->", color="#d73027", lw=0.9),
        )

    ax0.set_xlabel("Polysemanticity index  p = top2 / top1", fontsize=FS_TITLE)
    ax0.set_ylabel("Neuron count", fontsize=FS_TITLE)
    ax0.set_title("p = 0: monosemantic\np → 1: polysemantic", fontsize=FS_ANNOT + 1, pad=4)

    # ── centre: top1 vs top2 scatter ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1])
    pt_cols = [colours[o] for o in owner]
    ax1.scatter(top1, top2, c=pt_cols, s=3, alpha=0.35, linewidths=0)

    # y = x line (fully polysemantic)
    lim = max(top1.max(), top2.max()) * 1.02
    ax1.plot([0, lim], [0, lim], lw=1.0, ls="--", color="#999999",
             label="p = 1  (equal selectivity)")
    # y = 0.9x line (median)
    ax1.plot([0, lim], [0, 0.9 * lim], lw=1.0, ls=":", color="#d73027",
             label=f"p = {med:.2f}  (median)")

    # Assert neuron
    if len(assert_local):
        ai = assert_local[0]
        ax1.scatter([top1[ai]], [top2[ai]], s=60, color="#d73027",
                    zorder=5, marker="*")
        ax1.annotate(f"#{ASSERT_NEURON_IDX}", xy=(top1[ai], top2[ai]),
                     xytext=(top1[ai] - 0.3, top2[ai] + 0.15),
                     fontsize=FS_ANNOT - 1, color="#d73027",
                     arrowprops=dict(arrowstyle="->", color="#d73027", lw=0.8))

    ax1.set_xlim(0, lim); ax1.set_ylim(0, lim)
    ax1.set_xlabel("Top-1 selectivity  (z-score)", fontsize=FS_TITLE)
    ax1.set_ylabel("Top-2 selectivity  (z-score)", fontsize=FS_TITLE)
    ax1.set_title("Most neurons cluster near the\np = 1 diagonal", fontsize=FS_ANNOT + 1, pad=4)
    ax1.legend(fontsize=FS_ANNOT - 1, loc="upper left")

    # ── right: per-class median p ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[2])
    class_meds = []
    for c in range(C):
        owned = poly[owner == c]
        class_meds.append(float(np.median(owned)) if len(owned) > 0 else np.nan)

    valid = [(classes[c], class_meds[c], colours[c]) for c in range(C)
             if not np.isnan(class_meds[c])]
    valid.sort(key=lambda x: x[1])
    cls_names, cls_meds, cls_cols = zip(*valid)

    ypos = range(len(cls_names))
    ax2.barh(list(ypos), list(cls_meds), color=list(cls_cols), alpha=0.85)
    ax2.set_yticks(list(ypos))
    ax2.set_yticklabels(list(cls_names), fontsize=6)
    ax2.axvline(med, ls="--", lw=1.0, color="#d73027",
                label=f"Median = {med:.2f}")
    ax2.set_xlabel("Median p per class", fontsize=FS_TITLE)
    ax2.set_title("No class escapes\nsuperposition", fontsize=FS_ANNOT + 1, pad=4)
    ax2.legend(fontsize=FS_ANNOT - 1)

    fig.suptitle(
        "Polysemanticity in circuit_sparsity MLP neurons  "
        f"(n = {len(poly):,} pure neurons · median p = {med:.2f})",
        fontsize=FS_SUPTITLE, y=1.02,
    )

    fig.savefig(fig_dir / f"polysemanticity_dist.{fmt}", dpi=dpi,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fig_dir / f'polysemanticity_dist.{fmt}'}")

# ─────────────────────────────────────────────────────────────────────────────
# D — 3-D class similarity graph (Plotly HTML)
# ─────────────────────────────────────────────────────────────────────────────

def _embed_3d(sim: np.ndarray) -> np.ndarray:
    dist = np.clip(1.0 - sim, 0, None)
    np.fill_diagonal(dist, 0.0)
    mds = MDS(n_components=3, dissimilarity="precomputed",
              random_state=42, max_iter=500, n_init=4)
    return mds.fit_transform(dist).astype(np.float32)


def _mst_edges(sim: np.ndarray) -> list[tuple[int, int]]:
    dist = np.clip(1.0 - sim, 0, None)
    np.fill_diagonal(dist, 0.0)
    mst = minimum_spanning_tree(csr_matrix(dist)).toarray()
    edges = []
    n = dist.shape[0]
    for i in range(n):
        for j in range(n):
            if mst[i, j] > 0 or mst[j, i] > 0:
                edges.append((min(i, j), max(i, j)))
    return list(set(edges))


def plot_class_similarity_3d(sel_pure: np.ndarray,
                              classes: list[str],
                              owner: np.ndarray,
                              fig_dir: Path) -> None:
    """
    3-D Plotly graph where nodes = AST classes, edges = cosine similarity of
    their selectivity vectors.  Similar classes (sharing neurons) are close.
    """
    C = len(classes)

    # cosine similarity between class selectivity vectors
    norms = np.linalg.norm(sel_pure, axis=1, keepdims=True) + 1e-9
    sel_n  = sel_pure / norms
    sim    = (sel_n @ sel_n.T).astype(np.float64)   # (C, C)
    np.fill_diagonal(sim, 1.0)

    xyz      = _embed_3d(sim)                        # (C, 3)
    mst_edg  = _mst_edges(sim)

    # coreness = mean similarity to all other classes
    coreness = (sim.sum(axis=1) - 1.0) / (C - 1)
    core_min, core_max = coreness.min(), coreness.max()

    # node count = how many neurons each class owns
    n_owned  = np.array([(owner == c).sum() for c in range(C)])

    # ── Plotly traces ─────────────────────────────────────────────────────────
    traces = []

    # non-MST edges
    edge_x, edge_y, edge_z = [], [], []
    for i in range(C):
        for j in range(i + 1, C):
            if (i, j) in mst_edg:
                continue
            if sim[i, j] < EDGE_SIM_THRESH:
                continue
            for k in [i, j, None]:
                if k is None:
                    edge_x.append(None); edge_y.append(None); edge_z.append(None)
                else:
                    edge_x.append(float(xyz[k, 0]))
                    edge_y.append(float(xyz[k, 1]))
                    edge_z.append(float(xyz[k, 2]))

    traces.append(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode="lines",
        line=dict(color="rgba(80,80,200,0.35)", width=1.5),
        hoverinfo="none",
        name="other edges",
    ))

    # MST edges — one trace per edge, coloured by similarity
    sim_min = min(sim[i, j] for i, j in mst_edg) if mst_edg else 0
    sim_max = max(sim[i, j] for i, j in mst_edg) if mst_edg else 1
    for i, j in mst_edg:
        t = (sim[i, j] - sim_min) / (sim_max - sim_min + 1e-9)
        r = int(255 * (1 - t)); g = int(220 * t); b = 30
        colour = f"rgb({r},{g},{b})"
        traces.append(go.Scatter3d(
            x=[float(xyz[i, 0]), float(xyz[j, 0])],
            y=[float(xyz[i, 1]), float(xyz[j, 1])],
            z=[float(xyz[i, 2]), float(xyz[j, 2])],
            mode="lines",
            line=dict(color=colour, width=4),
            hoverinfo="none",
            name=f"MST {classes[i]}↔{classes[j]}",
            showlegend=False,
        ))

    # nodes
    node_size = 8 + 20 * (coreness - core_min) / (core_max - core_min + 1e-9)
    hover = [
        f"<b>{classes[c]}</b><br>"
        f"coreness: {coreness[c]:.3f}<br>"
        f"owned neurons: {n_owned[c]}<br>"
        f"poly median: {float(np.median(sim[c, [k for k in range(C) if k != c]])):.3f}"
        for c in range(C)
    ]

    traces.append(go.Scatter3d(
        x=xyz[:, 0].tolist(), y=xyz[:, 1].tolist(), z=xyz[:, 2].tolist(),
        mode="markers+text",
        marker=dict(
            size=node_size.tolist(),
            color=coreness.tolist(),
            colorscale="Viridis",
            reversescale=True,
            colorbar=dict(title="Coreness", thickness=12, len=0.5),
            line=dict(color="white", width=0.8),
        ),
        text=classes,
        textfont=dict(size=9, color="#111133"),
        textposition="top center",
        hovertext=hover,
        hoverinfo="text",
        name="AST classes",
    ))

    _axis = dict(
        showbackground=True, backgroundcolor="rgba(220,225,245,1.0)",
        gridcolor="#6666aa", gridwidth=1, showgrid=True,
        showline=True, linecolor="#111133", linewidth=2,
        showticklabels=True, tickfont=dict(size=9, color="#111133"),
        nticks=5, zeroline=True, zerolinecolor="#333366", zerolinewidth=2,
    )

    layout = go.Layout(
        title=dict(
            text="AST Class Similarity — Neuron Tuning Space<br>"
                 "<sup>Nodes = AST classes · Edge = shared neuron selectivity · "
                 "Proximity = similar neuron tuning · Colour = coreness</sup>",
            font=dict(size=14),
        ),
        scene=dict(
            xaxis=dict(**_axis, title=dict(text="MDS-1", font=dict(size=11))),
            yaxis=dict(**_axis, title=dict(text="MDS-2", font=dict(size=11))),
            zaxis=dict(**_axis, title=dict(text="MDS-3", font=dict(size=11))),
        ),
        updatemenus=[dict(
            type="buttons", direction="left",
            x=0.01, y=1.06, xanchor="left",
            buttons=[
                dict(label="All edges",
                     method="restyle",
                     args=[{"visible": [True] * len(traces)}]),
                dict(label="MST only",
                     method="restyle",
                     args=[{"visible": [False] + [True] * (len(traces) - 1)}]),
            ],
        )],
        margin=dict(l=0, r=0, t=60, b=0),
        legend=dict(x=0.01, y=0.98, font=dict(size=9)),
    )

    fig = go.Figure(data=traces, layout=layout)
    out = fig_dir / "class_similarity_3d.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"  saved {out}")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    in_dir  = Path(__file__).parent / args.in_dir
    out_dir = Path(__file__).parent / args.out_dir
    fig_dir = out_dir / "figures" / "polysemanticity"
    fig_dir.mkdir(parents=True, exist_ok=True)

    stem = args.stem
    fmt  = args.format

    print("Loading data …")
    sel_n   = np.load(out_dir / f"02g_{stem}_selectivity_neurons.npz")
    pm      = np.load(in_dir  / f"02_{stem}_purity_masks.npz")
    mlp     = np.load(in_dir  / f"01_{stem}_mlp_neurons.npy")
    meta    = json.load(open(in_dir / f"01_{stem}_meta.json"))
    j       = json.load(open(out_dir / f"02g_{stem}_class_neurons.json"))

    classes     = j["classes"]
    sel         = sel_n["sel"]              # (C, L, D)
    ast_pure_n  = pm["neurons_ast_pure"]    # (L, D)

    # best-layer selectivity and restrict to pure neurons
    sel_best     = sel.max(axis=1)          # (C, D)
    ast_pure_any = ast_pure_n.any(axis=0)   # (D,)
    pure_idx     = np.where(ast_pure_any)[0]
    sel_pure     = sel_best[:, pure_idx]    # (C, n_pure)
    owner        = sel_pure.argmax(axis=0)  # (n_pure,)

    print(f"  {len(classes)} classes · {len(pure_idx)} pure MLP neurons")

    print("Figure A: Assert neuron spotlight …")
    plot_assert_spotlight(mlp, meta, classes, fmt, args.dpi, fig_dir)

    print("Figure B: Selectivity waves …")
    plot_selectivity_waves(sel_pure, pure_idx, owner, classes,
                           fmt, args.dpi, fig_dir)

    print("Figure C: Polysemanticity distribution …")
    plot_polysemanticity_dist(sel_pure, pure_idx, classes,
                              fmt, args.dpi, fig_dir)

    print("Figure D: 3-D class similarity graph …")
    plot_class_similarity_3d(sel_pure, classes, owner, fig_dir)

    print("Done.")


if __name__ == "__main__":
    main()
