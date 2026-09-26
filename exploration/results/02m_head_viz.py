"""
02m_head_viz.py  (lives in results/)

High-impact visualisations of attention head class selectivity, inspired by
the "territory map" style of Lindsey et al. (2025).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIGURES PRODUCED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. head_tsne_profiles.pdf
   Global t-SNE of the 260 AST-pure heads in selectivity-profile space (each
   head is one point in R^C, reduced to 2D).  Coloured by owner class; labelled
   with "L{l}H{h}".  Same-colour clusters = heads that prefer the same AST type.

2. head_territory_L{l:02d}.pdf  (one per layer)
   "Territory map" for layer l — inspired by the head-trajectory figures in
   circuit-interpretability literature:
     • 2D PCA of the N×M_l attribution matrix (prompts as rows, pure heads as
       columns).  Each prompt becomes a point in this 2D space.
     • For each owned head, its owner class's prompts are highlighted as a
       coloured filled hull, labelled "L{l}H{h}".
     • The convex hull of ALL prompts is drawn as a thin grey "SUM" boundary.

3. head_territory_all.pdf
   Same as above but stacking all layers into one combined PCA space, with
   layer encoded by marker shape (or small-multiple panels in a 4×2 grid).

All colours are fixed to the global CLASS_COLOURS palette (index = sorted
class position) so every figure is consistent with 02l figures.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette — identical to 02l so all figures share the same mapping
# ─────────────────────────────────────────────────────────────────────────────

CLASS_COLOURS = [
    "#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00",
    "#a65628","#f781bf","#999999","#66c2a5","#fc8d62",
    "#8da0cb","#e78ac3","#a6d854","#ffd92f","#e5c494",
    "#b3b3b3","#1b9e77","#d95f02","#7570b3","#e7298a",
    "#66a61e","#e6ab02","#a6761d","#666666","#8dd3c7",
    "#ffffb3","#bebada","#fb8072","#80b1d3","#fdb462",
    "#b3de69","#fccde5",
]
UNOWNED_COL = "#aaaaaa"
LAYER_MARKERS = ["o","s","^","D","v","P","X","*"]  # one per layer

def cmap(classes: list) -> list[str]:
    return [CLASS_COLOURS[i % len(CLASS_COLOURS)] for i in range(len(classes))]

def hex_to_rgba(h: str, alpha: float = 1.0):
    r,g,b = [int(h.lstrip("#")[i:i+2],16)/255 for i in (0,2,4)]
    return (r,g,b,alpha)

def convex_hull_patch(points: np.ndarray, colour: str, alpha: float = 0.25,
                      edge_alpha: float = 0.7, lw: float = 1.2):
    """Return a filled matplotlib Polygon for the convex hull of `points`."""
    if len(points) < 3:
        return None
    try:
        hull = ConvexHull(points)
        verts = points[hull.vertices]
        patch = plt.Polygon(verts, closed=True,
                            facecolor=hex_to_rgba(colour, alpha),
                            edgecolor=hex_to_rgba(colour, edge_alpha),
                            linewidth=lw, zorder=2)
        return patch
    except Exception:
        return None

def legend_patches(classes, colours):
    return [Patch(facecolor=colours[c], label=classes[c])
            for c in range(len(classes))]

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(in_dir: Path, stem: str):
    head_attr  = np.load(in_dir / f"01_{stem}_head_attr.npy")   # (N, L, H)
    meta       = json.load(open(in_dir / f"01_{stem}_meta.json"))
    labels_str = np.array([m["ast_node"] for m in meta])
    classes    = sorted(set(labels_str))
    label_idx  = np.array([classes.index(s) for s in labels_str])
    # load precomputed selectivity / ownership from 02l
    d          = np.load(in_dir.parent / "results" / "output" /
                         f"02l_{stem}_head_sel.npz")
    sel        = d["sel"]           # (C, L, H)
    owner      = d["owner"]         # (L, H)
    ast_pure   = d["ast_pure"].astype(bool)  # (L, H)
    return head_attr, sel, owner, ast_pure, label_idx, classes

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: t-SNE of head selectivity profiles
# ─────────────────────────────────────────────────────────────────────────────

def plot_tsne_profiles(sel: np.ndarray, owner: np.ndarray,
                       ast_pure: np.ndarray, classes: list, colours: list,
                       fmt: str, dpi: int, fig_dir: Path):
    """
    Each AST-pure head = one point.  Feature vector = its C-dim selectivity profile.
    """
    C, L, H = sel.shape
    lh_list = [(l, h) for l in range(L) for h in range(H) if ast_pure[l, h]]
    if len(lh_list) < 10:
        return

    lh_arr   = np.array(lh_list)
    profiles = sel[:, lh_arr[:,0], lh_arr[:,1]].T    # (M, C)
    owners   = np.array([owner[l, h] for l, h in lh_list])

    # standardise across classes before t-SNE
    X = StandardScaler().fit_transform(profiles)
    perp = min(30, max(5, len(lh_list)//5))
    emb  = TSNE(n_components=2, perplexity=perp, random_state=42,
                max_iter=1000).fit_transform(X)   # (M, 2)

    fig, ax = plt.subplots(figsize=(11, 9))

    # background: all unowned in grey
    mask_un = owners < 0
    ax.scatter(emb[mask_un, 0], emb[mask_un, 1],
               c=UNOWNED_COL, s=28, alpha=0.5, linewidths=0, zorder=1,
               label="unowned")

    # owned heads: coloured, with label
    for c, cls in enumerate(classes):
        mask = owners == c
        if not mask.any():
            continue
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=colours[c], s=60, alpha=0.85, linewidths=0.4,
                   edgecolors="white", zorder=3)
        for i in np.where(mask)[0]:
            l, h = lh_arr[i]
            ax.text(emb[i,0]+0.5, emb[i,1], f"L{l}H{h}",
                    fontsize=5.5, color=colours[c], va="center",
                    fontweight="bold", zorder=4)

    ax.set_xlabel("t-SNE 1", fontsize=10)
    ax.set_ylabel("t-SNE 2", fontsize=10)
    ax.set_title("t-SNE of AST-pure attention heads\n"
                 "Each point = one head; position = selectivity profile in $\\mathbb{R}^C$;\n"
                 "colour = owner AST class", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False)

    handles = legend_patches(classes, colours)
    handles.append(Patch(facecolor=UNOWNED_COL, label="unowned"))
    ax.legend(handles=handles, fontsize=6.5, ncol=3, loc="lower left",
              frameon=True, framealpha=0.8)

    fig.tight_layout()
    out = fig_dir / f"head_tsne_profiles.{fmt}"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 & 3: territory maps
# ─────────────────────────────────────────────────────────────────────────────

def _territory_panel(ax, xy: np.ndarray, label_idx: np.ndarray,
                     head_indices: list[int],   # column indices into xy
                     lh_pairs: list[tuple],      # (l,h) for each column
                     owner_h: np.ndarray,        # owner per pure head at this layer
                     classes: list, colours: list,
                     title: str):
    """
    xy:          (N, 2)  PCA of prompt activations in this layer's pure-head space
    head_indices: list of column indices (not used here — xy is already 2D)
    owner_h:     (M,) owner class index per pure head
    """
    C = len(classes)
    N = xy.shape[0]

    # 1. Overall convex hull — "SUM" boundary
    try:
        hull_all = ConvexHull(xy)
        verts_all = xy[hull_all.vertices]
        ax.add_patch(plt.Polygon(np.vstack([verts_all, verts_all[0]]),
                                 closed=True,
                                 facecolor="none", edgecolor="0.5",
                                 linewidth=1.2, linestyle="--", zorder=1))
        # label SUM at topmost point
        top = verts_all[verts_all[:,1].argmax()]
        ax.text(top[0], top[1]+0.3, "SUM", fontsize=7, color="0.4",
                ha="center", va="bottom")
    except Exception:
        pass

    # 2. For each owned head: convex hull of its owner class's prompts
    #    colour = class colour, hull filled lightly, labelled with L{l}H{h}
    labelled_classes: set[int] = set()
    for idx, (l, h) in enumerate(lh_pairs):
        oc = owner_h[idx]
        if oc < 0:
            continue
        mask_c = label_idx == oc
        pts = xy[mask_c]
        patch = convex_hull_patch(pts, colours[oc], alpha=0.18, edge_alpha=0.7)
        if patch is not None:
            ax.add_patch(patch)
        # centroid label for this head
        cx, cy = pts.mean(axis=0) if len(pts) else (0, 0)
        # jitter slightly so multiple heads for same class don't stack
        rng_val = (l * 128 + h) / (9 * 128)
        jx = (rng_val - 0.5) * 1.5
        jy = (rng_val - 0.5) * 1.5
        ax.text(cx + jx, cy + jy, f"L{l}H{h}",
                fontsize=6, color=colours[oc], fontweight="bold",
                ha="center", va="center", zorder=5,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.6))

    # 3. Scatter all prompts, tiny, coloured by class
    for c in range(C):
        mask = label_idx == c
        ax.scatter(xy[mask, 0], xy[mask, 1],
                   c=colours[c], s=4, alpha=0.3, linewidths=0, zorder=2)

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=8, pad=3)
    for sp in ax.spines.values():
        sp.set_visible(False)


def plot_territory_per_layer(head_attr: np.ndarray, owner: np.ndarray,
                             ast_pure: np.ndarray, label_idx: np.ndarray,
                             classes: list, colours: list,
                             fmt: str, dpi: int, fig_dir: Path):
    N, L, H = head_attr.shape

    for l in range(L):
        pure_hs = [h for h in range(H) if ast_pure[l, h]]
        if len(pure_hs) < 3:
            print(f"  L{l}: skip territory (too few pure heads)")
            continue

        X_l = head_attr[:, l, :][:, pure_hs]   # (N, M_l)
        X_l = StandardScaler().fit_transform(X_l)
        pca = PCA(n_components=2, random_state=42)
        xy  = pca.fit_transform(X_l)            # (N, 2)
        var = pca.explained_variance_ratio_

        owner_h = np.array([owner[l, h] for h in pure_hs])  # (M_l,)
        lh_pairs = [(l, h) for h in pure_hs]

        fig, ax = plt.subplots(figsize=(7, 6))
        _territory_panel(ax, xy, label_idx,
                         list(range(len(pure_hs))), lh_pairs, owner_h,
                         classes, colours,
                         title=(f"Layer {l}  —  head attribution territory map\n"
                                f"PCA of {len(pure_hs)} pure heads  "
                                f"({var[0]*100:.1f}% + {var[1]*100:.1f}% var)"))

        # class legend (only classes present at this layer)
        present = set(owner_h[owner_h >= 0].tolist())
        handles = [Patch(facecolor=colours[c], label=classes[c])
                   for c in sorted(present)]
        ax.legend(handles=handles, fontsize=6, ncol=2, loc="lower right",
                  frameon=True, framealpha=0.85)

        fig.tight_layout()
        out = fig_dir / f"head_territory_L{l:02d}.{fmt}"
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out}")


def plot_territory_all_layers(head_attr: np.ndarray, owner: np.ndarray,
                              ast_pure: np.ndarray, label_idx: np.ndarray,
                              classes: list, colours: list,
                              fmt: str, dpi: int, fig_dir: Path):
    """
    4×2 grid of per-layer territory panels on one page.
    All panels share the same class→colour mapping and marker convention.
    """
    N, L, H = head_attr.shape
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes_flat = axes.flatten()

    for l in range(L):
        ax = axes_flat[l]
        pure_hs = [h for h in range(H) if ast_pure[l, h]]
        if len(pure_hs) < 3:
            ax.set_visible(False)
            continue

        X_l = head_attr[:, l, :][:, pure_hs]
        X_l = StandardScaler().fit_transform(X_l)
        pca = PCA(n_components=2, random_state=42)
        xy  = pca.fit_transform(X_l)
        var = pca.explained_variance_ratio_

        owner_h  = np.array([owner[l, h] for h in pure_hs])
        lh_pairs = [(l, h) for h in pure_hs]

        _territory_panel(ax, xy, label_idx,
                         list(range(len(pure_hs))), lh_pairs, owner_h,
                         classes, colours,
                         title=(f"Layer {l}  ({len(pure_hs)} pure heads, "
                                f"{var[0]*100:.0f}%+{var[1]*100:.0f}% var)"))

    # shared legend below the grid
    handles = legend_patches(classes, colours)
    fig.legend(handles=handles, fontsize=6.5, ncol=8,
               loc="lower center", bbox_to_anchor=(0.5, -0.02),
               frameon=False)
    fig.suptitle("Attention head attribution territory maps — all layers\n"
                 "Each coloured hull = prompts of that head's owner class; "
                 "dashed line = combined SUM hull",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    out = fig_dir / f"head_territory_all.{fmt}"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Sunburst — hierarchy: root → layer → AST class → individual heads
# ─────────────────────────────────────────────────────────────────────────────

def plot_sunburst(owner: np.ndarray, sel: np.ndarray, ast_pure: np.ndarray,
                  classes: list, colours: list, fig_dir: Path):
    """
    Interactive HTML sunburst (Plotly).
    Levels:
      centre  — "All heads"
      ring 1  — Layer (L0 … L7)
      ring 2  — AST class (coloured by class palette)
      ring 3  — Individual head (L{l}H{h}, coloured same as its class)
    Sector size = number of owned heads.
    Hover shows: layer, class, head index, peak selectivity z-score.
    """
    import plotly.graph_objects as go

    C, L, H = sel.shape

    ids, labels, parents, values, hover, colors_sun = [], [], [], [], [], []

    # root
    ids.append("root"); labels.append("All heads"); parents.append("")
    values.append(0); hover.append(""); colors_sun.append("#eeeeee")

    for l in range(L):
        layer_id  = f"L{l}"
        n_layer   = int((owner[l] >= 0).sum())
        ids.append(layer_id); labels.append(f"Layer {l}")
        parents.append("root"); values.append(n_layer)
        hover.append(f"Layer {l}: {n_layer} owned heads")
        # neutral grey per layer, lightening with depth
        grey = f"#{180 - l*10:02x}{180 - l*10:02x}{180 - l*10:02x}"
        colors_sun.append(grey)

        for c, cls in enumerate(classes):
            heads_lc = [h for h in range(H)
                        if ast_pure[l, h] and owner[l, h] == c]
            if not heads_lc:
                continue
            class_id = f"L{l}_{cls}"
            ids.append(class_id); labels.append(cls)
            parents.append(layer_id); values.append(len(heads_lc))
            hover.append(f"Layer {l} · {cls}: {len(heads_lc)} heads")
            colors_sun.append(colours[c])

            for h in heads_lc:
                head_id = f"L{l}H{h}"
                peak_sel = float(sel[c, l, h])
                ids.append(head_id); labels.append(f"H{h}")
                parents.append(class_id); values.append(1)
                hover.append(
                    f"<b>L{l}H{h}</b><br>Class: {cls}<br>"
                    f"Selectivity: {peak_sel:.3f}<br>Layer: {l}")
                colors_sun.append(colours[c])

    fig = go.Figure(go.Sunburst(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        customdata=hover,
        hovertemplate="%{customdata}<extra></extra>",
        marker=dict(colors=colors_sun, line=dict(color="white", width=0.8)),
        branchvalues="total",
        maxdepth=3,
        insidetextorientation="radial",
        textfont=dict(size=10),
    ))

    fig.update_layout(
        title=dict(
            text="Attention head ownership — sunburst<br>"
                 "<sup>Ring 1: layer · Ring 2: AST class · Ring 3: head index</sup>",
            x=0.5, font=dict(size=15)),
        margin=dict(t=80, l=10, r=10, b=10),
        width=850, height=850,
    )

    out = fig_dir / "head_sunburst.html"
    fig.write_html(str(out), include_plotlyjs=True)
    print(f"  saved {out}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--stem",    default="contrastive_stubs")
    p.add_argument("--in_dir",  default="../data_more")
    p.add_argument("--out_dir", default="../results/output")
    p.add_argument("--format",  default="pdf")
    p.add_argument("--dpi",     type=int, default=200)
    args = p.parse_args()

    in_dir  = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures" / "heads"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data …")
    head_attr, sel, owner, ast_pure, label_idx, classes = load_data(in_dir, args.stem)
    N, L, H = head_attr.shape
    C = len(classes)
    colours = cmap(classes)
    print(f"  {N} prompts · {L} layers · {H} heads · {C} classes")
    print(f"  pure heads: {ast_pure.sum()}")

    print("\nFigure 1: t-SNE of selectivity profiles …")
    plot_tsne_profiles(sel, owner, ast_pure, classes, colours,
                       args.format, args.dpi, fig_dir)

    print("Figure 2: per-layer territory maps …")
    plot_territory_per_layer(head_attr, owner, ast_pure, label_idx,
                             classes, colours, args.format, args.dpi, fig_dir)

    print("Figure 3: combined 4×2 territory map …")
    plot_territory_all_layers(head_attr, owner, ast_pure, label_idx,
                              classes, colours, args.format, args.dpi, fig_dir)

    print("Figure 4: sunburst …")
    plot_sunburst(owner, sel, ast_pure, classes, colours, fig_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
