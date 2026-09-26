"""
02m_head_viz.py  (submission version)

Produces the diagonal-strip ownership grid for all AST-pure attention heads.

FIGURE PRODUCED
  head_all_pure_grid_diagonal.pdf
    L x H ownership grid where multi-owned cells are split by diagonal lines.
    Colour = owning AST class; diagonal strips = polysemantic (multi-owned) heads.

Usage
-----
  python 02m_head_viz.py --stem contrastive_stubs --in_dir ../data_more
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch


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

def cmap(classes: list) -> list[str]:
    return [CLASS_COLOURS[i % len(CLASS_COLOURS)] for i in range(len(classes))]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(in_dir: Path, stem: str):
    """Load head_attr meta and precomputed selectivity/ownership from 02l."""
    meta       = json.load(open(in_dir / f"01_{stem}_meta.json"))
    labels_str = np.array([m["ast_node"] for m in meta])
    classes    = sorted(set(labels_str))
    # load precomputed selectivity / ownership from 02l
    d        = np.load(in_dir / f"02l_{stem}_head_sel.npz")
    sel      = d["sel"]                       # (C, L, H)
    ast_pure = d["ast_pure"].astype(bool)     # (L, H)
    return sel, ast_pure, classes


# ─────────────────────────────────────────────────────────────────────────────
# Figure: diagonal-strip ownership grid
# ─────────────────────────────────────────────────────────────────────────────

def plot_all_pure_grid_diagonal(sel: np.ndarray, ast_pure: np.ndarray,
                                classes: list, colours: list,
                                sel_thresh: float, fmt: str, dpi: int,
                                fig_dir: Path,
                                opacity: bool = False,
                                alpha_cap: float = 4.0,
                                gamma: float = 1.0):
    """
    L x H grid layout.  All AST-pure heads with >= 1 owner at sel_thresh shown.
    Monosemantic heads  -> solid colour cell.
    Multi-owned heads   -> cell split into diagonal bands (one per owner,
                           dominant -> least dominant, upper-left -> lower-right).
    Grey background     = below threshold / not AST-pure.
    """
    C, L, H = sel.shape

    def hex_to_rgb(hx):
        hx = hx.lstrip("#")
        return tuple(int(hx[i:i+2], 16) / 255 for i in (0, 2, 4))

    def clip_poly(poly, a, b, c):
        """Sutherland-Hodgman: clip CCW convex polygon by ax+by <= c."""
        result = []
        n = len(poly)
        for i in range(n):
            p1, p2 = poly[i], poly[(i + 1) % n]
            d1 = a * p1[0] + b * p1[1] - c
            d2 = a * p2[0] + b * p2[1] - c
            if d1 <= 0:
                result.append(p1)
            if d1 * d2 < 0:
                t = d1 / (d1 - d2)
                result.append((p1[0] + t * (p2[0] - p1[0]),
                                p1[1] + t * (p2[1] - p1[1])))
        return result

    def diagonal_strip(k, n, direction):
        """Polygon vertices for the k-th of n diagonal strips in [0,1]^2."""
        square = [(0, 0), (1, 0), (1, 1), (0, 1)]   # CCW
        if direction == 0:
            a = 2 * k / n
            b = 2 * (k + 1) / n
            poly = clip_poly(square,  1,  1,  b)
            poly = clip_poly(poly,   -1, -1, -a)
        else:
            a = -1 + 2 * k / n
            b = -1 + 2 * (k + 1) / n
            poly = clip_poly(square,  1, -1,  b)
            poly = clip_poly(poly,   -1,  1, -a)
        return poly

    # Build owner lists: list of (ci, sel_val) sorted by decreasing selectivity
    owners_grid: dict[tuple, list[tuple]] = {}
    for l in range(L):
        for h in range(H):
            if not ast_pure[l, h]:
                continue
            owned = sorted(
                [(ci, float(sel[ci, l, h])) for ci in range(C)
                 if float(sel[ci, l, h]) > sel_thresh],
                key=lambda x: -x[1]
            )
            if owned:
                owners_grid[(l, h)] = owned

    # Alpha scale (used only when opacity=True):
    #   sel >= alpha_cap  -> alpha 1.0 (fully opaque)
    #   sel == sel_thresh -> alpha 0.20 (still visible)
    ALPHA_MIN = 0.20

    def sel_to_alpha(s: float) -> float:
        t = min(1.0, s / alpha_cap) ** gamma
        return ALPHA_MIN + (1.0 - ALPHA_MIN) * t

    fig, ax = plt.subplots(figsize=(16, 3.5))
    ax.set_facecolor("#e0e0e0")
    ax.set_xlim(-0.5, H - 0.5)
    ax.set_ylim(L - 0.5, -0.5)   # y inverted so L0 is at top

    for (l, h), owners in owners_grid.items():
        n = len(owners)
        direction = (l + h) % 2
        for k, (ci, sel_val) in enumerate(owners):
            verts = diagonal_strip(k, n, direction)
            verts_t = [(h - 0.5 + x, l - 0.5 + y) for x, y in verts]
            if verts_t:
                r, g, b = hex_to_rgb(colours[ci])
                a = sel_to_alpha(sel_val) if opacity else 1.0
                ax.add_patch(plt.Polygon(
                    verts_t, closed=True,
                    facecolor=(r, g, b, a), edgecolor="none"
                ))

    ax.set_xlabel("Head index", fontsize=10)
    ax.set_ylabel("Layer", fontsize=10)
    ax.set_yticks(range(L))
    ax.set_yticklabels([f"L{l}" for l in range(L)], fontsize=8)
    ax.set_xticks(range(0, H, 8))
    ax.tick_params(axis="x", labelsize=7)

    for l in range(L):
        n_l = sum(1 for (ll, _) in owners_grid if ll == l)
        ax.text(H + 1, l, f"{n_l}", va="center", ha="left", fontsize=7, color="0.4")

    handles = [mpatches.Patch(facecolor=colours[ci], label=cls)
               for ci, cls in enumerate(classes)]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.04, 1.0),
              fontsize=6.5, ncol=2, frameon=False, borderaxespad=0)

    n_multi = sum(1 for v in owners_grid.values() if len(v) > 1)
    ax.set_title(
        f"Attention head ownership by AST class  "
        f"(grey = below SEL threshold={sel_thresh})  "
        f"— diagonal strips = multi-owned  ({n_multi}/{len(owners_grid)} polysemantic)",
        fontsize=9)

    fig.tight_layout(rect=[0, 0, 0.82, 1])
    out = fig_dir / f"head_all_pure_grid_diagonal.{fmt}"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stem",      default="contrastive_stubs")
    p.add_argument("--in_dir",    default="../data_more")
    p.add_argument("--out_dir",   default="../results/output")
    p.add_argument("--sel_thresh", type=float, default=0.4)
    p.add_argument("--format",    default="pdf")
    p.add_argument("--dpi",       type=int, default=200)
    p.add_argument("--opacity",   action="store_true",
                   help="Scale cell opacity by selectivity strength")
    p.add_argument("--alpha_cap", type=float, default=4.0,
                   help="sel value that maps to full opacity (default: 4.0)")
    p.add_argument("--gamma",     type=float, default=1.0,
                   help="Gamma for opacity curve: >1 emphasises high-sel cells, "
                        "<1 lifts weak cells (default: 1.0 = linear)")
    args = p.parse_args()

    in_dir  = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures" / "heads"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data …")
    sel, ast_pure, classes = load_data(in_dir, args.stem)
    C, L, H = sel.shape
    colours = cmap(classes)
    print(f"  {L} layers · {H} heads · {C} classes · {ast_pure.sum()} pure heads")

    print("Plotting diagonal ownership grid …")
    plot_all_pure_grid_diagonal(sel, ast_pure, classes, colours,
                                sel_thresh=args.sel_thresh,
                                fmt=args.format, dpi=args.dpi,
                                fig_dir=fig_dir,
                                opacity=args.opacity,
                                alpha_cap=args.alpha_cap,
                                gamma=args.gamma)

    print("\nDone.")


import argparse  # noqa: E402 — imported here so main() can reference it

if __name__ == "__main__":
    main()
