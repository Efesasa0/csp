"""
02l_head_selectivity.py  (lives in results/)

Identifies attention heads uniquely responsible for individual AST node classes
using logit-attribution scores, and clusters groups of co-selective heads.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY WE DO THIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Head attribution scores (01_*_head_attr.npy, shape N×L×H) measure each head's
contribution to the final logit.  VP (02_*_vp_heads.npz) already partitions
each head's variance into unique_ast / unique_builtin / shared / unexplained.

This script adds the *class* dimension: which specific AST node type does each
head preferentially encode?  We compute the same z-score selectivity index
used in 02g for residual dims:

  sel(h, c, l) = (mean_attr[h, prompts_c, l] - mean_attr[h, all, l])
                 ─────────────────────────────────────────────────────
                             std_attr[h, all, l]  +  ε

A head at (l, h) is "class-owned" by c if:
  • it is AST-pure  (unique_ast > ua_thresh  AND  unique_ast/r2_joint > purity_frac)
  • sel(h, c, l) > sel_thresh
  • argmax_c sel(h,·,l) == c

Threshold rationale
  ua_thresh=0.10 corresponds to the top 30% of heads by unique_ast — the median
  head has UA≈0.03, so 0.10 screens out the long low-signal tail.
  purity_frac=0.70 means at least 70% of the head's total explained variance must
  come from AST identity alone (not shared with builtins); among heads passing
  ua_thresh this is the median purity, so we keep the stricter half.

FIGURES
  head_ownership_grid.pdf          8-layer × 128-head grid coloured by owning class
  head_selectivity_heatmap.pdf     (C × pure-heads) selectivity matrix
  head_top_per_class.pdf           bubble chart: top heads per class per layer
  head_cluster_heatmap.pdf         hierarchical cluster heatmap (all layers combined)
  head_dendrogram_L{l:02d}.pdf     per-layer dendrogram, leaves coloured by owner
  head_dendrogram_all.pdf          overall dendrogram across all layers

OUTPUTS  (under out_dir)
  02l_<stem>_head_sel.npz         sel (C, L, H)
  02l_<stem>_head_ownership.json  per-class per-layer lists of (layer, head_idx)
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
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Attention head selectivity per AST class")
    p.add_argument("--stem",        default="contrastive_stubs")
    p.add_argument("--in_dir",      default="../data_more")
    p.add_argument("--out_dir",     default="../results/output")
    p.add_argument("--ua_thresh",   type=float, default=0.10,
                   help="Min unique_ast for AST-pure head (top ~30%)")
    p.add_argument("--purity_frac", type=float, default=0.70,
                   help="Min unique_ast/r2_joint for AST-pure head (median purity)")
    p.add_argument("--sel_thresh",  type=float, default=0.4,
                   help="Min selectivity z-score for class ownership")
    p.add_argument("--min_samples", type=int, default=10)
    p.add_argument("--top_k",       type=int, default=5,
                   help="Top heads per class shown in top-heads figure")
    p.add_argument("--n_clusters",  type=int, default=15,
                   help="K for hierarchical cluster heatmap")
    p.add_argument("--format",      default="pdf")
    p.add_argument("--dpi",         type=int, default=200)
    return p.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(in_dir: Path, stem: str):
    head_attr  = np.load(in_dir / f"01_{stem}_head_attr.npy")   # (N, L, H)
    vp_h       = np.load(in_dir / f"02_{stem}_vp_heads.npz")
    meta       = json.load(open(in_dir / f"01_{stem}_meta.json"))
    labels_str = np.array([m["ast_node"] for m in meta])
    classes    = sorted(set(labels_str))
    label_idx  = np.array([classes.index(s) for s in labels_str])
    return head_attr, vp_h, labels_str, label_idx, classes

# ─────────────────────────────────────────────────────────────────────────────
# Selectivity
# ─────────────────────────────────────────────────────────────────────────────

def compute_selectivity(head_attr: np.ndarray, label_idx: np.ndarray,
                        classes: list, min_samples: int) -> np.ndarray:
    """Returns sel (C, L, H)."""
    N, L, H = head_attr.shape
    C = len(classes)
    mu_all  = head_attr.mean(axis=0)        # (L, H)
    std_all = head_attr.std(axis=0) + 1e-8  # (L, H)
    sel = np.zeros((C, L, H), dtype=np.float32)
    for c in range(C):
        mask = label_idx == c
        if mask.sum() < min_samples:
            continue
        mu_c = head_attr[mask].mean(axis=0)
        sel[c] = (mu_c - mu_all) / std_all
    return sel

# ─────────────────────────────────────────────────────────────────────────────
# Ownership
# ─────────────────────────────────────────────────────────────────────────────

def build_ownership(sel: np.ndarray, ast_pure: np.ndarray,
                    sel_thresh: float) -> np.ndarray:
    """Returns owner (L, H) — int class index, or -1 if unowned."""
    top_sel = sel.max(axis=0)   # (L, H)
    top_cls = sel.argmax(axis=0)
    owned   = ast_pure & (top_sel > sel_thresh)
    return np.where(owned, top_cls, -1).astype(np.int32)

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette  — fixed mapping: class index → colour, consistent everywhere
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
UNOWNED_COL = "#cccccc"

def class_colours(classes: list) -> list[str]:
    return [CLASS_COLOURS[i % len(CLASS_COLOURS)] for i in range(len(classes))]

def owner_colour(c: int, colours: list) -> str:
    return colours[c] if c >= 0 else UNOWNED_COL

def legend_patches(classes: list, colours: list) -> list:
    return [Patch(facecolor=colours[c], label=classes[c]) for c in range(len(classes))]

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: ownership grid (L × H)
# ─────────────────────────────────────────────────────────────────────────────

def plot_ownership_grid(owner: np.ndarray, classes: list, colours: list,
                        fmt: str, dpi: int, fig_dir: Path):
    L, H = owner.shape
    C    = len(classes)
    img  = np.ones((L, H, 4))
    for c in range(C):
        r, g, b = [int(colours[c].lstrip("#")[i:i+2], 16)/255 for i in (0,2,4)]
        img[owner == c] = [r, g, b, 1.0]
    img[owner == -1] = [0.88, 0.88, 0.88, 1.0]

    fig, ax = plt.subplots(figsize=(16, 3.5))
    ax.imshow(img, aspect="auto", interpolation="nearest",
              extent=[-0.5, H-0.5, L-0.5, -0.5])
    ax.set_xlabel("Head index", fontsize=10)
    ax.set_ylabel("Layer", fontsize=10)
    ax.set_yticks(range(L))
    ax.set_yticklabels([f"L{l}" for l in range(L)], fontsize=8)
    ax.set_xticks(range(0, H, 8))
    ax.tick_params(axis="x", labelsize=7)
    for l in range(L):
        ax.text(H+1, l, f"{(owner[l]>=0).sum()}", va="center", ha="left",
                fontsize=7, color="0.4")
    ax.legend(handles=legend_patches(classes, colours),
              loc="upper left", bbox_to_anchor=(1.04, 1.0),
              fontsize=6.5, ncol=2, frameon=False, borderaxespad=0)
    ax.set_title("Attention head ownership by AST class  "
                 "(grey = below pure threshold)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    out = fig_dir / f"head_ownership_grid.{fmt}"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: selectivity heatmap — all pure heads × classes
# ─────────────────────────────────────────────────────────────────────────────

def plot_selectivity_heatmap(sel: np.ndarray, owner: np.ndarray,
                             ast_pure: np.ndarray, classes: list, colours: list,
                             fmt: str, dpi: int, fig_dir: Path):
    C, L, H = sel.shape
    lh_list = [(l, h) for l in range(L) for h in range(H) if ast_pure[l, h]]
    lh_list.sort(key=lambda x: (owner[x[0],x[1]] if owner[x[0],x[1]]>=0 else C,
                                 x[0], x[1]))
    if not lh_list:
        return
    lh_arr = np.array(lh_list)
    M = len(lh_arr)
    mat  = sel[:, lh_arr[:,0], lh_arr[:,1]]   # (C, M)
    vmax = np.percentile(np.abs(mat), 99)

    fig_w = min(max(10, M*0.05), 24)
    fig, ax = plt.subplots(figsize=(fig_w, max(4, 0.28*C)))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_yticks(range(C))
    ax.set_yticklabels(classes, fontsize=7)
    ax.set_xlabel("AST-pure attention heads (sorted by owner class)", fontsize=9)
    ax.set_title("Head selectivity z-score  (columns sorted by owner class)", fontsize=10)
    xtick_pos = list(range(0, M, max(1, M//20)))
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels([f"L{lh_arr[i,0]}H{lh_arr[i,1]}" for i in xtick_pos],
                       fontsize=5.5, rotation=60, ha="right")
    # class-block separators
    cur = owner[lh_arr[0,0], lh_arr[0,1]]
    for i, (l, h) in enumerate(lh_list[1:], 1):
        o = owner[l, h]
        if o != cur:
            ax.axvline(i-0.5, color="white", lw=1.0)
            cur = o
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="Selectivity z-score")
    fig.tight_layout()
    out = fig_dir / f"head_selectivity_heatmap.{fmt}"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: bubble chart — top heads per class per layer
# ─────────────────────────────────────────────────────────────────────────────

def plot_top_heads_per_class(sel: np.ndarray, owner: np.ndarray,
                             ast_pure: np.ndarray, classes: list, colours: list,
                             top_k: int, fmt: str, dpi: int, fig_dir: Path):
    C, L, H = sel.shape
    fig, ax = plt.subplots(figsize=(10, max(6, 0.35*C)))
    for c, cls in enumerate(classes):
        rows = [(l, h, float(sel[c,l,h]))
                for l in range(L) for h in range(H)
                if ast_pure[l,h] and owner[l,h]==c]
        if not rows:
            continue
        rows.sort(key=lambda x: -x[2])
        rows = rows[:top_k*L]
        xs = [l for l,h,s in rows]
        ys = [c + (h/H - 0.5)*0.6 for l,h,s in rows]
        ss = [max(20, s*60) for l,h,s in rows]
        ax.scatter(xs, ys, s=ss, color=colours[c], alpha=0.75,
                   linewidths=0, zorder=3)
        by_layer: dict[int,tuple] = {}
        for l,h,s in rows:
            if l not in by_layer or s > by_layer[l][2]:
                by_layer[l] = (l,h,s)
        for l,h,s in by_layer.values():
            ax.text(l, c+(h/H-0.5)*0.6, f"H{h}", fontsize=5,
                    ha="center", va="bottom", color="0.2")
    ax.set_xticks(range(L))
    ax.set_xticklabels([f"L{l}" for l in range(L)], fontsize=9)
    ax.set_yticks(range(C))
    ax.set_yticklabels(classes, fontsize=7)
    for c, col in enumerate(colours):
        ax.axhline(c, color=col, lw=0.4, alpha=0.3)
    ax.set_xlabel("Layer", fontsize=10)
    ax.set_title("Top AST-selective attention heads per class\n"
                 "(bubble size ∝ selectivity z-score; label = head index)", fontsize=10)
    ax.grid(axis="x", ls="--", alpha=0.3)
    fig.tight_layout()
    out = fig_dir / f"head_top_per_class.{fmt}"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: hierarchical cluster heatmap (all layers)
# ─────────────────────────────────────────────────────────────────────────────

def plot_cluster_heatmap(sel: np.ndarray, owner: np.ndarray,
                         ast_pure: np.ndarray, classes: list, colours: list,
                         n_clusters: int, fmt: str, dpi: int, fig_dir: Path):
    C, L, H = sel.shape
    lh_list = [(l, h) for l in range(L) for h in range(H) if ast_pure[l,h]]
    if len(lh_list) < n_clusters:
        return
    lh_arr   = np.array(lh_list)
    profiles = sel[:, lh_arr[:,0], lh_arr[:,1]].T   # (M, C)
    Z        = linkage(profiles, method="ward")
    cl_labs  = fcluster(Z, n_clusters, criterion="maxclust")
    order    = np.argsort(cl_labs)
    mat      = profiles[order]
    vmax     = np.percentile(np.abs(mat), 99)
    owner_ord = np.array([owner[lh_arr[i,0], lh_arr[i,1]] for i in order])

    fig_h = min(max(8, len(lh_list)*0.045), 22)
    fig, axes = plt.subplots(1, 2, figsize=(12, fig_h),
                             gridspec_kw={"width_ratios":[0.04, 1]})
    strip = np.array([matplotlib.colors.to_rgba(owner_colour(c, colours))
                      for c in owner_ord])
    axes[0].imshow(strip[:,np.newaxis,:], aspect="auto", interpolation="nearest",
                   extent=[-0.5,0.5, len(order)-0.5, -0.5])
    axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[0].set_ylabel("AST-pure attention heads (Ward clustering)", fontsize=9)
    im = axes[1].imshow(mat, aspect="auto", cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax, interpolation="nearest",
                        extent=[-0.5, C-0.5, len(order)-0.5, -0.5])
    axes[1].set_xticks(range(C))
    axes[1].set_xticklabels(classes, rotation=45, ha="right", fontsize=7)
    axes[1].set_yticks([])
    axes[1].set_xlabel("AST class", fontsize=9)
    axes[1].set_title(f"Head selectivity clusters (Ward, k={n_clusters})\n"
                      "Left strip = owner class colour", fontsize=10)
    prev = cl_labs[order[0]]
    for i, oi in enumerate(order):
        ci = cl_labs[oi]
        if ci != prev:
            axes[1].axhline(i-0.5, color="white", lw=1.0)
            prev = ci
    plt.colorbar(im, ax=axes[1], fraction=0.02, pad=0.01,
                 label="Selectivity z-score")
    fig.tight_layout()
    out = fig_dir / f"head_cluster_heatmap.{fmt}"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: dendrogram (per layer + overall)
# ─────────────────────────────────────────────────────────────────────────────

def _dendrogram_one(profiles: np.ndarray, labels: list[str],
                    leaf_colours: list[str], title: str,
                    fmt: str, dpi: int, out_path: Path,
                    classes: list, colours: list):
    """
    Draw a single dendrogram.
    profiles:     (M, C)  selectivity vectors for each head
    labels:       (M,)    tick labels  e.g. "L3H77"
    leaf_colours: (M,)    colour string per leaf (owner class colour)
    """
    M = len(labels)
    if M < 3:
        return

    Z = linkage(profiles, method="ward")

    # height: 0.22in per leaf, min 4
    fig_h = min(max(4.0, M * 0.22), 26)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    ddata = dendrogram(
        Z,
        ax=ax,
        orientation="left",
        labels=labels,
        color_threshold=0,          # disable scipy's own colouring
        above_threshold_color="0.5",
        leaf_font_size=6,
        no_plot=False,
    )

    # Colour the y-tick labels (leaves) by owner class
    # scipy's dendrogram reorders leaves; ivl gives the label order bottom→top
    leaf_order = ddata["ivl"]   # list of label strings in display order
    label_to_colour = dict(zip(labels, leaf_colours))
    for tick_label in ax.get_yticklabels():
        txt = tick_label.get_text()
        col = label_to_colour.get(txt, UNOWNED_COL)
        tick_label.set_color(col)

    ax.set_xlabel("Ward linkage distance", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    handles = legend_patches(classes, colours)
    handles.append(Patch(facecolor=UNOWNED_COL, label="unowned"))
    ax.legend(handles=handles, loc="lower right", fontsize=6,
              ncol=2, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_dendrograms(sel: np.ndarray, owner: np.ndarray,
                     ast_pure: np.ndarray, classes: list, colours: list,
                     fmt: str, dpi: int, fig_dir: Path):
    C, L, H = sel.shape

    all_profiles: list[np.ndarray] = []
    all_labels:   list[str]        = []
    all_lcolours: list[str]        = []

    for l in range(L):
        # heads that are pure at this layer
        hs = [h for h in range(H) if ast_pure[l, h]]
        if len(hs) < 3:
            print(f"  L{l}: too few pure heads ({len(hs)}), skipping dendrogram")
            continue

        profiles    = sel[:, l, hs].T              # (M, C)
        labels      = [f"L{l}H{h}" for h in hs]
        leaf_cols   = [owner_colour(owner[l, h], colours) for h in hs]

        # collect for overall dendrogram
        all_profiles.append(profiles)
        all_labels.extend(labels)
        all_lcolours.extend(leaf_cols)

        _dendrogram_one(
            profiles, labels, leaf_cols,
            title=f"Layer {l}  — head selectivity dendrogram  "
                  f"({len(hs)} AST-pure heads)",
            fmt=fmt, dpi=dpi,
            out_path=fig_dir / f"head_dendrogram_L{l:02d}.{fmt}",
            classes=classes, colours=colours,
        )

    # Overall dendrogram across all layers
    if len(all_labels) >= 3:
        all_mat = np.vstack(all_profiles)
        _dendrogram_one(
            all_mat, all_labels, all_lcolours,
            title=f"All layers — head selectivity dendrogram  "
                  f"({len(all_labels)} AST-pure heads)",
            fmt=fmt, dpi=dpi,
            out_path=fig_dir / f"head_dendrogram_all.{fmt}",
            classes=classes, colours=colours,
        )

# ─────────────────────────────────────────────────────────────────────────────
# Ownership JSON
# ─────────────────────────────────────────────────────────────────────────────

def build_ownership_json(owner: np.ndarray, sel: np.ndarray,
                         ua: np.ndarray, classes: list) -> dict:
    C, L, H = sel.shape
    result: dict = {cls: {} for cls in classes}
    for c, cls in enumerate(classes):
        for l in range(L):
            heads = [{"head": int(h), "layer": int(l),
                      "selectivity": float(sel[c,l,h]),
                      "unique_ast":  float(ua[l,h])}
                     for h in range(H) if owner[l,h]==c]
            heads.sort(key=lambda x: -x["selectivity"])
            if heads:
                result[cls][f"L{l}"] = heads
    return result

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    in_dir  = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures" / "heads"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data …")
    head_attr, vp_h, labels_str, label_idx, classes = load_data(in_dir, args.stem)
    N, L, H = head_attr.shape
    C = len(classes)
    print(f"  {N} prompts · {L} layers · {H} heads · {C} classes")

    ua      = vp_h["unique_ast"]    # (L, H)
    rj      = vp_h["r2_joint"]
    purity  = np.where(rj > 1e-6, ua / rj, 0.0)
    ast_pure = (ua > args.ua_thresh) & (purity > args.purity_frac)
    print(f"  AST-pure heads (UA>{args.ua_thresh}, purity>{args.purity_frac}): "
          f"{ast_pure.sum()} / {L*H}")
    for l in range(L):
        print(f"    L{l}: {ast_pure[l].sum()} pure heads")

    print("\nComputing selectivity …")
    sel   = compute_selectivity(head_attr, label_idx, classes, args.min_samples)
    owner = build_ownership(sel, ast_pure, args.sel_thresh)

    owned_total = (owner >= 0).sum()
    print(f"  Class-owned heads: {owned_total} / {ast_pure.sum()} pure")
    for c, cls in enumerate(classes):
        n = (owner == c).sum()
        if n > 0:
            print(f"    {cls:20s}: {n} heads")

    # Save arrays
    np.savez_compressed(out_dir / f"02l_{args.stem}_head_sel.npz",
                        sel=sel, owner=owner, ast_pure=ast_pure.astype(np.uint8))
    print(f"  saved {out_dir}/02l_{args.stem}_head_sel.npz")

    ownership_json = build_ownership_json(owner, sel, ua, classes)
    with open(out_dir / f"02l_{args.stem}_head_ownership.json", "w") as f:
        json.dump(ownership_json, f, indent=2)
    print(f"  saved {out_dir}/02l_{args.stem}_head_ownership.json")

    colours = class_colours(classes)

    print("\nPlotting ownership grid …")
    plot_ownership_grid(owner, classes, colours, args.format, args.dpi, fig_dir)

    print("Plotting selectivity heatmap …")
    plot_selectivity_heatmap(sel, owner, ast_pure, classes, colours,
                             args.format, args.dpi, fig_dir)

    print("Plotting top heads per class …")
    plot_top_heads_per_class(sel, owner, ast_pure, classes, colours,
                             args.top_k, args.format, args.dpi, fig_dir)

    print("Plotting cluster heatmap …")
    plot_cluster_heatmap(sel, owner, ast_pure, classes, colours,
                         args.n_clusters, args.format, args.dpi, fig_dir)

    print("Plotting dendrograms (per layer + overall) …")
    plot_dendrograms(sel, owner, ast_pure, classes, colours,
                     args.format, args.dpi, fig_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
