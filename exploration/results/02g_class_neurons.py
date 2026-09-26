"""
02g_class_neurons.py  (lives in results/)

Identifies which neurons are responsible for individual AST node classes,
separates them from generic "core" AST neurons, validates the assignment
causally via offline activation patching, and renders an AST Neuron Atlas.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY WE DO THIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 02 identified neurons with high unique_ast variance — but that score is
pooled across ALL 32 AST classes.  A neuron with high unique_ast could be:

  Core AST neuron      fires for every AST class (encodes "this is AST"
                       generically, not which class)
  Class-specific neuron fires selectively for one class (FunctionDef, For, …)

This script separates the two via a per-class selectivity index:

  sel(u, c, l) = (mean_act[u, prompts_c, l] − mean_act[u, all, l])
                 ─────────────────────────────────────────────────
                           std_act[u, all, l]  +  ε

  core neuron:          ast_pure  AND  std_c( sel(u,·,l) ) < core_thresh
  class-specific neuron: sel(u,c,l) > sel_thresh  AND  argmax_c sel == c
                          AND  NOT core

CAUSAL VALIDATION
  Offline activation patching at the residual stream level.  For each
  AST class c, replace the class-specific dims in a target prompt (class c')
  with the values from a source prompt (class c).  Measure how much the
  representation shifts toward class c's centroid.  Compare three conditions:

    specific : patch class-c-specific dims
    core     : patch core AST dims (same count, randomly sampled)
    random   : patch random non-AST dims (same count)

  If specific >> core ≈ random, those neurons causally encode class identity.

FIGURES
  neuron_atlas_L{ll:02d}.{fmt}   per-layer sorted selectivity heatmap
  neuron_atlas_all.{fmt}          headline figure: aggregated across layers
  patch_causal.{fmt}              patch-shift boxplot (3 conditions)

OUTPUTS  (under out_dir)
  02g_<stem>_selectivity_residual.npz   sel  (C, L+1, H)
  02g_<stem>_selectivity_neurons.npz    sel  (C, L, mlp_dim)
  02g_<stem>_class_neurons.json         per-class neuron lists (per layer + all)
  02g_<stem>_patch_results.npz          patch-shift arrays

Usage
-----
  python 02g_class_neurons.py
  python 02g_class_neurons.py --stem contrastive_stubs --in_dir ../data_more
  python 02g_class_neurons.py --sel_thresh 0.4 --top_k 25 --format pdf
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
from matplotlib.patches import Patch

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Class-specific neuron atlas")
    p.add_argument("--stem",       default="contrastive_stubs")
    p.add_argument("--in_dir",     default="../data_more",
                   help="Directory with 01_/02_/03_ output files")
    p.add_argument("--out_dir",    default="../results/output",
                   help="Root output directory")
    p.add_argument("--sel_thresh", type=float, default=0.5,
                   help="Min z-score for a neuron to be class-selective")
    p.add_argument("--core_thresh", type=float, default=0.25,
                   help="Max std across classes to be 'core' (not class-specific)")
    p.add_argument("--top_k",      type=int, default=20,
                   help="Top-K neurons per class shown in atlas")
    p.add_argument("--min_samples", type=int, default=10,
                   help="Minimum prompts per class; classes below this are skipped")
    p.add_argument("--n_patch",    type=int, default=40,
                   help="Source/target pairs per class for causal patching")
    p.add_argument("--patch_layer", type=int, default=-1,
                   help="Residual-stream layer index for patching (-1 = last)")
    p.add_argument("--format",     default="pdf",
                   help="Figure format: pdf or png")
    p.add_argument("--dpi",        type=int, default=200)
    return p.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(in_dir: Path, stem: str):
    resid  = np.load(in_dir / f"01_{stem}_residual_all.npy")    # (N, L+1, H)
    mlp    = np.load(in_dir / f"01_{stem}_mlp_neurons.npy")     # (N, L, mlp_dim)
    meta   = json.load(open(in_dir / f"01_{stem}_meta.json"))
    pm     = np.load(in_dir / f"02_{stem}_purity_masks.npz")
    vp_r   = np.load(in_dir / f"02_{stem}_vp_residual.npz")
    vp_n   = np.load(in_dir / f"02_{stem}_vp_mlp_neurons.npz")
    return resid, mlp, meta, pm, vp_r, vp_n

# ─────────────────────────────────────────────────────────────────────────────
# Selectivity
# ─────────────────────────────────────────────────────────────────────────────

def compute_selectivity(acts: np.ndarray, labels: np.ndarray,
                        classes: list[str], min_samples: int) -> np.ndarray:
    """
    acts:   (N, D)  activations at one layer
    labels: (N,)    int class index
    Returns sel (C, D) — z-score of each class's mean relative to the global dist.
    Classes with fewer than min_samples prompts are set to 0.
    """
    C = len(classes)
    D = acts.shape[1]
    mu_all  = acts.mean(axis=0)          # (D,)
    std_all = acts.std(axis=0) + 1e-8    # (D,)
    sel = np.zeros((C, D), dtype=np.float32)
    for c in range(C):
        mask = labels == c
        if mask.sum() < min_samples:
            continue
        mu_c = acts[mask].mean(axis=0)
        sel[c] = (mu_c - mu_all) / std_all
    return sel


def compute_selectivity_all_layers(acts_all: np.ndarray, labels: np.ndarray,
                                   classes: list[str], min_samples: int) -> np.ndarray:
    """
    acts_all: (N, L, D)
    Returns sel_all (C, L, D)
    """
    N, L, D = acts_all.shape
    C = len(classes)
    sel_all = np.zeros((C, L, D), dtype=np.float32)
    for l in range(L):
        sel_all[:, l, :] = compute_selectivity(acts_all[:, l, :], labels,
                                               classes, min_samples)
    return sel_all

# ─────────────────────────────────────────────────────────────────────────────
# Core / class-specific split
# ─────────────────────────────────────────────────────────────────────────────

def split_core_class(sel_all: np.ndarray,
                     ast_pure_mask: np.ndarray,
                     sel_thresh: float,
                     core_thresh: float):
    """
    sel_all:      (C, L, D)
    ast_pure_mask:(L, D)  bool
    Returns:
      core      (L, D)  bool  — ast_pure AND selectivity is uniform across classes
      class_nrn (C, L, D) bool — selective for exactly class c, not core
    """
    C, L, D = sel_all.shape

    # core: flagged ast_pure by VP, but fires similarly for all AST classes
    sel_std = sel_all.std(axis=0)          # (L, D)  std across classes
    core = ast_pure_mask & (sel_std < core_thresh)

    # class-specific: argmax class owns it, passes threshold, not core
    sel_max    = sel_all.max(axis=0)       # (L, D)
    sel_argmax = sel_all.argmax(axis=0)    # (L, D)  int

    class_nrn = np.zeros((C, L, D), dtype=bool)
    for c in range(C):
        class_nrn[c] = (sel_argmax == c) & (sel_max > sel_thresh) & ~core

    return core, class_nrn

# ─────────────────────────────────────────────────────────────────────────────
# Causal patching
# ─────────────────────────────────────────────────────────────────────────────

def _cos_sim_1d(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def run_patching(resid_all: np.ndarray,
                 labels: np.ndarray,
                 classes: list[str],
                 class_nrn_r: np.ndarray,
                 core_r: np.ndarray,
                 patch_layer: int,
                 n_patch: int,
                 rng: np.random.Generator) -> dict:
    """
    Offline activation patching on the residual stream.

    For each AST class c:
      - Draw n_patch (source, target) pairs: source from class c, target NOT c
      - Three conditions:
          specific : patch class-c-specific dims (class_nrn_r[c, patch_layer])
          core     : patch same-count randomly-sampled core dims
          random   : patch same-count randomly-sampled non-AST dims

    Returns dict with arrays:
      shifts_specific (C, n_patch)
      shifts_core     (C, n_patch)
      shifts_random   (C, n_patch)
      counts          (C,)  actual pairs evaluated per class
    """
    N, n_layers, H = resid_all.shape
    C = len(classes)
    l = patch_layer % n_layers

    acts_l = resid_all[:, l, :]      # (N, H)

    # class centroids at patch layer
    centroids = np.zeros((C, H), dtype=np.float32)
    for c in range(C):
        mask = labels == c
        if mask.sum() > 0:
            centroids[c] = acts_l[mask].mean(axis=0)

    # non-AST dims: not core, not any class-specific
    any_class = class_nrn_r[:, l, :].any(axis=0)  # (H,)
    non_ast_dims = np.where(~any_class & ~core_r[l])[0]

    shifts_specific = np.full((C, n_patch), np.nan, dtype=np.float32)
    shifts_core     = np.full((C, n_patch), np.nan, dtype=np.float32)
    shifts_random   = np.full((C, n_patch), np.nan, dtype=np.float32)
    counts = np.zeros(C, dtype=int)

    core_dims_l = np.where(core_r[l])[0]

    for c in range(C):
        src_idx = np.where(labels == c)[0]
        tgt_idx = np.where(labels != c)[0]
        if len(src_idx) < 2 or len(tgt_idx) < 2:
            continue

        specific_dims = np.where(class_nrn_r[c, l])[0]
        if len(specific_dims) == 0:
            continue

        k = len(specific_dims)   # match count across conditions

        # sample pairs
        n = min(n_patch, len(src_idx), len(tgt_idx))
        src_sample = rng.choice(src_idx, size=n, replace=False)
        tgt_sample = rng.choice(tgt_idx, size=n, replace=False)
        counts[c] = n

        centroid_c = centroids[c]

        for i in range(n):
            src_act = acts_l[src_sample[i]]
            tgt_act = acts_l[tgt_sample[i]]

            orig_sim = _cos_sim_1d(tgt_act, centroid_c)

            # specific
            patched = tgt_act.copy()
            patched[specific_dims] = src_act[specific_dims]
            shifts_specific[c, i] = _cos_sim_1d(patched, centroid_c) - orig_sim

            # core (same count k, randomly sampled from core dims)
            if len(core_dims_l) >= k:
                sel_core = rng.choice(core_dims_l, size=k, replace=False)
            else:
                sel_core = core_dims_l
            patched = tgt_act.copy()
            patched[sel_core] = src_act[sel_core]
            shifts_core[c, i] = _cos_sim_1d(patched, centroid_c) - orig_sim

            # random non-AST dims
            if len(non_ast_dims) >= k:
                sel_rand = rng.choice(non_ast_dims, size=k, replace=False)
            else:
                sel_rand = non_ast_dims
            patched = tgt_act.copy()
            patched[sel_rand] = src_act[sel_rand]
            shifts_random[c, i] = _cos_sim_1d(patched, centroid_c) - orig_sim

    return dict(
        shifts_specific=shifts_specific,
        shifts_core=shifts_core,
        shifts_random=shifts_random,
        counts=counts,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

FS_TITLE   = 11
FS_TICK    = 8
FS_SUPTITLE = 13
FS_CBAR    = 10
FS_ANNOT   = 7

# Grid layout constants — square grids with small black padding
# Nearest ceil(sqrt(D))² grid; padded cells rendered black so they are
# visually distinct from white (unowned) and grey (core) neurons.
GRID_SHAPE: dict[str, tuple[int, int]] = {
    "residual": (46, 46),    # 2048 → 46×46=2116, 68 cells padded black
    "mlp":      (91, 91),    # 8192 → 91×91=8281, 89 cells padded black
}
CELL_SIZE: dict[str, tuple[float, float]] = {
    "residual": (3.2, 3.2),
    "mlp":      (3.2, 3.2),
}
LEFT_IN   = 0.20
TOP_IN    = 0.75
LEGEND_IN = 1.80   # right margin reserved for the class-colour legend
N_OWN_COLS = 3     # layer columns in the ownership grid


def _atlas_heatmap(sel_CD: np.ndarray, classes: list[str],
                   core_mask_D: np.ndarray,
                   top_k: int, title: str, fmt: str, dpi: int,
                   out_path: Path):
    """
    sel_CD:     (C, D) — selectivity (one layer or aggregated)
    core_mask_D:(D,)   — which dims are core (marked with hatching column)
    Selects top_k neurons per class, deduplicates, sorts by argmax class,
    renders sorted heatmap with block structure.
    """
    C, D = sel_CD.shape

    # select neurons
    selected = set()
    for c in range(C):
        top = np.argsort(sel_CD[c])[-top_k:]
        selected.update(top.tolist())
    selected = np.array(sorted(selected))

    if len(selected) == 0:
        print(f"  [atlas] no neurons selected for {title}, skipping")
        return

    sub = sel_CD[:, selected]          # (C, K)
    is_core_sub = core_mask_D[selected]  # (K,) bool

    # sort columns by argmax class
    order = np.argsort(sub.argmax(axis=0), kind="stable")
    sub = sub[:, order]
    is_core_sub = is_core_sub[order]

    # figure dimensions
    K = sub.shape[1]
    fig_w = max(8.0, 0.12 * K + 2.0)
    fig_h = max(5.0, 0.22 * C + 1.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    vmax = max(2.0, float(np.abs(sub).max()))
    im = ax.imshow(sub, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")

    # mark core columns with a subtle overlay line
    core_col_idx = np.where(is_core_sub)[0]
    for ci in core_col_idx:
        ax.axvline(ci, color="#888800", lw=0.4, alpha=0.6)

    ax.set_yticks(range(C))
    ax.set_yticklabels(classes, fontsize=FS_TICK)
    ax.set_xlabel("Neurons (sorted by argmax class)", fontsize=FS_TITLE)
    ax.set_ylabel("AST node class", fontsize=FS_TITLE)
    ax.set_title(title, fontsize=FS_SUPTITLE, pad=8)

    # colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Selectivity  (z-score above global mean)", fontsize=FS_CBAR)

    # legend
    handles = [
        Patch(facecolor="#888800", alpha=0.6, label="core AST neuron"),
        Patch(facecolor="#d73027", label=f"class-selective  (z > {0:.1f})"),
    ]
    ax.legend(handles=handles, loc="upper left",
              fontsize=FS_ANNOT, framealpha=0.85)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_atlas_per_layer(sel_all: np.ndarray, classes: list[str],
                         core: np.ndarray, top_k: int,
                         qty_label: str, fmt: str, dpi: int,
                         fig_dir: Path):
    """sel_all (C, L, D), core (L, D)"""
    C, L, D = sel_all.shape
    for l in range(L):
        title = f"AST Neuron Atlas — {qty_label}  Layer {l}"
        out = fig_dir / f"neuron_atlas_{qty_label}_L{l:02d}.{fmt}"
        _atlas_heatmap(sel_all[:, l, :], classes, core[l],
                       top_k, title, fmt, dpi, out)


def plot_atlas_all_layers(sel_all: np.ndarray, classes: list[str],
                          core: np.ndarray, top_k: int,
                          qty_label: str, fmt: str, dpi: int,
                          fig_dir: Path):
    """Aggregate by max selectivity across layers."""
    sel_agg = sel_all.max(axis=1)           # (C, D)
    core_agg = core.any(axis=0)             # (D,)
    title = f"AST Neuron Atlas — {qty_label}  (all layers, max selectivity)"
    out = fig_dir / f"neuron_atlas_{qty_label}_all.{fmt}"
    _atlas_heatmap(sel_agg, classes, core_agg, top_k, title, fmt, dpi, out)


def plot_patch_causal(patch_res: dict, classes: list[str],
                      fmt: str, dpi: int, fig_dir: Path):
    """
    Boxplot comparing patch-shift distributions across three conditions.
    One subplot per condition; shared y-axis; per-class colours.
    Also a summary boxplot pooled across all classes.
    """
    ss = patch_res["shifts_specific"]   # (C, n_patch)
    sc = patch_res["shifts_core"]
    sr = patch_res["shifts_random"]
    counts = patch_res["counts"]

    # collect valid (non-nan) values pooled across classes
    def pool(arr):
        vals = arr[~np.isnan(arr)]
        return vals

    conditions = ["class-specific", "core AST", "random"]
    arrays_pooled = [pool(ss), pool(sc), pool(sr)]

    # per-class median for the class-specific condition
    per_class_med = []
    valid_classes = []
    for c, cls in enumerate(classes):
        v = ss[c][~np.isnan(ss[c])]
        if len(v) > 0:
            per_class_med.append(float(np.median(v)))
            valid_classes.append(cls)

    fig = plt.figure(figsize=(11, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.6], wspace=0.35)

    # ── left: pooled 3-condition boxplot ─────────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    bp = ax0.boxplot(
        arrays_pooled,
        labels=conditions,
        patch_artist=True,
        medianprops=dict(color="black", lw=2),
        whiskerprops=dict(lw=1.2),
        flierprops=dict(marker=".", markersize=2, alpha=0.4),
    )
    colours = ["#d73027", "#fdae61", "#abd9e9"]
    for patch, col in zip(bp["boxes"], colours):
        patch.set_facecolor(col)
        patch.set_alpha(0.85)

    ax0.axhline(0, ls="--", lw=1, color="grey")
    ax0.set_ylabel("Δ cos-sim toward class centroid", fontsize=FS_TITLE)
    ax0.set_title("Patch-shift by condition\n(pooled across classes)",
                  fontsize=FS_TITLE)
    ax0.tick_params(axis="x", labelsize=FS_TICK)

    # ── right: per-class bar of class-specific median shift ───────────────────
    ax1 = fig.add_subplot(gs[1])
    ypos = range(len(valid_classes))
    colours_bar = ["#d73027" if v > 0 else "#4393c3" for v in per_class_med]
    ax1.barh(list(ypos), per_class_med, color=colours_bar, alpha=0.80)
    ax1.set_yticks(list(ypos))
    ax1.set_yticklabels(valid_classes, fontsize=FS_TICK)
    ax1.axvline(0, ls="--", lw=1, color="grey")
    ax1.set_xlabel("Median Δ cos-sim (class-specific patch)", fontsize=FS_TITLE)
    ax1.set_title("Per-class causal patch shift\n(class-specific neurons)",
                  fontsize=FS_TITLE)

    fig.suptitle("Causal validation: class-specific neurons shift representations "
                 "toward target class", fontsize=FS_SUPTITLE, y=1.01)
    fig.tight_layout()
    out = fig_dir / f"patch_causal.{fmt}"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def plot_class_neuron_counts(class_nrn: np.ndarray, classes: list[str],
                             qty_label: str, fmt: str, dpi: int,
                             fig_dir: Path):
    """
    Stacked bar: for each layer, how many neurons are owned by each class.
    Gives an overview of where class specificity lives across depth.
    """
    C, L, D = class_nrn.shape
    counts = class_nrn.sum(axis=2)   # (C, L)

    cmap = plt.get_cmap("tab20")
    colours = [cmap(i % 20) for i in range(C)]

    fig, ax = plt.subplots(figsize=(8, 4))
    bottom = np.zeros(L)
    for c in range(C):
        ax.bar(range(L), counts[c], bottom=bottom,
               color=colours[c], label=classes[c])
        bottom += counts[c]

    ax.set_xlabel("Layer", fontsize=FS_TITLE)
    ax.set_ylabel("# class-specific neurons", fontsize=FS_TITLE)
    ax.set_title(f"Class-specific neuron count per layer  — {qty_label}",
                 fontsize=FS_SUPTITLE)
    ax.legend(fontsize=5, ncol=4, loc="upper left",
              framealpha=0.7, bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    out = fig_dir / f"class_neuron_counts_{qty_label}.{fmt}"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")

def get_overall_cluster_colors(features: np.ndarray, classes: list[str]) -> dict:
    """
    Computes overall clustering and assigns a rainbow color to each cluster,
    leaving the base tree default (blue).
    """
    from scipy.cluster.hierarchy import linkage, dendrogram, optimal_leaf_ordering
    from scipy.spatial.distance import pdist
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    valid_idx = np.where(np.abs(features).sum(axis=1) > 0)[0]
    if len(valid_idx) < 2:
        return {}

    valid_features = features[valid_idx]
    valid_classes = [classes[i] for i in valid_idx]

    dists = pdist(valid_features, metric='cosine')
    dists = np.nan_to_num(dists, nan=1.0)
    Z = linkage(dists, method='average')
    Z = optimal_leaf_ordering(Z, dists)

    R = dendrogram(Z, labels=valid_classes, no_plot=True, count_sort='descending')
    
    leaf_labels = R['ivl']
    leaf_colors = R['leaves_color_list']
    
    unique_clusters = []
    for c in leaf_colors:
        if c not in unique_clusters:
            unique_clusters.append(c)
            
    real_clusters = [c for c in unique_clusters if c not in ('C0', 'b', '#1f77b4', '#555555')]
    n_real = len(real_clusters)
    
    cmap = plt.get_cmap('rainbow')
    cluster_to_color = {}
    
    for c in unique_clusters:
        if c not in real_clusters:
            cluster_to_color[c] = '#1f77b4'  # Blue base tree
        else:
            idx = real_clusters.index(c)
            frac = idx / (n_real - 1) if n_real > 1 else 0.5
            cluster_to_color[c] = mcolors.to_hex(cmap(frac))

    class_to_color = {
        cls: cluster_to_color[color] 
        for cls, color in zip(leaf_labels, leaf_colors)
    }
    
    return class_to_color

def plot_ast_dendrogram(features: np.ndarray, classes: list[str],
                        qty_label: str, fmt: str, dpi: int, fig_dir: Path,
                        class_to_color: dict = None):
    """
    Hierarchical clustering of AST classes based on their selectivity profiles.
    Produces a dendrogram (family-tree like plot) to show which AST concepts
    use similar neural circuits.
    """
    from scipy.cluster.hierarchy import linkage, dendrogram, optimal_leaf_ordering
    from scipy.spatial.distance import pdist

    # Filter out classes with all-zero feature vectors (e.g., from insufficient samples)
    # which would otherwise result in NaN cosine distances.
    valid_idx = np.where(np.abs(features).sum(axis=1) > 0)[0]
    if len(valid_idx) < 2:
        print(f"  [dendrogram] Not enough valid classes to cluster for {qty_label}.")
        return

    valid_features = features[valid_idx]
    valid_classes = [classes[i] for i in valid_idx]

    # Compute pairwise cosine distances (1 - cosine similarity)
    dists = pdist(valid_features, metric='cosine')
    dists = np.nan_to_num(dists, nan=1.0)

    # Perform hierarchical/agglomerative clustering using average linkage
    Z = linkage(dists, method='average')
    Z = optimal_leaf_ordering(Z, dists)

    fig, ax = plt.subplots(figsize=(10, 5))

    link_color_func = None
    if class_to_color:
        N = len(valid_classes)
        node_colors = {}
        default_color = '#1f77b4'
        for i, cls in enumerate(valid_classes):
            node_colors[i] = class_to_color.get(cls, default_color)

        for i, row in enumerate(Z):
            idx1, idx2 = int(row[0]), int(row[1])
            c1 = node_colors.get(idx1, default_color)
            c2 = node_colors.get(idx2, default_color)
            if c1 == c2 and c1 != default_color:
                node_colors[N + i] = c1
            else:
                node_colors[N + i] = default_color

        link_color_func = lambda x: node_colors.get(x, default_color)

    dendrogram(
        Z,
        labels=valid_classes,
        leaf_rotation=90,
        leaf_font_size=FS_TICK,
        ax=ax,
        count_sort='descending',
        color_threshold=0 if class_to_color else None,
            above_threshold_color='#1f77b4',
        link_color_func=link_color_func
    )

    ax.set_title(f"Hierarchical Clustering of AST Classes ({qty_label})", fontsize=FS_SUPTITLE)
    ax.set_ylabel("Cosine Distance", fontsize=FS_TITLE)

    fig.tight_layout()
    out = fig_dir / f"ast_dendrogram_{qty_label}.{fmt}"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")

def plot_radial_dendrogram(features: np.ndarray, classes: list[str],
                           qty_label: str, fmt: str, dpi: int, fig_dir: Path,
                           class_to_color: dict = None):
    """
    Hierarchical clustering mapped to a polar coordinate system (Radial Dendrogram),
    similar to a sunburst/radial tree edge layout.
    """
    from scipy.cluster.hierarchy import linkage, dendrogram, optimal_leaf_ordering
    from scipy.spatial.distance import pdist

    valid_idx = np.where(np.abs(features).sum(axis=1) > 0)[0]
    if len(valid_idx) < 2:
        return

    valid_features = features[valid_idx]
    valid_classes = [classes[i] for i in valid_idx]

    dists = pdist(valid_features, metric='cosine')
    dists = np.nan_to_num(dists, nan=1.0)
    Z = linkage(dists, method='average')
    Z = optimal_leaf_ordering(Z, dists)

    # Create the color mapping function for the branches
    link_color_func = None
    if class_to_color:
        N = len(valid_classes)
        node_colors = {}
        default_color = '#1f77b4'
        for i, cls in enumerate(valid_classes):
            node_colors[i] = class_to_color.get(cls, default_color)
        for i, row in enumerate(Z):
            idx1, idx2 = int(row[0]), int(row[1])
            c1, c2 = node_colors.get(idx1, default_color), node_colors.get(idx2, default_color)
            node_colors[N + i] = c1 if c1 == c2 and c1 != default_color else default_color
        link_color_func = lambda x: node_colors.get(x, default_color)

    R = dendrogram(Z, no_plot=True, count_sort='descending', 
                   color_threshold=0 if class_to_color else None,
                   link_color_func=link_color_func)

    icoord, dcoord = np.array(R['icoord']), np.array(R['dcoord'])
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
    
    leaves = R['leaves']
    xmax = 10 * len(leaves)
    ymax = np.max(dcoord) if np.max(dcoord) > 0 else 1.0
    inner_r, outer_r = 0.15, 1.0

    # Draw the branches
    for xs, ys, color in zip(icoord, dcoord, R['color_list']):
        thetas = (xs / xmax) * 2 * np.pi
        rs = outer_r - (ys / ymax) * (outer_r - inner_r)
        ax.plot(thetas, rs, color=color, lw=1.2)

    # Add the leaf labels
    for i, leaf_idx in enumerate(leaves):
        name = valid_classes[leaf_idx]
        theta = ((10 * i + 5) / xmax) * 2 * np.pi
        rot = np.degrees(theta)
        if np.pi / 2 < theta < 3 * np.pi / 2:
            rot += 180
            ha = 'right'
        else:
            ha = 'left'
            
        c = class_to_color.get(name, 'black') if class_to_color else 'black'
        ax.text(theta, outer_r + 0.03, name, rotation=rot, ha=ha, va='center',
                rotation_mode='anchor', color=c, fontsize=FS_TICK)

    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.spines['polar'].set_visible(False)
    ax.set_title(f"Radial Dendrogram ({qty_label})", fontsize=FS_SUPTITLE, pad=20)

    fig.tight_layout()
    out = fig_dir / f"radial_dendrogram_{qty_label}.{fmt}"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def plot_tsne(features: np.ndarray, classes: list[str],
              qty_label: str, fmt: str, dpi: int, fig_dir: Path,
              class_to_color: dict = None):
    """
    Plots a t-SNE projection to visualize the uniqueness and relationships of AST features.
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("  [t-SNE] scikit-learn not found. Skipping t-SNE plot.")
        return
        
    valid_idx = np.where(np.abs(features).sum(axis=1) > 0)[0]
    if len(valid_idx) < 3:
        return

    valid_features = features[valid_idx]
    valid_classes = [classes[i] for i in valid_idx]

    perplexity = min(5, len(valid_features) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, metric='cosine', random_state=42, init='random')
    
    try:
        emb = tsne.fit_transform(valid_features)
    except Exception as e:
        print(f"  [t-SNE] Could not compute t-SNE for {qty_label}: {e}")
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    for i, cls_name in enumerate(valid_classes):
        c = class_to_color.get(cls_name, '#1f77b4') if class_to_color else '#1f77b4'
        ax.scatter(emb[i, 0], emb[i, 1], color=c, s=80, alpha=0.8, edgecolors='white', linewidth=0.5)
        ax.annotate(cls_name, (emb[i, 0], emb[i, 1]), xytext=(7, 0), textcoords='offset points',
                    fontsize=FS_ANNOT, color=c, va='center')

    ax.set_title(f"t-SNE of AST Classes ({qty_label})", fontsize=FS_SUPTITLE)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    out = fig_dir / f"tsne_{qty_label}.{fmt}"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def plot_dendrograms_for_qty(sel_all: np.ndarray, classes: list[str], qty_label: str,
                             fmt: str, dpi: int, fig_dir: Path, class_to_color: dict = None):
    """
    Plots an overall hierarchy (Dendrograms + t-SNE) and per-layer views for the given selectivity array.
    """
    C, L, D = sel_all.shape
    
    # Overall view (flattened across all layers)
    plot_ast_dendrogram(sel_all.reshape(C, -1), classes, f"{qty_label}_overall", fmt, dpi, fig_dir, class_to_color)
    plot_radial_dendrogram(sel_all.reshape(C, -1), classes, f"{qty_label}_overall", fmt, dpi, fig_dir, class_to_color)
    plot_tsne(sel_all.reshape(C, -1), classes, f"{qty_label}_overall", fmt, dpi, fig_dir, class_to_color)
    
    # Per-layer view
    for l in range(L):
        plot_ast_dendrogram(sel_all[:, l, :], classes, f"{qty_label}_L{l:02d}", fmt, dpi, fig_dir, class_to_color)
        plot_radial_dendrogram(sel_all[:, l, :], classes, f"{qty_label}_L{l:02d}", fmt, dpi, fig_dir, class_to_color)
        plot_tsne(sel_all[:, l, :], classes, f"{qty_label}_L{l:02d}", fmt, dpi, fig_dir, class_to_color)

# ─────────────────────────────────────────────────────────────────────────────
# Ownership grid  (neuron_grid-style, coloured by AST class)
# ─────────────────────────────────────────────────────────────────────────────

def _class_colours(n: int) -> list:
    """n distinct RGBA colours via tab20 + tab20b cycling."""
    c1 = plt.get_cmap("tab20")
    c2 = plt.get_cmap("tab20b")
    return [c1(i) if i < 20 else c2(i - 20) for i in range(n)]


def _build_ownership_image(sel_CD: np.ndarray,
                            class_nrn_D: np.ndarray,
                            core_D: np.ndarray,
                            top_k: int,
                            colours: list,
                            grid_shape: tuple[int, int],
                            top_k_only: bool = True,
                            pure_mask_D: np.ndarray | None = None):
    """
    sel_CD:      (C, D)   selectivity at one layer
    class_nrn_D: (C, D)   bool — which dims are owned by each class
    core_D:      (D,)     bool — which dims are core
    top_k_only:  if True, colour only the top-K neurons per class (not all owned)
    pure_mask_D: (D,) bool — when given, intersect ownership with ast_pure mask

    Padding: grid_shape may be larger than D (R×Cg > D). Padded cells are
    set to black to distinguish them from white (unowned) and grey (core).

    Returns:
      img     (R, Cg, 4)  RGBA float32
      markers list of (row, col, class_idx) for top-K per class
    """
    R, Cg = grid_shape
    D_pad = R * Cg
    D = len(core_D)

    img = np.ones((D_pad, 4), dtype=np.float32)       # white background
    img[D:] = [0.0, 0.0, 0.0, 1.0]                    # padded cells → black

    # core neurons → grey
    img[:D][core_D] = [0.72, 0.72, 0.72, 1.0]

    # optionally intersect with ast_pure mask
    effective = class_nrn_D.copy()  # (C, D)
    if pure_mask_D is not None:
        effective &= pure_mask_D[np.newaxis, :]

    # top-K per class: find which dims to colour
    markers: list[tuple[int, int, int]] = []
    for c, rgba in enumerate(colours):
        owned = np.where(effective[c])[0]
        if len(owned) == 0:
            continue
        k = min(top_k, len(owned))
        best = owned[np.argsort(sel_CD[c, owned])[-k:]]

        if top_k_only:
            # colour only the top-K, not all owned
            color_dims = best
        else:
            color_dims = owned

        img[:D][color_dims] = [rgba[0], rgba[1], rgba[2], 0.92]

        for d in best:
            markers.append((int(d // Cg), int(d % Cg), c))

    return img.reshape(R, Cg, 4), markers


def plot_ownership_grid(sel_all: np.ndarray,
                        class_nrn_all: np.ndarray,
                        core_all: np.ndarray,
                        classes: list[str],
                        qty: str,
                        top_k: int,
                        fmt: str,
                        dpi: int,
                        fig_dir: Path,
                        label: str = "",
                        top_k_only: bool = True,
                        ast_pure_all: np.ndarray | None = None) -> None:
    """
    One figure per qty: all layers tiled in a grid, each cell showing which
    neurons belong to which AST class.

      black  = padded cell (no neuron)
      white  = unowned neuron
      grey   = core AST neuron (fires for all classes)
      colour = owned by one AST class (tab20 palette)
      dot    = top-K most selective neurons per class (white face, class-colour edge)

    top_k_only: colour only the top-K neurons per class (not all owned dims)
    ast_pure_all: (L, D) bool — when given, intersect ownership with ast_pure
    label: suffix for the output filename  e.g. "top10_raw", "top10_pure"
    """
    from matplotlib.gridspec import GridSpec

    C, L, D = sel_all.shape
    grid_shape = GRID_SHAPE[qty]
    cell_w, cell_h = CELL_SIZE[qty]

    n_cols = min(N_OWN_COLS, L)
    n_rows = math.ceil(L / n_cols)
    colours = _class_colours(C)

    fig_w = LEFT_IN + cell_w * n_cols + LEGEND_IN
    fig_h = TOP_IN  + cell_h * n_rows

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    outer = GridSpec(
        n_rows, n_cols,
        figure = fig,
        left   = LEFT_IN  / fig_w,
        right  = 1.0 - LEGEND_IN / fig_w,
        top    = 1.0 - TOP_IN / fig_h,
        bottom = 0.02,
        wspace = 0.06,
        hspace = 0.10,
    )

    for l in range(L):
        ri = l // n_cols
        ci = l % n_cols
        ax = fig.add_subplot(outer[ri, ci])

        img, markers = _build_ownership_image(
            sel_all[:, l, :],
            class_nrn_all[:, l, :],
            core_all[l],
            top_k, colours, grid_shape,
            top_k_only=top_k_only,
            pure_mask_D=ast_pure_all[l] if ast_pure_all is not None else None,
        )
        ax.imshow(img, aspect="auto", interpolation="nearest", origin="upper")

        # top-K dots: white fill, class-colour edge — visible but not overwhelming
        if markers:
            xs = [m[1] for m in markers]
            ys = [m[0] for m in markers]
            ec = [colours[m[2]] for m in markers]
            ax.scatter(xs, ys, s=8, facecolors="white", edgecolors=ec,
                       linewidths=0.8, zorder=3)

        n_owned = int(class_nrn_all[:, l, :].sum())
        n_core  = int(core_all[l].sum())
        ax.set_title(
            f"L{l}   {n_owned} class-specific · {n_core} core",
            fontsize=FS_TITLE - 2 if qty == "mlp" else FS_TITLE,
            pad=3,
        )
        # first cell only: show corner neuron indices
        if l == 0:
            R_g, C_g = grid_shape
            ax.set_xticks([0, C_g - 1])
            ax.set_xticklabels(["n=0", f"n={C_g-1}"],
                               fontsize=FS_ANNOT - 1, color="#555")
            ax.tick_params(axis="x", length=2, pad=1,
                           bottom=False, top=True,
                           labelbottom=False, labeltop=True)
            ax.set_yticks([0, R_g - 1])
            ax.set_yticklabels(["", f"n={(R_g-1)*C_g}"],
                               fontsize=FS_ANNOT - 1, color="#555")
            ax.tick_params(axis="y", length=2, pad=1)
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("#bbbbbb")
            spine.set_linewidth(0.5)

    # ── legend ────────────────────────────────────────────────────────────────
    from matplotlib.patches import Patch as _Patch
    handles = [_Patch(facecolor=colours[c], label=classes[c]) for c in range(C)]
    handles += [
        _Patch(facecolor=(0.72, 0.72, 0.72, 1.0), label="core AST"),
        _Patch(facecolor="white", edgecolor="#999", label="unowned"),
    ]
    fig.legend(
        handles=handles,
        loc="center right",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=6,
        ncol=1,
        framealpha=0.9,
        handlelength=1.2,
        handleheight=0.9,
        borderpad=0.6,
        labelspacing=0.3,
        title="AST class",
        title_fontsize=7,
    )

    qty_label = {"residual": "Residual stream", "mlp": "MLP neurons"}[qty]
    fig.suptitle(
        f"AST Neuron Ownership Map — {qty_label}  "
        f"(colour = owning class · dot = top-{top_k} most selective)",
        fontsize=FS_SUPTITLE,
        y=1.01,
    )

    suffix = f"_{label}" if label else ""
    out = fig_dir / f"ownership_grid_{qty}{suffix}.{fmt}"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Serialise neuron lists
# ─────────────────────────────────────────────────────────────────────────────

def build_class_neurons_json(class_nrn_r: np.ndarray,
                              class_nrn_n: np.ndarray,
                              core_r: np.ndarray,
                              core_n: np.ndarray,
                              classes: list[str]) -> dict:
    """
    Returns a JSON-serialisable dict with per-class neuron lists
    for both residual and MLP, per layer and aggregated.
    """
    out = {"classes": classes, "residual": {}, "neurons": {}}

    for c, cls in enumerate(classes):
        # residual
        r_per_layer = {}
        for l in range(class_nrn_r.shape[1]):
            ids = np.where(class_nrn_r[c, l])[0].tolist()
            r_per_layer[f"L{l:02d}"] = ids
        r_all = np.where(class_nrn_r[c].any(axis=0))[0].tolist()
        out["residual"][cls] = {"per_layer": r_per_layer, "any_layer": r_all}

        # mlp neurons
        n_per_layer = {}
        for l in range(class_nrn_n.shape[1]):
            ids = np.where(class_nrn_n[c, l])[0].tolist()
            n_per_layer[f"L{l:02d}"] = ids
        n_all = np.where(class_nrn_n[c].any(axis=0))[0].tolist()
        out["neurons"][cls] = {"per_layer": n_per_layer, "any_layer": n_all}

    # core
    out["core_residual_per_layer"] = {
        f"L{l:02d}": np.where(core_r[l])[0].tolist()
        for l in range(core_r.shape[0])
    }
    out["core_neurons_per_layer"] = {
        f"L{l:02d}": np.where(core_n[l])[0].tolist()
        for l in range(core_n.shape[0])
    }
    return out

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    in_dir  = Path(__file__).parent / args.in_dir
    out_dir = Path(__file__).parent / args.out_dir
    fig_dir = out_dir / "figures" / "class_neurons"
    fig_dir.mkdir(parents=True, exist_ok=True)

    stem = args.stem
    fmt  = args.format

    # ── load ─────────────────────────────────────────────────────────────────
    print("Loading data …")
    resid, mlp, meta, pm, vp_r, vp_n = load_data(in_dir, stem)

    N, Lr, H        = resid.shape    # e.g. (14705, 9, 2048)
    N2, Ln, mlp_dim = mlp.shape      # e.g. (14705, 8, 8192)

    # label arrays
    classes  = sorted(set(m["ast_node"] for m in meta))
    cls_to_i = {c: i for i, c in enumerate(classes)}
    ast_idx  = np.array([cls_to_i[m["ast_node"]] for m in meta], dtype=np.int32)
    C = len(classes)

    print(f"  {N} prompts · {C} AST classes · residual ({Lr}, {H}) · mlp ({Ln}, {mlp_dim})")

    # purity masks
    ast_pure_r = pm["residual_ast_pure"]  # (Lr, H)
    ast_pure_n = pm["neurons_ast_pure"]   # (Ln, mlp_dim)

    # ── selectivity ──────────────────────────────────────────────────────────
    print("Computing selectivity (residual) …")
    sel_r = compute_selectivity_all_layers(resid, ast_idx, classes, args.min_samples)
    # sel_r shape: (C, Lr, H)

    print("Computing selectivity (MLP neurons, layer by layer) …")
    sel_n = compute_selectivity_all_layers(mlp,   ast_idx, classes, args.min_samples)
    # sel_n shape: (C, Ln, mlp_dim)

    # ── core / class-specific split ──────────────────────────────────────────
    print("Splitting core vs class-specific neurons …")
    core_r, class_nrn_r = split_core_class(sel_r, ast_pure_r,
                                            args.sel_thresh, args.core_thresh)
    core_n, class_nrn_n = split_core_class(sel_n, ast_pure_n,
                                            args.sel_thresh, args.core_thresh)

    # summary
    print(f"  Residual  core: {core_r.sum()} dims · class-specific: {class_nrn_r.sum()} total")
    print(f"  MLP       core: {core_n.sum()} dims · class-specific: {class_nrn_n.sum()} total")
    per_class_r = class_nrn_r.sum(axis=(1,2))
    per_class_n = class_nrn_n.sum(axis=(1,2))
    for i, cls in enumerate(classes):
        print(f"    {cls:20s}  resid={per_class_r[i]:4d}  mlp={per_class_n[i]:4d}")

    # ── save selectivity arrays ───────────────────────────────────────────────
    print("Saving selectivity arrays …")
    np.savez_compressed(
        out_dir / f"02g_{stem}_selectivity_residual.npz",
        sel=sel_r, core=core_r, class_neurons=class_nrn_r,
        classes=np.array(classes),
    )
    np.savez_compressed(
        out_dir / f"02g_{stem}_selectivity_neurons.npz",
        sel=sel_n, core=core_n, class_neurons=class_nrn_n,
        classes=np.array(classes),
    )

    # ── causal patching ───────────────────────────────────────────────────────
    patch_layer = args.patch_layer % Lr
    print(f"Running causal patching at residual layer {patch_layer} …")
    rng = np.random.default_rng(42)
    patch_res = run_patching(
        resid, ast_idx, classes,
        class_nrn_r, core_r,
        patch_layer=patch_layer,
        n_patch=args.n_patch,
        rng=rng,
    )
    np.savez_compressed(
        out_dir / f"02g_{stem}_patch_results.npz",
        **patch_res,
        classes=np.array(classes),
        patch_layer=np.array(patch_layer),
    )

    # ── save class neuron JSON ────────────────────────────────────────────────
    print("Saving class_neurons.json …")
    cn_json = build_class_neurons_json(class_nrn_r, class_nrn_n,
                                       core_r, core_n, classes)
    with open(out_dir / f"02g_{stem}_class_neurons.json", "w") as f:
        json.dump(cn_json, f, indent=2)

    # ── figures ───────────────────────────────────────────────────────────────
    print("Plotting per-layer atlases (residual) …")
    plot_atlas_per_layer(sel_r, classes, core_r, args.top_k,
                         "residual", fmt, args.dpi, fig_dir)

    print("Plotting per-layer atlases (MLP neurons) …")
    plot_atlas_per_layer(sel_n, classes, core_n, args.top_k,
                         "mlp", fmt, args.dpi, fig_dir)

    print("Plotting headline atlas (residual, all layers) …")
    plot_atlas_all_layers(sel_r, classes, core_r, args.top_k,
                          "residual", fmt, args.dpi, fig_dir)

    print("Plotting headline atlas (MLP, all layers) …")
    plot_atlas_all_layers(sel_n, classes, core_n, args.top_k,
                          "mlp", fmt, args.dpi, fig_dir)

    print("Plotting class neuron count bars …")
    plot_class_neuron_counts(class_nrn_r, classes, "residual", fmt, args.dpi, fig_dir)
    plot_class_neuron_counts(class_nrn_n, classes, "mlp",      fmt, args.dpi, fig_dir)

    print("Computing overall reference colors for consistent dendrogram plots …")
    comb_features = np.concatenate([sel_r.reshape(C, -1), sel_n.reshape(C, -1)], axis=1)
    class_to_color = get_overall_cluster_colors(comb_features, classes)

    print("Plotting AST dendrograms (residual: overall + per layer) …")
    plot_dendrograms_for_qty(sel_r, classes, "residual", fmt, args.dpi, fig_dir, class_to_color)

    print("Plotting AST dendrograms (MLP: overall + per layer) …")
    plot_dendrograms_for_qty(sel_n, classes, "mlp", fmt, args.dpi, fig_dir, class_to_color)

    print("Plotting combined AST dendrogram (Residual + MLP) …")
    plot_ast_dendrogram(comb_features, classes, "combined_res_mlp_overall", fmt, args.dpi, fig_dir, class_to_color)

    print("Plotting causal patching figure …")
    plot_patch_causal(patch_res, classes, fmt, args.dpi, fig_dir)

    print("Plotting ownership grid (residual, top-k raw) …")
    plot_ownership_grid(sel_r, class_nrn_r, core_r, classes,
                        "residual", args.top_k, fmt, args.dpi, fig_dir,
                        label="top_k_raw", top_k_only=True)

    print("Plotting ownership grid (residual, top-k pure) …")
    plot_ownership_grid(sel_r, class_nrn_r, core_r, classes,
                        "residual", args.top_k, fmt, args.dpi, fig_dir,
                        label="top_k_pure", top_k_only=True,
                        ast_pure_all=ast_pure_r)

    print("Plotting ownership grid (residual, all owned pure) …")
    plot_ownership_grid(sel_r, class_nrn_r, core_r, classes,
                        "residual", args.top_k, fmt, args.dpi, fig_dir,
                        label="all_owned_pure", top_k_only=False,
                        ast_pure_all=ast_pure_r)

    print("Plotting ownership grid (MLP, top-k raw) …")
    plot_ownership_grid(sel_n, class_nrn_n, core_n, classes,
                        "mlp", args.top_k, fmt, args.dpi, fig_dir,
                        label="top_k_raw", top_k_only=True)

    print("Plotting ownership grid (MLP, top-k pure) …")
    plot_ownership_grid(sel_n, class_nrn_n, core_n, classes,
                        "mlp", args.top_k, fmt, args.dpi, fig_dir,
                        label="top_k_pure", top_k_only=True,
                        ast_pure_all=ast_pure_n)

    print("Plotting ownership grid (MLP, all owned pure) …")
    plot_ownership_grid(sel_n, class_nrn_n, core_n, classes,
                        "mlp", args.top_k, fmt, args.dpi, fig_dir,
                        label="all_owned_pure", top_k_only=False,
                        ast_pure_all=ast_pure_n)

    print("Done.")


if __name__ == "__main__":
    main()