"""
02e_ast_overlap.py  (lives in results/)

Analyses whether AST node representations share a common central structure,
with individual node types as deviations from that centre.

Three complementary analyses, repeated per layer:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1.  NEURON OVERLAP MATRIX  (Jaccard on top-K active neurons per class)
    ─────────────────────────────────────────────────────────────────
    For each AST class c: rank neurons by mean activation → take top-K.
    Compute Jaccard similarity J(c1,c2) = |top_K(c1) ∩ top_K(c2)| / K.
    Plot as a clustermap.
    High off-diagonal values → classes share their "active" neurons.

2.  CENTROID DECOMPOSITION
    ──────────────────────
    Global mean:  μ  = mean over ALL AST prompts  (H,)
    Class mean:   μ_c = mean over prompts for class c  (H,)
    Deviation:    δ_c = μ_c − μ  (H,)

    Metrics per class:
      cos(μ_c, μ)         — how aligned is the class mean to the global mean?
                            → 1.0 = class is "the same" as the global structure
      |δ_c| / |μ|         — relative deviation magnitude
      |δ_c|_ast / |δ_c|   — fraction of deviation in the ast_pure subspace

    Plots:
      • Bar chart of cosine similarities (sorted)
      • Bar chart of relative deviation magnitudes
      • Scatter: cos(μ_c, μ) vs |δ_c|/|μ|  — the "central structure" plane
        (top-right = far from centre; bottom-left = close to centre)

3.  PCA OF CLASS MEANS  ("inter-class geometry")
    ──────────────────────────────────────────────
    Stack all class mean vectors → PCA.
    PC1 = the dominant shared direction (the "central AST structure").
    Remaining PCs = dimensions of class-specific variation.

    Plots:
      • Scree plot: variance explained per PC (how dominant is PC1?)
      • 2-D scatter of class means in PC1–PC2 space, labelled by class name
      • Loadings heatmap: which neurons drive PC1 vs PC2
        (shown as 32×64 activation grid, same layout as 02d)

4.  CORE vs PERIPHERAL NEURONS  (per layer)
    ─────────────────────────────────────────
    Core:       neurons in the top-K for ≥ core_frac of classes
    Peripheral: neurons in the top-K for ≤ 1 class

    Plots:
      • Grid heatmap (classes × top-K neurons) sorted by "coreness"
        → leftmost columns = neurons shared by many classes (core)
        → rightmost = class-specific (peripheral)
      • Summary bar: #core, #peripheral, #intermediate per layer

INPUTS (relative to --in_dir)
──────
  01_<stem>_residual_all.npy    (N, L+1, H)
  01_<stem>_meta.json
  02_<stem>_vp_residual.npz     keys: unique_ast, unique_builtin
  02_<stem>_purity_masks.npz    keys: residual_ast_pure, …

OUTPUTS  (all under --out_dir)
────────
  figures/ast_overlap/
    L<ll>_jaccard.png
    L<ll>_centroid.png
    L<ll>_pca_class_means.png
    L<ll>_core_peripheral.png
  ast_overlap_summary.json      scalar metrics per layer

Usage
-----
  python 02e_ast_overlap.py --stem contrastive_stubs
  python 02e_ast_overlap.py --stem contrastive_stubs --layer 5
  python 02e_ast_overlap.py --stem contrastive_stubs --top_k 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

TOP_K_DEFAULT    = 200    # neurons per class for Jaccard / core analysis
CORE_FRAC        = 0.75   # fraction of classes a neuron must appear in → "core"
PERIPHERAL_MAX   = 1      # max classes → "peripheral"
DPI              = 180
GRID_SHAPE       = (32, 64)   # 2048 = 32 × 64  (residual only)
PC_SHOW          = 8          # number of PCs in scree plot


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stem",    default="contrastive_stubs")
    p.add_argument("--in_dir",  "-d", default="../data",
                   help="Dir with 01_* and 02_* files (relative to script)")
    p.add_argument("--out_dir", "-o", default=None,
                   help="Output root (default: same as --in_dir)")
    p.add_argument("--layer",   type=int, default=None,
                   help="If set, only analyse this layer")
    p.add_argument("--top_k",   type=int, default=TOP_K_DEFAULT,
                   help=f"Top-K neurons per class for Jaccard/core (default {TOP_K_DEFAULT})")
    p.add_argument("--core_frac", type=float, default=CORE_FRAC,
                   help=f"Min class fraction to be 'core' (default {CORE_FRAC})")
    p.add_argument("--dpi",     type=int, default=DPI)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helper: class mean activations
# ─────────────────────────────────────────────────────────────────────────────

def class_means(
    act:        np.ndarray,    # (N, H)
    labels:     np.ndarray,   # (N,) string
    classes:    list[str],
) -> np.ndarray:
    """Return (C, H) array of per-class mean activations."""
    return np.stack([
        act[labels == c].mean(axis=0) if (labels == c).any()
        else np.zeros(act.shape[1], dtype=np.float32)
        for c in classes
    ])   # (C, H)


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 1: Jaccard overlap matrix
# ─────────────────────────────────────────────────────────────────────────────

def jaccard_matrix(mu_c: np.ndarray, top_k: int) -> np.ndarray:
    """
    (C, H) class means → (C, C) Jaccard matrix.
    J(i,j) = |top_k(i) ∩ top_k(j)| / k
    """
    C = mu_c.shape[0]
    top_sets = [set(np.argsort(mu_c[i])[-top_k:]) for i in range(C)]
    J = np.zeros((C, C), dtype=np.float32)
    for i in range(C):
        for j in range(i, C):
            inter = len(top_sets[i] & top_sets[j])
            J[i, j] = J[j, i] = inter / top_k
    return J


def plot_jaccard(
    J:       np.ndarray,
    classes: list[str],
    layer:   int,
    save:    Path,
    dpi:     int,
) -> None:
    C = len(classes)
    # Hierarchical clustering order
    dist = 1.0 - J
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    condensed = np.clip(condensed, 0, None)
    Z   = linkage(condensed, method="average")
    ord = leaves_list(Z)

    J_ord = J[np.ix_(ord, ord)]
    cls_ord = [classes[i] for i in ord]

    fig, ax = plt.subplots(figsize=(max(8, C * 0.28), max(7, C * 0.26)))
    im = ax.imshow(J_ord, vmin=0, vmax=1, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(C)); ax.set_xticklabels(cls_ord, rotation=90, fontsize=7)
    ax.set_yticks(range(C)); ax.set_yticklabels(cls_ord, fontsize=7)
    plt.colorbar(im, ax=ax, label="Jaccard overlap", shrink=0.8)
    ax.set_title(
        f"Layer {layer}  —  Jaccard overlap of top-{J.shape[0]} neuron sets per AST class\n"
        f"(higher = more shared active neurons)", fontsize=10
    )
    fig.tight_layout()
    fig.savefig(save, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    → {save.name}  (mean off-diag Jaccard = {_mean_offdiag(J):.3f})")


def _mean_offdiag(M: np.ndarray) -> float:
    C = M.shape[0]
    mask = ~np.eye(C, dtype=bool)
    return float(M[mask].mean())


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 2: Centroid decomposition
# ─────────────────────────────────────────────────────────────────────────────

def centroid_analysis(
    act:        np.ndarray,    # (N, H)
    labels:     np.ndarray,
    classes:    list[str],
    ast_mask:   np.ndarray | None,   # (H,) bool
) -> dict:
    """
    Returns per-class metrics dict:
      cos_to_global, delta_norm_rel, delta_in_ast_frac
    and scalar: global_norm
    """
    mu_global = act.mean(axis=0)                      # (H,)
    norm_g    = float(np.linalg.norm(mu_global)) + 1e-12

    mu_c    = class_means(act, labels, classes)       # (C, H)
    delta_c = mu_c - mu_global[None, :]               # (C, H)

    # Cosine of class mean to global mean
    norms_c = np.linalg.norm(mu_c, axis=1) + 1e-12
    cos_to_g = (mu_c @ mu_global) / (norms_c * norm_g)   # (C,)

    # Relative deviation magnitude
    delta_norms = np.linalg.norm(delta_c, axis=1)         # (C,)
    delta_rel   = delta_norms / norm_g                     # (C,)

    # Fraction of deviation in ast_pure subspace
    if ast_mask is not None and ast_mask.any():
        delta_ast = np.linalg.norm(delta_c[:, ast_mask], axis=1) + 1e-12
        ast_frac  = delta_ast / (delta_norms + 1e-12)
    else:
        ast_frac = np.full(len(classes), float("nan"))

    return {
        "classes":        classes,
        "cos_to_global":  cos_to_g.tolist(),
        "delta_rel":      delta_rel.tolist(),
        "ast_frac":       ast_frac.tolist(),
        "global_norm":    norm_g,
        "mu_global":      mu_global,
        "mu_c":           mu_c,
        "delta_c":        delta_c,
    }


def plot_centroid(res: dict, layer: int, save: Path, dpi: int) -> None:
    classes  = res["classes"]
    cos_g    = np.array(res["cos_to_global"])
    delta_r  = np.array(res["delta_rel"])
    ast_frac = np.array(res["ast_frac"])
    C        = len(classes)

    sort_idx = np.argsort(cos_g)[::-1]
    cls_s    = [classes[i] for i in sort_idx]
    cos_s    = cos_g[sort_idx]
    dr_s     = delta_r[sort_idx]
    af_s     = ast_frac[sort_idx]

    fig, axes = plt.subplots(1, 3, figsize=(18, max(4, C * 0.22)))

    # ── Panel 1: cosine to global mean ───────────────────────────────────────
    ax = axes[0]
    bars = ax.barh(range(C), cos_s, color=plt.cm.RdYlGn(cos_s))
    ax.set_yticks(range(C)); ax.set_yticklabels(cls_s, fontsize=7)
    ax.invert_yaxis()
    ax.axvline(cos_s.mean(), color="black", lw=1, linestyle="--",
               label=f"mean={cos_s.mean():.3f}")
    ax.set_xlabel("cos(μ_class, μ_global)", fontsize=9)
    ax.set_title("Alignment to global mean\n(1.0 = perfectly central)", fontsize=9)
    ax.legend(fontsize=7)
    ax.set_xlim(0, 1.05)
    ax.spines[["top", "right"]].set_visible(False)

    # ── Panel 2: relative deviation magnitude ────────────────────────────────
    ax = axes[1]
    ax.barh(range(C), dr_s, color=plt.cm.YlOrRd(dr_s / (dr_s.max() + 1e-9)))
    ax.set_yticks(range(C)); ax.set_yticklabels(cls_s, fontsize=7)
    ax.invert_yaxis()
    ax.axvline(dr_s.mean(), color="black", lw=1, linestyle="--",
               label=f"mean={dr_s.mean():.3f}")
    ax.set_xlabel("|δ_class| / |μ_global|", fontsize=9)
    ax.set_title("Relative deviation magnitude\n(small = close to centre)", fontsize=9)
    ax.legend(fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)

    # ── Panel 3: scatter cos vs delta ────────────────────────────────────────
    ax = axes[2]
    sc = ax.scatter(cos_g, delta_r,
                    c=af_s if not np.isnan(af_s).all() else "steelblue",
                    cmap="cool", vmin=0, vmax=1,
                    s=60, edgecolors="k", linewidths=0.4, zorder=3)
    for i, cls in enumerate(classes):
        ax.annotate(cls, (cos_g[i], delta_r[i]), fontsize=5,
                    xytext=(2, 2), textcoords="offset points")
    if not np.isnan(af_s).all():
        cb = plt.colorbar(sc, ax=ax, shrink=0.8)
        cb.set_label("fraction of δ in ast_pure", fontsize=7)
    ax.set_xlabel("cos(μ_class, μ_global)", fontsize=9)
    ax.set_ylabel("|δ_class| / |μ_global|", fontsize=9)
    ax.set_title("Central structure plane\n"
                 "bottom-right = central · top-left = outlier", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Layer {layer}  —  Centroid decomposition of AST class means",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(save, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    → {save.name}  "
          f"(mean cos={cos_g.mean():.3f}, mean δ/μ={delta_r.mean():.3f})")


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 3: PCA of class means
# ─────────────────────────────────────────────────────────────────────────────

def plot_pca_class_means(
    mu_c:    np.ndarray,    # (C, H)
    classes: list[str],
    layer:   int,
    save:    Path,
    dpi:     int,
) -> None:
    C, H = mu_c.shape
    n_pc = min(PC_SHOW, C - 1, H)

    pca = PCA(n_components=n_pc, random_state=42)
    Z   = pca.fit_transform(mu_c.astype(np.float32))    # (C, n_pc)
    ev  = pca.explained_variance_ratio_

    fig = plt.figure(figsize=(18, 10))
    gs  = fig.add_gridspec(2, 3, hspace=0.40, wspace=0.35)

    # ── Scree plot ────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.bar(range(1, n_pc + 1), ev * 100, color="steelblue", edgecolor="white")
    ax.plot(range(1, n_pc + 1), np.cumsum(ev) * 100, "o-",
            color="tomato", label="Cumulative")
    ax.set_xlabel("PC", fontsize=9); ax.set_ylabel("Variance explained (%)", fontsize=9)
    ax.set_title(f"Scree — L{layer}\nPC1={ev[0]*100:.1f}%", fontsize=9)
    ax.legend(fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)

    # ── PC1 vs PC2 scatter ───────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1:])
    cmap = plt.get_cmap("tab20", C)
    for i, cls in enumerate(classes):
        ax.scatter(Z[i, 0], Z[i, 1], color=cmap(i), s=60,
                   edgecolors="k", linewidths=0.4, zorder=3)
        ax.annotate(cls, (Z[i, 0], Z[i, 1]), fontsize=6,
                    xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="grey", lw=0.6); ax.axvline(0, color="grey", lw=0.6)
    ax.set_xlabel(f"PC1  ({ev[0]*100:.1f}%)", fontsize=9)
    ax.set_ylabel(f"PC2  ({ev[1]*100:.1f}%)" if n_pc > 1 else "PC2", fontsize=9)
    ax.set_title("Class means in PC1–PC2 space\n"
                 "spread along PC1 = shared structure · spread along PC2+ = class-specific",
                 fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # ── PC1 loadings as activation grid ──────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    pc1 = pca.components_[0]     # (H,)
    if H == GRID_SHAPE[0] * GRID_SHAPE[1]:
        pc1_img = pc1.reshape(GRID_SHAPE)
        vabs = np.abs(pc1_img).max()
        ax.imshow(pc1_img, cmap="RdBu_r", vmin=-vabs, vmax=vabs,
                  aspect="auto", origin="upper", interpolation="nearest")
        ax.set_title("PC1 loadings (neuron grid)\nred=+, blue=−", fontsize=8)
    else:
        ax.plot(pc1); ax.set_title("PC1 loadings", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[:].set_visible(False)

    # ── PC2 loadings ──────────────────────────────────────────────────────────
    if n_pc > 1:
        ax = fig.add_subplot(gs[1, 1])
        pc2 = pca.components_[1]
        if H == GRID_SHAPE[0] * GRID_SHAPE[1]:
            pc2_img = pc2.reshape(GRID_SHAPE)
            vabs = np.abs(pc2_img).max()
            ax.imshow(pc2_img, cmap="RdBu_r", vmin=-vabs, vmax=vabs,
                      aspect="auto", origin="upper", interpolation="nearest")
            ax.set_title("PC2 loadings (neuron grid)\nred=+, blue=−", fontsize=8)
        else:
            ax.plot(pc2); ax.set_title("PC2 loadings", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines[:].set_visible(False)

    # ── PC scores heatmap: classes × PCs ─────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    n_show = min(n_pc, 6)
    Z_show = Z[:, :n_show]
    vabs   = np.abs(Z_show).max()
    im     = ax.imshow(Z_show, cmap="RdBu_r", vmin=-vabs, vmax=vabs, aspect="auto")
    ax.set_xticks(range(n_show))
    ax.set_xticklabels([f"PC{i+1}" for i in range(n_show)], fontsize=8)
    ax.set_yticks(range(C)); ax.set_yticklabels(classes, fontsize=6)
    plt.colorbar(im, ax=ax, shrink=0.8, label="PC score")
    ax.set_title("PC scores per class\n(rows = classes, cols = PCs)", fontsize=8)

    fig.suptitle(
        f"Layer {layer}  —  PCA of per-class mean activations\n"
        f"PC1 = shared 'central AST structure'  |  PC2+ = class-specific deviations",
        fontsize=11,
    )
    fig.savefig(save, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    → {save.name}  (PC1={ev[0]*100:.1f}%, PC2={ev[1]*100:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 4: Core vs peripheral neurons
# ─────────────────────────────────────────────────────────────────────────────

def plot_core_peripheral(
    mu_c:       np.ndarray,    # (C, H)
    classes:    list[str],
    layer:      int,
    top_k:      int,
    core_frac:  float,
    save:       Path,
    dpi:        int,
) -> dict:
    C, H = mu_c.shape
    # Per-class top-K neuron sets (indices)
    top_sets = [set(np.argsort(mu_c[i])[-top_k:]) for i in range(C)]

    # "Coreness" of each neuron: number of classes it appears in top-K
    count_in_topk = np.zeros(H, dtype=np.int32)
    for s in top_sets:
        for idx in s:
            count_in_topk[idx] += 1

    core_thresh = int(np.ceil(core_frac * C))
    core_mask  = count_in_topk >= core_thresh
    periph_mask = count_in_topk <= PERIPHERAL_MAX
    inter_mask  = ~core_mask & ~periph_mask

    n_core  = int(core_mask.sum())
    n_periph = int(periph_mask.sum())
    n_inter  = int(inter_mask.sum())

    # ── Build class × neuron activation matrix for display ───────────────────
    # Only show top-K neurons; sort columns by coreness (desc), then by mean act
    union_idx = sorted(set().union(*top_sets))
    union_arr  = np.array(union_idx)
    act_sub    = mu_c[:, union_arr]           # (C, |union|)
    coreness   = count_in_topk[union_arr]     # (|union|,)
    # Sort: first by coreness desc, then by mean activation desc
    sort_order = np.lexsort((-act_sub.mean(axis=0), -coreness))
    act_sorted = act_sub[:, sort_order]
    core_sorted = coreness[sort_order]
    union_sorted = union_arr[sort_order]

    # Limit columns for display
    max_show = min(act_sorted.shape[1], 300)
    act_disp  = act_sorted[:, :max_show]
    core_disp = core_sorted[:, None][:max_show]

    fig, axes = plt.subplots(
        2, 1,
        figsize=(max(12, max_show * 0.06 + 2), max(6, C * 0.25 + 3)),
        gridspec_kw={"height_ratios": [1, 8]},
    )

    # ── Top strip: coreness ───────────────────────────────────────────────────
    ax = axes[0]
    ax.imshow(core_sorted[:max_show][None, :],
              cmap="YlOrRd", vmin=0, vmax=C,
              aspect="auto", interpolation="nearest")
    ax.set_yticks([0]); ax.set_yticklabels(["#classes\nin top-K"], fontsize=7)
    ax.set_xticks([]); ax.spines[:].set_visible(False)
    # Mark core / peripheral boundary
    n_core_disp = int((core_sorted[:max_show] >= core_thresh).sum())
    n_periph_disp = int((core_sorted[:max_show] <= PERIPHERAL_MAX).sum())
    if n_core_disp > 0:
        ax.axvline(n_core_disp - 0.5, color="blue", lw=1.2, label=f"core ({n_core})")
    if n_periph_disp > 0:
        ax.axvline(max_show - n_periph_disp - 0.5, color="red", lw=1.2,
                   label=f"peripheral ({n_periph})")
    ax.legend(fontsize=6, loc="upper right")

    # ── Main heatmap: classes × neurons ──────────────────────────────────────
    ax = axes[1]
    vmax = float(act_disp.max()) or 1.0
    im = ax.imshow(act_disp, cmap="viridis_r", vmin=0, vmax=vmax,
                   aspect="auto", interpolation="nearest", origin="upper")
    ax.set_yticks(range(C)); ax.set_yticklabels(classes, fontsize=7)
    ax.set_xlabel(
        f"Neurons (sorted by coreness → mean act)   "
        f"|  core={n_core}  inter={n_inter}  periph={n_periph}",
        fontsize=8,
    )
    ax.set_xticks([]); ax.spines[:].set_visible(False)
    plt.colorbar(im, ax=ax, label="mean act", shrink=0.6, pad=0.01)

    fig.suptitle(
        f"Layer {layer}  —  Core vs peripheral neurons  "
        f"(top-{top_k} per class, core = in top-K for ≥{core_frac*100:.0f}% of classes)",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(save, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    → {save.name}  (core={n_core}, inter={n_inter}, periph={n_periph})")

    return {"n_core": n_core, "n_inter": n_inter, "n_periph": n_periph,
            "core_neurons": union_sorted[:n_core_disp].tolist()}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args    = _parse_args()
    here    = Path(__file__).parent
    base    = here / args.in_dir
    out     = here / (args.out_dir if args.out_dir else args.in_dir)
    fig_dir = out / "figures" / "ast_overlap"
    fig_dir.mkdir(parents=True, exist_ok=True)

    stem = args.stem

    # ── Load meta ─────────────────────────────────────────────────────────────
    meta_path = base / f"01_{stem}_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    ast_labels = np.array([m["ast_node"] for m in meta])
    classes    = sorted(set(ast_labels))
    C          = len(classes)
    print(f"N={len(meta)}  AST classes={C}: {classes}")

    # ── Load activations ──────────────────────────────────────────────────────
    resid_path = base / f"01_{stem}_residual_all.npy"
    resid_all  = np.load(resid_path)          # (N, L+1, H)
    N, L1, H   = resid_all.shape
    print(f"Residual stream shape: {resid_all.shape}")

    # ── Load VP + purity masks ────────────────────────────────────────────────
    vp_r     = dict(np.load(base / f"02_{stem}_vp_residual.npz"))
    masks_npz = dict(np.load(base / f"02_{stem}_purity_masks.npz",
                              allow_pickle=True))

    # ── Layer selection ───────────────────────────────────────────────────────
    layers = list(range(L1))
    if args.layer is not None:
        l = args.layer if args.layer >= 0 else L1 + args.layer
        layers = [int(np.clip(l, 0, L1 - 1))]

    summary: list[dict] = []

    for l in layers:
        print(f"\n── Layer {l} ──")
        act      = resid_all[:, l, :].astype(np.float32)   # (N, H)
        ast_mask = masks_npz.get("residual_ast_pure")
        ast_mask = ast_mask[l].astype(bool) if ast_mask is not None else None

        mu_c = class_means(act, ast_labels, classes)     # (C, H)

        # 1. Jaccard
        J = jaccard_matrix(mu_c, args.top_k)
        plot_jaccard(J, classes, l,
                     fig_dir / f"L{l:02d}_jaccard.png", args.dpi)

        # 2. Centroid decomposition
        res = centroid_analysis(act, ast_labels, classes, ast_mask)
        plot_centroid(res, l,
                      fig_dir / f"L{l:02d}_centroid.png", args.dpi)

        # 3. PCA of class means
        plot_pca_class_means(mu_c, classes, l,
                             fig_dir / f"L{l:02d}_pca_class_means.png", args.dpi)

        # 4. Core / peripheral
        cp = plot_core_peripheral(mu_c, classes, l, args.top_k, args.core_frac,
                                  fig_dir / f"L{l:02d}_core_peripheral.png", args.dpi)

        summary.append({
            "layer":           l,
            "mean_jaccard":    float(_mean_offdiag(J)),
            "mean_cos_global": float(np.mean(res["cos_to_global"])),
            "mean_delta_rel":  float(np.mean(res["delta_rel"])),
            **cp,
        })

    # ── Save summary JSON ─────────────────────────────────────────────────────
    summary_path = out / "ast_overlap_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved → {summary_path}")

    # ── Summary layer trace ───────────────────────────────────────────────────
    if len(summary) > 1:
        ls  = [s["layer"]           for s in summary]
        mj  = [s["mean_jaccard"]    for s in summary]
        mc  = [s["mean_cos_global"] for s in summary]
        mdr = [s["mean_delta_rel"]  for s in summary]
        nc  = [s["n_core"]          for s in summary]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
        axes[0].plot(ls, mj, "o-", color="steelblue")
        axes[0].set_ylabel("Mean Jaccard overlap"); axes[0].set_xlabel("Layer")
        axes[0].set_title("Neuron sharing across AST classes\n(↑ = more central structure)")
        axes[0].spines[["top","right"]].set_visible(False)

        axes[1].plot(ls, mc, "o-", color="seagreen")
        axes[1].set_ylabel("Mean cos(μ_class, μ_global)"); axes[1].set_xlabel("Layer")
        axes[1].set_title("Alignment to global AST mean\n(↑ = stronger central structure)")
        axes[1].set_ylim(0, 1.05)
        axes[1].spines[["top","right"]].set_visible(False)

        axes[2].plot(ls, nc, "o-", color="tomato")
        axes[2].set_ylabel("#core neurons"); axes[2].set_xlabel("Layer")
        axes[2].set_title(f"Core neurons (in top-{args.top_k} for ≥{args.core_frac*100:.0f}% classes)")
        axes[2].spines[["top","right"]].set_visible(False)

        fig.suptitle("AST central structure — layer evolution", fontsize=11)
        fig.tight_layout()
        trace_path = fig_dir / "layer_trace.png"
        fig.savefig(trace_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Layer trace → {trace_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
