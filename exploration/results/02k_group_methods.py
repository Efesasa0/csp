"""
02k_group_methods.py  (lives in results/)

Three methods for finding groups of co-activating residual stream dimensions
that are collectively responsible for each AST class, per layer.

Each method produces one figure per layer:
  rows    = learned groups (components / clusters)
  columns = residual dims sorted by group ownership
  colour  = owning AST class (tab20 palette)
  cell    = component weight (how much that dim contributes to the group)

─────────────────────────────────────────────────────────────────────────────
METHOD A — NMF  (Non-negative Matrix Factorization)
─────────────────────────────────────────────────────────────────────────────
  Input : (N, D_pure)  — activations at ast_pure dims, positive-shifted
  Model : X ≈ W @ H   W∈R^{N×K}, H∈R^{K×D_pure}  all non-negative
  Group : row of H = one component = set of dims that co-activate additively
  Assign: component k → class c = argmax mean(W[prompts_c, k])
  Why   : non-negativity forces purely additive grouping; no cancellation;
          each dim belongs to at most the component with highest loading

─────────────────────────────────────────────────────────────────────────────
METHOD B — Sparse PCA
─────────────────────────────────────────────────────────────────────────────
  Input : (N, D_pure)  — zero-mean activations
  Model : each component = sparse linear direction in dim space
  Group : the non-zero dims in each component's loading vector
  Assign: same as NMF (mean score per class)
  Why   : unlike NMF, can capture both positive and negative co-activation;
          sparsity forces each component to "own" a small set of dims

─────────────────────────────────────────────────────────────────────────────
METHOD C — Co-activation Clustering
─────────────────────────────────────────────────────────────────────────────
  Input : (D_pure, N)  — transpose: cluster the dims not the prompts
  Model : K-means on dims; each cluster = a group of dims with similar
          activation profiles across prompts
  Assign: cluster k → class c = argmax mean activation of cluster dims
          when prompts of class c are present
  Why   : most direct; no linear decomposition assumption; dims in a cluster
          literally co-activate together across the dataset

─────────────────────────────────────────────────────────────────────────────
FIGURE (one per layer × method)
─────────────────────────────────────────────────────────────────────────────
  figures/group_methods/{method}/layer_{l:02d}.pdf

  Layout:
    Main heatmap  — rows=groups (sorted by class), cols=dims (sorted by group)
                    colour = component loading / cluster membership weight
    Left strip    — group label + class colour bar
    Right axis    — class label for each group row
    Title bar     — method, layer, n_pure dims

Usage
─────
  python 02k_group_methods.py
  python 02k_group_methods.py --n_components 48 --layer 5
  python 02k_group_methods.py --methods nmf spca  # skip clustering
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.decomposition import NMF, SparsePCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

FS_TITLE    = 11
FS_SUPTITLE = 13
FS_TICK     = 7
FS_ANNOT    = 6

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--stem",         default="contrastive_stubs")
    p.add_argument("--in_dir",       default="../data_more")
    p.add_argument("--out_dir",      default="../results/output")
    p.add_argument("--n_components", type=int, default=48,
                   help="Groups per layer (NMF / SparsePCA / K-means K)")
    p.add_argument("--spca_alpha",   type=float, default=1.0,
                   help="SparsePCA sparsity penalty")
    p.add_argument("--layer",        type=int, default=None,
                   help="Single layer only (default: all)")
    p.add_argument("--methods",      nargs="+",
                   default=["nmf", "spca", "clustering"],
                   choices=["nmf", "spca", "clustering"])
    p.add_argument("--top_dims",     type=int, default=30,
                   help="Top dims per component shown in figure")
    p.add_argument("--min_samples",  type=int, default=10)
    p.add_argument("--format",       default="pdf")
    p.add_argument("--dpi",          type=int, default=200)
    return p.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Colour helpers
# ─────────────────────────────────────────────────────────────────────────────

def _class_colours(n: int) -> list:
    c1 = plt.get_cmap("tab20")
    c2 = plt.get_cmap("tab20b")
    return [c1(i) if i < 20 else c2(i - 20) for i in range(n)]

# ─────────────────────────────────────────────────────────────────────────────
# Component → class assignment
# ─────────────────────────────────────────────────────────────────────────────

def assign_components_to_classes(scores: np.ndarray,
                                  ast_idx: np.ndarray,
                                  classes: list[str],
                                  min_samples: int) -> np.ndarray:
    """
    scores : (N, K) — per-prompt activation of each component
    Returns owner (K,) int — class index that most activates each component.
    """
    K = scores.shape[1]
    C = len(classes)
    owner = np.zeros(K, dtype=int)
    for k in range(K):
        means = np.array([
            scores[ast_idx == c, k].mean()
            if (ast_idx == c).sum() >= min_samples else -np.inf
            for c in range(C)
        ])
        owner[k] = int(np.argmax(means))
    return owner

# ─────────────────────────────────────────────────────────────────────────────
# METHOD A — NMF
# ─────────────────────────────────────────────────────────────────────────────

def run_nmf(acts: np.ndarray, K: int, rng_seed: int = 42):
    """
    acts : (N, D)  — will be shifted to non-negative before factorisation
    Returns W (N,K), H (K,D) — coefficient matrix and component matrix
    """
    # shift to non-negative: subtract per-dim minimum
    acts_nn = acts - acts.min(axis=0, keepdims=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = NMF(n_components=K, init="nndsvda", max_iter=400,
                    random_state=rng_seed)
        W = model.fit_transform(acts_nn)   # (N, K)
        H = model.components_             # (K, D)
    return W, H

# ─────────────────────────────────────────────────────────────────────────────
# METHOD B — Sparse PCA
# ─────────────────────────────────────────────────────────────────────────────

def run_spca(acts: np.ndarray, K: int, alpha: float, rng_seed: int = 42,
             n_subsample: int = 3000):
    """
    acts : (N, D)
    SparsePCA is slow — subsample prompts for fitting, transform all.
    Returns scores (N,K), components (K,D).
    """
    rng = np.random.default_rng(rng_seed)
    acts_c = acts - acts.mean(axis=0)
    idx = rng.choice(len(acts_c), size=min(n_subsample, len(acts_c)), replace=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SparsePCA(n_components=K, alpha=alpha, max_iter=200,
                          random_state=rng_seed, n_jobs=-1)
        model.fit(acts_c[idx])
    scores = model.transform(acts_c)      # (N, K)
    return scores, model.components_     # (K, D)

# ─────────────────────────────────────────────────────────────────────────────
# METHOD C — Co-activation Clustering
# ─────────────────────────────────────────────────────────────────────────────

def run_clustering(acts: np.ndarray, K: int, rng_seed: int = 42):
    """
    acts : (N, D)
    Cluster the D dims by their N-dim activation profiles → K clusters.
    Returns:
      labels     (D,)   int — cluster id for each dim
      centroids  (K, N) — cluster centroid activation profile
      scores     (N, K) — per-prompt mean activation within each cluster
    """
    # normalise each dim's activation vector to unit length before clustering
    dim_profiles = normalize(acts.T, norm="l2")   # (D, N)
    km = KMeans(n_clusters=K, n_init=5, max_iter=200, random_state=rng_seed)
    labels = km.fit_predict(dim_profiles)          # (D,)

    # cluster loading = mean activation across dims in the cluster
    scores = np.zeros((len(acts), K), dtype=np.float32)
    components = np.zeros((K, acts.shape[1]), dtype=np.float32)
    for k in range(K):
        mask = labels == k
        if mask.sum() > 0:
            scores[:, k]      = acts[:, mask].mean(axis=1)
            components[k, mask] = 1.0   # binary membership

    return scores, components, labels

# ─────────────────────────────────────────────────────────────────────────────
# Figure — one per layer per method
# ─────────────────────────────────────────────────────────────────────────────

def plot_group_heatmap(components: np.ndarray,
                       owner: np.ndarray,
                       pure_dim_ids: np.ndarray,
                       classes: list[str],
                       method: str,
                       layer: int,
                       n_pure: int,
                       top_dims: int,
                       fmt: str,
                       dpi: int,
                       fig_dir: Path) -> None:
    """
    components : (K, D_pure) — loading / weight of each dim in each group
    owner      : (K,)        — class index owning each group
    pure_dim_ids: (D_pure,)  — original residual dim indices

    Builds a heatmap:
      rows    = groups, sorted by owner class then descending max loading
      columns = top-top_dims dims per group (union, deduplicated, sorted by
                argmax group so each dim appears under its strongest group)
      colour  = component loading value (diverging for spca, sequential for nmf/clust)
    """
    K, D = components.shape
    C    = len(classes)
    colours = _class_colours(C)

    # ── select and order columns ──────────────────────────────────────────────
    # top-top_dims dims per group (by absolute loading)
    selected: set[int] = set()
    for k in range(K):
        top = np.argsort(np.abs(components[k]))[-top_dims:]
        selected.update(top.tolist())
    sel_arr = np.array(sorted(selected))          # indices into D_pure space

    sub = components[:, sel_arr]                  # (K, sel)
    # sort columns by argmax group (which group "owns" each dim most)
    col_order = np.argsort(np.abs(sub).argmax(axis=0), kind="stable")
    sub       = sub[:, col_order]
    dim_labels = pure_dim_ids[sel_arr[col_order]] # original residual dim IDs

    # ── sort rows by (owner class, descending max loading) ────────────────────
    row_order = sorted(range(K),
                       key=lambda k: (int(owner[k]),
                                      -float(np.abs(components[k]).max())))
    sub   = sub[row_order, :]
    owner_sorted = owner[row_order]

    # ── figure ────────────────────────────────────────────────────────────────
    row_h   = max(0.18, 4.0 / K)
    fig_h   = max(5.0, K * row_h + 2.0)
    fig_w   = max(10.0, 0.10 * sub.shape[1] + 3.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = "RdBu_r" if method == "spca" else "YlOrRd"
    vmax = float(np.abs(sub).max()) or 1.0
    vmin = -vmax if method == "spca" else 0.0
    im   = ax.imshow(sub, aspect="auto", cmap=cmap,
                     vmin=vmin, vmax=vmax, interpolation="nearest")

    # ── class colour strip on the left ───────────────────────────────────────
    strip_w = 0.018   # axes fraction
    for ri, (row_k, cls_i) in enumerate(zip(row_order, owner_sorted)):
        ax.add_patch(plt.Rectangle(
            (-strip_w * sub.shape[1], ri - 0.5),
            strip_w * sub.shape[1], 1.0,
            transform=ax.transData,
            color=colours[cls_i], clip_on=False, zorder=3,
        ))

    # class separator lines between row groups
    prev = owner_sorted[0]
    for ri in range(1, K):
        if owner_sorted[ri] != prev:
            ax.axhline(ri - 0.5, color="white", lw=1.0, zorder=4)
            prev = owner_sorted[ri]

    # ── axis labels ───────────────────────────────────────────────────────────
    ax.set_yticks(range(K))
    ax.set_yticklabels(
        [f"grp {row_order[k]}  [{classes[owner_sorted[k]]}]" for k in range(K)],
        fontsize=FS_ANNOT,
    )
    # colour each y-tick label by its class
    for tick, cls_i in zip(ax.get_yticklabels(), owner_sorted):
        tick.set_color(colours[cls_i])

    # x-ticks: every N-th dim label to avoid crowding
    n_xtick = max(1, sub.shape[1] // 30)
    xtick_pos = list(range(0, sub.shape[1], n_xtick))
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels([f"d{dim_labels[i]}" for i in xtick_pos],
                       rotation=90, fontsize=FS_ANNOT - 1)
    ax.set_xlabel("Residual stream dimension", fontsize=FS_TITLE)
    ax.set_ylabel("Group  (coloured by owning AST class)", fontsize=FS_TITLE)

    # colourbar
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cb.set_label("Component loading" if method != "clustering" else "Membership",
                 fontsize=FS_ANNOT)
    cb.ax.tick_params(labelsize=FS_ANNOT)

    # legend
    seen = sorted(set(owner_sorted.tolist()))
    handles = [Patch(facecolor=colours[c], label=classes[c]) for c in seen]
    ax.legend(handles=handles, fontsize=5, ncol=4,
              loc="upper left", bbox_to_anchor=(1.02, 1),
              framealpha=0.9, title="AST class", title_fontsize=6)

    method_labels = {"nmf": "NMF", "spca": "Sparse PCA",
                     "clustering": "Co-activation Clustering"}
    ax.set_title(
        f"{method_labels[method]}  ·  Layer {layer}  ·  "
        f"{n_pure} pure dims  ·  {K} groups  ·  {sub.shape[1]} dims shown\n"
        f"Rows sorted by class  ·  columns sorted by dominant group  ·  "
        f"colour = loading magnitude",
        fontsize=FS_ANNOT + 1, pad=6,
    )

    fig.tight_layout()
    out = fig_dir / method / f"layer_{layer:02d}.{fmt}"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {out}")

# ─────────────────────────────────────────────────────────────────────────────
# Summary JSON
# ─────────────────────────────────────────────────────────────────────────────

def save_group_dim_lists(all_results: dict, out_dir: Path, stem: str) -> None:
    """
    Saves a JSON: for each method → layer → class → list of residual dim IDs
    that are in that class's groups (sorted by total loading).
    """
    out = {}
    for method, layers in all_results.items():
        out[method] = {}
        for l, (components, owner, pure_ids) in layers.items():
            out[method][f"L{l:02d}"] = {}
            K = len(owner)
            C_max = int(owner.max()) + 1
            for c in range(C_max):
                grp_mask = owner == c
                if not grp_mask.any():
                    continue
                # aggregate: sum loadings across all groups owned by class c
                agg = np.abs(components[grp_mask]).sum(axis=0)  # (D_pure,)
                # take dims with non-zero total loading, sorted descending
                nonzero = np.where(agg > 0)[0]
                ranked  = nonzero[np.argsort(agg[nonzero])[::-1]]
                out[method][f"L{l:02d}"][str(c)] = pure_ids[ranked].tolist()

    path = out_dir / f"02k_{stem}_group_dim_lists.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  saved {path}")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    in_dir  = Path(__file__).parent / args.in_dir
    out_dir = Path(__file__).parent / args.out_dir
    fig_dir = out_dir / "figures" / "group_methods"
    fig_dir.mkdir(parents=True, exist_ok=True)

    stem = args.stem
    fmt  = args.format
    K    = args.n_components

    print("Loading data …")
    resid  = np.load(in_dir / f"01_{stem}_residual_all.npy")   # (N, L, H)
    pm     = np.load(in_dir / f"02_{stem}_purity_masks.npz")
    meta   = json.load(open(in_dir / f"01_{stem}_meta.json"))
    j      = json.load(open(out_dir / f"02g_{stem}_class_neurons.json"))

    classes     = j["classes"]
    C           = len(classes)
    ast_pure_r  = pm["residual_ast_pure"]  # (L, H)
    N, L, H     = resid.shape
    ast_idx     = np.array([classes.index(m["ast_node"]) for m in meta],
                            dtype=np.int32)

    layers = [args.layer] if args.layer is not None else list(range(L))

    # store results for JSON export: method → {layer: (components, owner, pure_ids)}
    all_results: dict[str, dict] = {m: {} for m in args.methods}

    for l in layers:
        pure_mask = ast_pure_r[l]              # (H,)
        pure_idx  = np.where(pure_mask)[0]     # actual dim indices
        n_pure    = len(pure_idx)

        if n_pure < K:
            print(f"  Layer {l}: only {n_pure} pure dims < K={K}, skipping")
            continue

        acts = resid[:, l, pure_idx].astype(np.float32)   # (N, D_pure)
        print(f"\nLayer {l}  ({n_pure} pure dims) …")

        # ── NMF ──────────────────────────────────────────────────────────────
        if "nmf" in args.methods:
            print("  NMF …")
            W, H_nmf = run_nmf(acts, K)
            owner = assign_components_to_classes(W, ast_idx, classes,
                                                  args.min_samples)
            plot_group_heatmap(H_nmf, owner, pure_idx, classes,
                               "nmf", l, n_pure, args.top_dims,
                               fmt, args.dpi, fig_dir)
            all_results["nmf"][l] = (H_nmf, owner, pure_idx)

        # ── Sparse PCA ───────────────────────────────────────────────────────
        if "spca" in args.methods:
            print("  Sparse PCA …")
            scores, comps = run_spca(acts, K, args.spca_alpha)
            owner = assign_components_to_classes(scores, ast_idx, classes,
                                                  args.min_samples)
            plot_group_heatmap(comps, owner, pure_idx, classes,
                               "spca", l, n_pure, args.top_dims,
                               fmt, args.dpi, fig_dir)
            all_results["spca"][l] = (comps, owner, pure_idx)

        # ── Co-activation Clustering ─────────────────────────────────────────
        if "clustering" in args.methods:
            print("  Co-activation clustering …")
            scores, comps, labels = run_clustering(acts, K)
            owner = assign_components_to_classes(scores, ast_idx, classes,
                                                  args.min_samples)
            plot_group_heatmap(comps, owner, pure_idx, classes,
                               "clustering", l, n_pure, args.top_dims,
                               fmt, args.dpi, fig_dir)
            all_results["clustering"][l] = (comps, owner, pure_idx)

    print("\nSaving group dim lists …")
    save_group_dim_lists(all_results, out_dir, stem)

    print("Done.")


if __name__ == "__main__":
    main()
