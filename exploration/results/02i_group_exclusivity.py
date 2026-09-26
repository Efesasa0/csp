"""
02i_group_exclusivity.py  (lives in results/)

Option 1: Group-level neuron exclusivity.

Even though no single neuron is exclusively owned by an AST class (except
Assert #2218), a *set* of neurons taken together may collectively encode
class identity in a separable way.

METHOD
──────
For each AST class c and each layer l:

  1. Candidate set S_c: top-K MLP neurons by selectivity z(u,c,l), restricted
     to ast_pure neurons.

  2. Group activation vector: for each prompt i,
       g_c(i) = mlp[i, l, S_c]   ∈ R^K

  3. Separability test (permutation MANOVA):
     Statistic T = mean cosine-sim within class c  −  mean cosine-sim outside c
                   using the group activation vectors g_c(i).

     Under H0 (class label permutation), T ~ null distribution.
     p-value = fraction of permutations where T_perm >= T_obs.

  4. Effect size: Cohen's d between within-class and between-class cosine sims.

  5. Cross-class confusion matrix at best layer: for each prompt, assign it to
     the class c* = argmax_c  cos_sim(g_c(prompt), centroid_c).
     Confusion matrix shows which classes are most similar in group-neuron space.

FIGURES
───────
  group_pvalue_heatmap.{fmt}    p-values and Cohen's d per (class, layer)
  group_confusion.{fmt}         classification confusion matrix at best layer
  group_separability_profile.{fmt}  per-class effect size across layers

OUTPUTS (under out_dir)
───────
  02i_<stem>_group_results.json    per-class, per-layer statistics

Usage
─────
  python 02i_group_exclusivity.py
  python 02i_group_exclusivity.py --stem contrastive_stubs --in_dir ../data_more
  python 02i_group_exclusivity.py --top_k 50 --n_perm 1000
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

FS_TITLE    = 11
FS_SUPTITLE = 13
FS_TICK     = 8
FS_ANNOT    = 7

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Group neuron exclusivity test")
    p.add_argument("--stem",      default="contrastive_stubs")
    p.add_argument("--in_dir",    default="../data_more")
    p.add_argument("--out_dir",   default="../results/output")
    p.add_argument("--top_k",     type=int,   default=50,
                   help="Neurons per class group")
    p.add_argument("--n_perm",    type=int,   default=500,
                   help="Permutation iterations for p-value")
    p.add_argument("--min_samples", type=int, default=10)
    p.add_argument("--format",    default="pdf")
    p.add_argument("--dpi",       type=int,   default=200)
    return p.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Core statistics
# ─────────────────────────────────────────────────────────────────────────────

def group_separability(acts_NK: np.ndarray,
                       labels: np.ndarray,
                       c: int,
                       n_perm: int,
                       rng: np.random.Generator,
                       n_subsample: int = 150) -> dict:
    """
    acts_NK:     (N, K) group activation vectors for class c's neuron set
    labels:      (N,)   class indices
    c:           target class
    n_subsample: max prompts to use per class (avoids O(N²) cost)

    Computes within/between cosine similarities directly (no full N×N matrix).
    Permutation test shuffles labels on the subsampled set.

    Returns dict with keys: T_obs, p_value, cohens_d, within_mean, between_mean
    """
    in_idx  = np.where(labels == c)[0]
    out_idx = np.where(labels != c)[0]

    if len(in_idx) < 2 or len(out_idx) < 2:
        return dict(T_obs=0.0, p_value=1.0, cohens_d=0.0,
                    within_mean=0.0, between_mean=0.0)

    # subsample to keep cost manageable
    n_in  = min(n_subsample, len(in_idx))
    n_out = min(n_subsample, len(out_idx))
    in_s  = rng.choice(in_idx,  n_in,  replace=False)
    out_s = rng.choice(out_idx, n_out, replace=False)

    A_in  = acts_NK[in_s].astype(np.float32)    # (n_in,  K)
    A_out = acts_NK[out_s].astype(np.float32)   # (n_out, K)

    def _norm(A):
        return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)

    An_in  = _norm(A_in)
    An_out = _norm(A_out)

    def _stats(Ain, Aout):
        # within: upper-triangle of (n_in × n_in) sim
        W = Ain @ Ain.T                    # (n_in, n_in)
        triu = np.triu_indices(n_in, k=1)
        w_sims = W[triu]
        w_mean = float(w_sims.mean()) if len(w_sims) > 0 else 0.0
        # between: all of (n_in × n_out) sim
        B = Ain @ Aout.T                   # (n_in, n_out)
        b_sims = B.flatten()
        b_mean = float(b_sims.mean())
        return w_mean - b_mean, w_sims, b_sims

    T_obs, w_sims, b_sims = _stats(An_in, An_out)

    # permutation null: shuffle labels within the subsampled pool
    pool      = np.concatenate([A_in, A_out])
    pool_norm = _norm(pool)
    lbl_pool  = np.array([1] * n_in + [0] * n_out)

    T_perm = np.zeros(n_perm, dtype=np.float32)
    for i in range(n_perm):
        sh = rng.permutation(len(lbl_pool))
        sl  = lbl_pool[sh]
        Pi  = pool_norm[sl == 1]
        Po  = pool_norm[sl == 0]
        if len(Pi) < 2 or len(Po) == 0:
            continue
        T_perm[i], _, _ = _stats(Pi, Po)

    p_value  = float((T_perm >= T_obs).mean())
    pooled   = np.sqrt((w_sims.var() + b_sims.var()) / 2 + 1e-9)
    cohens_d = float((w_sims.mean() - b_sims.mean()) / pooled)

    return dict(T_obs=float(T_obs), p_value=p_value,
                cohens_d=cohens_d,
                within_mean=float(w_sims.mean()),
                between_mean=float(b_sims.mean()))


def centroid_classify(group_acts: dict, labels: np.ndarray, classes: list[str]) -> np.ndarray:
    """
    group_acts: dict  c -> (N, K) group activation matrix at best layer
    Assign each prompt to class c* = argmax_c cos_sim(acts_c[i], centroid_c)
    Returns confusion matrix (C, C).
    """
    C = len(classes)
    N = labels.shape[0]
    centroids = {}
    for c in range(C):
        if c in group_acts:
            mask = labels == c
            if mask.sum() > 0:
                centroids[c] = group_acts[c][mask].mean(axis=0)

    # score matrix (N, C)
    score = np.full((N, C), -np.inf, dtype=np.float32)
    for c, mu in centroids.items():
        A = group_acts[c]
        nA = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
        nmu = mu / (np.linalg.norm(mu) + 1e-8)
        score[:, c] = nA @ nmu

    pred = score.argmax(axis=1)
    cm   = np.zeros((C, C), dtype=np.int32)
    for true, pr in zip(labels, pred):
        cm[true, pr] += 1
    return cm

# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def _class_colours(n):
    c1 = plt.get_cmap("tab20")
    c2 = plt.get_cmap("tab20b")
    return [c1(i) if i < 20 else c2(i - 20) for i in range(n)]


def plot_neuron_ownership_strips(sel_all: np.ndarray,
                                  ast_pure_all: np.ndarray,
                                  classes: list[str],
                                  top_k: int,
                                  fmt: str, dpi: int, fig_dir: Path) -> None:
    """
    One panel per layer.  Each panel is a heatmap:
      rows    = AST classes
      columns = top-K residual dims per class (pooled, sorted by argmax class)
      colour  = selectivity z-score

    This directly answers "which residual dims are the group neurons for each class?"
    The block-diagonal structure shows class-specific clustering.
    """
    C, L, H = sel_all.shape
    colours  = _class_colours(C)

    n_cols = min(3, L)
    n_rows = math.ceil(L / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 7, n_rows * (C * 0.22 + 1.2)))

    axes_flat = np.array(axes).flatten()

    for l in range(L):
        ax  = axes_flat[l]
        sel_l = sel_all[:, l, :]           # (C, H)
        pure  = ast_pure_all[l]            # (H,)
        pure_idx = np.where(pure)[0]

        # collect top-k per class (among pure dims only)
        selected = []
        for c in range(C):
            if len(pure_idx) == 0:
                continue
            k = min(top_k, len(pure_idx))
            best = pure_idx[np.argsort(sel_l[c, pure_idx])[-k:]]
            selected.extend(best.tolist())
        selected = np.array(sorted(set(selected)))

        if len(selected) == 0:
            ax.set_visible(False)
            continue

        sub   = sel_l[:, selected]                      # (C, K)
        order = np.argsort(sub.argmax(axis=0), kind="stable")
        sub   = sub[:, order]
        dim_ids = selected[order]

        vmax = max(2.0, float(sub.max()))
        im   = ax.imshow(sub, aspect="auto", cmap="RdBu_r",
                         vmin=-vmax, vmax=vmax, interpolation="nearest")

        # class separators
        owner_prev = sub.argmax(axis=0)[0]
        for k in range(1, len(order)):
            o = sub.argmax(axis=0)[k]
            if o != owner_prev:
                ax.axvline(k - 0.5, color="white", lw=0.5, alpha=0.7)
                owner_prev = o

        ax.set_yticks(range(C))
        ax.set_yticklabels(classes, fontsize=5)
        ax.set_xticks([])
        ax.set_title(
            f"Layer {l}  ·  {int(pure.sum())} pure dims  ·  "
            f"{len(selected)} shown",
            fontsize=FS_ANNOT + 1, pad=3,
        )
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01,
                     label="selectivity z").ax.tick_params(labelsize=6)

    # hide unused axes
    for ax in axes_flat[L:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Residual stream group neurons per AST class, per layer\n"
        f"(top-{top_k} pure dims per class · colour = selectivity z-score · "
        f"columns sorted by owning class → block diagonal = class-specific groups)",
        fontsize=FS_SUPTITLE, y=1.01,
    )
    fig.tight_layout()
    out = fig_dir / f"group_neuron_strips.{fmt}"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def plot_pvalue_heatmap(results: dict, classes: list[str],
                        fmt: str, dpi: int, fig_dir: Path) -> None:
    C = len(classes)
    layers = sorted({int(l) for c in results for l in results[c]})
    L = len(layers)

    pval  = np.ones((C, L))
    cohd  = np.zeros((C, L))
    for ci, cls in enumerate(classes):
        for li, l in enumerate(layers):
            r = results.get(cls, {}).get(str(l), {})
            pval[ci, li] = r.get("p_value", 1.0)
            cohd[ci, li] = r.get("cohens_d", 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, max(5, C * 0.28 + 1)),
                             gridspec_kw={"wspace": 0.08})

    for ax, data, label, cmap, vmin, vmax in [
        (axes[0], -np.log10(pval + 1e-3), "−log₁₀(p)", "YlOrRd", 0, 3),
        (axes[1], cohd,                    "Cohen's d",  "RdYlGn", -1, 3),
    ]:
        im = ax.imshow(data, aspect="auto", cmap=cmap,
                       vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_xticks(range(L))
        ax.set_xticklabels([f"L{l}" for l in layers], fontsize=FS_TICK)
        ax.set_yticks(range(C))
        ax.set_yticklabels(classes if ax is axes[0] else [], fontsize=FS_TICK)
        ax.set_xlabel("Layer", fontsize=FS_TITLE)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).set_label(label, fontsize=FS_ANNOT)

        # significance markers
        sig = pval < 0.05
        for ci in range(C):
            for li in range(L):
                if sig[ci, li]:
                    ax.text(li, ci, "*", ha="center", va="center",
                            fontsize=7, color="black", fontweight="bold")

    axes[0].set_title("Group separability  (p-value)", fontsize=FS_TITLE)
    axes[1].set_title("Effect size  (Cohen's d)", fontsize=FS_TITLE)
    fig.suptitle(f"Group neuron exclusivity — top-K neuron sets per AST class\n"
                 f"* = p < 0.05  (permutation test, n_perm iterations)",
                 fontsize=FS_SUPTITLE, y=1.01)
    fig.savefig(fig_dir / f"group_pvalue_heatmap.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fig_dir / f'group_pvalue_heatmap.{fmt}'}")


def plot_separability_profile(results: dict, classes: list[str],
                               fmt: str, dpi: int, fig_dir: Path) -> None:
    """Cohen's d per layer per class — line plot."""
    colours = _class_colours(len(classes))
    layers  = sorted({int(l) for c in results for l in results[c]})

    fig, ax = plt.subplots(figsize=(8, 5))
    for ci, cls in enumerate(classes):
        ds = [results.get(cls, {}).get(str(l), {}).get("cohens_d", 0.0)
              for l in layers]
        ax.plot(layers, ds, lw=1.2, color=colours[ci], alpha=0.7, label=cls)

    ax.axhline(0, lw=0.8, color="grey", ls="--")
    ax.set_xlabel("Layer", fontsize=FS_TITLE)
    ax.set_ylabel("Cohen's d  (group separability)", fontsize=FS_TITLE)
    ax.set_title("Group neuron separability profile per AST class across layers",
                 fontsize=FS_SUPTITLE)
    ax.legend(fontsize=5, ncol=4, loc="upper left",
              bbox_to_anchor=(1.01, 1), framealpha=0.8)
    fig.tight_layout()
    fig.savefig(fig_dir / f"group_separability_profile.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fig_dir / f'group_separability_profile.{fmt}'}")


def plot_confusion(cm: np.ndarray, classes: list[str],
                   best_layer: int, fmt: str, dpi: int, fig_dir: Path) -> None:
    C = len(classes)
    # row-normalise
    row_sum = cm.sum(axis=1, keepdims=True).clip(1)
    cm_norm = cm / row_sum

    fig, ax = plt.subplots(figsize=(max(8, C * 0.35), max(7, C * 0.32)))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02).set_label(
        "Fraction assigned to column class", fontsize=FS_ANNOT)

    ax.set_xticks(range(C))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(C))
    ax.set_yticklabels(classes, fontsize=6)
    ax.set_xlabel("Predicted class", fontsize=FS_TITLE)
    ax.set_ylabel("True class", fontsize=FS_TITLE)
    ax.set_title(
        f"Group centroid classification at layer {best_layer}\n"
        f"(centroid = mean group-neuron activation vector per class)",
        fontsize=FS_SUPTITLE,
    )

    # diagonal accuracy annotations
    for c in range(C):
        ax.text(c, c, f"{cm_norm[c, c]:.2f}",
                ha="center", va="center", fontsize=5,
                color="white" if cm_norm[c, c] > 0.5 else "black")

    fig.tight_layout()
    fig.savefig(fig_dir / f"group_confusion.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fig_dir / f'group_confusion.{fmt}'}")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    in_dir  = Path(__file__).parent / args.in_dir
    out_dir = Path(__file__).parent / args.out_dir
    fig_dir = out_dir / "figures" / "group_exclusivity"
    fig_dir.mkdir(parents=True, exist_ok=True)

    stem = args.stem
    fmt  = args.format

    print("Loading data …")
    sel_r   = np.load(out_dir / f"02g_{stem}_selectivity_residual.npz")
    pm      = np.load(in_dir  / f"02_{stem}_purity_masks.npz")
    resid   = np.load(in_dir  / f"01_{stem}_residual_all.npy")   # (N, L, H)
    meta    = json.load(open(in_dir / f"01_{stem}_meta.json"))
    j       = json.load(open(out_dir / f"02g_{stem}_class_neurons.json"))

    classes    = j["classes"]
    C          = len(classes)
    sel        = sel_r["sel"]             # (C, L, H)
    ast_pure_r = pm["residual_ast_pure"]  # (L, H)

    N, L, H = resid.shape
    ast_idx = np.array([classes.index(m["ast_node"]) for m in meta], dtype=np.int32)

    rng = np.random.default_rng(42)

    results: dict[str, dict] = {cls: {} for cls in classes}

    # ── per-class top-K dim lists (the main output) ───────────────────────────
    neuron_lists: dict = {cls: {"per_layer": {}, "any_layer": []} for cls in classes}

    print(f"Extracting top-{args.top_k} residual dims per class per layer …")
    for l in range(L):
        pure_mask = ast_pure_r[l]          # (H,)
        sel_l     = sel[:, l, :]           # (C, H)
        pure_idx  = np.where(pure_mask)[0]

        for ci, cls in enumerate(classes):
            if len(pure_idx) == 0:
                neuron_lists[cls]["per_layer"][f"L{l:02d}"] = []
                continue
            k    = min(args.top_k, len(pure_idx))
            best = pure_idx[np.argsort(sel_l[ci, pure_idx])[-k:]]
            # sort by descending selectivity for readability
            best = best[np.argsort(sel_l[ci, best])[::-1]]
            neuron_lists[cls]["per_layer"][f"L{l:02d}"] = best.tolist()

    for cls in classes:
        union = sorted(set(
            d for layer_dims in neuron_lists[cls]["per_layer"].values()
            for d in layer_dims
        ))
        neuron_lists[cls]["any_layer"] = union

    out_lists = out_dir / f"02i_{stem}_neuron_lists.json"
    with open(out_lists, "w") as f:
        json.dump({"classes": classes, "residual_dims": neuron_lists}, f, indent=2)
    print(f"  saved {out_lists}")

    # ── group separability stats ──────────────────────────────────────────────
    print(f"Running group separability (top_k={args.top_k}, n_perm={args.n_perm}) …")
    for l in range(L):
        print(f"  Layer {l} …")
        pure_mask = ast_pure_r[l]
        pure_idx  = np.where(pure_mask)[0]
        sel_l     = sel[:, l, :]
        acts_l    = resid[:, l, :]         # (N, H)

        for ci, cls in enumerate(classes):
            if len(pure_idx) == 0 or (ast_idx == ci).sum() < args.min_samples:
                continue
            k    = min(args.top_k, len(pure_idx))
            best = pure_idx[np.argsort(sel_l[ci, pure_idx])[-k:]]
            acts_NK = acts_l[:, best]
            stat = group_separability(acts_NK, ast_idx, ci, args.n_perm, rng)
            results[cls][str(l)] = stat

    out_json = out_dir / f"02i_{stem}_group_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  saved {out_json}")

    layer_ds = {l: float(np.mean([results[cls].get(str(l), {}).get("cohens_d", 0.0)
                                   for cls in classes])) for l in range(L)}
    best_layer = max(layer_ds, key=lambda l: layer_ds[l])
    print(f"  Best layer by mean Cohen's d: {best_layer}  (d={layer_ds[best_layer]:.3f})")

    # confusion matrix at best layer
    print(f"Building confusion matrix at layer {best_layer} …")
    pure_mask = ast_pure_r[best_layer]
    pure_idx  = np.where(pure_mask)[0]
    group_acts = {}
    for ci, cls in enumerate(classes):
        if len(pure_idx) == 0:
            continue
        k = min(args.top_k, len(pure_idx))
        best_dims = pure_idx[np.argsort(sel[:, best_layer, :][ci, pure_idx])[-k:]]
        group_acts[ci] = resid[:, best_layer, :][:, best_dims]

    cm = centroid_classify(group_acts, ast_idx, classes)
    acc = cm.diagonal().sum() / cm.sum()
    print(f"  Group centroid accuracy: {acc:.3f}")

    print("Plotting …")
    plot_neuron_ownership_strips(sel, ast_pure_r, classes, args.top_k,
                                 fmt, args.dpi, fig_dir)
    plot_separability_profile(results, classes, fmt, args.dpi, fig_dir)
    plot_confusion(cm, classes, best_layer, fmt, args.dpi, fig_dir)

    print("Done.")


if __name__ == "__main__":
    main()
