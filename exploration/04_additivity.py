"""
04_additivity.py

Step 4 of the AST x builtin mechanistic interpretability pipeline.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY WE DO THIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Steps 02 and 03 answered "which units carry each factor and how separable are
their geometries?"  Step 04 asks a sharper causal question:

  ADDITIVITY TEST
  ---------------
  If AST and builtin representations are independent (orthogonal subspaces),
  then the combined activation should equal the sum of the individual baseline
  activations:

      act_combined  ~  act_ast_baseline + act_builtin_baseline

  We measure this per (AST, builtin) pair at every layer:

      additivity_score(layer) = cos_sim(
          act_combined[layer],
          act_ast_baseline[layer] + act_builtin_baseline[layer]
      )

  score ~  1.0  =>  additive / independent: the model stores AST and
                    builtin information independently and combines them
                    linearly.  This is the "clean circuit" hypothesis.

  score << 1.0  =>  interaction circuit: the (AST, builtin) pair encodes
                    something that cannot be decomposed as a sum.  The model
                    has learned a joint concept rather than two separate ones.

  EXPLICIT VS PROXY TEST
  ----------------------
  Proxy stubs replace the builtin token with a semantically equivalent
  expression that does NOT use the builtin name (e.g. len(x) -> x.__len__(),
  str(x) -> f"{x}", list(x) -> [*x]).

  If the additivity score is SIMILAR between explicit and proxy stubs for the
  same (AST, builtin) pair, then the model encodes the SEMANTIC concept of the
  builtin rather than just recognising its token.

  If there is a large GAP (explicit >> proxy), the model is doing lexical
  pattern matching on the builtin token rather than understanding its semantics.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTAMINATION AND HOW WE CONTROL FOR IT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Raw additivity is biased if AST and builtin representations are entangled:
  - mu_ast[A] contains some builtin signal   (AST baseline prompts still
    fire builtin-related features in passing)
  - mu_builtin[B] contains some AST signal
  - So mu_ast + mu_builtin is inflated, and cos_sim may look high even when
    the representations are NOT truly additive.

Two decontamination strategies are run in addition to the raw test:

  OPTION A — Purity-mask projection  (primary, requires step 02 outputs)
  -----------------------------------------------------------------------
  Use the variance-partition purity masks from step 02 to restrict analysis
  to dimensions that are known to carry *clean* signal:

    union_mask[layer] = ast_pure_mask[layer] | builtin_pure_mask[layer]

  All activations and baselines are element-wise masked (non-pure dims zeroed)
  before computing cos_sim.  This removes the contaminated dimensions entirely.
  Baselines computed in the masked space are also free of cross-factor signal
  (by construction of the purity mask).  Additivity scores from this mode are
  the most trustworthy.

  OPTION B — Regression-direction removal  (aside, no extra inputs needed)
  -------------------------------------------------------------------------
  A rougher correction that does not need step 02.  For each baseline vector:

    mu_ast_corrected[A][l]  = mu_ast[A][l]
                              - proj(mu_ast[A][l],  onto mean_builtin_dir[l])
    mu_b_corrected[B][l]    = mu_builtin[B][l]
                              - proj(mu_b[B][l],    onto mean_ast_dir[l])

  where mean_builtin_dir / mean_ast_dir are the global mean directions across
  all builtin / AST baseline vectors.  This removes the shared "DC offset"
  direction from each baseline before summing.  Less principled than Option A
  but useful as a sanity check when step 02 outputs are not available.

  COMPARISON PLOT
  ---------------
  A three-way plot (raw / masked / regression-corrected) is produced showing
  mean additivity per layer for each method.  Convergence of all three lines
  indicates robust additivity.  Divergence (raw high, masked/corrected low)
  indicates the raw score was inflated by contamination.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW IT IS COMPUTED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Baseline mean vectors:
  For each AST node A:
      mu_ast[A][layer]     = mean over all baseline_ast stubs with ast_node==A
  For each builtin B:
      mu_builtin[B][layer] = mean over all baseline_builtin stubs with builtin==B

  "baseline_ast" stubs: simple prompts using AST node A but NO named builtin
  "baseline_builtin" stubs: simple prompts using builtin B in a neutral Assign

Additivity score per explicit stub (prompt_text t, ast_node A, builtin B):
  act_t         = residual_all[idx_t]            (L+1, H)
  predicted     = mu_ast[A] + mu_builtin[B]       (L+1, H)  — element-wise sum
  score[layer]  = cos_sim(act_t[layer], predicted[layer])

  Additional diagnostics per stub:
    ast_alignment[layer]     = cos_sim(act_t[layer], mu_ast[A][layer])
    builtin_alignment[layer] = cos_sim(act_t[layer], mu_builtin[B][layer])
    interaction_norm[layer]  = ||act_t[layer] - predicted[layer]||
    ast_builtin_ortho[layer] = cos_sim(mu_ast[A][layer], mu_builtin[B][layer])

Statistical significance:
  Permutation test: shuffle builtin labels, recompute mean additivity,
  build null distribution.  p-value = fraction of null >= observed.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUT FILES  (from step 01)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  01_<stem>_residual_all.npy    float32  (N, L+1, H)
  01_<stem>_meta.json           per-prompt metadata including variant_type,
                                ast_node, builtin_obj

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FILES  (prefix 04_<stem>_)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  04_<stem>_baselines_ast.npz           mu_ast per AST node    (n_ast, L+1, H)
  04_<stem>_baselines_builtin.npz       mu_builtin per builtin (n_b,   L+1, H)
  04_<stem>_additivity_scores.npz       per-stub scores (raw) at every layer
  04_<stem>_additivity_scores_masked.npz  per-stub scores (purity-masked, Opt A)
  04_<stem>_additivity_scores_corrected.npz per-stub scores (regression, Opt B)
  04_<stem>_pair_summary.json           aggregated score per (AST, builtin) pair
  04_<stem>_significance.json           permutation test p-values per layer
  04_<stem>_explicit_vs_proxy.json      per-pair gap: explicit score - proxy score

Usage
-----
  python 04_additivity.py --stem contrastive_stubs
  python 04_additivity.py --stem contrastive_stubs --plot --n_perm 2000
  python 04_additivity.py --stem contrastive_stubs --plot --layer -1
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import mannwhitneyu, ttest_rel, pearsonr
from scipy.spatial.distance import squareform


# ─────────────────────────────────────────────────────────────────────────────
# Cosine similarity (vectorised)
# ─────────────────────────────────────────────────────────────────────────────

def _cos_sim_vecs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between two (L+1, H) arrays, computed per layer.
    Returns 1-D array of length L+1.
    """
    dot   = (a * b).sum(axis=-1)                          # (L+1,)
    na    = np.linalg.norm(a, axis=-1).clip(1e-12)        # (L+1,)
    nb    = np.linalg.norm(b, axis=-1).clip(1e-12)        # (L+1,)
    return np.clip(dot / (na * nb), -1.0, 1.0)


def _cos_sim_flat(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / denom)


# ─────────────────────────────────────────────────────────────────────────────
# Baseline mean computation
# ─────────────────────────────────────────────────────────────────────────────

def build_baselines(
    resid_all: np.ndarray,    # (N, L+1, H)
    meta:      list[dict],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, list[int]]]:
    """
    Compute per-class mean residual vectors for AST baselines and builtin baselines.

    Returns
    -------
    mu_ast     : dict  ast_node -> (L+1, H)  mean activation over baseline_ast stubs
    mu_builtin : dict  builtin  -> (L+1, H)  mean activation over baseline_builtin stubs
    idx_map    : dict  "explicit" / "proxy" / "baseline_ast" / "baseline_builtin"
                       -> list of row indices in resid_all
    """
    # Partition indices by variant_type
    idx_map: dict[str, list[int]] = defaultdict(list)
    for i, m in enumerate(meta):
        vtype = m.get("variant_type", "explicit")
        idx_map[vtype].append(i)

    # AST baselines: group by ast_node
    ast_groups: dict[str, list[int]] = defaultdict(list)
    for i in idx_map["baseline_ast"]:
        ast_groups[meta[i]["ast_node"]].append(i)

    mu_ast: dict[str, np.ndarray] = {}
    for node, idxs in ast_groups.items():
        mu_ast[node] = resid_all[idxs].mean(axis=0)    # (L+1, H)

    # Builtin baselines: group by builtin_obj
    b_groups: dict[str, list[int]] = defaultdict(list)
    for i in idx_map["baseline_builtin"]:
        b_groups[meta[i]["builtin_obj"]].append(i)

    mu_builtin: dict[str, np.ndarray] = {}
    for b, idxs in b_groups.items():
        mu_builtin[b] = resid_all[idxs].mean(axis=0)   # (L+1, H)

    print(f"  AST baselines    : {len(mu_ast)} nodes  "
          f"({len(idx_map['baseline_ast'])} stubs)")
    print(f"  Builtin baselines: {len(mu_builtin)} builtins  "
          f"({len(idx_map['baseline_builtin'])} stubs)")
    print(f"  Explicit stubs   : {len(idx_map['explicit'])}")
    print(f"  Proxy stubs      : {len(idx_map['proxy'])}")

    return mu_ast, mu_builtin, dict(idx_map)


# ─────────────────────────────────────────────────────────────────────────────
# Additivity test — per stub, all layers
# ─────────────────────────────────────────────────────────────────────────────

def run_additivity(
    resid_all:  np.ndarray,          # (N, L+1, H)
    meta:       list[dict],
    mu_ast:     dict[str, np.ndarray],
    mu_builtin: dict[str, np.ndarray],
    variant_types: list[str] = ("explicit", "proxy"),
) -> list[dict]:
    """
    For every stub of the requested variant_type(s), compute per-layer:
      - additivity_score   : cos_sim(act, mu_ast + mu_builtin)
      - ast_alignment      : cos_sim(act, mu_ast)
      - builtin_alignment  : cos_sim(act, mu_builtin)
      - interaction_norm   : ||act - (mu_ast + mu_builtin)||_2
      - ast_builtin_ortho  : cos_sim(mu_ast, mu_builtin)

    Returns list of dicts, one per scored stub.  Each dict contains:
      prompt_id, ast_node, builtin_obj, variant_type, category
      additivity        : (L+1,) array
      ast_alignment     : (L+1,)
      builtin_alignment : (L+1,)
      interaction_norm  : (L+1,)
      ast_builtin_ortho : (L+1,)
    """
    results = []
    skipped = 0

    for i, m in enumerate(meta):
        if m.get("variant_type", "explicit") not in variant_types:
            continue

        ast_node = m["ast_node"]
        builtin  = m["builtin_obj"]

        if ast_node not in mu_ast or builtin not in mu_builtin:
            skipped += 1
            continue

        act       = resid_all[i]                        # (L+1, H)
        mu_a      = mu_ast[ast_node]                    # (L+1, H)
        mu_b      = mu_builtin[builtin]                 # (L+1, H)
        predicted = mu_a + mu_b                         # (L+1, H)

        results.append({
            "prompt_id":        m.get("prompt_id", i),
            "ast_node":         ast_node,
            "builtin_obj":      builtin,
            "variant_type":     m.get("variant_type", "explicit"),
            "category":         m.get("category", ""),
            "additivity":       _cos_sim_vecs(act, predicted),         # (L+1,)
            "ast_alignment":    _cos_sim_vecs(act, mu_a),              # (L+1,)
            "builtin_alignment":_cos_sim_vecs(act, mu_b),              # (L+1,)
            "interaction_norm": np.linalg.norm(act - predicted, axis=-1),  # (L+1,)
            "ast_builtin_ortho":_cos_sim_vecs(mu_a, mu_b),            # (L+1,)
        })

    print(f"  Scored {len(results)} stubs  ({skipped} skipped — missing baseline)")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation helpers
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_by_pair(
    results: list[dict],
    metric:  str = "additivity",
    layer:   int = -1,
) -> dict[tuple[str, str], float]:
    """Mean score per (ast_node, builtin_obj) pair at the given layer."""
    sums:   dict[tuple, float] = defaultdict(float)
    counts: dict[tuple, int]   = defaultdict(int)
    for r in results:
        key = (r["ast_node"], r["builtin_obj"])
        sums[key]   += float(r[metric][layer])
        counts[key] += 1
    return {k: sums[k] / counts[k] for k in sums}


def aggregate_by_layer(
    results: list[dict],
    metric:  str = "additivity",
) -> np.ndarray:
    """Mean score across all stubs, per layer.  Returns (L+1,)."""
    if not results:
        return np.array([])
    stack = np.stack([r[metric] for r in results])   # (n_stubs, L+1)
    return stack.mean(axis=0)


def top_k_pairs(
    pair_scores: dict[tuple, float],
    k:           int = 10,
    lowest:      bool = True,
) -> list[tuple[tuple, float]]:
    return sorted(pair_scores.items(), key=lambda x: x[1], reverse=not lowest)[:k]


# ─────────────────────────────────────────────────────────────────────────────
# Permutation significance test
# ─────────────────────────────────────────────────────────────────────────────

def permutation_significance(
    results:    list[dict],
    mu_ast:     dict[str, np.ndarray],
    mu_builtin: dict[str, np.ndarray],
    resid_all:  np.ndarray,
    meta:       list[dict],
    n_perm:     int = 1000,
    rng:        np.random.Generator | None = None,
) -> dict:
    """
    Null hypothesis: the additivity score is no better than chance.

    Method: shuffle the builtin labels within each explicit stub (keeping
    AST labels fixed), recompute predicted = mu_ast[A] + mu_builtin[B_shuffled],
    repeat n_perm times.

    Returns dict with:
      observed_mean   : (L+1,) observed mean additivity per layer
      null_mean       : (n_perm, L+1) null distribution
      p_values        : (L+1,) fraction of null >= observed
      significant     : (L+1,) bool array at alpha=0.05
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Collect explicit stubs that have both baselines
    valid = [
        (i, m) for i, m in enumerate(meta)
        if m.get("variant_type", "explicit") == "explicit"
        and m["ast_node"] in mu_ast
        and m["builtin_obj"] in mu_builtin
    ]
    if not valid:
        return {}

    idxs, metas = zip(*valid)
    acts      = resid_all[list(idxs)]          # (n_valid, L+1, H)
    ast_nodes = [m["ast_node"] for m in metas]
    builtins  = [m["builtin_obj"] for m in metas]
    all_builtins = list(mu_builtin.keys())

    def _mean_additivity(b_labels: list[str]) -> np.ndarray:
        scores = []
        for j, (ast_n, b_n) in enumerate(zip(ast_nodes, b_labels)):
            if b_n not in mu_builtin:
                continue
            pred  = mu_ast[ast_n] + mu_builtin[b_n]
            scores.append(_cos_sim_vecs(acts[j], pred))
        return np.mean(scores, axis=0) if scores else np.zeros(acts.shape[1])

    # Observed
    observed = _mean_additivity(builtins)

    # Null distribution
    null = np.zeros((n_perm, acts.shape[1]), dtype=np.float32)
    for p in range(n_perm):
        shuffled = rng.choice(all_builtins, size=len(builtins), replace=True).tolist()
        null[p]  = _mean_additivity(shuffled)
        if (p + 1) % 200 == 0:
            print(f"  permutation {p+1}/{n_perm}", end="\r")
    print()

    p_values   = ((null >= observed[None, :]).sum(axis=0) + 1) / (n_perm + 1)
    significant = p_values < 0.05

    return {
        "observed_mean": observed,
        "null_mean":     null,
        "p_values":      p_values,
        "significant":   significant,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Explicit vs proxy comparison
# ─────────────────────────────────────────────────────────────────────────────

def explicit_vs_proxy(
    explicit_results: list[dict],
    proxy_results:    list[dict],
    layer:            int = -1,
) -> list[dict]:
    """
    For each (ast_node, builtin_obj) pair that appears in BOTH explicit and
    proxy results, compute the gap in additivity score at the given layer:

      gap = mean_explicit_score - mean_proxy_score

    Positive gap  -> model uses the builtin TOKEN (lexical encoding).
    Gap near zero -> model encodes the builtin SEMANTICS regardless of token.
    Negative gap  -> proxy activations are actually MORE additive (unusual).

    Returns list of dicts sorted by gap (largest first).
    """
    ex_by_pair = defaultdict(list)
    for r in explicit_results:
        ex_by_pair[(r["ast_node"], r["builtin_obj"])].append(float(r["additivity"][layer]))

    pr_by_pair = defaultdict(list)
    for r in proxy_results:
        pr_by_pair[(r["ast_node"], r["builtin_obj"])].append(float(r["additivity"][layer]))

    common = sorted(set(ex_by_pair) & set(pr_by_pair))
    out = []
    for pair in common:
        ex_mean = float(np.mean(ex_by_pair[pair]))
        pr_mean = float(np.mean(pr_by_pair[pair]))
        out.append({
            "ast_node":      pair[0],
            "builtin_obj":   pair[1],
            "explicit_mean": ex_mean,
            "proxy_mean":    pr_mean,
            "gap":           ex_mean - pr_mean,
            "n_explicit":    len(ex_by_pair[pair]),
            "n_proxy":       len(pr_by_pair[pair]),
        })

    out.sort(key=lambda x: x["gap"], reverse=True)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Baseline cosine similarity matrices
# ─────────────────────────────────────────────────────────────────────────────

def baseline_cosine_matrices(
    mu_ast:     dict[str, np.ndarray],   # ast_node -> (L+1, H)
    mu_builtin: dict[str, np.ndarray],   # builtin -> (L+1, H)
    layer:      int = -1,
) -> tuple[list[str], np.ndarray, list[str], np.ndarray, np.ndarray]:
    """
    Compute cosine similarity matrices:
      ast_mat[i,j]     = cos_sim(mu_ast[i][layer], mu_ast[j][layer])
      builtin_mat[i,j] = cos_sim(mu_builtin[i][layer], mu_builtin[j][layer])
      cross_mat[i,j]   = cos_sim(mu_ast[i][layer], mu_builtin[j][layer])

    Near-diagonal ast_mat and builtin_mat -> each concept has a distinct
    activation direction -> representations are separable.
    High cross_mat values -> AST and builtin directions are aligned -> entangled.
    """
    ast_names = sorted(mu_ast.keys())
    b_names   = sorted(mu_builtin.keys())

    def _vec(mu: dict, name: str) -> np.ndarray:
        v = mu[name][layer]
        n = np.linalg.norm(v) + 1e-12
        return v / n

    ast_vecs = np.stack([_vec(mu_ast, n)    for n in ast_names])    # (n_ast, H)
    b_vecs   = np.stack([_vec(mu_builtin,n) for n in b_names])      # (n_b, H)

    ast_mat   = np.clip(ast_vecs @ ast_vecs.T, -1, 1).astype(np.float32)
    b_mat     = np.clip(b_vecs   @ b_vecs.T,   -1, 1).astype(np.float32)
    cross_mat = np.clip(ast_vecs @ b_vecs.T,   -1, 1).astype(np.float32)

    # Sort by numerical structure (seriation from hierarchical clustering), not alphabetically.
    # This creates a continuous flow in the heatmap.
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform

    def _seriate(mat: np.ndarray) -> np.ndarray:
        n = mat.shape[0]
        if n <= 2:
            return np.arange(n)

        # Use 1 - similarity as distance; ignore diagonal.
        dist_mat = 1.0 - mat
        np.fill_diagonal(dist_mat, 0.0)
        dist_condensed = squareform(dist_mat, checks=False)
        Z = linkage(dist_condensed, method="average")
        return leaves_list(Z)

    ast_order = _seriate(ast_mat)
    b_order   = _seriate(b_mat)

    ast_names = [ast_names[i] for i in ast_order]
    b_names   = [b_names[i]   for i in b_order]

    ast_mat   = ast_mat[ast_order, :][:, ast_order]
    b_mat     = b_mat[b_order, :][:, b_order]
    cross_mat = cross_mat[ast_order, :][:, b_order]

    return ast_names, ast_mat, b_names, b_mat, cross_mat


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

CMAP_DIV = LinearSegmentedColormap.from_list(
    "div2", ["#1565C0", "#FFFFFF", "#E65100"], N=256
)


def _heatmap(ax, mat, row_names, col_names, title, vmin=-0.2, vmax=1.0, annotate=True):
    im = ax.imshow(mat, aspect="auto", cmap=CMAP_DIV, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    ax.set_xticks(range(len(col_names)))
    ax.set_yticks(range(len(row_names)))
    fs = max(4, 8 - len(col_names) // 10)
    ax.set_xticklabels(col_names, rotation=45, ha="right", fontsize=fs)
    ax.set_yticklabels(row_names, fontsize=fs)
    ax.set_title(title, fontsize=9)
    if annotate and len(row_names) <= 20 and len(col_names) <= 20:
        for i in range(len(row_names)):
            for j in range(len(col_names)):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                        fontsize=5, color="black")
    return im


def plot_additivity_heatmap(
    explicit_results: list[dict],
    layer:     int = -1,
    title:     str = "Additivity score — explicit stubs",
    save_path: Path | None = None,
) -> None:
    """
    Heatmap of mean additivity score at `layer` for every (AST, builtin) pair.
    Green cells = additive (independent).  Red/blue cells = interaction circuit.
    Grey = no data.
    """
    from scipy.cluster.hierarchy import linkage, leaves_list

    pair_scores = aggregate_by_pair(explicit_results, "additivity", layer)
    pair_counts: dict[tuple, int] = defaultdict(int)
    for r in explicit_results:
        pair_counts[(r["ast_node"], r["builtin_obj"])] += 1

    ast_nodes = sorted({k[0] for k in pair_scores})
    builtins  = sorted({k[1] for k in pair_scores})

    grid = np.full((len(ast_nodes), len(builtins)), np.nan)
    for i, ast in enumerate(ast_nodes):
        for j, b in enumerate(builtins):
            if (ast, b) in pair_scores:
                grid[i, j] = pair_scores[(ast, b)]

    def _seriate_axis(mat: np.ndarray, names: list[str]) -> list[int]:
        if len(names) <= 2:
            return list(range(len(names)))
        arr = np.nan_to_num(mat, nan=np.nanmean(mat))
        dist = 1.0 - arr
        np.fill_diagonal(dist, 0.0)
        # condensed format
        d = squareform(dist, checks=False)
        z = linkage(d, method="average")
        return leaves_list(z).tolist()

    ast_order = _seriate_axis(grid, ast_nodes)
    builtin_order = _seriate_axis(grid.T, builtins)

    ast_nodes = [ast_nodes[i] for i in ast_order]
    builtins  = [builtins[i] for i in builtin_order]
    grid      = grid[np.ix_(ast_order, builtin_order)]

    grid = np.full((len(ast_nodes), len(builtins)), np.nan)
    for i, ast in enumerate(ast_nodes):
        for j, b in enumerate(builtins):
            if (ast, b) in pair_scores:
                grid[i, j] = pair_scores[(ast, b)]

    fig, ax = plt.subplots(figsize=(max(12, len(builtins) * 0.5),
                                     max(8,  len(ast_nodes) * 0.4)))
    cmap = plt.colormaps["RdYlGn"].copy()
    cmap.set_bad(color="#e0e0e0")
    im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax,
                 label="Additivity score (1.0 = fully additive / independent)")

    ax.set_xticks(range(len(builtins)))
    ax.set_xticklabels(builtins, rotation=45, ha="right",
                       fontsize=max(4, 8 - len(builtins) // 10))
    ax.set_yticks(range(len(ast_nodes)))
    ax.set_yticklabels(ast_nodes, fontsize=max(5, 9 - len(ast_nodes) // 10))
    ax.set_xlabel("Builtin")
    ax.set_ylabel("AST node")
    ax.set_title(title, fontsize=11)

    if len(ast_nodes) <= 20 and len(builtins) <= 20:
        for i in range(len(ast_nodes)):
            for j in range(len(builtins)):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, f"{grid[i,j]:.2f}", ha="center", va="center",
                            fontsize=5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    plt.close(fig)


def plot_layer_traces(
    explicit_results: list[dict],
    top_k:     int = 10,
    title:     str = "Additivity score across layers",
    save_path: Path | None = None,
) -> None:
    """
    Line plot of per-layer mean additivity for the top_k most and least
    additive (AST, builtin) pairs.  Shows where in the model interactions
    emerge (sharp dip = interaction formed at that layer).
    """
    final_scores = aggregate_by_pair(explicit_results, "additivity", layer=-1)
    worst_pairs  = top_k_pairs(final_scores, k=top_k, lowest=True)
    best_pairs   = top_k_pairs(final_scores, k=top_k, lowest=False)

    L1 = explicit_results[0]["additivity"].shape[0] if explicit_results else 0
    layers = np.arange(L1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, pairs, label, cmap_name in [
        (axes[0], worst_pairs, f"Bottom-{top_k}: strongest interaction circuits", "Reds"),
        (axes[1], best_pairs,  f"Top-{top_k}: most additive / independent pairs", "Greens"),
    ]:
        cmap = plt.colormaps[cmap_name]
        for idx, ((ast_n, b_n), _) in enumerate(pairs):
            subset = [r for r in explicit_results
                      if r["ast_node"] == ast_n and r["builtin_obj"] == b_n]
            if not subset:
                continue
            layer_means = np.mean([r["additivity"] for r in subset], axis=0)
            colour = cmap(0.35 + 0.55 * idx / max(len(pairs) - 1, 1))
            ax.plot(layers, layer_means, "-o", markersize=3, lw=1.5,
                    color=colour, label=f"{ast_n} x {b_n}")
        ax.set_xlabel("Layer")
        ax.set_ylim(-0.1, 1.1)
        ax.axhline(1.0, color="grey", lw=0.8, linestyle="--", alpha=0.5)
        ax.axhline(0.0, color="grey", lw=0.5, linestyle=":", alpha=0.4)
        ax.set_title(label, fontsize=9)
        ax.legend(fontsize=6, loc="lower left")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("Mean additivity score")
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    plt.close(fig)


def plot_baseline_cosine_matrices(
    ast_names:   list[str],
    ast_mat:     np.ndarray,
    b_names:     list[str],
    b_mat:       np.ndarray,
    cross_mat:   np.ndarray,
    layer:       int = -1,
    save_path:   Path | None = None,
) -> None:
    """
    Three panels:
      Left   — AST baseline cosine similarities  (are AST directions orthogonal?)
      Centre — Builtin baseline cosine similarities
      Right  — AST vs Builtin cross-similarity   (are the two factors aligned?)

    Ideal (independent representations):
      Left/Centre : near-diagonal (each class has its own direction)
      Right        : uniformly low (AST and builtin directions do not align)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, max(5, max(len(ast_names), len(b_names)) * 0.32)))

    panels = [
        (axes[0], ast_mat,   ast_names,  ast_names,
         f"AST baseline cosine sim (L={layer})"),
        (axes[1], b_mat,     b_names,    b_names,
         f"Builtin baseline cosine sim (L={layer})"),
        (axes[2], cross_mat, ast_names,  b_names,
         f"AST vs Builtin cross-sim (L={layer})"),
    ]

    ims = []
    for ax, mat, rnms, cnms, title in panels:
        im = _heatmap(ax, mat, rnms, cnms, title, vmin=-1.0, vmax=1.0, annotate=False)
        ims.append(im)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    fig.suptitle(
        "How orthogonal are AST / builtin directions in residual space?\n"
        "Blue=anti-correlated  |  White=orthogonal  |  Orange=aligned (entangled)",
        fontsize=10
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    plt.close(fig)


def plot_explicit_vs_proxy(
    ev_pairs:  list[dict],
    layer:     int = -1,
    title:     str = "Explicit vs Proxy additivity gap",
    save_path: Path | None = None,
) -> None:
    """
    Grouped bar chart: for each (AST, builtin) pair, shows explicit and proxy
    additivity scores side by side, sorted by gap size.

    Large gap  -> model relies on the builtin TOKEN (lexical encoding).
    Small gap  -> model encodes builtin SEMANTICS (generalises beyond the token).

    Also shows the gap distribution as a violin plot inset.
    """
    if not ev_pairs:
        print("  No explicit/proxy pairs to plot.")
        return

    pairs    = [f"{r['ast_node']}\nx\n{r['builtin_obj']}" for r in ev_pairs]
    ex_vals  = [r["explicit_mean"] for r in ev_pairs]
    pr_vals  = [r["proxy_mean"]    for r in ev_pairs]
    gaps     = [r["gap"]           for r in ev_pairs]

    x = np.arange(len(pairs))
    w = 0.35

    fig = plt.figure(figsize=(max(10, len(pairs) * 0.7 + 2), 6))
    gs  = fig.add_gridspec(1, 5)
    ax_bar  = fig.add_subplot(gs[0, :4])
    ax_viol = fig.add_subplot(gs[0, 4])

    ax_bar.bar(x - w/2, ex_vals, w, label="Explicit (builtin token present)",
               color="#1565C0", alpha=0.8)
    ax_bar.bar(x + w/2, pr_vals, w, label="Proxy (builtin token absent)",
               color="#E65100", alpha=0.8)

    # Gap arrows
    for xi, (ex, pr) in enumerate(zip(ex_vals, pr_vals)):
        if abs(ex - pr) > 0.02:
            ax_bar.annotate("", xy=(xi + w/2, pr), xytext=(xi + w/2, ex),
                            arrowprops=dict(arrowstyle="-|>", color="black",
                                           lw=0.8, mutation_scale=8))

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(pairs, fontsize=max(5, 8 - len(pairs) // 10))
    ax_bar.set_ylim(-0.05, 1.15)
    ax_bar.axhline(1.0, color="grey", lw=0.8, linestyle="--", alpha=0.5)
    ax_bar.set_ylabel("Additivity score")
    ax_bar.set_title(f"{title}  (layer {layer})", fontsize=10)
    ax_bar.legend(fontsize=8)

    # Violin of gap distribution
    parts = ax_viol.violinplot([gaps], positions=[0], showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("#9C27B0")
        pc.set_alpha(0.7)
    ax_viol.axhline(0, color="grey", lw=0.8, linestyle="--")
    ax_viol.set_ylabel("Gap (explicit - proxy)")
    ax_viol.set_xticks([0])
    ax_viol.set_xticklabels(["gap"], fontsize=8)
    ax_viol.set_title("Gap dist.", fontsize=9)

    # Significance: is mean gap significantly > 0?
    if len(gaps) >= 5:
        from scipy.stats import wilcoxon
        try:
            stat, p = wilcoxon(gaps, alternative="greater")
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
            ax_viol.text(0, max(gaps) * 1.05, f"p={p:.3f}\n{sig}",
                         ha="center", fontsize=8, color="black")
        except Exception:
            pass

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    plt.close(fig)


def plot_permutation_null(
    sig_results: dict,
    layer:       int = -1,
    save_path:   Path | None = None,
) -> None:
    """
    Histogram of null distribution at the given layer vs observed mean.
    Shows whether the observed additivity is above chance.
    """
    if not sig_results:
        return

    obs  = float(sig_results["observed_mean"][layer])
    null = sig_results["null_mean"][:, layer]
    p    = float(sig_results["p_values"][layer])
    sig  = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(null, bins=40, color="#9C27B0", alpha=0.7, density=True,
            label=f"Null (shuffled builtins, n={len(null)})")
    ax.axvline(obs, color="black", lw=2, label=f"Observed mean={obs:.3f}")
    ax.axvline(np.percentile(null, 95), color="red", lw=1.5, linestyle="--",
               label="95th pct null")
    ax.set_title(f"Additivity permutation test — layer {layer}  "
                 f"(p={p:.4f} {sig})", fontsize=10)
    ax.set_xlabel("Mean additivity score")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    plt.close(fig)


def plot_additivity_by_layer_overview(
    explicit_results: list[dict],
    sig_results:      dict,
    save_path:        Path | None = None,
) -> None:
    """
    Single-panel: mean additivity (with std band) across all layers,
    with permutation significance shaded and per-layer p-value markers.
    """
    if not explicit_results:
        return

    stack  = np.stack([r["additivity"] for r in explicit_results])  # (N, L+1)
    mean   = stack.mean(axis=0)
    std    = stack.std(axis=0)
    layers = np.arange(len(mean))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(layers, mean - std, mean + std, alpha=0.2, color="#1565C0")
    ax.plot(layers, mean, "o-", color="#1565C0", lw=2, label="Mean additivity ± std")

    if sig_results:
        null_95 = np.percentile(sig_results["null_mean"], 95, axis=0)
        ax.fill_between(layers,
                        np.percentile(sig_results["null_mean"], 5, axis=0),
                        null_95, alpha=0.15, color="#9C27B0",
                        label="Null 5-95th pct")
        ax.plot(layers, null_95, "--", color="#9C27B0", lw=1, alpha=0.6)

        # Significance markers
        for l in layers:
            p   = float(sig_results["p_values"][l])
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            if sig:
                ax.text(l, mean[l] + std[l] + 0.02, sig,
                        ha="center", va="bottom", fontsize=9, color="#1565C0")

    ax.axhline(1.0, color="grey", lw=0.8, linestyle="--", alpha=0.5)
    ax.axhline(0.0, color="grey", lw=0.5, linestyle=":", alpha=0.4)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Additivity score")
    ax.set_title("Mean additivity score across layers — "
                 "are AST+builtin representations additive?", fontsize=11)
    ax.set_xticks(layers)
    ax.set_ylim(-0.15, 1.2)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    plt.close(fig)


def plot_diagnostic_metrics(
    explicit_results: list[dict],
    layer:     int = -1,
    save_path: Path | None = None,
) -> None:
    """
    Four-panel diagnostic:
      (A) Distribution of additivity scores at the given layer
      (B) Distribution of AST baseline alignment
      (C) Distribution of builtin baseline alignment
      (D) Distribution of AST-builtin baseline orthogonality
          (how much do baseline directions correlate with each other?)

    These four together tell you:
      - Whether the combined prompt resembles the sum of its parts (A)
      - Whether it resembles just the AST baseline (B)
      - Whether it resembles just the builtin baseline (C)
      - Whether the two baseline directions are already correlated (D);
        if D is high, additivity scores may look good by accident
    """
    metrics = [
        ("additivity",       "A: Additivity score",              "#1565C0"),
        ("ast_alignment",    "B: AST baseline alignment",        "#2196F3"),
        ("builtin_alignment","C: Builtin baseline alignment",     "#E65100"),
        ("ast_builtin_ortho","D: AST vs Builtin baseline cosine", "#9C27B0"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
    for ax, (mkey, mlabel, col) in zip(axes, metrics):
        vals = [float(r[mkey][layer]) for r in explicit_results]
        ax.hist(vals, bins=30, color=col, alpha=0.75, density=True)
        ax.axvline(np.mean(vals), color="black", lw=1.5,
                   label=f"mean={np.mean(vals):.3f}")
        ax.axvline(np.median(vals), color="grey", lw=1.2, linestyle="--",
                   label=f"median={np.median(vals):.3f}")
        ax.set_xlabel("Cosine similarity / score")
        ax.set_title(mlabel, fontsize=9)
        ax.legend(fontsize=7)

    fig.suptitle(f"Diagnostic distributions — layer {layer}", fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    plt.close(fig)


def print_summary(
    explicit_results: list[dict],
    ev_pairs:         list[dict],
    sig_results:      dict,
    layer:            int = -1,
) -> None:
    """Print a text summary of the additivity and explicit-vs-proxy results."""
    scores = [float(r["additivity"][layer]) for r in explicit_results]
    if not scores:
        print("No results.")
        return

    print(f"\n{'='*65}")
    print(f"ADDITIVITY TEST SUMMARY  (layer {layer})")
    print(f"{'='*65}")
    print(f"  Stubs scored  : {len(scores)}")
    print(f"  Mean score    : {np.mean(scores):.4f}")
    print(f"  Std           : {np.std(scores):.4f}")
    print(f"  Median        : {np.median(scores):.4f}")
    print(f"  >= 0.9 (additive)  : {sum(s>=0.9 for s in scores)} "
          f"({100*sum(s>=0.9 for s in scores)/len(scores):.1f}%)")
    print(f"  < 0.5 (interaction): {sum(s<0.5  for s in scores)} "
          f"({100*sum(s<0.5  for s in scores)/len(scores):.1f}%)")

    if sig_results:
        p = float(sig_results["p_values"][layer])
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        print(f"\n  Permutation p-value: {p:.4f}  {sig}")
        if p < 0.05:
            print("  -> Additivity is SIGNIFICANTLY above chance.")
        else:
            print("  -> Additivity is NOT significantly above chance.")

    pair_scores = aggregate_by_pair(explicit_results, "additivity", layer)
    print(f"\n  Top-10 most ADDITIVE pairs (score -> 1.0 = independent):")
    for (a, b), s in top_k_pairs(pair_scores, 10, lowest=False):
        print(f"    {s:.4f}  {a:<20s} x {b}")
    print(f"\n  Top-10 strongest INTERACTIONS (score << 1.0 = joint circuit):")
    for (a, b), s in top_k_pairs(pair_scores, 10, lowest=True):
        print(f"    {s:.4f}  {a:<20s} x {b}")

    if ev_pairs:
        gaps = [r["gap"] for r in ev_pairs]
        print(f"\n{'='*65}")
        print(f"EXPLICIT vs PROXY SUMMARY  (layer {layer})")
        print(f"{'='*65}")
        print(f"  Pairs compared: {len(ev_pairs)}")
        print(f"  Mean gap (explicit - proxy): {np.mean(gaps):.4f}")
        print(f"  Std gap                     : {np.std(gaps):.4f}")
        print(f"  Pairs with gap > 0.1        : {sum(g>0.1 for g in gaps)} "
              f"(lexical encoding dominates)")
        print(f"  Pairs with |gap| < 0.05     : {sum(abs(g)<0.05 for g in gaps)} "
              f"(semantic encoding)")
        print(f"\n  Top-10 LEXICAL pairs (largest gap -> token-dependent):")
        for r in ev_pairs[:10]:
            print(f"    gap={r['gap']:+.4f}  ex={r['explicit_mean']:.4f}  "
                  f"pr={r['proxy_mean']:.4f}  "
                  f"{r['ast_node']:<18s} x {r['builtin_obj']}")
        print(f"\n  Top-10 SEMANTIC pairs (smallest gap -> token-independent):")
        for r in ev_pairs[-10:]:
            print(f"    gap={r['gap']:+.4f}  ex={r['explicit_mean']:.4f}  "
                  f"pr={r['proxy_mean']:.4f}  "
                  f"{r['ast_node']:<18s} x {r['builtin_obj']}")

    print(f"{'='*65}\n")


# ─────────────────────────────────────────────────────────────────────────────
# OPTION A — Purity-mask projected additivity  (requires step 02 outputs)
# ─────────────────────────────────────────────────────────────────────────────

def load_purity_masks(base: Path, stem: str) -> dict | None:
    """
    Load purity masks produced by 02_variance_partition.py.
    Returns dict with keys "ast_pure" and "builtin_pure", each shape (L+1, H),
    or None if the file does not exist.
    """
    path = base / f"02_{stem}_purity_masks.npz"
    if not path.exists():
        return None
    npz = np.load(path, allow_pickle=True)
    # keys are prefixed "residual_" in the npz
    result = {}
    for key in ("ast_pure", "builtin_pure"):
        full_key = f"residual_{key}"
        if full_key in npz.files:
            result[key] = npz[full_key].astype(bool)   # (L+1, H)
    if len(result) < 2:
        return None
    print(f"  Loaded purity masks from {path.name}  "
          f"(ast_pure dims per layer: "
          f"{result['ast_pure'].sum(axis=-1).tolist()[:5]}...)")
    return result


def _apply_mask_to_residuals(
    resid_all:  np.ndarray,    # (N, L+1, H)
    union_mask: np.ndarray,    # (L+1, H) bool
) -> np.ndarray:
    """
    Zero out dimensions NOT in union_mask at each layer.
    Returns (N, L+1, H) with contaminated dims set to 0.
    Operates in-place on a copy to avoid modifying the original.
    """
    masked = resid_all.copy()
    for l in range(resid_all.shape[1]):
        masked[:, l, ~union_mask[l]] = 0.0
    return masked


def run_additivity_masked(
    resid_all:   np.ndarray,           # (N, L+1, H)  raw residuals
    meta:        list[dict],
    mu_ast:      dict[str, np.ndarray],
    mu_builtin:  dict[str, np.ndarray],
    purity_masks: dict,                # from load_purity_masks()
    variant_types: list[str] = ("explicit", "proxy"),
) -> tuple[list[dict], np.ndarray]:
    """
    Option A: run the additivity test restricted to purity-masked dimensions.

    Strategy:
      union_mask[l] = ast_pure_mask[l] | builtin_pure_mask[l]

    All activations and baseline mean vectors are masked before computing
    cos_sim.  This removes contaminated (shared / noise) dimensions, so the
    additivity score reflects only the clean portion of each representation.

    Returns (results, union_mask) where results has the same structure as
    run_additivity() but with scores computed in the masked space.
    """
    ast_mask    = purity_masks["ast_pure"]      # (L+1, H)
    b_mask      = purity_masks["builtin_pure"]  # (L+1, H)
    union_mask  = ast_mask | b_mask             # (L+1, H)

    n_clean = union_mask.sum(axis=-1)
    print(f"  Clean dims per layer (union mask): {n_clean.tolist()[:8]}...")

    # Mask baseline mean vectors
    mu_ast_m: dict[str, np.ndarray] = {}
    for node, v in mu_ast.items():
        vm = v.copy()                   # (L+1, H)
        for l in range(v.shape[0]):
            vm[l, ~union_mask[l]] = 0.0
        mu_ast_m[node] = vm

    mu_b_m: dict[str, np.ndarray] = {}
    for b, v in mu_builtin.items():
        vm = v.copy()
        for l in range(v.shape[0]):
            vm[l, ~union_mask[l]] = 0.0
        mu_b_m[b] = vm

    # Mask prompt activations
    resid_masked = _apply_mask_to_residuals(resid_all, union_mask)

    # Reuse the same scoring logic with masked data
    results = run_additivity(resid_masked, meta, mu_ast_m, mu_b_m, variant_types)
    return results, union_mask


# ─────────────────────────────────────────────────────────────────────────────
# OPTION B — Regression-direction removal  (aside, no extra inputs needed)
# ─────────────────────────────────────────────────────────────────────────────

def _project_out(v: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """
    Remove from vector v the component along `direction`.
    direction must be a unit vector (or will be normalised).
    Works on arrays: v shape (..., H), direction shape (H,).
    """
    d = direction / (np.linalg.norm(direction) + 1e-12)
    coeff = (v * d).sum(axis=-1, keepdims=True)     # (..., 1)
    return v - coeff * d


def correct_baselines_regression(
    mu_ast:     dict[str, np.ndarray],   # node -> (L+1, H)
    mu_builtin: dict[str, np.ndarray],   # builtin -> (L+1, H)
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Option B (aside): remove each factor's global mean direction from the
    other factor's baseline vectors, layer by layer.

    For layer l:
      mean_ast_dir[l]  = mean over all mu_ast[A][l],  then normalised
      mean_b_dir[l]    = mean over all mu_builtin[B][l], then normalised

      mu_ast_corrected[A][l]  = mu_ast[A][l]  - proj(mu_ast[A][l], mean_b_dir[l])
      mu_b_corrected[B][l]    = mu_builtin[B][l] - proj(mu_builtin[B][l], mean_ast_dir[l])

    This strips out the "average builtin flavour" from AST baselines and vice
    versa.  It is a first-order approximation — it only removes the shared
    *mean* direction, not individual cross-contamination per pair.

    Returns (mu_ast_corrected, mu_b_corrected).
    """
    L1 = next(iter(mu_ast.values())).shape[0]

    # Compute global mean directions per layer
    ast_stack = np.stack(list(mu_ast.values()))      # (n_ast, L+1, H)
    b_stack   = np.stack(list(mu_builtin.values()))  # (n_b,   L+1, H)

    mean_ast_dir = ast_stack.mean(axis=0)   # (L+1, H)
    mean_b_dir   = b_stack.mean(axis=0)     # (L+1, H)

    # Correct AST baselines: project out mean builtin direction
    mu_ast_c: dict[str, np.ndarray] = {}
    for node, v in mu_ast.items():
        vc = v.copy()
        for l in range(L1):
            vc[l] = _project_out(vc[l], mean_b_dir[l])
        mu_ast_c[node] = vc

    # Correct builtin baselines: project out mean AST direction
    mu_b_c: dict[str, np.ndarray] = {}
    for b, v in mu_builtin.items():
        vc = v.copy()
        for l in range(L1):
            vc[l] = _project_out(vc[l], mean_ast_dir[l])
        mu_b_c[b] = vc

    # Report how much was removed (average angle change)
    ast_keys = list(mu_ast.keys())
    removed_ast = np.mean([
        np.linalg.norm(mu_ast[k][-1] - mu_ast_c[k][-1]) /
        (np.linalg.norm(mu_ast[k][-1]) + 1e-12)
        for k in ast_keys
    ])
    b_keys = list(mu_builtin.keys())
    removed_b = np.mean([
        np.linalg.norm(mu_builtin[k][-1] - mu_b_c[k][-1]) /
        (np.linalg.norm(mu_builtin[k][-1]) + 1e-12)
        for k in b_keys
    ])
    print(f"  [Option B] Mean fraction removed from AST baselines   : {removed_ast:.3f}")
    print(f"  [Option B] Mean fraction removed from builtin baselines: {removed_b:.3f}")

    return mu_ast_c, mu_b_c


# ─────────────────────────────────────────────────────────────────────────────
# Three-way comparison plots and summary
# ─────────────────────────────────────────────────────────────────────────────

def plot_additivity_three_way(
    raw_results:       list[dict],
    masked_results:    list[dict] | None,
    corrected_results: list[dict] | None,
    sig_raw:           dict,
    sig_masked:        dict,
    title:             str = "Additivity: raw vs decontaminated",
    save_path:         Path | None = None,
) -> None:
    """
    Three-panel comparison of mean additivity across layers:
      Panel 1  Raw additivity (may include crosstalk inflation)
      Panel 2  Option A — purity-masked  (clean dims only)
      Panel 3  Option B — regression-corrected baselines  (aside)

    Also overlays all three as thin lines on a single axis for direct
    visual comparison.  Convergence = contamination was not inflating scores.
    Divergence = raw scores were misleadingly high.

    Null bands are drawn for raw and masked if permutation results are available.
    """
    all_series = [
        (raw_results,       sig_raw,    "#1565C0", "Raw"),
        (masked_results,    sig_masked, "#2E7D32", "Masked (Opt A)"),
        (corrected_results, {},         "#E65100", "Regression-corr. (Opt B, aside)"),
    ]

    # Filter out None
    series = [(r, s, c, l) for r, s, c, l in all_series if r is not None and len(r) > 0]

    n_panels = len(series) + 1          # individual panels + overlay
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5), sharey=True)

    # Overlay panel (last)
    ax_ov = axes[-1]
    ax_ov.set_title("Overlay", fontsize=10)
    ax_ov.axhline(1.0, color="grey", lw=0.8, linestyle="--", alpha=0.4)
    ax_ov.axhline(0.0, color="grey", lw=0.5, linestyle=":", alpha=0.3)

    for panel_i, (results, sig, colour, label) in enumerate(series):
        ax = axes[panel_i]
        stack  = np.stack([r["additivity"] for r in results])
        mean   = stack.mean(axis=0)
        std    = stack.std(axis=0)
        layers = np.arange(len(mean))

        ax.fill_between(layers, mean - std, mean + std, alpha=0.18, color=colour)
        ax.plot(layers, mean, "o-", color=colour, lw=2, label=label)

        if sig:
            null_5  = np.percentile(sig["null_mean"],  5, axis=0)
            null_95 = np.percentile(sig["null_mean"], 95, axis=0)
            ax.fill_between(layers, null_5, null_95, alpha=0.12, color="#9C27B0",
                            label="Null 5-95th pct")
            for l in layers:
                p   = float(sig["p_values"][l])
                mk  = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
                if mk:
                    ax.text(l, mean[l] + std[l] + 0.02, mk,
                            ha="center", fontsize=8, color=colour)

        ax.axhline(1.0, color="grey", lw=0.8, linestyle="--", alpha=0.4)
        ax.axhline(0.0, color="grey", lw=0.5, linestyle=":", alpha=0.3)
        ax.set_xlabel("Layer")
        ax.set_ylim(-0.15, 1.25)
        ax.set_title(label, fontsize=10)
        ax.set_xticks(layers)
        ax.grid(True, alpha=0.2)
        if panel_i == 0:
            ax.set_ylabel("Mean additivity score")
        ax.legend(fontsize=7)

        # Overlay
        ax_ov.plot(layers, mean, "o-", color=colour, lw=2, label=label, alpha=0.8)
        ax_ov.fill_between(layers, mean - std, mean + std, alpha=0.1, color=colour)

    ax_ov.set_xlabel("Layer")
    ax_ov.set_xticks(np.arange(len(np.stack([r["additivity"] for r in series[0][0]]).shape[-1]
                                   if False else mean)))
    ax_ov.legend(fontsize=8)
    ax_ov.grid(True, alpha=0.2)
    ax_ov.set_ylim(-0.15, 1.25)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    plt.close(fig)


def plot_additivity_heatmap_comparison(
    raw_results:       list[dict],
    masked_results:    list[dict] | None,
    corrected_results: list[dict] | None,
    layer:             int = -1,
    save_path:         Path | None = None,
) -> None:
    """
    Side-by-side heatmaps of (AST x builtin) additivity score for all three
    methods at the chosen layer.

    Cells that are green in Raw but yellow/red in Masked reveal pairs where
    the raw additivity was contaminated by crosstalk — their circuits are NOT
    truly additive, the raw score was inflated by shared baseline directions.
    """
    series = [
        (raw_results,       "Raw"),
        (masked_results,    "Masked (Opt A)"),
        (corrected_results, "Regression-corr. (Opt B, aside)"),
    ]
    series = [(r, l) for r, l in series if r is not None and len(r) > 0]

    if not series:
        return

    # Collect all (AST, builtin) pairs from raw
    all_ast      = sorted({r["ast_node"]   for r in raw_results})
    all_builtins = sorted({r["builtin_obj"] for r in raw_results})

    def _grid(results):
        ps = aggregate_by_pair(results, "additivity", layer)
        grid = np.full((len(all_ast), len(all_builtins)), np.nan)
        for i, a in enumerate(all_ast):
            for j, b in enumerate(all_builtins):
                if (a, b) in ps:
                    grid[i, j] = ps[(a, b)]
        return grid

    fig, axes = plt.subplots(1, len(series),
                             figsize=(max(8, len(all_builtins) * 0.45) * len(series),
                                      max(6, len(all_ast) * 0.38)),
                             sharey=True)
    if len(series) == 1:
        axes = [axes]

    cmap = plt.colormaps["RdYlGn"].copy()
    cmap.set_bad(color="#e0e0e0")

    for ax, (results, label) in zip(axes, series):
        grid = _grid(results)
        im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(all_builtins)))
        ax.set_xticklabels(all_builtins, rotation=45, ha="right",
                           fontsize=max(4, 8 - len(all_builtins) // 10))
        ax.set_yticks(range(len(all_ast)))
        ax.set_yticklabels(all_ast, fontsize=max(4, 8 - len(all_ast) // 10))
        ax.set_title(label, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02,
                     label="Additivity score")

    fig.suptitle(
        f"Additivity heatmap comparison (layer {layer})\n"
        "Cells green in Raw but red in Masked = contamination was inflating the score",
        fontsize=11
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    plt.close(fig)


def print_decontamination_summary(
    raw_results:       list[dict],
    masked_results:    list[dict] | None,
    corrected_results: list[dict] | None,
    layer:             int = -1,
) -> None:
    """
    Compare mean additivity across all three methods at the given layer
    and draw a conclusion about whether raw scores are trustworthy.
    """
    def _mean(results):
        if not results:
            return None
        return float(np.mean([r["additivity"][layer] for r in results]))

    raw_m  = _mean(raw_results)
    mask_m = _mean(masked_results)
    corr_m = _mean(corrected_results)

    print(f"\n{'='*65}")
    print(f"DECONTAMINATION SUMMARY  (layer {layer})")
    print(f"{'='*65}")
    print(f"  Raw additivity mean              : {raw_m:.4f}" if raw_m  is not None else "  Raw: N/A")
    print(f"  Masked (Opt A) additivity mean   : {mask_m:.4f}" if mask_m is not None else "  Masked: N/A (step 02 not run)")
    print(f"  Regr.-corr. (Opt B) additivity   : {corr_m:.4f}" if corr_m is not None else "  Corrected: N/A")

    if raw_m is not None and mask_m is not None:
        drop = raw_m - mask_m
        print(f"\n  Drop (Raw - Masked)              : {drop:+.4f}")
        if drop > 0.10:
            print("  INTERPRETATION: Large drop — raw scores were SUBSTANTIALLY inflated")
            print("  by crosstalk between AST and builtin baselines.  Trust masked scores.")
        elif drop > 0.03:
            print("  INTERPRETATION: Moderate drop — some contamination present.")
            print("  Masked scores are more reliable; raw is a mild over-estimate.")
        else:
            print("  INTERPRETATION: Negligible drop — raw scores are NOT substantially")
            print("  contaminated.  AST and builtin baselines are largely orthogonal.")

    if raw_m is not None and corr_m is not None:
        drop_b = raw_m - corr_m
        print(f"\n  Drop (Raw - Regression-corr.)    : {drop_b:+.4f}  (Option B, aside)")
        if abs(drop_b - (raw_m - mask_m if mask_m else drop_b)) < 0.05:
            print("  Option A and B agree — result is robust.")
        else:
            print("  Option A and B disagree — use masked (Opt A) as primary.")

    print(f"{'='*65}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 4: Additivity test + Explicit vs Proxy analysis"
    )
    parser.add_argument("--stem",    default="contrastive_stubs",
                        help="File stem used in step 01 (default: contrastive_stubs)")
    parser.add_argument("--in_dir", "-d", default="data",
                        help="Directory containing 01_ output files (default: data/)")
    parser.add_argument("--out_dir", "-o", default=None,
                        help="Output directory for results (default: same as --in_dir)")
    parser.add_argument("--layer",   type=int, default=-1,
                        help="Layer index for detailed analysis (-1 = final layer)")
    parser.add_argument("--n_perm",  type=int, default=1000,
                        help="Number of permutations for significance test (default 1000)")
    parser.add_argument("--no_perm",  action="store_true",
                        help="Skip permutation test (fast mode)")
    parser.add_argument("--no_proj", action="store_true",
                        help="Skip Option A (purity-mask projection) even if "
                             "02_ files are present")
    parser.add_argument("--no_regr", action="store_true",
                        help="Skip Option B (regression-direction correction)")
    parser.add_argument("--plot",    action="store_true",
                        help="Generate and save all plots")
    args = parser.parse_args()

    base    = Path(__file__).parent / args.in_dir
    out     = Path(__file__).parent / (args.out_dir if args.out_dir else args.in_dir)
    out.mkdir(exist_ok=True)
    img_dir = out / "images"
    img_dir.mkdir(exist_ok=True)
    stem = args.stem

    # ── Load inputs ───────────────────────────────────────────────────────────
    print(f"\n[04] Loading inputs for stem='{stem}'")

    resid_path = base / f"01_{stem}_residual_all.npy"
    meta_path  = base / f"01_{stem}_meta.json"
    for p in [resid_path, meta_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}\nRun 01_extraction.py first.")

    resid_all = np.load(resid_path)           # (N, L+1, H)
    N, L1, H  = resid_all.shape
    print(f"  residual_all: {resid_all.shape}")

    with open(meta_path) as f:
        meta = json.load(f)

    variant_counts: dict[str, int] = defaultdict(int)
    for m in meta:
        variant_counts[m.get("variant_type", "explicit")] += 1
    print(f"  Variant type counts: {dict(variant_counts)}")

    layer_idx = L1 + args.layer if args.layer < 0 else args.layer
    layer_idx = int(np.clip(layer_idx, 0, L1 - 1))
    print(f"  Analysis layer: {layer_idx}")

    # ── Baselines ─────────────────────────────────────────────────────────────
    print("\n[04] Building baseline mean vectors...")
    mu_ast, mu_builtin, idx_map = build_baselines(resid_all, meta)

    if not mu_ast or not mu_builtin:
        print("ERROR: No baseline stubs found.  "
              "Ensure contrastive_stubs include 'baseline_ast' and "
              "'baseline_builtin' variant_type entries.")
        return

    # Save baselines
    np.savez_compressed(
        out / f"04_{stem}_baselines_ast.npz",
        **{k.replace(" ", "_"): v for k, v in mu_ast.items()}
    )
    np.savez_compressed(
        out / f"04_{stem}_baselines_builtin.npz",
        **{k.replace(" ", "_"): v for k, v in mu_builtin.items()}
    )
    print(f"  Saved baselines.")

    # ── Additivity test ───────────────────────────────────────────────────────
    print("\n[04] Running additivity test on explicit stubs...")
    explicit_results = run_additivity(
        resid_all, meta, mu_ast, mu_builtin, variant_types=["explicit"]
    )

    print("\n[04] Running additivity test on proxy stubs...")
    proxy_results = run_additivity(
        resid_all, meta, mu_ast, mu_builtin, variant_types=["proxy"]
    )

    # Save scored arrays
    if explicit_results:
        np.savez_compressed(
            out / f"04_{stem}_additivity_scores.npz",
            additivity        = np.stack([r["additivity"]        for r in explicit_results]),
            ast_alignment     = np.stack([r["ast_alignment"]     for r in explicit_results]),
            builtin_alignment = np.stack([r["builtin_alignment"] for r in explicit_results]),
            interaction_norm  = np.stack([r["interaction_norm"]  for r in explicit_results]),
            ast_builtin_ortho = np.stack([r["ast_builtin_ortho"] for r in explicit_results]),
        )

    # Pair summary JSON
    pair_scores = aggregate_by_pair(explicit_results, "additivity", layer_idx)
    pair_summary = [
        {"ast_node": k[0], "builtin_obj": k[1], "mean_additivity": v}
        for k, v in sorted(pair_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    with open(out / f"04_{stem}_pair_summary.json", "w") as f:
        json.dump(pair_summary, f, indent=2)
    print(f"  Saved 04_{stem}_pair_summary.json ({len(pair_summary)} pairs)")

    # ── Explicit vs Proxy ─────────────────────────────────────────────────────
    print("\n[04] Computing explicit vs proxy gap...")
    ev_pairs = explicit_vs_proxy(explicit_results, proxy_results, layer=layer_idx)
    with open(out / f"04_{stem}_explicit_vs_proxy.json", "w") as f:
        json.dump(ev_pairs, f, indent=2)
    print(f"  Saved 04_{stem}_explicit_vs_proxy.json ({len(ev_pairs)} pairs)")

    # ── Baseline cosine matrices ──────────────────────────────────────────────
    print("\n[04] Computing baseline cosine similarity matrices...")
    ast_names, ast_mat, b_names, b_mat, cross_mat = baseline_cosine_matrices(
        mu_ast, mu_builtin, layer=layer_idx
    )

    # ── Option A: purity-masked additivity ───────────────────────────────────
    masked_explicit = None
    sig_masked      = {}

    if not args.no_proj:
        print("\n[04] Option A: loading purity masks from step 02...")
        purity_masks = load_purity_masks(base, stem)
        if purity_masks is None:
            print("  Purity masks not found (02_ outputs missing) — skipping Option A.")
            print("  Run 02_variance_partition.py first, or pass --no_proj to suppress.")
        else:
            print("\n[04] Option A: running masked additivity test...")
            masked_explicit, union_mask = run_additivity_masked(
                resid_all, meta, mu_ast, mu_builtin, purity_masks,
                variant_types=["explicit"]
            )
            if masked_explicit:
                np.savez_compressed(
                    out / f"04_{stem}_additivity_scores_masked.npz",
                    additivity        = np.stack([r["additivity"]        for r in masked_explicit]),
                    ast_alignment     = np.stack([r["ast_alignment"]     for r in masked_explicit]),
                    builtin_alignment = np.stack([r["builtin_alignment"] for r in masked_explicit]),
                    interaction_norm  = np.stack([r["interaction_norm"]  for r in masked_explicit]),
                )
                print(f"  Saved 04_{stem}_additivity_scores_masked.npz")

            if not args.no_perm and masked_explicit:
                print(f"\n[04] Option A: running permutation test (n_perm={args.n_perm})...")
                # Re-mask residuals for the permutation test too
                resid_masked = _apply_mask_to_residuals(resid_all, union_mask)
                mu_ast_m  = {k: v * union_mask[:, None].T if False else  # done inside run_additivity_masked
                             np.where(union_mask[:, :, None].transpose(0,2,1)
                                      if False else True, v, v)
                             for k, v in mu_ast.items()}
                # Simpler: re-derive masked baselines inline for perm test
                _m_exp, _umask = run_additivity_masked(
                    resid_all, meta, mu_ast, mu_builtin, purity_masks,
                    variant_types=["explicit"]
                )
                # Extract masked mu_ast/mu_builtin by building from masked resid
                resid_m_all = _apply_mask_to_residuals(resid_all, union_mask)
                mu_ast_m2, mu_b_m2, _ = build_baselines(resid_m_all, meta)
                sig_masked = permutation_significance(
                    masked_explicit, mu_ast_m2, mu_b_m2, resid_m_all, meta,
                    n_perm=args.n_perm
                )
                if sig_masked:
                    with open(out / f"04_{stem}_significance_masked.json", "w") as f:
                        json.dump({
                            "p_values":      sig_masked["p_values"].tolist(),
                            "significant":   sig_masked["significant"].tolist(),
                            "observed_mean": sig_masked["observed_mean"].tolist(),
                        }, f, indent=2)
                    print(f"  Saved 04_{stem}_significance_masked.json")
    else:
        print("\n[04] Skipping Option A (--no_proj set).")

    # ── Option B: regression-corrected additivity (aside) ────────────────────
    corrected_explicit = None

    if not args.no_regr:
        print("\n[04] Option B (aside): regression-direction correction of baselines...")
        mu_ast_c, mu_b_c = correct_baselines_regression(mu_ast, mu_builtin)
        corrected_explicit = run_additivity(
            resid_all, meta, mu_ast_c, mu_b_c, variant_types=["explicit"]
        )
        if corrected_explicit:
            np.savez_compressed(
                out / f"04_{stem}_additivity_scores_corrected.npz",
                additivity        = np.stack([r["additivity"]        for r in corrected_explicit]),
                ast_alignment     = np.stack([r["ast_alignment"]     for r in corrected_explicit]),
                builtin_alignment = np.stack([r["builtin_alignment"] for r in corrected_explicit]),
                interaction_norm  = np.stack([r["interaction_norm"]  for r in corrected_explicit]),
            )
            print(f"  Saved 04_{stem}_additivity_scores_corrected.npz")
    else:
        print("\n[04] Skipping Option B (--no_regr set).")

    # ── Decontamination summary ───────────────────────────────────────────────
    print_decontamination_summary(
        explicit_results, masked_explicit, corrected_explicit, layer=layer_idx
    )

    # ── Permutation significance ──────────────────────────────────────────────
    sig_results = {}
    if not args.no_perm and explicit_results:
        print(f"\n[04] Running permutation significance test (n_perm={args.n_perm})...")
        sig_results = permutation_significance(
            explicit_results, mu_ast, mu_builtin, resid_all, meta,
            n_perm=args.n_perm
        )
        p_vals_out = {
            "p_values":   sig_results["p_values"].tolist(),
            "significant": sig_results["significant"].tolist(),
            "observed_mean": sig_results["observed_mean"].tolist(),
        }
        with open(out / f"04_{stem}_significance.json", "w") as f:
            json.dump(p_vals_out, f, indent=2)
        print(f"  Saved 04_{stem}_significance.json")

    # ── Text summary ──────────────────────────────────────────────────────────
    print_summary(explicit_results, ev_pairs, sig_results, layer=layer_idx)

    # ── Plots ─────────────────────────────────────────────────────────────────
    if args.plot:
        print(f"\n[04] Generating plots...")

        # 1. Additivity heatmap
        plot_additivity_heatmap(
            explicit_results, layer=layer_idx,
            title=f"Additivity score — explicit stubs (layer {layer_idx})",
            save_path=img_dir / f"04_{stem}_additivity_heatmap.png",
        )

        # 2. Layer traces (top-k most/least additive)
        plot_layer_traces(
            explicit_results, top_k=10,
            title="Additivity score across layers",
            save_path=img_dir / f"04_{stem}_layer_traces.png",
        )

        # 3. Overview: mean additivity across all layers with null band
        plot_additivity_by_layer_overview(
            explicit_results, sig_results,
            save_path=img_dir / f"04_{stem}_additivity_overview.png",
        )

        # 4. Baseline cosine similarity matrices (3-panel)
        plot_baseline_cosine_matrices(
            ast_names, ast_mat, b_names, b_mat, cross_mat,
            layer=layer_idx,
            save_path=img_dir / f"04_{stem}_baseline_cosine_matrices.png",
        )

        # 5. Explicit vs proxy bar chart
        plot_explicit_vs_proxy(
            ev_pairs, layer=layer_idx,
            save_path=img_dir / f"04_{stem}_explicit_vs_proxy.png",
        )

        # 6. Permutation null histogram
        if sig_results:
            plot_permutation_null(
                sig_results, layer=layer_idx,
                save_path=img_dir / f"04_{stem}_permutation_null.png",
            )

        # 7. Diagnostic distributions (raw)
        if explicit_results:
            plot_diagnostic_metrics(
                explicit_results, layer=layer_idx,
                save_path=img_dir / f"04_{stem}_diagnostic_distributions.png",
            )

        # 8. Three-way comparison: raw vs masked vs corrected
        plot_additivity_three_way(
            explicit_results,
            masked_explicit,
            corrected_explicit,
            sig_results,
            sig_masked,
            title="Additivity score: raw vs decontaminated (convergence = robust result)",
            save_path=img_dir / f"04_{stem}_additivity_three_way.png",
        )

        # 9. Three-way heatmap comparison at chosen layer
        plot_additivity_heatmap_comparison(
            explicit_results,
            masked_explicit,
            corrected_explicit,
            layer=layer_idx,
            save_path=img_dir / f"04_{stem}_heatmap_three_way.png",
        )

        # 10. Diagnostic distributions for masked (Option A) if available
        if masked_explicit:
            plot_diagnostic_metrics(
                masked_explicit, layer=layer_idx,
                save_path=img_dir / f"04_{stem}_diagnostic_distributions_masked.png",
            )

        print(f"\n  All plots saved to {out}/")

    print("\n[04] Done.")


if __name__ == "__main__":
    main()
