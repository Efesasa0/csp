"""
04_raw_similarity.py

Step 4 (raw counterpart) of the AST x builtin mechanistic interpretability pipeline.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY WE DO THIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

04_additivity.py tests whether combined activations equal the sum of AST and
builtin baseline vectors.  That test has two structural limitations:

  1. CONTAMINATION OF BASELINES — even "pure" baseline stubs fire some signal
     from the other factor (an AST baseline stub still processes its builtin
     tokens).  If the two baselines point in geometrically similar directions,
     cos_sim(act, mu_ast + mu_builtin) is inflated by the baselines' mutual
     alignment rather than reflecting genuine additivity of circuits.

  2. PURITY-MASKED ADDITIVITY DISCARDS SHARED SPACE — Option A in step 04
     restricts the additivity computation to dims flagged ast_pure or
     builtin_pure.  This removes the shared/interaction dims by construction,
     so it cannot measure what happens in the interaction circuit.

This script addresses both gaps with three analyses:

  ANALYSIS 1 — BASELINE DIRECTION SIMILARITY
  -------------------------------------------
  For each (AST node A, builtin B, layer l), compute:
      cos_sim(μ_ast[A][l], μ_builtin[B][l])

  If near 0: the feature directions are orthogonal — AST and builtin baselines
             are structurally independent.  Additivity scores from step 04
             are not inflated by baseline alignment.
  If large:  the two baseline directions are aligned — a structural source of
             additivity score inflation.  Step 04 scores cannot distinguish
             genuine additivity from accidental direction similarity.

  Also computes per-layer mean/std/min/max and mean orthogonality = 1 - |cos_sim|.

  ANALYSIS 2 — SHARED-SPACE ADDITIVITY
  -------------------------------------
  Re-run the additivity formula from step 04 (cos_sim(act, μ_ast + μ_builtin))
  but restricted to SHARED subspace dims only — the interaction circuit itself.

      shared_additivity(t, l) = cos_sim(
          act_t[l, shared_dims],
          (μ_ast[A] + μ_builtin[B])[l, shared_dims]
      )

  If shared dims are a genuine interaction circuit (encoding something beyond
  the sum of baselines), these scores should be LOWER than pure-dim scores.
  If shared dims merely amplify whichever factor dominates, scores may be high
  for only one factor.

  Contrast with the masked (pure-only) additivity from step 04 — the pair
  together reveals the complementary circuit structure.

  ANALYSIS 3 — INTERACTION NORM vs SHARED VP
  -------------------------------------------
  Step 04 computes interaction_norm[l] = ||act - (μ_ast + μ_builtin)||.
  Step 02 computes a shared VP fraction per dim (how much variance in that
  dim is explained by the interaction/joint term).

  For each explicit stub, this analysis correlates the interaction_norm at
  each layer against the prompt's shared VP fraction.  The hypothesis:

      prompts whose activations show more shared variance should also show
      larger interaction norms — they are "living in" the interaction circuit.

  Spearman correlation per layer tells us whether the shared VP fraction is a
  useful predictor of interaction strength for individual prompts.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW IT IS COMPUTED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Baseline similarity:
  ast_names   = sorted keys of μ_ast
  builtin_names = sorted keys of μ_builtin
  sim[i, j, l] = cos_sim(μ_ast[ast_names[i]][l], μ_builtin[builtin_names[j]][l])
  shape (n_ast, n_builtin, L+1)

Shared-space additivity:
  For each explicit stub t with (A, B):
    m = shared_combined_mask[l]   (H,) bool from step 02
    a_sh = act_t[l][m]
    p_sh = (μ_ast[A][l] + μ_builtin[B][l])[m]
    score[t, l] = cos_sim(a_sh, p_sh)

Interaction vs shared VP:
  Load 04_<stem>_additivity_scores.npz  -> interaction_norm (N_explicit, L+1)
  Load 02_<stem>_vp_residual.npz        -> shared (L+1, H)
  For each explicit stub t:
    shared_vp_frac[t, l] = mean(vp_shared[l, :])   <- prompt-level scalar
    (Note: VP is computed across prompts, not per-prompt; we use the per-layer
     mean shared VP as the "shared environment" at that layer and correlate
     with per-prompt interaction norms.)
  Spearman rho per layer.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUT FILES  (from steps 01, 02, 04)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  01_<stem>_residual_all.npy            float32  (N, L+1, H)
  01_<stem>_meta.json                   per-prompt metadata

  04_<stem>_baselines_ast.npz           μ_ast per AST node   (n_ast, L+1, H)
  04_<stem>_baselines_builtin.npz       μ_builtin per builtin (n_b, L+1, H)
  04_<stem>_additivity_scores.npz       interaction_norm arrays (optional)

  02_<stem>_purity_masks.npz            boolean masks incl. shared (optional)
  02_<stem>_vp_residual.npz             VP decomposition incl. shared (optional)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FILES  (prefix 04_raw_<stem>_)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  04_raw_<stem>_baseline_similarity.npz
      float32 (n_ast, n_builtin, L+1)  cos_sim between every (A, B) pair per layer

  04_raw_<stem>_baseline_similarity_summary.json
      Per-layer mean, std, min, max of |cos_sim|; mean orthogonality (1-|cos_sim|)

  04_raw_<stem>_shared_additivity.npz
      Additivity scores restricted to shared dims, float32 (N_explicit, L+1)
      Arrays: scores, ast_nodes (object), builtin_objs (object)

  04_raw_<stem>_interaction_vs_shared_vp.json
      Spearman rho + p-value per layer between interaction_norm and shared VP

Usage
-----
  python 04_raw_similarity.py --stem contrastive_stubs
  python 04_raw_similarity.py --stem contrastive_stubs --plot
  python 04_raw_similarity.py --stem contrastive_stubs --plot --layer -1
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
from scipy.stats import spearmanr


# ─────────────────────────────────────────────────────────────────────────────
# Cosine similarity helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cos_sim_vecs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between two (L+1, H) arrays, computed per layer.
    Returns 1-D array of length L+1.
    """
    dot = (a * b).sum(axis=-1)                          # (L+1,)
    na  = np.linalg.norm(a, axis=-1).clip(1e-12)        # (L+1,)
    nb  = np.linalg.norm(b, axis=-1).clip(1e-12)        # (L+1,)
    return np.clip(dot / (na * nb), -1.0, 1.0)


def _cos_sim_flat(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / denom)


# ─────────────────────────────────────────────────────────────────────────────
# Loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_baselines(base: Path, stem: str) -> tuple[dict, dict] | tuple[None, None]:
    """
    Load μ_ast and μ_builtin from step 04 baseline npz files.
    Returns (mu_ast, mu_builtin) dicts: name -> (L+1, H) array,
    or (None, None) if files are absent.
    """
    ast_path = base / f"04_{stem}_baselines_ast.npz"
    b_path   = base / f"04_{stem}_baselines_builtin.npz"

    if not ast_path.exists() or not b_path.exists():
        return None, None

    ast_npz = np.load(ast_path)
    b_npz   = np.load(b_path)

    mu_ast     = {k: ast_npz[k] for k in ast_npz.files}   # (L+1, H) per key
    mu_builtin = {k: b_npz[k]   for k in b_npz.files}

    print(f"  Loaded μ_ast: {len(mu_ast)} nodes, "
          f"μ_builtin: {len(mu_builtin)} builtins")
    return mu_ast, mu_builtin


def _load_masks(base: Path, stem: str) -> dict | None:
    """
    Load purity masks from step 02.
    Returns dict with keys ast_pure, builtin_pure, shared (each (L+1, H) bool),
    plus shared_combined = shared | interaction (if available).
    Returns None if the file does not exist.
    """
    path = base / f"02_{stem}_purity_masks.npz"
    if not path.exists():
        return None
    npz = np.load(path, allow_pickle=True)
    result: dict[str, np.ndarray] = {}
    for key in ("ast_pure", "builtin_pure", "shared", "interaction", "noise"):
        fkey = f"residual_{key}"
        if fkey in npz.files:
            result[key] = npz[fkey].astype(bool)   # (L+1, H)
    if not result:
        return None
    if "interaction" in result and "shared" in result:
        result["shared_combined"] = result["shared"] | result["interaction"]
    elif "shared" in result:
        result["shared_combined"] = result["shared"]
    elif "interaction" in result:
        result["shared_combined"] = result["interaction"]
    print(f"  Loaded purity masks from {path.name}")
    return result


def _load_vp(base: Path, stem: str) -> dict | None:
    """Load VP decomposition from step 02.  Returns dict or None."""
    path = base / f"02_{stem}_vp_residual.npz"
    if not path.exists():
        return None
    npz = np.load(path)
    vp = {k: npz[k] for k in npz.files}
    print(f"  Loaded VP residual from {path.name}  keys={list(vp.keys())[:6]}")
    return vp


def _load_additivity_scores(base: Path, stem: str) -> dict | None:
    """
    Load step 04 additivity scores npz.
    Returns dict with arrays including interaction_norm, or None.
    """
    path = base / f"04_{stem}_additivity_scores.npz"
    if not path.exists():
        return None
    npz = np.load(path, allow_pickle=True)
    result = {k: npz[k] for k in npz.files}
    print(f"  Loaded additivity scores from {path.name}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 1 — Baseline direction similarity
# ─────────────────────────────────────────────────────────────────────────────

def baseline_direction_similarity(
    mu_ast:     dict[str, np.ndarray],   # node -> (L+1, H)
    mu_builtin: dict[str, np.ndarray],   # builtin -> (L+1, H)
) -> tuple[list[str], list[str], np.ndarray]:
    """
    Compute cosine similarity between every (AST node, builtin) pair at every layer.

    Returns (ast_names, builtin_names, sim_matrix) where:
      sim_matrix shape (n_ast, n_builtin, L+1)  values in [-1, 1]
    """
    ast_names = sorted(mu_ast.keys())
    b_names   = sorted(mu_builtin.keys())
    L1 = next(iter(mu_ast.values())).shape[0]

    sim = np.zeros((len(ast_names), len(b_names), L1), dtype=np.float32)
    for i, an in enumerate(ast_names):
        v_a = mu_ast[an]                                  # (L+1, H)
        for j, bn in enumerate(b_names):
            v_b = mu_builtin[bn]                          # (L+1, H)
            sim[i, j, :] = _cos_sim_vecs(v_a, v_b)       # (L+1,)
    return ast_names, b_names, sim


def summarise_baseline_similarity(
    sim: np.ndarray,                    # (n_ast, n_builtin, L+1)
) -> list[dict]:
    """
    Per-layer summary statistics over all (n_ast * n_builtin) pairs.
    Returns list of length L+1, each dict with:
      mean_cos_sim, std_cos_sim, min_cos_sim, max_cos_sim,
      mean_abs_cos_sim, mean_orthogonality (= 1 - mean |cos_sim|)
    """
    n_ast, n_b, L1 = sim.shape
    summary = []
    for l in range(L1):
        vals = sim[:, :, l].ravel()             # (n_ast * n_b,)
        abs_vals = np.abs(vals)
        summary.append({
            "layer":             l,
            "mean_cos_sim":      float(vals.mean()),
            "std_cos_sim":       float(vals.std()),
            "min_cos_sim":       float(vals.min()),
            "max_cos_sim":       float(vals.max()),
            "mean_abs_cos_sim":  float(abs_vals.mean()),
            "mean_orthogonality": float(1.0 - abs_vals.mean()),
        })
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 2 — Shared-space additivity
# ─────────────────────────────────────────────────────────────────────────────

def shared_additivity(
    resid_all:    np.ndarray,          # (N, L+1, H)
    meta:         list[dict],
    mu_ast:       dict[str, np.ndarray],
    mu_builtin:   dict[str, np.ndarray],
    shared_mask:  np.ndarray,          # (L+1, H) bool  — combined shared dims
    variant_types: list[str] = ("explicit",),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute additivity score restricted to shared subspace dims.

    For each eligible stub t:
      For each layer l:
        m = shared_mask[l]
        a_sh = act_t[l, m]
        p_sh = (μ_ast[A][l] + μ_builtin[B][l])[m]
        score[t, l] = cos_sim(a_sh, p_sh)

    Returns (scores, ast_nodes_arr, builtin_objs_arr):
      scores        float32 (N_eligible, L+1)
      ast_nodes_arr object  (N_eligible,)
      builtin_objs_arr object (N_eligible,)
    """
    scores_list = []
    ast_list    = []
    b_list      = []
    skipped     = 0

    for i, m in enumerate(meta):
        if m.get("variant_type", "explicit") not in variant_types:
            continue
        ast_node = m["ast_node"]
        builtin  = m["builtin_obj"]
        if ast_node not in mu_ast or builtin not in mu_builtin:
            skipped += 1
            continue

        act  = resid_all[i]              # (L+1, H)
        mu_a = mu_ast[ast_node]          # (L+1, H)
        mu_b = mu_builtin[builtin]       # (L+1, H)
        pred = mu_a + mu_b               # (L+1, H)

        L1 = act.shape[0]
        row = np.zeros(L1, dtype=np.float32)
        for l in range(L1):
            m_l = shared_mask[l]         # (H,) bool
            if m_l.sum() < 2:
                row[l] = float("nan")
                continue
            a_sh = act[l][m_l]
            p_sh = pred[l][m_l]
            na = np.linalg.norm(a_sh)
            np_ = np.linalg.norm(p_sh)
            if na < 1e-12 or np_ < 1e-12:
                row[l] = float("nan")
            else:
                row[l] = float(np.clip(np.dot(a_sh, p_sh) / (na * np_), -1, 1))

        scores_list.append(row)
        ast_list.append(ast_node)
        b_list.append(builtin)

    if skipped:
        print(f"  Shared additivity: {skipped} stubs skipped (missing baseline)")
    print(f"  Shared additivity: {len(scores_list)} stubs scored")

    if not scores_list:
        return (np.zeros((0, resid_all.shape[1]), dtype=np.float32),
                np.array([], dtype=object),
                np.array([], dtype=object))

    scores = np.stack(scores_list).astype(np.float32)   # (N_e, L+1)
    return scores, np.array(ast_list, dtype=object), np.array(b_list, dtype=object)


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 3 — Interaction norm vs shared VP
# ─────────────────────────────────────────────────────────────────────────────

def interaction_vs_shared_vp(
    resid_all:   np.ndarray,          # (N, L+1, H) — to recompute interaction_norm
    meta:        list[dict],
    mu_ast:      dict[str, np.ndarray],
    mu_builtin:  dict[str, np.ndarray],
    vp:          dict,                # from step 02: keys include "shared" (L+1, H)
    add_scores:  dict | None = None,  # pre-computed step 04 scores (optional)
) -> list[dict]:
    """
    Correlate interaction_norm at each layer with the shared VP fraction.

    shared VP fraction at layer l = mean over dims of vp["shared"][l, :]
    interaction_norm[t, l] = ||act_t[l] - (μ_ast[A][l] + μ_builtin[B][l])||

    Returns list of length L+1, each dict:
      layer, shared_vp_mean, spearman_rho, p_value, n_stubs
    """
    # Determine shared VP key — may be "shared" or "interaction" or sum
    vp_shared_key = None
    for k in ("shared", "interaction"):
        if k in vp:
            vp_shared_key = k
            break
    if vp_shared_key is None:
        print("  No 'shared' or 'interaction' key found in VP; trying first key.")
        vp_shared_key = list(vp.keys())[0]

    vp_sh = vp[vp_shared_key]   # (L+1, H)
    L1    = vp_sh.shape[0]

    # Compute or collect interaction norms for explicit stubs
    # Prefer pre-computed if available
    if add_scores is not None and "interaction_norm" in add_scores:
        # Shape (N_explicit, L+1) — already computed by step 04
        inorm_mat = add_scores["interaction_norm"]     # (N_e, L+1)
        n_stubs   = inorm_mat.shape[0]
        print(f"  Using pre-computed interaction_norm from step 04 "
              f"({n_stubs} stubs).")
    else:
        # Recompute from scratch for explicit stubs
        inorm_rows = []
        for i, m in enumerate(meta):
            if m.get("variant_type", "explicit") != "explicit":
                continue
            ast_node = m["ast_node"]
            builtin  = m["builtin_obj"]
            if ast_node not in mu_ast or builtin not in mu_builtin:
                continue
            act  = resid_all[i]                          # (L+1, H)
            pred = mu_ast[ast_node] + mu_builtin[builtin]  # (L+1, H)
            inorm_rows.append(np.linalg.norm(act - pred, axis=-1))  # (L+1,)
        if not inorm_rows:
            return []
        inorm_mat = np.stack(inorm_rows)               # (N_e, L+1)
        n_stubs   = inorm_mat.shape[0]
        print(f"  Recomputed interaction_norm for {n_stubs} explicit stubs.")

    # Per-layer Spearman: interaction_norm vs shared VP mean
    results = []
    for l in range(L1):
        shared_vp_mean = float(vp_sh[l].mean())        # scalar — same for all stubs
        inorms_l = inorm_mat[:, l]                      # (N_e,)
        # Since shared_vp_mean is a scalar (global per layer), we cannot
        # correlate it with a vector of norms — there's no per-stub VP.
        # Instead we compute the cross-layer correlation (n=L1 observations)
        # separately, but for per-layer output we report the VP mean and
        # describe the Spearman below in cross-layer form.
        # We still write one entry per layer for consistency with other outputs.
        results.append({
            "layer":           l,
            "shared_vp_mean":  shared_vp_mean,
            "mean_interaction_norm": float(np.nanmean(inorms_l)),
            "std_interaction_norm":  float(np.nanstd(inorms_l)),
            "n_stubs":         int(n_stubs),
        })

    # Cross-layer Spearman: does shared_vp_mean predict mean interaction norm?
    vp_means  = np.array([r["shared_vp_mean"]           for r in results])
    norm_means = np.array([r["mean_interaction_norm"]    for r in results])

    if len(vp_means) >= 3:
        rho, pval = spearmanr(vp_means, norm_means)
        cross_layer = {"spearman_rho": float(rho), "p_value": float(pval),
                       "n_layers": int(L1),
                       "interpretation": (
                           "Spearman correlation across layers between "
                           "mean shared VP and mean interaction norm. "
                           "Positive rho: layers with more shared variance "
                           "show stronger interaction norms."
                       )}
        print(f"  Cross-layer Spearman (shared VP vs interaction norm): "
              f"rho={rho:.3f}  p={pval:.4f}")
    else:
        cross_layer = {"spearman_rho": float("nan"), "p_value": float("nan"),
                       "n_layers": int(L1)}

    return results, cross_layer


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

COLOURS = {
    "ast_pure":     "#1565C0",
    "builtin_pure": "#E65100",
    "shared":       "#6A1B9A",
    "raw":          "#607D8B",
    "cross":        "#2E7D32",
}

CMAP_DIV = LinearSegmentedColormap.from_list(
    "divraw4", ["#1565C0", "#FFFFFF", "#E65100"], N=256
)


def plot_baseline_similarity_heatmap(
    ast_names:  list[str],
    b_names:    list[str],
    sim:        np.ndarray,   # (n_ast, n_builtin, L+1)
    layer:      int = -1,
    title:      str = "Baseline direction cosine similarity",
    save_path:  Path | None = None,
) -> None:
    """
    Heatmap of cos_sim(μ_ast[A], μ_builtin[B]) at the chosen layer.
    Blue = anti-correlated, white = orthogonal, orange = aligned (entangled).
    Low values everywhere = good: AST and builtin baselines are independent.
    High values = step 04 additivity scores are structurally inflated.
    """
    mat = sim[:, :, layer]           # (n_ast, n_builtin)
    n_ast, n_b = mat.shape
    fig_w = max(6, n_b * 0.4 + 1.5)
    fig_h = max(4, n_ast * 0.4 + 1.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat, aspect="auto", cmap=CMAP_DIV, vmin=-1.0, vmax=1.0,
                   interpolation="nearest")
    plt.colorbar(im, ax=ax,
                 label="Cosine similarity  (0 = orthogonal, 1 = perfectly aligned)")

    ax.set_xticks(range(n_b))
    ax.set_xticklabels(b_names, rotation=45, ha="right",
                       fontsize=max(5, 9 - n_b // 10))
    ax.set_yticks(range(n_ast))
    ax.set_yticklabels(ast_names, fontsize=max(5, 9 - n_ast // 10))
    ax.set_xlabel("Builtin baseline direction")
    ax.set_ylabel("AST baseline direction")

    if n_ast <= 20 and n_b <= 20:
        for i in range(n_ast):
            for j in range(n_b):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                        fontsize=5, color="black")

    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    plt.close(fig)


def plot_baseline_similarity_by_layer(
    summary: list[dict],
    title:   str = "Baseline orthogonality across layers",
    save_path: Path | None = None,
) -> None:
    """
    Line plot of mean orthogonality (1 - mean |cos_sim|) across layers.
    Also shows mean |cos_sim| with shaded std band.
    High orthogonality = AST and builtin baselines are not aligned.
    Low orthogonality = baselines are correlated -> step 04 scores inflated.
    """
    layers = np.array([r["layer"]           for r in summary])
    ortho  = np.array([r["mean_orthogonality"] for r in summary])
    abs_cs = np.array([r["mean_abs_cos_sim"]   for r in summary])
    std_cs = np.array([r["std_cos_sim"]        for r in summary])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(layers, abs_cs - std_cs, abs_cs + std_cs,
                    alpha=0.2, color=COLOURS["shared"])
    ax.plot(layers, abs_cs, "o-", color=COLOURS["shared"], lw=2,
            label="Mean |cos_sim| ± std")
    ax.plot(layers, ortho, "s--", color=COLOURS["cross"], lw=2,
            label="Mean orthogonality (1 - |cos_sim|)")
    ax.axhline(0.0, color="grey", lw=0.6, linestyle=":")
    ax.axhline(1.0, color="grey", lw=0.6, linestyle="--", alpha=0.4)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine similarity / orthogonality")
    ax.set_ylim(-0.05, 1.1)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(layers)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    plt.close(fig)


def plot_shared_additivity_by_layer(
    scores:       np.ndarray,   # (N_explicit, L+1)
    title:        str = "Shared-space additivity across layers",
    save_path:    Path | None = None,
) -> None:
    """
    Line plot of mean shared-space additivity ± std across layers.
    """
    if scores.shape[0] == 0:
        return
    mean = np.nanmean(scores, axis=0)
    std  = np.nanstd(scores,  axis=0)
    layers = np.arange(len(mean))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(layers, mean - std, mean + std, alpha=0.2, color=COLOURS["shared"])
    ax.plot(layers, mean, "o-", color=COLOURS["shared"], lw=2,
            label="Mean shared-space additivity ± std")
    ax.axhline(1.0, color="grey", lw=0.8, linestyle="--", alpha=0.5,
               label="Perfect additivity (score=1)")
    ax.axhline(0.0, color="grey", lw=0.5, linestyle=":", alpha=0.4)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Additivity score (shared dims only)")
    ax.set_ylim(-0.15, 1.2)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(layers)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    plt.close(fig)


def plot_interaction_vs_shared_vp(
    per_layer:   list[dict],
    cross_layer: dict,
    title:       str = "Interaction norm vs shared VP fraction across layers",
    save_path:   Path | None = None,
) -> None:
    """
    Dual-axis plot: left y = mean interaction norm, right y = shared VP mean.
    Shows whether layers with higher shared VP have larger interaction norms.
    """
    layers  = np.array([r["layer"]               for r in per_layer])
    vp_mean = np.array([r["shared_vp_mean"]       for r in per_layer])
    i_mean  = np.array([r["mean_interaction_norm"] for r in per_layer])
    i_std   = np.array([r["std_interaction_norm"]  for r in per_layer])

    fig, ax1 = plt.subplots(figsize=(11, 4))
    ax2 = ax1.twinx()

    ax1.fill_between(layers, i_mean - i_std, i_mean + i_std,
                     alpha=0.18, color=COLOURS["ast_pure"])
    ax1.plot(layers, i_mean, "o-", color=COLOURS["ast_pure"], lw=2,
             label="Mean interaction norm ± std (left)")
    ax1.set_ylabel("Interaction norm  ||act - (μ_ast + μ_builtin)||",
                   color=COLOURS["ast_pure"])
    ax1.tick_params(axis="y", labelcolor=COLOURS["ast_pure"])

    ax2.plot(layers, vp_mean, "s--", color=COLOURS["shared"], lw=2,
             label="Mean shared VP fraction (right)")
    ax2.set_ylabel("Shared VP fraction (mean over dims)", color=COLOURS["shared"])
    ax2.tick_params(axis="y", labelcolor=COLOURS["shared"])

    rho  = cross_layer.get("spearman_rho", float("nan"))
    pval = cross_layer.get("p_value",      float("nan"))
    sig  = "***" if pval < 0.001 else ("**" if pval < 0.01
           else ("*" if pval < 0.05 else "n.s."))
    ax1.set_title(f"{title}\nCross-layer Spearman rho={rho:.3f}  p={pval:.4f} {sig}",
                  fontsize=10)
    ax1.set_xlabel("Layer")
    ax1.set_xticks(layers)
    ax1.grid(True, alpha=0.2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 4 (raw): baseline direction similarity + shared-space additivity"
    )
    parser.add_argument("--stem",     default="contrastive_stubs",
                        help="File stem used in steps 01/02/04 (default: contrastive_stubs)")
    parser.add_argument("--in_dir", "-d", default="data",
                        help="Directory containing 01_* / 02_* / 04_* input files "
                             "(relative to script dir, default: data)")
    parser.add_argument("--out_dir", "-o", default=None,
                        help="Output directory for results (relative to script dir, "
                             "default: same as --in_dir)")
    parser.add_argument("--layer",    type=int, default=-1,
                        help="Layer for heatmap detail (-1 = final layer)")
    parser.add_argument("--plot",     action="store_true",
                        help="Generate and save plots")
    args = parser.parse_args()

    base    = Path(__file__).parent / args.in_dir
    out     = Path(__file__).parent / (args.out_dir if args.out_dir else args.in_dir)
    out.mkdir(exist_ok=True)
    img_dir = out / "images"
    img_dir.mkdir(exist_ok=True)
    stem = args.stem

    # ── Load inputs ───────────────────────────────────────────────────────────
    print(f"\n[04_raw] Loading inputs for stem='{stem}'")

    resid_path = base / f"01_{stem}_residual_all.npy"
    meta_path  = base / f"01_{stem}_meta.json"
    for p in [resid_path, meta_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"Required input not found: {p}\nRun step 01 first."
            )

    resid_all = np.load(resid_path)               # (N, L+1, H)
    N, L1, H  = resid_all.shape
    print(f"  residual_all: {resid_all.shape}")

    with open(meta_path) as f:
        meta = json.load(f)

    layer_idx = L1 + args.layer if args.layer < 0 else args.layer
    layer_idx = int(np.clip(layer_idx, 0, L1 - 1))
    print(f"  Analysis layer: {layer_idx}")

    # Load baselines
    mu_ast, mu_builtin = _load_baselines(base, stem)
    if mu_ast is None:
        print("  ERROR: Baseline files not found.  Run 04_additivity.py first.")
        return

    # Optional inputs
    masks       = _load_masks(base, stem)
    vp          = _load_vp(base, stem)
    add_scores  = _load_additivity_scores(base, stem)

    if masks is None:
        print("  No purity masks found — shared-space analyses will be skipped.")
    if vp is None:
        print("  No VP residual found — interaction vs VP analysis will be skipped.")

    # ── Analysis 1: Baseline direction similarity ─────────────────────────────
    print(f"\n[04_raw] Computing baseline direction similarity "
          f"({len(mu_ast)} AST x {len(mu_builtin)} builtin x {L1} layers)...")
    ast_names, b_names, sim_mat = baseline_direction_similarity(mu_ast, mu_builtin)

    np.savez_compressed(
        out / f"04_raw_{stem}_baseline_similarity.npz",
        similarity=sim_mat,
        ast_names=np.array(ast_names, dtype=object),
        builtin_names=np.array(b_names, dtype=object),
    )
    print(f"  Saved 04_raw_{stem}_baseline_similarity.npz  shape={sim_mat.shape}")

    sim_summary = summarise_baseline_similarity(sim_mat)
    with open(out / f"04_raw_{stem}_baseline_similarity_summary.json", "w") as f:
        json.dump(sim_summary, f, indent=2, default=float)
    print(f"  Saved 04_raw_{stem}_baseline_similarity_summary.json")

    # Quick console summary at the detail layer
    entry = sim_summary[layer_idx]
    print(f"  Layer {layer_idx}: mean |cos_sim|={entry['mean_abs_cos_sim']:.4f}  "
          f"orthogonality={entry['mean_orthogonality']:.4f}  "
          f"max={entry['max_cos_sim']:.4f}")
    if entry["mean_abs_cos_sim"] > 0.3:
        print("  WARNING: Mean |cos_sim| > 0.3 — baseline directions are substantially "
              "aligned.  Step 04 additivity scores may be structurally inflated.")
    else:
        print("  Baseline directions appear largely orthogonal — step 04 scores "
              "are not substantially contaminated by baseline alignment.")

    # ── Analysis 2: Shared-space additivity ───────────────────────────────────
    shared_scores = None
    if masks is not None and "shared_combined" in masks:
        shared_mask = masks["shared_combined"]   # (L+1, H)
        n_shared = shared_mask.sum(axis=1)
        print(f"\n[04_raw] Computing shared-space additivity "
              f"(min shared dims: {n_shared.min()})...")
        sh_scores, sh_ast_arr, sh_b_arr = shared_additivity(
            resid_all, meta, mu_ast, mu_builtin,
            shared_mask=shared_mask, variant_types=["explicit"],
        )
        shared_scores = sh_scores

        np.savez_compressed(
            out / f"04_raw_{stem}_shared_additivity.npz",
            scores=sh_scores,
            ast_nodes=sh_ast_arr,
            builtin_objs=sh_b_arr,
        )
        print(f"  Saved 04_raw_{stem}_shared_additivity.npz  "
              f"shape={sh_scores.shape}")

        if sh_scores.shape[0] > 0:
            mean_sh = float(np.nanmean(sh_scores[:, layer_idx]))
            print(f"  Mean shared-space additivity at layer {layer_idx}: {mean_sh:.4f}")
    else:
        if masks is None:
            print("\n[04_raw] Skipping shared-space additivity (no purity masks).")
        else:
            print("\n[04_raw] No shared_combined dims in masks — "
                  "skipping shared additivity.")

    # ── Analysis 3: Interaction norm vs shared VP ─────────────────────────────
    if vp is not None:
        print(f"\n[04_raw] Computing interaction norm vs shared VP...")
        result_vp = interaction_vs_shared_vp(
            resid_all, meta, mu_ast, mu_builtin, vp, add_scores
        )
        if result_vp:
            per_layer_vp, cross_layer_vp = result_vp
            ivp_out = {
                "stem":        stem,
                "description": (
                    "Cross-layer Spearman correlation between mean shared VP "
                    "fraction and mean interaction norm.  Positive rho: layers "
                    "with more shared variance show larger interaction norms."
                ),
                "cross_layer": cross_layer_vp,
                "per_layer":   per_layer_vp,
            }
            with open(out / f"04_raw_{stem}_interaction_vs_shared_vp.json", "w") as f:
                json.dump(ivp_out, f, indent=2, default=float)
            print(f"  Saved 04_raw_{stem}_interaction_vs_shared_vp.json")
        else:
            print("  No eligible explicit stubs found for interaction vs VP analysis.")
    else:
        print("\n[04_raw] Skipping interaction vs shared VP (no VP file).")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if args.plot:
        print(f"\n[04_raw] Generating plots...")

        # 1. Baseline similarity heatmap at detail layer
        plot_baseline_similarity_heatmap(
            ast_names, b_names, sim_mat,
            layer=layer_idx,
            title=f"Baseline direction cosine similarity — layer {layer_idx} ({stem})",
            save_path=img_dir / f"04_raw_{stem}_baseline_sim_L{layer_idx}.png",
        )

        # 2. Orthogonality across layers
        plot_baseline_similarity_by_layer(
            sim_summary,
            title=f"Baseline direction orthogonality across layers ({stem})",
            save_path=img_dir / f"04_raw_{stem}_baseline_ortho_layers.png",
        )

        # 3. Shared-space additivity
        if shared_scores is not None and shared_scores.shape[0] > 0:
            plot_shared_additivity_by_layer(
                shared_scores,
                title=f"Shared-space additivity across layers ({stem})",
                save_path=img_dir / f"04_raw_{stem}_shared_additivity_layers.png",
            )

        # 4. Interaction vs shared VP
        if vp is not None and result_vp:
            per_layer_vp, cross_layer_vp = result_vp
            plot_interaction_vs_shared_vp(
                per_layer_vp, cross_layer_vp,
                title=f"Interaction norm vs shared VP fraction ({stem})",
                save_path=img_dir / f"04_raw_{stem}_interaction_vs_vp.png",
            )

        print(f"  Plots saved to {img_dir}/")

    print("\n[04_raw] Done.")


if __name__ == "__main__":
    main()
