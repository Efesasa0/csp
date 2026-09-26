"""
03_projection.py

Step 3 of the AST x builtin mechanistic interpretability pipeline.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY WE DO THIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 02 produced a *map* of which units in the model carry AST-unique,
builtin-unique, or shared variance.  What it did NOT produce is clean,
separated activations — every unit still contains noise from both factors.

Step 03 has two jobs:

  A. PROJECTION — use the purity masks from step 02 to extract clean subspace
     activations:
       AST subspace:     activations projected onto the dimensions flagged
                         "ast_pure" by the purity mask
       Builtin subspace: same using "builtin_pure" dimensions

     Projecting removes dimensions that carry builtin (or AST) variance,
     giving a representation that is as close as possible to "only AST" or
     "only builtin".

  B. RSA (Representational Similarity Analysis) — validate whether AST and
     builtin information are genuinely independent or entangled:
       1. Build N x N Representational Dissimilarity Matrices (RDMs) from
          (i)  raw activations, (ii) AST subspace, (iii) builtin subspace
       2. Build model RDMs: RDM_ast[i,j] = 0 if same AST node else 1;
          RDM_builtin[i,j] = 0 if same builtin else 1
       3. Compute partial Spearman: rho(neural_RDM, RDM_ast | RDM_builtin)
          A high partial correlation means the model has geometry that
          tracks AST independently of builtins, and vice versa.
       4. Test statistical significance via Mantel permutation test (label
          permutation under the null that geometry is random).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT "CLEAN CIRCUITS" MEANS AFTER THIS STEP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

After projection:
  - proj_ast[n]     is a vector living in the AST-pure subspace.
                    Prompts sharing the same AST node will cluster;
                    prompts differing only in builtin will be nearby.
  - proj_builtin[n] is a vector living in the builtin-pure subspace.
                    Prompts sharing the same builtin will cluster;
                    prompts differing only in AST node will be nearby.

If the RSA partial correlations are HIGH (rho > 0.3, p < 0.05):
  -> The model has representationally separated the two factors.
  -> The purity projection is capturing real signal.
  -> AST circuits and builtin circuits are, at least partially, modular.

If partial correlations are LOW (rho ~ 0, p >= 0.05):
  -> AST and builtin are entangled — the model does not cleanly separate them.
  -> Shared units (crosstalk) dominate, and the "clean circuit" hypothesis
     is not supported by the activation geometry.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW RSA WORKS (technical detail)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Neural RDM:
  D_neural[i,j] = 1 - cosine_similarity(act[i], act[j])
  Values range 0 (identical) to 2 (opposite).  Lower triangular only.

Model RDMs:
  D_ast[i,j]     = 0 if ast_node[i] == ast_node[j] else 1
  D_builtin[i,j] = 0 if builtin[i] == builtin[j] else 1

Partial Spearman correlation (controlling for the other RDM):
  1. Spearman-rank both D_neural and D_model vectors (lower-tri)
  2. rho_ast_partial = partial_corr(rank(D_neural), rank(D_ast),
                                    controlling_for=rank(D_builtin))
     = Pearson on residuals after regressing out D_builtin

Mantel permutation test:
  - Permute row/column indices of D_ast N_perm times
  - Each permutation gives a null rho value
  - p-value = fraction of null values >= observed rho

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUT FILES  (from steps 01 and 02)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

From 01_extraction.py:
  01_<stem>_residual_all.npy     float32  (N, L+1, H)
  01_<stem>_meta.json            prompt metadata (ast_node, builtin_obj per row)

From 02_variance_partition.py:
  02_<stem>_vp_residual.npz      VP decomposition (unique_ast, unique_builtin, ...)
  02_<stem>_purity_masks.npz     Boolean masks (ast_pure, builtin_pure, shared, ...)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FILES  (prefix 03_<stem>_)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  03_<stem>_proj_ast.npy          float32  (N, n_ast_dims)  AST subspace projections
  03_<stem>_proj_builtin.npy      float32  (N, n_builtin_dims) builtin subspace projections
  03_<stem>_rdm_neural_raw.npy    float32  (N, N)  cosine-distance RDM, raw activations
  03_<stem>_rdm_neural_ast.npy    float32  (N, N)  cosine-distance RDM, AST subspace
  03_<stem>_rdm_neural_builtin.npy float32 (N, N)  cosine-distance RDM, builtin subspace
  03_<stem>_rdm_model_ast.npy     float32  (N, N)  label-based RDM for AST factor
  03_<stem>_rdm_model_builtin.npy float32  (N, N)  label-based RDM for builtin factor
  03_<stem>_rsa_results.json      RSA scores per layer + p-values + conclusions
  03_<stem>_similarity_matrices.npz  pairwise mean similarity per (category, category) pair

Usage
-----
  python 03_projection.py --stem contrastive_stubs
  python 03_projection.py --stem contrastive_stubs --layer -1 --n_perm 2000 --plot
  python 03_projection.py --stem contrastive_stubs --all_layers --plot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")            # non-interactive backend — safe on remote machines
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import pearsonr, rankdata, spearmanr
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _lower_tri(M: np.ndarray) -> np.ndarray:
    """Return strictly lower-triangular elements as a 1-D vector."""
    idx = np.tril_indices(M.shape[0], k=-1)
    return M[idx]


def _cosine_rdm(X: np.ndarray) -> np.ndarray:
    """
    Build an N x N cosine-distance RDM.
    D[i,j] = 1 - cos_sim(X[i], X[j]),  range [0, 2].
    Uses normalised dot products — O(N^2) but vectorised.
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1e-12, norms)
    Xn    = X / norms                            # (N, D)  unit vectors
    cos_sim = Xn @ Xn.T                          # (N, N)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    return (1.0 - cos_sim).astype(np.float32)   # distance in [0, 2]


def _label_rdm(labels: list) -> np.ndarray:
    """
    Binary label RDM: D[i,j] = 0 if labels[i] == labels[j] else 1.
    """
    arr = np.array(labels)
    return (arr[:, None] != arr[None, :]).astype(np.float32)


def _spearman_rank(v: np.ndarray) -> np.ndarray:
    """Return rank-transformed vector (0-indexed ranks for tied values)."""
    from scipy.stats import rankdata
    return rankdata(v).astype(np.float64)


def _partial_spearman(
    neural_tri: np.ndarray,   # lower-tri of neural RDM, already ranked
    model_tri:  np.ndarray,   # lower-tri of primary model RDM (ranked)
    control_tri: np.ndarray,  # lower-tri of control RDM (ranked)
) -> tuple[float, float]:
    """
    Partial Spearman correlation: corr(neural, model | control).

    Method: regress out control from both neural and model (OLS residuals),
    then compute Pearson on residuals — equivalent to partial correlation.

    Returns (rho, p_value) where p_value is from the t-distribution
    (used here for a quick parametric estimate; Mantel permutation is
    the authoritative test).
    """
    def _resid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
        x = x - x.mean()
        coeff = np.dot(x, y) / (np.dot(x, x) + 1e-12)
        return y - coeff * x

    r_neural = _spearman_rank(neural_tri)
    r_model  = _spearman_rank(model_tri)
    r_ctrl   = _spearman_rank(control_tri)

    res_neural = _resid(r_neural, r_ctrl)
    res_model  = _resid(r_model,  r_ctrl)

    rho, pval = pearsonr(res_neural, res_model)
    return float(rho), float(pval)


def _mantel_permutation(
    neural_tri:  np.ndarray,   # lower-tri, raw (unranked)
    model_rdm:   np.ndarray,   # full N x N  (to be permuted)
    control_rdm: np.ndarray,   # full N x N  control
    n_perm:      int = 1000,
    rng:         np.random.Generator | None = None,
) -> tuple[float, float, np.ndarray]:
    """
    Mantel permutation test.  Permutes rows/cols of model_rdm and recomputes
    partial Spearman each time to build the null distribution.

    For binary model RDMs (label RDMs with only 0/1 values) the permuted
    rankdata result has only two possible rank values that can be pre-computed,
    so we replace the O(n log n) sort with an O(n) binary lookup — this is
    ~100× faster and the critical optimisation for large N.

    Returns (observed_rho, p_value, null_distribution).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    N = model_rdm.shape[0]
    idx_lower = np.tril_indices(N, k=-1)
    i_lo, j_lo = idx_lower

    ctrl_tri  = _lower_tri(control_rdm)
    model_tri = _lower_tri(model_rdm)
    obs_rho, _ = _partial_spearman(neural_tri, model_tri, ctrl_tri)

    # Pre-compute fixed quantities (invariant across all permutations)
    r_ctrl   = rankdata(ctrl_tri).astype(np.float64)
    r_neural = rankdata(neural_tri).astype(np.float64)
    ctrl_c   = r_ctrl - r_ctrl.mean()
    ctrl_cc  = np.dot(ctrl_c, ctrl_c) + 1e-12
    res_neural = r_neural - ctrl_c * (np.dot(ctrl_c, r_neural) / ctrl_cc)
    norm_rn    = np.linalg.norm(res_neural) + 1e-12

    # Check if model_rdm is binary (label RDM) — enables the fast path
    uniq = np.unique(model_rdm)
    is_binary = uniq.size <= 2 and uniq.min() >= 0 and uniq.max() <= 1

    if is_binary:
        # Pre-compute the two rank values for the binary permuted vectors.
        # A permuted binary tri-vector has the same counts of 0s and 1s as
        # the original (row/col permutation preserves the diagonal structure
        # for symmetric RDMs, so n0 and n1 are fixed).
        n_tri = len(model_tri)
        n0 = int((model_tri == 0).sum())
        n1 = n_tri - n0
        rank0 = (n0 + 1) / 2.0           # average rank of the n0 zeros
        rank1 = n0 + (n1 + 1) / 2.0      # average rank of the n1 ones
        # Build pre-ranked fixed vectors: r_perm = rank0 + (rank1-rank0)*is_one
        rank_delta = rank1 - rank0        # scalar
        # We also need the mean-centred ctrl contribution pre-computed
        # (it depends on r_perm but r_perm = rank0 + delta * binary_vec)
        rank0_vec = np.full(n_tri, rank0, dtype=np.float64)
        # res for the constant part (rank0 everywhere)
        res_const = rank0_vec - ctrl_c * (np.dot(ctrl_c, rank0_vec) / ctrl_cc)

        null_rhos = np.empty(n_perm, dtype=np.float64)
        for i in range(n_perm):
            p         = rng.permutation(N)
            # Binary vector: 1 where permuted labels differ
            bin_tri   = model_rdm[p[i_lo], p[j_lo]]          # already 0/1
            # Rank = rank0 + delta * bin_tri  (no sort needed)
            r_delta   = rank_delta * bin_tri                   # (n_tri,)
            dot_d_c   = np.dot(r_delta, ctrl_c)
            res_p     = res_const + r_delta - ctrl_c * (dot_d_c / ctrl_cc)
            denom     = np.linalg.norm(res_p) * norm_rn
            null_rhos[i] = np.dot(res_p, res_neural) / (denom if denom > 1e-12 else 1e-12)
    else:
        # General path: sequential loop, O(n_tril) memory per step
        null_rhos = np.empty(n_perm, dtype=np.float64)
        for i in range(n_perm):
            p        = rng.permutation(N)
            perm_tri = model_rdm[p[i_lo], p[j_lo]].astype(np.float64)
            r_perm   = rankdata(perm_tri)
            res_p    = r_perm - ctrl_c * (np.dot(ctrl_c, r_perm) / ctrl_cc)
            denom    = np.linalg.norm(res_p) * norm_rn
            null_rhos[i] = np.dot(res_p, res_neural) / (denom if denom > 1e-12 else 1e-12)

    p_val = float((null_rhos >= obs_rho).sum() + 1) / (n_perm + 1)
    return obs_rho, p_val, null_rhos


# ─────────────────────────────────────────────────────────────────────────────
# Projection onto purity subspaces
# ─────────────────────────────────────────────────────────────────────────────

def project_subspace(
    activations: np.ndarray,   # (N, H)  full-width activation at one layer
    mask:        np.ndarray,   # (H,) or (L, H) boolean — True = keep this dim
    layer:       int | None = None,
) -> np.ndarray:
    """
    Project activations onto the subspace defined by `mask`.

    If mask is 2-D (L, H), selects row `layer`.
    Returns the subset of dimensions: (N, sum(mask)), dtype float32.

    Note: this is a *selection projection* (dimension masking), not an
    orthogonal projection via QR.  It is appropriate here because the
    VP analysis identifies *which individual dimensions* carry the signal,
    rather than identifying a rotated subspace.  For a rotated subspace,
    use PCA on the selected dimensions downstream.
    """
    if mask.ndim == 2:
        assert layer is not None, "layer required for 2-D mask"
        m = mask[layer]
    else:
        m = mask
    if m.sum() == 0:
        return np.zeros((activations.shape[0], 1), dtype=np.float32)
    return activations[:, m].astype(np.float32)


def project_all_layers(
    resid_all:   np.ndarray,   # (N, L+1, H)
    masks:       dict,         # output of build_purity_masks["residual"]
    factor:      str,          # "ast_pure" or "builtin_pure"
) -> list[np.ndarray]:
    """
    Project activations at every layer onto the subspace for `factor`.
    Returns list of length L+1, each element (N, n_dims_l).
    """
    N, L1, H = resid_all.shape
    mask_all = masks[factor]                    # (L+1, H)
    results = []
    for l in range(L1):
        proj = project_subspace(resid_all[:, l, :], mask_all, layer=l)
        results.append(proj)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# RSA — one-layer entry point
# ─────────────────────────────────────────────────────────────────────────────

def rsa_one_layer(
    act:            np.ndarray,   # (N, D)  activations for this layer
    ast_labels:     list,
    builtin_labels: list,
    n_perm:         int = 1000,
    rng:            np.random.Generator | None = None,
) -> dict:
    """
    Full RSA pipeline for a single activation matrix.

    Returns dict with:
      rdm_neural          (N, N) float32
      rdm_ast             (N, N) float32
      rdm_builtin         (N, N) float32
      rho_ast_partial     float   partial Spearman controlling for builtin
      rho_builtin_partial float   partial Spearman controlling for AST
      p_ast               float   Mantel permutation p-value (AST partial)
      p_builtin           float   Mantel permutation p-value (builtin partial)
      null_ast            (n_perm,)  null distribution for AST
      null_builtin        (n_perm,)  null distribution for builtin
      rho_ast_raw         float   raw (non-partial) Spearman with AST RDM
      rho_builtin_raw     float   raw (non-partial) Spearman with builtin RDM
    """
    if rng is None:
        rng = np.random.default_rng(42)

    rdm_n = _cosine_rdm(act)
    rdm_a = _label_rdm(ast_labels)
    rdm_b = _label_rdm(builtin_labels)

    n_tri   = _lower_tri(rdm_n)
    a_tri   = _lower_tri(rdm_a)
    b_tri   = _lower_tri(rdm_b)

    # Raw Spearman (no partial)
    rho_a_raw, _ = spearmanr(n_tri, a_tri)
    rho_b_raw, _ = spearmanr(n_tri, b_tri)

    # Partial Spearman
    rho_a_p, _ = _partial_spearman(n_tri, a_tri, b_tri)
    rho_b_p, _ = _partial_spearman(n_tri, b_tri, a_tri)

    # Mantel permutation tests
    obs_a, p_a, null_a = _mantel_permutation(n_tri, rdm_a, rdm_b, n_perm, rng)
    obs_b, p_b, null_b = _mantel_permutation(n_tri, rdm_b, rdm_a, n_perm, rng)

    # Note: full N×N RDMs are NOT returned — they are ~256 MB each and
    # accumulate to several GB when called across all layers. The single-layer
    # RDMs needed for saving are computed separately in main().
    return {
        "rho_ast_partial":      rho_a_p,
        "rho_builtin_partial":  rho_b_p,
        "p_ast":                p_a,
        "p_builtin":            p_b,
        "null_ast":             null_a,
        "null_builtin":         null_b,
        "rho_ast_raw":          float(rho_a_raw),
        "rho_builtin_raw":      float(rho_b_raw),
    }


def rsa_all_layers(
    resid_all:      np.ndarray,   # (N, L+1, H)
    proj_ast_list:  list,         # per-layer AST subspace (N, n_ast_l)
    proj_b_list:    list,         # per-layer builtin subspace (N, n_b_l)
    ast_labels:     list,
    builtin_labels: list,
    n_perm:         int = 1000,
) -> list[dict]:
    """
    RSA at every layer: raw, AST-subspace, and builtin-subspace.
    Returns list of length L+1, each entry a dict from rsa_one_layer.
    """
    N, L1, H = resid_all.shape
    print(f"  RSA — {L1} layers × 3 views × 2 Mantel tests "
          f"(n_perm={n_perm})...")

    results = []
    for l in range(L1):
        rng   = np.random.default_rng(42 + l)
        raw   = rsa_one_layer(resid_all[:, l, :], ast_labels, builtin_labels, n_perm, rng)
        ast_r = rsa_one_layer(proj_ast_list[l],   ast_labels, builtin_labels, n_perm, rng)
        b_r   = rsa_one_layer(proj_b_list[l],     ast_labels, builtin_labels, n_perm, rng)
        results.append({"layer": l, "raw": raw, "ast_proj": ast_r, "builtin_proj": b_r})
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Pairwise similarity matrices (category x category)
# ─────────────────────────────────────────────────────────────────────────────

def pairwise_similarity_matrices(
    act:            np.ndarray,   # (N, D)
    ast_labels:     list,
    builtin_labels: list,
) -> dict:
    """
    Compute mean cosine similarity for all pairs of categories.

    Returns:
      ast_ast         (n_ast, n_ast)   mean sim between AST node classes
      builtin_builtin (n_b, n_b)       mean sim between builtin classes
      ast_builtin     (n_ast, n_b)     mean sim between AST and builtin classes
      ast_names       list of AST class names (rows of ast_ast)
      builtin_names   list of builtin class names
    """
    norms = np.linalg.norm(act, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1e-12, norms)
    Xn    = act / norms                         # (N, D) unit vectors

    ast_names     = sorted(set(ast_labels))
    builtin_names = sorted(set(builtin_labels))

    # Pre-compute per-class mean unit vector
    def _class_means(labels: list, names: list) -> np.ndarray:
        means = []
        for name in names:
            idx = [i for i, l in enumerate(labels) if l == name]
            means.append(Xn[idx].mean(axis=0))
        return np.array(means)   # (n_classes, D)

    ast_means = _class_means(ast_labels, ast_names)
    b_means   = _class_means(builtin_labels, builtin_names)

    # Normalise class means before dot product
    def _norm(M):
        n = np.linalg.norm(M, axis=1, keepdims=True)
        return M / np.where(n < 1e-12, 1e-12, n)

    ast_n = _norm(ast_means)   # (n_ast, D)
    b_n   = _norm(b_means)     # (n_b, D)

    ast_ast = np.clip(ast_n @ ast_n.T, -1, 1).astype(np.float32)  # (n_ast, n_ast)
    b_b     = np.clip(b_n   @ b_n.T,   -1, 1).astype(np.float32)  # (n_b, n_b)
    ab      = np.clip(ast_n @ b_n.T,   -1, 1).astype(np.float32)  # (n_ast, n_b)

    # Also compute within-class vs between-class distributions for significance
    cos_sim_all = np.clip(Xn @ Xn.T, -1, 1)  # (N, N)
    arr_ast = np.array(ast_labels)
    arr_b   = np.array(builtin_labels)

    within_ast  = cos_sim_all[arr_ast[:, None]  == arr_ast[None, :]]
    between_ast = cos_sim_all[arr_ast[:, None]  != arr_ast[None, :]]
    within_b    = cos_sim_all[arr_b[:, None]    == arr_b[None, :]]
    between_b   = cos_sim_all[arr_b[:, None]    != arr_b[None, :]]

    return {
        "ast_ast":         ast_ast,
        "builtin_builtin": b_b,
        "ast_builtin":     ab,
        "ast_names":       ast_names,
        "builtin_names":   builtin_names,
        "within_ast":      within_ast.astype(np.float32),
        "between_ast":     between_ast.astype(np.float32),
        "within_builtin":  within_b.astype(np.float32),
        "between_builtin": between_b.astype(np.float32),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Statistical conclusions helper
# ─────────────────────────────────────────────────────────────────────────────

def draw_conclusions(rsa_layer_results: list[dict], alpha: float = 0.05) -> dict:
    """
    Summarise RSA results across all layers and return human-readable
    conclusions about AST vs builtin orthogonality.

    Returns a dict with:
      conclusion_str      str  — multi-line human-readable summary
      ast_sig_layers      list of layer indices where AST RSA is significant
      builtin_sig_layers  list of layer indices
      best_ast_layer      int  — layer with highest partial rho for AST
      best_builtin_layer  int  — layer with highest partial rho for builtin
      are_orthogonal      bool — True if partial rho for one factor stays low
                                 even when the other is significant
    """
    ast_rhos   = [r["raw"]["rho_ast_partial"]    for r in rsa_layer_results]
    b_rhos     = [r["raw"]["rho_builtin_partial"] for r in rsa_layer_results]
    ast_ps     = [r["raw"]["p_ast"]               for r in rsa_layer_results]
    b_ps       = [r["raw"]["p_builtin"]           for r in rsa_layer_results]

    ast_sig  = [i for i, p in enumerate(ast_ps) if p < alpha]
    b_sig    = [i for i, p in enumerate(b_ps)   if p < alpha]

    best_ast = int(np.argmax(ast_rhos))
    best_b   = int(np.argmax(b_rhos))

    # Orthogonality check: if AST is significant but builtin correlation is
    # low at the SAME layer, representations are independent
    both_sig = set(ast_sig) & set(b_sig)

    lines = ["=" * 60, "RSA CONCLUSIONS", "=" * 60]

    if ast_sig:
        lines.append(
            f"AST factor: SIGNIFICANT at layers {ast_sig} "
            f"(best rho={ast_rhos[best_ast]:.3f} at layer {best_ast})"
        )
    else:
        lines.append("AST factor: NOT significant at any layer (p >= {:.2f} everywhere)".format(alpha))

    if b_sig:
        lines.append(
            f"Builtin factor: SIGNIFICANT at layers {b_sig} "
            f"(best rho={b_rhos[best_b]:.3f} at layer {best_b})"
        )
    else:
        lines.append("Builtin factor: NOT significant at any layer")

    if both_sig:
        # Check if partial rhos move together or independently
        rho_diff = [abs(ast_rhos[i] - b_rhos[i]) for i in both_sig]
        if np.mean(rho_diff) > 0.1:
            lines.append(
                f"Orthogonality: At layers where both are significant ({sorted(both_sig)}), "
                "the partial rhos differ substantially — representations are PARTIALLY SEPARABLE."
            )
            are_orthogonal = True
        else:
            lines.append(
                f"Orthogonality: At layers where both are significant ({sorted(both_sig)}), "
                "the partial rhos are similar — representations show CROSSTALK / ENTANGLEMENT."
            )
            are_orthogonal = False
    elif ast_sig and b_sig:
        lines.append(
            "Orthogonality: AST and builtin are significant at DIFFERENT layers — "
            "information is SEQUENTIALLY encoded (not simultaneous crosstalk)."
        )
        are_orthogonal = True
    elif ast_sig or b_sig:
        lines.append(
            "Orthogonality: Only one factor is significant — "
            "cannot conclude orthogonality vs entanglement."
        )
        are_orthogonal = False
    else:
        lines.append(
            "Orthogonality: Neither factor is significant — "
            "model may not separate the two factors in activation space at all."
        )
        are_orthogonal = False

    lines.append("=" * 60)
    conclusion_str = "\n".join(lines)
    print(conclusion_str)

    return {
        "conclusion_str":    conclusion_str,
        "ast_sig_layers":    ast_sig,
        "builtin_sig_layers": b_sig,
        "best_ast_layer":    best_ast,
        "best_builtin_layer": best_b,
        "are_orthogonal":    are_orthogonal,
        "ast_rhos":          ast_rhos,
        "builtin_rhos":      b_rhos,
        "ast_pvalues":       ast_ps,
        "builtin_pvalues":   b_ps,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

COLOURS = {
    "unique_ast":     "#2196F3",   # blue
    "unique_builtin": "#FF9800",   # orange
    "shared":         "#9C27B0",   # purple
    "noise":          "#BDBDBD",   # grey
    "ast_pure":       "#1565C0",   # dark blue
    "builtin_pure":   "#E65100",   # dark orange
    "interaction":    "#E91E63",   # pink
}


def plot_neuron_overlap(
    vp:          dict,          # vp_residual output  keys: unique_ast, unique_builtin, shared, (L+1, H)
    masks:       dict,          # purity_masks["residual"]  keys: ast_pure, builtin_pure, shared, noise
    layer:       int = -1,
    title:       str = "Neuron VP landscape",
    save_path:   Path | None = None,
) -> None:
    """
    2D scatter: x = unique_ast score per unit, y = unique_builtin score.
    Colour by category (ast_pure / builtin_pure / shared / noise).
    Dot SIZE proportional to local kernel density estimate (KDE).
    A unit landing near the top-right is entangled; near top-left is AST-dominant;
    near bottom-right is builtin-dominant; near bottom-left is noise.

    KDE density is approximated by binning into a 2-D grid and bilinear
    interpolating — avoids scipy.stats.gaussian_kde overhead on large arrays.
    """
    ua = vp["unique_ast"][layer]        # (H,)
    ub = vp["unique_builtin"][layer]    # (H,)
    sh = vp["shared"][layer]            # (H,)

    mask_ap = masks["ast_pure"][layer]
    mask_bp = masks["builtin_pure"][layer]
    mask_sh = masks["shared"][layer]
    mask_no = masks["noise"][layer]
    mask_ix = masks["interaction"][layer]

    # Compute density via 2D histogram
    bins = 40
    H_hist, xe, ye = np.histogram2d(ua, ub, bins=bins, range=[[0,1],[0,1]])
    H_smooth = H_hist / H_hist.max().clip(1e-6)

    def _lookup_density(x, y):
        xi = np.clip(((x - xe[0]) / (xe[-1] - xe[0])) * (bins - 1), 0, bins - 1).astype(int)
        yi = np.clip(((y - ye[0]) / (ye[-1] - ye[0])) * (bins - 1), 0, bins - 1).astype(int)
        return H_smooth[xi, yi]

    density = _lookup_density(ua, ub)       # (H,)
    # Map density to dot size: range [10, 250]
    dot_size = 10 + 240 * density

    fig, ax = plt.subplots(figsize=(8, 7))

    category_groups = [
        (mask_ap,  COLOURS["ast_pure"],     "AST-pure",      "s"),
        (mask_bp,  COLOURS["builtin_pure"], "Builtin-pure",  "D"),
        (mask_sh | mask_ix, COLOURS["shared"], "Shared / interaction", "^"),
        (mask_no,  COLOURS["noise"],        "Noise",         "."),
    ]

    for mask_c, colour, label, marker in category_groups:
        if mask_c.sum() == 0:
            continue
        ax.scatter(
            ua[mask_c], ub[mask_c],
            s=dot_size[mask_c],
            c=colour,
            alpha=0.55,
            marker=marker,
            label=f"{label} (n={mask_c.sum()})",
            linewidths=0,
        )

    # Marginal density: shared fraction as annotation
    n_total = len(ua)
    ax.text(0.97, 0.97,
            f"AST-pure: {mask_ap.sum()/n_total:.1%}\n"
            f"Builtin-pure: {mask_bp.sum()/n_total:.1%}\n"
            f"Shared: {(mask_sh|mask_ix).sum()/n_total:.1%}\n"
            f"Noise: {mask_no.sum()/n_total:.1%}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    ax.set_xlabel("Unique AST variance (R²)", fontsize=11)
    ax.set_ylabel("Unique Builtin variance (R²)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.3, label="Equal influence line")
    ax.legend(loc="upper left", fontsize=8, markerscale=1.2)

    # Colorbar-like size legend
    for val, label in [(0.2, "low density"), (0.6, "mid"), (1.0, "high")]:
        ax.scatter([], [], s=10 + 240 * val, c="#555555", alpha=0.6,
                   label=f"density {label}")
    ax.legend(loc="lower right", fontsize=7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    plt.close(fig)


def plot_rdm_matrix(
    rdm:         np.ndarray,   # (N, N)
    labels_ast:  list,
    labels_b:    list,
    sort_by:     str = "ast",  # "ast" | "builtin" | "combined"
    title:       str = "Neural RDM",
    save_path:   Path | None = None,
) -> None:
    """
    Display the N x N cosine-distance RDM sorted by category labels.
    Block structure in the matrix indicates representational similarity
    within categories (dark blocks = similar activations within a class).

    Three sort modes:
      "ast"      — sort prompts by AST node label
      "builtin"  — sort prompts by builtin label
      "combined" — sort prompts by (AST, builtin) pair
    """
    arr_ast = np.array(labels_ast)
    arr_b   = np.array(builtin_labels := labels_b)

    if sort_by == "ast":
        order = np.argsort(arr_ast, kind="stable")
        tick_labels = arr_ast[order]
    elif sort_by == "builtin":
        order = np.argsort(arr_b, kind="stable")
        tick_labels = arr_b[order]
    else:  # combined
        combined = [f"{a}|{b}" for a, b in zip(labels_ast, labels_b)]
        order = np.argsort(combined, kind="stable")
        tick_labels = np.array(combined)[order]

    rdm_sorted = rdm[np.ix_(order, order)]

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(rdm_sorted, aspect="auto", cmap="RdYlBu_r",
                   vmin=0, vmax=np.percentile(rdm, 95), interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Cosine distance (0=same, 2=opposite)")

    # Draw category boundary lines
    if sort_by in ("ast", "builtin"):
        uniq, counts = np.unique(tick_labels, return_counts=True)
        cumcounts = np.cumsum(counts)[:-1] - 0.5
        for c in cumcounts:
            ax.axhline(c, color="white", lw=0.6, alpha=0.7)
            ax.axvline(c, color="white", lw=0.6, alpha=0.7)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel(f"Prompts (sorted by {sort_by})")
    ax.set_ylabel(f"Prompts (sorted by {sort_by})")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    plt.close(fig)


def plot_pairwise_similarity_heatmaps(
    sim:       dict,           # output of pairwise_similarity_matrices
    factor:    str,            # "ast_ast" | "builtin_builtin" | "ast_builtin"
    title:     str = "",
    save_path: Path | None = None,
) -> None:
    """
    Annotated heatmap of mean cosine similarity between class mean vectors.
    Rows and columns are labelled with class names.

    Diagonal (same class) should be ~1 for well-separated categories.
    Off-diagonal entries reveal cross-category similarity.
    High off-diagonal = similar activation geometry = likely entangled.
    """
    mat       = sim[factor]
    row_names = sim["ast_names"] if factor.startswith("ast") else sim["builtin_names"]
    col_names = sim["ast_names"] if factor == "ast_ast" else sim["builtin_names"]

    n_rows, n_cols = mat.shape
    fig_w = max(6, n_cols * 0.35 + 1)
    fig_h = max(5, n_rows * 0.35 + 1)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    vmin = -0.2
    vmax = 1.0
    cmap = LinearSegmentedColormap.from_list(
        "rsa", ["#1565C0", "#FFFFFF", "#E65100"], N=256
    )
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Cosine similarity")

    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels(col_names, rotation=45, ha="right", fontsize=max(4, 8 - n_cols // 10))
    ax.set_yticklabels(row_names, fontsize=max(4, 8 - n_rows // 10))

    # Annotate cells if small enough
    if n_rows <= 20 and n_cols <= 20:
        for i in range(n_rows):
            for j in range(n_cols):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                        fontsize=6, color="black" if abs(mat[i,j]) < 0.5 else "white")

    ax.set_title(title or f"Class-mean cosine similarity: {factor}", fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    plt.close(fig)


def plot_within_between_distributions(
    sim:       dict,           # output of pairwise_similarity_matrices
    title:     str = "Within vs between class similarity",
    save_path: Path | None = None,
) -> None:
    """
    KDE / violin plots comparing within-class vs between-class cosine similarity
    for AST factor and builtin factor.

    A well-separated factor has:
      within-class sim  >> between-class sim  (large gap)
    An entangled factor has both distributions overlapping heavily.

    Includes a Mann-Whitney U test for statistical significance.
    """
    from scipy.stats import mannwhitneyu

    groups = [
        ("AST within",    sim["within_ast"],   COLOURS["ast_pure"]),
        ("AST between",   sim["between_ast"],  "#90CAF9"),
        ("Builtin within", sim["within_builtin"], COLOURS["builtin_pure"]),
        ("Builtin between", sim["between_builtin"], "#FFCC80"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax_idx, (factor, within_key, between_key, colour_w, colour_b) in enumerate([
        ("AST",     "within_ast",     "between_ast",     COLOURS["ast_pure"],     "#90CAF9"),
        ("Builtin", "within_builtin", "between_builtin", COLOURS["builtin_pure"], "#FFCC80"),
    ]):
        ax = axes[ax_idx]
        w = sim[within_key].ravel()
        b = sim[between_key].ravel()

        # Subsample for speed if huge
        rng = np.random.default_rng(42)
        max_n = 50_000
        if len(w) > max_n:
            w = rng.choice(w, max_n, replace=False)
        if len(b) > max_n:
            b = rng.choice(b, max_n, replace=False)

        bins = np.linspace(-0.2, 1.0, 60)
        ax.hist(w, bins=bins, alpha=0.6, color=colour_w, density=True,
                label=f"Within {factor} (n={sim[within_key].ravel().shape[0]:,})")
        ax.hist(b, bins=bins, alpha=0.5, color=colour_b, density=True,
                label=f"Between {factor} (n={sim[between_key].ravel().shape[0]:,})")

        ax.axvline(np.median(w), color=colour_w, lw=1.5, linestyle="--",
                   label=f"median within={np.median(w):.3f}")
        ax.axvline(np.median(b), color=colour_b, lw=1.5, linestyle="--",
                   label=f"median between={np.median(b):.3f}")

        # Statistical test
        stat, p = mannwhitneyu(w, b, alternative="greater")
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        ax.set_title(f"{factor}  (U={stat:.0f}, p={p:.2e}  {sig})", fontsize=10)
        ax.set_xlabel("Cosine similarity")
        ax.set_ylabel("Density" if ax_idx == 0 else "")
        ax.legend(fontsize=7)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    plt.close(fig)


def plot_rsa_layer_profile(
    rsa_results:  list[dict],
    conclusions:  dict,
    alpha:        float = 0.05,
    title:        str = "RSA partial Spearman by layer",
    save_path:    Path | None = None,
) -> None:
    """
    Line plot: partial Spearman rho for AST and builtin factors across layers,
    with significance markers (* p<0.05, ** p<0.01, *** p<0.001).
    Shaded band = null distribution 5-95th percentile.
    Three panels: raw activations | AST subspace projection | builtin subspace projection.
    """
    L = len(rsa_results)
    layers = np.arange(L)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    panel_keys = ["raw", "ast_proj", "builtin_proj"]
    panel_titles = ["Raw activations", "AST subspace projection", "Builtin subspace projection"]

    for ax, pkey, ptitle in zip(axes, panel_keys, panel_titles):
        rho_a = np.array([r[pkey]["rho_ast_partial"]    for r in rsa_results])
        rho_b = np.array([r[pkey]["rho_builtin_partial"] for r in rsa_results])
        p_a   = np.array([r[pkey]["p_ast"]              for r in rsa_results])
        p_b   = np.array([r[pkey]["p_builtin"]          for r in rsa_results])

        null_a_lo = np.array([np.percentile(r[pkey]["null_ast"],     5) for r in rsa_results])
        null_a_hi = np.array([np.percentile(r[pkey]["null_ast"],    95) for r in rsa_results])
        null_b_lo = np.array([np.percentile(r[pkey]["null_builtin"], 5) for r in rsa_results])
        null_b_hi = np.array([np.percentile(r[pkey]["null_builtin"],95) for r in rsa_results])

        ax.fill_between(layers, null_a_lo, null_a_hi, alpha=0.12, color=COLOURS["unique_ast"])
        ax.fill_between(layers, null_b_lo, null_b_hi, alpha=0.12, color=COLOURS["unique_builtin"])

        ax.plot(layers, rho_a, "o-", color=COLOURS["unique_ast"],     lw=2, label="AST (partial)")
        ax.plot(layers, rho_b, "s-", color=COLOURS["unique_builtin"], lw=2, label="Builtin (partial)")
        ax.axhline(0, color="k", lw=0.6, linestyle="--")

        # Significance markers
        for l in range(L):
            sig_a = "***" if p_a[l] < 0.001 else ("**" if p_a[l] < 0.01 else ("*" if p_a[l] < alpha else ""))
            sig_b = "***" if p_b[l] < 0.001 else ("**" if p_b[l] < 0.01 else ("*" if p_b[l] < alpha else ""))
            if sig_a:
                ax.text(l, rho_a[l] + 0.02, sig_a, ha="center", va="bottom",
                        fontsize=8, color=COLOURS["unique_ast"])
            if sig_b:
                ax.text(l, rho_b[l] - 0.04, sig_b, ha="center", va="top",
                        fontsize=8, color=COLOURS["unique_builtin"])

        ax.set_xticks(layers)
        ax.set_xlabel("Layer")
        ax.set_title(ptitle, fontsize=10)
        ax.legend(fontsize=8)
        ax.set_ylim(-0.5, 0.7)

    axes[0].set_ylabel("Partial Spearman rho", fontsize=11)
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    plt.close(fig)


def plot_mantel_null_distribution(
    rsa_layer:  dict,           # single layer entry from rsa_all_layers
    layer:      int,
    save_path:  Path | None = None,
) -> None:
    """
    For a chosen layer, plot histograms of Mantel null distributions
    for AST and builtin with observed rho marked as a vertical line.
    Shows clearly whether the observed RSA is above the null.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, key, colour, label in [
        (axes[0], "null_ast",     COLOURS["unique_ast"],     "AST"),
        (axes[1], "null_builtin", COLOURS["unique_builtin"], "Builtin"),
    ]:
        null = rsa_layer["raw"][key]
        obs  = rsa_layer["raw"][f"rho_{key.split('_')[1]}_partial"]
        p    = rsa_layer["raw"][f"p_{key.split('_')[1]}"]

        ax.hist(null, bins=40, color=colour, alpha=0.7, density=True,
                label=f"Null distribution (n={len(null)})")
        ax.axvline(obs, color="black", lw=2, linestyle="-",
                   label=f"Observed rho={obs:.3f}")
        ax.axvline(np.percentile(null, 95), color="red", lw=1.5,
                   linestyle="--", label="95th percentile null")
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        ax.set_title(f"{label} RSA — Layer {layer}  (p={p:.3f} {sig})", fontsize=10)
        ax.set_xlabel("Partial Spearman rho")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    plt.close(fig)


def plot_neuron_overlap_all_layers(
    vp:        dict,    # vp_residual  (L+1, H)
    masks:     dict,    # purity_masks["residual"]
    save_path: Path | None = None,
) -> None:
    """
    Grid of neuron overlap scatter plots, one panel per layer.
    Gives a quick scan of how unit allocation changes across depth.
    """
    n_layers = vp["unique_ast"].shape[0]
    ncols = min(6, n_layers)
    nrows = int(np.ceil(n_layers / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 2.8, nrows * 2.8),
                             sharex=True, sharey=True)
    axes_flat = np.array(axes).ravel()

    for l in range(n_layers):
        ax = axes_flat[l]
        ua = vp["unique_ast"][l]
        ub = vp["unique_builtin"][l]

        mask_ap = masks["ast_pure"][l]
        mask_bp = masks["builtin_pure"][l]
        mask_sh = masks["shared"][l] | masks["interaction"][l]
        mask_no = masks["noise"][l]

        ax.scatter(ua[mask_no], ub[mask_no], s=3,  c=COLOURS["noise"],       alpha=0.3)
        ax.scatter(ua[mask_ap], ub[mask_ap], s=8,  c=COLOURS["ast_pure"],    alpha=0.7)
        ax.scatter(ua[mask_bp], ub[mask_bp], s=8,  c=COLOURS["builtin_pure"],alpha=0.7)
        ax.scatter(ua[mask_sh], ub[mask_sh], s=8,  c=COLOURS["shared"],      alpha=0.7)

        ax.set_title(f"L{l}", fontsize=8)
        ax.set_xlim(0, 0.8)
        ax.set_ylim(0, 0.8)
        ax.tick_params(labelsize=6)

    for l in range(n_layers, len(axes_flat)):
        axes_flat[l].set_visible(False)

    fig.supxlabel("Unique AST variance", fontsize=10)
    fig.supylabel("Unique Builtin variance", fontsize=10)
    fig.suptitle("Neuron VP landscape across all layers", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    plt.close(fig)


def plot_all_pairwise_matrices(
    sim_raw:     dict,
    sim_ast:     dict,
    sim_b:       dict,
    save_path:   Path | None = None,
) -> None:
    """
    3 x 3 grid: rows = {raw, AST subspace, builtin subspace}
                cols = {AST vs AST, builtin vs builtin, AST vs builtin}

    Allows direct comparison of how the subspace projection affects
    each class's representational geometry.
    """
    row_labels = ["Raw activations", "AST subspace proj.", "Builtin subspace proj."]
    col_labels = ["AST vs AST", "Builtin vs Builtin", "AST vs Builtin"]
    sims       = [sim_raw, sim_ast, sim_b]
    sim_keys   = ["ast_ast", "builtin_builtin", "ast_builtin"]

    fig, axes = plt.subplots(3, 3, figsize=(15, 13))

    cmap = LinearSegmentedColormap.from_list(
        "rsa2", ["#1565C0", "#FFFFFF", "#E65100"], N=256
    )

    for row_i, sim in enumerate(sims):
        for col_i, skey in enumerate(sim_keys):
            ax = axes[row_i][col_i]
            mat = sim[skey]
            im  = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=-0.1, vmax=1.0,
                            interpolation="nearest")
            if row_i == 0:
                ax.set_title(col_labels[col_i], fontsize=10)
            if col_i == 0:
                ax.set_ylabel(row_labels[row_i], fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

            # Label axes with class names if small enough
            if skey == "ast_ast":
                names = sim["ast_names"]
            elif skey == "builtin_builtin":
                names = sim["builtin_names"]
            else:
                names = sim["builtin_names"]  # x-axis = builtin
            if len(names) <= 20:
                ax.set_xticks(range(len(names)))
                ax.set_xticklabels(names, rotation=45, ha="right", fontsize=5)

    fig.suptitle("Pairwise class-mean cosine similarity: raw vs projected subspaces", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Step 3: subspace projection + RSA")
    parser.add_argument("--stem",     default="contrastive_stubs",
                        help="File stem used in steps 01 and 02 (default: contrastive_stubs)")
    parser.add_argument("--in_dir", "-d", default="data",
                        help="Directory containing 01_ and 02_ output files (default: data/)")
    parser.add_argument("--out_dir", "-o", default=None,
                        help="Output directory for results (default: same as --in_dir)")
    parser.add_argument("--layer",    type=int, default=-1,
                        help="Which residual layer to use for detailed RDMs/plots. "
                             "-1 means final layer. Ignored when --all_layers is set.")
    parser.add_argument("--all_layers", action="store_true",
                        help="Run RSA at every layer (slow; produces one set of plots per layer)")
    parser.add_argument("--n_perm",  type=int, default=1000,
                        help="Number of Mantel permutations (default 1000)")
    parser.add_argument("--alpha",   type=float, default=0.05,
                        help="Significance threshold for conclusions (default 0.05)")
    parser.add_argument("--plot",    action="store_true",
                        help="Generate and save all plots")
    parser.add_argument("--no_rsa",  action="store_true",
                        help="Skip RSA (useful for large N where permutations are slow)")
    args = parser.parse_args()

    base    = Path(__file__).parent / args.in_dir
    out     = Path(__file__).parent / (args.out_dir if args.out_dir else args.in_dir)
    out.mkdir(exist_ok=True)
    img_dir = out / "images"
    img_dir.mkdir(exist_ok=True)
    stem  = args.stem

    # ── Load inputs ──────────────────────────────────────────────────────────
    print(f"\n[03] Loading inputs for stem='{stem}'")

    resid_all_path   = base / f"01_{stem}_residual_all.npy"
    meta_path        = base / f"01_{stem}_meta.json"
    vp_residual_path = base / f"02_{stem}_vp_residual.npz"
    purity_path      = base / f"02_{stem}_purity_masks.npz"

    for p in [resid_all_path, meta_path, vp_residual_path, purity_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"Required input not found: {p}\n"
                "Run steps 01 and 02 first."
            )

    resid_all = np.load(resid_all_path)                # (N, L+1, H)
    N, L1, H  = resid_all.shape
    print(f"  residual_all: {resid_all.shape}")

    with open(meta_path) as f:
        meta = json.load(f)
    ast_labels     = [m["ast_node"]    for m in meta]
    builtin_labels = [m["builtin_obj"] for m in meta]
    print(f"  N={N} prompts, L+1={L1} layers, H={H}")
    print(f"  AST categories:    {len(set(ast_labels))}")
    print(f"  Builtin categories:{len(set(builtin_labels))}")

    vp_npz = np.load(vp_residual_path)
    vp_residual = {k: vp_npz[k] for k in vp_npz.files}   # (L+1, H) per key

    pm_npz = np.load(purity_path, allow_pickle=True)
    # purity_masks["residual"]["ast_pure"] shape (L+1, H)
    masks_residual = {k: pm_npz[k] for k in pm_npz.files if k.startswith("residual")}
    # strip "residual_" prefix
    masks_residual = {k[len("residual_"):]: v for k, v in masks_residual.items()}

    # ── Projection ───────────────────────────────────────────────────────────
    print("\n[03] Projecting activations onto purity subspaces...")

    proj_ast_list = project_all_layers(resid_all, masks_residual, "ast_pure")
    proj_b_list   = project_all_layers(resid_all, masks_residual, "builtin_pure")

    layer_idx = L1 + args.layer if args.layer < 0 else args.layer
    layer_idx = int(np.clip(layer_idx, 0, L1 - 1))

    proj_ast_final = proj_ast_list[layer_idx]     # (N, n_ast_dims)
    proj_b_final   = proj_b_list[layer_idx]       # (N, n_b_dims)
    print(f"  Layer {layer_idx}: AST subspace dims={proj_ast_final.shape[1]}, "
          f"Builtin subspace dims={proj_b_final.shape[1]}")

    # Save projections
    np.save(out / f"03_{stem}_proj_ast.npy",     proj_ast_final)
    np.save(out / f"03_{stem}_proj_builtin.npy", proj_b_final)
    print(f"  Saved 03_{stem}_proj_ast.npy  and  03_{stem}_proj_builtin.npy")

    # ── RDMs ─────────────────────────────────────────────────────────────────
    print(f"\n[03] Building RDMs at layer {layer_idx}...")

    act_raw  = resid_all[:, layer_idx, :]      # (N, H)

    rdm_n_raw  = _cosine_rdm(act_raw)
    rdm_n_ast  = _cosine_rdm(proj_ast_final) if proj_ast_final.shape[1] > 1 else rdm_n_raw
    rdm_n_b    = _cosine_rdm(proj_b_final)   if proj_b_final.shape[1]   > 1 else rdm_n_raw
    rdm_m_ast  = _label_rdm(ast_labels)
    rdm_m_b    = _label_rdm(builtin_labels)

    np.save(out / f"03_{stem}_rdm_neural_raw.npy",     rdm_n_raw)
    np.save(out / f"03_{stem}_rdm_neural_ast.npy",     rdm_n_ast)
    np.save(out / f"03_{stem}_rdm_neural_builtin.npy", rdm_n_b)
    np.save(out / f"03_{stem}_rdm_model_ast.npy",      rdm_m_ast)
    np.save(out / f"03_{stem}_rdm_model_builtin.npy",  rdm_m_b)
    print("  RDMs saved.")

    # ── Pairwise similarity matrices ──────────────────────────────────────────
    print("\n[03] Computing pairwise similarity matrices...")
    sim_raw = pairwise_similarity_matrices(act_raw,        ast_labels, builtin_labels)
    sim_ast = pairwise_similarity_matrices(proj_ast_final, ast_labels, builtin_labels)
    sim_b   = pairwise_similarity_matrices(proj_b_final,   ast_labels, builtin_labels)

    np.savez_compressed(
        out / f"03_{stem}_similarity_matrices.npz",
        ast_ast_raw         = sim_raw["ast_ast"],
        builtin_builtin_raw = sim_raw["builtin_builtin"],
        ast_builtin_raw     = sim_raw["ast_builtin"],
        ast_ast_proj        = sim_ast["ast_ast"],
        builtin_builtin_proj= sim_b["builtin_builtin"],
        ast_builtin_proj    = sim_ast["ast_builtin"],
        ast_names   = np.array(sim_raw["ast_names"],     dtype=object),
        builtin_names = np.array(sim_raw["builtin_names"], dtype=object),
    )
    print(f"  Saved 03_{stem}_similarity_matrices.npz")

    # ── RSA ──────────────────────────────────────────────────────────────────
    rsa_layer_results = None
    conclusions       = None

    if not args.no_rsa:
        print(f"\n[03] Running RSA (n_perm={args.n_perm}) across all layers...")
        rsa_layer_results = rsa_all_layers(
            resid_all, proj_ast_list, proj_b_list,
            ast_labels, builtin_labels, n_perm=args.n_perm
        )

        conclusions = draw_conclusions(rsa_layer_results, alpha=args.alpha)

        # Serialise results (numpy arrays → lists for JSON)
        def _serialise(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _serialise(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_serialise(x) for x in obj]
            return obj

        rsa_to_save = []
        for r in rsa_layer_results:
            entry = {"layer": r["layer"]}
            for pkey in ["raw", "ast_proj", "builtin_proj"]:
                d = r[pkey]
                entry[pkey] = {
                    "rho_ast_partial":     d["rho_ast_partial"],
                    "rho_builtin_partial": d["rho_builtin_partial"],
                    "p_ast":               d["p_ast"],
                    "p_builtin":           d["p_builtin"],
                    "rho_ast_raw":         d["rho_ast_raw"],
                    "rho_builtin_raw":     d["rho_builtin_raw"],
                    # omit large null arrays from JSON; they're in plots
                }
            rsa_to_save.append(entry)

        rsa_json = {
            "stem":        stem,
            "layer_used":  layer_idx,
            "n_perm":      args.n_perm,
            "alpha":       args.alpha,
            "results_per_layer": rsa_to_save,
            "conclusions":       {k: _serialise(v)
                                  for k, v in conclusions.items()
                                  if k != "conclusion_str"},
            "conclusion_str": conclusions["conclusion_str"],
        }
        rsa_path = out / f"03_{stem}_rsa_results.json"
        with open(rsa_path, "w") as f:
            json.dump(rsa_json, f, indent=2)
        print(f"\n  Saved -> {rsa_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if args.plot:
        print(f"\n[03] Generating plots (layer={layer_idx})...")

        # 1. Neuron overlap scatter — one panel for chosen layer
        plot_neuron_overlap(
            vp_residual, masks_residual, layer=layer_idx,
            title=f"Neuron VP landscape — layer {layer_idx}",
            save_path=img_dir / f"03_{stem}_neuron_overlap_L{layer_idx}.png",
        )

        # 2. Neuron overlap across all layers (grid)
        plot_neuron_overlap_all_layers(
            vp_residual, masks_residual,
            save_path=img_dir / f"03_{stem}_neuron_overlap_all_layers.png",
        )

        # 3. Neural RDMs sorted three ways
        for sort_by in ("ast", "builtin", "combined"):
            plot_rdm_matrix(
                rdm_n_raw, ast_labels, builtin_labels, sort_by=sort_by,
                title=f"Neural RDM (raw, sorted by {sort_by}) — layer {layer_idx}",
                save_path=img_dir / f"03_{stem}_rdm_raw_sorted_{sort_by}.png",
            )

        # 4. Pairwise similarity heatmaps (raw)
        for fkey, ftitle in [
            ("ast_ast",         "AST vs AST — raw activations"),
            ("builtin_builtin", "Builtin vs Builtin — raw activations"),
            ("ast_builtin",     "AST vs Builtin — raw activations"),
        ]:
            plot_pairwise_similarity_heatmaps(
                sim_raw, fkey, title=ftitle,
                save_path=img_dir / f"03_{stem}_sim_{fkey}_raw.png",
            )

        # 5. All pairwise matrices in one grid (raw vs proj)
        plot_all_pairwise_matrices(
            sim_raw, sim_ast, sim_b,
            save_path=img_dir / f"03_{stem}_sim_all_comparison.png",
        )

        # 6. Within vs between class similarity distributions
        for sim, suffix in [(sim_raw, "raw"), (sim_ast, "ast_proj"), (sim_b, "builtin_proj")]:
            plot_within_between_distributions(
                sim, title=f"Within vs between class cosine similarity ({suffix})",
                save_path=img_dir / f"03_{stem}_within_between_{suffix}.png",
            )

        # 7. RSA layer profile + Mantel null (only if RSA was run)
        if rsa_layer_results is not None:
            plot_rsa_layer_profile(
                rsa_layer_results, conclusions, alpha=args.alpha,
                title=f"RSA partial Spearman by layer ({stem})",
                save_path=img_dir / f"03_{stem}_rsa_layer_profile.png",
            )

            # Mantel null distribution for chosen layer
            plot_mantel_null_distribution(
                rsa_layer_results[layer_idx], layer=layer_idx,
                save_path=img_dir / f"03_{stem}_mantel_null_L{layer_idx}.png",
            )

        print(f"\n  All plots saved to {out}/")

    print("\n[03] Done.")


if __name__ == "__main__":
    main()
