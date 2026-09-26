"""
03_raw_rsa.py

Step 3 (raw counterpart) of the AST x builtin mechanistic interpretability pipeline.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY WE DO THIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

03_projection.py runs RSA after projecting activations onto pure (ast_pure or
builtin_pure) subspaces.  The projection *discards* the shared subspace — the
exactly the dimensions where AST-builtin cross-factor interaction lives.  This
means 03_projection.py answers "are the pure circuits separable?" but cannot
address "does the shared subspace encode both factors jointly?".

This script answers three complementary questions:

  1. RAW RSA
     Run the same partial-Spearman RSA directly on all 2048 (H) dimensions,
     without any projection.  This is the "full picture" view: how well can
     the full representation geometry track each factor independently?  Raw
     RSA does NOT remove shared signal, so high rho here may reflect both
     pure AND shared contributions.

  2. SHARED SUBSPACE RSA
     Restrict activations to the shared dims (where BOTH AST and builtin have
     signal, as flagged by the VP purity masks from step 02).  If the shared
     subspace is a true interaction circuit, it should show significant partial
     Spearman for BOTH factors simultaneously.  If it is dominated by one
     factor, only that factor will be significant.

  3. CROSS-RDM CORRELATION
     Correlate the neural RDM built from AST-pure dims against the neural RDM
     built from builtin-pure dims.  A high correlation means the two
     representational geometries are themselves geometrically similar — i.e.
     the model organises AST concepts and builtin concepts in the same way
     in their respective subspaces.  This is a structural form of entanglement
     that survives purity projection.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW IT IS COMPUTED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Neural RDM:
  D_neural[i,j] = 1 - cosine_similarity(act[i], act[j])
  Values range 0 (identical) to 2 (opposite).  Lower triangular only.

Label RDMs:
  D_ast[i,j]     = 0 if ast_node[i] == ast_node[j] else 1
  D_builtin[i,j] = 0 if builtin[i] == builtin[j] else 1

Partial Spearman correlation (controlling for the other factor's label RDM):
  1. Spearman-rank both D_neural and D_label vectors (lower-tri)
  2. rho_ast_partial = partial_corr(rank(D_neural), rank(D_ast),
                                    controlling_for=rank(D_builtin))
     = Pearson on residuals after regressing out D_builtin

Mantel permutation test (per analysis type):
  - Permute row/column indices of D_label n_perm times
  - Each permutation gives a null rho
  - p-value = fraction of null rhos >= observed rho

Cross-RDM correlation:
  D_ast_neural  = _cosine_rdm(act[:, ast_pure_dims])
  D_b_neural    = _cosine_rdm(act[:, builtin_pure_dims])
  rho_cross     = Spearman(lower_tri(D_ast_neural), lower_tri(D_b_neural))
  Positive rho  -> the two pure subspaces organise prompts in the same way
                   (structural entanglement).
  Near-zero rho -> the two subspaces have independent representational
                   geometries (clean circuit hypothesis supported).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUT FILES  (from steps 01 and 02)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  01_<stem>_residual_all.npy     float32  (N, L+1, H)
  01_<stem>_meta.json            prompt metadata (ast_node, builtin_obj per row)
  02_<stem>_purity_masks.npz     boolean masks: ast_pure, builtin_pure, shared
                                 (optional; raw-only analysis if absent)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FILES  (prefix 03_raw_<stem>_)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  03_raw_<stem>_rsa_raw.json
      Partial Spearman rho + Mantel p-value per layer on raw (full-H) activations.
      Keys per layer: rho_ast_partial, rho_builtin_partial, p_ast, p_builtin,
                      rho_ast_raw (non-partial), rho_builtin_raw.

  03_raw_<stem>_rsa_shared.json
      Same structure, computed on shared subspace activations only.
      Only written if purity masks are available and shared dims exist.

  03_raw_<stem>_cross_rdm_correlation.json
      Per-layer Spearman correlation between the AST-pure neural RDM and the
      builtin-pure neural RDM.  Positive rho = structural entanglement.
      Only written if purity masks are available.

  03_raw_<stem>_rdm_shared.npy
      float32 (N, N) cosine-distance RDM on shared dims at the final layer.
      Useful as input to downstream analyses or visualisation.

Usage
-----
  python 03_raw_rsa.py --stem contrastive_stubs
  python 03_raw_rsa.py --stem contrastive_stubs --n_perm 2000 --plot
  python 03_raw_rsa.py --stem contrastive_stubs --layer -1 --n_perm 500
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr, pearsonr, rankdata
from joblib import Parallel, delayed

try:
    import psutil
except ImportError:
    psutil = None


# ─────────────────────────────────────────────────────────────────────────────
# Utility — identical signatures to 03_projection.py for reuse compatibility
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
    neural_tri:  np.ndarray,   # lower-tri of neural RDM, already ranked
    model_tri:   np.ndarray,   # lower-tri of primary model RDM (ranked)
    control_tri: np.ndarray,   # lower-tri of control RDM (ranked)
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

    Returns (observed_rho, p_value, null_distribution).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    N = model_rdm.shape[0]
    idx_lower = np.tril_indices(N, k=-1)

    ctrl_tri = _lower_tri(control_rdm)

    # Observed
    model_tri = _lower_tri(model_rdm)
    obs_rho, _ = _partial_spearman(neural_tri, model_tri, ctrl_tri)

    # Pre-compute fixed quantities (invariant across all permutations)
    r_ctrl   = rankdata(ctrl_tri).astype(np.float64)
    r_neural = rankdata(neural_tri).astype(np.float64)
    ctrl_c   = r_ctrl - r_ctrl.mean()
    ctrl_cc  = np.dot(ctrl_c, ctrl_c) + 1e-12
    res_neural = r_neural - ctrl_c * (np.dot(ctrl_c, r_neural) / ctrl_cc)
    norm_rn    = np.linalg.norm(res_neural) + 1e-12

    # Determine target memory footprint and adapt batch size
    n_tril = idx_lower[0].shape[0]
    row_bytes = n_tril * 8  # float64 size per row
    if psutil is not None:
        avail_mb = psutil.virtual_memory().available / (1024 * 1024)
    else:
        avail_mb = 1024 * 1024
    target_mb = max(512, avail_mb * 0.20)

    # Use two rows (perm_tri + prank) simultaneously
    max_batch = max(1, int((target_mb * 1024 * 1024) / (row_bytes * 2)))
    max_batch = min(max_batch, n_perm)

    if n_perm > 500:
        print(f"  WARNING: n_perm={n_perm} too high for this memory budget; reducing to 500.")
        n_perm = 500

    null_rhos = np.empty(n_perm, dtype=np.float64)
    CHUNK = max_batch
    for start in range(0, n_perm, CHUNK):
        end = min(start + CHUNK, n_perm)
        batch = end - start
        bperms = np.stack([rng.permutation(N) for _ in range(batch)])
        perm_tri = model_rdm[bperms[:, idx_lower[0]],
                             bperms[:, idx_lower[1]]]

        prank = rankdata(perm_tri, axis=1).astype(np.float32)
        dot_mc = prank @ ctrl_c
        res_m = prank - ctrl_c[None, :] * (dot_mc / ctrl_cc)[:, None]

        numer = res_m @ res_neural
        denom = np.linalg.norm(res_m, axis=1) * norm_rn
        null_rhos[start:end] = numer / np.where(denom < 1e-12, 1e-12, denom)

    p_val = float((null_rhos >= obs_rho).sum() + 1) / (n_perm + 1)
    return obs_rho, p_val, null_rhos


# ─────────────────────────────────────────────────────────────────────────────
# Purity mask loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_masks(base: Path, stem: str) -> dict | None:
    """
    Load purity masks from step 02.
    Returns dict with keys ast_pure, builtin_pure, shared (each (L+1, H) bool),
    or None if the file does not exist.
    """
    path = base / f"02_{stem}_purity_masks.npz"
    if not path.exists():
        return None
    npz = np.load(path, allow_pickle=True)
    result: dict[str, np.ndarray] = {}
    for key in ("ast_pure", "builtin_pure", "shared", "interaction"):
        fkey = f"residual_{key}"
        if fkey in npz.files:
            result[key] = npz[fkey].astype(bool)   # (L+1, H)
    if not result:
        return None
    print(f"  Loaded purity masks from {path.name}")
    # Merge shared + interaction into shared for convenience
    if "interaction" in result and "shared" in result:
        result["shared_combined"] = result["shared"] | result["interaction"]
    elif "shared" in result:
        result["shared_combined"] = result["shared"]
    elif "interaction" in result:
        result["shared_combined"] = result["interaction"]
    return result


# ─────────────────────────────────────────────────────────────────────────────
# RSA on one activation matrix
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
      rho_ast_partial     float   partial Spearman controlling for builtin
      rho_builtin_partial float   partial Spearman controlling for AST
      p_ast               float   Mantel permutation p-value (AST partial)
      p_builtin           float   Mantel permutation p-value (builtin partial)
      rho_ast_raw         float   raw (non-partial) Spearman with AST RDM
      rho_builtin_raw     float   raw (non-partial) Spearman with builtin RDM
      null_ast            (n_perm,)  null distribution for AST
      null_builtin        (n_perm,)  null distribution for builtin
      rdm_neural          (N, N) float32  neural RDM
    """
    if rng is None:
        rng = np.random.default_rng(42)

    rdm_n = _cosine_rdm(act)
    rdm_a = _label_rdm(ast_labels)
    rdm_b = _label_rdm(builtin_labels)

    n_tri = _lower_tri(rdm_n)
    a_tri = _lower_tri(rdm_a)
    b_tri = _lower_tri(rdm_b)

    # Raw Spearman (no partial)
    rho_a_raw, _ = spearmanr(n_tri, a_tri)
    rho_b_raw, _ = spearmanr(n_tri, b_tri)

    # Partial Spearman
    rho_a_p, _ = _partial_spearman(n_tri, a_tri, b_tri)
    rho_b_p, _ = _partial_spearman(n_tri, b_tri, a_tri)

    # Mantel permutation tests
    obs_a, p_a, null_a = _mantel_permutation(n_tri, rdm_a, rdm_b, n_perm, rng)
    obs_b, p_b, null_b = _mantel_permutation(n_tri, rdm_b, rdm_a, n_perm, rng)

    return {
        "rho_ast_partial":      obs_a,
        "rho_builtin_partial":  obs_b,
        "p_ast":                p_a,
        "p_builtin":            p_b,
        "rho_ast_raw":          float(rho_a_raw),
        "rho_builtin_raw":      float(rho_b_raw),
        "null_ast":             null_a,
        "null_builtin":         null_b,
        "rdm_neural":           rdm_n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cross-RDM correlation
# ─────────────────────────────────────────────────────────────────────────────

def cross_rdm_correlation_one_layer(
    act:            np.ndarray,   # (N, H)  full activations at this layer
    ast_mask:       np.ndarray,   # (H,) bool  AST-pure dims
    builtin_mask:   np.ndarray,   # (H,) bool  builtin-pure dims
) -> dict:
    """
    Compute the Spearman correlation between:
      RDM built from AST-pure dims  vs  RDM built from builtin-pure dims.

    A positive rho means the two subspaces organise prompts similarly
    (structural entanglement).  Near-zero rho means independent geometry.

    Returns dict: rho, p_value, n_ast_dims, n_builtin_dims.
    """
    n_ast = int(ast_mask.sum())
    n_b   = int(builtin_mask.sum())

    if n_ast < 2 or n_b < 2:
        return {"rho": float("nan"), "p_value": float("nan"),
                "n_ast_dims": n_ast, "n_builtin_dims": n_b}

    act_ast = act[:, ast_mask]
    act_b   = act[:, builtin_mask]

    rdm_ast = _cosine_rdm(act_ast)
    rdm_b   = _cosine_rdm(act_b)

    tri_ast = _lower_tri(rdm_ast)
    tri_b   = _lower_tri(rdm_b)

    rho, pval = spearmanr(tri_ast, tri_b)
    return {
        "rho":           float(rho),
        "p_value":       float(pval),
        "n_ast_dims":    n_ast,
        "n_builtin_dims": n_b,
    }


# ─────────────────────────────────────────────────────────────────────────────
# All-layer loops
# ─────────────────────────────────────────────────────────────────────────────

def rsa_all_layers_raw(
    resid_all:      np.ndarray,   # (N, L+1, H)
    ast_labels:     list,
    builtin_labels: list,
    n_perm:         int = 1000,
    label:          str = "raw",
) -> list[dict]:
    """
    RSA at every layer on a given activation representation.
    `resid_all` may be pre-subsetted (e.g. to shared dims) before calling.

    Returns list of length L+1, each entry a stripped dict (no large arrays).
    The raw RDM at the final layer is also returned in the last entry under
    key 'rdm_neural' for downstream use.
    """
    N, L1, H = resid_all.shape
    print(f"  RSA ({label}) — {L1} layers × 2 Mantel tests "
          f"(sequential, n_perm={n_perm})...")

    def _job(l):
        rng = np.random.default_rng(42 + l)
        r   = rsa_one_layer(resid_all[:, l, :], ast_labels, builtin_labels,
                            n_perm=n_perm, rng=rng)
        return l, {
            "layer":               l,
            "rho_ast_partial":     r["rho_ast_partial"],
            "rho_builtin_partial": r["rho_builtin_partial"],
            "p_ast":               r["p_ast"],
            "p_builtin":           r["p_builtin"],
            "rho_ast_raw":         r["rho_ast_raw"],
            "rho_builtin_raw":     r["rho_builtin_raw"],
            "_null_ast":           r["null_ast"],
            "_null_builtin":       r["null_builtin"],
            "_rdm_neural":         r["rdm_neural"] if l == L1 - 1 else None,
        }

    pairs = Parallel(n_jobs=1, prefer="threads")(
        delayed(_job)(l) for l in range(L1)
    )
    return [r for _, r in sorted(pairs, key=lambda x: x[0])]


def cross_rdm_all_layers(
    resid_all:    np.ndarray,    # (N, L+1, H)
    masks:        dict,          # from _load_masks; needs ast_pure, builtin_pure
) -> list[dict]:
    """
    Cross-RDM correlation at every layer.
    Returns list of length L+1, each a dict from cross_rdm_correlation_one_layer.
    """
    N, L1, H = resid_all.shape
    print(f"  Cross-RDM — {L1} layers (sequential)...")

    def _job(l):
        ast_m = masks["ast_pure"][l] if masks["ast_pure"].ndim == 2 else masks["ast_pure"]
        b_m   = masks["builtin_pure"][l] if masks["builtin_pure"].ndim == 2 else masks["builtin_pure"]
        r = cross_rdm_correlation_one_layer(resid_all[:, l, :], ast_m, b_m)
        r["layer"] = l
        return l, r

    pairs = Parallel(n_jobs=1, prefer="threads")(
        delayed(_job)(l) for l in range(L1)
    )
    return [r for _, r in sorted(pairs, key=lambda x: x[0])]


# ─────────────────────────────────────────────────────────────────────────────
# Conclusions helper
# ─────────────────────────────────────────────────────────────────────────────

def draw_conclusions(
    rsa_raw:    list[dict],
    rsa_shared: list[dict] | None,
    cross_rdm:  list[dict] | None,
    alpha:      float = 0.05,
) -> str:
    """
    Print and return a human-readable summary of the three analyses.
    """
    lines = ["=" * 68, "03_raw_rsa  CONCLUSIONS", "=" * 68]

    # Raw RSA
    ast_rhos = [r["rho_ast_partial"]    for r in rsa_raw]
    b_rhos   = [r["rho_builtin_partial"] for r in rsa_raw]
    ast_ps   = [r["p_ast"]              for r in rsa_raw]
    b_ps     = [r["p_builtin"]          for r in rsa_raw]

    ast_sig = [i for i, p in enumerate(ast_ps) if p < alpha]
    b_sig   = [i for i, p in enumerate(b_ps)   if p < alpha]

    lines.append("RAW RSA (full H dims):")
    best_a = int(np.argmax(ast_rhos))
    best_b = int(np.argmax(b_rhos))
    lines.append(f"  AST    sig layers: {ast_sig}  best rho={ast_rhos[best_a]:.3f} @ L{best_a}")
    lines.append(f"  Builtin sig layers: {b_sig}  best rho={b_rhos[best_b]:.3f} @ L{best_b}")

    # Shared RSA
    if rsa_shared:
        sh_ast_sig = [i for i, r in enumerate(rsa_shared) if r["p_ast"] < alpha]
        sh_b_sig   = [i for i, r in enumerate(rsa_shared) if r["p_builtin"] < alpha]
        sh_both    = sorted(set(sh_ast_sig) & set(sh_b_sig))
        lines.append("\nSHARED SUBSPACE RSA:")
        lines.append(f"  AST sig in shared     : {sh_ast_sig}")
        lines.append(f"  Builtin sig in shared : {sh_b_sig}")
        if sh_both:
            lines.append(
                f"  BOTH factors significant at layers {sh_both} in shared dims "
                "-> shared subspace is a TRUE INTERACTION CIRCUIT."
            )
        elif sh_ast_sig or sh_b_sig:
            dom = "AST-dominant" if len(sh_ast_sig) > len(sh_b_sig) else "builtin-dominant"
            lines.append(f"  Only one factor significant -> shared subspace is {dom}.")
        else:
            lines.append("  Neither factor significant in shared dims.")
    else:
        lines.append("\nSHARED SUBSPACE RSA: skipped (no purity masks).")

    # Cross-RDM
    if cross_rdm:
        rhos = [r["rho"] for r in cross_rdm if not np.isnan(r["rho"])]
        if rhos:
            mean_rho = float(np.mean(rhos))
            max_rho  = float(np.max(rhos))
            lines.append("\nCROSS-RDM CORRELATION (AST-pure RDM vs builtin-pure RDM):")
            lines.append(f"  Mean rho across layers: {mean_rho:.3f}  max: {max_rho:.3f}")
            if mean_rho > 0.2:
                lines.append(
                    "  -> STRUCTURAL ENTANGLEMENT: the two pure subspaces organise "
                    "prompts in geometrically similar ways."
                )
            elif mean_rho > 0.05:
                lines.append(
                    "  -> Mild structural similarity between pure subspace geometries."
                )
            else:
                lines.append(
                    "  -> Near-zero: the two pure subspaces have independent geometry."
                    "  Clean circuit hypothesis is geometrically consistent."
                )
    else:
        lines.append("\nCROSS-RDM CORRELATION: skipped (no purity masks).")

    lines.append("=" * 68)
    conclusion = "\n".join(lines)
    print(conclusion)
    return conclusion


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

COLOURS = {
    "ast":    "#1565C0",
    "builtin":"#E65100",
    "shared": "#6A1B9A",
    "cross":  "#2E7D32",
}

CMAP_DIV = LinearSegmentedColormap.from_list(
    "divraw", ["#1565C0", "#FFFFFF", "#E65100"], N=256
)


def plot_rsa_comparison(
    rsa_raw:    list[dict],
    rsa_shared: list[dict] | None,
    alpha:      float = 0.05,
    title:      str = "RSA partial Spearman (raw vs shared subspace)",
    save_path:  Path | None = None,
) -> None:
    """
    Two-panel plot: left = raw RSA, right = shared subspace RSA.
    Each panel shows partial Spearman rho for AST and builtin across layers,
    with significance markers and null-distribution bands.
    """
    n_panels = 2 if rsa_shared else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5), sharey=True)
    if n_panels == 1:
        axes = [axes]

    L = len(rsa_raw)
    layers = np.arange(L)

    def _panel(ax, results, panel_title):
        rho_a = np.array([r["rho_ast_partial"]    for r in results])
        rho_b = np.array([r["rho_builtin_partial"] for r in results])
        p_a   = np.array([r["p_ast"]              for r in results])
        p_b   = np.array([r["p_builtin"]          for r in results])

        # Null distribution bands
        if "_null_ast" in results[0] and results[0]["_null_ast"] is not None:
            null_a_lo = np.array([np.percentile(r["_null_ast"],    5) for r in results])
            null_a_hi = np.array([np.percentile(r["_null_ast"],   95) for r in results])
            null_b_lo = np.array([np.percentile(r["_null_builtin"], 5) for r in results])
            null_b_hi = np.array([np.percentile(r["_null_builtin"],95) for r in results])
            ax.fill_between(layers, null_a_lo, null_a_hi, alpha=0.12, color=COLOURS["ast"])
            ax.fill_between(layers, null_b_lo, null_b_hi, alpha=0.12, color=COLOURS["builtin"])

        ax.plot(layers, rho_a, "o-", color=COLOURS["ast"],     lw=2, label="AST (partial)")
        ax.plot(layers, rho_b, "s-", color=COLOURS["builtin"], lw=2, label="Builtin (partial)")
        ax.axhline(0, color="k", lw=0.6, linestyle="--")

        for l in range(L):
            sig_a = "***" if p_a[l] < 0.001 else ("**" if p_a[l] < 0.01
                    else ("*" if p_a[l] < alpha else ""))
            sig_b = "***" if p_b[l] < 0.001 else ("**" if p_b[l] < 0.01
                    else ("*" if p_b[l] < alpha else ""))
            if sig_a:
                ax.text(l, rho_a[l] + 0.02, sig_a, ha="center", va="bottom",
                        fontsize=8, color=COLOURS["ast"])
            if sig_b:
                ax.text(l, rho_b[l] - 0.04, sig_b, ha="center", va="top",
                        fontsize=8, color=COLOURS["builtin"])

        ax.set_xticks(layers)
        ax.set_xlabel("Layer")
        ax.set_title(panel_title, fontsize=10)
        ax.legend(fontsize=8)
        ax.set_ylim(-0.5, 0.7)
        ax.grid(True, alpha=0.2)

    _panel(axes[0], rsa_raw, "Raw activations (all H dims)")
    if rsa_shared:
        _panel(axes[1], rsa_shared, "Shared subspace activations")

    axes[0].set_ylabel("Partial Spearman rho", fontsize=11)
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    plt.close(fig)


def plot_cross_rdm(
    cross_rdm: list[dict],
    title:     str = "Cross-RDM correlation: AST-pure vs builtin-pure geometry",
    save_path: Path | None = None,
) -> None:
    """
    Line plot of per-layer cross-RDM Spearman rho.
    Positive values indicate structural entanglement between the two pure
    subspace geometries.
    """
    layers = np.arange(len(cross_rdm))
    rhos   = np.array([r["rho"] for r in cross_rdm], dtype=float)
    ps     = np.array([r["p_value"] for r in cross_rdm], dtype=float)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(layers, rhos, "o-", color=COLOURS["cross"], lw=2,
            label="Cross-RDM Spearman rho")
    ax.fill_between(layers, 0, rhos, where=rhos > 0,
                    alpha=0.15, color=COLOURS["cross"],
                    label="Positive (structural entanglement)")
    ax.axhline(0, color="grey", lw=0.8, linestyle="--")
    ax.axhline(0.2, color="orange", lw=0.8, linestyle=":",
               alpha=0.7, label="Moderate entanglement threshold (0.2)")

    for l in layers:
        sig = "***" if ps[l] < 0.001 else ("**" if ps[l] < 0.01
              else ("*" if ps[l] < 0.05 else ""))
        if sig:
            ax.text(l, rhos[l] + 0.015, sig, ha="center", va="bottom",
                    fontsize=8, color=COLOURS["cross"])

    ax.set_xlabel("Layer")
    ax.set_ylabel("Spearman rho")
    ax.set_xticks(layers)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    plt.close(fig)


def plot_shared_rdm(
    rdm:        np.ndarray,   # (N, N)
    ast_labels: list,
    title:      str = "Shared subspace RDM (final layer)",
    save_path:  Path | None = None,
) -> None:
    """
    Visualise the shared-subspace RDM sorted by AST label.
    Block structure indicates AST-based organisation within the shared space.
    """
    order = np.argsort(ast_labels, kind="stable")
    rdm_sorted = rdm[np.ix_(order, order)]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(rdm_sorted, aspect="auto", cmap="RdYlBu_r",
                   vmin=0, vmax=float(np.percentile(rdm, 95)),
                   interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Cosine distance (0=same, 2=opposite)")

    arr_ast = np.array(ast_labels)[order]
    uniq, counts = np.unique(arr_ast, return_counts=True)
    cumcounts = np.cumsum(counts)[:-1] - 0.5
    for c in cumcounts:
        ax.axhline(c, color="white", lw=0.6, alpha=0.7)
        ax.axvline(c, color="white", lw=0.6, alpha=0.7)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Prompts (sorted by AST node)")
    ax.set_ylabel("Prompts (sorted by AST node)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Serialisation helper
# ─────────────────────────────────────────────────────────────────────────────

def _serialise_rsa(results: list[dict]) -> list[dict]:
    """Strip large numpy arrays (null distributions, RDMs) for JSON output."""
    out = []
    for r in results:
        d = {k: v for k, v in r.items() if not k.startswith("_")}
        out.append(d)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 3 (raw counterpart): RSA on raw + shared subspace activations"
    )
    parser.add_argument("--stem",      default="contrastive_stubs",
                        help="File stem used in steps 01 and 02 (default: contrastive_stubs)")
    parser.add_argument("--in_dir", "-d", default="data",
                        help="Directory containing 01_* and 02_* input files "
                             "(relative to script dir, default: data)")
    parser.add_argument("--out_dir", "-o", default=None,
                        help="Output directory for results (relative to script dir, "
                             "default: same as --in_dir)")
    parser.add_argument("--layer",     type=int, default=-1,
                        help="Layer for detailed RDM output (-1 = final layer)")
    parser.add_argument("--n_perm",    type=int, default=1000,
                        help="Number of Mantel permutations (default 1000)")
    parser.add_argument("--alpha",     type=float, default=0.05,
                        help="Significance threshold (default 0.05)")
    parser.add_argument("--plot",      action="store_true",
                        help="Generate and save plots")
    args = parser.parse_args()

    base    = Path(__file__).parent / args.in_dir
    out     = Path(__file__).parent / (args.out_dir if args.out_dir else args.in_dir)
    out.mkdir(exist_ok=True)
    img_dir = out / "images"
    img_dir.mkdir(exist_ok=True)
    stem = args.stem

    # ── Load inputs ───────────────────────────────────────────────────────────
    print(f"\n[03_raw] Loading inputs for stem='{stem}'")

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
    ast_labels     = [m["ast_node"]    for m in meta]
    builtin_labels = [m["builtin_obj"] for m in meta]
    print(f"  N={N} prompts, L+1={L1} layers, H={H}")
    print(f"  AST categories:     {len(set(ast_labels))}")
    print(f"  Builtin categories: {len(set(builtin_labels))}")

    masks = _load_masks(base, stem)
    if masks is None:
        print("  No purity masks found — shared and cross-RDM analyses will be skipped.")

    layer_idx = L1 + args.layer if args.layer < 0 else args.layer
    layer_idx = int(np.clip(layer_idx, 0, L1 - 1))
    print(f"  Detail layer: {layer_idx}")

    # ── Raw RSA ───────────────────────────────────────────────────────────────
    print(f"\n[03_raw] Running RSA on raw activations (n_perm={args.n_perm})...")
    rsa_raw = rsa_all_layers_raw(
        resid_all, ast_labels, builtin_labels,
        n_perm=args.n_perm, label="raw",
    )

    rsa_raw_path = out / f"03_raw_{stem}_rsa_raw.json"
    with open(rsa_raw_path, "w") as f:
        json.dump({
            "stem":   stem,
            "n_perm": args.n_perm,
            "alpha":  args.alpha,
            "results_per_layer": _serialise_rsa(rsa_raw),
        }, f, indent=2, default=float)
    print(f"  Saved -> {rsa_raw_path}")

    # ── Shared subspace RSA ───────────────────────────────────────────────────
    rsa_shared = None
    if masks is not None and "shared_combined" in masks:
        print(f"\n[03_raw] Running RSA on shared subspace (n_perm={args.n_perm})...")
        # Build shared-only activation tensor
        # For each layer we subset columns — shapes differ per layer, so we
        # work layer-by-layer and store in a padded array
        shared_mask_3d = masks["shared_combined"]   # (L+1, H) bool
        n_shared_per_layer = shared_mask_3d.sum(axis=1)
        print(f"  Shared dims per layer: {n_shared_per_layer.tolist()[:8]}...")
        min_shared = int(n_shared_per_layer.min())
        if min_shared < 2:
            print(f"  WARNING: some layers have < 2 shared dims — "
                  f"shared RSA may be unreliable at those layers.")

        # Parallelise shared-subspace RSA across layers
        def _shared_job(l):
            m = shared_mask_3d[l]
            if m.sum() < 2:
                return l, {
                    "layer": l,
                    "rho_ast_partial":     float("nan"),
                    "rho_builtin_partial": float("nan"),
                    "p_ast":               float("nan"),
                    "p_builtin":           float("nan"),
                    "rho_ast_raw":         float("nan"),
                    "rho_builtin_raw":     float("nan"),
                }
            act_sh = resid_all[:, l, :][:, m]
            rng_l  = np.random.default_rng(42 + l)
            r = rsa_one_layer(act_sh, ast_labels, builtin_labels,
                              n_perm=args.n_perm, rng=rng_l)
            return l, {
                "layer":               l,
                "rho_ast_partial":     r["rho_ast_partial"],
                "rho_builtin_partial": r["rho_builtin_partial"],
                "p_ast":               r["p_ast"],
                "p_builtin":           r["p_builtin"],
                "rho_ast_raw":         r["rho_ast_raw"],
                "rho_builtin_raw":     r["rho_builtin_raw"],
                "_null_ast":           r["null_ast"],
                "_null_builtin":       r["null_builtin"],
                "_rdm_neural":         r["rdm_neural"] if l == L1 - 1 else None,
            }

        print(f"  RSA (shared) — {L1} layers (sequential, n_perm={args.n_perm})...")
        sh_pairs = Parallel(n_jobs=1, prefer="threads")(
            delayed(_shared_job)(l) for l in range(L1)
        )
        rsa_shared = [r for _, r in sorted(sh_pairs, key=lambda x: x[0])]

        rsa_shared_path = out / f"03_raw_{stem}_rsa_shared.json"
        with open(rsa_shared_path, "w") as f:
            json.dump({
                "stem":   stem,
                "n_perm": args.n_perm,
                "alpha":  args.alpha,
                "results_per_layer": _serialise_rsa(rsa_shared),
            }, f, indent=2, default=float)
        print(f"  Saved -> {rsa_shared_path}")

        # Save shared RDM at detail layer
        shared_rdm_entry = rsa_shared[layer_idx]
        if shared_rdm_entry.get("_rdm_neural") is None:
            # Recompute for the requested layer (may differ from final)
            m = shared_mask_3d[layer_idx]
            act_sh_l = resid_all[:, layer_idx, :][:, m] if m.sum() >= 2 else resid_all[:, layer_idx, :]
            rdm_shared_final = _cosine_rdm(act_sh_l)
        else:
            rdm_shared_final = shared_rdm_entry["_rdm_neural"]

        rdm_path = out / f"03_raw_{stem}_rdm_shared.npy"
        np.save(rdm_path, rdm_shared_final)
        print(f"  Saved -> {rdm_path}")
    else:
        if masks is not None:
            print("  No shared dims found in masks — skipping shared RSA.")

    # ── Cross-RDM correlation ─────────────────────────────────────────────────
    cross_rdm_results = None
    if masks is not None and "ast_pure" in masks and "builtin_pure" in masks:
        print(f"\n[03_raw] Running cross-RDM correlation (AST-pure vs builtin-pure)...")
        cross_rdm_results = cross_rdm_all_layers(resid_all, masks)

        cross_path = out / f"03_raw_{stem}_cross_rdm_correlation.json"
        with open(cross_path, "w") as f:
            json.dump({
                "stem":   stem,
                "description": (
                    "Per-layer Spearman correlation between the neural RDM built "
                    "from AST-pure dims and the neural RDM built from builtin-pure "
                    "dims.  Positive rho = structural entanglement."
                ),
                "results_per_layer": cross_rdm_results,
            }, f, indent=2, default=float)
        print(f"  Saved -> {cross_path}")
    else:
        if masks is None:
            print("  No purity masks — skipping cross-RDM correlation.")

    # ── Conclusions ───────────────────────────────────────────────────────────
    print()
    conclusion_str = draw_conclusions(
        rsa_raw, rsa_shared, cross_rdm_results, alpha=args.alpha
    )

    # ── Plots ─────────────────────────────────────────────────────────────────
    if args.plot:
        print(f"\n[03_raw] Generating plots...")

        # 1. RSA layer profile: raw vs shared
        plot_rsa_comparison(
            rsa_raw, rsa_shared,
            alpha=args.alpha,
            title=f"RSA partial Spearman — raw vs shared subspace ({stem})",
            save_path=img_dir / f"03_raw_{stem}_rsa_comparison.png",
        )

        # 2. Cross-RDM correlation
        if cross_rdm_results:
            plot_cross_rdm(
                cross_rdm_results,
                title=f"Cross-RDM: AST-pure vs builtin-pure geometry ({stem})",
                save_path=img_dir / f"03_raw_{stem}_cross_rdm.png",
            )

        # 3. Shared RDM at detail layer
        if rsa_shared is not None:
            rdm_sh = np.load(out / f"03_raw_{stem}_rdm_shared.npy")
            plot_shared_rdm(
                rdm_sh, ast_labels,
                title=f"Shared subspace RDM — layer {layer_idx} ({stem})",
                save_path=img_dir / f"03_raw_{stem}_rdm_shared_L{layer_idx}.png",
            )

        print(f"  Plots saved to {img_dir}/")

    print("\n[03_raw] Done.")


if __name__ == "__main__":
    main()
