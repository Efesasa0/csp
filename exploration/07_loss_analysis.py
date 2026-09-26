"""
07_loss_analysis.py

Extension analysis using the small_40x50x50_validated_prompts.json dataset.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY THIS DATASET IS VALUABLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The contrastive stubs (steps 01-06) are PURPOSE-BUILT for interpretability —
they have controlled baselines, proxy variants, and clean factorial structure.
But they are small (~9k prompts) and synthetic.

small_40x50x50_validated_prompts.json provides:
  - 40 AST nodes × 50 builtins × 50 natural variations = 100,000 prompts
  - A PERFECTLY BALANCED factorial design (equal N per cell)
  - sequence_loss — the model's actual cross-entropy loss on each prompt

sequence_loss is the critical addition.  Steps 01-06 told us WHAT the model
represents and HOW it is structured.  sequence_loss lets us ask:

  "Does representational geometry predict model behaviour?"

  If the model cleanly separates AST and builtin representations (high VP
  purity, high RSA partial rho), does that correlate with LOWER sequence_loss
  (easier prediction)?  Or does entanglement (high shared VP) predict HARDER
  prompts?

This closes the loop from representation → function, which is the core claim
of mechanistic interpretability.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SIX ANALYSES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. FACTORIAL ANOVA ON LOSS
     loss ~ C(ast_node) * C(builtin_obj)
     The interaction term identifies (AST, builtin) PAIRS that are
     intrinsically harder than the sum of their individual difficulties.
     These are the same interaction circuits found in step 05, now validated
     by actual model behaviour rather than just representation geometry.

  2. LOSS-VP CORRELATION
     For each hidden unit, correlate sequence_loss with its VP scores
     (unique_ast, unique_builtin, shared) across all prompts.
     Units where higher unique_ast predicts lower loss are doing USEFUL AST
     processing.  Units where higher shared predicts higher loss are the
     crosstalk units hurting performance.

  3. COMPONENT LOSS PREDICTION
     Regress sequence_loss on head_attr and mlp_attr (from step 01).
     Which attention heads / MLP layers best predict how hard a prompt is?
     Overlap with ast_pure / builtin_pure masks from step 02 = functional
     confirmation of the VP analysis.

  4. HARD vs EASY PROMPT GEOMETRY
     Split prompts by sequence_loss quartile.  For high-loss (hard) and
     low-loss (easy) prompts separately, compute RSA partial Spearman rho.
     If hard prompts have LOWER rho → the model struggles precisely where
     representations are entangled.  This is the key mechanistic claim.

  5. VARIATION ROBUSTNESS
     Within each (AST, builtin) cell (50 variations), compute:
       - std(sequence_loss)   → how sensitive is loss to surface form?
       - std(activations)     → how sensitive is the representation?
     High activation variance + low loss variance = robust generalisation.
     High both = model is confused by surface variation.
     Scatter plot of (loss_std vs activation_std) per cell.

  6. CROSS-DATASET VP REPLICATION
     Compare the VP maps from this dataset to those from the contrastive stubs.
     Spearman correlation of unique_ast / unique_builtin scores per unit,
     per layer.  High correlation = findings generalise beyond the synthetic
     stubs.  This is the replication argument for a paper.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUT FILES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  <input_json>                           original validated_prompts.json
                                         (provides sequence_loss)
  01_<stem>_residual_all.npy   (N,L+1,H)
  01_<stem>_head_attr.npy      (N,L,n_heads)
  01_<stem>_mlp_attr.npy       (N,L)
  01_<stem>_meta.json          (provides ast_node, builtin_obj, variation_id)

  Optional (for analyses 2, 4, 6):
  02_<stem>_vp_residual.npz    VP scores from step 02
  02_<stem>_purity_masks.npz   purity masks from step 02

  Optional (for analysis 6 — cross-dataset replication):
  02_<ref_stem>_vp_residual.npz  VP scores from a DIFFERENT stem to compare

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FILES  (prefix 07_<stem>_)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  07_<stem>_loss_anova.json            2-way ANOVA table + cell means matrix
  07_<stem>_loss_vp_corr.npz          per-unit Spearman r(loss, VP_score)
  07_<stem>_component_loss_pred.json  per-head / per-MLP loss regression betas
  07_<stem>_hardness_rsa.json         RSA rho per loss quartile per layer
  07_<stem>_robustness.json           per-cell loss_std and activation_std
  07_<stem>_vp_replication.json       cross-dataset VP correlation per layer

Usage
-----
  # Run steps 01-02 on the large dataset first:
  python 01_extraction.py --input small_40x50x50_validated_prompts.json
  python 02_variance_partition.py --stem small_40x50x50_validated_prompts

  # Then run this:
  python 07_loss_analysis.py --stem small_40x50x50_validated_prompts --plot

  # With cross-dataset replication (comparing to contrastive_stubs VP):
  python 07_loss_analysis.py --stem small_40x50x50_validated_prompts \\
                             --ref_stem contrastive_stubs --plot
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.stats import spearmanr, pearsonr, f as f_dist, mannwhitneyu
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_losses(
    meta:       list[dict],
    input_json: Path | None = None,
) -> np.ndarray:
    """
    Load sequence_loss values aligned to the meta list.

    Tries meta["sequence_loss"] first (saved by 01_extraction.py >= this fix).
    Falls back to reading the original input JSON matched by prompt_id.

    Returns float32 array of shape (N,).  NaN where loss is missing.
    """
    losses = np.array([m.get("sequence_loss") for m in meta], dtype=object)

    # Check how many are already present
    n_present = sum(1 for v in losses if v is not None)

    if n_present < len(meta) and input_json is not None and input_json.exists():
        print(f"  {len(meta) - n_present} prompts missing sequence_loss "
              f"— merging from {input_json.name}...")
        with open(input_json, encoding="utf-8") as f:
            raw = [json.loads(line) if input_json.suffix == ".jsonl"
                   else None for line in f]
        if raw[0] is None:
            with open(input_json, encoding="utf-8") as f:
                content = f.read().strip()
            if content.startswith("["):
                raw = json.loads(content)
            else:
                raw = [json.loads(line) for line in content.splitlines() if line.strip()]

        id_to_loss = {r["prompt_id"]: r.get("sequence_loss") for r in raw}
        for i, m in enumerate(meta):
            if losses[i] is None:
                losses[i] = id_to_loss.get(m.get("prompt_id"))

    n_found = sum(1 for v in losses if v is not None)
    print(f"  sequence_loss: {n_found}/{len(meta)} prompts have values")

    return np.array([float(v) if v is not None else np.nan
                     for v in losses], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 1 — Factorial ANOVA on sequence_loss
# ─────────────────────────────────────────────────────────────────────────────

def loss_factorial_anova(
    losses:         np.ndarray,   # (N,)
    ast_labels:     list,
    builtin_labels: list,
) -> dict:
    """
    2-way factorial ANOVA: loss ~ C(ast_node) * C(builtin_obj).

    Uses SS decomposition from cell means (Type I / balanced design).
    With 50 balanced observations per cell this is an exact balanced ANOVA.

    Returns:
      main_ast      dict(F, p, eta2)
      main_builtin  dict(F, p, eta2)
      interaction   dict(F, p, eta2)
      cell_means    (n_ast, n_b) mean loss per cell
      cell_stds     (n_ast, n_b) std loss per cell
      cell_counts   (n_ast, n_b) n per cell
      ast_names     list
      builtin_names list
      conclusion    str
    """
    valid = ~np.isnan(losses)
    losses_v = losses[valid]
    ast_v    = [ast_labels[i]     for i in range(len(losses)) if valid[i]]
    b_v      = [builtin_labels[i] for i in range(len(losses)) if valid[i]]

    ast_names     = sorted(set(ast_v))
    builtin_names = sorted(set(b_v))
    A, B = len(ast_names), len(builtin_names)

    ai = {n: i for i, n in enumerate(ast_names)}
    bi = {n: i for i, n in enumerate(builtin_names)}

    # Build cell arrays
    cells: list[list[list[float]]] = [[[] for _ in range(B)] for _ in range(A)]
    for loss, a, b in zip(losses_v, ast_v, b_v):
        cells[ai[a]][bi[b]].append(float(loss))

    # Cell means
    cell_means  = np.full((A, B), np.nan)
    cell_stds   = np.full((A, B), np.nan)
    cell_counts = np.zeros((A, B), dtype=int)
    for i in range(A):
        for j in range(B):
            if cells[i][j]:
                cell_means[i, j]  = np.mean(cells[i][j])
                cell_stds[i, j]   = np.std(cells[i][j])
                cell_counts[i, j] = len(cells[i][j])

    grand_mean = float(np.nanmean(cell_means))
    n_per_cell = float(np.nanmean(cell_counts))   # assume ~balanced

    # Marginal means
    row_means = np.nanmean(cell_means, axis=1)    # (A,)
    col_means = np.nanmean(cell_means, axis=0)    # (B,)

    # SS
    ss_a   = B * n_per_cell * np.nansum((row_means - grand_mean) ** 2)
    ss_b   = A * n_per_cell * np.nansum((col_means - grand_mean) ** 2)
    ss_ab  = n_per_cell * np.nansum(
        (cell_means - row_means[:, None] - col_means[None, :] + grand_mean) ** 2
    )
    # Within-cell SS (error)
    ss_err = sum(
        (x - cell_means[i, j]) ** 2
        for i in range(A) for j in range(B) for x in cells[i][j]
        if not np.isnan(cell_means[i, j])
    )

    df_a   = A - 1
    df_b   = B - 1
    df_ab  = df_a * df_b
    df_err = max(1, sum(max(0, len(cells[i][j]) - 1)
                        for i in range(A) for j in range(B)))
    ss_tot = ss_a + ss_b + ss_ab + ss_err

    ms_err = ss_err / df_err

    def _stat(ss, df):
        ms  = ss / max(df, 1)
        F   = ms / max(ms_err, 1e-12)
        p   = float(1 - f_dist.cdf(F, df, df_err))
        eta2 = ss / max(ss_tot, 1e-12)
        return {"F": float(F), "p": float(p), "eta2": float(eta2),
                "df_effect": int(df), "df_error": int(df_err)}

    res_a  = _stat(ss_a,  df_a)
    res_b  = _stat(ss_b,  df_b)
    res_ab = _stat(ss_ab, df_ab)

    # Find hardest and easiest cells
    hard_idx  = np.unravel_index(np.nanargmax(cell_means), cell_means.shape)
    easy_idx  = np.unravel_index(np.nanargmin(cell_means), cell_means.shape)

    lines = ["=" * 65, "FACTORIAL ANOVA: sequence_loss ~ AST × builtin", "=" * 65]
    for label, res in [("Main: AST node",        res_a),
                       ("Main: builtin",          res_b),
                       ("Interaction AST×builtin", res_ab)]:
        sig = "***" if res["p"] < 0.001 else ("**" if res["p"] < 0.01
              else ("*" if res["p"] < 0.05 else "n.s."))
        lines.append(f"  {label:<30s} F={res['F']:7.2f}  "
                     f"p={res['p']:.4f} {sig}  eta2={res['eta2']:.4f}")

    if res_ab["p"] < 0.05:
        lines.append(
            f"\n  Significant interaction: certain (AST, builtin) PAIRS are harder"
            f"\n  than predicted by individual factor difficulties."
            f"\n  Hardest cell: {ast_names[hard_idx[0]]} x {builtin_names[hard_idx[1]]}"
            f"  (loss={cell_means[hard_idx]:.4f})"
            f"\n  Easiest cell: {ast_names[easy_idx[0]]} x {builtin_names[easy_idx[1]]}"
            f"  (loss={cell_means[easy_idx]:.4f})"
        )
    else:
        lines.append("\n  No significant interaction: loss is explained by AST and "
                     "builtin main effects only.")
    lines.append("=" * 65)
    conclusion = "\n".join(lines)
    print(conclusion)

    return {
        "main_ast":     res_a,
        "main_builtin": res_b,
        "interaction":  res_ab,
        "cell_means":   cell_means.tolist(),
        "cell_stds":    cell_stds.tolist(),
        "cell_counts":  cell_counts.tolist(),
        "ast_names":    ast_names,
        "builtin_names": builtin_names,
        "grand_mean":   grand_mean,
        "conclusion":   conclusion,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 2 — Loss-VP correlation
# ─────────────────────────────────────────────────────────────────────────────

def loss_vp_correlation(
    losses:    np.ndarray,     # (N,)
    vp_npz:    dict,           # loaded 02_*_vp_residual.npz  keys: unique_ast, etc.
    layer:     int = -1,
) -> dict:
    """
    For each hidden unit at `layer`, compute Spearman r(loss, VP_score).

    VP_score per prompt is resid_all[:, layer, unit].  But we don't have
    per-prompt VP scores (VP is aggregate).  Instead we compute:
      For unique_ast: how much does unit u's activation predict loss?
      Split: high-unique_ast units vs low-unique_ast units.
      Report: mean |beta| from Ridge regression of loss on [high_ast_units].

    Actually: correlate the per-unit unique_ast score with the per-unit
    Spearman r(activation, loss) — this is a unit-level analysis:
      1. For each unit u: compute r_u = Spearman(resid[:, layer, u], loss)
      2. Correlate r_u with unique_ast[layer, u] across all units
         Positive → units with high unique_ast have activations that predict loss

    Returns per-unit correlation arrays and the meta-correlation (r_u vs VP score).
    """
    L1 = list(vp_npz.values())[0].shape[0]
    l  = L1 + layer if layer < 0 else int(np.clip(layer, 0, L1 - 1))

    vp_keys = ["unique_ast", "unique_builtin", "shared", "unexplained"]
    result  = {}

    for vk in vp_keys:
        if vk not in vp_npz:
            continue
        result[f"vp_{vk}_at_layer"] = vp_npz[vk][l]    # (H,)

    return result


def loss_activation_correlation(
    losses:    np.ndarray,     # (N,)
    resid_all: np.ndarray,     # (N, L+1, H)
    vp_npz:    dict | None,
    masks:     dict | None,
    layer:     int = -1,
) -> dict:
    """
    Full loss-activation correlation analysis at `layer`.

    Step 1: For each unit u, Spearman r(resid[:, l, u], losses) → r_unit (H,)
    Step 2: Correlate r_unit with VP scores (unique_ast, unique_builtin, shared)
            → meta-correlation: do high-unique_ast units predict loss more?
    Step 3: Compare mean |r_unit| for units classified as ast_pure /
            builtin_pure / shared / noise by purity masks.

    Returns:
      r_unit          (H,)  Spearman r per unit
      meta_corr       dict  r(r_unit, VP_score) for each VP component
      group_mean_r    dict  mean |r_unit| per purity class
    """
    L1 = resid_all.shape[1]
    l  = L1 + layer if layer < 0 else int(np.clip(layer, 0, L1 - 1))
    H  = resid_all.shape[2]

    act = resid_all[:, l, :]    # (N, H)
    valid = ~np.isnan(losses)
    act_v = act[valid]
    loss_v = losses[valid]

    print(f"  Computing per-unit Spearman r(activation, loss) for {H} units...")
    r_unit = np.zeros(H, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for u in range(H):
            r, _ = spearmanr(act_v[:, u], loss_v)
            r_unit[u] = r if np.isfinite(r) else 0.0
        if (u + 1) % 200 == 0:
            print(f"  {u+1}/{H}", end="\r")
    print()

    # Meta-correlation: r_unit vs VP scores
    meta_corr = {}
    if vp_npz is not None:
        for vk in ["unique_ast", "unique_builtin", "shared", "unexplained"]:
            if vk not in vp_npz:
                continue
            vp_scores = vp_npz[vk][l]       # (H,)
            r_meta, p_meta = spearmanr(r_unit, vp_scores)
            meta_corr[vk] = {"r": float(r_meta), "p": float(p_meta)}
            print(f"  r(r_unit, {vk:<15s}) = {r_meta:+.4f}  p={p_meta:.4f}")

    # Per-purity-class mean |r|
    group_mean_r = {}
    if masks is not None:
        for cls_name in ("ast_pure", "builtin_pure", "shared", "noise"):
            m = masks.get(cls_name)
            if m is None:
                continue
            m_l = m[l] if m.ndim == 2 else m
            if cls_name == "shared" and "interaction" in masks:
                m_l = m_l | (masks["interaction"][l] if masks["interaction"].ndim == 2
                              else masks["interaction"])
            if m_l.sum() > 0:
                group_mean_r[cls_name] = float(np.abs(r_unit[m_l]).mean())

    return {
        "r_unit":       r_unit,
        "meta_corr":    meta_corr,
        "group_mean_r": group_mean_r,
        "layer":        l,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 3 — Component loss prediction
# ─────────────────────────────────────────────────────────────────────────────

def component_loss_prediction(
    losses:    np.ndarray,    # (N,)
    head_attr: np.ndarray,    # (N, L, n_heads)
    mlp_attr:  np.ndarray,    # (N, L)
    masks:     dict | None,
    n_cv:      int = 5,
) -> dict:
    """
    Ridge regression: predict sequence_loss from head_attr and mlp_attr.

    Per-head importance: fit a univariate Ridge model for each head,
    report |beta| and R² from 5-fold CV.

    Also reports: which heads are in the ast_pure / builtin_pure circuit
    (from step 02 head VP masks) and whether loss-predictive heads overlap.

    Returns:
      head_r2       (L, n_heads)  cross-validated R² per head
      mlp_r2        (L,)          cross-validated R² per MLP layer
      head_beta     (L, n_heads)  Ridge coefficient magnitude
      best_head     (layer, head_idx, R²)
      overlap       dict  fraction of top-N loss heads in each purity class
    """
    valid = ~np.isnan(losses)
    y     = losses[valid]
    N, L, n_heads = head_attr.shape

    scaler = StandardScaler()
    y_sc   = scaler.fit_transform(y.reshape(-1, 1)).ravel()

    head_r2   = np.zeros((L, n_heads))
    head_beta = np.zeros((L, n_heads))
    mlp_r2    = np.zeros(L)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for l in range(L):
            # Per-head
            for h in range(n_heads):
                x = head_attr[valid, l, h:h+1]
                x_sc = scaler.fit_transform(x)
                clf  = Ridge(alpha=1.0)
                r2s  = cross_val_score(clf, x_sc, y_sc, cv=n_cv,
                                       scoring="r2")
                head_r2[l, h] = max(0.0, float(r2s.mean()))
                clf.fit(x_sc, y_sc)
                head_beta[l, h] = float(abs(clf.coef_[0]))

            # MLP layer
            x = mlp_attr[valid, l].reshape(-1, 1)
            x_sc = scaler.fit_transform(x)
            r2s  = cross_val_score(Ridge(alpha=1.0), x_sc, y_sc, cv=n_cv,
                                   scoring="r2")
            mlp_r2[l] = max(0.0, float(r2s.mean()))

            print(f"  Component loss prediction layer {l+1}/{L}", end="\r")
    print()

    best_flat = np.argmax(head_r2)
    best_l, best_h = np.unravel_index(best_flat, head_r2.shape)
    print(f"  Best head for loss prediction: layer={best_l} head={best_h} "
          f"R²={head_r2[best_l, best_h]:.4f}")

    # Overlap with purity masks (heads don't have purity masks in step 02
    # the same way — head VP is over (L, n_heads) not (L+1, H).
    # We report which layers' top heads also have high head VP unique_ast.)
    return {
        "head_r2":   head_r2,
        "head_beta": head_beta,
        "mlp_r2":    mlp_r2,
        "best_layer": int(best_l),
        "best_head":  int(best_h),
        "best_r2":    float(head_r2[best_l, best_h]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 3b — Logit attribution vs sequence_loss
# ─────────────────────────────────────────────────────────────────────────────

def logit_attribution_loss_correlation(
    losses:    np.ndarray,    # (N,)
    head_attr: np.ndarray,    # (N, L, n_heads)
    mlp_attr:  np.ndarray,    # (N, L)
    n_bins:    int = 4,
) -> dict:
    """
    Since sequence_cross_entropy loss is built from final logits, compare
    the reconstructed logit attribution to observed loss.

    Uses the per-head and per-MLP contributions from 01_extraction as a
    proxy for the final predicted logit (target token score).

    Returns:
      logit_proxy        (N,)          head_attr.sum + mlp_attr.sum
      pearson            dict(r,p)
      spearman           dict(r,p)
      layer_corr         (L,)         spearman per layer
      quartile_edges     (n_bins+1,)  on logit_proxy
      quartile_loss_mean (n_bins,)    mean loss per logit-bin
    """
    valid = ~np.isnan(losses)
    y = losses[valid]

    if head_attr is None or mlp_attr is None:
        raise ValueError("head_attr and mlp_attr are required for logit attribution analysis")

    proxy = head_attr.sum(axis=(1, 2)) + mlp_attr.sum(axis=1)
    proxy_v = proxy[valid]

    # Correlation across all prompts
    p_r, p_p = pearsonr(proxy_v, y)
    s_r, s_p = spearmanr(proxy_v, y)

    # Per-layer correlation
    L = head_attr.shape[1]
    layer_corr = []
    for l in range(L):
        layer_score = head_attr[:, l, :].sum(axis=1) + mlp_attr[:, l]
        r, _ = spearmanr(layer_score[valid], y)
        layer_corr.append(float(r if np.isfinite(r) else 0.0))

    # Logit-bin means
    edges = np.percentile(proxy_v, np.linspace(0, 100, n_bins + 1))
    bin_idx = np.digitize(proxy_v, edges, right=False) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    quartile_loss_mean = [float(np.mean(y[bin_idx == qi])) if (bin_idx == qi).any() else float('nan')
                          for qi in range(n_bins)]

    return {
        "logit_proxy":        proxy.tolist(),
        "pearson":            {"r": float(p_r), "p": float(p_p)},
        "spearman":           {"r": float(s_r), "p": float(s_p)},
        "layer_spearman":     layer_corr,
        "quartile_edges":     edges.tolist(),
        "quartile_loss_mean": quartile_loss_mean,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 4 — Hard vs easy prompt geometry (RSA by quartile)
# ─────────────────────────────────────────────────────────────────────────────

def hardness_rsa(
    losses:         np.ndarray,    # (N,)
    resid_all:      np.ndarray,    # (N, L+1, H)
    ast_labels:     list,
    builtin_labels: list,
    n_quartiles:    int = 4,
) -> dict:
    """
    Split prompts into loss quartiles.  For each quartile, at each layer,
    compute the RSA partial Spearman correlation (neural RDM vs AST label RDM,
    controlling for builtin, and vice versa).

    The hypothesis: hard prompts (high loss) have MORE ENTANGLED representations,
    leading to lower partial rho for both factors.

    Returns:
      quartile_edges   (n_quartiles+1,)  loss thresholds
      rho_ast          (n_quartiles, L+1)  partial rho for AST per quartile
      rho_builtin      (n_quartiles, L+1)  partial rho for builtin per quartile
      quartile_sizes   (n_quartiles,)  n prompts per quartile
    """
    valid = ~np.isnan(losses)
    l_arr = losses[valid]
    idx_valid = np.where(valid)[0]

    edges = np.nanpercentile(losses,
                             np.linspace(0, 100, n_quartiles + 1))
    edges[0]  -= 1e-6
    edges[-1] += 1e-6

    N, L1, H = resid_all.shape
    rho_ast  = np.zeros((n_quartiles, L1))
    rho_b    = np.zeros((n_quartiles, L1))
    q_sizes  = np.zeros(n_quartiles, dtype=int)

    def _lower_tri(M):
        idx = np.tril_indices(M.shape[0], k=-1)
        return M[idx]

    def _label_rdm(labels):
        arr = np.array(labels)
        return (arr[:, None] != arr[None, :]).astype(np.float32)

    def _partial_spearman(n_tri, m_tri, c_tri):
        from scipy.stats import rankdata
        rn = rankdata(n_tri).astype(float)
        rm = rankdata(m_tri).astype(float)
        rc = rankdata(c_tri).astype(float)
        def _resid(y, x):
            x = x - x.mean()
            return y - np.dot(x, y) / (np.dot(x, x) + 1e-12) * x
        r, _ = pearsonr(_resid(rn, rc), _resid(rm, rc))
        return float(r)

    quartile_loss_mean = np.zeros(n_quartiles, dtype=np.float32)

    for qi in range(n_quartiles):
        lo, hi = edges[qi], edges[qi + 1]
        q_idx = idx_valid[(l_arr > lo) & (l_arr <= hi)]
        q_sizes[qi] = len(q_idx)

        if len(q_idx) > 0:
            quartile_loss_mean[qi] = float(np.nanmean(losses[q_idx]))
        else:
            quartile_loss_mean[qi] = float('nan')

        if len(q_idx) < 10:
            continue

        ast_q = [ast_labels[i]     for i in q_idx]
        b_q   = [builtin_labels[i] for i in q_idx]
        rdm_a = _label_rdm(ast_q)
        rdm_b = _label_rdm(b_q)
        a_tri = _lower_tri(rdm_a)
        b_tri = _lower_tri(rdm_b)

        for l in range(L1):
            act  = resid_all[q_idx, l, :]
            norms= np.linalg.norm(act, axis=1, keepdims=True).clip(1e-12)
            act_n= act / norms
            cos  = np.clip(act_n @ act_n.T, -1, 1)
            rdm_n= (1 - cos).astype(np.float32)
            n_tri= _lower_tri(rdm_n)
            if n_tri.std() < 1e-8:
                continue
            rho_ast[qi, l] = _partial_spearman(n_tri, a_tri, b_tri)
            rho_b[qi, l]   = _partial_spearman(n_tri, b_tri, a_tri)

        print(f"  Quartile {qi+1}/{n_quartiles} (n={len(q_idx)}, "
              f"loss=[{lo:.3f},{hi:.3f}]) done")

    return {
        "quartile_edges":      edges.tolist(),
        "rho_ast":             rho_ast.tolist(),
        "rho_builtin":         rho_b.tolist(),
        "quartile_sizes":      q_sizes.tolist(),
        "quartile_loss_mean":  quartile_loss_mean.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 5 — Variation robustness
# ─────────────────────────────────────────────────────────────────────────────

def variation_robustness(
    losses:         np.ndarray,    # (N,)
    resid_all:      np.ndarray,    # (N, L+1, H)
    ast_labels:     list,
    builtin_labels: list,
    variation_ids:  list,
    layer:          int = -1,
) -> dict:
    """
    Within each (AST, builtin) cell, compute:
      loss_std:       std of sequence_loss across 50 variations
      act_std:        mean std of activations across variations (mean over units)
      loss_mean:      mean loss
      n:              number of variations present

    Scatter (act_std, loss_std) per cell reveals robustness clusters:
      Bottom-left: stable activations, stable loss → robust generalisation
      Top-right:   variable activations, variable loss → sensitive to surface form
      Top-left:    variable activations, stable loss → representation explores
                   but prediction is stable (interesting!)
      Bottom-right: stable activations, variable loss → shouldn't happen much

    Returns list of per-cell dicts.
    """
    L1 = resid_all.shape[1]
    l  = L1 + layer if layer < 0 else int(np.clip(layer, 0, L1 - 1))

    act = resid_all[:, l, :]    # (N, H)

    # Group by (ast_node, builtin_obj)
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, (a, b) in enumerate(zip(ast_labels, builtin_labels)):
        groups[(a, b)].append(i)

    results = []
    for (a, b), idxs in sorted(groups.items()):
        loss_vals = losses[idxs]
        act_vals  = act[idxs]             # (n, H)
        valid_loss = ~np.isnan(loss_vals)

        results.append({
            "ast_node":    a,
            "builtin_obj": b,
            "n":           len(idxs),
            "loss_mean":   float(np.nanmean(loss_vals)),
            "loss_std":    float(np.nanstd(loss_vals[valid_loss])) if valid_loss.sum() > 1 else 0.0,
            "act_std":     float(act_vals.std(axis=0).mean()),   # mean per-unit std
            "act_norm_mean": float(np.linalg.norm(act_vals, axis=1).mean()),
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 6 — Cross-dataset VP replication
# ─────────────────────────────────────────────────────────────────────────────

def cross_dataset_vp_correlation(
    vp_main: dict,    # VP from this dataset (stem)
    vp_ref:  dict,    # VP from reference dataset (ref_stem, e.g. contrastive_stubs)
) -> dict:
    """
    Compare variance partition maps between two datasets.

    For each layer and VP component, compute Spearman r between the per-unit
    scores across both datasets.  This requires both datasets to have been
    run on the SAME model (same H), so the unit indices correspond.

    If the two VP maps are correlated:
      → the model's representational structure generalises across datasets
      → the contrastive-stub findings are not artefacts of the dataset construction

    Returns per-layer correlation per VP component.
    """
    vp_keys = ["unique_ast", "unique_builtin", "shared", "unexplained"]
    L1_main = list(vp_main.values())[0].shape[0]
    L1_ref  = list(vp_ref.values())[0].shape[0]
    L1      = min(L1_main, L1_ref)

    result: dict[str, list[dict]] = {vk: [] for vk in vp_keys}

    for l in range(L1):
        for vk in vp_keys:
            if vk not in vp_main or vk not in vp_ref:
                continue
            v1 = vp_main[vk][l]    # (H,)
            v2 = vp_ref[vk][l]     # (H,)
            H  = min(len(v1), len(v2))
            r, p = spearmanr(v1[:H], v2[:H])
            result[vk].append({"layer": l, "r": float(r), "p": float(p)})
            print(f"  VP replication layer {l} {vk:<15s} r={r:+.4f}  p={p:.4f}")

    # Summary: mean r across layers per component
    summary = {}
    for vk in vp_keys:
        if result[vk]:
            rs = [d["r"] for d in result[vk]]
            summary[vk] = {"mean_r": float(np.mean(rs)), "min_r": float(np.min(rs)),
                           "max_r": float(np.max(rs))}

    return {"per_layer": result, "summary": summary}


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

COLOURS = {
    "ast_pure":     "#1565C0",
    "builtin_pure": "#E65100",
    "shared":       "#6A1B9A",
    "noise":        "#BDBDBD",
}
CMAP_LOSS = LinearSegmentedColormap.from_list(
    "loss", ["#1B5E20", "#FFF9C4", "#B71C1C"], N=256
)


def plot_loss_heatmap(
    anova:     dict,
    save_path: Path | None = None,
) -> None:
    """
    Heatmap of mean sequence_loss per (AST, builtin) cell.
    Annotated with interaction significance.
    Rows sorted by mean loss; columns sorted by mean loss.
    Reveals which (AST, builtin) combinations are intrinsically hard.
    """
    cell_means   = np.array(anova["cell_means"])
    ast_names    = anova["ast_names"]
    builtin_names= anova["builtin_names"]

    # Sort rows and cols by mean loss
    row_order = np.argsort(np.nanmean(cell_means, axis=1))
    col_order = np.argsort(np.nanmean(cell_means, axis=0))
    cell_sorted  = cell_means[np.ix_(row_order, col_order)]
    row_names    = [ast_names[i]     for i in row_order]
    col_names    = [builtin_names[j] for j in col_order]

    fig, axes = plt.subplots(1, 2, figsize=(18, max(6, len(ast_names) * 0.3)),
                              gridspec_kw={"width_ratios": [3, 1]})

    # Main heatmap
    ax = axes[0]
    vmin = np.nanpercentile(cell_sorted, 5)
    vmax = np.nanpercentile(cell_sorted, 95)
    im = ax.imshow(cell_sorted, aspect="auto", cmap=CMAP_LOSS,
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Mean sequence_loss")
    ax.set_xticks(range(len(col_names)))
    ax.set_xticklabels(col_names, rotation=45, ha="right",
                       fontsize=max(4, 8 - len(col_names) // 10))
    ax.set_yticks(range(len(row_names)))
    ax.set_yticklabels(row_names, fontsize=max(4, 8 - len(row_names) // 10))
    ax.set_xlabel("Builtin (sorted by mean loss)")
    ax.set_ylabel("AST node (sorted by mean loss)")

    inter = anova["interaction"]
    sig   = "***" if inter["p"] < 0.001 else ("**" if inter["p"] < 0.01
            else ("*" if inter["p"] < 0.05 else "n.s."))
    ax.set_title(
        f"sequence_loss per (AST, builtin) cell\n"
        f"Interaction: F={inter['F']:.1f}  p={inter['p']:.4f} {sig}  "
        f"eta²={inter['eta2']:.4f}",
        fontsize=10
    )

    # Marginal bars: mean loss per AST node (right panel)
    ax2 = axes[1]
    row_means = np.nanmean(cell_sorted, axis=1)
    ax2.barh(range(len(row_names)), row_means,
             color=[CMAP_LOSS((v - vmin) / max(vmax - vmin, 1e-6)) for v in row_means])
    ax2.set_yticks(range(len(row_names)))
    ax2.set_yticklabels(row_names, fontsize=max(4, 8 - len(row_names) // 10))
    ax2.set_xlabel("Mean loss")
    ax2.set_title("AST marginal means", fontsize=9)
    ax2.invert_yaxis()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    plt.close(fig)


def plot_loss_distribution_by_factor(
    losses:     np.ndarray,
    ast_labels: list,
    b_labels:   list,
    save_path:  Path | None = None,
) -> None:
    """
    Two violin plots: loss distribution per AST node (left) and per builtin (right).
    Sorted by median loss.  Shows how much individual factors explain difficulty.
    """
    ast_names = sorted(set(ast_labels),
                        key=lambda n: np.nanmedian([losses[i] for i, a
                                                    in enumerate(ast_labels) if a == n]))
    b_names   = sorted(set(b_labels),
                        key=lambda n: np.nanmedian([losses[i] for i, b
                                                    in enumerate(b_labels) if b == n]))

    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(b_names) * 0.35 + 2),
                                             max(5, len(ast_names) * 0.35)))

    for ax, names, labels, colour, xlabel in [
        (axes[0], ast_names,  ast_labels, COLOURS["ast_pure"],     "AST node"),
        (axes[1], b_names,    b_labels,   COLOURS["builtin_pure"], "Builtin"),
    ]:
        data = [[losses[i] for i, l in enumerate(labels) if l == n and not np.isnan(losses[i])]
                for n in names]
        data = [d if d else [np.nan] for d in data]
        parts = ax.violinplot(data, positions=range(len(names)),
                              showmedians=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(colour)
            pc.set_alpha(0.6)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right",
                           fontsize=max(5, 8 - len(names) // 8))
        ax.set_ylabel("sequence_loss")
        ax.set_xlabel(xlabel)
        ax.set_title(f"Loss by {xlabel} (sorted by median)", fontsize=10)
        ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle("sequence_loss distribution — main effects of AST and builtin",
                 fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    plt.close(fig)


def plot_loss_vp_scatter(
    r_unit:    np.ndarray,    # (H,)  per-unit Spearman r(activation, loss)
    vp_npz:    dict,
    masks:     dict | None,
    layer:     int,
    save_path: Path | None = None,
) -> None:
    """
    Scatter: x = unique_ast score per unit, y = Spearman r(activation, loss).
    Colour by purity class.

    Units with high unique_ast AND high |r| are doing AST-specific processing
    that is also functionally relevant (predicts model difficulty).

    Units with high shared VP AND high |r| are the crosstalk units hurting performance.
    """
    ua = vp_npz.get("unique_ast",     np.zeros_like(r_unit))[layer]
    ub = vp_npz.get("unique_builtin", np.zeros_like(r_unit))[layer]
    sh = vp_npz.get("shared",         np.zeros_like(r_unit))[layer]
    H  = len(r_unit)

    if masks is not None:
        ap = masks.get("ast_pure",     np.zeros((1, H), bool))[layer]
        bp = masks.get("builtin_pure", np.zeros((1, H), bool))[layer]
        sm = masks.get("shared",       np.zeros((1, H), bool))[layer]
        if "interaction" in masks:
            sm = sm | masks["interaction"][layer]
        no = ~(ap | bp | sm)
    else:
        ap = ua > 0.1;  bp = ub > 0.1
        sm = sh > 0.1;  no = ~(ap | bp | sm)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, vp_score, vp_label, xlabel in [
        (axes[0], ua, "unique_ast",     "Unique AST variance (R²)"),
        (axes[1], sh, "shared",         "Shared variance (R²)"),
    ]:
        for mask, colour, label, marker in [
            (ap, COLOURS["ast_pure"],     "AST-pure", "o"),
            (bp, COLOURS["builtin_pure"], "Builtin-pure", "s"),
            (sm, COLOURS["shared"],       "Shared/interaction", "^"),
            (no, COLOURS["noise"],        "Noise", "."),
        ]:
            if mask.sum() == 0:
                continue
            ax.scatter(vp_score[mask], r_unit[mask], s=8, c=colour,
                       alpha=0.5, marker=marker, label=f"{label} (n={mask.sum()})")

        ax.axhline(0, color="grey", lw=0.8, linestyle="--")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Spearman r(activation, loss)")
        ax.set_title(f"{vp_label} vs loss-predictiveness (layer {layer})", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.15)

    fig.suptitle(
        "Which units predict loss?  Colour = purity class from VP analysis\n"
        "High shared + high |r| = crosstalk units hurting model performance",
        fontsize=11
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    plt.close(fig)


def plot_component_loss_importance(
    comp:      dict,    # output of component_loss_prediction
    save_path: Path | None = None,
) -> None:
    """
    Heatmap of per-head R² for predicting sequence_loss,
    plus a line plot of per-MLP-layer R².
    Shows which components are most relevant to prediction difficulty.
    """
    head_r2 = np.array(comp["head_r2"])   # (L, n_heads)
    mlp_r2  = np.array(comp["mlp_r2"])    # (L,)
    L, n_heads = head_r2.shape
    layers = np.arange(L)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    im = ax.imshow(head_r2.T, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=head_r2.max().clip(1e-4),
                   interpolation="nearest")
    plt.colorbar(im, ax=ax, label="CV R² (predicting loss)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Attention head")
    ax.set_xticks(range(L))
    ax.set_title(f"Per-head loss prediction R²\n"
                 f"Best: L{comp['best_layer']} H{comp['best_head']} "
                 f"R²={comp['best_r2']:.4f}", fontsize=10)

    ax = axes[1]
    ax.bar(layers, mlp_r2, color="#6A1B9A", alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("CV R² (predicting loss)")
    ax.set_title("Per-MLP-layer loss prediction R²", fontsize=10)
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle("Which components predict sequence_loss?\n"
                 "High R² = component output correlates with prediction difficulty",
                 fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    plt.close(fig)


def plot_logit_loss_correlation(
    logit_corr: dict,
    losses: np.ndarray,
    save_path: Path | None = None,
) -> None:
    """
    Plot the proxy logit score vs sequence_loss and layerwise correlation.
    """
    proxy = np.array(logit_corr["logit_proxy"])
    rho = np.array(logit_corr.get("layer_spearman", []))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1, 1]})

    axes[0].scatter(proxy, losses, s=8, alpha=0.5, c="#1565C0")
    axes[0].set_xlabel("Logit proxy (head+mlp)")
    axes[0].set_ylabel("sequence_loss")
    axes[0].set_title(
        f"Logit proxy vs loss: Pearson r={logit_corr['pearson']['r']:+.3f}, "
        f"Spearman r={logit_corr['spearman']['r']:+.3f}"
    )

    if rho.size > 0:
        axes[1].plot(np.arange(len(rho)), rho, "o-", color="#E65100")
        axes[1].axhline(0, color="gray", lw=0.8, ls="--")
        axes[1].set_xlabel("Layer")
        axes[1].set_ylabel("Spearman r(layer logit proxy, loss)")
        axes[1].set_title("Layerwise logit-loss correlation")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    plt.close(fig)


def plot_hardness_rsa(
    hardness:   dict,
    save_path:  Path | None = None,
) -> None:
    """
    Line plot: RSA partial rho per layer for each loss quartile.
    Hard (high-loss) prompts should have lower rho if entanglement ↔ difficulty.

    Two panels: AST partial rho and builtin partial rho.
    Quartile 1 = easiest; quartile 4 = hardest.
    """
    rho_ast = np.array(hardness["rho_ast"])      # (Q, L+1)
    rho_b   = np.array(hardness["rho_builtin"])  # (Q, L+1)
    q_edges = hardness["quartile_edges"]
    q_sizes = hardness["quartile_sizes"]
    Q, L1   = rho_ast.shape
    layers  = np.arange(L1)

    cmap   = plt.colormaps["RdYlGn_r"]
    quartile_loss_mean = np.array(hardness.get("quartile_loss_mean", [np.nan] * Q))
    labels = [f"Q{q+1}: loss=[{q_edges[q]:.2f},{q_edges[q+1]:.2f}] "
              f"mean={quartile_loss_mean[q]:.3f} n={q_sizes[q]}"
              for q in range(Q)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, rho, target in [(axes[0], rho_ast, "AST"), (axes[1], rho_b, "Builtin")]:
        for qi in range(Q):
            colour = cmap(qi / max(Q - 1, 1))
            ax.plot(layers, rho[qi], "o-", color=colour, lw=2,
                    label=labels[qi], markersize=4)
        ax.axhline(0, color="grey", lw=0.8, linestyle="--")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Partial Spearman rho")
        ax.set_title(f"{target} RSA by loss quartile", fontsize=10)
        ax.legend(fontsize=7, loc="lower right")
        ax.set_xticks(layers)
        ax.grid(True, alpha=0.2)
        ax.set_ylim(-0.4, 0.7)

    fig.suptitle(
        "Hard vs easy prompt geometry: does entanglement predict difficulty?\n"
        "Green = easy (low loss), Red = hard (high loss).  "
        "Convergence = loss independent of geometry",
        fontsize=11
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    plt.close(fig)


def plot_robustness_scatter(
    robustness: list[dict],
    save_path:  Path | None = None,
) -> None:
    """
    Scatter: x = within-cell activation std, y = within-cell loss std.
    Each point = one (AST, builtin) cell.  Colour = mean loss (easy=green, hard=red).
    Quadrant annotations show what each region means.
    """
    xs      = np.array([r["act_std"]   for r in robustness])
    ys      = np.array([r["loss_std"]  for r in robustness])
    colours = np.array([r["loss_mean"] for r in robustness])
    labels  = [f"{r['ast_node']} x {r['builtin_obj']}" for r in robustness]

    vmin = np.percentile(colours, 5)
    vmax = np.percentile(colours, 95)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=np.median(colours), vmax=vmax)

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(xs, ys, c=colours, cmap=CMAP_LOSS, norm=norm,
                    s=40, alpha=0.75, linewidths=0)
    plt.colorbar(sc, ax=ax, label="Mean sequence_loss (green=easy, red=hard)")

    xmed = np.median(xs)
    ymed = np.median(ys)
    ax.axvline(xmed, color="grey", lw=0.8, linestyle="--", alpha=0.6)
    ax.axhline(ymed, color="grey", lw=0.8, linestyle="--", alpha=0.6)

    # Quadrant labels
    kw = dict(ha="center", fontsize=8, style="italic",
               bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
    ax.text(np.percentile(xs, 25), np.percentile(ys, 75),
            "Variable representation\nVariable loss\n(surface-sensitive)", **kw)
    ax.text(np.percentile(xs, 75), np.percentile(ys, 25),
            "Stable representation\nStable loss\n(robust)", **kw)
    ax.text(np.percentile(xs, 25), np.percentile(ys, 25),
            "Low variation\n(consistent cells)", **kw)
    ax.text(np.percentile(xs, 75), np.percentile(ys, 75),
            "Stable representation\nVariable loss\n(interesting!)", **kw)

    # Annotate extreme points
    top_n = 5
    top_idx = np.argsort(ys)[-top_n:]
    for i in top_idx:
        ax.annotate(labels[i], (xs[i], ys[i]), fontsize=5, alpha=0.7,
                    xytext=(4, 4), textcoords="offset points")

    r, p = spearmanr(xs, ys)
    sig  = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
    ax.set_xlabel("Within-cell activation std (representation variability)")
    ax.set_ylabel("Within-cell loss std (prediction variability)")
    ax.set_title(
        f"Variation robustness per (AST, builtin) cell\n"
        f"Spearman r={r:.3f}  p={p:.4f} {sig}",
        fontsize=11
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    plt.close(fig)


def plot_vp_replication(
    replication: dict,
    main_stem:   str,
    ref_stem:    str,
    save_path:   Path | None = None,
) -> None:
    """
    Line plot of Spearman r(VP_main, VP_ref) per layer for each VP component.
    High r → findings replicate across datasets.
    Low r → dataset-specific artefact.
    """
    per_layer = replication["per_layer"]
    summary   = replication.get("summary", {})

    vp_keys = [k for k in per_layer if per_layer[k]]
    colours  = {"unique_ast":     COLOURS["ast_pure"],
                "unique_builtin": COLOURS["builtin_pure"],
                "shared":         COLOURS["shared"],
                "unexplained":    COLOURS["noise"]}

    fig, ax = plt.subplots(figsize=(10, 5))
    for vk in vp_keys:
        data   = per_layer[vk]
        layers = [d["layer"] for d in data]
        rs     = [d["r"]     for d in data]
        ps     = [d["p"]     for d in data]
        mean_r = summary.get(vk, {}).get("mean_r", 0)
        ax.plot(layers, rs, "o-", color=colours.get(vk, "#607D8B"),
                lw=2, label=f"{vk}  (mean r={mean_r:.3f})")
        # Significance markers
        for l, r, p in zip(layers, rs, ps):
            if p < 0.05:
                ax.text(l, r + 0.02, "*" if p >= 0.01 else "**" if p >= 0.001 else "***",
                        ha="center", fontsize=8, color=colours.get(vk, "#607D8B"))

    ax.axhline(0, color="grey", lw=0.8, linestyle="--")
    ax.axhline(0.3, color="grey", lw=0.5, linestyle=":", alpha=0.5,
               label="Moderate agreement (r=0.3)")
    ax.axhline(0.7, color="grey", lw=0.5, linestyle=":", alpha=0.5,
               label="Strong agreement (r=0.7)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Spearman r (VP scores across datasets)")
    ax.set_title(
        f"VP replication: '{main_stem}' vs '{ref_stem}'\n"
        "High r = same units identified as ast_pure/builtin_pure/shared in both datasets",
        fontsize=10
    )
    ax.legend(fontsize=8)
    ax.set_ylim(-0.3, 1.05)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 7: Loss-based analysis on validated_prompts dataset"
    )
    parser.add_argument("--stem",       default="small_40x50x50_validated_prompts",
                        help="Stem of the 01_/02_ files for this dataset")
    parser.add_argument("--in_dir", "-d", default="data",
                        help="Directory containing pipeline output files (default: data/)")
    parser.add_argument("--out_dir", "-o", default=None,
                        help="Output directory for results (default: same as --in_dir)")
    parser.add_argument("--input_json", default=None,
                        help="Path to original .json file (for sequence_loss fallback)")
    parser.add_argument("--ref_stem",   default=None,
                        help="Stem of a second dataset for VP replication (e.g. contrastive_stubs)")
    parser.add_argument("--layer",      type=int, default=-1,
                        help="Layer for detailed unit-level analysis (-1 = final)")
    parser.add_argument("--n_quartiles",type=int, default=4)
    parser.add_argument("--no_rsa",     action="store_true",
                        help="Skip hardness RSA (slow for large N)")
    parser.add_argument("--no_comp",    action="store_true",
                        help="Skip component loss prediction")
    parser.add_argument("--plot",       action="store_true")
    args = parser.parse_args()

    base    = Path(__file__).parent / args.in_dir
    out     = Path(__file__).parent / (args.out_dir if args.out_dir else args.in_dir)
    out.mkdir(exist_ok=True)
    img_dir = out / "images"
    img_dir.mkdir(exist_ok=True)
    stem = args.stem

    # ── Load ─────────────────────────────────────────────────────────────────
    print(f"\n[07] Loading inputs for stem='{stem}'")

    resid_all  = np.load(base / f"01_{stem}_residual_all.npy")
    N, L1, H   = resid_all.shape
    with open(base / f"01_{stem}_meta.json") as f:
        meta = json.load(f)

    ast_labels     = [m["ast_node"]    for m in meta]
    builtin_labels = [m["builtin_obj"] for m in meta]
    variation_ids  = [m.get("variation_id", 0) for m in meta]
    print(f"  residual_all: {resid_all.shape}")

    # Load sequence_loss
    input_json = Path(args.input_json) if args.input_json else None
    losses = load_losses(meta, input_json)
    n_valid_loss = int((~np.isnan(losses)).sum())

    if n_valid_loss == 0:
        print("ERROR: No sequence_loss values found.")
        print("  Pass --input_json path/to/validated_prompts.json")
        return

    # Optional VP / masks
    vp_path    = base / f"02_{stem}_vp_residual.npz"
    masks_path = base / f"02_{stem}_purity_masks.npz"

    vp_npz  = dict(np.load(vp_path,    allow_pickle=True)) if vp_path.exists()    else None
    masks_npz = np.load(masks_path, allow_pickle=True)     if masks_path.exists() else None
    masks   = None
    if masks_npz is not None:
        masks = {k[len("residual_"):]: masks_npz[k].astype(bool)
                 for k in masks_npz.files if k.startswith("residual_")}

    head_attr_path = base / f"01_{stem}_head_attr.npy"
    mlp_attr_path  = base / f"01_{stem}_mlp_attr.npy"
    head_attr = np.load(head_attr_path) if head_attr_path.exists() else None
    mlp_attr  = np.load(mlp_attr_path)  if mlp_attr_path.exists()  else None

    layer_idx = L1 + args.layer if args.layer < 0 else args.layer
    layer_idx = int(np.clip(layer_idx, 0, L1 - 1))
    print(f"  Analysis layer: {layer_idx}")
    print(f"  AST categories: {len(set(ast_labels))}  "
          f"Builtin categories: {len(set(builtin_labels))}")

    # ── Analysis 1: Factorial ANOVA on loss ──────────────────────────────────
    print(f"\n[07] Analysis 1: Factorial ANOVA on sequence_loss...")
    anova = loss_factorial_anova(losses, ast_labels, builtin_labels)
    with open(out / f"07_{stem}_loss_anova.json", "w") as f:
        json.dump({k: v for k, v in anova.items() if k != "conclusion"},
                  f, indent=2, default=float)
    print(f"  Saved 07_{stem}_loss_anova.json")

    # ── Analysis 2: Loss-VP correlation ──────────────────────────────────────
    loss_act_corr = None
    if vp_npz is not None:
        print(f"\n[07] Analysis 2: Loss-activation correlation (layer {layer_idx})...")
        loss_act_corr = loss_activation_correlation(
            losses, resid_all, vp_npz, masks, layer=layer_idx
        )
        np.savez_compressed(
            out / f"07_{stem}_loss_vp_corr.npz",
            r_unit=loss_act_corr["r_unit"],
        )
        with open(out / f"07_{stem}_loss_vp_corr_meta.json", "w") as f:
            json.dump({"meta_corr":    loss_act_corr["meta_corr"],
                       "group_mean_r": loss_act_corr["group_mean_r"],
                       "layer":        layer_idx}, f, indent=2)
        print(f"  Saved loss-VP correlation outputs.")
    else:
        print(f"\n[07] Skipping Analysis 2 (no VP file — run 02_variance_partition.py first)")

    # ── Analysis 3: Component loss prediction ────────────────────────────────
    comp = None
    if not args.no_comp and head_attr is not None and mlp_attr is not None:
        print(f"\n[07] Analysis 3: Component loss prediction (Ridge CV)...")
        comp = component_loss_prediction(losses, head_attr, mlp_attr, masks)
        with open(out / f"07_{stem}_component_loss_pred.json", "w") as f:
            json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                       for k, v in comp.items()}, f, indent=2)
        print(f"  Saved 07_{stem}_component_loss_pred.json")
    else:
        if head_attr is None:
            print(f"\n[07] Skipping Analysis 3 (no head_attr — run 01_extraction.py first)")
        else:
            print(f"\n[07] Skipping Analysis 3 (--no_comp set)")

    # ── Analysis 3b: Logit attribution vs sequence_loss ───────────────────────
    logit_corr = None
    if head_attr is not None and mlp_attr is not None:
        print(f"\n[07] Analysis 3b: Logit attribution vs sequence_loss...")
        logit_corr = logit_attribution_loss_correlation(losses, head_attr, mlp_attr)
        with open(out / f"07_{stem}_logit_loss_corr.json", "w") as f:
            json.dump(logit_corr, f, indent=2)
        print(f"  Saved 07_{stem}_logit_loss_corr.json")
    else:
        print(f"\n[07] Skipping Analysis 3b (requires head_attr and mlp_attr)")

    # ── Analysis 4: Hard vs easy geometry ────────────────────────────────────
    hardness = None
    if not args.no_rsa:
        print(f"\n[07] Analysis 4: Hard vs easy RSA ({args.n_quartiles} quartiles)...")
        hardness = hardness_rsa(
            losses, resid_all, ast_labels, builtin_labels, args.n_quartiles
        )
        with open(out / f"07_{stem}_hardness_rsa.json", "w") as f:
            json.dump(hardness, f, indent=2)
        print(f"  Saved 07_{stem}_hardness_rsa.json")
    else:
        print(f"\n[07] Skipping Analysis 4 (--no_rsa set)")

    # ── Analysis 5: Variation robustness ─────────────────────────────────────
    print(f"\n[07] Analysis 5: Variation robustness (per cell)...")
    robustness = variation_robustness(
        losses, resid_all, ast_labels, builtin_labels, variation_ids, layer=layer_idx
    )
    with open(out / f"07_{stem}_robustness.json", "w") as f:
        json.dump(robustness, f, indent=2)
    print(f"  Saved 07_{stem}_robustness.json ({len(robustness)} cells)")

    # ── Analysis 6: Cross-dataset VP replication ──────────────────────────────
    replication = None
    if args.ref_stem and vp_npz is not None:
        ref_vp_path = base / f"02_{args.ref_stem}_vp_residual.npz"
        if ref_vp_path.exists():
            print(f"\n[07] Analysis 6: VP replication vs '{args.ref_stem}'...")
            ref_vp = dict(np.load(ref_vp_path, allow_pickle=True))
            replication = cross_dataset_vp_correlation(vp_npz, ref_vp)
            with open(out / f"07_{stem}_vp_replication.json", "w") as f:
                json.dump(replication, f, indent=2)
            print(f"  Saved 07_{stem}_vp_replication.json")
        else:
            print(f"\n[07] Skipping Analysis 6: ref VP file not found ({ref_vp_path.name})")
    elif args.ref_stem:
        print(f"\n[07] Skipping Analysis 6: no VP for main stem (run 02 first)")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if args.plot:
        print(f"\n[07] Generating plots...")

        plot_loss_heatmap(
            anova,
            save_path=img_dir / f"07_{stem}_loss_heatmap.png",
        )
        plot_loss_distribution_by_factor(
            losses, ast_labels, builtin_labels,
            save_path=img_dir / f"07_{stem}_loss_distributions.png",
        )

        if loss_act_corr is not None and vp_npz is not None:
            plot_loss_vp_scatter(
                loss_act_corr["r_unit"], vp_npz, masks, layer=layer_idx,
                save_path=img_dir / f"07_{stem}_loss_vp_scatter.png",
            )

        if comp is not None:
            plot_component_loss_importance(
                comp,
                save_path=img_dir / f"07_{stem}_component_loss_importance.png",
            )

        if logit_corr is not None:
            plot_logit_loss_correlation(
                logit_corr, losses,
                save_path=img_dir / f"07_{stem}_logit_loss_correlation.png",
            )

        if hardness is not None:
            plot_hardness_rsa(
                hardness,
                save_path=img_dir / f"07_{stem}_hardness_rsa.png",
            )

        plot_robustness_scatter(
            robustness,
            save_path=img_dir / f"07_{stem}_robustness_scatter.png",
        )

        if replication is not None:
            plot_vp_replication(
                replication, main_stem=stem, ref_stem=args.ref_stem,
                save_path=img_dir / f"07_{stem}_vp_replication.png",
            )

        print(f"  All plots saved to {out}/")

    print("\n[07] Done.")


if __name__ == "__main__":
    main()
