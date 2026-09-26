"""
05_probing.py

Step 5 of the AST x builtin mechanistic interpretability pipeline.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY WE DO THIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Steps 02-04 are correlational: they show which units covary with AST/builtin
labels and whether representations look additive.  Step 05 asks two sharper
questions:

  QUESTION 1 — INTERACTION CIRCUITS (Factorial ANOVA on probe accuracy)
  -----------------------------------------------------------------------
  If AST and builtin are encoded in separate subspaces, then:
    - The AST-pure subspace should predict AST node well but NOT builtin
    - The builtin-pure subspace should predict builtin well but NOT AST
    - The SHARED subspace should predict BOTH — it is the interaction circuit

  We test this with a 2-way factorial design:
    Factor A: subspace  {raw, ast_pure, builtin_pure, shared}
    Factor B: target    {ast_node, builtin_obj}
    Response: probe accuracy (k-fold cross-validated LogisticRegression)

  The INTERACTION TERM of the ANOVA is the key result:
    - Significant interaction → the shared subspace benefits one target more
      than the other → evidence of asymmetric interaction circuits
    - Non-significant → subspaces contribute equally to both targets →
      representations are fully entangled (no clean modular structure)

  Cross-probing:
    Train a probe on ast_pure dims to predict BUILTIN (should fail if clean).
    Train a probe on builtin_pure dims to predict AST (should fail if clean).
    Residual accuracy above chance = information leaks across subspaces.

  QUESTION 2 — CONCEPT PERMEATION (layer-by-layer concept tracing)
  -----------------------------------------------------------------
  For a chosen concept (e.g. AST node "If", or builtin "len"), track how
  strongly that concept's representation is present at every layer of the
  residual stream — for any prompt in the dataset.

  Method:
    concept_dir[l] = mean over all baseline stubs for that concept, at layer l
    permeation[prompt, l] = cos_sim(resid_all[prompt, l], concept_dir[l])

  The permeation curve reveals:
    - Flat and high  → concept is broadcast across all depths (stable encoding)
    - Rises sharply  → concept crystallises at a specific layer
    - Peak then drop → concept is transiently computed, then transformed
    - Low everywhere → this prompt does not activate the concept direction

  Positive vs negative control:
    Prompts sharing the concept (positive) vs prompts that do not (negative)
    are plotted as distribution bands per layer.  The AUROC at each layer
    quantifies how separable the concept is at that depth.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW IT IS COMPUTED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Linear probe:
  LogisticRegression(C=1.0, max_iter=1000, multi_class="multinomial")
  StratifiedKFold(n_splits=5) — each fold is one ANOVA replicate
  Accuracy = mean over folds; per-fold accuracy used for ANOVA

Factorial ANOVA:
  Implemented via statsmodels OLS formula interface:
    accuracy ~ C(subspace) * C(target)
  This gives:
    Main effect A (subspace): does subspace type change accuracy?
    Main effect B (target):   does target identity change accuracy?
    Interaction A*B:           does the effect of subspace depend on target?
  p-values from F-distribution; effect sizes as partial eta-squared.

Concept direction:
  Computed from baseline stubs (variant_type == "baseline_ast" or
  "baseline_builtin") grouped by concept label.  Normalised to unit norm
  per layer so that cosine similarity is well-defined.

AUROC per layer:
  Binary classification: positive = prompts with this concept,
  negative = all other prompts.  sklearn.metrics.roc_auc_score.
  AUROC = 0.5 → chance; AUROC = 1.0 → perfectly separable at this layer.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUT FILES  (from steps 01 and 02)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  01_<stem>_residual_all.npy     float32  (N, L+1, H)
  01_<stem>_meta.json            prompt metadata
  02_<stem>_purity_masks.npz     boolean masks (optional but recommended)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FILES  (prefix 05_<stem>_)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  05_<stem>_probe_accuracy.npz       accuracy per (subspace, target, layer)
  05_<stem>_probe_foldscores.npz     per-fold accuracy for ANOVA input
  05_<stem>_anova_results.json       ANOVA table (F, p, eta-squared)
  05_<stem>_cross_probe.json         cross-subspace probe accuracy
  05_<stem>_concept_permeation.npz   per-prompt permeation curves
  05_<stem>_concept_auroc.npz        AUROC per layer for each concept

Usage
-----
  python 05_probing.py --stem contrastive_stubs --plot
  python 05_probing.py --stem contrastive_stubs --concept If --concept_type ast
  python 05_probing.py --stem contrastive_stubs --concept len --concept_type builtin
  python 05_probing.py --stem contrastive_stubs --all_concepts --plot
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
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.stats import f_oneway


# ─────────────────────────────────────────────────────────────────────────────
# Subspace masking helpers
# ─────────────────────────────────────────────────────────────────────────────

SUBSPACES = ["raw", "ast_pure", "builtin_pure", "shared"]

def _load_masks(base: Path, stem: str) -> dict | None:
    """Load purity masks from step 02. Returns None if not available."""
    path = base / f"02_{stem}_purity_masks.npz"
    if not path.exists():
        return None
    npz = np.load(path, allow_pickle=True)
    result = {}
    for key in ("ast_pure", "builtin_pure", "shared", "interaction", "noise"):
        fkey = f"residual_{key}"
        if fkey in npz.files:
            result[key] = npz[fkey].astype(bool)   # (L+1, H)
    return result if result else None


def _select_dims(act: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Select columns where mask is True. Returns act unchanged if mask is None."""
    if mask is None or mask.sum() == 0:
        return act
    return act[:, mask]


def _apply_subspace(
    resid_all: np.ndarray,   # (N, L+1, H)
    layer:     int,
    masks:     dict | None,
    subspace:  str,          # "raw" | "ast_pure" | "builtin_pure" | "shared"
) -> np.ndarray:
    """
    Extract activations at `layer` restricted to `subspace`.
    Returns (N, D) where D <= H.
    """
    act = resid_all[:, layer, :]   # (N, H)
    if subspace == "raw" or masks is None:
        return act
    mask = masks.get(subspace)
    if subspace == "shared":
        # combined shared + interaction dims
        mask = masks.get("shared", np.zeros(act.shape[1], bool))
        if mask.ndim == 2:
            mask = mask[layer]
        if "interaction" in masks:
            imask = masks["interaction"]
            if imask.ndim == 2:
                imask = imask[layer]
            mask = mask | imask
    elif mask is not None and mask.ndim == 2:
        mask = mask[layer]
    else:
        return act
    if mask.sum() == 0:
        return act   # fallback to raw if mask is empty at this layer
    return act[:, mask]


# ─────────────────────────────────────────────────────────────────────────────
# Linear probe
# ─────────────────────────────────────────────────────────────────────────────

def train_probe_cv(
    X: np.ndarray,       # (N, D)  activations
    y: np.ndarray,       # (N,)    integer class labels
    n_splits: int = 5,
    C:        float = 1.0,
    seed:     int = 42,
) -> tuple[float, np.ndarray]:
    """
    Train a logistic regression probe with stratified k-fold CV.

    Returns (mean_accuracy, per_fold_accuracies).
    Uses liblinear solver for speed; multinomial for multi-class.
    Falls back to dummy accuracy if too few samples per class.
    """
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2 or counts.min() < n_splits:
        return float(1.0 / max(len(classes), 1)), np.full(n_splits, 1.0 / max(len(classes), 1))

    skf    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for train_idx, test_idx in skf.split(X, y):
            clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs",
                                     multi_class="multinomial", n_jobs=1)
            clf.fit(X[train_idx], y[train_idx])
            scores.append(accuracy_score(y[test_idx], clf.predict(X[test_idx])))
    return float(np.mean(scores)), np.array(scores)


def probe_one_layer(
    resid_all:      np.ndarray,   # (N, L+1, H)
    layer:          int,
    ast_labels:     list,
    builtin_labels: list,
    masks:          dict | None,
    n_splits:       int = 5,
) -> dict:
    """
    Run probes for all (subspace, target) combinations at one layer.

    Returns dict:
      accuracy[subspace][target]       float  mean CV accuracy
      fold_scores[subspace][target]    (n_splits,)  per-fold accuracy
    """
    le_ast = LabelEncoder().fit(ast_labels)
    le_b   = LabelEncoder().fit(builtin_labels)
    y_ast  = le_ast.transform(ast_labels)
    y_b    = le_b.transform(builtin_labels)

    accuracy    = {s: {} for s in SUBSPACES}
    fold_scores = {s: {} for s in SUBSPACES}

    for subspace in SUBSPACES:
        X = _apply_subspace(resid_all, layer, masks, subspace)
        if X.shape[1] == 0:
            for target, y in [("ast", y_ast), ("builtin", y_b)]:
                n_cls = len(np.unique(y))
                accuracy[subspace][target]    = 1.0 / n_cls
                fold_scores[subspace][target] = np.full(n_splits, 1.0 / n_cls)
            continue
        for target, y in [("ast", y_ast), ("builtin", y_b)]:
            acc, folds = train_probe_cv(X, y, n_splits)
            accuracy[subspace][target]    = acc
            fold_scores[subspace][target] = folds

    return {"accuracy": accuracy, "fold_scores": fold_scores}


def probe_all_layers(
    resid_all:      np.ndarray,
    ast_labels:     list,
    builtin_labels: list,
    masks:          dict | None,
    n_splits:       int = 5,
) -> dict:
    """
    Run probe_one_layer for every layer.

    Returns:
      accuracy_mat[subspace][target]     np.ndarray (L+1,)
      fold_mat[subspace][target]         np.ndarray (L+1, n_splits)
    """
    N, L1, H = resid_all.shape
    accuracy_mat = {s: {"ast": np.zeros(L1), "builtin": np.zeros(L1)}
                    for s in SUBSPACES}
    fold_mat     = {s: {"ast": np.zeros((L1, n_splits)),
                        "builtin": np.zeros((L1, n_splits))}
                    for s in SUBSPACES}

    for l in range(L1):
        print(f"  Probing layer {l+1}/{L1}...", end="\r")
        res = probe_one_layer(resid_all, l, ast_labels, builtin_labels, masks, n_splits)
        for s in SUBSPACES:
            for t in ("ast", "builtin"):
                accuracy_mat[s][t][l] = res["accuracy"][s][t]
                fold_mat[s][t][l]     = res["fold_scores"][s][t]
    print()
    return {"accuracy": accuracy_mat, "fold_scores": fold_mat}


# ─────────────────────────────────────────────────────────────────────────────
# Cross-probing (leakage test)
# ─────────────────────────────────────────────────────────────────────────────

def cross_probe(
    resid_all:      np.ndarray,
    ast_labels:     list,
    builtin_labels: list,
    masks:          dict | None,
    layer:          int = -1,
    n_splits:       int = 5,
) -> dict:
    """
    Cross-probe analysis at a chosen layer:
      Train on ast_pure dims  → predict BUILTIN  (should be chance if clean)
      Train on builtin_pure dims → predict AST   (should be chance if clean)

    Residual accuracy above chance = information leaks between subspaces.

    Returns dict with:
      ast_dims_predict_builtin   float  mean accuracy
      builtin_dims_predict_ast   float  mean accuracy
      ast_chance                 float  1/n_builtin_classes
      builtin_chance             float  1/n_ast_classes
      leak_ast_to_builtin        float  accuracy - chance
      leak_builtin_to_ast        float  accuracy - chance
    """
    L1 = resid_all.shape[1]
    l  = L1 + layer if layer < 0 else layer

    le_ast = LabelEncoder().fit(ast_labels)
    le_b   = LabelEncoder().fit(builtin_labels)
    y_ast  = le_ast.transform(ast_labels)
    y_b    = le_b.transform(builtin_labels)

    chance_ast = 1.0 / len(le_ast.classes_)
    chance_b   = 1.0 / len(le_b.classes_)

    def _probe(X, y):
        acc, _ = train_probe_cv(X, y, n_splits)
        return acc

    X_ast = _apply_subspace(resid_all, l, masks, "ast_pure")
    X_b   = _apply_subspace(resid_all, l, masks, "builtin_pure")

    acc_ast_predicts_b   = _probe(X_ast, y_b)
    acc_b_predicts_ast   = _probe(X_b,  y_ast)

    return {
        "layer":                   l,
        "ast_dims_predict_builtin": acc_ast_predicts_b,
        "builtin_dims_predict_ast": acc_b_predicts_ast,
        "ast_chance":              chance_b,
        "builtin_chance":          chance_ast,
        "leak_ast_to_builtin":     acc_ast_predicts_b  - chance_b,
        "leak_builtin_to_ast":     acc_b_predicts_ast  - chance_ast,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Factorial ANOVA
# ─────────────────────────────────────────────────────────────────────────────

def run_factorial_anova(
    fold_mat: dict,
    layer:    int = -1,
    subspaces: list[str] | None = None,
) -> dict:
    """
    2-way factorial ANOVA:  accuracy ~ C(subspace) * C(target)

    Each (subspace, target) combination has n_splits observations (CV folds).
    This gives us within-cell variance to test effects against.

    Returns dict with:
      main_subspace    dict(F, p, eta2)  main effect of subspace
      main_target      dict(F, p, eta2)  main effect of target
      interaction      dict(F, p, eta2)  interaction A*B
      conclusion       str

    Uses scipy.stats.f_oneway for marginal effects and manual SS decomposition
    for the interaction (statsmodels is optional).
    """
    if subspaces is None:
        subspaces = SUBSPACES

    L1 = list(list(fold_mat["fold_scores"].values())[0].values())[0].shape[0]
    l  = L1 + layer if layer < 0 else int(np.clip(layer, 0, L1-1))

    # Build cell arrays: shape (n_subspaces, n_targets, n_splits)
    targets = ["ast", "builtin"]
    cells   = np.zeros((len(subspaces), len(targets),
                        fold_mat["fold_scores"][subspaces[0]]["ast"].shape[1]))
    for si, s in enumerate(subspaces):
        for ti, t in enumerate(targets):
            cells[si, ti, :] = fold_mat["fold_scores"][s][t][l]

    n_s, n_t, n_rep = cells.shape
    grand_mean = cells.mean()

    # SS decomposition
    ss_total = ((cells - grand_mean) ** 2).sum()
    ss_s  = n_t * n_rep * ((cells.mean(axis=(1,2)) - grand_mean) ** 2).sum()
    ss_t  = n_s * n_rep * ((cells.mean(axis=(0,2)) - grand_mean) ** 2).sum()
    cell_means = cells.mean(axis=2)                    # (n_s, n_t)
    row_means  = cells.mean(axis=(1,2), keepdims=False)  # (n_s,)
    col_means  = cells.mean(axis=(0,2), keepdims=False)  # (n_t,)
    ss_st = n_rep * (((cell_means - row_means[:, None]
                        - col_means[None, :] + grand_mean) ** 2).sum())
    ss_err = ss_total - ss_s - ss_t - ss_st

    df_s   = n_s - 1
    df_t   = n_t - 1
    df_st  = df_s * df_t
    df_err = n_s * n_t * (n_rep - 1)

    ms_s   = ss_s   / max(df_s,   1)
    ms_t   = ss_t   / max(df_t,   1)
    ms_st  = ss_st  / max(df_st,  1)
    ms_err = ss_err / max(df_err, 1)

    from scipy.stats import f as f_dist

    def _f_p_eta(ss_effect, ms_effect, df_effect):
        F   = ms_effect / max(ms_err, 1e-12)
        p   = float(1 - f_dist.cdf(F, df_effect, df_err))
        eta2 = ss_effect / max(ss_total, 1e-12)
        return {"F": float(F), "p": float(p), "eta2": float(eta2),
                "df_effect": int(df_effect), "df_error": int(df_err)}

    res_s  = _f_p_eta(ss_s,  ms_s,  df_s)
    res_t  = _f_p_eta(ss_t,  ms_t,  df_t)
    res_st = _f_p_eta(ss_st, ms_st, df_st)

    # Cell means for interpretation
    cell_means = {s: {t: float(cells[si, ti].mean())
                      for ti, t in enumerate(targets)}
                  for si, s in enumerate(subspaces)}

    # Interpretation
    lines = [f"Factorial ANOVA — layer {l}"]
    for label, res in [("Main: subspace", res_s), ("Main: target", res_t),
                        ("Interaction subspace*target", res_st)]:
        sig = "***" if res["p"] < 0.001 else ("**" if res["p"] < 0.01
              else ("*" if res["p"] < 0.05 else "n.s."))
        lines.append(f"  {label:<35s} F={res['F']:.2f}  p={res['p']:.4f} {sig}  "
                     f"eta2={res['eta2']:.3f}")

    if res_st["p"] < 0.05:
        # Find which subspace benefits more from which target
        shared_ast = cell_means.get("shared", {}).get("ast", 0)
        shared_b   = cell_means.get("shared", {}).get("builtin", 0)
        if shared_ast > shared_b:
            lines.append("  INTERACTION: shared dims predict AST better than builtin "
                         "-> AST-dominant interaction circuit")
        else:
            lines.append("  INTERACTION: shared dims predict builtin better than AST "
                         "-> builtin-dominant interaction circuit")
    else:
        lines.append("  No significant interaction -> subspaces affect both targets equally")

    conclusion = "\n".join(lines)
    print(conclusion)

    return {
        "layer":        l,
        "main_subspace": res_s,
        "main_target":   res_t,
        "interaction":   res_st,
        "cell_means":    cell_means,
        "conclusion":    conclusion,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Concept permeation
# ─────────────────────────────────────────────────────────────────────────────

def build_concept_directions(
    resid_all:    np.ndarray,   # (N, L+1, H)
    meta:         list[dict],
    concept_name: str,          # e.g. "If", "len"
    concept_type: str,          # "ast" | "builtin"
) -> np.ndarray | None:
    """
    Compute the mean activation direction for a concept at each layer,
    using only the baseline stubs for that concept.

    baseline_ast stubs → concept_type == "ast"
    baseline_builtin stubs → concept_type == "builtin"

    Returns (L+1, H) unit-normalised concept direction per layer,
    or None if no baseline stubs are found.
    """
    vtype = "baseline_ast" if concept_type == "ast" else "baseline_builtin"
    label_key = "ast_node" if concept_type == "ast" else "builtin_obj"

    idxs = [i for i, m in enumerate(meta)
             if m.get("variant_type") == vtype and m.get(label_key) == concept_name]

    if not idxs:
        return None

    mean_vec = resid_all[idxs].mean(axis=0)    # (L+1, H)
    norms    = np.linalg.norm(mean_vec, axis=-1, keepdims=True).clip(1e-12)
    return (mean_vec / norms).astype(np.float32)


def concept_permeation_trace(
    resid_all:     np.ndarray,   # (N, L+1, H)
    meta:          list[dict],
    concept_name:  str,
    concept_type:  str,          # "ast" | "builtin"
    concept_dir:   np.ndarray,   # (L+1, H)  from build_concept_directions
) -> dict:
    """
    For every prompt in the dataset, compute the cosine similarity to
    `concept_dir` at each layer.

    Also computes per-layer AUROC (positive = prompts WITH the concept,
    negative = all others).

    Returns dict:
      sim_positive   (n_pos, L+1)  cos_sim for prompts WITH the concept
      sim_negative   (n_neg, L+1)  cos_sim for prompts WITHOUT the concept
      auroc          (L+1,)        AUROC per layer
      positive_ids   list of prompt_ids in the positive set
      concept_norm   (L+1,)        L2 norm of concept direction per layer
                                    (shows where the concept "lives" strongest)
    """
    label_key = "ast_node" if concept_type == "ast" else "builtin_obj"

    pos_idx = [i for i, m in enumerate(meta) if m.get(label_key) == concept_name]
    neg_idx = [i for i, m in enumerate(meta) if m.get(label_key) != concept_name]

    if not pos_idx:
        return {}

    # Cosine similarity: act (N, L+1, H) · dir (L+1, H) -> (N, L+1)
    # Normalise act per layer
    norms = np.linalg.norm(resid_all, axis=-1, keepdims=True).clip(1e-12)  # (N,L+1,1)
    act_n = resid_all / norms                                                # (N, L+1, H)
    # concept_dir already unit-normalised per layer
    sim   = (act_n * concept_dir[None, :, :]).sum(axis=-1)                  # (N, L+1)
    sim   = np.clip(sim, -1.0, 1.0).astype(np.float32)

    sim_pos = sim[pos_idx]  # (n_pos, L+1)
    sim_neg = sim[neg_idx]  # (n_neg, L+1)

    # AUROC per layer
    all_idx    = pos_idx + neg_idx
    y_true     = np.array([1]*len(pos_idx) + [0]*len(neg_idx))
    auroc      = np.zeros(resid_all.shape[1])
    for l in range(resid_all.shape[1]):
        scores_l = sim[all_idx, l]
        try:
            auroc[l] = roc_auc_score(y_true, scores_l)
        except Exception:
            auroc[l] = 0.5

    # Raw concept direction norms (before normalisation) — shows signal strength
    mean_vec = resid_all[pos_idx].mean(axis=0)      # (L+1, H)
    concept_norm = np.linalg.norm(mean_vec, axis=-1) # (L+1,)

    return {
        "sim_positive":  sim_pos,
        "sim_negative":  sim_neg,
        "auroc":         auroc.astype(np.float32),
        "positive_ids":  [meta[i].get("prompt_id", i) for i in pos_idx],
        "concept_norm":  concept_norm.astype(np.float32),
    }


def all_concept_permeation(
    resid_all: np.ndarray,
    meta:      list[dict],
) -> tuple[dict, dict]:
    """
    Run concept_permeation_trace for every distinct AST node and builtin.
    Returns (ast_permeation, builtin_permeation) dicts keyed by concept name.
    """
    ast_concepts     = sorted({m["ast_node"]    for m in meta})
    builtin_concepts = sorted({m["builtin_obj"] for m in meta})

    ast_perm = {}
    for c in ast_concepts:
        cdir = build_concept_directions(resid_all, meta, c, "ast")
        if cdir is not None:
            ast_perm[c] = concept_permeation_trace(resid_all, meta, c, "ast", cdir)
        print(f"  AST permeation: {c:<20s} "
              f"AUROC_final={ast_perm[c]['auroc'][-1]:.3f}" if c in ast_perm else
              f"  AST permeation: {c} — no baseline stubs")

    builtin_perm = {}
    for c in builtin_concepts:
        cdir = build_concept_directions(resid_all, meta, c, "builtin")
        if cdir is not None:
            builtin_perm[c] = concept_permeation_trace(resid_all, meta, c, "builtin", cdir)
        print(f"  Builtin permeation: {c:<20s} "
              f"AUROC_final={builtin_perm[c]['auroc'][-1]:.3f}" if c in builtin_perm else
              f"  Builtin permeation: {c} — no baseline stubs")

    return ast_perm, builtin_perm


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

COLOURS = {
    "raw":          "#607D8B",
    "ast_pure":     "#1565C0",
    "builtin_pure": "#E65100",
    "shared":       "#6A1B9A",
}
CMAP_DIV = LinearSegmentedColormap.from_list(
    "divprobe", ["#1565C0", "#FFFFFF", "#E65100"], N=256
)


def plot_probe_accuracy_by_layer(
    probe_results: dict,
    title:         str = "Linear probe accuracy by layer",
    save_path:     Path | None = None,
) -> None:
    """
    Two-panel line plot: left = predict AST node, right = predict builtin.
    Each subspace is one line.  Shows how decodability evolves across depth.
    A well-separated model has ast_pure high in left panel but low in right,
    and builtin_pure high in right but low in left.
    """
    acc = probe_results["accuracy"]
    L1  = len(acc["raw"]["ast"])
    layers = np.arange(L1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, target, tlabel in [
        (axes[0], "ast",     "Predict AST node"),
        (axes[1], "builtin", "Predict builtin"),
    ]:
        for subspace in SUBSPACES:
            vals = acc[subspace][target]
            ax.plot(layers, vals, "o-", lw=2, markersize=4,
                    color=COLOURS[subspace], label=subspace)
        ax.set_xlabel("Layer")
        ax.set_ylim(0, 1.05)
        ax.set_title(tlabel, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.set_xticks(layers)

    axes[0].set_ylabel("CV accuracy")
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    plt.close(fig)


def plot_anova_interaction(
    probe_results: dict,
    layer_results: list[dict],   # one dict per layer from run_factorial_anova
    save_path:     Path | None = None,
) -> None:
    """
    Two-panel ANOVA summary:
      Left  — Interaction F-statistic across layers with significance bands
      Right — Cell means at the best-interaction layer (2x4 grid:
               rows=targets, cols=subspaces), showing which cell "pops"
    """
    layers  = np.arange(len(layer_results))
    F_inter = np.array([r["interaction"]["F"] for r in layer_results])
    p_inter = np.array([r["interaction"]["p"] for r in layer_results])
    best_l  = int(np.argmax(F_inter))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: F-statistic per layer
    ax = axes[0]
    ax.plot(layers, F_inter, "o-", color="#6A1B9A", lw=2)
    ax.axhline(3.84, color="red", lw=1, linestyle="--", label="F critical (p=0.05, df=1)")
    for l in layers:
        sig = "***" if p_inter[l] < 0.001 else ("**" if p_inter[l] < 0.01
               else ("*" if p_inter[l] < 0.05 else ""))
        if sig:
            ax.text(l, F_inter[l] + 0.2, sig, ha="center", fontsize=9,
                    color="#6A1B9A")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Interaction F-statistic")
    ax.set_title("ANOVA interaction (subspace × target) by layer", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.2)

    # Panel 2: cell means heatmap at best layer
    ax = axes[1]
    cell = layer_results[best_l]["cell_means"]
    targets   = ["ast", "builtin"]
    mat = np.array([[cell[s][t] for s in SUBSPACES] for t in targets])

    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Probe accuracy")
    ax.set_xticks(range(len(SUBSPACES)))
    ax.set_xticklabels(SUBSPACES, fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Predict AST", "Predict builtin"], fontsize=9)
    for i in range(2):
        for j in range(len(SUBSPACES)):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=10)
    ax.set_title(f"Cell means — layer {best_l} (strongest interaction)", fontsize=10)

    fig.suptitle("Factorial ANOVA: does the shared subspace specialise for one factor?",
                 fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    plt.close(fig)


def plot_cross_probe(
    cross_results_by_layer: list[dict],
    save_path:              Path | None = None,
) -> None:
    """
    Line plot of cross-probe accuracy across layers (information leakage).
    Chance level is plotted as a dashed reference.
    Gap between accuracy and chance = how much each subspace leaks.
    """
    layers = np.arange(len(cross_results_by_layer))
    ab     = np.array([r["ast_dims_predict_builtin"] for r in cross_results_by_layer])
    ba     = np.array([r["builtin_dims_predict_ast"] for r in cross_results_by_layer])
    ch_ab  = cross_results_by_layer[0]["ast_chance"]
    ch_ba  = cross_results_by_layer[0]["builtin_chance"]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(layers, ab, "o-", color=COLOURS["ast_pure"],     lw=2,
            label="AST dims → predict builtin")
    ax.plot(layers, ba, "s-", color=COLOURS["builtin_pure"], lw=2,
            label="Builtin dims → predict AST")
    ax.axhline(ch_ab, color=COLOURS["ast_pure"],     lw=1, linestyle="--",
               alpha=0.6, label=f"Chance (builtin): {ch_ab:.3f}")
    ax.axhline(ch_ba, color=COLOURS["builtin_pure"], lw=1, linestyle=":",
               alpha=0.6, label=f"Chance (AST): {ch_ba:.3f}")
    ax.fill_between(layers, ch_ab, ab, alpha=0.15, color=COLOURS["ast_pure"])
    ax.fill_between(layers, ch_ba, ba, alpha=0.15, color=COLOURS["builtin_pure"])

    ax.set_xlabel("Layer")
    ax.set_ylabel("Cross-probe accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Information leakage between subspaces\n"
                 "Accuracy above chance = concepts are NOT cleanly separated", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    plt.close(fig)


def plot_concept_permeation(
    perm:         dict,           # output of concept_permeation_trace
    concept_name: str,
    concept_type: str,
    save_path:    Path | None = None,
) -> None:
    """
    Three-panel concept permeation plot for a single concept:
      Panel 1 — Mean cos_sim per layer: positive (concept present) vs
                negative (concept absent).  Shaded bands = ±1 std.
      Panel 2 — AUROC per layer.  Shows at which layer the concept is most
                separable.  AUROC=1 → perfectly decodable; 0.5 → chance.
      Panel 3 — Concept direction norm per layer.  Shows where the concept
                activates most strongly in absolute terms.
    """
    if not perm:
        return

    sim_pos  = perm["sim_positive"]    # (n_pos, L+1)
    sim_neg  = perm["sim_negative"]    # (n_neg, L+1)
    auroc    = perm["auroc"]           # (L+1,)
    cnorm    = perm["concept_norm"]    # (L+1,)
    layers   = np.arange(len(auroc))

    colour  = COLOURS["ast_pure"] if concept_type == "ast" else COLOURS["builtin_pure"]
    neg_col = "#BDBDBD"

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: cos_sim bands
    ax = axes[0]
    pos_mean = sim_pos.mean(axis=0)
    pos_std  = sim_pos.std(axis=0)
    neg_mean = sim_neg.mean(axis=0)
    neg_std  = sim_neg.std(axis=0)

    ax.fill_between(layers, pos_mean - pos_std, pos_mean + pos_std,
                    alpha=0.2, color=colour)
    ax.fill_between(layers, neg_mean - neg_std, neg_mean + neg_std,
                    alpha=0.15, color=neg_col)
    ax.plot(layers, pos_mean, "o-", color=colour, lw=2,
            label=f"Positive (n={sim_pos.shape[0]})")
    ax.plot(layers, neg_mean, "s--", color=neg_col, lw=1.5,
            label=f"Negative (n={sim_neg.shape[0]})")
    ax.axhline(0, color="grey", lw=0.6, linestyle=":")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine similarity to concept direction")
    ax.set_title(f"'{concept_name}' permeation: cos_sim", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.2)

    # Panel 2: AUROC
    ax = axes[1]
    ax.plot(layers, auroc, "o-", color=colour, lw=2)
    ax.axhline(0.5, color="grey", lw=1, linestyle="--", label="Chance (0.5)")
    ax.axhline(0.75, color="orange", lw=0.8, linestyle=":", alpha=0.7,
               label="Moderate (0.75)")
    ax.fill_between(layers, 0.5, auroc, where=auroc > 0.5,
                    alpha=0.2, color=colour)
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.3, 1.05)
    ax.set_title(f"'{concept_name}' separability (AUROC) by layer", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.2)

    # Panel 3: concept norm (signal strength)
    ax = axes[2]
    ax.bar(layers, cnorm, color=colour, alpha=0.75)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Concept direction L2 norm")
    ax.set_title(f"'{concept_name}' signal magnitude by layer", fontsize=10)
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle(
        f"Concept permeation: {concept_type.upper()} '{concept_name}'\n"
        f"Peak AUROC = {auroc.max():.3f} at layer {auroc.argmax()}",
        fontsize=11
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    plt.close(fig)


def plot_concept_permeation_grid(
    perm_dict:    dict,    # concept_name -> perm result
    concept_type: str,
    metric:       str = "auroc",    # "auroc" | "pos_mean"
    save_path:    Path | None = None,
) -> None:
    """
    Heatmap: rows = concepts, columns = layers, values = AUROC (or mean cos_sim).
    Shows at a glance which concepts are separable at which layers.
    Sorted by peak AUROC layer — reveals the temporal ordering of concept encoding.
    """
    concepts = [c for c, p in perm_dict.items() if p]
    if not concepts:
        return

    L1   = len(next(iter(perm_dict.values()))["auroc"])
    mat  = np.zeros((len(concepts), L1))
    for i, c in enumerate(concepts):
        p = perm_dict[c]
        if metric == "auroc":
            mat[i] = p["auroc"]
        else:
            mat[i] = p["sim_positive"].mean(axis=0)

    # Sort by layer of peak value
    peak_layers = mat.argmax(axis=1)
    order       = np.argsort(peak_layers)
    mat         = mat[order]
    labels      = [concepts[i] for i in order]

    colour = COLOURS["ast_pure"] if concept_type == "ast" else COLOURS["builtin_pure"]
    cmap   = LinearSegmentedColormap.from_list("perm", ["#FFFFFF", colour], N=256)

    fig, ax = plt.subplots(figsize=(max(8, L1 * 0.6), max(5, len(concepts) * 0.3)))
    vmax = 1.0 if metric == "auroc" else max(mat.max(), 0.5)
    vmin = 0.5 if metric == "auroc" else mat.min()
    im   = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                     interpolation="nearest")
    plt.colorbar(im, ax=ax, label="AUROC" if metric == "auroc" else "Mean cos_sim")
    ax.set_xticks(range(L1))
    ax.set_xticklabels([f"L{l}" for l in range(L1)], fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=max(5, 9 - len(labels) // 10))
    ax.set_xlabel("Layer")
    ax.set_title(
        f"{concept_type.upper()} concept permeation grid ({metric})\n"
        "Sorted by layer of peak separability",
        fontsize=11
    )
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
        description="Step 5: Factorial ANOVA probing + concept permeation"
    )
    parser.add_argument("--stem",         default="contrastive_stubs")
    parser.add_argument("--in_dir", "-d", default="data",
                        help="Directory containing pipeline output files (default: data/)")
    parser.add_argument("--out_dir", "-o", default=None,
                        help="Output directory for results (default: same as --in_dir)")
    parser.add_argument("--layer",        type=int, default=-1,
                        help="Layer for ANOVA and cross-probe detail (-1 = final)")
    parser.add_argument("--n_splits",     type=int, default=5,
                        help="CV folds for probe training (default 5)")
    parser.add_argument("--concept",      default=None,
                        help="Concept name for permeation trace (e.g. 'If', 'len')")
    parser.add_argument("--concept_type", default="ast", choices=["ast", "builtin"],
                        help="Whether --concept is an AST node or builtin (default ast)")
    parser.add_argument("--all_concepts", action="store_true",
                        help="Run permeation trace for ALL concepts (slow)")
    parser.add_argument("--no_probe",     action="store_true",
                        help="Skip linear probing / ANOVA (permeation only)")
    parser.add_argument("--plot",         action="store_true")
    args = parser.parse_args()

    base    = Path(__file__).parent / args.in_dir
    out     = Path(__file__).parent / (args.out_dir if args.out_dir else args.in_dir)
    out.mkdir(exist_ok=True)
    img_dir = out / "images"
    img_dir.mkdir(exist_ok=True)
    stem = args.stem

    # ── Load ─────────────────────────────────────────────────────────────────
    print(f"\n[05] Loading inputs for stem='{stem}'")
    resid_all = np.load(base / f"01_{stem}_residual_all.npy")
    N, L1, H  = resid_all.shape
    with open(base / f"01_{stem}_meta.json") as f:
        meta = json.load(f)
    ast_labels     = [m["ast_node"]    for m in meta]
    builtin_labels = [m["builtin_obj"] for m in meta]
    print(f"  Shape: {resid_all.shape}")

    masks = _load_masks(base, stem)
    if masks is None:
        print("  No purity masks found — probing on raw only.")
    else:
        print("  Purity masks loaded.")

    layer_idx = L1 + args.layer if args.layer < 0 else args.layer
    layer_idx = int(np.clip(layer_idx, 0, L1 - 1))

    # ── Linear probing + ANOVA ────────────────────────────────────────────────
    probe_results = None
    anova_by_layer = []
    cross_by_layer = []

    if not args.no_probe:
        print(f"\n[05] Running linear probes (n_splits={args.n_splits}) across all layers...")
        probe_results = probe_all_layers(
            resid_all, ast_labels, builtin_labels, masks, args.n_splits
        )

        # Save probe accuracy
        save_acc = {}
        save_folds = {}
        for s in SUBSPACES:
            for t in ("ast", "builtin"):
                save_acc[f"{s}_{t}"]   = probe_results["accuracy"][s][t]
                save_folds[f"{s}_{t}"] = probe_results["fold_scores"][s][t]
        np.savez_compressed(out / f"05_{stem}_probe_accuracy.npz",  **save_acc)
        np.savez_compressed(out / f"05_{stem}_probe_foldscores.npz", **save_folds)
        print(f"  Saved probe accuracy files.")

        # ANOVA at every layer
        print(f"\n[05] Running factorial ANOVA at each layer...")
        for l in range(L1):
            anova_by_layer.append(run_factorial_anova(probe_results, layer=l))

        with open(out / f"05_{stem}_anova_results.json", "w") as f:
            json.dump(anova_by_layer, f, indent=2, default=float)
        print(f"  Saved 05_{stem}_anova_results.json")

        # Cross-probe at every layer
        print(f"\n[05] Running cross-probe (leakage test) at each layer...")
        for l in range(L1):
            cross_by_layer.append(cross_probe(
                resid_all, ast_labels, builtin_labels, masks,
                layer=l, n_splits=args.n_splits
            ))
        with open(out / f"05_{stem}_cross_probe.json", "w") as f:
            json.dump(cross_by_layer, f, indent=2, default=float)
        print(f"  Saved 05_{stem}_cross_probe.json")

    # ── Concept permeation ────────────────────────────────────────────────────
    perm_result   = None
    ast_perm_all  = {}
    b_perm_all    = {}

    if args.concept:
        print(f"\n[05] Concept permeation: '{args.concept}' ({args.concept_type})")
        cdir = build_concept_directions(resid_all, meta, args.concept, args.concept_type)
        if cdir is None:
            print(f"  No baseline stubs found for '{args.concept}'. "
                  f"Check that baseline_{args.concept_type} stubs exist.")
        else:
            perm_result = concept_permeation_trace(
                resid_all, meta, args.concept, args.concept_type, cdir
            )
            print(f"  Peak AUROC = {perm_result['auroc'].max():.3f} "
                  f"at layer {perm_result['auroc'].argmax()}")

    if args.all_concepts:
        print(f"\n[05] Running permeation for ALL concepts...")
        ast_perm_all, b_perm_all = all_concept_permeation(resid_all, meta)

        # Save AUROC grids
        if ast_perm_all:
            np.savez_compressed(
                out / f"05_{stem}_concept_auroc_ast.npz",
                **{c.replace(" ", "_"): v["auroc"]
                   for c, v in ast_perm_all.items() if v}
            )
        if b_perm_all:
            np.savez_compressed(
                out / f"05_{stem}_concept_auroc_builtin.npz",
                **{c.replace(" ", "_"): v["auroc"]
                   for c, v in b_perm_all.items() if v}
            )
        print(f"  Saved concept AUROC files.")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if args.plot:
        print(f"\n[05] Generating plots...")

        if probe_results:
            plot_probe_accuracy_by_layer(
                probe_results,
                title=f"Linear probe accuracy by layer ({stem})",
                save_path=img_dir / f"05_{stem}_probe_accuracy.png",
            )

        if anova_by_layer:
            plot_anova_interaction(
                probe_results, anova_by_layer,
                save_path=img_dir / f"05_{stem}_anova_interaction.png",
            )

        if cross_by_layer:
            plot_cross_probe(
                cross_by_layer,
                save_path=img_dir / f"05_{stem}_cross_probe.png",
            )

        if perm_result:
            plot_concept_permeation(
                perm_result, args.concept, args.concept_type,
                save_path=img_dir / f"05_{stem}_permeation_{args.concept}.png",
            )

        if args.all_concepts:
            if ast_perm_all:
                plot_concept_permeation_grid(
                    ast_perm_all, "ast",
                    save_path=img_dir / f"05_{stem}_permeation_grid_ast.png",
                )
            if b_perm_all:
                plot_concept_permeation_grid(
                    b_perm_all, "builtin",
                    save_path=img_dir / f"05_{stem}_permeation_grid_builtin.png",
                )

        print(f"  All plots saved to {out}/")

    print("\n[05] Done.")


if __name__ == "__main__":
    main()
