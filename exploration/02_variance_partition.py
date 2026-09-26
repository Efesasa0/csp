"""
02_variance_partition.py

Step 2 of the AST x builtin mechanistic interpretability pipeline.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY WE DO THIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

In step 01 we extracted, for every prompt, five activation arrays:
  residual_all   (N, L+1, H)     — residual stream at every layer
  head_attr      (N, L, n_heads) — signed logit attribution per attention head
  mlp_attr       (N, L)          — signed logit attribution per MLP layer
  mlp_neurons    (N, L, mlp_dim) — pre-AbsTopK MLP neuron activations
  edge_graph     (static)        — model weight-sparsity graph

Each prompt is labelled with two categorical factors:
  ast_node   — the primary Python AST node used  (e.g. "For", "While", "If")
  builtin_obj — the Python builtin called         (e.g. "len", "sorted", "range")

The central question is:
  "Is AST information and builtin information stored in separate parts of the
   model, or are they entangled in the same units?"

Variance partitioning (VP) answers this *per unit*:

  For each unit u (hidden dim / head / neuron) in each layer:
    R²_ast(u)    = fraction of unit u's variance explained by AST label alone
    R²_builtin(u) = fraction explained by builtin label alone
    R²_joint(u)  = fraction explained by both labels together

  Then decompose:
    unique_ast(u)     = R²_joint(u) - R²_builtin(u)   [AST signal, not builtin]
    unique_builtin(u) = R²_joint(u) - R²_ast(u)       [builtin signal, not AST]
    shared(u)         = R²_ast(u) + R²_builtin(u) - R²_joint(u)  [crosstalk]
    unexplained(u)    = 1 - R²_joint(u)               [neither factor]

  This gives a four-way decomposition of every unit in the model.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DOES THIS GIVE "CLEAN" CIRCUITS?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Variance partitioning gives you a *map*, not a separation.

  What you DO get:
    - A per-unit score: how much of this unit's variance is uniquely AST,
      uniquely builtin, or shared.
    - A per-layer profile: at which layer does AST information crystallise?
      At which layer does builtin information crystallise?
    - A "purity mask": which units are "pure AST" (unique_ast > threshold
      and shared < threshold)?

  What you do NOT get directly:
    - Clean separated activations. A unit with high unique_ast still contains
      noise from other sources.

  To get "clean" activations, use the purity masks computed here as input to
  step 03 (projection / residualisation):
    - AST subspace  = span of units where unique_ast >> shared
    - Builtin subspace = span of units where unique_builtin >> shared
    - Project activations onto each subspace to get clean representations
    - OR: regress out the builtin mean-direction from each activation,
      then re-probe for AST — this is the residualised probe approach

  The shared units are the most interpretability-interesting ones:
  they are candidates for interaction circuits that encode the joint
  (AST, builtin) concept rather than either factor in isolation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW IT IS COMPUTED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

All R² values are computed via the hat-matrix (OLS projection) approach,
fully vectorised across all units in a layer simultaneously.

For a layer with activation matrix Y ∈ R^{N x D}:

  1. Build one-hot design matrices:
       X_ast     ∈ R^{N x n_ast}   (one-hot encoded AST labels)
       X_builtin ∈ R^{N x n_b}     (one-hot encoded builtin labels)
       X_joint   = [X_ast | X_builtin]  ∈ R^{N x (n_ast + n_b)}

  2. For each design matrix X, compute the hat matrix H = X(X'X)^{-1}X'
     and the fitted values Y_hat = H @ Y.

  3. Compute R² per unit:
       SS_tot(d)  = sum((Y[:,d] - mean(Y[:,d]))^2)
       SS_res(d)  = sum((Y[:,d] - Y_hat[:,d])^2)
       R²(d)      = 1 - SS_res(d) / SS_tot(d)

  This is equivalent to a one-way ANOVA R² per unit. No model training,
  no hyperparameters, no cross-validation needed — it is an exact
  decomposition of the observed variance.

  Note on collinearity: the one-hot matrices X_ast and X_builtin will
  share variance if the dataset is unbalanced (e.g. some (AST, builtin)
  pairs are overrepresented). The shared term captures this. For a perfectly
  balanced factorial design, shared = 0 by construction.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUT FILES  (all from 01_extraction.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  01_<stem>_residual_all.npy    float32  (N, L+1, H)
  01_<stem>_head_attr.npy       float32  (N, L, n_heads)
  01_<stem>_mlp_attr.npy        float32  (N, L)
  01_<stem>_mlp_neurons.npy     float32  (N, L, mlp_dim)
  01_<stem>_meta.json           prompt metadata with ast_node / builtin_obj labels
  01_<stem>_edge_graph.npz      static weight-sparsity graph

  The edge graph is treated differently: VP does not apply to static weights
  (they have no per-prompt variance). Instead we compute, per edge, how much
  its weight magnitude correlates with the AST-unique and builtin-unique
  activation patterns identified from the residual stream. This gives an
  "activation-weighted edge importance" score for each AST node and builtin.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FILES  (prefix 02_<stem>_)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  02_<stem>_vp_residual.npz    VP decomposition of residual stream per layer+dim
  02_<stem>_vp_heads.npz       VP decomposition of head attributions per layer+head
  02_<stem>_vp_mlp_attr.npz    VP decomposition of MLP layer attributions
  02_<stem>_vp_mlp_neurons.npz VP decomposition of MLP neuron pre-activations
  02_<stem>_vp_edge_weights.npz  Edge activation-importance scores
  02_<stem>_purity_masks.npz   Boolean masks: ast_pure, builtin_pure, shared, interaction
  02_<stem>_summary.json       Scalar summary statistics per quantity

  Each *_vp_*.npz contains arrays:
    unique_ast      same shape as the activation quantity, per unit
    unique_builtin  same shape
    shared          same shape
    unexplained     same shape
    r2_ast          same shape
    r2_builtin      same shape
    r2_joint        same shape

Usage
-----
  python 02_variance_partition.py --stem contrastive_stubs
  python 02_variance_partition.py --stem contrastive_stubs --threshold 0.1
  python 02_variance_partition.py --stem contrastive_stubs --plot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# ─────────────────────────────────────────────────────────────────────────────
# Core VP computation  (fully vectorised, no sklearn dependency)
# ─────────────────────────────────────────────────────────────────────────────

def _hat_project(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute Y_hat = H @ Y where H = X(X'X)^{-1}X' is the hat matrix.
    X : (N, k)  design matrix (should be full-rank; add tiny ridge if needed).
                N = number of samples/prompts, k = number of categorical features.
    Y : (N, D)  activation matrix.
                D = number of dimensions/units (e.g., hidden_dim, n_heads, mlp_dim).
    Returns Y_hat : (N, D)
    """
    # Least-squares solution: B = (X'X)^{-1} X'Y  →  Y_hat = X @ B
    B, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)   # (k, D)
    return X @ B                                       # (N, D)


def variance_partition(
    Y: np.ndarray,          # (N, D)  activation matrix for one layer/quantity
    ast_labels:   list,     # length N
    builtin_labels: list,   # length N
) -> dict[str, np.ndarray]:
    """
    Variance partitioning per unit (column of Y).

    Returns dict with keys:
      r2_ast, r2_builtin, r2_joint,
      unique_ast, unique_builtin, shared, unexplained
    Each value is a 1-D array of shape (D,).
    """
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder

    def _ohe(labels: list) -> np.ndarray:
        le  = LabelEncoder()
        y   = le.fit_transform(labels).reshape(-1, 1)
        enc = OneHotEncoder(sparse_output=False, drop="first")  # drop one column to avoid rank deficiency
        return enc.fit_transform(y)

    X_ast  = _ohe(ast_labels)               # (N, n_ast-1)
    X_b    = _ohe(builtin_labels)           # (N, n_b-1)
    X_both = np.hstack([X_ast, X_b])        # (N, n_ast-1 + n_b-1)

    # centre Y (required for R² interpretation)
    Y_c   = Y - Y.mean(axis=0, keepdims=True)           # (N, D)
    ss_tot = (Y_c ** 2).sum(axis=0)                      # (D,)  — total variance per unit
    ss_tot = np.where(ss_tot < 1e-12, 1e-12, ss_tot)    # avoid division by zero

    def _r2(X: np.ndarray) -> np.ndarray:
        Y_hat  = _hat_project(X, Y_c)                   # (N, D)
        ss_res = ((Y_c - Y_hat) ** 2).sum(axis=0)       # (D,)
        return np.clip(1.0 - ss_res / ss_tot, 0.0, 1.0) # (D,)

    r2_ast  = _r2(X_ast)
    r2_b    = _r2(X_b)
    r2_joint= _r2(X_both)

    unique_ast    = np.clip(r2_joint - r2_b,            0.0, 1.0)
    unique_builtin= np.clip(r2_joint - r2_ast,          0.0, 1.0)
    shared        = np.clip(r2_ast + r2_b - r2_joint,   0.0, 1.0)
    unexplained   = np.clip(1.0 - r2_joint,             0.0, 1.0)

    return {
        "r2_ast":         r2_ast,
        "r2_builtin":     r2_b,
        "r2_joint":       r2_joint,
        "unique_ast":     unique_ast,
        "unique_builtin": unique_builtin,
        "shared":         shared,
        "unexplained":    unexplained,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-quantity VP runners
# ─────────────────────────────────────────────────────────────────────────────

def vp_residual(
    resid_all:      np.ndarray,   # (N, L+1, H)
    ast_labels:     list,
    builtin_labels: list,
) -> dict[str, np.ndarray]:
    """
    VP on residual stream — applied at every layer independently.
    Returns arrays of shape (L+1, H) for each VP component.
    """
    N, n_layers_p1, H = resid_all.shape
    keys = ["r2_ast","r2_builtin","r2_joint","unique_ast","unique_builtin","shared","unexplained"]
    result = {k: np.zeros((n_layers_p1, H), dtype=np.float32) for k in keys}

    for l in range(n_layers_p1):
        vp = variance_partition(resid_all[:, l, :], ast_labels, builtin_labels)
        for k in keys:
            result[k][l] = vp[k]
        print(f"  residual layer {l+1}/{n_layers_p1} done", end="\r")
    print()
    return result


def vp_heads(
    head_attr:      np.ndarray,   # (N, L, n_heads)
    ast_labels:     list,
    builtin_labels: list,
) -> dict[str, np.ndarray]:
    """
    VP on per-head logit attribution.
    Returns arrays of shape (L, n_heads).
    """
    N, L, n_heads = head_attr.shape
    keys = ["r2_ast","r2_builtin","r2_joint","unique_ast","unique_builtin","shared","unexplained"]
    result = {k: np.zeros((L, n_heads), dtype=np.float32) for k in keys}

    for l in range(L):
        vp = variance_partition(head_attr[:, l, :], ast_labels, builtin_labels)
        for k in keys:
            result[k][l] = vp[k]
        print(f"  heads layer {l+1}/{L} done", end="\r")
    print()
    return result


def vp_mlp_attr(
    mlp_attr:       np.ndarray,   # (N, L)
    ast_labels:     list,
    builtin_labels: list,
) -> dict[str, np.ndarray]:
    """
    VP on per-layer MLP logit attribution.
    mlp_attr has one scalar per prompt per layer, so VP is computed
    across prompts for each layer independently — returns shape (L,).
    """
    N, L = mlp_attr.shape
    keys = ["r2_ast","r2_builtin","r2_joint","unique_ast","unique_builtin","shared","unexplained"]
    result = {k: np.zeros((L,), dtype=np.float32) for k in keys}

    for l in range(L):
        vp = variance_partition(mlp_attr[:, l:l+1], ast_labels, builtin_labels)
        for k in keys:
            result[k][l] = float(vp[k][0])
    return result


def vp_mlp_neurons(
    mlp_neurons:    np.ndarray,   # (N, L, mlp_dim)
    ast_labels:     list,
    builtin_labels: list,
) -> dict[str, np.ndarray]:
    """
    VP on MLP pre-AbsTopK neuron activations.
    Returns arrays of shape (L, mlp_dim).
    This is the most expensive computation (mlp_dim can be 4*H).
    """
    N, L, mlp_dim = mlp_neurons.shape
    keys = ["r2_ast","r2_builtin","r2_joint","unique_ast","unique_builtin","shared","unexplained"]
    result = {k: np.zeros((L, mlp_dim), dtype=np.float32) for k in keys}

    for l in range(L):
        vp = variance_partition(mlp_neurons[:, l, :], ast_labels, builtin_labels)
        for k in keys:
            result[k][l] = vp[k]
        print(f"  mlp_neurons layer {l+1}/{L} done", end="\r")
    print()
    return result


def vp_edge_weights(
    edge_npz:       dict,         # loaded from 01_*_edge_graph.npz
    resid_vp:       dict,         # output of vp_residual
    ast_labels:     list,
    builtin_labels: list,
) -> dict[str, np.ndarray]:
    """
    Edge activation-importance scores.

    Static edge weights have no per-prompt variance, so we cannot apply VP
    directly. Instead, for each edge (src_layer, src_unit) → (dst_layer, dst_unit),
    we look up the VP score of the source unit in the residual VP map and
    assign it as the edge's "AST importance" and "builtin importance".

    This gives every edge in the sparsity graph a decomposition:
      edge_ast_importance     = unique_ast[src_layer, src_unit]  * |weight|
      edge_builtin_importance = unique_builtin[src_layer, src_unit] * |weight|
      edge_shared             = shared[src_layer, src_unit]       * |weight|

    The magnitude-weighting ensures that large weights in AST-unique dimensions
    rank highest — i.e. these edges are the most likely to implement AST-specific
    circuits.
    """
    layers    = edge_npz["layer"].astype(int)
    src_idx   = edge_npz["src_idx"].astype(int)
    weights   = edge_npz["weight"].astype(np.float32)
    abs_w     = np.abs(weights)

    n_layers_p1, H = resid_vp["unique_ast"].shape
    E = len(weights)

    edge_ast     = np.zeros(E, dtype=np.float32)
    edge_builtin = np.zeros(E, dtype=np.float32)
    edge_shared  = np.zeros(E, dtype=np.float32)

    for i in range(E):
        l = layers[i]
        s = src_idx[i]
        # clip to valid range (embedding layer = -1 mapped to layer 0)
        layer_idx = max(0, min(l, n_layers_p1 - 1))
        dim_idx   = min(s, H - 1)
        edge_ast[i]     = resid_vp["unique_ast"][layer_idx, dim_idx]     * abs_w[i]
        edge_builtin[i] = resid_vp["unique_builtin"][layer_idx, dim_idx] * abs_w[i]
        edge_shared[i]  = resid_vp["shared"][layer_idx, dim_idx]         * abs_w[i]

    return {
        "src_name":             edge_npz["src_name"],
        "dst_name":             edge_npz["dst_name"],
        "layer":                layers,
        "src_idx":              src_idx,
        "dst_idx":              edge_npz["dst_idx"].astype(int),
        "weight":               weights,
        "edge_ast_importance":  edge_ast,
        "edge_builtin_importance": edge_builtin,
        "edge_shared":          edge_shared,
    }


def build_purity_masks(
    resid_vp:   dict,           # output of vp_residual  (L+1, H)
    head_vp:    dict,           # output of vp_heads     (L, n_heads)
    neuron_vp:  dict,           # output of vp_mlp_neurons (L, mlp_dim)
    threshold:  float = 0.05,   # minimum R² to count as "informative"
    purity:     float = 0.7,    # fraction of informative variance that must be unique
) -> dict[str, dict[str, np.ndarray]]:
    """
    Build boolean purity masks for each quantity.

    A unit is labelled:
      ast_pure      : unique_ast   > threshold  AND  unique_ast / r2_joint > purity
      builtin_pure  : unique_builtin > threshold AND  unique_builtin / r2_joint > purity
      shared        : shared > threshold  AND  neither pure
      interaction   : shared > threshold  AND  r2_joint > 2*threshold
                      (high joint R², significant shared component)
      noise         : r2_joint < threshold  (neither factor explains much)

    These masks are the inputs for step 03 (projection / subspace extraction).
    """
    masks = {}
    for name, vp in [("residual", resid_vp),
                     ("heads",    head_vp),
                     ("neurons",  neuron_vp)]:
        r2j  = vp["r2_joint"]
        ua   = vp["unique_ast"]
        ub   = vp["unique_builtin"]
        sh   = vp["shared"]
        safe = np.where(r2j > 1e-9, r2j, 1e-9)

        ast_pure     = (ua > threshold) & (ua / safe > purity)
        builtin_pure = (ub > threshold) & (ub / safe > purity)
        shared_mask  = (sh > threshold) & ~ast_pure & ~builtin_pure
        interaction  = (sh > threshold) & (r2j > 2 * threshold)
        noise        = r2j < threshold

        masks[name] = {
            "ast_pure":     ast_pure.astype(bool),
            "builtin_pure": builtin_pure.astype(bool),
            "shared":       shared_mask.astype(bool),
            "interaction":  interaction.astype(bool),
            "noise":        noise.astype(bool),
        }
    return masks


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

COLOURS = {
    "unique_ast":     "#2196F3",   # blue
    "unique_builtin": "#FF9800",   # orange
    "shared":         "#9C27B0",   # purple
    "unexplained":    "#EEEEEE",   # light grey
}


def plot_vp_stacked_bar(
    vp:       dict,           # e.g. output of vp_residual,  shape (L, D) per key
    title:    str,
    xlabel:   str = "Layer",
    save_path: Path | None = None,
) -> None:
    """
    Stacked bar: mean variance partition across units, per layer.
    Shows the global picture — how much variance is uniquely AST vs builtin
    vs shared at each layer.
    """
    # average across the unit axis (last axis)
    def _mean(key):
        arr = vp[key]
        return arr.mean(axis=-1) if arr.ndim > 1 else arr

    ua  = _mean("unique_ast")
    ub  = _mean("unique_builtin")
    sh  = _mean("shared")
    un  = _mean("unexplained")
    layers = np.arange(len(ua))

    fig, ax = plt.subplots(figsize=(max(8, len(layers) * 0.5), 4))
    ax.bar(layers, ua, label="Unique AST",     color=COLOURS["unique_ast"])
    ax.bar(layers, ub, bottom=ua,              label="Unique builtin", color=COLOURS["unique_builtin"])
    ax.bar(layers, sh, bottom=ua + ub,         label="Shared (crosstalk)", color=COLOURS["shared"])
    ax.bar(layers, un, bottom=ua + ub + sh,    label="Unexplained", color=COLOURS["unexplained"],
           edgecolor="#aaaaaa", linewidth=0.5)

    ax.set_xticks(layers)
    ax.set_xticklabels([str(l) for l in layers], fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean fraction of variance")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    plt.show()


def plot_vp_heatmap(
    vp:         dict,           # shape per key: (L, D)
    component:  str,            # "unique_ast" | "unique_builtin" | "shared"
    title:      str,
    max_units:  int = 200,      # truncate D for readability
    save_path:  Path | None = None,
) -> None:
    """
    Heatmap of VP component across (layer, unit).
    Rows = layers, columns = units (truncated to max_units for readability).
    Bright cells = high unique variance for the chosen component.
    Shows which specific units in which layers carry the signal.
    """
    arr = vp[component]                      # (L, D) or (L,)
    if arr.ndim == 1:
        arr = arr[:, None]
    arr = arr[:, :max_units]

    fig, ax = plt.subplots(figsize=(min(20, max_units * 0.08 + 2), max(4, arr.shape[0] * 0.4)))
    cmap = {"unique_ast": "Blues", "unique_builtin": "Oranges",
            "shared": "Purples", "unexplained": "Greys"}.get(component, "viridis")
    im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=0, vmax=0.5,
                   interpolation="nearest")
    plt.colorbar(im, ax=ax, label=f"R² ({component})")
    ax.set_xlabel(f"Unit index (first {max_units})")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    plt.show()


def plot_purity_masks(
    masks:    dict,           # output of build_purity_masks
    quantity: str,            # "residual" | "heads" | "neurons"
    save_path: Path | None = None,
) -> None:
    """
    Four-panel binary heatmap showing which units are ast_pure / builtin_pure /
    shared / interaction at each layer.
    """
    m     = masks[quantity]
    keys  = ["ast_pure", "builtin_pure", "shared", "interaction"]
    labels= ["AST-pure", "Builtin-pure", "Shared crosstalk", "Interaction circuit"]
    colours = ["#2196F3", "#FF9800", "#9C27B0", "#E91E63"]

    fig, axes = plt.subplots(1, 4, figsize=(16, max(3, list(m.values())[0].shape[0] * 0.35)))
    for ax, key, label, colour in zip(axes, keys, labels, colours):
        data = m[key]
        if data.ndim == 1:
            data = data[:, None]
        # use a two-colour map: white = False, colour = True
        cmap = plt.matplotlib.colors.ListedColormap(["#f5f5f5", colour])
        ax.imshow(data.astype(float), aspect="auto", cmap=cmap, vmin=0, vmax=1,
                  interpolation="nearest")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Unit index")
        ax.set_ylabel("Layer" if ax == axes[0] else "")
        frac = data.mean()
        ax.set_xlabel(f"Unit index\n({frac:.1%} of units)", fontsize=8)

    plt.suptitle(f"Purity masks — {quantity}\n"
                 "Coloured = unit meets purity threshold for that category",
                 fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    plt.show()


def plot_before_after_vp(
    Y_raw:      np.ndarray,   # (N, D)  raw activations for one layer
    vp:         dict,         # VP output for this layer (1-D arrays of shape D)
    ast_labels: list,
    builtin_labels: list,
    title:      str,
    n_components: int = 2,
    save_path:  Path | None = None,
) -> None:
    """
    PCA scatter of activations before and after VP-based projection.

    Left panel  — raw activations, coloured by AST node.
    Right panel — activations projected onto AST-unique dimensions only
                  (pure AST subspace), coloured by AST node.
    Middle panel— same with builtin-unique dimensions, coloured by builtin.

    If AST information is cleanly separated by VP, the right panel should
    show better cluster separation than the left.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import LabelEncoder

    le_ast = LabelEncoder()
    le_b   = LabelEncoder()
    y_ast  = le_ast.fit_transform(ast_labels)
    y_b    = le_b.fit_transform(builtin_labels)
    n_ast  = len(le_ast.classes_)
    n_b    = len(le_b.classes_)

    # AST-unique subspace: columns where unique_ast > median unique_ast
    # (using relative threshold so it works regardless of absolute R² scale)
    ua_thresh  = np.percentile(vp["unique_ast"],     75)
    ub_thresh  = np.percentile(vp["unique_builtin"], 75)

    ast_dims    = np.where(vp["unique_ast"]     >= ua_thresh)[0]
    builtin_dims= np.where(vp["unique_builtin"] >= ub_thresh)[0]

    Y_ast_sub = Y_raw[:, ast_dims]     if len(ast_dims) > 0     else Y_raw
    Y_b_sub   = Y_raw[:, builtin_dims] if len(builtin_dims) > 0 else Y_raw

    def _pca2(X):
        pca = PCA(n_components=min(n_components, X.shape[1]))
        return pca.fit_transform(X)

    Z_raw  = _pca2(Y_raw)
    Z_ast  = _pca2(Y_ast_sub)
    Z_b    = _pca2(Y_b_sub)

    cmap_ast = plt.colormaps["tab20"]
    cmap_b   = plt.colormaps["tab20b"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, Z, labels, le, n_cats, cmap, subtitle in [
        (axes[0], Z_raw,  y_ast, le_ast, n_ast, cmap_ast,
         f"Raw — coloured by AST\n({Y_raw.shape[1]} dims)"),
        (axes[1], Z_ast,  y_ast, le_ast, n_ast, cmap_ast,
         f"AST-unique subspace\n({len(ast_dims)} dims, top-25% unique_ast)"),
        (axes[2], Z_b,    y_b,   le_b,   n_b,   cmap_b,
         f"Builtin-unique subspace\n({len(builtin_dims)} dims, top-25% unique_builtin)"),
    ]:
        for c in range(n_cats):
            mask = labels == c
            ax.scatter(Z[mask, 0], Z[mask, 1] if Z.shape[1] > 1 else np.zeros(mask.sum()),
                       c=[cmap(c / max(n_cats - 1, 1))],
                       label=le.classes_[c], s=15, alpha=0.6)
        ax.set_title(subtitle, fontsize=8)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        if n_cats <= 15:
            ax.legend(fontsize=5, ncol=2, loc="best")

    plt.suptitle(title, fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    plt.show()


def plot_top_units(
    vp:        dict,          # (L, D) per key
    component: str,           # "unique_ast" | "unique_builtin" | "shared"
    top_k:     int = 20,
    title:     str = "",
    save_path: Path | None = None,
) -> None:
    """
    Bar chart of the top-k units (layer, dim) with highest VP score.
    Useful for identifying exactly which hidden dimensions carry AST/builtin info.
    """
    arr = vp[component]      # (L, D)
    if arr.ndim == 1:
        arr = arr[:, None]

    # flatten and get top-k indices
    flat    = arr.flatten()
    top_idx = np.argsort(flat)[::-1][:top_k]
    L, D    = arr.shape
    scores  = flat[top_idx]
    labels  = [f"L{idx // D}_u{idx % D}" for idx in top_idx]

    colour = COLOURS.get(component, "#607D8B")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(range(top_k), scores[::-1], color=colour)
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(labels[::-1], fontsize=7, fontfamily="monospace")
    ax.set_xlabel(f"R² ({component})")
    ax.set_title(title or f"Top-{top_k} units by {component}")
    ax.axvline(0.1, color="gray", linestyle="--", lw=0.8, label="threshold=0.1")
    ax.legend(fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    plt.show()


def plot_edge_importance(
    edge_vp:  dict,
    component: str = "edge_ast_importance",
    top_k:    int = 30,
    title:    str = "",
    save_path: Path | None = None,
) -> None:
    """
    Bar chart of top-k edges by activation-weighted importance.
    """
    scores = edge_vp[component]
    top_idx = np.argsort(scores)[::-1][:top_k]

    src_names = edge_vp["src_name"]
    dst_names = edge_vp["dst_name"]
    edge_labels = [f"{src_names[i]} -> {dst_names[i]}" for i in top_idx]

    colour = {"edge_ast_importance": COLOURS["unique_ast"],
              "edge_builtin_importance": COLOURS["unique_builtin"],
              "edge_shared": COLOURS["shared"]}.get(component, "#607D8B")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(range(top_k), scores[top_idx][::-1], color=colour)
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(edge_labels[::-1], fontsize=6, fontfamily="monospace")
    ax.set_xlabel("Activation-weighted importance")
    ax.set_title(title or f"Top-{top_k} edges by {component}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Edge pathway visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

# Canonical order and display names for the four intra-block pathway segments.
_PATHWAY_TAGS = ["embed", "attn_out", "residual", "mlp_fc"]
_PATHWAY_LABELS = {
    "embed":    "Resid→Attn",
    "attn_out": "Attn→Resid",
    "residual": "Resid→MLP",
    "mlp_fc":   "MLP→Resid",
}


def _parse_src_tag(src_name: str) -> str:
    """
    Extract the component type from a src_name string of the form
    'L{layer}_{tag}_{idx}' or 'tok_{n}' / 'residual_dim_{n}'.
    Returns one of the _PATHWAY_TAGS keys, or 'other'.
    """
    parts = src_name.split("_")
    if len(parts) >= 2 and parts[0].startswith("L"):
        return parts[1]      # e.g. 'embed', 'attn', 'residual', 'mlp'
    return "other"


def _aggregate_by_pathway(edge_vp: dict) -> dict:
    """
    Aggregate edge importance by (layer, src_tag) for the four main pathway
    segments.  Returns dict keyed by VP component:
      result[vp_key][(layer, pathway_tag)] = summed_importance
    """
    layers   = edge_vp["layer"]
    src_names = edge_vp["src_name"]
    vp_keys  = ["edge_ast_importance", "edge_builtin_importance", "edge_shared"]

    agg: dict[str, dict] = {k: {} for k in vp_keys}

    for i, (l, sn) in enumerate(zip(layers, src_names)):
        tag = _parse_src_tag(sn)
        if tag not in _PATHWAY_TAGS:
            continue
        key = (int(l), tag)
        for k in vp_keys:
            agg[k][key] = agg[k].get(key, 0.0) + float(edge_vp[k][i])

    return agg


def plot_edge_pathway_heatmap(
    edge_vp:   dict,
    save_path: Path | None = None,
) -> None:
    """
    Option B — Layer × pathway-segment heatmap.
    Three panels (AST / builtin / shared), rows = layers, columns = pathway
    segments. Cell colour = sum of VP-weighted edge importance.
    """
    agg = _aggregate_by_pathway(edge_vp)
    layers_present = sorted({l for (l, _) in next(iter(agg.values()))})
    pathways       = _PATHWAY_TAGS
    xlabels        = [_PATHWAY_LABELS[p] for p in pathways]
    ylabels        = ["emb" if l == 0 else str(l) for l in layers_present]

    vp_keys    = ["edge_ast_importance", "edge_builtin_importance", "edge_shared"]
    panel_titles = ["AST importance", "Builtin importance", "Shared importance"]
    cmaps        = ["Blues", "Oranges", "Purples"]

    fig, axes = plt.subplots(1, 3, figsize=(14, max(4, len(layers_present) * 0.55 + 1)))

    for ax, vk, title, cmap in zip(axes, vp_keys, panel_titles, cmaps):
        grid = np.array([
            [agg[vk].get((l, p), 0.0) for p in pathways]
            for l in layers_present
        ])
        im = ax.imshow(grid, aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(pathways)))
        ax.set_xticklabels(xlabels, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(len(layers_present)))
        ax.set_yticklabels(ylabels, fontsize=9)
        ax.set_xlabel("Pathway segment", fontsize=9)
        ax.set_ylabel("Layer", fontsize=9)
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="Σ VP-weighted importance")

    fig.suptitle(
        "Edge pathway importance by layer\n"
        "Each cell = total VP-weighted importance flowing through that segment",
        fontsize=11,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    plt.close(fig)


def plot_edge_pathway_flow(
    edge_vp:   dict,
    save_path: Path | None = None,
) -> None:
    """
    Option A — Per-layer pathway flow (Sankey-style line plot).
    Three panels (AST / builtin / shared), x = layer, one line per pathway
    segment. Shows which computational pathway carries each signal at each depth.
    Layer 0 is the raw embedding; x-tick labelled 'emb'.
    """
    agg = _aggregate_by_pathway(edge_vp)
    layers_present = sorted({l for (l, _) in next(iter(agg.values()))})
    xtick_labels   = ["emb" if l == 0 else str(l) for l in layers_present]

    vp_keys      = ["edge_ast_importance", "edge_builtin_importance", "edge_shared"]
    panel_titles = ["AST importance", "Builtin importance", "Shared importance"]
    pathway_colours = {
        "embed":    "#4C72B0",
        "attn_out": "#DD8452",
        "residual": "#55A868",
        "mlp_fc":   "#C44E52",
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=False)

    for ax, vk, title in zip(axes, vp_keys, panel_titles):
        for tag in _PATHWAY_TAGS:
            vals = [agg[vk].get((l, tag), 0.0) for l in layers_present]
            ax.plot(range(len(layers_present)), vals, "o-",
                    color=pathway_colours[tag], lw=2,
                    label=_PATHWAY_LABELS[tag])
            ax.fill_between(range(len(layers_present)), vals,
                            alpha=0.08, color=pathway_colours[tag])
        ax.set_xticks(range(len(layers_present)))
        ax.set_xticklabels(xtick_labels, fontsize=8)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Σ VP-weighted importance")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.2)

    fig.suptitle(
        "Edge pathway flow across layers\n"
        "Which computational pathway carries AST / builtin / shared signal at each depth?",
        fontsize=11,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Summary statistics
# ─────────────────────────────────────────────────────────────────────────────

def summarise(name: str, vp: dict) -> dict:
    """Scalar summary of a VP result dict."""
    keys = ["unique_ast","unique_builtin","shared","unexplained","r2_joint"]
    out  = {"quantity": name}
    for k in keys:
        arr = vp[k].flatten()
        out[f"{k}_mean"] = float(arr.mean())
        out[f"{k}_max"]  = float(arr.max())
        out[f"{k}_pct_above_0.1"] = float((arr > 0.1).mean())
    return out


def classify_builtins(builtin_names: list[str]) -> dict[str, str]:
    """
    Classify each unique builtin name as 'single' or 'multi' token
    using the GPT-2 BPE vocabulary (same as circuit_sparsity).
    Returns {name: 'single'|'multi'}.
    """
    try:
        from transformers import GPT2Tokenizer
        tok = GPT2Tokenizer.from_pretrained("gpt2")
        result = {}
        for name in builtin_names:
            ids = tok.encode(name, add_special_tokens=False)
            result[name] = "single" if len(ids) == 1 else "multi"
        return result
    except Exception as e:
        print(f"  [classify_builtins] Could not load GPT-2 tokenizer: {e}")
        return {name: "unknown" for name in builtin_names}


def vp_subset_summary(
    resid_all: np.ndarray,          # (N, L+1, H)
    ast_labels: list,
    builtin_labels: list,
    builtin_token_types: dict,      # {builtin_name: 'single'|'multi'}
    token_type: str,                # 'single' or 'multi'
) -> tuple:
    """
    Re-run residual VP on the subset of prompts where builtin is single- or
    multi-token, returning (scalar summary dict, raw vp dict).
    Raw vp dict is None if too few samples.
    """
    idx = [i for i, b in enumerate(builtin_labels)
           if builtin_token_types.get(b) == token_type]
    if len(idx) < 10:
        return {"n": len(idx), "note": "too few samples"}, None

    sub_resid  = resid_all[idx]
    sub_ast    = [ast_labels[i]     for i in idx]
    sub_builtin = [builtin_labels[i] for i in idx]

    vp = vp_residual(sub_resid, sub_ast, sub_builtin)
    s  = summarise(f"residual_{token_type}_token", vp)
    s["n"] = len(idx)
    s["n_builtin_classes"] = len(set(sub_builtin))
    return s, vp


def plot_builtin_token_type_comparison(
    single_summary: dict,
    multi_summary:  dict,
    builtin_token_types: dict,
    raw_single: dict = None,
    raw_multi:  dict = None,
    save_path=None,
):
    """
    Two-panel figure:
      Left : bar chart of unique_builtin_mean / unique_ast_mean / shared_mean
             for single-token vs multi-token builtins, with Mann-Whitney U
             significance brackets on each VP component (if raw arrays provided).
      Right: two-column list of builtin names by tokenisation type.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from scipy import stats as spstats

    def _sig_stars(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return "ns"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Left: grouped bar ──────────────────────────────────────────────────────
    ax = axes[0]
    vp_keys    = ["unique_ast",      "unique_builtin",      "shared",      "r2_joint"]
    sum_keys   = ["unique_ast_mean", "unique_builtin_mean", "shared_mean", "r2_joint_mean"]
    labels     = ["unique_ast",      "unique_builtin",      "shared",      "r2_joint"]
    x          = np.arange(len(vp_keys))
    w          = 0.35
    s_vals = [single_summary.get(k, 0) for k in sum_keys]
    m_vals = [multi_summary.get(k, 0)  for k in sum_keys]
    bars_s = ax.bar(x - w/2, s_vals, w, label=f"single-token (n={single_summary.get('n',0)})",
                    color="#4C72B0", alpha=0.85)
    bars_m = ax.bar(x + w/2, m_vals, w, label=f"multi-token  (n={multi_summary.get('n',0)})",
                    color="#DD8452", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Mean R² (across all layers × units)")
    ax.set_title("Residual VP: single- vs multi-token builtins")
    ax.legend(fontsize=9)
    y_top = max(max(s_vals), max(m_vals)) * 1.45
    ax.set_ylim(0, y_top)

    # ── Significance brackets ──────────────────────────────────────────────────
    if raw_single is not None and raw_multi is not None:
        bracket_h = y_top * 0.04   # height of the horizontal bracket line
        text_gap  = y_top * 0.015
        for i, vk in enumerate(vp_keys):
            a = raw_single[vk].flatten()
            b = raw_multi[vk].flatten()
            stat, p = spstats.mannwhitneyu(a, b, alternative="two-sided")
            stars = _sig_stars(p)
            bar_top = max(s_vals[i], m_vals[i])
            y1 = bar_top + y_top * 0.04
            y2 = y1 + bracket_h
            # bracket: left tip → top left → top right → right tip
            lx = i - w/2
            rx = i + w/2
            ax.plot([lx, lx, rx, rx], [y1, y2, y2, y1],
                    color="#333333", lw=0.9, clip_on=False)
            ax.text(i, y2 + text_gap, stars,
                    ha="center", va="bottom", fontsize=8,
                    color="#333333" if stars != "ns" else "#888888")
            # also print to stdout
            print(f"  MWU [{vk}]: U={stat:.0f}  p={p:.4g}  {stars}")

    # ── Right: clean two-column list ─────────────────────────────────────────
    ax2 = axes[1]
    ax2.axis("off")
    single_names = sorted(n for n, t in builtin_token_types.items() if t == "single")
    multi_names  = sorted(n for n, t in builtin_token_types.items() if t == "multi")
    n_rows = max(len(single_names), len(multi_names))

    ax2.text(0.25, 1.02, f"single-token ({len(single_names)})",
             ha="center", va="bottom", fontsize=9, fontweight="bold",
             color="#4C72B0", transform=ax2.transAxes)
    ax2.text(0.75, 1.02, f"multi-token ({len(multi_names)})",
             ha="center", va="bottom", fontsize=9, fontweight="bold",
             color="#DD8452", transform=ax2.transAxes)

    for i, name in enumerate(single_names):
        y = 1.0 - (i + 1) / (n_rows + 1)
        ax2.text(0.25, y, name, ha="center", va="center", fontsize=7.5,
                 color="#4C72B0", transform=ax2.transAxes)
    for i, name in enumerate(multi_names):
        y = 1.0 - (i + 1) / (n_rows + 1)
        ax2.text(0.75, y, name, ha="center", va="center", fontsize=7.5,
                 color="#DD8452", transform=ax2.transAxes)
    ax2.set_title("Builtin names by tokenisation type", pad=14)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Variance partitioning on 01_extraction.py outputs."
    )
    parser.add_argument("--stem", "-s", required=True,
                        help="Dataset stem, e.g. 'contrastive_stubs'. "
                             "All 01_<stem>_* files must be in the same directory.")
    parser.add_argument("--in_dir", "-d", default="data",
                        help="Directory containing the 01_* files. "
                             "Defaults to same directory as this script.")
    parser.add_argument("--out_dir", "-o", default=None,
                        help="Output directory for results (default: same as --in_dir)")
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="Min R² to count a unit as informative (default: 0.05).")
    parser.add_argument("--purity", type=float, default=0.70,
                        help="Min fraction of joint R² that must be unique for a "
                             "'pure' label (default: 0.70).")
    parser.add_argument("--plot", action="store_true",
                        help="Generate and save all plots.")
    parser.add_argument("--scatter_layer", type=int, default=None,
                        help="Layer index for before/after VP scatter. "
                             "Omit to generate one plot per layer (default). "
                             "Use -1 for the final layer only.")
    args = parser.parse_args()

    in_dir = Path(__file__).parent / args.in_dir
    out_dir = Path(__file__).parent / (args.out_dir if args.out_dir else args.in_dir)
    out_dir.mkdir(exist_ok=True)
    img_dir = Path(__file__).parent / out_dir / "images"
    img_dir.mkdir(exist_ok=True)
    stem = args.stem
    in_prefix  = f"01_{stem}_"
    out_prefix = f"02_{stem}_"

    # ── load 01 outputs ───────────────────────────────────────────────────────
    def _load(name):
        p = out_dir / f"{in_prefix}{name}"
        assert p.exists(), f"Missing input: {p}"
        return np.load(p)

    print(f"Loading 01 outputs for stem '{stem}' from {out_dir} ...")
    resid_all   = _load("residual_all.npy")      # (N, L+1, H)
    head_attr   = _load("head_attr.npy")          # (N, L, n_heads)
    mlp_attr    = _load("mlp_attr.npy")           # (N, L)
    mlp_neurons = _load("mlp_neurons.npy")        # (N, L, mlp_dim)

    edge_path = out_dir / f"{in_prefix}edge_graph.npz"
    has_edges = edge_path.exists()
    if has_edges:
        edge_npz = dict(np.load(edge_path, allow_pickle=True))
    else:
        print("  Edge graph not found — skipping edge VP.")

    with open(out_dir / f"{in_prefix}meta.json", encoding="utf-8") as f:
        meta = json.load(f)

    N = resid_all.shape[0]
    assert len(meta) == N, f"Meta length {len(meta)} != N={N}"

    ast_labels     = [r["ast_node"]    for r in meta]
    builtin_labels = [r["builtin_obj"] for r in meta]
    print(f"  N={N}  AST classes={len(set(ast_labels))}  "
          f"Builtin classes={len(set(builtin_labels))}")

    # ── classify builtins as single- or multi-token ───────────────────────────
    unique_builtins = sorted(set(builtin_labels))
    builtin_token_types = classify_builtins(unique_builtins)
    n_single = sum(1 for v in builtin_token_types.values() if v == "single")
    n_multi  = sum(1 for v in builtin_token_types.values() if v == "multi")
    print(f"  Builtin tokenisation: {n_single} single-token, {n_multi} multi-token")

    # ── variance partitioning ─────────────────────────────────────────────────
    summaries = []

    print("\n[1/4] Residual stream VP ...")
    vp_r = vp_residual(resid_all, ast_labels, builtin_labels)
    np.savez(out_dir / f"{out_prefix}vp_residual.npz", **vp_r)
    summaries.append(summarise("residual", vp_r))
    print(f"  Saved -> {out_prefix}vp_residual.npz")

    print("\n[2/4] Attention head attribution VP ...")
    vp_h = vp_heads(head_attr, ast_labels, builtin_labels)
    np.savez(out_dir / f"{out_prefix}vp_heads.npz", **vp_h)
    summaries.append(summarise("heads", vp_h))
    print(f"  Saved -> {out_prefix}vp_heads.npz")

    print("\n[3/4] MLP attribution VP ...")
    vp_m = vp_mlp_attr(mlp_attr, ast_labels, builtin_labels)
    np.savez(out_dir / f"{out_prefix}vp_mlp_attr.npz", **vp_m)
    summaries.append(summarise("mlp_attr", vp_m))
    print(f"  Saved -> {out_prefix}vp_mlp_attr.npz")

    print("\n[4/4] MLP neuron pre-activation VP ...")
    vp_n = vp_mlp_neurons(mlp_neurons, ast_labels, builtin_labels)
    np.savez(out_dir / f"{out_prefix}vp_mlp_neurons.npz", **vp_n)
    summaries.append(summarise("mlp_neurons", vp_n))
    print(f"  Saved -> {out_prefix}vp_mlp_neurons.npz")

    if has_edges:
        print("\n[+] Edge activation-importance scores ...")
        vp_e = vp_edge_weights(edge_npz, vp_r, ast_labels, builtin_labels)
        np.savez(out_dir / f"{out_prefix}vp_edge_weights.npz", **vp_e)
        print(f"  Saved -> {out_prefix}vp_edge_weights.npz")

    # ── purity masks ──────────────────────────────────────────────────────────
    print("\nBuilding purity masks ...")
    masks = build_purity_masks(vp_r, vp_h, vp_n,
                               threshold=args.threshold, purity=args.purity)
    save_masks = {}
    for qty, m in masks.items():
        for mtype, arr in m.items():
            save_masks[f"{qty}_{mtype}"] = arr
    np.savez(out_dir / f"{out_prefix}purity_masks.npz", **save_masks)
    print(f"  Saved -> {out_prefix}purity_masks.npz")

    for qty, m in masks.items():
        print(f"  {qty}:")
        for mtype, arr in m.items():
            print(f"    {mtype:<16s} {arr.mean():.1%} of units")

    # ── builtin token-type subset VP (residual only) ──────────────────────────
    print("\n[+] Builtin token-type subset VP (residual) ...")
    single_summary, single_vp_raw = vp_subset_summary(
        resid_all, ast_labels, builtin_labels, builtin_token_types, "single")
    multi_summary, multi_vp_raw = vp_subset_summary(
        resid_all, ast_labels, builtin_labels, builtin_token_types, "multi")
    print(f"  single-token builtins (n={single_summary.get('n',0)}): "
          f"unique_builtin={single_summary.get('unique_builtin_mean', 0):.3f}  "
          f"unique_ast={single_summary.get('unique_ast_mean', 0):.3f}")
    print(f"  multi-token  builtins (n={multi_summary.get('n',0)}): "
          f"unique_builtin={multi_summary.get('unique_builtin_mean', 0):.3f}  "
          f"unique_ast={multi_summary.get('unique_ast_mean', 0):.3f}")

    # ── summary JSON ──────────────────────────────────────────────────────────
    summary_path = out_dir / f"{out_prefix}summary.json"
    full_summary = {
        "vp": summaries,
        "builtin_token_types": builtin_token_types,
        "builtin_single_token_vp": single_summary,
        "builtin_multi_token_vp":  multi_summary,
    }
    with open(summary_path, "w") as f:
        json.dump(full_summary, f, indent=2)
    print(f"\nSummary -> {summary_path}")
    for s in summaries:
        print(f"  [{s['quantity']}]  "
              f"unique_ast={s['unique_ast_mean']:.3f}  "
              f"unique_builtin={s['unique_builtin_mean']:.3f}  "
              f"shared={s['shared_mean']:.3f}  "
              f"r2_joint={s['r2_joint_mean']:.3f}")
    print(f"  [single-token builtins]  unique_builtin={single_summary.get('unique_builtin_mean',0):.3f}")
    print(f"  [multi-token  builtins]  unique_builtin={multi_summary.get('unique_builtin_mean',0):.3f}")

    # ── plots ─────────────────────────────────────────────────────────────────
    if args.plot:
        print("\nGenerating plots ...")

        # 1. Stacked bar: mean VP per layer for each quantity
        for name, vp in [("residual", vp_r), ("heads", vp_h),
                         ("mlp_attr", vp_m), ("mlp_neurons", vp_n)]:
            plot_vp_stacked_bar(
                vp, title=f"Variance partition — {name}",
                xlabel="Layer",
                save_path=img_dir / f"{out_prefix}plot_stacked_{name}.png",
            )

        # 2. Heatmaps: unique_ast and unique_builtin for residual and neurons
        for name, vp in [("residual", vp_r), ("mlp_neurons", vp_n)]:
            for component in ["unique_ast", "unique_builtin", "shared"]:
                plot_vp_heatmap(
                    vp, component=component,
                    title=f"{component} — {name}  (each cell = one layer x unit)",
                    save_path=img_dir / f"{out_prefix}plot_heatmap_{name}_{component}.png",
                )

        # 3. Before/after VP scatter (one file per layer, or a specific layer)
        L = resid_all.shape[1]
        sl = args.scatter_layer
        scatter_indices = (
            range(L) if sl is None
            else [sl if sl >= 0 else L + sl]
        )
        for l_idx in scatter_indices:
            layer_vp = {k: v[l_idx] for k, v in vp_r.items()}   # (H,) arrays
            lbl = "emb" if l_idx == 0 else str(l_idx)
            plot_before_after_vp(
                Y_raw=resid_all[:, l_idx, :],
                vp=layer_vp,
                ast_labels=ast_labels,
                builtin_labels=builtin_labels,
                title=f"Before vs after VP projection — residual layer {lbl}",
                save_path=img_dir / f"{out_prefix}plot_before_after_layer{lbl}.png",
            )

        # 4. Top units
        for name, vp in [("residual", vp_r), ("heads", vp_h), ("neurons", vp_n)]:
            for comp in ["unique_ast", "unique_builtin", "shared"]:
                plot_top_units(
                    vp, component=comp,
                    title=f"Top units — {name} — {comp}",
                    save_path=img_dir / f"{out_prefix}plot_top_{name}_{comp}.png",
                )

        # 5. Purity masks
        for qty in ["residual", "heads", "neurons"]:
            plot_purity_masks(
                masks, quantity=qty,
                save_path=img_dir / f"{out_prefix}plot_purity_{qty}.png",
            )

        # 6. Builtin token-type comparison
        plot_builtin_token_type_comparison(
            single_summary, multi_summary, builtin_token_types,
            raw_single=single_vp_raw, raw_multi=multi_vp_raw,
            save_path=img_dir / f"{out_prefix}plot_builtin_token_type.png",
        )

        # 7. Edge importance (top-k bar charts, heatmap, flow)
        if has_edges:
            for comp in ["edge_ast_importance", "edge_builtin_importance", "edge_shared"]:
                plot_edge_importance(
                    vp_e, component=comp,
                    title=f"Top edges — {comp}",
                    save_path=img_dir / f"{out_prefix}plot_edge_{comp}.png",
                )
            plot_edge_pathway_heatmap(
                vp_e,
                save_path=img_dir / f"{out_prefix}plot_edge_pathway_heatmap.png",
            )
            plot_edge_pathway_flow(
                vp_e,
                save_path=img_dir / f"{out_prefix}plot_edge_pathway_flow.png",
            )

        print("  All plots saved.")

    # ── final file listing ────────────────────────────────────────────────────
    print("\nOutput files:")
    for p in sorted(out_dir.glob(f"{out_prefix}*")):
        print(f"  {p.name:<55s}  {p.stat().st_size / 1e6:.1f} MB")

    print("""
Load in downstream scripts (step 03):
  vp_r = np.load('02_{stem}_vp_residual.npz')
  # keys: unique_ast, unique_builtin, shared, unexplained, r2_ast, r2_builtin, r2_joint
  # shape per key: (L+1, H)

  masks = np.load('02_{stem}_purity_masks.npz')
  # keys: residual_ast_pure, residual_builtin_pure, residual_shared, ...
  # shape per key: (L+1, H)  — boolean

  # To get "clean" AST activations for layer L:
  ast_dims = masks['residual_ast_pure'][L]             # boolean (H,)
  Y_ast_clean = resid_all[:, L, ast_dims]              # (N, n_ast_dims)
""")


if __name__ == "__main__":
    main()
