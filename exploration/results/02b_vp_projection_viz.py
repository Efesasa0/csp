"""
02b_vp_projection_viz.py  (lives in results/)

"Before and After VP-based Projection" — 3-D scatter visualisation.

For every (quantity, layer, representation, dim_set, reduction) combination
this script produces a 2×2 grid of 3-D scatter plots:

  ┌──────────────────────────┬──────────────────────────┐
  │ [0,0] Combined           │ [0,1] AST only           │
  │   AST (○) + Builtin (▲)  │   coloured by AST        │
  │   on AST axes            │   on AST axes            │
  ├──────────────────────────┼──────────────────────────┤
  │ [1,0] Builtin            │ [1,1] Builtin            │
  │   coloured by builtin    │   coloured by builtin    │
  │   on AST axes            │   on Builtin axes        │
  └──────────────────────────┴──────────────────────────┘

"AST axes"     = top-3 PCA / UMAP components fitted on the AST-relevant
                 dimension subset of the data.
"Builtin axes" = same, fitted on the builtin-relevant subset.

The two dimension subsets are:
  Raw repr.  – top dims ranked by unique_ast / unique_builtin VP score.
  Pure repr. – dims flagged ast_pure / builtin_pure by the VP purity mask.

Two dim-set sizes are produced for each combination:
  full    – all dims in the subset (2048 for raw residual, all pure dims)
  top512  – the top-512 dims from the subset (by VP unique score)

In addition to the PNG plots, for every (quantity, layer) a JSON file is
written listing the active neuron IDs and their VP rank:

  <out_dir>/neuron_ids/<qty>_L<ll>_neuron_ids.json

  Each entry has the form:
    { "id": "L3[Residual]1024", "rank": 1, "vp_score": 0.312,
      "component": "unique_ast|unique_builtin", "pure": true|false }

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUANTITIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  residual   – 01_<stem>_residual_all.npy    (N, L+1, H=2048)
  mlp        – 01_<stem>_mlp_neurons.npy     (N, L,   mlp_dim=8192)

INPUTS (from steps 01 and 02, path relative to --in_dir)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  01_<stem>_residual_all.npy
  01_<stem>_mlp_neurons.npy
  01_<stem>_meta.json
  02_<stem>_vp_residual.npz     keys: unique_ast, unique_builtin  (L+1, H)
  02_<stem>_vp_mlp_neurons.npz  keys: unique_ast, unique_builtin  (L, mlp_dim)
  02_<stem>_purity_masks.npz    keys: residual_ast_pure, residual_builtin_pure,
                                      neurons_ast_pure,  neurons_builtin_pure

OUTPUT  (all under --out_dir, default: <script_dir>/output/)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  figures/<qty>/
      <qty>_L<ll>_<repr>_<dims>_<method>.png   (2×2 scatter)
  neuron_ids/
      <qty>_L<ll>_neuron_ids.json               (ranked active-neuron table)

Usage
-----
  python 02b_vp_projection_viz.py --stem contrastive_stubs
  python 02b_vp_projection_viz.py --stem contrastive_stubs --layer -1 --methods pca
  python 02b_vp_projection_viz.py --stem contrastive_stubs --qty mlp --skip_full_umap
  python 02b_vp_projection_viz.py --stem contrastive_stubs --ids_only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

try:
    import umap as _umap_mod
    _UMAP_OK = True
except ImportError:
    _UMAP_OK = False
    print("[warn] umap-learn not found — UMAP plots will be skipped. "
          "Install with: pip install umap-learn")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

N_COMP   = 3      # 3-D scatter
TOP_K    = 512    # "top-512" dim set
UMAP_MAX_DIM_FULL = 2048   # skip UMAP for 'full' configs with more dims than this

_MARK_AST = "o"   # circle     — AST labels
_MARK_B   = "^"   # triangle   — Builtin labels
_ALPHA    = 0.50
_SIZE     = 12

# Label used in neuron IDs
_QTY_LABEL = {"residual": "Residual", "mlp": "MLP"}


# ─────────────────────────────────────────────────────────────────────────────
# Neuron-ID export
# ─────────────────────────────────────────────────────────────────────────────

def _neuron_id(layer: int, qty: str, neuron: int) -> str:
    """Format a neuron identifier, e.g. 'L3[MLP]1024' or 'L0[Residual]512'."""
    return f"L{layer}{_QTY_LABEL[qty]}{neuron}"


def export_neuron_ids(
    layer:       int,
    qty:         str,
    vp_ua:       np.ndarray,        # (H,) unique_ast VP scores
    vp_ub:       np.ndarray,        # (H,) unique_builtin VP scores
    ast_mask:    np.ndarray | None, # (H,) bool  ast_pure
    b_mask:      np.ndarray | None, # (H,) bool  builtin_pure
    out_dir:     Path,
    act:         np.ndarray | None = None,  # (N, H)
    ast_labels:  list | None = None,
    b_labels:    list | None = None,
) -> Path:
    """
    Write <out_dir>/neuron_ids/<qty>_L<ll>_neuron_ids.json.

    Clean two-section structure — AST neurons never appear in the builtin
    section and vice versa:

    {
      "layer": 3,  "qty": "residual",

      "ast": {
        "overall_ranking": [
          { "id":"L3[Residual]512", "neuron":512, "vp_rank":1,
            "vp_score":0.31, "pure":true, "overall_mean_act":0.45,
            "overall_act_rank":2 },
          ...
        ],
        "by_class": {
          "For":   [ {"id":…,"neuron":…,"vp_rank":…,"vp_score":…,"pure":…,
                      "mean_act":0.67,"class_act_rank":1}, … ],
          "While": [ … ],
          ...
        }
      },

      "builtin": {
        "overall_ranking": [ … ],   ← builtin neurons only, ranked by
                                       mean_act across ALL prompts
        "by_class": {
          "len":    [ … ],
          "sorted": [ … ],
          ...
        }
      }
    }

    overall_ranking  — neurons with vp_score > 0, sorted by overall_mean_act
                       descending; overall_act_rank is that position.
    by_class         — for each class, the SAME neuron set but re-sorted by
                       that class's mean_act; class_act_rank is that position.
    """
    ast_classes     = sorted(set(ast_labels)) if ast_labels else []
    builtin_classes = sorted(set(b_labels))   if b_labels   else []
    ast_arr = np.array(ast_labels) if ast_labels else None
    b_arr   = np.array(b_labels)   if b_labels   else None

    def _build_section(
        scores:     np.ndarray,    # (H,) VP scores for this factor
        mask:       np.ndarray | None,
        class_list: list[str],
        label_arr:  np.ndarray | None,
    ) -> dict:
        # Neurons with any positive VP score
        vp_order = np.argsort(scores)[::-1]
        pure_set = set(np.where(mask)[0]) if mask is not None else set()

        base_rows: list[dict] = []
        for vp_rank, nidx in enumerate(vp_order, start=1):
            sc = float(scores[nidx])
            if sc <= 0:
                break
            base_rows.append({
                "id":       _neuron_id(layer, qty, int(nidx)),
                "neuron":   int(nidx),
                "vp_rank":  vp_rank,
                "vp_score": round(sc, 6),
                "pure":     int(nidx) in pure_set,
            })

        if not base_rows:
            return {"overall_ranking": [], "by_class": {}}

        neuron_indices = np.array([r["neuron"] for r in base_rows])

        # ── Overall ranking: sort by mean_act across ALL prompts ──────────────
        if act is not None:
            overall_mean = act[:, neuron_indices].mean(axis=0)   # (n_active,)
            act_order    = np.argsort(overall_mean)[::-1]
            overall_ranking = []
            for act_rank, pos in enumerate(act_order, start=1):
                overall_ranking.append({
                    **base_rows[pos],
                    "overall_mean_act":  round(float(overall_mean[pos]), 6),
                    "overall_act_rank":  act_rank,
                })
        else:
            overall_ranking = [{**r, "overall_mean_act": None,
                                 "overall_act_rank": None} for r in base_rows]

        # ── Per-class ranking: re-sort same neurons by class mean_act ─────────
        by_class: dict[str, list] = {}
        if act is not None and label_arr is not None:
            for cls in class_list:
                cls_idx  = np.where(label_arr == cls)[0]
                if len(cls_idx) == 0:
                    continue
                cls_mean = act[cls_idx, :][:, neuron_indices].mean(axis=0)
                act_ord  = np.argsort(cls_mean)[::-1]
                by_class[cls] = [
                    {**base_rows[pos],
                     "mean_act":       round(float(cls_mean[pos]), 6),
                     "class_act_rank": class_rank}
                    for class_rank, pos in enumerate(act_ord, start=1)
                ]

        return {"overall_ranking": overall_ranking, "by_class": by_class}

    result = {
        "layer": layer,
        "qty":   qty,
        "ast":     _build_section(vp_ua, ast_mask,  ast_classes,     ast_arr),
        "builtin": _build_section(vp_ub, b_mask,    builtin_classes, b_arr),
    }

    ids_dir   = out_dir / "neuron_ids"
    ids_dir.mkdir(parents=True, exist_ok=True)
    save_path = ids_dir / f"{qty}_L{layer:02d}_neuron_ids.json"
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2)

    na = len(result["ast"]["overall_ranking"])
    nb = len(result["builtin"]["overall_ranking"])
    print(f"    → neuron_ids/{save_path.name}  "
          f"(ast:{na} neurons, builtin:{nb} neurons)")
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# Colour / legend helpers
# ─────────────────────────────────────────────────────────────────────────────

def _label_colors(labels: list, cmap_name: str = "tab20"):
    """Map label strings → RGBA array, also return unique list and lut."""
    unique = sorted(set(labels))
    n      = max(len(unique), 2)
    cmap   = plt.get_cmap(cmap_name, n)
    lut    = {u: cmap(i / (n - 1)) for i, u in enumerate(unique)}
    colors = np.array([lut[lb] for lb in labels])
    return colors, unique, lut


def _legend_handles(unique: list, lut: dict, marker: str,
                    max_show: int = 18) -> list:
    handles = [
        Line2D([0], [0], marker=marker, color="w",
               markerfacecolor=lut[u], markersize=7, label=u)
        for u in unique[:max_show]
    ]
    if len(unique) > max_show:
        handles.append(Line2D([0], [0], color="none",
                               label=f"… +{len(unique) - max_show} more"))
    return handles


# ─────────────────────────────────────────────────────────────────────────────
# Dimension selection
# ─────────────────────────────────────────────────────────────────────────────

def _top_dims(scores: np.ndarray, n: int) -> np.ndarray:
    """Indices of the top-n dims by VP score (descending)."""
    if scores.shape[0] <= n:
        return np.arange(scores.shape[0])
    return np.argsort(scores)[::-1][:n]


# ─────────────────────────────────────────────────────────────────────────────
# Dimensionality reduction — PCA
# ─────────────────────────────────────────────────────────────────────────────

def _pca_fit(X: np.ndarray) -> tuple:
    """Fit PCA(3) on X, return (pca, Z)."""
    nc  = min(N_COMP, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=nc, random_state=42)
    Z   = pca.fit_transform(X.astype(np.float32))
    if Z.shape[1] < N_COMP:
        Z = np.pad(Z, ((0, 0), (0, N_COMP - Z.shape[1])))
    return pca, Z


# ─────────────────────────────────────────────────────────────────────────────
# Dimensionality reduction — UMAP
# ─────────────────────────────────────────────────────────────────────────────

def _umap_fit(X: np.ndarray, seed: int = 42) -> tuple:
    n_neighbors = min(15, X.shape[0] - 1)
    reducer = _umap_mod.UMAP(
        n_components=N_COMP,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        random_state=seed,
        verbose=False,
    )
    Z = reducer.fit_transform(X.astype(np.float32))
    return reducer, Z


# ─────────────────────────────────────────────────────────────────────────────
# 3-D scatter panel helper
# ─────────────────────────────────────────────────────────────────────────────

def _scatter3(ax, Z: np.ndarray, colors, marker: str,
              alpha: float = _ALPHA, s: float = _SIZE) -> None:
    ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2],
               c=colors, marker=marker,
               s=s, alpha=alpha, linewidths=0, depthshade=True)


def _style_ax(ax, method: str, title: str) -> None:
    label = "PC" if method == "pca" else "UMAP"
    ax.set_xlabel(f"{label}1", fontsize=7, labelpad=1)
    ax.set_ylabel(f"{label}2", fontsize=7, labelpad=1)
    ax.set_zlabel(f"{label}3", fontsize=7, labelpad=1)
    ax.tick_params(labelsize=5)
    ax.set_title(title, fontsize=8, pad=3)


# ─────────────────────────────────────────────────────────────────────────────
# 2×2 figure
# ─────────────────────────────────────────────────────────────────────────────

def _make_2x2(
    Z_ast:        np.ndarray,  # (N, 3) — embedding on AST axes
    Z_b:          np.ndarray,  # (N, 3) — embedding on Builtin axes
    ast_labels:   list,
    b_labels:     list,
    method:       str,         # "pca" or "umap"
    suptitle:     str,
    save_path:    Path,
    ast_dim_info: str = "",
    b_dim_info:   str = "",
) -> None:
    """
    2×2 grid of 3-D scatter plots:

      [0,0]  Combined  — AST (○) + Builtin (▲) both plotted on AST axes
      [0,1]  AST only  — AST labels, AST axes
      [1,0]  Builtin   — Builtin labels, AST axes  (leakage probe)
      [1,1]  Builtin   — Builtin labels, Builtin axes
    """
    ast_c, ast_u, ast_lut = _label_colors(ast_labels, "tab20")
    b_c,   b_u,   b_lut   = _label_colors(b_labels,   "tab20b")

    fig = plt.figure(figsize=(17, 12))
    fig.suptitle(suptitle, fontsize=9, y=0.995)

    # ── [0,0] Combined ────────────────────────────────────────────────────────
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    _scatter3(ax, Z_ast, ast_c, _MARK_AST, alpha=0.40, s=_SIZE)
    _scatter3(ax, Z_ast, b_c,   _MARK_B,   alpha=0.40, s=_SIZE)
    _style_ax(ax, method, f"Combined: AST (○) + Builtin (▲)\nAST axes  {ast_dim_info}")
    ax.legend(handles=[
        Line2D([0], [0], marker=_MARK_AST, color="w",
               markerfacecolor="steelblue", ms=8, label="AST labels (○)"),
        Line2D([0], [0], marker=_MARK_B,   color="w",
               markerfacecolor="darkorange", ms=8, label="Builtin labels (▲)"),
    ], fontsize=7, loc="upper left")

    # ── [0,1] AST only on AST axes ────────────────────────────────────────────
    ax = fig.add_subplot(2, 2, 2, projection="3d")
    _scatter3(ax, Z_ast, ast_c, _MARK_AST)
    _style_ax(ax, method, f"AST labels — AST axes  {ast_dim_info}")
    ax.legend(handles=_legend_handles(ast_u, ast_lut, _MARK_AST),
              fontsize=5, loc="upper left", ncol=2)

    # ── [1,0] Builtin on AST axes ─────────────────────────────────────────────
    ax = fig.add_subplot(2, 2, 3, projection="3d")
    _scatter3(ax, Z_ast, b_c, _MARK_B)
    _style_ax(ax, method, f"Builtin labels — AST axes  {ast_dim_info}")
    ax.legend(handles=_legend_handles(b_u, b_lut, _MARK_B),
              fontsize=5, loc="upper left", ncol=2)

    # ── [1,1] Builtin on Builtin axes ─────────────────────────────────────────
    ax = fig.add_subplot(2, 2, 4, projection="3d")
    _scatter3(ax, Z_b, b_c, _MARK_B)
    _style_ax(ax, method, f"Builtin labels — Builtin axes  {b_dim_info}")
    ax.legend(handles=_legend_handles(b_u, b_lut, _MARK_B),
              fontsize=5, loc="upper left", ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.985])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"    → figures/{save_path.parent.name}/{save_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Per-layer processor
# ─────────────────────────────────────────────────────────────────────────────

def _process_layer(
    act:            np.ndarray,          # (N, H) activations at this layer
    vp_ua:          np.ndarray,          # (H,) unique_ast VP scores
    vp_ub:          np.ndarray,          # (H,) unique_builtin VP scores
    ast_mask:       np.ndarray | None,   # (H,) bool ast_pure
    b_mask:         np.ndarray | None,   # (H,) bool builtin_pure
    ast_labels:     list,
    b_labels:       list,
    qty:            str,                 # "residual" or "mlp"
    layer:          int,
    out_dir:        Path,                # root output dir for this run
    methods:        list[str],
    skip_full_umap: bool,
    ids_only:       bool,
) -> None:
    """Compute neuron-ID export and all 2×2 figures for one (quantity, layer)."""
    N, H = act.shape

    # ── 1. Export neuron IDs with per-class breakdown (always) ────────────────
    export_neuron_ids(layer, qty, vp_ua, vp_ub, ast_mask, b_mask, out_dir,
                      act=act, ast_labels=ast_labels, b_labels=b_labels)

    if ids_only:
        return

    # ── 2. Build dimension configs ─────────────────────────────────────────────
    configs: list[tuple] = []

    # Raw: full
    configs.append((
        "raw", "full",
        act, act,
        f"(all {H} dims)", f"(all {H} dims)",
    ))

    # Raw: top-512 (by VP score)
    ast_top_idx = _top_dims(vp_ua, TOP_K)
    b_top_idx   = _top_dims(vp_ub, TOP_K)
    configs.append((
        "raw", "top512",
        act[:, ast_top_idx], act[:, b_top_idx],
        f"(top-{len(ast_top_idx)} by unique_ast)",
        f"(top-{len(b_top_idx)} by unique_builtin)",
    ))

    # Pure: full and top-512 (only if masks available and non-trivial)
    if ast_mask is not None and b_mask is not None:
        n_ap = int(ast_mask.sum())
        n_bp = int(b_mask.sum())
        if n_ap >= N_COMP and n_bp >= N_COMP:
            X_ap = act[:, ast_mask]
            X_bp = act[:, b_mask]
            configs.append((
                "pure", "full",
                X_ap, X_bp,
                f"({n_ap} ast_pure dims)", f"({n_bp} builtin_pure dims)",
            ))
            ap_top = _top_dims(vp_ua[ast_mask], TOP_K)
            bp_top = _top_dims(vp_ub[b_mask],   TOP_K)
            configs.append((
                "pure", "top512",
                X_ap[:, ap_top], X_bp[:, bp_top],
                f"(top-{len(ap_top)} of {n_ap} ast_pure)",
                f"(top-{len(bp_top)} of {n_bp} builtin_pure)",
            ))
        else:
            print(f"    [L{layer:02d}] Skipping pure configs — "
                  f"ast_pure={n_ap}, builtin_pure={n_bp} dims (need ≥ {N_COMP})")

    # ── 3. Fit + plot ──────────────────────────────────────────────────────────
    fig_dir = out_dir / "figures" / qty
    fig_dir.mkdir(parents=True, exist_ok=True)

    for method in methods:
        if method == "umap" and not _UMAP_OK:
            continue
        fit_fn = _pca_fit if method == "pca" else _umap_fit

        for repr_name, dim_set, X_ast, X_b, ast_info, b_info in configs:
            if (method == "umap" and skip_full_umap
                    and dim_set == "full" and X_ast.shape[1] > UMAP_MAX_DIM_FULL):
                print(f"    [L{layer:02d}] Skipping {repr_name}/{dim_set} UMAP "
                      f"(dims={X_ast.shape[1]} > {UMAP_MAX_DIM_FULL})")
                continue

            print(f"    [L{layer:02d}] {qty} {repr_name}/{dim_set} {method} "
                  f"(X_ast={X_ast.shape}, X_b={X_b.shape})", end=" ... ")

            try:
                _, Z_ast = fit_fn(X_ast)
                _, Z_b   = fit_fn(X_b)
            except Exception as e:
                print(f"FAILED: {e}")
                continue
            print("done")

            suptitle = (
                f"{qty.upper()}  |  Layer {layer}  |  "
                f"{repr_name.upper()} {dim_set}  |  {method.upper()}"
            )
            fname     = f"{qty}_L{layer:02d}_{repr_name}_{dim_set}_{method}.png"
            save_path = fig_dir / fname

            _make_2x2(
                Z_ast, Z_b,
                ast_labels, b_labels,
                method, suptitle, save_path,
                ast_info, b_info,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="3-D before/after VP projection visualisation (results/02b_vp_projection_viz.py)"
    )
    parser.add_argument("--stem",     "-s", required=True,
                        help="Dataset stem (e.g. contrastive_stubs)")
    parser.add_argument("--in_dir",  "-d", default="../data",
                        help="Directory with 01_* and 02_* files "
                             "(relative to this script; default: ../data)")
    parser.add_argument("--out_dir", "-o", default="output",
                        help="Output root (relative to this script; default: output/)")
    parser.add_argument("--qty",     default="both",
                        choices=["residual", "mlp", "both"],
                        help="Which quantity to visualise (default: both)")
    parser.add_argument("--layer",   type=int, default=None,
                        help="Single layer index (-1 = final). Omit for ALL layers.")
    parser.add_argument("--methods", nargs="+", default=["pca", "umap"],
                        choices=["pca", "umap"],
                        help="Reduction methods (default: pca umap)")
    parser.add_argument("--skip_full_umap", action="store_true",
                        help=f"Skip UMAP on full-dim configs with > "
                             f"{UMAP_MAX_DIM_FULL} dims")
    parser.add_argument("--ids_only", action="store_true",
                        help="Only export neuron_ids JSON — skip all plotting")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    base       = Path(args.in_dir)
    out_root   = (script_dir / args.out_dir).resolve()
    stem       = args.stem

    methods = list(args.methods)
    if "umap" in methods and not _UMAP_OK:
        print("[warn] umap-learn not available — removing 'umap' from methods.")
        methods = [m for m in methods if m != "umap"]
    if not methods and not args.ids_only:
        raise SystemExit("No reduction methods available.")

    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Input dir : {base}")
    print(f"Output dir: {out_root}")

    # ── Load metadata ──────────────────────────────────────────────────────────
    meta_path = base / f"01_{stem}_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing: {meta_path}")
    with open(meta_path) as f:
        meta = json.load(f)
    ast_labels = [m["ast_node"]    for m in meta]
    b_labels   = [m["builtin_obj"] for m in meta]
    N = len(meta)
    print(f"  N={N}  AST classes={len(set(ast_labels))}  "
          f"Builtin classes={len(set(b_labels))}")

    # ── Load VP scores ─────────────────────────────────────────────────────────
    vp_r, vp_n = None, None
    vp_r_path  = base / f"02_{stem}_vp_residual.npz"
    vp_n_path  = base / f"02_{stem}_vp_mlp_neurons.npz"
    if vp_r_path.exists():
        vp_r = dict(np.load(vp_r_path))
        print(f"  VP residual loaded  shape={vp_r['unique_ast'].shape}")
    elif args.qty in ("residual", "both"):
        print(f"  [warn] VP residual not found — dim ranking will use zeros")
    if vp_n_path.exists():
        vp_n = dict(np.load(vp_n_path))
        print(f"  VP mlp_neurons loaded  shape={vp_n['unique_ast'].shape}")
    elif args.qty in ("mlp", "both"):
        print(f"  [warn] VP mlp_neurons not found — dim ranking will use zeros")

    # ── Load purity masks ──────────────────────────────────────────────────────
    masks_npz  = None
    masks_path = base / f"02_{stem}_purity_masks.npz"
    if masks_path.exists():
        masks_npz = dict(np.load(masks_path, allow_pickle=True))
        print(f"  Purity masks loaded from {masks_path.name}")
    else:
        print(f"  [warn] Purity masks not found — pure configs will be skipped.")

    # ── Process residual stream ────────────────────────────────────────────────
    if args.qty in ("residual", "both"):
        resid_path = base / f"01_{stem}_residual_all.npy"
        if not resid_path.exists():
            print(f"  [skip] residual_all.npy not found at {resid_path}")
        else:
            resid_all = np.load(resid_path)        # (N, L+1, H)
            _, L1, H  = resid_all.shape
            print(f"\nResidual stream: shape={resid_all.shape}")

            layers_r = list(range(L1))
            if args.layer is not None:
                l_idx    = L1 + args.layer if args.layer < 0 else args.layer
                layers_r = [int(np.clip(l_idx, 0, L1 - 1))]

            for l in layers_r:
                print(f"  [residual] Layer {l}/{L1-1}")
                act   = resid_all[:, l, :]
                vp_ua = vp_r["unique_ast"][l]     if vp_r else np.zeros(H)
                vp_ub = vp_r["unique_builtin"][l] if vp_r else np.zeros(H)
                ast_m = (masks_npz["residual_ast_pure"][l].astype(bool)
                         if masks_npz and "residual_ast_pure" in masks_npz else None)
                b_m   = (masks_npz["residual_builtin_pure"][l].astype(bool)
                         if masks_npz and "residual_builtin_pure" in masks_npz else None)

                _process_layer(
                    act, vp_ua, vp_ub, ast_m, b_m,
                    ast_labels, b_labels,
                    qty="residual", layer=l,
                    out_dir=out_root,
                    methods=methods,
                    skip_full_umap=args.skip_full_umap,
                    ids_only=args.ids_only,
                )
            del resid_all

    # ── Process MLP neurons ────────────────────────────────────────────────────
    if args.qty in ("mlp", "both"):
        mlp_path = base / f"01_{stem}_mlp_neurons.npy"
        if not mlp_path.exists():
            print(f"  [skip] mlp_neurons.npy not found at {mlp_path}")
        else:
            mlp_all = np.load(mlp_path)            # (N, L, mlp_dim)
            _, L, M = mlp_all.shape
            print(f"\nMLP neurons: shape={mlp_all.shape}")

            layers_m = list(range(L))
            if args.layer is not None:
                l_idx    = L + args.layer if args.layer < 0 else args.layer
                layers_m = [int(np.clip(l_idx, 0, L - 1))]

            for l in layers_m:
                print(f"  [mlp] Layer {l}/{L-1}")
                act   = mlp_all[:, l, :]
                vp_ua = vp_n["unique_ast"][l]     if vp_n else np.zeros(M)
                vp_ub = vp_n["unique_builtin"][l] if vp_n else np.zeros(M)
                ast_m = (masks_npz["neurons_ast_pure"][l].astype(bool)
                         if masks_npz and "neurons_ast_pure" in masks_npz else None)
                b_m   = (masks_npz["neurons_builtin_pure"][l].astype(bool)
                         if masks_npz and "neurons_builtin_pure" in masks_npz else None)

                _process_layer(
                    act, vp_ua, vp_ub, ast_m, b_m,
                    ast_labels, b_labels,
                    qty="mlp", layer=l,
                    out_dir=out_root,
                    methods=methods,
                    skip_full_umap=args.skip_full_umap,
                    ids_only=args.ids_only,
                )

    print(f"\nDone.")
    print(f"  Figures  : {out_root / 'figures'}")
    print(f"  Neuron IDs: {out_root / 'neuron_ids'}")


if __name__ == "__main__":
    main()
