"""
02c_vp_interactive_viz.py  (lives in results/)

Interactive 3-D "Before and After VP Projection" visualiser.

Generates one self-contained HTML file per (quantity, layer) using Plotly.
Each file has four dropdown/button controls:

  Representation  ─ Raw  |  Pure  |  Both
  Factor          ─ AST  |  Builtin  |  Both
  Method          ─ PCA  |  UMAP
  Dim set         ─ Full  |  Top-512

Every unique (repr, method, dim_set, factor) combination is pre-computed and
stored as Plotly traces that are shown/hidden by the buttons without any
server round-trip.

MST overlay
-----------
For each category (AST node or builtin name) an MST (minimum spanning tree)
is computed on the 3-D embedding using Euclidean distances.  The MST edges
are added as semi-transparent line traces that visually "join" the cloud of
points belonging to the same category.

Output
------
  <out_dir>/interactive/
      <qty>_L<ll>_interactive.html

Usage
-----
  python 02c_vp_interactive_viz.py --stem contrastive_stubs
  python 02c_vp_interactive_viz.py --stem contrastive_stubs --layer -1 --methods pca
  python 02c_vp_interactive_viz.py --stem contrastive_stubs --qty mlp --skip_full_umap
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from itertools import product as iproduct

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio
    _PLOTLY_OK = True
except ImportError:
    _PLOTLY_OK = False
    print("[error] plotly not found.  Install with: pip install plotly")

try:
    import umap as _umap_mod
    _UMAP_OK = True
except ImportError:
    _UMAP_OK = False
    print("[warn] umap-learn not found — UMAP embeddings will be skipped. "
          "Install with: pip install umap-learn")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

N_COMP            = 3
TOP_K             = 512
UMAP_MAX_DIM_FULL = 2048   # skip UMAP when full-dim count exceeds this

_MARK_AST = "circle"
_MARK_B   = "diamond"
_OPACITY  = 0.70
_SIZE_PT  = 4
_MST_OPACITY = 0.18
_MST_WIDTH   = 1.0

# 29 tab-style colours for AST nodes (distinct enough for dense plots)
_PALETTE_AST = [
    "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
    "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
    "#aec7e8","#ffbb78","#98df8a","#ff9896","#c5b0d5",
    "#c49c94","#f7b6d2","#c7c7c7","#dbdb8d","#9edae5",
    "#393b79","#637939","#8c6d31","#843c39","#7b4173",
    "#5254a3","#8ca252","#bd9e39","#ad494a",
]
_PALETTE_B = [
    "#e6194b","#3cb44b","#ffe119","#4363d8","#f58231",
    "#911eb4","#46f0f0","#f032e6","#bcf60c","#fabebe",
    "#008080","#e6beff","#9a6324","#fffac8","#800000",
    "#aaffc3","#808000","#ffd8b1","#000075","#808080",
    "#ffffff","#000000","#a9a9a9","#00bfff","#ff69b4",
    "#7b68ee","#48d1cc","#ff6347","#40e0d0","#da70d6",
    "#98fb98","#ffdab9","#ff4500","#2e8b57","#b0c4de",
    "#dda0dd","#87ceeb","#778899","#b0e0e6","#ffa07a",
    "#20b2aa","#87cefa","#778899","#fffacd","#90ee90",
    "#ffb6c1","#ffa500","#ff8c00","#db7093","#c71585",
    "#b22222","#dc143c","#8b0000","#ff0000","#ff6347",
]


def _color_lut(names: list[str], palette: list[str]) -> dict[str, str]:
    return {n: palette[i % len(palette)] for i, n in enumerate(sorted(set(names)))}


# ─────────────────────────────────────────────────────────────────────────────
# Dimensionality reduction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _top_dims(scores: np.ndarray, n: int) -> np.ndarray:
    if scores.shape[0] <= n:
        return np.arange(scores.shape[0])
    return np.argsort(scores)[::-1][:n]


def _pca3(X: np.ndarray) -> np.ndarray:
    nc  = min(N_COMP, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=nc, random_state=42)
    Z   = pca.fit_transform(X.astype(np.float32))
    if Z.shape[1] < N_COMP:
        Z = np.pad(Z, ((0, 0), (0, N_COMP - Z.shape[1])))
    return Z


def _umap3(X: np.ndarray) -> np.ndarray:
    n_neighbors = min(15, X.shape[0] - 1)
    reducer = _umap_mod.UMAP(
        n_components=N_COMP,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        random_state=42,
        verbose=False,
    )
    return reducer.fit_transform(X.astype(np.float32))


# ─────────────────────────────────────────────────────────────────────────────
# MST edge generator
# ─────────────────────────────────────────────────────────────────────────────

def _mst_edges(Z: np.ndarray) -> list[tuple[int, int]]:
    """Return list of (i, j) pairs forming the MST of points Z."""
    if len(Z) < 2:
        return []
    D    = cdist(Z, Z, metric="euclidean")
    mst  = minimum_spanning_tree(D)
    cx   = mst.tocoo()
    return list(zip(cx.row.tolist(), cx.col.tolist()))


def _mst_traces_for_category(
    Z:      np.ndarray,   # (N, 3) all points
    mask:   np.ndarray,   # (N,) bool — which points belong to this category
    color:  str,
    name:   str,
    visible: bool = True,
) -> list:
    """Build Plotly line traces for the MST within one category's point cloud."""
    idx = np.where(mask)[0]
    if len(idx) < 2:
        return []
    Zc    = Z[idx]
    edges = _mst_edges(Zc)
    traces = []
    for (a, b) in edges:
        traces.append(go.Scatter3d(
            x=[float(Zc[a, 0]), float(Zc[b, 0]), None],
            y=[float(Zc[a, 1]), float(Zc[b, 1]), None],
            z=[float(Zc[a, 2]), float(Zc[b, 2]), None],
            mode="lines",
            line=dict(color=color, width=_MST_WIDTH),
            opacity=_MST_OPACITY,
            showlegend=False,
            hoverinfo="skip",
            name=f"MST:{name}",
            visible=visible,
        ))
    return traces


# ─────────────────────────────────────────────────────────────────────────────
# Scatter trace builder
# ─────────────────────────────────────────────────────────────────────────────

def _scatter_trace(
    Z:       np.ndarray,   # (N, 3)
    labels:  list[str],
    color_lut: dict[str, str],
    marker_sym: str,
    factor_label: str,      # "AST" or "Builtin"
    visible: bool = True,
) -> list:
    """One Scatter3d trace per unique label, coloured consistently."""
    traces = []
    for name in sorted(set(labels)):
        mask = np.array([lb == name for lb in labels])
        pts  = Z[mask]
        traces.append(go.Scatter3d(
            x=pts[:, 0].tolist(),
            y=pts[:, 1].tolist(),
            z=pts[:, 2].tolist(),
            mode="markers",
            name=f"{factor_label}: {name}",
            legendgroup=f"{factor_label}:{name}",
            marker=dict(
                size=_SIZE_PT,
                color=color_lut[name],
                symbol=marker_sym,
                opacity=_OPACITY,
                line=dict(width=0),
            ),
            hovertemplate=(
                f"<b>{factor_label}: {name}</b><br>"
                "x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>"
            ),
            visible=visible,
        ))
    return traces


# ─────────────────────────────────────────────────────────────────────────────
# Per-layer HTML builder
# ─────────────────────────────────────────────────────────────────────────────

def _axis_label(method: str, i: int) -> str:
    return f"{'PC' if method == 'pca' else 'UMAP'}{i}"


def build_interactive(
    configs:     dict,   # key=(repr_name, dim_set, method) → {"Z_ast":…,"Z_b":…}
    ast_labels:  list[str],
    b_labels:    list[str],
    qty:         str,
    layer:       int,
    out_path:    Path,
) -> None:
    """
    Build a single Plotly HTML with button panels for:
      Representation (raw | pure | both)
      Factor         (AST | Builtin | both)
      Method         (pca | umap)
      Dim set        (full | top512)

    Each button combination sets the visible property of all traces.
    MST edges are included at low opacity.
    """
    ast_lut = _color_lut(ast_labels, _PALETTE_AST)
    b_lut   = _color_lut(b_labels,   _PALETTE_B)

    # ── Build all traces, keyed by combo so we can toggle ──────────────────────
    # trace_meta: list of (trace_obj, repr_name, dim_set, method, factor_str)
    trace_meta: list[tuple] = []

    for (repr_name, dim_set, method), zdata in sorted(configs.items()):
        Z_ast = zdata["Z_ast"]   # (N, 3)
        Z_b   = zdata["Z_b"]     # (N, 3)
        # Start all hidden; buttons will flip visibility
        vis = False

        # AST scatter + MST
        for tr in _scatter_trace(Z_ast, ast_labels, ast_lut, _MARK_AST, "AST", vis):
            trace_meta.append((tr, repr_name, dim_set, method, "ast"))
        for name in sorted(set(ast_labels)):
            mask = np.array([lb == name for lb in ast_labels])
            for tr in _mst_traces_for_category(Z_ast, mask, ast_lut[name], name, vis):
                trace_meta.append((tr, repr_name, dim_set, method, "ast_mst"))

        # Builtin scatter + MST  (on builtin axes, i.e. Z_b)
        for tr in _scatter_trace(Z_b, b_labels, b_lut, _MARK_B, "Builtin", vis):
            trace_meta.append((tr, repr_name, dim_set, method, "builtin"))
        for name in sorted(set(b_labels)):
            mask = np.array([lb == name for lb in b_labels])
            for tr in _mst_traces_for_category(Z_b, mask, b_lut[name], name, vis):
                trace_meta.append((tr, repr_name, dim_set, method, "builtin_mst"))

    traces = [t for t, *_ in trace_meta]
    fig    = go.Figure(data=traces)

    # ── Collect available option values ────────────────────────────────────────
    repr_vals  = sorted({m[1] for m in trace_meta})
    dim_vals   = sorted({m[2] for m in trace_meta})
    meth_vals  = sorted({m[3] for m in trace_meta})

    def _vis_array(repr_sel, dim_sel, meth_sel, factor_sel):
        """Return list of True/False matching which traces to show."""
        result = []
        for _, rn, ds, mth, fac in trace_meta:
            repr_ok   = (repr_sel == "both") or (rn == repr_sel)
            dim_ok    = (ds == dim_sel)
            meth_ok   = (mth == meth_sel)
            if factor_sel == "ast":
                fac_ok = fac in ("ast", "ast_mst")
            elif factor_sel == "builtin":
                fac_ok = fac in ("builtin", "builtin_mst")
            else:  # both
                fac_ok = True
            result.append(repr_ok and dim_ok and meth_ok and fac_ok)
        return result

    def _axis_titles(meth_sel):
        return dict(
            xaxis_title=_axis_label(meth_sel, 1),
            yaxis_title=_axis_label(meth_sel, 2),
            zaxis_title=_axis_label(meth_sel, 3),
        )

    # ── Build button sets ──────────────────────────────────────────────────────
    # Default selection: first repr, first dim, first method, both factors
    def_repr  = repr_vals[0]
    def_dim   = dim_vals[0]
    def_meth  = meth_vals[0]
    def_fac   = "both"

    # Set default visibility
    for i, (_, rn, ds, mth, fac) in enumerate(trace_meta):
        traces[i].visible = _vis_array(def_repr, def_dim, def_meth, def_fac)[i]

    # Build repr buttons
    repr_btns = []
    for rv in repr_vals + (["both"] if len(repr_vals) > 1 else []):
        repr_btns.append(dict(
            label=rv.capitalize(),
            method="update",
            args=[{"visible": _vis_array(rv, def_dim, def_meth, def_fac)},
                  {"title": f"{qty.upper()} | L{layer} | repr={rv} | "
                            f"dims={def_dim} | {def_meth} | factor={def_fac}"}],
        ))

    # Build factor buttons
    factor_btns = []
    for fv, flabel in [("ast","AST"), ("builtin","Builtin"), ("both","Both")]:
        factor_btns.append(dict(
            label=flabel,
            method="update",
            args=[{"visible": _vis_array(def_repr, def_dim, def_meth, fv)},
                  {"title": f"{qty.upper()} | L{layer} | repr={def_repr} | "
                            f"dims={def_dim} | {def_meth} | factor={fv}"}],
        ))

    # Build method buttons (also update axis labels)
    meth_btns = []
    for mv in meth_vals:
        at = _axis_titles(mv)
        meth_btns.append(dict(
            label=mv.upper(),
            method="update",
            args=[{"visible": _vis_array(def_repr, def_dim, mv, def_fac)},
                  {"scene.xaxis.title": at["xaxis_title"],
                   "scene.yaxis.title": at["yaxis_title"],
                   "scene.zaxis.title": at["zaxis_title"],
                   "title": f"{qty.upper()} | L{layer} | repr={def_repr} | "
                            f"dims={def_dim} | {mv} | factor={def_fac}"}],
        ))

    # Build dim-set buttons
    dim_btns = []
    for dv in dim_vals:
        dim_btns.append(dict(
            label=dv,
            method="update",
            args=[{"visible": _vis_array(def_repr, dv, def_meth, def_fac)},
                  {"title": f"{qty.upper()} | L{layer} | repr={def_repr} | "
                            f"dims={dv} | {def_meth} | factor={def_fac}"}],
        ))

    at0 = _axis_titles(def_meth)
    fig.update_layout(
        title=dict(
            text=(f"{qty.upper()}  |  Layer {layer}  |  "
                  f"repr={def_repr}  |  dims={def_dim}  |  "
                  f"{def_meth}  |  factor={def_fac}"),
            font=dict(size=13),
        ),
        scene=dict(
            xaxis_title=at0["xaxis_title"],
            yaxis_title=at0["yaxis_title"],
            zaxis_title=at0["zaxis_title"],
            xaxis=dict(showgrid=True, zeroline=False),
            yaxis=dict(showgrid=True, zeroline=False),
            zaxis=dict(showgrid=True, zeroline=False),
        ),
        legend=dict(
            title="Category",
            font=dict(size=9),
            itemsizing="constant",
            tracegroupgap=2,
        ),
        updatemenus=[
            dict(
                buttons=repr_btns,
                direction="right",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.01, xanchor="left",
                y=1.18, yanchor="top",
                bgcolor="#EEEEEE",
                font=dict(size=11),
                active=0,
            ),
            dict(
                buttons=factor_btns,
                direction="right",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.01, xanchor="left",
                y=1.11, yanchor="top",
                bgcolor="#DDEEEE",
                font=dict(size=11),
                active=2,
            ),
            dict(
                buttons=meth_btns,
                direction="right",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.01, xanchor="left",
                y=1.04, yanchor="top",
                bgcolor="#EEDDEE",
                font=dict(size=11),
                active=0,
            ),
            dict(
                buttons=dim_btns,
                direction="right",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.01, xanchor="left",
                y=0.97, yanchor="top",
                bgcolor="#EEEEDD",
                font=dict(size=11),
                active=0,
            ),
        ],
        annotations=[
            dict(text="<b>Repr</b>",   x=0.0, xref="paper", y=1.18, yref="paper",
                 showarrow=False, font=dict(size=10)),
            dict(text="<b>Factor</b>", x=0.0, xref="paper", y=1.11, yref="paper",
                 showarrow=False, font=dict(size=10)),
            dict(text="<b>Method</b>", x=0.0, xref="paper", y=1.04, yref="paper",
                 showarrow=False, font=dict(size=10)),
            dict(text="<b>Dims</b>",   x=0.0, xref="paper", y=0.97, yref="paper",
                 showarrow=False, font=dict(size=10)),
        ],
        margin=dict(l=0, r=0, b=0, t=180),
        height=820,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pio.write_html(fig, str(out_path), include_plotlyjs="cdn",
                   full_html=True, auto_open=False)
    print(f"    → {out_path.name}  ({len(traces)} traces)")


# ─────────────────────────────────────────────────────────────────────────────
# Neuron-ID export (with per-class breakdown)
# ─────────────────────────────────────────────────────────────────────────────

_QTY_LABEL = {"residual": "Residual", "mlp": "MLP"}


def _neuron_id(layer: int, qty: str, neuron: int) -> str:
    return f"L{layer}[{_QTY_LABEL[qty]}]{neuron}"


def export_neuron_ids_by_class(
    layer:       int,
    qty:         str,
    vp_ua:       np.ndarray,        # (H,) unique_ast VP scores
    vp_ub:       np.ndarray,        # (H,) unique_builtin VP scores
    ast_mask:    np.ndarray | None,
    b_mask:      np.ndarray | None,
    act:         np.ndarray,        # (N, H) activations at this layer
    ast_labels:  list[str],
    b_labels:    list[str],
    out_dir:     Path,
) -> Path:
    """
    Write <out_dir>/neuron_ids/<qty>_L<ll>_neuron_ids.json.

    Clean two-section structure — AST neurons never appear in the builtin
    section and vice versa:

    {
      "layer": 3, "qty": "residual",
      "ast": {
        "overall_ranking": [
          { "id":"L3[Residual]512","neuron":512,"vp_rank":1,"vp_score":0.31,
            "pure":true,"overall_mean_act":0.45,"overall_act_rank":2 }, ...
        ],
        "by_class": {
          "For":   [ {"id":…,"neuron":…,"vp_rank":…,"vp_score":…,"pure":…,
                      "mean_act":0.67,"class_act_rank":1}, … ],
          "While": [ … ], ...
        }
      },
      "builtin": {
        "overall_ranking": [ … ],
        "by_class": { "len": [ … ], ... }
      }
    }
    """
    ast_classes     = sorted(set(ast_labels))
    builtin_classes = sorted(set(b_labels))
    ast_arr = np.array(ast_labels)
    b_arr   = np.array(b_labels)

    def _build_section(
        scores:     np.ndarray,
        mask:       np.ndarray | None,
        class_list: list[str],
        label_arr:  np.ndarray,
    ) -> dict:
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

        # Overall ranking by mean_act across ALL prompts
        overall_mean = act[:, neuron_indices].mean(axis=0)
        act_order    = np.argsort(overall_mean)[::-1]
        overall_ranking = [
            {**base_rows[pos],
             "overall_mean_act": round(float(overall_mean[pos]), 6),
             "overall_act_rank": act_rank}
            for act_rank, pos in enumerate(act_order, start=1)
        ]

        # Per-class: re-sort same neurons by class mean_act
        by_class: dict[str, list] = {}
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
# Per-layer processor
# ─────────────────────────────────────────────────────────────────────────────

def _process_layer(
    act:            np.ndarray,
    vp_ua:          np.ndarray,
    vp_ub:          np.ndarray,
    ast_mask:       np.ndarray | None,
    b_mask:         np.ndarray | None,
    ast_labels:     list[str],
    b_labels:       list[str],
    qty:            str,
    layer:          int,
    out_dir:        Path,
    methods:        list[str],
    skip_full_umap: bool,
    ids_only:       bool,
) -> None:
    N, H = act.shape

    # ── Neuron IDs with class breakdown ───────────────────────────────────────
    export_neuron_ids_by_class(
        layer, qty, vp_ua, vp_ub,
        ast_mask, b_mask,
        act, ast_labels, b_labels,
        out_dir,
    )

    if ids_only:
        return

    # ── Build dimension subsets ───────────────────────────────────────────────
    # Each config: (repr_name, dim_set, X_ast, X_b)
    dim_configs: list[tuple] = []

    # Raw full
    dim_configs.append(("raw", "full", act, act))

    # Raw top-512
    ai = _top_dims(vp_ua, TOP_K)
    bi = _top_dims(vp_ub, TOP_K)
    dim_configs.append(("raw", "top512", act[:, ai], act[:, bi]))

    # Pure full + top-512
    if ast_mask is not None and b_mask is not None:
        n_ap = int(ast_mask.sum())
        n_bp = int(b_mask.sum())
        if n_ap >= N_COMP and n_bp >= N_COMP:
            X_ap = act[:, ast_mask]
            X_bp = act[:, b_mask]
            dim_configs.append(("pure", "full", X_ap, X_bp))
            ap_t = _top_dims(vp_ua[ast_mask], TOP_K)
            bp_t = _top_dims(vp_ub[b_mask],   TOP_K)
            dim_configs.append(("pure", "top512", X_ap[:, ap_t], X_bp[:, bp_t]))
        else:
            print(f"    [L{layer:02d}] Skipping pure — "
                  f"ast_pure={n_ap}, builtin_pure={n_bp} (need ≥ {N_COMP})")

    # ── Compute embeddings ────────────────────────────────────────────────────
    # configs dict: (repr_name, dim_set, method) → {Z_ast, Z_b}
    configs: dict = {}

    for repr_name, dim_set, X_ast, X_b in dim_configs:
        for method in methods:
            if method == "umap" and not _UMAP_OK:
                continue
            if (method == "umap" and skip_full_umap
                    and dim_set == "full" and X_ast.shape[1] > UMAP_MAX_DIM_FULL):
                print(f"    [L{layer:02d}] Skipping {repr_name}/{dim_set} UMAP "
                      f"(dims={X_ast.shape[1]} > {UMAP_MAX_DIM_FULL})")
                continue

            print(f"    [L{layer:02d}] {qty} {repr_name}/{dim_set}/{method} "
                  f"X_ast={X_ast.shape} X_b={X_b.shape}", end=" ... ")
            try:
                fn    = _pca3 if method == "pca" else _umap3
                Z_ast = fn(X_ast)
                Z_b   = fn(X_b)
            except Exception as e:
                print(f"FAILED: {e}")
                continue
            print("done")

            configs[(repr_name, dim_set, method)] = {"Z_ast": Z_ast, "Z_b": Z_b}

    if not configs:
        print(f"    [L{layer:02d}] No embeddings computed — skipping HTML.")
        return

    # ── Build HTML ────────────────────────────────────────────────────────────
    html_dir  = out_dir / "interactive"
    html_dir.mkdir(parents=True, exist_ok=True)
    out_path  = html_dir / f"{qty}_L{layer:02d}_interactive.html"

    build_interactive(
        configs, ast_labels, b_labels,
        qty=qty, layer=layer,
        out_path=out_path,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if not _PLOTLY_OK:
        raise SystemExit("plotly is required.  Install with: pip install plotly")

    parser = argparse.ArgumentParser(
        description="Interactive 3-D VP projection visualiser (Plotly)"
    )
    parser.add_argument("--stem",     "-s", required=True)
    parser.add_argument("--in_dir",  "-d", default="../data",
                        help="Directory with 01_*/02_* files (default: ../data)")
    parser.add_argument("--out_dir", "-o", default="output",
                        help="Output root (default: output/ next to this script)")
    parser.add_argument("--qty",     default="both",
                        choices=["residual", "mlp", "both"])
    parser.add_argument("--layer",   type=int, default=None,
                        help="Single layer index (-1 = final). Omit for ALL layers.")
    parser.add_argument("--methods", nargs="+", default=["pca", "umap"],
                        choices=["pca", "umap"])
    parser.add_argument("--skip_full_umap", action="store_true",
                        help=f"Skip UMAP when full-dim count > {UMAP_MAX_DIM_FULL}")
    parser.add_argument("--ids_only", action="store_true",
                        help="Only export neuron_ids JSON, skip HTML")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    base       = Path(args.in_dir)
    out_root   = (script_dir / args.out_dir).resolve()
    stem       = args.stem

    methods = list(args.methods)
    if "umap" in methods and not _UMAP_OK:
        print("[warn] umap-learn unavailable — removing 'umap'.")
        methods = [m for m in methods if m != "umap"]
    if not methods and not args.ids_only:
        raise SystemExit("No reduction methods available.")

    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Input dir : {base}")
    print(f"Output dir: {out_root}")

    # ── Metadata ───────────────────────────────────────────────────────────────
    meta_path = base / f"01_{stem}_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing: {meta_path}")
    with open(meta_path) as f:
        meta = json.load(f)
    ast_labels = [m["ast_node"]    for m in meta]
    b_labels   = [m["builtin_obj"] for m in meta]
    print(f"  N={len(meta)}  AST={len(set(ast_labels))}  Builtin={len(set(b_labels))}")

    # ── VP scores ──────────────────────────────────────────────────────────────
    vp_r = vp_n = None
    for attr, path_tpl, qty_check in [
        ("vp_r", f"02_{stem}_vp_residual.npz",    ("residual","both")),
        ("vp_n", f"02_{stem}_vp_mlp_neurons.npz", ("mlp","both")),
    ]:
        p = base / path_tpl
        if p.exists():
            locals()[attr]  # touch
            if attr == "vp_r":
                vp_r = dict(np.load(p))
            else:
                vp_n = dict(np.load(p))
            print(f"  VP loaded: {p.name}")
        elif args.qty in qty_check:
            print(f"  [warn] {p.name} not found — dim ranking uses zeros")

    # ── Purity masks ───────────────────────────────────────────────────────────
    masks_npz  = None
    masks_path = base / f"02_{stem}_purity_masks.npz"
    if masks_path.exists():
        masks_npz = dict(np.load(masks_path, allow_pickle=True))
        print(f"  Purity masks loaded.")
    else:
        print(f"  [warn] {masks_path.name} not found — pure configs skipped.")

    # ── Residual ───────────────────────────────────────────────────────────────
    if args.qty in ("residual", "both"):
        rp = base / f"01_{stem}_residual_all.npy"
        if not rp.exists():
            print(f"  [skip] {rp.name} not found")
        else:
            resid_all = np.load(rp)
            _, L1, H  = resid_all.shape
            print(f"\nResidual: {resid_all.shape}")
            layers = list(range(L1))
            if args.layer is not None:
                li = L1 + args.layer if args.layer < 0 else args.layer
                layers = [int(np.clip(li, 0, L1-1))]
            for l in layers:
                print(f"  [residual] Layer {l}/{L1-1}")
                act   = resid_all[:, l, :]
                vp_ua = vp_r["unique_ast"][l]     if vp_r else np.zeros(H)
                vp_ub = vp_r["unique_builtin"][l] if vp_r else np.zeros(H)
                am = (masks_npz["residual_ast_pure"][l].astype(bool)
                      if masks_npz and "residual_ast_pure" in masks_npz else None)
                bm = (masks_npz["residual_builtin_pure"][l].astype(bool)
                      if masks_npz and "residual_builtin_pure" in masks_npz else None)
                _process_layer(act, vp_ua, vp_ub, am, bm,
                                ast_labels, b_labels, "residual", l,
                                out_root, methods, args.skip_full_umap, args.ids_only)
            del resid_all

    # ── MLP neurons ────────────────────────────────────────────────────────────
    if args.qty in ("mlp", "both"):
        mp = base / f"01_{stem}_mlp_neurons.npy"
        if not mp.exists():
            print(f"  [skip] {mp.name} not found")
        else:
            mlp_all = np.load(mp)
            _, L, M = mlp_all.shape
            print(f"\nMLP neurons: {mlp_all.shape}")
            layers = list(range(L))
            if args.layer is not None:
                li = L + args.layer if args.layer < 0 else args.layer
                layers = [int(np.clip(li, 0, L-1))]
            for l in layers:
                print(f"  [mlp] Layer {l}/{L-1}")
                act   = mlp_all[:, l, :]
                vp_ua = vp_n["unique_ast"][l]     if vp_n else np.zeros(M)
                vp_ub = vp_n["unique_builtin"][l] if vp_n else np.zeros(M)
                am = (masks_npz["neurons_ast_pure"][l].astype(bool)
                      if masks_npz and "neurons_ast_pure" in masks_npz else None)
                bm = (masks_npz["neurons_builtin_pure"][l].astype(bool)
                      if masks_npz and "neurons_builtin_pure" in masks_npz else None)
                _process_layer(act, vp_ua, vp_ub, am, bm,
                                ast_labels, b_labels, "mlp", l,
                                out_root, methods, args.skip_full_umap, args.ids_only)

    print(f"\nDone.")
    print(f"  Interactive HTML : {out_root / 'interactive'}")
    print(f"  Neuron IDs       : {out_root / 'neuron_ids'}")


if __name__ == "__main__":
    main()
