"""
02f_ast_graph.py  (lives in results/)

Interactive 3-D network graph of AST class similarity.

Layout philosophy
─────────────────
• Pairwise cosine similarity between per-class mean activations is the
  distance metric (1 − cos → distance).
• 3-D MDS embeds the classes so inter-node distances reflect those
  similarities as faithfully as possible.
• The "core" classes (highest mean similarity to all others) are
  translated to sit near the origin — peripheral classes radiate outward.
• An MST is overlaid as thick edges to show the dominant skeleton.
• All non-MST edges are drawn as faint lines; opacity ∝ similarity.

Per layer, one interactive Plotly HTML is written:

  figures/ast_graph/L<ll>_ast_graph.html

Controls in the HTML
────────────────────
  Layer selector : dropdown to jump between layers
                   (all layers are embedded in a single HTML file if
                    --all_layers is passed; otherwise one file per layer)
  Hover          : class name, coreness score, top-5 active neurons
  MST-only toggle: button to hide non-MST edges

Node encoding
─────────────
  size  ∝  coreness  (mean cosine similarity to all other classes)
  colour = coreness  (viridis: yellow=peripheral, purple=core)

Edge encoding
─────────────
  MST edges   : thick (width 4), coloured by similarity
  Other edges : thin (width 1), grey, opacity = similarity²

INPUTS (relative to --in_dir)
──────
  01_<stem>_residual_all.npy
  01_<stem>_meta.json

OUTPUTS
───────
  figures/ast_graph/L<ll>_ast_graph.html   (one per layer, or one combined)

Usage
-----
  python 02f_ast_graph.py --stem contrastive_stubs
  python 02f_ast_graph.py --stem contrastive_stubs --layer 5
  python 02f_ast_graph.py --stem contrastive_stubs --all_layers
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.manifold import MDS
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import squareform

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _PLOTLY_OK = True
except ImportError:
    _PLOTLY_OK = False
    print("[error] plotly not found — install with: pip install plotly")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

NODE_SIZE_MIN  = 8
NODE_SIZE_MAX  = 28
EDGE_SIM_THRESH = 0.10    # hide non-MST edges below this similarity
TOP_N_NEURONS  = 5        # shown in hover text


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stem",       default="contrastive_stubs")
    p.add_argument("--in_dir",  "-d", default="../data")
    p.add_argument("--out_dir", "-o", default=None)
    p.add_argument("--layer",      type=int, default=None,
                   help="Single layer (default: last layer)")
    p.add_argument("--all_layers", action="store_true",
                   help="Produce one HTML per layer")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def cosine_sim_matrix(mu_c: np.ndarray) -> np.ndarray:
    """(C, H) → (C, C) cosine similarity matrix."""
    norms = np.linalg.norm(mu_c, axis=1, keepdims=True) + 1e-12
    n = mu_c / norms
    return (n @ n.T).astype(np.float32)


def embed_3d(sim: np.ndarray) -> np.ndarray:
    """
    Cosine similarity → 3-D coordinates via metric MDS.
    Returns (C, 3) array.
    """
    dist = np.clip(1.0 - sim, 0, None)
    np.fill_diagonal(dist, 0.0)
    mds  = MDS(n_components=3, dissimilarity="precomputed",
               random_state=42, max_iter=500, n_init=4)
    return mds.fit_transform(dist).astype(np.float32)


def recenter_on_core(xyz: np.ndarray, coreness: np.ndarray,
                     core_frac: float = 0.25) -> np.ndarray:
    """
    Translate coordinates so the centroid of the top core_frac classes
    sits at the origin.
    """
    n_core  = max(1, int(np.ceil(core_frac * len(coreness))))
    core_idx = np.argsort(coreness)[-n_core:]
    centroid = xyz[core_idx].mean(axis=0)
    return xyz - centroid


def mst_edges(sim: np.ndarray) -> list[tuple[int, int, float]]:
    """
    Return list of (i, j, similarity) for MST edges.
    MST is computed on the distance matrix (1 − sim).
    """
    dist = np.clip(1.0 - sim, 0, None)
    np.fill_diagonal(dist, 0.0)
    csgraph = minimum_spanning_tree(dist)
    cx      = csgraph.tocoo()
    return [(int(i), int(j), float(sim[i, j]))
            for i, j in zip(cx.row, cx.col)]


def top_neurons(mu_c_row: np.ndarray, n: int = TOP_N_NEURONS) -> list[int]:
    return np.argsort(mu_c_row)[-n:][::-1].tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Plotly figure builder
# ─────────────────────────────────────────────────────────────────────────────

def build_figure(
    xyz:       np.ndarray,     # (C, 3) after recentering
    sim:       np.ndarray,     # (C, C) cosine similarity
    mu_c:      np.ndarray,     # (C, H) class mean activations
    classes:   list[str],
    layer:     int,
) -> go.Figure:

    C           = len(classes)
    coreness    = (sim.sum(axis=1) - 1.0) / (C - 1)   # mean sim to others ∈ [0,1]

    # Node sizes
    node_sizes  = NODE_SIZE_MIN + (NODE_SIZE_MAX - NODE_SIZE_MIN) * (
        (coreness - coreness.min()) / ((coreness.max() - coreness.min()) + 1e-9)
    )

    # Hover text
    hover = []
    for i, cls in enumerate(classes):
        top_n = top_neurons(mu_c[i])
        hover.append(
            f"<b>{cls}</b><br>"
            f"coreness: {coreness[i]:.3f}<br>"
            f"top neurons: {top_n}"
        )

    # ── MST edges ─────────────────────────────────────────────────────────────
    mst_set  = set()
    mst_edge_list = mst_edges(sim)
    for i, j, s in mst_edge_list:
        mst_set.add((min(i,j), max(i,j)))

    # Colour MST edges by similarity (green=high, red=low)
    sim_vals = [s for _, _, s in mst_edge_list]
    sim_min, sim_max = (min(sim_vals), max(sim_vals)) if sim_vals else (0, 1)

    def sim_to_rgb(s: float) -> str:
        t = (s - sim_min) / (sim_max - sim_min + 1e-9)
        r = int(220 * (1 - t))
        g = int(180 * t + 40)
        b = int(60)
        return f"rgb({r},{g},{b})"

    traces = []

    # ── Non-MST edges (faint, opacity ∝ sim²) ────────────────────────────────
    faint_x, faint_y, faint_z, faint_o = [], [], [], []
    for i in range(C):
        for j in range(i + 1, C):
            s = float(sim[i, j])
            if s < EDGE_SIM_THRESH:
                continue
            if (i, j) in mst_set:
                continue
            faint_x += [xyz[i,0], xyz[j,0], None]
            faint_y += [xyz[i,1], xyz[j,1], None]
            faint_z += [xyz[i,2], xyz[j,2], None]

    if faint_x:
        traces.append(go.Scatter3d(
            x=faint_x, y=faint_y, z=faint_z,
            mode="lines",
            line=dict(color="rgba(80,80,200,0.45)", width=1.5),
            hoverinfo="none",
            name="similarity edges",
            showlegend=True,
        ))

    # ── MST edges (one trace per edge so we can colour individually) ──────────
    for i, j, s in mst_edge_list:
        color = sim_to_rgb(s)
        traces.append(go.Scatter3d(
            x=[xyz[i,0], xyz[j,0]],
            y=[xyz[i,1], xyz[j,1]],
            z=[xyz[i,2], xyz[j,2]],
            mode="lines",
            line=dict(color=color, width=4),
            hoverinfo="none",
            showlegend=False,
        ))

    # ── Nodes ─────────────────────────────────────────────────────────────────
    traces.append(go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
        mode="markers+text",
        marker=dict(
            size=node_sizes.tolist(),
            color=coreness.tolist(),
            colorscale="Viridis",
            reversescale=True,             # purple=core, yellow=peripheral
            colorbar=dict(title="coreness", thickness=12, len=0.6),
            line=dict(color="white", width=0.5),
        ),
        text=classes,
        textposition="top center",
        textfont=dict(size=9, color="black"),
        hovertext=hover,
        hoverinfo="text",
        name="AST classes",
    ))

    # ── Layout ────────────────────────────────────────────────────────────────
    _axis_base = dict(
        showbackground  = True,
        backgroundcolor = "rgba(220,225,245,1.0)",
        gridcolor       = "#6666aa",
        gridwidth       = 1,
        showgrid        = True,
        showline        = True,
        linecolor       = "#111133",
        linewidth       = 2,
        showticklabels  = True,
        tickfont        = dict(size=9, color="#111133"),
        nticks          = 5,
        zeroline        = True,
        zerolinecolor   = "#333366",
        zerolinewidth   = 2,
    )

    def _axis(label: str) -> dict:
        return {**_axis_base, "title": dict(
            text=label,
            font=dict(size=11, color="#111133"),
        )}

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text=(
                f"<b>AST class similarity network  —  Layer {layer}</b><br>"
                "<sup>"
                "Axes: MDS-1/2/3 — 3-D metric MDS on pairwise cosine-distance "
                "(d = 1 − cos(μ<sub>c</sub>, μ<sub>c′</sub>)) between per-class "
                "mean residual activations; origin = centroid of top-25% most "
                "central classes<br>"
                "Node size/colour = coreness (mean cosine-sim to all other classes)  ·  "
                "thick edges = MST  ·  faint edges = similarity &gt; 0.1"
                "</sup>"
            ),
            x=0.5, xanchor="center",
        ),
        scene=dict(
            xaxis=_axis("MDS-1"),
            yaxis=_axis("MDS-2"),
            zaxis=_axis("MDS-3"),
            bgcolor="rgb(250,250,252)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
        ),
        margin=dict(l=0, r=0, t=110, b=0),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.7)",
                    bordercolor="lightgrey", borderwidth=1),
        paper_bgcolor="white",
        height=750,
    )

    # ── Button: toggle non-MST edges ─────────────────────────────────────────
    n_faint = 1 if faint_x else 0   # number of faint-edge traces
    n_mst   = len(mst_edge_list)
    n_node  = 1
    # trace order: [faint?] [mst_0 … mst_n] [nodes]
    visible_all  = [True]  * (n_faint + n_mst + n_node)
    visible_mst  = ([False] * n_faint) + ([True] * n_mst) + ([True] * n_node)

    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            showactive=True,
            x=0.01, y=0.92, xanchor="left",
            buttons=[
                dict(label="All edges",
                     method="restyle",
                     args=[{"visible": visible_all}]),
                dict(label="MST only",
                     method="restyle",
                     args=[{"visible": visible_mst}]),
            ],
        )]
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Per-layer processing
# ─────────────────────────────────────────────────────────────────────────────

def process_layer(
    act:      np.ndarray,    # (N, H)
    labels:   np.ndarray,   # (N,)
    classes:  list[str],
    layer:    int,
) -> go.Figure:
    """Build and return the figure for one layer (does not write HTML)."""
    mu_c = np.stack([
        act[labels == c].mean(axis=0) for c in classes
    ]).astype(np.float32)                     # (C, H)

    sim      = cosine_sim_matrix(mu_c)        # (C, C)
    coreness = (sim.sum(axis=1) - 1.0) / max(len(classes) - 1, 1)

    xyz = embed_3d(sim)                       # (C, 3)
    xyz = recenter_on_core(xyz, coreness)     # shift core → origin

    return build_figure(xyz, sim, mu_c, classes, layer)


# ─────────────────────────────────────────────────────────────────────────────
# HTML output helpers
# ─────────────────────────────────────────────────────────────────────────────

def write_combined_html(layer_figs: list[tuple[int, go.Figure]], out_path: Path) -> None:
    """
    Write a single HTML file containing all layers.

    Each layer is rendered by Plotly's own pio.to_html() (the same
    code path as write_html / standalone files), so all colorscales,
    colorbars, updatemenus, and numpy types are handled correctly.

    The divs are stacked; CSS show/hide + Plotly.Plots.resize() is used
    to switch layers without re-serialising or re-plotting anything.
    """
    import json as _json
    import plotly.io as pio

    layers      = [l for l, _ in layer_figs]
    options_html = "\n".join(
        f'      <option value="{l}">Layer {l}</option>'
        for l, _ in layer_figs
    )

    # Let Plotly generate each figure's div+script exactly as it would
    # for a standalone file.  CDN script tag only on the first figure.
    layer_divs = []
    for i, (layer, fig) in enumerate(layer_figs):
        include_js = "cdn" if i == 0 else False
        div_html   = pio.to_html(fig, full_html=False,
                                 include_plotlyjs=include_js)
        hidden     = ' style="display:none"' if i != 0 else ""
        layer_divs.append(
            f'<div id="layer-{layer}" class="layer-div"{hidden}>\n'
            f'{div_html}\n</div>'
        )

    html = (
        "<!DOCTYPE html>\n<html>\n<head>\n"
        '  <meta charset="utf-8">\n'
        "  <title>AST Class Similarity Graph</title>\n"
        "  <style>\n"
        "    * { box-sizing: border-box; margin: 0; padding: 0; }\n"
        "    body { font-family: Arial, sans-serif; background: #f8f8fc; }\n"
        "    #controls {\n"
        "      display: flex; align-items: center; gap: 14px;\n"
        "      padding: 10px 16px; background: #e8e8f4;\n"
        "      border-bottom: 2px solid #aaaacc;\n"
        "    }\n"
        "    #controls label { font-weight: bold; font-size: 14px; color: #222; }\n"
        "    #layer-select {\n"
        "      font-size: 14px; padding: 5px 10px;\n"
        "      border: 1px solid #aaaacc; border-radius: 4px;\n"
        "      background: white; cursor: pointer;\n"
        "    }\n"
        "    .layer-div { width: 100%; }\n"
        "  </style>\n"
        "</head>\n<body>\n"
        "  <div id=\"controls\">\n"
        "    <label>Layer:</label>\n"
        f"    <select id=\"layer-select\" onchange=\"switchLayer(+this.value)\">\n"
        f"{options_html}\n"
        "    </select>\n"
        "  </div>\n"
        + "\n".join(layer_divs) + "\n"
        "  <script>\n"
        f"    const LAYERS = {_json.dumps(layers)};\n"
        "    let currentLayer = LAYERS[0];\n"
        "\n"
        "    function switchLayer(layer) {\n"
        "      document.getElementById('layer-' + currentLayer).style.display = 'none';\n"
        "      const newDiv = document.getElementById('layer-' + layer);\n"
        "      newDiv.style.display = 'block';\n"
        "      const plotDiv = newDiv.querySelector('.plotly-graph-div');\n"
        "      if (plotDiv) Plotly.Plots.resize(plotDiv);\n"
        "      currentLayer = layer;\n"
        "    }\n"
        "  </script>\n"
        "</body>\n</html>\n"
    )

    out_path.write_text(html, encoding="utf-8")
    print(f"  → {out_path.name}  ({len(layer_figs)} layers embedded)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if not _PLOTLY_OK:
        return

    args    = _parse_args()
    here    = Path(__file__).parent
    base    = here / args.in_dir
    out     = here / (args.out_dir if args.out_dir else args.in_dir)
    fig_dir = out / "figures" / "ast_graph"
    fig_dir.mkdir(parents=True, exist_ok=True)

    stem = args.stem

    # Load
    with open(base / f"01_{stem}_meta.json") as f:
        meta = json.load(f)
    labels  = np.array([m["ast_node"] for m in meta])
    classes = sorted(set(labels))
    print(f"N={len(meta)}  C={len(classes)} classes")

    resid_all = np.load(base / f"01_{stem}_residual_all.npy")  # (N, L+1, H)
    N, L1, H  = resid_all.shape
    print(f"Residual: {resid_all.shape}")

    # Layer selection
    if args.all_layers:
        layers = list(range(L1))
    elif args.layer is not None:
        l = args.layer if args.layer >= 0 else L1 + args.layer
        layers = [int(np.clip(l, 0, L1 - 1))]
    else:
        layers = [L1 - 1]   # default: last layer

    layer_figs: list[tuple[int, go.Figure]] = []
    for l in layers:
        print(f"\n── Layer {l} ──")
        act = resid_all[:, l, :].astype(np.float32)
        fig = process_layer(act, labels, classes, l)
        layer_figs.append((l, fig))

    if len(layer_figs) == 1:
        # Single layer → standalone HTML
        l, fig = layer_figs[0]
        out_path = fig_dir / f"L{l:02d}_ast_graph.html"
        fig.write_html(str(out_path), include_plotlyjs="cdn")
        print(f"  → {out_path.name}")
    else:
        # Multiple layers → one combined HTML with layer dropdown
        out_path = fig_dir / "ast_graph_all_layers.html"
        write_combined_html(layer_figs, out_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
