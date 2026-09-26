"""
02f_ast_graph.py  (submission version)

Interactive 3-D network graph of AST class similarity — combined all-layers HTML.

Produces a single HTML file with a layer-selector dropdown embedding all layers:
  figures/ast_graph/ast_graph_all_layers.html

Each layer shows an interactive 3-D graph where:
  - Node position = 3-D metric MDS on pairwise cosine distance between
    per-class mean residual activations
  - Node size/colour = coreness (mean cosine-sim to all other classes)
  - Thick edges = MST skeleton
  - Faint edges = similarity > threshold
  - Buttons to toggle faint edges on/off

Usage
-----
  python 02f_ast_graph.py --stem contrastive_stubs --in_dir ../data_more
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.manifold import MDS
from scipy.sparse.csgraph import minimum_spanning_tree

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    _PLOTLY_OK = True
except ImportError:
    _PLOTLY_OK = False
    print("[error] plotly not found — install with: pip install plotly")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

NODE_SIZE_MIN    = 8
NODE_SIZE_MAX    = 28
EDGE_SIM_THRESH  = 0.10
TOP_N_NEURONS    = 5


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stem",    default="contrastive_stubs")
    p.add_argument("--in_dir",  default="../data_more")
    p.add_argument("--out_dir", default=None)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def cosine_sim_matrix(mu_c: np.ndarray) -> np.ndarray:
    """(C, H) -> (C, C) cosine similarity matrix."""
    norms = np.linalg.norm(mu_c, axis=1, keepdims=True) + 1e-12
    n = mu_c / norms
    return (n @ n.T).astype(np.float32)


def embed_3d(sim: np.ndarray) -> np.ndarray:
    dist = np.clip(1.0 - sim, 0, None)
    np.fill_diagonal(dist, 0.0)
    mds = MDS(n_components=3, dissimilarity="precomputed",
              random_state=42, max_iter=500, n_init=4)
    return mds.fit_transform(dist).astype(np.float32)


def recenter_on_core(xyz: np.ndarray, coreness: np.ndarray,
                     core_frac: float = 0.25) -> np.ndarray:
    n_core   = max(1, int(np.ceil(core_frac * len(coreness))))
    core_idx = np.argsort(coreness)[-n_core:]
    centroid = xyz[core_idx].mean(axis=0)
    return xyz - centroid


def mst_edges(sim: np.ndarray) -> list[tuple[int, int, float]]:
    dist = np.clip(1.0 - sim, 0, None)
    np.fill_diagonal(dist, 0.0)
    csgraph = minimum_spanning_tree(dist)
    cx = csgraph.tocoo()
    return [(int(i), int(j), float(sim[i, j]))
            for i, j in zip(cx.row, cx.col)]


def top_neurons(mu_c_row: np.ndarray, n: int = TOP_N_NEURONS) -> list[int]:
    return np.argsort(mu_c_row)[-n:][::-1].tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Plotly figure builder
# ─────────────────────────────────────────────────────────────────────────────

def build_figure(xyz: np.ndarray, sim: np.ndarray,
                 mu_c: np.ndarray, classes: list[str], layer: int) -> "go.Figure":
    C        = len(classes)
    coreness = (sim.sum(axis=1) - 1.0) / (C - 1)
    node_sizes = NODE_SIZE_MIN + (NODE_SIZE_MAX - NODE_SIZE_MIN) * (
        (coreness - coreness.min()) / ((coreness.max() - coreness.min()) + 1e-9)
    )

    hover = [
        f"<b>{cls}</b><br>coreness: {coreness[i]:.3f}<br>"
        f"top neurons: {top_neurons(mu_c[i])}"
        for i, cls in enumerate(classes)
    ]

    mst_set       = set()
    mst_edge_list = mst_edges(sim)
    for i, j, s in mst_edge_list:
        mst_set.add((min(i, j), max(i, j)))

    sim_vals = [s for _, _, s in mst_edge_list]
    sim_min, sim_max = (min(sim_vals), max(sim_vals)) if sim_vals else (0, 1)

    def sim_to_rgb(s: float) -> str:
        t = (s - sim_min) / (sim_max - sim_min + 1e-9)
        return f"rgb({int(220*(1-t))},{int(180*t+40)},60)"

    traces = []

    # Non-MST faint edges
    faint_x, faint_y, faint_z = [], [], []
    for i in range(C):
        for j in range(i + 1, C):
            s = float(sim[i, j])
            if s < EDGE_SIM_THRESH or (i, j) in mst_set:
                continue
            faint_x += [xyz[i, 0], xyz[j, 0], None]
            faint_y += [xyz[i, 1], xyz[j, 1], None]
            faint_z += [xyz[i, 2], xyz[j, 2], None]

    if faint_x:
        traces.append(go.Scatter3d(
            x=faint_x, y=faint_y, z=faint_z, mode="lines",
            line=dict(color="rgba(80,80,200,0.45)", width=1.5),
            hoverinfo="none", name="similarity edges", showlegend=True,
        ))

    # MST edges
    for i, j, s in mst_edge_list:
        traces.append(go.Scatter3d(
            x=[xyz[i, 0], xyz[j, 0]],
            y=[xyz[i, 1], xyz[j, 1]],
            z=[xyz[i, 2], xyz[j, 2]],
            mode="lines", line=dict(color=sim_to_rgb(s), width=4),
            hoverinfo="none", showlegend=False,
        ))

    # Nodes
    traces.append(go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
        mode="markers+text",
        marker=dict(
            size=node_sizes.tolist(),
            color=coreness.tolist(),
            colorscale="Viridis",
            reversescale=True,
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

    _axis_base = dict(
        showbackground=True,
        backgroundcolor="rgba(220,225,245,1.0)",
        gridcolor="#6666aa", gridwidth=1,
        showgrid=True, showline=True,
        linecolor="#111133", linewidth=2,
        showticklabels=True,
        tickfont=dict(size=9, color="#111133"),
        nticks=5, zeroline=True,
        zerolinecolor="#333366", zerolinewidth=2,
    )

    def _axis(label: str) -> dict:
        return {**_axis_base, "title": dict(text=label, font=dict(size=11, color="#111133"))}

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text=(
                f"<b>AST class similarity network  —  Layer {layer}</b><br>"
                "<sup>3-D metric MDS on pairwise cosine-distance; "
                "node size/colour = coreness; thick = MST; faint = similarity > 0.1</sup>"
            ),
            x=0.5, xanchor="center",
        ),
        scene=dict(
            xaxis=_axis("MDS-1"), yaxis=_axis("MDS-2"), zaxis=_axis("MDS-3"),
            bgcolor="rgb(250,250,252)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
        ),
        margin=dict(l=0, r=0, t=110, b=0),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.7)",
                    bordercolor="lightgrey", borderwidth=1),
        paper_bgcolor="white",
        height=750,
    )

    n_faint = 1 if faint_x else 0
    n_mst   = len(mst_edge_list)
    visible_all = [True]  * (n_faint + n_mst + 1)
    visible_mst = ([False] * n_faint) + ([True] * n_mst) + [True]

    fig.update_layout(
        updatemenus=[dict(
            type="buttons", showactive=True,
            x=0.01, y=0.92, xanchor="left",
            buttons=[
                dict(label="All edges",  method="restyle",
                     args=[{"visible": visible_all}]),
                dict(label="MST only",   method="restyle",
                     args=[{"visible": visible_mst}]),
            ],
        )]
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Per-layer processing
# ─────────────────────────────────────────────────────────────────────────────

def process_layer(act: np.ndarray, labels: np.ndarray,
                  classes: list[str], layer: int) -> "go.Figure":
    label_idx = [sorted(set(labels)).index(l) for l in labels]
    classes_sorted = sorted(set(labels))
    mu_c = np.stack([
        act[np.array(labels) == c].mean(axis=0) for c in classes_sorted
    ]).astype(np.float32)
    sim      = cosine_sim_matrix(mu_c)
    coreness = (sim.sum(axis=1) - 1.0) / max(len(classes_sorted) - 1, 1)
    xyz      = embed_3d(sim)
    xyz      = recenter_on_core(xyz, coreness)
    return build_figure(xyz, sim, mu_c, classes_sorted, layer)


# ─────────────────────────────────────────────────────────────────────────────
# Combined HTML writer
# ─────────────────────────────────────────────────────────────────────────────

def write_combined_html(layer_figs: list[tuple[int, "go.Figure"]],
                        out_path: Path) -> None:
    import json as _json

    layers       = [l for l, _ in layer_figs]
    options_html = "\n".join(
        f'      <option value="{l}">Layer {l}</option>'
        for l, _ in layer_figs
    )

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
    print(f"  -> {out_path.name}  ({len(layer_figs)} layers embedded)")


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

    with open(base / f"01_{stem}_meta.json") as f:
        meta = json.load(f)
    labels  = np.array([m["ast_node"] for m in meta])
    classes = sorted(set(labels))
    print(f"N={len(meta)}  C={len(classes)} classes")

    resid_all = np.load(base / f"01_{stem}_residual_all.npy")  # (N, L+1, H)
    N, L1, H  = resid_all.shape
    print(f"Residual: {resid_all.shape}")

    layers = list(range(L1))

    layer_figs: list[tuple[int, go.Figure]] = []
    for l in layers:
        print(f"\n-- Layer {l} --")
        act = resid_all[:, l, :].astype(np.float32)
        fig = process_layer(act, labels, classes, l)
        layer_figs.append((l, fig))

    out_path = fig_dir / "ast_graph_all_layers.html"
    write_combined_html(layer_figs, out_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
