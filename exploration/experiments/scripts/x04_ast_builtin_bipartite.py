#!/usr/bin/env python3
"""
X04 — AST↔Builtin Bipartite Graph

Hypothesis: AST nodes and builtins form a structured bipartite graph via
circuit overlap, revealing which constructs are semantically adjacent.

Offline — uses atlas HDF5 only.

Usage:
    python experiments/scripts/x04_ast_builtin_bipartite.py
    python experiments/scripts/x04_ast_builtin_bipartite.py \
        --atlas data/test_5x5x10_validated_prompts.h5 \
        --top-k 5
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "src"))

from analysis.concept_taxonomy import ast_family, builtin_domain
from module2.io_utils import load_atlas_hdf5
from module2.metrics import jaccard_similarity


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--atlas", default="data/small_dynamic_feature_atlas.h5")
    p.add_argument("--out", default="experiments/outputs/x04")
    p.add_argument("--top-k", type=int, default=5,
                   help="Max edges to keep per AST node (top-K by weight)")
    return p.parse_args()


def resolve(path: str) -> Path:
    pp = Path(path)
    return pp if pp.is_absolute() else ROOT / pp


def compute_similarity_matrix(ast_masks: dict, builtin_masks: dict) -> tuple:
    """Compute mean-over-layers Jaccard for all (ast, builtin) pairs."""
    ast_names = sorted(ast_masks.keys())
    blt_names = sorted(builtin_masks.keys())
    matrix = np.zeros((len(ast_names), len(blt_names)))

    for i, a in enumerate(ast_names):
        for j, b in enumerate(blt_names):
            layers_a = ast_masks[a]
            layers_b = builtin_masks[b]
            shared_layers = set(layers_a.keys()) & set(layers_b.keys())
            if not shared_layers:
                continue
            sims = [jaccard_similarity(layers_a[l], layers_b[l]) for l in shared_layers]
            matrix[i, j] = np.mean(sims)

    return ast_names, blt_names, matrix


def plot_similarity_matrix(matrix: np.ndarray, ast_names: list, blt_names: list, path: Path):
    fig, ax = plt.subplots(figsize=(max(6, len(blt_names) * 0.3),
                                    max(4, len(ast_names) * 0.3)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0)
    ax.set_xticks(range(len(blt_names)))
    ax.set_xticklabels(blt_names, rotation=90, fontsize=6)
    ax.set_yticks(range(len(ast_names)))
    ax.set_yticklabels(ast_names, fontsize=6)
    fig.colorbar(im, ax=ax, label="Mean Jaccard (layers)")
    ax.set_title("AST–Builtin similarity matrix")
    ax.set_xlabel("Builtin")
    ax.set_ylabel("AST node")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved: {path}")


def plot_bipartite_graph(edge_list: pd.DataFrame, path: Path):
    try:
        import networkx as nx
    except ImportError:
        print("  [skip graph] networkx not installed. pip install networkx")
        return

    G = nx.Graph()
    ast_nodes = edge_list["ast_node"].unique().tolist()
    blt_nodes = edge_list["builtin_obj"].unique().tolist()

    for n in ast_nodes:
        G.add_node(n, bipartite=0, family=ast_family(n) or "other")
    for n in blt_nodes:
        G.add_node(n, bipartite=1, domain=builtin_domain(n) or "other")

    for _, row in edge_list.iterrows():
        G.add_edge(row["ast_node"], row["builtin_obj"], weight=row["weight"])

    # Bipartite layout
    pos = {}
    for xi, n in enumerate(ast_nodes):
        pos[n] = (0, xi)
    for xi, n in enumerate(blt_nodes):
        pos[n] = (2, xi)

    fig, ax = plt.subplots(figsize=(10, max(6, max(len(ast_nodes), len(blt_nodes)) * 0.4)))

    # Color AST nodes by family
    families = list({G.nodes[n].get("family", "other") for n in ast_nodes})
    fam_cmap = plt.get_cmap("tab10", len(families))
    fam_color = {f: fam_cmap(i) for i, f in enumerate(families)}
    ast_colors = [fam_color[G.nodes[n].get("family", "other")] for n in ast_nodes]

    # Color builtin nodes by domain
    domains = list({G.nodes[n].get("domain", "other") for n in blt_nodes})
    dom_cmap = plt.get_cmap("Set2", len(domains))
    dom_color = {d: dom_cmap(i) for i, d in enumerate(domains)}
    blt_colors = [dom_color[G.nodes[n].get("domain", "other")] for n in blt_nodes]

    nx.draw_networkx_nodes(G, pos, nodelist=ast_nodes, node_color=ast_colors,
                           node_size=300, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=blt_nodes, node_color=blt_colors,
                           node_size=300, ax=ax, node_shape="s")
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=[w * 5 for w in weights],
                           alpha=0.6, ax=ax, edge_color="gray")
    nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)

    ax.set_title("AST–Builtin bipartite graph (circles=AST, squares=builtin)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved: {path}")


def main():
    args = parse_args()
    atlas_path = resolve(args.atlas)
    out_dir = resolve(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading atlas: {atlas_path}")
    atlas = load_atlas_hdf5(str(atlas_path))
    try:
        ast_masks = atlas["universal_masks"]["ast"]
        blt_masks = atlas["universal_masks"]["builtin"]
        print(f"  {len(ast_masks)} AST nodes, {len(blt_masks)} builtins")

        print("Computing similarity matrix...")
        ast_names, blt_names, matrix = compute_similarity_matrix(ast_masks, blt_masks)

        plot_similarity_matrix(matrix, ast_names, blt_names,
                               path=out_dir / "ast_builtin_similarity_matrix.png")

        # Build edge list: top-K per AST node
        edge_rows = []
        for i, a in enumerate(ast_names):
            row = matrix[i]
            top_k_idx = np.argsort(row)[::-1][:args.top_k]
            for j in top_k_idx:
                if row[j] > 0:
                    edge_rows.append({
                        "ast_node": a,
                        "builtin_obj": blt_names[j],
                        "weight": float(row[j]),
                        "ast_family": ast_family(a) or "other",
                        "builtin_domain": builtin_domain(blt_names[j]) or "other",
                    })

        edge_list = pd.DataFrame(edge_rows).sort_values("weight", ascending=False)
        csv_path = out_dir / "edge_list.csv"
        edge_list.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")
        print(edge_list.head(20).to_string(index=False))

        plot_bipartite_graph(edge_list, path=out_dir / "bipartite_graph.png")
    finally:
        atlas["handle"].close()


if __name__ == "__main__":
    main()
