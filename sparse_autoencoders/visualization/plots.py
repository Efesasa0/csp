"""Consolidated plotting utilities."""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

from core import load_config, get_path


def plot_tug_of_war(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    layers = [d["layer"] for d in data]
    plt.figure(figsize=(8, 4))
    plt.plot(layers, [d["ideal_logit"] for d in data], label="Logic", color="green")
    plt.plot(
        layers, [d["actual_logit"] for d in data], label="Bias", color="red", ls="--"
    )
    plt.title("Logit Tug-of-War")
    plt.xlabel("Layer")
    plt.ylabel("Logit")
    plt.legend()
    plt.show()


def generate_circuit_hierarchy():
    cfg = load_config()
    with open(get_path(cfg, "reports", cfg["paths"]["master_report"]), "r") as f:
        master_data = json.load(f)

    nodes = [k for k in master_data.keys() if not k.startswith("_")]

    max_head = 0
    for node in nodes:
        for edge in master_data[node]["upstream"]:
            head_idx = int(edge["source"].split("H")[-1])
            if head_idx > max_head:
                max_head = head_idx
    n_heads = max_head + 1

    fingerprints = []
    for node in nodes:
        vector = np.zeros(n_heads)
        for edge in master_data[node]["upstream"]:
            head_idx = int(edge["source"].split("H")[-1])
            vector[head_idx] = edge["weight"]
        fingerprints.append(vector)

    fp_matrix = np.array(fingerprints)
    dist_matrix = pdist(fp_matrix, metric="cosine")
    linked = linkage(dist_matrix, method="ward")

    fig_width = max(14, len(nodes) * 0.6)
    plt.figure(figsize=(fig_width, 8))
    dendrogram(linked, labels=nodes, leaf_rotation=90, leaf_font_size=8)
    layer = master_data.get("_meta", {}).get("layer", "?")
    plt.title(f"Mechanistic Hierarchy: AST Node Similarity by Layer {layer} Attribution")
    plt.ylabel("Ward Distance (Functional Divergence)")

    out_path = get_path(cfg, "results", "circuit_hierarchy.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Hierarchical graph saved to {out_path}")


def generate_coactivation_dendrogram(layer=7, site="mlp"):
    """Dendrogram of AST nodes clustered by co-activation fingerprints.

    Each node gets a vector of co-firing rates with every other node's
    best feature, taken from the cached co-activation analysis.  Nodes
    that share similar co-firing partners end up close in the tree.
    """
    cfg = load_config()
    cache_path = os.path.join(
        get_path(cfg, "cache", ""), "advanced",
        f"coact_L{layer}_{site}.json",
    )
    with open(cache_path, "r") as f:
        coact = json.load(f)

    nodes = sorted(k for k in coact.keys() if not k.startswith("_") and isinstance(coact[k], dict))
    if len(nodes) < 3:
        print(f"Skipping coactivation dendrogram: only {len(nodes)} nodes")
        return

    # Collect the union of all partner feature ids to build a common axis
    all_partners = set()
    for node in nodes:
        for entry in coact[node].get("top_cofiring", []):
            all_partners.add(entry["feature"])
    partner_list = sorted(all_partners)
    partner_idx = {f: i for i, f in enumerate(partner_list)}

    # Build fingerprint matrix: rows = nodes, cols = partner features
    matrix = np.zeros((len(nodes), len(partner_list)))
    for i, node in enumerate(nodes):
        for entry in coact[node].get("top_cofiring", []):
            matrix[i, partner_idx[entry["feature"]]] = entry["coact_rate"]

    dist = pdist(matrix, metric="cosine")
    # Replace NaNs from zero-vectors with max distance
    dist = np.nan_to_num(dist, nan=1.0)
    linked = linkage(dist, method="ward")

    fig_width = max(14, len(nodes) * 0.7)
    plt.figure(figsize=(fig_width, 8))
    dendrogram(linked, labels=nodes, leaf_rotation=90, leaf_font_size=9)
    plt.title(f"Co-activation Dendrogram: L{layer}/{site}")
    plt.ylabel("Ward Distance (Co-firing Divergence)")
    plt.tight_layout()

    out_path = get_path(cfg, "results", f"dendrogram_coactivation_L{layer}_{site}.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Co-activation dendrogram saved to {out_path}")


def generate_positional_dendrogram(layer=7, site="mlp"):
    """Dendrogram of AST nodes clustered by positional firing patterns.

    Each node's fingerprint is [mean_rel_position, std_rel_position,
    median_rel_position, q25, q50, q75].  Nodes that fire in similar
    sequence positions cluster together.
    """
    cfg = load_config()
    cache_path = os.path.join(
        get_path(cfg, "cache", ""), "advanced2",
        f"positional_L{layer}_{site}.json",
    )
    with open(cache_path, "r") as f:
        pos_data = json.load(f)

    nodes = sorted(pos_data.keys())
    if len(nodes) < 3:
        print(f"Skipping positional dendrogram: only {len(nodes)} nodes")
        return

    matrix = []
    for node in nodes:
        d = pos_data[node]
        q = d.get("position_quartiles", [0.25, 0.5, 0.75])
        vec = [
            d.get("mean_rel_position", 0.5),
            d.get("std_rel_position", 0.3),
            d.get("median_rel_position", 0.5),
        ] + list(q)
        matrix.append(vec)

    fp_matrix = np.array(matrix)
    dist = pdist(fp_matrix, metric="euclidean")
    linked = linkage(dist, method="ward")

    fig_width = max(14, len(nodes) * 0.7)
    plt.figure(figsize=(fig_width, 8))
    dendrogram(linked, labels=nodes, leaf_rotation=90, leaf_font_size=9)
    plt.title(f"Positional Firing Dendrogram: L{layer}/{site}")
    plt.ylabel("Ward Distance (Positional Divergence)")
    plt.tight_layout()

    out_path = get_path(cfg, "results", f"dendrogram_positional_L{layer}_{site}.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Positional dendrogram saved to {out_path}")


def generate_feature_similarity_dendrogram():
    """Dendrogram of AST nodes clustered by multi-feature fingerprint overlap.

    Uses the deep discovery report to build a binary fingerprint of which
    SAE features each node uses (top-5), then clusters by Jaccard distance.
    """
    cfg = load_config()
    report_path = get_path(cfg, "reports", "deep_discovery_report.json")
    if not os.path.exists(report_path):
        print("Skipping feature similarity dendrogram: no deep_discovery_report.json")
        return

    with open(report_path, "r") as f:
        deep = json.load(f)

    # Find fingerprints sub-dict (keyed like "fingerprints_L7_mlp")
    fp_data = None
    for key in deep:
        if key.startswith("fingerprints_"):
            fp_data = deep[key]
            break
    if fp_data is None:
        # Fallback: try master report format
        fp_data = {k: v for k, v in deep.items() if not k.startswith("_") and isinstance(v, dict)}

    nodes = []
    all_feats = set()
    node_feats = {}
    for node, data in fp_data.items():
        if node.startswith("_"):
            continue
        feats = set()
        if isinstance(data, dict):
            # fingerprints format: top_k_features list
            for f in data.get("top_k_features", data.get("top_features", data.get("all_winners", [])))[:10]:
                feats.add(int(f) if not isinstance(f, dict) else int(f.get("feature", 0)))
        if feats:
            nodes.append(node)
            node_feats[node] = feats
            all_feats |= feats

    if len(nodes) < 3:
        print(f"Skipping feature similarity dendrogram: only {len(nodes)} nodes")
        return

    feat_list = sorted(all_feats)
    feat_idx = {f: i for i, f in enumerate(feat_list)}

    matrix = np.zeros((len(nodes), len(feat_list)))
    for i, node in enumerate(nodes):
        for f in node_feats[node]:
            matrix[i, feat_idx[f]] = 1.0

    dist = pdist(matrix, metric="jaccard")
    dist = np.nan_to_num(dist, nan=1.0)
    linked = linkage(dist, method="ward")

    fig_width = max(14, len(nodes) * 0.7)
    plt.figure(figsize=(fig_width, 8))
    dendrogram(linked, labels=nodes, leaf_rotation=90, leaf_font_size=9)
    plt.title("Feature Fingerprint Dendrogram: AST Node Similarity by Shared SAE Features")
    plt.ylabel("Ward Distance (Jaccard)")
    plt.tight_layout()

    out_path = get_path(cfg, "results", "dendrogram_feature_similarity.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Feature similarity dendrogram saved to {out_path}")


def generate_all_dendrograms():
    """Generate all dendrogram variants."""
    print("\n=== Generating all dendrograms ===\n")

    # 1. Original attention-based hierarchy
    try:
        generate_circuit_hierarchy()
    except Exception as e:
        print(f"  circuit_hierarchy failed: {e}")

    # 2. Co-activation dendrograms for all cached layer/site combos
    cfg = load_config()
    for layer in range(4, 8):
        for site in ["mlp", "resid"]:
            cache = os.path.join(
                get_path(cfg, "cache", ""), "advanced",
                f"coact_L{layer}_{site}.json",
            )
            if os.path.exists(cache):
                try:
                    generate_coactivation_dendrogram(layer, site)
                except Exception as e:
                    print(f"  coact L{layer}/{site} failed: {e}")

    # 3. Positional dendrograms
    for layer in range(4, 8):
        for site in ["mlp", "resid"]:
            cache = os.path.join(
                get_path(cfg, "cache", ""), "advanced2",
                f"positional_L{layer}_{site}.json",
            )
            if os.path.exists(cache):
                try:
                    generate_positional_dendrogram(layer, site)
                except Exception as e:
                    print(f"  positional L{layer}/{site} failed: {e}")

    # 4. Feature similarity dendrogram
    try:
        generate_feature_similarity_dendrogram()
    except Exception as e:
        print(f"  feature_similarity failed: {e}")

    print("\n=== Dendrogram generation complete ===")


if __name__ == "__main__":
    generate_all_dendrograms()
