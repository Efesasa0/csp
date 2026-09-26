#!/usr/bin/env python3
"""
X03 — Layer Localization of Concepts

Hypothesis: Some programming concepts (e.g., iteration, exceptions) localize
to specific layers.

Offline — uses atlas HDF5 only.

Usage:
    python experiments/scripts/x03_layer_localization.py
    python experiments/scripts/x03_layer_localization.py \
        --atlas data/test_5x5x10_validated_prompts.h5
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "src"))

from analysis.mask_ops import circuit_density_by_layer
from module2.io_utils import load_atlas_hdf5


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--atlas", default="data/small_dynamic_feature_atlas.h5")
    p.add_argument("--out", default="experiments/outputs/x03")
    p.add_argument("--n-neurons", type=int, default=2048)
    return p.parse_args()


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def localization_entropy(density: np.ndarray) -> float:
    """H = -Σ p_l * log(p_l), where p_l = density[l] / sum(density). Low = localized."""
    total = density.sum()
    if total == 0:
        return float("nan")
    p = density / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def plot_density_heatmap(density_dict: dict, title: str, path: Path):
    concepts = sorted(density_dict.keys())
    n_layers = max(len(v) for v in density_dict.values())
    matrix = np.array([density_dict[c] for c in concepts])  # (n_concepts, n_layers)

    fig, ax = plt.subplots(figsize=(max(6, n_layers * 0.8), max(4, len(concepts) * 0.4)))
    im = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0)
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([f"L{i}" for i in range(n_layers)])
    ax.set_yticks(range(len(concepts)))
    ax.set_yticklabels(concepts, fontsize=7)
    fig.colorbar(im, ax=ax, label="Density (fraction of neurons active)")
    ax.set_title(title)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Concept")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved: {path}")


def plot_stacked_bar(density_dict: dict, title: str, path: Path):
    concepts = sorted(density_dict.keys())
    n_layers = max(len(v) for v in density_dict.values())
    matrix = np.array([density_dict[c] for c in concepts])  # (n_concepts, n_layers)

    fig, ax = plt.subplots(figsize=(max(6, n_layers * 0.8), 5))
    bottom = np.zeros(n_layers)
    cmap = plt.get_cmap("tab20", len(concepts))
    for idx, concept in enumerate(concepts):
        ax.bar(range(n_layers), matrix[idx], bottom=bottom,
               label=concept, color=cmap(idx), alpha=0.85)
        bottom += matrix[idx]

    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([f"L{i}" for i in range(n_layers)])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Total density (stacked)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=6, ncol=2)
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
        universal = atlas["universal_masks"]
        meta = atlas["metadata"]
        n_layers = int(meta.get("n_layers", 8))
        n_neurons = args.n_neurons
        print(f"  AST concepts: {len(universal['ast'])}, Builtin concepts: {len(universal['builtin'])}")

        rows = []
        for kind in ("ast", "builtin"):
            masks = universal[kind]
            density_dict = circuit_density_by_layer(masks, n_neurons=n_neurons)

            # Pad to n_layers in case some layers are missing
            for concept in density_dict:
                arr = density_dict[concept]
                if len(arr) < n_layers:
                    padded = np.zeros(n_layers)
                    padded[:len(arr)] = arr
                    density_dict[concept] = padded

            plot_density_heatmap(
                density_dict,
                title=f"Layer density — {kind} concepts",
                path=out_dir / f"concept_density_heatmap_{kind}.png",
            )
            plot_stacked_bar(
                density_dict,
                title=f"Stacked layer density — {kind} concepts",
                path=out_dir / f"stacked_bar_layer_density_{kind}.png",
            )

            for concept, density in density_dict.items():
                H = localization_entropy(density)
                peak_layer = int(np.argmax(density))
                rows.append({
                    "kind": kind,
                    "concept": concept,
                    "localization_entropy": H,
                    "peak_layer": peak_layer,
                    "total_density": float(density.sum()),
                    **{f"density_L{i}": float(density[i]) for i in range(len(density))},
                })

        df = pd.DataFrame(rows).sort_values(["kind", "localization_entropy"])
        csv_path = out_dir / "localization_scores.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")

        summary_cols = ["kind", "concept", "localization_entropy", "peak_layer", "total_density"]
        print(df[summary_cols].to_string(index=False))
    finally:
        atlas["handle"].close()


if __name__ == "__main__":
    main()
