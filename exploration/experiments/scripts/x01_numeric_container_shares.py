#!/usr/bin/env python3
"""
X01 — Numeric/Container Circuit Sharing

Hypothesis: numeric builtins (int, float, complex, bool) share circuits;
container builtins (list, tuple, dict, set, frozenset) share circuits.

Offline — uses atlas HDF5 only.

Usage:
    python experiments/scripts/x01_numeric_container_shares.py
    python experiments/scripts/x01_numeric_container_shares.py \
        --atlas data/test_5x5x10_validated_prompts.h5 \
        --out experiments/outputs/x01
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "src"))

from analysis.concept_taxonomy import BUILTIN_DOMAINS
from module2.io_utils import load_atlas_hdf5
from module2.metrics import compute_jaccard_matrix, jaccard_similarity


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--atlas", default="data/small_dynamic_feature_atlas.h5",
                   help="Path to atlas HDF5 (relative to repo root or absolute)")
    p.add_argument("--out", default="experiments/outputs/x01",
                   help="Output directory")
    p.add_argument("--groups", nargs="+", default=["numeric", "container"],
                   help="Builtin domain groups to analyse")
    return p.parse_args()


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def plot_jaccard_heatmap(matrix: np.ndarray, names: list, title: str, path: Path):
    fig, ax = plt.subplots(figsize=(max(4, len(names)), max(4, len(names))))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if matrix[i, j] < 0.5 else "black")
    fig.colorbar(im, ax=ax, label="Jaccard similarity")
    ax.set_title(title)
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
        universal_builtin = atlas["universal_masks"]["builtin"]
        meta = atlas["metadata"]
        n_layers = int(meta.get("n_layers", 8))
        print(f"  {len(universal_builtin)} builtins, {n_layers} layers")

        rows = []
        for group_name in args.groups:
            # Keep only builtins that are both in the taxonomy and in the atlas
            group_members = [b for b in BUILTIN_DOMAINS.get(group_name, [])
                             if b in universal_builtin]
            if len(group_members) < 2:
                print(f"  [skip] group '{group_name}': only {len(group_members)} members in atlas")
                continue

            group_masks = {b: universal_builtin[b] for b in group_members}
            print(f"\nGroup '{group_name}': {group_members}")

            for layer in range(n_layers):
                names = sorted(group_masks.keys())
                matrix = compute_jaccard_matrix(group_masks, layer)

                # Heatmap per layer per group
                plot_jaccard_heatmap(
                    matrix, names,
                    title=f"{group_name} — layer {layer}",
                    path=out_dir / f"heatmap_{group_name}_layer{layer}.png",
                )

                # Within-group mean (upper triangle, excluding diagonal)
                n = len(names)
                upper = [(matrix[i, j], names[i], names[j])
                         for i in range(n) for j in range(i + 1, n)]
                within_mean = np.mean([v for v, _, _ in upper]) if upper else float("nan")
                rows.append({
                    "group": group_name,
                    "layer": layer,
                    "scope": "within",
                    "mean_jaccard": within_mean,
                    "n_pairs": len(upper),
                })

            # Cross-group comparison: group vs all other groups
            other_groups = [g for g in args.groups if g != group_name]
            for other_name in other_groups:
                other_members = [b for b in BUILTIN_DOMAINS.get(other_name, [])
                                 if b in universal_builtin]
                if not other_members:
                    continue
                for layer in range(n_layers):
                    cross_sims = []
                    for a in group_members:
                        ma = universal_builtin[a].get(layer)
                        if ma is None:
                            continue
                        for b in other_members:
                            mb = universal_builtin[b].get(layer)
                            if mb is None:
                                continue
                            cross_sims.append(jaccard_similarity(ma, mb))
                    cross_mean = np.mean(cross_sims) if cross_sims else float("nan")
                    rows.append({
                        "group": f"{group_name}_vs_{other_name}",
                        "layer": layer,
                        "scope": "cross",
                        "mean_jaccard": cross_mean,
                        "n_pairs": len(cross_sims),
                    })

        df = pd.DataFrame(rows)
        csv_path = out_dir / "within_vs_cross_group_jaccard.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")
        print(df.to_string(index=False))
    finally:
        atlas["handle"].close()


if __name__ == "__main__":
    main()
