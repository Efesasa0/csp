#!/usr/bin/env python3
"""
X05 — Neuron Circuit Membership Heatmap

Hypothesis: Some neurons participate in many circuits (hubs); others are
circuit-specific. Distribution varies by layer.

Offline — uses atlas HDF5 only.

Usage:
    python experiments/scripts/x05_neuron_circuit_membership.py
    python experiments/scripts/x05_neuron_circuit_membership.py \
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

from analysis.mask_ops import neuron_membership_counts
from module2.io_utils import load_atlas_hdf5


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--atlas", default="data/small_dynamic_feature_atlas.h5")
    p.add_argument("--out", default="experiments/outputs/x05")
    p.add_argument("--n-neurons", type=int, default=2048)
    p.add_argument("--top-k", type=int, default=20,
                   help="Number of top neurons to report per layer")
    return p.parse_args()


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def main():
    args = parse_args()
    atlas_path = resolve(args.atlas)
    out_dir = resolve(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading atlas: {atlas_path}")
    atlas = load_atlas_hdf5(str(atlas_path))
    try:
        pair_masks = atlas["pair_masks"]
        meta = atlas["metadata"]
        n_layers = int(meta.get("n_layers", 8))
        n_neurons = args.n_neurons
        n_pairs = len(pair_masks)
        print(f"  {n_pairs} pairs, {n_layers} layers, {n_neurons} neurons")

        print("Computing neuron membership counts...")
        counts = neuron_membership_counts(pair_masks, n_layers=n_layers, n_neurons=n_neurons)
        # shape: (n_layers, n_neurons)

        prevalence = counts / n_pairs  # normalize

        # ── Heatmap: 8 layers × 2048 neurons ──────────────────────────────────
        fig, ax = plt.subplots(figsize=(16, 4))
        im = ax.imshow(counts, aspect="auto", cmap="hot_r", vmin=0)
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels([f"L{i}" for i in range(n_layers)])
        ax.set_xlabel("Neuron index")
        ax.set_ylabel("Layer")
        ax.set_title(f"Neuron circuit membership counts ({n_pairs} pairs)")
        fig.colorbar(im, ax=ax, label="# circuits")
        fig.tight_layout()
        heatmap_path = out_dir / "neuron_membership_heatmap.png"
        fig.savefig(heatmap_path, dpi=120)
        plt.close(fig)
        print(f"  saved: {heatmap_path}")

        # ── Per-layer histogram ────────────────────────────────────────────────
        fig, axes = plt.subplots(
            2, (n_layers + 1) // 2,
            figsize=(3 * ((n_layers + 1) // 2), 5),
            sharey=False,
        )
        axes_flat = axes.flatten()
        for lid in range(n_layers):
            ax = axes_flat[lid]
            ax.hist(counts[lid], bins=30, color="steelblue", edgecolor="none")
            ax.set_title(f"Layer {lid}", fontsize=9)
            ax.set_xlabel("# circuits", fontsize=8)
            ax.set_ylabel("# neurons", fontsize=8)
            ax.tick_params(labelsize=7)
        # hide unused subplots
        for idx in range(n_layers, len(axes_flat)):
            axes_flat[idx].set_visible(False)
        fig.suptitle("Per-layer neuron membership histogram", fontsize=11)
        fig.tight_layout()
        hist_path = out_dir / "per_layer_histogram.png"
        fig.savefig(hist_path, dpi=120)
        plt.close(fig)
        print(f"  saved: {hist_path}")

        # ── Top neurons per layer CSV ──────────────────────────────────────────
        top_rows = []
        for lid in range(n_layers):
            top_idx = np.argsort(counts[lid])[::-1][:args.top_k]
            for rank, nid in enumerate(top_idx):
                top_rows.append({
                    "layer": lid,
                    "rank": rank + 1,
                    "neuron_id": int(nid),
                    "n_circuits": int(counts[lid, nid]),
                    "prevalence": float(prevalence[lid, nid]),
                })
        df = pd.DataFrame(top_rows)
        csv_path = out_dir / "top_neurons_per_layer.csv"
        df.to_csv(csv_path, index=False)
        print(f"  saved: {csv_path}")
        print(df[df["rank"] <= 5].to_string(index=False))

        # ── Summary stats ─────────────────────────────────────────────────────
        print("\nSummary (mean membership count per layer):")
        for lid in range(n_layers):
            c = counts[lid]
            print(f"  Layer {lid}: mean={c.mean():.2f}, max={c.max()}, "
                  f"hub_neurons (>50% of pairs)={int((c > n_pairs * 0.5).sum())}")
    finally:
        atlas["handle"].close()


if __name__ == "__main__":
    main()
