#!/usr/bin/env python3
"""
X02-D Phase B — Positional JSD Analysis (CPU-only)

Reads the positional activations HDF5 and computes JSD at each token position,
revealing how domain sensitivity evolves along the sequence.

Usage:
    python experiments/scripts/x02d_positional_analysis.py
    python experiments/scripts/x02d_positional_analysis.py \
        --acts experiments/outputs/x02d/positional_activations.h5 \
        --out experiments/outputs/x02d
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[2]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--acts",
                   default="experiments/outputs/x02d/positional_activations.h5")
    p.add_argument("--out", default="experiments/outputs/x02d")
    return p.parse_args()


def resolve(path: str) -> Path:
    pp = Path(path)
    return pp if pp.is_absolute() else ROOT / pp


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def jensen_shannon_divergence(vectors: list[np.ndarray]) -> float:
    if len(vectors) < 2:
        return float("nan")
    distributions = np.array([softmax(np.abs(v).astype(np.float64)) for v in vectors])
    mean_dist = distributions.mean(axis=0)
    eps = 1e-12

    def entropy(p):
        p = p[p > eps]
        return -np.sum(p * np.log(p))

    H_mean = entropy(mean_dist)
    mean_H = np.mean([entropy(d) for d in distributions])
    return float(max(0.0, H_mean - mean_H))


def main():
    args = parse_args()
    acts_path = resolve(args.acts)
    out_dir = resolve(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not acts_path.exists():
        print(f"ERROR: {acts_path} not found — run x02d_extract_positional.py first.")
        sys.exit(1)

    print(f"Loading: {acts_path}")
    rows = []

    with h5py.File(str(acts_path), "r") as f:
        n_layers = int(f["metadata"].attrs.get("n_layers", 8))
        pos_labels = [x.decode() if isinstance(x, bytes) else x
                      for x in f["metadata/pos_labels"][:]]
        domain_names = [x.decode() if isinstance(x, bytes) else x
                        for x in f["domain_vocab/names"][:]]
        n_domains = len(domain_names)

        pairs = list(f["pairs"].keys())
        print(f"  {len(pairs)} pairs, {len(pos_labels)} positions, {n_layers} layers")

        for pair_key in pairs:
            pg = f[f"pairs/{pair_key}"]
            domain_ids = pg["prompt_meta/domain_id"][:]

            for pos_label in pos_labels:
                for lid in range(n_layers):
                    ds_key = f"pos_{pos_label}/layer_{lid}/acts_proj_f16"
                    if ds_key not in pg:
                        continue

                    acts = pg[ds_key][:].astype(np.float32)

                    domain_vecs = []
                    for did in range(n_domains):
                        idx = np.where(domain_ids == did)[0]
                        if len(idx) == 0:
                            continue
                        domain_vecs.append(acts[idx].mean(axis=0))

                    jsd = jensen_shannon_divergence(domain_vecs)
                    rows.append({
                        "pair": pair_key,
                        "position": pos_label,
                        "layer": lid,
                        "jsd": jsd,
                        "n_domains_present": len(domain_vecs),
                    })

    rdf = pd.DataFrame(rows)
    csv_out = out_dir / "jsd_by_position.csv"
    rdf.to_csv(csv_out, index=False)
    print(f"\nSaved: {csv_out}  ({len(rdf)} rows)")

    if not rdf.empty:
        pivot = rdf.groupby("position")["jsd"].mean()
        print("\nMean JSD by position:")
        for pos in pos_labels:
            if pos in pivot.index:
                print(f"  {pos:>5s}: {pivot[pos]:.6f}")


if __name__ == "__main__":
    main()
