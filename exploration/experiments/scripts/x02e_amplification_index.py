#!/usr/bin/env python3
"""
X02-E — Amplification Index

Computes AI = mean_cross_domain_cosine / resid_cross_domain_cosine per
(pair, layer). AI > 1 means the circuit amplifies cross-domain similarity
relative to the residual stream; AI < 1 means it suppresses it.

Usage:
    python experiments/scripts/x02e_amplification_index.py
    python experiments/scripts/x02e_amplification_index.py \
        --csv experiments/outputs/x02/ranked_circuits.csv \
        --out experiments/outputs/x02e
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[2]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="experiments/outputs/x02/ranked_circuits.csv")
    p.add_argument("--out", default="experiments/outputs/x02e")
    return p.parse_args()


def resolve(path: str) -> Path:
    pp = Path(path)
    return pp if pp.is_absolute() else ROOT / pp


def main():
    args = parse_args()
    csv_path = resolve(args.csv)
    out_dir = resolve(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found — run X02 Phase B first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    # Compute amplification index
    df["amplification_index"] = np.where(
        df["resid_cross_domain_cosine"] > 0,
        df["mean_cross_domain_cosine"] / df["resid_cross_domain_cosine"],
        float("nan"),
    )

    out_cols = ["pair", "layer", "mean_cross_domain_cosine",
                "resid_cross_domain_cosine", "amplification_index",
                "jsd", "interpretation"]
    out_df = df[out_cols].copy()
    csv_out = out_dir / "amplification_index.csv"
    out_df.to_csv(csv_out, index=False)
    print(f"Saved: {csv_out}  ({len(out_df)} rows)")

    valid = out_df.dropna(subset=["amplification_index"])
    if not valid.empty:
        print(f"  AI mean={valid['amplification_index'].mean():.4f}, "
              f"median={valid['amplification_index'].median():.4f}")
        amplifiers = (valid["amplification_index"] > 1).sum()
        suppressors = (valid["amplification_index"] < 1).sum()
        print(f"  amplifiers (AI>1): {amplifiers}, suppressors (AI<1): {suppressors}")


if __name__ == "__main__":
    main()
