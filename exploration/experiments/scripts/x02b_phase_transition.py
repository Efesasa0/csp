#!/usr/bin/env python3
"""
X02-B — Phase Transition Detection

Per pair: fit a sigmoid to JSD across 8 layers to detect syntax→meaning
transition points.

Usage:
    python experiments/scripts/x02b_phase_transition.py
    python experiments/scripts/x02b_phase_transition.py \
        --csv experiments/outputs/x02/ranked_circuits.csv \
        --out experiments/outputs/x02b
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[2]

warnings.filterwarnings("ignore", category=RuntimeWarning)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="experiments/outputs/x02/ranked_circuits.csv")
    p.add_argument("--out", default="experiments/outputs/x02b")
    return p.parse_args()


def resolve(path: str) -> Path:
    pp = Path(path)
    return pp if pp.is_absolute() else ROOT / pp


def sigmoid(x, L_trans, k, ymin, ymax):
    """Sigmoid: ymin + (ymax-ymin) / (1 + exp(-k*(x - L_trans)))"""
    return ymin + (ymax - ymin) / (1.0 + np.exp(-k * (x - L_trans)))


def main():
    from scipy.optimize import curve_fit

    args = parse_args()
    csv_path = resolve(args.csv)
    out_dir = resolve(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found — run X02 Phase B first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    pairs = df["pair"].unique()
    print(f"Fitting sigmoid to {len(pairs)} pairs across layers")

    results = []
    for pair in pairs:
        pdf = df[df["pair"] == pair].sort_values("layer")
        x = pdf["layer"].values.astype(float)
        y = pdf["jsd"].values.astype(float)

        if len(x) < 3:
            continue

        try:
            popt, _ = curve_fit(
                sigmoid, x, y,
                p0=[3.5, 1.0, y.min(), y.max()],
                bounds=([0, 0.01, 0, 0], [7, 50, y.max() + 0.01, y.max() * 10 + 0.01]),
                maxfev=5000,
            )
            L_trans, k_steep = popt[0], popt[1]
            y_pred = sigmoid(x, *popt)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        except (RuntimeError, ValueError):
            L_trans, k_steep, r2 = float("nan"), float("nan"), float("nan")

        results.append({
            "pair": pair,
            "L_transition": L_trans,
            "k_steepness": k_steep,
            "r_squared": r2,
            "jsd_min": float(y.min()),
            "jsd_max": float(y.max()),
        })

    rdf = pd.DataFrame(results)
    csv_out = out_dir / "transition_curves.csv"
    rdf.to_csv(csv_out, index=False)
    print(f"Saved: {csv_out}  ({len(rdf)} pairs)")

    # Histogram of transition layer
    valid = rdf.dropna(subset=["L_transition"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(valid["L_transition"], bins=20, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Transition Layer (L_transition)")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of Syntax→Meaning Transition Layer\n"
                 f"({len(valid)} pairs with valid sigmoid fit)")
    ax.axvline(valid["L_transition"].median(), color="red", linestyle="--",
               label=f"median = {valid['L_transition'].median():.2f}")
    ax.legend()
    fig.tight_layout()
    png_path = out_dir / "transition_histogram.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
