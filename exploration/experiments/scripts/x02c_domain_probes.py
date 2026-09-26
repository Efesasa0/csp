#!/usr/bin/env python3
"""
X02-C — Domain Probes (Linear Classifiability)

Per (pair, layer): 5-fold logistic regression to predict domain from
circuit-projected and residual activations.

Usage:
    python experiments/scripts/x02c_domain_probes.py
    python experiments/scripts/x02c_domain_probes.py \
        --acts experiments/outputs/x02/prompt_level_activations.h5 \
        --out experiments/outputs/x02c
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).parents[2]

warnings.filterwarnings("ignore", category=UserWarning)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--acts",
                   default="experiments/outputs/x02/prompt_level_activations.h5")
    p.add_argument("--out", default="experiments/outputs/x02c")
    return p.parse_args()


def resolve(path: str) -> Path:
    pp = Path(path)
    return pp if pp.is_absolute() else ROOT / pp


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    args = parse_args()
    acts_path = resolve(args.acts)
    out_dir = resolve(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not acts_path.exists():
        print(f"ERROR: {acts_path} not found — run X02 Phase A first.")
        sys.exit(1)

    print(f"Loading activations: {acts_path}")
    rows = []
    with h5py.File(str(acts_path), "r") as f:
        n_layers = int(f["metadata"].attrs.get("n_layers", 8))
        pairs = list(f["pairs"].keys())
        print(f"  {len(pairs)} pairs, {n_layers} layers")

        for pair_key in tqdm(pairs, desc="Probing pairs"):
            pg = f[f"pairs/{pair_key}"]
            domain_ids = pg["prompt_meta/domain_id"][:]
            n_classes = len(np.unique(domain_ids))

            if n_classes < 2:
                continue

            for lid in range(n_layers):
                layer_key = f"layer_{lid}"
                if layer_key not in pg:
                    continue

                acts = pg[f"{layer_key}/acts_proj_f16"][:].astype(np.float32)

                # Circuit probe
                try:
                    clf = LogisticRegression(max_iter=500, solver="saga",
                                            tol=1e-3, n_jobs=-1)
                    scores = cross_val_score(clf, acts, domain_ids, cv=3,
                                            scoring="accuracy")
                    circuit_acc = float(scores.mean())
                except Exception:
                    circuit_acc = float("nan")

                # Residual probe
                resid_acc = float("nan")
                if "resid_f16" in pg[layer_key]:
                    resid = pg[f"{layer_key}/resid_f16"][:].astype(np.float32)
                    try:
                        clf2 = LogisticRegression(max_iter=500, solver="saga",
                                                  tol=1e-3, n_jobs=-1)
                        scores2 = cross_val_score(clf2, resid, domain_ids, cv=3,
                                                  scoring="accuracy")
                        resid_acc = float(scores2.mean())
                    except Exception:
                        resid_acc = float("nan")

                delta = circuit_acc - resid_acc if not (
                    np.isnan(circuit_acc) or np.isnan(resid_acc)) else float("nan")

                rows.append({
                    "pair": pair_key,
                    "layer": lid,
                    "circuit_probe_acc": circuit_acc,
                    "resid_probe_acc": resid_acc,
                    "delta_acc": delta,
                })


    rdf = pd.DataFrame(rows)
    csv_out = out_dir / "probe_results.csv"
    rdf.to_csv(csv_out, index=False)
    print(f"\nSaved: {csv_out}  ({len(rdf)} rows)")

    if not rdf.empty:
        print(f"  circuit_probe_acc: mean={rdf['circuit_probe_acc'].mean():.3f}")
        print(f"  resid_probe_acc:   mean={rdf['resid_probe_acc'].mean():.3f}")
        print(f"  delta_acc:         mean={rdf['delta_acc'].mean():.3f}")


if __name__ == "__main__":
    main()
