"""
02g_class_neurons.py  (submission version)

Computes per-class selectivity for MLP neurons and residual stream dimensions,
splits them into core vs class-specific categories, and saves the two output
files consumed by downstream scripts.

OUTPUTS  (under out_dir)
  02g_<stem>_selectivity_neurons.npz   sel (C, L, mlp_dim), core, class_neurons, classes
  02g_<stem>_selectivity_residual.npz  sel (C, L+1, H), core, class_neurons, classes
  02g_<stem>_class_neurons.json        per-class neuron lists (used by check_neurons.py
                                       and 03n_circuit_bundling_interactive.py)

Usage
-----
  python 02g_class_neurons.py --stem contrastive_stubs --in_dir ../data_more
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Class-specific neuron selectivity arrays")
    p.add_argument("--stem",        default="contrastive_stubs")
    p.add_argument("--in_dir",      default="../data_more",
                   help="Directory with 01_/02_ output files")
    p.add_argument("--out_dir",     default="../results/output",
                   help="Root output directory")
    p.add_argument("--sel_thresh",  type=float, default=0.5,
                   help="Min z-score for a neuron to be class-selective")
    p.add_argument("--core_thresh", type=float, default=0.25,
                   help="Max std across classes to be considered 'core'")
    p.add_argument("--min_samples", type=int, default=10,
                   help="Minimum prompts per class; sparser classes are skipped")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(in_dir: Path, stem: str):
    resid = np.load(in_dir / f"01_{stem}_residual_all.npy")   # (N, L+1, H)
    mlp   = np.load(in_dir / f"01_{stem}_mlp_neurons.npy")    # (N, L, mlp_dim)
    meta  = json.load(open(in_dir / f"01_{stem}_meta.json"))
    pm    = np.load(in_dir / f"02_{stem}_purity_masks.npz")
    return resid, mlp, meta, pm


# ─────────────────────────────────────────────────────────────────────────────
# Selectivity
# ─────────────────────────────────────────────────────────────────────────────

def compute_selectivity(acts: np.ndarray, labels: np.ndarray,
                        classes: list[str], min_samples: int) -> np.ndarray:
    """
    acts:   (N, D)  activations at one layer
    labels: (N,)    int class index
    Returns sel (C, D) — z-score of each class's mean vs global distribution.
    """
    C = len(classes)
    D = acts.shape[1]
    mu_all  = acts.mean(axis=0)
    std_all = acts.std(axis=0) + 1e-8
    sel = np.zeros((C, D), dtype=np.float32)
    for c in range(C):
        mask = labels == c
        if mask.sum() < min_samples:
            continue
        sel[c] = (acts[mask].mean(axis=0) - mu_all) / std_all
    return sel


def compute_selectivity_all_layers(acts_all: np.ndarray, labels: np.ndarray,
                                   classes: list[str], min_samples: int) -> np.ndarray:
    """acts_all (N, L, D) -> sel_all (C, L, D)"""
    N, L, D = acts_all.shape
    C = len(classes)
    sel_all = np.zeros((C, L, D), dtype=np.float32)
    for l in range(L):
        sel_all[:, l, :] = compute_selectivity(acts_all[:, l, :], labels,
                                               classes, min_samples)
    return sel_all


# ─────────────────────────────────────────────────────────────────────────────
# Core / class-specific split
# ─────────────────────────────────────────────────────────────────────────────

def split_core_class(sel_all: np.ndarray,
                     ast_pure_mask: np.ndarray,
                     sel_thresh: float,
                     core_thresh: float):
    """
    sel_all       (C, L, D)
    ast_pure_mask (L, D) bool — from 02_variance_partition purity masks

    Returns
      core      (L, D) bool  — ast_pure AND fires uniformly for all classes
      class_nrn (C, L, D) bool — selective for exactly one class, not core
    """
    sel_std    = sel_all.std(axis=0)           # (L, D)
    core       = ast_pure_mask & (sel_std < core_thresh)

    sel_max    = sel_all.max(axis=0)
    sel_argmax = sel_all.argmax(axis=0)

    C = sel_all.shape[0]
    class_nrn = np.zeros_like(sel_all, dtype=bool)
    for c in range(C):
        class_nrn[c] = (sel_argmax == c) & (sel_max > sel_thresh) & ~core

    return core, class_nrn


# ─────────────────────────────────────────────────────────────────────────────
# Build JSON output
# ─────────────────────────────────────────────────────────────────────────────

def build_class_neurons_json(class_nrn_r: np.ndarray,
                              class_nrn_n: np.ndarray,
                              core_r: np.ndarray,
                              core_n: np.ndarray,
                              classes: list[str]) -> dict:
    """Returns a JSON-serialisable dict with per-class neuron lists."""
    out: dict = {"classes": classes, "residual": {}, "neurons": {}}

    for c, cls in enumerate(classes):
        out["residual"][cls] = {
            "per_layer": {
                f"L{l:02d}": np.where(class_nrn_r[c, l])[0].tolist()
                for l in range(class_nrn_r.shape[1])
            },
            "any_layer": np.where(class_nrn_r[c].any(axis=0))[0].tolist(),
        }
        out["neurons"][cls] = {
            "per_layer": {
                f"L{l:02d}": np.where(class_nrn_n[c, l])[0].tolist()
                for l in range(class_nrn_n.shape[1])
            },
            "any_layer": np.where(class_nrn_n[c].any(axis=0))[0].tolist(),
        }

    out["core_residual_per_layer"] = {
        f"L{l:02d}": np.where(core_r[l])[0].tolist()
        for l in range(core_r.shape[0])
    }
    out["core_neurons_per_layer"] = {
        f"L{l:02d}": np.where(core_n[l])[0].tolist()
        for l in range(core_n.shape[0])
    }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    in_dir  = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data …")
    resid, mlp, meta, pm = load_data(in_dir, args.stem)

    N, Lr, H        = resid.shape
    _,  Ln, mlp_dim = mlp.shape

    classes  = sorted(set(m["ast_node"] for m in meta))
    cls_to_i = {c: i for i, c in enumerate(classes)}
    ast_idx  = np.array([cls_to_i[m["ast_node"]] for m in meta], dtype=np.int32)
    C = len(classes)
    print(f"  {N} prompts · {C} AST classes · residual ({Lr},{H}) · mlp ({Ln},{mlp_dim})")

    ast_pure_r = pm["residual_ast_pure"]   # (Lr, H)
    ast_pure_n = pm["neurons_ast_pure"]    # (Ln, mlp_dim)

    print("Computing selectivity (residual) …")
    sel_r = compute_selectivity_all_layers(resid, ast_idx, classes, args.min_samples)

    print("Computing selectivity (MLP neurons) …")
    sel_n = compute_selectivity_all_layers(mlp, ast_idx, classes, args.min_samples)

    print("Splitting core vs class-specific …")
    core_r, class_nrn_r = split_core_class(sel_r, ast_pure_r,
                                            args.sel_thresh, args.core_thresh)
    core_n, class_nrn_n = split_core_class(sel_n, ast_pure_n,
                                            args.sel_thresh, args.core_thresh)

    print(f"  Residual  core: {core_r.sum()}  class-specific: {class_nrn_r.sum()}")
    print(f"  MLP       core: {core_n.sum()}  class-specific: {class_nrn_n.sum()}")
    per_r = class_nrn_r.sum(axis=(1, 2))
    per_n = class_nrn_n.sum(axis=(1, 2))
    for i, cls in enumerate(classes):
        if per_r[i] > 0 or per_n[i] > 0:
            print(f"    {cls:20s}  resid={per_r[i]:4d}  mlp={per_n[i]:4d}")

    print("Saving selectivity arrays …")
    np.savez_compressed(
        out_dir / f"02g_{args.stem}_selectivity_residual.npz",
        sel=sel_r, core=core_r, class_neurons=class_nrn_r,
        classes=np.array(classes),
    )
    print(f"  saved 02g_{args.stem}_selectivity_residual.npz")

    np.savez_compressed(
        out_dir / f"02g_{args.stem}_selectivity_neurons.npz",
        sel=sel_n, core=core_n, class_neurons=class_nrn_n,
        classes=np.array(classes),
    )
    print(f"  saved 02g_{args.stem}_selectivity_neurons.npz")

    print("Saving class_neurons.json …")
    cn_json = build_class_neurons_json(class_nrn_r, class_nrn_n,
                                       core_r, core_n, classes)
    with open(out_dir / f"02g_{args.stem}_class_neurons.json", "w") as f:
        json.dump(cn_json, f, indent=2)
    print(f"  saved 02g_{args.stem}_class_neurons.json")

    print("\nDone.")


if __name__ == "__main__":
    main()
