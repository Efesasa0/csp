"""
02l_head_selectivity.py  (submission version — complete)

Identifies attention heads uniquely responsible for individual AST node classes
using logit-attribution scores.

OUTPUTS  (under out_dir)
  02l_<stem>_head_sel.npz         sel (C, L, H), owner (L, H), ast_pure (L, H)
  02l_<stem>_head_ownership.json  per-class per-layer lists of owned heads

These files are consumed by 02m_head_viz.py and 03n_circuit_bundling_interactive.py.

Usage
-----
  python 02l_head_selectivity.py --stem contrastive_stubs --in_dir ../data_more
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.patches import Patch


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Attention head selectivity per AST class")
    p.add_argument("--stem",        default="contrastive_stubs")
    p.add_argument("--in_dir",      default="../data_more")
    p.add_argument("--out_dir",     default="../results/output")
    p.add_argument("--ua_thresh",   type=float, default=0.10,
                   help="Min unique_ast for AST-pure head (top ~30%%)")
    p.add_argument("--purity_frac", type=float, default=0.70,
                   help="Min unique_ast/r2_joint for AST-pure head")
    p.add_argument("--sel_thresh",  type=float, default=0.4,
                   help="Min selectivity z-score for class ownership")
    p.add_argument("--min_samples", type=int, default=10)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(in_dir: Path, stem: str):
    head_attr  = np.load(in_dir / f"01_{stem}_head_attr.npy")   # (N, L, H)
    vp_h       = np.load(in_dir / f"02_{stem}_vp_heads.npz")
    meta       = json.load(open(in_dir / f"01_{stem}_meta.json"))
    labels_str = np.array([m["ast_node"] for m in meta])
    classes    = sorted(set(labels_str))
    label_idx  = np.array([classes.index(s) for s in labels_str])
    return head_attr, vp_h, labels_str, label_idx, classes


# ─────────────────────────────────────────────────────────────────────────────
# Selectivity
# ─────────────────────────────────────────────────────────────────────────────

def compute_selectivity(head_attr: np.ndarray, label_idx: np.ndarray,
                        classes: list, min_samples: int) -> np.ndarray:
    """Returns sel (C, L, H)."""
    N, L, H = head_attr.shape
    C = len(classes)
    mu_all  = head_attr.mean(axis=0)        # (L, H)
    std_all = head_attr.std(axis=0) + 1e-8  # (L, H)
    sel = np.zeros((C, L, H), dtype=np.float32)
    for c in range(C):
        mask = label_idx == c
        if mask.sum() < min_samples:
            continue
        sel[c] = (head_attr[mask].mean(axis=0) - mu_all) / std_all
    return sel


# ─────────────────────────────────────────────────────────────────────────────
# Ownership
# ─────────────────────────────────────────────────────────────────────────────

def build_ownership(sel: np.ndarray, ast_pure: np.ndarray,
                    sel_thresh: float) -> np.ndarray:
    """Returns owner (L, H) — int class index, or -1 if unowned."""
    top_sel = sel.max(axis=0)    # (L, H)
    top_cls = sel.argmax(axis=0)
    owned   = ast_pure & (top_sel > sel_thresh)
    return np.where(owned, top_cls, -1).astype(np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# Ownership JSON
# ─────────────────────────────────────────────────────────────────────────────

def build_ownership_json(owner: np.ndarray, sel: np.ndarray,
                         ua: np.ndarray, classes: list) -> dict:
    C, L, H = sel.shape
    result: dict = {cls: {} for cls in classes}
    for c, cls in enumerate(classes):
        for l in range(L):
            heads = [{"head": int(h), "layer": int(l),
                      "selectivity": float(sel[c, l, h]),
                      "unique_ast":  float(ua[l, h])}
                     for h in range(H) if owner[l, h] == c]
            heads.sort(key=lambda x: -x["selectivity"])
            if heads:
                result[cls][f"L{l}"] = heads
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    in_dir  = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data …")
    head_attr, vp_h, labels_str, label_idx, classes = load_data(in_dir, args.stem)
    N, L, H = head_attr.shape
    C = len(classes)
    print(f"  {N} prompts · {L} layers · {H} heads · {C} classes")

    ua     = vp_h["unique_ast"]    # (L, H)
    rj     = vp_h["r2_joint"]
    purity = np.where(rj > 1e-6, ua / rj, 0.0)
    ast_pure = (ua > args.ua_thresh) & (purity > args.purity_frac)
    print(f"  AST-pure heads (UA>{args.ua_thresh}, purity>{args.purity_frac}): "
          f"{ast_pure.sum()} / {L * H}")
    for l in range(L):
        print(f"    L{l}: {ast_pure[l].sum()} pure heads")

    print("\nComputing selectivity …")
    sel   = compute_selectivity(head_attr, label_idx, classes, args.min_samples)
    owner = build_ownership(sel, ast_pure, args.sel_thresh)

    owned_total = (owner >= 0).sum()
    print(f"  Class-owned heads: {owned_total} / {ast_pure.sum()} pure")
    for c, cls in enumerate(classes):
        n = (owner == c).sum()
        if n > 0:
            print(f"    {cls:20s}: {n} heads")

    np.savez_compressed(out_dir / f"02l_{args.stem}_head_sel.npz",
                        sel=sel, owner=owner, ast_pure=ast_pure.astype(np.uint8))
    print(f"  saved 02l_{args.stem}_head_sel.npz")

    ownership_json = build_ownership_json(owner, sel, ua, classes)
    with open(out_dir / f"02l_{args.stem}_head_ownership.json", "w") as f:
        json.dump(ownership_json, f, indent=2)
    print(f"  saved 02l_{args.stem}_head_ownership.json")

    print("\nDone.")


if __name__ == "__main__":
    main()
