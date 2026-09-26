import argparse
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--in_dir", default="data")
in_dir = Path(parser.parse_args().in_dir)

data    = json.load(open(in_dir / "02g_constructor_stubs_class_neurons.json"))
sel     = np.load(in_dir / "02g_constructor_stubs_selectivity_neurons.npz")["sel"]  # (32, 8, 8192)
classes = data["classes"]

for (l, n) in [(5, 2218), (6, 1409), (6, 6013)]:
    vals = [(classes[i], float(sel[i, l, n])) for i in range(len(classes))]
    vals_sorted = sorted(vals, key=lambda x: -x[1])
    top1_score  = vals_sorted[0][1]
    top2_score  = vals_sorted[1][1]
    exactly_zero = sum(1 for _, v in vals if v == 0.0)
    print(f"L{l}N{n:4d}:")
    print(f"  top1={vals_sorted[0][0]:20s}  sel={top1_score:.4f}")
    print(f"  top2={vals_sorted[1][0]:20s}  sel={top2_score:.6f}  (gap={top1_score - top2_score:.4f})")
    print(f"  exactly_zero={exactly_zero}/32,  >0.01={sum(1 for _,v in vals if v>0.01)}/32")
    print()
