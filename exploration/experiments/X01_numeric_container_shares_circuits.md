# X01

## Summary

Compares Jaccard similarity of neuron masks within numeric builtins (int, float, complex, bool) and container builtins (list, tuple, dict, set, frozenset) to test whether semantically related builtins share circuit wiring. If within-group similarity is higher than cross-group similarity, it suggests the model organizes related Python types into shared neural circuits. Key outputs: per-layer Jaccard heatmaps and a CSV of within-group vs cross-group mean Jaccard scores.

The test should be conducted to answer the following question:

1. Does numeric builtins share circuts within its conceptual domain?
2. Does container builtins share circuts within its conceptual domain?


---

## Implementation

**Script**: `experiments/scripts/x01_numeric_container_shares.py`

**How to run**

```bash
# Default (full atlas)
python experiments/scripts/x01_numeric_container_shares.py

# Smoke test (small test atlas)
python experiments/scripts/x01_numeric_container_shares.py \
    --atlas data/test_5x5x10_validated_prompts.h5 \
    --out experiments/outputs/x01
```

Via the runner:
```bash
bash scripts/run_experiments.sh                                  # offline only
bash scripts/run_experiments.sh --atlas=data/test_5x5x10_validated_prompts.h5  # smoke test
```

**What it computes**

1. Loads the atlas HDF5 and reads the `universal_masks/builtin` dictionary, which maps each builtin name to a per-layer boolean neuron mask.
2. For each requested group (default: `numeric` = {int, float, complex, bool}; `container` = {list, tuple, dict, set, frozenset}), filters to members that actually appear in the atlas.
3. For each layer of each group, computes the full pairwise Jaccard similarity matrix across all group members using `compute_jaccard_matrix`. Saves one heatmap PNG per (group, layer) combination.
4. Records the mean within-group Jaccard (upper-triangle average, excluding the diagonal) for every (group, layer).
5. For every pair of groups (e.g., numeric vs. container), computes pairwise cross-group Jaccard at each layer and records the mean.
6. Writes a single CSV summarising within-group and cross-group mean Jaccard per layer for all groups.

**Outputs** (`experiments/outputs/x01/`)

| File | Description |
|------|-------------|
| `heatmap_{group}_layer{N}.png` | Annotated Jaccard heatmap for the group at layer N |
| `within_vs_cross_group_jaccard.csv` | Mean Jaccard per (group, layer, scope) with pair counts |

**Dependencies**

Offline only — requires the atlas HDF5 file, no GPU or model needed.
