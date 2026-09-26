# X03

## Summary

Measures circuit density per layer for every AST and builtin concept to discover which layers host which programming concepts. Uses localization entropy to quantify how concentrated or spread out each concept's circuit is across layers — low entropy means the concept is localized to a few layers, high entropy means it is distributed. Key outputs: density heatmaps (concepts x layers), stacked bar charts of per-layer concept density, and a CSV of localization entropy and peak layer for each concept.

The test should be conducted to answer the following question:

1. Test wheter some programming concepts specifically localize in certain
   layers.

To test this, do something like this:

In which layers are the circuits mostly dense? This could show by layer, how
concepts evolve.

I imagine plotting a bar chart of layer on x and relevance on y with concepts as
the bars on top of each other, sorted per layer fashion.

---

## Implementation

**Script**: `experiments/scripts/x03_layer_localization.py`

**How to run**

```bash
# Default
python experiments/scripts/x03_layer_localization.py

# Smoke test
python experiments/scripts/x03_layer_localization.py \
    --atlas data/test_5x5x10_validated_prompts.h5
```

Via the runner:
```bash
bash scripts/run_experiments.sh
bash scripts/run_experiments.sh --atlas=data/test_5x5x10_validated_prompts.h5
```

**What it computes**

1. Loads the atlas HDF5 and reads the `universal_masks` for both `ast` and `builtin` concept types.
2. For each concept type, calls `circuit_density_by_layer`, which computes the fraction of the `n_neurons` (default 2048) that are active in the circuit mask at each layer. The result is a dict mapping concept name to a density array of length `n_layers`.
3. Computes a localization entropy for each concept: `H = -sum(p_l * log(p_l))` where `p_l` is the density at layer `l` normalized to sum to 1. Low entropy means the concept's active neurons concentrate in few layers (localized); high entropy means spread across layers.
4. Records the peak layer (argmax of density array) for each concept.
5. Produces two plots per concept type: a heatmap of density (concepts × layers) and a stacked bar chart (layers on x-axis, stacked concept densities on y-axis).
6. Writes a CSV with localization entropy, peak layer, total density, and per-layer density columns for every concept.

**Outputs** (`experiments/outputs/x03/`)

| File | Description |
|------|-------------|
| `concept_density_heatmap_ast.png` | Heatmap: AST concepts × layers, coloured by circuit density |
| `concept_density_heatmap_builtin.png` | Same for builtin concepts |
| `stacked_bar_layer_density_ast.png` | Stacked bar chart of AST concept densities per layer |
| `stacked_bar_layer_density_builtin.png` | Same for builtin concepts |
| `localization_scores.csv` | One row per concept: `kind`, `concept`, `localization_entropy`, `peak_layer`, `total_density`, `density_L{N}` columns |

**Dependencies**

Offline only — requires the atlas HDF5 file, no GPU or model needed.
