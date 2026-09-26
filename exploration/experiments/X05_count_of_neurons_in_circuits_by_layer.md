# X05

## Summary

Counts how many circuits each neuron participates in at every layer, producing a membership heatmap across the full neuron space. This identifies "hub neurons" — neurons that are active in many circuits and may serve as shared computational infrastructure. It also shows whether circuit membership is sparse (most neurons belong to few circuits) or dense (many neurons are reused). Key outputs: a full neuron-membership heatmap, per-layer histograms of membership counts, and a CSV of the top-K hub neurons per layer with their prevalence scores.

The test should be conducted to answer the following question:

1. I want to see for each neuron on each layer, how many circuits they belon too
   in a heat map or something else.

---

## Implementation

**Script**: `experiments/scripts/x05_neuron_circuit_membership.py`

**How to run**

```bash
# Default
python experiments/scripts/x05_neuron_circuit_membership.py

# Smoke test
python experiments/scripts/x05_neuron_circuit_membership.py \
    --atlas data/test_5x5x10_validated_prompts.h5
```

Via the runner:
```bash
bash scripts/run_experiments.sh
bash scripts/run_experiments.sh --atlas=data/test_5x5x10_validated_prompts.h5
```

**What it computes**

1. Loads the atlas HDF5 and reads `pair_masks`, which maps each (AST node, builtin) pair to per-layer boolean neuron masks.
2. Calls `neuron_membership_counts` from `analysis.mask_ops`, which accumulates a count matrix of shape (n_layers, n_neurons). Each cell records how many pair circuits include that neuron in that layer.
3. Normalizes the count matrix by the total number of pairs to get a prevalence fraction.
4. Produces a full (n_layers × n_neurons) heatmap using a `hot_r` colormap, showing circuit membership density across the entire neuron space.
5. Produces a grid of per-layer histograms showing the distribution of membership counts (i.e., how many neurons belong to 0 circuits, 1 circuit, 2 circuits, etc.) for each layer.
6. Identifies hub neurons: neurons whose membership count exceeds 50% of the total number of pairs at a given layer.
7. Writes a CSV of the top-K neurons per layer (default K=20) ranked by membership count, including their prevalence fraction.

**Outputs** (`experiments/outputs/x05/`)

| File | Description |
|------|-------------|
| `neuron_membership_heatmap.png` | Full (n_layers × n_neurons) heatmap of circuit membership counts |
| `per_layer_histogram.png` | Grid of per-layer histograms: x = number of circuits, y = number of neurons |
| `top_neurons_per_layer.csv` | Top-K neurons per layer: `layer`, `rank`, `neuron_id`, `n_circuits`, `prevalence` |

**Dependencies**

Offline only — requires the atlas HDF5 file, no GPU or model needed.
