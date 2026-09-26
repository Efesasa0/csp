# X06

## Summary

Deep-dives into the activation geometry for 5 representative AST-builtin pairs (For+list, If+int, Return+len, Assign+dict, Call+zip), testing whether internal representations are compositional (AST component + builtin component) or entangled (pair-specific). For each pair, controlled prompt groups are built — same-pair variants, same-AST/different-builtin, same-builtin/different-AST, unrelated controls, and close neighbors — and full activation vectors are extracted per layer. The analysis compares four reconstruction models (AST-only, builtin-only, additive, pair-specific) and measures the interaction residual to quantify compositionality. Key outputs: reconstruction R-squared by layer, interaction residual norms, probe accuracies, PCA dimensionality, and neighborhood distance plots.

---

### Research question

To what extent are code concepts represented as low-dimensional, causally meaningful activation subspaces? Is the activation geometry for an AST-builtin pair better explained by AST identity alone, builtin identity alone, additive AST+builtin structure, or a full pair-specific representation?

### Design overview

For a target pair (A, B), the experiment builds 5 prompt groups: (1) same pair with surface variants, (2) same AST with different builtins, (3) same builtin with different ASTs, (4) unrelated controls, and (5) close neighbor pairs. Activations are extracted at every layer, centroids are computed for each group, and four reconstruction models are compared. The difference between the additive reconstruction and the true pair centroid (the interaction residual) measures how compositional or entangled the pair is at each layer. Additional analyses include linear probes, PCA dimensionality, and neighborhood geometry.

---

## Implementation

X06 is split into two scripts mirroring the X02 pattern:

### Phase A: `experiments/scripts/x06_extract_subspace.py` (GPU)

Extracts **full** 2048-dim activation vectors (not mask-projected like X02) for a target pair and its contrast groups.

**CLI:**
```bash
python experiments/scripts/x06_extract_subspace.py \
    --prompts data/small_40x50x50_validated_prompts.parquet \
    --target-pair For__list \
    --n-contrast 5 --m-per-contrast 10 --n-control 5 \
    --neighbors For__tuple,While__list,ListComp__list \
    --out experiments/outputs/x06/For__list_activations.h5 \
    --batch-size 8 --device auto --seed 42
```

**Prompt groups** built by `build_prompt_groups()`:
- **target**: all prompts for (A, B) from the parquet
- **same_ast**: `n-contrast` builtins B' != B that pair with A, `m-per-contrast` prompts each
- **same_builtin**: `n-contrast` ASTs A' != A that pair with B, `m-per-contrast` prompts each
- **control**: `n-control` pairs (A', B') where both differ, `m-per-contrast` prompts each
- **neighbor**: explicit pairs from `--neighbors`, `m-per-contrast` prompts each

**HDF5 output schema:**
```
/ (attrs): model_id, n_layers, n_neurons, target_ast, target_builtin
/prompts/text:         string[N]
/prompts/ast_node:     string[N]
/prompts/builtin_obj:  string[N]
/prompts/group:        string[N]
/prompts/variation_id: int32[N]
/activations/layer_{l}: float16[N, 2048]   (full vectors, gzip compressed)
```

Model loading reuses `_patch_circuit_sparsity()` and `load_model()` from X02.

### Phase B: `experiments/scripts/x06_subspace_geometry.py` (offline)

Reads the HDF5 and runs the full geometry analysis per layer.

**CLI:**
```bash
python experiments/scripts/x06_subspace_geometry.py \
    --acts experiments/outputs/x06/For__list_activations.h5 \
    --out experiments/outputs/x06 \
    --probe-cv 5 --pca-components 50
```

**Analysis pipeline (per layer):**

| Step | Function | Output |
|------|----------|--------|
| 1-2 | `compute_centroids()` | global_mean, ast/builtin/pair centroids |
| 3-4 | `reconstruction_metrics()` | R² and cosine for 4 models per prompt |
| 5 | `interaction_residual()` | pair_centroid - additive; norm |
| 6 | `pair_stability()` | within-pair vs between-pair distances |
| 7 | `train_probes()` | LogisticRegression + StratifiedKFold for AST/builtin/pair |
| 8 | `pca_analysis()` | explained variance, effective rank, participation ratio |
| 9 | `neighborhood_distances()` | pairwise centroid distances |

**Reconstruction models:**
- **A (AST-only):** `h_hat = ast_centroid[a]`
- **B (builtin-only):** `h_hat = builtin_centroid[b]`
- **C (additive):** `h_hat = global_mean + (ast_centroid[a] - global_mean) + (builtin_centroid[b] - global_mean)`
- **D (pair-specific):** `h_hat = pair_centroid[(a,b)]`

R² = `1 - ||h - h_hat||² / ||h - global_mean||²`

**Outputs:**
- `reconstruction_metrics.csv` — full per-prompt, per-layer, per-model table
- `pair_stability.csv` — within/between distances per pair per layer
- `probe_accuracy.csv` — AST/builtin/pair probe accuracy per layer
- `neighborhood_distances.csv` — pairwise centroid distances per layer
- `model_comparison_by_layer.png` — R²/cosine for 4 models × 8 layers
- `interaction_residual_by_layer.png` — ||residual|| across layers
- `pair_stability_by_layer.png` — within vs between distances
- `probe_accuracy_by_layer.png` — AST/builtin/pair probe accuracy
- `pca_explained_variance.png` — cumulative variance + dimensionality measures
- `neighborhood_distances.png` — centroid distance heatmap
- `pca_2d_layer{L}.png` — 2D scatter at layers 0, 4, 7

### Running via `run_experiments.sh`

```bash
bash scripts/run_experiments.sh --with-x06
bash scripts/run_experiments.sh --with-x06 --batch-size=16
```

The runner loops through all 5 representative pairs automatically (For__list, If__int, Return__len, Assign__dict, Call__zip).

