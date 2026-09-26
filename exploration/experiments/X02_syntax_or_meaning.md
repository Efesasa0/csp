# X02

## Summary

Tests whether each circuit responds primarily to syntactic structure (AST form) or semantic content (domain/meaning) by measuring Jensen-Shannon divergence of circuit activations across 5 code domains. Prompts sharing the same AST-builtin pair but drawn from different domains are run through the model, and per-circuit activation distributions are compared. Low JSD means the circuit fires consistently regardless of domain (syntax-dominant); high JSD means it is sensitive to what the code is about (meaning-sensitive). Circuits are labeled syntax-dominant or meaning-sensitive relative to the median JSD across all circuits. Key outputs: a ranked CSV of circuits by JSD and a domain-JSD heatmap.

The test should be conducted to answer the following question:

1. For each circuit, what do they respond to? syntax or meaning?

To test this, do something like this:

for x in items:
    vs
for element in dataset:

essentially same AST but different tokens.

I think of this as KL divergence or soemthing like that. Ideally we would want
similarity such that the model is capable of storing concepts.

---

## Implementation

This experiment runs in two sequential phases.

---

### Phase A — Per-prompt activation extraction

**Script**: `experiments/scripts/x02_extract_prompt_level.py`

**How to run**

```bash
# Default
python experiments/scripts/x02_extract_prompt_level.py

# Limit to two pairs for a quick test
python experiments/scripts/x02_extract_prompt_level.py \
    --atlas data/small_dynamic_feature_atlas.h5 \
    --prompts data/small_40x50x50_validated_prompts.parquet \
    --out experiments/outputs/x02/prompt_level_activations.h5 \
    --pairs For__list If__int \
    --batch-size 8
```

Via the runner:
```bash
bash scripts/run_experiments.sh --with-x02
bash scripts/run_experiments.sh --with-x02 --atlas=data/test_5x5x10_validated_prompts.h5
```

**What it computes**

1. Loads the atlas HDF5 (`pair_masks`) and a prompts parquet containing columns `ast_node`, `builtin_obj`, `prompt_text`, `domain`, and optionally `variation_id`.
2. Downloads and patches the `openai/circuit-sparsity` custom GPT architecture from HuggingFace (caches locally under `~/.cache/huggingface/modules`), then builds an `ActivationExtractor` around it.
3. For each (AST node, builtin) pair in the atlas, filters the prompts parquet to that pair. Runs the model in batches over all matching prompts, extracting MLP activations at the last token position for every layer.
4. For each layer, reads the circuit boolean mask from the atlas for that pair, converts it to a list of active neuron indices, and projects the full-layer activation vector down to just those neurons (stores as float16 to save space).
5. Writes the projected activations, per-prompt domain IDs, and variation IDs into a structured HDF5 file grouped by pair and layer.

**Outputs** (`experiments/outputs/x02/`)

| File | Description |
|------|-------------|
| `prompt_level_activations.h5` | Intermediate HDF5; contains `pairs/{pair}/layer_{N}/acts_proj_f16` (N_prompts × K_circuit_neurons, float16), `mask_indices`, and `prompt_meta/{variation_id,domain_id}` |

**Dependencies**

Requires GPU + model (`openai/circuit-sparsity` via HuggingFace). Python packages: `torch`, `transformers`, `huggingface_hub`, `pandas`, `h5py`.

---

### Phase B — Syntax vs. Meaning analysis

**Script**: `experiments/scripts/x02_syntax_vs_meaning.py`

**How to run**

```bash
# Default (reads output from Phase A)
python experiments/scripts/x02_syntax_vs_meaning.py

# Explicit paths
python experiments/scripts/x02_syntax_vs_meaning.py \
    --acts experiments/outputs/x02/prompt_level_activations.h5 \
    --out  experiments/outputs/x02
```

Phase A must complete before Phase B is run.

**What it computes**

1. Reads the intermediate HDF5 written by Phase A.
2. For each (pair, layer), groups the projected activation vectors by domain ID and computes the per-domain mean activation vector.
3. Computes Jensen-Shannon divergence (JSD) across the domain mean vectors. Each vector is treated as an unnormalized distribution over circuit neurons (softmax-normalized by absolute value before computing JSD). High JSD indicates the circuit responds differently across domains (meaning-sensitive); low JSD indicates consistent activation regardless of domain (syntax-dominant).
4. Also computes the mean pairwise cosine similarity across domain mean vectors as a complementary syntax-stability metric.
5. Labels each (pair, layer) record as `syntax-dominant` or `meaning-sensitive` by comparing its JSD to the median JSD across all records.
6. Produces a ranked CSV and a heatmap of JSD values indexed by circuit pair and layer.

**Outputs** (`experiments/outputs/x02/`)

| File | Description |
|------|-------------|
| `ranked_circuits.csv` | One row per (pair, layer): JSD, mean cross-domain cosine, n_domains_present, n_neurons_in_circuit, interpretation label |
| `domain_jsd_heatmap.png` | Heatmap of JSD values (rows = circuit pairs, columns = layers; green = syntax-stable, red = meaning-sensitive) |

**Dependencies**

Offline only (reads the HDF5 produced by Phase A). No GPU required. Python packages: `h5py`, `numpy`, `pandas`, `matplotlib`.
