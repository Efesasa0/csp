# Constructor-Stubs: Closing-Delimiter Control Experiment

This submission contains the full pipeline for the **constructor-stubs** control experiment,
which tests whether observed AST-node groupings in the original contrastive-stubs dataset
are genuine semantic signals or artefacts of closing-delimiter token identity.

In the original dataset, `DictComp`/`SetComp` prompts end with `}` and
`ListComp`/`Subscript` prompts end with `]` — the last-token extraction captures
these delimiters.  The constructor-stubs dataset rewrites all four nodes so every
prompt ends with `)`:
- `ListComp`  → `list(B for I in D)`
- `DictComp`  → `dict((k, B) for k, I in enumerate(D))`
- `SetComp`   → `set(B for I in D)`
- `Subscript` → `B.__getitem__(0)`

---

## Directory structure

```
submission/
├── README.md
├── main.py                          # run full pipeline in one command
├── constructor_stubs.jsonl          # prompt dataset (included — skip Step 0)
├── scripts/
│   ├── 00_generate_constructor_stubs.py   # Step 0 – generate prompts
│   ├── 01_extraction.py                   # Step 1 – extract activations
│   ├── 02_variance_partition.py           # Step 2 – variance partitioning (VP)
│   ├── 02g_class_neurons.py               # Step 3 – MLP neuron selectivity
│   ├── check_neurons.py                   # Step 3b – verify monosemantic neurons
│   ├── 02l_head_selectivity.py            # Step 4 – attention head selectivity
│   ├── 02m_head_viz.py                    # Step 5 – head grid figure
│   ├── 02p_group_selectivity.py           # Step 6 – residual stream grouping
│   ├── 02f_ast_graph.py                   # Step 7 – AST similarity graph (HTML)
│   └── 03n_circuit_bundling_interactive.py # Step 8 – edge-bundling (HTML)
├── data/                            # created at runtime by extraction step
└── figures/                         # final outputs (see below)
```

---

## Pipeline overview

| Step | Script | Inputs | Key outputs |
|------|--------|--------|-------------|
| 0 | `00_generate_constructor_stubs.py` | — | `constructor_stubs.jsonl` |
| 1 | `01_extraction.py` | `.jsonl` | `data/01_constructor_stubs_residual_all.npy`, `*_head_attr.npy`, `*_mlp_neurons.npy`, `*_meta.json` |
| 2 | `02_variance_partition.py` | `01_*` arrays | `data/02_constructor_stubs_vp_residual.npz`, `*_heads.npz` |
| 3 | `02g_class_neurons.py` | `01_*` + `02_*` | `data/02g_*_selectivity_neurons.npz`, `02g_*_class_neurons.json` |
| 3b | `check_neurons.py` | `02g_*` files | Prints monosemantic MLP neuron table (stdout) |
| 4 | `02l_head_selectivity.py` | `01_*` + `02_*` | `data/02l_*_head_sel.npz`, `02l_*_head_ownership.json` |
| 5 | `02m_head_viz.py` | `02l_*` files | `data/figures/heads/head_all_pure_grid_diagonal.pdf` |
| 6 | `02p_group_selectivity.py` | `02g_*` selectivity | `data/figures/group_selectivity/residual_group_heatmap_k14.pdf`, `*_summary_k14.json` |
| 7 | `02f_ast_graph.py` | `01_*` residual | `data/figures/ast_graph/ast_graph_all_layers.html` |
| 8 | `03n_circuit_bundling_interactive.py` | `02g_*` + `02l_*` | `data/figures/head_neuron_bundling/heads_edge_bundling_interactive.html` |

---

## Prerequisites

```bash
# Activate the environment
micromamba activate comp0087_cps

# Required packages: torch, transformer_lens, numpy, scipy,
#                    scikit-learn, matplotlib, plotly, umap-learn
```

The model used is `circuit_sparsity` (loaded inside `01_extraction.py`).
Make sure model weights are accessible from your environment.

---

## Running the full pipeline

```bash
# One-command run (from submission/ directory):
python main.py

# With custom batch size and GPU:
python main.py --batch 512 --device cuda
```

---

## Running steps individually

All analysis scripts (`02g` onwards) are run from the `scripts/` directory.
Data is read from and written to `../data` by default.

```bash
cd scripts/

# Step 0 – generate prompts (skip if using the included constructor_stubs.jsonl)
python 00_generate_constructor_stubs.py
# output: ../constructor_stubs.jsonl

# Step 1 – extract activations  (slow; use --batch 1024 on GPU)
python 01_extraction.py \
    --input ../constructor_stubs.jsonl \
    --out_dir ../data \
    --batch 1024

# Step 2 – variance partitioning
python 02_variance_partition.py \
    --stem constructor_stubs \
    --in_dir ../data \
    --out_dir ../data

# Step 3 – MLP neuron selectivity
python 02g_class_neurons.py \
    --stem constructor_stubs \
    --in_dir ../data \
    --out_dir ../data

# Step 3b – verify monosemantic MLP neurons (run from submission/ not scripts/)
cd ..
python scripts/check_neurons.py
cd scripts/

# Step 4 – attention head selectivity
python 02l_head_selectivity.py \
    --stem constructor_stubs \
    --in_dir ../data \
    --out_dir ../data

# Step 5 – head diagonal-grid figure  →  head_all_pure_grid_diagonal.pdf
python 02m_head_viz.py \
    --stem constructor_stubs \
    --in_dir ../data \
    --out_dir ../data

# Step 6 – residual stream grouping (k=14)  →  residual_group_heatmap_k14.pdf
python 02p_group_selectivity.py \
    --stem constructor_stubs \
    --in_dir ../data \
    --k_list 14

# Step 7 – AST similarity graph  →  ast_graph_all_layers.html
python 02f_ast_graph.py \
    --stem constructor_stubs \
    --in_dir ../data

# Step 8 – edge-bundling interactive figure  →  heads_edge_bundling_interactive.html
python 03n_circuit_bundling_interactive.py \
    --stem constructor_stubs \
    --in_dir ../data
```

---

## Output figures

| File | Description |
|------|-------------|
| `data/figures/heads/head_all_pure_grid_diagonal.pdf` | AST-pure attention head ownership grid |
| `data/figures/group_selectivity/residual_group_heatmap_k14.pdf` | Residual stream group selectivity heatmap (k=14) |
| `data/figures/ast_graph/ast_graph_all_layers.html` | Interactive AST similarity graph (all layers) |
| `data/figures/head_neuron_bundling/heads_edge_bundling_interactive.html` | Interactive attention head edge-bundling |
