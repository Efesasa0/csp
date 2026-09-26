# X04

## Summary

Builds a bipartite graph connecting AST nodes and builtins, where edge weights reflect how similar their circuit masks are (mean Jaccard similarity across shared layers). This reveals which syntactic structures and built-in types share underlying neural wiring — for example, whether `For` loops are wired similarly to `list` or `tuple`. Key outputs: a full similarity heatmap, a sparse edge-list CSV (top-K edges per AST node), and a bipartite graph visualization with nodes colored by AST family and builtin domain.

The test should be conducted to answer the following question:

1. 43 ASTs and 63 Builtins, I want to see the relationship between the two.

To test this, do something like this:

Create a bi-partite graph where edge weight is the similarity of the concepts

---

## Implementation

**Script**: `experiments/scripts/x04_ast_builtin_bipartite.py`

**How to run**

```bash
# Default
python experiments/scripts/x04_ast_builtin_bipartite.py

# Smoke test, limit to top-5 edges per AST node
python experiments/scripts/x04_ast_builtin_bipartite.py \
    --atlas data/test_5x5x10_validated_prompts.h5 \
    --top-k 5
```

Via the runner:
```bash
bash scripts/run_experiments.sh
bash scripts/run_experiments.sh --atlas=data/test_5x5x10_validated_prompts.h5
```

**What it computes**

1. Loads the atlas HDF5 and reads `universal_masks` for both `ast` and `builtin` concept types.
2. Computes a full (n_ast × n_builtin) similarity matrix. For each (AST node, builtin) pair, finds the layers present in both masks, computes pairwise Jaccard similarity at each shared layer, and takes the mean over layers as the edge weight.
3. Saves a heatmap of the full similarity matrix.
4. Builds a sparse edge list by keeping only the top-K highest-weight builtins for each AST node (default K=5, configurable via `--top-k`). Edges with weight 0 are discarded.
5. Annotates each edge with the AST node's family (from `analysis.concept_taxonomy.ast_family`) and the builtin's domain (from `builtin_domain`).
6. Writes the edge list as CSV, then renders a bipartite graph using NetworkX (if installed): AST nodes on the left (circles, coloured by family), builtin nodes on the right (squares, coloured by domain), edge width proportional to Jaccard weight.

**Outputs** (`experiments/outputs/x04/`)

| File | Description |
|------|-------------|
| `ast_builtin_similarity_matrix.png` | Full n_ast × n_builtin mean-Jaccard heatmap |
| `edge_list.csv` | Sparse edge list: `ast_node`, `builtin_obj`, `weight`, `ast_family`, `builtin_domain` |
| `bipartite_graph.png` | NetworkX bipartite layout (requires `networkx`; skipped if not installed) |

**Dependencies**

Offline only — requires the atlas HDF5 file, no GPU or model needed. `networkx` is optional; the graph plot is skipped if it is not installed.
