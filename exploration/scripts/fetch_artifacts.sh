#!/usr/bin/env bash
set -euo pipefail

# Download all pipeline artifacts from Hugging Face
# Usage: bash scripts/fetch_artifacts.sh

REPO="AST-Revisited/contrastive-stubs"
BASE="https://huggingface.co/datasets/${REPO}/resolve/main"

# ── Helper ───────────────────────────────────────────────────────────────────
fetch() {
  local dest="$1"
  local remote="${2:-$1}"
  if [[ -f "$dest" ]]; then
    echo "SKIP (exists): $dest"
  else
    mkdir -p "$(dirname "$dest")"
    echo "Downloading $dest ..."
    curl -fL "${BASE}/${remote}" -o "$dest"
  fi
}

# ── Input data ───────────────────────────────────────────────────────────────
fetch data/contrastive_stubs.json
fetch data/contrastive_stubs_v2.json
fetch data/extraction_stats.json
fetch data/small_40x50x50_stats.json
fetch data/small_40x50x50_validated_prompts.parquet
fetch data/test_5x5x10_stats.json
fetch data/test_5x5x10_validated_prompts.parquet

# ── 01 Extraction ────────────────────────────────────────────────────────────
fetch 01_contrastive_stubs_edge_graph.npz
fetch 01_contrastive_stubs_head_attr.npy
fetch 01_contrastive_stubs_meta.json
fetch 01_contrastive_stubs_mlp_attr.npy
fetch 01_contrastive_stubs_mlp_neurons.npy
fetch 01_contrastive_stubs_residual_all.npy
fetch 01_contrastive_stubs_residual_final.npy
fetch contrastive_stubs.json

# ── 02 Variance Partition ────────────────────────────────────────────────────
fetch 02_contrastive_stubs_purity_masks.npz
fetch 02_contrastive_stubs_summary.json
fetch 02_contrastive_stubs_vp_edge_weights.npz
fetch 02_contrastive_stubs_vp_heads.npz
fetch 02_contrastive_stubs_vp_mlp_attr.npz
fetch 02_contrastive_stubs_vp_mlp_neurons.npz
fetch 02_contrastive_stubs_vp_residual.npz

# ── 03 RSA ───────────────────────────────────────────────────────────────────
fetch 03_contrastive_stubs_proj_ast.npy
fetch 03_contrastive_stubs_proj_builtin.npy
fetch 03_contrastive_stubs_rdm_model_ast.npy
fetch 03_contrastive_stubs_rdm_model_builtin.npy
fetch 03_contrastive_stubs_rdm_neural_ast.npy
fetch 03_contrastive_stubs_rdm_neural_builtin.npy
fetch 03_contrastive_stubs_rdm_neural_raw.npy
fetch 03_contrastive_stubs_similarity_matrices.npz

# ── 04 Additivity ───────────────────────────────────────────────────────────
fetch 04_contrastive_stubs_additivity_heatmap.png
fetch 04_contrastive_stubs_additivity_overview.png
fetch 04_contrastive_stubs_additivity_scores.npz
fetch 04_contrastive_stubs_additivity_scores_corrected.npz
fetch 04_contrastive_stubs_additivity_scores_masked.npz
fetch 04_contrastive_stubs_additivity_three_way.png
fetch 04_contrastive_stubs_baseline_cosine_matrices.png
fetch 04_contrastive_stubs_baselines_ast.npz
fetch 04_contrastive_stubs_baselines_builtin.npz
fetch 04_contrastive_stubs_diagnostic_distributions.png
fetch 04_contrastive_stubs_diagnostic_distributions_masked.png
fetch 04_contrastive_stubs_explicit_vs_proxy.json
fetch 04_contrastive_stubs_explicit_vs_proxy.png
fetch 04_contrastive_stubs_heatmap_three_way.png
fetch 04_contrastive_stubs_layer_traces.png
fetch 04_contrastive_stubs_pair_summary.json
fetch 04_contrastive_stubs_permutation_null.png
fetch 04_contrastive_stubs_significance.json
fetch 04_contrastive_stubs_significance_masked.json

# ── 05 Probing ──────────────────────────────────────────────────────────────
fetch 05_contrastive_stubs_anova_interaction.png
fetch 05_contrastive_stubs_anova_results.json
fetch 05_contrastive_stubs_concept_auroc_ast.npz
fetch 05_contrastive_stubs_concept_auroc_builtin.npz
fetch 05_contrastive_stubs_cross_probe.json
fetch 05_contrastive_stubs_cross_probe.png
fetch 05_contrastive_stubs_permeation_grid_ast.png
fetch 05_contrastive_stubs_permeation_grid_builtin.png
fetch 05_contrastive_stubs_probe_accuracy.npz
fetch 05_contrastive_stubs_probe_accuracy.png
fetch 05_contrastive_stubs_probe_foldscores.npz

# ── 06 Ablation ─────────────────────────────────────────────────────────────
fetch 06_contrastive_stubs_ablation_by_category.png
fetch 06_contrastive_stubs_ablation_effects.npz
fetch 06_contrastive_stubs_ablation_effects.png
fetch 06_contrastive_stubs_ablation_summary.json
fetch 06_contrastive_stubs_circuit_ast.json
fetch 06_contrastive_stubs_circuit_builtin.json
fetch 06_contrastive_stubs_circuit_interaction.json
fetch 06_contrastive_stubs_circuit_summary.png
fetch 06_contrastive_stubs_patch_ast_heatmap.png
fetch 06_contrastive_stubs_patch_ast_layers.png
fetch 06_contrastive_stubs_patch_builtin_layers.png
fetch 06_contrastive_stubs_patch_summary.json

# ── Experiment output CSVs (for Streamlit pages) ────────────────────────────
fetch experiments/outputs/x01/within_vs_cross_group_jaccard.csv
fetch experiments/outputs/x02/ranked_circuits.csv
fetch experiments/outputs/x02/ranked_circuits_by_domain_pair.csv
fetch experiments/outputs/x02a/ablation_results.csv
fetch experiments/outputs/x02b/transition_curves.csv
fetch experiments/outputs/x02c/probe_results.csv
fetch experiments/outputs/x02d/jsd_by_position.csv
fetch experiments/outputs/x02e/amplification_index.csv
fetch experiments/outputs/x03/localization_scores.csv
fetch experiments/outputs/x04/edge_list.csv
fetch experiments/outputs/x05/top_neurons_per_layer.csv
fetch experiments/outputs/x06/Assign__dict/neighborhood_distances.csv
fetch experiments/outputs/x06/Assign__dict/pair_stability.csv
fetch experiments/outputs/x06/Assign__dict/probe_accuracy.csv
fetch experiments/outputs/x06/Assign__dict/reconstruction_metrics.csv
fetch experiments/outputs/x06/Call__zip/neighborhood_distances.csv
fetch experiments/outputs/x06/Call__zip/pair_stability.csv
fetch experiments/outputs/x06/Call__zip/probe_accuracy.csv
fetch experiments/outputs/x06/Call__zip/reconstruction_metrics.csv
fetch experiments/outputs/x06/For__list/neighborhood_distances.csv
fetch experiments/outputs/x06/For__list/pair_stability.csv
fetch experiments/outputs/x06/For__list/probe_accuracy.csv
fetch experiments/outputs/x06/For__list/reconstruction_metrics.csv
fetch experiments/outputs/x06/If__int/neighborhood_distances.csv
fetch experiments/outputs/x06/If__int/pair_stability.csv
fetch experiments/outputs/x06/If__int/probe_accuracy.csv
fetch experiments/outputs/x06/If__int/reconstruction_metrics.csv
fetch experiments/outputs/x06/Return__len/neighborhood_distances.csv
fetch experiments/outputs/x06/Return__len/pair_stability.csv
fetch experiments/outputs/x06/Return__len/probe_accuracy.csv
fetch experiments/outputs/x06/Return__len/reconstruction_metrics.csv
fetch experiments/outputs/x07/additivity_results.csv
fetch experiments/outputs/x07/layer_additivity.csv
fetch experiments/outputs/x07/pair_additivity.csv
fetch experiments/outputs/x07/top_additive.csv
fetch experiments/outputs/x07/top_interactions.csv

echo ""
echo "All artifacts downloaded."
