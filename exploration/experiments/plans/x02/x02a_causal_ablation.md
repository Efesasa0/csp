# X02-A: Causal Ablation Validation

## Title
X02-A: Causal Ablation Validation

## Motivation
X02 currently classifies circuits as syntax-dominant or meaning-sensitive using JSD, which is correlational. This plan validates causality by ablating candidate circuits and directly measuring how model outputs shift across domains.

## Method
- Select the top 10 syntax-dominant circuits and top 10 meaning-sensitive circuits from `ranked_circuits.csv`.
- For each selected circuit, zero-ablate its circuit neurons at the corresponding layer during forward pass.
- Measure output distribution shift per domain using KL divergence on next-token logits (ablated vs non-ablated).
- Evaluate pattern consistency:
- Syntax-dominant circuits should produce similar KL shifts across domains (low domain variance).
- Meaning-sensitive circuits should produce domain-dependent KL shifts (high domain variance).

## Implementation Sketch
- Create script: `experiments/scripts/x02a_causal_ablation.py`.
- Use GPU execution for efficient forward passes with ablation hooks.
- Inputs:
- `experiments/outputs/x02/ranked_circuits.csv`
- Atlas masks (circuit neuron masks used in prior X02 phases)
- Prompt dataset parquet used for domain prompts
- Compute:
- Per-circuit, per-domain KL divergence shift in next-token logits under zero-ablation
- Optional aggregate statistics per circuit (mean and variance across domains)
- Output:
- `experiments/outputs/x02a/ablation_results.csv`
- Required columns: `pair, layer, interpretation, domain, kl_divergence_shift`
- Add Streamlit page:
- `pages/2a_X02A_Causal_Ablation.py`
- Include tables and variance-focused comparison plots by interpretation class.

## Expected Output
If the X02 interpretation is causally valid, meaning-sensitive circuits should exhibit high cross-domain variance in KL shift, while syntax-dominant circuits should exhibit low variance and more uniform shifts.

## Dependencies
- X02 Phase A+B completed
- `ranked_circuits.csv` available with interpretation labels
- Atlas masks and prompt parquet available from prior pipeline
- GPU runtime for ablation forward passes
