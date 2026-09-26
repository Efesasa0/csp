# X02-C: Domain Discriminability Probes

## Title
X02-C: Domain Discriminability Probes

## Motivation
Cosine similarity measures geometric closeness but does not directly test whether domain identity is linearly decodable from activations. A linear probe quantifies accessible domain information in both circuit-projected and residual representations.

## Method
- For each `(pair, layer)`, train a 5-way logistic regression probe to predict domain from:
- Circuit-projected activations (`K` circuit neurons)
- Full residual stream (1024 dimensions)
- Use 5-fold cross-validation and report mean accuracy.
- Compare circuit vs residual probe performance:
- `circuit_probe_acc << resid_probe_acc`: circuit filters domain information (syntax-dominant behavior)
- `circuit_probe_acc >> resid_probe_acc`: circuit amplifies domain signal

## Implementation Sketch
- Create script: `experiments/scripts/x02c_domain_probes.py`.
- CPU-only pipeline.
- Use `sklearn.linear_model.LogisticRegression` and cross-validation utilities.
- Input:
- X02 Phase A HDF5 activations (including `resid_f16`)
- Compute:
- Per `(pair, layer)` cross-validated probe accuracy for circuit and residual representations
- `delta_acc = circuit_probe_acc - resid_probe_acc`
- Outputs:
- `experiments/outputs/x02c/probe_results.csv` with columns `pair, layer, circuit_probe_acc, resid_probe_acc, delta_acc`
- Scatter plot of `circuit_probe_acc` vs `resid_probe_acc`, colored by X02 interpretation class.

## Expected Output
Syntax-dominant circuits should underperform the residual baseline on domain decoding, while meaning-sensitive circuits should match or exceed residual decoding accuracy.

## Dependencies
- X02 Phase A completed
- `prompt_level_activations.h5` available with `resid_f16`
- `scikit-learn` installed in the analysis environment
