# X02-B: Layer-wise Syntax->Meaning Phase Transition

## Title
X02-B: Layer-wise Syntax->Meaning Phase Transition

## Motivation
X02 reports JSD independently per `(pair, layer)`, but many circuits may transition gradually from syntax stability in earlier layers to semantic sensitivity in later layers. Quantifying this transition identifies each pair's meaning emergence layer.

## Method
- For each pair, collect JSD across all 8 layers into a trajectory vector.
- Fit a sigmoid model to each trajectory:
- `JSD(layer) = a / (1 + exp(-k * (layer - L_transition)))`
- Interpret `L_transition` (sigmoid inflection point) as the meaning emergence layer.
- Cluster pairs by `L_transition`.
- Compute aggregate statistics: mean transition layer, variance, and bimodality test over transition layers.

## Implementation Sketch
- Create script: `experiments/scripts/x02b_phase_transition.py`.
- CPU-only analysis (no new model inference required).
- Input:
- `experiments/outputs/x02/ranked_circuits.csv`
- Compute:
- Per-pair sigmoid fit parameters (`L_transition`, `k_steepness`, goodness-of-fit)
- Clustering and summary statistics over transition layers
- Outputs:
- `experiments/outputs/x02b/transition_curves.csv` with columns `pair, L_transition, k_steepness, r_squared`
- `experiments/outputs/x02b/transition_histogram.png`
- Streamlit integration:
- Add to existing X02 page or create `pages/2b_X02B_Phase_Transition.py`.

## Expected Output
Most pairs should transition in layers 3-6. Pairs involving more complex builtins (for example `hasattr`, `zip`) may transition earlier than simple builtins (for example `int`, `list`).

## Dependencies
- X02 Phase B completed
- `ranked_circuits.csv` present with per-layer JSD data or fields sufficient to reconstruct trajectories
