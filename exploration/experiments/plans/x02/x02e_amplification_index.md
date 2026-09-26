# X02-E: Circuit Amplification Index

## Title
X02-E: Circuit Amplification Index

## Motivation
Given both circuit-projected and residual cross-domain cosine similarity, a single ratio can summarize whether a circuit suppresses or amplifies domain differences relative to baseline model behavior. This can be easier to interpret than JSD alone.

## Method
- Define amplification index:
- `AI = mean_cross_domain_cosine / resid_cross_domain_cosine`
- Interpret AI:
- `AI > 1`: circuit is more domain-invariant than residual baseline (suppresses domain info; syntax-dominant)
- `AI < 1`: circuit is less domain-invariant than residual baseline (amplifies domain differences; meaning-sensitive)
- `AI ≈ 1`: circuit mirrors baseline behavior
- Rank circuits by AI and compare against JSD-based interpretation labels.

## Implementation Sketch
- Option 1: extend `experiments/scripts/x02_syntax_vs_meaning.py` to compute and persist AI in `ranked_circuits.csv`.
- Option 2: add lightweight post-processing script `experiments/scripts/x02e_amplification_index.py`.
- Inputs:
- `ranked_circuits.csv` with `mean_cross_domain_cosine` and `resid_cross_domain_cosine`
- Compute:
- `AI` per circuit
- Rank order and correlation with JSD (expected negative relationship)
- Outputs:
- Updated `ranked_circuits.csv` or `experiments/outputs/x02e/amplification_index.csv`
- Streamlit scatter: `AI` (x-axis) vs `JSD` (y-axis), colored by interpretation.

## Expected Output
AI should correlate negatively with JSD. High-AI circuits should align with strong syntax-dominant candidates, and low-AI circuits should align with strong meaning-sensitive candidates.

## Dependencies
- X02 Phase B completed
- `ranked_circuits.csv` includes `resid_cross_domain_cosine`
- Existing plotting/Streamlit infrastructure for comparative scatter visualization
