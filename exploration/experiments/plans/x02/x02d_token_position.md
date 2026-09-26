# X02-D: Token-Position Sensitivity Analysis

## Title
X02-D: Token-Position Sensitivity Analysis

## Motivation
Current X02 extraction uses only the last token, which may miss when domain information first appears. Measuring multiple token positions can reveal the temporal point where representations shift from syntax-driven to domain-sensitive.

## Method
- Re-run activation extraction at token positions `[0, N//4, N//2, 3N//4, N-1]` for each prompt.
- For each `(pair, layer, position)`, compute JSD and cross-domain cosine.
- Plot JSD across token position.
- Expected pattern: low early-position JSD, then increase after domain-specific cues appear.
- Correlate JSD spikes with token identity, especially the first domain keyword position.

## Implementation Sketch
- Create/modify extraction script: `experiments/scripts/x02d_extract_positional.py`.
- Reuse X02 Phase A extraction architecture with configurable multi-position `token_pos` handling.
- GPU required due to increased extraction volume.
- Output extraction artifact:
- `experiments/outputs/x02d/positional_activations.h5`
- Create analysis script: `experiments/scripts/x02d_positional_analysis.py` (CPU).
- Compute:
- Per `(pair, layer, position)` JSD and cross-domain cosine
- Position-wise trends and optional keyword-aligned spike statistics
- Outputs:
- `experiments/outputs/x02d/jsd_by_position.csv`
- `experiments/outputs/x02d/jsd_position_curves.png`

## Expected Output
JSD should be near zero at position 0 and increase as more domain-specific context is consumed. The slope and onset layer/position characterize how quickly each circuit absorbs domain information.

## Dependencies
- X02 Phase A extraction pipeline available for reuse
- Access to prompts/tokenization metadata for token identity alignment
- GPU for multi-position extraction
