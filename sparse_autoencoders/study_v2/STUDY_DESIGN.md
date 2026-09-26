# study_v2 — Factorial SAE specificity study

**Question.** For SAE features at layer L, site S (mlp/resid), does a
discovered "winner" feature actually encode the target AST construct, or
is it an artifact of co-occurring builtin tokens / surface strings?

## Datasets

| | rows | variants | role |
|---|---:|---|---|
| `constructor_stubs` (new) | 14,683 | `explicit` (12,963), `baseline_ast` (372), `baseline_builtin` (1,257), `proxy` (91) | primary — split 50/50 (stratified by `ast_node × variant_type`) into **discovery** and **held-out** |
| `contrastive_stubs` (old) | 8,897 | `explicit` only | **external replication** only (SAEs were trained on this, so own-domain baseline) |

## Pipeline

1. **Harvest** (`harvest_all.py`): one model forward pass per prompt hooks
   all 8 layers × {mlp, resid} simultaneously. Caches 32 activation
   tensors + two metadata parquets + `splits.json`. ~20 min.
2. **Analysis** (`run_study.py`): for each (layer, site, AST node):
   - Top-10 candidate features on discovery split by
     `mean(target_explicit) - mean(other_explicit)`.
   - For each candidate, on the held-out split:
     - Mean activation on: target-explicit, other-explicit,
       `baseline_ast(node)`, `baseline_builtin` (any), `proxy(node)`.
     - Mann-Whitney U (one-sided, greater): target vs. other-explicit;
       explicit vs. `baseline_builtin`.
     - Cohen's d, 95% CI via 1,000-bootstrap.
     - Cross-dataset replication on old: same target-vs-other, pass if
       `p < 0.05` and direction matches.
     - Classification:
       - AST-specific: fires on `baseline_ast`, silent on `baseline_builtin`
       - Builtin-specific: inverse
       - Both-respond / Joint-required / Ambiguous
       - Surface-token vs. Concept from proxy comparison
   - Null distribution: 500 random SAE features → specificity ratio
     distribution. Winner's percentile vs. null reported.
3. **Retrain** (`retrain_new.py`): train SAEs on new-dataset activations
   at all 8 layers × 2 sites, same architecture as existing SAEs. Output
   to `data/study_v2/saes/`. Existing `data/saes/` untouched.
4. **Re-analysis** with new SAEs: `run_study.py --sae-dir data/study_v2/saes --tag ideal`.
5. **Comparison** (`compare.py`): per-(layer, site), count validated
   circuits (AST-specific + replicates on old + p<0.05) and mean |d|.
   Reports delta old→ideal.

## Robustness defenses

- **Winner bias**: discovery/held-out split.
- **Confound**: baseline_ast / baseline_builtin isolate factors.
- **Surface-vs-concept**: proxy rewrites.
- **External validity**: old-dataset replication required for "validated".
- **Chance**: null distribution from random features + Bonferroni-style
  thinking (29 nodes × top-10 candidates — raw p-values reported).
- **Layer-invariance**: full 8-layer × 2-site sweep, not cherry-picked L7.
- **SAE quality confound**: retrain on new data → re-run → compare.

## Output locations (no overwrites outside `study_v2/`)

```
data/study_v2/
  cache/
    acts_{new,old}_L{0..7}_{mlp,resid}.pt
    metadata_{new,old}.parquet
    splits.json
  reports/
    per_layer/L{0..7}_{mlp,resid}_{old_saes,ideal}.json
    summary_{old_saes,ideal}.json
    comparison.json
  saes/
    sae_layer_{0..7}.pt         (new-data-trained MLP)
    sae_layer_{0..7}_resid.pt   (new-data-trained resid)
    train_summary.json
  log/*.log
```

## Known limits (so we don't overclaim)

- Proxy rows only exist for ~6 AST nodes; proxy test is N/A elsewhere.
- Stratification couldn't balance all 32 × 4 cells (some have very few
  rows); statistical power varies by node.
- Cross-dataset replication only uses `explicit` rows (old has no
  variants) — confound tests remain new-dataset-only.
- Last-token activation used throughout (consistent with existing
  pipeline); multi-token aggregation is future work.
