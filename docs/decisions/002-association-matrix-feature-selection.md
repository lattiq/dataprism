# ADR-002: Smart feature selection for association matrix

## Status
Accepted

## Context

The association matrix (N×N heatmap of Pearson, Theil's U, and Eta values) was gated by a hard-coded check: `if len(features) <= 25`. Datasets with more than 25 features — which is the common case in production (200+ features) — got no matrix at all. This was an all-or-nothing tradeoff between output size and usefulness.

Options considered:
1. **Raise the hard-coded limit** (e.g. to 100) — simple but still fails at scale and produces unwieldy matrices
2. **Random sampling** — fast but misses important features and is non-deterministic
3. **Target-correlation-only ranking** — ignores feature-feature redundancy, biased toward continuous features
4. **Composite scoring with redundancy pruning** — balances relevance, diversity, and output size

## Decision

Replaced the hard-coded limit with a configurable `max_association_features` parameter (default 50) and a composite feature selection strategy in `CorrelationEngine.select_features_for_association()`.

The selection pipeline:

1. **Always include target variable** if present
2. **Quality filter** — remove constant features and features flagged as not recommended for modeling
3. **Score each candidate**:
   - With target: prefer IV, fall back to |Pearson| > |Spearman| > Theil's U > Eta
   - Without target: average absolute association strength across all matrices
4. **Cardinality penalty** — reduce scores for high-cardinality categoricals (>50% unique: ×0.3, 10–50%: ×0.7) since they produce noisy associations and inflate matrix size
5. **Sort by score** descending
6. **Redundancy pruning** — greedy forward selection, skip any feature with |association| > 0.9 to an already-selected feature
7. **Backfill** — if pruning removed too many, fill remaining slots from the scored list

Setting `max_association_features=0` disables the matrix entirely.

## Consequences

- Datasets of any size now produce an association matrix (previously skipped above 25 features)
- The default of 50 means existing small datasets (<50 features) see no behavioral change
- Previously skipped datasets (26–50 features) now get a full matrix with no selection
- Large datasets (51+ features) get a curated matrix that prioritizes predictive, non-redundant, low-cardinality features
- Output includes `feature_selection` metadata when selection is applied, so consumers know it happened
- The 0.9 redundancy threshold and cardinality penalty ratios are hard-coded — if these need tuning, they can be promoted to parameters in a future iteration
