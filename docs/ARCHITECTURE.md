# DataPrism Architecture

## Overview

DataPrism is a Python library for exploratory data analysis. It provides automated feature profiling, target analysis, stability testing, and data quality scoring.

## Module Structure

```
dataprism/
├── __init__.py          # Public API surface
├── eda.py               # EDARunner — main orchestrator
├── schema.py            # DatasetSchema, ColumnConfig, Sentinels, enums
├── processor.py         # FeatureProcessor — routes columns to analyzers
├── exceptions.py        # Exception hierarchy (10 domain-specific exceptions)
├── analyzers/
│   ├── base.py          # BaseAnalyzer — type detection, shared analysis logic
│   ├── basic.py         # Dataset-level stats (row counts, memory, duplicates)
│   ├── continuous.py    # Continuous feature analysis (stats, outliers, bins)
│   ├── categorical.py   # Categorical feature analysis (value counts, entropy)
│   ├── correlation.py   # CorrelationEngine — Pearson, Theil's U, Eta, association matrix
│   ├── stability.py     # PSI-based cohort and time-based stability
│   ├── match_rates.py   # Provider-level match rate analysis
│   ├── target_analysis.py  # IV, WoE, target correlation
│   └── types.py         # Analyzer type definitions (DatasetInfo, ContinuousStats, etc.)
├── data/
│   ├── loader.py        # DataLoader — CSV/Parquet loading, schema parsing
│   └── sentinels.py     # Sentinel value replacement (not_found/missing → nullable types)
├── output/
│   ├── formatter.py     # JSONFormatter — output composition, summary metrics, data quality
│   └── mapper.py        # ResultMapper — transforms analyzer output to final schema
├── viewer/
│   ├── server.py        # HTTP server for interactive dashboard
│   └── template.html    # Single-page viewer (Summary, Catalog, Deep Dive, Associations)
└── utils/
    ├── logger.py        # Logging configuration
    └── performance.py   # CacheManager, ChunkedCSVReader
```

## Public API

All public imports live in `__init__.py`:

```python
from dataprism import (
    EDARunner,          # Main orchestrator
    DataLoader,         # CSV/Parquet/schema loading
    DatasetSchema,      # Schema container with filtered queries
    ColumnConfig,       # Per-column configuration
    ColumnType,         # continuous | categorical | ordinal | binary
    ColumnRole,         # feature | target | identifier | split | observation_date
    Sentinels,          # Provider sentinel codes
    DataPrismError,      # Base exception (catch-all)
)
```

## Data Flow

```
DataLoader ──→ EDARunner.run() ──→ JSON output
```

1. **Load** — `DataLoader` reads CSV/Parquet + schema JSON into DataFrame + `DatasetSchema`
2. **Configure** — `EDARunner` resolves column types from schema, filters features, identifies target/split/time columns by role
3. **Preprocess** — Sentinel values (`not_found`, `missing`) replaced with nulls via `replace_sentinel_values_with_nulls()`. Integer columns are converted to pandas nullable types (e.g. `Int64`) to avoid silent float upcasting when NaN is introduced.
4. **Basic analysis** — `BasicStatsAnalyzer` computes dataset-level stats (row counts, memory, duplicates, missing cells). Runs on feature columns only (excludes metadata like cohort/time columns).
5. **Correlations** — `CorrelationEngine` computes three association types:
   - **Pearson** correlation matrix (continuous × continuous)
   - **Theil's U** matrix (categorical × categorical, asymmetric — U(X→Y) may differ from U(Y→X))
   - **Eta / Correlation Ratio** (categorical ↔ continuous)
6. **Feature analysis** — `FeatureProcessor` routes each column to `ContinuousAnalyzer` or `CategoricalAnalyzer`, adds correlations (including reverse Theil's U), target relationships (IV/WoE), and quality flags. Per-feature analysis is wrapped in try/except for resilience. `ResultMapper` transforms each feature to the output schema.
7. **Stability** — Optional PSI computation: cohort-based (train/test split) or time-based (monthly/weekly/quarterly windows). Wrapped in try/except so failures don't abort the run.
8. **Match rates** — `compute_provider_match_rates()` calculates provider-level match statistics
9. **Association matrix** — `CorrelationEngine.build_association_matrix()` merges Pearson, Theil's U, and Eta into a single N×N structure with per-cell method labels. Only computed when ≤25 features.
10. **Format** — `JSONFormatter` assembles results into 3-section JSON with summary metrics, data quality scores, feature counts (16+ categories), and feature rankings

## Schema Design

Schema is defined in `schema.py` using immutable dataclasses:

- **`ColumnType`**: `continuous`, `categorical`, `ordinal`, `binary`
- **`ColumnRole`**: `feature`, `target`, `identifier`, `split`, `observation_date`
- **`Sentinels`**: Provider-specific codes (`not_found`, `missing`) with `to_dict()`/`from_dict()` serialization
- **`ColumnConfig`**: Per-column configuration (name, type, role, description, provider, sentinels)
- **`DatasetSchema`**: Collection of ColumnConfigs with O(1) lookup by name and filtered queries:
  - `get_features(column_type=)` — feature columns, optionally filtered by type
  - `get_feature_names(column_type=)` — shortcut returning names only
  - `get_by_role(role)` — all columns with a given role
  - `target` property — the single target column (raises if none/multiple)
  - `columns` property — dict of all configs keyed by name

Schema JSON format:
```json
{
  "columns": [
    {
      "name": "feature_name",
      "type": "continuous",
      "role": "feature",
      "description": "Human-readable description",
      "provider": "bureau",
      "sentinels": { "not_found": "-9999", "missing": "-1" }
    }
  ]
}
```

## Output Schema

`JSONFormatter.format_results()` produces a dict with three top-level sections:

```json
{
  "metadata": {
    "timestamp": "...",
    "execution_time_seconds": 1.23,
    "target_variable": "target",
    "schema_available": true
  },
  "summary": {
    "total_features": 50,
    "feature_types": { "continuous": 30, "categorical": 20 },
    "avg_missing_percentage": 5.2,
    "feature_counts": {
      "high_correlation": {}, "redundant_features": {}, "high_iv": {}, "high_stability": {},
      "high_missing": {}, "constant_features": {}, "low_variance": {}, "not_recommended": {},
      "highly_skewed": {}, "high_kurtosis": {}, "high_cardinality": {}, "suspected_leakage": {},
      "predictive_power": {}, "significant_correlations": {},
      "increasing_drift": {}, "volatile_stability": {}, "minor_shift": {}, "major_shift": {}
    },
    "highest_metrics": {},
    "top_features_by_statistical_score": [],
    "data_quality": { "overall_score": 8.5, "recommended_actions": [] },
    "dataset_info": {},
    "provider_match_rates": {},
    "association_matrix": {
      "features": ["target", "feat_a", "feat_b"],
      "values": [[null, 0.85, 0.32], [0.85, null, -0.12], [0.32, -0.12, null]],
      "methods": [["", "eta", "eta"], ["eta", "", "pearson"], ["eta", "pearson", ""]]
    },
    "stability_analysis": {}
  },
  "features": [
    {
      "feature_name": "...",
      "type": "Continuous",
      "statistics": {},
      "outliers": {},
      "distribution": {},
      "correlations": {},
      "target_relationship": {},
      "quality": {},
      "stability": {}
    }
  ]
}
```

All values are sanitized for JSON (`NaN`/`Infinity` → `null`, `np.ndarray` → list, `datetime` → ISO string) via `sanitize_for_json()`.

## Exception Hierarchy

All exceptions inherit from `DataPrismError` for catch-all handling:

```
DataPrismError
├── DataLoadError              # File not found, read failures
├── DataValidationError        # Missing columns, invalid types, insufficient data
├── AnalysisError              # Correlation/statistical computation failures
├── ConfigurationError         # Invalid parameters, conflicting options
├── FeatureTypeError           # Type detection issues, incompatible operations
├── MissingDataError           # All values missing, invalid imputation config
├── StabilityAnalysisError     # Insufficient cohorts, invalid time periods
├── OutputFormattingError      # JSON serialization failures
└── TargetAnalysisError        # Non-binary target for IV, insufficient samples
```

## EDARunner Configuration

`EDARunner.__init__()` accepts:

| Parameter | Default | Purpose |
|---|---|---|
| `max_categories` | 50 | Max unique values before treating as continuous |
| `sample_size` | None | Limit rows analyzed (head-based sampling) |
| `top_correlations` | 10 | Number of top correlated features per feature |
| `max_correlation_features` | None | Limit features in correlation matrix |
| `calculate_stability` | False | Enable cohort-based PSI stability |
| `cohort_column` | None | Column containing cohort labels |
| `baseline_cohort` | None | Label for baseline cohort (default: "training") |
| `comparison_cohort` | None | Label for comparison cohort (default: "test") |
| `time_based_stability` | False | Enable time-based PSI stability |
| `time_column` | None | Column containing timestamps |
| `time_window_strategy` | "monthly" | Window strategy: monthly, weekly, quarterly |
| `baseline_period` | "first" | Baseline time window |
| `comparison_periods` | "all" | Comparison time windows |
| `min_samples_per_period` | 100 | Minimum samples required per time period |

## Statistical Methods

### Entropy
Shannon entropy in **bits** (log base 2) via `scipy.stats.entropy(probabilities, base=2)`. Applies to both categorical value distributions and continuous histogram bins. Zero-count bins are excluded from the calculation.

### Weight of Evidence (WoE)
Follows the Siddiqi (2006) convention: `WoE = ln(bad% / good%)`. Positive WoE indicates a higher proportion of bad outcomes (higher risk). Laplace smoothing with `0.5 / num_categories` per bin prevents log(0).

### Information Value (IV)
`IV = Σ (bad% - good%) × WoE` over all bins/categories. Uses the same Siddiqi convention as WoE above.

### PSI (Population Stability Index)
Bin distributions are normalized to proportions. Zero bins are replaced with `ε = 1e-4` to avoid log(0). Continuous features use quantile-based binning with edges extended to `[-∞, +∞]` to capture out-of-range comparison values. Constant-value features are detected and handled separately.

### Theil's U (Uncertainty Coefficient)
Asymmetric measure of categorical association: `U(X→Y)` measures how much knowing X reduces uncertainty about Y. Constant columns (zero entropy) produce `U = 0.0` on the diagonal rather than undefined values. Both forward and reverse directions are stored in feature correlations.

### Correlation Ratio (Eta)
Measures association between a categorical and continuous variable. Infinite values in the continuous variable are filtered before computation.

### Column Type Inference
When a `ColumnConfig` exists but has `type=None`, inference falls through to automatic detection instead of returning `None`. The unique ratio denominator uses non-null count (not total count) to handle columns with missing values. All-null columns default to categorical.

## Interactive Viewer

The `dataprism.viewer` module provides a single-page dashboard served via a local HTTP server.

### Architecture
- `server.py` — Starts `HTTPServer` on localhost, injects JSON data into the HTML template
- `template.html` — Self-contained SPA using Tailwind CSS and Chart.js (loaded via CDN)

### Tabs
- **Summary** — KPI cards, insights table (highest metrics + feature counts), top features by IV, provider match rates, data quality score
- **Catalog** — Paginated, sortable, filterable feature table (excludes target)
- **Deep Dive** — Split-pane: feature list (left) + detail panel (right) with statistics, box plot, distribution chart, target associations, and correlations. Target variable has a dedicated selector above the feature list.
- **Associations** — Canvas-rendered N×N heatmap from the association matrix. Shape encodes method (circle = Pearson, square = Theil's U, triangle = Eta). Size encodes magnitude. Color encodes sign and strength. Interactive tooltips and click-to-navigate.
