# DataPrism Usage Guide

Reference documentation for advanced features. For quick start and basic usage, see the [README](../README.md).

## DatasetSchema

`DatasetSchema` enables schema-aware profiling by defining column types, roles, providers, and sentinel values.

### Loading from JSON

```python
from dataprism import DataPrism, DataLoader

df = DataLoader.load_csv("data.csv")
schema = DataLoader.load_schema("schema.json")

prism = DataPrism()
results = prism.analyze(
    data=df,
    schema=schema,
    target_variable="target",
    output_path="eda_results.json"
)
```

**Schema JSON format** (`schema.json`):

```json
{
  "columns": [
    {
      "name": "age",
      "type": "continuous",
      "role": "feature",
      "provider": "demographics",
      "description": "User age",
      "sentinels": { "not_found": "-1", "missing": null }
    }
  ]
}
```

Supported types: `continuous`, `categorical`, `ordinal`, `binary`. Supported roles: `feature`, `target`, `identifier`, `split`, `observation_date`.

### Programmatic Schema

```python
from dataprism import (
    DataPrism, DataLoader,
    ColumnConfig, ColumnType, ColumnRole, Sentinels, DatasetSchema,
)

df = DataLoader.load_csv("data.csv")

schema = DatasetSchema([
    ColumnConfig('age', ColumnType.CONTINUOUS, ColumnRole.FEATURE,
                 provider='demographics', description='User age',
                 sentinels=Sentinels(not_found='-1')),
    ColumnConfig('zip_code', ColumnType.CATEGORICAL, ColumnRole.FEATURE,
                 provider='address', description='ZIP code',
                 sentinels=Sentinels(not_found='', missing='00000')),
    ColumnConfig('target', ColumnType.BINARY, ColumnRole.TARGET),
])

prism = DataPrism()
results = prism.analyze(data=df, schema=schema, target_variable="target")
```

## Stability Analysis

DataPrism supports PSI-based drift detection in two modes: cohort-based and time-based.

### Cohort-Based (Train/Test)

Compare distributions between two cohorts (e.g. training vs test split):

```python
from dataprism import DataPrism, DataLoader

df = DataLoader.load_parquet("data.parquet")
schema = DataLoader.load_schema("schema.json")

prism = DataPrism(
    calculate_stability=True,
    cohort_column='dataTag',
    baseline_cohort='training',
    comparison_cohort='test'
)

results = prism.analyze(data=df, schema=schema)
```

### Time-Based

Track distribution drift over time windows:

```python
from dataprism import DataPrism, DataLoader

df = DataLoader.load_parquet("data.parquet")
schema = DataLoader.load_schema("schema.json")

prism = DataPrism(
    time_based_stability=True,
    time_column='onboarding_time',
    time_window_strategy='monthly',  # or 'weekly', 'quartiles', 'custom'
    baseline_period='first',
    comparison_periods='all',
    min_samples_per_period=100
)

results = prism.analyze(data=df, schema=schema)
```

## Advanced Configuration

### Correlation Settings

```python
prism = DataPrism(
    top_correlations=10,           # Top N correlations per feature
    max_correlation_features=500   # Limit features in correlation matrix
)
```

### Sampling for Large Datasets

```python
prism = DataPrism(
    sample_size=10000  # Analyze sample of 10K rows
)
```

### Custom Column Selection

```python
df = DataLoader.load_csv("data.csv")
results = prism.analyze(
    data=df,
    columns=['age', 'income', 'zip_code']  # Analyze specific columns
)
```

### Compact JSON Output

```python
df = DataLoader.load_csv("data.csv")
results = prism.analyze(
    data=df,
    output_path="results.json",
    compact_json=True  # Minimize JSON size
)
```

### Parquet File Benefits

Parquet format offers significant advantages:
- **Faster loading**: Columnar format with efficient compression
- **Smaller file size**: Typically 50-80% smaller than CSV
- **Type preservation**: Maintains data types (no type inference needed)
- **Column selection**: Read only needed columns (reduces memory usage)

```python
# Convert CSV to Parquet (one-time operation)
import pandas as pd
df = pd.read_csv("data.csv")
df.to_parquet("data.parquet", index=False)

# Then use Parquet for faster analysis
df = DataLoader.load_parquet("data.parquet")
prism = DataPrism()
results = prism.analyze(data=df)
```

## Provider Match Rates

DataPrism automatically computes provider match rates (hit rates) to measure data coverage from third-party providers.

Match rates are detected using one of two methods:

**Method 1: `<provider>_record_not_found` columns (Preferred)** — If your dataset includes columns like `payu_record_not_found`, DataPrism will automatically detect and use them.

**Method 2: DatasetSchema (Fallback)** — If no `record_not_found` columns exist, set `provider` on columns in your schema to group features by source.

```python
prism = DataPrism()
results = prism.analyze(data=df, schema=schema)

provider_stats = results['summary']['provider_match_rates']
```

See [examples/example_06_provider_match_rates.py](../examples/example_06_provider_match_rates.py) for a complete working example.

## Feature Counts Reference

DataPrism automatically computes feature counts across 16+ categories in `results['summary']['feature_counts']`.

| Category | Threshold | Description |
|----------|-----------|-------------|
| **High Correlation** | best association > 0.1 | Features associated with target (Pearson, Eta, or Theil's U) |
| **High IV** | IV > 0.1 | Features with strong predictive power |
| **Significant Correlations** | p-value < 0.05 | Statistically significant target correlations |
| **Suspected Leakage** | IV > 0.5 | Suspiciously high predictive power |
| **Redundant Features** | correlation > 0.7 | Highly correlated with another feature |
| **High Missing** | > 30% | Features with substantial missing values |
| **Constant Features** | 1 unique value | Zero-variance features |
| **Low Variance** | low CV | Very low coefficient of variation |
| **Not Recommended** | composite | Flagged as unsuitable for modeling |
| **Highly Skewed** | \|skewness\| > 1.0 | Heavy distributional skew |
| **High Kurtosis** | kurtosis > 3.0 | Outlier-prone features |
| **High Cardinality** | — | Categoricals with high unique-value ratio |
| **High Stability** | PSI < 0.1 | Stable distribution across cohorts/time |
| **Minor Shift** | 0.1 ≤ PSI < 0.2 | Minor distribution drift |
| **Major Shift** | PSI ≥ 0.2 | Major distribution drift |
| **Increasing Drift** | — | Worsening distribution drift over time |
| **Volatile Stability** | — | Inconsistent stability across periods |
| **Predictive Power** | — | Count by IV class: unpredictive, weak, medium, strong, very strong |
