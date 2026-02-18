# DataPrism Usage Guide

Reference documentation for advanced features. For quick start and basic usage, see the [README](../README.md).

## DatasetSchema Reference

`DatasetSchema` enables advanced functionality by defining column types, roles, providers, and sentinel values.

### Variable Type Override

Override automatic type inference:
```json
{
  "name": "customer_id",
  "type": "categorical",
  "role": "feature"
}
```

### Sentinel Values

Define values that should be treated as missing:
```json
{
  "name": "income",
  "type": "continuous",
  "role": "feature",
  "sentinels": {
    "not_found": "-1",
    "missing": "0"
  }
}
```

### Provider Tracking

Track data sources:
```json
{
  "name": "credit_score",
  "type": "continuous",
  "role": "feature",
  "provider": "bureau_provider",
  "description": "FICO credit score"
}
```

## Provider Match Rates

DataPrism automatically computes provider match rates (also called "hit rates") to help you understand data coverage from different third-party data providers.

### Automatic Detection

Provider match rates are computed automatically during EDA using one of two methods:

#### Method 1: Using `<provider>_record_not_found` columns (Preferred)

If your dataset includes columns like `payu_record_not_found`, `truecaller_record_not_found`, etc., DataPrism will automatically detect and use them:

```python
runner = EDARunner()
df = DataLoader.load_csv("data.csv")
results = runner.run(data=df)

# Access provider stats
provider_stats = results['summary']['provider_match_rates']
```

#### Method 2: Using DatasetSchema (Fallback)

If no `record_not_found` columns exist, you can use a schema to group features by provider:

```python
df = DataLoader.load_csv("data.csv")
schema = DataLoader.load_schema("schema.json")

runner = EDARunner()
results = runner.run(data=df, schema=schema)

# Provider stats show match rates based on feature null analysis
provider_stats = results['summary']['provider_match_rates']
```

### Example

See [examples/example_06_provider_match_rates.py](../examples/example_06_provider_match_rates.py) for a complete working example.

## Feature Counts

DataPrism automatically computes feature counts across 16+ categories — useful for dashboards, feature selection, and data quality monitoring.

### Automatic Computation

Feature counts are computed automatically during EDA and included in the results:

```python
from dataprism import EDARunner, DataLoader

runner = EDARunner()
df = DataLoader.load_csv("data.csv")
results = runner.run(
    data=df,
    target_variable="target"  # Required for correlation and IV
)

# Access feature counts
feature_counts = results['summary']['feature_counts']

print(f"High Correlation: {feature_counts['high_correlation']['count']}")
print(f"Redundant Features: {feature_counts['redundant_features']['count']}")
print(f"High IV: {feature_counts['high_iv']['count']}")
print(f"High Stability: {feature_counts['high_stability']['count']}")
```

### Categories

**Target Relationship**

| Category | Threshold | Description |
|----------|-----------|-------------|
| **High Correlation** | best association > 0.1 | Features associated with target (Pearson, Eta, or Theil's U) |
| **High IV** | IV > 0.1 | Features with strong predictive power |
| **Significant Correlations** | p-value < 0.05 | Statistically significant target correlations |
| **Suspected Leakage** | IV > 0.5 | Features with suspiciously high predictive power |

**Feature Quality**

| Category | Threshold | Description |
|----------|-----------|-------------|
| **Redundant Features** | correlation > 0.7 | Highly correlated with another feature |
| **High Missing** | > 30% | Features with substantial missing values |
| **Constant Features** | 1 unique value | Zero-variance features |
| **Low Variance** | low CV | Features with very low coefficient of variation |
| **Not Recommended** | composite | Features flagged as unsuitable for modeling |
| **Highly Skewed** | \|skewness\| > 1.0 | Features with heavy distributional skew |
| **High Kurtosis** | kurtosis > 3.0 | Outlier-prone features |
| **High Cardinality** | — | Categoricals with high unique-value ratio |

**Predictive Power Breakdown**

| Category | Description |
|----------|-------------|
| **Predictive Power** | Count of features by IV class: unpredictive, weak, medium, strong, very strong |

**Stability**

| Category | Threshold | Description |
|----------|-----------|-------------|
| **High Stability** | PSI < 0.1 | Stable distribution across cohorts/time |
| **Minor Shift** | 0.1 ≤ PSI < 0.2 | Minor distribution drift |
| **Major Shift** | PSI ≥ 0.2 | Major distribution drift |
| **Increasing Drift** | — | Worsening distribution drift over time |
| **Volatile Stability** | — | Inconsistent stability across periods |

## Advanced Configuration

### Correlation Settings

```python
runner = EDARunner(
    top_correlations=10,           # Top N correlations per feature
    max_correlation_features=500   # Limit features in correlation matrix
)
```

### Sampling for Large Datasets

```python
runner = EDARunner(
    sample_size=10000  # Analyze sample of 10K rows
)
```

### Custom Column Selection

```python
df = DataLoader.load_csv("data.csv")
results = runner.run(
    data=df,
    columns=['age', 'income', 'zip_code']  # Analyze specific columns
)
```

### Compact JSON Output

```python
df = DataLoader.load_csv("data.csv")
results = runner.run(
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
runner = EDARunner()
results = runner.run(data=df)
```
