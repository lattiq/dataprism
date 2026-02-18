# DataPrism

A comprehensive Python library for exploratory data analysis with advanced features for data profiling, quality assessment, and stability monitoring.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Interactive Viewer

DataPrism includes a built-in interactive dashboard to explore your analysis results in the browser.

```python
from dataprism.viewer.server import serve_results

# From a saved JSON file
serve_results("eda_results.json")

# Or directly from EDA results
results = runner.run(data=df, schema=schema, target_variable="target")
serve_results(results)
```

**Summary** — Dataset overview, insights, top features by IV, data quality score, and provider match rates.

![Summary](docs/images/viewer-summary.png)

**Catalog** — Sortable feature table with type, provider, target correlation, IV, and PSI at a glance.

![Catalog](docs/images/viewer-catalog.png)

**Deep Dive** — Per-feature detail view with statistics, box plots, distribution charts, target associations, and correlations.

![Deep Dive](docs/images/viewer-deepdive.png)

**Associations** — Mixed-method heatmap (Pearson, Theil's U, Eta) showing relationships across all features.

![Associations](docs/images/viewer-associations.png)

## Features

- **Automated Feature Analysis** — Continuous and categorical profiling with automatic type inference and missing value detection
- **Target Relationship Analysis** — Information Value (IV), Weight of Evidence (WoE), optimal binning, predictive power classification
- **Correlation & Association Analysis** — Pearson, Spearman, Theil's U, Eta with unified association matrix across all feature types
- **Quality Assessment** — Automated scoring (0-10), per-feature quality flags, actionable recommendations
- **Sentinel Value Handling** — Automatic detection and replacement of no-hit values with nullable type preservation
- **Cohort-Based Stability** — PSI and KS test for train/test drift detection
- **Time-Based Stability** — Monthly, weekly, quartile, or custom time windows with temporal trend analysis
- **Provider Match Rates** — Automatic data coverage statistics by provider
- **Large Dataset Support** — CSV and Parquet formats, chunked reading, configurable sampling

## Installation

```bash
pip install dataprism
```

## Quick Start

### Basic Usage

```python
from dataprism import EDARunner, DataLoader
import pandas as pd

# Option 1: Load from file using DataLoader
df = DataLoader.load_csv("data.csv")

# Option 2: Use existing DataFrame
df = pd.read_csv("data.csv")  # or from database, etc.

# Initialize runner
runner = EDARunner(
    max_categories=50,
    top_correlations=10
)

# Run analysis
results = runner.run(
    data=df,
    output_path="eda_results.json"
)
```

### With DatasetSchema

```python
from dataprism import (
    EDARunner, DataLoader,
    ColumnConfig, ColumnType, ColumnRole, Sentinels, DatasetSchema,
)

# Load data and schema
df = DataLoader.load_csv("data.csv")
schema = DataLoader.load_schema("schema.json")

# Or create schema programmatically
schema = DatasetSchema([
    ColumnConfig('age', ColumnType.CONTINUOUS, ColumnRole.FEATURE,
                 provider='demographics', description='User age',
                 sentinels=Sentinels(not_found='-1')),
    ColumnConfig('zip_code', ColumnType.CATEGORICAL, ColumnRole.FEATURE,
                 provider='address', description='ZIP code',
                 sentinels=Sentinels(not_found='', missing='00000')),
    ColumnConfig('target', ColumnType.BINARY, ColumnRole.TARGET),
])

# Run with schema
runner = EDARunner()
results = runner.run(
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
      "sentinels": {
        "not_found": "-1",
        "missing": null
      }
    }
  ]
}
```

### Stability Analysis

#### Cohort-Based (Train/Test)

```python
from dataprism import EDARunner, DataLoader

# Load data and schema
df = DataLoader.load_parquet("data.parquet")
schema = DataLoader.load_schema("schema.json")

# Configure for stability analysis
runner = EDARunner(
    calculate_stability=True,
    cohort_column='dataTag',
    baseline_cohort='training',
    comparison_cohort='test'
)

results = runner.run(
    data=df,
    schema=schema
)
```

#### Time-Based

```python
from dataprism import EDARunner, DataLoader

# Load data and schema
df = DataLoader.load_parquet("data.parquet")
schema = DataLoader.load_schema("schema.json")

# Configure for time-based stability
runner = EDARunner(
    time_based_stability=True,
    time_column='onboarding_time',
    time_window_strategy='monthly',  # or 'weekly', 'quartiles', 'custom'
    baseline_period='first',
    comparison_periods='all',
    min_samples_per_period=100
)

results = runner.run(
    data=df,
    schema=schema
)
```

## Development

```bash
pip install -e .           # Install for development
python -m build            # Build package
python -m pytest tests/    # Run tests
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — internals, module structure, data flow
- [Usage Guide](docs/USAGE.md) — advanced configuration, provider match rates, feature counts reference
- [Decision Records](docs/decisions/) — key design decisions and rationale
- [Examples](examples/) — usage examples and demos

## Requirements

- Python 3.9+
- pandas >= 2.0.0
- numpy >= 1.24.0
- scipy >= 1.10.0
- pyarrow >= 10.0.0 (for Parquet support)

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contact

For questions or suggestions:
- Email: dev@lattiq.com
- GitHub: [https://github.com/lattiq/dataprism](https://github.com/lattiq/dataprism)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
