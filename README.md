# DataPrism

A Python library for exploratory data analysis with data profiling, quality assessment, and stability monitoring.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Interactive Viewer

DataPrism includes a built-in interactive dashboard to explore your analysis results in the browser.

```python
from dataprism import DataPrism, DataLoader

# Load data from CSV or Parquet
df = DataLoader.load_csv("data.csv")
# df = DataLoader.load_parquet("data.parquet")

# Run analysis and launch viewer
prism = DataPrism()
prism.analyze(
    data=df,
    target_variable="target",
    exclude_columns=["id", "split", "onboarding_date"],
    output_path="eda_results.json",
)
prism.view()
```

**Summary** — Dataset overview, insights, top features by IV, data quality score, and provider match rates.

![Summary](docs/images/viewer-summary.png)

**Catalog** — Sortable feature table with type, provider, target correlation, IV, and PSI at a glance.

![Catalog](docs/images/viewer-catalog.png)

**Deep Dive** — Per-feature detail view with statistics, violin plots, distribution charts, PSI trend analysis, target associations, and correlations.

![Deep Dive](docs/images/viewer-deepdive.png)

**Associations** — Mixed-method heatmap (Pearson, Theil's U, Eta) showing relationships across all features.

![Associations](docs/images/viewer-associations.png)

## How DataPrism Compares

| Capability                  | DataPrism | ydata-profiling | Sweetviz | D-Tale | AutoViz | DataPrep |
| --------------------------- | --------- | --------------- | -------- | ------ | ------- | -------- |
| Predictive power (IV / WoE) | ✅         | ➖               | 🟡       | 🟡     | ➖       | ➖        |
| Drift detection (PSI)       | ✅         | 🟡              | 🟡       | ➖      | ➖       | 🟡       |
| Data quality score          | ✅         | ➖               | ➖        | ➖      | ➖       | ➖        |
| Multi-source match rates    | ✅         | ➖               | ➖        | ➖      | ➖       | ➖        |
| Schema-aware profiling      | ✅         | 🟡              | 🟡       | 🟡     | ➖       | 🟡       |
| Structured JSON output      | ✅         | ✅               | ➖        | 🟡     | ➖       | 🟡       |
| Interactive explorer        | ✅         | ✅               | 🟡       | ✅      | 🟡      | ✅        |

> ✅ Supported  🟡 Partial  ➖ Not supported

## Installation

```bash
pip install dataprism
```

## Quick Start

```python
from dataprism import DataPrism, DataLoader

df = DataLoader.load_csv("data.csv")

prism = DataPrism()
results = prism.analyze(
    data=df,
    exclude_columns=["customer_id", "created_at"],
    target_variable="target",
    output_path="eda_results.json"
)
```

For schema-aware profiling, stability analysis, and advanced configuration, see the [Usage Guide](docs/USAGE.md).

## Roadmap

DataPrism is being built for the AI era — where data analysis is increasingly driven by LLM agents, automated pipelines, and programmatic consumers rather than humans clicking through dashboards.

### AI-Native Analysis

- **Natural language insights** — Auto-generated plain-English summaries of each feature, anomalies, and recommendations that LLMs can directly incorporate into reports.

### Closing the Gaps

- **Dataset comparison** — Side-by-side train/test/production profiling with automatic drift highlights.
- **Scatter & pair plots** — Interactive scatter matrices for continuous feature pairs with target coloring.
- **Auto-visualization** — One-line generation of per-feature visual summaries exportable as images.
- **Spark/Dask support** — Distributed computation for datasets that don't fit in memory.
- **Streaming analysis** — Incremental profiling for real-time data pipelines without re-analyzing the full dataset.

### Deeper Intelligence

- **Automated feature recommendations** — Go beyond flagging issues to suggesting transformations (log, binning, encoding) based on distribution shape and target relationship.
- **Anomaly explanations** — When outliers or drift are detected, surface the likely cause (data pipeline issues, population shift, seasonality).
- **Cross-dataset lineage** — Track how feature distributions evolve across model versions and data refreshes.

## Documentation

- [Usage Guide](docs/USAGE.md) — schema, stability analysis, advanced configuration, provider match rates
- [Architecture](docs/ARCHITECTURE.md) — internals, module structure, data flow
- [Decision Records](docs/decisions/) — key design decisions and rationale
- [Examples](examples/) — usage examples and demos

## Development

```bash
pip install -e .           # Install for development
python -m build            # Build package
python -m pytest tests/    # Run tests
```

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

- Email: [dev@lattiq.com](mailto:dev@lattiq.com)
- GitHub: [https://github.com/lattiq/dataprism](https://github.com/lattiq/dataprism)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
