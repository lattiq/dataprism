"""Example 4: EDA with Parquet file support."""

from pathlib import Path
from dataprism import DataPrism, DataLoader

DATA_DIR = Path(__file__).parent

print("=" * 80)
print("Example 4: Parquet File Support")
print("=" * 80)

schema = DataLoader.load_schema(DATA_DIR / "credit_risk_schema.json")

# --- Part 1: Basic EDA with Parquet ---
print("\n1. Basic EDA with Parquet file:")
print("-" * 80)

prism = DataPrism(max_categories=20)
df = DataLoader.load_parquet(DATA_DIR / "credit_risk_dataset.parquet")

results = prism.analyze(
    data=df,
    schema=schema,
    target_variable="loan_status",
    output_path="tmp/eda_parquet.json"
)

print(f"   Analyzed {results['summary']['total_features']} features")
print(f"   Dataset: {results['summary']['dataset_info']['rows']:,} rows")

# --- Part 2: Cohort stability with Parquet ---
print("\n2. Stability analysis with Parquet:")
print("-" * 80)

prism_stability = DataPrism(
    max_categories=20,
    calculate_stability=True,
    cohort_column="split",
    baseline_cohort="train",
    comparison_cohort="test"
)

results_stability = prism_stability.analyze(
    data=df,
    schema=schema,
    target_variable="loan_status",
    output_path="tmp/eda_parquet_stability.json"
)

print(f"   Stability analysis completed")
print(f"   Features: {results_stability['summary']['total_features']}")

print("\n" + "=" * 80)
print("Parquet advantages: faster loading, smaller files, preserves dtypes")
print("=" * 80)
