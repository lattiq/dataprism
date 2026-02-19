"""Example 1: Basic EDA without stability analysis."""

from pathlib import Path
from dataprism import DataPrism, DataLoader

DATA_DIR = Path(__file__).parent

print("=" * 80)
print("Example 1: Basic EDA (No Stability)")
print("=" * 80)

# Load data and metadata
df = DataLoader.load_csv(DATA_DIR / "credit_risk_dataset.csv")
schema = DataLoader.load_schema(DATA_DIR / "credit_risk_schema.json")

# Run EDA
prism = DataPrism(max_categories=20)

results = prism.analyze(
    data=df,
    schema=schema,
    target_variable="loan_status",
    output_path="tmp/eda_credit_risk.json"
)

print(f"✅ Analyzed {results['summary']['total_features']} features")
print(f"   Dataset size: {results['summary']['dataset_info']['rows']:,} rows")
print(f"   Avg missing: {results['summary']['avg_missing_percentage']:.2f}%")
print(f"   Data quality score: {results['summary']['data_quality']['overall_score']:.1f}/10")

print("\n" + "=" * 80)
print("Output saved to: tmp/eda_credit_risk.json")
print("=" * 80)
