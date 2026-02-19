"""Example 2: Cohort-based stability analysis (Train/Test validation)."""

from pathlib import Path
from dataprism import DataPrism, DataLoader

DATA_DIR = Path(__file__).parent

print("=" * 80)
print("Example 2: Cohort-Based Stability (Train/Test Validation)")
print("=" * 80)

prism = DataPrism(
    max_categories=20,
    calculate_stability=True,
    cohort_column='split',
    baseline_cohort='train',
    comparison_cohort='test'
)
# Load data
df = DataLoader.load_csv(DATA_DIR / "credit_risk_dataset.csv")
schema = DataLoader.load_schema(DATA_DIR / "credit_risk_schema.json")


results = prism.analyze(
    data=df,
    schema=schema,
    target_variable="loan_status",
    output_path="tmp/eda_cohort_stability.json"
)

print(f"✅ Cohort-based stability calculated")
print(f"   Total features: {results['summary']['total_features']}")
print(f"   Data quality score: {results['summary']['data_quality']['overall_score']:.1f}/10")

if 'highest_stability' in results['summary']['highest_metrics']:
    hs = results['summary']['highest_metrics']['highest_stability']
    if hs.get('feature_name'):
        print(f"   Most stable feature: {hs['feature_name']} (PSI: {hs['value']:.4f})")
    else:
        print(f"   Stability: {hs.get('note')}")

print("\n" + "=" * 80)
print("Output saved to: tmp/eda_cohort_stability.json")
print("=" * 80)
