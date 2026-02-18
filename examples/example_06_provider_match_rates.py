"""Example 5: Provider match rates and feature counts for dashboards."""

from pathlib import Path
from dataprism import EDARunner, DataLoader

DATA_DIR = Path(__file__).parent

print("=" * 80)
print("Example 5: Provider Match Rates & Feature Counts")
print("=" * 80)

# Run EDA with stability to get feature counts including stability
runner = EDARunner(
    max_categories=20,
    calculate_stability=True,
    cohort_column='split',
    baseline_cohort='train',
    comparison_cohort='test'
)

df = DataLoader.load_csv(DATA_DIR / "credit_risk_dataset.csv")
schema = DataLoader.load_schema(DATA_DIR / "credit_risk_schema.json")

results = runner.run(
    data=df,
    schema=schema,
    target_variable="loan_status",
    output_path="tmp/eda_provider_feature_counts.json"
)

# --- Part 1: Provider match rates ---
print("\nProvider Match Rates:")
print("-" * 80)

if 'provider_match_rates' in results:
    for provider, stats in results['provider_match_rates'].items():
        print(f"  {provider.upper()}: {stats['match_rate']*100:.1f}% "
              f"({stats['matched_records']:,}/{stats['total_records']:,} records)")
        if stats.get('total_features', 0) > 0:
            print(f"    Features: {stats['total_features']}")
else:
    print("  No provider match rates (requires provider metadata in schema)")

# --- Part 2: Feature counts ---
print("\nFeature Counts:")
print("-" * 80)

fc = results['summary']['feature_counts']

categories = {
    'high_correlation': 'High Correlation',
    'redundant_features': 'Redundant Features',
    'high_iv': 'High IV',
    'high_stability': 'High Stability',
}

for key, label in categories.items():
    data = fc.get(key, {})
    count = data.get('count', 0)
    threshold = data.get('threshold', 'N/A')
    print(f"  {label}: {count} (threshold: {threshold})")

# --- Part 3: Feature details per category ---
print("\nTop High-IV Features:")
high_iv_features = fc.get('high_iv', {}).get('features', [])
for i, feat in enumerate(high_iv_features[:5], 1):
    print(f"  {i}. {feat['feature_name']:<30} IV: {feat['information_value']:.4f} ({feat['predictive_power']})")

print("\n" + "=" * 80)
print("Output saved to: tmp/eda_provider_feature_counts.json")
print("=" * 80)
