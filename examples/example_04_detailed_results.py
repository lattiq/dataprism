"""Example 4: Exploring detailed results and inspecting specific features."""

from pathlib import Path
from dataprism import EDARunner, DataLoader

DATA_DIR = Path(__file__).parent

print("=" * 80)
print("Example 4: Detailed Results & Feature Inspection")
print("=" * 80)

# Run with both cohort and time-based stability
runner = EDARunner(
    max_categories=20,
    calculate_stability=True,
    cohort_column='split',
    baseline_cohort='train',
    comparison_cohort='test',
    time_based_stability=True,
    time_column='onboarding_date',
    time_window_strategy='monthly',
    baseline_period='first',
    comparison_periods='all',
    min_samples_per_period=100,
)

df = DataLoader.load_csv(DATA_DIR / "credit_risk_dataset.csv")
schema = DataLoader.load_schema(DATA_DIR / "credit_risk_schema.json")

results = runner.run(
    data=df,
    schema=schema,
    target_variable="loan_status",
    output_path="tmp/eda_detailed.json",
)

# --- Part 1: Top features by Information Value ---
print("\nTop 5 Features by Information Value:")
for feat in results['summary']['top_features_by_statistical_score'][:5]:
    psi_info = f"psi={feat['psi']:.4f}" if feat['psi'] else "psi=N/A"
    print(f"  {feat['rank']}. {feat['feature_name']}")
    print(f"     iv={feat['iv']:.4f}, power={feat['predictive_power']}, "
          f"assoc={feat['association']:.4f}, {psi_info}")

# --- Part 2: Highest metrics ---
print("\nHighest Metrics:")
hm = results['summary']['highest_metrics']
if 'highest_association' in hm:
    print(f"  Association: {hm['highest_association']['feature_name']} ({hm['highest_association']['value']:.4f})")
print(f"  IV: {hm['highest_iv']['feature_name']} ({hm['highest_iv']['value']:.4f})")

# --- Part 3: Data quality ---
dq = results['summary']['data_quality']
print(f"\nData Quality: {dq['overall_score']:.1f}/10")
for i, rec in enumerate(dq['recommended_actions'][:3], 1):
    print(f"  {i}. {rec}")

# --- Part 4: Inspect a specific feature ---
feature_name = "loan_int_rate"
feature = next((f for f in results['features'] if f['feature_name'] == feature_name), None)

if feature:
    print(f"\nFeature Detail: {feature_name}")
    print(f"  Type: {feature['type']}")
    print(f"  Provider: {feature.get('source', {}).get('provider', 'N/A')}")

    if 'target_relationship' in feature:
        tr = feature['target_relationship']
        print(f"  Pearson: {tr.get('correlation_pearson')}")
        print(f"  IV: {tr.get('information_value')}")
        print(f"  Predictive Power: {tr.get('predictive_power')}")

    if 'statistics' in feature:
        stats = feature['statistics']
        if 'mean' in stats:
            print(f"  Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
            print(f"  Range: [{stats['min']}, {stats['max']}]")

    if 'quality' in feature:
        q = feature['quality']
        print(f"  Recommended for Modeling: {q.get('recommended_for_modeling')}")

    if 'stability' in feature:
        stab = feature['stability']
        if 'cohort_based' in stab:
            cb = stab['cohort_based']
            print(f"  Cohort Stability: PSI={cb['psi']}, {cb['stability']} — {cb['interpretation']}")
        if 'time_based' in stab:
            tb = stab['time_based']
            print(f"  Time Stability: PSI={tb['psi']}, {tb['stability']}, trend={tb['trend']}")
            print(f"    {tb['interpretation']}")

print("\n" + "=" * 80)
print("Output saved to: tmp/eda_detailed.json")
print("=" * 80)
