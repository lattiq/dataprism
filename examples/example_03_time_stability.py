"""Example 3: Time-based stability analysis (monthly, weekly, quartile windows)."""

from pathlib import Path
from dataprism import EDARunner, DataLoader

DATA_DIR = Path(__file__).parent

print("=" * 80)
print("Example 3: Time-Based Stability (Drift Detection)")
print("=" * 80)

df = DataLoader.load_csv(DATA_DIR / "credit_risk_dataset.csv")
schema = DataLoader.load_schema(DATA_DIR / "credit_risk_schema.json")

# --- Strategy 1: Monthly windows ---
print("\n1. Monthly Windows:")
print("-" * 80)

runner_monthly = EDARunner(
    max_categories=20,
    time_based_stability=True,
    time_column='onboarding_date',
    time_window_strategy='monthly',
    baseline_period='first',
    comparison_periods='all',
    min_samples_per_period=100
)

results_monthly = runner_monthly.run(
    data=df,
    schema=schema,
    target_variable="loan_status",
    output_path="tmp/eda_time_monthly.json"
)

# Show per-feature stability from the formatted output
print("\n   Per-feature time-based stability:")
for feat in results_monthly.get('features', [])[:5]:
    stab = feat.get('stability', {}).get('time_based')
    if stab:
        print(f"   {feat['feature_name']}: PSI={stab['psi']}, {stab['stability']}, {stab['trend']}")

# --- Strategy 2: Weekly windows ---
print("\n2. Weekly Windows:")
print("-" * 80)

runner_weekly = EDARunner(
    max_categories=20,
    time_based_stability=True,
    time_column='onboarding_date',
    time_window_strategy='weekly',
    baseline_period='first',
    comparison_periods='all',
    min_samples_per_period=50
)

results_weekly = runner_weekly.run(
    data=df,
    schema=schema,
    target_variable="loan_status",
    output_path="tmp/eda_time_weekly.json"
)

print("\n   Per-feature time-based stability:")
for feat in results_weekly.get('features', [])[:5]:
    stab = feat.get('stability', {}).get('time_based')
    if stab:
        print(f"   {feat['feature_name']}: PSI={stab['psi']}, {stab['stability']}, {stab['trend']}")

# --- Strategy 3: Quartile windows (equal sample sizes) ---
print("\n3. Quartile Windows (Equal Sample Sizes):")
print("-" * 80)

runner_quartile = EDARunner(
    max_categories=20,
    time_based_stability=True,
    time_column='onboarding_date',
    time_window_strategy='quartiles',
    baseline_period='first',
    comparison_periods='all',
    min_samples_per_period=100
)

results_quartile = runner_quartile.run(
    data=df,
    schema=schema,
    target_variable="loan_status",
    output_path="tmp/eda_time_quartiles.json"
)

print("\n   Per-feature time-based stability:")
for feat in results_quartile.get('features', [])[:5]:
    stab = feat.get('stability', {}).get('time_based')
    if stab:
        print(f"   {feat['feature_name']}: PSI={stab['psi']}, {stab['stability']}, {stab['trend']}")

print("\n" + "=" * 80)
print("Outputs saved to: tmp/eda_time_{monthly,weekly,quartiles}.json")
print("=" * 80)
