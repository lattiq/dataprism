"""Test time-based stability calculation with real dataset."""

import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataprism import DataPrism, DataLoader


def test_time_based_stability():
    """Test time-based stability with onboarding_time column."""
    base_path = Path(__file__).parent.parent / 'examples'

    print("Testing time-based stability with onboarding_time...")
    print(f"Dataset path: {base_path / 'credit_risk_dataset.csv'}")

    # Load sample
    print("\nLoading dataset sample (5,000 rows)...")
    df_full = pd.read_csv(base_path / 'credit_risk_dataset.csv', nrows=5000)

    print(f"Dataset shape: {df_full.shape}")

    # Check onboarding_time column
    if 'onboarding_time' in df_full.columns:
        df_full['onboarding_time'] = pd.to_datetime(df_full['onboarding_time'], errors='coerce')
        print(f"\nonboarding_time distribution:")
        print(f"  Min: {df_full['onboarding_time'].min()}")
        print(f"  Max: {df_full['onboarding_time'].max()}")
        print(f"  Span: {(df_full['onboarding_time'].max() - df_full['onboarding_time'].min()).days} days")
    else:
        print("\nERROR: onboarding_time column not found!")
        return

    # Save sample
    sample_csv = base_path / 'dataset_sample_time_stability.csv'
    df_full.to_csv(sample_csv, index=False)

    # Test 1: Monthly windows
    print("\n" + "="*80)
    print("TEST 1: MONTHLY TIME WINDOWS")
    print("="*80)

    prism = DataPrism(
        time_based_stability=True,
        time_column='onboarding_time',
        time_window_strategy='monthly',
        baseline_period='first',
        comparison_periods='all',
        min_samples_per_period=50,
        max_correlation_features=50
    )

    results = prism.analyze(
        data=DataLoader.load_csv(sample_csv),
        schema=DataLoader.load_schema(base_path / 'credit_risk_schema.json'),
        target_variable='loan_status',
        output_path=base_path / 'time_stability_monthly.json'
    )

    # Display results
    print("\n" + "="*80)
    print("MONTHLY STABILITY RESULTS")
    print("="*80)

    if 'summary' in results.get('stability_analysis', {}):
        sa = results['stability_analysis']
        print(f"\nMethod: {sa.get('method')}")
        print(f"Time column: {sa.get('time_column')}")
        print(f"Window strategy: {sa.get('window_strategy')}")

        print(f"\nBaseline period:")
        bp = sa['baseline_period']
        print(f"  {bp['start']} to {bp['end']}")
        print(f"  Samples: {bp['sample_count']}")

        print(f"\nComparison periods: {len(sa['comparison_periods'])}")
        for period in sa['comparison_periods'][:3]:
            print(f"  Period {period['period_id']}: {period['start']} to {period['end']} ({period['sample_count']} samples)")

        summary = sa['summary']
        print(f"\nTemporal Drift Summary:")
        print(f"  Total features analyzed: {summary['total_features_analyzed']}")
        print(f"  Stable features: {summary['stable_features']}")
        print(f"  Minor drift: {summary['minor_drift_features']}")
        print(f"  Major drift: {summary['major_drift_features']}")
        print(f"  Features with increasing drift: {len(summary['features_with_increasing_drift'])}")
        if summary['features_with_increasing_drift']:
            print(f"    Examples: {summary['features_with_increasing_drift'][:5]}")

    if 'highest_metrics' in results:
        hm = results['highest_metrics']
        if 'highest_stability' in hm:
            hs = hm['highest_stability']
            print(f"\n✅ Highest Stability (Most Stable Feature):")
            if hs.get('feature_name'):
                print(f"   Feature: {hs['feature_name']}")
                print(f"   Avg PSI: {hs['value']}")
                print(f"   Status: {hs.get('stability', 'N/A')}")
                if hs.get('trend'):
                    print(f"   Trend: {hs.get('trend')}")
            else:
                print(f"   {hs.get('note', 'Not available')}")

    # Test 2: Weekly windows
    print("\n" + "="*80)
    print("TEST 2: WEEKLY TIME WINDOWS")
    print("="*80)

    prism_weekly = DataPrism(
        time_based_stability=True,
        time_column='onboarding_time',
        time_window_strategy='weekly',
        baseline_period='first',
        comparison_periods='all',
        min_samples_per_period=50,
        max_correlation_features=50
    )

    results_weekly = prism_weekly.analyze(
        data=DataLoader.load_csv(sample_csv),
        schema=DataLoader.load_schema(base_path / 'credit_risk_schema.json'),
        target_variable='loan_status',
        output_path=base_path / 'time_stability_weekly.json'
    )

    if 'summary' in results_weekly.get('stability_analysis', {}):
        sa = results_weekly['stability_analysis']
        summary = sa['summary']
        print(f"\nWeekly Analysis:")
        print(f"  Total periods analyzed: {summary['total_periods_analyzed']}")
        print(f"  Stable features: {summary['stable_features']}")
        print(f"  Drifting features: {summary['minor_drift_features'] + summary['major_drift_features']}")

    # Test 3: Quartile-based windows
    print("\n" + "="*80)
    print("TEST 3: QUARTILE-BASED WINDOWS (Equal Sample Sizes)")
    print("="*80)

    prism_quartile = DataPrism(
        time_based_stability=True,
        time_column='onboarding_time',
        time_window_strategy='quartiles',
        baseline_period='first',
        comparison_periods='all',
        min_samples_per_period=50,
        max_correlation_features=50
    )

    results_quartile = prism_quartile.analyze(
        data=DataLoader.load_csv(sample_csv),
        schema=DataLoader.load_schema(base_path / 'credit_risk_schema.json'),
        target_variable='loan_status',
        output_path=base_path / 'time_stability_quartiles.json'
    )

    if 'summary' in results_quartile.get('stability_analysis', {}):
        sa = results_quartile['stability_analysis']
        print(f"\nQuartile periods:")
        for period in sa['comparison_periods']:
            print(f"  Q{period['period_id']}: {period['start']} to {period['end']} ({period['sample_count']} samples)")

    print("\n" + "="*80)
    print("✅ All tests completed!")
    print(f"\nOutputs saved to:")
    print(f"  - {base_path / 'time_stability_monthly.json'}")
    print(f"  - {base_path / 'time_stability_weekly.json'}")
    print(f"  - {base_path / 'time_stability_quartiles.json'}")


if __name__ == '__main__':
    test_time_based_stability()
