"""Test stability calculation with real dataset."""

import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataprism import EDARunner, DataLoader


def test_stability():
    """Test stability with train/test cohorts."""
    base_path = Path(__file__).parent.parent / 'examples'
    out_path = Path(__file__).parent.parent / 'tmp'
    out_path.mkdir(exist_ok=True)

    print("Testing stability calculation with train/test cohorts...")
    print(f"Dataset path: {base_path / 'credit_risk_dataset.csv'}")

    # Load sample
    print("\nLoading dataset sample (5,000 rows)...")
    df_full = pd.read_csv(base_path / 'credit_risk_dataset.csv', nrows=5000)

    print(f"Dataset shape: {df_full.shape}")
    print(f"\nsplit distribution:")
    print(df_full['split'].value_counts())

    # Save sample
    sample_csv = out_path / 'dataset_sample_stability.csv'
    df_full.to_csv(sample_csv, index=False)

    # Run EDA with stability calculation
    print("\n" + "="*80)
    print("Running EDA with stability calculation...")
    print("="*80)

    runner = EDARunner(
        calculate_stability=True,
        cohort_column='split',
        baseline_cohort='train',
        comparison_cohort='test',
        max_correlation_features=50  # Limit for speed
    )

    results = runner.run(
        data=DataLoader.load_csv(sample_csv),
        schema=DataLoader.load_schema(base_path / 'credit_risk_schema.json'),
        target_variable='loan_status',
        output_path=out_path / 'stability_test_output.json'
    )

    # Display results
    print("\n" + "="*80)
    print("HIGHEST METRICS (WITH STABILITY)")
    print("="*80)

    hm = results['summary']['highest_metrics']

    if 'highest_correlation' in hm:
        hc = hm['highest_correlation']
        print(f"\nHighest Correlation: {hc['feature_name']}")
        print(f"   Value: {hc['value']}")

    if 'highest_iv' in hm:
        hi = hm['highest_iv']
        print(f"\nHighest IV: {hi['feature_name']}")
        print(f"   Value: {hi['value']}")

    if 'highest_statistical_score' in hm:
        hi = hm['highest_statistical_score']
        print(f"\nHighest Statistical Score: {hi['feature_name']}")
        print(f"   Score: {hi['value']}")

    if 'highest_stability' in hm:
        hs = hm['highest_stability']
        print(f"\nHighest Stability:")
        if hs.get('feature_name'):
            print(f"   Feature: {hs['feature_name']}")
            print(f"   PSI: {hs['value']} ({hs.get('stability', 'N/A')})")
            print(f"   Interpretation: {hs.get('interpretation', 'N/A')}")
        else:
            print(f"   Status: {hs.get('note', 'Not available')}")

    print("\n" + "="*80)
    print("Test completed!")
    print(f"Output saved to: {out_path / 'stability_test_output.json'}")


if __name__ == '__main__':
    test_stability()
