"""Test with real dataset sample."""

import pandas as pd
from pathlib import Path
import sys
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataprism import EDARunner, DataLoader


def test_real_dataset_sample():
    """Test EDA with real dataset sample."""
    base_path = Path(__file__).parent.parent / 'examples'
    out_path = Path(__file__).parent.parent / 'tmp'
    out_path.mkdir(exist_ok=True)

    print("Testing with real dataset sample...")
    print(f"Dataset path: {base_path / 'credit_risk_dataset.csv'}")
    print(f"Schema: {base_path / 'credit_risk_schema.json'}")

    # Load sample (first 10k rows to speed up testing)
    print("\nLoading dataset sample (10,000 rows)...")
    df_full = pd.read_csv(base_path / 'credit_risk_dataset.csv', nrows=10000)

    print(f"Dataset shape: {df_full.shape}")
    print(f"Columns: {df_full.columns[:10].tolist()}...")

    target = 'loan_status'
    print(f"\nUsing target column: {target}")

    # Save sample
    sample_csv = out_path / 'dataset_sample_10k.csv'
    df_full.to_csv(sample_csv, index=False)
    print(f"Saved sample to: {sample_csv}")

    # Run EDA with schema
    print("\nRunning EDA with feature metadata...")
    start_time = time.time()

    runner = EDARunner(
        max_correlation_features=100  # Limit correlation matrix for speed
    )

    results = runner.run(
        data=DataLoader.load_csv(sample_csv),
        schema=DataLoader.load_schema(base_path / 'credit_risk_schema.json'),
        target_variable=target,
        output_path=out_path / 'real_dataset_eda_enhanced.json'
    )

    elapsed = time.time() - start_time

    # Print results
    print(f"\n✅ EDA completed in {elapsed:.2f} seconds")
    print("\n" + "="*80)
    print("SUMMARY METRICS")
    print("="*80)

    summary = results['summary']
    dataset_info = summary['dataset_info']
    print(f"Total Features: {summary['total_features']}")
    print(f"Total Rows: {dataset_info['rows']}")
    print(f"Provider Features: {summary['provider_features']}")
    print(f"Memory Usage: {dataset_info['memory_mb']:.2f} MB")
    print(f"Avg Missing %: {summary['avg_missing_percentage']}")
    print(f"Avg Outliers %: {summary['avg_outliers_percentage']}")
    print(f"\nFeature Type Breakdown:")
    for ftype, count in summary['feature_types'].items():
        print(f"  {ftype}: {count}")

    print(f"\nFeature Counts:")
    fc = summary['feature_counts']
    print(f"  High Correlation Features: {fc['high_correlation']['count']}")
    print(f"  Redundant Features: {fc['redundant_features']['count']}")
    print(f"  High IV Features: {fc['high_iv']['count']}")

    print("\n" + "="*80)
    print("HIGHEST METRICS")
    print("="*80)

    hm = summary['highest_metrics']

    if 'highest_association' in hm:
        ha = hm['highest_association']
        print(f"Highest Association: {ha['feature_name']}")
        print(f"  Value: {ha['value']}")
        print(f"  Target: {ha['target']}")

    if 'highest_iv' in hm:
        hi = hm['highest_iv']
        print(f"\nHighest Information Value: {hi['feature_name']}")
        print(f"  IV: {hi['value']}")

    if 'highest_stability' in hm:
        hs = hm['highest_stability']
        if hs['feature_name']:
            print(f"\nHighest Stability: {hs['feature_name']}")
            print(f"  Score: {hs['value']}")
        else:
            print(f"\nHighest Stability: {hs.get('note', 'Not available')}")

    print("\n" + "="*80)
    print("TOP 10 FEATURES BY INFORMATION VALUE")
    print("="*80)

    for feat in summary['top_features_by_statistical_score'][:10]:
        print(f"{feat['rank']:2d}. {feat['feature_name']:40s} "
              f"iv={feat['iv']:7.4f}  power={feat['predictive_power'] or 'N/A':15s}  assoc={feat['association']:7.4f}")

    print("\n" + "="*80)
    print("DATA QUALITY")
    print("="*80)

    dq = summary['data_quality']
    print(f"Overall Score: {dq['overall_score']}/10")
    print(f"Features with high missing: {dq['features_with_high_missing_count']}")
    print(f"Features with low variance: {dq['features_with_low_variance_count']}")
    print(f"Features with outliers: {dq['features_with_outliers_count']}")
    print(f"\nRecommendations ({len(dq['recommended_actions'])}):")
    for i, rec in enumerate(dq['recommended_actions'][:5], 1):
        print(f"  {i}. {rec}")
    if len(dq['recommended_actions']) > 5:
        print(f"  ... and {len(dq['recommended_actions']) - 5} more")

    print("\n" + "="*80)
    print("SAMPLE FEATURE DETAILS")
    print("="*80)

    # Show details of top feature
    if results['features']:
        top_feat = results['features'][0]
        print(f"\nFeature: {top_feat['feature_name']}")
        print(f"  Type: {top_feat['type']}")
        print(f"  Provider: {top_feat.get('source', {}).get('provider', 'N/A')}")
        print(f"  Description: {top_feat.get('description', 'N/A')[:80]}...")

        if 'statistics' in top_feat:
            stats = top_feat['statistics']
            print(f"\n  Statistics:")
            if 'mean' in stats:
                print(f"    Mean: {stats.get('mean')}")
                print(f"    Std: {stats.get('std')}")
                print(f"    Min/Max: {stats.get('min')} / {stats.get('max')}")
            if 'unique' in stats:
                print(f"    Unique Values: {stats.get('unique')}")
                print(f"    Mode: {stats.get('mode')}")

        if 'target_relationship' in top_feat:
            tr = top_feat['target_relationship']
            print(f"\n  Target Relationship:")
            print(f"    Pearson Corr: {tr.get('correlation_pearson')}")
            print(f"    Spearman Corr: {tr.get('correlation_spearman')}")
            print(f"    Information Value: {tr.get('information_value')}")
            print(f"    Predictive Power: {tr.get('predictive_power')}")

        if 'quality' in top_feat:
            quality = top_feat['quality']
            print(f"\n  Quality Flags:")
            print(f"    High Missing: {quality.get('has_high_missing')}")
            print(f"    Low Variance: {quality.get('has_low_variance')}")
            print(f"    Has Outliers: {quality.get('has_outliers')}")
            print(f"    Recommended for Modeling: {quality.get('recommended_for_modeling')}")

    print("\n" + "="*80)
    print(f"\n✅ Full output saved to: {out_path / 'real_dataset_eda_enhanced.json'}")
    print(f"File size: {(out_path / 'real_dataset_eda_enhanced.json').stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == '__main__':
    test_real_dataset_sample()
