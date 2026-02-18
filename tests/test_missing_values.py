"""Test missing value handling and provider match rates."""

import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from dataprism.data.sentinels import replace_sentinel_values_with_nulls
from dataprism.analyzers.match_rates import compute_provider_match_rates
from dataprism.schema import ColumnConfig, ColumnRole, ColumnType, Sentinels

def test_sentinel_replacement():
    """Test replacing sentinel values with nulls."""
    # Create test data
    df = pd.DataFrame({
        'age': [25, -1, 30, -1, 35],
        'income': [50000, 60000, -1, 70000, -1],
        'name': ['Alice', '', 'Bob', '', 'Charlie'],
        'city': ['NYC', 'LA', '', 'SF', 'NYC']
    })

    col_configs = {
        'age': ColumnConfig('age', ColumnType.CONTINUOUS, ColumnRole.FEATURE, sentinels=Sentinels(not_found='-1')),
        'income': ColumnConfig('income', ColumnType.CONTINUOUS, ColumnRole.FEATURE, sentinels=Sentinels(not_found='-1')),
        'name': ColumnConfig('name', ColumnType.CATEGORICAL, ColumnRole.FEATURE, sentinels=Sentinels(not_found='')),
        'city': ColumnConfig('city', ColumnType.CATEGORICAL, ColumnRole.FEATURE, sentinels=Sentinels(not_found='')),
    }

    # Replace sentinels
    df_clean = replace_sentinel_values_with_nulls(df, col_configs)

    # Verify
    assert df_clean['age'].isna().sum() == 2, "Should have 2 nulls in age"
    assert df_clean['income'].isna().sum() == 2, "Should have 2 nulls in income"
    assert df_clean['name'].isna().sum() == 2, "Should have 2 nulls in name"
    assert df_clean["city"].isna().sum() == 1, "Should have 1 null in city"


def test_provider_match_rates():
    """Test computing provider match rates."""
    # Create test data (already cleaned)
    df = pd.DataFrame({
        'age': [25, None, 30, None, 35],
        'income': [50000, 60000, None, 70000, None],
        'name': ['Alice', None, 'Bob', None, 'Charlie'],
        'credit_score': [700, 750, None, 680, None]
    })

    col_configs = {
        'age': ColumnConfig('age', ColumnType.CONTINUOUS, ColumnRole.FEATURE, provider='bureau'),
        'income': ColumnConfig('income', ColumnType.CONTINUOUS, ColumnRole.FEATURE, provider='bureau'),
        'name': ColumnConfig('name', ColumnType.CATEGORICAL, ColumnRole.FEATURE, provider='kyc'),
        'credit_score': ColumnConfig('credit_score', ColumnType.CONTINUOUS, ColumnRole.FEATURE, provider='bureau'),
    }

    # Compute match rates
    provider_stats = compute_provider_match_rates(df, col_configs)

    # Verify bureau: 3 features, all 5 records have at least one bureau feature
    assert provider_stats['bureau']['total_features'] == 3
    assert provider_stats["bureau"]["matched_records"] == 5

    # Verify KYC: 1 feature, 3/5 records have name
    assert provider_stats['kyc']['total_features'] == 1
    assert provider_stats['kyc']['matched_records'] == 3


def test_with_real_data():
    """Test with real dataset sample (requires tmp/ data files in new schema format)."""
    import json
    import pytest

    base_path = Path(__file__).parent.parent / 'examples'

    dataset_path = base_path / 'credit_risk_dataset.csv'
    schema_path = base_path / 'credit_risk_schema.json'

    if not dataset_path.exists() or not schema_path.exists():
        pytest.skip("Real data files not available in tmp/")

    with open(schema_path) as f:
        data = json.load(f)
    if "columns" not in data:
        pytest.skip("credit_risk_schema.json is not in DatasetSchema format")

    # Load small sample
    df = pd.read_csv(dataset_path, nrows=1000)

    # Load schema using DataLoader
    from dataprism.data.loader import DataLoader
    schema = DataLoader.load_schema(schema_path)

    # Build col_configs dict
    col_configs = schema.columns

    # Check before
    null_counts_before = df.isna().sum().sum()

    # Replace sentinels
    df_clean = replace_sentinel_values_with_nulls(df, col_configs)

    # Check after
    null_counts_after = df_clean.isna().sum().sum()
    assert null_counts_after >= null_counts_before, "Should have same or more nulls after replacement"

    # Compute provider stats
    provider_stats = compute_provider_match_rates(df_clean, col_configs)
    # Just verify it runs without error


def test_provider_match_rates_from_columns_fallback():
    """Test schema-less fallback: detects <provider>_record_not_found columns."""
    df = pd.DataFrame({
        'bureau_record_not_found': [0, 0, 1, 0, 1],
        'kyc_record_not_found': [0, 1, 1, 0, 0],
        'age': [25, 30, None, 35, None],
        'name': ['Alice', None, None, 'Bob', 'Charlie'],
    })

    # No col_configs — should fall back to column detection
    provider_stats = compute_provider_match_rates(df)

    assert 'bureau' in provider_stats, "Should detect bureau provider"
    assert 'kyc' in provider_stats, "Should detect kyc provider"

    # bureau: rows 0,1,3 have record_not_found==0 → 3 matched
    assert provider_stats['bureau']['matched_records'] == 3
    assert provider_stats['bureau']['match_rate'] == 0.6
    assert provider_stats['bureau']['computation_method'] == 'record_not_found_column'
    assert provider_stats['bureau']['total_records'] == 5

    # kyc: rows 0,3,4 have record_not_found==0 → 3 matched
    assert provider_stats['kyc']['matched_records'] == 3
    assert provider_stats['kyc']['match_rate'] == 0.6

    # total_features is 0 since no schema provides feature mappings
    assert provider_stats['bureau']['total_features'] == 0
    assert provider_stats['kyc']['total_features'] == 0


def test_provider_match_rates_empty_fallback():
    """Test schema-less fallback with no record_not_found columns returns empty."""
    df = pd.DataFrame({
        'age': [25, 30, 35],
        'income': [50000, 60000, 70000],
    })

    provider_stats = compute_provider_match_rates(df)
    assert provider_stats == {}, "Should return empty dict when no providers detected"


if __name__ == '__main__':
    test_sentinel_replacement()
    test_provider_match_rates()
    test_provider_match_rates_from_columns_fallback()
    test_provider_match_rates_empty_fallback()
    test_with_real_data()
