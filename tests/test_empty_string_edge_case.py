"""Test edge case: numeric sentinels configured but data contains empty strings."""

import pandas as pd
import numpy as np
import pytest

from dataprism.data.sentinels import replace_sentinel_values_with_nulls
from dataprism.schema import ColumnConfig, ColumnRole, ColumnType, Sentinels


def test_numeric_sentinels_with_empty_strings():
    """
    Test that empty strings are NOT replaced when only numeric sentinels are configured.
    Empty strings are treated as valid values unless explicitly configured as sentinels.
    """
    # Create DataFrame with mixed types: numbers, numeric sentinels, and empty strings
    df = pd.DataFrame({
        'age': [25, -1, '', -9999, 30, '', 35, -1, 40],
        'income': ['50000', '-1', '', '60000', '-9999', '70000', '', '80000', '-1']
    })

    col_configs = {
        'age': ColumnConfig('age', ColumnType.CONTINUOUS, ColumnRole.FEATURE,
                            sentinels=Sentinels(missing='-9999', not_found='-1')),
        'income': ColumnConfig('income', ColumnType.CONTINUOUS, ColumnRole.FEATURE,
                               sentinels=Sentinels(missing='-9999', not_found='-1')),
    }

    # Apply sentinel replacement
    df_clean = replace_sentinel_values_with_nulls(df, col_configs)

    # Verify only numeric sentinels were replaced (empty strings remain)
    assert df_clean['age'].isna().sum() == 3, \
        "Should have 3 nulls in age: 2x -1 + 1x -9999 (empty strings NOT replaced)"

    # Verify only numeric sentinels were replaced in 'income'
    assert df_clean['income'].isna().sum() == 3, \
        "Should have 3 nulls in income: 2x '-1' + 1x '-9999' (empty strings NOT replaced)"

    # Verify empty strings remain as valid values
    assert (df_clean['age'] == '').sum() == 2, "Should have 2 empty strings in age"
    assert (df_clean['income'] == '').sum() == 2, "Should have 2 empty strings in income"

    # Verify other valid values remain
    valid_ages = df_clean['age'].dropna()
    assert len(valid_ages) == 6, "Should have 6 valid age values (including 2 empty strings)"
    assert set(valid_ages) == {25, 30, 35, 40, ''}, "Valid ages should include empty string"

    valid_incomes = df_clean['income'].dropna()
    assert len(valid_incomes) == 6, "Should have 6 valid income values (including 2 empty strings)"


def test_only_empty_strings_no_numeric_sentinels():
    """
    Test that empty strings are NOT replaced when no sentinels are configured.
    Empty strings are treated as valid values.
    """
    df = pd.DataFrame({
        'name': ['Alice', '', 'Bob', '', 'Charlie'],
        'city': ['NYC', 'LA', '', 'SF', '']
    })

    # Config with no sentinels defined
    col_configs = {
        'name': ColumnConfig('name', ColumnType.CATEGORICAL, ColumnRole.FEATURE),
        'city': ColumnConfig('city', ColumnType.CATEGORICAL, ColumnRole.FEATURE),
    }

    df_clean = replace_sentinel_values_with_nulls(df, col_configs)

    # Empty strings should NOT be replaced (treated as valid values)
    assert df_clean['name'].isna().sum() == 0, "Should NOT replace empty strings"
    assert df_clean['city'].isna().sum() == 0, "Should NOT replace empty strings"

    # Verify empty strings remain
    assert (df_clean['name'] == '').sum() == 2, "Should have 2 empty strings in name"
    assert (df_clean['city'] == '').sum() == 2, "Should have 2 empty strings in city"


def test_empty_string_sentinel_explicitly_configured():
    """
    Test when empty string is explicitly configured as sentinel.
    Should not double-count replacements.
    """
    df = pd.DataFrame({
        'status': ['active', '', 'inactive', '', 'pending']
    })

    col_configs = {
        'status': ColumnConfig('status', ColumnType.CATEGORICAL, ColumnRole.FEATURE,
                               sentinels=Sentinels(not_found='')),
    }

    df_clean = replace_sentinel_values_with_nulls(df, col_configs)

    # Should replace 2 empty strings (not double-counted)
    assert df_clean['status'].isna().sum() == 2, "Should replace 2 empty strings"

    valid_statuses = df_clean['status'].dropna()
    assert len(valid_statuses) == 3
    assert set(valid_statuses) == {'active', 'inactive', 'pending'}


def test_all_three_types_mixed():
    """
    Test column with empty string explicitly configured as sentinel along with numeric sentinels.
    """
    df = pd.DataFrame({
        'score': [100, '', -1, 200, -9999, '', 300, -1, 400, -9999]
    })

    # Explicitly configure empty string as a sentinel
    col_configs = {
        'score': ColumnConfig('score', ColumnType.CONTINUOUS, ColumnRole.FEATURE,
                              sentinels=Sentinels(missing='-9999', not_found='')),
    }

    df_clean = replace_sentinel_values_with_nulls(df, col_configs)

    # Should replace: 2 empty strings + 2x -9999 = 4 total
    assert df_clean['score'].isna().sum() == 4, \
        "Should replace 4 values: 2 empty strings + 2x -9999"

    # Valid values should include -1 (not configured as sentinel)
    valid_scores = df_clean['score'].dropna()
    assert len(valid_scores) == 6  # 100, -1, 200, 300, -1, 400
    assert set(valid_scores) == {100, 200, 300, 400, -1}


def test_numeric_column_no_empty_strings():
    """
    Test that numeric columns (dtype int64/float64) work normally
    without empty string handling.
    """
    df = pd.DataFrame({
        'count': [10, 20, -1, 30, -9999, 40]
    })

    # Column is pure numeric (no empty strings)
    assert df['count'].dtype in ['int64', 'float64']

    col_configs = {
        'count': ColumnConfig('count', ColumnType.CONTINUOUS, ColumnRole.FEATURE,
                              sentinels=Sentinels(missing='-9999', not_found='-1')),
    }

    df_clean = replace_sentinel_values_with_nulls(df, col_configs)

    # Should replace only the numeric sentinels
    assert df_clean['count'].isna().sum() == 2, "Should replace -1 and -9999"

    valid_counts = df_clean['count'].dropna()
    assert len(valid_counts) == 4
    assert list(valid_counts) == [10, 20, 30, 40]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
