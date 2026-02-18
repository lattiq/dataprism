"""Test feature filtering logic - only analyze features with provider set."""

import pandas as pd
import numpy as np
import pytest

from dataprism import EDARunner, DataLoader, ColumnConfig, ColumnType, ColumnRole, DatasetSchema


def test_feature_filtering_with_schema():
    """Test that schema controls which features are analyzed."""

    np.random.seed(42)
    df = pd.DataFrame({
        'feature_with_provider': np.random.normal(0, 1, 100),
        'feature_without_provider': np.random.normal(0, 1, 100),
        'target_variable': np.random.choice([0, 1], 100),
    })

    # Schema only includes feature_with_provider as a FEATURE
    schema = DatasetSchema([
        ColumnConfig('feature_with_provider', ColumnType.CONTINUOUS, ColumnRole.FEATURE,
                     provider='provider_a', description='Feature with provider'),
        ColumnConfig('target_variable', ColumnType.BINARY, ColumnRole.TARGET),
    ])

    runner = EDARunner()
    results = runner.run(
        data=df,
        schema=schema,
        target_variable='target_variable'
    )

    analyzed_features = [f['feature_name'] for f in results['features']]

    # Should include feature_with_provider and target_variable
    assert 'feature_with_provider' in analyzed_features
    assert 'target_variable' in analyzed_features

    # Should NOT include feature_without_provider (not in schema features)
    assert 'feature_without_provider' not in analyzed_features

    # Verify target variable has is_target flag
    target_feature = next(f for f in results['features'] if f['feature_name'] == 'target_variable')
    assert target_feature.get('is_target') is True


def test_feature_filtering_with_stability():
    """Test that metadata columns are excluded from analysis but used for stability."""

    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n),
        'dataTag': ['training'] * 100 + ['test'] * 100,
        'target_variable': np.random.choice([0, 1], n),
    })

    schema = DatasetSchema([
        ColumnConfig('feature1', ColumnType.CONTINUOUS, ColumnRole.FEATURE,
                     provider='provider_a'),
        ColumnConfig('dataTag', ColumnType.CATEGORICAL, ColumnRole.SPLIT),
        ColumnConfig('target_variable', ColumnType.BINARY, ColumnRole.TARGET),
    ])

    runner = EDARunner(
        calculate_stability=True,
        cohort_column='dataTag',
        baseline_cohort='training',
        comparison_cohort='test'
    )

    results = runner.run(
        data=df,
        schema=schema,
        target_variable='target_variable'
    )

    analyzed_features = [f['feature_name'] for f in results['features']]

    assert 'feature1' in analyzed_features
    assert 'target_variable' in analyzed_features
    # dataTag is SPLIT role, not FEATURE, so not analyzed
    assert 'dataTag' not in analyzed_features
