"""Test new EDA schema implementation."""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataprism import EDARunner, DataLoader
from dataprism.schema import (
    ColumnConfig, ColumnRole, ColumnType, DatasetSchema, Sentinels
)


def create_test_dataset():
    """Create a small test dataset with target variable."""
    np.random.seed(42)
    n_samples = 1000

    df = pd.DataFrame({
        'feature_continuous_1': np.random.normal(100, 15, n_samples),
        'feature_continuous_2': np.random.uniform(0, 1, n_samples),
        'feature_categorical_1': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
        'feature_categorical_2': np.random.choice(['Low', 'Medium', 'High'], n_samples),
        'target': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    })

    # Add some missing values
    df.loc[np.random.choice(df.index, 50), 'feature_continuous_1'] = np.nan
    df.loc[np.random.choice(df.index, 30), 'feature_categorical_1'] = np.nan

    # Add some outliers
    df.loc[np.random.choice(df.index, 20), 'feature_continuous_1'] = 500

    return df


def test_new_schema():
    """Test EDA with new schema."""
    print("Creating test dataset...")
    df = create_test_dataset()

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Run EDA directly on DataFrame (not file path)
    print("\nRunning EDA...")
    runner = EDARunner()
    results = runner.run(
        data=df,
        target_variable='target',
    )

    # Verify structure
    print("\nVerifying output structure...")
    assert 'summary' in results, "Missing 'summary' section"
    assert 'features' in results, "Missing 'features' section"
    assert isinstance(results['features'], list), "'features' should be a list"

    # Check summary fields
    summary = results['summary']
    assert summary.get('total_features') == 4, "Should have 4 features (excludes target)"

    print("\nAll tests passed!")


def test_schema_round_trip_with_sentinels():
    """Test DatasetSchema serialization round-trip preserves sentinel values."""
    original = DatasetSchema([
        ColumnConfig(
            name='age',
            type=ColumnType.CONTINUOUS,
            role=ColumnRole.FEATURE,
            description='Customer age',
            provider='bureau',
            sentinels=Sentinels(not_found='-9999', missing='-1'),
        ),
        ColumnConfig(
            name='city',
            type=ColumnType.CATEGORICAL,
            role=ColumnRole.FEATURE,
            provider='kyc',
            sentinels=Sentinels(not_found=''),
        ),
        ColumnConfig(
            name='target',
            type=ColumnType.BINARY,
            role=ColumnRole.TARGET,
        ),
    ])

    # Serialize → deserialize
    data = original.to_dict()
    restored = DatasetSchema.from_dict(data)

    # Same number of columns
    assert len(restored) == len(original)

    # Check sentinel values survived the round trip
    age = restored['age']
    assert age.sentinels is not None
    assert age.sentinels.not_found == '-9999'
    assert age.sentinels.missing == '-1'
    assert age.provider == 'bureau'
    assert age.description == 'Customer age'

    city = restored['city']
    assert city.sentinels is not None
    assert city.sentinels.not_found == ''
    assert city.sentinels.missing is None

    target = restored['target']
    assert target.sentinels is None
    assert target.role is ColumnRole.TARGET
    assert target.type is ColumnType.BINARY


def test_column_config_is_type():
    """Test ColumnConfig.is_type() with single and multiple type arguments."""
    continuous = ColumnConfig('age', ColumnType.CONTINUOUS, ColumnRole.FEATURE)
    categorical = ColumnConfig('city', ColumnType.CATEGORICAL, ColumnRole.FEATURE)
    binary = ColumnConfig('flag', ColumnType.BINARY, ColumnRole.FEATURE)
    ordinal = ColumnConfig('rank', ColumnType.ORDINAL, ColumnRole.FEATURE)

    # Single type match
    assert continuous.is_type(ColumnType.CONTINUOUS) is True
    assert continuous.is_type(ColumnType.CATEGORICAL) is False

    # Multi-type match (as used in CategoricalAnalyzer.can_analyze)
    assert categorical.is_type(ColumnType.CATEGORICAL, ColumnType.ORDINAL, ColumnType.BINARY) is True
    assert binary.is_type(ColumnType.CATEGORICAL, ColumnType.ORDINAL, ColumnType.BINARY) is True
    assert ordinal.is_type(ColumnType.CATEGORICAL, ColumnType.ORDINAL, ColumnType.BINARY) is True
    assert continuous.is_type(ColumnType.CATEGORICAL, ColumnType.ORDINAL, ColumnType.BINARY) is False

    # No types passed
    assert continuous.is_type() is False


if __name__ == '__main__':
    test_new_schema()
    test_schema_round_trip_with_sentinels()
    test_column_config_is_type()
