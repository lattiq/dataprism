"""Test handling of not_found and missing sentinel values."""

import pandas as pd
import pytest

from dataprism.schema import ColumnConfig, ColumnRole, ColumnType, Sentinels
from dataprism.data.sentinels import replace_sentinel_values_with_nulls
from dataprism.analyzers.continuous import ContinuousAnalyzer
from dataprism.analyzers.categorical import CategoricalAnalyzer


def _clean_series(series: pd.Series, config: ColumnConfig) -> pd.Series:
    """Helper: run sentinel replacement and return the cleaned series."""
    df = series.to_frame()
    df_clean = replace_sentinel_values_with_nulls(df, {config.name: config})
    return df_clean[config.name]


class TestDefaultAndNotFoundValues:
    """Test that not_found and missing values are treated as missing."""

    def test_continuous_with_missing_value(self):
        """Test continuous feature with missing sentinel value."""
        data = [1.0, 2.0, 3.0, -999.0, 4.0, 5.0, -999.0]
        series = pd.Series(data, name="test_feature")

        config = ColumnConfig(
            name="test_feature",
            type=ColumnType.CONTINUOUS,
            role=ColumnRole.FEATURE,
            sentinels=Sentinels(missing="-999.0"),
        )

        cleaned = _clean_series(series, config)
        analyzer = ContinuousAnalyzer()
        result = analyzer.analyze(cleaned, column_config=config)

        assert result.data['missing']['count'] == 2
        assert result.data['missing']['percent'] == pytest.approx(28.57, abs=0.1)
        assert result.data['stats']['count'] == 5
        assert result.data['stats']['mean'] == pytest.approx(3.0, abs=0.1)

    def test_continuous_with_not_found_value(self):
        """Test continuous feature with not_found value."""
        data = [10.0, 20.0, 30.0, 0.0, 40.0, 50.0]
        series = pd.Series(data, name="test_feature")

        config = ColumnConfig(
            name="test_feature",
            type=ColumnType.CONTINUOUS,
            role=ColumnRole.FEATURE,
            sentinels=Sentinels(not_found="0"),
        )

        cleaned = _clean_series(series, config)
        analyzer = ContinuousAnalyzer()
        result = analyzer.analyze(cleaned, column_config=config)

        assert result.data['missing']['count'] == 1
        assert result.data['missing']['percent'] == pytest.approx(16.67, abs=0.1)
        assert result.data['stats']['count'] == 5
        assert result.data['stats']['mean'] == pytest.approx(30.0, abs=0.1)

    def test_continuous_with_both_default_and_not_found(self):
        """Test continuous feature with both missing and not_found values."""
        data = [1.0, 2.0, -999.0, 3.0, 0.0, 4.0, -999.0, 5.0]
        series = pd.Series(data, name="test_feature")

        config = ColumnConfig(
            name="test_feature",
            type=ColumnType.CONTINUOUS,
            role=ColumnRole.FEATURE,
            sentinels=Sentinels(missing="-999.0", not_found="0.0"),
        )

        cleaned = _clean_series(series, config)
        analyzer = ContinuousAnalyzer()
        result = analyzer.analyze(cleaned, column_config=config)

        # 3 total: 2x -999, 1x 0
        assert result.data['missing']['count'] == 3
        assert result.data['missing']['percent'] == pytest.approx(37.5, abs=0.1)
        assert result.data['stats']['count'] == 5

    def test_categorical_with_missing_value(self):
        """Test categorical feature with missing sentinel value."""
        data = ["A", "B", "C", "unknown", "A", "B", "unknown", "C"]
        series = pd.Series(data, name="test_category")

        config = ColumnConfig(
            name="test_category",
            type=ColumnType.CATEGORICAL,
            role=ColumnRole.FEATURE,
            sentinels=Sentinels(missing="unknown"),
        )

        cleaned = _clean_series(series, config)
        analyzer = CategoricalAnalyzer()
        result = analyzer.analyze(cleaned, column_config=config)

        assert result.data['missing']['count'] == 2
        assert result.data['missing']['percent'] == pytest.approx(25.0, abs=0.1)
        assert result.data['stats']['count'] == 6
        assert result.data['stats']['unique'] == 3  # A, B, C

    def test_categorical_with_not_found_value(self):
        """Test categorical feature with not_found value."""
        data = ["Red", "Blue", "N/A", "Red", "Green", "N/A", "Blue"]
        series = pd.Series(data, name="test_category")

        config = ColumnConfig(
            name="test_category",
            type=ColumnType.CATEGORICAL,
            role=ColumnRole.FEATURE,
            sentinels=Sentinels(not_found="N/A"),
        )

        cleaned = _clean_series(series, config)
        analyzer = CategoricalAnalyzer()
        result = analyzer.analyze(cleaned, column_config=config)

        assert result.data['missing']['count'] == 2
        assert result.data['missing']['percent'] == pytest.approx(28.57, abs=0.1)
        assert result.data['stats']['count'] == 5
        assert result.data['stats']['unique'] == 3  # Red, Blue, Green

    def test_no_metadata_provided(self):
        """Test that analysis works normally without metadata."""
        data = [1.0, 2.0, 3.0, -999.0, 4.0, 5.0]
        series = pd.Series(data, name="test_feature")

        analyzer = ContinuousAnalyzer()
        result = analyzer.analyze(series, column_config=None)

        # Without config, -999 should NOT be treated as missing
        assert result.data['missing']['count'] == 0
        assert result.data['stats']['count'] == 6
        assert result.data['stats']['mean'] < 0  # Will be negative due to -999

    def test_integer_missing_value(self):
        """Test that integer missing sentinel values work correctly."""
        data = [1, 2, 3, -1, 4, 5, -1, 6]
        series = pd.Series(data, name="test_feature")

        config = ColumnConfig(
            name="test_feature",
            type=ColumnType.CONTINUOUS,
            role=ColumnRole.FEATURE,
            sentinels=Sentinels(missing="-1"),
        )

        cleaned = _clean_series(series, config)
        analyzer = ContinuousAnalyzer()
        result = analyzer.analyze(cleaned, column_config=config)

        assert result.data['missing']['count'] == 2
        assert result.data['stats']['count'] == 6
        assert result.data['stats']['mean'] == pytest.approx(3.5, abs=0.1)
