"""Analyzer for categorical features."""

from typing import Any, Dict

import pandas as pd
from scipy import stats

from dataprism.analyzers.base import BaseAnalyzer
from dataprism.analyzers.types import CategoricalStats, MissingInfo
from dataprism.schema import ColumnType


class CategoricalAnalyzer(BaseAnalyzer):
    """Analyzer for categorical features."""

    def __init__(self, max_categories: int = 50):
        """
        Initialize categorical analyzer.
        
        Args:
            max_categories: Maximum number of categories to include in value counts
        """
        self.max_categories = max_categories

    @property
    def analyzer_name(self) -> str:
        return "categorical"

    def can_analyze(self, series: pd.Series, column_config=None) -> bool:
        """
        Check if series can be analyzed as categorical.

        Args:
            series: pandas Series to check
            column_config: Optional ColumnConfig object

        Returns:
            True if can analyze as categorical
        """
        if column_config is not None and column_config.type is not None:
            return column_config.is_type(ColumnType.CATEGORICAL, ColumnType.ORDINAL, ColumnType.BINARY)

        # Infer from data when no config or config has no type
        return (
            pd.api.types.is_object_dtype(series) or
            pd.api.types.is_string_dtype(series) or
            isinstance(series.dtype, pd.CategoricalDtype) or
            (pd.api.types.is_numeric_dtype(series) and series.nunique() < 20)
        )

    def _analyze_impl(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze categorical feature."""
        # Remove NaN values for calculations
        clean_series = series.dropna()

        if len(clean_series) == 0:
            return self._empty_stats(len(series))

        # Value counts
        value_counts = clean_series.value_counts()
        total_count = len(clean_series)

        # Limit categories if too many
        if len(value_counts) > self.max_categories:
            top_categories = value_counts.head(self.max_categories)
            other_count = value_counts[self.max_categories:].sum()
            if other_count > 0:
                top_categories['_other_'] = other_count
            value_counts = top_categories

        # Calculate percentages
        value_percentages = {
            str(k): round(v / total_count * 100, 2)
            for k, v in value_counts.items()
        }

        # Convert value counts to regular dict with string keys
        value_counts_dict = {
            str(k): int(v) for k, v in value_counts.items()
        }

        # Basic statistics — derive mode from value_counts (already sorted desc)
        # to ensure mode and mode_count always refer to the same value
        mode_value = value_counts.index[0] if len(value_counts) > 0 else None

        categorical_stats = CategoricalStats(
            count=len(clean_series),
            unique=int(clean_series.nunique()),
            mode=str(mode_value) if mode_value is not None else None,
            mode_count=int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
            value_counts=value_counts_dict,
            value_percentages=value_percentages
        )

        # Missing values
        missing_count = series.isnull().sum()
        missing_info = MissingInfo(
            count=int(missing_count),
            percent=round(missing_count / len(series) * 100, 2) if len(series) > 0 else 0
        )

        # Calculate entropy (Shannon entropy in bits using log2)
        probabilities = value_counts / total_count
        entropy = stats.entropy(probabilities, base=2)

        # Cardinality analysis
        cardinality = clean_series.nunique()
        cardinality_ratio = cardinality / len(clean_series) if len(clean_series) > 0 else 0

        return {
            "type": "categorical",
            "missing": {
                "count": missing_info.count,
                "percent": missing_info.percent
            },
            "stats": {
                "count": categorical_stats.count,
                "unique": categorical_stats.unique,
                "mode": categorical_stats.mode,
                "mode_count": categorical_stats.mode_count,
                "mode_frequency": round(categorical_stats.mode_count / categorical_stats.count * 100, 2),
                "unique_values": cardinality,
                "unique_ratio": round(cardinality_ratio, 4)
            },
            "distribution": {
                "type": "bar",
                "value_counts": categorical_stats.value_counts,
                "value_percentages": categorical_stats.value_percentages,
                "top_n": min(10, len(value_counts_dict)),
                "counts": list(value_counts_dict.values()),
                "percentages": list(value_percentages.values()),
                "entropy": round(float(entropy), 4)
            },
            "cardinality": {
                "unique_values": cardinality,
                "cardinality_ratio": round(cardinality_ratio, 4),
                "is_high_cardinality": cardinality_ratio > 0.5,
                "entropy": round(float(entropy), 4)
            }
        }

    def _empty_stats(self, total_count: int = 0) -> Dict[str, Any]:
        """Return empty statistics for series with no valid values."""
        return {
            "type": "categorical",
            "missing": {"count": total_count, "percent": 100.0 if total_count > 0 else 0.0},
            "stats": {
                "count": 0,
                "unique": 0,
                "mode": None,
                "mode_count": 0,
                "mode_frequency": 0.0,
                "unique_values": 0,
                "unique_ratio": 0.0
            },
            "distribution": {
                "value_counts": {},
                "value_percentages": {},
                "top_n": 0
            },
            "cardinality": {
                "unique_values": 0,
                "cardinality_ratio": 0.0,
                "is_high_cardinality": False,
                "entropy": 0.0
            }
        }
