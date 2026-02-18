"""Base analyzer interface and common functionality."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd

from dataprism.schema import ColumnConfig, ColumnType


@dataclass
class AnalysisResult:
    """Container for analysis results with metadata."""
    analyzer_name: str
    column_name: Optional[str]
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "analyzer": self.analyzer_name,
            "column": self.column_name,
            "data": self.data,
            "metadata": self.metadata,
            "execution_time": self.execution_time
        }


class BaseAnalyzer(ABC):
    """Abstract base class for all data analyzers."""

    @property
    @abstractmethod
    def analyzer_name(self) -> str:
        """Return the name of this analyzer."""
        pass

    @abstractmethod
    def can_analyze(
        self, series: pd.Series, column_config: Optional[ColumnConfig] = None
    ) -> bool:
        """Check if this analyzer can process the given series."""
        pass

    @abstractmethod
    def _analyze_impl(self, series: pd.Series) -> Dict[str, Any]:
        """Core analysis implementation."""
        pass

    def analyze(self, series: pd.Series, column_config: Optional[ColumnConfig] = None) -> AnalysisResult:
        """
        Main analysis method implementing the template pattern.

        Args:
            series: Pandas series to analyze
            column_config: Optional column config containing sentinel values

        Returns:
            Analysis result with processed data
        """
        start_time = time.perf_counter()

        if not self.can_analyze(series, column_config):
            raise ValueError(f"{self.analyzer_name} cannot analyze series '{series.name}'")

        results = self._analyze_impl(series)


        execution_time = time.perf_counter() - start_time

        return AnalysisResult(
            analyzer_name=self.analyzer_name,
            column_name=series.name,
            data=results,
            metadata={
                "dtype": str(series.dtype),
                "size": len(series),
                "null_count": series.isnull().sum(),
            },
            execution_time=execution_time,
        )

    @staticmethod
    def determine_column_type(
        series: pd.Series,
        column_config: Optional[ColumnConfig] = None
    ) -> ColumnType:
        """
        Determine column type from config or infer from data.

        If config has a type, return it directly.
        Otherwise fall through to infer_column_type().
        """
        if column_config is not None and column_config.type is not None:
            return column_config.type

        return BaseAnalyzer.infer_column_type(series)

    @staticmethod
    def infer_column_type(series: pd.Series) -> ColumnType:
        """
        Infer the column type from pandas series data characteristics.

        Numeric with low cardinality -> CATEGORICAL
        Numeric -> CONTINUOUS
        Everything else -> CATEGORICAL
        """
        if len(series) == 0:
            return ColumnType.CATEGORICAL

        if pd.api.types.is_numeric_dtype(series):
            non_null_count = series.count()
            if non_null_count == 0:
                return ColumnType.CATEGORICAL
            unique_ratio = series.nunique() / non_null_count
            if unique_ratio < 0.05 and series.nunique() < 20:
                return ColumnType.CATEGORICAL
            return ColumnType.CONTINUOUS
        else:
            return ColumnType.CATEGORICAL
