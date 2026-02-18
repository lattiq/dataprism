"""Basic statistics analyzer for dataset overview."""

from typing import Any, Dict, Optional

import pandas as pd

from dataprism.analyzers.base import BaseAnalyzer
from dataprism.schema import ColumnConfig, ColumnType
from dataprism.analyzers.types import DatasetInfo


class BasicStatsAnalyzer:
    """Analyzer for basic dataset statistics and overview.

    Operates on an entire DataFrame (not a single Series), so it does not
    follow the BaseAnalyzer template-method contract.
    """

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        col_configs: Optional[Dict[str, ColumnConfig]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze entire DataFrame for basic statistics.

        Args:
            df: DataFrame to analyze
            col_configs: Optional dict mapping column names to ColumnConfig

        Returns:
            Dictionary with dataset overview
        """
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()

        dataset_info = DatasetInfo(
            rows=len(df),
            columns=len(df.columns),
            memory_mb=df.memory_usage(deep=True).sum() / 1024 / 1024,
            missing_cells=missing_cells,
            missing_percentage=(missing_cells / total_cells * 100) if total_cells > 0 else 0,
            duplicate_rows=df.duplicated().sum()
        )

        # Analyze column types — use config if available
        col_configs = col_configs or {}
        column_types = {}
        for col in df.columns:
            config = col_configs.get(col)
            column_types[col] = BaseAnalyzer.determine_column_type(df[col], config)

        # Count by type
        type_counts = {
            "continuous": sum(1 for t in column_types.values() if t is ColumnType.CONTINUOUS),
            "categorical": sum(1 for t in column_types.values() if t in (ColumnType.CATEGORICAL, ColumnType.ORDINAL, ColumnType.BINARY)),
        }

        return {
            "dataset_info": {
                "rows": int(dataset_info.rows),
                "columns": int(dataset_info.columns),
                "memory_mb": round(dataset_info.memory_mb, 2),
                "missing_cells": int(dataset_info.missing_cells),
                "missing_percentage": round(dataset_info.missing_percentage, 2),
                "duplicate_rows": int(dataset_info.duplicate_rows)
            },
            "feature_types": type_counts,
            "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()}
        }
