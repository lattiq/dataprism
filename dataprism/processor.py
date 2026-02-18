"""Feature analysis orchestration — routes each column to the right analyzer."""

from typing import Any, Dict, List, Optional

import pandas as pd

from dataprism.analyzers.categorical import CategoricalAnalyzer
from dataprism.analyzers.continuous import ContinuousAnalyzer
from dataprism.analyzers.target_analysis import TargetAnalyzer
from dataprism.analyzers.base import BaseAnalyzer
from dataprism.analyzers.correlation import CorrelationEngine
from dataprism.utils.logger import get_logger
from dataprism.schema import ColumnConfig, ColumnType
from dataprism.output.formatter import safe_round
from dataprism.output.mapper import ResultMapper

logger = get_logger(__name__)


class FeatureProcessor:
    """Handles feature analysis, quality assessment, and transformation."""

    def __init__(
        self,
        max_categories: int = 50,
        correlation_engine: Optional[CorrelationEngine] = None,
    ):
        self.max_categories = max_categories
        self.correlation_engine = correlation_engine

        # Initialize analyzers
        self.continuous_analyzer = ContinuousAnalyzer()
        self.categorical_analyzer = CategoricalAnalyzer(max_categories)
        self.target_analyzer = TargetAnalyzer()

    def analyze_features(
        self,
        df: pd.DataFrame,
        col_configs: Dict[str, ColumnConfig],
        target_variable: Optional[str] = None,
        correlation_matrix: Optional[pd.DataFrame] = None,
        precomputed_correlations: Optional[Dict[str, Any]] = None,
        columns_to_analyze: Optional[List[str]] = None,
        theils_u_matrix: Optional[pd.DataFrame] = None,
        eta_matrix: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze features in the dataset.

        Args:
            df: DataFrame with all data (including metadata columns)
            col_configs: Dict mapping column names to ColumnConfig
            target_variable: Name of target variable
            correlation_matrix: Precomputed Pearson correlation matrix
            precomputed_correlations: Precomputed correlation data from metadata
            columns_to_analyze: Specific columns to analyze (excludes metadata columns like time/cohort)
            theils_u_matrix: Theil's U matrix for categorical associations
            eta_matrix: Correlation Ratio dict for cat↔cont associations

        Returns:
            Dictionary of analyzed features
        """
        features = {}

        # Determine which columns to analyze
        cols_to_process = columns_to_analyze if columns_to_analyze is not None else df.columns.tolist()

        for col in cols_to_process:
            try:
                # Determine column type — use config if available, otherwise infer
                col_config = col_configs.get(col)
                column_type = self._determine_column_type(col, df, col_configs)

                # Choose appropriate analyzer and analyze
                feature_data = self._analyze_single_feature(col, df, column_type, col_config)

                if feature_data is None:
                    logger.warning("Skipping feature '%s': analyzer returned no results (type=%s)", col, column_type)
                    continue

                # Set actual data type from the column
                feature_data["data_type"] = str(df[col].dtype)

                # Mark if this is the target variable
                if col == target_variable:
                    feature_data["is_target"] = True

                # Add metadata if available
                if col_config is not None:
                    self._add_column_metadata(feature_data, col_config)

                # Add correlations / associations
                if self.correlation_engine:
                    correlations_data = self.correlation_engine.get_feature_correlations(
                        col, correlation_matrix, precomputed_correlations, target_variable,
                        theils_u_matrix=theils_u_matrix,
                        eta_matrix=eta_matrix,
                        column_type=column_type,
                    )
                    feature_data["correlations"] = correlations_data

                # Add target relationship analysis (IV, WoE, enhanced correlations)
                if target_variable and target_variable in df.columns and col != target_variable:
                    try:
                        target_rel = self.target_analyzer.analyze_target_relationship(
                            df[col], df[target_variable], column_type.value,
                            distribution=feature_data.get("distribution")
                        )
                        feature_data["target_relationship"] = target_rel
                    except Exception as e:
                        logger.warning("Target relationship analysis failed for '%s': %s", col, e)
                        feature_data["target_relationship"] = None

                # Add quality flags
                feature_data["quality"] = self._assess_feature_quality(feature_data)

                # Transform to final output schema in a single efficient pass
                features[col] = ResultMapper.map_to_output_schema(
                    feature_data=feature_data,
                    feature_name=col,
                    col_metadata=col_config,
                )
            except Exception as e:
                logger.error("Failed to analyze feature '%s': %s", col, e, exc_info=True)
                continue

        return features

    def _determine_column_type(
        self, col: str, df: pd.DataFrame, col_configs: Dict[str, ColumnConfig]
    ) -> ColumnType:
        """Determine column type from config or infer from data."""
        col_config = col_configs.get(col) if col_configs else None
        return BaseAnalyzer.determine_column_type(df[col], col_config)

    def _analyze_single_feature(
        self,
        col: str,
        df: pd.DataFrame,
        column_type: ColumnType,
        column_config: Optional[ColumnConfig],
    ) -> Optional[Dict[str, Any]]:
        """Analyze a single feature using appropriate analyzer."""
        if column_type is ColumnType.CONTINUOUS:
            analyzer = self.continuous_analyzer
        elif column_type in (ColumnType.CATEGORICAL, ColumnType.ORDINAL, ColumnType.BINARY):
            analyzer = self.categorical_analyzer
        else:
            return None

        if analyzer.can_analyze(df[col], column_config):
            result = analyzer.analyze(df[col], column_config=column_config)
            return result.data

        return None

    def _add_column_metadata(self, feature_data: Dict[str, Any], config: ColumnConfig) -> None:
        """Add metadata fields to feature data from ColumnConfig."""
        feature_data["provider"] = config.provider
        feature_data["description"] = config.description

        if config.sentinels is not None:
            if config.sentinels.missing is not None:
                feature_data["missing_sentinel"] = config.sentinels.missing
            if config.sentinels.not_found is not None:
                feature_data["not_found_sentinel"] = config.sentinels.not_found

    def _assess_feature_quality(self, feature_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess quality flags for a feature."""
        missing_pct = feature_data.get("missing", {}).get("percent", 0)
        outlier_pct = feature_data.get("outliers", {}).get("percent", 0)
        unique_count = feature_data.get("stats", {}).get("unique_values", 0)

        # Check for low variance (for continuous features)
        has_low_variance = False
        if feature_data.get("type") == "continuous":
            stats = feature_data.get("stats", {})
            mean = stats.get("mean")
            std = stats.get("std")
            if mean is not None and std is not None and mean != 0:
                cv = abs(std / mean)
                has_low_variance = cv < 0.01

        is_constant = unique_count == 1

        recommended = not (
            missing_pct > 30
            or has_low_variance
            or is_constant
            or outlier_pct > 50
        )

        return {
            "has_high_missing": missing_pct > 30,
            "has_low_variance": has_low_variance,
            "has_outliers": outlier_pct > 0,
            "outlier_percentage": safe_round(outlier_pct, 2),
            "is_constant": is_constant,
            "recommended_for_modeling": recommended,
        }
