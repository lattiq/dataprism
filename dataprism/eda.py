"""Main EDA orchestrator."""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from dataprism.analyzers.base import BaseAnalyzer
from dataprism.analyzers.basic import BasicStatsAnalyzer
from dataprism.analyzers.match_rates import compute_provider_match_rates
from dataprism.analyzers.stability import StabilityAnalyzer
from dataprism.analyzers.correlation import CorrelationEngine
from dataprism.data.sentinels import replace_sentinel_values_with_nulls
from dataprism.processor import FeatureProcessor
from dataprism.utils.logger import get_logger
from dataprism.schema import ColumnConfig, ColumnType, DatasetSchema
from dataprism.output.formatter import JSONFormatter

# Setup module logger
logger = get_logger(__name__)


class EDARunner:
    """Main orchestrator for EDA pipeline."""

    def __init__(
        self,
        max_categories: int = 50,
        sample_size: Optional[int] = None,
        top_correlations: int = 10,
        max_correlation_features: Optional[int] = None,
        # Cohort-based stability
        calculate_stability: bool = False,
        cohort_column: Optional[str] = None,
        baseline_cohort: Optional[str] = None,
        comparison_cohort: Optional[str] = None,
        # Time-based stability
        time_based_stability: bool = False,
        time_column: Optional[str] = None,
        time_window_strategy: str = 'monthly',
        baseline_period: Union[str, tuple] = 'first',
        comparison_periods: Union[str, List[tuple]] = 'all',
        min_samples_per_period: int = 100
    ):
        self.max_categories = max_categories
        self.sample_size = sample_size
        self.top_correlations = top_correlations
        self.max_correlation_features = max_correlation_features

        # Cohort-based stability
        self.calculate_stability = calculate_stability
        self.cohort_column = cohort_column
        self.baseline_cohort = baseline_cohort
        self.comparison_cohort = comparison_cohort

        # Time-based stability
        self.time_based_stability = time_based_stability
        self.time_column = time_column
        self.time_window_strategy = time_window_strategy
        self.baseline_period = baseline_period
        self.comparison_periods = comparison_periods
        self.min_samples_per_period = min_samples_per_period

        # Initialize analyzers and processors
        self.basic_analyzer = BasicStatsAnalyzer()
        self.stability_analyzer = StabilityAnalyzer()
        self.correlation_engine = CorrelationEngine(top_correlations, max_correlation_features)
        self.feature_processor = FeatureProcessor(max_categories, self.correlation_engine)

        # JSON formatter
        self.formatter = JSONFormatter()

    def run(
        self,
        data: pd.DataFrame,
        schema: Optional[DatasetSchema] = None,
        output_path: Optional[Union[str, Path]] = None,
        compact_json: bool = False,
        columns: Optional[List[str]] = None,
        target_variable: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run complete EDA pipeline.

        Args:
            data: DataFrame to analyze
            schema: Optional DatasetSchema with column configs
            output_path: Optional path to save JSON output
            compact_json: If True, minimize JSON size
            columns: Optional list of columns to analyze (overrides schema)
            target_variable: Name of target variable column

        Returns:
            Complete EDA results as dictionary
        """
        start_time = time.perf_counter()

        # Use provided DataFrame
        logger.info("Running EDA analysis...")
        df = data
        if self.sample_size and len(df) > self.sample_size:
            logger.info("Sampling %s rows from %s total rows", self.sample_size, len(df))
            df = df.head(self.sample_size)

        # Build column configs dict from schema for per-column lookups
        column_configs: Dict[str, ColumnConfig] = {}
        if schema is not None:
            column_configs = schema.columns

        requested_columns = columns
        available_columns = None
        if column_configs and not requested_columns:
            # Use schema feature names when columns not explicitly provided
            requested_columns = schema.get_feature_names()
            logger.info("Found %s features in schema", len(requested_columns))
        elif requested_columns:
            logger.info("Using explicitly provided columns list with %s features", len(requested_columns))

        # Filter columns if specified (either from schema or explicit list)
        if requested_columns:
            available_columns = [col for col in requested_columns if col in df.columns]
            missing_from_df = [col for col in requested_columns if col not in df.columns]
            if missing_from_df:
                logger.warning("Requested columns not found in DataFrame: %s", missing_from_df)

            # Log schema coverage
            if column_configs:
                with_schema = sum(1 for col in available_columns if col in column_configs)
                without_schema = len(available_columns) - with_schema
                logger.info("Schema defined for %s columns, %s columns will use auto-detected types", with_schema, without_schema)

            # Exclude time_column and cohort_column from feature analysis
            if self.cohort_column and self.cohort_column in available_columns:
                available_columns.remove(self.cohort_column)
                logger.info("Excluded cohort column from feature analysis: %s", self.cohort_column)

            if self.time_column and self.time_column in available_columns:
                available_columns.remove(self.time_column)
                logger.info("Excluded time column from feature analysis: %s", self.time_column)

            logger.info("Analyzing %s available features", len(available_columns))

            # Include target variable if specified and not already included
            if target_variable and target_variable in df.columns and target_variable not in available_columns:
                available_columns.append(target_variable)
                logger.info("Added target variable: %s", target_variable)

            # Build list of columns to retain in dataframe (features + metadata columns)
            columns_to_retain = available_columns.copy()

            # Include cohort column for cohort-based stability calculation if needed
            if (
                self.calculate_stability
                and self.cohort_column
                and self.cohort_column in df.columns
                and self.cohort_column not in columns_to_retain
            ):
                columns_to_retain.append(self.cohort_column)
                logger.info("Keeping cohort column for stability calculation: %s", self.cohort_column)

            # Include time column for time-based stability calculation if needed
            if (
                self.time_based_stability
                and self.time_column
                and self.time_column in df.columns
                and self.time_column not in columns_to_retain
            ):
                columns_to_retain.append(self.time_column)
                logger.info("Keeping time column for stability calculation: %s", self.time_column)

            df = df[columns_to_retain]

        # Replace sentinel values with proper nulls
        if column_configs:
            logger.info("\nReplacing sentinel values (not_found, missing) with nulls...")
            df = replace_sentinel_values_with_nulls(df, column_configs)

        # Run basic analysis (on feature columns only, excluding metadata like cohort/time)
        logger.info("\nRunning basic dataset analysis...")
        analysis_df = df[available_columns] if available_columns else df
        basic_analysis = self.basic_analyzer.analyze_dataframe(analysis_df, column_configs)

        # Calculate correlations for continuous features (schema-aware, excludes metadata columns)
        logger.info("Computing feature correlations...")
        correlation_df = df[available_columns] if available_columns else df
        correlation_matrix = self.correlation_engine.compute_correlation_matrix(
            df=correlation_df, target_variable=target_variable, column_configs=column_configs
        )

        # Compute categorical association matrices (Theil's U and Eta)
        logger.info("Computing categorical associations...")
        theils_u_matrix, eta_matrix = self.correlation_engine.compute_association_matrices(
            df=correlation_df, col_configs=column_configs
        )

        # Analyze each feature (only those in available_columns, excluding metadata columns)
        logger.info("Analyzing %s features...", len(available_columns) if available_columns else len(df.columns))
        features = self.feature_processor.analyze_features(
            df=df,
            col_configs=column_configs,
            target_variable=target_variable,
            correlation_matrix=correlation_matrix,
            precomputed_correlations=None,
            columns_to_analyze=available_columns,
            theils_u_matrix=theils_u_matrix,
            eta_matrix=eta_matrix,
        )

        # Calculate stability if requested
        stability_results = None

        # Cohort-based stability
        if self.calculate_stability and self.cohort_column:
            if self.cohort_column in df.columns:
                try:
                    logger.info("\nCalculating cohort-based stability using '%s'...", self.cohort_column)

                    feature_types = {}
                    for name, feature_result in features.items():
                        ftype = feature_result.get('type')
                        if ftype:
                            ftype = ftype.lower()
                        else:
                            col_config = column_configs.get(name)
                            inferred = BaseAnalyzer.determine_column_type(df[name], col_config)
                            ftype = 'continuous' if inferred is ColumnType.CONTINUOUS else 'categorical'
                        feature_types[name] = ftype

                    baseline = self.baseline_cohort or 'training'
                    comparison = self.comparison_cohort or 'test'

                    features_to_analyze = [f for f in features.keys() if f != target_variable]

                    stability_results = self.stability_analyzer.calculate_stability_for_dataset(
                        df=df,
                        cohort_column=self.cohort_column,
                        baseline_cohort=baseline,
                        comparison_cohort=comparison,
                        features_to_analyze=features_to_analyze,
                        feature_types=feature_types
                    )
                except Exception as e:
                    logger.error("Cohort-based stability analysis failed: %s", e, exc_info=True)
            else:
                logger.warning(" Cohort column '%s' not found. Skipping cohort-based stability calculation.", self.cohort_column)

        # Time-based stability
        if self.time_based_stability and self.time_column:
            if self.time_column in df.columns:
                try:
                    logger.info("\nCalculating time-based stability using '%s'...", self.time_column)

                    feature_types = {}
                    for name, feature_result in features.items():
                        ftype = feature_result.get('type')
                        if ftype:
                            ftype = ftype.lower()
                        else:
                            col_config = column_configs.get(name)
                            inferred = BaseAnalyzer.determine_column_type(df[name], col_config)
                            ftype = 'continuous' if inferred is ColumnType.CONTINUOUS else 'categorical'
                        feature_types[name] = ftype

                    features_to_analyze = [
                        f for f in features
                        if f != self.time_column and f != target_variable
                    ]

                    time_stability_results = self.stability_analyzer.calculate_stability_time_based(
                        df=df,
                        time_column=self.time_column,
                        baseline_window=self.baseline_period,
                        comparison_windows=self.comparison_periods,
                        features_to_analyze=features_to_analyze,
                        feature_types=feature_types,
                        window_strategy=self.time_window_strategy,
                        min_samples_per_period=self.min_samples_per_period
                    )

                    if stability_results is None:
                        stability_results = time_stability_results
                    else:
                        stability_results['time_based_analysis'] = time_stability_results
                except Exception as e:
                    logger.error("Time-based stability analysis failed: %s", e, exc_info=True)
            else:
                logger.warning(" Time column '%s' not found. Skipping time-based stability calculation.", self.time_column)

        # Build unified association matrix from the already-computed matrices
        # Only include when <= 25 features to keep output size reasonable
        association_matrix = None
        if len(features) <= 25:
            # Target variable first, then remaining features in original order
            assoc_features = list(features.keys())
            if target_variable and target_variable in assoc_features:
                assoc_features.remove(target_variable)
                assoc_features.insert(0, target_variable)
            association_matrix = self.correlation_engine.build_association_matrix(
                feature_names=assoc_features,
                correlation_matrix=correlation_matrix,
                theils_u_matrix=theils_u_matrix,
                eta_matrix=eta_matrix,
            )

        # Compute provider match rates before formatting
        logger.info("\nComputing provider match rates...")
        provider_match_rates = compute_provider_match_rates(df, column_configs)
        if provider_match_rates:
            logger.info("  Computed match rates for %s providers", len(provider_match_rates))
        else:
            logger.info("  No provider match rates computed (no metadata or record_not_found columns found)")

        # Format results
        execution_time = time.perf_counter() - start_time
        results = self.formatter.format_results(
            dataset_info=basic_analysis['dataset_info'],
            stability_results=stability_results,
            provider_match_rates=provider_match_rates,
            association_matrix=association_matrix,
            features=features,
            metadata={
                "target_variable": target_variable,
                "max_categories": self.max_categories,
                "sample_size": self.sample_size,
                "top_correlations": self.top_correlations,
                "max_correlation_features": self.max_correlation_features,
                "schema_available": bool(column_configs),
                "feature_types": basic_analysis.get('feature_types', {}),
                "correlation_config": {
                    "top_correlations": self.top_correlations,
                    "correlation_threshold": 0.1
                },
                **({
                    "cohort_stability_config": {
                        "cohort_column": self.cohort_column,
                        "baseline_cohort": self.baseline_cohort,
                        "comparison_cohort": self.comparison_cohort
                    }
                } if self.calculate_stability else {}),
                **({
                    "time_stability_config": {
                        "time_window_strategy": self.time_window_strategy,
                        "baseline_period": self.baseline_period,
                        "comparison_periods": self.comparison_periods,
                        "min_samples_per_period": self.min_samples_per_period,
                        "time_column": self.time_column
                    }
                } if self.time_based_stability else {}),
            },
            execution_time=execution_time
        )

        # Save if output path provided
        if output_path:
            logger.info("Saving results to %s...", output_path)
            self.formatter.save_json(results, output_path, compact=compact_json)

        logger.info("EDA completed in %.2f seconds", execution_time)
        return results
