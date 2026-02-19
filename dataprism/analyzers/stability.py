"""Stability analysis for features across cohorts and time periods."""

from datetime import datetime
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from dataprism.utils.logger import get_logger
from dataprism.output.formatter import safe_round

# Setup module logger
logger = get_logger(__name__)


class StabilityAnalyzer:
    """Analyzer for feature stability across cohorts or time periods."""

    def __init__(self, bins: int = 10, min_samples: int = 100):
        """
        Initialize stability analyzer.

        Args:
            bins: Number of bins for continuous variables
            min_samples: Minimum samples required in each cohort
        """
        self.bins = bins
        self.min_samples = min_samples

    def calculate_psi(
        self,
        baseline_series: pd.Series,
        comparison_series: pd.Series,
        feature_type: str
    ) -> Dict[str, Any]:
        """
        Calculate Population Stability Index (PSI).

        PSI = Σ [(Actual% - Expected%) × ln(Actual% / Expected%)]

        PSI Interpretation:
        - PSI < 0.1: No significant change (stable)
        - 0.1 <= PSI < 0.2: Minor shift (monitor)
        - PSI >= 0.2: Major shift (investigate/retrain)

        Args:
            baseline_series: Baseline/reference distribution (e.g., training)
            comparison_series: Comparison distribution (e.g., test/production)
            feature_type: 'continuous' or 'categorical'

        Returns:
            Dictionary with PSI score and interpretation
        """
        # Remove NaN values
        baseline_clean = baseline_series.dropna()
        comparison_clean = comparison_series.dropna()

        # Check minimum sample size
        if len(baseline_clean) < self.min_samples or len(comparison_clean) < self.min_samples:
            return {
                "psi": None,
                "stability": None,
                "interpretation": "insufficient_data",
                "note": f"Requires at least {self.min_samples} samples in each cohort"
            }

        try:
            if feature_type == 'continuous':
                psi, details = self._calculate_psi_continuous(baseline_clean, comparison_clean)
            else:  # categorical
                psi, details = self._calculate_psi_categorical(baseline_clean, comparison_clean)

            # Interpret PSI
            if psi < 0.1:
                stability = "stable"
                interpretation = "No significant distribution change"
            elif psi < 0.2:
                stability = "minor_shift"
                interpretation = "Minor distribution shift, monitor closely"
            else:
                stability = "major_shift"
                interpretation = "Major distribution shift, investigate or retrain"

            return {
                "psi": safe_round(psi, 4),
                "stability": stability,
                "interpretation": interpretation,
                "details": details
            }

        except (ValueError, TypeError, FloatingPointError, ZeroDivisionError) as e:
            return {
                "psi": None,
                "stability": None,
                "interpretation": "error",
                "note": f"Error calculating PSI: {str(e)}"
            }

    def _calculate_psi_continuous(
        self,
        baseline: pd.Series,
        comparison: pd.Series
    ) -> tuple:
        """Calculate PSI for continuous features."""
        # Handle constant-value features
        if baseline.nunique() <= 1:
            # Constant baseline — PSI is 0 if comparison is also constant with same value
            if comparison.nunique() <= 1 and baseline.iloc[0] == comparison.iloc[0]:
                return 0.0, {"note": "constant_distribution", "num_bins": 1}
            # Otherwise treat as categorical (single bin)
            return self._calculate_psi_categorical(baseline, comparison)

        # Create bins from baseline distribution
        try:
            # Use quantile-based binning
            bin_edges = np.percentile(baseline, np.linspace(0, 100, self.bins + 1))
            bin_edges = np.unique(bin_edges)  # Remove duplicates
        except (ValueError, IndexError):
            # Fallback to equal-width binning
            bin_edges = np.linspace(baseline.min(), baseline.max(), self.bins + 1)

        # Extend edges to -inf/+inf to capture out-of-range comparison values
        bin_edges = np.concatenate([[-np.inf], bin_edges[1:-1], [np.inf]])

        # Get distributions
        baseline_counts, _ = np.histogram(baseline, bins=bin_edges)
        comparison_counts, _ = np.histogram(comparison, bins=bin_edges)

        # Calculate PSI
        psi, details = self._calculate_psi_from_counts(
            baseline_counts, comparison_counts, bin_edges
        )

        return psi, details

    def _calculate_psi_categorical(
        self,
        baseline: pd.Series,
        comparison: pd.Series
    ) -> tuple:
        """Calculate PSI for categorical features."""
        # Get all unique categories from both distributions
        all_categories = sorted(set(baseline.unique()) | set(comparison.unique()))

        # Get value counts
        baseline_counts = baseline.value_counts()
        comparison_counts = comparison.value_counts()

        # Create aligned count arrays - explicitly convert to int to avoid dtype issues
        baseline_array = np.array([baseline_counts.get(cat, 0) for cat in all_categories], dtype=np.int64)
        comparison_array = np.array([comparison_counts.get(cat, 0) for cat in all_categories], dtype=np.int64)

        # Calculate PSI
        psi, details = self._calculate_psi_from_counts(
            baseline_array, comparison_array, all_categories
        )

        return psi, details

    def _calculate_psi_from_counts(
        self,
        baseline_counts: np.ndarray,
        comparison_counts: np.ndarray,
        labels: Any
    ) -> tuple:
        """Calculate PSI from count arrays."""
        # Ensure counts are numeric
        baseline_counts = np.asarray(baseline_counts, dtype=np.float64)
        comparison_counts = np.asarray(comparison_counts, dtype=np.float64)

        # Normalize to proportions, replacing zero bins with epsilon to avoid log(0)
        eps = 1e-4
        baseline_total = baseline_counts.sum()
        comparison_total = comparison_counts.sum()
        if baseline_total == 0 or comparison_total == 0:
            return 0.0, {"note": "empty_distribution", "num_bins": len(baseline_counts)}
        baseline_pct = baseline_counts / baseline_total
        comparison_pct = comparison_counts / comparison_total
        baseline_pct = np.where(baseline_pct == 0, eps, baseline_pct)
        comparison_pct = np.where(comparison_pct == 0, eps, comparison_pct)

        # Calculate PSI
        psi_components = (comparison_pct - baseline_pct) * np.log(comparison_pct / baseline_pct)
        psi = np.sum(psi_components)

        # Create details
        details = {
            "baseline_distribution": baseline_pct.tolist(),
            "comparison_distribution": comparison_pct.tolist(),
            "psi_components": [safe_round(x, 4) for x in psi_components],
            "num_bins": len(baseline_counts)
        }

        return abs(psi), details

    def calculate_stability_for_dataset(
        self,
        df: pd.DataFrame,
        cohort_column: str,
        baseline_cohort: str,
        comparison_cohort: str,
        features_to_analyze: list,
        feature_types: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Calculate stability for multiple features across cohorts.

        Args:
            df: DataFrame with data
            cohort_column: Column name for cohort identification (e.g., 'dataTag')
            baseline_cohort: Baseline cohort value (e.g., 'training')
            comparison_cohort: Comparison cohort value (e.g., 'test')
            features_to_analyze: List of feature names
            feature_types: Dict mapping feature names to types ('continuous'/'categorical')

        Returns:
            Dictionary with stability metrics for all features
        """
        baseline_df = df[df[cohort_column] == baseline_cohort]
        comparison_df = df[df[cohort_column] == comparison_cohort]

        logger.info("\nCalculating stability metrics:")
        logger.info("  Baseline cohort '%s': %s samples", baseline_cohort, len(baseline_df))
        logger.info("  Comparison cohort '%s': %s samples", comparison_cohort, len(comparison_df))

        results = {}
        stable_count = 0
        minor_shift_count = 0
        major_shift_count = 0

        for feature in features_to_analyze:
            if feature not in df.columns or feature == cohort_column:
                continue

            feature_type = feature_types.get(feature, 'continuous')

            psi_result = self.calculate_psi(
                baseline_df[feature],
                comparison_df[feature],
                feature_type
            )

            results[feature] = psi_result

            # Count stability categories
            if psi_result['stability'] == 'stable':
                stable_count += 1
            elif psi_result['stability'] == 'minor_shift':
                minor_shift_count += 1
            elif psi_result['stability'] == 'major_shift':
                major_shift_count += 1

        # Summary
        summary = {
            "total_features_analyzed": len(results),
            "stable_features": stable_count,
            "minor_shift_features": minor_shift_count,
            "major_shift_features": major_shift_count,
            "baseline_cohort": baseline_cohort,
            "comparison_cohort": comparison_cohort,
            "baseline_samples": len(baseline_df),
            "comparison_samples": len(comparison_df)
        }

        logger.info("  Stable features: %s", stable_count)
        logger.info("  Minor shift: %s", minor_shift_count)
        logger.info("  Major shift: %s", major_shift_count)

        return {
            "summary": summary,
            "feature_stability": results
        }

    def calculate_stability_time_based(
        self,
        df: pd.DataFrame,
        time_column: str,
        baseline_window: Union[str, Tuple[str, str]],
        comparison_windows: Union[str, List[Tuple[str, str]]],
        features_to_analyze: List[str],
        feature_types: Dict[str, str],
        window_strategy: str = 'monthly',
        min_samples_per_period: int = 100
    ) -> Dict[str, Any]:
        """
        Calculate stability across time windows.

        Args:
            df: DataFrame with data
            time_column: Column name with timestamps
            baseline_window: 'first_month', 'first_week', 'first_quartile', or tuple ('start', 'end')
            comparison_windows: 'all', 'auto', or list of tuples [('start1', 'end1'), ...]
            features_to_analyze: List of feature names
            feature_types: Dict mapping feature names to types
            window_strategy: 'monthly', 'weekly', 'quartiles', 'custom'
            min_samples_per_period: Minimum samples required per period

        Returns:
            Dictionary with temporal stability metrics
        """
        # Convert time column to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
            df = df.copy()
            df[time_column] = pd.to_datetime(df[time_column], errors='coerce')

        # Auto-detect windows if needed
        if isinstance(baseline_window, str) or comparison_windows == 'all' or comparison_windows == 'auto':
            baseline_window, comparison_windows = self._auto_detect_time_windows(
                df, time_column, window_strategy, baseline_window, comparison_windows, min_samples_per_period
            )

        # Get baseline data
        baseline_df = self._filter_by_time_window(df, time_column, baseline_window)

        if len(baseline_df) < min_samples_per_period:
            return {
                "error": "insufficient_baseline_data",
                "note": f"Baseline period has {len(baseline_df)} samples, requires at least {min_samples_per_period}"
            }

        logger.info("\nCalculating time-based stability:")
        logger.info("  Baseline period: %s to %s (%s samples)", baseline_window[0], baseline_window[1], len(baseline_df))

        # Calculate stability for each comparison window
        period_results = []
        feature_stability_by_period = {feature: [] for feature in features_to_analyze}

        for i, comp_window in enumerate(comparison_windows, 1):
            comp_df = self._filter_by_time_window(df, time_column, comp_window)

            if len(comp_df) < min_samples_per_period:
                logger.info("  Period %s: Skipping (only %s samples)", i, len(comp_df))
                continue

            logger.info("  Period %s: %s to %s (%s samples)", i, comp_window[0], comp_window[1], len(comp_df))

            period_stability = {}
            for feature in features_to_analyze:
                if feature not in df.columns or feature == time_column:
                    continue

                feature_type = feature_types.get(feature, 'continuous')
                psi_result = self.calculate_psi(
                    baseline_df[feature],
                    comp_df[feature],
                    feature_type
                )

                period_stability[feature] = psi_result
                feature_stability_by_period[feature].append(psi_result['psi'])

            period_results.append({
                "period_id": i,
                "start": str(comp_window[0]),
                "end": str(comp_window[1]),
                "sample_count": len(comp_df),
                "stability": period_stability
            })

        # Compute temporal trends for each feature
        feature_temporal_analysis = {}
        for feature, psi_values in feature_stability_by_period.items():
            if not any(v is not None for v in psi_values):
                continue

            analysis = self._analyze_temporal_trend(psi_values)
            feature_temporal_analysis[feature] = analysis

        # Summary statistics
        summary = self._compute_temporal_summary(period_results, feature_temporal_analysis)

        return {
            "method": "time_based",
            "time_column": time_column,
            "window_strategy": window_strategy,
            "baseline_period": {
                "start": str(baseline_window[0]),
                "end": str(baseline_window[1]),
                "sample_count": len(baseline_df)
            },
            "comparison_periods": [
                {
                    "period_id": p["period_id"],
                    "start": p["start"],
                    "end": p["end"],
                    "sample_count": p["sample_count"]
                }
                for p in period_results
            ],
            "feature_stability": feature_temporal_analysis,
            "summary": summary
        }

    def _auto_detect_time_windows(
        self,
        df: pd.DataFrame,
        time_column: str,
        strategy: str,
        baseline_spec: str,
        comparison_spec: str,
        min_samples: int
    ) -> Tuple[Tuple[datetime, datetime], List[Tuple[datetime, datetime]]]:
        """Auto-detect optimal time windows based on strategy."""
        time_series = df[time_column].dropna()
        min_date = time_series.min()
        max_date = time_series.max()

        if strategy == 'monthly':
            windows = self._create_monthly_windows(df, time_column, min_date, max_date, min_samples)
        elif strategy == 'weekly':
            windows = self._create_weekly_windows(df, time_column, min_date, max_date, min_samples)
        elif strategy == 'quartiles':
            windows = self._create_quartile_windows(df, time_column, min_samples)
        else:  # custom - user should provide tuples
            raise ValueError(f"Custom strategy requires explicit window tuples, not '{baseline_spec}' and '{comparison_spec}'")

        if len(windows) < 2:
            raise ValueError(f"Insufficient time windows detected. Need at least 2, found {len(windows)}")

        # Baseline is first window
        baseline_window = windows[0]

        # Comparison windows are the rest
        comparison_windows = windows[1:]

        return baseline_window, comparison_windows

    def _create_monthly_windows(
        self,
        df: pd.DataFrame,
        time_column: str,
        min_date: datetime,
        max_date: datetime,
        min_samples: int
    ) -> List[Tuple[datetime, datetime]]:
        """Create monthly time windows."""
        windows = []
        current_date = min_date.replace(day=1)

        while current_date <= max_date:
            # Get next month
            if current_date.month == 12:
                next_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                next_date = current_date.replace(month=current_date.month + 1)

            # Check sample count
            mask = (df[time_column] >= current_date) & (df[time_column] < next_date)
            count = mask.sum()

            if count >= min_samples:
                windows.append((current_date, next_date))

            current_date = next_date

        return windows

    def _create_weekly_windows(
        self,
        df: pd.DataFrame,
        time_column: str,
        min_date: datetime,
        max_date: datetime,
        min_samples: int
    ) -> List[Tuple[datetime, datetime]]:
        """Create weekly time windows."""
        windows = []
        current_date = min_date

        while current_date <= max_date:
            next_date = current_date + pd.Timedelta(days=7)

            # Check sample count
            mask = (df[time_column] >= current_date) & (df[time_column] < next_date)
            count = mask.sum()

            if count >= min_samples:
                windows.append((current_date, next_date))

            current_date = next_date

        return windows

    def _create_quartile_windows(
        self,
        df: pd.DataFrame,
        time_column: str,
        min_samples: int
    ) -> List[Tuple[datetime, datetime]]:
        """Create quartile-based time windows (equal sample sizes)."""
        time_series = df[time_column].dropna().sort_values()

        # Split into 4 equal parts
        quartile_size = len(time_series) // 4

        if quartile_size < min_samples:
            raise ValueError(f"Quartile size {quartile_size} is less than minimum {min_samples}")

        windows = []
        for i in range(4):
            start_idx = i * quartile_size
            start_date = time_series.iloc[start_idx]

            if i < 3:
                # Use start of next quartile as exclusive end boundary
                end_date = time_series.iloc[(i + 1) * quartile_size]
            else:
                # Last window: include the final observation by nudging past it
                end_date = time_series.iloc[-1] + pd.Timedelta(milliseconds=1)

            windows.append((start_date, end_date))

        return windows

    def _filter_by_time_window(
        self,
        df: pd.DataFrame,
        time_column: str,
        window: Tuple[Union[str, datetime], Union[str, datetime]]
    ) -> pd.DataFrame:
        """Filter DataFrame by time window."""
        start, end = window

        # Convert strings to datetime if needed
        if isinstance(start, str):
            start = pd.to_datetime(start)
        if isinstance(end, str):
            end = pd.to_datetime(end)

        mask = (df[time_column] >= start) & (df[time_column] < end)
        return df[mask]

    def _analyze_temporal_trend(self, psi_values: List) -> Dict[str, Any]:
        """Analyze temporal trend in PSI values.

        psi_values may contain None entries for periods where a feature
        had insufficient samples.  We keep the full list (with Nones) in
        ``psi_by_period`` so it stays index-aligned with comparison_periods,
        but compute stats only on valid (non-None) values.
        """
        valid = [v for v in psi_values if v is not None]
        if len(valid) == 0:
            return {"trend": "unknown", "note": "No PSI values available"}

        avg_psi = np.mean(valid)
        max_psi = np.max(valid)
        min_psi = np.min(valid)

        # Determine trend
        if len(valid) >= 2:
            # Simple linear trend: compare first half vs second half
            mid = len(valid) // 2
            first_half_avg = np.mean(valid[:mid])
            second_half_avg = np.mean(valid[mid:])

            if second_half_avg > first_half_avg * 1.5:
                trend = "increasing_drift"
            elif second_half_avg < first_half_avg * 0.67:
                trend = "decreasing_drift"
            elif np.std(valid) > avg_psi * 0.5:
                trend = "volatile"
            else:
                trend = "stable"
        else:
            trend = "stable" if avg_psi < 0.1 else "drifting"

        # Count stable/drifting periods
        stable_periods = sum(1 for p in valid if p < 0.1)
        minor_shift_periods = sum(1 for p in valid if 0.1 <= p < 0.2)
        major_shift_periods = sum(1 for p in valid if p >= 0.2)

        # Overall stability classification
        if avg_psi < 0.1:
            stability = "stable"
        elif avg_psi < 0.2:
            stability = "minor_shift"
        else:
            stability = "major_shift"

        return {
            "psi_by_period": [safe_round(p, 4) if p is not None else None for p in psi_values],
            "avg_psi": safe_round(avg_psi, 4),
            "max_psi": safe_round(max_psi, 4),
            "min_psi": safe_round(min_psi, 4),
            "stability": stability,
            "trend": trend,
            "stable_periods": stable_periods,
            "minor_shift_periods": minor_shift_periods,
            "major_shift_periods": major_shift_periods
        }

    def _compute_temporal_summary(
        self,
        period_results: List[Dict],
        feature_analysis: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """Compute summary statistics for temporal stability."""
        total_features = len(feature_analysis)

        stable_features = sum(1 for f in feature_analysis.values() if f.get('stability') == 'stable')
        minor_shift_features = sum(1 for f in feature_analysis.values() if f.get('stability') == 'minor_shift')
        major_shift_features = sum(1 for f in feature_analysis.values() if f.get('stability') == 'major_shift')

        # Find features with increasing drift
        increasing_drift = [
            fname for fname, data in feature_analysis.items()
            if data.get('trend') == 'increasing_drift'
        ]

        # Find worst drift period (period with most major shifts)
        worst_period = None
        max_major_shifts = 0
        for period in period_results:
            major_shifts_in_period = sum(
                1 for f, psi in period.get('stability', {}).items()
                if psi.get('psi') is not None and psi.get('psi') >= 0.2
            )
            if major_shifts_in_period > max_major_shifts:
                max_major_shifts = major_shifts_in_period
                worst_period = period['period_id']

        return {
            "total_features_analyzed": total_features,
            "stable_features": stable_features,
            "minor_drift_features": minor_shift_features,
            "major_drift_features": major_shift_features,
            "features_with_increasing_drift": increasing_drift[:10],  # Top 10
            "worst_drift_period": worst_period,
            "total_periods_analyzed": len(period_results)
        }
