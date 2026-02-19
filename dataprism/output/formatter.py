"""JSON output formatter for EDA results."""

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np


def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively sanitize data structure for JSON serialization.
    Converts NaN, Infinity, and -Infinity to None (null in JSON).

    Args:
        obj: Object to sanitize

    Returns:
        Sanitized object safe for JSON serialization
    """

    if isinstance(obj, dict):
        return {key: sanitize_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif hasattr(obj, 'isoformat'):
        # Handle pd.Timestamp, datetime, date objects
        return obj.isoformat()
    else:
        return obj


def safe_round(value: Any, decimals: int = 4) -> Optional[float]:
    """
    Safely round a numeric value, returning None for NaN or Infinity.

    Args:
        value: Value to round
        decimals: Number of decimal places

    Returns:
        Rounded value or None if value is NaN/Infinity
    """
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            if math.isnan(value) or math.isinf(value):
                return None
            return round(float(value), decimals)
        return value
    except (ValueError, TypeError):
        return None


class SafeJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that converts NaN and Infinity to null."""

    def default(self, obj):
        """Handle objects that can't be serialized."""
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
        return super().default(obj)

    def encode(self, obj):
        """Encode object after sanitizing."""
        # Sanitize the entire object tree first
        sanitized = sanitize_for_json(obj)
        return super().encode(sanitized)

    def iterencode(self, obj, _one_shot=False):
        """Iterate over encoded chunks after sanitizing."""
        # Sanitize the entire object tree first
        sanitized = sanitize_for_json(obj)
        return super().iterencode(sanitized, _one_shot)


class JSONFormatter:
    """Formats EDA results for JSON output."""

    @staticmethod
    def format_results(
        dataset_info: Dict[str, Any],
        features: Dict[str, Any],
        stability_results: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        execution_time: float = 0.0,
        provider_match_rates: Optional[Dict[str, Any]] = None,
        association_matrix: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Format EDA results into structured JSON with 3 main sections:
        - metadata: execution metadata and configuration
        - summary: all aggregate statistics and metrics
        - features: individual feature-level analysis

        Args:
            dataset_info: Basic dataset information
            features: Feature analysis results
            correlations: Correlation analysis results (deprecated - embedded in features)
            stability_results: Stability analysis results (PSI/CSI)
            metadata: Additional metadata
            execution_time: Total execution time
            provider_match_rates: Provider match rate statistics

        Returns:
            Formatted dictionary ready for JSON serialization
        """
        # Convert features dict to list and add feature_name
        features_list = []
        for name, data in features.items():
            # Remove metadata subsection from individual features
            feature_data = data.copy()
            if 'metadata' in feature_data:
                del feature_data['metadata']

            feature_data['feature_name'] = name

            # Merge stability data into feature if available
            if stability_results:
                stability_data = {}
                is_time_only = stability_results.get('method') == 'time_based'

                # Add cohort-based stability if available (only key fields)
                if not is_time_only and 'feature_stability' in stability_results:
                    cohort_stability = stability_results['feature_stability'].get(name)
                    if cohort_stability:
                        stability_data['cohort_based'] = {
                            'psi': cohort_stability.get('psi'),
                            'stability': cohort_stability.get('stability'),
                            'interpretation': cohort_stability.get('interpretation')
                        }

                # Add time-based stability if available (only key fields)
                time_source = None
                if is_time_only and 'feature_stability' in stability_results:
                    time_source = stability_results['feature_stability']
                elif 'time_based_analysis' in stability_results:
                    tba = stability_results['time_based_analysis']
                    if 'feature_stability' in tba:
                        time_source = tba['feature_stability']

                if time_source:
                    time_stability = time_source.get(name)
                    if time_stability:
                        stability_data['time_based'] = {
                            'psi': time_stability.get('avg_psi'),
                            'max_psi': time_stability.get('max_psi'),
                            'min_psi': time_stability.get('min_psi'),
                            'stability': time_stability.get('stability'),
                            'trend': time_stability.get('trend'),
                            'psi_by_period': time_stability.get('psi_by_period'),
                            'interpretation': JSONFormatter._time_stability_interpretation(time_stability)
                        }

                # If we have any stability data, add it to the feature
                if stability_data:
                    feature_data['stability'] = stability_data

            features_list.append(feature_data)

        # Compute summary metrics (derived insights only)
        summary_stats = JSONFormatter._compute_summary(features_list, metadata)

        # Compute highest metrics (with stability if available)
        highest_metrics = JSONFormatter._compute_highest_metrics(features_list, stability_results)

        # Compute top 10 features by statistical score
        top_features_by_score = JSONFormatter._compute_top_features_by_statistical_score(features_list, stability_results)

        # Compute data quality summary
        data_quality = JSONFormatter._compute_data_quality(dataset_info, features_list)

        # Compute feature counts for dashboard
        target_variable = metadata.get("target_variable") if metadata else None
        feature_counts = JSONFormatter._compute_feature_counts(features_list, stability_results, target_variable)

        # Build metadata section with execution info
        metadata_section = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "execution_time_seconds": round(execution_time, 2),
        }
        if metadata:
            metadata_section.update(metadata)

        # Build summary section with structured subsections
        summary_section = {
            **summary_stats,  # Derived insights (total_features, avg_missing_percentage, etc.)
            "feature_counts": feature_counts,
            "highest_metrics": highest_metrics,
            "top_features_by_statistical_score": top_features_by_score,
            "data_quality": data_quality,
            "dataset_info": dataset_info  # Raw dataset statistics (single source of truth)
        }

        # Add provider_match_rates to summary if available
        if provider_match_rates:
            summary_section["provider_match_rates"] = provider_match_rates

        # Add unified association matrix if available
        if association_matrix:
            summary_section["association_matrix"] = association_matrix

        # Add stability period metadata for viewer time-series charts
        if stability_results:
            stability_summary = {'time_based_available': False}
            time_src = (
                stability_results if stability_results.get('method') == 'time_based'
                else stability_results.get('time_based_analysis')
            )
            if time_src:
                periods = time_src.get('comparison_periods', [])
                stability_summary['time_based_available'] = True
                stability_summary['time_periods'] = [
                    {'label': p.get('start', ''), 'end': p.get('end', '')}
                    for p in periods
                ]
            summary_section['stability_analysis'] = stability_summary

        # Build output with only 3 sections
        output = {
            "metadata": metadata_section,
            "summary": summary_section,
            "features": features_list
        }

        # Sanitize all data to ensure valid JSON
        return sanitize_for_json(output)

    @staticmethod
    def save_json(
        data: Dict[str, Any],
        filepath: Union[str, Path],
        indent: int = 2,
        compact: bool = False
    ) -> None:
        """
        Save formatted data to JSON file.

        Args:
            data: Data to save
            filepath: Output file path
            indent: JSON indentation (None for compact)
            compact: If True, minimize JSON size
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            if compact:
                json.dump(data, f, separators=(',', ':'), cls=SafeJSONEncoder)
            else:
                json.dump(data, f, indent=indent, cls=SafeJSONEncoder)

    @staticmethod
    def to_json_string(
        data: Dict[str, Any],
        indent: Optional[int] = 2,
        compact: bool = False
    ) -> str:
        """
        Convert data to JSON string.

        Args:
            data: Data to convert
            indent: JSON indentation
            compact: If True, minimize JSON size

        Returns:
            JSON string
        """
        if compact:
            return json.dumps(data, separators=(',', ':'), cls=SafeJSONEncoder)
        else:
            return json.dumps(data, indent=indent, cls=SafeJSONEncoder)

    @staticmethod
    def compress_features(features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compress feature data for efficient storage.
        
        Removes redundant information and optimizes structure.
        
        Args:
            features: Feature analysis results
            
        Returns:
            Compressed feature data
        """
        compressed = {}

        for name, data in features.items():
            if data.get('type') == 'continuous':
                compressed[name] = {
                    't': 'c',  # type: continuous
                    'm': data.get('missing', {}),  # missing
                    's': {  # stats (abbreviated keys)
                        'mean': data['stats'].get('mean'),
                        'std': data['stats'].get('std'),
                        'min': data['stats'].get('min'),
                        'max': data['stats'].get('max'),
                        'med': data['stats'].get('median')
                    },
                    'o': data.get('outliers', {}).get('count', 0)  # outlier count
                }
            elif data.get('type') == 'categorical':
                compressed[name] = {
                    't': 'cat',  # type: categorical
                    'm': data.get('missing', {}),  # missing
                    'u': data['stats'].get('unique'),  # unique values
                    'mode': data['stats'].get('mode'),
                    'top5': dict(list(data['distribution']['value_counts'].items())[:5])
                }

        return compressed

    @staticmethod
    def _time_stability_interpretation(ts: Dict[str, Any]) -> str:
        """Build a human-readable interpretation for time-based stability."""
        stability = ts.get('stability', 'unknown')
        trend = ts.get('trend', 'unknown')
        avg_psi = ts.get('avg_psi')

        trend_labels = {
            'stable': 'stable trend',
            'increasing_drift': 'increasing drift over time',
            'decreasing_drift': 'decreasing drift over time',
            'volatile': 'volatile across periods',
            'drifting': 'drifting',
        }
        trend_text = trend_labels.get(trend, trend)
        psi_str = f"{avg_psi:.4f}" if avg_psi is not None else "N/A"

        if stability == 'stable':
            return f"Stable distribution (avg PSI {psi_str}), {trend_text}"
        elif stability == 'minor_shift':
            return f"Minor distribution shift (avg PSI {psi_str}), {trend_text}"
        else:
            return f"Major distribution shift (avg PSI {psi_str}), {trend_text}"

    @staticmethod
    def _get_feature_stability_map(stability_results: Optional[Dict[str, Any]]) -> Dict[str, Dict]:
        """Extract a flat feature_name -> stability dict from stability_results.

        Works for cohort-only, time-only, and combined stability.
        For time-based, maps avg_psi to 'psi' for uniform access.
        """
        psi_map: Dict[str, Dict] = {}
        if not stability_results:
            return psi_map

        is_time_only = stability_results.get('method') == 'time_based'

        # Cohort-based stability (top-level feature_stability when not time-only)
        if not is_time_only and 'feature_stability' in stability_results:
            for fname, data in stability_results['feature_stability'].items():
                psi_map[fname] = data

        # Time-based stability
        time_source = None
        if is_time_only and 'feature_stability' in stability_results:
            time_source = stability_results['feature_stability']
        elif 'time_based_analysis' in stability_results:
            tba = stability_results['time_based_analysis']
            if 'feature_stability' in tba:
                time_source = tba['feature_stability']

        if time_source:
            for fname, data in time_source.items():
                if fname not in psi_map:
                    # Normalize: use avg_psi as psi for uniform access
                    psi_map[fname] = {**data, 'psi': data.get('avg_psi')}
                # If cohort already present, don't overwrite

        return psi_map

    @staticmethod
    def _compute_summary(features: list, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute summary metrics - DERIVED INSIGHTS ONLY.

        Raw dataset statistics are in dataset_info subsection.
        Feature counts are in feature_counts subsection.
        Execution metadata is in metadata section.

        Args:
            features: List of analyzed features
            metadata: Optional metadata dictionary

        Returns:
            Dictionary with derived summary metrics
        """
        # Exclude target variable from feature count
        total_features = sum(1 for f in features if not f.get('is_target', False))

        # Count feature types
        feature_type_counts = metadata.get('feature_types', {}) if metadata else {}

        # Calculate averages across features (derived insights)
        missing_percentages = [f.get('statistics', {}).get('missing_percentage', 0) for f in features]
        outlier_percentages = [f.get('outliers', {}).get('percent', 0) for f in features if f.get('outliers')]

        avg_missing = sum(missing_percentages) / len(missing_percentages) if missing_percentages else 0
        avg_outliers = sum(outlier_percentages) / len(outlier_percentages) if outlier_percentages else 0

        # Count provider features (derived from feature metadata)
        provider_features = sum(1 for f in features if f.get('source', {}).get('provider'))
        derived_features = sum(1 for f in features if f.get('is_derived', False))

        return {
            "total_features": total_features,
            "feature_types": feature_type_counts,
            "provider_features": provider_features,
            "derived_features": derived_features,
            "avg_missing_percentage": safe_round(avg_missing, 2),
            "avg_outliers_percentage": safe_round(avg_outliers, 2),
        }

    @staticmethod
    def _compute_highest_metrics(features: list, stability_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Find features with highest metrics."""
        highest_metrics = {}

        # Highest target association (best of Pearson, Correlation Ratio, Theil's U)
        features_with_assoc = []
        for f in features:
            target_rel = f.get('target_relationship', {})
            pearson = target_rel.get('correlation_pearson')
            corr_ratio = target_rel.get('correlation_ratio')
            theils_u = target_rel.get('theils_u')
            candidates = [abs(pearson) if pearson is not None else None, corr_ratio, theils_u]
            best = max((c for c in candidates if c is not None), default=None)
            if best is not None and best > 0:
                features_with_assoc.append((f.get('feature_name'), best))
        if features_with_assoc:
            top_assoc = max(features_with_assoc, key=lambda x: x[1])
            highest_metrics['highest_association'] = {
                'feature_name': top_assoc[0],
                'value': safe_round(top_assoc[1], 4),
                'target': features[0].get('target_relationship', {}).get('target_variable') if features else None
            }

        # Highest IV
        features_with_iv = [
            (f.get('feature_name'), f.get('target_relationship', {}).get('information_value', 0))
            for f in features
            if f.get('target_relationship', {}).get('information_value') is not None
        ]
        if features_with_iv:
            top_iv = max(features_with_iv, key=lambda x: x[1])
            highest_metrics['highest_iv'] = {
                'feature_name': top_iv[0],
                'value': safe_round(top_iv[1], 4)
            }

        # Highest stability (use unified stability map)
        stability_map = JSONFormatter._get_feature_stability_map(stability_results)
        if stability_map:
            stable_features = [
                (fname, data.get('psi'))
                for fname, data in stability_map.items()
                if data.get('psi') is not None and data.get('stability') == 'stable'
            ]

            if stable_features:
                top_stable = min(stable_features, key=lambda x: x[1])
                highest_metrics['highest_stability'] = {
                    'feature_name': top_stable[0],
                    'value': safe_round(top_stable[1], 4),
                    'stability': 'stable',
                    'interpretation': 'Most stable feature (lowest PSI)'
                }
            else:
                highest_metrics['highest_stability'] = {
                    'feature_name': None,
                    'value': None,
                    'note': 'No stable features found (all PSI >= 0.1)'
                }
        else:
            highest_metrics['highest_stability'] = {
                'feature_name': None,
                'value': None,
                'note': 'Stability metrics require cohort or time-based data'
            }

        return highest_metrics

    @staticmethod
    def _compute_top_features_by_statistical_score(features: list, stability_results: Optional[Dict] = None, top_n: int = 10) -> list:
        """Compute top N features ranked by Information Value (IV).

        IV is the industry-standard univariate metric for feature screening
        in credit risk / scorecard development (Siddiqi, 2006). It works
        equally for continuous and categorical features.

        Association strength (Pearson, Correlation Ratio, or Theil's U) and
        stability (PSI) are included as context columns but do not affect rank.
        """
        # Extract stability PSI by feature name using unified map
        stability_map = JSONFormatter._get_feature_stability_map(stability_results)
        stability_psi = {}
        for fname, data in stability_map.items():
            psi = data.get('psi')
            if psi is not None:
                stability_psi[fname] = psi

        scored_features = []
        for f in features:
            target_rel = f.get('target_relationship', {})
            iv = target_rel.get('information_value')
            if iv is None:
                continue

            # Best available target association for context
            pearson = target_rel.get('correlation_pearson')
            corr_ratio = target_rel.get('correlation_ratio')
            theils_u = target_rel.get('theils_u')
            association = abs(pearson) if pearson is not None else (corr_ratio or theils_u or 0)

            feature_name = f.get('feature_name')
            scored_features.append({
                'feature_name': feature_name,
                'iv': iv,
                'association': safe_round(association, 4),
                'psi': stability_psi.get(feature_name),
                'predictive_power': target_rel.get('predictive_power'),
            })

        # Rank by IV descending
        scored_features.sort(key=lambda x: x['iv'], reverse=True)
        top_features = scored_features[:top_n]

        result = []
        for i, f in enumerate(top_features, 1):
            result.append({
                'rank': i,
                'feature_name': f['feature_name'],
                'iv': safe_round(f['iv'], 4),
                'predictive_power': f['predictive_power'],
                'association': f['association'],
                'psi': safe_round(f['psi'], 4) if f['psi'] is not None else None,
            })

        return result

    @staticmethod
    def _compute_feature_counts(features: list, stability_results: Optional[Dict[str, Any]] = None, target_variable: Optional[str] = None) -> Dict[str, Any]:
        """
        Compute feature counts for dashboard metrics.

        Returns counts for:
        - High correlation features (>0.1)
        - Redundant features (>0.7)
        - High IV features (>0.1)
        - High stability features (PSI < 0.5)

        Args:
            features: List of feature analysis results
            stability_results: Optional stability analysis results

        Returns:
            Dictionary with counts, thresholds, and descriptions for each category
        """
        # Count high correlation features (correlation with target > 0.1)
        high_correlation_count = 0
        high_correlation_threshold = 0.1
        for f in features:
            target_rel = f.get('target_relationship', {})
            corr = target_rel.get('correlation_pearson')
            corr_ratio = target_rel.get('correlation_ratio')
            theils_u = target_rel.get('theils_u')
            candidates = [abs(corr) if corr is not None else None, corr_ratio, theils_u]
            best = max((c for c in candidates if c is not None), default=0)
            if best > high_correlation_threshold:
                high_correlation_count += 1

        # Count redundant features (high correlation with other features > 0.7)
        redundant_count = 0
        redundancy_threshold = 0.7
        for f in features:
            correlations = f.get('correlations', {})
            top_correlated = correlations.get('top_correlated_features', [])
            for corr_data in top_correlated:
                corr_feat = corr_data.get('feature')
                if target_variable and corr_feat == target_variable:
                    continue
                if abs(corr_data.get('correlation', 0)) > redundancy_threshold:
                    redundant_count += 1
                    break

        # Count high IV features (IV > 0.1)
        high_iv_count = 0
        high_iv_threshold = 0.1
        for f in features:
            iv = f.get('target_relationship', {}).get('information_value')
            if iv is not None and iv > high_iv_threshold:
                high_iv_count += 1

        # Count stable features (PSI < 0.1)
        high_stability_count = 0
        stability_threshold = 0.1
        stability_map = JSONFormatter._get_feature_stability_map(stability_results)
        for stability_data in stability_map.values():
            psi = stability_data.get('psi')
            if psi is not None and psi < stability_threshold:
                high_stability_count += 1

        # Count data quality flags
        high_missing_count = sum(1 for f in features if f.get('quality', {}).get('has_high_missing', False))
        constant_count = sum(1 for f in features if f.get('quality', {}).get('is_constant', False))
        low_variance_count = sum(1 for f in features if f.get('quality', {}).get('has_low_variance', False))
        not_recommended_count = sum(1 for f in features if not f.get('quality', {}).get('recommended_for_modeling', True))

        # Count distribution shape issues (continuous features only)
        highly_skewed_count = 0
        high_kurtosis_count = 0
        for f in features:
            stats = f.get('stats', {})
            skew = stats.get('skewness')
            kurt = stats.get('kurtosis')
            if skew is not None and abs(skew) > 1.0:
                highly_skewed_count += 1
            if kurt is not None and kurt > 3.0:
                high_kurtosis_count += 1

        # Count high-cardinality categoricals
        high_cardinality_count = sum(1 for f in features if f.get('cardinality', {}).get('is_high_cardinality', False))

        # Count suspected leakage (IV > 0.5)
        suspected_leakage_count = 0
        leakage_threshold = 0.5
        for f in features:
            iv = f.get('target_relationship', {}).get('information_value')
            if iv is not None and iv > leakage_threshold:
                suspected_leakage_count += 1

        # Predictive power breakdown by IV class
        pp_counts = {"unpredictive": 0, "weak": 0, "medium": 0, "strong": 0, "very_strong": 0}
        for f in features:
            pp = f.get('target_relationship', {}).get('predictive_power')
            if pp and pp in pp_counts:
                pp_counts[pp] += 1

        # Statistically significant correlations (p-value < 0.05)
        significant_count = 0
        for f in features:
            tr = f.get('target_relationship', {})
            pearson_p = tr.get('correlation_pearson_pvalue')
            spearman_p = tr.get('correlation_spearman_pvalue')
            if (pearson_p is not None and pearson_p < 0.05) or (spearman_p is not None and spearman_p < 0.05):
                significant_count += 1

        # Stability trends (from stability map — works for both cohort and time-based)
        increasing_drift_count = 0
        volatile_count = 0
        minor_shift_count_stability = 0
        major_shift_count_stability = 0
        for data in stability_map.values():
            trend = data.get('trend')
            stab = data.get('stability')
            if trend == 'increasing_drift':
                increasing_drift_count += 1
            if trend == 'volatile':
                volatile_count += 1
            if stab == 'minor_shift':
                minor_shift_count_stability += 1
            if stab == 'major_shift':
                major_shift_count_stability += 1

        return {
            "high_correlation": {
                "count": high_correlation_count,
                "threshold": high_correlation_threshold,
                "description": f"Features with absolute correlation > {high_correlation_threshold}"
            },
            "redundant_features": {
                "count": redundant_count,
                "threshold": redundancy_threshold,
                "description": f"Features with correlation > {redundancy_threshold} with another feature"
            },
            "high_iv": {
                "count": high_iv_count,
                "threshold": high_iv_threshold,
                "description": f"Features with Information Value > {high_iv_threshold}"
            },
            "high_stability": {
                "count": high_stability_count,
                "threshold": stability_threshold,
                "description": f"Features with PSI < {stability_threshold} (stable distribution)",
                "note": "Lower PSI values indicate higher stability" if high_stability_count > 0 else "Stability analysis not performed or no stable features found"
            },
            "high_missing": {
                "count": high_missing_count,
                "threshold": 0.3,
                "description": "Features with >30% missing values"
            },
            "constant_features": {
                "count": constant_count,
                "description": "Features with only one unique value"
            },
            "low_variance": {
                "count": low_variance_count,
                "description": "Features with very low coefficient of variation"
            },
            "not_recommended": {
                "count": not_recommended_count,
                "description": "Features not recommended for modeling"
            },
            "highly_skewed": {
                "count": highly_skewed_count,
                "threshold": 1.0,
                "description": "Features with |skewness| > 1.0"
            },
            "high_kurtosis": {
                "count": high_kurtosis_count,
                "threshold": 3.0,
                "description": "Features with kurtosis > 3.0 (outlier-prone)"
            },
            "high_cardinality": {
                "count": high_cardinality_count,
                "description": "Categorical features with high unique-value ratio"
            },
            "suspected_leakage": {
                "count": suspected_leakage_count,
                "threshold": leakage_threshold,
                "description": f"Features with IV > {leakage_threshold} (possible information leakage)"
            },
            "predictive_power": {
                "unpredictive": pp_counts["unpredictive"],
                "weak": pp_counts["weak"],
                "medium": pp_counts["medium"],
                "strong": pp_counts["strong"],
                "very_strong": pp_counts["very_strong"],
                "description": "Feature count by IV-based predictive power class"
            },
            "significant_correlations": {
                "count": significant_count,
                "threshold": 0.05,
                "description": "Features with statistically significant correlation (p < 0.05)"
            },
            "increasing_drift": {
                "count": increasing_drift_count,
                "description": "Features with worsening distribution drift over time"
            },
            "volatile_stability": {
                "count": volatile_count,
                "description": "Features with inconsistent stability across periods"
            },
            "minor_shift": {
                "count": minor_shift_count_stability,
                "description": "Features with minor distribution shift (0.1 <= PSI < 0.2)"
            },
            "major_shift": {
                "count": major_shift_count_stability,
                "description": "Features with major distribution shift (PSI >= 0.2)"
            },
        }

    @staticmethod
    def _compute_data_quality(dataset_info: Dict[str, Any], features: list) -> Dict[str, Any]:
        """Compute data quality summary."""
        # Find features with quality issues
        features_with_high_missing = [
            f.get('feature_name') for f in features
            if f.get('quality', {}).get('has_high_missing', False)
        ]

        features_with_low_variance = [
            f.get('feature_name') for f in features
            if f.get('quality', {}).get('has_low_variance', False)
        ]

        features_with_outliers = [
            f.get('feature_name') for f in features
            if f.get('quality', {}).get('has_outliers', False)
        ]

        # Calculate overall quality score (0-10)
        # Based on: missing data, duplicates, low variance features, outliers
        total_features = sum(1 for f in features if not f.get('is_target', False))
        score = 10.0

        # Penalize for high missing
        if features_with_high_missing and total_features > 0:
            score -= min(2.0, len(features_with_high_missing) / total_features * 5)

        # Penalize for low variance
        if features_with_low_variance and total_features > 0:
            score -= min(1.5, len(features_with_low_variance) / total_features * 3)

        # Penalize for duplicates
        duplicate_rows = dataset_info.get('duplicate_rows', 0)
        total_rows = dataset_info.get('rows', 1)
        if duplicate_rows > 0:
            score -= min(2.0, duplicate_rows / total_rows * 5)

        # Penalize for too many outliers
        outlier_ratio = len(features_with_outliers) / total_features if total_features > 0 else 0
        if outlier_ratio > 0.5:
            score -= 1.5

        score = max(0.0, score)  # Cap at 0

        # Generate recommendations
        recommendations = []
        if features_with_high_missing:
            recommendations.append(f"Consider imputation for {len(features_with_high_missing)} features with >30% missing values")
        if features_with_outliers:
            recommendations.append(f"Review outliers in {len(features_with_outliers)} features")
        if features_with_low_variance:
            recommendations.append(f"Consider removing {len(features_with_low_variance)} low-variance features")
        if duplicate_rows > 0:
            recommendations.append(f"Remove {duplicate_rows} duplicate rows")

        return {
            "overall_score": safe_round(score, 1),
            "total_missing_cells": dataset_info.get('missing_cells', 0),
            "features_with_high_missing_count": len(features_with_high_missing),
            "features_with_low_variance_count": len(features_with_low_variance),
            "features_with_outliers_count": len(features_with_outliers),
            "duplicate_rows": dataset_info.get('duplicate_rows', 0),
            "recommended_actions": recommendations
        }
