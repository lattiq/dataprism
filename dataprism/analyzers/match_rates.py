"""Provider-level match rate analysis."""

import re
from typing import Any, Dict, Optional

import pandas as pd

from dataprism.schema import ColumnConfig


def compute_provider_match_rates(
    df: pd.DataFrame,
    col_configs: Optional[Dict[str, ColumnConfig]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Compute match rates by provider.

    Args:
        df: DataFrame with features (after sentinel value replacement)
        col_configs: Dictionary mapping column names to ColumnConfig objects.

    Returns:
        Dictionary mapping provider names to their statistics.
    """
    if col_configs is None or not col_configs:
        return _compute_provider_match_rates_from_columns(df)

    # Group features by provider
    provider_features: Dict[str, list] = {}
    for feature_name, config in col_configs.items():
        provider = config.provider

        if not provider or not feature_name or feature_name not in df.columns:
            continue

        if provider not in provider_features:
            provider_features[provider] = []

        provider_features[provider].append(feature_name)

    # Compute match rates for each provider
    provider_stats = {}
    total_records = len(df)

    for provider, features in provider_features.items():
        record_not_found_col = f"{provider}_record_not_found"

        if record_not_found_col in df.columns:
            matched_records = (df[record_not_found_col] == 0).sum()
            overall_match_rate = matched_records / total_records if total_records > 0 else 0

            total_cells = len(features) * total_records
            non_null_cells = sum(df[feature].notna().sum() for feature in features)
            feature_match_rate = non_null_cells / total_cells if total_cells > 0 else 0

            provider_stats[provider] = {
                "match_rate": round(overall_match_rate, 4),
                "total_features": len(features),
                "total_records": total_records,
                "matched_records": int(matched_records),
                "computation_method": "record_not_found_column",
                "feature_match_rate": round(feature_match_rate, 4)
            }

        else:
            provider_matched_per_record = []
            total_cells = len(features) * total_records
            non_null_cells = 0

            for feature in features:
                non_null_count = df[feature].notna().sum()
                non_null_cells += non_null_count
                provider_matched_per_record.append(df[feature].notna())

            if provider_matched_per_record:
                any_match_per_record = pd.concat(provider_matched_per_record, axis=1).any(axis=1)
                matched_records = any_match_per_record.sum()
                overall_match_rate = matched_records / total_records if total_records > 0 else 0
            else:
                matched_records = 0
                overall_match_rate = 0.0

            feature_match_rate = non_null_cells / total_cells if total_cells > 0 else 0

            provider_stats[provider] = {
                "match_rate": round(overall_match_rate, 4),
                "total_features": len(features),
                "total_records": total_records,
                "matched_records": int(matched_records),
                "computation_method": "feature_analysis",
                "feature_match_rate": round(feature_match_rate, 4)
            }

    return provider_stats


def _compute_provider_match_rates_from_columns(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Detect and compute provider match rates from <provider>_record_not_found columns.

    This is a fallback method when no col_configs is provided.
    """
    provider_stats = {}
    total_records = len(df)

    record_not_found_pattern = re.compile(r'^(.+)_record_not_found$')

    for col in df.columns:
        match = record_not_found_pattern.match(col)
        if match:
            provider = match.group(1)

            matched_records = (df[col] == 0).sum()
            overall_match_rate = matched_records / total_records if total_records > 0 else 0

            provider_stats[provider] = {
                "match_rate": round(overall_match_rate, 4),
                "total_features": 0,
                "total_records": total_records,
                "matched_records": int(matched_records),
                "computation_method": "record_not_found_column",
                "feature_match_rate": 0.0
            }

    return provider_stats
