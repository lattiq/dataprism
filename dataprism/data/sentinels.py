"""Sentinel value replacement — converts provider sentinel codes to proper nulls."""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from dataprism.utils.logger import get_logger
from dataprism.schema import ColumnConfig

# Setup module logger
logger = get_logger(__name__)

# Map numpy int dtypes to pandas nullable equivalents
_NULLABLE_INT_MAP = {
    np.dtype("int8"): pd.Int8Dtype(),
    np.dtype("int16"): pd.Int16Dtype(),
    np.dtype("int32"): pd.Int32Dtype(),
    np.dtype("int64"): pd.Int64Dtype(),
    np.dtype("uint8"): pd.UInt8Dtype(),
    np.dtype("uint16"): pd.UInt16Dtype(),
    np.dtype("uint32"): pd.UInt32Dtype(),
    np.dtype("uint64"): pd.UInt64Dtype(),
}


def replace_sentinel_values_with_nulls(
    df: pd.DataFrame,
    col_configs: Optional[Dict[str, ColumnConfig]] = None
) -> pd.DataFrame:
    """
    Replace sentinel values (missing, not_found) with proper nulls.

    Args:
        df: Input DataFrame
        col_configs: Dictionary mapping column names to ColumnConfig objects.

    Returns:
        DataFrame with sentinel values replaced by nulls.
        Integer columns are converted to pandas nullable integer types
        (e.g. Int64) to avoid silent upcasting to float64.
    """
    if col_configs is None or not col_configs:
        return df.copy()

    # Create a copy to avoid modifying the original
    df_clean = df.copy()

    replacement_count = 0
    features_processed = []

    for feature_name, config in col_configs.items():
        if not feature_name or feature_name not in df_clean.columns:
            continue

        if config.sentinels is None:
            continue

        sentinel_values = []

        if config.sentinels.missing is not None:
            sentinel_values.append(config.sentinels.missing)

        if config.sentinels.not_found is not None:
            sentinel_values.append(config.sentinels.not_found)

        if not sentinel_values:
            continue

        # Remember original dtype to preserve integer types
        original_dtype = df_clean[feature_name].dtype
        feature_replaced = False

        for sentinel in sentinel_values:
            try:
                if pd.api.types.is_numeric_dtype(df_clean[feature_name]):
                    if isinstance(sentinel, str):
                        try:
                            sentinel_numeric = float(sentinel)
                            mask = df_clean[feature_name] == sentinel_numeric
                        except (ValueError, TypeError):
                            mask = df_clean[feature_name].astype(str) == sentinel
                    else:
                        mask = df_clean[feature_name] == sentinel
                else:
                    if sentinel == '':
                        mask = (df_clean[feature_name] == '') & df_clean[feature_name].notna()
                    else:
                        mask = df_clean[feature_name].astype(str) == str(sentinel)

                if mask.any():
                    count_before = df_clean[feature_name].isna().sum()
                    df_clean.loc[mask, feature_name] = np.nan
                    count_after = df_clean[feature_name].isna().sum()
                    replaced = count_after - count_before

                    if replaced > 0:
                        replacement_count += replaced
                        feature_replaced = True

            except (ValueError, TypeError, KeyError) as e:
                logger.warning("  Warning: Could not replace sentinel '%s' in feature '%s': %s", sentinel, feature_name, e)
                continue

        # Convert int columns that got upcasted to float back to nullable int
        if feature_replaced and original_dtype in _NULLABLE_INT_MAP:
            df_clean[feature_name] = df_clean[feature_name].astype(_NULLABLE_INT_MAP[original_dtype])

        if feature_replaced:
            features_processed.append(feature_name)

    if replacement_count > 0:
        logger.info("\nReplaced %d sentinel values with nulls across %d features", replacement_count, len(features_processed))

    return df_clean
