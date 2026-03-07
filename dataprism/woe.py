"""Weight of Evidence (WoE) and Information Value (IV) computation.

Provides a single canonical implementation of crosstab → WoE/IV math
used by both DataPrism EDA and the studio-processor pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class WoEResult:
    """Result of a WoE/IV computation for one feature."""

    woe_mapping: Dict[str, float]
    """Category (or bin label) → WoE value."""

    iv: float
    """Total Information Value."""

    iv_contributions: Dict[str, float]
    """Category → IV contribution."""

    predictive_power: str
    """Siddiqi classification: unpredictive / weak / medium / strong / very_strong."""


def compute_woe_iv(
    feature: pd.Series,
    target: pd.Series,
    smoothing: float = 0.5,
) -> Optional[WoEResult]:
    """Compute WoE mapping and IV from a feature and binary target.

    IV = Σ (% of goods - % of bads) * WoE
    WoE = ln(% of goods / % of bads)

    Args:
        feature: Categorical series (raw categories or bin labels).
        target: Binary target series (0/1).
        smoothing: Laplace smoothing value added to counts.
            When > 0, applies Laplace smoothing:
                rate = (count + smoothing) / (total + smoothing * num_categories)
            When == 0, uses epsilon floor (1e-10) on zero rates.

    Returns:
        A ``WoEResult``, or ``None`` when the crosstab is degenerate
        (fewer than 2 target classes represented).
    """
    # Create crosstab
    df = pd.DataFrame({"feature": feature, "target": target})

    # Get counts of goods (0) and bads (1)
    crosstab = pd.crosstab(df["feature"], df["target"])

    # Handle case where target only has one class in some bins
    if crosstab.shape[1] < 2:
        return None

    # Assume binary target: column 0 is good, column 1 is bad
    # Get the actual column names (could be 0/1, True/False, etc.)
    cols = sorted(crosstab.columns)
    goods_col = cols[0]  # Typically 0 or False (negative class)
    bads_col = cols[1]   # Typically 1 or True (positive class)

    goods = crosstab[goods_col]
    bads = crosstab[bads_col]

    # Calculate distributions
    total_goods = goods.sum()
    total_bads = bads.sum()

    if total_goods == 0 or total_bads == 0:
        return None

    # Calculate WoE and IV for each category
    num_categories = len(crosstab.index)
    woe_mapping: Dict[str, float] = {}
    iv_contributions: Dict[str, float] = {}
    total_iv = 0.0

    for category in crosstab.index:
        good_count = goods[category]
        bad_count = bads[category]

        if smoothing > 0:
            # Laplace smoothing: add smoothing per category, normalize denominator
            good_pct = (good_count + smoothing) / (total_goods + smoothing * num_categories)
            bad_pct = (bad_count + smoothing) / (total_bads + smoothing * num_categories)
        else:
            # Epsilon floor: replace zero rates with 1e-10
            eps = 1e-10
            good_pct = good_count / total_goods if good_count > 0 else eps
            bad_pct = bad_count / total_bads if bad_count > 0 else eps

        # WoE = ln(% of goods / % of bads) — Siddiqi (2006) convention
        # Positive WoE = lower risk, negative WoE = higher risk
        woe = float(np.log(good_pct / bad_pct))

        # IV contribution = (% of goods - % of bads) * WoE
        iv_contrib = float((good_pct - bad_pct) * woe)

        woe_mapping[str(category)] = woe
        iv_contributions[str(category)] = iv_contrib
        total_iv += iv_contrib

    predictive_power = _classify_predictive_power(total_iv)

    return WoEResult(
        woe_mapping=woe_mapping,
        iv=total_iv,
        iv_contributions=iv_contributions,
        predictive_power=predictive_power,
    )


def _classify_predictive_power(iv: float) -> str:
    """Classify predictive power based on IV (Siddiqi thresholds)."""
    if iv < 0.02:
        return "unpredictive"
    elif iv < 0.1:
        return "weak"
    elif iv < 0.3:
        return "medium"
    elif iv < 0.5:
        return "strong"
    else:
        return "very_strong"
