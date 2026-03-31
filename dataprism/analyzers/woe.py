"""Weight of Evidence (WoE) and Information Value (IV) computation.

Provides a single canonical implementation of crosstab → WoE/IV math
used by both DataPrism EDA and the studio-processor pipeline.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class WoEResult:
    """Result of a WoE/IV computation for one feature."""

    woe_mapping: dict[str, float]
    """Category (or bin label) → WoE value."""

    iv: float
    """Total Information Value."""

    iv_contributions: dict[str, float]
    """Category → IV contribution."""

    predictive_power: str
    """Siddiqi classification: unpredictive / weak / medium / strong / very_strong."""


class WoEAnalyzer:
    """Weight of Evidence and Information Value computation.

    Args:
        smoothing: Laplace smoothing value added to counts.
            When > 0, applies Laplace smoothing:
                rate = (count + smoothing) / (total + smoothing * num_categories)
            When == 0, uses epsilon floor (1e-10) on zero rates.
    """

    def __init__(self, smoothing: float = 0.5) -> None:
        self.smoothing = smoothing

    def compute(
        self,
        feature: pd.Series,
        target: pd.Series,
    ) -> WoEResult | None:
        """Compute WoE mapping and IV from a feature and binary target.

        IV = Σ (% of goods - % of bads) * WoE
        WoE = ln(% of goods / % of bads)

        Args:
            feature: Categorical series (raw categories or bin labels).
            target: Binary target series (0/1).

        Returns:
            A ``WoEResult``, or ``None`` when the crosstab is degenerate
            (fewer than 2 target classes represented).
        """
        df = pd.DataFrame({"feature": feature, "target": target})
        crosstab = pd.crosstab(df["feature"], df["target"])

        if crosstab.shape[1] < 2:
            return None

        # Assume binary target: column 0 is good, column 1 is bad
        cols = sorted(crosstab.columns)
        goods_col = cols[0]  # Typically 0 or False (negative class)
        bads_col = cols[1]  # Typically 1 or True (positive class)

        goods = crosstab[goods_col]
        bads = crosstab[bads_col]

        total_goods = goods.sum()
        total_bads = bads.sum()

        if total_goods == 0 or total_bads == 0:
            return None

        num_categories = len(crosstab.index)
        woe_mapping: dict[str, float] = {}
        iv_contributions: dict[str, float] = {}
        total_iv = 0.0

        for category in crosstab.index:
            good_count = goods[category]
            bad_count = bads[category]

            if self.smoothing > 0:
                s, n = self.smoothing, num_categories
                good_pct = (good_count + s) / (total_goods + s * n)
                bad_pct = (bad_count + s) / (total_bads + s * n)
            else:
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

        return WoEResult(
            woe_mapping=woe_mapping,
            iv=total_iv,
            iv_contributions=iv_contributions,
            predictive_power=self.classify_predictive_power(total_iv),
        )

    @staticmethod
    def classify_predictive_power(iv: float) -> str:
        """Classify predictive power based on IV (Siddiqi thresholds)."""
        if iv < 0.02:
            return "unpredictive"
        if iv < 0.1:
            return "weak"
        if iv < 0.3:
            return "medium"
        if iv < 0.5:
            return "strong"
        return "very_strong"
