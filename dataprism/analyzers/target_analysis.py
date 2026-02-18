"""Target relationship analysis including IV and WoE calculation."""

from typing import Any, Dict, Optional

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from dataprism.analyzers.correlation import CorrelationEngine
from dataprism.utils.logger import get_logger
from dataprism.output.formatter import safe_round

# Setup module logger
logger = get_logger(__name__)


class TargetAnalyzer:
    """Analyzer for target variable relationships."""

    def __init__(self, max_bins: int = 10):
        """
        Initialize target analyzer.

        Args:
            max_bins: Maximum number of bins for continuous variables
        """
        self.max_bins = max_bins

    def analyze_target_relationship(
        self,
        feature_series: pd.Series,
        target_series: pd.Series,
        feature_type: str,
        distribution: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze relationship between feature and target variable.

        Args:
            feature_series: Feature values
            target_series: Target variable values (binary)
            feature_type: 'continuous' or 'categorical'
            distribution: Pre-computed distribution data (bin_edges for continuous, value_counts for categorical)

        Returns:
            Dictionary with correlation, IV, WoE, and predictive power
        """
        result = {
            "target_variable": target_series.name,
            "correlation_pearson": None,
            "correlation_pearson_pvalue": None,
            "correlation_spearman": None,
            "correlation_spearman_pvalue": None,
            "theils_u": None,
            "correlation_ratio": None,
            "information_value": None,
            "predictive_power": None,
            "woe_mapping": None,
            "iv_contribution": None,
            "target_mean_per_bin": None
        }

        # Remove rows where either feature or target is missing
        valid_idx = feature_series.notna() & target_series.notna()
        feature_clean = feature_series[valid_idx]
        target_clean = target_series[valid_idx]

        if len(feature_clean) < 10:  # Need minimum sample size
            return result

        # Check if target is binary
        unique_targets = target_clean.nunique()
        if unique_targets != 2:
            # IV only works for binary classification
            result["note"] = f"Target has {unique_targets} unique values. IV requires binary target."

        # Compute correlations based on feature type
        if feature_type == "continuous":
            result.update(self._compute_correlations(feature_clean, target_clean))
            # Correlation Ratio: continuous feature → categorical target
            result["correlation_ratio"] = safe_round(
                CorrelationEngine._correlation_ratio(target_clean, feature_clean), 4
            )
        else:
            # Categorical feature
            # Theil's U: categorical feature → categorical target
            result["theils_u"] = safe_round(
                CorrelationEngine._theils_u(feature_clean, target_clean), 4
            )
            # Correlation Ratio: categorical feature → continuous-like target
            result["correlation_ratio"] = safe_round(
                CorrelationEngine._correlation_ratio(feature_clean, target_clean), 4
            )

        # Compute Information Value and WoE
        if unique_targets == 2:
            if feature_type == "continuous":
                iv_result = self._compute_iv_continuous(feature_clean, target_clean)
            else:  # categorical
                iv_result = self._compute_iv_categorical(feature_clean, target_clean)

            result.update(iv_result)

            # Classify predictive power based on IV
            result["predictive_power"] = self._classify_predictive_power(
                result["information_value"]
            )

        # Compute target mean per bin (works for any numeric target)
        result["target_mean_per_bin"] = self._compute_target_mean_per_bin(
            feature_clean, target_clean, feature_type, distribution
        )

        return result

    def _compute_correlations(
        self,
        feature: pd.Series,
        target: pd.Series
    ) -> Dict[str, Any]:
        """Compute Pearson and Spearman correlations with p-values."""
        result = {}

        # Check if feature is constant (no variance)
        if feature.nunique() <= 1:
            logger.debug("Skipping correlation for constant feature: %s", feature.name)
            result["correlation_pearson"] = None
            result["correlation_pearson_pvalue"] = None
            result["correlation_spearman"] = None
            result["correlation_spearman_pvalue"] = None
            return result

        try:
            # Pearson correlation - suppress ConstantInputWarning
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                pearson_corr, pearson_pval = stats.pearsonr(feature, target)
                if w:
                    logger.debug("Suppressed ConstantInputWarning for Pearson correlation on feature: %s", feature.name)
            result["correlation_pearson"] = safe_round(pearson_corr, 4)
            result["correlation_pearson_pvalue"] = safe_round(pearson_pval, 4)
        except (ValueError, TypeError, FloatingPointError) as e:
            logger.warning("Pearson correlation failed for '%s': %s", feature.name, e)
            result["correlation_pearson"] = None
            result["correlation_pearson_pvalue"] = None

        try:
            # Spearman correlation - suppress ConstantInputWarning
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                spearman_corr, spearman_pval = stats.spearmanr(feature, target)
                if w:
                    logger.debug("Suppressed ConstantInputWarning for Spearman correlation on feature: %s", feature.name)
            result["correlation_spearman"] = safe_round(spearman_corr, 4)
            result["correlation_spearman_pvalue"] = safe_round(spearman_pval, 4)
        except (ValueError, TypeError, FloatingPointError) as e:
            logger.warning("Spearman correlation failed for '%s': %s", feature.name, e)
            result["correlation_spearman"] = None
            result["correlation_spearman_pvalue"] = None

        return result

    def _compute_iv_continuous(
        self,
        feature: pd.Series,
        target: pd.Series
    ) -> Dict[str, Any]:
        """
        Compute Information Value for continuous features.
        Uses optimal binning based on quantiles.
        """
        # Bin the continuous feature
        try:
            # Use quantile-based binning
            binned_feature, bin_edges = pd.qcut(
                feature,
                q=min(self.max_bins, len(feature.unique())),
                duplicates='drop',
                retbins=True
            )
        except (ValueError, TypeError) as e:
            # Fall back to equal-width binning if quantile fails
            logger.debug("qcut failed for '%s': %s, falling back to equal-width bins", feature.name, e)
            try:
                binned_feature, bin_edges = pd.cut(
                    feature,
                    bins=self.max_bins,
                    duplicates='drop',
                    retbins=True
                )
            except (ValueError, TypeError) as e:
                logger.warning("IV binning failed for '%s': %s", feature.name, e)
                return {
                    "information_value": None,
                    "woe_mapping": None,
                    "iv_contribution": None
                }

        # Compute IV for binned feature
        return self._compute_iv_categorical(binned_feature, target)

    def _compute_iv_categorical(
        self,
        feature: pd.Series,
        target: pd.Series
    ) -> Dict[str, Any]:
        """
        Compute Information Value and WoE for categorical features.

        IV = Σ (% of goods - % of bads) * WoE
        WoE = ln(% of goods / % of bads)
        """
        # Create crosstab
        df = pd.DataFrame({'feature': feature, 'target': target})

        # Get counts of goods (0) and bads (1)
        crosstab = pd.crosstab(df['feature'], df['target'])

        # Handle case where target only has one class in some bins
        if crosstab.shape[1] < 2:
            return {
                "information_value": None,
                "woe_mapping": None,
                "iv_contribution": None
            }

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
            return {
                "information_value": None,
                "woe_mapping": None,
                "iv_contribution": None
            }

        # Calculate WoE and IV for each category
        woe_mapping = {}
        iv_contribution = {}
        total_iv = 0.0
        num_categories = len(crosstab.index)

        for category in crosstab.index:
            good_count = goods[category]
            bad_count = bads[category]

            # Laplace smoothing: add 0.5 per category, normalize denominator accordingly
            good_pct = (good_count + 0.5) / (total_goods + 0.5 * num_categories)
            bad_pct = (bad_count + 0.5) / (total_bads + 0.5 * num_categories)

            # WoE = ln(% of bads / % of goods) — standard convention (Siddiqi 2006)
            # Positive WoE = higher proportion of bads (higher risk)
            woe = np.log(bad_pct / good_pct)

            # IV contribution = (% of bads - % of goods) * WoE
            iv_contrib = (bad_pct - good_pct) * woe

            woe_mapping[str(category)] = safe_round(woe, 4)
            iv_contribution[str(category)] = safe_round(iv_contrib, 4)
            total_iv += iv_contrib

        # Add default WoE for unseen categories (use 0)
        woe_mapping["_default"] = 0.0

        return {
            "information_value": safe_round(total_iv, 4),
            "woe_mapping": woe_mapping,
            "iv_contribution": iv_contribution
        }

    def _compute_target_mean_per_bin(
        self,
        feature: pd.Series,
        target: pd.Series,
        feature_type: str,
        distribution: Optional[Dict[str, Any]] = None
    ) -> Optional[list]:
        """
        Compute mean target value per distribution bin.

        For continuous features, uses the same bin edges as the histogram.
        For categorical features, uses the top categories (sorted by frequency descending).

        Returns:
            List of mean target values aligned with the distribution bins/categories,
            or None if computation fails.
        """
        try:
            if feature_type == "continuous":
                bin_edges = None
                if distribution and distribution.get("bin_edges"):
                    bin_edges = distribution["bin_edges"]

                if bin_edges is None:
                    # Fall back to computing our own 10-bin histogram edges
                    _, bin_edges = np.histogram(feature.dropna(), bins=10)
                    bin_edges = bin_edges.tolist()

                # Assign each value to a bin using the same edges
                bin_indices = np.digitize(feature.values, bin_edges, right=False)
                # np.digitize returns 1-based indices; bin 0 = below first edge, bin len = above last
                # Clamp to valid range [1, num_bins]
                num_bins = len(bin_edges) - 1
                bin_indices = np.clip(bin_indices, 1, num_bins)

                means = []
                for b in range(1, num_bins + 1):
                    mask = bin_indices == b
                    if mask.sum() > 0:
                        means.append(safe_round(float(target.values[mask].mean()), 4))
                    else:
                        means.append(None)
                return means

            else:
                # Categorical: align with top value_counts (sorted desc by frequency)
                if distribution and distribution.get("value_counts"):
                    categories = sorted(
                        distribution["value_counts"].items(),
                        key=lambda x: x[1], reverse=True
                    )[:20]
                    cat_names = [c[0] for c in categories]
                else:
                    cat_names = feature.value_counts().head(20).index.tolist()

                means = []
                for cat in cat_names:
                    mask = feature.astype(str) == str(cat)
                    if mask.sum() > 0:
                        means.append(safe_round(float(target[mask].mean()), 4))
                    else:
                        means.append(None)
                return means

        except Exception as e:
            logger.debug("Target mean per bin failed for '%s': %s", feature.name, e)
            return None

    def _classify_predictive_power(self, iv: Optional[float]) -> Optional[str]:
        """
        Classify predictive power based on Information Value.

        Thresholds follow the Siddiqi classification (Credit Risk
        Scorecards, 2006) — the industry-standard reference for
        scorecard development. IV >= 0.5 is flagged as suspicious
        because it often indicates data leakage rather than
        genuinely strong predictive power.

        IV < 0.02: Unpredictive
        0.02 <= IV < 0.1: Weak
        0.1 <= IV < 0.3: Medium
        0.3 <= IV < 0.5: Strong
        IV >= 0.5: Very Strong / Suspicious
        """
        if iv is None:
            return None

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

    def compute_vif(self, df: pd.DataFrame, feature_col: str) -> Optional[float]:
        """
        Compute Variance Inflation Factor for a feature.

        VIF measures how much a feature's variance is inflated by
        correlation with other features (multicollinearity).
        VIF=1 means no correlation; VIF>5-10 signals redundancy —
        the feature is largely explained by others and may be dropped.

        VIF = 1 / (1 - R²)
        where R² is from regressing this feature on all other features.

        Args:
            df: DataFrame with numeric features
            feature_col: Feature to compute VIF for

        Returns:
            VIF value or None if cannot compute
        """
        try:
            from sklearn.linear_model import LinearRegression

            # Get all numeric columns except the feature
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if feature_col not in numeric_cols:
                return None

            other_cols = [col for col in numeric_cols if col != feature_col]
            if len(other_cols) < 1:
                return None

            # Prepare data
            X = df[other_cols].fillna(df[other_cols].mean())
            y = df[feature_col].fillna(df[feature_col].mean())

            # Fit regression
            model = LinearRegression()
            model.fit(X, y)

            # Calculate R²
            r_squared = model.score(X, y)

            # Calculate VIF
            if r_squared >= 0.9999:  # Avoid division by near-zero
                return None

            vif = 1 / (1 - r_squared)
            return safe_round(vif, 2)

        except ImportError:
            logger.warning("VIF requires scikit-learn. Install with: pip install scikit-learn")
            return None
        except (ValueError, np.linalg.LinAlgError) as e:
            logger.debug("VIF computation failed for '%s': %s", feature_col, e)
            return None
