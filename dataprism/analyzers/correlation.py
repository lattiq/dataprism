"""Correlation analysis engine for EDA."""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from dataprism.schema import ColumnConfig, ColumnType
from dataprism.utils.logger import get_logger
from dataprism.output.formatter import safe_round

logger = get_logger(__name__)


class CorrelationEngine:
    """Handles correlation computation and analysis for features."""

    def __init__(self, top_correlations: int = 10, max_correlation_features: Optional[int] = None):
        """
        Initialize correlation engine.

        Args:
            top_correlations: Number of top correlations to show per feature
            max_correlation_features: Maximum features for correlation matrix. None = no limit
        """
        self.top_correlations = top_correlations
        self.max_correlation_features = max_correlation_features

    def compute_correlation_matrix(
        self,
        df: pd.DataFrame,
        target_variable: Optional[str] = None,
        column_configs: Optional[Dict[str, ColumnConfig]] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Compute correlation matrix for continuous features.

        Args:
            df: Input dataframe
            target_variable: Optional target variable to include
            column_configs: Optional column configs to determine continuous columns from schema

        Returns:
            Correlation matrix as DataFrame or None if no continuous features
        """
        # Per-column decision: use schema type if available, else fall back to dtype
        numeric_cols = []
        for col in df.columns:
            config = column_configs.get(col) if column_configs else None
            if config is not None:
                if config.type is ColumnType.CONTINUOUS:
                    numeric_cols.append(col)
            elif df[col].dtype.kind in "iufb":
                numeric_cols.append(col)

        if not numeric_cols:
            logger.warning("No continuous columns found for correlation calculation")
            return None

        # If max_correlation_features is set, select features based on target correlation
        if self.max_correlation_features and len(numeric_cols) > self.max_correlation_features:
            numeric_cols = self._select_correlated_features(
                df, numeric_cols, target_variable, self.max_correlation_features
            )

        logger.info("Computing correlation matrix for %s numeric features", len(numeric_cols))
        return df[numeric_cols].corr()

    def _select_correlated_features(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
        target_variable: Optional[str],
        max_features: int,
    ) -> List[str]:
        """
        Select top features based on correlation with target.

        Args:
            df: Input dataframe
            numeric_cols: List of numeric column names
            target_variable: Target variable name
            max_features: Maximum number of features to select

        Returns:
            List of selected feature names
        """
        if not target_variable or target_variable not in numeric_cols:
            logger.info(
                f"No valid target variable for correlation selection. Using first {max_features} numeric features."
            )
            return numeric_cols[:max_features]

        # Compute correlations with target
        target_corr = df[numeric_cols].corrwith(df[target_variable]).abs()

        # Always include target variable
        top_features = [target_variable]

        # Select top correlated features (excluding target itself)
        other_features = target_corr.drop(target_variable, errors="ignore").nlargest(
            max_features - 1
        )
        top_features.extend(other_features.index.tolist())

        logger.info(
            f"Selected {len(top_features)} features with highest correlation to target '{target_variable}'"
        )
        return top_features

    def select_features_for_association(
        self,
        feature_names: List[str],
        features_data: Dict[str, Any],
        target_variable: Optional[str],
        correlation_matrix: Optional[pd.DataFrame],
        theils_u_matrix: Optional[pd.DataFrame],
        eta_matrix: Optional[Dict[str, Dict[str, float]]],
        max_features: int,
    ) -> List[str]:
        """
        Select the most informative features for the association matrix.

        Uses composite scoring: IV ranking (or association strength without target),
        cardinality penalty for high-cardinality categoricals, and redundancy pruning
        to remove near-duplicate features.

        Args:
            feature_names: All candidate feature names
            features_data: Per-feature analysis results dict
            target_variable: Target variable name (or None)
            correlation_matrix: Pearson correlation matrix (cont×cont)
            theils_u_matrix: Theil's U matrix (cat×cat)
            eta_matrix: Eta dict (cat↔cont)
            max_features: Maximum features to select

        Returns:
            List of selected feature names
        """
        # Always include target variable
        selected_target = None
        if target_variable and target_variable in feature_names:
            selected_target = target_variable

        # Quality filter: remove constant or not-recommended features
        candidates = []
        for fname in feature_names:
            if fname == target_variable:
                continue
            fdata = features_data.get(fname, {})
            quality = fdata.get("quality", {})
            if quality.get("is_constant", False):
                continue
            if quality.get("recommended_for_modeling") is False:
                continue
            candidates.append(fname)

        # Score each candidate
        scored = []
        for fname in candidates:
            fdata = features_data.get(fname, {})
            if target_variable:
                score = self._score_with_target(fdata)
            else:
                score = self._score_without_target(
                    fname, correlation_matrix, theils_u_matrix, eta_matrix
                )
            score = self._apply_cardinality_penalty(score, fdata)
            scored.append((fname, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Redundancy pruning: greedy forward selection
        budget = max_features - (1 if selected_target else 0)
        selected = []
        for fname, _score in scored:
            if len(selected) >= budget:
                break
            if not self._is_redundant(
                fname, selected, correlation_matrix, theils_u_matrix, eta_matrix
            ):
                selected.append(fname)

        # Backfill if pruning removed too many
        if len(selected) < budget:
            selected_set = set(selected)
            for fname, _score in scored:
                if len(selected) >= budget:
                    break
                if fname not in selected_set:
                    selected.append(fname)
                    selected_set.add(fname)

        result = ([selected_target] if selected_target else []) + selected
        logger.info(
            "Association matrix feature selection: %d/%d features selected",
            len(result), len(feature_names),
        )
        return result

    @staticmethod
    def _score_with_target(fdata: Dict[str, Any]) -> float:
        """Score a feature by its relationship with the target variable.

        Priority: IV > |Pearson| > |Spearman| > Theil's U > Eta > 0.0
        """
        tr = fdata.get("target_relationship", {})
        if not tr:
            return 0.0

        iv = tr.get("iv")
        if iv is not None and not (isinstance(iv, float) and np.isnan(iv)):
            return float(iv)

        pearson = tr.get("pearson_correlation")
        if pearson is not None and not (isinstance(pearson, float) and np.isnan(pearson)):
            return abs(float(pearson))

        spearman = tr.get("spearman_correlation")
        if spearman is not None and not (isinstance(spearman, float) and np.isnan(spearman)):
            return abs(float(spearman))

        theils_u = tr.get("theils_u")
        if theils_u is not None and not (isinstance(theils_u, float) and np.isnan(theils_u)):
            return float(theils_u)

        eta = tr.get("eta")
        if eta is not None and not (isinstance(eta, float) and np.isnan(eta)):
            return float(eta)

        return 0.0

    @staticmethod
    def _score_without_target(
        fname: str,
        correlation_matrix: Optional[pd.DataFrame],
        theils_u_matrix: Optional[pd.DataFrame],
        eta_matrix: Optional[Dict[str, Dict[str, float]]],
    ) -> float:
        """Score a feature by average absolute association strength with all other features."""
        values = []

        if correlation_matrix is not None and fname in correlation_matrix.columns:
            row = correlation_matrix[fname].drop(fname, errors="ignore").dropna()
            values.extend(row.abs().tolist())

        if theils_u_matrix is not None and fname in theils_u_matrix.index:
            row = theils_u_matrix.loc[fname].drop(fname, errors="ignore").dropna()
            values.extend(row.abs().tolist())

        if eta_matrix and fname in eta_matrix:
            values.extend(abs(v) for v in eta_matrix[fname].values())

        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def _apply_cardinality_penalty(score: float, fdata: Dict[str, Any]) -> float:
        """Penalize high-cardinality categorical features.

        High cardinality (ratio > 0.5): score × 0.3
        Medium cardinality (0.1–0.5): score × 0.7
        Low cardinality or continuous: no penalty
        """
        # Only penalize categorical features — continuous features naturally
        # have high cardinality ratios and should not be penalized for it.
        ftype = fdata.get("type", "")
        if ftype not in ("categorical",):
            return score

        cardinality = fdata.get("cardinality")
        if cardinality is None:
            return score

        ratio = cardinality.get("cardinality_ratio")
        if ratio is None:
            return score

        if ratio > 0.5:
            return score * 0.3
        elif ratio > 0.1:
            return score * 0.7
        return score

    @staticmethod
    def _is_redundant(
        fname: str,
        selected: List[str],
        correlation_matrix: Optional[pd.DataFrame],
        theils_u_matrix: Optional[pd.DataFrame],
        eta_matrix: Optional[Dict[str, Dict[str, float]]],
        threshold: float = 0.9,
    ) -> bool:
        """Check if a feature is redundant with any already-selected feature.

        Returns True if any association (Pearson, Theil's U in both directions, or Eta)
        exceeds the threshold with an already-selected feature.
        """
        for sel in selected:
            # Pearson
            if correlation_matrix is not None:
                if fname in correlation_matrix.columns and sel in correlation_matrix.columns:
                    val = correlation_matrix.loc[fname, sel]
                    if not (isinstance(val, float) and np.isnan(val)) and abs(val) > threshold:
                        return True

            # Theil's U: both directions
            if theils_u_matrix is not None:
                if fname in theils_u_matrix.index and sel in theils_u_matrix.columns:
                    val = theils_u_matrix.loc[fname, sel]
                    if not (isinstance(val, float) and np.isnan(val)) and abs(val) > threshold:
                        return True
                if sel in theils_u_matrix.index and fname in theils_u_matrix.columns:
                    val = theils_u_matrix.loc[sel, fname]
                    if not (isinstance(val, float) and np.isnan(val)) and abs(val) > threshold:
                        return True

            # Eta
            if eta_matrix:
                if fname in eta_matrix and sel in eta_matrix[fname]:
                    if abs(eta_matrix[fname][sel]) > threshold:
                        return True
                if sel in eta_matrix and fname in eta_matrix[sel]:
                    if abs(eta_matrix[sel][fname]) > threshold:
                        return True

        return False

    @staticmethod
    def _theils_u(x: pd.Series, y: pd.Series) -> float:
        """
        Compute Theil's U (Uncertainty Coefficient) U(X→Y).

        Measures how much knowing X reduces uncertainty about Y.
        Asymmetric: U(X→Y) != U(Y→X). Range [0, 1].

        Args:
            x: Categorical series (predictor)
            y: Categorical series (response)

        Returns:
            Theil's U value, or 0.0 if undefined
        """
        # Drop rows where either value is NaN
        mask = x.notna() & y.notna()
        x, y = x[mask], y[mask]
        if len(x) == 0:
            return 0.0

        # Contingency table
        contingency = pd.crosstab(x, y)
        n = contingency.values.sum()
        if n == 0:
            return 0.0

        # H(Y) — entropy of y
        py = contingency.values.sum(axis=0) / n
        py = py[py > 0]
        h_y = -np.sum(py * np.log(py))
        if h_y == 0:
            return 0.0

        # H(Y|X) — conditional entropy of y given x
        px = contingency.values.sum(axis=1) / n
        h_y_given_x = 0.0
        for i in range(contingency.shape[0]):
            row = contingency.values[i]
            row_sum = row.sum()
            if row_sum == 0:
                continue
            p_y_given_xi = row / row_sum
            p_y_given_xi = p_y_given_xi[p_y_given_xi > 0]
            h_y_given_x -= px[i] * np.sum(p_y_given_xi * np.log(p_y_given_xi))

        u = (h_y - h_y_given_x) / h_y
        return max(0.0, min(1.0, float(u)))

    @staticmethod
    def _correlation_ratio(categories: pd.Series, measurements: pd.Series) -> float:
        """
        Compute Correlation Ratio (Eta) between a categorical and continuous variable.

        Eta = sqrt(SS_between / SS_total). Range [0, 1].

        Args:
            categories: Categorical series
            measurements: Continuous (numeric) series

        Returns:
            Correlation ratio value, or 0.0 if undefined
        """
        mask = categories.notna() & measurements.notna()
        categories, measurements = categories[mask], measurements[mask]
        # Filter out infinite values
        finite_mask = np.isfinite(measurements)
        categories, measurements = categories[finite_mask], measurements[finite_mask]
        if len(categories) == 0:
            return 0.0

        grand_mean = measurements.mean()
        ss_total = np.sum((measurements.values - grand_mean) ** 2)
        if ss_total == 0:
            return 0.0

        ss_between = 0.0
        for _, group in measurements.groupby(categories):
            group_mean = group.mean()
            ss_between += len(group) * (group_mean - grand_mean) ** 2

        ratio = ss_between / ss_total
        return float(max(0.0, min(1.0, math.sqrt(ratio))))

    def compute_association_matrices(
        self,
        df: pd.DataFrame,
        col_configs: Optional[Dict[str, ColumnConfig]] = None,
    ) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Dict[str, float]]]]:
        """
        Compute Theil's U matrix (cat↔cat) and Eta dict (cat↔cont).

        Args:
            df: Input dataframe
            col_configs: Column configs to determine types

        Returns:
            (theils_u_matrix, eta_matrix) where:
            - theils_u_matrix: asymmetric DataFrame indexed by categorical cols, or None
            - eta_matrix: nested dict {col: {other_col: eta}} for cat↔cont pairs, or None
        """
        # Classify columns
        cat_cols: List[str] = []
        cont_cols: List[str] = []
        for col in df.columns:
            config = col_configs.get(col) if col_configs else None
            if config is not None:
                if config.type is ColumnType.CONTINUOUS:
                    cont_cols.append(col)
                else:
                    cat_cols.append(col)
            elif df[col].dtype.kind in "iufb":
                cont_cols.append(col)
            else:
                cat_cols.append(col)

        theils_u_matrix = None
        eta_matrix = None

        # Theil's U: categorical × categorical
        if len(cat_cols) >= 2:
            logger.info("Computing Theil's U matrix for %s categorical features", len(cat_cols))
            data = {}
            for cx in cat_cols:
                row = {}
                is_constant_x = df[cx].dropna().nunique() <= 1
                for cy in cat_cols:
                    if cx == cy:
                        # Constant column has zero entropy → U is undefined; use 0.0
                        row[cy] = 0.0 if is_constant_x else 1.0
                    else:
                        row[cy] = self._theils_u(df[cx], df[cy])
                data[cx] = row
            theils_u_matrix = pd.DataFrame(data).T  # rows = X, cols = Y

        # Eta: categorical ↔ continuous
        if cat_cols and cont_cols:
            logger.info(
                "Computing Correlation Ratio (Eta) for %s categorical × %s continuous features",
                len(cat_cols), len(cont_cols),
            )
            eta_matrix = {}
            for cat in cat_cols:
                for cont in cont_cols:
                    val = self._correlation_ratio(df[cat], df[cont])
                    eta_matrix.setdefault(cat, {})[cont] = val
                    eta_matrix.setdefault(cont, {})[cat] = val

        return theils_u_matrix, eta_matrix

    @staticmethod
    def build_association_matrix(
        feature_names: List[str],
        correlation_matrix: Optional[pd.DataFrame] = None,
        theils_u_matrix: Optional[pd.DataFrame] = None,
        eta_matrix: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Merge precomputed Pearson, Theil's U, and Eta into a single N×N
        structure suitable for JSON serialization and heatmap rendering.

        Args:
            feature_names: Ordered list of features (determines row/col order)
            correlation_matrix: Pearson correlations (cont×cont)
            theils_u_matrix: Theil's U (cat×cat, asymmetric)
            eta_matrix: Correlation Ratio dict (cat↔cont)

        Returns:
            Dict with ``features``, ``values`` (N×N), and ``methods`` (N×N),
            or None if no data is available.
        """
        n = len(feature_names)
        if n == 0:
            return None

        values: List[List[Optional[float]]] = [[None] * n for _ in range(n)]
        methods: List[List[str]] = [[""] * n for _ in range(n)]

        idx = {name: i for i, name in enumerate(feature_names)}

        # Pearson (symmetric, range -1..1)
        if correlation_matrix is not None:
            for row_name in correlation_matrix.index:
                if row_name not in idx:
                    continue
                ri = idx[row_name]
                for col_name in correlation_matrix.columns:
                    if col_name not in idx or row_name == col_name:
                        continue
                    ci = idx[col_name]
                    val = correlation_matrix.loc[row_name, col_name]
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        values[ri][ci] = safe_round(float(val), 4)
                        methods[ri][ci] = "pearson"

        # Theil's U (asymmetric, range 0..1)
        # values[i][j] = U(feature_i → feature_j)
        if theils_u_matrix is not None:
            for row_name in theils_u_matrix.index:
                if row_name not in idx:
                    continue
                ri = idx[row_name]
                for col_name in theils_u_matrix.columns:
                    if col_name not in idx or row_name == col_name:
                        continue
                    ci = idx[col_name]
                    val = theils_u_matrix.loc[row_name, col_name]
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        values[ri][ci] = safe_round(float(val), 4)
                        methods[ri][ci] = "theil_u"

        # Eta / Correlation Ratio (symmetric, range 0..1)
        if eta_matrix:
            for row_name, row_dict in eta_matrix.items():
                if row_name not in idx:
                    continue
                ri = idx[row_name]
                for col_name, val in row_dict.items():
                    if col_name not in idx or row_name == col_name:
                        continue
                    ci = idx[col_name]
                    # Don't overwrite Pearson or Theil's U already placed
                    if methods[ri][ci] == "":
                        values[ri][ci] = safe_round(float(val), 4)
                        methods[ri][ci] = "eta"

        # Check if we produced anything
        has_data = any(values[i][j] is not None for i in range(n) for j in range(n))
        if not has_data:
            return None

        return {
            "features": feature_names,
            "values": values,
            "methods": methods,
        }

    def get_feature_correlations(
        self,
        feature_name: str,
        correlation_matrix: Optional[pd.DataFrame],
        precomputed_correlations: Optional[Dict[str, Any]],
        target_variable: Optional[str] = None,
        theils_u_matrix: Optional[pd.DataFrame] = None,
        eta_matrix: Optional[Dict[str, Dict[str, float]]] = None,
        column_type: Optional[ColumnType] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get correlation information for a feature.

        Merges Pearson (cont↔cont), Theil's U (cat↔cat), and Eta (cat↔cont)
        into a single sorted list with a ``method`` field on each entry.

        Args:
            feature_name: Name of the feature
            correlation_matrix: Precomputed Pearson correlation matrix
            precomputed_correlations: Precomputed correlation data
            target_variable: Optional target variable name
            theils_u_matrix: Theil's U matrix (cat↔cat)
            eta_matrix: Eta dict (cat↔cont and cont↔cat)
            column_type: Resolved ColumnType for this feature

        Returns:
            Dictionary with correlation data or None
        """
        correlations: List[Dict[str, Any]] = []
        seen: set = set()

        # --- Pearson (precomputed or matrix) ---
        if precomputed_correlations:
            precomp = self._get_precomputed_correlations(
                feature_name, precomputed_correlations, target_variable
            )
            if precomp:
                for entry in precomp:
                    entry.setdefault("method", "pearson")
                    seen.add(entry["feature"])
                correlations.extend(precomp)

        if correlation_matrix is not None and feature_name in correlation_matrix.columns:
            pearson = self._compute_feature_correlations(
                feature_name, correlation_matrix, target_variable
            )
            for entry in pearson:
                if entry["feature"] not in seen:
                    entry["method"] = "pearson"
                    seen.add(entry["feature"])
                    correlations.append(entry)

        # --- Theil's U (cat↔cat) ---
        # Forward: U(this_feature → other) — "this feature provides information on..."
        if theils_u_matrix is not None and feature_name in theils_u_matrix.index:
            row = theils_u_matrix.loc[feature_name].drop(feature_name, errors="ignore").dropna()
            for feat, val in row.items():
                if feat not in seen and val > 0:
                    correlations.append({
                        "feature": feat,
                        "correlation": safe_round(float(val), 4),
                        "method": "theil_u",
                    })
                    seen.add(feat)

        # Reverse: U(other → this_feature) — "these features give information on this feature"
        if theils_u_matrix is not None and feature_name in theils_u_matrix.columns:
            col = theils_u_matrix[feature_name].drop(feature_name, errors="ignore").dropna()
            for feat, val in col.items():
                if val > 0:
                    correlations.append({
                        "feature": feat,
                        "correlation": safe_round(float(val), 4),
                        "method": "theil_u_reverse",
                    })

        # --- Eta (cat↔cont) ---
        if eta_matrix and feature_name in eta_matrix:
            for feat, val in eta_matrix[feature_name].items():
                if feat not in seen and val > 0:
                    correlations.append({
                        "feature": feat,
                        "correlation": safe_round(float(val), 4),
                        "method": "eta",
                    })
                    seen.add(feat)

        if not correlations:
            return None

        # Sort by absolute value (high→low) and trim to top_correlations
        correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        correlations = correlations[: self.top_correlations]

        return {"top_correlated_features": correlations}

    def _get_precomputed_correlations(
        self,
        feature_name: str,
        precomputed_correlations: Dict[str, Any],
        target_variable: Optional[str] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get precomputed correlations from metadata.

        Args:
            feature_name: Name of the feature
            precomputed_correlations: Dictionary of precomputed correlations
            target_variable: Optional target variable name

        Returns:
            List of correlation dictionaries sorted by absolute value (high to low) or None if not found
        """
        if feature_name not in precomputed_correlations:
            return None

        feature_corr = precomputed_correlations[feature_name]

        # Check if it has the expected structure
        if not isinstance(feature_corr, dict):
            return None

        correlations = []

        # Add target correlation if available and target is specified
        if target_variable and "target_correlation" in feature_corr:
            correlations.append(
                {
                    "feature": target_variable,
                    "correlation": safe_round(feature_corr["target_correlation"], 4),
                }
            )

        # Add top correlations if available
        if "top_correlations" in feature_corr:
            top_corr = feature_corr["top_correlations"]
            if isinstance(top_corr, list):
                correlations.extend(top_corr)
            elif isinstance(top_corr, dict):
                # Convert dict format to list format
                for feat, corr_val in top_corr.items():
                    if feat != feature_name:  # Exclude self-correlation
                        correlations.append({"feature": feat, "correlation": safe_round(corr_val, 4)})

        # Sort by absolute value of correlation (high to low), preserving the sign
        correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        return correlations if correlations else None

    def _compute_feature_correlations(
        self, feature_name: str, correlation_matrix: pd.DataFrame, target_variable: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Compute correlations for a feature from correlation matrix.

        Args:
            feature_name: Name of the feature
            correlation_matrix: Precomputed correlation matrix
            target_variable: Optional target variable name

        Returns:
            List of top correlations sorted by absolute value (high to low), preserving sign
        """
        # Get all correlations for this feature (excluding self)
        feature_corr = correlation_matrix[feature_name].drop(feature_name, errors="ignore")

        # Remove NaN values
        feature_corr = feature_corr.dropna()

        # Get top correlations by absolute value (sorted high to low)
        top_corr = feature_corr.abs().nlargest(self.top_correlations)

        correlations = []
        for feat in top_corr.index:
            # Keep original sign of correlation
            original_corr = correlation_matrix.loc[feat, feature_name]
            correlations.append({"feature": feat, "correlation": safe_round(original_corr, 4)})

        # Already sorted by absolute value from nlargest(), but ensuring it's explicit
        # This maintains high to low order by absolute value while preserving signs
        return correlations
