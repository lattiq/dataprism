"""Maps analysis results to the EDA output schema."""

from typing import Any, Dict, Optional

from dataprism.schema import ColumnConfig


class ResultMapper:
    """Maps analyzer output format to final output schema format."""

    @staticmethod
    def map_to_output_schema(
        feature_data: Dict[str, Any],
        feature_name: str,
        col_metadata: Optional[ColumnConfig] = None,
    ) -> Dict[str, Any]:
        """
        Transform analyzer output to final schema in a single pass.

        Args:
            feature_data: Raw output from analyzer
            feature_name: Name of the feature
            col_metadata: Optional feature metadata

        Returns:
            Feature data in final output schema format
        """
        # Extract commonly used values once
        stats = feature_data.get("stats", {})
        missing = feature_data.get("missing", {})
        feature_type = feature_data.get("type", "categorical")

        # Build output schema efficiently
        output = {
            "feature_name": feature_name,
            "display_name": feature_data.get("description") or feature_name,
            "description": feature_data.get("description", ""),
            "type": "Continuous" if feature_type == "continuous" else "Categorical",
            "data_type": feature_data.get("data_type", "object"),
            "is_derived": False,
            "is_target": feature_data.get("is_target", False),
        }

        # Add source section
        output["source"] = {
            "type": "provider",
            "provider": feature_data.get("provider"),
        }

        # Add config section
        output["config"] = {
            "missing_sentinel": feature_data.get("missing_sentinel"),
            "not_found_sentinel": feature_data.get("not_found_sentinel"),
        }

        # Build statistics section efficiently
        # Note: stats already contains all analyzer stats including:
        # - count, unique, mode, etc. (categorical)
        # - count, mean, std, min, max, etc. (continuous)
        # - unique_values and unique_ratio (both types)
        # Only add missing-related stats which come from a separate dict
        non_null_count = stats.get("count", 0)
        missing_count = missing.get("count", 0)
        output["statistics"] = {
            **stats,  # Include all original stats from analyzers (count = non-null)
            "total_count": non_null_count + missing_count,
            "missing_count": missing_count,
            "missing_percentage": missing.get("percent", 0.0),
        }

        # Add optional sections if present and not None
        if "outliers" in feature_data and feature_data["outliers"] is not None:
            output["outliers"] = feature_data["outliers"]

        if "cardinality" in feature_data and feature_data["cardinality"] is not None:
            output["cardinality"] = feature_data["cardinality"]

        if "distribution" in feature_data and feature_data["distribution"] is not None:
            output["distribution"] = feature_data["distribution"]

        if "target_relationship" in feature_data and feature_data["target_relationship"] is not None:
            output["target_relationship"] = feature_data["target_relationship"]

        if "correlations" in feature_data and feature_data["correlations"] is not None:
            output["correlations"] = feature_data["correlations"]

        if "quality" in feature_data and feature_data["quality"] is not None:
            output["quality"] = feature_data["quality"]

        return output
