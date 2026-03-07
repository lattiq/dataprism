"""Tests for dataprism.woe — compute_woe_iv and WoEResult."""

import numpy as np
import pandas as pd
import pytest

from dataprism.woe import WoEResult, compute_woe_iv, _classify_predictive_power


@pytest.fixture
def binary_data():
    """Simple dataset with known WoE properties."""
    rng = np.random.default_rng(42)
    n = 500
    cats = rng.choice(["A", "B", "C"], n)
    # Make A mostly good, C mostly bad
    target = np.where(cats == "A", rng.choice([0, 1], n, p=[0.8, 0.2]),
             np.where(cats == "C", rng.choice([0, 1], n, p=[0.3, 0.7]),
                      rng.choice([0, 1], n, p=[0.5, 0.5])))
    return pd.Series(cats, name="feature"), pd.Series(target, name="target")


class TestComputeWoeIv:
    def test_returns_woe_result(self, binary_data):
        feature, target = binary_data
        result = compute_woe_iv(feature, target)
        assert isinstance(result, WoEResult)
        assert set(result.woe_mapping.keys()) == {"A", "B", "C"}
        assert set(result.iv_contributions.keys()) == {"A", "B", "C"}
        assert result.iv > 0
        assert result.predictive_power in (
            "unpredictive", "weak", "medium", "strong", "very_strong"
        )

    def test_woe_direction(self, binary_data):
        """A (mostly good=0) should have positive WoE; C (mostly bad=1) should have negative."""
        feature, target = binary_data
        result = compute_woe_iv(feature, target)
        assert result.woe_mapping["A"] > 0
        assert result.woe_mapping["C"] < 0

    def test_iv_sum_matches_contributions(self, binary_data):
        feature, target = binary_data
        result = compute_woe_iv(feature, target)
        assert abs(result.iv - sum(result.iv_contributions.values())) < 1e-10

    def test_laplace_smoothing_default(self, binary_data):
        feature, target = binary_data
        result_default = compute_woe_iv(feature, target)
        result_laplace = compute_woe_iv(feature, target, smoothing=0.5)
        assert result_default.woe_mapping == result_laplace.woe_mapping

    def test_epsilon_smoothing(self, binary_data):
        feature, target = binary_data
        result = compute_woe_iv(feature, target, smoothing=0)
        assert isinstance(result, WoEResult)
        assert result.iv > 0

    def test_different_smoothing_gives_different_results(self, binary_data):
        feature, target = binary_data
        r1 = compute_woe_iv(feature, target, smoothing=0.5)
        r2 = compute_woe_iv(feature, target, smoothing=0)
        # Values should differ due to different smoothing
        assert r1.woe_mapping["A"] != r2.woe_mapping["A"]

    def test_returns_none_single_target_class(self):
        feature = pd.Series(["A", "B", "C"])
        target = pd.Series([0, 0, 0])
        result = compute_woe_iv(feature, target)
        assert result is None

    def test_returns_none_single_crosstab_column(self):
        feature = pd.Series(["A", "A", "A"])
        target = pd.Series([1, 1, 1])
        result = compute_woe_iv(feature, target)
        assert result is None

    def test_epsilon_handles_zero_count(self):
        """With epsilon smoothing, a category with zero goods should not fail."""
        feature = pd.Series(["A", "A", "A", "B", "B"])
        target = pd.Series([0, 0, 0, 1, 1])
        # B has 0 goods — epsilon floor should kick in
        result = compute_woe_iv(feature, target, smoothing=0)
        assert result is not None
        assert "B" in result.woe_mapping

    def test_woe_mapping_keys_are_strings(self, binary_data):
        feature, target = binary_data
        result = compute_woe_iv(feature, target)
        for k in result.woe_mapping:
            assert isinstance(k, str)

    def test_numeric_categories(self):
        feature = pd.Series([1, 2, 3, 1, 2, 3])
        target = pd.Series([0, 0, 1, 1, 0, 1])
        result = compute_woe_iv(feature, target)
        assert result is not None
        # Keys should be string representations
        assert all(isinstance(k, str) for k in result.woe_mapping)


class TestClassifyPredictivePower:
    @pytest.mark.parametrize("iv,expected", [
        (0.0, "unpredictive"),
        (0.01, "unpredictive"),
        (0.02, "weak"),
        (0.05, "weak"),
        (0.1, "medium"),
        (0.25, "medium"),
        (0.3, "strong"),
        (0.45, "strong"),
        (0.5, "very_strong"),
        (1.0, "very_strong"),
    ])
    def test_thresholds(self, iv, expected):
        assert _classify_predictive_power(iv) == expected
