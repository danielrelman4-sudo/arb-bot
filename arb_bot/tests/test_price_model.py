"""Tests for Monte Carlo price model."""

from __future__ import annotations

import math

import numpy as np
import pytest

from arb_bot.crypto.price_model import PriceModel, ProbabilityEstimate


class TestProbabilityEstimate:
    def test_frozen(self) -> None:
        pe = ProbabilityEstimate(0.5, 0.45, 0.55, 0.05, 1000)
        with pytest.raises(AttributeError):
            pe.probability = 0.6  # type: ignore[misc]


class TestEstimateVolatility:
    def test_constant_price_zero_vol(self) -> None:
        model = PriceModel(seed=42)
        returns = [0.0] * 30
        vol = model.estimate_volatility(returns, interval_seconds=60)
        assert vol == 0.0

    def test_positive_returns_positive_vol(self) -> None:
        model = PriceModel(seed=42)
        rng = np.random.default_rng(42)
        returns = list(rng.normal(0.0, 0.001, 100))
        vol = model.estimate_volatility(returns, interval_seconds=60)
        assert vol > 0

    def test_too_few_returns(self) -> None:
        model = PriceModel(seed=42)
        assert model.estimate_volatility([0.001]) == 0.0
        assert model.estimate_volatility([]) == 0.0

    def test_annualization_scales(self) -> None:
        """Smaller interval → more observations/year → higher annualized vol."""
        model = PriceModel(seed=42)
        returns = [0.001, -0.0005, 0.002, -0.001, 0.0015] * 10
        vol_60s = model.estimate_volatility(returns, interval_seconds=60)
        vol_300s = model.estimate_volatility(returns, interval_seconds=300)
        # Same std, but 60s has sqrt(5x) more observations → higher annualized
        assert vol_60s > vol_300s


class TestGeneratePaths:
    def test_shape(self) -> None:
        model = PriceModel(num_paths=500, seed=42)
        paths = model.generate_paths(97000.0, 0.60, 15.0)
        assert paths.shape == (500,)

    def test_num_paths_override(self) -> None:
        model = PriceModel(num_paths=100, seed=42)
        paths = model.generate_paths(97000.0, 0.60, 15.0, num_paths=50)
        assert paths.shape == (50,)

    def test_all_positive(self) -> None:
        model = PriceModel(num_paths=1000, seed=42)
        paths = model.generate_paths(97000.0, 0.60, 15.0)
        assert np.all(paths > 0)

    def test_zero_vol_returns_current(self) -> None:
        model = PriceModel(num_paths=100, seed=42)
        paths = model.generate_paths(97000.0, 0.0, 15.0)
        assert np.allclose(paths, 97000.0)

    def test_zero_horizon_returns_current(self) -> None:
        model = PriceModel(num_paths=100, seed=42)
        paths = model.generate_paths(97000.0, 0.60, 0.0)
        assert np.allclose(paths, 97000.0)

    def test_mean_near_current_zero_drift(self) -> None:
        """With zero drift, mean terminal price ≈ current price."""
        model = PriceModel(num_paths=50000, seed=42)
        paths = model.generate_paths(97000.0, 0.60, 15.0)
        mean_price = float(np.mean(paths))
        # Should be within ~2% of current price for 15-min horizon
        assert abs(mean_price - 97000.0) / 97000.0 < 0.02

    def test_deterministic_with_seed(self) -> None:
        model1 = PriceModel(num_paths=100, seed=123)
        model2 = PriceModel(num_paths=100, seed=123)
        p1 = model1.generate_paths(97000.0, 0.60, 15.0)
        p2 = model2.generate_paths(97000.0, 0.60, 15.0)
        np.testing.assert_array_equal(p1, p2)

    def test_different_seeds_different_paths(self) -> None:
        model1 = PriceModel(num_paths=100, seed=1)
        model2 = PriceModel(num_paths=100, seed=2)
        p1 = model1.generate_paths(97000.0, 0.60, 15.0)
        p2 = model2.generate_paths(97000.0, 0.60, 15.0)
        assert not np.allclose(p1, p2)

    def test_higher_vol_wider_distribution(self) -> None:
        model = PriceModel(num_paths=10000, seed=42)
        low_vol = model.generate_paths(97000.0, 0.30, 60.0)
        # Recreate for fresh RNG state
        model2 = PriceModel(num_paths=10000, seed=42)
        high_vol = model2.generate_paths(97000.0, 1.20, 60.0)
        assert float(np.std(high_vol)) > float(np.std(low_vol))


class TestProbabilityAbove:
    def test_all_above(self) -> None:
        model = PriceModel(seed=42)
        paths = np.array([100.0, 110.0, 120.0, 130.0])
        pe = model.probability_above(paths, 50.0)
        assert pe.probability == 1.0
        assert pe.num_paths == 4

    def test_none_above(self) -> None:
        model = PriceModel(seed=42)
        paths = np.array([100.0, 110.0, 120.0])
        pe = model.probability_above(paths, 200.0)
        assert pe.probability == 0.0

    def test_half_above(self) -> None:
        model = PriceModel(seed=42)
        paths = np.array([90.0, 95.0, 105.0, 110.0])
        pe = model.probability_above(paths, 100.0)
        assert pe.probability == 0.5

    def test_ci_bounds(self) -> None:
        model = PriceModel(seed=42)
        paths = np.array([90.0, 95.0, 105.0, 110.0])
        pe = model.probability_above(paths, 100.0)
        assert 0.0 <= pe.ci_lower <= pe.probability
        assert pe.probability <= pe.ci_upper <= 1.0
        assert pe.uncertainty >= 0.0

    def test_large_sample_tight_ci(self) -> None:
        model = PriceModel(num_paths=10000, seed=42)
        paths = model.generate_paths(100.0, 0.60, 15.0)
        pe = model.probability_above(paths, 100.0)
        # With 10k paths, CI should be tight
        assert pe.uncertainty < 0.02

    def test_empty_paths(self) -> None:
        model = PriceModel(seed=42)
        pe = model.probability_above(np.array([]), 100.0)
        assert pe.probability == 0.5
        assert pe.num_paths == 0


class TestProbabilityUp:
    def test_roughly_half(self) -> None:
        """With zero drift, P(up) ≈ 0.50."""
        model = PriceModel(num_paths=10000, seed=42)
        paths = model.generate_paths(97000.0, 0.60, 15.0)
        pe = model.probability_up(paths, 97000.0)
        assert abs(pe.probability - 0.5) < 0.05


class TestProbabilityBelow:
    def test_complement(self) -> None:
        model = PriceModel(seed=42)
        paths = np.array([90.0, 95.0, 105.0, 110.0])
        above = model.probability_above(paths, 100.0)
        below = model.probability_below(paths, 100.0)
        assert abs(above.probability + below.probability - 1.0) < 1e-10


class TestWilsonCI:
    def test_extreme_p_zero(self) -> None:
        model = PriceModel(seed=42)
        pe = model._wilson_ci(0.0, 100)
        assert pe.ci_lower >= 0.0
        assert pe.ci_upper > 0.0  # Wilson interval is non-zero at extremes

    def test_extreme_p_one(self) -> None:
        model = PriceModel(seed=42)
        pe = model._wilson_ci(1.0, 100)
        assert pe.ci_lower < 1.0
        assert pe.ci_upper <= 1.0

    def test_larger_n_tighter(self) -> None:
        model = PriceModel(seed=42)
        small = model._wilson_ci(0.5, 50)
        large = model._wilson_ci(0.5, 5000)
        assert large.uncertainty < small.uncertainty
