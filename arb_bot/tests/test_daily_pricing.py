"""Tests for v43 daily pricing model methods (price_model.py additions).

Covers:
  1A: variance_ratio, realized_vol_at_horizon, vol_with_variance_ratio_correction
  1B: probability_above_merton_series
  1C: compute_probability_bounds, clamp_probability
  2A: probability_above_ou, estimate_ou_params
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from arb_bot.crypto.price_model import PriceModel, ProbabilityEstimate


@pytest.fixture
def model() -> PriceModel:
    return PriceModel(num_paths=500, seed=42)


# ── 1A: Multi-scale realized vol ─────────────────────────────────


class TestVarianceRatio:
    """variance_ratio()"""

    def test_iid_returns_vr_near_one(self, model: PriceModel) -> None:
        """IID normal returns should give VR ≈ 1.0."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.001, 5000).tolist()
        vr = model.variance_ratio(returns, q=30)
        # Should be within ~10% of 1.0 for large samples
        assert 0.85 <= vr <= 1.15, f"VR={vr} not near 1.0 for IID"

    def test_mean_reverting_returns_vr_below_one(self, model: PriceModel) -> None:
        """Alternating (mean-reverting) returns should give VR < 1."""
        n = 3000
        # Strongly mean-reverting: each return is negatively correlated with previous
        returns = []
        r = 0.001
        for _ in range(n):
            returns.append(r)
            r = -r * 0.8 + np.random.default_rng(42).normal(0, 0.0001)
        vr = model.variance_ratio(returns, q=10)
        assert vr < 0.8, f"VR={vr} should be < 0.8 for mean-reverting"

    def test_trending_returns_vr_above_one(self, model: PriceModel) -> None:
        """Positively autocorrelated returns should give VR > 1."""
        rng = np.random.default_rng(42)
        n = 3000
        # Trending: cumulative momentum
        returns = []
        r = 0.0
        for _ in range(n):
            r = 0.7 * r + rng.normal(0, 0.001)
            returns.append(r)
        vr = model.variance_ratio(returns, q=10)
        assert vr > 1.2, f"VR={vr} should be > 1.2 for trending"

    def test_insufficient_data_returns_one(self, model: PriceModel) -> None:
        """Too few data points should return VR=1.0 (random walk default)."""
        vr = model.variance_ratio([0.01, 0.02, 0.03], q=5)
        assert vr == 1.0

    def test_q_less_than_two_returns_one(self, model: PriceModel) -> None:
        vr = model.variance_ratio([0.01] * 100, q=1)
        assert vr == 1.0

    def test_clamped_range(self, model: PriceModel) -> None:
        """VR should be clamped to [0.01, 5.0]."""
        vr = model.variance_ratio([0.01] * 200, q=5)
        assert 0.01 <= vr <= 5.0


class TestRealizedVolAtHorizon:
    """realized_vol_at_horizon()"""

    def test_returns_annualized_vol(self, model: PriceModel) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.001, 600).tolist()  # 600 1-min returns
        vol = model.realized_vol_at_horizon(returns, horizon_minutes=30)
        assert vol is not None
        assert vol > 0

    def test_insufficient_data_returns_none(self, model: PriceModel) -> None:
        returns = [0.001] * 50  # Only 50 returns, need 10 blocks of 30 = 300
        vol = model.realized_vol_at_horizon(returns, horizon_minutes=30)
        assert vol is None

    def test_consistent_with_direct_vol(self, model: PriceModel) -> None:
        """For IID returns, realized_vol_at_horizon ≈ √T-scaled vol."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.001, 3000).tolist()
        rv_vol = model.realized_vol_at_horizon(returns, horizon_minutes=10)
        direct_vol = model.estimate_volatility(returns, interval_seconds=60)
        assert rv_vol is not None
        # Should be within 20% for IID
        ratio = rv_vol / direct_vol if direct_vol > 0 else 0
        assert 0.7 < ratio < 1.3, f"ratio={ratio}"


class TestVolWithVRCorrection:
    """vol_with_variance_ratio_correction()"""

    def test_vr_one_no_change(self) -> None:
        vol = PriceModel.vol_with_variance_ratio_correction(0.5, 30.0, 1.0)
        assert abs(vol - 0.5) < 1e-10

    def test_vr_below_one_reduces_vol(self) -> None:
        vol = PriceModel.vol_with_variance_ratio_correction(0.5, 30.0, 0.7)
        assert vol < 0.5

    def test_vr_above_one_increases_vol(self) -> None:
        vol = PriceModel.vol_with_variance_ratio_correction(0.5, 30.0, 1.5)
        assert vol > 0.5


# ── 1B: Merton jump-diffusion series ────────────────────────────


class TestMertonSeries:
    """probability_above_merton_series()"""

    def test_atm_returns_near_half(self, model: PriceModel) -> None:
        """ATM contract with zero drift should give P ≈ 0.5."""
        est = model.probability_above_merton_series(
            current_price=100_000, strike=100_000,
            volatility=0.50, horizon_minutes=30.0,
            drift=0.0, lambda_daily=3.0,
        )
        assert 0.4 <= est.probability <= 0.6, f"P={est.probability}"
        assert est.num_paths == 0  # analytical

    def test_deep_itm_high_probability(self, model: PriceModel) -> None:
        """Deep ITM: price >> strike → P close to 1."""
        est = model.probability_above_merton_series(
            current_price=100_000, strike=95_000,
            volatility=0.30, horizon_minutes=30.0,
        )
        assert est.probability > 0.8, f"P={est.probability}"

    def test_deep_otm_low_probability(self, model: PriceModel) -> None:
        """Deep OTM: price << strike → P close to 0."""
        est = model.probability_above_merton_series(
            current_price=95_000, strike=100_000,
            volatility=0.30, horizon_minutes=30.0,
        )
        assert est.probability < 0.2, f"P={est.probability}"

    def test_drift_increases_probability(self, model: PriceModel) -> None:
        """Positive drift should increase P(above)."""
        p_no_drift = model.probability_above_merton_series(
            100_000, 100_000, 0.5, 30.0, drift=0.0,
        ).probability
        p_with_drift = model.probability_above_merton_series(
            100_000, 100_000, 0.5, 30.0, drift=1.0,
        ).probability
        assert p_with_drift > p_no_drift

    def test_higher_jump_intensity_widens_distribution(self, model: PriceModel) -> None:
        """More jumps → wider distribution → P moves toward 0.5 for ATM."""
        p_low_jumps = model.probability_above_merton_series(
            100_000, 100_000, 0.5, 30.0, lambda_daily=0.1,
        ).probability
        p_high_jumps = model.probability_above_merton_series(
            100_000, 100_000, 0.5, 30.0, lambda_daily=20.0,
        ).probability
        # Both should be near 0.5 for ATM, but higher jumps → more uncertainty
        assert abs(p_high_jumps - 0.5) < 0.15

    def test_convergence_with_mc_simulation(self, model: PriceModel) -> None:
        """Merton series should roughly agree with MC jump-diffusion."""
        mc_paths = model.generate_paths_jump_diffusion(
            100_000, 0.50, 30.0, drift=0.0,
            jump_intensity=3.0, jump_mean=0.0, jump_vol=0.02,
        )
        mc_prob = model.probability_above(mc_paths, 100_000).probability

        series_prob = model.probability_above_merton_series(
            100_000, 100_000, 0.50, 30.0, drift=0.0,
            lambda_daily=3.0, jump_mean=0.0, jump_vol=0.02,
        ).probability

        # Should agree within ~10% (MC has noise)
        assert abs(mc_prob - series_prob) < 0.15, (
            f"MC={mc_prob:.3f} vs series={series_prob:.3f}"
        )

    def test_degenerate_inputs(self, model: PriceModel) -> None:
        """Degenerate inputs should return P=0.5."""
        est = model.probability_above_merton_series(0, 100, 0.5, 30.0)
        assert est.probability == 0.5
        est = model.probability_above_merton_series(100, 100, 0.5, 0)
        assert est.probability == 0.5

    def test_series_truncation_uncertainty(self, model: PriceModel) -> None:
        """With enough terms, truncation error should be tiny."""
        est = model.probability_above_merton_series(
            100_000, 100_000, 0.5, 30.0, n_terms=20,
        )
        assert est.uncertainty < 0.1  # should converge well


# ── 1C: Probability floors/ceilings ─────────────────────────────


class TestProbabilityBounds:
    """compute_probability_bounds() + clamp_probability()"""

    def test_returns_floor_ceiling_tuple(self, model: PriceModel) -> None:
        floor, ceiling = model.compute_probability_bounds(
            lambda_daily=3.0, horizon_minutes=30.0,
            volatility=0.50, moneyness_sigma=2.0,
        )
        assert floor > 0
        assert ceiling < 1
        assert floor < ceiling

    def test_deeper_otm_narrower_bounds(self, model: PriceModel) -> None:
        """Deeper OTM → smaller jump flip probability → tighter bounds."""
        floor_2s, ceil_2s = model.compute_probability_bounds(
            3.0, 30.0, 0.50, moneyness_sigma=2.0,
        )
        floor_5s, ceil_5s = model.compute_probability_bounds(
            3.0, 30.0, 0.50, moneyness_sigma=5.0,
        )
        # 5σ should have lower floor than 2σ
        assert floor_5s <= floor_2s

    def test_higher_intensity_wider_bounds(self, model: PriceModel) -> None:
        """More jumps → higher chance of adverse flip → wider bounds."""
        floor_low, _ = model.compute_probability_bounds(
            lambda_daily=1.0, horizon_minutes=30.0,
            volatility=0.50, moneyness_sigma=3.0,
        )
        floor_high, _ = model.compute_probability_bounds(
            lambda_daily=10.0, horizon_minutes=30.0,
            volatility=0.50, moneyness_sigma=3.0,
        )
        assert floor_high >= floor_low

    def test_floor_respects_minimum(self, model: PriceModel) -> None:
        floor, _ = model.compute_probability_bounds(
            0.01, 1.0, 0.50, 10.0, floor_min=0.005,
        )
        assert floor >= 0.005

    def test_ceiling_respects_maximum(self, model: PriceModel) -> None:
        _, ceiling = model.compute_probability_bounds(
            0.01, 1.0, 0.50, 10.0, ceiling_max=0.995,
        )
        assert ceiling <= 0.995

    def test_degenerate_inputs(self, model: PriceModel) -> None:
        floor, ceiling = model.compute_probability_bounds(
            3.0, 0, 0.50, 2.0,
        )
        assert floor == 0.005  # default floor_min
        assert ceiling == 0.995  # default ceiling_max


class TestClampProbability:
    def test_clamp_below_floor(self) -> None:
        assert PriceModel.clamp_probability(0.001, 0.01, 0.99) == 0.01

    def test_clamp_above_ceiling(self) -> None:
        assert PriceModel.clamp_probability(0.999, 0.01, 0.99) == 0.99

    def test_no_clamp_in_range(self) -> None:
        assert PriceModel.clamp_probability(0.5, 0.01, 0.99) == 0.5


# ── 2A: Ornstein-Uhlenbeck closed-form ──────────────────────────


class TestProbabilityAboveOU:
    """probability_above_ou()"""

    def test_atm_returns_near_half(self, model: PriceModel) -> None:
        """ATM with price at mean → P ≈ 0.5."""
        est = model.probability_above_ou(
            current_price=100_000, strike=100_000,
            horizon_minutes=30.0, theta=5.0,
            mu=math.log(100_000), sigma=0.50,
        )
        assert 0.4 <= est.probability <= 0.6, f"P={est.probability}"

    def test_price_above_strike_high_theta(self, model: PriceModel) -> None:
        """Price well above strike with high mean-reversion → P stays high."""
        est = model.probability_above_ou(
            current_price=105_000, strike=100_000,
            horizon_minutes=30.0, theta=5.0,
            mu=math.log(105_000), sigma=0.30,
        )
        assert est.probability > 0.7, f"P={est.probability}"

    def test_mean_reversion_pulls_toward_mu(self, model: PriceModel) -> None:
        """Price above mu but mu below strike → mean-reversion pulls P down."""
        # Price is at 105k, mu is at 95k, strike is at 100k
        # High theta → strong pull toward 95k → P(above 100k) should be low
        est = model.probability_above_ou(
            current_price=105_000, strike=100_000,
            horizon_minutes=60.0, theta=20.0,
            mu=math.log(95_000), sigma=0.30,
        )
        # With strong mean-reversion toward 95k, should pull below 100k
        assert est.probability < 0.7, f"P={est.probability}"

    def test_higher_vol_wider_distribution(self, model: PriceModel) -> None:
        """Higher sigma → wider distribution → P moves toward 0.5."""
        p_low_vol = model.probability_above_ou(
            100_000, 100_000, 30.0, 5.0, math.log(105_000), 0.10,
        ).probability
        p_high_vol = model.probability_above_ou(
            100_000, 100_000, 30.0, 5.0, math.log(105_000), 1.0,
        ).probability
        # Higher vol → more uncertainty → closer to 0.5
        assert abs(p_high_vol - 0.5) < abs(p_low_vol - 0.5) + 0.05

    def test_degenerate_inputs(self, model: PriceModel) -> None:
        est = model.probability_above_ou(0, 100_000, 30.0, 5.0, 11.0, 0.5)
        assert est.probability == 0.5

    def test_returns_probability_estimate(self, model: PriceModel) -> None:
        est = model.probability_above_ou(
            100_000, 100_000, 30.0, 5.0, math.log(100_000), 0.5,
        )
        assert isinstance(est, ProbabilityEstimate)
        assert est.num_paths == 0
        assert 0 <= est.ci_lower <= est.probability <= est.ci_upper <= 1


class TestEstimateOUParams:
    """estimate_ou_params()"""

    def test_returns_three_params(self, model: PriceModel) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.001, 100).tolist()
        result = model.estimate_ou_params(returns, interval_seconds=60)
        assert result is not None
        theta, mu, sigma = result
        assert theta > 0
        assert sigma > 0

    def test_insufficient_data_returns_none(self, model: PriceModel) -> None:
        result = model.estimate_ou_params([0.01] * 10, interval_seconds=60)
        assert result is None

    def test_mean_reverting_series_high_theta(self, model: PriceModel) -> None:
        """OU-generated series should yield a reasonable theta."""
        rng = np.random.default_rng(42)
        n = 500
        # Generate OU-like returns with negative autocorrelation
        returns = []
        r = 0.0
        for _ in range(n):
            r = -0.3 * r + rng.normal(0, 0.001)
            returns.append(r)
        result = model.estimate_ou_params(returns, interval_seconds=60)
        assert result is not None
        theta, _, _ = result
        assert theta > 0.5, f"theta={theta} should indicate mean-reversion"

    def test_theta_clamped(self, model: PriceModel) -> None:
        """Theta should be clamped to [0.1, 50.0]."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.001, 100).tolist()
        result = model.estimate_ou_params(returns, interval_seconds=60)
        if result is not None:
            theta, _, _ = result
            assert 0.1 <= theta <= 50.0


# ── Config loading ───────────────────────────────────────────────


class TestDailyConfigFields:
    """Verify v43 config fields exist and have correct defaults."""

    def test_config_defaults(self) -> None:
        from arb_bot.crypto.config import CryptoSettings

        settings = CryptoSettings()
        assert settings.daily_model_enabled is False
        assert settings.variance_ratio_enabled is True
        assert settings.variance_ratio_min_samples == 50
        assert settings.merton_enabled is True
        assert settings.merton_jump_mean == 0.0
        assert settings.merton_jump_vol == 0.02
        assert settings.merton_default_intensity == 3.0
        assert settings.merton_n_terms == 15
        assert settings.probability_floor_enabled is True
        assert settings.probability_floor_min == 0.005
        assert settings.probability_ceiling_max == 0.995
        assert settings.daily_ou_weight_mean_reverting == 0.7
        assert settings.daily_gbm_weight_trending == 0.7
        assert settings.daily_merton_weight_deep == 0.8
        assert settings.daily_regime_transition_tau_minutes == 5.0
        assert settings.daily_moneyness_deep_threshold == 3.0
        assert settings.daily_moneyness_atm_threshold == 1.0
        assert settings.daily_ofi_weights == "0.1,0.2,0.3,0.4"

    def test_load_crypto_settings(self) -> None:
        from arb_bot.crypto.config import load_crypto_settings

        settings = load_crypto_settings()
        # Should load without error and have the new fields
        assert hasattr(settings, "daily_model_enabled")
        assert hasattr(settings, "merton_jump_vol")
        assert hasattr(settings, "daily_ofi_weights")
