"""Tests for empirical CDF bootstrap probability model.

Tests cover:
- Core probability_above_empirical() behaviour (fallbacks, edge cases,
  determinism, CI properties)
- Comparison with parametric MC-GBM model
- CryptoSettings empirical_* configuration fields
"""

from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pytest

from arb_bot.crypto.config import CryptoSettings, load_crypto_settings
from arb_bot.crypto.price_model import PriceModel, ProbabilityEstimate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _symmetric_returns(n: int = 200, seed: int = 42) -> list[float]:
    """Small mean-zero normal returns (typical 1-min crypto scale)."""
    rng = np.random.RandomState(seed)
    return rng.normal(0.0, 0.001, n).tolist()


def _positive_returns(n: int = 200) -> list[float]:
    """All returns are positive (price going up every minute)."""
    return [0.001] * n


def _negative_returns(n: int = 200) -> list[float]:
    """All returns are negative (price going down every minute)."""
    return [-0.001] * n


# ===========================================================================
# TestProbabilityAboveEmpirical
# ===========================================================================


class TestProbabilityAboveEmpirical:
    """Core tests for probability_above_empirical()."""

    def test_atm_near_50_percent(self):
        """ATM strike with symmetric returns should give P near 50%."""
        model = PriceModel(num_paths=1000, seed=42)
        returns = _symmetric_returns(200, seed=99)
        result = model.probability_above_empirical(
            returns=returns,
            current_price=100.0,
            strike=100.0,
            horizon_steps=10,
            bootstrap_paths=5000,
        )
        assert 0.35 < result.probability < 0.65, (
            f"ATM probability {result.probability} not near 0.5"
        )

    def test_deep_itm_near_100_percent(self):
        """Strike far below current price -> P near 1.0."""
        model = PriceModel(num_paths=1000, seed=42)
        returns = _symmetric_returns(200)
        result = model.probability_above_empirical(
            returns=returns,
            current_price=100.0,
            strike=90.0,
            horizon_steps=10,
            bootstrap_paths=3000,
        )
        assert result.probability > 0.90, (
            f"Deep ITM probability {result.probability} should be > 0.90"
        )

    def test_deep_otm_near_0_percent(self):
        """Strike far above current price -> P near 0."""
        model = PriceModel(num_paths=1000, seed=42)
        returns = _symmetric_returns(200)
        result = model.probability_above_empirical(
            returns=returns,
            current_price=100.0,
            strike=110.0,
            horizon_steps=10,
            bootstrap_paths=3000,
        )
        assert result.probability < 0.10, (
            f"Deep OTM probability {result.probability} should be < 0.10"
        )

    def test_insufficient_data_fallback(self):
        """Fewer than min_samples returns -> deterministic fallback."""
        model = PriceModel(num_paths=1000, seed=42)
        returns = [0.001] * 20  # only 20, default min_samples is 30
        result = model.probability_above_empirical(
            returns=returns,
            current_price=100.0,
            strike=101.0,
            horizon_steps=10,
        )
        assert result == ProbabilityEstimate(0.5, 0.0, 1.0, 0.5, 0)

    def test_insufficient_data_custom_min(self):
        """Custom min_samples threshold controls fallback."""
        model = PriceModel(num_paths=1000, seed=42)
        returns = _symmetric_returns(25)

        # 25 returns with min_samples=20 -> should work (not fallback)
        ok = model.probability_above_empirical(
            returns=returns,
            current_price=100.0,
            strike=100.0,
            horizon_steps=5,
            min_samples=20,
        )
        assert ok.num_paths > 0, "Should not fallback with 25 >= 20"

        # 15 returns with min_samples=20 -> should fallback
        short = model.probability_above_empirical(
            returns=returns[:15],
            current_price=100.0,
            strike=100.0,
            horizon_steps=5,
            min_samples=20,
        )
        assert short == ProbabilityEstimate(0.5, 0.0, 1.0, 0.5, 0)

    def test_single_step_horizon(self):
        """horizon_steps=1 is a single-step simulation."""
        model = PriceModel(num_paths=1000, seed=42)
        returns = _symmetric_returns(200)
        result = model.probability_above_empirical(
            returns=returns,
            current_price=100.0,
            strike=100.0,
            horizon_steps=1,
            bootstrap_paths=2000,
        )
        # With mean-zero returns and strike==price, single step should be ~50%
        assert 0.35 < result.probability < 0.65

    def test_multi_step_horizon(self):
        """horizon_steps>1 produces a wider distribution than single step."""
        model_a = PriceModel(num_paths=1000, seed=42)
        model_b = PriceModel(num_paths=1000, seed=42)
        returns = _symmetric_returns(200)

        # For a slightly OTM strike, longer horizon should give higher P
        # because the distribution widens with more steps
        single = model_a.probability_above_empirical(
            returns=returns,
            current_price=100.0,
            strike=100.05,  # slightly OTM
            horizon_steps=1,
            bootstrap_paths=5000,
        )
        multi = model_b.probability_above_empirical(
            returns=returns,
            current_price=100.0,
            strike=100.05,
            horizon_steps=30,
            bootstrap_paths=5000,
        )
        # The multi-step result should differ from single-step
        # (wider distribution -> more mass in tails)
        assert single.probability != multi.probability, (
            "Single-step and multi-step should produce different P"
        )

    def test_deterministic_with_seed(self):
        """Same seed + same inputs -> identical results."""
        returns = _symmetric_returns(200)
        r1 = PriceModel(seed=42).probability_above_empirical(
            returns=returns,
            current_price=100.0,
            strike=100.5,
            horizon_steps=10,
            bootstrap_paths=2000,
        )
        r2 = PriceModel(seed=42).probability_above_empirical(
            returns=returns,
            current_price=100.0,
            strike=100.5,
            horizon_steps=10,
            bootstrap_paths=2000,
        )
        assert r1.probability == r2.probability
        assert r1.uncertainty == r2.uncertainty
        assert r1.ci_lower == r2.ci_lower
        assert r1.ci_upper == r2.ci_upper

    def test_uncertainty_floor(self):
        """min_uncertainty sets a floor on reported uncertainty."""
        model = PriceModel(num_paths=1000, seed=42)
        returns = _symmetric_returns(200)
        result = model.probability_above_empirical(
            returns=returns,
            current_price=100.0,
            strike=100.0,
            horizon_steps=10,
            bootstrap_paths=10000,
            min_uncertainty=0.05,
        )
        assert result.uncertainty >= 0.05, (
            f"Uncertainty {result.uncertainty} should be >= 0.05 floor"
        )

    def test_wilson_ci_bounds(self):
        """CI bounds must satisfy 0 <= ci_lower <= P <= ci_upper <= 1."""
        model = PriceModel(num_paths=1000, seed=42)
        returns = _symmetric_returns(200)
        result = model.probability_above_empirical(
            returns=returns,
            current_price=100.0,
            strike=100.0,
            horizon_steps=10,
            bootstrap_paths=2000,
        )
        assert 0.0 <= result.ci_lower <= result.probability
        assert result.probability <= result.ci_upper <= 1.0

    def test_num_paths_matches(self):
        """num_paths in result should equal bootstrap_paths."""
        model = PriceModel(num_paths=1000, seed=42)
        returns = _symmetric_returns(200)
        for bp in [500, 2000, 8000]:
            result = model.probability_above_empirical(
                returns=returns,
                current_price=100.0,
                strike=100.0,
                horizon_steps=10,
                bootstrap_paths=bp,
            )
            assert result.num_paths == bp

    def test_all_positive_returns_high_prob(self):
        """All positive returns -> elevated P even for slightly OTM."""
        model = PriceModel(num_paths=1000, seed=42)
        returns = _positive_returns(200)
        result = model.probability_above_empirical(
            returns=returns,
            current_price=100.0,
            strike=100.5,
            horizon_steps=10,
            bootstrap_paths=3000,
        )
        # 10 steps of +0.001 log return -> cumulative ~ +0.01
        # ln(100.5/100) ~ 0.004987, so most paths exceed it
        assert result.probability > 0.50, (
            f"All-positive returns should give P > 0.50, got {result.probability}"
        )

    def test_all_negative_returns_low_prob(self):
        """All negative returns -> low P for strike above current."""
        model = PriceModel(num_paths=1000, seed=42)
        returns = _negative_returns(200)
        result = model.probability_above_empirical(
            returns=returns,
            current_price=100.0,
            strike=100.5,
            horizon_steps=10,
            bootstrap_paths=3000,
        )
        # Every simulated path goes down -> P(above OTM strike) ~ 0
        assert result.probability < 0.05, (
            f"All-negative returns should give P < 0.05, got {result.probability}"
        )

    def test_zero_current_price_fallback(self):
        """current_price=0 -> deterministic fallback."""
        model = PriceModel(num_paths=1000, seed=42)
        returns = _symmetric_returns(200)
        result = model.probability_above_empirical(
            returns=returns,
            current_price=0.0,
            strike=100.0,
            horizon_steps=10,
        )
        assert result == ProbabilityEstimate(0.5, 0.0, 1.0, 0.5, 0)

    def test_zero_strike_fallback(self):
        """strike=0 -> deterministic fallback."""
        model = PriceModel(num_paths=1000, seed=42)
        returns = _symmetric_returns(200)
        result = model.probability_above_empirical(
            returns=returns,
            current_price=100.0,
            strike=0.0,
            horizon_steps=10,
        )
        assert result == ProbabilityEstimate(0.5, 0.0, 1.0, 0.5, 0)


# ===========================================================================
# TestEmpiricalVsParametric
# ===========================================================================


class TestEmpiricalVsParametric:
    """Compare empirical bootstrap vs parametric MC-GBM behaviour."""

    @staticmethod
    def _crypto_returns(seed: int = 42, n: int = 200, std: float = 0.001824):
        """Generate typical 1-min crypto log returns."""
        rng = np.random.RandomState(seed)
        return rng.normal(0.0, std, n).tolist()

    def test_empirical_lower_otm_than_mc(self):
        """Empirical should give lower OTM P than MC with inflated vol.

        MC-GBM annualizes the 1-min std to a very large vol (~132%),
        which inflates tail probabilities.  The empirical model draws
        directly from the raw returns and does NOT annualize, so it
        should produce a more conservative (lower) OTM probability.
        """
        returns = self._crypto_returns(seed=42, n=200, std=0.001824)
        model = PriceModel(num_paths=5000, seed=42)

        price = 100.0
        strike = 101.5  # 1.5% OTM
        horizon_steps = 44  # 44 minutes

        # Empirical
        emp = model.probability_above_empirical(
            returns=returns,
            current_price=price,
            strike=strike,
            horizon_steps=horizon_steps,
            bootstrap_paths=5000,
        )

        # MC-GBM: annualize 1-min std
        arr = np.array(returns)
        std_1m = float(np.std(arr, ddof=1))
        # Crypto minutes per year: 365.25 * 24 * 60
        minutes_per_year = 365.25 * 24.0 * 60.0
        vol_annual = std_1m * math.sqrt(minutes_per_year)

        paths = model.generate_paths(
            current_price=price,
            volatility=vol_annual,
            horizon_minutes=horizon_steps,
            num_paths=5000,
            drift=0.0,
        )
        mc = model.probability_above(paths, strike)

        # The key assertion: empirical should be lower or both very small
        if mc.probability > 0.15:
            assert emp.probability < mc.probability, (
                f"Empirical P={emp.probability:.4f} should be < MC P={mc.probability:.4f}"
            )
        else:
            # Both are small enough that ordering may not hold deterministically
            assert emp.probability < 0.15 and mc.probability < 0.15

    def test_atm_roughly_similar(self):
        """At the money, both models should be near 50%."""
        returns = self._crypto_returns(seed=42, n=200)
        model = PriceModel(num_paths=5000, seed=42)

        price = 100.0
        strike = 100.0
        horizon_steps = 10

        emp = model.probability_above_empirical(
            returns=returns,
            current_price=price,
            strike=strike,
            horizon_steps=horizon_steps,
            bootstrap_paths=5000,
        )

        arr = np.array(returns)
        std_1m = float(np.std(arr, ddof=1))
        minutes_per_year = 365.25 * 24.0 * 60.0
        vol_annual = std_1m * math.sqrt(minutes_per_year)

        paths = model.generate_paths(
            current_price=price,
            volatility=vol_annual,
            horizon_minutes=horizon_steps,
            num_paths=5000,
        )
        mc = model.probability_above(paths, strike)

        # Both should be in the vicinity of 50%
        assert 0.35 < emp.probability < 0.65
        assert 0.35 < mc.probability < 0.65

    def test_bootstrap_paths_affects_precision(self):
        """More bootstrap paths -> smaller uncertainty."""
        returns = self._crypto_returns(seed=42, n=200)

        low = PriceModel(seed=42).probability_above_empirical(
            returns=returns,
            current_price=100.0,
            strike=100.5,
            horizon_steps=10,
            bootstrap_paths=500,
        )
        high = PriceModel(seed=42).probability_above_empirical(
            returns=returns,
            current_price=100.0,
            strike=100.5,
            horizon_steps=10,
            bootstrap_paths=5000,
        )
        # Wilson CI shrinks with sqrt(n), so 10x more paths -> ~3x tighter
        assert high.uncertainty <= low.uncertainty, (
            f"More paths should give <= uncertainty: "
            f"{high.uncertainty} vs {low.uncertainty}"
        )

    def test_longer_horizon_wider_distribution(self):
        """Longer horizon increases P(above) for moderately OTM strike.

        More steps -> wider distribution of cumulative returns ->
        more mass in the upper tail past the OTM strike.
        """
        returns = self._crypto_returns(seed=42, n=200)

        short = PriceModel(seed=42).probability_above_empirical(
            returns=returns,
            current_price=100.0,
            strike=100.2,  # slightly OTM
            horizon_steps=5,
            bootstrap_paths=5000,
        )
        long = PriceModel(seed=42).probability_above_empirical(
            returns=returns,
            current_price=100.0,
            strike=100.2,
            horizon_steps=60,
            bootstrap_paths=5000,
        )
        assert long.probability > short.probability, (
            f"Longer horizon P={long.probability:.4f} should > "
            f"short horizon P={short.probability:.4f}"
        )

    def test_returns_distribution_shape_matters(self):
        """Fat-tailed returns (Student-t nu=3) vs normal of same std
        should produce different tail probabilities."""
        rng = np.random.RandomState(42)
        std = 0.002

        normal_returns = rng.normal(0, std, 500).tolist()

        # Student-t with nu=3 (fat tails), scaled to match std
        # Var of t(nu) = nu/(nu-2), so scale = std / sqrt(nu/(nu-2))
        t_scale = std / math.sqrt(3.0 / (3.0 - 2.0))
        t_raw = rng.standard_t(df=3, size=500)
        fat_returns = (t_raw * t_scale).tolist()

        # Compare OTM (tail) probabilities
        strike = 101.0  # 1% OTM
        price = 100.0

        norm_est = PriceModel(seed=42).probability_above_empirical(
            returns=normal_returns,
            current_price=price,
            strike=strike,
            horizon_steps=30,
            bootstrap_paths=5000,
        )
        fat_est = PriceModel(seed=42).probability_above_empirical(
            returns=fat_returns,
            current_price=price,
            strike=strike,
            horizon_steps=30,
            bootstrap_paths=5000,
        )
        # They should not be identical -- fat tails change the result
        assert norm_est.probability != fat_est.probability, (
            "Normal vs fat-tailed returns should give different P"
        )


# ===========================================================================
# TestEmpiricalConfig
# ===========================================================================


class TestEmpiricalConfig:
    """Tests for CryptoSettings empirical_* configuration fields."""

    def test_default_values(self):
        """CryptoSettings() has correct defaults for all empirical fields."""
        cfg = CryptoSettings()
        assert cfg.empirical_window_minutes == 120
        assert cfg.empirical_min_samples == 30
        assert cfg.empirical_bootstrap_paths == 2000
        assert cfg.empirical_min_uncertainty == 0.02
        assert cfg.empirical_uncertainty_multiplier == 1.5
        assert cfg.empirical_return_interval_seconds == 60

    def test_env_var_loading(self):
        """Environment variables are read by load_crypto_settings()."""
        env = {
            "ARB_CRYPTO_EMPIRICAL_WINDOW_MINUTES": "60",
            "ARB_CRYPTO_EMPIRICAL_MIN_SAMPLES": "50",
            "ARB_CRYPTO_EMPIRICAL_BOOTSTRAP_PATHS": "5000",
            "ARB_CRYPTO_EMPIRICAL_MIN_UNCERTAINTY": "0.03",
            "ARB_CRYPTO_EMPIRICAL_UNCERTAINTY_MULTIPLIER": "2.0",
            "ARB_CRYPTO_EMPIRICAL_RETURN_INTERVAL_SECONDS": "30",
        }
        with patch.dict("os.environ", env, clear=False):
            cfg = load_crypto_settings()
        assert cfg.empirical_window_minutes == 60
        assert cfg.empirical_min_samples == 50
        assert cfg.empirical_bootstrap_paths == 5000
        assert cfg.empirical_min_uncertainty == pytest.approx(0.03)
        assert cfg.empirical_uncertainty_multiplier == pytest.approx(2.0)
        assert cfg.empirical_return_interval_seconds == 30

    def test_valid_model_names(self):
        """'empirical' and 'ab_empirical' are accepted model names."""
        cfg_emp = CryptoSettings(probability_model="empirical")
        assert cfg_emp.probability_model == "empirical"

        cfg_ab = CryptoSettings(probability_model="ab_empirical")
        assert cfg_ab.probability_model == "ab_empirical"

    def test_custom_bootstrap_paths(self):
        """Custom empirical_bootstrap_paths is stored correctly."""
        cfg = CryptoSettings(empirical_bootstrap_paths=500)
        assert cfg.empirical_bootstrap_paths == 500

    def test_custom_window_minutes(self):
        """Custom empirical_window_minutes is stored correctly."""
        cfg = CryptoSettings(empirical_window_minutes=60)
        assert cfg.empirical_window_minutes == 60
