"""Tests for the Student-t analytical probability model.

Covers PriceModel.fit_nu_from_returns, estimate_vol_stderr,
probability_above_student_t, side-by-side comparison with GBM
analytical, and CryptoSettings config defaults.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import t as student_t_dist

from arb_bot.crypto.config import CryptoSettings
from arb_bot.crypto.price_model import PriceModel, ProbabilityEstimate


# ── Helpers ─────────────────────────────────────────────────────────


def _make_model(seed: int = 42) -> PriceModel:
    return PriceModel(num_paths=1000, seed=seed)


# ── TestFitNuFromReturns ────────────────────────────────────────────


class TestFitNuFromReturns:
    """Tests for PriceModel.fit_nu_from_returns()."""

    def test_returns_tuple(self):
        model = _make_model()
        rng = np.random.default_rng(42)
        returns = list(rng.normal(0, 0.01, size=200))
        result = model.fit_nu_from_returns(returns)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_normal_returns_high_nu(self):
        """N(0, 0.01) returns should give high nu (near-normal)."""
        model = _make_model()
        rng = np.random.default_rng(42)
        returns = list(rng.normal(0, 0.01, size=200))
        nu, nu_stderr = model.fit_nu_from_returns(returns)
        assert nu > 15, f"Expected nu > 15 for normal data, got {nu}"

    def test_heavy_tailed_returns_low_nu(self):
        """t(3) returns should give low nu (heavy tails)."""
        model = _make_model()
        returns = list(student_t_dist.rvs(df=3, size=200, random_state=42) * 0.01)
        nu, nu_stderr = model.fit_nu_from_returns(returns)
        assert nu < 8, f"Expected nu < 8 for t(3) data, got {nu}"

    def test_insufficient_data_returns_ceiling(self):
        """Fewer than min_samples should return (ceiling, 0.0)."""
        model = _make_model()
        returns = list(np.random.default_rng(42).normal(0, 0.01, size=10))
        nu, nu_stderr = model.fit_nu_from_returns(returns, min_samples=30)
        assert nu == 30.0
        assert nu_stderr == 0.0

    def test_nu_clamped_to_floor(self):
        """Fitted nu must be >= nu_floor."""
        model = _make_model()
        # Extremely heavy-tailed data
        returns = list(student_t_dist.rvs(df=2.1, size=500, random_state=99) * 0.01)
        nu, _ = model.fit_nu_from_returns(returns, nu_floor=2.5)
        assert nu >= 2.5, f"nu should be >= 2.5 (floor), got {nu}"

    def test_nu_clamped_to_ceiling(self):
        """Fitted nu must be <= nu_ceiling."""
        model = _make_model()
        rng = np.random.default_rng(42)
        returns = list(rng.normal(0, 0.01, size=200))
        nu, _ = model.fit_nu_from_returns(returns, nu_ceiling=30.0)
        assert nu <= 30.0, f"nu should be <= 30.0 (ceiling), got {nu}"

    def test_nu_stderr_positive(self):
        """With bootstrap_n=50 and enough data, nu_stderr should be > 0."""
        model = _make_model()
        returns = list(student_t_dist.rvs(df=5, size=200, random_state=42) * 0.01)
        _, nu_stderr = model.fit_nu_from_returns(returns, bootstrap_n=50)
        assert nu_stderr > 0, f"Expected positive nu_stderr, got {nu_stderr}"

    def test_empty_returns(self):
        """Empty returns list should return (ceiling, 0.0)."""
        model = _make_model()
        nu, nu_stderr = model.fit_nu_from_returns([])
        assert nu == 30.0
        assert nu_stderr == 0.0


# ── TestEstimateVolStderr ───────────────────────────────────────────


class TestEstimateVolStderr:
    """Tests for PriceModel.estimate_vol_stderr()."""

    def test_more_data_lower_stderr(self):
        """200 returns should give lower stderr than 30 returns (same vol)."""
        model = _make_model()
        rng = np.random.default_rng(42)
        returns_200 = list(rng.normal(0, 0.01, size=200))
        returns_30 = list(rng.normal(0, 0.01, size=30))
        stderr_200 = model.estimate_vol_stderr(returns_200)
        stderr_30 = model.estimate_vol_stderr(returns_30)
        assert stderr_200 < stderr_30, (
            f"stderr with 200 points ({stderr_200}) should be < "
            f"stderr with 30 points ({stderr_30})"
        )

    def test_insufficient_data_returns_zero(self):
        """Fewer than 3 returns should return 0.0."""
        model = _make_model()
        assert model.estimate_vol_stderr([0.01, -0.01]) == 0.0
        assert model.estimate_vol_stderr([0.01]) == 0.0
        assert model.estimate_vol_stderr([]) == 0.0

    def test_positive_for_valid_data(self):
        """50+ returns should give a positive stderr."""
        model = _make_model()
        rng = np.random.default_rng(42)
        returns = list(rng.normal(0, 0.01, size=50))
        stderr = model.estimate_vol_stderr(returns)
        assert stderr > 0, f"Expected positive stderr, got {stderr}"


# ── TestProbabilityAboveStudentT ────────────────────────────────────


class TestProbabilityAboveStudentT:
    """Tests for PriceModel.probability_above_student_t()."""

    def test_returns_probability_estimate(self):
        """Result should be a ProbabilityEstimate with expected fields."""
        model = _make_model()
        result = model.probability_above_student_t(
            current_price=100.0,
            strike=100.0,
            volatility=0.5,
            horizon_minutes=15.0,
        )
        assert isinstance(result, ProbabilityEstimate)
        assert hasattr(result, "probability")
        assert hasattr(result, "ci_lower")
        assert hasattr(result, "ci_upper")
        assert hasattr(result, "uncertainty")
        assert hasattr(result, "num_paths")
        assert result.num_paths == 0

    def test_at_the_money_near_half(self):
        """S=K=100, drift=0, vol=0.5, horizon=15 -> ~0.5."""
        model = _make_model()
        result = model.probability_above_student_t(
            current_price=100.0,
            strike=100.0,
            volatility=0.5,
            horizon_minutes=15.0,
            drift=0.0,
            nu=5.0,
        )
        assert result.probability == pytest.approx(0.5, abs=0.02)

    def test_deep_itm_near_one(self):
        """S=100, K=50 -> P(above) close to 1.0."""
        model = _make_model()
        result = model.probability_above_student_t(
            current_price=100.0,
            strike=50.0,
            volatility=0.5,
            horizon_minutes=15.0,
        )
        assert result.probability > 0.95

    def test_deep_otm_near_zero(self):
        """S=100, K=200 -> P(above) close to 0.0."""
        model = _make_model()
        result = model.probability_above_student_t(
            current_price=100.0,
            strike=200.0,
            volatility=0.5,
            horizon_minutes=15.0,
        )
        assert result.probability < 0.05

    def test_matches_analytical_for_high_nu(self):
        """With nu=100 (near-normal), should match GBM analytical within 0.01."""
        model = _make_model()
        s, k, vol, t_min, drift = 100.0, 105.0, 0.5, 15.0, 0.0
        p_student = model.probability_above_student_t(
            current_price=s,
            strike=k,
            volatility=vol,
            horizon_minutes=t_min,
            drift=drift,
            nu=100.0,
        )
        p_normal = model.probability_above_analytical(
            current_price=s,
            strike=k,
            volatility=vol,
            horizon_minutes=t_min,
            drift=drift,
        )
        assert p_student.probability == pytest.approx(p_normal, abs=0.01), (
            f"Student-t(nu=100) = {p_student.probability}, "
            f"GBM analytical = {p_normal}"
        )

    def test_heavier_tails_differ_from_normal(self):
        """nu=3 vs nu=100 should produce different probabilities for OTM strikes."""
        model = _make_model()
        # Use a moderately OTM strike so both probabilities are in a
        # numerically distinguishable range (not both ~0).
        kwargs = dict(
            current_price=100.0,
            strike=101.0,
            volatility=0.5,
            horizon_minutes=15.0,
            drift=0.0,
        )
        p_heavy = model.probability_above_student_t(**kwargs, nu=3.0)
        p_normal = model.probability_above_student_t(**kwargs, nu=100.0)
        assert abs(p_heavy.probability - p_normal.probability) > 0.001, (
            f"Expected different probabilities: nu=3 -> {p_heavy.probability}, "
            f"nu=100 -> {p_normal.probability}"
        )

    def test_num_paths_is_zero(self):
        """Analytical model sets num_paths=0."""
        model = _make_model()
        result = model.probability_above_student_t(
            current_price=100.0,
            strike=100.0,
            volatility=0.5,
            horizon_minutes=15.0,
        )
        assert result.num_paths == 0

    def test_uncertainty_from_nu_stderr(self):
        """Higher nu_stderr should increase uncertainty."""
        model = _make_model()
        kwargs = dict(
            current_price=100.0,
            strike=105.0,
            volatility=0.5,
            horizon_minutes=15.0,
            drift=0.0,
            nu=5.0,
            vol_stderr=0.0,
        )
        r_low = model.probability_above_student_t(**kwargs, nu_stderr=0.5)
        r_high = model.probability_above_student_t(**kwargs, nu_stderr=3.0)
        assert r_high.uncertainty >= r_low.uncertainty, (
            f"Higher nu_stderr should give >= uncertainty: "
            f"nu_stderr=0.5 -> {r_low.uncertainty}, "
            f"nu_stderr=3.0 -> {r_high.uncertainty}"
        )

    def test_uncertainty_floor(self):
        """Even with nu_stderr=0 and vol_stderr=0, uncertainty >= min_uncertainty."""
        model = _make_model()
        result = model.probability_above_student_t(
            current_price=100.0,
            strike=105.0,
            volatility=0.5,
            horizon_minutes=15.0,
            nu_stderr=0.0,
            vol_stderr=0.0,
            min_uncertainty=0.02,
        )
        assert result.uncertainty >= 0.02

    def test_degenerate_inputs_return_half(self):
        """vol=0, horizon=0, or price=0 should return probability=0.5."""
        model = _make_model()

        # vol=0
        r1 = model.probability_above_student_t(
            current_price=100.0, strike=100.0, volatility=0.0, horizon_minutes=15.0,
        )
        assert r1.probability == 0.5

        # horizon=0
        r2 = model.probability_above_student_t(
            current_price=100.0, strike=100.0, volatility=0.5, horizon_minutes=0.0,
        )
        assert r2.probability == 0.5

        # price=0
        r3 = model.probability_above_student_t(
            current_price=0.0, strike=100.0, volatility=0.5, horizon_minutes=15.0,
        )
        assert r3.probability == 0.5

    def test_drift_affects_probability(self):
        """Positive drift should increase P(above) for OTM strike."""
        model = _make_model()
        kwargs = dict(
            current_price=100.0,
            strike=105.0,
            volatility=0.5,
            horizon_minutes=15.0,
            nu=5.0,
        )
        p_no_drift = model.probability_above_student_t(**kwargs, drift=0.0)
        p_pos_drift = model.probability_above_student_t(**kwargs, drift=5.0)
        assert p_pos_drift.probability > p_no_drift.probability, (
            f"Positive drift should increase P(above): "
            f"drift=0 -> {p_no_drift.probability}, "
            f"drift=5 -> {p_pos_drift.probability}"
        )


# ── TestStudentTVsGBM ──────────────────────────────────────────────


class TestStudentTVsGBM:
    """Side-by-side comparison between Student-t and GBM analytical."""

    def test_fat_tails_increase_otm_probability(self):
        """For OTM strike, Student-t (nu=4) should give HIGHER prob than GBM.

        Fat tails mean extreme moves are more likely, so the probability
        of reaching a far-away strike is higher under Student-t.
        """
        model = _make_model()
        s, k, vol, t_min = 100.0, 115.0, 0.5, 15.0

        p_student = model.probability_above_student_t(
            current_price=s, strike=k, volatility=vol,
            horizon_minutes=t_min, nu=4.0,
        )
        p_gbm = model.probability_above_analytical(
            current_price=s, strike=k, volatility=vol,
            horizon_minutes=t_min,
        )
        assert p_student.probability > p_gbm, (
            f"Student-t(nu=4) should give higher OTM probability: "
            f"Student-t={p_student.probability}, GBM={p_gbm}"
        )

    def test_atm_probability_similar(self):
        """At the money (S=K), both models should agree closely (within 0.02)."""
        model = _make_model()
        s, k, vol, t_min = 100.0, 100.0, 0.5, 15.0

        p_student = model.probability_above_student_t(
            current_price=s, strike=k, volatility=vol,
            horizon_minutes=t_min, nu=5.0,
        )
        p_gbm = model.probability_above_analytical(
            current_price=s, strike=k, volatility=vol,
            horizon_minutes=t_min,
        )
        assert p_student.probability == pytest.approx(p_gbm, abs=0.02), (
            f"ATM should be similar: Student-t={p_student.probability}, GBM={p_gbm}"
        )

    def test_probability_symmetry(self):
        """P(above, K) + P(below, K) should sum to ~1.0."""
        model = _make_model()
        s, k, vol, t_min = 100.0, 105.0, 0.5, 15.0

        p_above = model.probability_above_student_t(
            current_price=s, strike=k, volatility=vol,
            horizon_minutes=t_min, nu=5.0,
        )
        # P(below) = 1 - P(above) by complement
        p_below = 1.0 - p_above.probability
        total = p_above.probability + p_below
        assert total == pytest.approx(1.0, abs=1e-12), (
            f"P(above) + P(below) should be 1.0, got {total}"
        )


# ── TestStudentTConfig ──────────────────────────────────────────────


class TestStudentTConfig:
    """Tests for Student-t config defaults in CryptoSettings."""

    def test_default_probability_model_is_mc_gbm(self):
        """CryptoSettings() should have probability_model='mc_gbm'."""
        settings = CryptoSettings()
        assert settings.probability_model == "mc_gbm"

    def test_student_t_params_have_defaults(self):
        """All student_t_* params should have sensible defaults."""
        settings = CryptoSettings()
        assert settings.student_t_nu_default == 5.0
        assert settings.student_t_nu_floor == 2.5
        assert settings.student_t_nu_ceiling == 30.0
        assert settings.student_t_fit_window_minutes == 120
        assert settings.student_t_fit_min_samples == 30
        assert settings.student_t_refit_every_cycles == 20
        assert settings.student_t_min_uncertainty == 0.02
        assert settings.student_t_uncertainty_multiplier == 1.5

    def test_ab_test_is_valid_model(self):
        """'ab_test' should be accepted as probability_model value."""
        settings = CryptoSettings(probability_model="ab_test")
        assert settings.probability_model == "ab_test"

    def test_student_t_is_valid_model(self):
        """'student_t' should be accepted as probability_model value."""
        settings = CryptoSettings(probability_model="student_t")
        assert settings.probability_model == "student_t"
