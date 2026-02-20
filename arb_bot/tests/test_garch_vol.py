"""Tests for the GARCH(1,1) volatility forecaster."""

from __future__ import annotations

import math

import numpy as np
import pytest

from arb_bot.crypto.garch_vol import GarchForecaster, GarchParams


# ── Helpers ────────────────────────────────────────────────────────

def _simulate_garch(
    omega: float,
    alpha: float,
    beta: float,
    n: int,
    seed: int = 42,
) -> np.ndarray:
    """Simulate GARCH(1,1) returns with known parameters."""
    rng = np.random.default_rng(seed)
    sigma2 = np.empty(n)
    returns = np.empty(n)

    long_run_var = omega / (1.0 - alpha - beta)
    sigma2[0] = long_run_var
    returns[0] = math.sqrt(sigma2[0]) * rng.standard_normal()

    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]
        returns[t] = math.sqrt(sigma2[t]) * rng.standard_normal()

    return returns


# ── Fitting tests ─────────────────────────────────────────────────

class TestGarchFit:
    def test_recover_known_params(self) -> None:
        """Fit on synthetic data should approximately recover true params."""
        true_omega = 1e-6
        true_alpha = 0.08
        true_beta = 0.90

        returns = _simulate_garch(true_omega, true_alpha, true_beta, n=3000, seed=42)
        forecaster = GarchForecaster(min_obs=100)
        params = forecaster.fit(returns)

        assert params is not None
        # Allow wide tolerance — MLE on 3000 obs is noisy.
        assert 0.02 <= params.alpha <= 0.20
        assert 0.75 <= params.beta <= 0.97
        assert params.alpha + params.beta < 1.0

    def test_stationarity_enforced(self) -> None:
        """Alpha + beta must be < 1."""
        returns = _simulate_garch(1e-6, 0.08, 0.90, n=1000)
        forecaster = GarchForecaster(min_obs=100)
        params = forecaster.fit(returns)

        assert params is not None
        assert params.alpha + params.beta < 1.0

    def test_long_run_variance_correct(self) -> None:
        """long_run_var should equal omega / (1 - alpha - beta)."""
        returns = _simulate_garch(1e-6, 0.05, 0.90, n=2000)
        forecaster = GarchForecaster(min_obs=100)
        params = forecaster.fit(returns)

        assert params is not None
        expected = params.omega / (1.0 - params.alpha - params.beta)
        assert abs(params.long_run_var - expected) < 1e-15

    def test_min_obs_guard(self) -> None:
        """Returns None with fewer than min_obs returns."""
        returns = np.random.default_rng(42).normal(0, 0.01, 50)
        forecaster = GarchForecaster(min_obs=100)
        params = forecaster.fit(returns)
        assert params is None

    def test_constant_series_returns_none(self) -> None:
        """Returns None on constant (zero-variance) series."""
        returns = np.zeros(200)
        forecaster = GarchForecaster(min_obs=100)
        params = forecaster.fit(returns)
        assert params is None

    def test_lookback_trims(self) -> None:
        """Uses only the last lookback_obs observations."""
        returns = _simulate_garch(1e-6, 0.08, 0.90, n=5000)
        forecaster = GarchForecaster(min_obs=100, lookback_obs=500)
        params = forecaster.fit(returns)

        assert params is not None
        assert params.n_obs == 500

    def test_grid_search_fallback(self) -> None:
        """Grid search produces reasonable params on noisy data."""
        rng = np.random.default_rng(99)
        returns = rng.normal(0, 0.005, 200)
        forecaster = GarchForecaster(min_obs=100)
        params = forecaster.fit(returns)

        assert params is not None
        assert params.alpha > 0
        assert params.beta > 0
        assert params.omega > 0

    def test_fit_caches_last_params(self) -> None:
        """After fit, last_params and last_sigma2 are set."""
        returns = _simulate_garch(1e-6, 0.08, 0.90, n=500)
        forecaster = GarchForecaster(min_obs=100)
        forecaster.fit(returns)

        assert forecaster.last_params is not None
        assert forecaster.last_sigma2 is not None
        assert forecaster.last_sigma2 > 0


# ── Forecast tests ────────────────────────────────────────────────

class TestGarchForecast:
    def test_forecast_converges_to_long_run(self) -> None:
        """Multi-step forecast converges to unconditional variance."""
        returns = _simulate_garch(1e-6, 0.08, 0.90, n=1000)
        forecaster = GarchForecaster(min_obs=100)
        params = forecaster.fit(returns)
        assert params is not None

        # Short horizon.
        short = forecaster.forecast(params, returns, horizon_steps=5)
        # Long horizon.
        long = forecaster.forecast(params, returns, horizon_steps=10000)

        # At very long horizon, per-step vol should converge to unconditional.
        long_per_step_var = (long.sigma_horizon ** 2) / long.forecast_steps
        assert abs(long_per_step_var - params.long_run_var) / params.long_run_var < 0.01

    def test_short_horizon_above_long_run_after_shock(self) -> None:
        """After a large return, 1-step vol > long-run vol."""
        returns = _simulate_garch(1e-6, 0.10, 0.85, n=500, seed=42)
        # Append a big shock.
        returns = np.append(returns, 0.05)  # 5% 1-minute return = huge

        forecaster = GarchForecaster(min_obs=100)
        params = forecaster.fit(returns[:-1])  # Fit on pre-shock data
        assert params is not None

        fc = forecaster.forecast(params, returns, horizon_steps=1)
        long_run_sigma = math.sqrt(params.long_run_var)
        assert fc.sigma_1step > long_run_sigma

    def test_forecast_returns_positive_vol(self) -> None:
        """All volatility outputs are positive."""
        returns = _simulate_garch(1e-6, 0.08, 0.90, n=500)
        forecaster = GarchForecaster(min_obs=100)
        params = forecaster.fit(returns)
        assert params is not None

        for h in [1, 5, 15, 60, 1440]:
            fc = forecaster.forecast(params, returns, horizon_steps=h)
            assert fc.sigma_annualized > 0
            assert fc.sigma_horizon > 0
            assert fc.sigma_1step > 0

    def test_annualization_consistent(self) -> None:
        """sigma_annualized and sigma_horizon should be consistent."""
        returns = _simulate_garch(1e-6, 0.08, 0.90, n=500)
        forecaster = GarchForecaster(min_obs=100, interval_seconds=60)
        params = forecaster.fit(returns)
        assert params is not None

        h = 15  # 15-minute horizon
        fc = forecaster.forecast(params, returns, horizon_steps=h)

        # sigma_annualized = sigma_horizon * sqrt(obs_per_year / h)
        obs_per_year = 365.25 * 24.0 * 60.0  # 1-min intervals
        expected_annual = fc.sigma_horizon * math.sqrt(obs_per_year / h)
        assert abs(fc.sigma_annualized - expected_annual) / expected_annual < 1e-6


# ── Online update tests ───────────────────────────────────────────

class TestOnlineUpdate:
    def test_update_matches_formula(self) -> None:
        """Online sigma2 update matches the GARCH formula directly."""
        returns = _simulate_garch(1e-6, 0.08, 0.90, n=300)
        forecaster = GarchForecaster(min_obs=100)
        params = forecaster.fit(returns[:200])
        assert params is not None

        # Compute sigma2 at t=200 from the full series.
        sigma2_series = forecaster._compute_sigma2_series(params, returns[:200])
        sigma2_prev = float(sigma2_series[-1])

        # Online update for one new return.
        new_r = float(returns[200])
        actual = forecaster.update_online(new_r, sigma2_prev)

        # Expected from formula: omega + alpha * r^2 + beta * sigma2_prev.
        expected = params.omega + params.alpha * new_r ** 2 + params.beta * sigma2_prev

        assert abs(actual - expected) < 1e-15

    def test_needs_refit_tracks_observations(self) -> None:
        """needs_refit becomes True after refit_interval observations."""
        forecaster = GarchForecaster(min_obs=100, refit_interval=10)
        returns = _simulate_garch(1e-6, 0.08, 0.90, n=200)
        forecaster.fit(returns)

        assert not forecaster.needs_refit

        sigma2 = forecaster.last_sigma2
        for i in range(9):
            sigma2 = forecaster.update_online(0.001, sigma2)
            assert not forecaster.needs_refit

        sigma2 = forecaster.update_online(0.001, sigma2)
        assert forecaster.needs_refit


# ── Edge case tests ───────────────────────────────────────────────

class TestEdgeCases:
    def test_very_small_returns(self) -> None:
        """Handles very small returns without numerical issues."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 1e-6, 200)
        forecaster = GarchForecaster(min_obs=100)
        params = forecaster.fit(returns)
        # Should either fit or return None, not crash.
        if params is not None:
            assert params.omega > 0

    def test_large_returns(self) -> None:
        """Handles large returns (e.g., flash crash)."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 200)
        returns[100] = 0.10  # 10% 1-minute return
        returns[101] = -0.08
        forecaster = GarchForecaster(min_obs=100)
        params = forecaster.fit(returns)
        assert params is not None
        # Vol should spike after the shock.
        fc = forecaster.forecast(params, returns, horizon_steps=15)
        assert fc.sigma_annualized > 0

    def test_sigma2_series_all_positive(self) -> None:
        """Conditional variance series should always be positive."""
        returns = _simulate_garch(1e-6, 0.08, 0.90, n=500)
        forecaster = GarchForecaster(min_obs=100)
        params = forecaster.fit(returns)
        assert params is not None

        sigma2 = forecaster._compute_sigma2_series(params, returns)
        assert np.all(sigma2 > 0)
