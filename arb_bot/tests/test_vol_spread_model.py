"""Tests for the vol-spread signal generator."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm as _norm

from arb_bot.crypto.garch_vol import GarchForecaster, GarchParams
from arb_bot.crypto.price_model import ProbabilityEstimate
from arb_bot.crypto.vol_spread_model import VolSpreadModel, VolSpreadSignal


# ── Helpers ────────────────────────────────────────────────────────

_MINUTES_PER_YEAR = 365.25 * 24.0 * 60.0


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


def _bs_binary_price(
    spot: float,
    strike: float,
    vol: float,
    tte_minutes: float,
    drift: float = 0.0,
) -> float:
    """Black-Scholes binary P(above) for testing."""
    dt = tte_minutes / _MINUTES_PER_YEAR
    d2 = (math.log(spot / strike) + (drift - 0.5 * vol**2) * dt) / (
        vol * math.sqrt(dt)
    )
    return float(_norm.cdf(d2))


# ── IV round-trip tests ──────────────────────────────────────────

class TestImpliedVol:
    """Verify IV extraction round-trips with analytical pricing."""

    def test_iv_roundtrip_atm(self) -> None:
        """IV extraction should recover the true vol for ATM contracts."""
        model = VolSpreadModel()
        true_vol = 0.80
        spot = 100_000.0
        strike = 100_000.0
        tte = 15.0

        market_price = _bs_binary_price(spot, strike, true_vol, tte)
        iv = model._extract_iv(market_price, strike, spot, tte, 0.0, "above")

        assert iv is not None
        assert abs(iv - true_vol) < 0.01

    def test_iv_roundtrip_otm(self) -> None:
        """IV extraction should recover vol for OTM contracts."""
        model = VolSpreadModel()
        true_vol = 1.20
        spot = 100_000.0
        strike = 101_000.0  # 1% OTM
        tte = 30.0

        market_price = _bs_binary_price(spot, strike, true_vol, tte)
        iv = model._extract_iv(market_price, strike, spot, tte, 0.0, "above")

        assert iv is not None
        assert abs(iv - true_vol) < 0.01

    def test_iv_roundtrip_below_direction(self) -> None:
        """IV extraction should work for 'below' contracts."""
        model = VolSpreadModel()
        true_vol = 0.60
        spot = 100_000.0
        strike = 99_500.0  # Slightly ITM for "below"
        tte = 15.0

        p_above = _bs_binary_price(spot, strike, true_vol, tte)
        market_price_below = 1.0 - p_above

        iv = model._extract_iv(market_price_below, strike, spot, tte, 0.0, "below")

        assert iv is not None
        assert abs(iv - true_vol) < 0.01

    def test_iv_extreme_price_returns_none(self) -> None:
        """Prices near 0 or 1 should return None."""
        model = VolSpreadModel()
        assert model._extract_iv(0.005, 100_000, 100_000, 15, 0, "above") is None
        assert model._extract_iv(0.995, 100_000, 100_000, 15, 0, "above") is None

    def test_iv_degenerate_inputs(self) -> None:
        """Invalid inputs should return None."""
        model = VolSpreadModel()
        assert model._extract_iv(0.5, 0, 100_000, 15, 0, "above") is None  # zero strike
        assert model._extract_iv(0.5, 100_000, 0, 15, 0, "above") is None  # zero spot


# ── GARCH probability tests ─────────────────────────────────────

class TestGarchProbability:
    """Test GARCH-based probability computation."""

    def test_atm_near_50_pct(self) -> None:
        """ATM binary should have ~50% probability."""
        model = VolSpreadModel()
        prob = model._garch_probability(
            spot=100_000, strike=100_000, garch_vol=0.80,
            tte_minutes=15, drift=0.0, direction="above",
        )
        # Should be close to 50% for ATM with zero drift
        assert 0.45 < prob < 0.55

    def test_deep_itm_high_prob(self) -> None:
        """Deep ITM (spot >> strike) should have high probability."""
        model = VolSpreadModel()
        prob = model._garch_probability(
            spot=102_000, strike=100_000, garch_vol=0.30,
            tte_minutes=15, drift=0.0, direction="above",
        )
        assert prob > 0.90

    def test_deep_otm_low_prob(self) -> None:
        """Deep OTM (spot << strike) should have low probability."""
        model = VolSpreadModel()
        prob = model._garch_probability(
            spot=98_000, strike=100_000, garch_vol=0.30,
            tte_minutes=15, drift=0.0, direction="above",
        )
        assert prob < 0.10

    def test_below_direction_complementary(self) -> None:
        """P(below) should equal 1 - P(above)."""
        model = VolSpreadModel()
        p_above = model._garch_probability(
            spot=100_500, strike=100_000, garch_vol=0.80,
            tte_minutes=15, drift=0.0, direction="above",
        )
        p_below = model._garch_probability(
            spot=100_500, strike=100_000, garch_vol=0.80,
            tte_minutes=15, drift=0.0, direction="below",
        )
        assert abs(p_above + p_below - 1.0) < 1e-10

    def test_degenerate_returns_half(self) -> None:
        """Degenerate inputs should return 0.5."""
        model = VolSpreadModel()
        assert model._garch_probability(0, 100_000, 0.80, 15, 0, "above") == 0.5
        assert model._garch_probability(100_000, 0, 0.80, 15, 0, "above") == 0.5
        assert model._garch_probability(100_000, 100_000, 0, 15, 0, "above") == 0.5


# ── Probability estimate output ──────────────────────────────────

class TestProbabilityEstimate:
    """Test compute_garch_probability returns valid ProbabilityEstimate."""

    def test_returns_probability_estimate(self) -> None:
        """Output should be a ProbabilityEstimate dataclass."""
        model = VolSpreadModel()
        result = model.compute_garch_probability(
            spot=100_000, strike=100_500, garch_vol=0.80,
            tte_minutes=15, drift=0.0, direction="above",
        )
        assert isinstance(result, ProbabilityEstimate)
        assert 0 <= result.probability <= 1
        assert result.ci_lower <= result.probability
        assert result.probability <= result.ci_upper
        assert result.uncertainty > 0
        assert result.num_paths == 0  # No MC used

    def test_uncertainty_decreases_with_history(self) -> None:
        """More spread history should reduce uncertainty."""
        model = VolSpreadModel()

        # No history
        est1 = model.compute_garch_probability(
            100_000, 100_500, 0.80, 15, 0.0, "above",
        )

        # Add some spread history
        for i in range(200):
            model._spread_history.append(0.01 * (i % 10 - 5))

        est2 = model.compute_garch_probability(
            100_000, 100_500, 0.80, 15, 0.0, "above",
        )

        assert est2.uncertainty < est1.uncertainty


# ── Z-score tests ────────────────────────────────────────────────

class TestZScore:
    """Test z-score computation."""

    def test_insufficient_history_returns_zero(self) -> None:
        """Z-score should be 0 with < 20 observations."""
        model = VolSpreadModel()
        for i in range(19):
            model._spread_history.append(float(i))
        assert model._zscore(10.0) == 0.0

    def test_zscore_centering(self) -> None:
        """Z-score of the mean should be ~0."""
        model = VolSpreadModel()
        rng = np.random.default_rng(42)
        for _ in range(100):
            model._spread_history.append(rng.normal(0.05, 0.02))

        arr = np.array(model._spread_history)
        mean = float(np.mean(arr))
        z = model._zscore(mean)
        assert abs(z) < 0.1

    def test_zscore_outlier_large(self) -> None:
        """An outlier should have a large |z-score|."""
        model = VolSpreadModel()
        for _ in range(100):
            model._spread_history.append(0.0)
        # Add a tiny bit of variance so std > 0
        model._spread_history.append(0.001)

        z = model._zscore(1.0)  # Way above mean
        assert abs(z) > 5.0

    def test_constant_history_returns_zero(self) -> None:
        """Zero std should return z-score of 0."""
        model = VolSpreadModel()
        for _ in range(50):
            model._spread_history.append(0.05)
        assert model._zscore(0.10) == 0.0


# ── Moneyness filter tests ───────────────────────────────────────

class TestMoneynessFilter:
    """Test the moneyness sigma filter."""

    def test_atm_rejected(self) -> None:
        """ATM (moneyness ~0) should be rejected by min_moneyness."""
        model = VolSpreadModel(min_moneyness_sigma=0.3)
        # ATM: spot == strike → log_moneyness = 0
        assert not model._moneyness_in_range(100_000, 100_000, 0.80, 15)

    def test_moderate_otm_accepted(self) -> None:
        """Moderately OTM should be accepted."""
        model = VolSpreadModel(min_moneyness_sigma=0.3, max_moneyness_sigma=2.5)
        # ~1 sigma OTM
        vol = 0.80
        tte = 15.0
        dt = tte / _MINUTES_PER_YEAR
        sigma_t = vol * math.sqrt(dt)
        strike = 100_000 * math.exp(1.0 * sigma_t)  # 1 sigma above

        assert model._moneyness_in_range(100_000, strike, vol, tte)

    def test_deep_otm_rejected(self) -> None:
        """Very deep OTM (> max_moneyness_sigma) should be rejected."""
        model = VolSpreadModel(max_moneyness_sigma=2.5)
        vol = 0.80
        tte = 15.0
        dt = tte / _MINUTES_PER_YEAR
        sigma_t = vol * math.sqrt(dt)
        strike = 100_000 * math.exp(3.0 * sigma_t)  # 3 sigma above

        assert not model._moneyness_in_range(100_000, strike, vol, tte)


# ── Signal direction tests ───────────────────────────────────────

class TestSignalDirection:
    """Test signal direction from z-score."""

    def _build_model_with_history(
        self, n: int = 200, mean: float = 0.0, std: float = 0.02,
    ) -> VolSpreadModel:
        """Create a model with pre-seeded spread history."""
        model = VolSpreadModel(entry_z_threshold=1.5, min_moneyness_sigma=0.0)
        rng = np.random.default_rng(42)
        for _ in range(n):
            model._spread_history.append(rng.normal(mean, std))
        return model

    def test_buy_signal_on_high_zscore(self) -> None:
        """GARCH vol >> IV should produce 'buy' signal (z > threshold)."""
        model = self._build_model_with_history(mean=0.0, std=0.02)

        # Manually check: a spread of +0.10 should be z ≈ 5 (far above 1.5)
        z = model._zscore(0.10)
        assert z > 1.5

    def test_sell_signal_on_low_zscore(self) -> None:
        """GARCH vol << IV should produce 'sell' signal (z < -threshold)."""
        model = self._build_model_with_history(mean=0.0, std=0.02)

        z = model._zscore(-0.10)
        assert z < -1.5

    def test_neutral_in_range(self) -> None:
        """Small spread should produce 'neutral' signal."""
        model = self._build_model_with_history(mean=0.0, std=0.10)

        z = model._zscore(0.01)  # Tiny deviation
        assert abs(z) < 1.5


# ── Full signal pipeline tests ───────────────────────────────────

class TestComputeSignal:
    """Test the full compute_signal pipeline."""

    def test_full_pipeline_produces_signal(self) -> None:
        """End-to-end: synthetic data should produce a valid signal."""
        returns = _simulate_garch(1e-6, 0.08, 0.90, n=1000, seed=42)

        garch = GarchForecaster(min_obs=100)
        model = VolSpreadModel(
            garch_forecaster=garch,
            min_moneyness_sigma=0.0,  # Accept ATM for testing
            entry_z_threshold=1.5,
        )

        # Seed spread history for z-scoring
        for _ in range(50):
            model._spread_history.append(0.01)

        # Spot / strike / market price that make sense
        spot = 100_000.0
        strike = 100_500.0
        tte = 15.0
        # Compute a market price from a slightly different vol
        market_price = _bs_binary_price(spot, strike, 0.70, tte)

        signal = model.compute_signal(
            returns, spot, strike, tte, market_price, "above",
        )

        assert signal is not None
        assert isinstance(signal, VolSpreadSignal)
        assert signal.garch_vol > 0
        assert signal.implied_vol > 0
        assert signal.signal_direction in ("buy", "sell", "neutral")
        assert 0 <= signal.garch_probability <= 1
        assert 0 <= signal.confidence <= 1

    def test_insufficient_data_returns_none(self) -> None:
        """Too few returns should return None."""
        returns = np.random.default_rng(42).normal(0, 0.01, 50)
        garch = GarchForecaster(min_obs=100)
        model = VolSpreadModel(garch_forecaster=garch)

        signal = model.compute_signal(
            returns, 100_000, 100_500, 15, 0.45, "above",
        )
        assert signal is None

    def test_extreme_market_price_returns_none(self) -> None:
        """Market price near 0 or 1 should fail IV extraction → None."""
        returns = _simulate_garch(1e-6, 0.08, 0.90, n=500, seed=42)
        garch = GarchForecaster(min_obs=100)
        model = VolSpreadModel(
            garch_forecaster=garch,
            min_moneyness_sigma=0.0,
        )

        signal = model.compute_signal(
            returns, 100_000, 100_500, 15, 0.001, "above",
        )
        assert signal is None

    def test_degenerate_inputs_return_none(self) -> None:
        """Zero spot/strike/tte should return None."""
        returns = _simulate_garch(1e-6, 0.08, 0.90, n=500)
        garch = GarchForecaster(min_obs=100)
        model = VolSpreadModel(garch_forecaster=garch)

        assert model.compute_signal(returns, 0, 100_000, 15, 0.5, "above") is None
        assert model.compute_signal(returns, 100_000, 0, 15, 0.5, "above") is None
        assert model.compute_signal(returns, 100_000, 100_000, 0, 0.5, "above") is None

    def test_spread_history_grows(self) -> None:
        """Each valid signal computation should add to spread history."""
        returns = _simulate_garch(1e-6, 0.08, 0.90, n=1000, seed=42)
        garch = GarchForecaster(min_obs=100)
        model = VolSpreadModel(
            garch_forecaster=garch,
            min_moneyness_sigma=0.0,
        )

        spot = 100_000.0
        strike = 100_500.0
        tte = 15.0
        market_price = _bs_binary_price(spot, strike, 0.80, tte)

        initial_len = model.spread_history_size
        signal = model.compute_signal(
            returns, spot, strike, tte, market_price, "above",
        )

        if signal is not None:
            assert model.spread_history_size == initial_len + 1

    def test_garch_refit_triggered(self) -> None:
        """Model should refit GARCH when needs_refit is True."""
        returns = _simulate_garch(1e-6, 0.08, 0.90, n=1000, seed=42)
        garch = GarchForecaster(min_obs=100, refit_interval=5)
        model = VolSpreadModel(
            garch_forecaster=garch,
            min_moneyness_sigma=0.0,
        )

        # Initial fit
        garch.fit(returns[:500])

        # Burn through refit interval
        sigma2 = garch.last_sigma2
        for _ in range(5):
            sigma2 = garch.update_online(0.001, sigma2)

        assert garch.needs_refit

        # compute_signal should trigger a refit
        spot = 100_000.0
        strike = 100_500.0
        tte = 15.0
        market_price = _bs_binary_price(spot, strike, 0.80, tte)

        signal = model.compute_signal(
            returns, spot, strike, tte, market_price, "above",
        )

        # After refit, needs_refit should be False
        assert not garch.needs_refit


# ── Signal confidence tests ──────────────────────────────────────

class TestSignalConfidence:
    """Test confidence scoring."""

    def test_higher_zscore_higher_confidence(self) -> None:
        """Larger |z-score| should give higher confidence."""
        model = VolSpreadModel()
        for _ in range(100):
            model._spread_history.append(0.0)

        params = GarchParams(
            omega=1e-6, alpha=0.08, beta=0.90,
            long_run_var=1e-6 / 0.02,
            log_likelihood=-100, n_obs=500, converged=True,
        )

        c_low = model._signal_confidence(0.5, params, 500)
        c_high = model._signal_confidence(2.5, params, 500)

        assert c_high > c_low

    def test_converged_higher_than_grid(self) -> None:
        """Converged MLE should give higher confidence than grid search."""
        model = VolSpreadModel()
        for _ in range(100):
            model._spread_history.append(0.0)

        params_conv = GarchParams(
            omega=1e-6, alpha=0.08, beta=0.90,
            long_run_var=1e-6 / 0.02,
            log_likelihood=-100, n_obs=500, converged=True,
        )
        params_grid = GarchParams(
            omega=1e-6, alpha=0.08, beta=0.90,
            long_run_var=1e-6 / 0.02,
            log_likelihood=-100, n_obs=500, converged=False,
        )

        c_conv = model._signal_confidence(2.0, params_conv, 500)
        c_grid = model._signal_confidence(2.0, params_grid, 500)

        assert c_conv > c_grid

    def test_more_data_higher_confidence(self) -> None:
        """More return data should give higher confidence."""
        model = VolSpreadModel()
        for _ in range(100):
            model._spread_history.append(0.0)

        params = GarchParams(
            omega=1e-6, alpha=0.08, beta=0.90,
            long_run_var=1e-6 / 0.02,
            log_likelihood=-100, n_obs=500, converged=True,
        )

        c_small = model._signal_confidence(2.0, params, 200)
        c_large = model._signal_confidence(2.0, params, 2000)

        assert c_large > c_small
