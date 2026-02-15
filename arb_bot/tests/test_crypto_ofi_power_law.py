"""Tests for the power-law OFI impact model (Task 1).

Verifies:
- Config field ofi_impact_exponent exists and defaults to 0.5
- _apply_ofi_impact() produces correct power-law drift
- OFICalibrator returns theta and recovers power-law exponents
"""
from __future__ import annotations

import math
import os
from unittest import mock

import numpy as np
import pytest

from arb_bot.crypto.config import CryptoSettings, load_crypto_settings
from arb_bot.crypto.engine import CryptoEngine
from arb_bot.crypto.ofi_calibrator import OFICalibrationResult, OFICalibrator


# ── Helpers ────────────────────────────────────────────────────────

def _make_settings(**overrides) -> CryptoSettings:
    defaults = dict(
        enabled=True,
        paper_mode=True,
        bankroll=1000.0,
        mc_num_paths=100,
        min_edge_pct=0.05,
        min_edge_cents=0.02,
        max_model_uncertainty=0.15,
        kelly_fraction_cap=0.10,
        max_position_per_market=100.0,
        max_concurrent_positions=10,
        scan_interval_seconds=0.01,
        paper_slippage_cents=0.0,
        confidence_level=0.95,
        mc_vol_window_minutes=30,
        price_feed_symbols=["btcusdt"],
        symbols=["KXBTC"],
        min_minutes_to_expiry=2,
        max_minutes_to_expiry=60,
    )
    defaults.update(overrides)
    return CryptoSettings(**defaults)


# ── Config tests ──────────────────────────────────────────────────

class TestPowerLawConfig:
    def test_default_exponent(self) -> None:
        """ofi_impact_exponent defaults to 0.5."""
        s = CryptoSettings()
        assert s.ofi_impact_exponent == 0.5

    def test_configurable_exponent(self) -> None:
        """ofi_impact_exponent can be set to a custom value."""
        s = CryptoSettings(ofi_impact_exponent=0.7)
        assert s.ofi_impact_exponent == 0.7

    def test_exponent_from_env(self) -> None:
        """ofi_impact_exponent can be loaded from environment variable."""
        env = {"ARB_CRYPTO_OFI_IMPACT_EXPONENT": "0.8"}
        with mock.patch.dict(os.environ, env, clear=True):
            s = load_crypto_settings()
        assert s.ofi_impact_exponent == 0.8

    def test_exponent_default_from_env(self) -> None:
        """Without env var, load_crypto_settings uses default 0.5."""
        with mock.patch.dict(os.environ, {}, clear=True):
            s = load_crypto_settings()
        assert s.ofi_impact_exponent == 0.5


# ── Drift impact tests ───────────────────────────────────────────

class TestPowerLawDrift:
    def test_sqrt_law_dampens_extremes(self) -> None:
        """With theta=0.5, large OFI values are dampened (sqrt law)."""
        settings = _make_settings(ofi_impact_exponent=0.5)
        engine = CryptoEngine(settings)

        alpha = 1.0
        # Linear: drift = 1.0 * 100 = 100
        # Sqrt:   drift = 1.0 * sqrt(100) = 10
        drift = engine._apply_ofi_impact(100.0, alpha)
        assert abs(drift - 10.0) < 1e-10

    def test_preserves_sign_positive(self) -> None:
        """Positive OFI produces positive drift."""
        settings = _make_settings(ofi_impact_exponent=0.5)
        engine = CryptoEngine(settings)
        drift = engine._apply_ofi_impact(4.0, 1.0)
        assert drift > 0
        assert abs(drift - 2.0) < 1e-10  # 1.0 * sqrt(4) = 2.0

    def test_preserves_sign_negative(self) -> None:
        """Negative OFI produces negative drift."""
        settings = _make_settings(ofi_impact_exponent=0.5)
        engine = CryptoEngine(settings)
        drift = engine._apply_ofi_impact(-4.0, 1.0)
        assert drift < 0
        assert abs(drift - (-2.0)) < 1e-10  # 1.0 * -sqrt(4) = -2.0

    def test_theta_one_is_linear(self) -> None:
        """With theta=1.0, power law reduces to linear model."""
        settings = _make_settings(ofi_impact_exponent=1.0)
        engine = CryptoEngine(settings)
        drift = engine._apply_ofi_impact(5.0, 0.3)
        assert abs(drift - 0.3 * 5.0) < 1e-10

    def test_zero_ofi_gives_zero(self) -> None:
        """OFI of zero should always give zero drift."""
        settings = _make_settings(ofi_impact_exponent=0.5)
        engine = CryptoEngine(settings)
        drift = engine._apply_ofi_impact(0.0, 0.5)
        assert drift == 0.0

    def test_zero_alpha_gives_zero(self) -> None:
        """Alpha of zero should always give zero drift."""
        settings = _make_settings(ofi_impact_exponent=0.5)
        engine = CryptoEngine(settings)
        drift = engine._apply_ofi_impact(10.0, 0.0)
        assert drift == 0.0

    def test_negative_alpha_flips_sign(self) -> None:
        """Negative alpha flips the sign of drift."""
        settings = _make_settings(ofi_impact_exponent=0.5)
        engine = CryptoEngine(settings)
        drift = engine._apply_ofi_impact(4.0, -1.0)
        assert drift < 0
        assert abs(drift - (-2.0)) < 1e-10  # -1.0 * sqrt(4) = -2.0

    def test_fractional_ofi(self) -> None:
        """Small fractional OFI is amplified under sqrt law."""
        settings = _make_settings(ofi_impact_exponent=0.5)
        engine = CryptoEngine(settings)
        # sqrt(0.01) = 0.1, so drift = 1.0 * 0.1 = 0.1
        drift = engine._apply_ofi_impact(0.01, 1.0)
        assert abs(drift - 0.1) < 1e-10


# ── Calibrator tests ─────────────────────────────────────────────

class TestPowerLawCalibrator:
    def test_calibration_result_has_theta(self) -> None:
        """OFICalibrationResult must have a theta field."""
        result = OFICalibrationResult(alpha=0.1, theta=0.5, r_squared=0.8, n_samples=50)
        assert result.theta == 0.5

    def test_recovers_sqrt_from_synthetic_data(self) -> None:
        """Given y = alpha * sgn(x) * |x|^0.5, calibrator should recover theta ~ 0.5."""
        rng = np.random.default_rng(42)
        cal = OFICalibrator(max_samples=5000, min_samples=30, min_r_squared=0.01)

        true_alpha = 0.3
        true_theta = 0.5
        for _ in range(500):
            ofi = rng.uniform(-2.0, 2.0)
            fwd = true_alpha * math.copysign(abs(ofi) ** true_theta, ofi) + rng.normal(0, 0.005)
            cal.record_sample(ofi, fwd)

        result = cal.calibrate()
        assert abs(result.theta - 0.5) < 0.15, f"theta={result.theta}, expected ~0.5"
        assert abs(result.alpha - 0.3) < 0.1, f"alpha={result.alpha}, expected ~0.3"
        assert result.r_squared > 0.5

    def test_insufficient_samples_returns_defaults(self) -> None:
        """With too few samples, theta should default to 0.5 and alpha to 0.0."""
        cal = OFICalibrator(min_samples=30)
        for i in range(5):
            cal.record_sample(float(i) * 0.1, float(i) * 0.05)

        result = cal.calibrate()
        assert result.alpha == 0.0
        assert result.theta == 0.5
        assert result.n_samples == 5

    def test_pure_noise_returns_default_theta(self) -> None:
        """When data is pure noise, alpha should be 0.0 and theta defaults to 0.5."""
        rng = np.random.default_rng(42)
        cal = OFICalibrator(max_samples=5000, min_samples=30, min_r_squared=0.01)

        for _ in range(100):
            ofi = rng.uniform(-1.0, 1.0)
            fwd = rng.normal(0, 1.0)  # pure noise
            cal.record_sample(ofi, fwd)

        result = cal.calibrate()
        assert result.alpha == 0.0
        assert result.theta == 0.5  # default when signal is noise

    def test_linear_data_recovers_theta_one(self) -> None:
        """Given y = alpha * x (theta=1.0), calibrator should recover theta ~ 1.0."""
        rng = np.random.default_rng(42)
        cal = OFICalibrator(max_samples=5000, min_samples=30, min_r_squared=0.01)

        true_alpha = 0.5
        for _ in range(500):
            ofi = rng.uniform(-2.0, 2.0)
            fwd = true_alpha * ofi + rng.normal(0, 0.005)
            cal.record_sample(ofi, fwd)

        result = cal.calibrate()
        assert abs(result.theta - 1.0) < 0.15, f"theta={result.theta}, expected ~1.0"
        assert abs(result.alpha - 0.5) < 0.1, f"alpha={result.alpha}, expected ~0.5"

    def test_zero_ofi_data_returns_defaults(self) -> None:
        """All-zero OFI data should give alpha=0.0, theta=0.5 (degenerate)."""
        rng = np.random.default_rng(42)
        cal = OFICalibrator(min_samples=10)

        for _ in range(50):
            cal.record_sample(0.0, rng.normal(0, 0.01))

        result = cal.calibrate()
        assert result.alpha == 0.0
        assert result.theta == 0.5
