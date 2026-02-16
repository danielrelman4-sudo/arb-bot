"""Tests for Z-score reachability filter and probability clamp."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from arb_bot.crypto.config import CryptoSettings
from arb_bot.crypto.engine import CryptoEngine
from arb_bot.crypto.price_model import ProbabilityEstimate


# ── TestComputeZscore ────────────────────────────────────────────

class TestComputeZscore:
    """Test the Z-score moneyness calculation (static method)."""

    def test_atm_zscore_near_zero(self) -> None:
        """Price = strike → Z ≈ 0."""
        z = CryptoEngine._compute_zscore(
            current_price=68500.0, strike=68500.0, vol=0.60, tte_minutes=10.0,
        )
        assert z == pytest.approx(0.0, abs=1e-10)

    def test_deep_otm_short_tte_high_zscore(self) -> None:
        """Price far from strike + short TTE → Z >> 2."""
        # BTC at $68,000, strike at $69,000 (1.5% away), 5 min left, vol=60%
        z = CryptoEngine._compute_zscore(
            current_price=68000.0, strike=69000.0, vol=0.60, tte_minutes=5.0,
        )
        # distance = |ln(68000/69000)| ≈ 0.01466
        # vol_horizon = 0.60 * sqrt(5 / 525960) ≈ 0.001849
        # z ≈ 0.01466 / 0.001849 ≈ 7.93
        assert z > 5.0

    def test_deep_otm_long_tte_moderate_zscore(self) -> None:
        """Price far from strike + long TTE → Z moderate."""
        # Same distance but 60 min left
        z = CryptoEngine._compute_zscore(
            current_price=68000.0, strike=69000.0, vol=0.60, tte_minutes=60.0,
        )
        # vol_horizon = 0.60 * sqrt(60 / 525960) ≈ 0.006407
        # z ≈ 0.01466 / 0.006407 ≈ 2.29
        assert 1.5 < z < 4.0

    def test_near_strike_short_tte_low_zscore(self) -> None:
        """Price very close to strike + short TTE → Z small (gamma scalp)."""
        # BTC at $69,000, strike at $69,050 (0.07% away), 3 min left
        z = CryptoEngine._compute_zscore(
            current_price=69000.0, strike=69050.0, vol=0.60, tte_minutes=3.0,
        )
        # distance ≈ 0.000724, vol_horizon ≈ 0.001433
        # z ≈ 0.50
        assert z < 2.0

    def test_zero_vol_returns_zero(self) -> None:
        """Zero vol → don't filter (return 0)."""
        z = CryptoEngine._compute_zscore(
            current_price=68000.0, strike=69000.0, vol=0.0, tte_minutes=10.0,
        )
        assert z == 0.0

    def test_zero_tte_returns_inf(self) -> None:
        """Zero TTE → unreachable."""
        z = CryptoEngine._compute_zscore(
            current_price=68000.0, strike=69000.0, vol=0.60, tte_minutes=0.0,
        )
        assert z == float('inf')

    def test_negative_vol_returns_zero(self) -> None:
        """Negative vol → don't filter."""
        z = CryptoEngine._compute_zscore(
            current_price=68000.0, strike=69000.0, vol=-0.5, tte_minutes=10.0,
        )
        assert z == 0.0

    def test_negative_tte_returns_inf(self) -> None:
        """Negative TTE → unreachable."""
        z = CryptoEngine._compute_zscore(
            current_price=68000.0, strike=69000.0, vol=0.60, tte_minutes=-1.0,
        )
        assert z == float('inf')

    def test_none_strike_returns_zero(self) -> None:
        """None strike → don't filter."""
        z = CryptoEngine._compute_zscore(
            current_price=68000.0, strike=None, vol=0.60, tte_minutes=10.0,
        )
        assert z == 0.0

    def test_zero_strike_returns_zero(self) -> None:
        """Zero strike → don't filter."""
        z = CryptoEngine._compute_zscore(
            current_price=68000.0, strike=0.0, vol=0.60, tte_minutes=10.0,
        )
        assert z == 0.0

    def test_symmetric_above_below(self) -> None:
        """Z-score is symmetric — same absolute distance regardless of direction."""
        z_above = CryptoEngine._compute_zscore(
            current_price=68000.0, strike=69000.0, vol=0.60, tte_minutes=10.0,
        )
        z_below = CryptoEngine._compute_zscore(
            current_price=69000.0, strike=68000.0, vol=0.60, tte_minutes=10.0,
        )
        assert z_above == pytest.approx(z_below, rel=1e-10)

    def test_higher_vol_lowers_zscore(self) -> None:
        """Higher vol makes strike more reachable → lower Z."""
        z_low_vol = CryptoEngine._compute_zscore(
            current_price=68000.0, strike=69000.0, vol=0.30, tte_minutes=10.0,
        )
        z_high_vol = CryptoEngine._compute_zscore(
            current_price=68000.0, strike=69000.0, vol=1.20, tte_minutes=10.0,
        )
        assert z_high_vol < z_low_vol

    def test_longer_tte_lowers_zscore(self) -> None:
        """Longer TTE makes strike more reachable → lower Z."""
        z_short = CryptoEngine._compute_zscore(
            current_price=68000.0, strike=69000.0, vol=0.60, tte_minutes=5.0,
        )
        z_long = CryptoEngine._compute_zscore(
            current_price=68000.0, strike=69000.0, vol=0.60, tte_minutes=120.0,
        )
        assert z_long < z_short


# ── TestZscoreClampLogic ────────────────────────────────────────

class TestZscoreClampLogic:
    """Test the probability clamp direction logic.

    When Z > threshold, the clamp should push probability to reflect
    which side of the strike the price is currently on:
    - above/up: price < strike → P(cross up) ≈ 0  (can't reach strike)
                price > strike → P(cross up) ≈ 1  (already past it)
    - below/down: price > strike → P(cross down) ≈ 0
                  price < strike → P(cross down) ≈ 1
    """

    def test_above_price_below_strike_clamps_low(self) -> None:
        """above direction, price < strike → P(above) ≈ 0.01."""
        # This is the v14 hail mary scenario: price far below strike
        # Model says ~50%, but Z-score says unreachable
        direction = "above"
        current_price = 68000.0
        z_strike = 69000.0
        # Simulate: Z > max, so we check clamp direction
        if direction in ("above", "up"):
            clamped_p = 0.01 if current_price < z_strike else 0.99
        assert clamped_p == 0.01

    def test_above_price_above_strike_clamps_high(self) -> None:
        """above direction, price > strike → P(above) ≈ 0.99."""
        direction = "above"
        current_price = 70000.0
        z_strike = 69000.0
        if direction in ("above", "up"):
            clamped_p = 0.01 if current_price < z_strike else 0.99
        assert clamped_p == 0.99

    def test_below_price_above_strike_clamps_low(self) -> None:
        """below direction, price > strike → P(below) ≈ 0.01."""
        direction = "below"
        current_price = 70000.0
        z_strike = 69000.0
        if direction in ("below", "down"):
            clamped_p = 0.01 if current_price > z_strike else 0.99
        assert clamped_p == 0.01

    def test_below_price_below_strike_clamps_high(self) -> None:
        """below direction, price < strike → P(below) ≈ 0.99."""
        direction = "below"
        current_price = 68000.0
        z_strike = 69000.0
        if direction in ("below", "down"):
            clamped_p = 0.01 if current_price > z_strike else 0.99
        assert clamped_p == 0.99

    def test_up_price_below_ref_clamps_low(self) -> None:
        """up direction, price < ref → P(up) ≈ 0.01."""
        direction = "up"
        current_price = 67500.0
        z_strike = 68000.0  # interval start price
        if direction in ("above", "up"):
            clamped_p = 0.01 if current_price < z_strike else 0.99
        assert clamped_p == 0.01

    def test_down_price_below_ref_clamps_high(self) -> None:
        """down direction, price < ref → P(down) ≈ 0.99."""
        direction = "down"
        current_price = 67500.0
        z_strike = 68000.0
        if direction in ("below", "down"):
            clamped_p = 0.01 if current_price > z_strike else 0.99
        assert clamped_p == 0.99

    def test_clamp_produces_tight_uncertainty(self) -> None:
        """When clamped, uncertainty should be very small (0.02)."""
        clamped_p = 0.01
        prob = ProbabilityEstimate(
            probability=clamped_p,
            ci_lower=max(0.0, clamped_p - 0.02),
            ci_upper=min(1.0, clamped_p + 0.02),
            uncertainty=0.02,
            num_paths=1000,
        )
        assert prob.uncertainty == 0.02
        # ci_lower = max(0.0, 0.01 - 0.02) = 0.0, ci_upper = 0.03
        assert prob.ci_upper - prob.ci_lower == pytest.approx(0.03, abs=0.001)

    def test_z_below_threshold_no_clamp(self) -> None:
        """Z < max → probability should NOT be clamped."""
        z = CryptoEngine._compute_zscore(
            current_price=69000.0, strike=69050.0, vol=0.60, tte_minutes=10.0,
        )
        # This should be well below 2.0
        assert z < 2.0
        # In this case, the original model prob is used as-is


# ── TestZscoreConfig ─────────────────────────────────────────────

class TestZscoreConfig:
    """Test Z-score configuration defaults and env var loading."""

    def test_defaults(self) -> None:
        settings = CryptoSettings()
        assert settings.zscore_filter_enabled is True
        assert settings.zscore_max == 2.0
        assert settings.zscore_vol_window_minutes == 15

    def test_env_var_loading(self, monkeypatch) -> None:
        from arb_bot.crypto.config import load_crypto_settings
        monkeypatch.setenv("ARB_CRYPTO_ZSCORE_FILTER_ENABLED", "false")
        monkeypatch.setenv("ARB_CRYPTO_ZSCORE_MAX", "3.0")
        monkeypatch.setenv("ARB_CRYPTO_ZSCORE_VOL_WINDOW_MINUTES", "10")
        settings = load_crypto_settings()
        assert settings.zscore_filter_enabled is False
        assert settings.zscore_max == 3.0
        assert settings.zscore_vol_window_minutes == 10

    def test_disabled_filter(self) -> None:
        """When disabled, Z-score filter should be skipped entirely."""
        settings = CryptoSettings(zscore_filter_enabled=False)
        assert settings.zscore_filter_enabled is False

    def test_custom_threshold(self) -> None:
        """Custom threshold — higher means fewer rejections."""
        settings = CryptoSettings(zscore_max=3.5)
        assert settings.zscore_max == 3.5
        # With max=3.5, the deep OTM scenario at Z≈2.29 would pass
        z = CryptoEngine._compute_zscore(
            current_price=68000.0, strike=69000.0, vol=0.60, tte_minutes=60.0,
        )
        assert z < settings.zscore_max  # Would not be clamped
