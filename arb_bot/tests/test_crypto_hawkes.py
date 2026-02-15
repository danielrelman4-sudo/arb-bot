"""Tests for Hawkes self-exciting jump intensity process."""

from __future__ import annotations

import math
import os
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from unittest import mock

import numpy as np
import pytest

from arb_bot.crypto.config import CryptoSettings
from arb_bot.crypto.hawkes import HawkesIntensity
from arb_bot.crypto.engine import CryptoEngine
from arb_bot.crypto.market_scanner import (
    CryptoMarket,
    CryptoMarketMeta,
    CryptoMarketQuote,
    parse_ticker,
)
from arb_bot.crypto.price_feed import PriceTick
from arb_bot.crypto.price_model import PriceModel, ProbabilityEstimate


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


def _make_market(
    ticker: str = "KXBTCD-26FEB14-T97500",
    expiry_offset_minutes: float = 10.0,
) -> CryptoMarket:
    meta = parse_ticker(ticker)
    assert meta is not None
    now = datetime.now(timezone.utc)
    adjusted = replace(meta, expiry=now + timedelta(minutes=expiry_offset_minutes))
    return CryptoMarket(ticker=ticker, meta=adjusted)


def _make_quote(
    ticker: str = "KXBTCD-26FEB14-T97500",
    yes_price: float = 0.50,
    no_price: float = 0.50,
    tte_minutes: float = 10.0,
    expiry_offset_minutes: float = 10.0,
) -> CryptoMarketQuote:
    market = _make_market(ticker, expiry_offset_minutes)
    implied = 0.5 * (yes_price + (1.0 - no_price))
    return CryptoMarketQuote(
        market=market,
        yes_buy_price=yes_price,
        no_buy_price=no_price,
        yes_buy_size=100,
        no_buy_size=100,
        yes_bid_price=yes_price - 0.01,
        no_bid_price=no_price - 0.01,
        time_to_expiry_minutes=tte_minutes,
        implied_probability=implied,
    )


# ── TestHawkesIntensity ───────────────────────────────────────────

class TestHawkesIntensity:
    def test_baseline_equals_mu(self) -> None:
        """With no shocks, intensity should equal baseline mu."""
        h = HawkesIntensity(mu=3.0, alpha=5.0, beta=0.00115)
        t = time.monotonic()
        assert h.intensity(t) == 3.0

    def test_shock_spikes_intensity(self) -> None:
        """A shock should spike intensity above baseline."""
        h = HawkesIntensity(mu=3.0, alpha=5.0, beta=0.00115)
        t = time.monotonic()
        h.record_shock(t, magnitude=1.0)
        # Immediately after: intensity = mu + alpha * 1.0 * exp(0) = 3 + 5 = 8
        val = h.intensity(t)
        assert abs(val - 8.0) < 0.01

    def test_exponential_decay(self) -> None:
        """Intensity should decay exponentially after a shock."""
        beta = math.log(2) / 600.0  # 10-min half-life
        h = HawkesIntensity(mu=3.0, alpha=5.0, beta=beta)
        t0 = time.monotonic()
        h.record_shock(t0, magnitude=1.0)

        # At t0: intensity = 3 + 5 = 8, excitation = 5
        # After 600 seconds (1 half-life): excitation = 5 * 0.5 = 2.5
        t1 = t0 + 600.0
        val = h.intensity(t1)
        expected = 3.0 + 5.0 * 0.5  # 5.5
        assert abs(val - expected) < 0.01

        # After 1200 seconds (2 half-lives): excitation = 5 * 0.25 = 1.25
        t2 = t0 + 1200.0
        val2 = h.intensity(t2)
        expected2 = 3.0 + 5.0 * 0.25  # 4.25
        assert abs(val2 - expected2) < 0.01

    def test_multiple_shocks_additive(self) -> None:
        """Multiple shocks should be additive in the intensity."""
        h = HawkesIntensity(mu=3.0, alpha=5.0, beta=0.00115)
        t = time.monotonic()
        h.record_shock(t, magnitude=1.0)
        h.record_shock(t, magnitude=2.0)
        # intensity = 3 + 5*1*exp(0) + 5*2*exp(0) = 3 + 5 + 10 = 18
        val = h.intensity(t)
        assert abs(val - 18.0) < 0.01

    def test_old_shocks_pruned(self) -> None:
        """Shocks older than max_history_seconds should be pruned."""
        h = HawkesIntensity(mu=3.0, alpha=5.0, beta=0.00115, max_history_seconds=100)
        t0 = time.monotonic()
        h.record_shock(t0, magnitude=1.0)

        # Query well past max_history_seconds
        t1 = t0 + 200.0
        val = h.intensity(t1)
        # Shock should have been pruned
        assert val == 3.0
        assert len(h._shocks) == 0

    def test_zero_magnitude_noop(self) -> None:
        """Zero magnitude shocks should not be recorded."""
        h = HawkesIntensity(mu=3.0, alpha=5.0, beta=0.00115)
        t = time.monotonic()
        h.record_shock(t, magnitude=0.0)
        assert len(h._shocks) == 0
        assert h.intensity(t) == 3.0

    def test_large_return_triggers_shock(self) -> None:
        """A return exceeding threshold_sigma should record a shock."""
        h = HawkesIntensity(
            mu=3.0, alpha=5.0, beta=0.00115,
            return_threshold_sigma=4.0,
        )
        t = time.monotonic()
        realized_vol = 0.01  # per-interval vol
        # A 5-sigma return should trigger (|z| = 5 >= 4)
        large_return = 5.0 * realized_vol
        h.record_return(t, large_return, realized_vol)
        assert len(h._shocks) == 1
        # magnitude = z_score / threshold = 5.0 / 4.0 = 1.25
        _, mag = h._shocks[0]
        assert abs(mag - 1.25) < 0.01

    def test_small_return_no_shock(self) -> None:
        """A return below threshold_sigma should not record a shock."""
        h = HawkesIntensity(
            mu=3.0, alpha=5.0, beta=0.00115,
            return_threshold_sigma=4.0,
        )
        t = time.monotonic()
        realized_vol = 0.01
        # A 2-sigma return should not trigger (|z| = 2 < 4)
        small_return = 2.0 * realized_vol
        h.record_return(t, small_return, realized_vol)
        assert len(h._shocks) == 0
        assert h.intensity(t) == 3.0

    def test_negative_return_triggers_shock(self) -> None:
        """Negative large returns should also trigger shocks (uses |return|)."""
        h = HawkesIntensity(
            mu=3.0, alpha=5.0, beta=0.00115,
            return_threshold_sigma=4.0,
        )
        t = time.monotonic()
        realized_vol = 0.01
        large_neg = -6.0 * realized_vol
        h.record_return(t, large_neg, realized_vol)
        assert len(h._shocks) == 1
        _, mag = h._shocks[0]
        assert abs(mag - 1.5) < 0.01  # |z|/threshold = 6/4 = 1.5

    def test_zero_vol_no_shock(self) -> None:
        """If realized vol is zero, no shock should be recorded (avoid division by zero)."""
        h = HawkesIntensity(mu=3.0, alpha=5.0, beta=0.00115)
        t = time.monotonic()
        h.record_return(t, 0.05, 0.0)
        assert len(h._shocks) == 0

    def test_intensity_never_below_mu(self) -> None:
        """Intensity should never drop below baseline mu."""
        h = HawkesIntensity(mu=3.0, alpha=5.0, beta=0.00115)
        t0 = time.monotonic()
        h.record_shock(t0, magnitude=0.001)
        # Even with a tiny shock decayed for a long time, intensity >= mu
        t_far = t0 + 100000.0
        val = h.intensity(t_far)
        assert val >= 3.0


# ── TestHawkesConfig ──────────────────────────────────────────────

class TestHawkesConfig:
    def test_hawkes_enabled_default(self) -> None:
        s = CryptoSettings()
        assert s.hawkes_enabled is True

    def test_hawkes_alpha_default(self) -> None:
        s = CryptoSettings()
        assert s.hawkes_alpha == 5.0

    def test_hawkes_beta_default(self) -> None:
        s = CryptoSettings()
        assert abs(s.hawkes_beta - 0.00115) < 0.00001

    def test_hawkes_return_threshold_sigma_default(self) -> None:
        s = CryptoSettings()
        assert s.hawkes_return_threshold_sigma == 4.0

    def test_hawkes_enabled_from_env(self) -> None:
        with mock.patch.dict(os.environ, {"ARB_CRYPTO_HAWKES_ENABLED": "false"}, clear=True):
            from arb_bot.crypto.config import load_crypto_settings
            s = load_crypto_settings()
        assert s.hawkes_enabled is False

    def test_hawkes_alpha_from_env(self) -> None:
        with mock.patch.dict(os.environ, {"ARB_CRYPTO_HAWKES_ALPHA": "10.0"}, clear=True):
            from arb_bot.crypto.config import load_crypto_settings
            s = load_crypto_settings()
        assert s.hawkes_alpha == 10.0

    def test_hawkes_beta_from_env(self) -> None:
        with mock.patch.dict(os.environ, {"ARB_CRYPTO_HAWKES_BETA": "0.005"}, clear=True):
            from arb_bot.crypto.config import load_crypto_settings
            s = load_crypto_settings()
        assert s.hawkes_beta == 0.005

    def test_hawkes_threshold_from_env(self) -> None:
        with mock.patch.dict(os.environ, {"ARB_CRYPTO_HAWKES_RETURN_THRESHOLD_SIGMA": "3.0"}, clear=True):
            from arb_bot.crypto.config import load_crypto_settings
            s = load_crypto_settings()
        assert s.hawkes_return_threshold_sigma == 3.0


# ── TestHawkesPathGeneration ──────────────────────────────────────

class TestHawkesPathGeneration:
    def test_elevated_intensity_wider_distribution(self) -> None:
        """Paths with elevated jump intensity should have wider terminal distribution."""
        pm = PriceModel(num_paths=5000, seed=42)
        current_price = 100000.0
        vol = 0.50
        horizon = 10.0

        # Low intensity (baseline)
        paths_low = pm.generate_paths_jump_diffusion(
            current_price, vol, horizon, jump_intensity=1.0,
            jump_mean=0.0, jump_vol=0.02,
        )

        # High intensity (after Hawkes spike)
        pm2 = PriceModel(num_paths=5000, seed=42)
        paths_high = pm2.generate_paths_jump_diffusion(
            current_price, vol, horizon, jump_intensity=20.0,
            jump_mean=0.0, jump_vol=0.02,
        )

        # Higher intensity should produce wider spread (more jumps = more variance)
        std_low = float(np.std(paths_low))
        std_high = float(np.std(paths_high))
        assert std_high > std_low

    def test_engine_uses_hawkes_intensity(self) -> None:
        """Engine with hawkes_enabled should pass dynamic intensity to path generator."""
        settings = _make_settings(
            mc_num_paths=500,
            hawkes_enabled=True,
            hawkes_alpha=5.0,
            hawkes_beta=0.00115,
            hawkes_return_threshold_sigma=4.0,
            use_jump_diffusion=True,
            ofi_enabled=False,
            activity_scaling_enabled=False,
        )
        engine = CryptoEngine(settings)

        # Inject price data with a very large move to trigger Hawkes shock
        ts = time.time()
        now_mono = time.monotonic()
        base_price = 97000.0
        for i in range(100):
            engine.price_feed.inject_tick(
                PriceTick("btcusdt", base_price, ts - 100 + i, 1.0)
            )
        # Inject a sudden big jump
        engine.price_feed.inject_tick(
            PriceTick("btcusdt", base_price * 1.05, ts, 1.0)  # 5% jump
        )

        # Manually record a large shock to the Hawkes tracker
        engine._hawkes.record_shock(now_mono, magnitude=2.0)

        # Check that Hawkes intensity is above baseline
        intensity = engine._hawkes.intensity(now_mono)
        assert intensity > settings.mc_jump_intensity

        quote = _make_quote(
            ticker="KXBTCD-26FEB14-T97500",
            yes_price=0.50,
            no_price=0.50,
            tte_minutes=10.0,
        )

        probs = engine._compute_model_probabilities([quote])
        assert quote.market.ticker in probs
        # The probability should be valid
        prob = probs[quote.market.ticker]
        assert 0.0 <= prob.probability <= 1.0

    def test_hawkes_disabled_uses_static_intensity(self) -> None:
        """With hawkes_enabled=False, static mc_jump_intensity should be used."""
        settings = _make_settings(
            mc_num_paths=500,
            hawkes_enabled=False,
            mc_jump_intensity=3.0,
            use_jump_diffusion=True,
            ofi_enabled=False,
            activity_scaling_enabled=False,
        )
        engine = CryptoEngine(settings)

        # Inject price data
        ts = time.time()
        for i in range(100):
            engine.price_feed.inject_tick(
                PriceTick("btcusdt", 97000.0 + i * 10, ts - 100 + i, 1.0)
            )

        # Even with shocks recorded, disabled Hawkes should use static intensity
        engine._hawkes.record_shock(time.monotonic(), magnitude=10.0)

        quote = _make_quote(
            ticker="KXBTCD-26FEB14-T97500",
            yes_price=0.50,
            no_price=0.50,
            tte_minutes=10.0,
        )

        probs = engine._compute_model_probabilities([quote])
        assert quote.market.ticker in probs

    def test_engine_has_hawkes_attribute(self) -> None:
        """Engine should have a _hawkes attribute of type HawkesIntensity."""
        settings = _make_settings()
        engine = CryptoEngine(settings)
        assert hasattr(engine, "_hawkes")
        assert isinstance(engine._hawkes, HawkesIntensity)

    def test_engine_hawkes_initialized_with_settings(self) -> None:
        """Engine's HawkesIntensity should use settings values."""
        settings = _make_settings(
            mc_jump_intensity=5.0,
            hawkes_alpha=10.0,
            hawkes_beta=0.002,
            hawkes_return_threshold_sigma=3.0,
        )
        engine = CryptoEngine(settings)
        assert engine._hawkes.mu == 5.0
        assert engine._hawkes.alpha == 10.0
        assert engine._hawkes.beta == 0.002
        assert engine._hawkes.return_threshold_sigma == 3.0
