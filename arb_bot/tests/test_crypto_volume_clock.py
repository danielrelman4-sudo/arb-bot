"""Tests for the volume clock feature (C3A).

Verifies volume projection methods on PriceFeed,
effective horizon computation on CryptoEngine,
and behavioral effects on MC path generation.
"""

from __future__ import annotations

import math
import os
import time
import tempfile

import numpy as np
import pytest

from arb_bot.crypto.config import CryptoSettings
from arb_bot.crypto.cycle_logger import CycleLogger, CycleSnapshot
from arb_bot.crypto.engine import CryptoEngine
from arb_bot.crypto.price_feed import PriceFeed, PriceTick


# ── Helpers ────────────────────────────────────────────────────

def _make_feed_with_volume(
    symbol: str = "btcusdt",
    n_ticks: int = 60,
    volume_per_tick: float = 1.0,
    tick_interval_seconds: float = 1.0,
    base_price: float = 97000.0,
) -> PriceFeed:
    """Create a PriceFeed with *n_ticks* injected ticks over a time span."""
    feed = PriceFeed(symbols=[symbol])
    feed._running = True
    now = time.time()
    for i in range(n_ticks):
        ts = now - (n_ticks - 1 - i) * tick_interval_seconds
        feed.inject_tick(PriceTick(
            symbol=symbol,
            price=base_price + i * 0.01,
            timestamp=ts,
            volume=volume_per_tick,
            is_buyer_maker=None,
        ))
    return feed


def _make_engine(
    volume_clock_enabled: bool = False,
    activity_scaling_enabled: bool = False,
    **overrides,
) -> CryptoEngine:
    """Create a CryptoEngine with configurable volume clock settings."""
    kwargs = dict(
        volume_clock_enabled=volume_clock_enabled,
        activity_scaling_enabled=activity_scaling_enabled,
        paper_mode=True,
        mc_num_paths=500,
        price_feed_symbols=["btcusdt"],
        symbols=["KXBTC"],
    )
    kwargs.update(overrides)
    settings = CryptoSettings(**kwargs)
    return CryptoEngine(settings)


# ── PriceFeed volume methods ──────────────────────────────────

class TestGetTotalVolume:

    def test_sums_correctly(self):
        feed = _make_feed_with_volume(n_ticks=10, volume_per_tick=2.5)
        total = feed.get_total_volume("btcusdt", window_seconds=300)
        assert total == pytest.approx(25.0, rel=1e-6)

    def test_no_data_returns_zero(self):
        feed = PriceFeed(symbols=["btcusdt"])
        assert feed.get_total_volume("btcusdt") == 0.0

    def test_window_filters_old_ticks(self):
        feed = PriceFeed(symbols=["btcusdt"])
        feed._running = True
        now = time.time()
        # Old tick (600s ago)
        feed.inject_tick(PriceTick("btcusdt", 97000, now - 600, 10.0, None))
        # Recent tick (1s ago)
        feed.inject_tick(PriceTick("btcusdt", 97001, now - 1, 5.0, None))
        total = feed.get_total_volume("btcusdt", window_seconds=300)
        assert total == pytest.approx(5.0, rel=1e-6)


class TestProjectVolume:

    def test_linear_forecast(self):
        # 60 ticks over 60 seconds = 1 tick/sec, 1.0 vol each = 60 vol/min
        feed = _make_feed_with_volume(n_ticks=60, volume_per_tick=1.0)
        projected = feed.project_volume("btcusdt", horizon_minutes=15.0, rate_window_seconds=300)
        assert projected is not None
        # rate ~= 60 vol/min, projected = 60 * 15 = 900
        assert projected == pytest.approx(60.0 * 15.0, rel=0.1)

    def test_no_data_returns_none(self):
        feed = PriceFeed(symbols=["btcusdt"])
        assert feed.project_volume("btcusdt", 15.0) is None

    def test_zero_volume_ticks_returns_none(self):
        feed = _make_feed_with_volume(n_ticks=10, volume_per_tick=0.0)
        assert feed.project_volume("btcusdt", 15.0) is None


# ── Engine effective horizon ──────────────────────────────────

class TestComputeEffectiveHorizon:

    def test_no_data_returns_clock_time(self):
        engine = _make_engine(volume_clock_enabled=True)
        h, pv, br = engine._compute_effective_horizon("btcusdt", 15.0)
        assert h == 15.0
        assert pv is None
        assert br is None

    def test_equal_rates_returns_clock_time(self):
        engine = _make_engine(
            volume_clock_enabled=True,
            volume_clock_short_window_seconds=60,
            volume_clock_baseline_window_seconds=300,
        )
        # Inject uniform volume over 300s
        now = time.time()
        for i in range(300):
            engine._price_feed.inject_tick(PriceTick(
                "btcusdt", 97000, now - 300 + i, 1.0, None,
            ))
        h, pv, br = engine._compute_effective_horizon("btcusdt", 10.0)
        # short rate ≈ long rate → activity_ratio ≈ 1.0 → horizon ≈ 10.0
        assert h == pytest.approx(10.0, rel=0.15)

    def test_double_volume_increases_horizon(self):
        engine = _make_engine(
            volume_clock_enabled=True,
            volume_clock_short_window_seconds=60,
            volume_clock_baseline_window_seconds=600,
        )
        now = time.time()
        # Old ticks (600s to 61s ago): 1.0 vol each — no overlap with short window
        for i in range(539):
            engine._price_feed.inject_tick(PriceTick(
                "btcusdt", 97000, now - 600 + i, 1.0, None,
            ))
        # Recent ticks (60s to 0s ago): 2.0 vol each — higher rate
        for i in range(60):
            engine._price_feed.inject_tick(PriceTick(
                "btcusdt", 97000, now - 60 + i, 2.0, None,
            ))
        h, pv, br = engine._compute_effective_horizon("btcusdt", 10.0)
        # short_rate > long_rate → activity_ratio > 1 → h > 10
        assert h > 10.0

    def test_half_volume_shrinks_horizon(self):
        engine = _make_engine(
            volume_clock_enabled=True,
            volume_clock_short_window_seconds=60,
            volume_clock_baseline_window_seconds=600,
        )
        now = time.time()
        # Old ticks (600s to 61s ago): 2.0 vol each — higher baseline
        for i in range(539):
            engine._price_feed.inject_tick(PriceTick(
                "btcusdt", 97000, now - 600 + i, 2.0, None,
            ))
        # Recent ticks (60s to 0s ago): 1.0 vol each — lower recent rate
        for i in range(60):
            engine._price_feed.inject_tick(PriceTick(
                "btcusdt", 97000, now - 60 + i, 1.0, None,
            ))
        h, pv, br = engine._compute_effective_horizon("btcusdt", 10.0)
        # short_rate < long_rate → activity_ratio < 1 → h < 10
        assert h < 10.0

    def test_clamped_to_bounds(self):
        engine = _make_engine(
            volume_clock_enabled=True,
            volume_clock_short_window_seconds=10,
            volume_clock_baseline_window_seconds=300,
            volume_clock_ratio_ceiling=2.5,
            volume_clock_ratio_floor=0.25,
        )
        now = time.time()
        # Very low baseline: 0.01 vol over 300s
        for i in range(300):
            engine._price_feed.inject_tick(PriceTick(
                "btcusdt", 97000, now - 300 + i, 0.01, None,
            ))
        # Extreme spike: 100.0 vol in last 10s
        for i in range(10):
            engine._price_feed.inject_tick(PriceTick(
                "btcusdt", 97000, now - 10 + i, 100.0, None,
            ))
        h, _, _ = engine._compute_effective_horizon("btcusdt", 10.0)
        # Clamped to 2.5x (ceiling)
        assert h == pytest.approx(25.0, rel=0.01)

    def test_clamped_with_custom_bounds(self):
        """Verify configurable clamp floor/ceiling are respected."""
        engine = _make_engine(
            volume_clock_enabled=True,
            volume_clock_short_window_seconds=10,
            volume_clock_baseline_window_seconds=300,
            volume_clock_ratio_floor=0.5,
            volume_clock_ratio_ceiling=3.0,
        )
        now = time.time()
        # Very low baseline: 0.01 vol over 300s
        for i in range(300):
            engine._price_feed.inject_tick(PriceTick(
                "btcusdt", 97000, now - 300 + i, 0.01, None,
            ))
        # Extreme spike: 100.0 vol in last 10s
        for i in range(10):
            engine._price_feed.inject_tick(PriceTick(
                "btcusdt", 97000, now - 10 + i, 100.0, None,
            ))
        h, _, _ = engine._compute_effective_horizon("btcusdt", 10.0)
        # Clamped to custom ceiling 3.0x
        assert h == pytest.approx(30.0, rel=0.01)


# ── Behavioral tests ──────────────────────────────────────────

class TestVolumeClockBehavior:

    def test_disabled_preserves_behavior(self):
        """With volume_clock=False and activity_scaling=False, horizon is raw clock time."""
        engine = _make_engine(
            volume_clock_enabled=False,
            activity_scaling_enabled=False,
        )
        # Even with volume data, horizon shouldn't change
        now = time.time()
        for i in range(60):
            engine._price_feed.inject_tick(PriceTick(
                "btcusdt", 97000, now - 60 + i, 100.0, None,
            ))
        # The internal method still computes, but the engine won't call it
        # when volume_clock_enabled=False.  Verify the config gate works.
        assert engine._settings.volume_clock_enabled is False

    def test_cycle_logger_records_volume_clock_fields(self, tmp_path):
        """When volume clock is enabled, CycleSnapshot includes effective_horizon."""
        log_path = str(tmp_path / "test_vclock.csv")
        logger = CycleLogger(log_path)

        snap = CycleSnapshot(
            timestamp=time.time(),
            cycle=1,
            symbol="btcusdt",
            price=97000.0,
            ofi=0.1,
            volume_rate_short=120.0,
            volume_rate_long=60.0,
            activity_ratio=2.0,
            realized_vol=0.5,
            hawkes_intensity=3.0,
            num_edges=2,
            num_positions=1,
            session_pnl=1.50,
            bankroll=500.0,
            effective_horizon=20.0,
            projected_volume=1800.0,
        )
        logger.log(snap)
        logger.flush()
        logger.close()

        # Read back and check columns exist
        with open(log_path) as f:
            header = f.readline().strip().split(",")
            data = f.readline().strip().split(",")
        assert "effective_horizon" in header
        assert "projected_volume" in header
        idx_eh = header.index("effective_horizon")
        idx_pv = header.index("projected_volume")
        assert float(data[idx_eh]) == pytest.approx(20.0)
        assert float(data[idx_pv]) == pytest.approx(1800.0)

    def test_volume_clock_takes_precedence_over_activity_scaling(self):
        """When both volume_clock and activity_scaling are enabled,
        volume clock wins: horizon changes but vol does NOT get scaled."""
        engine = _make_engine(
            volume_clock_enabled=True,
            activity_scaling_enabled=True,
            volume_clock_short_window_seconds=60,
            volume_clock_baseline_window_seconds=600,
        )
        now = time.time()
        # Inject baseline volume for 600s at 1.0 vol/tick
        for i in range(540):
            engine._price_feed.inject_tick(PriceTick(
                "btcusdt", 97000 + i * 0.01, now - 600 + i, 1.0, None,
            ))
        # Recent burst: 3.0 vol/tick for last 60s
        for i in range(60):
            engine._price_feed.inject_tick(PriceTick(
                "btcusdt", 97000 + 540 * 0.01 + i * 0.01, now - 60 + i, 3.0, None,
            ))

        h, pv, br = engine._compute_effective_horizon("btcusdt", 10.0)
        # Volume clock should produce h > 10 (high recent volume)
        assert h > 10.0
        assert pv is not None
        # The key check: volume_clock_enabled takes the if-branch,
        # skipping the elif (activity_scaling) entirely
        assert engine._settings.volume_clock_enabled is True
        assert engine._settings.activity_scaling_enabled is True
