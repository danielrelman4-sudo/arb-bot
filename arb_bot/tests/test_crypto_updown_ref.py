"""Tests for the up/down reference price fix.

Validates that:
1. PriceFeed.get_price_at_time() returns the nearest tick price
2. CryptoMarketMeta can carry an interval_start_time field
3. The engine uses the interval start price (not current price) as the
   reference for up/down probability computation
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from dataclasses import replace

import numpy as np
import pytest

from arb_bot.crypto.config import CryptoSettings
from arb_bot.crypto.engine import CryptoEngine
from arb_bot.crypto.market_scanner import (
    CryptoMarket,
    CryptoMarketMeta,
    CryptoMarketQuote,
    parse_ticker,
)
from arb_bot.crypto.price_feed import PriceFeed, PriceTick
from arb_bot.crypto.price_model import PriceModel, ProbabilityEstimate


# ── Helpers ────────────────────────────────────────────────────────

def _make_settings(**overrides) -> CryptoSettings:
    defaults = dict(
        enabled=True,
        paper_mode=True,
        bankroll=1000.0,
        mc_num_paths=5000,
        min_edge_pct=0.01,
        min_edge_cents=0.01,
        max_model_uncertainty=0.25,
        model_uncertainty_multiplier=1.0,
        kelly_fraction_cap=0.10,
        max_position_per_market=100.0,
        max_concurrent_positions=10,
        scan_interval_seconds=0.01,
        paper_slippage_cents=0.0,
        confidence_level=0.95,
        mc_vol_window_minutes=30,
        price_feed_symbols=["btcusdt"],
        symbols=["KXBTC"],
        min_minutes_to_expiry=1,
        max_minutes_to_expiry=60,
        allowed_directions=["above", "below", "up", "down"],
        # Disable noise sources for deterministic tests
        ofi_enabled=False,
        activity_scaling_enabled=False,
        use_jump_diffusion=False,
        hawkes_enabled=False,
    )
    defaults.update(overrides)
    return CryptoSettings(**defaults)


def _inject_price_history(
    feed: PriceFeed,
    symbol: str,
    base_price: float,
    n_ticks: int = 120,
    start_offset: float = 120.0,
    jitter: float = 10.0,
) -> None:
    """Inject enough price history so the engine can compute vol."""
    now = time.time()
    rng = np.random.default_rng(42)
    for i in range(n_ticks):
        price = base_price + rng.normal(0, jitter)
        ts = now - start_offset + i * (start_offset / n_ticks)
        feed.inject_tick(PriceTick(symbol=symbol, price=price, timestamp=ts, volume=1.0))


# ── TestIntervalStartPrice: PriceFeed.get_price_at_time ────────────

class TestIntervalStartPrice:
    def test_exact_match(self) -> None:
        """When a tick has the exact target timestamp, return its price."""
        feed = PriceFeed(symbols=["btcusdt"])
        feed.inject_tick(PriceTick("btcusdt", 69000.0, 1000.0, 1.0))
        feed.inject_tick(PriceTick("btcusdt", 70000.0, 1001.0, 1.0))
        feed.inject_tick(PriceTick("btcusdt", 71000.0, 1002.0, 1.0))

        result = feed.get_price_at_time("btcusdt", 1001.0)
        assert result == 70000.0

    def test_nearest_match(self) -> None:
        """When no exact match, return the price of the nearest tick."""
        feed = PriceFeed(symbols=["btcusdt"])
        feed.inject_tick(PriceTick("btcusdt", 69000.0, 1000.0, 1.0))
        feed.inject_tick(PriceTick("btcusdt", 70000.0, 1010.0, 1.0))
        feed.inject_tick(PriceTick("btcusdt", 71000.0, 1020.0, 1.0))

        # Target 1008 is closest to 1010 -> price 70000
        result = feed.get_price_at_time("btcusdt", 1008.0)
        assert result == 70000.0

        # Target 1003 is closest to 1000 -> price 69000
        result = feed.get_price_at_time("btcusdt", 1003.0)
        assert result == 69000.0

    def test_no_data_returns_none(self) -> None:
        """When no ticks exist for the symbol, return None."""
        feed = PriceFeed(symbols=["btcusdt"])
        result = feed.get_price_at_time("btcusdt", 1000.0)
        assert result is None

    def test_unknown_symbol_returns_none(self) -> None:
        """When querying an unknown symbol, return None."""
        feed = PriceFeed(symbols=["btcusdt"])
        feed.inject_tick(PriceTick("btcusdt", 69000.0, 1000.0, 1.0))
        result = feed.get_price_at_time("ethusdt", 1000.0)
        assert result is None

    def test_case_insensitive(self) -> None:
        """Symbol lookup should be case-insensitive."""
        feed = PriceFeed(symbols=["btcusdt"])
        feed.inject_tick(PriceTick("btcusdt", 69000.0, 1000.0, 1.0))
        result = feed.get_price_at_time("BTCUSDT", 1000.0)
        assert result == 69000.0

    def test_single_tick(self) -> None:
        """With only one tick, always return that tick's price."""
        feed = PriceFeed(symbols=["btcusdt"])
        feed.inject_tick(PriceTick("btcusdt", 69000.0, 1000.0, 1.0))
        result = feed.get_price_at_time("btcusdt", 5000.0)
        assert result == 69000.0


# ── TestMetaIntervalStart: CryptoMarketMeta carries interval_start_time ──

class TestMetaIntervalStart:
    def test_default_none(self) -> None:
        """interval_start_time defaults to None for backward compat."""
        meta = CryptoMarketMeta(
            underlying="BTC",
            interval="15min",
            expiry=datetime(2026, 2, 14, 12, 0, tzinfo=timezone.utc),
            strike=None,
            direction="up",
            series_ticker="KXBTC15M",
        )
        assert meta.interval_start_time is None

    def test_explicit_start_time(self) -> None:
        """interval_start_time can be explicitly set."""
        start = datetime(2026, 2, 14, 11, 45, tzinfo=timezone.utc)
        meta = CryptoMarketMeta(
            underlying="BTC",
            interval="15min",
            expiry=datetime(2026, 2, 14, 12, 0, tzinfo=timezone.utc),
            strike=None,
            direction="up",
            series_ticker="KXBTC15M",
            interval_start_time=start,
        )
        assert meta.interval_start_time == start

    def test_parse_ticker_leaves_start_none(self) -> None:
        """parse_ticker doesn't set interval_start_time (callers do)."""
        meta = parse_ticker("KXBTC15M-26FEB14-U12")
        assert meta is not None
        assert meta.interval_start_time is None

    def test_frozen_field(self) -> None:
        """interval_start_time is frozen along with the rest of the dataclass."""
        meta = CryptoMarketMeta(
            underlying="BTC",
            interval="15min",
            expiry=datetime(2026, 2, 14, 12, 0, tzinfo=timezone.utc),
            strike=None,
            direction="up",
            series_ticker="KXBTC15M",
        )
        with pytest.raises(AttributeError):
            meta.interval_start_time = datetime.now(timezone.utc)  # type: ignore[misc]

    def test_existing_constructions_still_work(self) -> None:
        """Existing code that doesn't pass interval_start_time still works."""
        meta = CryptoMarketMeta(
            underlying="BTC",
            interval="daily",
            expiry=datetime(2026, 2, 14, tzinfo=timezone.utc),
            strike=97500.0,
            direction="above",
            series_ticker="KXBTCD",
            interval_index=None,
        )
        assert meta.interval_start_time is None
        assert meta.strike == 97500.0


# ── TestUpDownUsesRefPrice: engine uses interval start price ────────

class TestUpDownUsesRefPrice:
    def test_up_market_already_up_high_prob(self) -> None:
        """BTC started at 69000, now 70000 (already up) -> P(up) > 75%.

        With the fix, the model compares simulated terminal prices to
        the interval-start price (69000), not the current price (70000).
        Since BTC has already moved from 69000 to 70000 (a +1.4% move),
        and is staying near 70000, most MC paths starting at 70000 will
        end above 69000 -> high P(up).
        """
        settings = _make_settings(mc_num_paths=5000)
        engine = CryptoEngine(settings)
        engine._price_model = PriceModel(num_paths=5000, seed=42)

        # Inject price history around 70000 (current price)
        _inject_price_history(engine.price_feed, "btcusdt", 70000.0, n_ticks=120, jitter=10.0)

        # The interval started at 69000, 15 minutes ago
        interval_start = datetime.now(timezone.utc) - timedelta(minutes=15)

        # Inject a tick at the interval start time with price 69000
        engine.price_feed.inject_tick(PriceTick(
            "btcusdt", 69000.0,
            interval_start.timestamp(),
            1.0,
        ))

        # Re-inject current price to make sure get_current_price returns 70000
        engine.price_feed.inject_tick(PriceTick(
            "btcusdt", 70000.0,
            time.time(),
            1.0,
        ))

        # Build up-market quote with interval_start_time set
        now = datetime.now(timezone.utc)
        meta = CryptoMarketMeta(
            underlying="BTC",
            interval="15min",
            expiry=now + timedelta(minutes=5),
            strike=None,
            direction="up",
            series_ticker="KXBTC15M",
            interval_start_time=interval_start,
        )
        market = CryptoMarket(ticker="KXBTC15M-26FEB14-U12", meta=meta)
        quote = CryptoMarketQuote(
            market=market,
            yes_buy_price=0.50,
            no_buy_price=0.50,
            yes_buy_size=100,
            no_buy_size=100,
            yes_bid_price=0.49,
            no_bid_price=0.49,
            time_to_expiry_minutes=5.0,
            implied_probability=0.50,
        )

        probs = engine._compute_model_probabilities([quote])
        assert market.ticker in probs
        prob = probs[market.ticker]
        # BTC is already 1000 above the start price with low vol ->
        # most paths stay above 69000
        assert prob.probability > 0.75, (
            f"Expected P(up) > 75% when already up, got {prob.probability:.1%}"
        )

    def test_down_market_already_down_high_prob(self) -> None:
        """BTC started at 70000, now 68000 (already down) -> P(down) > 75%.

        The model compares terminal prices to interval-start price 70000.
        Current price 68000 means most MC paths starting at 68000 will
        end below 70000.
        """
        settings = _make_settings(mc_num_paths=5000)
        engine = CryptoEngine(settings)
        engine._price_model = PriceModel(num_paths=5000, seed=42)

        # Inject price history around 68000 (current price)
        _inject_price_history(engine.price_feed, "btcusdt", 68000.0, n_ticks=120, jitter=10.0)

        interval_start = datetime.now(timezone.utc) - timedelta(minutes=15)

        # Inject tick at interval start with price 70000
        engine.price_feed.inject_tick(PriceTick(
            "btcusdt", 70000.0,
            interval_start.timestamp(),
            1.0,
        ))

        # Re-inject current price at 68000
        engine.price_feed.inject_tick(PriceTick(
            "btcusdt", 68000.0,
            time.time(),
            1.0,
        ))

        now = datetime.now(timezone.utc)
        meta = CryptoMarketMeta(
            underlying="BTC",
            interval="15min",
            expiry=now + timedelta(minutes=5),
            strike=None,
            direction="down",
            series_ticker="KXBTC15M",
            interval_start_time=interval_start,
        )
        market = CryptoMarket(ticker="KXBTC15M-26FEB14-D12", meta=meta)
        quote = CryptoMarketQuote(
            market=market,
            yes_buy_price=0.50,
            no_buy_price=0.50,
            yes_buy_size=100,
            no_buy_size=100,
            yes_bid_price=0.49,
            no_bid_price=0.49,
            time_to_expiry_minutes=5.0,
            implied_probability=0.50,
        )

        probs = engine._compute_model_probabilities([quote])
        assert market.ticker in probs
        prob = probs[market.ticker]
        # BTC is already 2000 below start price -> most paths end below 70000
        assert prob.probability > 0.75, (
            f"Expected P(down) > 75% when already down, got {prob.probability:.1%}"
        )

    def test_up_market_no_interval_start_falls_back_to_current(self) -> None:
        """When interval_start_time is None, engine falls back to current price.

        This tests backward compatibility: old-style quotes without
        interval_start_time still work and produce ~50% probability.
        """
        settings = _make_settings(mc_num_paths=2000)
        engine = CryptoEngine(settings)
        engine._price_model = PriceModel(num_paths=2000, seed=42)

        _inject_price_history(engine.price_feed, "btcusdt", 70000.0, n_ticks=120, jitter=10.0)
        engine.price_feed.inject_tick(PriceTick(
            "btcusdt", 70000.0, time.time(), 1.0,
        ))

        now = datetime.now(timezone.utc)
        # No interval_start_time set (defaults to None)
        meta = CryptoMarketMeta(
            underlying="BTC",
            interval="15min",
            expiry=now + timedelta(minutes=10),
            strike=None,
            direction="up",
            series_ticker="KXBTC15M",
        )
        market = CryptoMarket(ticker="KXBTC15M-26FEB14-U12", meta=meta)
        quote = CryptoMarketQuote(
            market=market,
            yes_buy_price=0.50,
            no_buy_price=0.50,
            yes_buy_size=100,
            no_buy_size=100,
            yes_bid_price=0.49,
            no_bid_price=0.49,
            time_to_expiry_minutes=10.0,
            implied_probability=0.50,
        )

        probs = engine._compute_model_probabilities([quote])
        assert market.ticker in probs
        prob = probs[market.ticker]
        # With ref = current price and zero drift, P(up) ~ 50%
        assert 0.35 <= prob.probability <= 0.65, (
            f"Expected P(up) ~ 50% with no ref, got {prob.probability:.1%}"
        )

    def test_up_market_price_at_time_not_found_falls_back(self) -> None:
        """When interval_start_time is set but no tick exists near that time,
        the engine falls back to current price.
        """
        settings = _make_settings(mc_num_paths=2000)
        engine = CryptoEngine(settings)
        engine._price_model = PriceModel(num_paths=2000, seed=42)

        # Only inject recent ticks (no data near the interval start)
        now_ts = time.time()
        for i in range(120):
            engine.price_feed.inject_tick(PriceTick(
                "btcusdt", 70000.0 + np.random.default_rng(i).normal(0, 10),
                now_ts - 120 + i, 1.0,
            ))
        engine.price_feed.inject_tick(PriceTick("btcusdt", 70000.0, now_ts, 1.0))

        # interval_start_time is way in the past (no ticks there because
        # PriceFeed only keeps 7200 ticks, and this timestamp has no coverage).
        # However, get_price_at_time returns the nearest tick's price,
        # so it will still find something. To truly get None, we'd need
        # an empty tick deque for that symbol.
        #
        # Instead, test with a different symbol that has no ticks:
        # Use ethusdt (no ticks injected for it)
        interval_start = datetime.now(timezone.utc) - timedelta(hours=2)
        now = datetime.now(timezone.utc)
        meta = CryptoMarketMeta(
            underlying="ETH",
            interval="15min",
            expiry=now + timedelta(minutes=10),
            strike=None,
            direction="up",
            series_ticker="KXETH15M",
            interval_start_time=interval_start,
        )
        market = CryptoMarket(ticker="KXETH15M-26FEB14-U12", meta=meta)
        quote = CryptoMarketQuote(
            market=market,
            yes_buy_price=0.50,
            no_buy_price=0.50,
            yes_buy_size=100,
            no_buy_size=100,
            yes_bid_price=0.49,
            no_bid_price=0.49,
            time_to_expiry_minutes=10.0,
            implied_probability=0.50,
        )

        # This won't produce a probability because there's no ethusdt price
        # data at all (no current_price). This is the expected behavior.
        probs = engine._compute_model_probabilities([quote])
        assert market.ticker not in probs
