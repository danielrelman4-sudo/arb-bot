"""Tests for crypto price feed."""

from __future__ import annotations

import json
import math
import time

import pytest

from arb_bot.crypto.price_feed import PriceFeed, PriceTick


class TestPriceTick:
    def test_frozen(self) -> None:
        t = PriceTick(symbol="btcusdt", price=97000.0, timestamp=1000.0)
        with pytest.raises(AttributeError):
            t.price = 98000.0  # type: ignore[misc]

    def test_default_volume(self) -> None:
        t = PriceTick(symbol="ethusdt", price=3200.0, timestamp=1000.0)
        assert t.volume == 0.0

    def test_price_tick_has_buyer_maker_flag(self) -> None:
        # Default is None
        t1 = PriceTick(symbol="btcusdt", price=97000.0, timestamp=1000.0)
        assert t1.is_buyer_maker is None

        # Explicitly set to True (sell pressure)
        t2 = PriceTick(symbol="btcusdt", price=97000.0, timestamp=1000.0, is_buyer_maker=True)
        assert t2.is_buyer_maker is True

        # Explicitly set to False (buy pressure)
        t3 = PriceTick(symbol="btcusdt", price=97000.0, timestamp=1000.0, is_buyer_maker=False)
        assert t3.is_buyer_maker is False


class TestPriceFeedInject:
    def test_inject_updates_current_price(self) -> None:
        feed = PriceFeed(symbols=["btcusdt"])
        assert feed.get_current_price("btcusdt") is None
        feed.inject_tick(PriceTick("btcusdt", 97000.0, time.time()))
        assert feed.get_current_price("btcusdt") == 97000.0

    def test_inject_case_insensitive(self) -> None:
        feed = PriceFeed(symbols=["btcusdt"])
        feed.inject_tick(PriceTick("BTCUSDT", 97000.0, time.time()))
        assert feed.get_current_price("btcusdt") == 97000.0
        assert feed.get_current_price("BTCUSDT") == 97000.0

    def test_inject_multiple_symbols(self) -> None:
        feed = PriceFeed(symbols=["btcusdt", "ethusdt"])
        feed.inject_tick(PriceTick("btcusdt", 97000.0, time.time()))
        feed.inject_tick(PriceTick("ethusdt", 3200.0, time.time()))
        assert feed.get_current_price("btcusdt") == 97000.0
        assert feed.get_current_price("ethusdt") == 3200.0

    def test_unknown_symbol_returns_none(self) -> None:
        feed = PriceFeed(symbols=["btcusdt"])
        assert feed.get_current_price("solusdt") is None


class TestPriceHistory:
    def _make_feed_with_ticks(self, n: int = 10) -> PriceFeed:
        feed = PriceFeed(symbols=["btcusdt"])
        now = time.time()
        for i in range(n):
            feed.inject_tick(PriceTick("btcusdt", 97000.0 + i * 10, now - (n - i) * 60))
        return feed

    def test_get_all_history(self) -> None:
        feed = self._make_feed_with_ticks(10)
        history = feed.get_price_history("btcusdt")
        assert len(history) == 10

    def test_filter_by_minutes(self) -> None:
        feed = self._make_feed_with_ticks(10)
        # Only last 3 minutes
        history = feed.get_price_history("btcusdt", minutes=3)
        assert len(history) <= 4  # ~3 ticks at 60s intervals

    def test_empty_symbol(self) -> None:
        feed = PriceFeed(symbols=["btcusdt"])
        assert feed.get_price_history("ethusdt") == []


class TestReturns:
    def test_basic_returns(self) -> None:
        feed = PriceFeed(symbols=["btcusdt"])
        now = time.time()
        # Inject 5 ticks at 60-second intervals
        prices = [100.0, 110.0, 105.0, 115.0, 120.0]
        for i, p in enumerate(prices):
            feed.inject_tick(PriceTick("btcusdt", p, now - (len(prices) - i) * 60))

        returns = feed.get_returns("btcusdt", interval_seconds=60)
        assert len(returns) == 4
        assert abs(returns[0] - math.log(110 / 100)) < 1e-10
        assert abs(returns[1] - math.log(105 / 110)) < 1e-10

    def test_too_few_ticks(self) -> None:
        feed = PriceFeed(symbols=["btcusdt"])
        feed.inject_tick(PriceTick("btcusdt", 97000.0, time.time()))
        returns = feed.get_returns("btcusdt")
        assert returns == []

    def test_no_ticks(self) -> None:
        feed = PriceFeed(symbols=["btcusdt"])
        assert feed.get_returns("btcusdt") == []

    def test_returns_with_window(self) -> None:
        feed = PriceFeed(symbols=["btcusdt"])
        now = time.time()
        # Old ticks (> 5 min ago)
        for i in range(5):
            feed.inject_tick(PriceTick("btcusdt", 90000.0 + i * 100, now - 600 + i * 60))
        # Recent ticks (< 3 min ago)
        for i in range(3):
            feed.inject_tick(PriceTick("btcusdt", 97000.0 + i * 100, now - 120 + i * 60))

        all_returns = feed.get_returns("btcusdt", interval_seconds=60)
        recent_returns = feed.get_returns("btcusdt", interval_seconds=60, window_minutes=3)
        assert len(recent_returns) <= len(all_returns)


class TestHandleMessage:
    def test_valid_trade_message(self) -> None:
        feed = PriceFeed(symbols=["btcusdt"])
        msg = json.dumps({
            "e": "trade",
            "s": "BTCUSDT",
            "p": "97500.50",
            "q": "0.001",
            "T": int(time.time() * 1000),
        })
        feed._handle_message(msg)
        assert feed.get_current_price("btcusdt") == 97500.50

    def test_non_trade_event_ignored(self) -> None:
        feed = PriceFeed(symbols=["btcusdt"])
        msg = json.dumps({"e": "aggTrade", "s": "BTCUSDT", "p": "97500.50"})
        feed._handle_message(msg)
        assert feed.get_current_price("btcusdt") is None

    def test_invalid_json_ignored(self) -> None:
        feed = PriceFeed(symbols=["btcusdt"])
        feed._handle_message("not json")
        assert feed.get_current_price("btcusdt") is None

    def test_missing_fields_ignored(self) -> None:
        feed = PriceFeed(symbols=["btcusdt"])
        msg = json.dumps({"e": "trade", "s": "BTCUSDT"})
        feed._handle_message(msg)
        assert feed.get_current_price("btcusdt") is None

    def test_handle_message_tracks_buy_sell(self) -> None:
        feed = PriceFeed(symbols=["btcusdt"])
        now_ms = int(time.time() * 1000)
        # Buy trade (is_buyer_maker=False means buyer is taker = buy pressure)
        msg_buy = json.dumps({
            "e": "trade", "s": "BTCUSDT",
            "p": "97500.00", "q": "1.0",
            "T": now_ms, "m": False,
        })
        # Sell trade (is_buyer_maker=True means seller is maker = sell pressure)
        msg_sell = json.dumps({
            "e": "trade", "s": "BTCUSDT",
            "p": "97400.00", "q": "2.0",
            "T": now_ms + 1000, "m": True,
        })
        feed._handle_message(msg_buy)
        feed._handle_message(msg_sell)
        buy_vol, sell_vol = feed.get_buy_sell_volume("btcusdt", window_seconds=60)
        assert buy_vol == pytest.approx(1.0)
        assert sell_vol == pytest.approx(2.0)


class TestBuySellVolume:
    def test_price_feed_tracks_buy_sell_volume(self) -> None:
        feed = PriceFeed(symbols=["btcusdt"])
        now = time.time()
        # Inject 3 buy ticks and 2 sell ticks
        feed.inject_tick(PriceTick("btcusdt", 97000.0, now - 10, volume=1.5, is_buyer_maker=False))
        feed.inject_tick(PriceTick("btcusdt", 97100.0, now - 8, volume=2.0, is_buyer_maker=False))
        feed.inject_tick(PriceTick("btcusdt", 97050.0, now - 6, volume=0.5, is_buyer_maker=True))
        feed.inject_tick(PriceTick("btcusdt", 97200.0, now - 4, volume=1.0, is_buyer_maker=False))
        feed.inject_tick(PriceTick("btcusdt", 96900.0, now - 2, volume=3.0, is_buyer_maker=True))

        buy_vol, sell_vol = feed.get_buy_sell_volume("btcusdt", window_seconds=60)
        # Buys: 1.5 + 2.0 + 1.0 = 4.5  (is_buyer_maker=False -> buy)
        assert buy_vol == pytest.approx(4.5)
        # Sells: 0.5 + 3.0 = 3.5  (is_buyer_maker=True -> sell)
        assert sell_vol == pytest.approx(3.5)

    def test_inject_tick_with_buyer_maker_tracks_buy_sell(self) -> None:
        feed = PriceFeed(symbols=["btcusdt"])
        now = time.time()
        # Tick without is_buyer_maker should NOT appear in buy/sell tracking
        feed.inject_tick(PriceTick("btcusdt", 97000.0, now - 5, volume=10.0))
        buy_vol, sell_vol = feed.get_buy_sell_volume("btcusdt", window_seconds=60)
        assert buy_vol == 0.0
        assert sell_vol == 0.0

        # Tick with is_buyer_maker should appear
        feed.inject_tick(PriceTick("btcusdt", 97100.0, now - 3, volume=2.0, is_buyer_maker=False))
        buy_vol, sell_vol = feed.get_buy_sell_volume("btcusdt", window_seconds=60)
        assert buy_vol == pytest.approx(2.0)
        assert sell_vol == 0.0

    def test_buy_sell_volume_no_data_returns_zeros(self) -> None:
        feed = PriceFeed(symbols=["btcusdt"])
        buy_vol, sell_vol = feed.get_buy_sell_volume("btcusdt", window_seconds=60)
        assert buy_vol == 0.0
        assert sell_vol == 0.0

    def test_buy_sell_volume_unknown_symbol_returns_zeros(self) -> None:
        feed = PriceFeed(symbols=["btcusdt"])
        buy_vol, sell_vol = feed.get_buy_sell_volume("solusdt", window_seconds=60)
        assert buy_vol == 0.0
        assert sell_vol == 0.0

    def test_buy_sell_volume_respects_window(self) -> None:
        feed = PriceFeed(symbols=["btcusdt"])
        now = time.time()
        # Old tick outside window
        feed.inject_tick(PriceTick("btcusdt", 97000.0, now - 400, volume=5.0, is_buyer_maker=False))
        # Recent tick inside window
        feed.inject_tick(PriceTick("btcusdt", 97100.0, now - 10, volume=1.0, is_buyer_maker=False))

        buy_vol, sell_vol = feed.get_buy_sell_volume("btcusdt", window_seconds=300)
        # Only the recent tick should count
        assert buy_vol == pytest.approx(1.0)
        assert sell_vol == 0.0


class TestOFI:
    def test_ofi_computation(self) -> None:
        """3 buy ticks (vol 1.0 each) + 1 sell tick (vol 1.0) => OFI = (3-1)/(3+1) = 0.5."""
        feed = PriceFeed(symbols=["btcusdt"])
        now = time.time()
        # 3 buy ticks
        feed.inject_tick(PriceTick("btcusdt", 97000.0, now - 10, volume=1.0, is_buyer_maker=False))
        feed.inject_tick(PriceTick("btcusdt", 97100.0, now - 8, volume=1.0, is_buyer_maker=False))
        feed.inject_tick(PriceTick("btcusdt", 97200.0, now - 6, volume=1.0, is_buyer_maker=False))
        # 1 sell tick
        feed.inject_tick(PriceTick("btcusdt", 96900.0, now - 4, volume=1.0, is_buyer_maker=True))

        ofi = feed.get_ofi("btcusdt", window_seconds=60)
        assert ofi == pytest.approx(0.5)

    def test_ofi_no_data_returns_zero(self) -> None:
        feed = PriceFeed(symbols=["btcusdt"])
        assert feed.get_ofi("btcusdt") == 0.0

    def test_ofi_all_buys_returns_one(self) -> None:
        feed = PriceFeed(symbols=["btcusdt"])
        now = time.time()
        feed.inject_tick(PriceTick("btcusdt", 97000.0, now - 5, volume=2.0, is_buyer_maker=False))
        feed.inject_tick(PriceTick("btcusdt", 97100.0, now - 3, volume=3.0, is_buyer_maker=False))
        assert feed.get_ofi("btcusdt", window_seconds=60) == pytest.approx(1.0)

    def test_ofi_all_sells_returns_negative_one(self) -> None:
        feed = PriceFeed(symbols=["btcusdt"])
        now = time.time()
        feed.inject_tick(PriceTick("btcusdt", 97000.0, now - 5, volume=2.0, is_buyer_maker=True))
        feed.inject_tick(PriceTick("btcusdt", 96900.0, now - 3, volume=3.0, is_buyer_maker=True))
        assert feed.get_ofi("btcusdt", window_seconds=60) == pytest.approx(-1.0)
