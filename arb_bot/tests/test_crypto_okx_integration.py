"""Tests for OKX trade and funding rate integration.

Covers:
- OKX trade message parsing in PriceFeed._handle_okx_trade_message
- OKX symbol mappings (Binance <-> OKX spot)
- FundingRateTracker OKX swap mappings and core interface
- OKX trade -> OFI integration (end-to-end)
"""

from __future__ import annotations

import time

import pytest

from arb_bot.crypto.price_feed import (
    PriceFeed,
    PriceTick,
    _BINANCE_TO_OKX_SPOT,
    _OKX_TO_BINANCE_SPOT,
)
from arb_bot.crypto.funding_rate import (
    FundingRateTracker,
    FundingSignal,
    _BINANCE_TO_OKX_SWAP,
)


# ── helpers ───────────────────────────────────────────────────────


def _okx_trade_msg(
    inst_id: str = "BTC-USDT",
    px: str = "97500.00",
    sz: str = "0.15",
    ts: str = "1700000000000",
    side: str = "buy",
) -> dict:
    """Build a minimal OKX trade WebSocket message."""
    return {
        "arg": {"channel": "trades", "instId": inst_id},
        "data": [
            {
                "instId": inst_id,
                "tradeId": "12345",
                "px": px,
                "sz": sz,
                "side": side,
                "ts": ts,
            }
        ],
    }


# ── OKX Trade Message Parsing ────────────────────────────────────


class TestHandleOkxTradeMessageBuy:
    """Test _handle_okx_trade_message with a valid buy trade."""

    def test_buy_trade_creates_tick(self):
        feed = PriceFeed(symbols=["btcusdt"])
        msg = _okx_trade_msg(
            inst_id="BTC-USDT",
            px="97500.00",
            sz="0.15",
            ts="1700000000000",
            side="buy",
        )
        feed._handle_okx_trade_message(msg)

        ticks = feed.get_price_history("btcusdt")
        assert len(ticks) == 1
        tick = ticks[0]
        assert tick.symbol == "btcusdt"
        assert tick.price == 97500.00
        assert tick.volume == 0.15
        assert tick.timestamp == 1700000000.0  # ms -> sec
        # buy side: taker bought -> is_buyer_maker=False
        assert tick.is_buyer_maker is False

    def test_buy_trade_updates_current_price(self):
        feed = PriceFeed(symbols=["btcusdt"])
        feed._handle_okx_trade_message(
            _okx_trade_msg(px="97500.00", side="buy")
        )
        assert feed.get_current_price("btcusdt") == 97500.00


class TestHandleOkxTradeMessageSell:
    """Test _handle_okx_trade_message with a valid sell trade."""

    def test_sell_trade_is_buyer_maker_true(self):
        feed = PriceFeed(symbols=["btcusdt"])
        msg = _okx_trade_msg(side="sell", px="97400.00", sz="0.25")
        feed._handle_okx_trade_message(msg)

        ticks = feed.get_price_history("btcusdt")
        assert len(ticks) == 1
        tick = ticks[0]
        assert tick.is_buyer_maker is True
        assert tick.price == 97400.00
        assert tick.volume == 0.25


class TestHandleOkxTradeMessageEdgeCases:
    """Edge cases: unknown symbol, missing fields, empty data."""

    def test_unknown_inst_id_silently_skipped(self):
        feed = PriceFeed(symbols=["btcusdt"])
        msg = _okx_trade_msg(inst_id="DOGE-USDT")
        feed._handle_okx_trade_message(msg)

        ticks = feed.get_price_history("btcusdt")
        assert len(ticks) == 0

    def test_missing_fields_no_crash(self):
        feed = PriceFeed(symbols=["btcusdt"])
        # Message with data entry missing required fields
        msg = {
            "data": [
                {"instId": "BTC-USDT"}  # missing px, sz, ts
            ]
        }
        feed._handle_okx_trade_message(msg)
        assert len(feed.get_price_history("btcusdt")) == 0

    def test_empty_data_array_no_crash(self):
        feed = PriceFeed(symbols=["btcusdt"])
        msg = {"data": []}
        feed._handle_okx_trade_message(msg)
        assert len(feed.get_price_history("btcusdt")) == 0

    def test_no_data_key_no_crash(self):
        feed = PriceFeed(symbols=["btcusdt"])
        msg = {"arg": {"channel": "trades"}}
        feed._handle_okx_trade_message(msg)
        assert len(feed.get_price_history("btcusdt")) == 0

    def test_data_none_no_crash(self):
        feed = PriceFeed(symbols=["btcusdt"])
        msg = {"data": None}
        feed._handle_okx_trade_message(msg)
        assert len(feed.get_price_history("btcusdt")) == 0


class TestOkxTradeInjectTick:
    """Verify inject_tick is called and _buy_sells deque is populated."""

    def test_buy_sells_populated_for_buy(self):
        feed = PriceFeed(symbols=["btcusdt"])
        ts = time.time() * 1000  # current time in ms
        msg = _okx_trade_msg(
            px="97500.00", sz="0.10", ts=str(int(ts)), side="buy",
        )
        feed._handle_okx_trade_message(msg)

        # _buy_sells should have one entry: (ts, vol, is_buy=True)
        dq = feed._buy_sells.get("btcusdt")
        assert dq is not None
        assert len(dq) == 1
        entry_ts, entry_vol, entry_is_buy = dq[0]
        assert entry_vol == pytest.approx(0.10)
        # side="buy" -> is_buyer_maker=False -> is_buy = not False = True
        assert entry_is_buy is True

    def test_buy_sells_populated_for_sell(self):
        feed = PriceFeed(symbols=["btcusdt"])
        ts = time.time() * 1000
        msg = _okx_trade_msg(
            px="97400.00", sz="0.20", ts=str(int(ts)), side="sell",
        )
        feed._handle_okx_trade_message(msg)

        dq = feed._buy_sells.get("btcusdt")
        assert dq is not None
        assert len(dq) == 1
        _, _, entry_is_buy = dq[0]
        # side="sell" -> is_buyer_maker=True -> is_buy = not True = False
        assert entry_is_buy is False


# ── OKX Symbol Mappings ──────────────────────────────────────────


class TestBinanceToOkxSpotMapping:
    """Test _BINANCE_TO_OKX_SPOT mapping."""

    def test_has_all_three_symbols(self):
        assert "btcusdt" in _BINANCE_TO_OKX_SPOT
        assert "ethusdt" in _BINANCE_TO_OKX_SPOT
        assert "solusdt" in _BINANCE_TO_OKX_SPOT

    def test_btcusdt_maps_correctly(self):
        assert _BINANCE_TO_OKX_SPOT["btcusdt"] == "BTC-USDT"

    def test_ethusdt_maps_correctly(self):
        assert _BINANCE_TO_OKX_SPOT["ethusdt"] == "ETH-USDT"

    def test_solusdt_maps_correctly(self):
        assert _BINANCE_TO_OKX_SPOT["solusdt"] == "SOL-USDT"


class TestOkxToBinanceSpotMapping:
    """Test _OKX_TO_BINANCE_SPOT reverse mapping."""

    def test_reverse_mapping_correct(self):
        assert _OKX_TO_BINANCE_SPOT["BTC-USDT"] == "btcusdt"
        assert _OKX_TO_BINANCE_SPOT["ETH-USDT"] == "ethusdt"
        assert _OKX_TO_BINANCE_SPOT["SOL-USDT"] == "solusdt"

    def test_round_trip_all_symbols(self):
        """binance -> okx -> binance should give original symbol."""
        for binance_sym in ["btcusdt", "ethusdt", "solusdt"]:
            okx_sym = _BINANCE_TO_OKX_SPOT[binance_sym]
            round_tripped = _OKX_TO_BINANCE_SPOT[okx_sym]
            assert round_tripped == binance_sym


# ── Funding Rate OKX Parsing ─────────────────────────────────────


class TestBinanceToOkxSwapMapping:
    """Test _BINANCE_TO_OKX_SWAP mapping."""

    def test_has_all_three_symbols(self):
        assert "BTCUSDT" in _BINANCE_TO_OKX_SWAP
        assert "ETHUSDT" in _BINANCE_TO_OKX_SWAP
        assert "SOLUSDT" in _BINANCE_TO_OKX_SWAP

    def test_btcusdt_swap(self):
        assert _BINANCE_TO_OKX_SWAP["BTCUSDT"] == "BTC-USDT-SWAP"

    def test_ethusdt_swap(self):
        assert _BINANCE_TO_OKX_SWAP["ETHUSDT"] == "ETH-USDT-SWAP"

    def test_solusdt_swap(self):
        assert _BINANCE_TO_OKX_SWAP["SOLUSDT"] == "SOL-USDT-SWAP"


class TestFundingRateTrackerDefaults:
    """Test FundingRateTracker default configuration."""

    def test_default_api_url_is_okx(self):
        tracker = FundingRateTracker()
        assert tracker._api_url == "https://www.okx.com/api/v5/public/funding-rate"


class TestFundingRateInjectAndQuery:
    """Test inject_rate, get_current_rate, get_rolling_avg."""

    def test_inject_rate_then_get_current(self):
        tracker = FundingRateTracker(symbols=["BTCUSDT"])
        tracker.inject_rate("BTCUSDT", 0.0003)
        assert tracker.get_current_rate("BTCUSDT") == pytest.approx(0.0003)

    def test_inject_rate_case_insensitive(self):
        tracker = FundingRateTracker(symbols=["BTCUSDT"])
        tracker.inject_rate("btcusdt", 0.0005)
        # inject_rate uppercases internally
        assert tracker.get_current_rate("btcusdt") == pytest.approx(0.0005)

    def test_get_current_rate_none_before_inject(self):
        tracker = FundingRateTracker(symbols=["BTCUSDT"])
        assert tracker.get_current_rate("BTCUSDT") is None

    def test_get_rolling_avg_single_sample(self):
        tracker = FundingRateTracker(symbols=["BTCUSDT"])
        now = time.time()
        tracker.inject_rate("BTCUSDT", 0.0004, timestamp=now)
        avg = tracker.get_rolling_avg("BTCUSDT", window_hours=8.0)
        assert avg == pytest.approx(0.0004)

    def test_get_rolling_avg_multiple_samples(self):
        tracker = FundingRateTracker(symbols=["BTCUSDT"])
        now = time.time()
        rates = [0.0002, 0.0004, 0.0006, 0.0008, 0.0010]
        for i, rate in enumerate(rates):
            tracker.inject_rate("BTCUSDT", rate, timestamp=now - (len(rates) - 1 - i))
        avg = tracker.get_rolling_avg("BTCUSDT", window_hours=8.0)
        expected = sum(rates) / len(rates)
        assert avg == pytest.approx(expected)

    def test_get_rolling_avg_excludes_old_samples(self):
        tracker = FundingRateTracker(symbols=["BTCUSDT"])
        now = time.time()
        # Old sample outside 8h window
        tracker.inject_rate("BTCUSDT", 0.0100, timestamp=now - 10 * 3600)
        # Recent samples
        tracker.inject_rate("BTCUSDT", 0.0002, timestamp=now - 60)
        tracker.inject_rate("BTCUSDT", 0.0004, timestamp=now)
        avg = tracker.get_rolling_avg("BTCUSDT", window_hours=8.0)
        # Should only average the two recent samples
        assert avg == pytest.approx(0.0003)


class TestFundingSignalGeneration:
    """Test get_funding_signal produces valid FundingSignal."""

    def test_returns_none_with_no_data(self):
        tracker = FundingRateTracker(symbols=["BTCUSDT"])
        sig = tracker.get_funding_signal("BTCUSDT")
        assert sig is None

    def test_produces_valid_signal(self):
        tracker = FundingRateTracker(symbols=["BTCUSDT"])
        now = time.time()
        tracker.inject_rate("BTCUSDT", 0.0003, timestamp=now)
        sig = tracker.get_funding_signal("BTCUSDT")
        assert sig is not None
        assert isinstance(sig, FundingSignal)
        assert sig.rate == pytest.approx(0.0003)
        assert isinstance(sig.rolling_avg_8h, float)
        assert isinstance(sig.rate_of_change, float)
        assert isinstance(sig.drift_adjustment, float)
        assert isinstance(sig.is_extreme_long, bool)
        assert isinstance(sig.is_extreme_short, bool)

    def test_extreme_long_detected(self):
        tracker = FundingRateTracker(symbols=["BTCUSDT"], extreme_threshold=0.0005)
        now = time.time()
        tracker.inject_rate("BTCUSDT", 0.001, timestamp=now)
        sig = tracker.get_funding_signal("BTCUSDT")
        assert sig is not None
        assert sig.is_extreme_long is True
        assert sig.is_extreme_short is False
        # Drift should be negative (expect pullback from crowded longs)
        assert sig.drift_adjustment < 0

    def test_extreme_short_detected(self):
        tracker = FundingRateTracker(symbols=["BTCUSDT"], extreme_threshold=0.0005)
        now = time.time()
        tracker.inject_rate("BTCUSDT", -0.001, timestamp=now)
        sig = tracker.get_funding_signal("BTCUSDT")
        assert sig is not None
        assert sig.is_extreme_long is False
        assert sig.is_extreme_short is True
        # Drift should be positive (expect squeeze from crowded shorts)
        assert sig.drift_adjustment > 0


# ── OKX Trade -> OFI Integration ─────────────────────────────────


class TestOkxTradeOfiIntegration:
    """End-to-end: OKX trade messages -> OFI signal."""

    def _inject_trades(self, feed: PriceFeed, n_buys: int, n_sells: int) -> None:
        """Inject a mix of buy and sell trades via _handle_okx_trade_message."""
        now = time.time()
        for i in range(n_buys):
            msg = _okx_trade_msg(
                px="97500.00",
                sz="1.0",
                ts=str(int((now - n_buys - n_sells + i) * 1000)),
                side="buy",
            )
            feed._handle_okx_trade_message(msg)
        for i in range(n_sells):
            msg = _okx_trade_msg(
                px="97500.00",
                sz="1.0",
                ts=str(int((now - n_sells + i) * 1000)),
                side="sell",
            )
            feed._handle_okx_trade_message(msg)

    def test_get_ofi_nonzero_after_okx_trades(self):
        feed = PriceFeed(symbols=["btcusdt"])
        # 10 buys + 5 sells -> net buy pressure -> positive OFI
        self._inject_trades(feed, n_buys=10, n_sells=5)
        ofi = feed.get_ofi("btcusdt", window_seconds=600)
        assert ofi != 0.0
        # More buys than sells -> positive OFI
        assert ofi > 0.0

    def test_get_ofi_negative_when_more_sells(self):
        feed = PriceFeed(symbols=["btcusdt"])
        # 3 buys + 12 sells -> net sell pressure -> negative OFI
        self._inject_trades(feed, n_buys=3, n_sells=12)
        ofi = feed.get_ofi("btcusdt", window_seconds=600)
        assert ofi < 0.0

    def test_get_ofi_multiscale_nonzero_all_windows(self):
        feed = PriceFeed(symbols=["btcusdt"])
        # Inject enough trades
        self._inject_trades(feed, n_buys=20, n_sells=10)
        result = feed.get_ofi_multiscale("btcusdt", windows=[30, 60, 120, 300])
        assert isinstance(result, dict)
        for window in [30, 60, 120, 300]:
            assert window in result
            # All windows should show non-zero OFI
            assert result[window] != 0.0
