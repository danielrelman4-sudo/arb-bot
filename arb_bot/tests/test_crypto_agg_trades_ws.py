"""Tests for the aggTrade WebSocket streaming feature (Task 4).

Verifies:
- _handle_agg_trade_message() correctly parses buy and sell trades
- OFI computed from streamed data matches expected values
- Unknown symbols are silently ignored
- Malformed messages do not crash the handler
- CryptoSettings.agg_trades_ws_enabled field exists and defaults to True
"""
from __future__ import annotations

import os
import time
from unittest import mock

import pytest

from arb_bot.crypto.config import CryptoSettings, load_crypto_settings
from arb_bot.crypto.price_feed import PriceFeed, PriceTick


# ── Helpers ────────────────────────────────────────────────────────

def _make_feed(symbols: list[str] | None = None) -> PriceFeed:
    """Create a PriceFeed for testing with given symbols."""
    return PriceFeed(symbols=symbols or ["btcusdt"])


def _agg_trade_msg(
    symbol: str,
    price: float,
    qty: float,
    is_buyer_maker: bool,
    ts_ms: int | None = None,
) -> dict:
    """Build a Binance aggTrade WebSocket message dict."""
    if ts_ms is None:
        ts_ms = int(time.time() * 1000)
    return {
        "e": "aggTrade",
        "E": ts_ms,
        "s": symbol.upper(),
        "a": 123456,
        "p": str(price),
        "q": str(qty),
        "f": 100,
        "l": 105,
        "T": ts_ms,
        "m": is_buyer_maker,
        "M": True,
    }


# ── Buy/sell parsing tests ────────────────────────────────────────

class TestHandleAggTradeMessage:
    def test_parses_buy_trade(self) -> None:
        """msg with m=False (buyer is taker) should increase buy volume."""
        feed = _make_feed(["btcusdt"])
        ts_ms = int(time.time() * 1000)

        msg = _agg_trade_msg("BTCUSDT", 97500.0, 0.5, is_buyer_maker=False, ts_ms=ts_ms)
        feed._handle_agg_trade_message(msg)

        buy_vol, sell_vol = feed.get_buy_sell_volume("btcusdt", window_seconds=60)
        assert buy_vol == pytest.approx(0.5)
        assert sell_vol == pytest.approx(0.0)

    def test_parses_sell_trade(self) -> None:
        """msg with m=True (buyer is maker = sell pressure) should increase sell volume."""
        feed = _make_feed(["btcusdt"])
        ts_ms = int(time.time() * 1000)

        msg = _agg_trade_msg("BTCUSDT", 97500.0, 0.3, is_buyer_maker=True, ts_ms=ts_ms)
        feed._handle_agg_trade_message(msg)

        buy_vol, sell_vol = feed.get_buy_sell_volume("btcusdt", window_seconds=60)
        assert buy_vol == pytest.approx(0.0)
        assert sell_vol == pytest.approx(0.3)

    def test_updates_current_price(self) -> None:
        """After handling a message, current price should be updated."""
        feed = _make_feed(["btcusdt"])
        ts_ms = int(time.time() * 1000)

        msg = _agg_trade_msg("BTCUSDT", 98000.0, 0.1, is_buyer_maker=False, ts_ms=ts_ms)
        feed._handle_agg_trade_message(msg)

        assert feed.get_current_price("btcusdt") == pytest.approx(98000.0)

    def test_appends_to_tick_history(self) -> None:
        """Handler should append to tick history."""
        feed = _make_feed(["btcusdt"])
        ts_ms = int(time.time() * 1000)

        msg = _agg_trade_msg("BTCUSDT", 97000.0, 0.2, is_buyer_maker=True, ts_ms=ts_ms)
        feed._handle_agg_trade_message(msg)

        ticks = feed.get_price_history("btcusdt")
        assert len(ticks) == 1
        assert ticks[0].price == pytest.approx(97000.0)
        assert ticks[0].volume == pytest.approx(0.2)
        assert ticks[0].is_buyer_maker is True


# ── OFI from stream data ─────────────────────────────────────────

class TestOFIFromStream:
    def test_ofi_from_stream_data(self) -> None:
        """7 buys + 3 sells -> OFI = (7-3)/10 = 0.4."""
        feed = _make_feed(["btcusdt"])
        ts_ms = int(time.time() * 1000)

        # 7 buy trades (is_buyer_maker=False -> buyer is taker -> buy pressure)
        for i in range(7):
            msg = _agg_trade_msg("BTCUSDT", 97500.0, 1.0, is_buyer_maker=False, ts_ms=ts_ms + i)
            feed._handle_agg_trade_message(msg)

        # 3 sell trades (is_buyer_maker=True -> buyer is maker -> sell pressure)
        for i in range(3):
            msg = _agg_trade_msg("BTCUSDT", 97500.0, 1.0, is_buyer_maker=True, ts_ms=ts_ms + 7 + i)
            feed._handle_agg_trade_message(msg)

        ofi = feed.get_ofi("btcusdt", window_seconds=60)
        assert ofi == pytest.approx(0.4, abs=1e-6)

    def test_all_buys_ofi_one(self) -> None:
        """All buys should give OFI = +1.0."""
        feed = _make_feed(["btcusdt"])
        ts_ms = int(time.time() * 1000)

        for i in range(5):
            msg = _agg_trade_msg("BTCUSDT", 97500.0, 1.0, is_buyer_maker=False, ts_ms=ts_ms + i)
            feed._handle_agg_trade_message(msg)

        ofi = feed.get_ofi("btcusdt", window_seconds=60)
        assert ofi == pytest.approx(1.0)

    def test_all_sells_ofi_neg_one(self) -> None:
        """All sells should give OFI = -1.0."""
        feed = _make_feed(["btcusdt"])
        ts_ms = int(time.time() * 1000)

        for i in range(5):
            msg = _agg_trade_msg("BTCUSDT", 97500.0, 1.0, is_buyer_maker=True, ts_ms=ts_ms + i)
            feed._handle_agg_trade_message(msg)

        ofi = feed.get_ofi("btcusdt", window_seconds=60)
        assert ofi == pytest.approx(-1.0)


# ── Filtering / robustness ────────────────────────────────────────

class TestFilteringAndRobustness:
    def test_unknown_symbol_ignored(self) -> None:
        """XRPUSDT message when only btcusdt subscribed should be silently ignored."""
        feed = _make_feed(["btcusdt"])
        ts_ms = int(time.time() * 1000)

        msg = _agg_trade_msg("XRPUSDT", 0.50, 100.0, is_buyer_maker=False, ts_ms=ts_ms)
        feed._handle_agg_trade_message(msg)

        # btcusdt should have no data
        assert feed.get_current_price("btcusdt") is None
        # xrpusdt should also have no data (not subscribed)
        assert feed.get_current_price("xrpusdt") is None

    def test_malformed_message_empty_dict(self) -> None:
        """Empty dict should not crash."""
        feed = _make_feed(["btcusdt"])
        feed._handle_agg_trade_message({})
        # No error, no data
        assert feed.get_current_price("btcusdt") is None

    def test_malformed_message_partial_dict(self) -> None:
        """Dict with some fields missing should not crash."""
        feed = _make_feed(["btcusdt"])
        feed._handle_agg_trade_message({"s": "BTCUSDT", "p": "97500.0"})
        # Missing 'q' and 'T' fields -> should be silently dropped
        assert feed.get_current_price("btcusdt") is None

    def test_malformed_message_bad_types(self) -> None:
        """Non-numeric price/qty should not crash."""
        feed = _make_feed(["btcusdt"])
        msg = {
            "s": "BTCUSDT",
            "p": "not_a_number",
            "q": "1.0",
            "T": int(time.time() * 1000),
            "m": False,
        }
        feed._handle_agg_trade_message(msg)
        assert feed.get_current_price("btcusdt") is None

    def test_multi_symbol_routing(self) -> None:
        """Messages for multiple subscribed symbols are routed correctly."""
        feed = _make_feed(["btcusdt", "ethusdt"])
        ts_ms = int(time.time() * 1000)

        msg_btc = _agg_trade_msg("BTCUSDT", 97500.0, 1.0, is_buyer_maker=False, ts_ms=ts_ms)
        msg_eth = _agg_trade_msg("ETHUSDT", 3500.0, 2.0, is_buyer_maker=True, ts_ms=ts_ms)
        feed._handle_agg_trade_message(msg_btc)
        feed._handle_agg_trade_message(msg_eth)

        assert feed.get_current_price("btcusdt") == pytest.approx(97500.0)
        assert feed.get_current_price("ethusdt") == pytest.approx(3500.0)


# ── Symbols set ───────────────────────────────────────────────────

class TestSymbolsSet:
    def test_symbols_set_matches_symbols(self) -> None:
        """_symbols_set should contain lowercase versions of all symbols."""
        feed = _make_feed(["BTCUSDT", "ETHUSDT"])
        assert feed._symbols_set == {"btcusdt", "ethusdt"}

    def test_symbols_set_is_set_type(self) -> None:
        """_symbols_set should be a set for O(1) lookups."""
        feed = _make_feed(["btcusdt"])
        assert isinstance(feed._symbols_set, set)


# ── Config tests ──────────────────────────────────────────────────

class TestAggTradesWSConfig:
    def test_has_agg_trades_ws_field(self) -> None:
        """CryptoSettings should have agg_trades_ws_enabled field defaulting to True."""
        s = CryptoSettings()
        assert s.agg_trades_ws_enabled is True

    def test_can_disable_agg_trades_ws(self) -> None:
        """agg_trades_ws_enabled can be set to False."""
        s = CryptoSettings(agg_trades_ws_enabled=False)
        assert s.agg_trades_ws_enabled is False

    def test_agg_trades_ws_from_env_true(self) -> None:
        """agg_trades_ws_enabled can be loaded from environment variable."""
        env = {"ARB_CRYPTO_AGG_TRADES_WS_ENABLED": "true"}
        with mock.patch.dict(os.environ, env, clear=True):
            s = load_crypto_settings()
        assert s.agg_trades_ws_enabled is True

    def test_agg_trades_ws_from_env_false(self) -> None:
        """agg_trades_ws_enabled=false from env disables the feature."""
        env = {"ARB_CRYPTO_AGG_TRADES_WS_ENABLED": "false"}
        with mock.patch.dict(os.environ, env, clear=True):
            s = load_crypto_settings()
        assert s.agg_trades_ws_enabled is False

    def test_agg_trades_ws_default_from_env(self) -> None:
        """Without env var, load_crypto_settings defaults to True."""
        with mock.patch.dict(os.environ, {}, clear=True):
            s = load_crypto_settings()
        assert s.agg_trades_ws_enabled is True
