"""Tests for crypto market scanner and ticker parsing."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from arb_bot.crypto.market_scanner import (
    CryptoMarket,
    CryptoMarketMeta,
    MarketScanner,
    parse_ticker,
)


class TestParseTicker:
    def test_daily_above(self) -> None:
        meta = parse_ticker("KXBTCD-26FEB14-T97500")
        assert meta is not None
        assert meta.underlying == "BTC"
        assert meta.interval == "daily"
        assert meta.direction == "above"
        assert meta.strike == 97500.0
        assert meta.interval_index is None
        assert meta.expiry.year == 2026
        assert meta.expiry.month == 2
        assert meta.expiry.day == 14

    def test_daily_below(self) -> None:
        meta = parse_ticker("KXBTCD-26FEB14-B97500")
        assert meta is not None
        assert meta.direction == "below"
        assert meta.strike == 97500.0

    def test_15min_up(self) -> None:
        meta = parse_ticker("KXBTC15M-26FEB14-U12")
        assert meta is not None
        assert meta.underlying == "BTC"
        assert meta.interval == "15min"
        assert meta.direction == "up"
        assert meta.strike is None
        assert meta.interval_index == 12

    def test_15min_down(self) -> None:
        meta = parse_ticker("KXBTC15M-26FEB14-D12")
        assert meta is not None
        assert meta.direction == "down"
        assert meta.interval_index == 12

    def test_1hour_up(self) -> None:
        meta = parse_ticker("KXBTC1H-26FEB14-U5")
        assert meta is not None
        assert meta.interval == "1hour"
        assert meta.direction == "up"
        assert meta.interval_index == 5

    def test_eth_daily(self) -> None:
        meta = parse_ticker("KXETHD-26FEB14-T3200")
        assert meta is not None
        assert meta.underlying == "ETH"
        assert meta.interval == "daily"
        assert meta.strike == 3200.0

    def test_eth_15min(self) -> None:
        meta = parse_ticker("KXETH15M-26FEB14-U8")
        assert meta is not None
        assert meta.underlying == "ETH"
        assert meta.interval == "15min"

    def test_sol_daily(self) -> None:
        meta = parse_ticker("KXSOLD-26FEB14-T150")
        assert meta is not None
        assert meta.underlying == "SOL"

    def test_invalid_ticker(self) -> None:
        assert parse_ticker("INVALID-TICKER") is None
        assert parse_ticker("") is None
        assert parse_ticker("KXBTCD-BADDATE-T100") is None

    def test_case_insensitive(self) -> None:
        meta = parse_ticker("kxbtcd-26feb14-t97500")
        assert meta is not None
        assert meta.underlying == "BTC"

    def test_series_ticker_preserved(self) -> None:
        meta = parse_ticker("KXBTC15M-26FEB14-U12")
        assert meta is not None
        assert meta.series_ticker == "KXBTC15M"


class TestMarketScanner:
    def _make_market(
        self,
        ticker: str = "KXBTCD-26FEB14-T97500",
        expiry_offset_minutes: float = 10.0,
    ) -> CryptoMarket:
        meta = parse_ticker(ticker)
        assert meta is not None
        # Override expiry to be relative to now
        from dataclasses import replace
        now = datetime.now(timezone.utc)
        adjusted = replace(meta, expiry=now + timedelta(minutes=expiry_offset_minutes))
        return CryptoMarket(ticker=ticker, meta=adjusted)

    def test_filter_by_symbol(self) -> None:
        scanner = MarketScanner(symbols=["KXBTC"])
        btc = self._make_market("KXBTCD-26FEB14-T97500")
        eth = self._make_market("KXETHD-26FEB14-T3200")
        result = scanner.filter_markets([btc, eth])
        assert len(result) == 1
        assert result[0].ticker.startswith("KXBTCD")

    def test_filter_by_time_to_expiry_too_soon(self) -> None:
        scanner = MarketScanner(min_minutes_to_expiry=2)
        mkt = self._make_market(expiry_offset_minutes=1.0)
        assert len(scanner.filter_markets([mkt])) == 0

    def test_filter_by_time_to_expiry_too_far(self) -> None:
        scanner = MarketScanner(max_minutes_to_expiry=14)
        mkt = self._make_market(expiry_offset_minutes=20.0)
        assert len(scanner.filter_markets([mkt])) == 0

    def test_filter_passes_in_range(self) -> None:
        scanner = MarketScanner(min_minutes_to_expiry=2, max_minutes_to_expiry=14)
        mkt = self._make_market(expiry_offset_minutes=10.0)
        assert len(scanner.filter_markets([mkt])) == 1

    def test_parse_markets_from_tickers(self) -> None:
        scanner = MarketScanner()
        tickers = [
            "KXBTCD-26FEB14-T97500",
            "KXETH15M-26FEB14-U12",
            "INVALID-TICKER",
        ]
        markets = scanner.parse_markets_from_tickers(tickers)
        assert len(markets) == 2

    def test_compute_time_to_expiry(self) -> None:
        scanner = MarketScanner()
        mkt = self._make_market(expiry_offset_minutes=10.0)
        tte = scanner.compute_time_to_expiry(mkt)
        assert 9.0 < tte < 11.0  # Approximate due to test execution time
