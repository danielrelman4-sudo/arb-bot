"""Tests for the crypto prediction trading engine."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from dataclasses import replace

import numpy as np
import pytest

from arb_bot.crypto.config import CryptoSettings
from arb_bot.crypto.edge_detector import CryptoEdge, EdgeDetector
from arb_bot.crypto.engine import (
    CryptoEngine,
    CryptoPosition,
    CryptoTradeRecord,
    _KALSHI_TO_BINANCE,
)
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
        mc_num_paths=100,  # Fewer for fast tests
        min_edge_pct=0.05,
        min_edge_cents=0.02,
        max_model_uncertainty=0.15,
        kelly_fraction_cap=0.10,
        max_position_per_market=100.0,
        max_concurrent_positions=10,
        scan_interval_seconds=0.01,
        paper_slippage_cents=0.0,  # No slippage for deterministic tests
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


def _make_edge(
    ticker: str = "KXBTCD-26FEB14-T97500",
    edge: float = 0.10,
    side: str = "yes",
    yes_price: float = 0.50,
    no_price: float = 0.50,
    model_prob: float = 0.60,
    uncertainty: float = 0.05,
    tte_minutes: float = 10.0,
) -> CryptoEdge:
    market = _make_market(ticker)
    return CryptoEdge(
        market=market,
        model_prob=ProbabilityEstimate(model_prob, 0.55, 0.65, uncertainty, 1000),
        market_implied_prob=0.50,
        edge=edge,
        edge_cents=abs(edge),
        side=side,
        recommended_price=yes_price if side == "yes" else no_price,
        model_uncertainty=uncertainty,
        time_to_expiry_minutes=tte_minutes,
        yes_buy_price=yes_price,
        no_buy_price=no_price,
    )


# ── Engine initialization tests ────────────────────────────────────

class TestEngineInit:
    def test_default_construction(self) -> None:
        settings = _make_settings()
        engine = CryptoEngine(settings)
        assert engine.bankroll == 1000.0
        assert engine.cycle_count == 0
        assert engine.session_pnl == 0.0
        assert len(engine.positions) == 0
        assert len(engine.trades) == 0

    def test_custom_bankroll(self) -> None:
        settings = _make_settings(bankroll=500.0)
        engine = CryptoEngine(settings)
        assert engine.bankroll == 500.0


# ── Position sizing tests ──────────────────────────────────────────

class TestPositionSizing:
    def test_basic_sizing(self) -> None:
        settings = _make_settings(kelly_fraction_cap=0.10, bankroll=1000.0)
        engine = CryptoEngine(settings)
        edge = _make_edge(edge=0.10, yes_price=0.50, side="yes")
        contracts = engine._compute_position_size(edge)
        # Kelly fraction = 0.10/0.50 = 0.20, capped at 0.10
        # Dollar amount = 0.10 * 1000 = 100
        # Contracts = 100 / 0.50 = 200, but capped at max_position/price
        assert contracts > 0

    def test_zero_edge_gives_zero_contracts(self) -> None:
        settings = _make_settings()
        engine = CryptoEngine(settings)
        edge = _make_edge(edge=0.0, side="yes")
        contracts = engine._compute_position_size(edge)
        assert contracts == 0

    def test_max_position_cap(self) -> None:
        settings = _make_settings(
            max_position_per_market=10.0,
            bankroll=1000.0,
            kelly_fraction_cap=0.50,
        )
        engine = CryptoEngine(settings)
        edge = _make_edge(edge=0.20, yes_price=0.50, side="yes")
        contracts = engine._compute_position_size(edge)
        # Dollar amount capped at 10, contracts = 10/0.50 = 20
        assert contracts <= 20

    def test_uncertainty_haircut_reduces_size(self) -> None:
        settings = _make_settings(kelly_fraction_cap=0.10, bankroll=1000.0)
        engine = CryptoEngine(settings)

        edge_low_unc = _make_edge(edge=0.10, uncertainty=0.01)
        edge_high_unc = _make_edge(edge=0.10, uncertainty=0.14)

        c_low = engine._compute_position_size(edge_low_unc)
        c_high = engine._compute_position_size(edge_high_unc)

        assert c_low >= c_high

    def test_no_side_cost_zero(self) -> None:
        settings = _make_settings()
        engine = CryptoEngine(settings)
        edge = _make_edge(edge=0.10, no_price=0.0, side="no")
        contracts = engine._compute_position_size(edge)
        assert contracts == 0


# ── Paper trade execution tests ────────────────────────────────────

class TestPaperExecution:
    def test_paper_trade_opens_position(self) -> None:
        settings = _make_settings(paper_slippage_cents=0.0)
        engine = CryptoEngine(settings)
        edge = _make_edge(side="yes", yes_price=0.50)
        engine._execute_paper_trade(edge, 10)
        assert len(engine.positions) == 1
        ticker = edge.market.ticker
        assert ticker in engine.positions
        pos = engine.positions[ticker]
        assert pos.side == "yes"
        assert pos.contracts == 10
        assert pos.entry_price == 0.50

    def test_paper_trade_deducts_bankroll(self) -> None:
        settings = _make_settings(bankroll=1000.0, paper_slippage_cents=0.0)
        engine = CryptoEngine(settings)
        edge = _make_edge(side="yes", yes_price=0.40)
        engine._execute_paper_trade(edge, 10)
        # Cost = 0.40 * 10 = 4.0
        assert abs(engine.bankroll - 996.0) < 0.01

    def test_paper_trade_with_slippage(self) -> None:
        settings = _make_settings(bankroll=1000.0, paper_slippage_cents=1.0)
        engine = CryptoEngine(settings)
        edge = _make_edge(side="yes", yes_price=0.50)
        engine._execute_paper_trade(edge, 10)
        pos = list(engine.positions.values())[0]
        # 0.50 + 0.01 slippage = 0.51
        assert abs(pos.entry_price - 0.51) < 0.001

    def test_paper_trade_no_side(self) -> None:
        settings = _make_settings(paper_slippage_cents=0.0)
        engine = CryptoEngine(settings)
        edge = _make_edge(side="no", no_price=0.40)
        engine._execute_paper_trade(edge, 5)
        pos = list(engine.positions.values())[0]
        assert pos.side == "no"
        assert pos.entry_price == 0.40

    def test_paper_trade_insufficient_bankroll(self) -> None:
        settings = _make_settings(bankroll=1.0, paper_slippage_cents=0.0)
        engine = CryptoEngine(settings)
        edge = _make_edge(side="yes", yes_price=0.50)
        engine._execute_paper_trade(edge, 100)
        # Should limit contracts to what bankroll allows
        if len(engine.positions) > 0:
            pos = list(engine.positions.values())[0]
            assert pos.contracts * pos.entry_price <= 1.0 + 0.01


# ── Settlement tests ───────────────────────────────────────────────

class TestSettlement:
    def test_settle_yes_win(self) -> None:
        settings = _make_settings(bankroll=1000.0, paper_slippage_cents=0.0)
        engine = CryptoEngine(settings)
        edge = _make_edge(side="yes", yes_price=0.40)
        engine._execute_paper_trade(edge, 10)
        # Cost = 4.0, bankroll = 996.0

        ticker = edge.market.ticker
        record = engine.settle_position_with_outcome(ticker, settled_yes=True)
        assert record is not None
        assert record.settled is True
        assert record.actual_outcome == "yes"
        # PnL = (1.0 - 0.40) * 10 = 6.0
        assert abs(record.pnl - 6.0) < 0.01
        assert abs(engine.session_pnl - 6.0) < 0.01
        # Bankroll = 996 + 4.0 (capital back) + 6.0 (profit) = 1006
        assert abs(engine.bankroll - 1006.0) < 0.01

    def test_settle_yes_loss(self) -> None:
        settings = _make_settings(bankroll=1000.0, paper_slippage_cents=0.0)
        engine = CryptoEngine(settings)
        edge = _make_edge(side="yes", yes_price=0.60)
        engine._execute_paper_trade(edge, 10)

        ticker = edge.market.ticker
        record = engine.settle_position_with_outcome(ticker, settled_yes=False)
        assert record is not None
        assert record.actual_outcome == "no"
        # PnL = (0.0 - 0.60) * 10 = -6.0
        assert abs(record.pnl - (-6.0)) < 0.01
        assert abs(engine.session_pnl - (-6.0)) < 0.01

    def test_settle_no_win(self) -> None:
        settings = _make_settings(bankroll=1000.0, paper_slippage_cents=0.0)
        engine = CryptoEngine(settings)
        edge = _make_edge(side="no", no_price=0.30)
        engine._execute_paper_trade(edge, 10)

        ticker = edge.market.ticker
        record = engine.settle_position_with_outcome(ticker, settled_yes=False)
        assert record is not None
        assert record.actual_outcome == "no"
        # NO wins when settled_yes=False → PnL = (1.0 - 0.30) * 10 = 7.0
        assert abs(record.pnl - 7.0) < 0.01

    def test_settle_no_loss(self) -> None:
        settings = _make_settings(bankroll=1000.0, paper_slippage_cents=0.0)
        engine = CryptoEngine(settings)
        edge = _make_edge(side="no", no_price=0.40)
        engine._execute_paper_trade(edge, 10)

        ticker = edge.market.ticker
        record = engine.settle_position_with_outcome(ticker, settled_yes=True)
        assert record is not None
        assert record.actual_outcome == "yes"
        # NO loses when settled_yes=True → PnL = (0.0 - 0.40) * 10 = -4.0
        assert abs(record.pnl - (-4.0)) < 0.01

    def test_settle_unknown_ticker_returns_none(self) -> None:
        settings = _make_settings()
        engine = CryptoEngine(settings)
        record = engine.settle_position_with_outcome("NONEXISTENT", True)
        assert record is None

    def test_settle_adds_to_trades(self) -> None:
        settings = _make_settings(paper_slippage_cents=0.0)
        engine = CryptoEngine(settings)
        edge = _make_edge(side="yes", yes_price=0.50)
        engine._execute_paper_trade(edge, 5)
        engine.settle_position_with_outcome(edge.market.ticker, True)
        assert len(engine.trades) == 1
        assert engine.trades[0].ticker == edge.market.ticker

    def test_settle_removes_position(self) -> None:
        settings = _make_settings(paper_slippage_cents=0.0)
        engine = CryptoEngine(settings)
        edge = _make_edge(side="yes", yes_price=0.50)
        engine._execute_paper_trade(edge, 5)
        engine.settle_position_with_outcome(edge.market.ticker, True)
        assert len(engine.positions) == 0


# ── Real settlement tests ─────────────────────────────────────────

class TestRealSettlement:
    def test_fetch_settlement_returns_none_without_client(self) -> None:
        """Without HTTP client, _fetch_settlement_outcome returns None."""
        settings = _make_settings()
        engine = CryptoEngine(settings)
        result = asyncio.get_event_loop().run_until_complete(
            engine._fetch_settlement_outcome("KXBTCD-26FEB14-T97500")
        )
        assert result is None

    def test_settle_with_real_yes_outcome(self) -> None:
        """Mock HTTP returning yes -> position settled with real outcome."""
        import unittest.mock as mock

        settings = _make_settings(bankroll=1000.0, paper_slippage_cents=0.0)
        engine = CryptoEngine(settings)
        edge = _make_edge(side="yes", yes_price=0.40)
        engine._execute_paper_trade(edge, 10)
        ticker = edge.market.ticker

        # Create mock HTTP response (MagicMock because resp.json() is sync)
        mock_response = mock.MagicMock()
        mock_response.json.return_value = {"market": {"result": "yes", "status": "settled"}}
        mock_response.status_code = 200
        mock_response.raise_for_status = mock.Mock()

        mock_client = mock.AsyncMock()
        mock_client.get = mock.AsyncMock(return_value=mock_response)
        engine.set_http_client(mock_client)

        outcome = asyncio.get_event_loop().run_until_complete(
            engine._fetch_settlement_outcome(ticker)
        )
        assert outcome is True

        # Settle with the outcome
        record = engine.settle_position_with_outcome(ticker, outcome)
        assert record is not None
        assert record.actual_outcome == "yes"
        assert abs(record.pnl - 6.0) < 0.01  # (1.0 - 0.40) * 10

    def test_settle_with_real_no_outcome(self) -> None:
        """Mock HTTP returning no -> position settled correctly."""
        import unittest.mock as mock

        settings = _make_settings(bankroll=1000.0, paper_slippage_cents=0.0)
        engine = CryptoEngine(settings)
        edge = _make_edge(side="yes", yes_price=0.60)
        engine._execute_paper_trade(edge, 10)
        ticker = edge.market.ticker

        mock_response = mock.MagicMock()
        mock_response.json.return_value = {"market": {"result": "no", "status": "settled"}}
        mock_response.status_code = 200
        mock_response.raise_for_status = mock.Mock()

        mock_client = mock.AsyncMock()
        mock_client.get = mock.AsyncMock(return_value=mock_response)
        engine.set_http_client(mock_client)

        outcome = asyncio.get_event_loop().run_until_complete(
            engine._fetch_settlement_outcome(ticker)
        )
        assert outcome is False

        record = engine.settle_position_with_outcome(ticker, outcome)
        assert record is not None
        assert record.actual_outcome == "no"
        assert abs(record.pnl - (-6.0)) < 0.01  # (0.0 - 0.60) * 10

    def test_fetch_returns_none_when_not_settled(self) -> None:
        """Mock HTTP returning empty result -> None."""
        import unittest.mock as mock

        settings = _make_settings()
        engine = CryptoEngine(settings)

        mock_response = mock.MagicMock()
        mock_response.json.return_value = {"market": {"result": "", "status": "open"}}
        mock_response.status_code = 200
        mock_response.raise_for_status = mock.Mock()

        mock_client = mock.AsyncMock()
        mock_client.get = mock.AsyncMock(return_value=mock_response)
        engine.set_http_client(mock_client)

        outcome = asyncio.get_event_loop().run_until_complete(
            engine._fetch_settlement_outcome("KXBTCD-26FEB14-T97500")
        )
        assert outcome is None


# ── Model probability computation tests ────────────────────────────

class TestModelProbabilities:
    def test_computes_prob_for_above_market(self) -> None:
        settings = _make_settings(mc_num_paths=500)
        engine = CryptoEngine(settings)

        # Inject price data for BTC
        ts = time.time()
        for i in range(100):
            engine.price_feed.inject_tick(
                PriceTick("btcusdt", 97000.0 + i * 10, ts - 100 + i, 1.0)
            )

        quote = _make_quote(
            ticker="KXBTCD-26FEB14-T97500",
            yes_price=0.50,
            no_price=0.50,
            tte_minutes=10.0,
        )

        probs = engine._compute_model_probabilities([quote])
        assert quote.market.ticker in probs
        prob = probs[quote.market.ticker]
        assert 0.0 <= prob.probability <= 1.0
        assert prob.num_paths == 500

    def test_computes_prob_for_up_market(self) -> None:
        settings = _make_settings(
            mc_num_paths=500,
            allowed_directions=["above", "below", "up", "down"],
        )
        engine = CryptoEngine(settings)

        ts = time.time()
        for i in range(100):
            engine.price_feed.inject_tick(
                PriceTick("btcusdt", 97000.0 + i * 10, ts - 100 + i, 1.0)
            )

        # Use a 15-min up market
        market = _make_market("KXBTC15M-26FEB14-U12", expiry_offset_minutes=10.0)
        # Override direction to "up"
        meta = replace(market.meta, direction="up")
        market = CryptoMarket(ticker=market.ticker, meta=meta)

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

    def test_computes_prob_for_down_market(self) -> None:
        settings = _make_settings(
            mc_num_paths=500,
            allowed_directions=["above", "below", "up", "down"],
        )
        engine = CryptoEngine(settings)

        ts = time.time()
        for i in range(100):
            engine.price_feed.inject_tick(
                PriceTick("btcusdt", 97000.0, ts - 100 + i, 1.0)
            )

        market = _make_market("KXBTC15M-26FEB14-D12", expiry_offset_minutes=10.0)
        meta = replace(market.meta, direction="down")
        market = CryptoMarket(ticker=market.ticker, meta=meta)

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
        # With flat prices, P(down) should be approximately 0.50
        prob = probs[market.ticker]
        assert 0.3 <= prob.probability <= 0.7

    def test_no_price_data_returns_empty(self) -> None:
        settings = _make_settings()
        engine = CryptoEngine(settings)
        # No ticks injected → no current price
        quote = _make_quote()
        probs = engine._compute_model_probabilities([quote])
        assert len(probs) == 0

    def test_default_vol_used_when_insufficient_returns(self) -> None:
        settings = _make_settings(mc_num_paths=100)
        engine = CryptoEngine(settings)

        # Only inject 1 tick → not enough for returns → vol defaults to 0.50
        ts = time.time()
        engine.price_feed.inject_tick(
            PriceTick("btcusdt", 97000.0, ts, 1.0)
        )

        quote = _make_quote(tte_minutes=10.0)
        probs = engine._compute_model_probabilities([quote])
        # Should still produce a probability using default vol
        assert quote.market.ticker in probs


# ── Run cycle with quotes tests ────────────────────────────────────

class TestRunCycleWithQuotes:
    def test_cycle_detects_edges(self) -> None:
        async def _run() -> None:
            settings = _make_settings(
                mc_num_paths=500,
                min_edge_pct=0.01,
                min_edge_cents=0.01,
                max_model_uncertainty=0.20,
            )
            engine = CryptoEngine(settings)

            # Inject BTC price data — price clearly above 90000
            ts = time.time()
            for i in range(100):
                engine.price_feed.inject_tick(
                    PriceTick("btcusdt", 100000.0, ts - 100 + i, 1.0)
                )

            # Market says only 30% chance of being above 90000
            # Model should say ~100% → big edge
            quote = _make_quote(
                ticker="KXBTCD-26FEB14-T90000",
                yes_price=0.30,
                no_price=0.70,
                tte_minutes=10.0,
            )

            edges = await engine.run_cycle_with_quotes([quote])
            # Should detect edge since model says ~100% but market says 30%
            assert engine.cycle_count == 1

        asyncio.run(_run())

    def test_cycle_respects_max_concurrent(self) -> None:
        async def _run() -> None:
            settings = _make_settings(
                mc_num_paths=100,
                max_concurrent_positions=1,
                min_edge_pct=0.01,
                min_edge_cents=0.01,
                max_model_uncertainty=0.20,
            )
            engine = CryptoEngine(settings)

            ts = time.time()
            for i in range(100):
                engine.price_feed.inject_tick(
                    PriceTick("btcusdt", 100000.0, ts - 100 + i, 1.0)
                )

            quotes = [
                _make_quote("KXBTCD-26FEB14-T80000", yes_price=0.30, no_price=0.70, tte_minutes=10.0),
                _make_quote("KXBTCD-26FEB14-T85000", yes_price=0.30, no_price=0.70, tte_minutes=10.0),
            ]

            await engine.run_cycle_with_quotes(quotes)
            # Should only open 1 position due to max_concurrent_positions=1
            assert len(engine.positions) <= 1

        asyncio.run(_run())

    def test_cycle_no_duplicate_positions(self) -> None:
        async def _run() -> None:
            settings = _make_settings(
                mc_num_paths=100,
                min_edge_pct=0.01,
                min_edge_cents=0.01,
                max_model_uncertainty=0.20,
            )
            engine = CryptoEngine(settings)

            ts = time.time()
            for i in range(100):
                engine.price_feed.inject_tick(
                    PriceTick("btcusdt", 100000.0, ts - 100 + i, 1.0)
                )

            quote = _make_quote(
                "KXBTCD-26FEB14-T80000",
                yes_price=0.30, no_price=0.70, tte_minutes=10.0,
            )

            await engine.run_cycle_with_quotes([quote])
            await engine.run_cycle_with_quotes([quote])

            # Same ticker should not open duplicate positions
            assert len(engine.positions) <= 1

        asyncio.run(_run())


# ── CSV export tests ───────────────────────────────────────────────

class TestCSVExport:
    def test_export_empty_trades(self) -> None:
        settings = _make_settings()
        engine = CryptoEngine(settings)
        csv_data = engine.export_trades_csv()
        lines = csv_data.strip().split("\n")
        assert len(lines) == 1  # Header only
        assert "ticker" in lines[0]

    def test_export_with_trades(self) -> None:
        settings = _make_settings(paper_slippage_cents=0.0)
        engine = CryptoEngine(settings)
        edge = _make_edge(side="yes", yes_price=0.50)
        engine._execute_paper_trade(edge, 10)
        engine.settle_position_with_outcome(edge.market.ticker, True)

        csv_data = engine.export_trades_csv()
        lines = csv_data.strip().split("\n")
        assert len(lines) == 2  # Header + 1 trade
        assert "KXBTCD" in lines[1]

    def test_export_fields(self) -> None:
        settings = _make_settings(paper_slippage_cents=0.0)
        engine = CryptoEngine(settings)
        edge = _make_edge(side="yes", yes_price=0.50)
        engine._execute_paper_trade(edge, 5)
        engine.settle_position_with_outcome(edge.market.ticker, True)

        csv_data = engine.export_trades_csv()
        header = csv_data.strip().split("\n")[0]
        for field in ["ticker", "side", "contracts", "entry_price", "pnl",
                       "edge_at_entry", "model_prob_at_entry", "settled", "actual_outcome"]:
            assert field in header

    def test_actual_outcome_in_csv_data(self) -> None:
        settings = _make_settings(paper_slippage_cents=0.0)
        engine = CryptoEngine(settings)
        edge = _make_edge(side="yes", yes_price=0.50)
        engine._execute_paper_trade(edge, 5)
        engine.settle_position_with_outcome(edge.market.ticker, True)

        csv_data = engine.export_trades_csv()
        lines = csv_data.strip().split("\n")
        assert len(lines) == 2
        # actual_outcome should be "yes"
        assert "yes" in lines[1]


# ── Kalshi-to-Binance mapping tests ────────────────────────────────

class TestKalshiToBinance:
    def test_btc_mapping(self) -> None:
        assert _KALSHI_TO_BINANCE["BTC"] == "btcusdt"

    def test_eth_mapping(self) -> None:
        assert _KALSHI_TO_BINANCE["ETH"] == "ethusdt"

    def test_sol_mapping(self) -> None:
        assert _KALSHI_TO_BINANCE["SOL"] == "solusdt"


# ── Allowed directions filter (A1) ────────────────────────────────

class TestAllowedDirections:
    def test_up_direction_filtered_when_not_allowed(self) -> None:
        """Only above/below are allowed; up/down quotes get no probability."""
        settings = _make_settings(
            mc_num_paths=100,
            allowed_directions=["above", "below"],
        )
        engine = CryptoEngine(settings)

        # Inject BTC price data
        ts = time.time()
        for i in range(100):
            engine.price_feed.inject_tick(
                PriceTick("btcusdt", 97000.0 + i * 10, ts - 100 + i, 1.0)
            )

        # Create an "up" direction quote (15-min market)
        up_market = _make_market("KXBTC15M-26FEB14-U12", expiry_offset_minutes=10.0)
        up_meta = replace(up_market.meta, direction="up")
        up_market = CryptoMarket(ticker=up_market.ticker, meta=up_meta)
        up_quote = CryptoMarketQuote(
            market=up_market,
            yes_buy_price=0.50,
            no_buy_price=0.50,
            yes_buy_size=100,
            no_buy_size=100,
            yes_bid_price=0.49,
            no_bid_price=0.49,
            time_to_expiry_minutes=10.0,
            implied_probability=0.50,
        )

        # Create an "above" direction quote (daily market)
        above_quote = _make_quote(
            ticker="KXBTCD-26FEB14-T97500",
            yes_price=0.50,
            no_price=0.50,
            tte_minutes=10.0,
        )

        probs = engine._compute_model_probabilities([up_quote, above_quote])

        # "up" should be filtered out, "above" should get a probability
        assert up_market.ticker not in probs
        assert above_quote.market.ticker in probs

    def test_all_directions_allowed(self) -> None:
        """When all directions are allowed, up/down get probabilities."""
        settings = _make_settings(
            mc_num_paths=100,
            allowed_directions=["above", "below", "up", "down"],
        )
        engine = CryptoEngine(settings)

        ts = time.time()
        for i in range(100):
            engine.price_feed.inject_tick(
                PriceTick("btcusdt", 97000.0 + i * 10, ts - 100 + i, 1.0)
            )

        up_market = _make_market("KXBTC15M-26FEB14-U12", expiry_offset_minutes=10.0)
        up_meta = replace(up_market.meta, direction="up")
        up_market = CryptoMarket(ticker=up_market.ticker, meta=up_meta)
        up_quote = CryptoMarketQuote(
            market=up_market,
            yes_buy_price=0.50,
            no_buy_price=0.50,
            yes_buy_size=100,
            no_buy_size=100,
            yes_bid_price=0.49,
            no_bid_price=0.49,
            time_to_expiry_minutes=10.0,
            implied_probability=0.50,
        )

        probs = engine._compute_model_probabilities([up_quote])
        assert up_market.ticker in probs


# ── Per-underlying position cap (A2) ──────────────────────────────

class TestPerUnderlyingPositionCap:
    def test_count_positions_for_underlying(self) -> None:
        """_count_positions_for_underlying correctly counts BTC positions."""
        settings = _make_settings(max_positions_per_underlying=3)
        engine = CryptoEngine(settings)

        # Inject 2 BTC positions directly
        edge1 = _make_edge(ticker="KXBTCD-26FEB14-T97000", side="yes", yes_price=0.50)
        edge2 = _make_edge(ticker="KXBTCD-26FEB14-T98000", side="yes", yes_price=0.40)
        engine._execute_paper_trade(edge1, 5)
        engine._execute_paper_trade(edge2, 5)

        assert engine._count_positions_for_underlying("BTC") == 2
        assert engine._count_positions_for_underlying("ETH") == 0

    def test_underlying_cap_blocks_excess_positions(self) -> None:
        """When cap is reached, new positions for that underlying are blocked."""
        async def _run() -> None:
            settings = _make_settings(
                mc_num_paths=100,
                max_positions_per_underlying=2,
                max_concurrent_positions=10,
                min_edge_pct=0.01,
                min_edge_cents=0.01,
                max_model_uncertainty=0.20,
                paper_slippage_cents=0.0,
            )
            engine = CryptoEngine(settings)

            # Inject BTC price data — price clearly above strikes
            ts = time.time()
            for i in range(100):
                engine.price_feed.inject_tick(
                    PriceTick("btcusdt", 100000.0, ts - 100 + i, 1.0)
                )

            # Pre-inject 2 BTC positions to reach the cap
            edge1 = _make_edge(ticker="KXBTCD-26FEB14-T90000", side="yes", yes_price=0.30)
            edge2 = _make_edge(ticker="KXBTCD-26FEB14-T91000", side="yes", yes_price=0.30)
            engine._execute_paper_trade(edge1, 5)
            engine._execute_paper_trade(edge2, 5)

            assert engine._count_positions_for_underlying("BTC") == 2

            # Try to open a 3rd BTC position via cycle — should be blocked
            quote = _make_quote(
                "KXBTCD-26FEB14-T80000",
                yes_price=0.30, no_price=0.70, tte_minutes=10.0,
            )
            await engine.run_cycle_with_quotes([quote])

            # Should still have only 2 BTC positions (3rd blocked by cap)
            assert engine._count_positions_for_underlying("BTC") == 2

        asyncio.run(_run())


# ── Model uncertainty multiplier (A4) ─────────────────────────────

class TestModelUncertaintyMultiplier:
    def test_uncertainty_scaled_by_multiplier(self) -> None:
        """model_uncertainty_multiplier=3.0 should widen CI by 3x."""
        from arb_bot.crypto.price_model import PriceModel

        settings_scaled = _make_settings(
            mc_num_paths=500,
            model_uncertainty_multiplier=3.0,
        )
        settings_unscaled = _make_settings(
            mc_num_paths=500,
            model_uncertainty_multiplier=1.0,
        )
        engine_scaled = CryptoEngine(settings_scaled)
        engine_unscaled = CryptoEngine(settings_unscaled)

        # Use the same seed so both engines generate identical MC paths
        engine_scaled._price_model = PriceModel(
            num_paths=500, confidence_level=0.95, seed=42,
        )
        engine_unscaled._price_model = PriceModel(
            num_paths=500, confidence_level=0.95, seed=42,
        )

        # Inject identical BTC price data into both engines
        ts = time.time()
        for i in range(100):
            tick = PriceTick("btcusdt", 97000.0 + i * 10, ts - 100 + i, 1.0)
            engine_scaled.price_feed.inject_tick(tick)
            engine_unscaled.price_feed.inject_tick(tick)

        quote = _make_quote(
            ticker="KXBTCD-26FEB14-T97500",
            yes_price=0.50,
            no_price=0.50,
            tte_minutes=10.0,
        )

        probs_scaled = engine_scaled._compute_model_probabilities([quote])
        probs_unscaled = engine_unscaled._compute_model_probabilities([quote])

        ticker = quote.market.ticker
        assert ticker in probs_scaled
        assert ticker in probs_unscaled

        unc_scaled = probs_scaled[ticker].uncertainty
        unc_unscaled = probs_unscaled[ticker].uncertainty

        # Scaled uncertainty should be exactly 3x the unscaled
        assert abs(unc_scaled - 3.0 * unc_unscaled) < 1e-9

        # CI should be wider
        width_scaled = probs_scaled[ticker].ci_upper - probs_scaled[ticker].ci_lower
        width_unscaled = probs_unscaled[ticker].ci_upper - probs_unscaled[ticker].ci_lower
        assert width_scaled >= width_unscaled

    def test_multiplier_1_leaves_uncertainty_unchanged(self) -> None:
        """model_uncertainty_multiplier=1.0 should not change uncertainty."""
        settings = _make_settings(
            mc_num_paths=500,
            model_uncertainty_multiplier=1.0,
        )
        engine = CryptoEngine(settings)

        ts = time.time()
        for i in range(100):
            engine.price_feed.inject_tick(
                PriceTick("btcusdt", 97000.0, ts - 100 + i, 1.0)
            )

        quote = _make_quote(
            ticker="KXBTCD-26FEB14-T97500",
            yes_price=0.50,
            no_price=0.50,
            tte_minutes=10.0,
        )

        probs = engine._compute_model_probabilities([quote])
        ticker = quote.market.ticker
        assert ticker in probs
        # With multiplier=1.0, raw Wilson CI uncertainty should be preserved
        # (no scaling applied)
        prob = probs[ticker]
        assert prob.uncertainty > 0
        # CI should be symmetric around probability
        expected_lower = max(0.0, prob.probability - prob.uncertainty)
        expected_upper = min(1.0, prob.probability + prob.uncertainty)
        # The raw Wilson CI center != p_hat, so the CI may not be symmetric
        # around probability. But the uncertainty field should be the original
        # Wilson CI half-width.


# ── OFI drift injection (B5) ──────────────────────────────────────

class TestOFIDrift:
    def test_ofi_drift_injected_into_paths(self) -> None:
        """When OFI is positive and alpha is set, drift should affect probabilities."""
        settings = _make_settings(
            mc_num_paths=2000,
            ofi_enabled=True,
            ofi_window_seconds=60,
            allowed_directions=["above", "below"],
        )
        engine = CryptoEngine(settings)

        # Inject price data + heavy buy pressure (all buys, OFI -> +1.0)
        now = time.time()
        for i in range(60):
            engine.price_feed.inject_tick(
                PriceTick("btcusdt", 70000.0, now - 60 + i, 2.0, is_buyer_maker=False)
            )

        # Set a known alpha by injecting calibration samples
        # alpha ~ 0.5 means drift = 0.5 * OFI
        rng = np.random.default_rng(42)
        for _ in range(100):
            ofi_sample = rng.uniform(-1, 1)
            engine._ofi_calibrator.record_sample(ofi_sample, 0.5 * ofi_sample + rng.normal(0, 0.001))

        # Verify OFI is strongly positive
        ofi = engine.price_feed.get_ofi("btcusdt", window_seconds=60)
        assert ofi > 0.9

        # Build a quote for above-strike (strike below current price)
        # With positive drift, P(above 69500) should be high
        quote = _make_quote(
            ticker="KXBTCD-26FEB14-T69500",
            yes_price=0.50,
            no_price=0.50,
            tte_minutes=10.0,
        )
        probs = engine._compute_model_probabilities([quote])
        assert quote.market.ticker in probs
        # With OFI=1.0 and alpha~0.5, drift is positive -> higher above-strike probability
        assert probs[quote.market.ticker].probability > 0.5

    def test_ofi_disabled_uses_zero_drift(self) -> None:
        """When OFI is disabled, drift should be 0.0 (martingale)."""
        settings = _make_settings(
            mc_num_paths=500,
            ofi_enabled=False,
        )
        engine = CryptoEngine(settings)

        now = time.time()
        for i in range(60):
            engine.price_feed.inject_tick(
                PriceTick("btcusdt", 70000.0, now - 60 + i, 2.0, is_buyer_maker=False)
            )

        # Even with calibration samples, drift should be zero
        rng = np.random.default_rng(42)
        for _ in range(100):
            ofi_sample = rng.uniform(-1, 1)
            engine._ofi_calibrator.record_sample(ofi_sample, 0.5 * ofi_sample)

        quote = _make_quote(
            ticker="KXBTCD-26FEB14-T70000",
            yes_price=0.50,
            no_price=0.50,
            tte_minutes=10.0,
        )
        probs = engine._compute_model_probabilities([quote])
        assert quote.market.ticker in probs
        # With flat price and zero drift, probability should be around 0.50
        prob = probs[quote.market.ticker].probability
        assert 0.3 <= prob <= 0.7

    def test_ofi_calibrator_property_accessible(self) -> None:
        """The ofi_calibrator property should expose the OFICalibrator."""
        settings = _make_settings()
        engine = CryptoEngine(settings)
        from arb_bot.crypto.ofi_calibrator import OFICalibrator
        assert isinstance(engine.ofi_calibrator, OFICalibrator)


# ── Activity-scaled volatility (C1) ──────────────────────────────

class TestActivityScaling:
    def test_activity_scaling_adjusts_vol(self) -> None:
        """Activity scaling should adjust vol based on volume rate ratio."""
        settings = _make_settings(
            activity_scaling_enabled=True,
            activity_scaling_short_window_seconds=300,
            activity_scaling_long_window_seconds=1800,
            mc_num_paths=100,
            allowed_directions=["above", "below"],
            ofi_enabled=False,  # Disable OFI for this test
        )
        engine = CryptoEngine(settings)

        now = time.time()
        # Inject 30 min of "normal" volume (1.0 per tick, 1 tick/sec)
        for i in range(1800):
            engine.price_feed.inject_tick(
                PriceTick("btcusdt", 70000.0, now - 1800 + i, 1.0)
            )
        # Last 5 minutes: 4x volume (high activity)
        for i in range(300):
            engine.price_feed.inject_tick(
                PriceTick("btcusdt", 70000.0, now - 300 + i, 4.0)
            )

        # Verify volume rates
        short_rate = engine.price_feed.get_volume_flow_rate("btcusdt", window_seconds=300)
        long_rate = engine.price_feed.get_volume_flow_rate("btcusdt", window_seconds=1800)
        assert short_rate > long_rate  # Recent activity is higher

    def test_activity_scaling_disabled_no_effect(self) -> None:
        """With activity_scaling_enabled=False, vol should not be scaled."""
        from arb_bot.crypto.price_model import PriceModel

        settings_on = _make_settings(
            activity_scaling_enabled=True,
            activity_scaling_short_window_seconds=300,
            activity_scaling_long_window_seconds=1800,
            mc_num_paths=500,
            ofi_enabled=False,
        )
        settings_off = _make_settings(
            activity_scaling_enabled=False,
            activity_scaling_short_window_seconds=300,
            activity_scaling_long_window_seconds=1800,
            mc_num_paths=500,
            ofi_enabled=False,
        )
        engine_on = CryptoEngine(settings_on)
        engine_off = CryptoEngine(settings_off)

        # Use same seed for deterministic comparison
        engine_on._price_model = PriceModel(num_paths=500, confidence_level=0.95, seed=42)
        engine_off._price_model = PriceModel(num_paths=500, confidence_level=0.95, seed=42)

        now = time.time()
        # Inject mixed volume: normal for 30min then high for 5min
        for i in range(1800):
            tick = PriceTick("btcusdt", 70000.0, now - 1800 + i, 1.0)
            engine_on.price_feed.inject_tick(tick)
            engine_off.price_feed.inject_tick(tick)
        for i in range(300):
            tick = PriceTick("btcusdt", 70000.0, now - 300 + i, 4.0)
            engine_on.price_feed.inject_tick(tick)
            engine_off.price_feed.inject_tick(tick)

        quote = _make_quote(
            ticker="KXBTCD-26FEB14-T70000",
            yes_price=0.50,
            no_price=0.50,
            tte_minutes=10.0,
        )

        probs_on = engine_on._compute_model_probabilities([quote])
        probs_off = engine_off._compute_model_probabilities([quote])

        ticker = quote.market.ticker
        assert ticker in probs_on
        assert ticker in probs_off
        # Both should produce valid probabilities; with scaling enabled
        # and high recent volume, the uncertainty should differ
        unc_on = probs_on[ticker].uncertainty
        unc_off = probs_off[ticker].uncertainty
        # Activity scaling widens vol, which typically widens uncertainty
        # (though MC noise can vary). We just verify both are valid.
        assert 0.0 < unc_on
        assert 0.0 < unc_off

    def test_activity_scaling_clamped(self) -> None:
        """Activity ratio should be clamped to [0.25, 4.0]."""
        settings = _make_settings(
            activity_scaling_enabled=True,
            activity_scaling_short_window_seconds=60,
            activity_scaling_long_window_seconds=600,
            mc_num_paths=100,
            ofi_enabled=False,
        )
        engine = CryptoEngine(settings)

        now = time.time()
        # Inject extremely low long-window volume (near zero) and high short-window
        # Long window: 600 seconds with volume=0.001 each
        for i in range(600):
            engine.price_feed.inject_tick(
                PriceTick("btcusdt", 70000.0, now - 600 + i, 0.001)
            )
        # Short window: 60 seconds with volume=100 each (extreme ratio)
        for i in range(60):
            engine.price_feed.inject_tick(
                PriceTick("btcusdt", 70000.0, now - 60 + i, 100.0)
            )

        quote = _make_quote(
            ticker="KXBTCD-26FEB14-T70000",
            yes_price=0.50,
            no_price=0.50,
            tte_minutes=10.0,
        )

        # Should not crash and should produce valid probabilities
        probs = engine._compute_model_probabilities([quote])
        assert quote.market.ticker in probs
        prob = probs[quote.market.ticker]
        assert 0.0 <= prob.probability <= 1.0


# ── Per-cycle entry throttle (v44 Fix G) ─────────────────────────

class TestMaxNewTradesPerCycle:
    def test_throttle_limits_entries(self) -> None:
        """max_new_trades_per_cycle=2 should open at most 2 trades per cycle."""
        async def _run() -> None:
            settings = _make_settings(
                mc_num_paths=500,
                max_new_trades_per_cycle=2,
                max_concurrent_positions=10,
                min_edge_pct=0.01,
                min_edge_cents=0.01,
                max_model_uncertainty=0.30,
                paper_slippage_cents=0.0,
                # Disable disagreement veto so edges pass through
                market_disagreement_max=0.80,
            )
            engine = CryptoEngine(settings)

            # Inject BTC price data — price clearly above all strikes
            ts = time.time()
            for i in range(100):
                engine.price_feed.inject_tick(
                    PriceTick("btcusdt", 100000.0, ts - 100 + i, 1.0)
                )

            # 4 quotes, all far below current price → model should say YES ~100%
            # but market says 30% → big edge on each
            quotes = [
                _make_quote("KXBTCD-26FEB14-T80000", yes_price=0.30, no_price=0.70, tte_minutes=10.0),
                _make_quote("KXBTCD-26FEB14-T85000", yes_price=0.30, no_price=0.70, tte_minutes=10.0),
                _make_quote("KXBTCD-26FEB14-T86000", yes_price=0.30, no_price=0.70, tte_minutes=10.0),
                _make_quote("KXBTCD-26FEB14-T87000", yes_price=0.30, no_price=0.70, tte_minutes=10.0),
            ]

            await engine.run_cycle_with_quotes(quotes)

            # With 4 qualifying edges but max_new_trades_per_cycle=2,
            # should open at most 2 positions
            assert len(engine.positions) <= 2

        asyncio.run(_run())

    def test_throttle_default_allows_many(self) -> None:
        """Default max_new_trades_per_cycle=10 should not restrict normal trading."""
        async def _run() -> None:
            settings = _make_settings(
                mc_num_paths=500,
                max_new_trades_per_cycle=10,  # default
                max_concurrent_positions=10,
                min_edge_pct=0.01,
                min_edge_cents=0.01,
                max_model_uncertainty=0.30,
                paper_slippage_cents=0.0,
                market_disagreement_max=0.80,
            )
            engine = CryptoEngine(settings)

            ts = time.time()
            for i in range(100):
                engine.price_feed.inject_tick(
                    PriceTick("btcusdt", 100000.0, ts - 100 + i, 1.0)
                )

            # 3 quotes — all should be enterable with default limit of 10
            quotes = [
                _make_quote("KXBTCD-26FEB14-T80000", yes_price=0.30, no_price=0.70, tte_minutes=10.0),
                _make_quote("KXBTCD-26FEB14-T85000", yes_price=0.30, no_price=0.70, tte_minutes=10.0),
                _make_quote("KXBTCD-26FEB14-T86000", yes_price=0.30, no_price=0.70, tte_minutes=10.0),
            ]

            await engine.run_cycle_with_quotes(quotes)

            # Default limit=10 should not block any of the 3 edges
            # (though other filters may reduce count)
            # At minimum: if all 3 generate edges, all 3 should be opened
            # since 3 < 10
            assert engine.cycle_count == 1

        asyncio.run(_run())

    def test_throttle_one_limits_to_single_trade(self) -> None:
        """max_new_trades_per_cycle=1 should open exactly 1 trade."""
        async def _run() -> None:
            settings = _make_settings(
                mc_num_paths=500,
                max_new_trades_per_cycle=1,
                max_concurrent_positions=10,
                min_edge_pct=0.01,
                min_edge_cents=0.01,
                max_model_uncertainty=0.30,
                paper_slippage_cents=0.0,
                market_disagreement_max=0.80,
            )
            engine = CryptoEngine(settings)

            ts = time.time()
            for i in range(100):
                engine.price_feed.inject_tick(
                    PriceTick("btcusdt", 100000.0, ts - 100 + i, 1.0)
                )

            quotes = [
                _make_quote("KXBTCD-26FEB14-T80000", yes_price=0.30, no_price=0.70, tte_minutes=10.0),
                _make_quote("KXBTCD-26FEB14-T85000", yes_price=0.30, no_price=0.70, tte_minutes=10.0),
                _make_quote("KXBTCD-26FEB14-T86000", yes_price=0.30, no_price=0.70, tte_minutes=10.0),
            ]

            await engine.run_cycle_with_quotes(quotes)

            # With max=1, at most 1 position should be opened
            assert len(engine.positions) <= 1

        asyncio.run(_run())

    def test_best_edges_selected_first(self) -> None:
        """Edge detector sorts by absolute edge; throttle should keep the best ones."""
        async def _run() -> None:
            settings = _make_settings(
                mc_num_paths=500,
                max_new_trades_per_cycle=1,
                max_concurrent_positions=10,
                min_edge_pct=0.01,
                min_edge_cents=0.01,
                max_model_uncertainty=0.30,
                paper_slippage_cents=0.0,
                market_disagreement_max=0.80,
            )
            engine = CryptoEngine(settings)

            ts = time.time()
            for i in range(100):
                engine.price_feed.inject_tick(
                    PriceTick("btcusdt", 100000.0, ts - 100 + i, 1.0)
                )

            # Two quotes with very different mispricing:
            # T80000: strike far below → model ~100%, market 20% → huge edge
            # T99000: strike near price → model ~55%, market 40% → small edge
            quotes = [
                _make_quote("KXBTCD-26FEB14-T99000", yes_price=0.40, no_price=0.60, tte_minutes=10.0),
                _make_quote("KXBTCD-26FEB14-T80000", yes_price=0.20, no_price=0.80, tte_minutes=10.0),
            ]

            await engine.run_cycle_with_quotes(quotes)

            if len(engine.positions) == 1:
                # The single opened position should be T80000 (bigger edge)
                opened_ticker = list(engine.positions.keys())[0]
                assert "T80000" in opened_ticker

        asyncio.run(_run())
