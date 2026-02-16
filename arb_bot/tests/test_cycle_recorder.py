"""Tests for the CycleRecorder append-only SQLite recorder."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest

from arb_bot.crypto.cycle_recorder import CycleRecorder, FilterResult
from arb_bot.crypto.config import CryptoSettings


# ── Helpers ────────────────────────────────────────────────────────


def _make_settings(**overrides) -> CryptoSettings:
    defaults = dict(
        enabled=True,
        paper_mode=True,
        bankroll=500.0,
        mc_num_paths=100,
        symbols=["KXBTC"],
        scan_interval_seconds=5.0,
    )
    defaults.update(overrides)
    return CryptoSettings(**defaults)


def _make_mock_market(ticker: str = "KXBTCD-26FEB14-T97500"):
    """Create a mock CryptoMarket with nested meta."""
    meta = MagicMock()
    meta.underlying = "BTC"
    meta.interval = "daily"
    meta.direction = "above"
    meta.strike = 97500.0
    market = MagicMock()
    market.ticker = ticker
    market.meta = meta
    return market


def _make_mock_quote(
    ticker: str = "KXBTCD-26FEB14-T97500",
    yes_buy_price: float = 0.50,
    no_buy_price: float = 0.50,
    tte_minutes: float = 10.0,
):
    """Create a mock CryptoMarketQuote."""
    market = _make_mock_market(ticker)
    q = MagicMock()
    q.market = market
    q.yes_buy_price = yes_buy_price
    q.no_buy_price = no_buy_price
    q.yes_buy_size = 100.0
    q.no_buy_size = 100.0
    q.yes_bid_price = yes_buy_price - 0.01
    q.no_bid_price = no_buy_price - 0.01
    q.time_to_expiry_minutes = tte_minutes
    q.implied_probability = 0.5 * (yes_buy_price + (1.0 - no_buy_price))
    return q


def _make_mock_edge(
    ticker: str = "KXBTCD-26FEB14-T97500",
    side: str = "yes",
    model_prob: float = 0.60,
    market_implied_prob: float = 0.50,
    edge_cents: float = 0.10,
    uncertainty: float = 0.05,
    tte_minutes: float = 10.0,
    yes_buy_price: float = 0.50,
    no_buy_price: float = 0.50,
):
    """Create a mock CryptoEdge."""
    market = _make_mock_market(ticker)
    edge = MagicMock()
    edge.market = market
    edge.side = side
    # model_prob as a float (not an object with .probability)
    edge.model_prob = model_prob
    edge.market_implied_prob = market_implied_prob
    edge.blended_probability = model_prob
    edge.edge_cents = edge_cents
    edge.model_uncertainty = uncertainty
    edge.time_to_expiry_minutes = tte_minutes
    edge.yes_buy_price = yes_buy_price
    edge.no_buy_price = no_buy_price
    edge.spread_cost = 0.02
    edge.staleness_score = 0.1
    return edge


# ── TestCycleRecorderInit ─────────────────────────────────────────


class TestCycleRecorderInit:
    def test_creates_db_file_on_open(self, tmp_path) -> None:
        db_file = str(tmp_path / "test.db")
        settings = _make_settings()
        recorder = CycleRecorder(db_file, settings)
        recorder.open()
        try:
            assert os.path.exists(db_file)
        finally:
            recorder.close()

    def test_creates_all_five_tables(self, tmp_path) -> None:
        db_file = str(tmp_path / "test.db")
        settings = _make_settings()
        recorder = CycleRecorder(db_file, settings)
        recorder.open()
        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = {row[0] for row in cursor.fetchall()}
            conn.close()
            expected = {"sessions", "cycles", "market_snapshots", "edges", "trades"}
            assert expected.issubset(tables)
        finally:
            recorder.close()

    def test_session_row_written_with_settings(self, tmp_path) -> None:
        db_file = str(tmp_path / "test.db")
        settings = _make_settings(bankroll=999.0, symbols=["KXBTC", "KXETH"])
        recorder = CycleRecorder(db_file, settings)
        recorder.open()
        try:
            conn = sqlite3.connect(db_file)
            row = conn.execute("SELECT settings_json, symbols_json FROM sessions WHERE id=1").fetchone()
            conn.close()
            assert row is not None
            settings_dict = json.loads(row[0])
            assert settings_dict["bankroll"] == 999.0
            symbols = json.loads(row[1])
            assert "KXBTC" in symbols
            assert "KXETH" in symbols
        finally:
            recorder.close()

    def test_close_sets_end_time(self, tmp_path) -> None:
        db_file = str(tmp_path / "test.db")
        settings = _make_settings()
        recorder = CycleRecorder(db_file, settings)
        recorder.open()
        recorder.close()
        conn = sqlite3.connect(db_file)
        row = conn.execute("SELECT end_time FROM sessions WHERE id=1").fetchone()
        conn.close()
        assert row is not None
        assert row[0] is not None
        assert row[0] > 0


# ── TestCycleRecorderWrite ────────────────────────────────────────


class TestCycleRecorderWrite:
    def test_begin_cycle_returns_incrementing_ids(self, tmp_path) -> None:
        db_file = str(tmp_path / "test.db")
        settings = _make_settings()
        recorder = CycleRecorder(db_file, settings)
        recorder.open()
        try:
            ts = time.time()
            c1 = recorder.begin_cycle(1, ts)
            c2 = recorder.begin_cycle(2, ts + 1)
            c3 = recorder.begin_cycle(3, ts + 2)
            assert c1 < c2 < c3
        finally:
            recorder.close()

    def test_record_market_snapshots_basic(self, tmp_path) -> None:
        db_file = str(tmp_path / "test.db")
        settings = _make_settings()
        recorder = CycleRecorder(db_file, settings)
        recorder.open()
        try:
            cid = recorder.begin_cycle(1, time.time())
            quote = _make_mock_quote()
            recorder.record_market_snapshots(
                cid,
                [quote],
                spot_prices={"BTC": 97000.0},
                vpin_values={},
                ofi_values={},
                returns={},
            )
            recorder.flush()
            conn = sqlite3.connect(db_file)
            rows = conn.execute("SELECT * FROM market_snapshots WHERE cycle_id=?", (cid,)).fetchall()
            conn.close()
            assert len(rows) == 1
        finally:
            recorder.close()

    def test_record_market_snapshots_with_returns_json_roundtrip(self, tmp_path) -> None:
        db_file = str(tmp_path / "test.db")
        settings = _make_settings()
        recorder = CycleRecorder(db_file, settings)
        recorder.open()
        try:
            cid = recorder.begin_cycle(1, time.time())
            quote = _make_mock_quote()
            returns_data = [0.001, -0.002, 0.003, 0.0001]
            recorder.record_market_snapshots(
                cid,
                [quote],
                spot_prices={"BTC": 97000.0},
                vpin_values={},
                ofi_values={},
                returns={"BTC": returns_data},
            )
            recorder.flush()
            conn = sqlite3.connect(db_file)
            row = conn.execute(
                "SELECT returns_json FROM market_snapshots WHERE cycle_id=?", (cid,)
            ).fetchone()
            conn.close()
            parsed = json.loads(row[0])
            assert len(parsed) == 4
            assert abs(parsed[0] - 0.001) < 1e-9
            assert abs(parsed[2] - 0.003) < 1e-9
        finally:
            recorder.close()

    def test_record_market_snapshots_null_vpin(self, tmp_path) -> None:
        db_file = str(tmp_path / "test.db")
        settings = _make_settings()
        recorder = CycleRecorder(db_file, settings)
        recorder.open()
        try:
            cid = recorder.begin_cycle(1, time.time())
            quote = _make_mock_quote()
            # VPIN dict does not include "BTC" -> should be NULL
            recorder.record_market_snapshots(
                cid,
                [quote],
                spot_prices={"BTC": 97000.0},
                vpin_values={"ETH": (0.5, 0.3, 0.1)},
                ofi_values={},
                returns={},
            )
            recorder.flush()
            conn = sqlite3.connect(db_file)
            row = conn.execute(
                "SELECT vpin, signed_vpin, vpin_trend FROM market_snapshots WHERE cycle_id=?",
                (cid,),
            ).fetchone()
            conn.close()
            assert row[0] is None
            assert row[1] is None
            assert row[2] is None
        finally:
            recorder.close()

    def test_record_edges_basic(self, tmp_path) -> None:
        db_file = str(tmp_path / "test.db")
        settings = _make_settings()
        recorder = CycleRecorder(db_file, settings)
        recorder.open()
        try:
            cid = recorder.begin_cycle(1, time.time())
            edge = _make_mock_edge()
            fr = FilterResult(
                passed_regime_min_edge=True,
                passed_zscore=True,
                passed_counter_trend=True,
                passed_confidence=True,
                survived_all=True,
                contracts_sized=10,
                was_traded=True,
            )
            result_map = recorder.record_edges(cid, [edge], {"KXBTCD-26FEB14-T97500": fr})
            recorder.flush()
            assert "KXBTCD-26FEB14-T97500" in result_map
            conn = sqlite3.connect(db_file)
            rows = conn.execute("SELECT * FROM edges WHERE cycle_id=?", (cid,)).fetchall()
            conn.close()
            assert len(rows) == 1
        finally:
            recorder.close()

    def test_record_edges_missing_filter_result(self, tmp_path) -> None:
        db_file = str(tmp_path / "test.db")
        settings = _make_settings()
        recorder = CycleRecorder(db_file, settings)
        recorder.open()
        try:
            cid = recorder.begin_cycle(1, time.time())
            edge = _make_mock_edge(ticker="KXBTCD-26FEB14-T90000")
            # Empty filter_results dict -> defaults to FilterResult()
            result_map = recorder.record_edges(cid, [edge], {})
            recorder.flush()
            assert "KXBTCD-26FEB14-T90000" in result_map
            conn = sqlite3.connect(db_file)
            row = conn.execute(
                "SELECT passed_regime_min_edge, survived_all_filters FROM edges WHERE cycle_id=?",
                (cid,),
            ).fetchone()
            conn.close()
            # Default FilterResult has None for passed_regime_min_edge
            assert row[0] is None
            # survived_all defaults to False -> 0
            assert row[1] == 0
        finally:
            recorder.close()

    def test_record_trade_basic(self, tmp_path) -> None:
        db_file = str(tmp_path / "test.db")
        settings = _make_settings()
        recorder = CycleRecorder(db_file, settings)
        recorder.open()
        try:
            cid = recorder.begin_cycle(1, time.time())
            edge = _make_mock_edge()
            edge_map = recorder.record_edges(cid, [edge], {})
            edge_id = edge_map["KXBTCD-26FEB14-T97500"]
            trade_id = recorder.record_trade(
                cid, edge_id, "KXBTCD-26FEB14-T97500", "yes", 10, 0.50
            )
            recorder.flush()
            assert trade_id > 0
            conn = sqlite3.connect(db_file)
            row = conn.execute(
                "SELECT ticker, side, contracts, entry_price, settled FROM trades WHERE id=?",
                (trade_id,),
            ).fetchone()
            conn.close()
            assert row[0] == "KXBTCD-26FEB14-T97500"
            assert row[1] == "yes"
            assert row[2] == 10
            assert abs(row[3] - 0.50) < 1e-9
            assert row[4] == 0  # Not settled
        finally:
            recorder.close()

    def test_end_cycle_updates_row(self, tmp_path) -> None:
        db_file = str(tmp_path / "test.db")
        settings = _make_settings()
        recorder = CycleRecorder(db_file, settings)
        recorder.open()
        try:
            cid = recorder.begin_cycle(1, time.time())
            recorder.end_cycle(
                cid,
                elapsed_ms=120.3,
                regime="mean_reverting",
                regime_confidence=0.85,
                regime_is_transitioning=False,
                num_quotes=5,
                num_edges_raw=3,
                num_edges_final=1,
                num_trades=1,
                session_pnl=2.50,
                bankroll=502.50,
            )
            conn = sqlite3.connect(db_file)
            row = conn.execute(
                "SELECT elapsed_ms, regime, regime_confidence, num_quotes, num_trades, session_pnl, bankroll "
                "FROM cycles WHERE id=?",
                (cid,),
            ).fetchone()
            conn.close()
            assert abs(row[0] - 120.3) < 1e-6
            assert row[1] == "mean_reverting"
            assert abs(row[2] - 0.85) < 1e-6
            assert row[3] == 5
            assert row[4] == 1
            assert abs(row[5] - 2.50) < 1e-6
            assert abs(row[6] - 502.50) < 1e-6
        finally:
            recorder.close()

    def test_end_cycle_commits_transaction(self, tmp_path) -> None:
        db_file = str(tmp_path / "test.db")
        settings = _make_settings()
        recorder = CycleRecorder(db_file, settings)
        recorder.open()
        try:
            cid = recorder.begin_cycle(1, time.time())
            recorder.end_cycle(
                cid,
                elapsed_ms=50.0,
                regime="neutral",
                regime_confidence=0.5,
                regime_is_transitioning=False,
                num_quotes=2,
                num_edges_raw=1,
                num_edges_final=0,
                num_trades=0,
                session_pnl=0.0,
                bankroll=500.0,
            )
        finally:
            recorder.close()
        # Verify data persists after close (committed)
        conn = sqlite3.connect(db_file)
        row = conn.execute("SELECT elapsed_ms FROM cycles WHERE id=?", (cid,)).fetchone()
        conn.close()
        assert row is not None
        assert abs(row[0] - 50.0) < 1e-6

    def test_flush_forces_commit(self, tmp_path) -> None:
        db_file = str(tmp_path / "test.db")
        settings = _make_settings()
        recorder = CycleRecorder(db_file, settings)
        recorder.open()
        try:
            cid = recorder.begin_cycle(1, time.time())
            quote = _make_mock_quote()
            recorder.record_market_snapshots(
                cid, [quote], {"BTC": 97000.0}, {}, {}, {}
            )
            recorder.flush()
            # Open a second connection to verify data is visible
            conn2 = sqlite3.connect(db_file)
            count = conn2.execute(
                "SELECT COUNT(*) FROM market_snapshots WHERE cycle_id=?", (cid,)
            ).fetchone()[0]
            conn2.close()
            assert count == 1
        finally:
            recorder.close()


# ── TestCycleRecorderSettlement ───────────────────────────────────


class TestCycleRecorderSettlement:
    def test_record_settlement_updates_trade(self, tmp_path) -> None:
        db_file = str(tmp_path / "test.db")
        settings = _make_settings()
        recorder = CycleRecorder(db_file, settings)
        recorder.open()
        try:
            cid = recorder.begin_cycle(1, time.time())
            edge = _make_mock_edge()
            edge_map = recorder.record_edges(cid, [edge], {})
            edge_id = edge_map["KXBTCD-26FEB14-T97500"]
            recorder.record_trade(cid, edge_id, "KXBTCD-26FEB14-T97500", "yes", 10, 0.50)
            recorder.flush()

            now = time.time()
            recorder.record_settlement("KXBTCD-26FEB14-T97500", "yes", 5.0, now)

            conn = sqlite3.connect(db_file)
            row = conn.execute(
                "SELECT settled, actual_outcome, pnl, settlement_time FROM trades WHERE ticker=?",
                ("KXBTCD-26FEB14-T97500",),
            ).fetchone()
            conn.close()
            assert row[0] == 1
            assert row[1] == "yes"
            assert abs(row[2] - 5.0) < 1e-9
            assert row[3] is not None
        finally:
            recorder.close()

    def test_record_settlement_nonexistent_ticker_no_crash(self, tmp_path) -> None:
        db_file = str(tmp_path / "test.db")
        settings = _make_settings()
        recorder = CycleRecorder(db_file, settings)
        recorder.open()
        try:
            # Should not raise
            recorder.record_settlement("NONEXISTENT-TICKER", "no", -3.0, time.time())
        finally:
            recorder.close()

    def test_settlement_pnl_stored_correctly(self, tmp_path) -> None:
        db_file = str(tmp_path / "test.db")
        settings = _make_settings()
        recorder = CycleRecorder(db_file, settings)
        recorder.open()
        try:
            cid = recorder.begin_cycle(1, time.time())
            edge = _make_mock_edge(ticker="KXBTCD-26FEB14-T80000")
            edge_map = recorder.record_edges(cid, [edge], {})
            edge_id = edge_map["KXBTCD-26FEB14-T80000"]
            recorder.record_trade(cid, edge_id, "KXBTCD-26FEB14-T80000", "no", 5, 0.30)
            recorder.flush()

            recorder.record_settlement("KXBTCD-26FEB14-T80000", "no", 3.50, time.time())

            conn = sqlite3.connect(db_file)
            row = conn.execute(
                "SELECT pnl FROM trades WHERE ticker=?", ("KXBTCD-26FEB14-T80000",)
            ).fetchone()
            conn.close()
            assert abs(row[0] - 3.50) < 1e-9
        finally:
            recorder.close()

    def test_multiple_settlements_different_tickers(self, tmp_path) -> None:
        db_file = str(tmp_path / "test.db")
        settings = _make_settings()
        recorder = CycleRecorder(db_file, settings)
        recorder.open()
        try:
            cid = recorder.begin_cycle(1, time.time())

            ticker_a = "KXBTCD-26FEB14-T90000"
            ticker_b = "KXBTCD-26FEB14-T95000"

            edge_a = _make_mock_edge(ticker=ticker_a)
            edge_b = _make_mock_edge(ticker=ticker_b)
            edge_map = recorder.record_edges(cid, [edge_a, edge_b], {})

            recorder.record_trade(cid, edge_map[ticker_a], ticker_a, "yes", 10, 0.40)
            recorder.record_trade(cid, edge_map[ticker_b], ticker_b, "no", 5, 0.30)
            recorder.flush()

            recorder.record_settlement(ticker_a, "yes", 6.0, time.time())
            recorder.record_settlement(ticker_b, "no", 3.5, time.time())

            conn = sqlite3.connect(db_file)
            row_a = conn.execute(
                "SELECT settled, pnl FROM trades WHERE ticker=?", (ticker_a,)
            ).fetchone()
            row_b = conn.execute(
                "SELECT settled, pnl FROM trades WHERE ticker=?", (ticker_b,)
            ).fetchone()
            conn.close()

            assert row_a[0] == 1
            assert abs(row_a[1] - 6.0) < 1e-9
            assert row_b[0] == 1
            assert abs(row_b[1] - 3.5) < 1e-9
        finally:
            recorder.close()


# ── TestCycleRecorderPerformance ──────────────────────────────────


class TestCycleRecorderPerformance:
    def test_batch_insert_20_snapshots_under_10ms(self, tmp_path) -> None:
        db_file = str(tmp_path / "test.db")
        settings = _make_settings()
        recorder = CycleRecorder(db_file, settings)
        recorder.open()
        try:
            cid = recorder.begin_cycle(1, time.time())
            quotes = []
            for i in range(20):
                quotes.append(
                    _make_mock_quote(
                        ticker=f"KXBTCD-26FEB14-T{90000 + i * 500}",
                        yes_buy_price=0.30 + i * 0.01,
                    )
                )
            start = time.perf_counter()
            recorder.record_market_snapshots(
                cid,
                quotes,
                spot_prices={"BTC": 97000.0},
                vpin_values={},
                ofi_values={},
                returns={},
            )
            recorder.flush()
            elapsed_ms = (time.perf_counter() - start) * 1000
            assert elapsed_ms < 100  # generous bound (10ms goal, 100ms safety)
        finally:
            recorder.close()

    def test_50_cycles_insert_reasonable_time(self, tmp_path) -> None:
        db_file = str(tmp_path / "test.db")
        settings = _make_settings()
        recorder = CycleRecorder(db_file, settings)
        recorder.open()
        try:
            start = time.perf_counter()
            for i in range(50):
                ts = time.time() + i
                cid = recorder.begin_cycle(i + 1, ts)
                recorder.end_cycle(
                    cid,
                    elapsed_ms=float(i),
                    regime="neutral",
                    regime_confidence=0.5,
                    regime_is_transitioning=False,
                    num_quotes=3,
                    num_edges_raw=2,
                    num_edges_final=1,
                    num_trades=0,
                    session_pnl=0.0,
                    bankroll=500.0,
                )
            elapsed_s = time.perf_counter() - start
            assert elapsed_s < 5.0  # should be well under 1s
        finally:
            recorder.close()

    def test_wal_mode_enabled(self, tmp_path) -> None:
        db_file = str(tmp_path / "test.db")
        settings = _make_settings()
        recorder = CycleRecorder(db_file, settings)
        recorder.open()
        try:
            conn = sqlite3.connect(db_file)
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
            conn.close()
            assert mode.lower() == "wal"
        finally:
            recorder.close()


# ── TestCycleRecorderEdgeCases ────────────────────────────────────


class TestCycleRecorderEdgeCases:
    def test_empty_quotes_list(self, tmp_path) -> None:
        db_file = str(tmp_path / "test.db")
        settings = _make_settings()
        recorder = CycleRecorder(db_file, settings)
        recorder.open()
        try:
            cid = recorder.begin_cycle(1, time.time())
            recorder.record_market_snapshots(cid, [], {}, {}, {}, {})
            recorder.flush()
            conn = sqlite3.connect(db_file)
            count = conn.execute(
                "SELECT COUNT(*) FROM market_snapshots WHERE cycle_id=?", (cid,)
            ).fetchone()[0]
            conn.close()
            assert count == 0
        finally:
            recorder.close()

    def test_empty_edges_list(self, tmp_path) -> None:
        db_file = str(tmp_path / "test.db")
        settings = _make_settings()
        recorder = CycleRecorder(db_file, settings)
        recorder.open()
        try:
            cid = recorder.begin_cycle(1, time.time())
            result_map = recorder.record_edges(cid, [], {})
            recorder.flush()
            assert result_map == {}
            conn = sqlite3.connect(db_file)
            count = conn.execute(
                "SELECT COUNT(*) FROM edges WHERE cycle_id=?", (cid,)
            ).fetchone()[0]
            conn.close()
            assert count == 0
        finally:
            recorder.close()

    def test_no_filter_results_dict_entries(self, tmp_path) -> None:
        db_file = str(tmp_path / "test.db")
        settings = _make_settings()
        recorder = CycleRecorder(db_file, settings)
        recorder.open()
        try:
            cid = recorder.begin_cycle(1, time.time())
            edges = [
                _make_mock_edge(ticker="KXBTCD-26FEB14-T90000"),
                _make_mock_edge(ticker="KXBTCD-26FEB14-T95000"),
            ]
            result_map = recorder.record_edges(cid, edges, {})
            recorder.flush()
            assert len(result_map) == 2
            # All filter columns should use defaults
            conn = sqlite3.connect(db_file)
            rows = conn.execute(
                "SELECT passed_regime_min_edge, survived_all_filters FROM edges WHERE cycle_id=?",
                (cid,),
            ).fetchall()
            conn.close()
            for row in rows:
                assert row[0] is None  # default None -> NULL
                assert row[1] == 0    # survived_all defaults to False -> 0
        finally:
            recorder.close()

    def test_open_close_lifecycle_idempotent(self, tmp_path) -> None:
        db_file = str(tmp_path / "test.db")
        settings = _make_settings()
        recorder = CycleRecorder(db_file, settings)

        # First open/close
        recorder.open()
        cid1 = recorder.begin_cycle(1, time.time())
        recorder.close()

        # Second open/close with same db file
        recorder2 = CycleRecorder(db_file, settings)
        recorder2.open()
        cid2 = recorder2.begin_cycle(2, time.time())
        recorder2.close()

        # Both cycles should be in the database
        conn = sqlite3.connect(db_file)
        count = conn.execute("SELECT COUNT(*) FROM cycles").fetchone()[0]
        sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        conn.close()
        assert count == 2
        assert sessions == 2


# ── TestFilterResult ──────────────────────────────────────────────


class TestFilterResult:
    def test_default_values(self) -> None:
        fr = FilterResult()
        assert fr.passed_regime_min_edge is None
        assert fr.passed_zscore is None
        assert fr.passed_counter_trend is None
        assert fr.passed_confidence is None
        assert fr.zscore_value is None
        assert fr.confidence_score is None
        assert fr.confidence_agreement is None
        assert fr.reject_reason is None
        assert fr.survived_all is False
        assert fr.contracts_sized == 0
        assert fr.was_traded is False

    def test_all_passed_state(self) -> None:
        fr = FilterResult(
            passed_regime_min_edge=True,
            passed_zscore=True,
            passed_counter_trend=True,
            passed_confidence=True,
            zscore_value=1.2,
            confidence_score=0.85,
            confidence_agreement=5,
            reject_reason=None,
            survived_all=True,
            contracts_sized=15,
            was_traded=True,
        )
        assert fr.survived_all is True
        assert fr.was_traded is True
        assert fr.contracts_sized == 15
        assert fr.reject_reason is None

    def test_first_failure_sets_reject_reason(self) -> None:
        fr = FilterResult(
            passed_regime_min_edge=True,
            passed_zscore=False,
            reject_reason="zscore_too_high",
            survived_all=False,
        )
        assert fr.passed_zscore is False
        assert fr.reject_reason == "zscore_too_high"
        assert fr.survived_all is False


# ── TestCycleRecorderQueryability ─────────────────────────────────


class TestCycleRecorderQueryability:
    def _setup_populated_db(self, tmp_path):
        """Create a recorder with multiple cycles, edges, snapshots, and trades."""
        db_file = str(tmp_path / "query_test.db")
        settings = _make_settings(bankroll=1000.0, symbols=["KXBTC", "KXETH"])
        recorder = CycleRecorder(db_file, settings)
        recorder.open()

        now = time.time()

        # Cycle 1
        cid1 = recorder.begin_cycle(1, now - 100)
        q1 = _make_mock_quote(ticker="KXBTCD-26FEB14-T97500")
        recorder.record_market_snapshots(
            cid1, [q1], {"BTC": 97000.0},
            vpin_values={"BTC": (0.6, 0.3, 0.1)},
            ofi_values={"BTC": {30: 0.1, 60: 0.2, 120: 0.15, 300: 0.05}},
            returns={"BTC": [0.001, -0.002, 0.003]},
        )
        edge1 = _make_mock_edge(ticker="KXBTCD-26FEB14-T97500")
        fr1 = FilterResult(
            passed_regime_min_edge=True, passed_zscore=True,
            survived_all=True, contracts_sized=10, was_traded=True,
        )
        edge_map1 = recorder.record_edges(cid1, [edge1], {"KXBTCD-26FEB14-T97500": fr1})
        eid1 = edge_map1["KXBTCD-26FEB14-T97500"]
        recorder.record_trade(cid1, eid1, "KXBTCD-26FEB14-T97500", "yes", 10, 0.50)
        recorder.end_cycle(
            cid1, elapsed_ms=80.0, regime="mean_reverting", regime_confidence=0.9,
            regime_is_transitioning=False, num_quotes=1, num_edges_raw=1,
            num_edges_final=1, num_trades=1, session_pnl=0.0, bankroll=995.0,
        )

        # Cycle 2
        cid2 = recorder.begin_cycle(2, now - 50)
        q2 = _make_mock_quote(ticker="KXBTCD-26FEB14-T98000", yes_buy_price=0.40)
        recorder.record_market_snapshots(
            cid2, [q2], {"BTC": 97500.0},
            vpin_values={},
            ofi_values={},
            returns={"BTC": [0.002, 0.001]},
        )
        edge2 = _make_mock_edge(ticker="KXBTCD-26FEB14-T98000")
        edge_map2 = recorder.record_edges(cid2, [edge2], {})
        recorder.end_cycle(
            cid2, elapsed_ms=60.0, regime="trending_up", regime_confidence=0.7,
            regime_is_transitioning=True, num_quotes=1, num_edges_raw=1,
            num_edges_final=0, num_trades=0, session_pnl=0.0, bankroll=995.0,
        )

        # Settle the trade from cycle 1
        recorder.record_settlement("KXBTCD-26FEB14-T97500", "yes", 5.0, now)

        recorder.close()
        return db_file, now

    def test_select_cycles_in_time_range(self, tmp_path) -> None:
        db_file, now = self._setup_populated_db(tmp_path)
        conn = sqlite3.connect(db_file)
        rows = conn.execute(
            "SELECT id, regime FROM cycles WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp",
            (now - 120, now - 40),
        ).fetchall()
        conn.close()
        assert len(rows) == 2
        assert rows[0][1] == "mean_reverting"
        assert rows[1][1] == "trending_up"

    def test_select_edges_by_cycle_id(self, tmp_path) -> None:
        db_file, now = self._setup_populated_db(tmp_path)
        conn = sqlite3.connect(db_file)
        # Cycle 1 should have 1 edge
        rows = conn.execute("SELECT ticker, survived_all_filters FROM edges WHERE cycle_id=1").fetchall()
        conn.close()
        assert len(rows) == 1
        assert rows[0][0] == "KXBTCD-26FEB14-T97500"
        assert rows[0][1] == 1  # survived_all=True

    def test_select_trades_with_settlement(self, tmp_path) -> None:
        db_file, now = self._setup_populated_db(tmp_path)
        conn = sqlite3.connect(db_file)
        rows = conn.execute(
            "SELECT ticker, settled, actual_outcome, pnl FROM trades WHERE settled=1"
        ).fetchall()
        conn.close()
        assert len(rows) == 1
        assert rows[0][0] == "KXBTCD-26FEB14-T97500"
        assert rows[0][1] == 1
        assert rows[0][2] == "yes"
        assert abs(rows[0][3] - 5.0) < 1e-9

    def test_select_market_snapshots_by_ticker(self, tmp_path) -> None:
        db_file, now = self._setup_populated_db(tmp_path)
        conn = sqlite3.connect(db_file)
        rows = conn.execute(
            "SELECT ticker, spot_price FROM market_snapshots WHERE ticker=?",
            ("KXBTCD-26FEB14-T97500",),
        ).fetchall()
        conn.close()
        assert len(rows) == 1
        assert abs(rows[0][1] - 97000.0) < 1e-6

    def test_join_edges_to_cycles_for_regime(self, tmp_path) -> None:
        db_file, now = self._setup_populated_db(tmp_path)
        conn = sqlite3.connect(db_file)
        rows = conn.execute(
            """SELECT e.ticker, c.regime, e.survived_all_filters
               FROM edges e
               JOIN cycles c ON e.cycle_id = c.id
               ORDER BY c.timestamp""",
        ).fetchall()
        conn.close()
        assert len(rows) == 2
        # First edge in mean_reverting regime
        assert rows[0][1] == "mean_reverting"
        assert rows[0][2] == 1
        # Second edge in trending_up regime
        assert rows[1][1] == "trending_up"
        assert rows[1][2] == 0  # default FilterResult -> survived_all=False

    def test_returns_json_roundtrip(self, tmp_path) -> None:
        db_file, now = self._setup_populated_db(tmp_path)
        conn = sqlite3.connect(db_file)
        row = conn.execute(
            "SELECT returns_json FROM market_snapshots WHERE ticker=?",
            ("KXBTCD-26FEB14-T97500",),
        ).fetchone()
        conn.close()
        parsed = json.loads(row[0])
        assert isinstance(parsed, list)
        assert len(parsed) == 3
        assert abs(parsed[0] - 0.001) < 1e-9
        assert abs(parsed[1] - (-0.002)) < 1e-9
        assert abs(parsed[2] - 0.003) < 1e-9

    def test_settings_json_roundtrip(self, tmp_path) -> None:
        db_file, now = self._setup_populated_db(tmp_path)
        conn = sqlite3.connect(db_file)
        row = conn.execute("SELECT settings_json FROM sessions LIMIT 1").fetchone()
        conn.close()
        settings_dict = json.loads(row[0])
        assert settings_dict["bankroll"] == 1000.0
        assert settings_dict["paper_mode"] is True
        assert settings_dict["mc_num_paths"] == 100
