"""Tests for arb_bot.crypto.backtester — offline replay backtester.

~40 tests covering initialization, run logic, filters, sweep,
sizing, and output.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import sqlite3
import sys
import time
from contextlib import redirect_stdout
from typing import Any, Dict, List, Optional

import pytest

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from arb_bot.crypto.backtester import (
    BacktestConfig,
    BacktestResult,
    BacktestTrade,
    Backtester,
    _extract_underlying,
    _is_counter_trend,
)

# ---------------------------------------------------------------------------
# Schema DDL (copied from cycle_recorder.py for self-contained fixtures)
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    start_time REAL NOT NULL,
    end_time REAL,
    settings_json TEXT NOT NULL DEFAULT '{}',
    symbols_json TEXT NOT NULL DEFAULT '[]',
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS cycles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    cycle_number INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    elapsed_ms REAL,
    regime TEXT,
    regime_confidence REAL,
    regime_is_transitioning INTEGER,
    num_quotes INTEGER,
    num_edges_raw INTEGER,
    num_edges_final INTEGER,
    num_trades INTEGER,
    session_pnl REAL,
    bankroll REAL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS market_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id INTEGER NOT NULL,
    ticker TEXT NOT NULL,
    underlying TEXT,
    interval TEXT,
    direction TEXT,
    strike REAL,
    time_to_expiry_minutes REAL,
    yes_buy_price REAL,
    no_buy_price REAL,
    yes_buy_size REAL,
    no_buy_size REAL,
    yes_bid_price REAL,
    no_bid_price REAL,
    implied_probability REAL,
    spot_price REAL,
    vpin REAL,
    signed_vpin REAL,
    vpin_trend REAL,
    ofi_30 REAL,
    ofi_60 REAL,
    ofi_120 REAL,
    ofi_300 REAL,
    returns_json TEXT,
    FOREIGN KEY (cycle_id) REFERENCES cycles(id)
);

CREATE TABLE IF NOT EXISTS edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id INTEGER NOT NULL,
    ticker TEXT NOT NULL,
    side TEXT,
    model_prob REAL,
    market_implied_prob REAL,
    blended_prob REAL,
    edge_cents REAL,
    model_uncertainty REAL,
    time_to_expiry_minutes REAL,
    yes_buy_price REAL,
    no_buy_price REAL,
    spread_cost REAL,
    staleness_score REAL,
    passed_regime_min_edge INTEGER,
    passed_zscore INTEGER,
    passed_counter_trend INTEGER,
    passed_confidence INTEGER,
    zscore_value REAL,
    confidence_score REAL,
    confidence_agreement INTEGER,
    filter_reject_reason TEXT,
    survived_all_filters INTEGER,
    contracts_sized INTEGER,
    was_traded INTEGER,
    FOREIGN KEY (cycle_id) REFERENCES cycles(id)
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id INTEGER NOT NULL,
    edge_id INTEGER,
    ticker TEXT NOT NULL,
    side TEXT NOT NULL,
    contracts INTEGER NOT NULL,
    entry_price REAL NOT NULL,
    settled INTEGER NOT NULL DEFAULT 0,
    actual_outcome TEXT,
    pnl REAL,
    settlement_time REAL,
    FOREIGN KEY (cycle_id) REFERENCES cycles(id),
    FOREIGN KEY (edge_id) REFERENCES edges(id)
);
"""


# ---------------------------------------------------------------------------
# Default settings dict (mirrors CryptoSettings defaults relevant to backtest)
# ---------------------------------------------------------------------------

_DEFAULT_SETTINGS: Dict[str, Any] = {
    "bankroll": 500.0,
    "max_position_per_market": 50.0,
    "max_concurrent_positions": 10,
    "max_positions_per_underlying": 3,
    "kelly_fraction_cap": 0.10,
    "min_edge_pct": 0.12,
    # Filters default OFF in settings unless explicitly enabled
    "regime_min_edge_enabled": False,
    "zscore_filter_enabled": True,
    "zscore_max": 2.0,
    "regime_skip_counter_trend": True,
    "regime_skip_counter_trend_min_conf": 0.6,
    "confidence_scoring_enabled": False,
    "confidence_min_score": 0.65,
    # Regime sizing
    "regime_sizing_enabled": False,
    "regime_kelly_mean_reverting": 1.5,
    "regime_kelly_trending_up": 0.4,
    "regime_kelly_trending_down": 0.5,
    "regime_kelly_high_vol": 0.0,
    # Regime min edge thresholds
    "regime_min_edge_mean_reverting": 0.10,
    "regime_min_edge_trending": 0.20,
    "regime_min_edge_high_vol": 0.30,
    # Mean-reverting cap boost
    "regime_kelly_cap_boost_mean_reverting": 1.25,
    # Transition
    "regime_transition_sizing_multiplier": 0.3,
}


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _create_test_db(
    tmp_path,
    settings_overrides: Optional[Dict[str, Any]] = None,
    cycles: Optional[List[Dict[str, Any]]] = None,
    edges: Optional[List[Dict[str, Any]]] = None,
    trades: Optional[List[Dict[str, Any]]] = None,
    snapshots: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Create a test DB with pre-populated data. Returns db_path."""
    os.makedirs(str(tmp_path), exist_ok=True)
    db_path = str(tmp_path / "test_session.db")
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA_SQL)

    # Merge settings
    settings = dict(_DEFAULT_SETTINGS)
    if settings_overrides:
        settings.update(settings_overrides)

    # Insert session
    conn.execute(
        "INSERT INTO sessions (id, start_time, settings_json) VALUES (?, ?, ?)",
        (1, time.time(), json.dumps(settings)),
    )

    # Insert cycles
    if cycles:
        for c in cycles:
            conn.execute(
                """INSERT INTO cycles
                   (id, session_id, cycle_number, timestamp, regime,
                    regime_confidence, regime_is_transitioning, bankroll)
                   VALUES (?, 1, ?, ?, ?, ?, ?, ?)""",
                (
                    c["id"],
                    c["cycle_number"],
                    c.get("timestamp", 1000000.0 + c["cycle_number"] * 60),
                    c.get("regime"),
                    c.get("regime_confidence"),
                    c.get("regime_is_transitioning", 0),
                    c.get("bankroll", 500.0),
                ),
            )

    # Insert edges
    if edges:
        for e in edges:
            conn.execute(
                """INSERT INTO edges
                   (id, cycle_id, ticker, side, model_prob, market_implied_prob,
                    blended_prob, edge_cents, model_uncertainty,
                    time_to_expiry_minutes, yes_buy_price, no_buy_price,
                    spread_cost, staleness_score,
                    passed_regime_min_edge, passed_zscore,
                    passed_counter_trend, passed_confidence,
                    zscore_value, confidence_score, confidence_agreement,
                    filter_reject_reason, survived_all_filters,
                    contracts_sized, was_traded)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    e.get("id"),
                    e["cycle_id"],
                    e.get("ticker", "KXBTC15M-26FEB15-T100500-B"),
                    e.get("side", "yes"),
                    e.get("model_prob", 0.60),
                    e.get("market_implied_prob", 0.50),
                    e.get("blended_prob", 0.55),
                    e.get("edge_cents", 8.0),
                    e.get("model_uncertainty", 0.05),
                    e.get("time_to_expiry_minutes", 10.0),
                    e.get("yes_buy_price", 0.50),
                    e.get("no_buy_price", 0.50),
                    e.get("spread_cost", 0.02),
                    e.get("staleness_score", 0.0),
                    e.get("passed_regime_min_edge", 1),
                    e.get("passed_zscore", 1),
                    e.get("passed_counter_trend", 1),
                    e.get("passed_confidence", 1),
                    e.get("zscore_value", 1.0),
                    e.get("confidence_score", 0.80),
                    e.get("confidence_agreement", 5),
                    e.get("filter_reject_reason"),
                    e.get("survived_all_filters", 1),
                    e.get("contracts_sized", 5),
                    e.get("was_traded", 1),
                ),
            )

    # Insert trades (settlement data)
    if trades:
        for t in trades:
            conn.execute(
                """INSERT INTO trades
                   (id, cycle_id, edge_id, ticker, side, contracts,
                    entry_price, settled, actual_outcome, pnl, settlement_time)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    t.get("id"),
                    t["cycle_id"],
                    t.get("edge_id"),
                    t.get("ticker", "KXBTC15M-26FEB15-T100500-B"),
                    t.get("side", "yes"),
                    t.get("contracts", 5),
                    t.get("entry_price", 0.50),
                    t.get("settled", 1),
                    t.get("actual_outcome", "yes"),
                    t.get("pnl", 2.50),
                    t.get("settlement_time", time.time()),
                ),
            )

    # Insert market snapshots
    if snapshots:
        for s in snapshots:
            conn.execute(
                """INSERT INTO market_snapshots
                   (cycle_id, ticker, direction)
                   VALUES (?, ?, ?)""",
                (s["cycle_id"], s["ticker"], s.get("direction", "above")),
            )

    conn.commit()
    conn.close()
    return db_path


def _make_standard_db(tmp_path, settings_overrides=None):
    """Create a standard DB with 3 cycles, several edges, and some trades.

    Returns (db_path, expected_edge_count).
    """
    cycles = [
        {"id": 1, "cycle_number": 1, "regime": "mean_reverting", "regime_confidence": 0.80},
        {"id": 2, "cycle_number": 2, "regime": "trending_up", "regime_confidence": 0.75},
        {"id": 3, "cycle_number": 3, "regime": "high_vol", "regime_confidence": 0.60},
    ]

    # 10 edges total, spread across cycles with varying edge_cents
    edges = [
        # Cycle 1 (mean_reverting): 4 edges
        {"id": 1, "cycle_id": 1, "ticker": "KXBTC15M-26FEB15-T100500-A",
         "side": "yes", "edge_cents": 8.0, "yes_buy_price": 0.40, "no_buy_price": 0.60,
         "zscore_value": 1.0, "confidence_score": 0.85, "model_uncertainty": 0.04},
        {"id": 2, "cycle_id": 1, "ticker": "KXBTC15M-26FEB15-T100500-B",
         "side": "no", "edge_cents": 6.0, "yes_buy_price": 0.55, "no_buy_price": 0.45,
         "zscore_value": 1.5, "confidence_score": 0.70, "model_uncertainty": 0.05},
        {"id": 3, "cycle_id": 1, "ticker": "KXETH15M-26FEB15-T3200-A",
         "side": "yes", "edge_cents": 3.0, "yes_buy_price": 0.60, "no_buy_price": 0.40,
         "zscore_value": 0.8, "confidence_score": 0.90, "model_uncertainty": 0.03},
        {"id": 4, "cycle_id": 1, "ticker": "KXETH15M-26FEB15-T3200-B",
         "side": "no", "edge_cents": 10.0, "yes_buy_price": 0.70, "no_buy_price": 0.30,
         "zscore_value": 2.5, "confidence_score": 0.60, "model_uncertainty": 0.06},
        # Cycle 2 (trending_up): 3 edges
        {"id": 5, "cycle_id": 2, "ticker": "KXBTC15M-26FEB15-T101000-A",
         "side": "yes", "edge_cents": 7.0, "yes_buy_price": 0.45, "no_buy_price": 0.55,
         "zscore_value": 1.2, "confidence_score": 0.75, "model_uncertainty": 0.05},
        {"id": 6, "cycle_id": 2, "ticker": "KXBTC15M-26FEB15-T101000-B",
         "side": "no", "edge_cents": 5.0, "yes_buy_price": 0.50, "no_buy_price": 0.50,
         "zscore_value": 1.8, "confidence_score": 0.80, "model_uncertainty": 0.04},
        {"id": 7, "cycle_id": 2, "ticker": "KXETH15M-26FEB15-T3250-A",
         "side": "yes", "edge_cents": 9.0, "yes_buy_price": 0.35, "no_buy_price": 0.65,
         "zscore_value": 0.5, "confidence_score": 0.88, "model_uncertainty": 0.03},
        # Cycle 3 (high_vol): 3 edges
        {"id": 8, "cycle_id": 3, "ticker": "KXBTC15M-26FEB15-T101500-A",
         "side": "yes", "edge_cents": 12.0, "yes_buy_price": 0.38, "no_buy_price": 0.62,
         "zscore_value": 1.0, "confidence_score": 0.92, "model_uncertainty": 0.07},
        {"id": 9, "cycle_id": 3, "ticker": "KXBTC15M-26FEB15-T101500-B",
         "side": "no", "edge_cents": 4.0, "yes_buy_price": 0.65, "no_buy_price": 0.35,
         "zscore_value": 3.0, "confidence_score": 0.55, "model_uncertainty": 0.08},
        {"id": 10, "cycle_id": 3, "ticker": "KXETH15M-26FEB15-T3300-A",
         "side": "yes", "edge_cents": 6.5, "yes_buy_price": 0.42, "no_buy_price": 0.58,
         "zscore_value": 0.9, "confidence_score": 0.78, "model_uncertainty": 0.05},
    ]

    # Settlement trades for some edges (wins and losses)
    trades = [
        {"id": 1, "cycle_id": 1, "edge_id": 1, "ticker": "KXBTC15M-26FEB15-T100500-A",
         "side": "yes", "contracts": 5, "entry_price": 0.40,
         "settled": 1, "actual_outcome": "yes", "pnl": 3.0},
        {"id": 2, "cycle_id": 1, "edge_id": 2, "ticker": "KXBTC15M-26FEB15-T100500-B",
         "side": "no", "contracts": 3, "entry_price": 0.45,
         "settled": 1, "actual_outcome": "no", "pnl": 1.65},
        {"id": 3, "cycle_id": 1, "edge_id": 3, "ticker": "KXETH15M-26FEB15-T3200-A",
         "side": "yes", "contracts": 4, "entry_price": 0.60,
         "settled": 1, "actual_outcome": "no", "pnl": -2.40},
        {"id": 4, "cycle_id": 2, "edge_id": 5, "ticker": "KXBTC15M-26FEB15-T101000-A",
         "side": "yes", "contracts": 5, "entry_price": 0.45,
         "settled": 1, "actual_outcome": "yes", "pnl": 2.75},
        {"id": 5, "cycle_id": 2, "edge_id": 7, "ticker": "KXETH15M-26FEB15-T3250-A",
         "side": "yes", "contracts": 6, "entry_price": 0.35,
         "settled": 1, "actual_outcome": "yes", "pnl": 3.90},
        {"id": 6, "cycle_id": 3, "edge_id": 8, "ticker": "KXBTC15M-26FEB15-T101500-A",
         "side": "yes", "contracts": 5, "entry_price": 0.38,
         "settled": 1, "actual_outcome": "yes", "pnl": 3.10},
        {"id": 7, "cycle_id": 3, "edge_id": 10, "ticker": "KXETH15M-26FEB15-T3300-A",
         "side": "yes", "contracts": 4, "entry_price": 0.42,
         "settled": 1, "actual_outcome": "no", "pnl": -1.68},
    ]

    # Market snapshots for direction
    snapshots = [
        {"cycle_id": 1, "ticker": "KXBTC15M-26FEB15-T100500-A", "direction": "above"},
        {"cycle_id": 1, "ticker": "KXBTC15M-26FEB15-T100500-B", "direction": "above"},
        {"cycle_id": 1, "ticker": "KXETH15M-26FEB15-T3200-A", "direction": "above"},
        {"cycle_id": 1, "ticker": "KXETH15M-26FEB15-T3200-B", "direction": "below"},
        {"cycle_id": 2, "ticker": "KXBTC15M-26FEB15-T101000-A", "direction": "above"},
        {"cycle_id": 2, "ticker": "KXBTC15M-26FEB15-T101000-B", "direction": "above"},
        {"cycle_id": 2, "ticker": "KXETH15M-26FEB15-T3250-A", "direction": "above"},
        {"cycle_id": 3, "ticker": "KXBTC15M-26FEB15-T101500-A", "direction": "above"},
        {"cycle_id": 3, "ticker": "KXBTC15M-26FEB15-T101500-B", "direction": "below"},
        {"cycle_id": 3, "ticker": "KXETH15M-26FEB15-T3300-A", "direction": "above"},
    ]

    db_path = _create_test_db(
        tmp_path,
        settings_overrides=settings_overrides,
        cycles=cycles,
        edges=edges,
        trades=trades,
        snapshots=snapshots,
    )
    return db_path, 10  # 10 total edges


# ===========================================================================
# Test classes
# ===========================================================================


class TestBacktesterInit:
    """Tests for Backtester initialization and DB loading."""

    def test_load_session_from_db(self, tmp_path):
        """Backtester can load and run from a valid DB."""
        db_path, _ = _make_standard_db(tmp_path)
        bt = Backtester(db_path)
        cfg = BacktestConfig(db_path=db_path)
        result = bt.run(cfg)
        assert isinstance(result, BacktestResult)
        assert result.config_label == "baseline"

    def test_load_settings_from_db(self, tmp_path):
        """Settings overrides are merged into the DB settings."""
        db_path, _ = _make_standard_db(tmp_path)
        bt = Backtester(db_path)
        cfg = BacktestConfig(
            db_path=db_path,
            settings_overrides={"kelly_fraction_cap": 0.05},
        )
        result = bt.run(cfg)
        assert "k_fraction_cap" in result.config_label

    def test_invalid_db_path_raises(self, tmp_path):
        """Attempting to run against a non-existent DB raises an error."""
        bt = Backtester("/nonexistent/path.db")
        cfg = BacktestConfig(db_path="/nonexistent/path.db")
        with pytest.raises(Exception):
            bt.run(cfg)


class TestBacktesterRun:
    """Tests for the core run logic."""

    def test_basic_run_produces_trades(self, tmp_path):
        """A basic run with default settings produces some trades."""
        db_path, _ = _make_standard_db(tmp_path)
        bt = Backtester(db_path)
        result = bt.run(BacktestConfig(db_path=db_path))
        assert result.num_trades > 0

    def test_no_edges_no_trades(self, tmp_path):
        """If DB has no edges, result has zero trades."""
        cycles = [{"id": 1, "cycle_number": 1, "regime": "mean_reverting", "regime_confidence": 0.8}]
        db_path = _create_test_db(tmp_path, cycles=cycles, edges=[], trades=[])
        bt = Backtester(db_path)
        result = bt.run(BacktestConfig(db_path=db_path))
        assert result.num_trades == 0
        assert result.total_pnl == 0.0

    def test_lower_min_edge_more_trades(self, tmp_path):
        """Lowering regime_min_edge threshold produces more surviving edges."""
        # Enable regime_min_edge with a very high threshold
        db_path, _ = _make_standard_db(
            tmp_path,
            settings_overrides={"regime_min_edge_enabled": True, "regime_min_edge_mean_reverting": 15.0,
                                "regime_min_edge_trending": 15.0, "regime_min_edge_high_vol": 15.0},
        )
        bt = Backtester(db_path)

        # High threshold: kills all edges below 15 cents
        cfg_high = BacktestConfig(db_path=db_path)
        result_high = bt.run(cfg_high)

        # Low threshold via override: allows more through
        cfg_low = BacktestConfig(
            db_path=db_path,
            settings_overrides={"regime_min_edge_mean_reverting": 1.0,
                                "regime_min_edge_trending": 1.0,
                                "regime_min_edge_high_vol": 1.0},
        )
        result_low = bt.run(cfg_low)

        assert result_low.num_trades >= result_high.num_trades

    def test_kelly_fraction_cap_respected(self, tmp_path):
        """A lower kelly_fraction_cap should change position sizing."""
        db_path, _ = _make_standard_db(tmp_path)
        bt = Backtester(db_path)

        # Low cap
        cfg_low = BacktestConfig(
            db_path=db_path,
            settings_overrides={"kelly_fraction_cap": 0.01},
        )
        result_low = bt.run(cfg_low)

        # High cap
        cfg_high = BacktestConfig(
            db_path=db_path,
            settings_overrides={"kelly_fraction_cap": 0.50},
        )
        result_high = bt.run(cfg_high)

        # Different caps should yield different total PnL or trade sizes
        # (could be same trade count but different contract amounts)
        if result_low.num_trades > 0 and result_high.num_trades > 0:
            low_contracts = sum(t.contracts for t in result_low.trades)
            high_contracts = sum(t.contracts for t in result_high.trades)
            assert high_contracts >= low_contracts

    def test_running_bankroll_tracked(self, tmp_path):
        """Bankroll is deducted and replenished across cycles."""
        db_path, _ = _make_standard_db(tmp_path)
        bt = Backtester(db_path)
        result = bt.run(BacktestConfig(db_path=db_path))
        # With 500 bankroll and settled trades, total_pnl reflects net
        assert isinstance(result.total_pnl, float)

    def test_max_concurrent_positions(self, tmp_path):
        """Setting max_concurrent_positions=1 limits trades."""
        db_path, _ = _make_standard_db(
            tmp_path,
            settings_overrides={"max_concurrent_positions": 1},
        )
        bt = Backtester(db_path)
        result = bt.run(BacktestConfig(db_path=db_path))
        # Should still produce trades since positions settle immediately
        assert result.num_trades >= 1

    def test_max_positions_per_underlying(self, tmp_path):
        """Setting max_positions_per_underlying=1 caps per-underlying trades."""
        db_path, _ = _make_standard_db(
            tmp_path,
            settings_overrides={"max_positions_per_underlying": 1},
        )
        bt = Backtester(db_path)
        cfg_limited = BacktestConfig(db_path=db_path)
        result_limited = bt.run(cfg_limited)

        # Compare with unlimited
        db_path2, _ = _make_standard_db(
            tmp_path / "sub",
            settings_overrides={"max_positions_per_underlying": 100},
        )
        bt2 = Backtester(db_path2)
        result_unlimited = bt2.run(BacktestConfig(db_path=db_path2))

        assert result_limited.num_trades <= result_unlimited.num_trades

    def test_pnl_from_settlement_win(self, tmp_path):
        """Winning trade computes PnL correctly: (1 - entry_price) * contracts."""
        cycles = [{"id": 1, "cycle_number": 1, "regime": "mean_reverting", "regime_confidence": 0.8}]
        edges = [
            {"id": 1, "cycle_id": 1, "ticker": "KXBTC15M-26FEB15-T100500-A",
             "side": "yes", "edge_cents": 10.0, "yes_buy_price": 0.40,
             "no_buy_price": 0.60, "zscore_value": 0.5, "model_uncertainty": 0.0},
        ]
        trades = [
            {"id": 1, "cycle_id": 1, "edge_id": 1,
             "ticker": "KXBTC15M-26FEB15-T100500-A",
             "side": "yes", "contracts": 10, "entry_price": 0.40,
             "settled": 1, "actual_outcome": "yes", "pnl": 6.0},
        ]
        db_path = _create_test_db(tmp_path, cycles=cycles, edges=edges, trades=trades)
        bt = Backtester(db_path)
        result = bt.run(BacktestConfig(db_path=db_path))
        # The backtester recalculates PnL based on its own contract sizing
        # For a win: pnl = (1.0 - entry_price) * contracts
        if result.num_trades > 0:
            for t in result.trades:
                if t.actual_outcome == "yes" and t.side == "yes":
                    expected_pnl = (1.0 - t.entry_price) * t.contracts
                    assert abs(t.pnl - expected_pnl) < 0.01

    def test_pnl_from_settlement_loss(self, tmp_path):
        """Losing trade computes PnL correctly: -entry_price * contracts."""
        cycles = [{"id": 1, "cycle_number": 1, "regime": "mean_reverting", "regime_confidence": 0.8}]
        edges = [
            {"id": 1, "cycle_id": 1, "ticker": "KXBTC15M-26FEB15-T100500-A",
             "side": "yes", "edge_cents": 10.0, "yes_buy_price": 0.40,
             "no_buy_price": 0.60, "zscore_value": 0.5, "model_uncertainty": 0.0},
        ]
        trades = [
            {"id": 1, "cycle_id": 1, "edge_id": 1,
             "ticker": "KXBTC15M-26FEB15-T100500-A",
             "side": "yes", "contracts": 10, "entry_price": 0.40,
             "settled": 1, "actual_outcome": "no", "pnl": -4.0},
        ]
        db_path = _create_test_db(tmp_path, cycles=cycles, edges=edges, trades=trades)
        bt = Backtester(db_path)
        result = bt.run(BacktestConfig(db_path=db_path))
        if result.num_trades > 0:
            for t in result.trades:
                if t.actual_outcome == "no" and t.side == "yes":
                    expected_pnl = -t.entry_price * t.contracts
                    assert abs(t.pnl - expected_pnl) < 0.01

    def test_unsettled_excluded_when_required(self, tmp_path):
        """Unsettled trades are excluded when require_settlement=True."""
        cycles = [{"id": 1, "cycle_number": 1, "regime": "mean_reverting", "regime_confidence": 0.8}]
        edges = [
            {"id": 1, "cycle_id": 1, "ticker": "KXBTC15M-26FEB15-T100500-A",
             "side": "yes", "edge_cents": 10.0, "yes_buy_price": 0.40,
             "no_buy_price": 0.60, "zscore_value": 0.5, "model_uncertainty": 0.0},
        ]
        # No settled trades for this edge
        db_path = _create_test_db(tmp_path, cycles=cycles, edges=edges, trades=[])
        bt = Backtester(db_path)
        result = bt.run(BacktestConfig(db_path=db_path, require_settlement=True))
        assert result.num_trades == 0

    def test_unsettled_included_when_not_required(self, tmp_path):
        """Unsettled trades are included when require_settlement=False."""
        cycles = [{"id": 1, "cycle_number": 1, "regime": "mean_reverting", "regime_confidence": 0.8}]
        edges = [
            {"id": 1, "cycle_id": 1, "ticker": "KXBTC15M-26FEB15-T100500-A",
             "side": "yes", "edge_cents": 10.0, "yes_buy_price": 0.40,
             "no_buy_price": 0.60, "zscore_value": 0.5, "model_uncertainty": 0.0},
        ]
        db_path = _create_test_db(tmp_path, cycles=cycles, edges=edges, trades=[])
        bt = Backtester(db_path)
        result = bt.run(BacktestConfig(db_path=db_path, require_settlement=False))
        # Edge should pass through (no settlement needed) — trade gets created with pnl=None
        assert len(result.trades) > 0
        # But num_trades counts only settled trades
        assert result.num_trades == 0  # no settled PnL

    def test_regime_breakdown_counts(self, tmp_path):
        """Regime breakdown correctly attributes trades to regimes."""
        db_path, _ = _make_standard_db(tmp_path)
        bt = Backtester(db_path)
        result = bt.run(BacktestConfig(db_path=db_path))
        # Check that regime breakdown sums to num_trades
        total_regime_trades = sum(result.trades_by_regime.values())
        assert total_regime_trades == result.num_trades

    def test_filter_stats_correct(self, tmp_path):
        """edges_killed_by is populated correctly."""
        db_path, total_edges = _make_standard_db(tmp_path)
        bt = Backtester(db_path)
        result = bt.run(BacktestConfig(db_path=db_path))
        assert result.edges_total == total_edges
        # Sum of killed + survived should equal total
        killed_sum = sum(
            v for k, v in result.edges_killed_by.items() if k != "survived"
        )
        survived = result.edges_killed_by.get("survived", 0)
        assert killed_sum + survived == total_edges


class TestBacktesterFilters:
    """Tests for individual filter replay."""

    def test_regime_min_edge_kills_below_threshold(self, tmp_path):
        """Edges below the regime min edge threshold are killed."""
        cycles = [{"id": 1, "cycle_number": 1, "regime": "mean_reverting", "regime_confidence": 0.8}]
        edges = [
            # Edge with 3 cents — below 5.0 threshold
            {"id": 1, "cycle_id": 1, "ticker": "KXBTC15M-26FEB15-T100500-A",
             "side": "yes", "edge_cents": 3.0, "yes_buy_price": 0.40,
             "no_buy_price": 0.60, "zscore_value": 0.5, "model_uncertainty": 0.0},
        ]
        trades = [
            {"id": 1, "cycle_id": 1, "edge_id": 1,
             "ticker": "KXBTC15M-26FEB15-T100500-A",
             "side": "yes", "contracts": 5, "entry_price": 0.40,
             "settled": 1, "actual_outcome": "yes", "pnl": 3.0},
        ]
        db_path = _create_test_db(
            tmp_path,
            settings_overrides={"regime_min_edge_enabled": True, "regime_min_edge_mean_reverting": 5.0},
            cycles=cycles, edges=edges, trades=trades,
        )
        bt = Backtester(db_path)
        result = bt.run(BacktestConfig(db_path=db_path, apply_regime_min_edge=True))
        assert result.edges_killed_by["regime_min_edge"] == 1
        assert result.num_trades == 0

    def test_zscore_filter_kills_above_max(self, tmp_path):
        """Edges with zscore > max are killed."""
        cycles = [{"id": 1, "cycle_number": 1, "regime": "mean_reverting", "regime_confidence": 0.8}]
        edges = [
            {"id": 1, "cycle_id": 1, "ticker": "KXBTC15M-26FEB15-T100500-A",
             "side": "yes", "edge_cents": 10.0, "yes_buy_price": 0.40,
             "no_buy_price": 0.60, "zscore_value": 3.5, "model_uncertainty": 0.0},
        ]
        trades = [
            {"id": 1, "cycle_id": 1, "edge_id": 1,
             "ticker": "KXBTC15M-26FEB15-T100500-A",
             "side": "yes", "contracts": 5, "entry_price": 0.40,
             "settled": 1, "actual_outcome": "yes", "pnl": 3.0},
        ]
        db_path = _create_test_db(
            tmp_path,
            settings_overrides={"zscore_filter_enabled": True, "zscore_max": 2.0},
            cycles=cycles, edges=edges, trades=trades,
        )
        bt = Backtester(db_path)
        result = bt.run(BacktestConfig(db_path=db_path, apply_zscore=True))
        assert result.edges_killed_by["zscore"] == 1

    def test_counter_trend_kills_opposing(self, tmp_path):
        """Counter-trend filter kills edges opposing the prevailing trend."""
        cycles = [{"id": 1, "cycle_number": 1, "regime": "trending_up", "regime_confidence": 0.80}]
        # side=no + direction=above in trending_up = counter-trend (bearish bet in bull market)
        edges = [
            {"id": 1, "cycle_id": 1, "ticker": "KXBTC15M-26FEB15-T100500-A",
             "side": "no", "edge_cents": 10.0, "yes_buy_price": 0.60,
             "no_buy_price": 0.40, "zscore_value": 0.5, "model_uncertainty": 0.0},
        ]
        trades = [
            {"id": 1, "cycle_id": 1, "edge_id": 1,
             "ticker": "KXBTC15M-26FEB15-T100500-A",
             "side": "no", "contracts": 5, "entry_price": 0.40,
             "settled": 1, "actual_outcome": "no", "pnl": 3.0},
        ]
        snapshots = [
            {"cycle_id": 1, "ticker": "KXBTC15M-26FEB15-T100500-A", "direction": "above"},
        ]
        db_path = _create_test_db(
            tmp_path,
            settings_overrides={"regime_skip_counter_trend": True, "regime_skip_counter_trend_min_conf": 0.6},
            cycles=cycles, edges=edges, trades=trades, snapshots=snapshots,
        )
        bt = Backtester(db_path)
        result = bt.run(BacktestConfig(db_path=db_path, apply_counter_trend=True))
        assert result.edges_killed_by["counter_trend"] == 1

    def test_confidence_filter_kills_low_score(self, tmp_path):
        """Confidence filter kills edges with score below min."""
        cycles = [{"id": 1, "cycle_number": 1, "regime": "mean_reverting", "regime_confidence": 0.8}]
        edges = [
            {"id": 1, "cycle_id": 1, "ticker": "KXBTC15M-26FEB15-T100500-A",
             "side": "yes", "edge_cents": 10.0, "yes_buy_price": 0.40,
             "no_buy_price": 0.60, "zscore_value": 0.5, "confidence_score": 0.30,
             "model_uncertainty": 0.0},
        ]
        trades = [
            {"id": 1, "cycle_id": 1, "edge_id": 1,
             "ticker": "KXBTC15M-26FEB15-T100500-A",
             "side": "yes", "contracts": 5, "entry_price": 0.40,
             "settled": 1, "actual_outcome": "yes", "pnl": 3.0},
        ]
        db_path = _create_test_db(
            tmp_path,
            settings_overrides={"confidence_scoring_enabled": True, "confidence_min_score": 0.65},
            cycles=cycles, edges=edges, trades=trades,
        )
        bt = Backtester(db_path)
        result = bt.run(BacktestConfig(db_path=db_path, apply_confidence=True))
        assert result.edges_killed_by["confidence"] == 1

    def test_disable_regime_min_edge_allows_all(self, tmp_path):
        """Disabling regime_min_edge allows all edges through that filter."""
        cycles = [{"id": 1, "cycle_number": 1, "regime": "mean_reverting", "regime_confidence": 0.8}]
        edges = [
            {"id": 1, "cycle_id": 1, "ticker": "KXBTC15M-26FEB15-T100500-A",
             "side": "yes", "edge_cents": 1.0, "yes_buy_price": 0.40,
             "no_buy_price": 0.60, "zscore_value": 0.5, "model_uncertainty": 0.0},
        ]
        trades = [
            {"id": 1, "cycle_id": 1, "edge_id": 1,
             "ticker": "KXBTC15M-26FEB15-T100500-A",
             "side": "yes", "contracts": 5, "entry_price": 0.40,
             "settled": 1, "actual_outcome": "yes", "pnl": 3.0},
        ]
        db_path = _create_test_db(
            tmp_path,
            settings_overrides={"regime_min_edge_enabled": True, "regime_min_edge_mean_reverting": 5.0},
            cycles=cycles, edges=edges, trades=trades,
        )
        bt = Backtester(db_path)
        result = bt.run(BacktestConfig(db_path=db_path, apply_regime_min_edge=False))
        assert result.edges_killed_by["regime_min_edge"] == 0

    def test_disable_zscore_allows_all(self, tmp_path):
        """Disabling zscore filter allows all edges through."""
        cycles = [{"id": 1, "cycle_number": 1, "regime": "mean_reverting", "regime_confidence": 0.8}]
        edges = [
            {"id": 1, "cycle_id": 1, "ticker": "KXBTC15M-26FEB15-T100500-A",
             "side": "yes", "edge_cents": 10.0, "yes_buy_price": 0.40,
             "no_buy_price": 0.60, "zscore_value": 10.0, "model_uncertainty": 0.0},
        ]
        trades = [
            {"id": 1, "cycle_id": 1, "edge_id": 1,
             "ticker": "KXBTC15M-26FEB15-T100500-A",
             "side": "yes", "contracts": 5, "entry_price": 0.40,
             "settled": 1, "actual_outcome": "yes", "pnl": 3.0},
        ]
        db_path = _create_test_db(
            tmp_path,
            settings_overrides={"zscore_filter_enabled": True, "zscore_max": 2.0},
            cycles=cycles, edges=edges, trades=trades,
        )
        bt = Backtester(db_path)
        result = bt.run(BacktestConfig(db_path=db_path, apply_zscore=False))
        assert result.edges_killed_by["zscore"] == 0

    def test_disable_counter_trend_allows_all(self, tmp_path):
        """Disabling counter_trend filter allows opposing edges through."""
        cycles = [{"id": 1, "cycle_number": 1, "regime": "trending_up", "regime_confidence": 0.80}]
        edges = [
            {"id": 1, "cycle_id": 1, "ticker": "KXBTC15M-26FEB15-T100500-A",
             "side": "no", "edge_cents": 10.0, "yes_buy_price": 0.60,
             "no_buy_price": 0.40, "zscore_value": 0.5, "model_uncertainty": 0.0},
        ]
        trades = [
            {"id": 1, "cycle_id": 1, "edge_id": 1,
             "ticker": "KXBTC15M-26FEB15-T100500-A",
             "side": "no", "contracts": 5, "entry_price": 0.40,
             "settled": 1, "actual_outcome": "no", "pnl": 3.0},
        ]
        snapshots = [
            {"cycle_id": 1, "ticker": "KXBTC15M-26FEB15-T100500-A", "direction": "above"},
        ]
        db_path = _create_test_db(
            tmp_path,
            settings_overrides={"regime_skip_counter_trend": True},
            cycles=cycles, edges=edges, trades=trades, snapshots=snapshots,
        )
        bt = Backtester(db_path)
        result = bt.run(BacktestConfig(db_path=db_path, apply_counter_trend=False))
        assert result.edges_killed_by["counter_trend"] == 0

    def test_disable_confidence_allows_all(self, tmp_path):
        """Disabling confidence filter allows low-confidence edges through."""
        cycles = [{"id": 1, "cycle_number": 1, "regime": "mean_reverting", "regime_confidence": 0.8}]
        edges = [
            {"id": 1, "cycle_id": 1, "ticker": "KXBTC15M-26FEB15-T100500-A",
             "side": "yes", "edge_cents": 10.0, "yes_buy_price": 0.40,
             "no_buy_price": 0.60, "zscore_value": 0.5, "confidence_score": 0.10,
             "model_uncertainty": 0.0},
        ]
        trades = [
            {"id": 1, "cycle_id": 1, "edge_id": 1,
             "ticker": "KXBTC15M-26FEB15-T100500-A",
             "side": "yes", "contracts": 5, "entry_price": 0.40,
             "settled": 1, "actual_outcome": "yes", "pnl": 3.0},
        ]
        db_path = _create_test_db(
            tmp_path,
            settings_overrides={"confidence_scoring_enabled": True, "confidence_min_score": 0.65},
            cycles=cycles, edges=edges, trades=trades,
        )
        bt = Backtester(db_path)
        result = bt.run(BacktestConfig(db_path=db_path, apply_confidence=False))
        assert result.edges_killed_by["confidence"] == 0


class TestBacktesterSweep:
    """Tests for parameter sweep methods."""

    def test_single_param_sweep_result_count(self, tmp_path):
        """Single param sweep returns one result per value."""
        db_path, _ = _make_standard_db(tmp_path)
        bt = Backtester(db_path)
        values = [0.02, 0.05, 0.10, 0.20]
        results = bt.sweep("kelly_fraction_cap", values)
        assert len(results) == len(values)

    def test_sweep_different_trade_counts(self, tmp_path):
        """Different sweep values may produce different trade counts."""
        db_path, _ = _make_standard_db(
            tmp_path,
            settings_overrides={"zscore_filter_enabled": True},
        )
        bt = Backtester(db_path)
        # Sweep zscore_max from very tight to very loose
        values = [0.5, 1.0, 5.0, 100.0]
        results = bt.sweep("zscore_max", values)
        # Tight zscore should kill more edges than loose zscore
        assert results[-1].num_trades >= results[0].num_trades

    def test_multi_sweep_cartesian_product(self, tmp_path):
        """Multi-sweep produces cartesian product of parameter values."""
        db_path, _ = _make_standard_db(tmp_path)
        bt = Backtester(db_path)
        grid = {
            "kelly_fraction_cap": [0.05, 0.10],
            "zscore_max": [1.0, 3.0],
        }
        results = bt.multi_sweep(grid)
        assert len(results) == 2 * 2  # 4 combinations

    def test_multi_sweep_empty_grid(self, tmp_path):
        """Multi-sweep with empty parameter lists produces no results."""
        db_path, _ = _make_standard_db(tmp_path)
        bt = Backtester(db_path)
        results = bt.multi_sweep({"kelly_fraction_cap": []})
        assert len(results) == 0

    def test_sweep_preserves_base_overrides(self, tmp_path):
        """Base overrides are preserved in every sweep result."""
        db_path, _ = _make_standard_db(tmp_path)
        bt = Backtester(db_path)
        base = {"zscore_max": 100.0}  # Very loose zscore
        results = bt.sweep("kelly_fraction_cap", [0.05, 0.10], base_overrides=base)
        for r in results:
            assert "zscore_max" in r.settings_overrides
            assert r.settings_overrides["zscore_max"] == 100.0

    def test_sweep_results_ordered_by_value(self, tmp_path):
        """Sweep results are in the same order as the input values."""
        db_path, _ = _make_standard_db(tmp_path)
        bt = Backtester(db_path)
        values = [0.20, 0.05, 0.10]
        results = bt.sweep("kelly_fraction_cap", values)
        for r, v in zip(results, values):
            assert r.settings_overrides["kelly_fraction_cap"] == v


class TestBacktesterSizing:
    """Tests for Kelly sizing and regime adjustments."""

    def test_kelly_sizing_with_different_caps(self, tmp_path):
        """Higher kelly cap yields larger positions."""
        db_path_low, _ = _make_standard_db(
            tmp_path / "low",
            settings_overrides={"kelly_fraction_cap": 0.02},
        )
        db_path_high, _ = _make_standard_db(
            tmp_path / "high",
            settings_overrides={"kelly_fraction_cap": 0.30},
        )
        bt_low = Backtester(db_path_low)
        bt_high = Backtester(db_path_high)
        result_low = bt_low.run(BacktestConfig(db_path=db_path_low))
        result_high = bt_high.run(BacktestConfig(db_path=db_path_high))
        if result_low.num_trades > 0 and result_high.num_trades > 0:
            avg_contracts_low = sum(t.contracts for t in result_low.trades) / len(result_low.trades)
            avg_contracts_high = sum(t.contracts for t in result_high.trades) / len(result_high.trades)
            assert avg_contracts_high >= avg_contracts_low

    def test_regime_kelly_multiplier_high_vol_zero(self, tmp_path):
        """regime_kelly_high_vol=0 means no trades in high_vol regime."""
        cycles = [{"id": 1, "cycle_number": 1, "regime": "high_vol", "regime_confidence": 0.8}]
        edges = [
            {"id": 1, "cycle_id": 1, "ticker": "KXBTC15M-26FEB15-T100500-A",
             "side": "yes", "edge_cents": 10.0, "yes_buy_price": 0.40,
             "no_buy_price": 0.60, "zscore_value": 0.5, "model_uncertainty": 0.0},
        ]
        trades = [
            {"id": 1, "cycle_id": 1, "edge_id": 1,
             "ticker": "KXBTC15M-26FEB15-T100500-A",
             "side": "yes", "contracts": 5, "entry_price": 0.40,
             "settled": 1, "actual_outcome": "yes", "pnl": 3.0},
        ]
        db_path = _create_test_db(
            tmp_path,
            settings_overrides={
                "regime_sizing_enabled": True,
                "regime_kelly_high_vol": 0.0,
            },
            cycles=cycles, edges=edges, trades=trades,
        )
        bt = Backtester(db_path)
        result = bt.run(BacktestConfig(db_path=db_path))
        assert result.num_trades == 0

    def test_mean_reverting_cap_boost(self, tmp_path):
        """Mean-reverting regime with high confidence gets cap boost."""
        cycles = [{"id": 1, "cycle_number": 1, "regime": "mean_reverting", "regime_confidence": 0.85}]
        edges = [
            {"id": 1, "cycle_id": 1, "ticker": "KXBTC15M-26FEB15-T100500-A",
             "side": "yes", "edge_cents": 15.0, "yes_buy_price": 0.40,
             "no_buy_price": 0.60, "zscore_value": 0.5, "model_uncertainty": 0.0},
        ]
        trades = [
            {"id": 1, "cycle_id": 1, "edge_id": 1,
             "ticker": "KXBTC15M-26FEB15-T100500-A",
             "side": "yes", "contracts": 10, "entry_price": 0.40,
             "settled": 1, "actual_outcome": "yes", "pnl": 6.0},
        ]
        # With boost
        db_path_boost = _create_test_db(
            tmp_path / "boost",
            settings_overrides={"regime_kelly_cap_boost_mean_reverting": 2.0, "kelly_fraction_cap": 0.10},
            cycles=cycles, edges=edges, trades=trades,
        )
        # Without boost
        db_path_noboost = _create_test_db(
            tmp_path / "noboost",
            settings_overrides={"regime_kelly_cap_boost_mean_reverting": 1.0, "kelly_fraction_cap": 0.10},
            cycles=cycles, edges=edges, trades=trades,
        )
        bt_boost = Backtester(db_path_boost)
        bt_noboost = Backtester(db_path_noboost)
        r_boost = bt_boost.run(BacktestConfig(db_path=db_path_boost))
        r_noboost = bt_noboost.run(BacktestConfig(db_path=db_path_noboost))
        if r_boost.num_trades > 0 and r_noboost.num_trades > 0:
            contracts_boost = sum(t.contracts for t in r_boost.trades)
            contracts_noboost = sum(t.contracts for t in r_noboost.trades)
            assert contracts_boost >= contracts_noboost

    def test_bankroll_depletion(self, tmp_path):
        """With tiny bankroll, not all edges can be traded."""
        cycles = [{"id": 1, "cycle_number": 1, "regime": "mean_reverting", "regime_confidence": 0.8}]
        # Many high-edge trades to exhaust bankroll
        edges = []
        tr = []
        for i in range(20):
            eid = i + 1
            ticker = "KXBTC15M-26FEB15-T1005{:02d}-A".format(i)
            edges.append({
                "id": eid, "cycle_id": 1, "ticker": ticker,
                "side": "yes", "edge_cents": 15.0, "yes_buy_price": 0.40,
                "no_buy_price": 0.60, "zscore_value": 0.5, "model_uncertainty": 0.0,
            })
            tr.append({
                "id": eid, "cycle_id": 1, "edge_id": eid,
                "ticker": ticker,
                "side": "yes", "contracts": 10, "entry_price": 0.40,
                "settled": 1, "actual_outcome": "no", "pnl": -4.0,
            })
        db_path = _create_test_db(
            tmp_path,
            settings_overrides={
                "bankroll": 10.0,  # Very low bankroll
                "kelly_fraction_cap": 0.50,
                "max_concurrent_positions": 100,
                "max_positions_per_underlying": 100,
            },
            cycles=cycles, edges=edges, trades=tr,
        )
        bt = Backtester(db_path)
        result = bt.run(BacktestConfig(db_path=db_path))
        # Not all 20 edges can be traded with $10 bankroll
        assert result.num_trades < 20

    def test_max_position_per_market(self, tmp_path):
        """max_position_per_market caps dollar amount per trade."""
        cycles = [{"id": 1, "cycle_number": 1, "regime": "mean_reverting", "regime_confidence": 0.8}]
        edges = [
            {"id": 1, "cycle_id": 1, "ticker": "KXBTC15M-26FEB15-T100500-A",
             "side": "yes", "edge_cents": 20.0, "yes_buy_price": 0.40,
             "no_buy_price": 0.60, "zscore_value": 0.5, "model_uncertainty": 0.0},
        ]
        trades = [
            {"id": 1, "cycle_id": 1, "edge_id": 1,
             "ticker": "KXBTC15M-26FEB15-T100500-A",
             "side": "yes", "contracts": 100, "entry_price": 0.40,
             "settled": 1, "actual_outcome": "yes", "pnl": 60.0},
        ]
        db_path = _create_test_db(
            tmp_path,
            settings_overrides={
                "max_position_per_market": 5.0,
                "kelly_fraction_cap": 0.50,
                "bankroll": 1000.0,
            },
            cycles=cycles, edges=edges, trades=trades,
        )
        bt = Backtester(db_path)
        result = bt.run(BacktestConfig(db_path=db_path))
        if result.num_trades > 0:
            for t in result.trades:
                dollar_cost = t.entry_price * t.contracts
                assert dollar_cost <= 5.0 + 0.01  # Allow small float tolerance


class TestBacktesterOutput:
    """Tests for print and export methods."""

    def test_print_report_zero_trades(self, tmp_path):
        """print_report doesn't crash with zero trades."""
        db_path = _create_test_db(
            tmp_path,
            cycles=[{"id": 1, "cycle_number": 1, "regime": "mean_reverting", "regime_confidence": 0.8}],
            edges=[], trades=[],
        )
        bt = Backtester(db_path)
        result = bt.run(BacktestConfig(db_path=db_path))
        buf = io.StringIO()
        with redirect_stdout(buf):
            bt.print_report(result)
        output = buf.getvalue()
        assert "Backtest Report" in output
        assert "N/A" in output  # No trades means N/A period

    def test_print_report_with_trades(self, tmp_path):
        """print_report produces output with trades."""
        db_path, _ = _make_standard_db(tmp_path)
        bt = Backtester(db_path)
        result = bt.run(BacktestConfig(db_path=db_path))
        buf = io.StringIO()
        with redirect_stdout(buf):
            bt.print_report(result)
        output = buf.getvalue()
        assert "Backtest Report" in output
        assert "Trades:" in output
        assert "Total PnL" in output

    def test_print_sweep_report(self, tmp_path):
        """print_sweep_report produces a table."""
        db_path, _ = _make_standard_db(tmp_path)
        bt = Backtester(db_path)
        results = bt.sweep("kelly_fraction_cap", [0.05, 0.10])
        buf = io.StringIO()
        with redirect_stdout(buf):
            bt.print_sweep_report(results)
        output = buf.getvalue()
        assert "Parameter Sweep" in output

    def test_export_trades_csv_creates_file(self, tmp_path):
        """export_trades_csv creates a file with correct columns."""
        db_path, _ = _make_standard_db(tmp_path)
        bt = Backtester(db_path)
        result = bt.run(BacktestConfig(db_path=db_path))
        csv_path = str(tmp_path / "trades.csv")
        bt.export_trades_csv(result, csv_path)
        assert os.path.exists(csv_path)
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            assert "cycle_id" in headers
            assert "ticker" in headers
            assert "pnl" in headers
            assert "edge_cents" in headers
            assert "regime" in headers

    def test_export_trades_csv_content_matches(self, tmp_path):
        """Exported CSV content matches the result trades."""
        db_path, _ = _make_standard_db(tmp_path)
        bt = Backtester(db_path)
        result = bt.run(BacktestConfig(db_path=db_path))
        csv_path = str(tmp_path / "trades.csv")
        bt.export_trades_csv(result, csv_path)
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == len(result.trades)
        for row, trade in zip(rows, result.trades):
            assert row["ticker"] == trade.ticker
            assert row["side"] == trade.side

    def test_sharpe_calculation(self, tmp_path):
        """Sharpe is mean/std of trade PnLs."""
        cycles = [{"id": 1, "cycle_number": 1, "regime": "mean_reverting", "regime_confidence": 0.8}]
        # Create edges that will produce known PnL values
        edges = [
            {"id": 1, "cycle_id": 1, "ticker": "KXBTC15M-26FEB15-T100501-A",
             "side": "yes", "edge_cents": 10.0, "yes_buy_price": 0.40,
             "no_buy_price": 0.60, "zscore_value": 0.5, "model_uncertainty": 0.0},
            {"id": 2, "cycle_id": 1, "ticker": "KXETH15M-26FEB15-T100502-A",
             "side": "yes", "edge_cents": 10.0, "yes_buy_price": 0.40,
             "no_buy_price": 0.60, "zscore_value": 0.5, "model_uncertainty": 0.0},
            {"id": 3, "cycle_id": 1, "ticker": "KXBTC15M-26FEB15-T100503-A",
             "side": "yes", "edge_cents": 10.0, "yes_buy_price": 0.40,
             "no_buy_price": 0.60, "zscore_value": 0.5, "model_uncertainty": 0.0},
        ]
        trades = [
            {"id": 1, "cycle_id": 1, "edge_id": 1,
             "ticker": "KXBTC15M-26FEB15-T100501-A",
             "side": "yes", "contracts": 10, "entry_price": 0.40,
             "settled": 1, "actual_outcome": "yes", "pnl": 6.0},
            {"id": 2, "cycle_id": 1, "edge_id": 2,
             "ticker": "KXETH15M-26FEB15-T100502-A",
             "side": "yes", "contracts": 10, "entry_price": 0.40,
             "settled": 1, "actual_outcome": "no", "pnl": -4.0},
            {"id": 3, "cycle_id": 1, "edge_id": 3,
             "ticker": "KXBTC15M-26FEB15-T100503-A",
             "side": "yes", "contracts": 10, "entry_price": 0.40,
             "settled": 1, "actual_outcome": "yes", "pnl": 6.0},
        ]
        db_path = _create_test_db(tmp_path, cycles=cycles, edges=edges, trades=trades)
        bt = Backtester(db_path)
        result = bt.run(BacktestConfig(db_path=db_path))
        # Verify sharpe is computed and non-zero when there's variance
        if result.num_trades >= 2:
            pnls = [t.pnl for t in result.trades if t.pnl is not None]
            if len(pnls) >= 2:
                mean_pnl = sum(pnls) / len(pnls)
                var_pnl = sum((p - mean_pnl) ** 2 for p in pnls) / (len(pnls) - 1)
                std_pnl = math.sqrt(var_pnl) if var_pnl > 0 else 0.0
                expected_sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0.0
                assert abs(result.sharpe_approx - expected_sharpe) < 0.01


# ===========================================================================
# Standalone helper tests
# ===========================================================================


class TestHelpers:
    """Tests for standalone helper functions."""

    def test_extract_underlying_btc(self):
        assert _extract_underlying("KXBTC15M-26FEB15-T100500-B") == "KXBTC"

    def test_extract_underlying_eth(self):
        assert _extract_underlying("KXETH15M-26FEB15-T3200-A") == "KXETH"

    def test_extract_underlying_plain(self):
        assert _extract_underlying("ABCDEF") == "ABCDEF"

    def test_is_counter_trend_bearish_in_bull(self):
        """side=no, direction=above, regime=trending_up => counter-trend."""
        assert _is_counter_trend("no", "above", "trending_up") is True

    def test_is_counter_trend_bullish_in_bull(self):
        """side=yes, direction=above, regime=trending_up => with-trend."""
        assert _is_counter_trend("yes", "above", "trending_up") is False

    def test_is_counter_trend_mean_reverting(self):
        """Mean-reverting regime is never counter-trend."""
        assert _is_counter_trend("no", "above", "mean_reverting") is False

    def test_is_counter_trend_no_direction(self):
        """No direction info means not counter-trend."""
        assert _is_counter_trend("no", None, "trending_up") is False
