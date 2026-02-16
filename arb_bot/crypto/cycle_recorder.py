"""Append-only SQLite recorder for full engine cycle state.

Captures market snapshots, edges, filter decisions, trades, and
settlements for offline backtesting and Monte Carlo simulation replay.

Uses WAL mode, single-transaction-per-cycle bracketing, and
``executemany()`` for bulk inserts.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Filter result dataclass
# ---------------------------------------------------------------------------


@dataclass
class FilterResult:
    """Per-edge filter decision trail."""

    passed_regime_min_edge: Optional[bool] = None
    passed_zscore: Optional[bool] = None
    passed_counter_trend: Optional[bool] = None
    passed_confidence: Optional[bool] = None
    zscore_value: Optional[float] = None
    confidence_score: Optional[float] = None
    confidence_agreement: Optional[int] = None
    reject_reason: Optional[str] = None
    survived_all: bool = False
    contracts_sized: int = 0
    was_traded: bool = False


# ---------------------------------------------------------------------------
# Schema DDL
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

CREATE INDEX IF NOT EXISTS idx_cycles_timestamp ON cycles(timestamp);
CREATE INDEX IF NOT EXISTS idx_cycles_session ON cycles(session_id);

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

CREATE INDEX IF NOT EXISTS idx_snapshots_cycle ON market_snapshots(cycle_id);

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

CREATE INDEX IF NOT EXISTS idx_edges_cycle ON edges(cycle_id);
CREATE INDEX IF NOT EXISTS idx_edges_ticker ON edges(ticker);

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

CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades(ticker);
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_settings_dict(settings: Any) -> Dict[str, Any]:
    """Convert a CryptoSettings dataclass to a JSON-safe dict.

    Handles non-serializable types by falling back to ``str()``.
    """
    try:
        raw = dataclasses.asdict(settings)
    except Exception:
        return {}
    return _make_json_safe(raw)


def _make_json_safe(obj: Any) -> Any:
    """Recursively ensure all values are JSON-serializable."""
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # Fallback: stringify
    return str(obj)


def _bool_to_int(val: Optional[bool]) -> Optional[int]:
    if val is None:
        return None
    return 1 if val else 0


# ---------------------------------------------------------------------------
# CycleRecorder
# ---------------------------------------------------------------------------


class CycleRecorder:
    """Append-only SQLite recorder for engine cycle state.

    Usage::

        recorder = CycleRecorder("/path/to/recording.db", settings)
        recorder.open()

        cid = recorder.begin_cycle(1, time.time())
        recorder.record_market_snapshots(cid, quotes, spots, vpins, ofis, rets)
        edge_ids = recorder.record_edges(cid, edges, filter_results)
        recorder.record_trade(cid, edge_ids["KXBTC15M-..."], "KXBTC15M-...", "yes", 5, 0.42)
        recorder.end_cycle(cid, elapsed_ms=120.3, ...)

        recorder.close()
    """

    def __init__(self, db_path: str, settings: Any) -> None:
        self._db_path = db_path
        self._settings = settings
        self._conn: Optional[sqlite3.Connection] = None
        self._session_id: Optional[int] = None

    # -- lifecycle ----------------------------------------------------------

    def open(self) -> None:
        """Open the database, create tables, and write a session row."""
        if self._conn is not None:
            return
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

        # Write session row
        settings_json = json.dumps(_safe_settings_dict(self._settings))
        symbols_json = "[]"
        try:
            symbols_json = json.dumps(list(self._settings.symbols))
        except Exception:
            pass

        cursor = self._conn.execute(
            """INSERT INTO sessions (start_time, settings_json, symbols_json)
               VALUES (?, ?, ?)""",
            (time.time(), settings_json, symbols_json),
        )
        self._session_id = cursor.lastrowid
        self._conn.commit()

    def close(self) -> None:
        """Set session end_time and close the connection."""
        if self._conn is not None:
            if self._session_id is not None:
                try:
                    self._conn.execute(
                        "UPDATE sessions SET end_time = ? WHERE id = ?",
                        (time.time(), self._session_id),
                    )
                    self._conn.commit()
                except Exception:
                    LOGGER.warning("Failed to update session end_time", exc_info=True)
            self._conn.close()
            self._conn = None

    def _ensure_open(self) -> sqlite3.Connection:
        if self._conn is None:
            self.open()
        assert self._conn is not None
        return self._conn

    # -- cycle bracket ------------------------------------------------------

    def begin_cycle(self, cycle_number: int, timestamp: float) -> int:
        """Begin a new cycle. Returns the cycle_id (row id).

        Opens an implicit transaction that is committed by ``end_cycle``.
        """
        conn = self._ensure_open()
        cursor = conn.execute(
            """INSERT INTO cycles (session_id, cycle_number, timestamp)
               VALUES (?, ?, ?)""",
            (self._session_id, cycle_number, timestamp),
        )
        cycle_id = cursor.lastrowid
        assert cycle_id is not None
        return cycle_id

    def end_cycle(
        self,
        cycle_id: int,
        elapsed_ms: float,
        regime: Optional[str],
        regime_confidence: Optional[float],
        regime_is_transitioning: Optional[bool],
        num_quotes: int,
        num_edges_raw: int,
        num_edges_final: int,
        num_trades: int,
        session_pnl: float,
        bankroll: float,
    ) -> None:
        """Finalize the cycle row with summary stats and commit."""
        conn = self._ensure_open()
        conn.execute(
            """UPDATE cycles SET
                 elapsed_ms = ?,
                 regime = ?,
                 regime_confidence = ?,
                 regime_is_transitioning = ?,
                 num_quotes = ?,
                 num_edges_raw = ?,
                 num_edges_final = ?,
                 num_trades = ?,
                 session_pnl = ?,
                 bankroll = ?
               WHERE id = ?""",
            (
                elapsed_ms,
                regime,
                regime_confidence,
                _bool_to_int(regime_is_transitioning),
                num_quotes,
                num_edges_raw,
                num_edges_final,
                num_trades,
                session_pnl,
                bankroll,
                cycle_id,
            ),
        )
        conn.commit()

    # -- market snapshots ---------------------------------------------------

    def record_market_snapshots(
        self,
        cycle_id: int,
        quotes: List[Any],
        spot_prices: Dict[str, float],
        vpin_values: Dict[str, Tuple[float, float, float]],
        ofi_values: Dict[str, Dict[int, float]],
        returns: Dict[str, List[float]],
    ) -> None:
        """Bulk-insert market snapshots for a cycle.

        Parameters
        ----------
        quotes : list[CryptoMarketQuote]
        spot_prices : {underlying: price}
        vpin_values : {underlying: (vpin, signed_vpin, trend)}
        ofi_values : {underlying: {30: val, 60: val, 120: val, 300: val}}
        returns : {underlying: [log_return_1, log_return_2, ...]}
        """
        conn = self._ensure_open()
        rows = []
        for q in quotes:
            ticker = q.market.ticker
            underlying = q.market.meta.underlying
            interval = q.market.meta.interval
            direction = q.market.meta.direction
            strike = q.market.meta.strike

            spot = spot_prices.get(underlying)
            vpin_tuple = vpin_values.get(underlying, (None, None, None))
            vpin_val = vpin_tuple[0] if vpin_tuple[0] is not None else None
            signed_vpin = vpin_tuple[1] if len(vpin_tuple) > 1 and vpin_tuple[1] is not None else None
            vpin_trend = vpin_tuple[2] if len(vpin_tuple) > 2 and vpin_tuple[2] is not None else None

            ofi = ofi_values.get(underlying, {})
            rets = returns.get(underlying, [])

            rows.append((
                cycle_id,
                ticker,
                underlying,
                interval,
                direction,
                strike,
                q.time_to_expiry_minutes,
                q.yes_buy_price,
                q.no_buy_price,
                q.yes_buy_size,
                q.no_buy_size,
                q.yes_bid_price,
                q.no_bid_price,
                q.implied_probability,
                spot,
                vpin_val,
                signed_vpin,
                vpin_trend,
                ofi.get(30),
                ofi.get(60),
                ofi.get(120),
                ofi.get(300),
                json.dumps(rets) if rets else "[]",
            ))

        if rows:
            conn.executemany(
                """INSERT INTO market_snapshots
                   (cycle_id, ticker, underlying, interval, direction, strike,
                    time_to_expiry_minutes, yes_buy_price, no_buy_price,
                    yes_buy_size, no_buy_size, yes_bid_price, no_bid_price,
                    implied_probability, spot_price,
                    vpin, signed_vpin, vpin_trend,
                    ofi_30, ofi_60, ofi_120, ofi_300,
                    returns_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )

    # -- edges --------------------------------------------------------------

    def record_edges(
        self,
        cycle_id: int,
        edges: List[Any],
        filter_results: Dict[str, "FilterResult"],
    ) -> Dict[str, int]:
        """Insert edges with filter decision columns. Returns {ticker: edge_id}."""
        conn = self._ensure_open()
        result_map: Dict[str, int] = {}

        for edge in edges:
            ticker = edge.market.ticker
            fr = filter_results.get(ticker, FilterResult())

            cursor = conn.execute(
                """INSERT INTO edges
                   (cycle_id, ticker, side, model_prob, market_implied_prob,
                    blended_prob, edge_cents, model_uncertainty,
                    time_to_expiry_minutes, yes_buy_price, no_buy_price,
                    spread_cost, staleness_score,
                    passed_regime_min_edge, passed_zscore,
                    passed_counter_trend, passed_confidence,
                    zscore_value, confidence_score, confidence_agreement,
                    filter_reject_reason, survived_all_filters,
                    contracts_sized, was_traded)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    cycle_id,
                    ticker,
                    edge.side,
                    edge.model_prob.probability if hasattr(edge.model_prob, "probability") else edge.model_prob,
                    edge.market_implied_prob,
                    edge.blended_probability,
                    edge.edge_cents,
                    edge.model_uncertainty,
                    edge.time_to_expiry_minutes,
                    edge.yes_buy_price,
                    edge.no_buy_price,
                    edge.spread_cost,
                    edge.staleness_score,
                    _bool_to_int(fr.passed_regime_min_edge),
                    _bool_to_int(fr.passed_zscore),
                    _bool_to_int(fr.passed_counter_trend),
                    _bool_to_int(fr.passed_confidence),
                    fr.zscore_value,
                    fr.confidence_score,
                    fr.confidence_agreement,
                    fr.reject_reason,
                    _bool_to_int(fr.survived_all),
                    fr.contracts_sized,
                    _bool_to_int(fr.was_traded),
                ),
            )
            edge_id = cursor.lastrowid
            assert edge_id is not None
            result_map[ticker] = edge_id

        return result_map

    # -- trades -------------------------------------------------------------

    def record_trade(
        self,
        cycle_id: int,
        edge_id: int,
        ticker: str,
        side: str,
        contracts: int,
        entry_price: float,
    ) -> int:
        """Record a trade execution. Returns the trade_id."""
        conn = self._ensure_open()
        cursor = conn.execute(
            """INSERT INTO trades
               (cycle_id, edge_id, ticker, side, contracts, entry_price)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (cycle_id, edge_id, ticker, side, contracts, entry_price),
        )
        trade_id = cursor.lastrowid
        assert trade_id is not None
        return trade_id

    # -- settlement ---------------------------------------------------------

    def record_settlement(
        self,
        ticker: str,
        actual_outcome: str,
        pnl: float,
        settlement_time: float,
    ) -> None:
        """Update the most recent unsettled trade for this ticker."""
        conn = self._ensure_open()
        # Find the most recent unsettled trade for this ticker
        row = conn.execute(
            """SELECT id FROM trades
               WHERE ticker = ? AND settled = 0
               ORDER BY id DESC LIMIT 1""",
            (ticker,),
        ).fetchone()
        if row is None:
            LOGGER.warning("No unsettled trade found for ticker %s", ticker)
            return
        conn.execute(
            """UPDATE trades SET
                 settled = 1,
                 actual_outcome = ?,
                 pnl = ?,
                 settlement_time = ?
               WHERE id = ?""",
            (actual_outcome, pnl, settlement_time, row[0]),
        )
        conn.commit()

    # -- flush --------------------------------------------------------------

    def flush(self) -> None:
        """Force a commit on the current connection."""
        conn = self._ensure_open()
        conn.commit()
