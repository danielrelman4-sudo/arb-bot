"""Long-run storage + analytics pipeline (Phase 2G).

Provides a lightweight analytics store for recording decisions, fills,
and run summaries. Uses SQLite (no DuckDB dependency) with structured
tables for daily summaries, lane conversion tracking, and fill-quality
drift monitoring.

The store is append-only for decision and fill records, with periodic
summary aggregation. Designed for offline analysis and KPI tracking.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalyticsConfig:
    """Configuration for the analytics store.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file. Default "analytics.db".
    retention_days:
        Number of days to retain raw records. 0 = no expiry. Default 90.
    enabled:
        Master enable/disable. Default True.
    """

    db_path: str = "analytics.db"
    retention_days: int = 90
    enabled: bool = True


# ---------------------------------------------------------------------------
# Record types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecisionRecord:
    """A single opportunity detection + sizing decision."""

    timestamp: float
    kind: str            # e.g., "cross_venue", "intra_venue"
    execution_style: str  # e.g., "taker", "maker_estimate"
    market_key: str
    gross_edge: float
    net_edge: float
    contracts: int
    capital_required: float
    accepted: bool       # Whether the plan passed risk gates.
    reject_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FillRecord:
    """A completed (or failed) execution record."""

    timestamp: float
    intent_id: str
    kind: str
    market_key: str
    planned_contracts: int
    filled_contracts: int
    planned_edge: float
    realized_pnl: float
    slippage: float      # realized - planned per contract
    fill_rate: float     # filled / planned
    venue: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DailySummary:
    """Aggregated daily summary."""

    date: str            # ISO date "2025-06-01"
    total_decisions: int
    total_accepted: int
    total_rejected: int
    total_fills: int
    total_contracts: int
    total_realized_pnl: float
    avg_fill_rate: float
    avg_slippage: float
    decisions_by_kind: Dict[str, int] = field(default_factory=dict)
    pnl_by_kind: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Analytics store
# ---------------------------------------------------------------------------


class AnalyticsStore:
    """SQLite-backed analytics store for long-run KPI tracking.

    Usage::

        store = AnalyticsStore(AnalyticsConfig())
        store.record_decision(decision)
        store.record_fill(fill)
        summary = store.daily_summary("2025-06-01")
    """

    def __init__(self, config: AnalyticsConfig | None = None) -> None:
        self._config = config or AnalyticsConfig()
        self._conn: Optional[sqlite3.Connection] = None

    @property
    def config(self) -> AnalyticsConfig:
        return self._config

    def open(self) -> None:
        """Open the database connection and create tables."""
        if self._conn is not None:
            return
        self._conn = sqlite3.connect(self._config.db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _ensure_open(self) -> sqlite3.Connection:
        if self._conn is None:
            self.open()
        assert self._conn is not None
        return self._conn

    def _create_tables(self) -> None:
        conn = self._ensure_open()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                date TEXT NOT NULL,
                kind TEXT NOT NULL,
                execution_style TEXT NOT NULL,
                market_key TEXT NOT NULL,
                gross_edge REAL NOT NULL,
                net_edge REAL NOT NULL,
                contracts INTEGER NOT NULL,
                capital_required REAL NOT NULL,
                accepted INTEGER NOT NULL,
                reject_reason TEXT NOT NULL DEFAULT '',
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_decisions_date ON decisions(date);
            CREATE INDEX IF NOT EXISTS idx_decisions_kind ON decisions(kind);

            CREATE TABLE IF NOT EXISTS fills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                date TEXT NOT NULL,
                intent_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                market_key TEXT NOT NULL,
                planned_contracts INTEGER NOT NULL,
                filled_contracts INTEGER NOT NULL,
                planned_edge REAL NOT NULL,
                realized_pnl REAL NOT NULL,
                slippage REAL NOT NULL,
                fill_rate REAL NOT NULL,
                venue TEXT NOT NULL DEFAULT '',
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_fills_date ON fills(date);
            CREATE INDEX IF NOT EXISTS idx_fills_intent ON fills(intent_id);
        """)
        conn.commit()

    def record_decision(self, record: DecisionRecord) -> None:
        """Store a decision record."""
        if not self._config.enabled:
            return
        conn = self._ensure_open()
        date = datetime.fromtimestamp(record.timestamp, tz=timezone.utc).strftime("%Y-%m-%d")
        conn.execute(
            """INSERT INTO decisions
               (timestamp, date, kind, execution_style, market_key,
                gross_edge, net_edge, contracts, capital_required,
                accepted, reject_reason, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.timestamp,
                date,
                record.kind,
                record.execution_style,
                record.market_key,
                record.gross_edge,
                record.net_edge,
                record.contracts,
                record.capital_required,
                1 if record.accepted else 0,
                record.reject_reason,
                json.dumps(record.metadata),
            ),
        )
        conn.commit()

    def record_fill(self, record: FillRecord) -> None:
        """Store a fill record."""
        if not self._config.enabled:
            return
        conn = self._ensure_open()
        date = datetime.fromtimestamp(record.timestamp, tz=timezone.utc).strftime("%Y-%m-%d")
        conn.execute(
            """INSERT INTO fills
               (timestamp, date, intent_id, kind, market_key,
                planned_contracts, filled_contracts, planned_edge,
                realized_pnl, slippage, fill_rate, venue, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.timestamp,
                date,
                record.intent_id,
                record.kind,
                record.market_key,
                record.planned_contracts,
                record.filled_contracts,
                record.planned_edge,
                record.realized_pnl,
                record.slippage,
                record.fill_rate,
                record.venue,
                json.dumps(record.metadata),
            ),
        )
        conn.commit()

    def daily_summary(self, date: str) -> DailySummary:
        """Compute summary for a given date (ISO format "YYYY-MM-DD")."""
        conn = self._ensure_open()

        # Decision stats.
        row = conn.execute(
            """SELECT
                 COUNT(*),
                 SUM(CASE WHEN accepted=1 THEN 1 ELSE 0 END),
                 SUM(CASE WHEN accepted=0 THEN 1 ELSE 0 END)
               FROM decisions WHERE date=?""",
            (date,),
        ).fetchone()
        total_decisions = row[0] or 0
        total_accepted = row[1] or 0
        total_rejected = row[2] or 0

        # Decisions by kind.
        decisions_by_kind: Dict[str, int] = {}
        for kind_row in conn.execute(
            "SELECT kind, COUNT(*) FROM decisions WHERE date=? GROUP BY kind",
            (date,),
        ):
            decisions_by_kind[kind_row[0]] = kind_row[1]

        # Fill stats.
        fill_row = conn.execute(
            """SELECT
                 COUNT(*),
                 COALESCE(SUM(filled_contracts), 0),
                 COALESCE(SUM(realized_pnl), 0),
                 COALESCE(AVG(fill_rate), 0),
                 COALESCE(AVG(slippage), 0)
               FROM fills WHERE date=?""",
            (date,),
        ).fetchone()
        total_fills = fill_row[0] or 0
        total_contracts = fill_row[1] or 0
        total_realized_pnl = fill_row[2] or 0.0
        avg_fill_rate = fill_row[3] or 0.0
        avg_slippage = fill_row[4] or 0.0

        # PnL by kind.
        pnl_by_kind: Dict[str, float] = {}
        for pnl_row in conn.execute(
            "SELECT kind, SUM(realized_pnl) FROM fills WHERE date=? GROUP BY kind",
            (date,),
        ):
            pnl_by_kind[pnl_row[0]] = pnl_row[1]

        return DailySummary(
            date=date,
            total_decisions=total_decisions,
            total_accepted=total_accepted,
            total_rejected=total_rejected,
            total_fills=total_fills,
            total_contracts=total_contracts,
            total_realized_pnl=total_realized_pnl,
            avg_fill_rate=avg_fill_rate,
            avg_slippage=avg_slippage,
            decisions_by_kind=decisions_by_kind,
            pnl_by_kind=pnl_by_kind,
        )

    def decision_count(self, date: str | None = None) -> int:
        """Count decisions, optionally filtered by date."""
        conn = self._ensure_open()
        if date:
            row = conn.execute(
                "SELECT COUNT(*) FROM decisions WHERE date=?", (date,)
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()
        return row[0] or 0

    def fill_count(self, date: str | None = None) -> int:
        """Count fills, optionally filtered by date."""
        conn = self._ensure_open()
        if date:
            row = conn.execute(
                "SELECT COUNT(*) FROM fills WHERE date=?", (date,)
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) FROM fills").fetchone()
        return row[0] or 0

    def purge_old_records(self, before_date: str) -> tuple[int, int]:
        """Remove records older than the given date.

        Returns (decisions_removed, fills_removed).
        """
        conn = self._ensure_open()
        d_cursor = conn.execute("DELETE FROM decisions WHERE date < ?", (before_date,))
        f_cursor = conn.execute("DELETE FROM fills WHERE date < ?", (before_date,))
        conn.commit()
        return d_cursor.rowcount, f_cursor.rowcount
