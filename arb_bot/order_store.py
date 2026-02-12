"""Persistent SQLite store for order intents, leg executions, and positions.

Provides crash-safe order tracking: write intent BEFORE submitting to exchange,
update on fill/fail, and recover state on restart by reading the store.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

LOGGER = logging.getLogger(__name__)

DB_FILENAME = "arb_orders.db"

SCHEMA_VERSION = 1

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS order_intents (
    intent_id       TEXT PRIMARY KEY,
    plan_json       TEXT NOT NULL,
    kind            TEXT NOT NULL,
    execution_style TEXT NOT NULL,
    contracts       INTEGER NOT NULL,
    capital_required REAL NOT NULL,
    capital_required_by_venue TEXT NOT NULL,
    expected_profit REAL NOT NULL,
    edge_per_contract REAL NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    total_filled    INTEGER NOT NULL DEFAULT 0,
    error           TEXT,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS leg_executions (
    leg_id          TEXT PRIMARY KEY,
    intent_id       TEXT NOT NULL REFERENCES order_intents(intent_id),
    leg_index       INTEGER NOT NULL,
    venue           TEXT NOT NULL,
    market_id       TEXT NOT NULL,
    side            TEXT NOT NULL,
    contracts       INTEGER NOT NULL,
    limit_price     REAL NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    order_id        TEXT,
    filled_contracts INTEGER NOT NULL DEFAULT 0,
    average_price   REAL,
    raw_json        TEXT,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS positions (
    position_id     TEXT PRIMARY KEY,
    intent_id       TEXT NOT NULL REFERENCES order_intents(intent_id),
    venue           TEXT NOT NULL,
    market_id       TEXT NOT NULL,
    side            TEXT NOT NULL,
    net_contracts   INTEGER NOT NULL,
    average_entry   REAL NOT NULL,
    status          TEXT NOT NULL DEFAULT 'open',
    opened_at       REAL NOT NULL,
    closed_at       REAL
);

CREATE INDEX IF NOT EXISTS idx_order_intents_status ON order_intents(status);
CREATE INDEX IF NOT EXISTS idx_leg_executions_intent ON leg_executions(intent_id);
CREATE INDEX IF NOT EXISTS idx_leg_executions_status ON leg_executions(status);
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
CREATE INDEX IF NOT EXISTS idx_positions_venue ON positions(venue, market_id);
"""


class IntentStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    FAILED = "failed"
    CANCELLED = "cancelled"


class LegStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PositionStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"


@dataclass
class StoredIntent:
    intent_id: str
    plan_json: str
    kind: str
    execution_style: str
    contracts: int
    capital_required: float
    capital_required_by_venue: dict[str, float]
    expected_profit: float
    edge_per_contract: float
    status: str
    total_filled: int
    error: str | None
    created_at: float
    updated_at: float


@dataclass
class StoredLeg:
    leg_id: str
    intent_id: str
    leg_index: int
    venue: str
    market_id: str
    side: str
    contracts: int
    limit_price: float
    status: str
    order_id: str | None
    filled_contracts: int
    average_price: float | None
    raw_json: str | None
    created_at: float
    updated_at: float


@dataclass
class StoredPosition:
    position_id: str
    intent_id: str
    venue: str
    market_id: str
    side: str
    net_contracts: int
    average_entry: float
    status: str
    opened_at: float
    closed_at: float | None


class OrderStore:
    """SQLite-backed persistent store for order lifecycle tracking."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            db_path = Path("data") / DB_FILENAME
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._open()

    def _open(self) -> None:
        self._conn = sqlite3.connect(
            str(self._db_path),
            isolation_level="DEFERRED",
            check_same_thread=False,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._apply_schema()

    def _apply_schema(self) -> None:
        assert self._conn is not None
        cur = self._conn.cursor()
        cur.executescript(_SCHEMA_SQL)
        row = cur.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
        if row is None:
            cur.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
        self._conn.commit()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Cursor]:
        assert self._conn is not None
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Intent lifecycle
    # ------------------------------------------------------------------

    def create_intent(self, plan: Any) -> str:
        """Write an order intent BEFORE submitting to exchange. Returns intent_id."""
        intent_id = uuid.uuid4().hex
        now = time.time()

        plan_dict = _plan_to_dict(plan)
        plan_json = json.dumps(plan_dict, default=str)
        cap_by_venue_json = json.dumps(plan.capital_required_by_venue)

        with self._tx() as cur:
            cur.execute(
                """INSERT INTO order_intents
                   (intent_id, plan_json, kind, execution_style, contracts,
                    capital_required, capital_required_by_venue, expected_profit,
                    edge_per_contract, status, total_filled, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    intent_id,
                    plan_json,
                    plan.kind.value,
                    plan.execution_style.value,
                    plan.contracts,
                    plan.capital_required,
                    cap_by_venue_json,
                    plan.expected_profit,
                    plan.edge_per_contract,
                    IntentStatus.PENDING.value,
                    0,
                    now,
                    now,
                ),
            )

            for idx, leg in enumerate(plan.legs):
                leg_id = uuid.uuid4().hex
                cur.execute(
                    """INSERT INTO leg_executions
                       (leg_id, intent_id, leg_index, venue, market_id, side,
                        contracts, limit_price, status, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        leg_id,
                        intent_id,
                        idx,
                        leg.venue,
                        leg.market_id,
                        leg.side.value if hasattr(leg.side, "value") else str(leg.side),
                        leg.contracts,
                        leg.limit_price,
                        LegStatus.PENDING.value,
                        now,
                        now,
                    ),
                )

        LOGGER.debug("Created intent %s for %s (%d legs)", intent_id, plan.kind.value, len(plan.legs))
        return intent_id

    def mark_intent_submitted(self, intent_id: str) -> None:
        now = time.time()
        with self._tx() as cur:
            cur.execute(
                "UPDATE order_intents SET status = ?, updated_at = ? WHERE intent_id = ?",
                (IntentStatus.SUBMITTED.value, now, intent_id),
            )

    def mark_leg_submitted(self, intent_id: str, leg_index: int, order_id: str | None = None) -> None:
        now = time.time()
        with self._tx() as cur:
            cur.execute(
                """UPDATE leg_executions SET status = ?, order_id = ?, updated_at = ?
                   WHERE intent_id = ? AND leg_index = ?""",
                (LegStatus.SUBMITTED.value, order_id, now, intent_id, leg_index),
            )

    def record_leg_result(
        self,
        intent_id: str,
        leg_index: int,
        *,
        success: bool,
        order_id: str | None = None,
        filled_contracts: int = 0,
        average_price: float | None = None,
        raw: dict[str, Any] | None = None,
    ) -> None:
        now = time.time()
        if success and filled_contracts > 0:
            leg_status = LegStatus.FILLED.value
        elif success and filled_contracts == 0:
            leg_status = LegStatus.PARTIALLY_FILLED.value
        else:
            leg_status = LegStatus.FAILED.value

        raw_json = json.dumps(raw, default=str) if raw else None

        with self._tx() as cur:
            cur.execute(
                """UPDATE leg_executions
                   SET status = ?, order_id = COALESCE(?, order_id),
                       filled_contracts = ?, average_price = ?,
                       raw_json = ?, updated_at = ?
                   WHERE intent_id = ? AND leg_index = ?""",
                (leg_status, order_id, filled_contracts, average_price,
                 raw_json, now, intent_id, leg_index),
            )

    def finalize_intent(self, intent_id: str) -> StoredIntent:
        """Compute final intent status from leg results and update. Returns the updated intent."""
        now = time.time()
        with self._tx() as cur:
            legs = self._fetch_legs(cur, intent_id)

            all_filled = all(l.status == LegStatus.FILLED.value for l in legs)
            any_failed = any(l.status == LegStatus.FAILED.value for l in legs)
            total_filled = min(l.filled_contracts for l in legs) if legs else 0

            if all_filled and total_filled > 0:
                status = IntentStatus.FILLED.value
            elif any_failed:
                status = IntentStatus.FAILED.value
                if any(l.filled_contracts > 0 for l in legs):
                    status = IntentStatus.PARTIALLY_FILLED.value
            elif total_filled > 0:
                status = IntentStatus.PARTIALLY_FILLED.value
            else:
                status = IntentStatus.FAILED.value

            error = None
            if status in (IntentStatus.FAILED.value, IntentStatus.PARTIALLY_FILLED.value):
                failed_legs = [l for l in legs if l.status == LegStatus.FAILED.value]
                if failed_legs:
                    errors = []
                    for fl in failed_legs:
                        if fl.raw_json:
                            raw = json.loads(fl.raw_json)
                            errors.append(raw.get("error", "unknown"))
                        else:
                            errors.append("unknown")
                    error = "; ".join(errors)

            cur.execute(
                """UPDATE order_intents
                   SET status = ?, total_filled = ?, error = ?, updated_at = ?
                   WHERE intent_id = ?""",
                (status, total_filled, error, now, intent_id),
            )

        return self.get_intent(intent_id)

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def open_position(
        self,
        intent_id: str,
        venue: str,
        market_id: str,
        side: str,
        net_contracts: int,
        average_entry: float,
    ) -> str:
        position_id = uuid.uuid4().hex
        now = time.time()
        with self._tx() as cur:
            cur.execute(
                """INSERT INTO positions
                   (position_id, intent_id, venue, market_id, side,
                    net_contracts, average_entry, status, opened_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (position_id, intent_id, venue, market_id, side,
                 net_contracts, average_entry, PositionStatus.OPEN.value, now),
            )
        return position_id

    def close_position(self, position_id: str) -> None:
        now = time.time()
        with self._tx() as cur:
            cur.execute(
                "UPDATE positions SET status = ?, closed_at = ? WHERE position_id = ?",
                (PositionStatus.CLOSED.value, now, position_id),
            )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_intent(self, intent_id: str) -> StoredIntent:
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT * FROM order_intents WHERE intent_id = ?", (intent_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Intent {intent_id} not found")
        return _row_to_intent(row)

    def get_legs(self, intent_id: str) -> list[StoredLeg]:
        assert self._conn is not None
        rows = self._conn.execute(
            "SELECT * FROM leg_executions WHERE intent_id = ? ORDER BY leg_index",
            (intent_id,),
        ).fetchall()
        return [_row_to_leg(r) for r in rows]

    def get_open_intents(self) -> list[StoredIntent]:
        assert self._conn is not None
        rows = self._conn.execute(
            "SELECT * FROM order_intents WHERE status IN (?, ?) ORDER BY created_at",
            (IntentStatus.PENDING.value, IntentStatus.SUBMITTED.value),
        ).fetchall()
        return [_row_to_intent(r) for r in rows]

    def get_open_positions(self) -> list[StoredPosition]:
        assert self._conn is not None
        rows = self._conn.execute(
            "SELECT * FROM positions WHERE status = ? ORDER BY opened_at",
            (PositionStatus.OPEN.value,),
        ).fetchall()
        return [_row_to_position(r) for r in rows]

    def get_positions_by_venue(self, venue: str) -> list[StoredPosition]:
        assert self._conn is not None
        rows = self._conn.execute(
            "SELECT * FROM positions WHERE venue = ? AND status = ? ORDER BY opened_at",
            (venue, PositionStatus.OPEN.value),
        ).fetchall()
        return [_row_to_position(r) for r in rows]

    def get_filled_intents_since(self, since_ts: float) -> list[StoredIntent]:
        assert self._conn is not None
        rows = self._conn.execute(
            "SELECT * FROM order_intents WHERE status = ? AND updated_at >= ? ORDER BY updated_at",
            (IntentStatus.FILLED.value, since_ts),
        ).fetchall()
        return [_row_to_intent(r) for r in rows]

    def compute_locked_capital_by_venue(self) -> dict[str, float]:
        """Sum capital committed in open positions by venue."""
        positions = self.get_open_positions()
        locked: dict[str, float] = {}
        for pos in positions:
            locked[pos.venue] = locked.get(pos.venue, 0.0) + (pos.net_contracts * pos.average_entry)
        return locked

    def compute_open_markets_by_venue(self) -> dict[str, set[str]]:
        """Get set of market IDs with open positions by venue."""
        positions = self.get_open_positions()
        markets: dict[str, set[str]] = {}
        for pos in positions:
            markets.setdefault(pos.venue, set()).add(pos.market_id)
        return markets

    def _fetch_legs(self, cur: sqlite3.Cursor, intent_id: str) -> list[StoredLeg]:
        rows = cur.execute(
            "SELECT * FROM leg_executions WHERE intent_id = ? ORDER BY leg_index",
            (intent_id,),
        ).fetchall()
        return [_row_to_leg(r) for r in rows]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _plan_to_dict(plan: Any) -> dict[str, Any]:
    legs = []
    for leg in plan.legs:
        legs.append({
            "venue": leg.venue,
            "market_id": leg.market_id,
            "side": leg.side.value if hasattr(leg.side, "value") else str(leg.side),
            "contracts": leg.contracts,
            "limit_price": leg.limit_price,
            "metadata": leg.metadata if hasattr(leg, "metadata") else {},
        })
    return {
        "kind": plan.kind.value,
        "execution_style": plan.execution_style.value,
        "legs": legs,
        "contracts": plan.contracts,
        "capital_required": plan.capital_required,
        "capital_required_by_venue": plan.capital_required_by_venue,
        "expected_profit": plan.expected_profit,
        "edge_per_contract": plan.edge_per_contract,
        "metadata": plan.metadata if hasattr(plan, "metadata") else {},
    }


def _row_to_intent(row: tuple) -> StoredIntent:
    return StoredIntent(
        intent_id=row[0],
        plan_json=row[1],
        kind=row[2],
        execution_style=row[3],
        contracts=row[4],
        capital_required=row[5],
        capital_required_by_venue=json.loads(row[6]),
        expected_profit=row[7],
        edge_per_contract=row[8],
        status=row[9],
        total_filled=row[10],
        error=row[11],
        created_at=row[12],
        updated_at=row[13],
    )


def _row_to_leg(row: tuple) -> StoredLeg:
    return StoredLeg(
        leg_id=row[0],
        intent_id=row[1],
        leg_index=row[2],
        venue=row[3],
        market_id=row[4],
        side=row[5],
        contracts=row[6],
        limit_price=row[7],
        status=row[8],
        order_id=row[9],
        filled_contracts=row[10],
        average_price=row[11],
        raw_json=row[12],
        created_at=row[13],
        updated_at=row[14],
    )


def _row_to_position(row: tuple) -> StoredPosition:
    return StoredPosition(
        position_id=row[0],
        intent_id=row[1],
        venue=row[2],
        market_id=row[3],
        side=row[4],
        net_contracts=row[5],
        average_entry=row[6],
        status=row[7],
        opened_at=row[8],
        closed_at=row[9],
    )
