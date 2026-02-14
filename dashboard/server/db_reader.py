from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


def _find_db(name: str) -> str | None:
    """Search common locations for the SQLite database."""
    candidates = [
        Path(name),
        Path("data") / name,
        Path("..") / "data" / name,
        Path("..") / ".." / name,
        Path("..") / ".." / "data" / name,
    ]
    for p in candidates:
        if p.exists():
            return str(p.resolve())
    return None


class DbReader:
    """Read-only access to the bot's SQLite databases."""

    def __init__(
        self,
        analytics_path: str | None = None,
        orders_path: str | None = None,
    ) -> None:
        self._analytics_path = analytics_path or _find_db("analytics.db")
        self._orders_path = orders_path or _find_db("arb_orders.db")

    def _analytics_conn(self) -> sqlite3.Connection:
        if not self._analytics_path:
            raise FileNotFoundError("analytics.db not found")
        conn = sqlite3.connect(self._analytics_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _orders_conn(self) -> sqlite3.Connection:
        if not self._orders_path:
            raise FileNotFoundError("arb_orders.db not found")
        conn = sqlite3.connect(self._orders_path)
        conn.row_factory = sqlite3.Row
        return conn

    # --- Analytics queries ---

    def daily_pnl(self, limit: int = 30) -> list[dict[str, Any]]:
        conn = self._analytics_conn()
        try:
            rows = conn.execute(
                "SELECT date, SUM(realized_pnl) as pnl, COUNT(*) as trades, "
                "SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins, "
                "AVG(slippage) as avg_slippage, AVG(fill_rate) as avg_fill_rate "
                "FROM fills GROUP BY date ORDER BY date DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def pnl_by_lane(self) -> list[dict[str, Any]]:
        conn = self._analytics_conn()
        try:
            rows = conn.execute(
                "SELECT kind, SUM(realized_pnl) as pnl, COUNT(*) as trades, "
                "SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins "
                "FROM fills GROUP BY kind ORDER BY pnl DESC"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def pnl_by_bucket(self, limit: int = 50) -> list[dict[str, Any]]:
        conn = self._analytics_conn()
        try:
            rows = conn.execute(
                "SELECT market_key as group_id, SUM(realized_pnl) as pnl, "
                "COUNT(*) as trades, "
                "SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins, "
                "AVG(realized_pnl) as avg_pnl "
                "FROM fills WHERE kind = 'structural_bucket' "
                "GROUP BY market_key ORDER BY pnl DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def equity_curve(self) -> list[dict[str, Any]]:
        conn = self._analytics_conn()
        try:
            rows = conn.execute(
                "SELECT timestamp, realized_pnl, kind, market_key "
                "FROM fills ORDER BY timestamp ASC"
            ).fetchall()
            curve = []
            cumulative = 0.0
            high_water = 0.0
            for r in rows:
                cumulative += r["realized_pnl"]
                high_water = max(high_water, cumulative)
                curve.append({
                    "ts": r["timestamp"],
                    "pnl": round(cumulative, 4),
                    "trade_pnl": round(r["realized_pnl"], 4),
                    "high_water": round(high_water, 4),
                    "drawdown": round(cumulative - high_water, 4),
                    "kind": r["kind"],
                })
            return curve
        finally:
            conn.close()

    # --- Order queries ---

    def open_positions(self) -> list[dict[str, Any]]:
        conn = self._orders_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM positions WHERE status = 'open' ORDER BY opened_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def recent_trades(self, limit: int = 50) -> list[dict[str, Any]]:
        conn = self._orders_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM order_intents ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()
