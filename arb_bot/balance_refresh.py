"""Automatic balance refresh with stale capital reconciliation (Phase 0G).

Provides periodic venue balance refresh on a timer and auto-release of
stale locked capital by reconciling actual exchange balance against the
engine's in-memory locked capital tracking.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from arb_bot.models import EngineState

LOGGER = logging.getLogger(__name__)


@dataclass
class BalanceSnapshot:
    """Point-in-time balance snapshot from a venue."""
    venue: str
    available_cash: float
    fetched_at: float


@dataclass
class ReconciliationResult:
    """Outcome of reconciling engine state against exchange balances."""
    venue: str
    exchange_balance: float
    engine_cash: float
    engine_locked: float
    engine_total: float
    discrepancy: float
    released_locked: float = 0.0


class BalanceRefresher:
    """Periodically refreshes venue balances and reconciles locked capital.

    Parameters
    ----------
    refresh_interval:
        Seconds between balance refreshes. Default 60s.
    stale_threshold:
        If locked capital exceeds exchange balance by more than this
        fraction, release the excess. Default 0.1 (10%).
    """

    def __init__(
        self,
        refresh_interval: float = 60.0,
        stale_threshold: float = 0.1,
    ) -> None:
        self._refresh_interval = max(5.0, refresh_interval)
        self._stale_threshold = max(0.0, stale_threshold)
        self._last_refresh_ts: float = -1e9  # ensure first check always triggers
        self._snapshots: dict[str, BalanceSnapshot] = {}

    @property
    def refresh_interval(self) -> float:
        return self._refresh_interval

    @property
    def last_refresh_ts(self) -> float:
        return self._last_refresh_ts

    def should_refresh(self) -> bool:
        """Check if enough time has passed for a refresh."""
        return (time.monotonic() - self._last_refresh_ts) >= self._refresh_interval

    def record_balance(self, venue: str, available_cash: float) -> BalanceSnapshot:
        """Record a balance fetched from an exchange adapter."""
        snap = BalanceSnapshot(
            venue=venue,
            available_cash=available_cash,
            fetched_at=time.monotonic(),
        )
        self._snapshots[venue] = snap
        return snap

    def mark_refreshed(self) -> None:
        """Mark that a refresh cycle has completed."""
        self._last_refresh_ts = time.monotonic()

    def reconcile(self, state: EngineState) -> list[ReconciliationResult]:
        """Reconcile engine state against exchange balances.

        If the engine thinks it has more locked capital than the exchange
        shows as total value, releases the excess (stale locks from
        positions that may have resolved).

        Returns a list of per-venue reconciliation results.
        """
        results = []
        for venue, snap in self._snapshots.items():
            engine_cash = state.cash_for(venue)
            engine_locked = state.locked_for(venue)
            engine_total = engine_cash + engine_locked
            exchange_balance = snap.available_cash

            discrepancy = engine_total - exchange_balance

            result = ReconciliationResult(
                venue=venue,
                exchange_balance=exchange_balance,
                engine_cash=engine_cash,
                engine_locked=engine_locked,
                engine_total=engine_total,
                discrepancy=discrepancy,
            )

            # If engine thinks we have more locked than exchange shows total,
            # release the excess locked capital.
            if engine_locked > 0 and discrepancy > 0:
                threshold_amount = exchange_balance * self._stale_threshold
                if discrepancy > threshold_amount:
                    release = min(engine_locked, discrepancy)
                    state.locked_capital_by_venue[venue] = max(
                        0.0, engine_locked - release
                    )
                    state.cash_by_venue[venue] = exchange_balance - max(
                        0.0, engine_locked - release
                    )
                    result.released_locked = release
                    LOGGER.warning(
                        "Released %.2f stale locked capital on %s "
                        "(exchange=%.2f, engine_total=%.2f)",
                        release, venue, exchange_balance, engine_total,
                    )
                else:
                    # Small discrepancy within threshold — just sync cash.
                    state.cash_by_venue[venue] = max(
                        0.0, exchange_balance - engine_locked
                    )
            elif discrepancy > 0:
                # No locked capital but engine cash > exchange — sync cash down.
                state.cash_by_venue[venue] = exchange_balance
            elif discrepancy < 0:
                # Exchange shows more than engine — update cash upward.
                state.cash_by_venue[venue] = exchange_balance - engine_locked

            results.append(result)

        return results

    def get_snapshot(self, venue: str) -> BalanceSnapshot | None:
        return self._snapshots.get(venue)
