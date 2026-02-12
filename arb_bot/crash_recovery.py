"""Crash recovery startup logic (Phase 0F).

On boot, reads the persistent order store and rebuilds engine state.
Detects unhedged positions (partially filled intents) and enters safe mode
to block new trades until the operator acknowledges.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from arb_bot.models import EngineState
from arb_bot.order_store import IntentStatus, OrderStore

LOGGER = logging.getLogger(__name__)


@dataclass
class RecoveryReport:
    """Summary of what was recovered from the persistent store."""
    open_intents: int = 0
    partially_filled_intents: int = 0
    open_positions: int = 0
    locked_capital_by_venue: dict[str, float] = field(default_factory=dict)
    open_markets_by_venue: dict[str, set[str]] = field(default_factory=dict)
    unhedged_exposure: list[dict[str, Any]] = field(default_factory=list)
    safe_mode: bool = False
    safe_mode_reason: str = ""


def recover_state(
    store: OrderStore,
    initial_cash: dict[str, float],
) -> tuple[EngineState, RecoveryReport]:
    """Rebuild EngineState from the persistent store.

    Parameters
    ----------
    store:
        The persistent order store (must already be open).
    initial_cash:
        Default cash balances per venue (from config or exchange API).

    Returns
    -------
    Tuple of (rebuilt EngineState, RecoveryReport).
    If unhedged positions are found, ``report.safe_mode`` is True.
    """
    report = RecoveryReport()

    # Rebuild locked capital and open markets from positions.
    locked = store.compute_locked_capital_by_venue()
    open_markets = store.compute_open_markets_by_venue()

    report.locked_capital_by_venue = dict(locked)
    report.open_markets_by_venue = {v: set(m) for v, m in open_markets.items()}

    # Adjust cash for locked capital.
    cash = dict(initial_cash)
    for venue, amount in locked.items():
        cash[venue] = max(0.0, cash.get(venue, 0.0) - amount)

    # Check for open intents (pending/submitted but not finalized).
    open_intents = store.get_open_intents()
    report.open_intents = len(open_intents)

    # Check for partially filled intents (unhedged exposure).
    partial_intents = store.get_partially_filled_intents()
    report.partially_filled_intents = len(partial_intents)

    for intent in partial_intents:
        exposure = store.compute_hedge_exposure(intent.intent_id)
        if exposure["exposed"] > 0:
            report.unhedged_exposure.append({
                "intent_id": intent.intent_id,
                "kind": intent.kind,
                **exposure,
            })

    # Check open positions.
    open_positions = store.get_open_positions()
    report.open_positions = len(open_positions)

    # Enter safe mode if there are unhedged positions or dangling open intents.
    reasons = []
    if report.unhedged_exposure:
        reasons.append(
            f"{len(report.unhedged_exposure)} unhedged position(s)"
        )
    if report.open_intents > 0:
        reasons.append(f"{report.open_intents} open intent(s) from prior session")

    if reasons:
        report.safe_mode = True
        report.safe_mode_reason = "; ".join(reasons)
        LOGGER.warning(
            "SAFE MODE ACTIVATED: %s — no new trades until operator acknowledges",
            report.safe_mode_reason,
        )

    state = EngineState(
        cash_by_venue=cash,
        locked_capital_by_venue=dict(locked),
        open_markets_by_venue={v: set(m) for v, m in open_markets.items()},
    )

    LOGGER.info(
        "Recovery complete: %d open intents, %d partial fills, %d positions, safe_mode=%s",
        report.open_intents,
        report.partially_filled_intents,
        report.open_positions,
        report.safe_mode,
    )

    return state, report
