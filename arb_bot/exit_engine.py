"""Hold-time / exit discipline engine (Phase 6D).

Close/hedge/abandon logic on edge decay or risk rise, with
time-stop and EV-stop exits.

Usage::

    engine = ExitEngine(config)
    engine.open_position("pos1", entry_edge=0.05, entry_time=100.0)
    action = engine.evaluate("pos1", current_edge=0.02, now=200.0)
    if action.should_exit:
        # Execute exit action.action_type
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Exit action types
# ---------------------------------------------------------------------------


class ExitAction(str, Enum):
    """Type of exit action."""

    HOLD = "hold"  # Continue holding.
    CLOSE = "close"  # Close position.
    HEDGE = "hedge"  # Hedge rather than full close.
    ABANDON = "abandon"  # Abandon (let expire).
    TIME_STOP = "time_stop"  # Time limit reached.
    EV_STOP = "ev_stop"  # EV threshold breached.
    EDGE_DECAY = "edge_decay"  # Edge decayed below minimum.
    RISK_STOP = "risk_stop"  # Risk limit breached.


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExitEngineConfig:
    """Configuration for exit discipline engine.

    Parameters
    ----------
    max_hold_seconds:
        Maximum hold time before time-stop (seconds). Default 3600.
    min_edge:
        Minimum edge to continue holding. Default 0.005.
    edge_decay_ratio:
        Exit if current_edge / entry_edge < this ratio. Default 0.25.
    ev_stop_threshold:
        Exit if estimated EV drops below this. Default -0.02.
    risk_stop_threshold:
        Exit if risk metric exceeds this. Default 0.10.
    hedge_threshold:
        Edge ratio below which to hedge instead of full close.
        Default 0.50.
    abandon_time_ratio:
        Time ratio above which to abandon rather than close
        (if edge is very low). Default 0.90.
    abandon_edge_threshold:
        Max edge at which abandoning is preferred over closing.
        Default 0.001.
    evaluation_cooldown:
        Minimum seconds between evaluations per position.
        Default 5.0.
    """

    max_hold_seconds: float = 3600.0
    min_edge: float = 0.005
    edge_decay_ratio: float = 0.25
    ev_stop_threshold: float = -0.02
    risk_stop_threshold: float = 0.10
    hedge_threshold: float = 0.50
    abandon_time_ratio: float = 0.90
    abandon_edge_threshold: float = 0.001
    evaluation_cooldown: float = 5.0


# ---------------------------------------------------------------------------
# Position state
# ---------------------------------------------------------------------------


@dataclass
class PositionState:
    """State for a tracked position."""

    position_id: str
    entry_edge: float
    entry_time: float
    last_edge: float = 0.0
    last_ev: float = 0.0
    last_risk: float = 0.0
    last_evaluated_at: float = 0.0
    evaluation_count: int = 0
    exited: bool = False
    exit_action: ExitAction = ExitAction.HOLD
    exit_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Evaluation result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExitEvaluation:
    """Result of evaluating a position for exit."""

    position_id: str
    should_exit: bool
    action_type: ExitAction
    current_edge: float
    edge_ratio: float  # current_edge / entry_edge.
    hold_seconds: float
    hold_ratio: float  # hold_seconds / max_hold_seconds.
    reason: str = ""


# ---------------------------------------------------------------------------
# Exit report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExitReport:
    """Summary of all positions."""

    total_positions: int
    open_positions: int
    exited_positions: int
    avg_hold_seconds: float
    exits_by_action: Dict[str, int]


# ---------------------------------------------------------------------------
# Exit engine
# ---------------------------------------------------------------------------


class ExitEngine:
    """Manages position exit discipline.

    Tracks open positions and evaluates them against time-stop,
    EV-stop, edge-decay, and risk-stop criteria to recommend
    close/hedge/abandon actions.
    """

    def __init__(self, config: ExitEngineConfig | None = None) -> None:
        self._config = config or ExitEngineConfig()
        self._positions: Dict[str, PositionState] = {}

    @property
    def config(self) -> ExitEngineConfig:
        return self._config

    def open_position(
        self,
        position_id: str,
        entry_edge: float,
        entry_time: float | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        """Register a new open position."""
        if entry_time is None:
            entry_time = time.time()
        self._positions[position_id] = PositionState(
            position_id=position_id,
            entry_edge=entry_edge,
            entry_time=entry_time,
            last_edge=entry_edge,
            metadata=dict(metadata or {}),
        )

    def evaluate(
        self,
        position_id: str,
        current_edge: float = 0.0,
        current_ev: float = 0.0,
        current_risk: float = 0.0,
        now: float | None = None,
    ) -> ExitEvaluation:
        """Evaluate a position for exit.

        Parameters
        ----------
        position_id:
            Position to evaluate.
        current_edge:
            Current estimated edge.
        current_ev:
            Current estimated expected value.
        current_risk:
            Current risk metric.
        now:
            Current timestamp.
        """
        if now is None:
            now = time.time()
        cfg = self._config

        state = self._positions.get(position_id)
        if state is None:
            return ExitEvaluation(
                position_id=position_id,
                should_exit=False,
                action_type=ExitAction.HOLD,
                current_edge=0.0,
                edge_ratio=0.0,
                hold_seconds=0.0,
                hold_ratio=0.0,
                reason="not_found",
            )

        if state.exited:
            return ExitEvaluation(
                position_id=position_id,
                should_exit=False,
                action_type=state.exit_action,
                current_edge=state.last_edge,
                edge_ratio=0.0,
                hold_seconds=0.0,
                hold_ratio=0.0,
                reason="already_exited",
            )

        # Update state.
        state.last_edge = current_edge
        state.last_ev = current_ev
        state.last_risk = current_risk
        state.last_evaluated_at = now
        state.evaluation_count += 1

        hold_seconds = now - state.entry_time
        hold_ratio = hold_seconds / cfg.max_hold_seconds if cfg.max_hold_seconds > 0 else 0.0
        edge_ratio = current_edge / state.entry_edge if state.entry_edge != 0 else 0.0

        def _result(action: ExitAction, reason: str) -> ExitEvaluation:
            return ExitEvaluation(
                position_id=position_id,
                should_exit=action != ExitAction.HOLD,
                action_type=action,
                current_edge=current_edge,
                edge_ratio=edge_ratio,
                hold_seconds=hold_seconds,
                hold_ratio=hold_ratio,
                reason=reason,
            )

        # 1. Time stop — highest priority.
        if hold_seconds >= cfg.max_hold_seconds:
            self._mark_exit(state, ExitAction.TIME_STOP, now)
            return _result(ExitAction.TIME_STOP, "max_hold_time_exceeded")

        # 2. Risk stop.
        if current_risk > cfg.risk_stop_threshold:
            self._mark_exit(state, ExitAction.RISK_STOP, now)
            return _result(ExitAction.RISK_STOP, "risk_threshold_exceeded")

        # 3. EV stop.
        if current_ev < cfg.ev_stop_threshold:
            self._mark_exit(state, ExitAction.EV_STOP, now)
            return _result(ExitAction.EV_STOP, "ev_below_threshold")

        # 4. Edge decay — abandon if near expiry and very low edge.
        if edge_ratio < cfg.edge_decay_ratio:
            if hold_ratio >= cfg.abandon_time_ratio and current_edge <= cfg.abandon_edge_threshold:
                self._mark_exit(state, ExitAction.ABANDON, now)
                return _result(ExitAction.ABANDON, "edge_decayed_near_expiry")

            # Hedge if edge still above hedge threshold ratio.
            if edge_ratio >= cfg.hedge_threshold * cfg.edge_decay_ratio:
                self._mark_exit(state, ExitAction.HEDGE, now)
                return _result(ExitAction.HEDGE, "edge_decayed_partial")

            self._mark_exit(state, ExitAction.EDGE_DECAY, now)
            return _result(ExitAction.EDGE_DECAY, "edge_decayed_below_ratio")

        # 5. Minimum edge check.
        if current_edge < cfg.min_edge:
            self._mark_exit(state, ExitAction.CLOSE, now)
            return _result(ExitAction.CLOSE, "below_min_edge")

        # Hold.
        return _result(ExitAction.HOLD, "")

    def _mark_exit(
        self, state: PositionState, action: ExitAction, now: float
    ) -> None:
        """Mark a position as exited."""
        state.exited = True
        state.exit_action = action
        state.exit_time = now

    def close_position(self, position_id: str, now: float | None = None) -> None:
        """Manually close a position."""
        if now is None:
            now = time.time()
        state = self._positions.get(position_id)
        if state is not None and not state.exited:
            self._mark_exit(state, ExitAction.CLOSE, now)

    def get_position(self, position_id: str) -> PositionState | None:
        """Get position state."""
        return self._positions.get(position_id)

    def open_positions(self) -> List[str]:
        """List open position IDs."""
        return [
            pid for pid, state in self._positions.items()
            if not state.exited
        ]

    def exited_positions(self) -> List[str]:
        """List exited position IDs."""
        return [
            pid for pid, state in self._positions.items()
            if state.exited
        ]

    def report(self, now: float | None = None) -> ExitReport:
        """Generate an exit discipline report."""
        if now is None:
            now = time.time()

        total = len(self._positions)
        open_count = 0
        exited_count = 0
        total_hold = 0.0
        action_counts: Dict[str, int] = {}

        for state in self._positions.values():
            if state.exited:
                exited_count += 1
                hold = state.exit_time - state.entry_time
                total_hold += hold
                action_counts[state.exit_action.value] = (
                    action_counts.get(state.exit_action.value, 0) + 1
                )
            else:
                open_count += 1
                total_hold += now - state.entry_time

        avg_hold = total_hold / total if total > 0 else 0.0

        return ExitReport(
            total_positions=total,
            open_positions=open_count,
            exited_positions=exited_count,
            avg_hold_seconds=avg_hold,
            exits_by_action=action_counts,
        )

    def position_count(self) -> int:
        """Total positions (open + exited)."""
        return len(self._positions)

    def clear(self) -> None:
        """Clear all positions."""
        self._positions.clear()
