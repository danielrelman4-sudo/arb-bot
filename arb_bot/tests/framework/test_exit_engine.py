"""Tests for exit_engine module (Phase 6D).

This module controls real-money exit decisions. Tests are designed to
thoroughly validate priority ordering of exit signals, boundary conditions,
and that no position can slip through without proper evaluation.
"""

from __future__ import annotations

import pytest

from arb_bot.framework.exit_engine import (
    ExitAction,
    ExitEngine,
    ExitEngineConfig,
    ExitEvaluation,
    ExitReport,
    PositionState,
)


def _engine(**kw: object) -> ExitEngine:
    return ExitEngine(ExitEngineConfig(**kw))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self) -> None:
        cfg = ExitEngineConfig()
        assert cfg.max_hold_seconds == 3600.0
        assert cfg.min_edge == 0.005
        assert cfg.edge_decay_ratio == 0.25
        assert cfg.ev_stop_threshold == -0.02
        assert cfg.risk_stop_threshold == 0.10
        assert cfg.hedge_threshold == 0.50
        assert cfg.abandon_time_ratio == 0.90
        assert cfg.abandon_edge_threshold == 0.001
        assert cfg.evaluation_cooldown == 5.0

    def test_frozen(self) -> None:
        cfg = ExitEngineConfig()
        with pytest.raises(AttributeError):
            cfg.max_hold_seconds = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Open position
# ---------------------------------------------------------------------------


class TestOpenPosition:
    def test_basic(self) -> None:
        e = _engine()
        e.open_position("pos1", entry_edge=0.05, entry_time=100.0)
        assert e.position_count() == 1
        state = e.get_position("pos1")
        assert state is not None
        assert state.entry_edge == 0.05
        assert state.entry_time == 100.0
        assert state.exited is False

    def test_with_metadata(self) -> None:
        e = _engine()
        e.open_position("pos1", entry_edge=0.05, entry_time=100.0,
                         metadata={"venue": "kalshi", "lane": "cross"})
        state = e.get_position("pos1")
        assert state is not None
        assert state.metadata["venue"] == "kalshi"

    def test_multiple_positions(self) -> None:
        e = _engine()
        e.open_position("a", entry_edge=0.05, entry_time=100.0)
        e.open_position("b", entry_edge=0.03, entry_time=101.0)
        assert e.position_count() == 2
        assert set(e.open_positions()) == {"a", "b"}

    def test_overwrite_position(self) -> None:
        e = _engine()
        e.open_position("pos1", entry_edge=0.05, entry_time=100.0)
        e.open_position("pos1", entry_edge=0.10, entry_time=200.0)
        state = e.get_position("pos1")
        assert state is not None
        assert state.entry_edge == 0.10  # Overwritten.


# ---------------------------------------------------------------------------
# Hold — no exit needed
# ---------------------------------------------------------------------------


class TestHold:
    def test_healthy_position_holds(self) -> None:
        """Position with good edge, low risk, positive EV should hold."""
        e = _engine(max_hold_seconds=3600.0, min_edge=0.005)
        e.open_position("pos1", entry_edge=0.05, entry_time=100.0)
        result = e.evaluate("pos1", current_edge=0.04, current_ev=0.03,
                             current_risk=0.01, now=200.0)
        assert result.should_exit is False
        assert result.action_type == ExitAction.HOLD
        assert result.hold_seconds == 100.0

    def test_edge_ratio_above_decay(self) -> None:
        """Edge at 50% of entry (above 25% decay ratio) should hold."""
        e = _engine(edge_decay_ratio=0.25, min_edge=0.005)
        e.open_position("pos1", entry_edge=0.10, entry_time=100.0)
        result = e.evaluate("pos1", current_edge=0.05, now=200.0)  # 50% ratio.
        assert result.should_exit is False
        assert abs(result.edge_ratio - 0.5) < 0.01


# ---------------------------------------------------------------------------
# Time stop — priority 1
# ---------------------------------------------------------------------------


class TestTimeStop:
    def test_time_stop_triggers(self) -> None:
        e = _engine(max_hold_seconds=60.0)
        e.open_position("pos1", entry_edge=0.05, entry_time=100.0)
        result = e.evaluate("pos1", current_edge=0.05, now=161.0)  # 61s > 60s.
        assert result.should_exit is True
        assert result.action_type == ExitAction.TIME_STOP
        assert result.reason == "max_hold_time_exceeded"

    def test_time_stop_exact_boundary(self) -> None:
        """Exactly at max hold should trigger."""
        e = _engine(max_hold_seconds=60.0)
        e.open_position("pos1", entry_edge=0.05, entry_time=100.0)
        result = e.evaluate("pos1", current_edge=0.05, now=160.0)  # Exactly 60s.
        assert result.should_exit is True
        assert result.action_type == ExitAction.TIME_STOP

    def test_time_stop_just_before(self) -> None:
        """Just below max hold should NOT trigger."""
        e = _engine(max_hold_seconds=60.0)
        e.open_position("pos1", entry_edge=0.05, entry_time=100.0)
        result = e.evaluate("pos1", current_edge=0.05, now=159.9)  # 59.9s < 60s.
        assert result.action_type != ExitAction.TIME_STOP

    def test_time_stop_overrides_good_edge(self) -> None:
        """Time stop fires even if edge is excellent."""
        e = _engine(max_hold_seconds=60.0)
        e.open_position("pos1", entry_edge=0.05, entry_time=100.0)
        result = e.evaluate("pos1", current_edge=0.10, now=161.0)  # Great edge!
        assert result.should_exit is True
        assert result.action_type == ExitAction.TIME_STOP


# ---------------------------------------------------------------------------
# Risk stop — priority 2
# ---------------------------------------------------------------------------


class TestRiskStop:
    def test_risk_stop_triggers(self) -> None:
        e = _engine(risk_stop_threshold=0.10)
        e.open_position("pos1", entry_edge=0.05, entry_time=100.0)
        result = e.evaluate("pos1", current_edge=0.05, current_risk=0.15, now=200.0)
        assert result.should_exit is True
        assert result.action_type == ExitAction.RISK_STOP
        assert result.reason == "risk_threshold_exceeded"

    def test_risk_exactly_at_threshold(self) -> None:
        """Risk at exactly the threshold should NOT trigger (> not >=)."""
        e = _engine(risk_stop_threshold=0.10)
        e.open_position("pos1", entry_edge=0.05, entry_time=100.0)
        result = e.evaluate("pos1", current_edge=0.05, current_risk=0.10, now=200.0)
        # 0.10 is not > 0.10, so should hold.
        assert result.action_type != ExitAction.RISK_STOP

    def test_risk_just_above_threshold(self) -> None:
        e = _engine(risk_stop_threshold=0.10)
        e.open_position("pos1", entry_edge=0.05, entry_time=100.0)
        result = e.evaluate("pos1", current_edge=0.05, current_risk=0.101, now=200.0)
        assert result.action_type == ExitAction.RISK_STOP

    def test_time_stop_beats_risk_stop(self) -> None:
        """Time stop should trigger before risk stop if both conditions met."""
        e = _engine(max_hold_seconds=60.0, risk_stop_threshold=0.10)
        e.open_position("pos1", entry_edge=0.05, entry_time=100.0)
        result = e.evaluate("pos1", current_edge=0.05, current_risk=0.15, now=161.0)
        assert result.action_type == ExitAction.TIME_STOP  # Higher priority.


# ---------------------------------------------------------------------------
# EV stop — priority 3
# ---------------------------------------------------------------------------


class TestEvStop:
    def test_ev_stop_triggers(self) -> None:
        e = _engine(ev_stop_threshold=-0.02)
        e.open_position("pos1", entry_edge=0.05, entry_time=100.0)
        result = e.evaluate("pos1", current_edge=0.05, current_ev=-0.03, now=200.0)
        assert result.should_exit is True
        assert result.action_type == ExitAction.EV_STOP

    def test_ev_exactly_at_threshold(self) -> None:
        """EV at exactly threshold should NOT trigger (< not <=)."""
        e = _engine(ev_stop_threshold=-0.02)
        e.open_position("pos1", entry_edge=0.05, entry_time=100.0)
        result = e.evaluate("pos1", current_edge=0.05, current_ev=-0.02, now=200.0)
        assert result.action_type != ExitAction.EV_STOP

    def test_risk_beats_ev(self) -> None:
        """Risk stop should fire before EV stop."""
        e = _engine(risk_stop_threshold=0.10, ev_stop_threshold=-0.02)
        e.open_position("pos1", entry_edge=0.05, entry_time=100.0)
        result = e.evaluate("pos1", current_edge=0.05, current_ev=-0.05,
                             current_risk=0.15, now=200.0)
        assert result.action_type == ExitAction.RISK_STOP


# ---------------------------------------------------------------------------
# Edge decay — priority 4
# ---------------------------------------------------------------------------


class TestEdgeDecay:
    def test_edge_decay_close(self) -> None:
        """Edge decayed below ratio → close."""
        e = _engine(edge_decay_ratio=0.25, min_edge=0.001)
        e.open_position("pos1", entry_edge=0.10, entry_time=100.0)
        # 0.01 / 0.10 = 0.10 < 0.25 ratio.
        result = e.evaluate("pos1", current_edge=0.01, now=200.0)
        assert result.should_exit is True
        assert result.action_type == ExitAction.EDGE_DECAY

    def test_edge_decay_abandon_near_expiry(self) -> None:
        """Edge decayed + near expiry + very low edge → abandon."""
        e = _engine(
            max_hold_seconds=100.0,
            edge_decay_ratio=0.25,
            abandon_time_ratio=0.90,
            abandon_edge_threshold=0.001,
        )
        e.open_position("pos1", entry_edge=0.10, entry_time=100.0)
        # hold_ratio = 95/100 = 0.95 ≥ 0.90, edge=0.0005 ≤ 0.001.
        result = e.evaluate("pos1", current_edge=0.0005, now=195.0)
        assert result.should_exit is True
        assert result.action_type == ExitAction.ABANDON
        assert "near_expiry" in result.reason

    def test_edge_exact_at_decay_ratio(self) -> None:
        """Edge ratio exactly at decay ratio should hold (< not <=)."""
        e = _engine(edge_decay_ratio=0.25, min_edge=0.001)
        e.open_position("pos1", entry_edge=0.10, entry_time=100.0)
        # 0.025 / 0.10 = 0.25, exactly at ratio — NOT below.
        result = e.evaluate("pos1", current_edge=0.025, now=200.0)
        assert result.action_type != ExitAction.EDGE_DECAY

    def test_edge_just_below_decay_ratio(self) -> None:
        e = _engine(edge_decay_ratio=0.25, min_edge=0.001)
        e.open_position("pos1", entry_edge=0.10, entry_time=100.0)
        # 0.024 / 0.10 = 0.24 < 0.25.
        result = e.evaluate("pos1", current_edge=0.024, now=200.0)
        assert result.should_exit is True


# ---------------------------------------------------------------------------
# Min edge — priority 5
# ---------------------------------------------------------------------------


class TestMinEdge:
    def test_below_min_edge_close(self) -> None:
        e = _engine(min_edge=0.005, edge_decay_ratio=0.01)  # Low decay ratio so it doesn't trigger first.
        e.open_position("pos1", entry_edge=0.05, entry_time=100.0)
        result = e.evaluate("pos1", current_edge=0.004, now=200.0)
        assert result.should_exit is True
        assert result.action_type == ExitAction.CLOSE
        assert result.reason == "below_min_edge"

    def test_at_min_edge_holds(self) -> None:
        """Edge at exactly min_edge should hold (< not <=)."""
        e = _engine(min_edge=0.005, edge_decay_ratio=0.01)
        e.open_position("pos1", entry_edge=0.05, entry_time=100.0)
        result = e.evaluate("pos1", current_edge=0.005, now=200.0)
        assert result.action_type != ExitAction.CLOSE


# ---------------------------------------------------------------------------
# Exit is permanent
# ---------------------------------------------------------------------------


class TestExitPermanence:
    def test_already_exited(self) -> None:
        """Once exited, further evaluations return already_exited."""
        e = _engine(max_hold_seconds=60.0)
        e.open_position("pos1", entry_edge=0.05, entry_time=100.0)
        e.evaluate("pos1", current_edge=0.05, now=161.0)  # Triggers time stop.

        result = e.evaluate("pos1", current_edge=0.10, now=200.0)  # Re-evaluate.
        assert result.should_exit is False  # Already handled.
        assert result.reason == "already_exited"

    def test_exited_state_recorded(self) -> None:
        e = _engine(max_hold_seconds=60.0)
        e.open_position("pos1", entry_edge=0.05, entry_time=100.0)
        e.evaluate("pos1", current_edge=0.05, now=161.0)
        state = e.get_position("pos1")
        assert state is not None
        assert state.exited is True
        assert state.exit_action == ExitAction.TIME_STOP
        assert state.exit_time == 161.0


# ---------------------------------------------------------------------------
# Manual close
# ---------------------------------------------------------------------------


class TestManualClose:
    def test_manual_close(self) -> None:
        e = _engine()
        e.open_position("pos1", entry_edge=0.05, entry_time=100.0)
        e.close_position("pos1", now=200.0)
        state = e.get_position("pos1")
        assert state is not None
        assert state.exited is True
        assert state.exit_action == ExitAction.CLOSE

    def test_manual_close_already_exited(self) -> None:
        """Manual close on already-exited position is a no-op."""
        e = _engine(max_hold_seconds=60.0)
        e.open_position("pos1", entry_edge=0.05, entry_time=100.0)
        e.evaluate("pos1", current_edge=0.05, now=161.0)  # Time stop.
        e.close_position("pos1", now=200.0)  # Should not change exit action.
        state = e.get_position("pos1")
        assert state is not None
        assert state.exit_action == ExitAction.TIME_STOP  # Preserved.


# ---------------------------------------------------------------------------
# Nonexistent position
# ---------------------------------------------------------------------------


class TestNonexistent:
    def test_evaluate_nonexistent(self) -> None:
        e = _engine()
        result = e.evaluate("missing", current_edge=0.05, now=100.0)
        assert result.should_exit is False
        assert result.reason == "not_found"


# ---------------------------------------------------------------------------
# Open / exited lists
# ---------------------------------------------------------------------------


class TestPositionLists:
    def test_open_and_exited(self) -> None:
        e = _engine(max_hold_seconds=60.0)
        e.open_position("a", entry_edge=0.05, entry_time=100.0)
        e.open_position("b", entry_edge=0.05, entry_time=100.0)
        e.evaluate("a", current_edge=0.05, now=161.0)  # Exits a.

        assert e.open_positions() == ["b"]
        assert e.exited_positions() == ["a"]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


class TestReport:
    def test_empty(self) -> None:
        e = _engine()
        r = e.report(now=100.0)
        assert r.total_positions == 0
        assert r.avg_hold_seconds == 0.0

    def test_report_with_positions(self) -> None:
        e = _engine(max_hold_seconds=60.0, min_edge=0.01)
        e.open_position("a", entry_edge=0.05, entry_time=100.0)
        e.open_position("b", entry_edge=0.05, entry_time=100.0)
        e.evaluate("a", current_edge=0.05, now=161.0)  # Time stop at 61s.
        # edge_ratio = 0.001/0.05 = 0.02 < edge_decay_ratio (0.25),
        # so edge_decay fires before min_edge.
        e.evaluate("b", current_edge=0.001, now=150.0)

        r = e.report(now=200.0)
        assert r.total_positions == 2
        assert r.open_positions == 0
        assert r.exited_positions == 2
        assert r.exits_by_action.get("time_stop", 0) == 1
        assert r.exits_by_action.get("edge_decay", 0) == 1


# ---------------------------------------------------------------------------
# Evaluation state tracking
# ---------------------------------------------------------------------------


class TestStateTracking:
    def test_evaluation_count(self) -> None:
        e = _engine()
        e.open_position("pos1", entry_edge=0.05, entry_time=100.0)
        e.evaluate("pos1", current_edge=0.04, now=110.0)
        e.evaluate("pos1", current_edge=0.03, now=120.0)
        e.evaluate("pos1", current_edge=0.02, now=130.0)
        state = e.get_position("pos1")
        assert state is not None
        assert state.evaluation_count == 3

    def test_last_edge_updated(self) -> None:
        e = _engine()
        e.open_position("pos1", entry_edge=0.05, entry_time=100.0)
        e.evaluate("pos1", current_edge=0.03, current_ev=0.01, current_risk=0.02, now=200.0)
        state = e.get_position("pos1")
        assert state is not None
        assert state.last_edge == 0.03
        assert state.last_ev == 0.01
        assert state.last_risk == 0.02
        assert state.last_evaluated_at == 200.0


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clear(self) -> None:
        e = _engine()
        e.open_position("a", entry_edge=0.05, entry_time=100.0)
        e.open_position("b", entry_edge=0.03, entry_time=100.0)
        e.clear()
        assert e.position_count() == 0


# ---------------------------------------------------------------------------
# Priority ordering comprehensive
# ---------------------------------------------------------------------------


class TestPriorityOrdering:
    """Verify the full priority chain: time > risk > EV > edge_decay > min_edge > hold."""

    def _engine(self) -> ExitEngine:
        return _engine(
            max_hold_seconds=100.0,
            risk_stop_threshold=0.10,
            ev_stop_threshold=-0.02,
            edge_decay_ratio=0.25,
            min_edge=0.005,
        )

    def test_all_bad_time_wins(self) -> None:
        """All exit conditions met → time stop wins."""
        e = self._engine()
        e.open_position("p", entry_edge=0.10, entry_time=100.0)
        result = e.evaluate("p", current_edge=0.001, current_ev=-0.05,
                             current_risk=0.20, now=201.0)
        assert result.action_type == ExitAction.TIME_STOP

    def test_risk_ev_edge_risk_wins(self) -> None:
        """Risk, EV, edge all bad but within time → risk wins."""
        e = self._engine()
        e.open_position("p", entry_edge=0.10, entry_time=100.0)
        result = e.evaluate("p", current_edge=0.001, current_ev=-0.05,
                             current_risk=0.20, now=150.0)
        assert result.action_type == ExitAction.RISK_STOP

    def test_ev_edge_bad_ev_wins(self) -> None:
        """EV and edge bad, risk OK → EV wins."""
        e = self._engine()
        e.open_position("p", entry_edge=0.10, entry_time=100.0)
        result = e.evaluate("p", current_edge=0.001, current_ev=-0.05,
                             current_risk=0.01, now=150.0)
        assert result.action_type == ExitAction.EV_STOP

    def test_edge_bad_only_hedge(self) -> None:
        """Edge decayed but still above hedge sub-threshold → hedge."""
        e = self._engine()
        e.open_position("p", entry_edge=0.10, entry_time=100.0)
        result = e.evaluate("p", current_edge=0.02, current_ev=0.01,
                             current_risk=0.01, now=150.0)
        # edge_ratio = 0.02/0.10 = 0.20 < 0.25 (decay ratio), but
        # 0.20 >= 0.50 * 0.25 = 0.125 (hedge threshold) → HEDGE.
        assert result.action_type == ExitAction.HEDGE

    def test_edge_fully_decayed(self) -> None:
        """Edge decayed well below hedge sub-threshold → full edge decay close."""
        e = self._engine()
        e.open_position("p", entry_edge=0.10, entry_time=100.0)
        result = e.evaluate("p", current_edge=0.01, current_ev=0.01,
                             current_risk=0.01, now=150.0)
        # edge_ratio = 0.01/0.10 = 0.10 < 0.125 (hedge threshold) → EDGE_DECAY.
        assert result.action_type == ExitAction.EDGE_DECAY


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_position_lifecycle(self) -> None:
        """Full lifecycle: open → evaluate (hold) → edge decays → close."""
        e = _engine(
            max_hold_seconds=3600.0,
            min_edge=0.005,
            edge_decay_ratio=0.25,
        )

        # Open with 5% edge.
        e.open_position("trade1", entry_edge=0.05, entry_time=1000.0,
                         metadata={"lane": "cross", "venue": "kalshi"})

        # T+60s: Edge still healthy.
        r1 = e.evaluate("trade1", current_edge=0.04, now=1060.0)
        assert r1.should_exit is False
        assert r1.hold_seconds == 60.0

        # T+120s: Edge declining.
        r2 = e.evaluate("trade1", current_edge=0.02, now=1120.0)
        assert r2.should_exit is False  # 0.02/0.05=0.40 > 0.25.

        # T+180s: Edge collapsed.
        r3 = e.evaluate("trade1", current_edge=0.005, now=1180.0)
        # 0.005/0.05 = 0.10 < 0.25 → edge decay.
        assert r3.should_exit is True
        assert r3.action_type == ExitAction.EDGE_DECAY

        # Position is now exited.
        assert "trade1" in e.exited_positions()
        assert "trade1" not in e.open_positions()

        # Report shows the exit.
        report = e.report(now=1200.0)
        assert report.exited_positions == 1
        assert report.exits_by_action.get("edge_decay") == 1
