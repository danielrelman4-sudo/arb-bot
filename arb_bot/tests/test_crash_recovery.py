"""Tests for Phase 0F: Crash recovery startup."""

from __future__ import annotations

from pathlib import Path

import pytest

from arb_bot.crash_recovery import RecoveryReport, recover_state
from arb_bot.models import ExecutionStyle, OpportunityKind, Side, TradeLegPlan, TradePlan
from arb_bot.order_store import OrderStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cross_plan(contracts: int = 10) -> TradePlan:
    return TradePlan(
        kind=OpportunityKind.CROSS_VENUE,
        execution_style=ExecutionStyle.TAKER,
        legs=(
            TradeLegPlan(venue="kalshi", market_id="K1", side=Side.YES, contracts=contracts, limit_price=0.45),
            TradeLegPlan(venue="polymarket", market_id="P1", side=Side.NO, contracts=contracts, limit_price=0.50),
        ),
        contracts=contracts,
        capital_required=9.5,
        capital_required_by_venue={"kalshi": 4.5, "polymarket": 5.0},
        expected_profit=0.50,
        edge_per_contract=0.05,
    )


def _intra_plan() -> TradePlan:
    return TradePlan(
        kind=OpportunityKind.INTRA_VENUE,
        execution_style=ExecutionStyle.TAKER,
        legs=(
            TradeLegPlan(venue="kalshi", market_id="M1", side=Side.YES, contracts=10, limit_price=0.40),
            TradeLegPlan(venue="kalshi", market_id="M1", side=Side.NO, contracts=10, limit_price=0.40),
        ),
        contracts=10,
        capital_required=8.0,
        capital_required_by_venue={"kalshi": 8.0},
        expected_profit=1.0,
        edge_per_contract=0.10,
    )


@pytest.fixture()
def store(tmp_path: Path) -> OrderStore:
    db_path = tmp_path / "test_recovery.db"
    s = OrderStore(db_path=db_path)
    yield s
    s.close()


INITIAL_CASH = {"kalshi": 100.0, "polymarket": 100.0}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRecoveryReport:
    def test_defaults(self) -> None:
        r = RecoveryReport()
        assert r.safe_mode is False
        assert r.safe_mode_reason == ""
        assert r.open_intents == 0


class TestRecoverCleanState:
    def test_empty_store(self, store: OrderStore) -> None:
        state, report = recover_state(store, INITIAL_CASH)

        assert state.cash_by_venue["kalshi"] == 100.0
        assert state.cash_by_venue["polymarket"] == 100.0
        assert state.locked_capital_by_venue == {}
        assert state.open_markets_by_venue == {}
        assert report.safe_mode is False
        assert report.open_intents == 0

    def test_fully_filled_intent_no_safe_mode(self, store: OrderStore) -> None:
        plan = _cross_plan()
        iid = store.create_intent(plan)
        store.record_leg_result(iid, 0, success=True, filled_contracts=10, average_price=0.45)
        store.record_leg_result(iid, 1, success=True, filled_contracts=10, average_price=0.50)
        store.finalize_intent(iid)
        store.open_position(iid, "kalshi", "K1", "yes", 10, 0.45)
        store.open_position(iid, "polymarket", "P1", "no", 10, 0.50)

        state, report = recover_state(store, INITIAL_CASH)

        assert report.safe_mode is False
        assert report.open_positions == 2
        assert state.locked_capital_by_venue["kalshi"] == pytest.approx(4.5)
        assert state.locked_capital_by_venue["polymarket"] == pytest.approx(5.0)
        assert state.cash_by_venue["kalshi"] == pytest.approx(95.5)
        assert state.cash_by_venue["polymarket"] == pytest.approx(95.0)

    def test_closed_positions_not_locked(self, store: OrderStore) -> None:
        plan = _intra_plan()
        iid = store.create_intent(plan)
        store.record_leg_result(iid, 0, success=True, filled_contracts=10, average_price=0.40)
        store.record_leg_result(iid, 1, success=True, filled_contracts=10, average_price=0.40)
        store.finalize_intent(iid)
        pid = store.open_position(iid, "kalshi", "M1", "yes", 10, 0.40)
        store.close_position(pid)

        state, report = recover_state(store, INITIAL_CASH)

        assert state.locked_capital_by_venue == {}
        assert state.cash_by_venue["kalshi"] == 100.0
        assert report.safe_mode is False


class TestRecoverWithSafeMode:
    def test_pending_intent_triggers_safe_mode(self, store: OrderStore) -> None:
        plan = _cross_plan()
        store.create_intent(plan)  # Pending, never finalized

        _, report = recover_state(store, INITIAL_CASH)

        assert report.safe_mode is True
        assert "open intent" in report.safe_mode_reason

    def test_submitted_intent_triggers_safe_mode(self, store: OrderStore) -> None:
        plan = _cross_plan()
        iid = store.create_intent(plan)
        store.mark_intent_submitted(iid)

        _, report = recover_state(store, INITIAL_CASH)

        assert report.safe_mode is True
        assert "open intent" in report.safe_mode_reason

    def test_partial_fill_triggers_safe_mode(self, store: OrderStore) -> None:
        plan = _cross_plan()
        iid = store.create_intent(plan)
        store.record_leg_result(iid, 0, success=True, filled_contracts=10, average_price=0.45)
        store.record_leg_result(iid, 1, success=False, raw={"error": "timeout"})
        store.finalize_intent(iid)

        _, report = recover_state(store, INITIAL_CASH)

        assert report.safe_mode is True
        assert "unhedged" in report.safe_mode_reason
        assert len(report.unhedged_exposure) == 1
        assert report.unhedged_exposure[0]["exposed"] == 10

    def test_multiple_issues(self, store: OrderStore) -> None:
        plan = _cross_plan()

        # One pending intent.
        store.create_intent(plan)

        # One partially filled.
        iid2 = store.create_intent(plan)
        store.record_leg_result(iid2, 0, success=True, filled_contracts=10, average_price=0.45)
        store.record_leg_result(iid2, 1, success=False, raw={"error": "fail"})
        store.finalize_intent(iid2)

        _, report = recover_state(store, INITIAL_CASH)

        assert report.safe_mode is True
        assert "unhedged" in report.safe_mode_reason
        assert "open intent" in report.safe_mode_reason

    def test_failed_intent_no_safe_mode(self, store: OrderStore) -> None:
        """Both legs failed → no unhedged exposure, no safe mode (just a failed trade)."""
        plan = _cross_plan()
        iid = store.create_intent(plan)
        store.record_leg_result(iid, 0, success=False, raw={"error": "err1"})
        store.record_leg_result(iid, 1, success=False, raw={"error": "err2"})
        store.finalize_intent(iid)

        _, report = recover_state(store, INITIAL_CASH)

        assert report.safe_mode is False


class TestRecoverOpenMarkets:
    def test_rebuilds_open_markets(self, store: OrderStore) -> None:
        plan = _cross_plan()
        iid = store.create_intent(plan)
        store.record_leg_result(iid, 0, success=True, filled_contracts=10, average_price=0.45)
        store.record_leg_result(iid, 1, success=True, filled_contracts=10, average_price=0.50)
        store.finalize_intent(iid)
        store.open_position(iid, "kalshi", "K1", "yes", 10, 0.45)
        store.open_position(iid, "polymarket", "P1", "no", 10, 0.50)

        state, _ = recover_state(store, INITIAL_CASH)

        assert "K1" in state.open_markets_by_venue.get("kalshi", set())
        assert "P1" in state.open_markets_by_venue.get("polymarket", set())


class TestRecoverPersistence:
    def test_survives_store_reopen(self, tmp_path: Path) -> None:
        """Full round-trip: create store, add data, close, reopen, recover."""
        db_path = tmp_path / "persist_recovery.db"

        store1 = OrderStore(db_path=db_path)
        plan = _cross_plan()
        iid = store1.create_intent(plan)
        store1.record_leg_result(iid, 0, success=True, filled_contracts=10, average_price=0.45)
        store1.record_leg_result(iid, 1, success=True, filled_contracts=10, average_price=0.50)
        store1.finalize_intent(iid)
        store1.open_position(iid, "kalshi", "K1", "yes", 10, 0.45)
        store1.close()

        store2 = OrderStore(db_path=db_path)
        state, report = recover_state(store2, INITIAL_CASH)
        store2.close()

        assert state.locked_capital_by_venue["kalshi"] == pytest.approx(4.5)
        assert report.safe_mode is False
        assert report.open_positions == 1
