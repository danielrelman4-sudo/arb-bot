"""Tests for Phase 0E: Partial fill handling."""

from __future__ import annotations

from pathlib import Path

import pytest

from arb_bot.models import ExecutionStyle, OpportunityKind, Side, TradeLegPlan, TradePlan
from arb_bot.order_store import IntentStatus, LegStatus, OrderStore


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


@pytest.fixture()
def store(tmp_path: Path) -> OrderStore:
    db_path = tmp_path / "test_partial.db"
    s = OrderStore(db_path=db_path)
    yield s
    s.close()


# ---------------------------------------------------------------------------
# update_leg_fill
# ---------------------------------------------------------------------------


class TestUpdateLegFill:
    def test_partial_fill_updates_status(self, store: OrderStore) -> None:
        plan = _cross_plan()
        iid = store.create_intent(plan)
        store.mark_leg_submitted(iid, 0, order_id="A")

        store.update_leg_fill(iid, 0, filled_contracts=5, average_price=0.44)

        legs = store.get_legs(iid)
        assert legs[0].filled_contracts == 5
        assert legs[0].average_price == pytest.approx(0.44)
        assert legs[0].status == LegStatus.PARTIALLY_FILLED.value

    def test_full_fill_updates_to_filled(self, store: OrderStore) -> None:
        plan = _cross_plan()
        iid = store.create_intent(plan)
        store.mark_leg_submitted(iid, 0, order_id="A")

        store.update_leg_fill(iid, 0, filled_contracts=10, average_price=0.45)

        legs = store.get_legs(iid)
        assert legs[0].status == LegStatus.FILLED.value
        assert legs[0].filled_contracts == 10

    def test_zero_fill_keeps_current_status(self, store: OrderStore) -> None:
        plan = _cross_plan()
        iid = store.create_intent(plan)
        store.mark_leg_submitted(iid, 0, order_id="A")

        store.update_leg_fill(iid, 0, filled_contracts=0)

        legs = store.get_legs(iid)
        assert legs[0].status == LegStatus.SUBMITTED.value
        assert legs[0].filled_contracts == 0

    def test_missing_leg_is_noop(self, store: OrderStore) -> None:
        plan = _cross_plan()
        iid = store.create_intent(plan)

        # No error for non-existent leg_index.
        store.update_leg_fill(iid, 99, filled_contracts=5)

    def test_preserves_existing_average_price(self, store: OrderStore) -> None:
        plan = _cross_plan()
        iid = store.create_intent(plan)
        store.record_leg_result(iid, 0, success=True, order_id="A", filled_contracts=5, average_price=0.43)

        # Update fill count but no price → should keep 0.43.
        store.update_leg_fill(iid, 0, filled_contracts=7)

        legs = store.get_legs(iid)
        assert legs[0].average_price == pytest.approx(0.43)
        assert legs[0].filled_contracts == 7


# ---------------------------------------------------------------------------
# compute_hedge_exposure
# ---------------------------------------------------------------------------


class TestComputeHedgeExposure:
    def test_both_legs_fully_filled(self, store: OrderStore) -> None:
        plan = _cross_plan()
        iid = store.create_intent(plan)
        store.record_leg_result(iid, 0, success=True, filled_contracts=10, average_price=0.45)
        store.record_leg_result(iid, 1, success=True, filled_contracts=10, average_price=0.50)

        exposure = store.compute_hedge_exposure(iid)
        assert exposure["hedged"] == 10
        assert exposure["exposed"] == 0
        assert exposure["unfilled"] == 0

    def test_one_leg_partial(self, store: OrderStore) -> None:
        plan = _cross_plan()
        iid = store.create_intent(plan)
        store.record_leg_result(iid, 0, success=True, filled_contracts=10, average_price=0.45)
        store.update_leg_fill(iid, 1, filled_contracts=6)

        exposure = store.compute_hedge_exposure(iid)
        assert exposure["hedged"] == 6
        assert exposure["exposed"] == 4  # 10 - 6
        assert exposure["unfilled"] == 0  # max(0, 10 - 10)

    def test_both_legs_partial(self, store: OrderStore) -> None:
        plan = _cross_plan()
        iid = store.create_intent(plan)
        store.update_leg_fill(iid, 0, filled_contracts=7)
        store.update_leg_fill(iid, 1, filled_contracts=4)

        exposure = store.compute_hedge_exposure(iid)
        assert exposure["hedged"] == 4
        assert exposure["exposed"] == 3  # 7 - 4
        assert exposure["unfilled"] == 3  # 10 - 7

    def test_no_fills(self, store: OrderStore) -> None:
        plan = _cross_plan()
        iid = store.create_intent(plan)

        exposure = store.compute_hedge_exposure(iid)
        assert exposure["hedged"] == 0
        assert exposure["exposed"] == 0
        assert exposure["unfilled"] == 10

    def test_one_leg_failed_other_filled(self, store: OrderStore) -> None:
        plan = _cross_plan()
        iid = store.create_intent(plan)
        store.record_leg_result(iid, 0, success=True, filled_contracts=10, average_price=0.45)
        store.record_leg_result(iid, 1, success=False)

        exposure = store.compute_hedge_exposure(iid)
        assert exposure["hedged"] == 0
        assert exposure["exposed"] == 10
        assert exposure["unfilled"] == 0

    def test_empty_intent(self, store: OrderStore) -> None:
        """Edge case: no legs found (shouldn't happen, but handle gracefully)."""
        exposure = store.compute_hedge_exposure("nonexistent")
        assert exposure["hedged"] == 0
        assert exposure["exposed"] == 0
        assert exposure["unfilled"] == 0


# ---------------------------------------------------------------------------
# get_partially_filled_intents
# ---------------------------------------------------------------------------


class TestGetPartiallyFilledIntents:
    def test_returns_partial_intents(self, store: OrderStore) -> None:
        plan = _cross_plan()
        id1 = store.create_intent(plan)
        id2 = store.create_intent(plan)

        # id1: one leg filled, one failed → partially_filled
        store.record_leg_result(id1, 0, success=True, filled_contracts=10, average_price=0.45)
        store.record_leg_result(id1, 1, success=False, raw={"error": "timeout"})
        store.finalize_intent(id1)

        # id2: both filled → filled
        store.record_leg_result(id2, 0, success=True, filled_contracts=10, average_price=0.45)
        store.record_leg_result(id2, 1, success=True, filled_contracts=10, average_price=0.50)
        store.finalize_intent(id2)

        partial = store.get_partially_filled_intents()
        assert len(partial) == 1
        assert partial[0].intent_id == id1
        assert partial[0].status == IntentStatus.PARTIALLY_FILLED.value

    def test_empty_when_none_partial(self, store: OrderStore) -> None:
        plan = _cross_plan()
        iid = store.create_intent(plan)
        store.record_leg_result(iid, 0, success=True, filled_contracts=10, average_price=0.45)
        store.record_leg_result(iid, 1, success=True, filled_contracts=10, average_price=0.50)
        store.finalize_intent(iid)

        partial = store.get_partially_filled_intents()
        assert len(partial) == 0


# ---------------------------------------------------------------------------
# Integration: update_leg_fill + finalize shows correct exposure
# ---------------------------------------------------------------------------


class TestPartialFillIntegration:
    def test_poll_update_then_finalize(self, store: OrderStore) -> None:
        """Simulate: leg A fills fully, leg B partially fills via polling, then finalize."""
        plan = _cross_plan()
        iid = store.create_intent(plan)

        # Leg A fills immediately.
        store.record_leg_result(iid, 0, success=True, order_id="A", filled_contracts=10, average_price=0.45)

        # Leg B partially fills (from polling).
        store.mark_leg_submitted(iid, 1, order_id="B")
        store.update_leg_fill(iid, 1, filled_contracts=7, average_price=0.49)

        # Check exposure before finalize.
        exposure = store.compute_hedge_exposure(iid)
        assert exposure["hedged"] == 7
        assert exposure["exposed"] == 3

        # Now update to full fill.
        store.update_leg_fill(iid, 1, filled_contracts=10, average_price=0.50)
        exposure = store.compute_hedge_exposure(iid)
        assert exposure["hedged"] == 10
        assert exposure["exposed"] == 0

    def test_progressive_fill_tracking(self, store: OrderStore) -> None:
        """Simulate progressive fills from multiple poll updates."""
        plan = _cross_plan()
        iid = store.create_intent(plan)

        store.mark_leg_submitted(iid, 0, order_id="A")
        store.mark_leg_submitted(iid, 1, order_id="B")

        # Poll 1: A has 3, B has 1.
        store.update_leg_fill(iid, 0, filled_contracts=3)
        store.update_leg_fill(iid, 1, filled_contracts=1)
        exposure = store.compute_hedge_exposure(iid)
        assert exposure["hedged"] == 1
        assert exposure["exposed"] == 2  # 3 - 1

        # Poll 2: A has 7, B has 5.
        store.update_leg_fill(iid, 0, filled_contracts=7)
        store.update_leg_fill(iid, 1, filled_contracts=5)
        exposure = store.compute_hedge_exposure(iid)
        assert exposure["hedged"] == 5
        assert exposure["exposed"] == 2  # 7 - 5

        # Poll 3: Both filled.
        store.update_leg_fill(iid, 0, filled_contracts=10)
        store.update_leg_fill(iid, 1, filled_contracts=10)
        exposure = store.compute_hedge_exposure(iid)
        assert exposure["hedged"] == 10
        assert exposure["exposed"] == 0
