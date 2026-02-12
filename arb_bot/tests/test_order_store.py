"""Tests for persistent order/position store (Phase 0A)."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from arb_bot.models import (
    ExecutionStyle,
    OpportunityKind,
    Side,
    TradeLegPlan,
    TradePlan,
)
from arb_bot.order_store import (
    IntentStatus,
    LegStatus,
    OrderStore,
    PositionStatus,
)


def _make_plan(
    kind: OpportunityKind = OpportunityKind.INTRA_VENUE,
    contracts: int = 10,
    capital: float = 8.0,
) -> TradePlan:
    return TradePlan(
        kind=kind,
        execution_style=ExecutionStyle.TAKER,
        legs=(
            TradeLegPlan(
                venue="kalshi",
                market_id="M1",
                side=Side.YES,
                contracts=contracts,
                limit_price=0.40,
            ),
            TradeLegPlan(
                venue="kalshi",
                market_id="M1",
                side=Side.NO,
                contracts=contracts,
                limit_price=0.40,
            ),
        ),
        contracts=contracts,
        capital_required=capital,
        capital_required_by_venue={"kalshi": capital},
        expected_profit=1.0,
        edge_per_contract=0.10,
    )


def _make_cross_plan() -> TradePlan:
    return TradePlan(
        kind=OpportunityKind.CROSS_VENUE,
        execution_style=ExecutionStyle.TAKER,
        legs=(
            TradeLegPlan(
                venue="kalshi",
                market_id="K1",
                side=Side.YES,
                contracts=5,
                limit_price=0.45,
            ),
            TradeLegPlan(
                venue="polymarket",
                market_id="P1",
                side=Side.NO,
                contracts=5,
                limit_price=0.50,
            ),
        ),
        contracts=5,
        capital_required=4.75,
        capital_required_by_venue={"kalshi": 2.25, "polymarket": 2.50},
        expected_profit=0.25,
        edge_per_contract=0.05,
        metadata={"match_key": "cross-pair-1"},
    )


@pytest.fixture()
def store(tmp_path: Path) -> OrderStore:
    db_path = tmp_path / "test_orders.db"
    s = OrderStore(db_path=db_path)
    yield s
    s.close()


# ------------------------------------------------------------------
# Intent creation and retrieval
# ------------------------------------------------------------------


def test_create_intent_returns_id(store: OrderStore) -> None:
    plan = _make_plan()
    intent_id = store.create_intent(plan)
    assert isinstance(intent_id, str)
    assert len(intent_id) == 32  # hex uuid4


def test_create_intent_stores_plan_fields(store: OrderStore) -> None:
    plan = _make_plan(kind=OpportunityKind.CROSS_VENUE, contracts=7, capital=5.6)
    intent_id = store.create_intent(plan)
    intent = store.get_intent(intent_id)

    assert intent.kind == "cross_venue"
    assert intent.execution_style == "taker"
    assert intent.contracts == 7
    assert intent.capital_required == pytest.approx(5.6)
    assert intent.expected_profit == pytest.approx(1.0)
    assert intent.edge_per_contract == pytest.approx(0.10)
    assert intent.status == IntentStatus.PENDING.value
    assert intent.total_filled == 0
    assert intent.error is None


def test_create_intent_stores_legs(store: OrderStore) -> None:
    plan = _make_plan()
    intent_id = store.create_intent(plan)
    legs = store.get_legs(intent_id)

    assert len(legs) == 2
    assert legs[0].leg_index == 0
    assert legs[0].venue == "kalshi"
    assert legs[0].market_id == "M1"
    assert legs[0].side == "yes"
    assert legs[0].contracts == 10
    assert legs[0].limit_price == pytest.approx(0.40)
    assert legs[0].status == LegStatus.PENDING.value
    assert legs[0].order_id is None

    assert legs[1].leg_index == 1
    assert legs[1].side == "no"


def test_create_intent_stores_plan_json(store: OrderStore) -> None:
    plan = _make_plan()
    intent_id = store.create_intent(plan)
    intent = store.get_intent(intent_id)

    plan_data = json.loads(intent.plan_json)
    assert plan_data["kind"] == "intra_venue"
    assert len(plan_data["legs"]) == 2
    assert plan_data["contracts"] == 10


def test_create_intent_stores_capital_by_venue(store: OrderStore) -> None:
    plan = _make_cross_plan()
    intent_id = store.create_intent(plan)
    intent = store.get_intent(intent_id)

    assert intent.capital_required_by_venue == {"kalshi": 2.25, "polymarket": 2.50}


def test_get_intent_raises_on_missing(store: OrderStore) -> None:
    with pytest.raises(KeyError):
        store.get_intent("nonexistent")


# ------------------------------------------------------------------
# Leg execution tracking
# ------------------------------------------------------------------


def test_mark_leg_submitted(store: OrderStore) -> None:
    plan = _make_plan()
    intent_id = store.create_intent(plan)

    store.mark_leg_submitted(intent_id, 0, order_id="ORD-123")
    legs = store.get_legs(intent_id)
    assert legs[0].status == LegStatus.SUBMITTED.value
    assert legs[0].order_id == "ORD-123"
    assert legs[1].status == LegStatus.PENDING.value


def test_record_leg_result_success(store: OrderStore) -> None:
    plan = _make_plan()
    intent_id = store.create_intent(plan)

    store.record_leg_result(
        intent_id, 0,
        success=True,
        order_id="ORD-123",
        filled_contracts=10,
        average_price=0.40,
        raw={"exchange_order_id": "abc"},
    )

    legs = store.get_legs(intent_id)
    assert legs[0].status == LegStatus.FILLED.value
    assert legs[0].order_id == "ORD-123"
    assert legs[0].filled_contracts == 10
    assert legs[0].average_price == pytest.approx(0.40)
    raw = json.loads(legs[0].raw_json)
    assert raw["exchange_order_id"] == "abc"


def test_record_leg_result_failure(store: OrderStore) -> None:
    plan = _make_plan()
    intent_id = store.create_intent(plan)

    store.record_leg_result(
        intent_id, 0,
        success=False,
        raw={"error": "insufficient funds"},
    )

    legs = store.get_legs(intent_id)
    assert legs[0].status == LegStatus.FAILED.value
    assert legs[0].filled_contracts == 0


# ------------------------------------------------------------------
# Intent finalization
# ------------------------------------------------------------------


def test_finalize_intent_all_filled(store: OrderStore) -> None:
    plan = _make_plan()
    intent_id = store.create_intent(plan)

    store.record_leg_result(intent_id, 0, success=True, order_id="A", filled_contracts=10, average_price=0.40)
    store.record_leg_result(intent_id, 1, success=True, order_id="B", filled_contracts=10, average_price=0.40)

    intent = store.finalize_intent(intent_id)
    assert intent.status == IntentStatus.FILLED.value
    assert intent.total_filled == 10
    assert intent.error is None


def test_finalize_intent_one_leg_failed(store: OrderStore) -> None:
    plan = _make_plan()
    intent_id = store.create_intent(plan)

    store.record_leg_result(intent_id, 0, success=True, order_id="A", filled_contracts=10, average_price=0.40)
    store.record_leg_result(intent_id, 1, success=False, raw={"error": "timeout"})

    intent = store.finalize_intent(intent_id)
    assert intent.status == IntentStatus.PARTIALLY_FILLED.value
    assert intent.total_filled == 0  # min(10, 0)
    assert "timeout" in intent.error


def test_finalize_intent_all_failed(store: OrderStore) -> None:
    plan = _make_plan()
    intent_id = store.create_intent(plan)

    store.record_leg_result(intent_id, 0, success=False, raw={"error": "err1"})
    store.record_leg_result(intent_id, 1, success=False, raw={"error": "err2"})

    intent = store.finalize_intent(intent_id)
    assert intent.status == IntentStatus.FAILED.value
    assert intent.total_filled == 0


def test_finalize_intent_partial_fills(store: OrderStore) -> None:
    plan = _make_plan()
    intent_id = store.create_intent(plan)

    store.record_leg_result(intent_id, 0, success=True, order_id="A", filled_contracts=10, average_price=0.40)
    store.record_leg_result(intent_id, 1, success=True, order_id="B", filled_contracts=7, average_price=0.40)

    intent = store.finalize_intent(intent_id)
    assert intent.status == IntentStatus.FILLED.value
    assert intent.total_filled == 7  # min(10, 7)


# ------------------------------------------------------------------
# Positions
# ------------------------------------------------------------------


def test_open_and_close_position(store: OrderStore) -> None:
    plan = _make_plan()
    intent_id = store.create_intent(plan)

    pos_id = store.open_position(intent_id, "kalshi", "M1", "yes", 10, 0.40)
    positions = store.get_open_positions()
    assert len(positions) == 1
    assert positions[0].position_id == pos_id
    assert positions[0].venue == "kalshi"
    assert positions[0].market_id == "M1"
    assert positions[0].side == "yes"
    assert positions[0].net_contracts == 10
    assert positions[0].average_entry == pytest.approx(0.40)
    assert positions[0].status == PositionStatus.OPEN.value

    store.close_position(pos_id)
    positions = store.get_open_positions()
    assert len(positions) == 0


def test_get_positions_by_venue(store: OrderStore) -> None:
    plan = _make_cross_plan()
    intent_id = store.create_intent(plan)

    store.open_position(intent_id, "kalshi", "K1", "yes", 5, 0.45)
    store.open_position(intent_id, "polymarket", "P1", "no", 5, 0.50)

    kalshi_positions = store.get_positions_by_venue("kalshi")
    assert len(kalshi_positions) == 1
    assert kalshi_positions[0].market_id == "K1"

    poly_positions = store.get_positions_by_venue("polymarket")
    assert len(poly_positions) == 1
    assert poly_positions[0].market_id == "P1"


# ------------------------------------------------------------------
# State reconstruction
# ------------------------------------------------------------------


def test_compute_locked_capital_by_venue(store: OrderStore) -> None:
    plan = _make_cross_plan()
    intent_id = store.create_intent(plan)

    store.open_position(intent_id, "kalshi", "K1", "yes", 5, 0.45)
    store.open_position(intent_id, "polymarket", "P1", "no", 5, 0.50)

    locked = store.compute_locked_capital_by_venue()
    assert locked["kalshi"] == pytest.approx(5 * 0.45)
    assert locked["polymarket"] == pytest.approx(5 * 0.50)


def test_compute_open_markets_by_venue(store: OrderStore) -> None:
    plan = _make_cross_plan()
    intent_id = store.create_intent(plan)

    store.open_position(intent_id, "kalshi", "K1", "yes", 5, 0.45)
    store.open_position(intent_id, "kalshi", "K2", "no", 3, 0.55)
    store.open_position(intent_id, "polymarket", "P1", "no", 5, 0.50)

    markets = store.compute_open_markets_by_venue()
    assert markets["kalshi"] == {"K1", "K2"}
    assert markets["polymarket"] == {"P1"}


def test_compute_locked_excludes_closed(store: OrderStore) -> None:
    plan = _make_plan()
    intent_id = store.create_intent(plan)

    pos_id = store.open_position(intent_id, "kalshi", "M1", "yes", 10, 0.40)
    store.close_position(pos_id)

    locked = store.compute_locked_capital_by_venue()
    assert locked == {}


# ------------------------------------------------------------------
# Open intents query (for crash recovery)
# ------------------------------------------------------------------


def test_get_open_intents(store: OrderStore) -> None:
    plan = _make_plan()
    id1 = store.create_intent(plan)  # pending
    id2 = store.create_intent(plan)  # will be submitted
    id3 = store.create_intent(plan)  # will be filled

    store.mark_intent_submitted(id2)

    store.record_leg_result(id3, 0, success=True, filled_contracts=10, average_price=0.40)
    store.record_leg_result(id3, 1, success=True, filled_contracts=10, average_price=0.40)
    store.finalize_intent(id3)

    open_intents = store.get_open_intents()
    open_ids = {i.intent_id for i in open_intents}
    assert id1 in open_ids  # pending
    assert id2 in open_ids  # submitted
    assert id3 not in open_ids  # filled


# ------------------------------------------------------------------
# Persistence across reopen
# ------------------------------------------------------------------


def test_data_survives_close_and_reopen(tmp_path: Path) -> None:
    db_path = tmp_path / "persist_test.db"
    plan = _make_plan()

    store1 = OrderStore(db_path=db_path)
    intent_id = store1.create_intent(plan)
    store1.open_position(intent_id, "kalshi", "M1", "yes", 10, 0.40)
    store1.close()

    store2 = OrderStore(db_path=db_path)
    intent = store2.get_intent(intent_id)
    assert intent.kind == "intra_venue"

    positions = store2.get_open_positions()
    assert len(positions) == 1
    assert positions[0].net_contracts == 10

    locked = store2.compute_locked_capital_by_venue()
    assert locked["kalshi"] == pytest.approx(4.0)
    store2.close()


# ------------------------------------------------------------------
# Intent submitted status
# ------------------------------------------------------------------


def test_mark_intent_submitted(store: OrderStore) -> None:
    plan = _make_plan()
    intent_id = store.create_intent(plan)

    store.mark_intent_submitted(intent_id)
    intent = store.get_intent(intent_id)
    assert intent.status == IntentStatus.SUBMITTED.value


# ------------------------------------------------------------------
# Filled intents since timestamp
# ------------------------------------------------------------------


def test_get_filled_intents_since(store: OrderStore) -> None:
    import time

    plan = _make_plan()
    before = time.time()

    id1 = store.create_intent(plan)
    store.record_leg_result(id1, 0, success=True, filled_contracts=10, average_price=0.40)
    store.record_leg_result(id1, 1, success=True, filled_contracts=10, average_price=0.40)
    store.finalize_intent(id1)

    filled = store.get_filled_intents_since(before)
    assert len(filled) == 1
    assert filled[0].intent_id == id1

    future = time.time() + 1000
    filled = store.get_filled_intents_since(future)
    assert len(filled) == 0


# ------------------------------------------------------------------
# Multiple intents isolation
# ------------------------------------------------------------------


def test_multiple_intents_isolated(store: OrderStore) -> None:
    plan1 = _make_plan(contracts=5, capital=4.0)
    plan2 = _make_cross_plan()

    id1 = store.create_intent(plan1)
    id2 = store.create_intent(plan2)

    legs1 = store.get_legs(id1)
    legs2 = store.get_legs(id2)
    assert len(legs1) == 2
    assert len(legs2) == 2
    assert legs1[0].venue == "kalshi"
    assert legs2[1].venue == "polymarket"

    store.record_leg_result(id1, 0, success=True, filled_contracts=5, average_price=0.40)
    store.record_leg_result(id1, 1, success=True, filled_contracts=5, average_price=0.40)
    store.finalize_intent(id1)

    intent1 = store.get_intent(id1)
    intent2 = store.get_intent(id2)
    assert intent1.status == IntentStatus.FILLED.value
    assert intent2.status == IntentStatus.PENDING.value
