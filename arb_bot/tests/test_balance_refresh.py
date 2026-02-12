"""Tests for Phase 0G: Automatic balance refresh."""

from __future__ import annotations

import time

import pytest

from arb_bot.balance_refresh import BalanceRefresher, BalanceSnapshot, ReconciliationResult
from arb_bot.models import EngineState


# ---------------------------------------------------------------------------
# BalanceSnapshot
# ---------------------------------------------------------------------------


class TestBalanceSnapshot:
    def test_fields(self) -> None:
        snap = BalanceSnapshot(venue="kalshi", available_cash=95.0, fetched_at=1000.0)
        assert snap.venue == "kalshi"
        assert snap.available_cash == 95.0


# ---------------------------------------------------------------------------
# BalanceRefresher basics
# ---------------------------------------------------------------------------


class TestRefresherBasics:
    def test_initial_should_refresh(self) -> None:
        r = BalanceRefresher(refresh_interval=60.0)
        assert r.should_refresh() is True

    def test_mark_refreshed_prevents_immediate_recheck(self) -> None:
        r = BalanceRefresher(refresh_interval=60.0)
        r.mark_refreshed()
        assert r.should_refresh() is False

    def test_refresh_interval_floor(self) -> None:
        r = BalanceRefresher(refresh_interval=1.0)
        assert r.refresh_interval == 5.0  # floored at 5s

    def test_record_balance(self) -> None:
        r = BalanceRefresher()
        snap = r.record_balance("kalshi", 95.0)
        assert snap.venue == "kalshi"
        assert snap.available_cash == 95.0
        assert r.get_snapshot("kalshi") is snap

    def test_get_snapshot_missing(self) -> None:
        r = BalanceRefresher()
        assert r.get_snapshot("nonexistent") is None


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------


class TestReconciliation:
    def test_no_discrepancy(self) -> None:
        """Engine and exchange agree — no changes."""
        state = EngineState(
            cash_by_venue={"kalshi": 90.0},
            locked_capital_by_venue={"kalshi": 10.0},
        )
        r = BalanceRefresher()
        r.record_balance("kalshi", 100.0)  # 90 + 10 = 100

        results = r.reconcile(state)

        assert len(results) == 1
        assert results[0].discrepancy == pytest.approx(0.0)
        assert results[0].released_locked == 0.0
        assert state.cash_by_venue["kalshi"] == pytest.approx(90.0)

    def test_stale_locked_released(self) -> None:
        """Engine shows more locked than exchange total → release stale locks."""
        state = EngineState(
            cash_by_venue={"kalshi": 50.0},
            locked_capital_by_venue={"kalshi": 60.0},
        )
        # Engine total = 110, exchange = 80 → discrepancy = 30
        r = BalanceRefresher(stale_threshold=0.1)
        r.record_balance("kalshi", 80.0)

        results = r.reconcile(state)

        assert len(results) == 1
        assert results[0].released_locked == pytest.approx(30.0)
        assert state.locked_capital_by_venue["kalshi"] == pytest.approx(30.0)
        assert state.cash_by_venue["kalshi"] == pytest.approx(50.0)  # 80 - 30

    def test_small_discrepancy_within_threshold(self) -> None:
        """Small discrepancy within threshold — sync cash, don't release locked."""
        state = EngineState(
            cash_by_venue={"kalshi": 50.0},
            locked_capital_by_venue={"kalshi": 10.0},
        )
        # Engine total = 60, exchange = 59 → discrepancy = 1
        # threshold = 59 * 0.1 = 5.9 → 1 < 5.9, so no release
        r = BalanceRefresher(stale_threshold=0.1)
        r.record_balance("kalshi", 59.0)

        results = r.reconcile(state)

        assert results[0].released_locked == 0.0
        assert state.locked_capital_by_venue["kalshi"] == pytest.approx(10.0)
        assert state.cash_by_venue["kalshi"] == pytest.approx(49.0)  # 59 - 10

    def test_exchange_higher_than_engine(self) -> None:
        """Exchange shows more cash → update engine cash upward."""
        state = EngineState(
            cash_by_venue={"kalshi": 50.0},
            locked_capital_by_venue={"kalshi": 10.0},
        )
        # Engine total = 60, exchange = 70 → discrepancy = -10
        r = BalanceRefresher()
        r.record_balance("kalshi", 70.0)

        results = r.reconcile(state)

        assert results[0].discrepancy == pytest.approx(-10.0)
        assert state.cash_by_venue["kalshi"] == pytest.approx(60.0)  # 70 - 10

    def test_no_locked_capital(self) -> None:
        """No locked capital — just sync cash."""
        state = EngineState(
            cash_by_venue={"kalshi": 100.0},
        )
        r = BalanceRefresher()
        r.record_balance("kalshi", 95.0)

        results = r.reconcile(state)

        # engine_total = 100, exchange = 95 → discrepancy = 5
        # But no locked capital to release, so just sync cash.
        assert results[0].released_locked == 0.0
        assert state.cash_by_venue["kalshi"] == pytest.approx(95.0)

    def test_multi_venue(self) -> None:
        """Reconcile across multiple venues."""
        state = EngineState(
            cash_by_venue={"kalshi": 90.0, "polymarket": 80.0},
            locked_capital_by_venue={"kalshi": 10.0, "polymarket": 20.0},
        )
        r = BalanceRefresher()
        r.record_balance("kalshi", 100.0)  # matches
        r.record_balance("polymarket", 110.0)  # exchange higher

        results = r.reconcile(state)

        assert len(results) == 2
        kalshi = next(x for x in results if x.venue == "kalshi")
        poly = next(x for x in results if x.venue == "polymarket")

        assert kalshi.discrepancy == pytest.approx(0.0)
        assert poly.discrepancy == pytest.approx(-10.0)
        assert state.cash_by_venue["polymarket"] == pytest.approx(90.0)  # 110 - 20

    def test_release_capped_at_locked(self) -> None:
        """Release can't exceed the locked amount."""
        state = EngineState(
            cash_by_venue={"kalshi": 90.0},
            locked_capital_by_venue={"kalshi": 5.0},
        )
        # Engine total = 95, exchange = 80 → discrepancy = 15
        # But only 5 locked to release.
        r = BalanceRefresher(stale_threshold=0.0)
        r.record_balance("kalshi", 80.0)

        results = r.reconcile(state)

        assert results[0].released_locked == pytest.approx(5.0)
        assert state.locked_capital_by_venue["kalshi"] == pytest.approx(0.0)

    def test_zero_threshold_always_releases(self) -> None:
        """With threshold=0, any discrepancy triggers release."""
        state = EngineState(
            cash_by_venue={"kalshi": 50.0},
            locked_capital_by_venue={"kalshi": 10.0},
        )
        r = BalanceRefresher(stale_threshold=0.0)
        r.record_balance("kalshi", 59.0)  # discrepancy = 1

        results = r.reconcile(state)

        assert results[0].released_locked == pytest.approx(1.0)
        assert state.locked_capital_by_venue["kalshi"] == pytest.approx(9.0)


# ---------------------------------------------------------------------------
# ReconciliationResult
# ---------------------------------------------------------------------------


class TestReconciliationResult:
    def test_defaults(self) -> None:
        r = ReconciliationResult(
            venue="x",
            exchange_balance=100.0,
            engine_cash=90.0,
            engine_locked=10.0,
            engine_total=100.0,
            discrepancy=0.0,
        )
        assert r.released_locked == 0.0
