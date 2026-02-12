"""Tests for Phase 5F: Dual-venue bootstrap budget enforcement."""

from __future__ import annotations

import pytest

from arb_bot.bootstrap_budget import (
    BootstrapBudget,
    BootstrapBudgetConfig,
    BootstrapBudgetReport,
    VenueBudgetState,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bb(**kw) -> BootstrapBudget:
    return BootstrapBudget(BootstrapBudgetConfig(**kw))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self) -> None:
        cfg = BootstrapBudgetConfig()
        assert cfg.overlap_reserve_fraction == 0.30
        assert cfg.discovery_fraction == 0.20
        assert cfg.min_per_venue == 20
        assert cfg.enforce_balance is True
        assert cfg.balance_max_fraction == 0.65

    def test_frozen(self) -> None:
        cfg = BootstrapBudgetConfig()
        with pytest.raises(AttributeError):
            cfg.overlap_reserve_fraction = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Register venue
# ---------------------------------------------------------------------------


class TestRegister:
    def test_register(self) -> None:
        bb = _bb()
        bb.register_venue("kalshi", budget=200)
        assert "kalshi" in bb.registered_venues()

    def test_budget_and_reserves(self) -> None:
        bb = _bb(overlap_reserve_fraction=0.30, discovery_fraction=0.20)
        bb.register_venue("kalshi", budget=100)
        state = bb.get_state("kalshi")
        assert state is not None
        assert state.total_budget == 100
        assert state.overlap_reserve == 30
        assert state.discovery_reserve == 20

    def test_min_budget(self) -> None:
        bb = _bb(min_per_venue=50)
        bb.register_venue("kalshi", budget=10)
        state = bb.get_state("kalshi")
        assert state is not None
        assert state.total_budget == 50


# ---------------------------------------------------------------------------
# Reserve overlap
# ---------------------------------------------------------------------------


class TestReserveOverlap:
    def test_reserves_updated(self) -> None:
        bb = _bb(overlap_reserve_fraction=0.10)
        bb.register_venue("kalshi", budget=200)
        bb.reserve_overlap(["BTC", "ETH", "SOL"], per_symbol=10)
        state = bb.get_state("kalshi")
        assert state is not None
        # 3 symbols * 10 = 30 > original 10% of 200 = 20.
        assert state.overlap_reserve >= 30

    def test_reserves_capped_at_budget(self) -> None:
        bb = _bb(overlap_reserve_fraction=0.10, discovery_fraction=0.10)
        bb.register_venue("kalshi", budget=50)
        bb.reserve_overlap([f"sym_{i}" for i in range(100)], per_symbol=10)
        state = bb.get_state("kalshi")
        assert state is not None
        # Can't exceed total_budget - discovery_reserve.
        assert state.overlap_reserve <= 50


# ---------------------------------------------------------------------------
# Can spend / spend — overlap
# ---------------------------------------------------------------------------


class TestSpendOverlap:
    def test_overlap_allowed(self) -> None:
        bb = _bb(enforce_balance=False)
        bb.register_venue("kalshi", budget=100)
        assert bb.can_spend("kalshi", category="overlap", count=1) is True
        assert bb.spend("kalshi", category="overlap", count=1) is True

    def test_overlap_exhausted(self) -> None:
        bb = _bb(overlap_reserve_fraction=0.10, enforce_balance=False)
        bb.register_venue("kalshi", budget=100)
        # Overlap reserve = 10.
        for _ in range(10):
            bb.spend("kalshi", category="overlap", count=1)
        assert bb.can_spend("kalshi", category="overlap", count=1) is False

    def test_overlap_tracks_separately(self) -> None:
        bb = _bb(overlap_reserve_fraction=0.30, enforce_balance=False)
        bb.register_venue("kalshi", budget=100)
        bb.spend("kalshi", category="overlap", count=5)
        state = bb.get_state("kalshi")
        assert state is not None
        assert state.spent_overlap == 5
        assert state.overlap_remaining == 25


# ---------------------------------------------------------------------------
# Can spend / spend — discovery
# ---------------------------------------------------------------------------


class TestSpendDiscovery:
    def test_discovery_allowed(self) -> None:
        bb = _bb(enforce_balance=False)
        bb.register_venue("kalshi", budget=100)
        assert bb.can_spend("kalshi", category="discovery", count=1) is True

    def test_discovery_exhausted(self) -> None:
        bb = _bb(discovery_fraction=0.10, enforce_balance=False)
        bb.register_venue("kalshi", budget=100)
        for _ in range(10):
            bb.spend("kalshi", category="discovery", count=1)
        assert bb.can_spend("kalshi", category="discovery", count=1) is False


# ---------------------------------------------------------------------------
# Can spend / spend — general
# ---------------------------------------------------------------------------


class TestSpendGeneral:
    def test_general_allowed(self) -> None:
        bb = _bb(
            overlap_reserve_fraction=0.30,
            discovery_fraction=0.20,
            enforce_balance=False,
        )
        bb.register_venue("kalshi", budget=100)
        # General budget = 100 - 30 - 20 = 50.
        assert bb.can_spend("kalshi", category="general", count=1) is True

    def test_general_cant_eat_reserves(self) -> None:
        bb = _bb(
            overlap_reserve_fraction=0.40,
            discovery_fraction=0.40,
            enforce_balance=False,
        )
        bb.register_venue("kalshi", budget=100)
        # General budget = 100 - 40 - 40 = 20.
        for _ in range(20):
            bb.spend("kalshi", category="general", count=1)
        assert bb.can_spend("kalshi", category="general", count=1) is False
        # But overlap and discovery still available.
        assert bb.can_spend("kalshi", category="overlap", count=1) is True
        assert bb.can_spend("kalshi", category="discovery", count=1) is True

    def test_total_budget_exhausted(self) -> None:
        bb = _bb(
            overlap_reserve_fraction=0.0,
            discovery_fraction=0.0,
            enforce_balance=False,
            min_per_venue=10,
        )
        bb.register_venue("kalshi", budget=10)
        for _ in range(10):
            bb.spend("kalshi", category="general", count=1)
        assert bb.can_spend("kalshi", category="general", count=1) is False


# ---------------------------------------------------------------------------
# Balance enforcement
# ---------------------------------------------------------------------------


class TestBalanceEnforcement:
    def test_balance_prevents_hogging(self) -> None:
        bb = _bb(
            overlap_reserve_fraction=0.0,
            discovery_fraction=0.0,
            enforce_balance=True,
            balance_max_fraction=0.60,
        )
        bb.register_venue("kalshi", budget=100)
        bb.register_venue("polymarket", budget=100)
        # Total = 200. Max per venue = 120.
        for _ in range(120):
            bb.spend("kalshi", category="general", count=1)
        assert bb.can_spend("kalshi", category="general", count=1) is False

    def test_balance_not_enforced_when_disabled(self) -> None:
        bb = _bb(
            overlap_reserve_fraction=0.0,
            discovery_fraction=0.0,
            enforce_balance=False,
        )
        bb.register_venue("kalshi", budget=200)
        bb.register_venue("polymarket", budget=200)
        for _ in range(200):
            bb.spend("kalshi", category="general", count=1)
        # All 200 spent — allowed without balance check.
        state = bb.get_state("kalshi")
        assert state is not None
        assert state.total_spent == 200

    def test_balance_single_venue_no_check(self) -> None:
        bb = _bb(
            overlap_reserve_fraction=0.0,
            discovery_fraction=0.0,
            enforce_balance=True,
            balance_max_fraction=0.50,
        )
        bb.register_venue("kalshi", budget=100)
        # Single venue — balance check skipped.
        assert bb.can_spend("kalshi", category="general", count=100) is True


# ---------------------------------------------------------------------------
# Nonexistent venue
# ---------------------------------------------------------------------------


class TestNonexistent:
    def test_can_spend_false(self) -> None:
        bb = _bb()
        assert bb.can_spend("nope", count=1) is False

    def test_spend_false(self) -> None:
        bb = _bb()
        assert bb.spend("nope", count=1) is False


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


class TestReport:
    def test_initial_report(self) -> None:
        bb = _bb()
        bb.register_venue("kalshi", budget=100)
        bb.register_venue("polymarket", budget=100)
        report = bb.report()
        assert report.total_budget == 200
        assert report.total_spent == 0
        assert report.total_remaining == 200
        assert report.any_venue_starved is False

    def test_with_overlap(self) -> None:
        bb = _bb()
        bb.register_venue("kalshi", budget=100)
        bb.reserve_overlap(["BTC", "ETH"])
        report = bb.report()
        assert report.overlap_symbols_count == 2


# ---------------------------------------------------------------------------
# VenueBudgetState properties
# ---------------------------------------------------------------------------


class TestVenueState:
    def test_total_spent(self) -> None:
        bb = _bb(enforce_balance=False)
        bb.register_venue("kalshi", budget=100)
        bb.spend("kalshi", category="overlap", count=5)
        bb.spend("kalshi", category="discovery", count=3)
        bb.spend("kalshi", category="general", count=2)
        state = bb.get_state("kalshi")
        assert state is not None
        assert state.total_spent == 10

    def test_remaining(self) -> None:
        bb = _bb(enforce_balance=False)
        bb.register_venue("kalshi", budget=100)
        bb.spend("kalshi", category="general", count=40)
        state = bb.get_state("kalshi")
        assert state is not None
        assert state.remaining == 60


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clears_all(self) -> None:
        bb = _bb()
        bb.register_venue("kalshi", budget=100)
        bb.reserve_overlap(["BTC"])
        bb.clear()
        assert bb.registered_venues() == []


# ---------------------------------------------------------------------------
# Config property
# ---------------------------------------------------------------------------


class TestConfigProperty:
    def test_config_accessible(self) -> None:
        cfg = BootstrapBudgetConfig(min_per_venue=50)
        bb = BootstrapBudget(cfg)
        assert bb.config.min_per_venue == 50


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_balanced_bootstrap(self) -> None:
        """Both venues bootstrap without starvation."""
        bb = _bb(
            overlap_reserve_fraction=0.30,
            discovery_fraction=0.20,
            enforce_balance=True,
            balance_max_fraction=0.60,
        )
        bb.register_venue("kalshi", budget=100)
        bb.register_venue("polymarket", budget=100)
        bb.reserve_overlap(["BTC-50K", "ETH-3K"], per_symbol=5)

        # Discovery phase.
        for _ in range(20):
            bb.spend("kalshi", category="discovery", count=1)
            bb.spend("polymarket", category="discovery", count=1)

        # Overlap loading.
        for _ in range(10):
            bb.spend("kalshi", category="overlap", count=1)
            bb.spend("polymarket", category="overlap", count=1)

        # General loading.
        for _ in range(20):
            bb.spend("kalshi", category="general", count=1)
            bb.spend("polymarket", category="general", count=1)

        report = bb.report()
        assert report.any_venue_starved is False

        # Both venues spent equally.
        k_state = bb.get_state("kalshi")
        p_state = bb.get_state("polymarket")
        assert k_state is not None and p_state is not None
        assert k_state.total_spent == p_state.total_spent == 50
