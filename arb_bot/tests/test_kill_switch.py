"""Tests for Phase 1A: Kill switches and operational risk controls."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from arb_bot.config import RiskSettings
from arb_bot.kill_switch import (
    ConsecutiveFailureTracker,
    DailyLossTracker,
    KillSwitchManager,
    KillSwitchState,
)
from arb_bot.models import EngineState, ExecutionStyle, OpportunityKind, Side, TradeLegPlan, TradePlan
from arb_bot.risk import RiskManager


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


# ---------------------------------------------------------------------------
# DailyLossTracker
# ---------------------------------------------------------------------------


class TestDailyLossTracker:
    def test_no_cap_never_breached(self) -> None:
        t = DailyLossTracker(daily_loss_cap_usd=0.0)
        t.record_pnl(-1000.0)
        assert t.is_breached() is False

    def test_cap_breached_on_loss(self) -> None:
        t = DailyLossTracker(daily_loss_cap_usd=50.0)
        t.record_pnl(-30.0)
        assert t.is_breached() is False
        t.record_pnl(-25.0)  # total = -55
        assert t.is_breached() is True

    def test_cap_not_breached_with_profits(self) -> None:
        t = DailyLossTracker(daily_loss_cap_usd=50.0)
        t.record_pnl(-30.0)
        t.record_pnl(20.0)  # net = -10
        assert t.is_breached() is False

    def test_exact_cap_is_breached(self) -> None:
        t = DailyLossTracker(daily_loss_cap_usd=50.0)
        t.record_pnl(-50.0)
        assert t.is_breached() is True

    def test_realized_pnl_property(self) -> None:
        t = DailyLossTracker(daily_loss_cap_usd=100.0)
        t.record_pnl(-10.0)
        t.record_pnl(5.0)
        assert t.realized_pnl == pytest.approx(-5.0)


# ---------------------------------------------------------------------------
# ConsecutiveFailureTracker
# ---------------------------------------------------------------------------


class TestConsecutiveFailureTracker:
    def test_no_cap_never_breached(self) -> None:
        t = ConsecutiveFailureTracker(max_consecutive_failures=0)
        for _ in range(100):
            t.record_failure()
        assert t.is_breached() is False

    def test_breached_at_threshold(self) -> None:
        t = ConsecutiveFailureTracker(max_consecutive_failures=3)
        t.record_failure()
        t.record_failure()
        assert t.is_breached() is False
        t.record_failure()  # 3rd
        assert t.is_breached() is True

    def test_success_resets_count(self) -> None:
        t = ConsecutiveFailureTracker(max_consecutive_failures=3)
        t.record_failure()
        t.record_failure()
        t.record_success()
        assert t.count == 0
        assert t.is_breached() is False

    def test_reset(self) -> None:
        t = ConsecutiveFailureTracker(max_consecutive_failures=3)
        t.record_failure()
        t.record_failure()
        t.record_failure()
        assert t.is_breached() is True
        t.reset()
        assert t.is_breached() is False


# ---------------------------------------------------------------------------
# KillSwitchManager — global file kill switch
# ---------------------------------------------------------------------------


class TestGlobalFileKillSwitch:
    def test_no_file_not_halted(self, tmp_path: Path) -> None:
        ks = KillSwitchManager(kill_switch_file=tmp_path / "nonexistent")
        assert ks.is_halted() is False

    def test_file_exists_halted(self, tmp_path: Path) -> None:
        kill_file = tmp_path / ".kill"
        kill_file.touch()
        ks = KillSwitchManager(kill_switch_file=kill_file)
        state = ks.check()
        assert state.halted is True
        assert state.global_file_kill is True
        assert "kill switch file" in state.reason


# ---------------------------------------------------------------------------
# KillSwitchManager — global env kill switch
# ---------------------------------------------------------------------------


class TestGlobalEnvKillSwitch:
    def test_env_not_set_not_halted(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.delenv("ARB_KILL_SWITCH", raising=False)
        ks = KillSwitchManager(kill_switch_file=tmp_path / "nope")
        assert ks.is_halted() is False

    def test_env_set_halted(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("ARB_KILL_SWITCH", "1")
        ks = KillSwitchManager(kill_switch_file=tmp_path / "nope")
        state = ks.check()
        assert state.halted is True
        assert state.global_env_kill is True
        assert "ARB_KILL_SWITCH" in state.reason

    def test_env_set_false_not_halted(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("ARB_KILL_SWITCH", "false")
        ks = KillSwitchManager(kill_switch_file=tmp_path / "nope")
        assert ks.is_halted() is False


# ---------------------------------------------------------------------------
# KillSwitchManager — per-venue kill switch
# ---------------------------------------------------------------------------


class TestVenueKillSwitch:
    def test_venue_kill_halts_for_that_venue(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("ARB_KILL_KALSHI", "1")
        ks = KillSwitchManager(kill_switch_file=tmp_path / "nope")
        state = ks.check(venues={"kalshi", "polymarket"})
        assert state.halted is True
        assert state.venue_kills["kalshi"] is True
        assert state.venue_kills["polymarket"] is False

    def test_no_venue_kill_not_halted(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.delenv("ARB_KILL_KALSHI", raising=False)
        monkeypatch.delenv("ARB_KILL_POLYMARKET", raising=False)
        ks = KillSwitchManager(kill_switch_file=tmp_path / "nope")
        assert ks.is_halted(venues={"kalshi", "polymarket"}) is False

    def test_no_venues_skips_venue_check(self, tmp_path: Path) -> None:
        ks = KillSwitchManager(kill_switch_file=tmp_path / "nope")
        state = ks.check(venues=None)
        assert state.venue_kills == {}


# ---------------------------------------------------------------------------
# KillSwitchManager — daily loss cap
# ---------------------------------------------------------------------------


class TestDailyLossCap:
    def test_daily_loss_halts(self, tmp_path: Path) -> None:
        ks = KillSwitchManager(
            kill_switch_file=tmp_path / "nope",
            daily_loss_cap_usd=50.0,
        )
        ks.daily_loss.record_pnl(-60.0)
        state = ks.check()
        assert state.halted is True
        assert state.daily_loss_breached is True
        assert "daily loss cap" in state.reason

    def test_no_cap_no_halt(self, tmp_path: Path) -> None:
        ks = KillSwitchManager(
            kill_switch_file=tmp_path / "nope",
            daily_loss_cap_usd=0.0,
        )
        ks.daily_loss.record_pnl(-1000.0)
        assert ks.is_halted() is False


# ---------------------------------------------------------------------------
# KillSwitchManager — consecutive failures
# ---------------------------------------------------------------------------


class TestConsecutiveFailuresKillSwitch:
    def test_failures_halt(self, tmp_path: Path) -> None:
        ks = KillSwitchManager(
            kill_switch_file=tmp_path / "nope",
            max_consecutive_failures=3,
        )
        ks.failures.record_failure()
        ks.failures.record_failure()
        ks.failures.record_failure()
        state = ks.check()
        assert state.halted is True
        assert state.consecutive_failures_breached is True
        assert "consecutive failures" in state.reason

    def test_success_resets_no_halt(self, tmp_path: Path) -> None:
        ks = KillSwitchManager(
            kill_switch_file=tmp_path / "nope",
            max_consecutive_failures=3,
        )
        ks.failures.record_failure()
        ks.failures.record_failure()
        ks.failures.record_success()
        ks.failures.record_failure()
        assert ks.is_halted() is False


# ---------------------------------------------------------------------------
# KillSwitchManager — safe mode
# ---------------------------------------------------------------------------


class TestSafeMode:
    def test_safe_mode_halts(self, tmp_path: Path) -> None:
        ks = KillSwitchManager(kill_switch_file=tmp_path / "nope")
        ks.activate_safe_mode("unhedged position")
        state = ks.check()
        assert state.halted is True
        assert state.safe_mode_active is True
        assert "safe mode" in state.reason

    def test_deactivate_safe_mode(self, tmp_path: Path) -> None:
        ks = KillSwitchManager(kill_switch_file=tmp_path / "nope")
        ks.activate_safe_mode("test")
        ks.deactivate_safe_mode()
        assert ks.is_halted() is False
        assert ks.safe_mode is False


# ---------------------------------------------------------------------------
# KillSwitchManager — canary mode
# ---------------------------------------------------------------------------


class TestCanaryMode:
    def test_canary_mode_properties(self, tmp_path: Path) -> None:
        ks = KillSwitchManager(
            kill_switch_file=tmp_path / "nope",
            canary_mode=True,
            canary_max_dollars_per_trade=5.0,
            canary_max_contracts_per_trade=10,
        )
        assert ks.canary_mode is True
        assert ks.canary_max_dollars_per_trade == 5.0
        assert ks.canary_max_contracts_per_trade == 10

    def test_canary_mode_does_not_halt(self, tmp_path: Path) -> None:
        """Canary mode restricts sizing, but doesn't halt trading."""
        ks = KillSwitchManager(
            kill_switch_file=tmp_path / "nope",
            canary_mode=True,
        )
        assert ks.is_halted() is False


# ---------------------------------------------------------------------------
# KillSwitchManager — multiple reasons
# ---------------------------------------------------------------------------


class TestMultipleReasons:
    def test_multiple_halt_reasons(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        kill_file = tmp_path / ".kill"
        kill_file.touch()
        monkeypatch.setenv("ARB_KILL_SWITCH", "true")

        ks = KillSwitchManager(
            kill_switch_file=kill_file,
            daily_loss_cap_usd=10.0,
            max_consecutive_failures=2,
        )
        ks.daily_loss.record_pnl(-15.0)
        ks.failures.record_failure()
        ks.failures.record_failure()
        ks.activate_safe_mode("test reason")

        state = ks.check()
        assert state.halted is True
        assert state.global_file_kill is True
        assert state.global_env_kill is True
        assert state.daily_loss_breached is True
        assert state.consecutive_failures_breached is True
        assert state.safe_mode_active is True
        # All reasons concatenated
        assert "kill switch file" in state.reason
        assert "ARB_KILL_SWITCH" in state.reason
        assert "daily loss cap" in state.reason
        assert "consecutive failures" in state.reason
        assert "safe mode" in state.reason


# ---------------------------------------------------------------------------
# RiskManager integration with KillSwitchManager
# ---------------------------------------------------------------------------


class TestRiskManagerKillSwitchIntegration:
    def test_precheck_passes_without_kill_switch(self) -> None:
        """Backward compatibility: no kill switch manager = old behavior."""
        risk = RiskManager(
            RiskSettings(
                max_exposure_per_venue_usd=1000,
                max_open_markets_per_venue=20,
                market_cooldown_seconds=0,
            ),
        )
        state = EngineState(cash_by_venue={"kalshi": 500, "polymarket": 500})
        allowed, reason = risk.precheck(_cross_plan(), state)
        assert allowed, reason

    def test_precheck_blocked_by_kill_switch(self, tmp_path: Path) -> None:
        kill_file = tmp_path / ".kill"
        kill_file.touch()

        ks = KillSwitchManager(kill_switch_file=kill_file)
        risk = RiskManager(
            RiskSettings(
                max_exposure_per_venue_usd=1000,
                max_open_markets_per_venue=20,
                market_cooldown_seconds=0,
            ),
            kill_switch=ks,
        )
        state = EngineState(cash_by_venue={"kalshi": 500, "polymarket": 500})
        allowed, reason = risk.precheck(_cross_plan(), state)
        assert not allowed
        assert "kill switch" in reason
        assert "kill switch file" in reason

    def test_precheck_blocked_by_daily_loss(self, tmp_path: Path) -> None:
        ks = KillSwitchManager(
            kill_switch_file=tmp_path / "nope",
            daily_loss_cap_usd=20.0,
        )
        ks.daily_loss.record_pnl(-25.0)

        risk = RiskManager(
            RiskSettings(
                max_exposure_per_venue_usd=1000,
                max_open_markets_per_venue=20,
                market_cooldown_seconds=0,
            ),
            kill_switch=ks,
        )
        state = EngineState(cash_by_venue={"kalshi": 500, "polymarket": 500})
        allowed, reason = risk.precheck(_cross_plan(), state)
        assert not allowed
        assert "daily loss cap" in reason

    def test_precheck_blocked_by_venue_kill(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("ARB_KILL_KALSHI", "yes")

        ks = KillSwitchManager(kill_switch_file=tmp_path / "nope")
        risk = RiskManager(
            RiskSettings(
                max_exposure_per_venue_usd=1000,
                max_open_markets_per_venue=20,
                market_cooldown_seconds=0,
            ),
            kill_switch=ks,
        )
        state = EngineState(cash_by_venue={"kalshi": 500, "polymarket": 500})
        allowed, reason = risk.precheck(_cross_plan(), state)
        assert not allowed
        assert "venue kill switch" in reason
        assert "kalshi" in reason

    def test_precheck_passes_when_kill_switch_clear(self, tmp_path: Path) -> None:
        ks = KillSwitchManager(kill_switch_file=tmp_path / "nope")
        risk = RiskManager(
            RiskSettings(
                max_exposure_per_venue_usd=1000,
                max_open_markets_per_venue=20,
                market_cooldown_seconds=0,
            ),
            kill_switch=ks,
        )
        state = EngineState(cash_by_venue={"kalshi": 500, "polymarket": 500})
        allowed, reason = risk.precheck(_cross_plan(), state)
        assert allowed, reason

    def test_kill_switch_property(self, tmp_path: Path) -> None:
        ks = KillSwitchManager(kill_switch_file=tmp_path / "nope")
        risk = RiskManager(RiskSettings(), kill_switch=ks)
        assert risk.kill_switch is ks

    def test_kill_switch_property_none(self) -> None:
        risk = RiskManager(RiskSettings())
        assert risk.kill_switch is None


# ---------------------------------------------------------------------------
# KillSwitchState dataclass
# ---------------------------------------------------------------------------


class TestKillSwitchState:
    def test_defaults(self) -> None:
        state = KillSwitchState(halted=False, reason="ok")
        assert state.global_file_kill is False
        assert state.global_env_kill is False
        assert state.venue_kills == {}
        assert state.daily_loss_breached is False
        assert state.consecutive_failures_breached is False
        assert state.safe_mode_active is False
