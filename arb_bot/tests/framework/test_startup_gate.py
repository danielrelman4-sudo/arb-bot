"""Tests for Phase 1E: Pre-run hard gate wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from arb_bot.framework.kill_switch import KillSwitchManager
from arb_bot.framework.startup_gate import (
    GateChainResult,
    GateResult,
    StartupGate,
    StartupGateChain,
    config_sanity_gate,
    kill_switch_gate,
    recovery_gate,
    safe_mode_gate,
    venue_connectivity_gate,
)


# ---------------------------------------------------------------------------
# GateResult
# ---------------------------------------------------------------------------


class TestGateResult:
    def test_fields(self) -> None:
        r = GateResult(name="test", passed=True, message="ok", elapsed_seconds=0.01)
        assert r.name == "test"
        assert r.passed is True
        assert r.message == "ok"


# ---------------------------------------------------------------------------
# GateChainResult
# ---------------------------------------------------------------------------


class TestGateChainResult:
    def test_passed_summary(self) -> None:
        results = (
            GateResult(name="a", passed=True, message="ok"),
            GateResult(name="b", passed=True, message="ok"),
        )
        r = GateChainResult(passed=True, results=results, elapsed_seconds=0.05)
        assert "All 2 gates passed" in r.summary
        assert r.failed_gates == []

    def test_failed_summary(self) -> None:
        results = (
            GateResult(name="a", passed=True, message="ok"),
            GateResult(name="b", passed=False, message="fail"),
        )
        r = GateChainResult(passed=False, results=results)
        assert "FAILED" in r.summary
        assert "b" in r.summary
        assert len(r.failed_gates) == 1


# ---------------------------------------------------------------------------
# StartupGate
# ---------------------------------------------------------------------------


class TestStartupGate:
    def test_passing_gate(self) -> None:
        gate = StartupGate("test", lambda: (True, "ok"))
        result = gate.check()
        assert result.passed is True
        assert result.message == "ok"
        assert result.name == "test"

    def test_failing_gate(self) -> None:
        gate = StartupGate("test", lambda: (False, "bad"))
        result = gate.check()
        assert result.passed is False

    def test_exception_in_gate(self) -> None:
        def _raise() -> tuple[bool, str]:
            raise RuntimeError("boom")

        gate = StartupGate("test", _raise)
        result = gate.check()
        assert result.passed is False
        assert "boom" in result.message

    def test_required_property(self) -> None:
        gate = StartupGate("test", lambda: (True, "ok"), required=False)
        assert gate.required is False


# ---------------------------------------------------------------------------
# StartupGateChain — basic
# ---------------------------------------------------------------------------


class TestGateChainBasic:
    def test_all_pass(self) -> None:
        chain = StartupGateChain()
        chain.add("a", lambda: (True, "ok"))
        chain.add("b", lambda: (True, "fine"))
        result = chain.run()
        assert result.passed is True
        assert len(result.results) == 2

    def test_one_required_fails(self) -> None:
        chain = StartupGateChain()
        chain.add("a", lambda: (True, "ok"))
        chain.add("b", lambda: (False, "fail"))
        result = chain.run()
        assert result.passed is False
        assert len(result.failed_gates) == 1

    def test_optional_failure_still_passes(self) -> None:
        chain = StartupGateChain()
        chain.add("a", lambda: (True, "ok"))
        chain.add("b", lambda: (False, "warn"), required=False)
        result = chain.run()
        assert result.passed is True  # Optional gate doesn't block

    def test_empty_chain_passes(self) -> None:
        chain = StartupGateChain()
        result = chain.run()
        assert result.passed is True
        assert len(result.results) == 0

    def test_gate_count(self) -> None:
        chain = StartupGateChain()
        chain.add("a", lambda: (True, "ok"))
        chain.add("b", lambda: (True, "ok"))
        assert chain.gate_count == 2


# ---------------------------------------------------------------------------
# StartupGateChain — fail fast
# ---------------------------------------------------------------------------


class TestGateChainFailFast:
    def test_fail_fast_stops_at_first_failure(self) -> None:
        call_log: list[str] = []

        def _gate_a() -> tuple[bool, str]:
            call_log.append("a")
            return False, "fail"

        def _gate_b() -> tuple[bool, str]:
            call_log.append("b")
            return True, "ok"

        chain = StartupGateChain(fail_fast=True)
        chain.add("a", _gate_a)
        chain.add("b", _gate_b)
        result = chain.run()

        assert result.passed is False
        assert call_log == ["a"]  # b was never called
        assert len(result.results) == 1

    def test_fail_fast_skips_optional_failures(self) -> None:
        call_log: list[str] = []

        def _opt() -> tuple[bool, str]:
            call_log.append("opt")
            return False, "warn"

        def _req() -> tuple[bool, str]:
            call_log.append("req")
            return True, "ok"

        chain = StartupGateChain(fail_fast=True)
        chain.add("optional", _opt, required=False)
        chain.add("required", _req)
        result = chain.run()

        assert result.passed is True
        assert call_log == ["opt", "req"]  # Both ran


# ---------------------------------------------------------------------------
# Built-in gate: kill_switch_gate
# ---------------------------------------------------------------------------


class TestKillSwitchGate:
    def test_passes_when_clear(self, tmp_path: Path) -> None:
        ks = KillSwitchManager(kill_switch_file=tmp_path / "nope")
        gate_fn = kill_switch_gate(ks)
        passed, msg = gate_fn()
        assert passed is True

    def test_fails_when_halted(self, tmp_path: Path) -> None:
        kill_file = tmp_path / ".kill"
        kill_file.touch()
        ks = KillSwitchManager(kill_switch_file=kill_file)
        gate_fn = kill_switch_gate(ks)
        passed, msg = gate_fn()
        assert passed is False
        assert "kill switch" in msg


# ---------------------------------------------------------------------------
# Built-in gate: safe_mode_gate
# ---------------------------------------------------------------------------


class TestSafeModeGate:
    def test_passes_when_not_active(self, tmp_path: Path) -> None:
        ks = KillSwitchManager(kill_switch_file=tmp_path / "nope")
        gate_fn = safe_mode_gate(ks)
        passed, msg = gate_fn()
        assert passed is True

    def test_fails_when_active(self, tmp_path: Path) -> None:
        ks = KillSwitchManager(kill_switch_file=tmp_path / "nope")
        ks.activate_safe_mode("test reason")
        gate_fn = safe_mode_gate(ks)
        passed, msg = gate_fn()
        assert passed is False
        assert "safe mode" in msg


# ---------------------------------------------------------------------------
# Built-in gate: venue_connectivity_gate
# ---------------------------------------------------------------------------


class TestVenueConnectivityGate:
    def test_none_adapter_fails(self) -> None:
        gate_fn = venue_connectivity_gate("kalshi", None)
        passed, msg = gate_fn()
        assert passed is False
        assert "None" in msg

    def test_connected_adapter_passes(self) -> None:
        class FakeAdapter:
            def is_connected(self) -> bool:
                return True

        gate_fn = venue_connectivity_gate("kalshi", FakeAdapter())
        passed, msg = gate_fn()
        assert passed is True

    def test_disconnected_adapter_fails(self) -> None:
        class FakeAdapter:
            def is_connected(self) -> bool:
                return False

        gate_fn = venue_connectivity_gate("kalshi", FakeAdapter())
        passed, msg = gate_fn()
        assert passed is False
        assert "not connected" in msg

    def test_adapter_without_method_passes(self) -> None:
        gate_fn = venue_connectivity_gate("kalshi", object())
        passed, msg = gate_fn()
        assert passed is True
        assert "no connectivity check" in msg


# ---------------------------------------------------------------------------
# Built-in gate: config_sanity_gate
# ---------------------------------------------------------------------------


class TestConfigSanityGate:
    def test_passes_with_valid_config(self) -> None:
        gate_fn = config_sanity_gate(
            live_mode=True,
            bankroll_by_venue={"kalshi": 100.0, "polymarket": 100.0},
            required_venues=["kalshi", "polymarket"],
        )
        passed, msg = gate_fn()
        assert passed is True

    def test_fails_with_zero_bankroll(self) -> None:
        gate_fn = config_sanity_gate(
            live_mode=True,
            bankroll_by_venue={"kalshi": 0.0},
            required_venues=["kalshi"],
        )
        passed, msg = gate_fn()
        assert passed is False
        assert "kalshi" in msg

    def test_passes_in_dry_run(self) -> None:
        """In dry run mode, bankroll validation is skipped."""
        gate_fn = config_sanity_gate(
            live_mode=False,
            bankroll_by_venue={},
            required_venues=["kalshi"],
        )
        passed, msg = gate_fn()
        assert passed is True


# ---------------------------------------------------------------------------
# Built-in gate: recovery_gate
# ---------------------------------------------------------------------------


class TestRecoveryGate:
    def test_passes_with_no_report(self) -> None:
        gate_fn = recovery_gate(None)
        passed, msg = gate_fn()
        assert passed is True
        assert "fresh start" in msg

    def test_passes_with_clean_report(self) -> None:
        @dataclass
        class FakeReport:
            safe_mode: bool = False
            open_positions: int = 2

        gate_fn = recovery_gate(FakeReport())
        passed, msg = gate_fn()
        assert passed is True
        assert "2 open positions" in msg

    def test_fails_with_safe_mode_report(self) -> None:
        @dataclass
        class FakeReport:
            safe_mode: bool = True
            safe_mode_reason: str = "unhedged positions"

        gate_fn = recovery_gate(FakeReport())
        passed, msg = gate_fn()
        assert passed is False
        assert "unhedged" in msg


# ---------------------------------------------------------------------------
# Integration: full gate chain
# ---------------------------------------------------------------------------


class TestFullGateChain:
    def test_realistic_gate_chain(self, tmp_path: Path) -> None:
        ks = KillSwitchManager(kill_switch_file=tmp_path / "nope")

        chain = StartupGateChain()
        chain.add("kill_switch", kill_switch_gate(ks))
        chain.add("safe_mode", safe_mode_gate(ks))
        chain.add("config", config_sanity_gate(
            live_mode=False,
            bankroll_by_venue={},
        ))
        chain.add("recovery", recovery_gate(None))

        result = chain.run()
        assert result.passed is True
        assert len(result.results) == 4
