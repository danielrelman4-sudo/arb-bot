"""Pre-run hard gate wrapper (Phase 1E).

A chain of startup gates that must ALL pass before the engine begins
trading. Each gate is a lightweight check that validates a specific
precondition (connectivity, data freshness, config sanity, etc.).

If any gate fails, the engine aborts startup with a clear error message
identifying which gate(s) failed and why.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class GateResult:
    """Result of a single gate check."""

    name: str
    passed: bool
    message: str
    elapsed_seconds: float = 0.0


@dataclass(frozen=True)
class GateChainResult:
    """Result of running the entire gate chain."""

    passed: bool
    results: tuple[GateResult, ...]
    elapsed_seconds: float = 0.0

    @property
    def failed_gates(self) -> list[GateResult]:
        return [r for r in self.results if not r.passed]

    @property
    def summary(self) -> str:
        if self.passed:
            return f"All {len(self.results)} gates passed ({self.elapsed_seconds:.2f}s)"
        failed = self.failed_gates
        names = ", ".join(g.name for g in failed)
        return f"{len(failed)}/{len(self.results)} gates FAILED: {names}"


# Type alias for a gate check function.
# Returns (passed: bool, message: str).
GateCheckFn = Callable[[], tuple[bool, str]]


class StartupGate:
    """A single named gate with a check function."""

    def __init__(self, name: str, check_fn: GateCheckFn, required: bool = True) -> None:
        self._name = name
        self._check_fn = check_fn
        self._required = required

    @property
    def name(self) -> str:
        return self._name

    @property
    def required(self) -> bool:
        return self._required

    def check(self) -> GateResult:
        start = time.monotonic()
        try:
            passed, message = self._check_fn()
        except Exception as exc:
            passed = False
            message = f"gate raised exception: {exc}"
        elapsed = time.monotonic() - start
        return GateResult(
            name=self._name,
            passed=passed,
            message=message,
            elapsed_seconds=elapsed,
        )


class StartupGateChain:
    """Ordered chain of startup gates.

    Gates run in registration order. If `fail_fast` is True, the chain
    stops at the first required gate failure. Otherwise, all gates run
    and a combined result is returned.
    """

    def __init__(self, fail_fast: bool = False) -> None:
        self._gates: list[StartupGate] = []
        self._fail_fast = fail_fast

    def add(self, name: str, check_fn: GateCheckFn, required: bool = True) -> None:
        """Register a gate."""
        self._gates.append(StartupGate(name, check_fn, required))

    def run(self) -> GateChainResult:
        """Run all gates and return the combined result."""
        start = time.monotonic()
        results: list[GateResult] = []
        overall_pass = True

        for gate in self._gates:
            result = gate.check()
            results.append(result)

            if result.passed:
                LOGGER.info("Gate '%s' PASSED: %s (%.3fs)", gate.name, result.message, result.elapsed_seconds)
            else:
                level = "ERROR" if gate.required else "WARNING"
                LOGGER.log(
                    logging.ERROR if gate.required else logging.WARNING,
                    "Gate '%s' FAILED: %s (%.3fs)",
                    gate.name,
                    result.message,
                    result.elapsed_seconds,
                )
                if gate.required:
                    overall_pass = False
                    if self._fail_fast:
                        break

        elapsed = time.monotonic() - start
        chain_result = GateChainResult(
            passed=overall_pass,
            results=tuple(results),
            elapsed_seconds=elapsed,
        )

        if overall_pass:
            LOGGER.info("Startup gate chain: %s", chain_result.summary)
        else:
            LOGGER.error("Startup gate chain: %s", chain_result.summary)

        return chain_result

    @property
    def gate_count(self) -> int:
        return len(self._gates)


# ---------------------------------------------------------------------------
# Built-in gate factories
# ---------------------------------------------------------------------------


def kill_switch_gate(kill_switch_manager: Any) -> GateCheckFn:
    """Gate that checks the kill switch is not engaged."""
    def _check() -> tuple[bool, str]:
        if kill_switch_manager.is_halted():
            state = kill_switch_manager.check()
            return False, f"kill switch engaged: {state.reason}"
        return True, "kill switch clear"
    return _check


def venue_connectivity_gate(venue: str, adapter: Any) -> GateCheckFn:
    """Gate that verifies we can reach a venue's API.

    Expects the adapter to have a `ping()` or `is_connected()` method.
    Falls back to checking if the adapter exists.
    """
    def _check() -> tuple[bool, str]:
        if adapter is None:
            return False, f"{venue} adapter is None"
        if hasattr(adapter, "is_connected"):
            if adapter.is_connected():
                return True, f"{venue} connected"
            return False, f"{venue} not connected"
        # If no connectivity check available, pass by default.
        return True, f"{venue} adapter present (no connectivity check)"
    return _check


def safe_mode_gate(kill_switch_manager: Any) -> GateCheckFn:
    """Gate that checks safe mode is not active."""
    def _check() -> tuple[bool, str]:
        if kill_switch_manager.safe_mode:
            return False, "safe mode active — operator must acknowledge before trading"
        return True, "safe mode not active"
    return _check


def config_sanity_gate(
    live_mode: bool,
    bankroll_by_venue: dict[str, float],
    required_venues: list[str] | None = None,
) -> GateCheckFn:
    """Gate that validates config sanity before live trading."""
    def _check() -> tuple[bool, str]:
        issues: list[str] = []

        if live_mode:
            for venue in (required_venues or []):
                bal = bankroll_by_venue.get(venue, 0.0)
                if bal <= 0:
                    issues.append(f"{venue} bankroll is {bal}")

        if issues:
            return False, "; ".join(issues)
        return True, "config ok"
    return _check


def recovery_gate(recovery_report: Any) -> GateCheckFn:
    """Gate that checks crash recovery didn't find critical issues."""
    def _check() -> tuple[bool, str]:
        if recovery_report is None:
            return True, "no recovery report (fresh start)"
        if recovery_report.safe_mode:
            return False, f"recovery found issues: {recovery_report.safe_mode_reason}"
        return True, f"recovery clean ({recovery_report.open_positions} open positions)"
    return _check
