"""Lane readiness watchdog enforcement (Phase 4J).

Gates trade execution on per-lane readiness checks. Each lane must
pass all readiness conditions (data freshness, calibration health,
circuit breaker status) before it's allowed to size and execute.

Usage::

    wd = LaneWatchdog(config)
    wd.register_check("cross_venue", "data_fresh", lambda: time.time() - last_update < 30)
    wd.register_check("cross_venue", "calibrated", lambda: cal_samples > 20)
    result = wd.check_lane("cross_venue")
    if result.ready:
        # proceed with sizing
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LaneWatchdogConfig:
    """Configuration for lane readiness watchdog.

    Parameters
    ----------
    grace_period_seconds:
        After a lane fails, it must pass continuously for this
        duration before being marked ready again. Default 60.
    max_consecutive_failures:
        After this many consecutive check failures, the lane is
        locked out until manual reset. Default 10.
    check_interval_seconds:
        Minimum time between checks for the same lane. Default 5.
    """

    grace_period_seconds: float = 60.0
    max_consecutive_failures: int = 10
    check_interval_seconds: float = 5.0


# ---------------------------------------------------------------------------
# Check result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LaneCheckResult:
    """Result of a lane readiness check."""

    lane: str
    ready: bool
    failed_checks: Tuple[str, ...]
    passed_checks: Tuple[str, ...]
    in_grace_period: bool
    locked_out: bool
    consecutive_failures: int


# ---------------------------------------------------------------------------
# Lane state
# ---------------------------------------------------------------------------


@dataclass
class _LaneState:
    """Internal state for a lane."""

    checks: Dict[str, Callable[[], bool]]
    consecutive_failures: int = 0
    last_fail_time: float = 0.0
    last_check_time: float = 0.0
    locked_out: bool = False


# ---------------------------------------------------------------------------
# Watchdog
# ---------------------------------------------------------------------------


class LaneWatchdog:
    """Enforces lane readiness before allowing sizing/execution.

    Lanes must register readiness checks. All checks must pass
    for a lane to be considered ready. Failed lanes enter a
    grace period and may be locked out after consecutive failures.
    """

    def __init__(self, config: LaneWatchdogConfig | None = None) -> None:
        self._config = config or LaneWatchdogConfig()
        self._lanes: Dict[str, _LaneState] = {}

    @property
    def config(self) -> LaneWatchdogConfig:
        return self._config

    def register_check(
        self,
        lane: str,
        check_name: str,
        check_fn: Callable[[], bool],
    ) -> None:
        """Register a readiness check for a lane.

        Parameters
        ----------
        lane:
            Lane identifier.
        check_name:
            Name of the check (e.g., "data_fresh", "calibrated").
        check_fn:
            Callable returning True if the check passes.
        """
        if lane not in self._lanes:
            self._lanes[lane] = _LaneState(checks={})
        self._lanes[lane].checks[check_name] = check_fn

    def check_lane(self, lane: str, now: float | None = None) -> LaneCheckResult:
        """Run all readiness checks for a lane.

        Parameters
        ----------
        lane:
            Lane identifier.
        now:
            Current timestamp. If None, uses time.time().
        """
        if now is None:
            now = time.time()

        state = self._lanes.get(lane)
        if state is None:
            # No checks registered → not ready.
            return LaneCheckResult(
                lane=lane, ready=False,
                failed_checks=("no_checks_registered",),
                passed_checks=(),
                in_grace_period=False,
                locked_out=False,
                consecutive_failures=0,
            )

        if state.locked_out:
            return LaneCheckResult(
                lane=lane, ready=False,
                failed_checks=("locked_out",),
                passed_checks=(),
                in_grace_period=False,
                locked_out=True,
                consecutive_failures=state.consecutive_failures,
            )

        state.last_check_time = now

        # Run all checks.
        passed: List[str] = []
        failed: List[str] = []
        for name, fn in state.checks.items():
            try:
                if fn():
                    passed.append(name)
                else:
                    failed.append(name)
            except Exception:
                failed.append(name)

        all_passed = len(failed) == 0

        if not all_passed:
            state.consecutive_failures += 1
            state.last_fail_time = now

            if state.consecutive_failures >= self._config.max_consecutive_failures:
                state.locked_out = True

            return LaneCheckResult(
                lane=lane, ready=False,
                failed_checks=tuple(failed),
                passed_checks=tuple(passed),
                in_grace_period=False,
                locked_out=state.locked_out,
                consecutive_failures=state.consecutive_failures,
            )

        # All passed — check grace period.
        in_grace = False
        if state.last_fail_time > 0:
            elapsed = now - state.last_fail_time
            if elapsed < self._config.grace_period_seconds:
                in_grace = True

        if not in_grace:
            state.consecutive_failures = 0

        return LaneCheckResult(
            lane=lane,
            ready=not in_grace,
            failed_checks=(),
            passed_checks=tuple(passed),
            in_grace_period=in_grace,
            locked_out=False,
            consecutive_failures=state.consecutive_failures,
        )

    def reset_lane(self, lane: str) -> None:
        """Reset a lane's failure state (clear lockout)."""
        state = self._lanes.get(lane)
        if state:
            state.consecutive_failures = 0
            state.last_fail_time = 0.0
            state.locked_out = False

    def registered_lanes(self) -> List[str]:
        """List all registered lanes."""
        return list(self._lanes.keys())

    def is_locked_out(self, lane: str) -> bool:
        """Check if a lane is locked out."""
        state = self._lanes.get(lane)
        return state.locked_out if state else False

    def clear(self) -> None:
        """Clear all lanes and checks."""
        self._lanes.clear()
