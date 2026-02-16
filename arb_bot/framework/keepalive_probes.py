"""Cross/parity keepalive probe set (Phase 5E).

Low-rate probes that periodically verify mapped pairs and parity
rules are still reachable and returning valid data. Alerts on
coverage floor breach.

Usage::

    probes = KeepaliveProbeSet(config)
    probes.register_probe("kalshi:BTC|poly:BTC", check_fn=lambda: fetch_both())
    due = probes.due_probes(now)
    for probe in due:
        result = probe.check_fn()
        probes.record_result(probe.probe_id, success=result, now=now)
    report = probes.report()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KeepaliveProbeConfig:
    """Configuration for keepalive probe set.

    Parameters
    ----------
    probe_interval_seconds:
        Default interval between probes. Default 30.
    failure_threshold:
        Number of consecutive failures before marking probe dead.
        Default 3.
    coverage_floor:
        Minimum fraction of probes that must be alive. Default 0.70.
    max_probes_per_cycle:
        Maximum probes to run per check cycle. Default 20.
    backoff_factor:
        Multiplier for probe interval after failure. Default 1.5.
    max_probe_interval:
        Maximum probe interval after backoff. Default 300.
    recovery_probes:
        Number of consecutive successes to recover from dead.
        Default 2.
    """

    probe_interval_seconds: float = 30.0
    failure_threshold: int = 3
    coverage_floor: float = 0.70
    max_probes_per_cycle: int = 20
    backoff_factor: float = 1.5
    max_probe_interval: float = 300.0
    recovery_probes: int = 2


# ---------------------------------------------------------------------------
# Probe state
# ---------------------------------------------------------------------------


@dataclass
class ProbeState:
    """State for a single keepalive probe."""

    probe_id: str
    check_fn: Callable[[], bool]
    interval: float
    alive: bool = True
    last_probe_time: float = 0.0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    total_probes: int = 0
    total_failures: int = 0


# ---------------------------------------------------------------------------
# Due probe
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DueProbe:
    """A probe that is due for execution."""

    probe_id: str
    check_fn: Callable[[], bool]
    overdue_seconds: float


# ---------------------------------------------------------------------------
# Probe report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeReport:
    """Overall probe health report."""

    total_probes: int
    alive_count: int
    dead_count: int
    coverage: float
    coverage_floor: float
    floor_breached: bool
    dead_probes: Tuple[str, ...]


# ---------------------------------------------------------------------------
# Probe set
# ---------------------------------------------------------------------------


class KeepaliveProbeSet:
    """Low-rate keepalive probes for mapped pairs and parity rules.

    Each probe periodically calls its check function. Failed probes
    back off. Probes that exceed the failure threshold are marked
    dead. An alert fires when alive coverage drops below the floor.
    """

    def __init__(self, config: KeepaliveProbeConfig | None = None) -> None:
        self._config = config or KeepaliveProbeConfig()
        self._probes: Dict[str, ProbeState] = {}

    @property
    def config(self) -> KeepaliveProbeConfig:
        return self._config

    def register_probe(
        self,
        probe_id: str,
        check_fn: Callable[[], bool],
        interval: float | None = None,
    ) -> None:
        """Register a keepalive probe.

        Parameters
        ----------
        probe_id:
            Unique probe identifier.
        check_fn:
            Callable returning True if the probe succeeds.
        interval:
            Probe interval override. If None, uses config default.
        """
        if probe_id not in self._probes:
            self._probes[probe_id] = ProbeState(
                probe_id=probe_id,
                check_fn=check_fn,
                interval=interval or self._config.probe_interval_seconds,
            )

    def unregister_probe(self, probe_id: str) -> None:
        """Remove a probe."""
        self._probes.pop(probe_id, None)

    def due_probes(self, now: float | None = None) -> List[DueProbe]:
        """Get probes that are due for execution.

        Returns at most max_probes_per_cycle items, sorted by
        most overdue first.
        """
        if now is None:
            now = time.monotonic()

        due: List[DueProbe] = []
        for state in self._probes.values():
            if state.last_probe_time == 0.0:
                # Never probed.
                overdue = state.interval
            else:
                elapsed = now - state.last_probe_time
                if elapsed >= state.interval:
                    overdue = elapsed - state.interval
                else:
                    continue

            due.append(DueProbe(
                probe_id=state.probe_id,
                check_fn=state.check_fn,
                overdue_seconds=overdue,
            ))

        due.sort(key=lambda d: d.overdue_seconds, reverse=True)
        return due[: self._config.max_probes_per_cycle]

    def record_result(
        self,
        probe_id: str,
        success: bool,
        now: float | None = None,
    ) -> None:
        """Record the result of a probe execution.

        Parameters
        ----------
        probe_id:
            Probe identifier.
        success:
            Whether the probe succeeded.
        now:
            Current timestamp.
        """
        if now is None:
            now = time.monotonic()
        state = self._probes.get(probe_id)
        if state is None:
            return

        state.last_probe_time = now
        state.total_probes += 1
        cfg = self._config

        if success:
            state.consecutive_failures = 0
            state.consecutive_successes += 1
            # Recover from dead after enough successes.
            if not state.alive and state.consecutive_successes >= cfg.recovery_probes:
                state.alive = True
                state.interval = cfg.probe_interval_seconds
        else:
            state.total_failures += 1
            state.consecutive_failures += 1
            state.consecutive_successes = 0
            # Back off.
            state.interval = min(
                state.interval * cfg.backoff_factor,
                cfg.max_probe_interval,
            )
            # Mark dead after threshold.
            if state.consecutive_failures >= cfg.failure_threshold:
                state.alive = False

    def get_state(self, probe_id: str) -> ProbeState | None:
        """Get state for a probe."""
        return self._probes.get(probe_id)

    def report(self) -> ProbeReport:
        """Generate probe health report."""
        total = len(self._probes)
        alive = sum(1 for s in self._probes.values() if s.alive)
        dead = total - alive
        coverage = alive / total if total > 0 else 1.0
        dead_ids = tuple(s.probe_id for s in self._probes.values() if not s.alive)

        return ProbeReport(
            total_probes=total,
            alive_count=alive,
            dead_count=dead,
            coverage=coverage,
            coverage_floor=self._config.coverage_floor,
            floor_breached=coverage < self._config.coverage_floor,
            dead_probes=dead_ids,
        )

    def probe_count(self) -> int:
        """Total registered probes."""
        return len(self._probes)

    def clear(self) -> None:
        """Clear all probes."""
        self._probes.clear()
