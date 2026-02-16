"""Initialization speed and startup progress (Phase 5H).

Tracks startup phases with timing, progress, and fail-fast behavior.
Provides structured progress reporting for startup diagnostics.

Usage::

    tracker = StartupTracker(config)
    tracker.begin_phase("discovery")
    # ... do work ...
    tracker.complete_phase("discovery")
    tracker.begin_phase("bootstrap")
    tracker.update_progress("bootstrap", current=50, total=200)
    tracker.complete_phase("bootstrap")
    report = tracker.report()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Phase status
# ---------------------------------------------------------------------------


class PhaseStatus(str, Enum):
    """Status of a startup phase."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    SKIPPED = "skipped"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StartupTrackerConfig:
    """Configuration for startup tracker.

    Parameters
    ----------
    default_timeout:
        Default timeout per phase (seconds). Default 60.
    fail_fast:
        If True, abort remaining phases on first failure.
        Default True.
    max_total_startup_time:
        Maximum total startup time (seconds). Default 300.
    """

    default_timeout: float = 60.0
    fail_fast: bool = True
    max_total_startup_time: float = 300.0


# ---------------------------------------------------------------------------
# Phase state
# ---------------------------------------------------------------------------


@dataclass
class PhaseState:
    """State for a single startup phase."""

    name: str
    status: PhaseStatus = PhaseStatus.PENDING
    start_time: float = 0.0
    end_time: float = 0.0
    timeout: float = 60.0
    current: int = 0
    total: int = 0
    message: str = ""
    error: str = ""

    @property
    def elapsed(self) -> float:
        if self.start_time == 0.0:
            return 0.0
        end = self.end_time if self.end_time > 0 else time.monotonic()
        return end - self.start_time

    @property
    def progress_fraction(self) -> float:
        if self.total <= 0:
            return 0.0
        return min(1.0, self.current / self.total)


# ---------------------------------------------------------------------------
# Startup report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StartupReport:
    """Overall startup progress report."""

    phases: Dict[str, PhaseState]
    total_elapsed: float
    completed_count: int
    failed_count: int
    pending_count: int
    overall_status: PhaseStatus
    failed_phases: Tuple[str, ...]
    is_ready: bool


# ---------------------------------------------------------------------------
# Startup tracker
# ---------------------------------------------------------------------------


class StartupTracker:
    """Tracks startup phases with timing and progress.

    Phases are defined upfront and executed in order. Each phase
    has a timeout, progress tracking, and fail-fast behavior.
    """

    def __init__(self, config: StartupTrackerConfig | None = None) -> None:
        self._config = config or StartupTrackerConfig()
        self._phases: Dict[str, PhaseState] = {}
        self._phase_order: List[str] = []
        self._start_time: float = 0.0
        self._aborted: bool = False

    @property
    def config(self) -> StartupTrackerConfig:
        return self._config

    def define_phase(
        self,
        name: str,
        timeout: float | None = None,
        total: int = 0,
    ) -> None:
        """Define a startup phase.

        Parameters
        ----------
        name:
            Phase name (e.g., "discovery", "bootstrap", "warmup").
        timeout:
            Phase timeout. If None, uses default_timeout.
        total:
            Total work items for progress tracking. 0 = unknown.
        """
        if name not in self._phases:
            self._phases[name] = PhaseState(
                name=name,
                timeout=timeout or self._config.default_timeout,
                total=total,
            )
            self._phase_order.append(name)

    def begin_phase(self, name: str, now: float | None = None) -> bool:
        """Begin a startup phase.

        Returns False if startup has been aborted.
        """
        if now is None:
            now = time.monotonic()
        if self._start_time == 0.0:
            self._start_time = now

        if self._aborted:
            return False

        # Check total startup timeout.
        if now - self._start_time > self._config.max_total_startup_time:
            self._aborted = True
            return False

        state = self._phases.get(name)
        if state is None:
            self.define_phase(name)
            state = self._phases[name]

        state.status = PhaseStatus.RUNNING
        state.start_time = now
        return True

    def update_progress(
        self,
        name: str,
        current: int = 0,
        total: int | None = None,
        message: str = "",
    ) -> None:
        """Update progress for a running phase."""
        state = self._phases.get(name)
        if state is None:
            return
        state.current = current
        if total is not None:
            state.total = total
        if message:
            state.message = message

    def complete_phase(self, name: str, now: float | None = None) -> None:
        """Mark a phase as completed."""
        if now is None:
            now = time.monotonic()
        state = self._phases.get(name)
        if state is None:
            return
        state.status = PhaseStatus.COMPLETED
        state.end_time = now
        if state.total > 0:
            state.current = state.total

    def fail_phase(
        self,
        name: str,
        error: str = "",
        now: float | None = None,
    ) -> None:
        """Mark a phase as failed."""
        if now is None:
            now = time.monotonic()
        state = self._phases.get(name)
        if state is None:
            return
        state.status = PhaseStatus.FAILED
        state.end_time = now
        state.error = error

        if self._config.fail_fast:
            self._aborted = True

    def timeout_phase(
        self,
        name: str,
        now: float | None = None,
    ) -> None:
        """Mark a phase as timed out."""
        if now is None:
            now = time.monotonic()
        state = self._phases.get(name)
        if state is None:
            return
        state.status = PhaseStatus.TIMED_OUT
        state.end_time = now

        if self._config.fail_fast:
            self._aborted = True

    def skip_phase(self, name: str) -> None:
        """Mark a phase as skipped."""
        state = self._phases.get(name)
        if state is None:
            return
        state.status = PhaseStatus.SKIPPED

    def check_timeout(self, name: str, now: float | None = None) -> bool:
        """Check if a running phase has timed out.

        Returns True if timed out (and marks it).
        """
        if now is None:
            now = time.monotonic()
        state = self._phases.get(name)
        if state is None:
            return False
        if state.status != PhaseStatus.RUNNING:
            return False
        if state.start_time > 0 and (now - state.start_time) > state.timeout:
            self.timeout_phase(name, now)
            return True
        return False

    def is_aborted(self) -> bool:
        """Check if startup has been aborted."""
        return self._aborted

    def get_phase(self, name: str) -> PhaseState | None:
        """Get state for a specific phase."""
        return self._phases.get(name)

    def report(self, now: float | None = None) -> StartupReport:
        """Generate startup progress report."""
        if now is None:
            now = time.monotonic()

        elapsed = now - self._start_time if self._start_time > 0 else 0.0

        completed = sum(
            1 for p in self._phases.values()
            if p.status == PhaseStatus.COMPLETED
        )
        failed = sum(
            1 for p in self._phases.values()
            if p.status in (PhaseStatus.FAILED, PhaseStatus.TIMED_OUT)
        )
        pending = sum(
            1 for p in self._phases.values()
            if p.status in (PhaseStatus.PENDING, PhaseStatus.RUNNING)
        )

        failed_names = tuple(
            p.name for p in self._phases.values()
            if p.status in (PhaseStatus.FAILED, PhaseStatus.TIMED_OUT)
        )

        if failed > 0:
            overall = PhaseStatus.FAILED
        elif pending > 0:
            overall = PhaseStatus.RUNNING
        else:
            overall = PhaseStatus.COMPLETED

        is_ready = (
            overall == PhaseStatus.COMPLETED
            and not self._aborted
        )

        return StartupReport(
            phases=dict(self._phases),
            total_elapsed=elapsed,
            completed_count=completed,
            failed_count=failed,
            pending_count=pending,
            overall_status=overall,
            failed_phases=failed_names,
            is_ready=is_ready,
        )

    def clear(self) -> None:
        """Reset all state."""
        self._phases.clear()
        self._phase_order.clear()
        self._start_time = 0.0
        self._aborted = False
