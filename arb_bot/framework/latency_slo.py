"""Latency SLO + auto-throttle (Phase 3E).

First-class SLOs (Service Level Objectives) for quote age, decision
latency, and execution latency. Tracks percentile performance and
automatically throttles when persistent breaches occur.

Usage::

    tracker = LatencyTracker(config)
    tracker.record("quote_age", 0.15)
    tracker.record("decision_latency", 0.05)
    report = tracker.evaluate()
    if report.should_throttle:
        # reduce load
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LatencySLO:
    """A single latency SLO definition.

    Parameters
    ----------
    name:
        SLO identifier (e.g. "quote_age", "decision_latency").
    target_p50:
        Target 50th percentile in seconds. 0 = not tracked.
    target_p95:
        Target 95th percentile in seconds. 0 = not tracked.
    target_p99:
        Target 99th percentile in seconds. 0 = not tracked.
    breach_window:
        Number of consecutive evaluation periods with breaches
        before triggering throttle. Default 3.
    """

    name: str
    target_p50: float = 0.0
    target_p95: float = 0.0
    target_p99: float = 0.0
    breach_window: int = 3


@dataclass(frozen=True)
class LatencyTrackerConfig:
    """Configuration for the latency tracker.

    Parameters
    ----------
    slos:
        SLO definitions. Default empty.
    window_size:
        Number of samples to keep per SLO. Default 1000.
    throttle_cooldown_evaluations:
        Number of passing evaluations required to exit throttle.
        Default 5.
    """

    slos: tuple[LatencySLO, ...] = ()
    window_size: int = 1000
    throttle_cooldown_evaluations: int = 5


# ---------------------------------------------------------------------------
# SLO evaluation results
# ---------------------------------------------------------------------------


class SLOStatus(Enum):
    PASSING = "passing"
    BREACHED = "breached"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass(frozen=True)
class SLOEvaluation:
    """Evaluation result for a single SLO."""

    name: str
    status: SLOStatus
    sample_count: int
    p50: float
    p95: float
    p99: float
    breaches: Dict[str, bool]
    consecutive_breach_count: int


@dataclass(frozen=True)
class LatencyReport:
    """Overall latency SLO evaluation report."""

    evaluations: tuple[SLOEvaluation, ...]
    should_throttle: bool
    throttle_reasons: tuple[str, ...]
    is_throttled: bool

    @property
    def all_passing(self) -> bool:
        return all(
            e.status == SLOStatus.PASSING
            for e in self.evaluations
        )

    @property
    def any_breached(self) -> bool:
        return any(
            e.status == SLOStatus.BREACHED
            for e in self.evaluations
        )


# ---------------------------------------------------------------------------
# Latency tracker
# ---------------------------------------------------------------------------


class LatencyTracker:
    """Tracks latency samples and evaluates against SLOs.

    Records latency observations per metric name, computes
    percentiles, and determines if auto-throttle should engage.
    """

    def __init__(self, config: LatencyTrackerConfig | None = None) -> None:
        self._config = config or LatencyTrackerConfig()
        self._slo_map: Dict[str, LatencySLO] = {
            s.name: s for s in self._config.slos
        }
        self._samples: Dict[str, list[float]] = {}
        self._breach_counts: Dict[str, int] = {}
        self._is_throttled: bool = False
        self._passing_count: int = 0

    @property
    def config(self) -> LatencyTrackerConfig:
        return self._config

    @property
    def is_throttled(self) -> bool:
        return self._is_throttled

    def record(self, name: str, latency_seconds: float) -> None:
        """Record a latency observation."""
        if name not in self._samples:
            self._samples[name] = []

        samples = self._samples[name]
        samples.append(latency_seconds)

        # Trim to window.
        if len(samples) > self._config.window_size:
            self._samples[name] = samples[-self._config.window_size:]

    def sample_count(self, name: str) -> int:
        """Number of samples recorded for a metric."""
        return len(self._samples.get(name, []))

    def percentile(self, name: str, pct: float) -> float | None:
        """Compute a percentile for a metric.

        Returns None if no samples exist.
        """
        samples = self._samples.get(name)
        if not samples:
            return None
        return _percentile(samples, pct)

    def evaluate(self) -> LatencyReport:
        """Evaluate all SLOs and determine throttle state."""
        evaluations: list[SLOEvaluation] = []
        throttle_reasons: list[str] = []
        any_breach = False

        for slo in self._config.slos:
            evaluation = self._evaluate_slo(slo)
            evaluations.append(evaluation)

            if evaluation.status == SLOStatus.BREACHED:
                any_breach = True
                if evaluation.consecutive_breach_count >= slo.breach_window:
                    throttle_reasons.append(
                        f"{slo.name}: {evaluation.consecutive_breach_count} "
                        f"consecutive breaches (window={slo.breach_window})"
                    )

        # Throttle logic.
        should_throttle = len(throttle_reasons) > 0

        if should_throttle:
            self._is_throttled = True
            self._passing_count = 0
        elif self._is_throttled:
            if not any_breach:
                self._passing_count += 1
                if self._passing_count >= self._config.throttle_cooldown_evaluations:
                    self._is_throttled = False
                    self._passing_count = 0

        return LatencyReport(
            evaluations=tuple(evaluations),
            should_throttle=should_throttle,
            throttle_reasons=tuple(throttle_reasons),
            is_throttled=self._is_throttled,
        )

    def reset(self) -> None:
        """Clear all samples and state."""
        self._samples.clear()
        self._breach_counts.clear()
        self._is_throttled = False
        self._passing_count = 0

    def _evaluate_slo(self, slo: LatencySLO) -> SLOEvaluation:
        samples = self._samples.get(slo.name, [])
        count = len(samples)

        if count == 0:
            return SLOEvaluation(
                name=slo.name,
                status=SLOStatus.INSUFFICIENT_DATA,
                sample_count=0,
                p50=0.0, p95=0.0, p99=0.0,
                breaches={},
                consecutive_breach_count=self._breach_counts.get(slo.name, 0),
            )

        p50 = _percentile(samples, 50.0)
        p95 = _percentile(samples, 95.0)
        p99 = _percentile(samples, 99.0)

        breaches: Dict[str, bool] = {}
        is_breached = False

        if slo.target_p50 > 0:
            breached = p50 > slo.target_p50
            breaches["p50"] = breached
            if breached:
                is_breached = True

        if slo.target_p95 > 0:
            breached = p95 > slo.target_p95
            breaches["p95"] = breached
            if breached:
                is_breached = True

        if slo.target_p99 > 0:
            breached = p99 > slo.target_p99
            breaches["p99"] = breached
            if breached:
                is_breached = True

        # Track consecutive breaches.
        if is_breached:
            self._breach_counts[slo.name] = self._breach_counts.get(slo.name, 0) + 1
        else:
            self._breach_counts[slo.name] = 0

        status = SLOStatus.BREACHED if is_breached else SLOStatus.PASSING

        return SLOEvaluation(
            name=slo.name,
            status=status,
            sample_count=count,
            p50=p50,
            p95=p95,
            p99=p99,
            breaches=breaches,
            consecutive_breach_count=self._breach_counts.get(slo.name, 0),
        )


# ---------------------------------------------------------------------------
# Percentile helper
# ---------------------------------------------------------------------------


def _percentile(samples: list[float], pct: float) -> float:
    """Compute percentile using linear interpolation."""
    sorted_samples = sorted(samples)
    n = len(sorted_samples)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_samples[0]

    # Use 0-based index.
    rank = (pct / 100.0) * (n - 1)
    lower = int(math.floor(rank))
    upper = min(lower + 1, n - 1)
    fraction = rank - lower
    return sorted_samples[lower] + fraction * (sorted_samples[upper] - sorted_samples[lower])
