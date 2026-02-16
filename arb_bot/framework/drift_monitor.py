"""Expected-vs-realized drift monitoring (Phase 4I).

Tracks the gap between expected and realized outcomes across
sizing inputs (edge, fill rate, cost, PnL). Raises alerts
when drift exceeds thresholds and can trigger sizing reductions.

Usage::

    mon = DriftMonitor(config)
    mon.record("edge", expected=0.03, realized=0.025)
    mon.record("fill_rate", expected=0.80, realized=0.65)
    report = mon.report()
    if report.any_alert:
        multiplier = report.sizing_multiplier
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DriftMonitorConfig:
    """Configuration for drift monitoring.

    Parameters
    ----------
    alert_threshold:
        Absolute drift threshold that triggers an alert.
        Default 0.05 (5%).
    relative_alert_threshold:
        Relative drift threshold (drift / expected).
        Default 0.30 (30%).
    window_size:
        Number of recent observations per metric. Default 50.
    sizing_reduction_per_alert:
        Fraction to reduce sizing for each metric in alert.
        Default 0.10 (10% per alerting metric).
    min_sizing_multiplier:
        Floor for sizing multiplier. Default 0.25.
    min_observations:
        Minimum observations before alerts fire. Default 5.
    """

    alert_threshold: float = 0.05
    relative_alert_threshold: float = 0.30
    window_size: int = 50
    sizing_reduction_per_alert: float = 0.10
    min_sizing_multiplier: float = 0.25
    min_observations: int = 5


# ---------------------------------------------------------------------------
# Per-metric drift stats
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricDrift:
    """Drift statistics for a single metric."""

    name: str
    observation_count: int
    mean_expected: float
    mean_realized: float
    mean_drift: float  # realized - expected
    abs_drift: float
    relative_drift: float  # drift / expected (if expected != 0)
    alert: bool
    alert_reason: str


# ---------------------------------------------------------------------------
# Overall report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DriftReport:
    """Overall drift monitoring report."""

    metrics: Dict[str, MetricDrift]
    alert_count: int
    any_alert: bool
    sizing_multiplier: float
    alerting_metrics: Tuple[str, ...]


# ---------------------------------------------------------------------------
# Drift monitor
# ---------------------------------------------------------------------------


class DriftMonitor:
    """Monitors expected-vs-realized drift across sizing inputs.

    Records paired (expected, realized) observations per metric
    and raises alerts when systematic drift is detected.
    """

    def __init__(self, config: DriftMonitorConfig | None = None) -> None:
        self._config = config or DriftMonitorConfig()
        self._observations: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

    @property
    def config(self) -> DriftMonitorConfig:
        return self._config

    def record(self, metric: str, expected: float, realized: float) -> None:
        """Record an expected-vs-realized observation.

        Parameters
        ----------
        metric:
            Metric name (e.g., "edge", "fill_rate", "cost", "pnl").
        expected:
            Expected value at time of sizing.
        realized:
            Realized value after execution.
        """
        self._observations[metric].append((expected, realized))
        window = self._config.window_size
        if len(self._observations[metric]) > window:
            self._observations[metric] = self._observations[metric][-window:]

    def observation_count(self, metric: str) -> int:
        """Number of observations for a metric."""
        return len(self._observations.get(metric, []))

    def metric_drift(self, metric: str) -> MetricDrift:
        """Compute drift for a single metric."""
        cfg = self._config
        obs = self._observations.get(metric, [])
        n = len(obs)

        if n == 0:
            return MetricDrift(
                name=metric, observation_count=0,
                mean_expected=0.0, mean_realized=0.0,
                mean_drift=0.0, abs_drift=0.0,
                relative_drift=0.0, alert=False, alert_reason="",
            )

        mean_exp = sum(e for e, _ in obs) / n
        mean_real = sum(r for _, r in obs) / n
        mean_drift = mean_real - mean_exp
        abs_drift = abs(mean_drift)

        relative_drift = 0.0
        if abs(mean_exp) > 1e-10:
            relative_drift = abs(mean_drift / mean_exp)

        alert = False
        alert_reason = ""
        if n >= cfg.min_observations:
            if abs_drift >= cfg.alert_threshold:
                alert = True
                alert_reason = "absolute_drift"
            elif relative_drift >= cfg.relative_alert_threshold:
                alert = True
                alert_reason = "relative_drift"

        return MetricDrift(
            name=metric,
            observation_count=n,
            mean_expected=mean_exp,
            mean_realized=mean_real,
            mean_drift=mean_drift,
            abs_drift=abs_drift,
            relative_drift=relative_drift,
            alert=alert,
            alert_reason=alert_reason,
        )

    def report(self) -> DriftReport:
        """Generate overall drift report."""
        cfg = self._config
        metrics: Dict[str, MetricDrift] = {}
        alerting: List[str] = []

        for metric in self._observations:
            drift = self.metric_drift(metric)
            metrics[metric] = drift
            if drift.alert:
                alerting.append(metric)

        alert_count = len(alerting)
        reduction = alert_count * cfg.sizing_reduction_per_alert
        multiplier = max(cfg.min_sizing_multiplier, 1.0 - reduction)

        return DriftReport(
            metrics=metrics,
            alert_count=alert_count,
            any_alert=alert_count > 0,
            sizing_multiplier=multiplier,
            alerting_metrics=tuple(alerting),
        )

    def clear(self) -> None:
        """Clear all observations."""
        self._observations.clear()
