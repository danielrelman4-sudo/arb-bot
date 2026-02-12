"""Metrics export + alert dashboard (Phase 2H).

Provides a lightweight in-process metrics registry for counters,
gauges, and histograms. Supports alert threshold evaluation and
structured metric snapshots for export (Prometheus text format,
JSON, or custom integrations).

No external dependencies — designed to be consumed by any exporter
(Prometheus push gateway, StatsD, log file, HTTP endpoint, etc.).
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric types
# ---------------------------------------------------------------------------


class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"


@dataclass
class MetricValue:
    """A single metric with its current value and metadata."""

    name: str
    metric_type: MetricType
    value: float = 0.0
    labels: Dict[str, str] = field(default_factory=dict)
    updated_at: float = 0.0

    def increment(self, amount: float = 1.0, now: float | None = None) -> None:
        self.value += amount
        self.updated_at = now or time.monotonic()

    def set(self, value: float, now: float | None = None) -> None:
        self.value = value
        self.updated_at = now or time.monotonic()

    def reset(self) -> None:
        self.value = 0.0
        self.updated_at = 0.0


# ---------------------------------------------------------------------------
# Alert config
# ---------------------------------------------------------------------------


class AlertSeverity(Enum):
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class AlertThreshold:
    """Defines an alert condition on a metric."""

    metric_name: str
    condition: str         # "above" or "below"
    threshold: float
    severity: AlertSeverity = AlertSeverity.WARNING
    message: str = ""


@dataclass(frozen=True)
class Alert:
    """A triggered alert."""

    metric_name: str
    current_value: float
    threshold: float
    condition: str
    severity: AlertSeverity
    message: str
    triggered_at: float


# ---------------------------------------------------------------------------
# Metric snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricSnapshot:
    """A point-in-time snapshot of all metrics."""

    timestamp: float
    metrics: Dict[str, float]
    alerts: tuple[Alert, ...]

    @property
    def has_alerts(self) -> bool:
        return len(self.alerts) > 0

    @property
    def has_critical_alerts(self) -> bool:
        return any(a.severity == AlertSeverity.CRITICAL for a in self.alerts)

    def to_prometheus_text(self) -> str:
        """Export metrics in Prometheus text exposition format."""
        lines: list[str] = []
        for name, value in sorted(self.metrics.items()):
            # Prometheus naming: replace dots with underscores.
            prom_name = name.replace(".", "_").replace("-", "_")
            lines.append(f"{prom_name} {value}")
        return "\n".join(lines) + "\n" if lines else ""

    def to_dict(self) -> Dict[str, Any]:
        """Export as a plain dict for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "metrics": dict(self.metrics),
            "alerts": [
                {
                    "metric": a.metric_name,
                    "value": a.current_value,
                    "threshold": a.threshold,
                    "severity": a.severity.value,
                    "message": a.message,
                }
                for a in self.alerts
            ],
        }


# ---------------------------------------------------------------------------
# Metrics registry
# ---------------------------------------------------------------------------


class MetricsRegistry:
    """Central metrics registry with alert evaluation.

    Usage::

        registry = MetricsRegistry()
        registry.counter("orders_placed").increment()
        registry.gauge("quote_cache_size").set(150)
        registry.add_alert(AlertThreshold(
            metric_name="error_rate",
            condition="above",
            threshold=0.1,
            severity=AlertSeverity.CRITICAL,
        ))
        snapshot = registry.snapshot()
    """

    def __init__(self) -> None:
        self._metrics: Dict[str, MetricValue] = {}
        self._alerts: list[AlertThreshold] = []

    def counter(self, name: str, **labels: str) -> MetricValue:
        """Get or create a counter metric."""
        if name not in self._metrics:
            self._metrics[name] = MetricValue(
                name=name,
                metric_type=MetricType.COUNTER,
                labels=labels,
            )
        return self._metrics[name]

    def gauge(self, name: str, **labels: str) -> MetricValue:
        """Get or create a gauge metric."""
        if name not in self._metrics:
            self._metrics[name] = MetricValue(
                name=name,
                metric_type=MetricType.GAUGE,
                labels=labels,
            )
        return self._metrics[name]

    def get(self, name: str) -> MetricValue | None:
        """Get a metric by name."""
        return self._metrics.get(name)

    def add_alert(self, threshold: AlertThreshold) -> None:
        """Register an alert threshold."""
        self._alerts.append(threshold)

    def evaluate_alerts(self, now: float | None = None) -> list[Alert]:
        """Evaluate all alert thresholds against current values."""
        if now is None:
            now = time.monotonic()

        triggered: list[Alert] = []
        for at in self._alerts:
            metric = self._metrics.get(at.metric_name)
            if metric is None:
                continue

            fired = False
            if at.condition == "above" and metric.value > at.threshold:
                fired = True
            elif at.condition == "below" and metric.value < at.threshold:
                fired = True

            if fired:
                alert = Alert(
                    metric_name=at.metric_name,
                    current_value=metric.value,
                    threshold=at.threshold,
                    condition=at.condition,
                    severity=at.severity,
                    message=at.message or (
                        f"{at.metric_name} is {at.condition} threshold "
                        f"({metric.value:.4f} vs {at.threshold:.4f})"
                    ),
                    triggered_at=now,
                )
                triggered.append(alert)
                LOGGER.warning(
                    "Alert triggered: %s (%s)",
                    alert.message,
                    alert.severity.value,
                )

        return triggered

    def snapshot(self, now: float | None = None) -> MetricSnapshot:
        """Take a point-in-time snapshot of all metrics and alerts."""
        if now is None:
            now = time.monotonic()

        metrics = {name: mv.value for name, mv in self._metrics.items()}
        alerts = tuple(self.evaluate_alerts(now=now))

        return MetricSnapshot(
            timestamp=now,
            metrics=metrics,
            alerts=alerts,
        )

    def metric_names(self) -> list[str]:
        return sorted(self._metrics.keys())

    def reset_all(self) -> None:
        """Reset all metric values to zero."""
        for mv in self._metrics.values():
            mv.reset()

    def clear(self) -> None:
        """Remove all metrics and alerts."""
        self._metrics.clear()
        self._alerts.clear()
