"""Tests for Phase 2H: Metrics export + alert dashboard."""

from __future__ import annotations

import json

import pytest

from arb_bot.framework.metrics import (
    Alert,
    AlertSeverity,
    AlertThreshold,
    MetricSnapshot,
    MetricType,
    MetricValue,
    MetricsRegistry,
)


# ---------------------------------------------------------------------------
# MetricValue
# ---------------------------------------------------------------------------


class TestMetricValue:
    def test_counter_increment(self) -> None:
        mv = MetricValue(name="test", metric_type=MetricType.COUNTER)
        mv.increment(now=100.0)
        assert mv.value == 1.0
        mv.increment(5.0, now=101.0)
        assert mv.value == 6.0
        assert mv.updated_at == 101.0

    def test_gauge_set(self) -> None:
        mv = MetricValue(name="test", metric_type=MetricType.GAUGE)
        mv.set(42.0, now=100.0)
        assert mv.value == 42.0
        mv.set(10.0, now=101.0)
        assert mv.value == 10.0

    def test_reset(self) -> None:
        mv = MetricValue(name="test", metric_type=MetricType.COUNTER, value=10.0)
        mv.reset()
        assert mv.value == 0.0
        assert mv.updated_at == 0.0

    def test_labels(self) -> None:
        mv = MetricValue(
            name="test",
            metric_type=MetricType.COUNTER,
            labels={"venue": "kalshi"},
        )
        assert mv.labels["venue"] == "kalshi"


# ---------------------------------------------------------------------------
# MetricsRegistry — counters and gauges
# ---------------------------------------------------------------------------


class TestRegistryCountersGauges:
    def test_counter(self) -> None:
        reg = MetricsRegistry()
        c = reg.counter("orders_placed")
        c.increment(now=100.0)
        c.increment(now=101.0)
        assert c.value == 2.0

    def test_gauge(self) -> None:
        reg = MetricsRegistry()
        g = reg.gauge("cache_size")
        g.set(150.0, now=100.0)
        assert g.value == 150.0

    def test_get_existing(self) -> None:
        reg = MetricsRegistry()
        reg.counter("test")
        assert reg.get("test") is not None
        assert reg.get("test").value == 0.0

    def test_get_nonexistent(self) -> None:
        reg = MetricsRegistry()
        assert reg.get("nope") is None

    def test_same_name_returns_same_metric(self) -> None:
        reg = MetricsRegistry()
        c1 = reg.counter("test")
        c2 = reg.counter("test")
        assert c1 is c2

    def test_metric_names(self) -> None:
        reg = MetricsRegistry()
        reg.counter("b_counter")
        reg.gauge("a_gauge")
        assert reg.metric_names() == ["a_gauge", "b_counter"]

    def test_reset_all(self) -> None:
        reg = MetricsRegistry()
        reg.counter("c").increment(5.0, now=100.0)
        reg.gauge("g").set(10.0, now=100.0)
        reg.reset_all()
        assert reg.get("c").value == 0.0
        assert reg.get("g").value == 0.0

    def test_clear(self) -> None:
        reg = MetricsRegistry()
        reg.counter("c")
        reg.add_alert(AlertThreshold(
            metric_name="c", condition="above", threshold=1.0
        ))
        reg.clear()
        assert reg.metric_names() == []


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------


class TestAlerts:
    def test_above_threshold_fires(self) -> None:
        reg = MetricsRegistry()
        reg.counter("errors").increment(10.0, now=100.0)
        reg.add_alert(AlertThreshold(
            metric_name="errors",
            condition="above",
            threshold=5.0,
            severity=AlertSeverity.CRITICAL,
        ))
        alerts = reg.evaluate_alerts(now=100.0)
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.CRITICAL
        assert alerts[0].current_value == 10.0

    def test_below_threshold_fires(self) -> None:
        reg = MetricsRegistry()
        reg.gauge("coverage").set(0.3, now=100.0)
        reg.add_alert(AlertThreshold(
            metric_name="coverage",
            condition="below",
            threshold=0.5,
            severity=AlertSeverity.WARNING,
        ))
        alerts = reg.evaluate_alerts(now=100.0)
        assert len(alerts) == 1
        assert alerts[0].condition == "below"

    def test_no_alert_when_within(self) -> None:
        reg = MetricsRegistry()
        reg.counter("errors").increment(3.0, now=100.0)
        reg.add_alert(AlertThreshold(
            metric_name="errors",
            condition="above",
            threshold=5.0,
        ))
        alerts = reg.evaluate_alerts(now=100.0)
        assert len(alerts) == 0

    def test_missing_metric_no_alert(self) -> None:
        reg = MetricsRegistry()
        reg.add_alert(AlertThreshold(
            metric_name="nonexistent",
            condition="above",
            threshold=1.0,
        ))
        alerts = reg.evaluate_alerts(now=100.0)
        assert len(alerts) == 0

    def test_custom_message(self) -> None:
        reg = MetricsRegistry()
        reg.counter("errors").increment(10.0, now=100.0)
        reg.add_alert(AlertThreshold(
            metric_name="errors",
            condition="above",
            threshold=5.0,
            message="Too many errors!",
        ))
        alerts = reg.evaluate_alerts(now=100.0)
        assert alerts[0].message == "Too many errors!"

    def test_default_message(self) -> None:
        reg = MetricsRegistry()
        reg.counter("errors").increment(10.0, now=100.0)
        reg.add_alert(AlertThreshold(
            metric_name="errors",
            condition="above",
            threshold=5.0,
        ))
        alerts = reg.evaluate_alerts(now=100.0)
        assert "errors" in alerts[0].message
        assert "above" in alerts[0].message


# ---------------------------------------------------------------------------
# MetricSnapshot
# ---------------------------------------------------------------------------


class TestMetricSnapshot:
    def test_no_alerts(self) -> None:
        snap = MetricSnapshot(
            timestamp=100.0,
            metrics={"a": 1.0, "b": 2.0},
            alerts=(),
        )
        assert snap.has_alerts is False
        assert snap.has_critical_alerts is False

    def test_with_alerts(self) -> None:
        alert = Alert(
            metric_name="errors",
            current_value=10.0,
            threshold=5.0,
            condition="above",
            severity=AlertSeverity.WARNING,
            message="test",
            triggered_at=100.0,
        )
        snap = MetricSnapshot(
            timestamp=100.0,
            metrics={"errors": 10.0},
            alerts=(alert,),
        )
        assert snap.has_alerts is True
        assert snap.has_critical_alerts is False

    def test_critical_alert(self) -> None:
        alert = Alert(
            metric_name="errors",
            current_value=10.0,
            threshold=5.0,
            condition="above",
            severity=AlertSeverity.CRITICAL,
            message="test",
            triggered_at=100.0,
        )
        snap = MetricSnapshot(
            timestamp=100.0,
            metrics={"errors": 10.0},
            alerts=(alert,),
        )
        assert snap.has_critical_alerts is True


# ---------------------------------------------------------------------------
# Prometheus export
# ---------------------------------------------------------------------------


class TestPrometheusExport:
    def test_basic_export(self) -> None:
        snap = MetricSnapshot(
            timestamp=100.0,
            metrics={"orders.placed": 5.0, "cache-size": 100.0},
            alerts=(),
        )
        text = snap.to_prometheus_text()
        assert "orders_placed 5.0" in text
        assert "cache_size 100.0" in text

    def test_empty_export(self) -> None:
        snap = MetricSnapshot(timestamp=100.0, metrics={}, alerts=())
        assert snap.to_prometheus_text() == ""


# ---------------------------------------------------------------------------
# Dict export
# ---------------------------------------------------------------------------


class TestDictExport:
    def test_basic_export(self) -> None:
        alert = Alert(
            metric_name="errors",
            current_value=10.0,
            threshold=5.0,
            condition="above",
            severity=AlertSeverity.WARNING,
            message="too many",
            triggered_at=100.0,
        )
        snap = MetricSnapshot(
            timestamp=100.0,
            metrics={"errors": 10.0},
            alerts=(alert,),
        )
        d = snap.to_dict()
        assert d["timestamp"] == 100.0
        assert d["metrics"]["errors"] == 10.0
        assert len(d["alerts"]) == 1
        assert d["alerts"][0]["severity"] == "warning"

    def test_json_serializable(self) -> None:
        snap = MetricSnapshot(
            timestamp=100.0,
            metrics={"a": 1.0},
            alerts=(),
        )
        # Should not raise.
        text = json.dumps(snap.to_dict())
        assert "a" in text


# ---------------------------------------------------------------------------
# Snapshot integration
# ---------------------------------------------------------------------------


class TestSnapshotIntegration:
    def test_full_snapshot(self) -> None:
        reg = MetricsRegistry()
        reg.counter("orders_placed").increment(5.0, now=100.0)
        reg.gauge("cache_size").set(150.0, now=100.0)
        reg.counter("errors").increment(10.0, now=100.0)
        reg.add_alert(AlertThreshold(
            metric_name="errors",
            condition="above",
            threshold=5.0,
            severity=AlertSeverity.CRITICAL,
        ))

        snap = reg.snapshot(now=100.0)
        assert snap.metrics["orders_placed"] == 5.0
        assert snap.metrics["cache_size"] == 150.0
        assert snap.has_critical_alerts is True
        assert len(snap.alerts) == 1

    def test_multiple_alerts(self) -> None:
        reg = MetricsRegistry()
        reg.counter("errors").increment(10.0, now=100.0)
        reg.gauge("coverage").set(0.2, now=100.0)
        reg.add_alert(AlertThreshold(
            metric_name="errors", condition="above", threshold=5.0,
        ))
        reg.add_alert(AlertThreshold(
            metric_name="coverage", condition="below", threshold=0.5,
        ))

        snap = reg.snapshot(now=100.0)
        assert len(snap.alerts) == 2
