"""Tests for Phase 4I: Expected-vs-realized drift monitoring."""

from __future__ import annotations

import pytest

from arb_bot.framework.drift_monitor import (
    DriftMonitor,
    DriftMonitorConfig,
    DriftReport,
    MetricDrift,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mon(**kw) -> DriftMonitor:
    return DriftMonitor(DriftMonitorConfig(**kw))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self) -> None:
        cfg = DriftMonitorConfig()
        assert cfg.alert_threshold == 0.05
        assert cfg.relative_alert_threshold == 0.30
        assert cfg.window_size == 50
        assert cfg.sizing_reduction_per_alert == 0.10
        assert cfg.min_sizing_multiplier == 0.25
        assert cfg.min_observations == 5

    def test_frozen(self) -> None:
        cfg = DriftMonitorConfig()
        with pytest.raises(AttributeError):
            cfg.alert_threshold = 0.1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Record observations
# ---------------------------------------------------------------------------


class TestRecord:
    def test_records(self) -> None:
        m = _mon()
        m.record("edge", expected=0.03, realized=0.025)
        assert m.observation_count("edge") == 1

    def test_window_trimming(self) -> None:
        m = _mon(window_size=5)
        for i in range(10):
            m.record("edge", expected=float(i), realized=float(i))
        assert m.observation_count("edge") == 5

    def test_unknown_metric(self) -> None:
        m = _mon()
        assert m.observation_count("nonexistent") == 0


# ---------------------------------------------------------------------------
# Metric drift — no alert
# ---------------------------------------------------------------------------


class TestMetricDriftNoAlert:
    def test_zero_drift(self) -> None:
        m = _mon(min_observations=3)
        for _ in range(5):
            m.record("edge", expected=0.03, realized=0.03)
        drift = m.metric_drift("edge")
        assert drift.mean_drift == pytest.approx(0.0)
        assert drift.alert is False

    def test_below_threshold(self) -> None:
        # Both absolute AND relative thresholds must not be exceeded.
        m = _mon(alert_threshold=0.05, relative_alert_threshold=0.90,
                 min_observations=3)
        for _ in range(5):
            m.record("edge", expected=0.03, realized=0.01)
        drift = m.metric_drift("edge")
        assert drift.abs_drift == pytest.approx(0.02)
        assert drift.alert is False  # abs 0.02 < 0.05, relative 66% < 90%

    def test_below_min_observations(self) -> None:
        m = _mon(min_observations=10, alert_threshold=0.001)
        for _ in range(5):
            m.record("edge", expected=0.03, realized=0.0)
        drift = m.metric_drift("edge")
        assert drift.alert is False  # Only 5 < 10 observations.

    def test_empty_metric(self) -> None:
        m = _mon()
        drift = m.metric_drift("edge")
        assert drift.observation_count == 0
        assert drift.alert is False


# ---------------------------------------------------------------------------
# Metric drift — alert
# ---------------------------------------------------------------------------


class TestMetricDriftAlert:
    def test_absolute_drift_alert(self) -> None:
        m = _mon(alert_threshold=0.05, min_observations=3)
        for _ in range(5):
            m.record("edge", expected=0.10, realized=0.03)
        drift = m.metric_drift("edge")
        assert drift.abs_drift == pytest.approx(0.07)
        assert drift.alert is True
        assert drift.alert_reason == "absolute_drift"

    def test_relative_drift_alert(self) -> None:
        m = _mon(alert_threshold=1.0, relative_alert_threshold=0.30,
                 min_observations=3)
        for _ in range(5):
            m.record("fill_rate", expected=0.80, realized=0.50)
        drift = m.metric_drift("fill_rate")
        # drift = -0.30, relative = 0.30/0.80 = 0.375 > 0.30
        assert drift.alert is True
        assert drift.alert_reason == "relative_drift"

    def test_negative_drift(self) -> None:
        m = _mon(alert_threshold=0.05, min_observations=3)
        for _ in range(5):
            m.record("cost", expected=0.50, realized=0.60)
        drift = m.metric_drift("cost")
        assert drift.mean_drift == pytest.approx(0.10)
        assert drift.alert is True


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


class TestReport:
    def test_no_alerts(self) -> None:
        m = _mon(min_observations=3)
        for _ in range(5):
            m.record("edge", expected=0.03, realized=0.03)
        report = m.report()
        assert report.any_alert is False
        assert report.alert_count == 0
        assert report.sizing_multiplier == 1.0

    def test_one_alert(self) -> None:
        m = _mon(alert_threshold=0.05, min_observations=3,
                 sizing_reduction_per_alert=0.10)
        for _ in range(5):
            m.record("edge", expected=0.10, realized=0.03)
        report = m.report()
        assert report.any_alert is True
        assert report.alert_count == 1
        assert report.sizing_multiplier == pytest.approx(0.90)
        assert "edge" in report.alerting_metrics

    def test_multiple_alerts(self) -> None:
        m = _mon(alert_threshold=0.05, min_observations=3,
                 sizing_reduction_per_alert=0.15)
        for _ in range(5):
            m.record("edge", expected=0.10, realized=0.03)
            m.record("fill_rate", expected=0.80, realized=0.50)
        report = m.report()
        assert report.alert_count == 2
        # 1.0 - 2 * 0.15 = 0.70
        assert report.sizing_multiplier == pytest.approx(0.70)

    def test_multiplier_floored(self) -> None:
        m = _mon(alert_threshold=0.01, min_observations=3,
                 sizing_reduction_per_alert=0.30,
                 min_sizing_multiplier=0.25)
        for _ in range(5):
            m.record("a", expected=1.0, realized=0.0)
            m.record("b", expected=1.0, realized=0.0)
            m.record("c", expected=1.0, realized=0.0)
        report = m.report()
        # 1.0 - 3 * 0.30 = 0.10, floored to 0.25
        assert report.sizing_multiplier == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clears_all(self) -> None:
        m = _mon()
        m.record("edge", expected=0.03, realized=0.02)
        m.clear()
        assert m.observation_count("edge") == 0
        report = m.report()
        assert report.alert_count == 0


# ---------------------------------------------------------------------------
# Config property
# ---------------------------------------------------------------------------


class TestConfigProperty:
    def test_config_accessible(self) -> None:
        cfg = DriftMonitorConfig(alert_threshold=0.10)
        m = DriftMonitor(cfg)
        assert m.config.alert_threshold == 0.10


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_gradual_drift(self) -> None:
        """Simulate drift building up over time."""
        m = _mon(alert_threshold=0.05, min_observations=5)

        # Initially accurate.
        for _ in range(5):
            m.record("edge", expected=0.03, realized=0.029)
        assert m.report().any_alert is False

        # Drift increases.
        for _ in range(10):
            m.record("edge", expected=0.03, realized=0.01)
        report = m.report()
        assert report.any_alert is True
        assert report.sizing_multiplier < 1.0
