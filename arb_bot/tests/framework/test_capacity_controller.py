"""Tests for Phase 5J: Dynamic capacity controller."""

from __future__ import annotations

import pytest

from arb_bot.framework.capacity_controller import (
    CapacityController,
    CapacityControllerConfig,
    CapacityLimits,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ctrl(**kw) -> CapacityController:
    return CapacityController(CapacityControllerConfig(**kw))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self) -> None:
        cfg = CapacityControllerConfig()
        assert cfg.cpu_target_pct == 70.0
        assert cfg.memory_target_pct == 75.0
        assert cfg.api_target_pct == 80.0
        assert cfg.latency_target_seconds == 1.0
        assert cfg.base_universe_size == 500
        assert cfg.base_scoring_depth == 100
        assert cfg.base_scan_interval == 5.0

    def test_frozen(self) -> None:
        cfg = CapacityControllerConfig()
        with pytest.raises(AttributeError):
            cfg.cpu_target_pct = 50.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# No pressure — returns base limits
# ---------------------------------------------------------------------------


class TestNoPressure:
    def test_base_limits(self) -> None:
        ctrl = _ctrl()
        ctrl.update_metrics(cpu_pct=50, memory_pct=50, api_usage_pct=50, latency_p95=0.5)
        limits = ctrl.compute_limits()
        assert limits.max_universe_size == 500
        assert limits.scoring_depth == 100
        assert limits.scan_interval == pytest.approx(5.0)
        assert limits.pressure_score == 0.0

    def test_at_target(self) -> None:
        ctrl = _ctrl(cpu_target_pct=70)
        ctrl.update_metrics(cpu_pct=70)
        limits = ctrl.compute_limits()
        assert limits.pressure_score == 0.0


# ---------------------------------------------------------------------------
# CPU pressure
# ---------------------------------------------------------------------------


class TestCPUPressure:
    def test_above_target(self) -> None:
        ctrl = _ctrl(cpu_target_pct=70)
        ctrl.update_metrics(cpu_pct=90)
        limits = ctrl.compute_limits()
        assert limits.pressure_score > 0
        assert limits.limiting_resource == "cpu"
        assert limits.max_universe_size < 500

    def test_high_cpu_reduces_depth(self) -> None:
        ctrl = _ctrl(cpu_target_pct=70, adjustment_speed=0.5)
        ctrl.update_metrics(cpu_pct=95)
        limits = ctrl.compute_limits()
        assert limits.scoring_depth < 100

    def test_high_cpu_increases_interval(self) -> None:
        ctrl = _ctrl(cpu_target_pct=70, adjustment_speed=0.5)
        ctrl.update_metrics(cpu_pct=95)
        limits = ctrl.compute_limits()
        assert limits.scan_interval > 5.0


# ---------------------------------------------------------------------------
# API pressure
# ---------------------------------------------------------------------------


class TestAPIPressure:
    def test_above_api_target(self) -> None:
        ctrl = _ctrl(api_target_pct=80)
        ctrl.update_metrics(api_usage_pct=95)
        limits = ctrl.compute_limits()
        assert limits.pressure_score > 0
        assert limits.limiting_resource == "api"


# ---------------------------------------------------------------------------
# Latency pressure
# ---------------------------------------------------------------------------


class TestLatencyPressure:
    def test_above_latency_target(self) -> None:
        ctrl = _ctrl(latency_target_seconds=1.0)
        ctrl.update_metrics(latency_p95=2.0)
        limits = ctrl.compute_limits()
        assert limits.pressure_score > 0
        assert limits.limiting_resource == "latency"


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------


class TestRecovery:
    def test_gradual_recovery(self) -> None:
        ctrl = _ctrl(cpu_target_pct=70, adjustment_speed=0.5)
        # Push down.
        ctrl.update_metrics(cpu_pct=95)
        limits_pressured = ctrl.compute_limits()
        universe_down = limits_pressured.max_universe_size

        # Relieve pressure.
        ctrl.update_metrics(cpu_pct=50)
        limits_recovering = ctrl.compute_limits()
        # Should recover (increase) but maybe not fully.
        assert limits_recovering.max_universe_size >= universe_down

    def test_recovery_capped_at_base(self) -> None:
        ctrl = _ctrl()
        # No pressure — should not exceed base.
        for _ in range(100):
            ctrl.update_metrics(cpu_pct=10, memory_pct=10)
            ctrl.compute_limits()
        limits = ctrl.compute_limits()
        assert limits.max_universe_size <= 500
        assert limits.scoring_depth <= 100


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------


class TestBounds:
    def test_min_universe_size(self) -> None:
        ctrl = _ctrl(
            min_universe_size=50,
            cpu_target_pct=50,
            adjustment_speed=1.0,
        )
        # Extreme pressure.
        for _ in range(20):
            ctrl.update_metrics(cpu_pct=100)
            ctrl.compute_limits()
        limits = ctrl.compute_limits()
        assert limits.max_universe_size >= 50

    def test_min_scoring_depth(self) -> None:
        ctrl = _ctrl(
            min_scoring_depth=10,
            cpu_target_pct=50,
            adjustment_speed=1.0,
        )
        for _ in range(20):
            ctrl.update_metrics(cpu_pct=100)
            ctrl.compute_limits()
        limits = ctrl.compute_limits()
        assert limits.scoring_depth >= 10

    def test_max_scan_interval(self) -> None:
        ctrl = _ctrl(
            max_scan_interval=60.0,
            cpu_target_pct=50,
            adjustment_speed=1.0,
        )
        for _ in range(20):
            ctrl.update_metrics(cpu_pct=100)
            ctrl.compute_limits()
        limits = ctrl.compute_limits()
        assert limits.scan_interval <= 60.0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_get_metrics(self) -> None:
        ctrl = _ctrl()
        ctrl.update_metrics(cpu_pct=75, memory_pct=60, now=100.0)
        m = ctrl.get_metrics()
        assert m.cpu_pct == 75
        assert m.memory_pct == 60


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_reset(self) -> None:
        ctrl = _ctrl(cpu_target_pct=50, adjustment_speed=1.0)
        ctrl.update_metrics(cpu_pct=100)
        ctrl.compute_limits()
        ctrl.clear()
        limits = ctrl.compute_limits()
        assert limits.max_universe_size == 500


# ---------------------------------------------------------------------------
# Config property
# ---------------------------------------------------------------------------


class TestConfigProperty:
    def test_config_accessible(self) -> None:
        cfg = CapacityControllerConfig(base_universe_size=300)
        ctrl = CapacityController(cfg)
        assert ctrl.config.base_universe_size == 300


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_pressure_cycle(self) -> None:
        """Simulate: normal → pressure → reduce → relieve → recover."""
        ctrl = _ctrl(
            cpu_target_pct=70,
            base_universe_size=500,
            min_universe_size=100,
            adjustment_speed=0.3,
        )

        # Normal operation.
        ctrl.update_metrics(cpu_pct=50)
        limits = ctrl.compute_limits()
        assert limits.max_universe_size == 500

        # CPU spike.
        for _ in range(5):
            ctrl.update_metrics(cpu_pct=95)
            limits = ctrl.compute_limits()
        reduced = limits.max_universe_size
        assert reduced < 500

        # CPU normalizes — gradual recovery.
        for _ in range(10):
            ctrl.update_metrics(cpu_pct=50)
            limits = ctrl.compute_limits()
        recovered = limits.max_universe_size
        assert recovered > reduced
