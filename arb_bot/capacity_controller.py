"""Dynamic capacity controller (Phase 5J).

Auto-adjusts universe size, scoring depth, and scan cadence based
on CPU, memory, API, and latency budgets.

Usage::

    ctrl = CapacityController(config)
    ctrl.update_metrics(cpu_pct=75, memory_pct=60, api_usage_pct=80, latency_p95=0.5)
    limits = ctrl.compute_limits()
    # limits.max_universe_size, limits.scoring_depth, etc.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CapacityControllerConfig:
    """Configuration for capacity controller.

    Parameters
    ----------
    cpu_target_pct:
        Target CPU usage percentage. Default 70.
    memory_target_pct:
        Target memory usage percentage. Default 75.
    api_target_pct:
        Target API budget usage percentage. Default 80.
    latency_target_seconds:
        Target p95 latency. Default 1.0.
    base_universe_size:
        Default maximum universe size. Default 500.
    base_scoring_depth:
        Default scoring depth (top-N to evaluate). Default 100.
    base_scan_interval:
        Default scan interval (seconds). Default 5.0.
    min_universe_size:
        Minimum universe size. Default 50.
    min_scoring_depth:
        Minimum scoring depth. Default 10.
    max_scan_interval:
        Maximum scan interval (seconds). Default 60.0.
    adjustment_speed:
        How quickly to adjust (0-1). 0.1 = slow. Default 0.2.
    """

    cpu_target_pct: float = 70.0
    memory_target_pct: float = 75.0
    api_target_pct: float = 80.0
    latency_target_seconds: float = 1.0
    base_universe_size: int = 500
    base_scoring_depth: int = 100
    base_scan_interval: float = 5.0
    min_universe_size: int = 50
    min_scoring_depth: int = 10
    max_scan_interval: float = 60.0
    adjustment_speed: float = 0.2


# ---------------------------------------------------------------------------
# Resource metrics
# ---------------------------------------------------------------------------


@dataclass
class ResourceMetrics:
    """Current resource usage metrics."""

    cpu_pct: float = 0.0
    memory_pct: float = 0.0
    api_usage_pct: float = 0.0
    latency_p95: float = 0.0
    timestamp: float = 0.0


# ---------------------------------------------------------------------------
# Capacity limits
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CapacityLimits:
    """Computed capacity limits."""

    max_universe_size: int
    scoring_depth: int
    scan_interval: float
    pressure_score: float  # 0 = no pressure, 1 = max pressure.
    limiting_resource: str


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class CapacityController:
    """Auto-adjusts capacity based on resource usage.

    Monitors CPU, memory, API, and latency metrics, and adjusts
    universe size, scoring depth, and scan cadence to stay within
    budgets.
    """

    def __init__(self, config: CapacityControllerConfig | None = None) -> None:
        self._config = config or CapacityControllerConfig()
        self._metrics = ResourceMetrics()
        self._current_universe: float = float(self._config.base_universe_size)
        self._current_depth: float = float(self._config.base_scoring_depth)
        self._current_interval: float = self._config.base_scan_interval
        self._history: List[ResourceMetrics] = []

    @property
    def config(self) -> CapacityControllerConfig:
        return self._config

    def update_metrics(
        self,
        cpu_pct: float = 0.0,
        memory_pct: float = 0.0,
        api_usage_pct: float = 0.0,
        latency_p95: float = 0.0,
        now: float | None = None,
    ) -> None:
        """Update resource metrics."""
        if now is None:
            now = time.monotonic()
        self._metrics = ResourceMetrics(
            cpu_pct=cpu_pct,
            memory_pct=memory_pct,
            api_usage_pct=api_usage_pct,
            latency_p95=latency_p95,
            timestamp=now,
        )
        self._history.append(self._metrics)
        if len(self._history) > 100:
            self._history = self._history[-100:]

    def _compute_pressure(self) -> tuple[float, str]:
        """Compute pressure score from metrics."""
        cfg = self._config
        m = self._metrics

        pressures: Dict[str, float] = {}

        if cfg.cpu_target_pct > 0:
            pressures["cpu"] = max(0.0, (m.cpu_pct - cfg.cpu_target_pct) / (100 - cfg.cpu_target_pct))

        if cfg.memory_target_pct > 0:
            pressures["memory"] = max(0.0, (m.memory_pct - cfg.memory_target_pct) / (100 - cfg.memory_target_pct))

        if cfg.api_target_pct > 0:
            pressures["api"] = max(0.0, (m.api_usage_pct - cfg.api_target_pct) / (100 - cfg.api_target_pct))

        if cfg.latency_target_seconds > 0:
            pressures["latency"] = max(0.0, (m.latency_p95 - cfg.latency_target_seconds) / cfg.latency_target_seconds)

        if not pressures:
            return 0.0, "none"

        max_resource = max(pressures, key=pressures.get)  # type: ignore
        return min(1.0, pressures[max_resource]), max_resource

    def compute_limits(self) -> CapacityLimits:
        """Compute current capacity limits based on resource pressure."""
        cfg = self._config
        speed = cfg.adjustment_speed

        pressure, limiting = self._compute_pressure()

        if pressure > 0:
            # Reduce capacity proportionally to pressure.
            reduction = pressure * speed
            self._current_universe *= (1.0 - reduction)
            self._current_depth *= (1.0 - reduction)
            self._current_interval *= (1.0 + reduction)
        else:
            # Gradually recover.
            recovery = speed * 0.5  # Recover slower than reduce.
            max_universe = float(cfg.base_universe_size)
            max_depth = float(cfg.base_scoring_depth)
            min_interval = cfg.base_scan_interval

            self._current_universe = min(
                max_universe,
                self._current_universe * (1.0 + recovery),
            )
            self._current_depth = min(
                max_depth,
                self._current_depth * (1.0 + recovery),
            )
            self._current_interval = max(
                min_interval,
                self._current_interval * (1.0 - recovery),
            )

        # Apply bounds.
        universe = max(cfg.min_universe_size, int(self._current_universe))
        universe = min(cfg.base_universe_size, universe)
        depth = max(cfg.min_scoring_depth, int(self._current_depth))
        depth = min(cfg.base_scoring_depth, depth)
        interval = max(cfg.base_scan_interval, self._current_interval)
        interval = min(cfg.max_scan_interval, interval)

        return CapacityLimits(
            max_universe_size=universe,
            scoring_depth=depth,
            scan_interval=interval,
            pressure_score=pressure,
            limiting_resource=limiting,
        )

    def get_metrics(self) -> ResourceMetrics:
        """Get current metrics."""
        return self._metrics

    def clear(self) -> None:
        """Reset to defaults."""
        cfg = self._config
        self._metrics = ResourceMetrics()
        self._current_universe = float(cfg.base_universe_size)
        self._current_depth = float(cfg.base_scoring_depth)
        self._current_interval = cfg.base_scan_interval
        self._history.clear()
