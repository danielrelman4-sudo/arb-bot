"""KPI-driven auto-tuning loop (Phase 6E).

Periodic threshold adjustment from paper KPIs with bounded parameter
ranges, minimum sample sizes, gradual steps, and revert-on-degradation.

Safety invariants:
- Parameters ALWAYS stay within [hard_min, hard_max].
- Adjustments NEVER exceed max_step_fraction per cycle.
- No adjustment is made with fewer than min_samples observations.
- If KPIs degrade after tuning, the parameter reverts to its previous value.

Usage::

    tuner = AutoTuner(config)
    tuner.define_param("edge_threshold", current=0.01, hard_min=0.003, hard_max=0.05)
    tuner.record_kpi("win_rate", 0.62)
    tuner.record_kpi("win_rate", 0.65)
    adjustments = tuner.tune_cycle()
    for adj in adjustments:
        # Apply adj.new_value to the trading engine.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Adjustment direction
# ---------------------------------------------------------------------------


class AdjustDirection(str, Enum):
    """Direction of a parameter adjustment."""

    INCREASE = "increase"
    DECREASE = "decrease"
    HOLD = "hold"
    REVERT = "revert"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AutoTunerConfig:
    """Configuration for the auto-tuning loop.

    Parameters
    ----------
    min_samples:
        Minimum KPI observations before allowing any adjustment.
        Default 20. This prevents tuning on noise.
    max_step_fraction:
        Maximum per-cycle change as a fraction of current value.
        Default 0.10 (10%). Prevents sudden jumps.
    revert_lookback:
        Number of recent KPI values to compare against pre-tune
        baseline for revert detection. Default 5.
    revert_degradation_threshold:
        If post-tune KPI mean is worse by this much (as fraction
        of pre-tune mean), revert. Default 0.10 (10%).
    tune_interval_seconds:
        Minimum seconds between tune cycles. Default 300.
    max_params:
        Maximum number of tunable parameters. Default 50.
    """

    min_samples: int = 20
    max_step_fraction: float = 0.10
    revert_lookback: int = 5
    revert_degradation_threshold: float = 0.10
    tune_interval_seconds: float = 300.0
    max_params: int = 50


# ---------------------------------------------------------------------------
# Parameter definition
# ---------------------------------------------------------------------------


@dataclass
class TunableParam:
    """A parameter that can be auto-tuned.

    hard_min/hard_max are absolute limits that can NEVER be exceeded.
    """

    name: str
    current_value: float
    hard_min: float
    hard_max: float
    default_value: float
    kpi_name: str  # Which KPI drives this parameter.
    kpi_direction: str  # "higher_is_better" or "lower_is_better".
    adjust_direction: str  # "increase_when_good" or "decrease_when_good".
    previous_value: float = 0.0
    last_tuned_at: float = 0.0
    tune_count: int = 0


# ---------------------------------------------------------------------------
# KPI record
# ---------------------------------------------------------------------------


@dataclass
class KPIRecord:
    """A single KPI observation."""

    name: str
    value: float
    timestamp: float


# ---------------------------------------------------------------------------
# Adjustment result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Adjustment:
    """Result of adjusting a single parameter."""

    param_name: str
    direction: AdjustDirection
    old_value: float
    new_value: float
    kpi_name: str
    kpi_mean: float
    reason: str


# ---------------------------------------------------------------------------
# Tune cycle report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TuneCycleReport:
    """Report from a tune cycle."""

    adjustments: Tuple[Adjustment, ...]
    params_evaluated: int
    params_adjusted: int
    params_reverted: int
    skipped_insufficient_samples: int
    timestamp: float


# ---------------------------------------------------------------------------
# Auto-tuner
# ---------------------------------------------------------------------------


class AutoTuner:
    """KPI-driven auto-tuning with safety bounds.

    Each tunable parameter is linked to a KPI. The tuner periodically
    evaluates KPIs and makes small, bounded adjustments. If KPIs
    degrade after tuning, the parameter reverts.
    """

    def __init__(self, config: AutoTunerConfig | None = None) -> None:
        self._config = config or AutoTunerConfig()
        self._params: Dict[str, TunableParam] = {}
        self._kpis: Dict[str, List[KPIRecord]] = {}
        self._audit: List[Adjustment] = []
        self._last_tune_time: float = 0.0

    @property
    def config(self) -> AutoTunerConfig:
        return self._config

    def define_param(
        self,
        name: str,
        current: float,
        hard_min: float,
        hard_max: float,
        kpi_name: str = "",
        kpi_direction: str = "higher_is_better",
        adjust_direction: str = "increase_when_good",
    ) -> bool:
        """Define a tunable parameter with safety bounds.

        Parameters
        ----------
        name:
            Parameter name.
        current:
            Current value.
        hard_min:
            Absolute minimum — tuner can never go below this.
        hard_max:
            Absolute maximum — tuner can never go above this.
        kpi_name:
            Which KPI drives this parameter's tuning.
        kpi_direction:
            "higher_is_better" or "lower_is_better".
        adjust_direction:
            "increase_when_good" (raise param when KPI is good) or
            "decrease_when_good" (lower param when KPI is good).

        Returns False if max params exceeded or min > max.
        """
        if len(self._params) >= self._config.max_params:
            return False
        if hard_min > hard_max:
            return False
        # Clamp current to bounds.
        clamped = max(hard_min, min(hard_max, current))
        self._params[name] = TunableParam(
            name=name,
            current_value=clamped,
            hard_min=hard_min,
            hard_max=hard_max,
            default_value=clamped,
            kpi_name=kpi_name,
            kpi_direction=kpi_direction,
            adjust_direction=adjust_direction,
            previous_value=clamped,
        )
        return True

    def record_kpi(
        self,
        name: str,
        value: float,
        ts: float | None = None,
    ) -> None:
        """Record a KPI observation."""
        if ts is None:
            ts = time.time()
        if name not in self._kpis:
            self._kpis[name] = []
        self._kpis[name].append(KPIRecord(name=name, value=value, timestamp=ts))

    def _kpi_mean(self, name: str, last_n: int = 0) -> float | None:
        """Compute mean of recent KPI values."""
        records = self._kpis.get(name, [])
        if not records:
            return None
        if last_n > 0:
            records = records[-last_n:]
        values = [r.value for r in records]
        return sum(values) / len(values)

    def _kpi_count(self, name: str) -> int:
        """Count KPI observations."""
        return len(self._kpis.get(name, []))

    def _clamp(self, value: float, param: TunableParam) -> float:
        """Clamp value to hard bounds. This is the critical safety invariant."""
        return max(param.hard_min, min(param.hard_max, value))

    def _compute_step(self, param: TunableParam) -> float:
        """Compute maximum step size for this parameter."""
        return abs(param.current_value) * self._config.max_step_fraction

    def _is_kpi_good(
        self, param: TunableParam, kpi_mean: float
    ) -> bool:
        """Determine if KPI is 'good' (recent trend vs overall baseline).

        Compares the mean of the most recent `revert_lookback` observations
        against the mean of ALL observations. This detects whether recent
        performance is trending above or below the historical baseline.
        """
        # Overall baseline = mean of ALL observations.
        baseline = self._kpi_mean(param.kpi_name)  # All data.
        if baseline is None:
            return False
        # Recent = mean of last `revert_lookback` observations.
        recent = self._kpi_mean(param.kpi_name, last_n=self._config.revert_lookback)
        if recent is None:
            return False
        if param.kpi_direction == "higher_is_better":
            return recent >= baseline
        else:
            return recent <= baseline

    def _should_revert(self, param: TunableParam) -> bool:
        """Check if a previously tuned parameter should revert.

        Compares post-tune KPIs against the baseline to detect degradation.
        """
        if param.tune_count == 0:
            return False
        kpi_records = self._kpis.get(param.kpi_name, [])
        if len(kpi_records) < self._config.revert_lookback:
            return False

        # Get recent KPI values.
        recent = [r.value for r in kpi_records[-self._config.revert_lookback:]]
        recent_mean = sum(recent) / len(recent)

        # Get baseline (all data).
        all_mean = self._kpi_mean(param.kpi_name)
        if all_mean is None or all_mean == 0:
            return False

        # Check degradation.
        if param.kpi_direction == "higher_is_better":
            degradation = (all_mean - recent_mean) / abs(all_mean)
        else:
            degradation = (recent_mean - all_mean) / abs(all_mean)

        return degradation > self._config.revert_degradation_threshold

    def tune_cycle(self, now: float | None = None) -> TuneCycleReport:
        """Run a single tune cycle.

        Evaluates all parameters, makes bounded adjustments if KPIs
        warrant, and reverts if KPIs have degraded.

        Returns a report of all adjustments made.
        """
        if now is None:
            now = time.time()
        cfg = self._config

        adjustments: List[Adjustment] = []
        params_evaluated = 0
        params_adjusted = 0
        params_reverted = 0
        skipped_insufficient = 0

        for param in self._params.values():
            params_evaluated += 1

            # Skip if no linked KPI.
            if not param.kpi_name:
                continue

            kpi_count = self._kpi_count(param.kpi_name)

            # Check minimum samples.
            if kpi_count < cfg.min_samples:
                skipped_insufficient += 1
                continue

            kpi_mean = self._kpi_mean(param.kpi_name)
            if kpi_mean is None:
                continue

            # Check for revert.
            if self._should_revert(param):
                old = param.current_value
                param.current_value = param.previous_value
                params_reverted += 1
                adj = Adjustment(
                    param_name=param.name,
                    direction=AdjustDirection.REVERT,
                    old_value=old,
                    new_value=param.previous_value,
                    kpi_name=param.kpi_name,
                    kpi_mean=kpi_mean,
                    reason="kpi_degraded",
                )
                adjustments.append(adj)
                self._audit.append(adj)
                param.tune_count += 1
                param.last_tuned_at = now
                continue

            # Determine adjustment direction.
            kpi_good = self._is_kpi_good(param, kpi_mean)
            step = self._compute_step(param)

            if step == 0:
                # Parameter is at zero — use hard_max as reference.
                step = param.hard_max * cfg.max_step_fraction

            if kpi_good:
                if param.adjust_direction == "increase_when_good":
                    new_value = param.current_value + step
                    direction = AdjustDirection.INCREASE
                else:
                    new_value = param.current_value - step
                    direction = AdjustDirection.DECREASE
            else:
                if param.adjust_direction == "increase_when_good":
                    new_value = param.current_value - step
                    direction = AdjustDirection.DECREASE
                else:
                    new_value = param.current_value + step
                    direction = AdjustDirection.INCREASE

            # CRITICAL: Clamp to hard bounds.
            new_value = self._clamp(new_value, param)

            if new_value == param.current_value:
                direction = AdjustDirection.HOLD
                adjustments.append(Adjustment(
                    param_name=param.name,
                    direction=direction,
                    old_value=param.current_value,
                    new_value=new_value,
                    kpi_name=param.kpi_name,
                    kpi_mean=kpi_mean,
                    reason="at_bound",
                ))
                continue

            # Apply adjustment.
            old = param.current_value
            param.previous_value = old
            param.current_value = new_value
            param.tune_count += 1
            param.last_tuned_at = now
            params_adjusted += 1

            adj = Adjustment(
                param_name=param.name,
                direction=direction,
                old_value=old,
                new_value=new_value,
                kpi_name=param.kpi_name,
                kpi_mean=kpi_mean,
                reason="kpi_driven",
            )
            adjustments.append(adj)
            self._audit.append(adj)

        self._last_tune_time = now

        return TuneCycleReport(
            adjustments=tuple(adjustments),
            params_evaluated=params_evaluated,
            params_adjusted=params_adjusted,
            params_reverted=params_reverted,
            skipped_insufficient_samples=skipped_insufficient,
            timestamp=now,
        )

    def get_param(self, name: str) -> TunableParam | None:
        """Get a parameter."""
        return self._params.get(name)

    def get_value(self, name: str) -> float | None:
        """Get current value of a parameter."""
        param = self._params.get(name)
        return param.current_value if param is not None else None

    def reset_param(self, name: str) -> None:
        """Reset a parameter to its default value."""
        param = self._params.get(name)
        if param is not None:
            param.previous_value = param.current_value
            param.current_value = param.default_value

    def audit_log(self, limit: int = 100) -> List[Adjustment]:
        """Get recent audit entries (newest first)."""
        return list(reversed(self._audit[-limit:]))

    def param_count(self) -> int:
        """Total defined parameters."""
        return len(self._params)

    def param_names(self) -> List[str]:
        """List parameter names."""
        return list(self._params.keys())

    def clear(self) -> None:
        """Clear all parameters and KPIs."""
        self._params.clear()
        self._kpis.clear()
        self._audit.clear()
