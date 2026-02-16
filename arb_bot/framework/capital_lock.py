"""Time-to-resolution capital lock penalty (Phase 4E).

Penalizes position sizing based on how long capital will be locked.
Shorter-duration markets get a smaller penalty; longer-duration
markets are sized down to reflect opportunity cost.

Usage::

    lock = CapitalLockPenalty(config)
    result = lock.compute(
        kelly_fraction=0.20, hours_to_resolution=48.0,
        annual_opportunity_rate=0.10,
    )
    adjusted = result.adjusted_fraction
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CapitalLockConfig:
    """Configuration for capital lock penalty.

    Parameters
    ----------
    penalty_per_hour:
        Base penalty per hour of lock-up. Applied as a fraction
        reduction. Default 0.001 (0.1% per hour).
    max_penalty_fraction:
        Maximum total penalty (caps penalty at this level).
        Default 0.50 (50% reduction maximum).
    free_hours:
        Hours below which no penalty applies (short-duration
        markets are "free"). Default 1.0.
    use_opportunity_cost:
        If True, incorporate annualized opportunity cost rate.
        Default True.
    hours_per_year:
        Hours in a year for annualization. Default 8760.
    max_lock_hours:
        Hard gate: reject trades with resolution beyond this.
        Default 720 (30 days).
    """

    penalty_per_hour: float = 0.001
    max_penalty_fraction: float = 0.50
    free_hours: float = 1.0
    use_opportunity_cost: bool = True
    hours_per_year: float = 8760.0
    max_lock_hours: float = 720.0


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CapitalLockResult:
    """Result of capital lock penalty computation."""

    input_fraction: float
    adjusted_fraction: float
    hours_to_resolution: float
    penalty_fraction: float
    opportunity_cost_penalty: float
    total_penalty: float
    blocked: bool
    block_reason: str


# ---------------------------------------------------------------------------
# Penalty manager
# ---------------------------------------------------------------------------


class CapitalLockPenalty:
    """Computes capital lock-up penalties for position sizing.

    Reduces Kelly fractions based on expected time-to-resolution
    and the opportunity cost of locked capital.
    """

    def __init__(self, config: CapitalLockConfig | None = None) -> None:
        self._config = config or CapitalLockConfig()
        self._history: List[CapitalLockResult] = []
        self._resolution_actuals: Dict[str, List[float]] = {}

    @property
    def config(self) -> CapitalLockConfig:
        return self._config

    def compute(
        self,
        kelly_fraction: float,
        hours_to_resolution: float,
        annual_opportunity_rate: float = 0.10,
    ) -> CapitalLockResult:
        """Compute the capital lock penalty.

        Parameters
        ----------
        kelly_fraction:
            Input Kelly fraction to adjust.
        hours_to_resolution:
            Expected hours until the market resolves.
        annual_opportunity_rate:
            Annualized opportunity cost rate. Default 10%.
        """
        cfg = self._config

        # Hard gate: too long.
        if hours_to_resolution > cfg.max_lock_hours:
            result = CapitalLockResult(
                input_fraction=kelly_fraction,
                adjusted_fraction=0.0,
                hours_to_resolution=hours_to_resolution,
                penalty_fraction=1.0,
                opportunity_cost_penalty=0.0,
                total_penalty=1.0,
                blocked=True,
                block_reason="exceeds_max_lock_hours",
            )
            self._history.append(result)
            return result

        # Compute base penalty.
        effective_hours = max(0.0, hours_to_resolution - cfg.free_hours)
        base_penalty = effective_hours * cfg.penalty_per_hour

        # Opportunity cost penalty.
        opp_penalty = 0.0
        if cfg.use_opportunity_cost and cfg.hours_per_year > 0:
            lock_fraction_of_year = hours_to_resolution / cfg.hours_per_year
            opp_penalty = annual_opportunity_rate * lock_fraction_of_year

        total_penalty = min(base_penalty + opp_penalty, cfg.max_penalty_fraction)
        total_penalty = max(0.0, total_penalty)

        adjusted = kelly_fraction * (1.0 - total_penalty)
        adjusted = max(0.0, adjusted)

        result = CapitalLockResult(
            input_fraction=kelly_fraction,
            adjusted_fraction=adjusted,
            hours_to_resolution=hours_to_resolution,
            penalty_fraction=base_penalty,
            opportunity_cost_penalty=opp_penalty,
            total_penalty=total_penalty,
            blocked=False,
            block_reason="",
        )
        self._history.append(result)
        return result

    def record_actual_resolution(
        self, market_id: str, actual_hours: float
    ) -> None:
        """Record actual resolution time for future calibration."""
        if market_id not in self._resolution_actuals:
            self._resolution_actuals[market_id] = []
        self._resolution_actuals[market_id].append(actual_hours)

    def avg_penalty(self, n: int = 50) -> float:
        """Average penalty over recent computations."""
        recent = self._history[-n:] if self._history else []
        if not recent:
            return 0.0
        return sum(r.total_penalty for r in recent) / len(recent)

    def clear(self) -> None:
        """Clear history and actuals."""
        self._history.clear()
        self._resolution_actuals.clear()
