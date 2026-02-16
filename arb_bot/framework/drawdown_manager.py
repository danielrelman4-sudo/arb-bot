"""Drawdown-adaptive de-risking (Phase 4F).

Reduces position sizing when the portfolio experiences drawdowns
from its equity high-water mark. Implements graduated de-risking
tiers and full shutdown at critical drawdown levels.

Usage::

    dd = DrawdownManager(config)
    dd.update_equity(10_000.0)
    dd.update_equity(9_500.0)
    result = dd.compute_multiplier()
    adjusted_fraction = kelly * result.multiplier
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DrawdownConfig:
    """Configuration for drawdown-adaptive sizing.

    Parameters
    ----------
    tiers:
        List of (drawdown_pct, multiplier) pairs, sorted by drawdown.
        When drawdown exceeds a tier threshold, the corresponding
        multiplier is applied. Example: [(0.05, 0.75), (0.10, 0.50)].
        Default tiers: 5% → 0.75x, 10% → 0.50x, 15% → 0.25x.
    shutdown_drawdown:
        Drawdown level that triggers full shutdown (multiplier = 0).
        Default 0.20 (20%).
    recovery_buffer:
        After hitting shutdown, equity must recover this fraction
        of the drawdown before trading resumes. Prevents oscillation.
        Default 0.50 (must recover 50% of drawdown).
    min_equity_history:
        Minimum equity updates before drawdown is meaningful.
        Default 1.
    """

    tiers: Tuple[Tuple[float, float], ...] = (
        (0.05, 0.75),
        (0.10, 0.50),
        (0.15, 0.25),
    )
    shutdown_drawdown: float = 0.20
    recovery_buffer: float = 0.50
    min_equity_history: int = 1


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DrawdownResult:
    """Result of drawdown de-risking computation."""

    current_equity: float
    high_water_mark: float
    drawdown_pct: float
    multiplier: float
    active_tier: int  # -1 = none, 0-indexed tier index
    shutdown: bool
    recovering: bool


# ---------------------------------------------------------------------------
# Drawdown manager
# ---------------------------------------------------------------------------


class DrawdownManager:
    """Manages drawdown-adaptive position sizing.

    Tracks equity high-water mark and applies graduated
    multipliers to reduce sizing during drawdowns.
    """

    def __init__(self, config: DrawdownConfig | None = None) -> None:
        self._config = config or DrawdownConfig()
        self._high_water_mark: float = 0.0
        self._current_equity: float = 0.0
        self._equity_count: int = 0
        self._shutdown_active: bool = False
        self._shutdown_equity: float = 0.0
        self._history: List[DrawdownResult] = []

    @property
    def config(self) -> DrawdownConfig:
        return self._config

    @property
    def high_water_mark(self) -> float:
        return self._high_water_mark

    @property
    def current_equity(self) -> float:
        return self._current_equity

    @property
    def drawdown_pct(self) -> float:
        if self._high_water_mark <= 0:
            return 0.0
        return max(0.0, (self._high_water_mark - self._current_equity) / self._high_water_mark)

    @property
    def is_shutdown(self) -> bool:
        return self._shutdown_active

    def update_equity(self, equity: float) -> None:
        """Update current equity. Automatically updates HWM."""
        self._current_equity = equity
        self._equity_count += 1

        if equity > self._high_water_mark:
            self._high_water_mark = equity

            # Check recovery from shutdown.
            if self._shutdown_active:
                recovery_target = self._shutdown_equity + (
                    self._high_water_mark - self._shutdown_equity
                ) * self._config.recovery_buffer
                # If we've set a new HWM above shutdown level, we've recovered.
                self._shutdown_active = False

        # Check for shutdown.
        if not self._shutdown_active and self._high_water_mark > 0:
            dd = self.drawdown_pct
            if dd >= self._config.shutdown_drawdown:
                self._shutdown_active = True
                self._shutdown_equity = equity

    def compute_multiplier(self) -> DrawdownResult:
        """Compute the current drawdown multiplier."""
        dd = self.drawdown_pct
        cfg = self._config

        # Check recovery from shutdown.
        if self._shutdown_active:
            # Need equity to recover above threshold.
            if self._high_water_mark > 0:
                recovery_target = (
                    self._shutdown_equity
                    + (self._high_water_mark - self._shutdown_equity)
                    * cfg.recovery_buffer
                )
                if self._current_equity >= recovery_target:
                    self._shutdown_active = False
                else:
                    result = DrawdownResult(
                        current_equity=self._current_equity,
                        high_water_mark=self._high_water_mark,
                        drawdown_pct=dd,
                        multiplier=0.0,
                        active_tier=-1,
                        shutdown=True,
                        recovering=True,
                    )
                    self._history.append(result)
                    return result

        # Not enough history.
        if self._equity_count < cfg.min_equity_history:
            result = DrawdownResult(
                current_equity=self._current_equity,
                high_water_mark=self._high_water_mark,
                drawdown_pct=dd,
                multiplier=1.0,
                active_tier=-1,
                shutdown=False,
                recovering=False,
            )
            self._history.append(result)
            return result

        # Find active tier.
        active_tier = -1
        multiplier = 1.0
        sorted_tiers = sorted(cfg.tiers, key=lambda t: t[0])
        for i, (threshold, mult) in enumerate(sorted_tiers):
            if dd >= threshold:
                active_tier = i
                multiplier = mult

        result = DrawdownResult(
            current_equity=self._current_equity,
            high_water_mark=self._high_water_mark,
            drawdown_pct=dd,
            multiplier=multiplier,
            active_tier=active_tier,
            shutdown=False,
            recovering=False,
        )
        self._history.append(result)
        return result

    def reset(self) -> None:
        """Reset all state."""
        self._high_water_mark = 0.0
        self._current_equity = 0.0
        self._equity_count = 0
        self._shutdown_active = False
        self._shutdown_equity = 0.0
        self._history.clear()
