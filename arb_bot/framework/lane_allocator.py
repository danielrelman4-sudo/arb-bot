"""Dynamic per-lane bankroll allocation (Phase 4G).

Allocates bankroll across detection lanes based on historical
Sharpe ratio, win rate, and configurable caps. Lanes that
perform well get more capital; underperforming lanes are reduced.

Usage::

    alloc = LaneAllocator(config)
    alloc.record_trade("cross_venue", pnl=0.05)
    alloc.record_trade("parity", pnl=-0.02)
    allocations = alloc.compute_allocations(bankroll=10_000)
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
class LaneAllocatorConfig:
    """Configuration for per-lane bankroll allocation.

    Parameters
    ----------
    default_allocation:
        Equal allocation used when a lane has no history.
        Expressed as a fraction. Default 0.20 (20%).
    min_allocation:
        Minimum allocation for any active lane.
        Default 0.05 (5%).
    max_allocation:
        Maximum allocation for any single lane.
        Default 0.40 (40%).
    sharpe_weight:
        Weight given to Sharpe ratio in scoring. Default 0.60.
    win_rate_weight:
        Weight given to win rate in scoring. Default 0.40.
    min_trades_for_scoring:
        Minimum trades before a lane gets scored. Below this,
        default allocation is used. Default 10.
    window_size:
        Rolling window of trades for scoring. Default 100.
    smoothing_factor:
        Exponential smoothing between current and new allocation
        to prevent sudden jumps. Default 0.3.
    """

    default_allocation: float = 0.20
    min_allocation: float = 0.05
    max_allocation: float = 0.40
    sharpe_weight: float = 0.60
    win_rate_weight: float = 0.40
    min_trades_for_scoring: int = 10
    window_size: int = 100
    smoothing_factor: float = 0.3


# ---------------------------------------------------------------------------
# Lane stats
# ---------------------------------------------------------------------------


@dataclass
class LaneStats:
    """Computed statistics for a lane."""

    lane: str
    trade_count: int
    win_rate: float
    mean_pnl: float
    std_pnl: float
    sharpe: float
    score: float
    allocation: float


# ---------------------------------------------------------------------------
# Allocation result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AllocationResult:
    """Result of bankroll allocation computation."""

    total_bankroll: float
    allocations: Dict[str, float]  # lane → dollar amount
    fractions: Dict[str, float]  # lane → fraction
    stats: Dict[str, LaneStats]


# ---------------------------------------------------------------------------
# Lane allocator
# ---------------------------------------------------------------------------


class LaneAllocator:
    """Allocates bankroll across detection lanes based on performance.

    Uses a weighted score of Sharpe ratio and win rate to
    determine relative allocation. Smooths between periods
    to avoid sudden jumps.
    """

    def __init__(self, config: LaneAllocatorConfig | None = None) -> None:
        self._config = config or LaneAllocatorConfig()
        self._trades: Dict[str, List[float]] = defaultdict(list)
        self._current_fractions: Dict[str, float] = {}

    @property
    def config(self) -> LaneAllocatorConfig:
        return self._config

    def record_trade(self, lane: str, pnl: float) -> None:
        """Record a trade PnL for a lane."""
        self._trades[lane].append(pnl)
        window = self._config.window_size
        if len(self._trades[lane]) > window:
            self._trades[lane] = self._trades[lane][-window:]

    def trade_count(self, lane: str) -> int:
        """Number of recorded trades for a lane."""
        return len(self._trades.get(lane, []))

    def compute_allocations(
        self,
        bankroll: float,
        active_lanes: List[str] | None = None,
    ) -> AllocationResult:
        """Compute bankroll allocation across lanes.

        Parameters
        ----------
        bankroll:
            Total bankroll to allocate.
        active_lanes:
            List of lanes to allocate to. If None, uses all
            lanes with recorded trades.
        """
        cfg = self._config
        if active_lanes is None:
            active_lanes = list(self._trades.keys())

        if not active_lanes:
            return AllocationResult(
                total_bankroll=bankroll,
                allocations={},
                fractions={},
                stats={},
            )

        # Compute stats per lane.
        stats: Dict[str, LaneStats] = {}
        for lane in active_lanes:
            stats[lane] = self._compute_lane_stats(lane)

        # Compute raw scores.
        scored_lanes = [
            lane for lane in active_lanes
            if self.trade_count(lane) >= cfg.min_trades_for_scoring
        ]
        unscored_lanes = [
            lane for lane in active_lanes
            if self.trade_count(lane) < cfg.min_trades_for_scoring
        ]

        # Allocate: scored lanes get performance-weighted allocation,
        # unscored lanes get default allocation.
        fractions: Dict[str, float] = {}

        if scored_lanes:
            scores = {lane: max(stats[lane].score, 0.001) for lane in scored_lanes}
            total_score = sum(scores.values())

            # Remaining budget after unscored defaults.
            unscored_budget = len(unscored_lanes) * cfg.default_allocation
            scored_budget = max(0.0, 1.0 - unscored_budget)

            for lane in scored_lanes:
                raw_frac = (scores[lane] / total_score) * scored_budget
                fractions[lane] = max(cfg.min_allocation, min(cfg.max_allocation, raw_frac))

            for lane in unscored_lanes:
                fractions[lane] = cfg.default_allocation
        else:
            # All lanes get equal allocation.
            equal = 1.0 / len(active_lanes) if active_lanes else 0.0
            for lane in active_lanes:
                fractions[lane] = max(cfg.min_allocation, min(cfg.max_allocation, equal))

        # Normalize to sum ≤ 1.0.
        total = sum(fractions.values())
        if total > 1.0:
            for lane in fractions:
                fractions[lane] /= total

        # Smooth with previous allocations.
        alpha = cfg.smoothing_factor
        for lane in fractions:
            prev = self._current_fractions.get(lane, fractions[lane])
            fractions[lane] = prev * (1.0 - alpha) + fractions[lane] * alpha

        # Update current fractions.
        self._current_fractions = dict(fractions)

        # Update stats with final allocation.
        for lane in stats:
            stats[lane].allocation = fractions.get(lane, 0.0)

        # Dollar allocations.
        allocations = {lane: bankroll * frac for lane, frac in fractions.items()}

        return AllocationResult(
            total_bankroll=bankroll,
            allocations=allocations,
            fractions=fractions,
            stats=stats,
        )

    def _compute_lane_stats(self, lane: str) -> LaneStats:
        """Compute statistics for a lane."""
        trades = self._trades.get(lane, [])
        n = len(trades)
        if n == 0:
            return LaneStats(
                lane=lane, trade_count=0, win_rate=0.0,
                mean_pnl=0.0, std_pnl=0.0, sharpe=0.0,
                score=0.0, allocation=0.0,
            )

        win_rate = sum(1 for t in trades if t > 0) / n
        mean_pnl = sum(trades) / n

        if n < 2:
            std_pnl = 0.0
        else:
            variance = sum((t - mean_pnl) ** 2 for t in trades) / (n - 1)
            std_pnl = math.sqrt(variance)

        sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0.0

        cfg = self._config
        score = (
            cfg.sharpe_weight * max(0.0, sharpe)
            + cfg.win_rate_weight * win_rate
        )

        return LaneStats(
            lane=lane,
            trade_count=n,
            win_rate=win_rate,
            mean_pnl=mean_pnl,
            std_pnl=std_pnl,
            sharpe=sharpe,
            score=score,
            allocation=0.0,
        )

    def clear(self) -> None:
        """Clear all state."""
        self._trades.clear()
        self._current_fractions.clear()
