"""Poll budget allocator with fairness (Phase 5C).

Allocates a per-cycle API call budget across lanes and venues.
Prevents starvation by guaranteeing minimum allocations, enforces
ceilings per lane/venue, and supports burst allowances.

Usage::

    alloc = PollBudgetAllocator(config)
    alloc.register_lane("cross_venue", weight=2.0)
    alloc.register_lane("intra_venue", weight=1.0)
    budget = alloc.allocate(total_budget=100)
    # budget["cross_venue"] → 60 (approx, after min/max constraints)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PollBudgetConfig:
    """Configuration for poll budget allocator.

    Parameters
    ----------
    default_weight:
        Default weight for lanes. Default 1.0.
    min_allocation:
        Minimum polls guaranteed per lane per cycle. Default 2.
    max_allocation_fraction:
        Maximum fraction of total budget any single lane can
        receive. Default 0.50.
    burst_multiplier:
        Burst allowance multiplier (applies when lane has unspent
        budget from previous cycle). Default 1.5.
    burst_decay:
        How quickly burst credit decays (0-1). 0 = no decay,
        1 = full decay each cycle. Default 0.5.
    starvation_boost:
        Extra weight multiplier for lanes that received zero polls
        last cycle. Default 2.0.
    """

    default_weight: float = 1.0
    min_allocation: int = 2
    max_allocation_fraction: float = 0.50
    burst_multiplier: float = 1.5
    burst_decay: float = 0.5
    starvation_boost: float = 2.0


# ---------------------------------------------------------------------------
# Lane state
# ---------------------------------------------------------------------------


@dataclass
class LaneBudgetState:
    """Budget state for a single lane."""

    lane: str
    weight: float
    last_allocation: int = 0
    last_used: int = 0
    burst_credit: float = 0.0
    total_allocated: int = 0
    total_used: int = 0
    starved_cycles: int = 0


# ---------------------------------------------------------------------------
# Allocation result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BudgetAllocation:
    """Result of budget allocation for a single cycle."""

    allocations: Dict[str, int]
    total_budget: int
    total_allocated: int
    starvation_boosted: Tuple[str, ...]
    burst_boosted: Tuple[str, ...]


# ---------------------------------------------------------------------------
# Allocator
# ---------------------------------------------------------------------------


class PollBudgetAllocator:
    """Allocates per-cycle poll budgets across lanes with fairness.

    Weighted proportional allocation with minimum guarantees,
    maximum ceilings, starvation prevention, and burst credits
    for lanes that under-used their previous allocation.
    """

    def __init__(self, config: PollBudgetConfig | None = None) -> None:
        self._config = config or PollBudgetConfig()
        self._lanes: Dict[str, LaneBudgetState] = {}

    @property
    def config(self) -> PollBudgetConfig:
        return self._config

    def register_lane(self, lane: str, weight: float | None = None) -> None:
        """Register a lane for budget allocation.

        Parameters
        ----------
        lane:
            Lane identifier.
        weight:
            Allocation weight. If None, uses default_weight.
        """
        if lane not in self._lanes:
            self._lanes[lane] = LaneBudgetState(
                lane=lane,
                weight=weight or self._config.default_weight,
            )
        else:
            if weight is not None:
                self._lanes[lane].weight = weight

    def unregister_lane(self, lane: str) -> None:
        """Remove a lane from budget allocation."""
        self._lanes.pop(lane, None)

    def record_usage(self, lane: str, polls_used: int) -> None:
        """Record actual polls used by a lane in the current cycle.

        Parameters
        ----------
        lane:
            Lane identifier.
        polls_used:
            Number of polls actually used.
        """
        state = self._lanes.get(lane)
        if state is not None:
            state.last_used = polls_used
            state.total_used += polls_used

    def allocate(self, total_budget: int) -> BudgetAllocation:
        """Allocate polls across registered lanes.

        Parameters
        ----------
        total_budget:
            Total API call budget for this cycle.
        """
        cfg = self._config
        n = len(self._lanes)

        if n == 0:
            return BudgetAllocation(
                allocations={},
                total_budget=total_budget,
                total_allocated=0,
                starvation_boosted=(),
                burst_boosted=(),
            )

        # Step 1: Compute effective weights (with starvation boost).
        starvation_boosted: List[str] = []
        burst_boosted: List[str] = []
        effective_weights: Dict[str, float] = {}

        for lane, state in self._lanes.items():
            w = state.weight

            # Starvation boost: if lane got zero polls last cycle.
            if state.last_used == 0 and state.total_allocated > 0:
                w *= cfg.starvation_boost
                state.starved_cycles += 1
                starvation_boosted.append(lane)

            # Burst credit from unused allocation.
            if state.last_allocation > 0:
                unused = max(0, state.last_allocation - state.last_used)
                state.burst_credit = (
                    state.burst_credit * (1.0 - cfg.burst_decay)
                    + unused * cfg.burst_decay
                )

            effective_weights[lane] = w

        total_weight = sum(effective_weights.values())
        if total_weight <= 0:
            total_weight = 1.0

        # Step 2: Proportional allocation.
        max_per_lane = max(1, int(total_budget * cfg.max_allocation_fraction))
        allocations: Dict[str, int] = {}
        remaining = total_budget

        for lane, state in self._lanes.items():
            ew = effective_weights[lane]
            raw = (ew / total_weight) * total_budget

            # Apply burst bonus.
            if state.burst_credit > 0:
                burst_bonus = state.burst_credit * (cfg.burst_multiplier - 1.0)
                raw += burst_bonus
                if burst_bonus > 0.5:
                    burst_boosted.append(lane)

            # Apply min/max constraints.
            alloc = max(cfg.min_allocation, int(round(raw)))
            alloc = min(alloc, max_per_lane)
            allocations[lane] = alloc

        # Step 3: Adjust to fit within total budget.
        total_alloc = sum(allocations.values())
        if total_alloc > total_budget:
            # Scale down proportionally, keeping minimums.
            excess = total_alloc - total_budget
            flexible = {
                lane: alloc - cfg.min_allocation
                for lane, alloc in allocations.items()
                if alloc > cfg.min_allocation
            }
            flex_total = sum(flexible.values())
            if flex_total > 0:
                for lane, flex in flexible.items():
                    reduction = int(round(excess * (flex / flex_total)))
                    allocations[lane] = max(
                        cfg.min_allocation,
                        allocations[lane] - reduction,
                    )

        # Step 4: Update state.
        for lane, alloc in allocations.items():
            state = self._lanes[lane]
            state.last_allocation = alloc
            state.total_allocated += alloc

        total_allocated = sum(allocations.values())

        return BudgetAllocation(
            allocations=allocations,
            total_budget=total_budget,
            total_allocated=total_allocated,
            starvation_boosted=tuple(starvation_boosted),
            burst_boosted=tuple(burst_boosted),
        )

    def get_state(self, lane: str) -> LaneBudgetState | None:
        """Get budget state for a lane."""
        return self._lanes.get(lane)

    def registered_lanes(self) -> List[str]:
        """List registered lanes."""
        return list(self._lanes.keys())

    def clear(self) -> None:
        """Clear all lanes."""
        self._lanes.clear()
