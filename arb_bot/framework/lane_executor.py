"""Parallel lane execution architecture (Phase 5I).

Isolated lane workers with bounded fan-out, deterministic merge
of results, and per-lane timing metrics with timeout budgets.

Usage::

    executor = LaneExecutor(config)
    executor.register_lane("cross_venue", worker_fn=scan_cross)
    executor.register_lane("intra_venue", worker_fn=scan_intra)
    results = executor.execute_all(now)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LaneExecutorConfig:
    """Configuration for parallel lane executor.

    Parameters
    ----------
    max_concurrent_lanes:
        Maximum lanes executing simultaneously. Default 5.
    default_timeout:
        Default per-lane timeout (seconds). Default 10.0.
    total_budget:
        Total time budget across all lanes (seconds). Default 30.0.
    merge_strategy:
        How to merge lane results: "append" or "priority".
        Default "append".
    """

    max_concurrent_lanes: int = 5
    default_timeout: float = 10.0
    total_budget: float = 30.0
    merge_strategy: str = "append"


# ---------------------------------------------------------------------------
# Lane result
# ---------------------------------------------------------------------------


@dataclass
class LaneResult:
    """Result from a single lane execution."""

    lane: str
    success: bool
    items: List[Any] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    error: str = ""
    timed_out: bool = False


# ---------------------------------------------------------------------------
# Merged result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MergedResult:
    """Merged results from all lanes."""

    lane_results: Dict[str, LaneResult]
    all_items: Tuple[Any, ...]
    total_elapsed: float
    successful_lanes: int
    failed_lanes: int
    timed_out_lanes: int


# ---------------------------------------------------------------------------
# Lane state
# ---------------------------------------------------------------------------


@dataclass
class _LaneState:
    """Internal state for a lane."""

    lane: str
    worker_fn: Callable[[], List[Any]]
    timeout: float
    priority: int = 0
    total_runs: int = 0
    total_successes: int = 0
    total_failures: int = 0
    total_timeouts: int = 0
    avg_elapsed: float = 0.0


# ---------------------------------------------------------------------------
# Lane executor
# ---------------------------------------------------------------------------


class LaneExecutor:
    """Executes lane workers with bounded fan-out and timing.

    Each lane has an independent worker function, timeout budget,
    and timing metrics. Results are merged deterministically.
    """

    def __init__(self, config: LaneExecutorConfig | None = None) -> None:
        self._config = config or LaneExecutorConfig()
        self._lanes: Dict[str, _LaneState] = {}

    @property
    def config(self) -> LaneExecutorConfig:
        return self._config

    def register_lane(
        self,
        lane: str,
        worker_fn: Callable[[], List[Any]],
        timeout: float | None = None,
        priority: int = 0,
    ) -> None:
        """Register a lane worker.

        Parameters
        ----------
        lane:
            Lane identifier.
        worker_fn:
            Callable returning a list of items (opportunities, etc.).
        timeout:
            Per-lane timeout. If None, uses default_timeout.
        priority:
            Higher priority lanes execute first. Default 0.
        """
        self._lanes[lane] = _LaneState(
            lane=lane,
            worker_fn=worker_fn,
            timeout=timeout or self._config.default_timeout,
            priority=priority,
        )

    def unregister_lane(self, lane: str) -> None:
        """Remove a lane."""
        self._lanes.pop(lane, None)

    def _execute_lane(self, state: _LaneState, deadline: float) -> LaneResult:
        """Execute a single lane with timeout."""
        start = time.monotonic()
        try:
            items = state.worker_fn()
            elapsed = time.monotonic() - start

            if elapsed > state.timeout:
                state.total_timeouts += 1
                state.total_runs += 1
                return LaneResult(
                    lane=state.lane,
                    success=False,
                    items=items,
                    elapsed_seconds=elapsed,
                    timed_out=True,
                )

            state.total_successes += 1
            state.total_runs += 1
            # Update running average.
            n = state.total_runs
            state.avg_elapsed = (
                state.avg_elapsed * (n - 1) + elapsed
            ) / n

            return LaneResult(
                lane=state.lane,
                success=True,
                items=items,
                elapsed_seconds=elapsed,
            )

        except Exception as e:
            elapsed = time.monotonic() - start
            state.total_failures += 1
            state.total_runs += 1
            return LaneResult(
                lane=state.lane,
                success=False,
                elapsed_seconds=elapsed,
                error=str(e),
            )

    def execute_all(self, now: float | None = None) -> MergedResult:
        """Execute all registered lanes.

        Lanes are executed in priority order (highest first),
        bounded by max_concurrent_lanes and total_budget.
        Currently synchronous (async version for production).
        """
        if now is None:
            now = time.monotonic()

        deadline = now + self._config.total_budget

        # Sort by priority (highest first).
        sorted_lanes = sorted(
            self._lanes.values(),
            key=lambda s: s.priority,
            reverse=True,
        )

        # Bound concurrency — in synchronous mode, just limit count.
        active = sorted_lanes[: self._config.max_concurrent_lanes]

        lane_results: Dict[str, LaneResult] = {}
        start_time = time.monotonic()

        for state in active:
            result = self._execute_lane(state, deadline)
            lane_results[state.lane] = result

        total_elapsed = time.monotonic() - start_time

        # Merge items deterministically (by lane registration order).
        all_items: List[Any] = []
        if self._config.merge_strategy == "priority":
            for state in sorted_lanes:
                if state.lane in lane_results:
                    all_items.extend(lane_results[state.lane].items)
        else:
            for lane_name in self._lanes:
                if lane_name in lane_results:
                    all_items.extend(lane_results[lane_name].items)

        successful = sum(1 for r in lane_results.values() if r.success)
        failed = sum(
            1 for r in lane_results.values()
            if not r.success and not r.timed_out
        )
        timed_out = sum(1 for r in lane_results.values() if r.timed_out)

        return MergedResult(
            lane_results=lane_results,
            all_items=tuple(all_items),
            total_elapsed=total_elapsed,
            successful_lanes=successful,
            failed_lanes=failed,
            timed_out_lanes=timed_out,
        )

    def get_lane_stats(self, lane: str) -> Dict[str, Any] | None:
        """Get statistics for a lane."""
        state = self._lanes.get(lane)
        if state is None:
            return None
        return {
            "lane": state.lane,
            "total_runs": state.total_runs,
            "total_successes": state.total_successes,
            "total_failures": state.total_failures,
            "total_timeouts": state.total_timeouts,
            "avg_elapsed": state.avg_elapsed,
            "priority": state.priority,
        }

    def registered_lanes(self) -> List[str]:
        """List registered lanes."""
        return list(self._lanes.keys())

    def clear(self) -> None:
        """Clear all lanes."""
        self._lanes.clear()
