"""Execution-cost completeness (Phase 3C).

Complete fee and friction accounting per venue and order type.
Supports tiered fee schedules, maker/taker distinction, settlement
fees, and reconciliation against actual venue statements.

Usage::

    model = FeeModel(config)
    cost = model.estimate("kalshi", OrderType.TAKER, price=0.55, contracts=10)
    model.record_actual("kalshi", OrderType.TAKER, 10, actual_fee=0.07)
    report = model.reconciliation_report()
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class OrderType(Enum):
    TAKER = "taker"
    MAKER = "maker"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VenueFeeSchedule:
    """Fee schedule for a single venue.

    Parameters
    ----------
    venue:
        Venue identifier (e.g. "kalshi", "polymarket").
    taker_fee_per_contract:
        Flat taker fee per contract. Default 0.0.
    maker_fee_per_contract:
        Flat maker fee per contract (often negative = rebate).
        Default 0.0.
    taker_fee_rate:
        Proportional taker fee as fraction of price. Default 0.0.
    maker_fee_rate:
        Proportional maker fee as fraction of price. Default 0.0.
    settlement_fee_per_contract:
        Fee charged on settlement/expiry. Default 0.0.
    min_fee_per_order:
        Minimum fee floor per order. Default 0.0.
    max_fee_per_order:
        Maximum fee cap per order. 0 = uncapped. Default 0.0.
    """

    venue: str
    taker_fee_per_contract: float = 0.0
    maker_fee_per_contract: float = 0.0
    taker_fee_rate: float = 0.0
    maker_fee_rate: float = 0.0
    settlement_fee_per_contract: float = 0.0
    min_fee_per_order: float = 0.0
    max_fee_per_order: float = 0.0


@dataclass(frozen=True)
class FeeModelConfig:
    """Configuration for the fee model.

    Parameters
    ----------
    venues:
        Fee schedules per venue. Default empty.
    default_taker_fee:
        Fallback taker fee for unknown venues. Default 0.0.
    reconciliation_tolerance:
        Acceptable absolute error between estimated and actual fees
        before flagging a discrepancy. Default 0.005.
    """

    venues: tuple[VenueFeeSchedule, ...] = ()
    default_taker_fee: float = 0.0
    reconciliation_tolerance: float = 0.005


# ---------------------------------------------------------------------------
# Fee estimate result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeeEstimate:
    """Estimated fees for an order."""

    venue: str
    order_type: OrderType
    contracts: int
    per_contract_fee: float
    total_fee: float
    settlement_fee: float
    breakdown: Dict[str, float]


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeeReconciliationEntry:
    """A single estimated vs actual fee comparison."""

    venue: str
    order_type: OrderType
    contracts: int
    estimated_fee: float
    actual_fee: float
    error: float  # estimated - actual


@dataclass(frozen=True)
class FeeDiscrepancy:
    """A fee discrepancy that exceeds tolerance."""

    venue: str
    order_type: str
    mean_error: float
    sample_count: int
    message: str


@dataclass(frozen=True)
class FeeReconciliationReport:
    """Summary of fee estimation accuracy."""

    sample_count: int
    mean_error: float
    mae: float
    total_estimated: float
    total_actual: float
    discrepancies: tuple[FeeDiscrepancy, ...]
    by_venue: Dict[str, "VenueFeeReconciliation"]

    @property
    def is_reconciled(self) -> bool:
        return len(self.discrepancies) == 0


@dataclass(frozen=True)
class VenueFeeReconciliation:
    """Per-venue fee reconciliation stats."""

    venue: str
    sample_count: int
    mean_error: float
    mae: float
    total_estimated: float
    total_actual: float


# ---------------------------------------------------------------------------
# Fee model
# ---------------------------------------------------------------------------


class FeeModel:
    """Complete fee and friction model with reconciliation.

    Computes per-order fees using venue-specific schedules and
    tracks estimated vs actual for reconciliation.
    """

    def __init__(self, config: FeeModelConfig | None = None) -> None:
        self._config = config or FeeModelConfig()
        self._schedules: Dict[str, VenueFeeSchedule] = {
            s.venue: s for s in self._config.venues
        }
        self._reconciliation_entries: list[FeeReconciliationEntry] = []

    @property
    def config(self) -> FeeModelConfig:
        return self._config

    def get_schedule(self, venue: str) -> VenueFeeSchedule | None:
        return self._schedules.get(venue)

    def estimate(
        self,
        venue: str,
        order_type: OrderType,
        contracts: int,
        price: float = 0.0,
    ) -> FeeEstimate:
        """Estimate fees for a proposed order.

        Parameters
        ----------
        venue:
            Venue identifier.
        order_type:
            TAKER or MAKER.
        contracts:
            Number of contracts.
        price:
            Per-contract price (used for proportional fees).
        """
        schedule = self._schedules.get(venue)
        breakdown: Dict[str, float] = {}

        if schedule is None:
            # Fallback to default.
            per_contract = self._config.default_taker_fee if order_type == OrderType.TAKER else 0.0
            total = per_contract * contracts
            breakdown["flat"] = total
            return FeeEstimate(
                venue=venue,
                order_type=order_type,
                contracts=contracts,
                per_contract_fee=per_contract,
                total_fee=total,
                settlement_fee=0.0,
                breakdown=breakdown,
            )

        # Flat fee component.
        if order_type == OrderType.TAKER:
            flat_per = schedule.taker_fee_per_contract
        else:
            flat_per = schedule.maker_fee_per_contract
        flat_total = flat_per * contracts
        breakdown["flat"] = flat_total

        # Proportional fee component.
        if order_type == OrderType.TAKER:
            rate = schedule.taker_fee_rate
        else:
            rate = schedule.maker_fee_rate
        prop_total = rate * price * contracts
        breakdown["proportional"] = prop_total

        # Raw total before min/max.
        raw_total = flat_total + prop_total

        # Apply min/max caps.
        if schedule.min_fee_per_order > 0 and raw_total < schedule.min_fee_per_order:
            raw_total = schedule.min_fee_per_order
            breakdown["min_floor_applied"] = schedule.min_fee_per_order
        if schedule.max_fee_per_order > 0 and raw_total > schedule.max_fee_per_order:
            raw_total = schedule.max_fee_per_order
            breakdown["max_cap_applied"] = schedule.max_fee_per_order

        # Settlement fee (separate from trading fee).
        settlement = schedule.settlement_fee_per_contract * contracts
        breakdown["settlement"] = settlement

        per_contract = raw_total / max(1, contracts)

        return FeeEstimate(
            venue=venue,
            order_type=order_type,
            contracts=contracts,
            per_contract_fee=per_contract,
            total_fee=raw_total,
            settlement_fee=settlement,
            breakdown=breakdown,
        )

    def total_cost(
        self,
        venue: str,
        order_type: OrderType,
        contracts: int,
        price: float,
    ) -> float:
        """Convenience: price + fees per contract."""
        est = self.estimate(venue, order_type, contracts, price)
        return price + est.per_contract_fee

    def record_actual(
        self,
        venue: str,
        order_type: OrderType,
        contracts: int,
        actual_fee: float,
        price: float = 0.0,
    ) -> None:
        """Record an actual fee for reconciliation.

        The estimated fee is computed internally for comparison.
        """
        est = self.estimate(venue, order_type, contracts, price)
        entry = FeeReconciliationEntry(
            venue=venue,
            order_type=order_type,
            contracts=contracts,
            estimated_fee=est.total_fee,
            actual_fee=actual_fee,
            error=est.total_fee - actual_fee,
        )
        self._reconciliation_entries.append(entry)

    def reconciliation_report(self, min_samples: int = 1) -> FeeReconciliationReport | None:
        """Compute reconciliation report.

        Returns None if fewer than min_samples entries exist.
        """
        entries = self._reconciliation_entries
        if len(entries) < min_samples:
            return None

        errors = [e.error for e in entries]
        mean_error = statistics.mean(errors)
        mae = statistics.mean(abs(e) for e in errors)
        total_est = sum(e.estimated_fee for e in entries)
        total_act = sum(e.actual_fee for e in entries)

        # Group by venue.
        venue_groups: Dict[str, list[FeeReconciliationEntry]] = {}
        for e in entries:
            venue_groups.setdefault(e.venue, []).append(e)

        by_venue: Dict[str, VenueFeeReconciliation] = {}
        discrepancies: list[FeeDiscrepancy] = []
        tolerance = self._config.reconciliation_tolerance

        for venue, group in venue_groups.items():
            v_errors = [e.error for e in group]
            v_mean = statistics.mean(v_errors)
            v_mae = statistics.mean(abs(e) for e in v_errors)
            v_est = sum(e.estimated_fee for e in group)
            v_act = sum(e.actual_fee for e in group)

            by_venue[venue] = VenueFeeReconciliation(
                venue=venue,
                sample_count=len(group),
                mean_error=v_mean,
                mae=v_mae,
                total_estimated=v_est,
                total_actual=v_act,
            )

            if abs(v_mean) > tolerance:
                direction = "over" if v_mean > 0 else "under"
                discrepancies.append(FeeDiscrepancy(
                    venue=venue,
                    order_type="all",
                    mean_error=v_mean,
                    sample_count=len(group),
                    message=f"{venue} fees {direction}-estimated by {abs(v_mean):.4f} avg",
                ))

        return FeeReconciliationReport(
            sample_count=len(entries),
            mean_error=mean_error,
            mae=mae,
            total_estimated=total_est,
            total_actual=total_act,
            discrepancies=tuple(discrepancies),
            by_venue=by_venue,
        )

    def clear_reconciliation(self) -> None:
        """Clear all reconciliation entries."""
        self._reconciliation_entries.clear()

    @property
    def reconciliation_count(self) -> int:
        return len(self._reconciliation_entries)
