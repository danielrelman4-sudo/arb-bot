"""Maker-first live execution policy (Phase 3D).

Decides whether to attempt passive (maker) execution first, with
automatic fallback to aggressive (taker) execution when the maker
attempt times out or fill probability is too low.

The decision is based on the execution model's fill/cost estimates
and the fee model's maker vs taker cost comparison.

Usage::

    policy = ExecutionPolicy(config)
    decision = policy.decide(market_state)
    # decision.order_type == OrderType.MAKER or OrderType.TAKER
    # decision.fallback_after_seconds == 5.0 (for maker)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ExecutionMode(Enum):
    """Overall execution mode."""
    MAKER_FIRST = "maker_first"
    TAKER_ONLY = "taker_only"
    MAKER_ONLY = "maker_only"


class FallbackReason(Enum):
    """Why the policy chose taker over maker."""
    LOW_FILL_PROBABILITY = "low_fill_probability"
    HIGH_ADVERSE_SELECTION = "high_adverse_selection"
    SPREAD_TOO_NARROW = "spread_too_narrow"
    INSUFFICIENT_EDGE = "insufficient_edge"
    MODE_TAKER_ONLY = "mode_taker_only"
    TIMEOUT = "timeout"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionPolicyConfig:
    """Configuration for the execution policy.

    Parameters
    ----------
    mode:
        Overall execution mode. Default MAKER_FIRST.
    min_maker_fill_probability:
        Minimum fill probability to attempt maker order.
        Below this, skip to taker. Default 0.3.
    min_maker_edge_advantage:
        Minimum edge improvement (maker vs taker) in price units
        to justify waiting for maker fill. Default 0.005.
    max_adverse_selection_score:
        Maximum adverse selection score before giving up on maker.
        Default 0.5.
    min_spread_for_maker:
        Minimum bid-ask spread to attempt maker entry.
        Below this, market is too tight for passive orders.
        Default 0.02.
    maker_timeout_seconds:
        How long to wait for maker fill before falling back
        to taker. Default 5.0.
    maker_timeout_extension_seconds:
        Extra time for maker if partial fill is observed.
        Default 3.0.
    max_maker_attempts:
        Maximum number of maker attempts before giving up.
        Default 2.
    """

    mode: ExecutionMode = ExecutionMode.MAKER_FIRST
    min_maker_fill_probability: float = 0.3
    min_maker_edge_advantage: float = 0.005
    max_adverse_selection_score: float = 0.5
    min_spread_for_maker: float = 0.02
    maker_timeout_seconds: float = 5.0
    maker_timeout_extension_seconds: float = 3.0
    max_maker_attempts: int = 2


# ---------------------------------------------------------------------------
# Market state input
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketState:
    """Market conditions at decision time."""

    venue: str
    market_id: str
    side: str
    bid_price: float
    ask_price: float
    available_size: float
    maker_fill_probability: float = 0.5
    adverse_selection_score: float = 0.0
    maker_fee_per_contract: float = 0.0
    taker_fee_per_contract: float = 0.0
    edge_per_contract: float = 0.0

    @property
    def spread(self) -> float:
        return max(0.0, self.ask_price - self.bid_price)

    @property
    def maker_cost(self) -> float:
        """Cost per contract with maker pricing (buy at bid + fee)."""
        return self.bid_price + self.maker_fee_per_contract

    @property
    def taker_cost(self) -> float:
        """Cost per contract with taker pricing (buy at ask + fee)."""
        return self.ask_price + self.taker_fee_per_contract

    @property
    def maker_edge_advantage(self) -> float:
        """How much cheaper maker is vs taker per contract."""
        return self.taker_cost - self.maker_cost


# ---------------------------------------------------------------------------
# Decision result
# ---------------------------------------------------------------------------


class OrderTypeDecision(Enum):
    MAKER = "maker"
    TAKER = "taker"


@dataclass(frozen=True)
class ExecutionDecision:
    """Result of the execution policy decision."""

    order_type: OrderTypeDecision
    price: float
    fallback_after_seconds: float
    fallback_reason: FallbackReason | None
    maker_edge_advantage: float
    maker_fill_probability: float
    max_maker_attempts: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_maker(self) -> bool:
        return self.order_type == OrderTypeDecision.MAKER

    @property
    def is_taker(self) -> bool:
        return self.order_type == OrderTypeDecision.TAKER


# ---------------------------------------------------------------------------
# Execution policy
# ---------------------------------------------------------------------------


class ExecutionPolicy:
    """Decides maker vs taker execution strategy.

    Evaluates market conditions against configurable thresholds
    to determine whether passive (maker) execution has positive EV
    compared to immediate aggressive (taker) execution.
    """

    def __init__(self, config: ExecutionPolicyConfig | None = None) -> None:
        self._config = config or ExecutionPolicyConfig()

    @property
    def config(self) -> ExecutionPolicyConfig:
        return self._config

    def decide(self, state: MarketState) -> ExecutionDecision:
        """Decide execution strategy for a given market state.

        Returns an ExecutionDecision with the recommended order type,
        price, and fallback parameters.
        """
        cfg = self._config

        # Taker-only mode: always taker.
        if cfg.mode == ExecutionMode.TAKER_ONLY:
            return self._taker_decision(
                state, FallbackReason.MODE_TAKER_ONLY,
            )

        # Check maker eligibility.
        reason = self._check_maker_eligibility(state)
        if reason is not None:
            if cfg.mode == ExecutionMode.MAKER_ONLY:
                # Even in maker-only mode, if ineligible we still return maker
                # but flag the reason.
                return self._maker_decision(state, ineligible_reason=reason)
            return self._taker_decision(state, reason)

        # Maker-only mode: always maker if eligible.
        if cfg.mode == ExecutionMode.MAKER_ONLY:
            return self._maker_decision(state)

        # Maker-first mode: use maker with taker fallback.
        return self._maker_decision(state)

    def _check_maker_eligibility(self, state: MarketState) -> FallbackReason | None:
        """Check if maker execution is viable.

        Returns a FallbackReason if maker should be skipped, else None.
        """
        cfg = self._config

        # Spread too narrow for passive entry.
        if state.spread < cfg.min_spread_for_maker:
            return FallbackReason.SPREAD_TOO_NARROW

        # Fill probability too low.
        if state.maker_fill_probability < cfg.min_maker_fill_probability:
            return FallbackReason.LOW_FILL_PROBABILITY

        # High adverse selection.
        if state.adverse_selection_score > cfg.max_adverse_selection_score:
            return FallbackReason.HIGH_ADVERSE_SELECTION

        # Maker edge advantage too small.
        if state.maker_edge_advantage < cfg.min_maker_edge_advantage:
            return FallbackReason.INSUFFICIENT_EDGE

        return None

    def _maker_decision(
        self,
        state: MarketState,
        ineligible_reason: FallbackReason | None = None,
    ) -> ExecutionDecision:
        cfg = self._config
        # Maker posts at bid (or slightly inside).
        price = state.bid_price

        return ExecutionDecision(
            order_type=OrderTypeDecision.MAKER,
            price=price,
            fallback_after_seconds=cfg.maker_timeout_seconds,
            fallback_reason=ineligible_reason,
            maker_edge_advantage=state.maker_edge_advantage,
            maker_fill_probability=state.maker_fill_probability,
            max_maker_attempts=cfg.max_maker_attempts,
        )

    def _taker_decision(
        self,
        state: MarketState,
        reason: FallbackReason,
    ) -> ExecutionDecision:
        return ExecutionDecision(
            order_type=OrderTypeDecision.TAKER,
            price=state.ask_price,
            fallback_after_seconds=0.0,
            fallback_reason=reason,
            maker_edge_advantage=state.maker_edge_advantage,
            maker_fill_probability=state.maker_fill_probability,
            max_maker_attempts=0,
        )


# ---------------------------------------------------------------------------
# Fallback tracker
# ---------------------------------------------------------------------------


class MakerFallbackTracker:
    """Tracks maker attempt outcomes and decides when to fall back.

    Used at execution time to manage the maker-then-taker lifecycle.
    """

    def __init__(
        self,
        max_attempts: int = 2,
        timeout_seconds: float = 5.0,
        extension_seconds: float = 3.0,
    ) -> None:
        self._max_attempts = max_attempts
        self._timeout_seconds = timeout_seconds
        self._extension_seconds = extension_seconds
        self._attempts: int = 0
        self._partial_fill_seen: bool = False
        self._start_time: float | None = None

    @property
    def attempts(self) -> int:
        return self._attempts

    @property
    def max_attempts(self) -> int:
        return self._max_attempts

    @property
    def partial_fill_seen(self) -> bool:
        return self._partial_fill_seen

    def start_attempt(self, now: float) -> None:
        """Start a new maker attempt."""
        self._attempts += 1
        self._start_time = now
        self._partial_fill_seen = False

    def record_partial_fill(self) -> None:
        """Record that a partial fill was observed."""
        self._partial_fill_seen = True

    def should_fallback(self, now: float) -> bool:
        """Check if we should give up on maker and switch to taker."""
        if self._start_time is None:
            return False

        elapsed = now - self._start_time
        timeout = self._timeout_seconds
        if self._partial_fill_seen:
            timeout += self._extension_seconds

        if elapsed >= timeout:
            return True

        return False

    def can_retry(self) -> bool:
        """Check if another maker attempt is allowed."""
        return self._attempts < self._max_attempts

    def should_give_up(self, now: float) -> bool:
        """Check if we should give up on maker entirely (all attempts exhausted + timed out)."""
        if not self.should_fallback(now):
            return False
        return not self.can_retry()
