"""Tests for Phase 3D: Maker-first live execution policy."""

from __future__ import annotations

import pytest

from arb_bot.framework.execution_policy import (
    ExecutionDecision,
    ExecutionMode,
    ExecutionPolicy,
    ExecutionPolicyConfig,
    FallbackReason,
    MakerFallbackTracker,
    MarketState,
    OrderTypeDecision,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state(
    venue: str = "kalshi",
    bid: float = 0.50,
    ask: float = 0.55,
    size: float = 50.0,
    fill_prob: float = 0.6,
    adverse: float = 0.1,
    maker_fee: float = -0.02,
    taker_fee: float = 0.07,
    edge: float = 0.03,
) -> MarketState:
    return MarketState(
        venue=venue,
        market_id="M1",
        side="yes",
        bid_price=bid,
        ask_price=ask,
        available_size=size,
        maker_fill_probability=fill_prob,
        adverse_selection_score=adverse,
        maker_fee_per_contract=maker_fee,
        taker_fee_per_contract=taker_fee,
        edge_per_contract=edge,
    )


# ---------------------------------------------------------------------------
# MarketState
# ---------------------------------------------------------------------------


class TestMarketState:
    def test_spread(self) -> None:
        s = _state(bid=0.50, ask=0.55)
        assert s.spread == pytest.approx(0.05)

    def test_spread_zero(self) -> None:
        s = _state(bid=0.55, ask=0.55)
        assert s.spread == pytest.approx(0.0)

    def test_maker_cost(self) -> None:
        s = _state(bid=0.50, maker_fee=-0.02)
        assert s.maker_cost == pytest.approx(0.48)

    def test_taker_cost(self) -> None:
        s = _state(ask=0.55, taker_fee=0.07)
        assert s.taker_cost == pytest.approx(0.62)

    def test_maker_edge_advantage(self) -> None:
        s = _state(bid=0.50, ask=0.55, maker_fee=-0.02, taker_fee=0.07)
        # Taker: 0.62, Maker: 0.48 → advantage = 0.14
        assert s.maker_edge_advantage == pytest.approx(0.14)


# ---------------------------------------------------------------------------
# ExecutionPolicyConfig
# ---------------------------------------------------------------------------


class TestExecutionPolicyConfig:
    def test_defaults(self) -> None:
        cfg = ExecutionPolicyConfig()
        assert cfg.mode == ExecutionMode.MAKER_FIRST
        assert cfg.min_maker_fill_probability == 0.3
        assert cfg.maker_timeout_seconds == 5.0
        assert cfg.max_maker_attempts == 2


# ---------------------------------------------------------------------------
# Taker-only mode
# ---------------------------------------------------------------------------


class TestTakerOnlyMode:
    def test_always_taker(self) -> None:
        policy = ExecutionPolicy(ExecutionPolicyConfig(
            mode=ExecutionMode.TAKER_ONLY,
        ))
        decision = policy.decide(_state())
        assert decision.is_taker
        assert decision.fallback_reason == FallbackReason.MODE_TAKER_ONLY
        assert decision.fallback_after_seconds == 0.0

    def test_taker_price_is_ask(self) -> None:
        policy = ExecutionPolicy(ExecutionPolicyConfig(
            mode=ExecutionMode.TAKER_ONLY,
        ))
        decision = policy.decide(_state(ask=0.60))
        assert decision.price == pytest.approx(0.60)


# ---------------------------------------------------------------------------
# Maker-first mode — eligible
# ---------------------------------------------------------------------------


class TestMakerFirstEligible:
    def test_maker_when_eligible(self) -> None:
        policy = ExecutionPolicy(ExecutionPolicyConfig(
            mode=ExecutionMode.MAKER_FIRST,
            min_maker_fill_probability=0.3,
            min_spread_for_maker=0.02,
            min_maker_edge_advantage=0.005,
        ))
        decision = policy.decide(_state(
            bid=0.50, ask=0.55, fill_prob=0.6, adverse=0.1,
        ))
        assert decision.is_maker
        assert decision.price == pytest.approx(0.50)
        assert decision.fallback_after_seconds == 5.0

    def test_maker_attempts(self) -> None:
        policy = ExecutionPolicy(ExecutionPolicyConfig(
            max_maker_attempts=3,
        ))
        decision = policy.decide(_state())
        assert decision.max_maker_attempts == 3


# ---------------------------------------------------------------------------
# Maker-first mode — ineligible fallbacks
# ---------------------------------------------------------------------------


class TestMakerFirstFallback:
    def test_spread_too_narrow(self) -> None:
        policy = ExecutionPolicy(ExecutionPolicyConfig(
            min_spread_for_maker=0.05,
        ))
        decision = policy.decide(_state(bid=0.53, ask=0.55))
        assert decision.is_taker
        assert decision.fallback_reason == FallbackReason.SPREAD_TOO_NARROW

    def test_low_fill_probability(self) -> None:
        policy = ExecutionPolicy(ExecutionPolicyConfig(
            min_maker_fill_probability=0.5,
        ))
        decision = policy.decide(_state(fill_prob=0.2))
        assert decision.is_taker
        assert decision.fallback_reason == FallbackReason.LOW_FILL_PROBABILITY

    def test_high_adverse_selection(self) -> None:
        policy = ExecutionPolicy(ExecutionPolicyConfig(
            max_adverse_selection_score=0.3,
        ))
        decision = policy.decide(_state(adverse=0.8))
        assert decision.is_taker
        assert decision.fallback_reason == FallbackReason.HIGH_ADVERSE_SELECTION

    def test_insufficient_edge(self) -> None:
        policy = ExecutionPolicy(ExecutionPolicyConfig(
            min_maker_edge_advantage=0.20,
        ))
        decision = policy.decide(_state(
            bid=0.50, ask=0.55, maker_fee=-0.02, taker_fee=0.07,
        ))
        # Advantage = 0.14 < 0.20
        assert decision.is_taker
        assert decision.fallback_reason == FallbackReason.INSUFFICIENT_EDGE


# ---------------------------------------------------------------------------
# Maker-only mode
# ---------------------------------------------------------------------------


class TestMakerOnlyMode:
    def test_maker_when_eligible(self) -> None:
        policy = ExecutionPolicy(ExecutionPolicyConfig(
            mode=ExecutionMode.MAKER_ONLY,
        ))
        decision = policy.decide(_state())
        assert decision.is_maker

    def test_maker_even_when_ineligible(self) -> None:
        policy = ExecutionPolicy(ExecutionPolicyConfig(
            mode=ExecutionMode.MAKER_ONLY,
            min_maker_fill_probability=0.9,
        ))
        decision = policy.decide(_state(fill_prob=0.1))
        # Still returns maker, but flags the reason.
        assert decision.is_maker
        assert decision.fallback_reason == FallbackReason.LOW_FILL_PROBABILITY


# ---------------------------------------------------------------------------
# ExecutionDecision properties
# ---------------------------------------------------------------------------


class TestDecisionProperties:
    def test_is_maker(self) -> None:
        decision = ExecutionDecision(
            order_type=OrderTypeDecision.MAKER,
            price=0.50, fallback_after_seconds=5.0,
            fallback_reason=None, maker_edge_advantage=0.10,
            maker_fill_probability=0.6, max_maker_attempts=2,
        )
        assert decision.is_maker is True
        assert decision.is_taker is False

    def test_is_taker(self) -> None:
        decision = ExecutionDecision(
            order_type=OrderTypeDecision.TAKER,
            price=0.55, fallback_after_seconds=0.0,
            fallback_reason=FallbackReason.SPREAD_TOO_NARROW,
            maker_edge_advantage=0.01, maker_fill_probability=0.2,
            max_maker_attempts=0,
        )
        assert decision.is_taker is True
        assert decision.is_maker is False


# ---------------------------------------------------------------------------
# MakerFallbackTracker
# ---------------------------------------------------------------------------


class TestFallbackTracker:
    def test_initial_state(self) -> None:
        tracker = MakerFallbackTracker()
        assert tracker.attempts == 0
        assert tracker.partial_fill_seen is False
        assert tracker.can_retry() is True

    def test_start_attempt(self) -> None:
        tracker = MakerFallbackTracker(max_attempts=2)
        tracker.start_attempt(now=100.0)
        assert tracker.attempts == 1
        assert tracker.can_retry() is True

    def test_max_attempts_exhausted(self) -> None:
        tracker = MakerFallbackTracker(max_attempts=2)
        tracker.start_attempt(now=100.0)
        tracker.start_attempt(now=105.0)
        assert tracker.attempts == 2
        assert tracker.can_retry() is False

    def test_timeout_triggers_fallback(self) -> None:
        tracker = MakerFallbackTracker(timeout_seconds=5.0)
        tracker.start_attempt(now=100.0)
        assert tracker.should_fallback(now=104.0) is False
        assert tracker.should_fallback(now=105.0) is True

    def test_partial_fill_extends_timeout(self) -> None:
        tracker = MakerFallbackTracker(
            timeout_seconds=5.0, extension_seconds=3.0,
        )
        tracker.start_attempt(now=100.0)
        tracker.record_partial_fill()
        # Without extension: timeout at 105.
        # With extension: timeout at 108.
        assert tracker.should_fallback(now=106.0) is False
        assert tracker.should_fallback(now=108.0) is True

    def test_should_give_up(self) -> None:
        tracker = MakerFallbackTracker(max_attempts=1, timeout_seconds=5.0)
        tracker.start_attempt(now=100.0)
        assert tracker.should_give_up(now=104.0) is False  # Not timed out.
        assert tracker.should_give_up(now=106.0) is True  # Timed out + no retries.

    def test_not_give_up_if_retries_available(self) -> None:
        tracker = MakerFallbackTracker(max_attempts=3, timeout_seconds=5.0)
        tracker.start_attempt(now=100.0)
        assert tracker.should_give_up(now=106.0) is False  # Has retries.

    def test_no_start_no_fallback(self) -> None:
        tracker = MakerFallbackTracker()
        assert tracker.should_fallback(now=1000.0) is False


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_full_maker_first_lifecycle(self) -> None:
        """Simulate a maker-first attempt with fallback."""
        policy = ExecutionPolicy(ExecutionPolicyConfig(
            mode=ExecutionMode.MAKER_FIRST,
            min_maker_fill_probability=0.3,
            min_spread_for_maker=0.02,
            maker_timeout_seconds=5.0,
            max_maker_attempts=2,
        ))

        state = _state(
            bid=0.50, ask=0.55,
            fill_prob=0.6, adverse=0.1,
            maker_fee=-0.02, taker_fee=0.07,
        )

        # Policy decides maker.
        decision = policy.decide(state)
        assert decision.is_maker
        assert decision.price == pytest.approx(0.50)

        # Tracker manages the lifecycle.
        tracker = MakerFallbackTracker(
            max_attempts=decision.max_maker_attempts,
            timeout_seconds=decision.fallback_after_seconds,
        )

        # Attempt 1: starts at t=100, times out at t=105.
        tracker.start_attempt(now=100.0)
        assert tracker.should_fallback(now=104.0) is False
        assert tracker.should_fallback(now=105.0) is True
        assert tracker.can_retry() is True

        # Attempt 2: starts at t=106, partial fill at t=107.
        tracker.start_attempt(now=106.0)
        tracker.record_partial_fill()
        assert tracker.should_fallback(now=110.0) is False  # Extended.
        assert tracker.should_fallback(now=114.0) is True
        assert tracker.should_give_up(now=114.0) is True  # No more retries.

    def test_config_property(self) -> None:
        cfg = ExecutionPolicyConfig(mode=ExecutionMode.TAKER_ONLY)
        policy = ExecutionPolicy(cfg)
        assert policy.config.mode == ExecutionMode.TAKER_ONLY
