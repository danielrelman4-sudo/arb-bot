"""Tests for Phase 4B: Liquidity impact curve integration."""

from __future__ import annotations

import math

import pytest

from arb_bot.liquidity_impact import (
    ImpactEstimate,
    ImpactFillRecord,
    LiquidityImpactConfig,
    LiquidityImpactModel,
    VenueImpactParams,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _model(**kw) -> LiquidityImpactModel:
    return LiquidityImpactModel(LiquidityImpactConfig(**kw))


# ---------------------------------------------------------------------------
# LiquidityImpactConfig
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self) -> None:
        cfg = LiquidityImpactConfig()
        assert cfg.default_impact_coefficient == 0.10
        assert cfg.default_impact_exponent == 1.5
        assert cfg.max_depth_fraction == 0.5
        assert cfg.min_book_depth == 5
        assert cfg.learning_rate == 0.1
        assert cfg.window_size == 200

    def test_custom(self) -> None:
        cfg = LiquidityImpactConfig(default_impact_coefficient=0.2, min_book_depth=10)
        assert cfg.default_impact_coefficient == 0.2
        assert cfg.min_book_depth == 10

    def test_frozen(self) -> None:
        cfg = LiquidityImpactConfig()
        with pytest.raises(AttributeError):
            cfg.default_impact_coefficient = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Estimate impact — blocking
# ---------------------------------------------------------------------------


class TestEstimateBlocking:
    def test_insufficient_depth(self) -> None:
        m = _model(min_book_depth=5)
        est = m.estimate_impact("kalshi", contracts=5, book_depth=3)
        assert est.blocked is True
        assert est.block_reason == "insufficient_depth"
        assert est.estimated_impact == float("inf")

    def test_exceeds_max_depth_fraction(self) -> None:
        m = _model(max_depth_fraction=0.5)
        est = m.estimate_impact("kalshi", contracts=60, book_depth=100)
        assert est.blocked is True
        assert est.block_reason == "exceeds_max_depth_fraction"

    def test_exactly_at_max_depth_not_blocked(self) -> None:
        m = _model(max_depth_fraction=0.5)
        est = m.estimate_impact("kalshi", contracts=50, book_depth=100)
        assert est.blocked is False


# ---------------------------------------------------------------------------
# Estimate impact — successful
# ---------------------------------------------------------------------------


class TestEstimateSuccess:
    def test_small_order_low_impact(self) -> None:
        m = _model(default_impact_coefficient=0.10, default_impact_exponent=1.5)
        est = m.estimate_impact("kalshi", contracts=5, book_depth=100)
        assert est.blocked is False
        assert est.estimated_impact > 0.0
        assert est.depth_fraction == pytest.approx(0.05)

    def test_larger_order_higher_impact(self) -> None:
        m = _model()
        small = m.estimate_impact("kalshi", contracts=5, book_depth=100)
        large = m.estimate_impact("kalshi", contracts=30, book_depth=100)
        assert large.estimated_impact > small.estimated_impact

    def test_convex_impact_curve(self) -> None:
        # With exponent > 1, impact grows faster than linearly.
        m = _model(default_impact_exponent=2.0)
        est10 = m.estimate_impact("v", contracts=10, book_depth=100)
        est20 = m.estimate_impact("v", contracts=20, book_depth=100)
        # Impact at 2x size should be more than 2x impact at 1x size.
        assert est20.estimated_impact > 2.0 * est10.estimated_impact

    def test_linear_impact_exponent_1(self) -> None:
        m = _model(default_impact_exponent=1.0, default_impact_coefficient=0.10)
        est10 = m.estimate_impact("v", contracts=10, book_depth=100)
        est20 = m.estimate_impact("v", contracts=20, book_depth=100)
        # Linear: 2x size = 2x impact.
        assert est20.estimated_impact == pytest.approx(2.0 * est10.estimated_impact)

    def test_zero_contracts(self) -> None:
        m = _model()
        est = m.estimate_impact("kalshi", contracts=0, book_depth=100)
        assert est.estimated_impact == 0.0
        assert est.marginal_impact == 0.0

    def test_impact_fields_populated(self) -> None:
        m = _model()
        est = m.estimate_impact("kalshi", contracts=10, book_depth=100)
        assert est.venue == "kalshi"
        assert est.contracts == 10
        assert est.book_depth == 100
        assert est.coefficient == 0.10
        assert est.exponent == 1.5

    def test_marginal_impact_positive(self) -> None:
        m = _model()
        est = m.estimate_impact("kalshi", contracts=10, book_depth=100)
        assert est.marginal_impact > 0.0


# ---------------------------------------------------------------------------
# Max contracts for edge
# ---------------------------------------------------------------------------


class TestMaxContractsForEdge:
    def test_positive_edge(self) -> None:
        m = _model(default_impact_coefficient=0.10, default_impact_exponent=1.5)
        max_c = m.max_contracts_for_edge("kalshi", edge=0.03, cost=0.55, book_depth=100)
        assert max_c > 0
        # Verify impact at max_c is within edge.
        est = m.estimate_impact("kalshi", contracts=max_c, book_depth=100)
        assert est.estimated_impact <= 0.03 / 0.55 + 0.01  # Small tolerance.

    def test_zero_edge(self) -> None:
        m = _model()
        assert m.max_contracts_for_edge("k", edge=0.0, cost=0.55, book_depth=100) == 0

    def test_negative_edge(self) -> None:
        m = _model()
        assert m.max_contracts_for_edge("k", edge=-0.01, cost=0.55, book_depth=100) == 0

    def test_zero_cost(self) -> None:
        m = _model()
        assert m.max_contracts_for_edge("k", edge=0.03, cost=0.0, book_depth=100) == 0

    def test_insufficient_depth(self) -> None:
        m = _model(min_book_depth=10)
        assert m.max_contracts_for_edge("k", edge=0.03, cost=0.55, book_depth=5) == 0

    def test_capped_by_max_depth_fraction(self) -> None:
        m = _model(max_depth_fraction=0.3, default_impact_coefficient=0.001)
        # Very low impact → would allow many contracts, but capped.
        max_c = m.max_contracts_for_edge("k", edge=0.10, cost=0.50, book_depth=100)
        assert max_c <= 30  # 0.3 * 100

    def test_higher_edge_allows_more(self) -> None:
        m = _model()
        low = m.max_contracts_for_edge("k", edge=0.01, cost=0.55, book_depth=100)
        high = m.max_contracts_for_edge("k", edge=0.05, cost=0.55, book_depth=100)
        assert high >= low

    def test_deeper_book_allows_more(self) -> None:
        m = _model()
        shallow = m.max_contracts_for_edge("k", edge=0.03, cost=0.55, book_depth=50)
        deep = m.max_contracts_for_edge("k", edge=0.03, cost=0.55, book_depth=200)
        assert deep >= shallow


# ---------------------------------------------------------------------------
# Record fill and learning
# ---------------------------------------------------------------------------


class TestRecordFill:
    def test_records_fill(self) -> None:
        m = _model()
        m.record_fill("kalshi", contracts=10, book_depth=100,
                       expected_price=0.55, actual_price=0.56)
        assert m.fill_count("kalshi") == 1

    def test_multiple_fills(self) -> None:
        m = _model()
        for i in range(5):
            m.record_fill("kalshi", contracts=10, book_depth=100,
                           expected_price=0.55, actual_price=0.55 + i * 0.001)
        assert m.fill_count("kalshi") == 5

    def test_window_trimming(self) -> None:
        m = _model(window_size=5)
        for i in range(10):
            m.record_fill("kalshi", contracts=10, book_depth=100,
                           expected_price=0.55, actual_price=0.56)
        assert m.fill_count("kalshi") == 5

    def test_learning_updates_coefficient(self) -> None:
        m = _model(learning_rate=0.5)
        initial = m.get_venue_params("kalshi").coefficient
        # Record a fill with significant slippage.
        m.record_fill("kalshi", contracts=10, book_depth=100,
                       expected_price=0.55, actual_price=0.60)
        updated = m.get_venue_params("kalshi").coefficient
        # Coefficient should change from default.
        assert updated != initial

    def test_learning_smooths(self) -> None:
        m = _model(learning_rate=0.1)
        # Record many fills with consistent small slippage.
        for _ in range(50):
            m.record_fill("kalshi", contracts=10, book_depth=100,
                           expected_price=0.55, actual_price=0.551)
        params = m.get_venue_params("kalshi")
        # Coefficient should have moved toward the observed value.
        assert params.sample_count == 50

    def test_no_update_on_zero_contracts(self) -> None:
        m = _model()
        m.record_fill("kalshi", contracts=0, book_depth=100,
                       expected_price=0.55, actual_price=0.56)
        # Should not create venue params.
        assert "kalshi" not in m._venue_params

    def test_no_update_on_insufficient_depth(self) -> None:
        m = _model(min_book_depth=10)
        m.record_fill("kalshi", contracts=5, book_depth=3,
                       expected_price=0.55, actual_price=0.56)
        assert "kalshi" not in m._venue_params


# ---------------------------------------------------------------------------
# Venue params
# ---------------------------------------------------------------------------


class TestVenueParams:
    def test_default_params(self) -> None:
        m = _model()
        params = m.get_venue_params("unknown")
        assert params.coefficient == 0.10
        assert params.exponent == 1.5
        assert params.sample_count == 0

    def test_learned_params(self) -> None:
        m = _model()
        m.record_fill("kalshi", contracts=10, book_depth=100,
                       expected_price=0.55, actual_price=0.56)
        params = m.get_venue_params("kalshi")
        assert params.sample_count == 1

    def test_independent_venues(self) -> None:
        m = _model(learning_rate=0.5)
        m.record_fill("kalshi", contracts=10, book_depth=100,
                       expected_price=0.55, actual_price=0.60)
        m.record_fill("poly", contracts=10, book_depth=100,
                       expected_price=0.55, actual_price=0.551)
        k_params = m.get_venue_params("kalshi")
        p_params = m.get_venue_params("poly")
        assert k_params.coefficient != p_params.coefficient


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clears_all(self) -> None:
        m = _model()
        m.record_fill("kalshi", contracts=10, book_depth=100,
                       expected_price=0.55, actual_price=0.56)
        m.clear()
        assert m.fill_count("kalshi") == 0
        assert "kalshi" not in m._venue_params


# ---------------------------------------------------------------------------
# Config property
# ---------------------------------------------------------------------------


class TestConfigProperty:
    def test_config_accessible(self) -> None:
        cfg = LiquidityImpactConfig(min_book_depth=10)
        m = LiquidityImpactModel(cfg)
        assert m.config.min_book_depth == 10


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_full_lifecycle(self) -> None:
        """Simulate realistic impact modeling with learning."""
        m = _model(
            default_impact_coefficient=0.10,
            default_impact_exponent=1.5,
            max_depth_fraction=0.5,
            learning_rate=0.2,
        )

        # Pre-learning: estimate with defaults.
        est_pre = m.estimate_impact("kalshi", contracts=10, book_depth=100)
        assert est_pre.blocked is False

        # Record fills that show less impact than default.
        for _ in range(20):
            m.record_fill("kalshi", contracts=10, book_depth=100,
                           expected_price=0.55, actual_price=0.551)

        # Post-learning: coefficient should decrease (less impact).
        params = m.get_venue_params("kalshi")
        assert params.coefficient < 0.10

        # Estimate should show lower impact now.
        est_post = m.estimate_impact("kalshi", contracts=10, book_depth=100)
        assert est_post.estimated_impact < est_pre.estimated_impact

        # Max contracts should increase with lower impact.
        max_pre = m.max_contracts_for_edge("poly", edge=0.03, cost=0.55, book_depth=100)
        max_post = m.max_contracts_for_edge("kalshi", edge=0.03, cost=0.55, book_depth=100)
        assert max_post >= max_pre

    def test_adverse_fills_increase_impact(self) -> None:
        """Fills with high slippage should increase impact estimates."""
        m = _model(learning_rate=0.3)

        # Record fills with significant adverse slippage.
        for _ in range(20):
            m.record_fill("kalshi", contracts=20, book_depth=100,
                           expected_price=0.55, actual_price=0.58)

        params = m.get_venue_params("kalshi")
        # Coefficient should have increased above default.
        assert params.coefficient > 0.10
