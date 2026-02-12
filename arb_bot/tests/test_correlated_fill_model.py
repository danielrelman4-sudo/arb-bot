"""Tests for Phase 8A: Correlated multi-leg fill model.

Validates that same-venue structural opportunities use correlation-adjusted
fill probability instead of the independent product that causes the multi-leg
death spiral (3 legs at 0.54 → 0.157 independent vs ~0.374 with rho=0.7).
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from arb_bot.config import FillModelSettings
from arb_bot.fill_model import (
    CorrelationMode,
    FillEstimate,
    FillModel,
    _combine_fill_probabilities,
)
from arb_bot.models import (
    ArbitrageOpportunity,
    ExecutionStyle,
    OpportunityKind,
    OpportunityLeg,
    Side,
    TradeLegPlan,
    TradePlan,
)

NOW = datetime.now(timezone.utc)


def _make_settings(**overrides: object) -> FillModelSettings:
    defaults = dict(
        enabled=True,
        min_fill_probability=0.2,
        queue_depth_factor=1.0,
        stale_quote_half_life_seconds=60.0,
        spread_penalty_weight=0.0,
        transform_source_penalty=0.0,
        partial_fill_penalty_per_contract=0.01,
        min_expected_realized_profit_usd=0.0,
        same_venue_fill_correlation=0.7,
    )
    defaults.update(overrides)
    return FillModelSettings(**defaults)


def _make_leg(venue: str = "kalshi", market_id: str = "A", side: Side = Side.YES,
              buy_price: float = 0.50, buy_size: float = 100.0,
              metadata: dict | None = None) -> OpportunityLeg:
    return OpportunityLeg(
        venue=venue,
        market_id=market_id,
        side=side,
        buy_price=buy_price,
        buy_size=buy_size,
        metadata=metadata or {},
    )


def _make_opportunity(
    kind: OpportunityKind = OpportunityKind.STRUCTURAL_BUCKET,
    legs: tuple[OpportunityLeg, ...] | None = None,
    edge: float = 0.05,
) -> ArbitrageOpportunity:
    if legs is None:
        legs = (
            _make_leg("kalshi", "A", Side.YES),
            _make_leg("kalshi", "B", Side.YES),
            _make_leg("kalshi", "C", Side.YES),
        )
    return ArbitrageOpportunity(
        kind=kind,
        execution_style=ExecutionStyle.TAKER,
        legs=legs,
        gross_edge_per_contract=edge + 0.01,
        net_edge_per_contract=edge,
        fee_per_contract=0.01,
        observed_at=NOW,
        match_key="-".join(leg.market_id for leg in legs),
    )


def _make_plan(opportunity: ArbitrageOpportunity, contracts: int = 10) -> TradePlan:
    return TradePlan(
        kind=opportunity.kind,
        execution_style=opportunity.execution_style,
        legs=tuple(
            TradeLegPlan(
                leg.venue, leg.market_id, leg.side, contracts,
                leg.buy_price, metadata=leg.metadata,
            )
            for leg in opportunity.legs
        ),
        contracts=contracts,
        capital_required=contracts * sum(leg.buy_price for leg in opportunity.legs),
        capital_required_by_venue={
            leg.venue: contracts * leg.buy_price for leg in opportunity.legs
        },
        expected_profit=contracts * opportunity.net_edge_per_contract,
        edge_per_contract=opportunity.net_edge_per_contract,
    )


# ===================================================================
# Tests for _combine_fill_probabilities (unit function)
# ===================================================================


class TestCombineFillProbabilities:
    """Tests for the core probability combination function."""

    def test_empty_list_returns_one(self) -> None:
        assert _combine_fill_probabilities([], CorrelationMode.INDEPENDENT, 0.7) == 1.0
        assert _combine_fill_probabilities([], CorrelationMode.SAME_VENUE, 0.7) == 1.0

    def test_single_leg_returns_same_probability(self) -> None:
        assert _combine_fill_probabilities([0.8], CorrelationMode.INDEPENDENT, 0.7) == 0.8
        assert _combine_fill_probabilities([0.8], CorrelationMode.SAME_VENUE, 0.7) == 0.8

    def test_independent_mode_is_simple_product(self) -> None:
        probs = [0.9, 0.8, 0.7]
        expected = 0.9 * 0.8 * 0.7
        result = _combine_fill_probabilities(probs, CorrelationMode.INDEPENDENT, 0.7)
        assert abs(result - expected) < 1e-12

    def test_same_venue_higher_than_independent_for_multi_leg(self) -> None:
        """The whole point: same-venue correlation raises combined probability."""
        probs = [0.54, 0.54, 0.54]
        independent = _combine_fill_probabilities(probs, CorrelationMode.INDEPENDENT, 0.7)
        correlated = _combine_fill_probabilities(probs, CorrelationMode.SAME_VENUE, 0.7)
        # Independent: 0.54^3 = 0.157
        assert abs(independent - 0.54 ** 3) < 1e-10
        # Correlated should be significantly higher
        assert correlated > independent
        # With rho=0.7, effective_N = 1 + 2*0.3 = 1.6
        # geom_mean = 0.54, combined = 0.54^1.6 ≈ 0.374
        expected = 0.54 ** 1.6
        assert abs(correlated - expected) < 1e-10

    def test_correlation_zero_equals_independent(self) -> None:
        """When rho=0.0, same_venue mode degrades to independent product."""
        probs = [0.6, 0.7, 0.5]
        independent = _combine_fill_probabilities(probs, CorrelationMode.INDEPENDENT, 0.0)
        same_venue = _combine_fill_probabilities(probs, CorrelationMode.SAME_VENUE, 0.0)
        assert abs(independent - same_venue) < 1e-12

    def test_correlation_one_equals_geometric_mean(self) -> None:
        """When rho=1.0, combined = geometric mean (single fill event)."""
        probs = [0.5, 0.6, 0.7]
        geom_mean = (0.5 * 0.6 * 0.7) ** (1.0 / 3.0)
        result = _combine_fill_probabilities(probs, CorrelationMode.SAME_VENUE, 1.0)
        assert abs(result - geom_mean) < 1e-10

    def test_zero_leg_probability_gives_zero(self) -> None:
        """If any leg has zero fill prob, combined is zero regardless of mode."""
        probs = [0.8, 0.0, 0.9]
        assert _combine_fill_probabilities(probs, CorrelationMode.SAME_VENUE, 0.7) == 0.0
        assert _combine_fill_probabilities(probs, CorrelationMode.INDEPENDENT, 0.7) == 0.0

    def test_all_ones_gives_one(self) -> None:
        probs = [1.0, 1.0, 1.0]
        assert _combine_fill_probabilities(probs, CorrelationMode.SAME_VENUE, 0.7) == 1.0
        assert _combine_fill_probabilities(probs, CorrelationMode.INDEPENDENT, 0.7) == 1.0

    def test_correlation_clamped_above_one(self) -> None:
        """Correlation > 1 should be clamped to 1.0."""
        probs = [0.5, 0.6, 0.7]
        result_at_1 = _combine_fill_probabilities(probs, CorrelationMode.SAME_VENUE, 1.0)
        result_at_2 = _combine_fill_probabilities(probs, CorrelationMode.SAME_VENUE, 2.0)
        assert abs(result_at_1 - result_at_2) < 1e-12

    def test_correlation_clamped_below_zero(self) -> None:
        """Negative correlation should be clamped to 0.0 (= independent product)."""
        probs = [0.6, 0.7, 0.5]
        independent = _combine_fill_probabilities(probs, CorrelationMode.INDEPENDENT, 0.0)
        result_neg = _combine_fill_probabilities(probs, CorrelationMode.SAME_VENUE, -0.5)
        assert abs(independent - result_neg) < 1e-12

    def test_two_legs_formula(self) -> None:
        """Verify math for 2-leg case: effective_N = 1 + (2-1)*0.3 = 1.3."""
        probs = [0.6, 0.8]
        rho = 0.7
        geom_mean = math.sqrt(0.6 * 0.8)
        effective_n = 1.0 + 1.0 * 0.3  # 1.3
        expected = geom_mean ** effective_n
        result = _combine_fill_probabilities(probs, CorrelationMode.SAME_VENUE, rho)
        assert abs(result - expected) < 1e-10

    def test_many_legs_formula(self) -> None:
        """Verify with 5 legs at different probabilities."""
        probs = [0.5, 0.6, 0.7, 0.4, 0.8]
        rho = 0.7
        n = 5
        geom_mean = math.exp(sum(math.log(p) for p in probs) / n)
        effective_n = 1.0 + (n - 1.0) * (1.0 - rho)  # 1 + 4*0.3 = 2.2
        expected = geom_mean ** effective_n
        result = _combine_fill_probabilities(probs, CorrelationMode.SAME_VENUE, rho)
        assert abs(result - expected) < 1e-10

    def test_result_bounded_zero_one(self) -> None:
        """Result should always be in [0, 1]."""
        probs = [0.99, 0.98, 0.97]
        result = _combine_fill_probabilities(probs, CorrelationMode.SAME_VENUE, 0.7)
        assert 0.0 <= result <= 1.0


# ===================================================================
# Tests for FillModel.estimate() with correlation modes
# ===================================================================


class TestFillModelCorrelation:
    """Tests for FillModel.estimate() using different correlation modes."""

    def test_independent_mode_is_default(self) -> None:
        """Default call should use independent mode (backward compatible)."""
        model = FillModel(_make_settings())
        opp = _make_opportunity()
        plan = _make_plan(opp)
        result = model.estimate(opp, plan, now=NOW)
        assert result.correlation_mode == "independent"

    def test_same_venue_mode_produces_higher_fill_probability(self) -> None:
        """Same-venue mode should produce higher all_fill_probability for multi-leg."""
        model = FillModel(_make_settings())
        # Use constrained liquidity so per-leg probs are < 1.0
        legs = (
            _make_leg("kalshi", "A", Side.YES, buy_size=8.0),
            _make_leg("kalshi", "B", Side.YES, buy_size=8.0),
            _make_leg("kalshi", "C", Side.YES, buy_size=8.0),
        )
        opp = _make_opportunity(legs=legs)
        plan = _make_plan(opp)

        independent = model.estimate(opp, plan, now=NOW, correlation_mode="independent")
        correlated = model.estimate(opp, plan, now=NOW, correlation_mode="same_venue")

        assert correlated.all_fill_probability > independent.all_fill_probability
        assert correlated.correlation_mode == "same_venue"
        assert independent.correlation_mode == "independent"

    def test_same_venue_higher_realized_profit(self) -> None:
        """Higher fill probability should lead to higher expected realized profit."""
        model = FillModel(_make_settings())
        # Use constrained liquidity so per-leg probs are < 1.0
        legs = (
            _make_leg("kalshi", "A", Side.YES, buy_size=8.0),
            _make_leg("kalshi", "B", Side.YES, buy_size=8.0),
            _make_leg("kalshi", "C", Side.YES, buy_size=8.0),
        )
        opp = _make_opportunity(edge=0.05, legs=legs)
        plan = _make_plan(opp)

        independent = model.estimate(opp, plan, now=NOW, correlation_mode="independent")
        correlated = model.estimate(opp, plan, now=NOW, correlation_mode="same_venue")

        assert correlated.expected_realized_profit > independent.expected_realized_profit

    def test_leg_probabilities_unchanged_between_modes(self) -> None:
        """Per-leg fill probabilities should be identical — only combination differs."""
        model = FillModel(_make_settings())
        opp = _make_opportunity()
        plan = _make_plan(opp)

        independent = model.estimate(opp, plan, now=NOW, correlation_mode="independent")
        correlated = model.estimate(opp, plan, now=NOW, correlation_mode="same_venue")

        assert independent.leg_fill_probabilities == correlated.leg_fill_probabilities

    def test_string_correlation_mode_accepted(self) -> None:
        """String mode should be accepted and converted."""
        model = FillModel(_make_settings())
        opp = _make_opportunity()
        plan = _make_plan(opp)

        result = model.estimate(opp, plan, now=NOW, correlation_mode="same_venue")
        assert result.correlation_mode == "same_venue"

    def test_enum_correlation_mode_accepted(self) -> None:
        model = FillModel(_make_settings())
        opp = _make_opportunity()
        plan = _make_plan(opp)

        result = model.estimate(opp, plan, now=NOW, correlation_mode=CorrelationMode.SAME_VENUE)
        assert result.correlation_mode == "same_venue"

    def test_zero_contracts_returns_zero_fill(self) -> None:
        model = FillModel(_make_settings())
        opp = _make_opportunity()
        plan = TradePlan(
            kind=opp.kind,
            execution_style=opp.execution_style,
            legs=tuple(
                TradeLegPlan(leg.venue, leg.market_id, leg.side, 0, leg.buy_price)
                for leg in opp.legs
            ),
            contracts=0,
            capital_required=0.0,
            capital_required_by_venue={},
            expected_profit=0.0,
            edge_per_contract=0.0,
        )
        result = model.estimate(opp, plan, now=NOW, correlation_mode="same_venue")
        assert result.all_fill_probability == 0.0
        assert result.correlation_mode == "same_venue"

    def test_death_spiral_example(self) -> None:
        """The motivating example: 3 legs at ~0.54 each.

        Independent: 0.54^3 ≈ 0.157 (fails 0.70 threshold)
        Correlated (rho=0.7): 0.54^1.6 ≈ 0.374 (passes 0.70 threshold)
        """
        # Create settings that produce leg probs near 0.54
        # With queue_depth_factor=1.0, buy_size=5 and contracts=10:
        # depth_ratio = min(1, 5/(10*1)) = 0.5
        # With some staleness adjustment, we can get close to 0.54
        settings = _make_settings(
            queue_depth_factor=1.0,
            spread_penalty_weight=0.0,
        )
        model = FillModel(settings)

        legs = (
            _make_leg("kalshi", "A", Side.YES, buy_size=5.0),
            _make_leg("kalshi", "B", Side.YES, buy_size=5.0),
            _make_leg("kalshi", "C", Side.YES, buy_size=5.0),
        )
        opp = _make_opportunity(legs=legs)
        plan = _make_plan(opp, contracts=10)

        independent = model.estimate(opp, plan, now=NOW, correlation_mode="independent")
        correlated = model.estimate(opp, plan, now=NOW, correlation_mode="same_venue")

        # Each leg should have depth_ratio = 5/(10*1) = 0.5
        for prob in independent.leg_fill_probabilities:
            assert abs(prob - 0.5) < 0.01

        # Independent: ~0.5^3 = 0.125 — fails any reasonable threshold
        assert independent.all_fill_probability < 0.20

        # Correlated: ~0.5^1.6 ≈ 0.33 — much more reasonable
        assert correlated.all_fill_probability > 0.30
        assert correlated.all_fill_probability > independent.all_fill_probability * 2.0


# ===================================================================
# Tests for _choose_correlation_mode (engine integration)
# ===================================================================


class TestChooseCorrelationMode:
    """Tests for the engine's correlation mode selection logic."""

    def test_structural_bucket_same_venue_uses_correlated(self) -> None:
        from arb_bot.engine import _choose_correlation_mode

        opp = _make_opportunity(
            kind=OpportunityKind.STRUCTURAL_BUCKET,
            legs=(
                _make_leg("kalshi", "A"),
                _make_leg("kalshi", "B"),
                _make_leg("kalshi", "C"),
            ),
        )
        assert _choose_correlation_mode(opp) == CorrelationMode.SAME_VENUE

    def test_structural_parity_same_venue_uses_correlated(self) -> None:
        from arb_bot.engine import _choose_correlation_mode

        opp = _make_opportunity(
            kind=OpportunityKind.STRUCTURAL_PARITY,
            legs=(
                _make_leg("kalshi", "X"),
                _make_leg("kalshi", "Y"),
            ),
        )
        assert _choose_correlation_mode(opp) == CorrelationMode.SAME_VENUE

    def test_structural_event_tree_same_venue_uses_correlated(self) -> None:
        from arb_bot.engine import _choose_correlation_mode

        opp = _make_opportunity(
            kind=OpportunityKind.STRUCTURAL_EVENT_TREE,
            legs=(
                _make_leg("polymarket", "P1"),
                _make_leg("polymarket", "P2"),
                _make_leg("polymarket", "P3"),
            ),
        )
        assert _choose_correlation_mode(opp) == CorrelationMode.SAME_VENUE

    def test_structural_bucket_cross_venue_uses_independent(self) -> None:
        """Structural bucket with legs on different venues → independent."""
        from arb_bot.engine import _choose_correlation_mode

        opp = _make_opportunity(
            kind=OpportunityKind.STRUCTURAL_BUCKET,
            legs=(
                _make_leg("kalshi", "A"),
                _make_leg("polymarket", "B"),
            ),
        )
        assert _choose_correlation_mode(opp) == CorrelationMode.INDEPENDENT

    def test_intra_venue_uses_independent(self) -> None:
        """Intra-venue (2-leg YES/NO) should stay independent."""
        from arb_bot.engine import _choose_correlation_mode

        opp = _make_opportunity(
            kind=OpportunityKind.INTRA_VENUE,
            legs=(
                _make_leg("kalshi", "A", Side.YES),
                _make_leg("kalshi", "A", Side.NO),
            ),
        )
        assert _choose_correlation_mode(opp) == CorrelationMode.INDEPENDENT

    def test_cross_venue_uses_independent(self) -> None:
        from arb_bot.engine import _choose_correlation_mode

        opp = _make_opportunity(
            kind=OpportunityKind.CROSS_VENUE,
            legs=(
                _make_leg("kalshi", "A"),
                _make_leg("polymarket", "B"),
            ),
        )
        assert _choose_correlation_mode(opp) == CorrelationMode.INDEPENDENT

    def test_single_leg_structural_uses_independent(self) -> None:
        """Single-leg structural (degenerate) → independent (no combination needed)."""
        from arb_bot.engine import _choose_correlation_mode

        opp = _make_opportunity(
            kind=OpportunityKind.STRUCTURAL_BUCKET,
            legs=(_make_leg("kalshi", "A"),),
        )
        assert _choose_correlation_mode(opp) == CorrelationMode.INDEPENDENT


# ===================================================================
# Tests for config loading
# ===================================================================


class TestCorrelationConfig:
    """Tests for the same_venue_fill_correlation config parameter."""

    def test_default_value(self) -> None:
        settings = FillModelSettings()
        assert settings.same_venue_fill_correlation == 0.7

    def test_custom_value(self) -> None:
        settings = FillModelSettings(same_venue_fill_correlation=0.5)
        assert settings.same_venue_fill_correlation == 0.5

    def test_fill_estimate_has_correlation_mode_field(self) -> None:
        est = FillEstimate(
            leg_fill_probabilities=(0.8,),
            all_fill_probability=0.8,
            partial_fill_probability=0.2,
            expected_slippage_per_contract=0.0,
            fill_quality_score=0.0,
            adverse_selection_flag=False,
            expected_realized_edge_per_contract=0.01,
            expected_realized_profit=0.1,
            correlation_mode="same_venue",
        )
        assert est.correlation_mode == "same_venue"

    def test_fill_estimate_default_correlation_mode(self) -> None:
        est = FillEstimate(
            leg_fill_probabilities=(0.8,),
            all_fill_probability=0.8,
            partial_fill_probability=0.2,
            expected_slippage_per_contract=0.0,
            fill_quality_score=0.0,
            adverse_selection_flag=False,
            expected_realized_edge_per_contract=0.01,
            expected_realized_profit=0.1,
        )
        assert est.correlation_mode == "independent"

    def test_correlation_mode_enum_values(self) -> None:
        assert CorrelationMode.INDEPENDENT.value == "independent"
        assert CorrelationMode.SAME_VENUE.value == "same_venue"
        assert CorrelationMode("independent") is CorrelationMode.INDEPENDENT
        assert CorrelationMode("same_venue") is CorrelationMode.SAME_VENUE

    def test_invalid_correlation_mode_raises(self) -> None:
        with pytest.raises(ValueError):
            CorrelationMode("invalid_mode")
