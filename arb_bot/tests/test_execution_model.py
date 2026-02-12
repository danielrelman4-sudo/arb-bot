"""Tests for Phase 3A: Execution realism upgrades."""

from __future__ import annotations

import pytest

from arb_bot.execution_model import (
    ExecutionEstimate,
    ExecutionModel,
    ExecutionModelConfig,
    LegFillEstimate,
    LegInput,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _leg(
    venue: str = "kalshi",
    market_id: str = "M1",
    side: str = "yes",
    buy_price: float = 0.55,
    available_size: float = 50.0,
    spread: float = 0.02,
) -> LegInput:
    return LegInput(
        venue=venue,
        market_id=market_id,
        side=side,
        buy_price=buy_price,
        available_size=available_size,
        spread=spread,
    )


# ---------------------------------------------------------------------------
# ExecutionModelConfig
# ---------------------------------------------------------------------------


class TestExecutionModelConfig:
    def test_defaults(self) -> None:
        cfg = ExecutionModelConfig()
        assert cfg.queue_decay_half_life_seconds == 5.0
        assert cfg.latency_seconds == 0.2
        assert cfg.market_impact_factor == 0.01
        assert cfg.max_market_impact == 0.5
        assert cfg.min_fill_fraction == 0.1
        assert cfg.fill_fraction_steps == 5
        assert cfg.sequential_leg_delay_seconds == 1.0
        assert cfg.enable_queue_decay is True
        assert cfg.enable_market_impact is True


# ---------------------------------------------------------------------------
# Queue position decay
# ---------------------------------------------------------------------------


class TestQueueDecay:
    def test_fresh_quote_high_score(self) -> None:
        model = ExecutionModel(ExecutionModelConfig(
            queue_decay_half_life_seconds=5.0,
            latency_seconds=0.0,
        ))
        est = model.simulate([_leg(available_size=100.0)], contracts=10, staleness_seconds=0.0)
        # Fresh quote, plenty of liquidity → high fill probability.
        assert est.legs[0].fill_probability > 0.9

    def test_stale_quote_lower_score(self) -> None:
        model = ExecutionModel(ExecutionModelConfig(
            queue_decay_half_life_seconds=5.0,
            latency_seconds=0.0,
        ))
        fresh = model.simulate([_leg()], contracts=10, staleness_seconds=0.0)
        stale = model.simulate([_leg()], contracts=10, staleness_seconds=10.0)
        assert stale.legs[0].fill_probability < fresh.legs[0].fill_probability

    def test_decay_disabled(self) -> None:
        model = ExecutionModel(ExecutionModelConfig(
            enable_queue_decay=False,
            latency_seconds=0.0,
        ))
        fresh = model.simulate([_leg()], contracts=10, staleness_seconds=0.0)
        stale = model.simulate([_leg()], contracts=10, staleness_seconds=100.0)
        # Without decay, staleness doesn't affect queue position.
        assert stale.legs[0].queue_position_score == pytest.approx(
            fresh.legs[0].queue_position_score
        )

    def test_latency_affects_score(self) -> None:
        low_lat = ExecutionModel(ExecutionModelConfig(latency_seconds=0.1))
        high_lat = ExecutionModel(ExecutionModelConfig(latency_seconds=2.0))
        est_low = low_lat.simulate([_leg()], contracts=10)
        est_high = high_lat.simulate([_leg()], contracts=10)
        assert est_high.legs[0].queue_position_score < est_low.legs[0].queue_position_score


# ---------------------------------------------------------------------------
# Market impact
# ---------------------------------------------------------------------------


class TestMarketImpact:
    def test_small_order_low_impact(self) -> None:
        model = ExecutionModel(ExecutionModelConfig(
            market_impact_factor=0.01,
            enable_queue_decay=False,
            latency_seconds=0.0,
        ))
        est = model.simulate(
            [_leg(available_size=1000.0, spread=0.02)],
            contracts=1,
        )
        assert est.legs[0].market_impact < 0.001

    def test_large_order_higher_impact(self) -> None:
        model = ExecutionModel(ExecutionModelConfig(
            market_impact_factor=0.01,
            enable_queue_decay=False,
            latency_seconds=0.0,
        ))
        small = model.simulate([_leg(available_size=100.0, spread=0.02)], contracts=1)
        large = model.simulate([_leg(available_size=100.0, spread=0.02)], contracts=50)
        assert large.legs[0].market_impact > small.legs[0].market_impact

    def test_impact_capped(self) -> None:
        model = ExecutionModel(ExecutionModelConfig(
            market_impact_factor=1.0,
            max_market_impact=0.5,
            enable_queue_decay=False,
            latency_seconds=0.0,
        ))
        est = model.simulate(
            [_leg(available_size=10.0, spread=0.10)],
            contracts=100,
        )
        assert est.legs[0].market_impact <= 0.10 * 0.5  # spread * max_market_impact

    def test_impact_disabled(self) -> None:
        model = ExecutionModel(ExecutionModelConfig(
            enable_market_impact=False,
        ))
        est = model.simulate([_leg(spread=0.10)], contracts=50)
        assert est.legs[0].market_impact == 0.0

    def test_zero_spread_no_impact(self) -> None:
        model = ExecutionModel(ExecutionModelConfig(
            enable_market_impact=True,
        ))
        est = model.simulate([_leg(spread=0.0)], contracts=50)
        assert est.legs[0].market_impact == 0.0


# ---------------------------------------------------------------------------
# Slippage
# ---------------------------------------------------------------------------


class TestSlippage:
    def test_perfect_fill_low_slippage(self) -> None:
        model = ExecutionModel(ExecutionModelConfig(
            enable_queue_decay=False,
            latency_seconds=0.0,
            enable_market_impact=False,
        ))
        est = model.simulate(
            [_leg(available_size=100.0, spread=0.02)],
            contracts=10,
        )
        # High queue score → low slippage.
        assert est.expected_slippage_per_contract < 0.01

    def test_poor_fill_higher_slippage(self) -> None:
        model = ExecutionModel(ExecutionModelConfig(
            enable_queue_decay=False,
            latency_seconds=0.0,
            enable_market_impact=False,
        ))
        good = model.simulate([_leg(available_size=100.0, spread=0.05)], contracts=10)
        bad = model.simulate([_leg(available_size=10.0, spread=0.05)], contracts=50)
        assert bad.expected_slippage_per_contract > good.expected_slippage_per_contract


# ---------------------------------------------------------------------------
# Sequential execution
# ---------------------------------------------------------------------------


class TestSequentialExecution:
    def test_sequential_degrades_later_legs(self) -> None:
        model = ExecutionModel(ExecutionModelConfig(
            sequential_leg_delay_seconds=2.0,
        ))
        legs = [_leg(market_id="M1"), _leg(market_id="M2", venue="polymarket")]
        est = model.simulate(legs, contracts=10, sequential=True)
        # Second leg has higher time offset.
        assert est.legs[1].time_offset_seconds > est.legs[0].time_offset_seconds
        # Second leg should have lower fill probability.
        assert est.legs[1].fill_probability <= est.legs[0].fill_probability

    def test_parallel_same_offset(self) -> None:
        model = ExecutionModel()
        legs = [_leg(market_id="M1"), _leg(market_id="M2", venue="polymarket")]
        est = model.simulate(legs, contracts=10, sequential=False)
        # Both legs should have same time offset.
        assert est.legs[0].time_offset_seconds == pytest.approx(
            est.legs[1].time_offset_seconds
        )


# ---------------------------------------------------------------------------
# Fill fraction
# ---------------------------------------------------------------------------


class TestFillFraction:
    def test_high_liquidity_full_fill(self) -> None:
        model = ExecutionModel(ExecutionModelConfig(
            enable_queue_decay=False,
            latency_seconds=0.0,
        ))
        est = model.simulate([_leg(available_size=100.0)], contracts=10)
        assert est.expected_fill_fraction > 0.8

    def test_low_liquidity_partial_fill(self) -> None:
        model = ExecutionModel(ExecutionModelConfig(
            enable_queue_decay=False,
            latency_seconds=0.0,
        ))
        est = model.simulate([_leg(available_size=5.0)], contracts=50)
        assert est.expected_fill_fraction < 0.5

    def test_zero_contracts(self) -> None:
        model = ExecutionModel()
        est = model.simulate([_leg()], contracts=0)
        assert est.expected_fill_fraction == 0.0

    def test_below_min_fraction_zeroed(self) -> None:
        model = ExecutionModel(ExecutionModelConfig(
            min_fill_fraction=0.5,
            enable_queue_decay=False,
            latency_seconds=0.0,
        ))
        est = model.simulate([_leg(available_size=5.0)], contracts=100)
        # Very thin liquidity → fill fraction below min → zeroed out.
        assert est.legs[0].expected_fill_fraction == 0.0


# ---------------------------------------------------------------------------
# All-fill probability
# ---------------------------------------------------------------------------


class TestAllFillProbability:
    def test_single_leg(self) -> None:
        model = ExecutionModel(ExecutionModelConfig(
            enable_queue_decay=False,
            latency_seconds=0.0,
        ))
        est = model.simulate([_leg(available_size=100.0)], contracts=10)
        assert est.all_fill_probability == pytest.approx(
            est.legs[0].fill_probability
        )

    def test_two_legs_product(self) -> None:
        model = ExecutionModel(ExecutionModelConfig(
            enable_queue_decay=False,
            latency_seconds=0.0,
        ))
        legs = [
            _leg(market_id="M1", available_size=100.0),
            _leg(market_id="M2", available_size=100.0, venue="polymarket"),
        ]
        est = model.simulate(legs, contracts=10)
        expected = est.legs[0].fill_probability * est.legs[1].fill_probability
        assert est.all_fill_probability == pytest.approx(expected)

    def test_partial_fill_probability(self) -> None:
        model = ExecutionModel(ExecutionModelConfig(
            enable_queue_decay=False,
            latency_seconds=0.0,
        ))
        est = model.simulate([_leg()], contracts=10)
        assert est.partial_fill_probability == pytest.approx(
            1.0 - est.all_fill_probability
        )


# ---------------------------------------------------------------------------
# Graduated fill distribution
# ---------------------------------------------------------------------------


class TestGraduatedDistribution:
    def test_distribution_sums_to_one(self) -> None:
        model = ExecutionModel(ExecutionModelConfig(fill_fraction_steps=5))
        est = model.simulate([_leg()], contracts=10)
        total_prob = sum(prob for _, prob in est.graduated_fill_distribution)
        assert total_prob == pytest.approx(1.0)

    def test_distribution_has_correct_levels(self) -> None:
        model = ExecutionModel(ExecutionModelConfig(fill_fraction_steps=4))
        est = model.simulate([_leg()], contracts=10)
        fractions = [frac for frac, _ in est.graduated_fill_distribution]
        assert fractions[0] == pytest.approx(0.0)
        assert fractions[-1] == pytest.approx(1.0)
        assert len(fractions) == 5  # steps + 1

    def test_empty_legs(self) -> None:
        model = ExecutionModel()
        est = model.simulate([], contracts=10)
        assert est.graduated_fill_distribution == ((0.0, 1.0),)


# ---------------------------------------------------------------------------
# ExecutionEstimate properties
# ---------------------------------------------------------------------------


class TestExecutionEstimate:
    def test_total_cost_adjustment(self) -> None:
        model = ExecutionModel(ExecutionModelConfig(
            enable_queue_decay=False,
            latency_seconds=0.0,
        ))
        est = model.simulate([_leg(spread=0.05)], contracts=10)
        assert est.expected_total_cost_adjustment == pytest.approx(
            est.expected_slippage_per_contract + est.expected_market_impact_per_contract
        )


# ---------------------------------------------------------------------------
# LegFillEstimate
# ---------------------------------------------------------------------------


class TestLegFillEstimate:
    def test_fields(self) -> None:
        est = LegFillEstimate(
            venue="kalshi",
            market_id="M1",
            side="yes",
            fill_probability=0.8,
            expected_fill_fraction=0.9,
            queue_position_score=0.85,
            market_impact=0.001,
            expected_slippage=0.005,
            time_offset_seconds=0.2,
        )
        assert est.venue == "kalshi"
        assert est.fill_probability == 0.8


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_realistic_cross_venue(self) -> None:
        """Simulate a realistic cross-venue arbitrage execution."""
        model = ExecutionModel(ExecutionModelConfig(
            queue_decay_half_life_seconds=5.0,
            latency_seconds=0.15,
            market_impact_factor=0.01,
        ))

        legs = [
            _leg(
                venue="kalshi",
                market_id="K1",
                side="yes",
                buy_price=0.55,
                available_size=30.0,
                spread=0.02,
            ),
            _leg(
                venue="polymarket",
                market_id="P1",
                side="no",
                buy_price=0.42,
                available_size=50.0,
                spread=0.03,
            ),
        ]

        est = model.simulate(legs, contracts=10, staleness_seconds=3.0, sequential=True)

        # Should produce reasonable estimates.
        assert 0.0 < est.all_fill_probability < 1.0
        assert 0.0 < est.expected_fill_fraction <= 1.0
        assert est.expected_slippage_per_contract >= 0.0
        assert est.expected_market_impact_per_contract >= 0.0
        assert len(est.graduated_fill_distribution) > 0

        # Sequential: second leg should be worse.
        assert est.legs[1].time_offset_seconds > est.legs[0].time_offset_seconds

    def test_config_property(self) -> None:
        cfg = ExecutionModelConfig(latency_seconds=0.5)
        model = ExecutionModel(cfg)
        assert model.config.latency_seconds == 0.5
