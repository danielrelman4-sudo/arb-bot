"""Tests for Monte Carlo execution simulation (B2)."""
from __future__ import annotations

import math
from collections import Counter
from datetime import datetime, timezone

import pytest

from arb_bot.models import (
    ArbitrageOpportunity,
    ExecutionStyle,
    OpportunityKind,
    OpportunityLeg,
    Side,
    TradePlan,
    TradeLegPlan,
)
from arb_bot.monte_carlo_sim import MonteCarloResult, MonteCarloSettings, MonteCarloSimulator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_opportunity(
    kind: OpportunityKind = OpportunityKind.CROSS_VENUE,
    n_legs: int = 2,
    edge: float = 0.05,
) -> ArbitrageOpportunity:
    legs = []
    for i in range(n_legs):
        legs.append(
            OpportunityLeg(
                venue=f"venue{i}",
                market_id=f"market{i}",
                side=Side.YES if i % 2 == 0 else Side.NO,
                buy_price=0.50 - edge / n_legs * (1 if i % 2 == 0 else -1),
                buy_size=100.0,
                metadata={},
            )
        )
    return ArbitrageOpportunity(
        kind=kind,
        execution_style=ExecutionStyle.TAKER,
        legs=tuple(legs),
        gross_edge_per_contract=edge,
        net_edge_per_contract=edge,
        fee_per_contract=0.0,
        observed_at=datetime.now(timezone.utc),
        match_key="test",
        match_score=1.0,
        payout_per_contract=1.0,
        metadata={},
    )


def _make_plan(
    contracts: int = 10,
    edge: float = 0.05,
    kind: OpportunityKind = OpportunityKind.CROSS_VENUE,
) -> TradePlan:
    return TradePlan(
        kind=kind,
        execution_style=ExecutionStyle.TAKER,
        contracts=contracts,
        edge_per_contract=edge,
        expected_profit=edge * contracts,
        capital_required=0.50 * contracts,
        capital_required_by_venue={"venue0": 0.25 * contracts, "venue1": 0.25 * contracts},
        legs=(
            TradeLegPlan(venue="venue0", market_id="market0", side=Side.YES, contracts=contracts, limit_price=0.50),
            TradeLegPlan(venue="venue1", market_id="market1", side=Side.NO, contracts=contracts, limit_price=0.50),
        ),
    )


# ---------------------------------------------------------------------------
# Tests: MonteCarloSettings
# ---------------------------------------------------------------------------

class TestMonteCarloSettings:
    def test_defaults(self):
        s = MonteCarloSettings()
        assert s.enabled is True
        assert s.legging_loss_fraction == 0.03
        assert s.adverse_selection_probability == 0.15
        assert s.resolution_success_rate == 0.95
        assert s.seed is None

    def test_disabled(self):
        s = MonteCarloSettings(enabled=False)
        sim = MonteCarloSimulator(s)
        assert sim.enabled is False


# ---------------------------------------------------------------------------
# Tests: Deterministic seeded simulation
# ---------------------------------------------------------------------------

class TestDeterministicSeed:
    def test_same_seed_same_results(self):
        """Two simulators with the same seed produce identical outcomes."""
        opp = _make_opportunity()
        plan = _make_plan()

        sim1 = MonteCarloSimulator(MonteCarloSettings(seed=42))
        sim2 = MonteCarloSimulator(MonteCarloSettings(seed=42))

        r1 = sim1.simulate_trade(opp, plan, all_fill_probability=0.8)
        r2 = sim2.simulate_trade(opp, plan, all_fill_probability=0.8)

        assert r1.leg_fills == r2.leg_fills
        assert r1.simulated_pnl == r2.simulated_pnl
        assert r1.adverse_selection_hit == r2.adverse_selection_hit

    def test_different_seeds_different_results(self):
        """Different seeds produce different outcomes (with high probability)."""
        opp = _make_opportunity()
        plan = _make_plan()

        results = []
        for seed in range(10):
            sim = MonteCarloSimulator(MonteCarloSettings(seed=seed))
            r = sim.simulate_trade(opp, plan, all_fill_probability=0.8)
            results.append(r.simulated_pnl)

        # At least 2 distinct PnL values in 10 different seeds.
        assert len(set(results)) >= 2


# ---------------------------------------------------------------------------
# Tests: Fill simulation
# ---------------------------------------------------------------------------

class TestFillSimulation:
    def test_all_legs_fill_high_probability(self):
        """With fill prob = 1.0, all legs should always fill."""
        sim = MonteCarloSimulator(MonteCarloSettings(seed=42))
        opp = _make_opportunity()
        plan = _make_plan()

        r = sim.simulate_trade(opp, plan, all_fill_probability=1.0)
        assert r.all_legs_filled is True
        assert r.filled_leg_count == 2
        assert r.simulated_filled_contracts == plan.contracts

    def test_no_legs_fill_zero_probability(self):
        """With fill prob = 0.0, no legs should fill."""
        sim = MonteCarloSimulator(MonteCarloSettings(seed=42))
        opp = _make_opportunity()
        plan = _make_plan()

        r = sim.simulate_trade(opp, plan, all_fill_probability=0.0)
        assert r.all_legs_filled is False
        assert r.filled_leg_count == 0
        assert r.simulated_filled_contracts == 0
        assert r.simulated_pnl == 0.0

    def test_per_leg_probabilities_used(self):
        """Per-leg probabilities are used when provided."""
        sim = MonteCarloSimulator(MonteCarloSettings(seed=42))
        opp = _make_opportunity()
        plan = _make_plan()

        # One leg certain, one leg impossible.
        r = sim.simulate_trade(
            opp, plan,
            leg_fill_probabilities=(1.0, 0.0),
            all_fill_probability=0.5,
        )
        # Leg 0 fills (prob=1.0), leg 1 doesn't (prob=0.0).
        assert r.leg_fills[0] is True
        assert r.leg_fills[1] is False
        assert r.all_legs_filled is False
        assert r.filled_leg_count == 1

    def test_fill_distribution_realistic(self):
        """Over many trials, fill rate approximates the input probability."""
        fill_prob = 0.7
        n_trials = 1000
        fill_count = 0

        for i in range(n_trials):
            sim = MonteCarloSimulator(MonteCarloSettings(seed=i))
            opp = _make_opportunity(n_legs=1)
            plan = TradePlan(
                kind=OpportunityKind.CROSS_VENUE,
                execution_style=ExecutionStyle.TAKER,
                contracts=10,
                edge_per_contract=0.05,
                expected_profit=0.50,
                capital_required=5.0,
                capital_required_by_venue={"venue0": 5.0},
                legs=(
                    TradeLegPlan(venue="venue0", market_id="market0", side=Side.YES, contracts=10, limit_price=0.50),
                ),
            )
            r = sim.simulate_trade(opp, plan, all_fill_probability=fill_prob)
            if r.all_legs_filled:
                fill_count += 1

        actual_rate = fill_count / n_trials
        assert abs(actual_rate - fill_prob) < 0.08, (
            f"Expected fill rate ~{fill_prob}, got {actual_rate}"
        )


# ---------------------------------------------------------------------------
# Tests: Legging risk
# ---------------------------------------------------------------------------

class TestLeggingRisk:
    def test_legging_loss_when_partial_fill(self):
        """When one leg fills and another doesn't, legging loss is incurred."""
        sim = MonteCarloSimulator(MonteCarloSettings(seed=42))
        opp = _make_opportunity()
        plan = _make_plan()

        # Force one leg to fill and one not.
        r = sim.simulate_trade(
            opp, plan,
            leg_fill_probabilities=(1.0, 0.0),
        )
        assert not r.all_legs_filled
        assert r.filled_leg_count == 1
        assert r.legging_loss > 0.0
        assert r.simulated_pnl < 0.0  # Net loss from legging risk.

    def test_legging_loss_fraction_scales(self):
        """Legging loss scales with the configured fraction."""
        opp = _make_opportunity()
        plan = _make_plan()

        sim_low = MonteCarloSimulator(MonteCarloSettings(seed=42, legging_loss_fraction=0.01))
        sim_high = MonteCarloSimulator(MonteCarloSettings(seed=42, legging_loss_fraction=0.10))

        r_low = sim_low.simulate_trade(opp, plan, leg_fill_probabilities=(1.0, 0.0))
        r_high = sim_high.simulate_trade(opp, plan, leg_fill_probabilities=(1.0, 0.0))

        assert r_high.legging_loss > r_low.legging_loss

    def test_no_legging_loss_when_all_fill(self):
        """No legging loss when all legs fill."""
        sim = MonteCarloSimulator(MonteCarloSettings(seed=42))
        opp = _make_opportunity()
        plan = _make_plan()

        r = sim.simulate_trade(opp, plan, all_fill_probability=1.0)
        assert r.legging_loss == 0.0


# ---------------------------------------------------------------------------
# Tests: Adverse selection
# ---------------------------------------------------------------------------

class TestAdverseSelection:
    def test_adverse_selection_sometimes_occurs(self):
        """Over many trials, adverse selection hits at expected rate."""
        n_trials = 1000
        hit_count = 0

        for i in range(n_trials):
            sim = MonteCarloSimulator(MonteCarloSettings(
                seed=i,
                adverse_selection_probability=0.15,
            ))
            opp = _make_opportunity()
            plan = _make_plan()
            r = sim.simulate_trade(opp, plan, all_fill_probability=1.0)
            if r.adverse_selection_hit:
                hit_count += 1

        rate = hit_count / n_trials
        assert 0.08 < rate < 0.25, f"Adverse selection rate {rate} outside expected range"

    def test_adverse_selection_reduces_pnl(self):
        """When adverse selection occurs, PnL is lower than without."""
        opp = _make_opportunity()
        plan = _make_plan()

        # Force adverse selection by setting probability to 1.0.
        sim = MonteCarloSimulator(MonteCarloSettings(
            seed=42,
            adverse_selection_probability=1.0,
            adverse_selection_edge_loss=0.5,
        ))
        r = sim.simulate_trade(opp, plan, all_fill_probability=1.0)
        assert r.adverse_selection_hit is True
        assert r.adverse_selection_cost > 0.0

    def test_no_adverse_selection_when_disabled(self):
        """No adverse selection when probability is 0."""
        sim = MonteCarloSimulator(MonteCarloSettings(
            seed=42,
            adverse_selection_probability=0.0,
        ))
        opp = _make_opportunity()
        plan = _make_plan()

        for _ in range(50):
            r = sim.simulate_trade(opp, plan, all_fill_probability=1.0)
            assert r.adverse_selection_hit is False
            assert r.adverse_selection_cost == 0.0


# ---------------------------------------------------------------------------
# Tests: Slippage
# ---------------------------------------------------------------------------

class TestSlippage:
    def test_slippage_is_nonnegative(self):
        """Slippage should always be >= 0."""
        for seed in range(100):
            sim = MonteCarloSimulator(MonteCarloSettings(seed=seed))
            opp = _make_opportunity()
            plan = _make_plan()
            r = sim.simulate_trade(opp, plan, all_fill_probability=1.0)
            assert r.slippage_cost >= 0.0

    def test_slippage_bounded_by_max(self):
        """Slippage per contract should never exceed max_cents."""
        for seed in range(100):
            sim = MonteCarloSimulator(MonteCarloSettings(
                seed=seed,
                slippage_max_cents=2.0,
            ))
            opp = _make_opportunity()
            plan = _make_plan(contracts=10)
            r = sim.simulate_trade(opp, plan, all_fill_probability=1.0)
            if r.simulated_filled_contracts > 0:
                per_contract = r.slippage_cost / r.simulated_filled_contracts
                assert per_contract <= 0.021  # 2.0 cents + floating point margin


# ---------------------------------------------------------------------------
# Tests: Edge decay
# ---------------------------------------------------------------------------

class TestEdgeDecay:
    def test_edge_decay_reduces_pnl(self):
        """Edge decay from latency should reduce PnL."""
        sim = MonteCarloSimulator(MonteCarloSettings(
            seed=42,
            expected_latency_seconds=5.0,
            edge_decay_half_life_seconds=5.0,
            adverse_selection_probability=0.0,
        ))
        opp = _make_opportunity(edge=0.10)
        plan = _make_plan(edge=0.10)

        r = sim.simulate_trade(opp, plan, all_fill_probability=1.0)
        assert r.edge_decay_cost > 0.0
        # PnL should be less than gross edge due to decay.
        max_possible = plan.edge_per_contract * plan.contracts
        assert r.simulated_pnl < max_possible


# ---------------------------------------------------------------------------
# Tests: Resolution risk
# ---------------------------------------------------------------------------

class TestResolutionRisk:
    def test_resolution_failure_occasionally(self):
        """With 95% success rate, ~5% of trades should fail to resolve."""
        n_trials = 1000
        failure_count = 0

        for i in range(n_trials):
            sim = MonteCarloSimulator(MonteCarloSettings(
                seed=i,
                resolution_success_rate=0.95,
                adverse_selection_probability=0.0,
            ))
            opp = _make_opportunity()
            plan = _make_plan()
            r = sim.simulate_trade(opp, plan, all_fill_probability=1.0)
            if r.resolution_pnl < 0:
                failure_count += 1

        rate = failure_count / n_trials
        assert 0.02 < rate < 0.10, f"Resolution failure rate {rate} outside expected range"


# ---------------------------------------------------------------------------
# Tests: PnL distribution
# ---------------------------------------------------------------------------

class TestPnLDistribution:
    def test_pnl_has_wins_and_losses(self):
        """Monte Carlo should produce both wins and losses."""
        wins = 0
        losses = 0
        n_trials = 500

        for i in range(n_trials):
            sim = MonteCarloSimulator(MonteCarloSettings(seed=i))
            opp = _make_opportunity()
            plan = _make_plan()
            r = sim.simulate_trade(opp, plan, all_fill_probability=0.85)
            if r.simulated_pnl > 0:
                wins += 1
            elif r.simulated_pnl < 0:
                losses += 1

        assert wins > 0, "Should have some winning trades"
        assert losses > 0, "Should have some losing trades"

    def test_average_pnl_less_than_ev(self):
        """Average simulated PnL should be less than pure EV."""
        total_pnl = 0.0
        n_trials = 1000

        for i in range(n_trials):
            sim = MonteCarloSimulator(MonteCarloSettings(seed=i))
            opp = _make_opportunity(edge=0.05)
            plan = _make_plan(contracts=10, edge=0.05)
            r = sim.simulate_trade(
                opp, plan,
                all_fill_probability=0.85,
                expected_realized_profit=0.50,
            )
            total_pnl += r.simulated_pnl

        avg_pnl = total_pnl / n_trials
        ev_pnl = 0.50  # expected_realized_profit
        # Average MC PnL should be substantially below pure EV
        # because of legging risk, slippage, adverse selection, etc.
        assert avg_pnl < ev_pnl, (
            f"Average MC PnL ({avg_pnl:.4f}) should be less than EV ({ev_pnl:.4f})"
        )


# ---------------------------------------------------------------------------
# Tests: Settlement simulation
# ---------------------------------------------------------------------------

class TestSettlementSimulation:
    def test_settlement_with_zero_contracts(self):
        sim = MonteCarloSimulator(MonteCarloSettings(seed=42))
        opp = _make_opportunity()
        result = sim.simulate_settlement(
            expected_realized_profit=0.50,
            filled_contracts=0,
            opportunity=opp,
        )
        assert result == 0.0

    def test_settlement_produces_variation(self):
        """Settlement should produce variation around expected value."""
        opp = _make_opportunity()
        results = []
        for i in range(100):
            sim = MonteCarloSimulator(MonteCarloSettings(seed=i))
            r = sim.simulate_settlement(
                expected_realized_profit=1.0,
                filled_contracts=10,
                opportunity=opp,
            )
            results.append(r)

        # Should have some variation.
        assert len(set(round(r, 4) for r in results)) > 5
        # Some should be negative (resolution failures).
        assert any(r < 0 for r in results), "Expected some negative settlements"

    def test_settlement_average_near_expected(self):
        """Average settlement should be near but below expected profit."""
        opp = _make_opportunity()
        total = 0.0
        n = 1000
        for i in range(n):
            sim = MonteCarloSimulator(MonteCarloSettings(seed=i))
            total += sim.simulate_settlement(1.0, 10, opp)

        avg = total / n
        # Should be positive but less than 1.0 due to adverse selection and resolution risk.
        assert 0.5 < avg < 1.0, f"Average settlement {avg:.4f} out of expected range"


# ---------------------------------------------------------------------------
# Tests: Multi-leg opportunities
# ---------------------------------------------------------------------------

class TestMultiLeg:
    def test_three_leg_structural(self):
        """Three-leg structural bucket opportunity."""
        sim = MonteCarloSimulator(MonteCarloSettings(seed=42))
        opp = _make_opportunity(
            kind=OpportunityKind.STRUCTURAL_BUCKET,
            n_legs=3,
            edge=0.10,
        )
        plan = TradePlan(
            kind=OpportunityKind.STRUCTURAL_BUCKET,
            execution_style=ExecutionStyle.TAKER,
            contracts=5,
            edge_per_contract=0.10,
            expected_profit=0.50,
            capital_required=2.50,
            capital_required_by_venue={"venue0": 0.83, "venue1": 0.83, "venue2": 0.84},
            legs=(
                TradeLegPlan(venue="venue0", market_id="market0", side=Side.YES, contracts=5, limit_price=0.50),
                TradeLegPlan(venue="venue1", market_id="market1", side=Side.NO, contracts=5, limit_price=0.50),
                TradeLegPlan(venue="venue2", market_id="market2", side=Side.YES, contracts=5, limit_price=0.50),
            ),
        )
        r = sim.simulate_trade(opp, plan, all_fill_probability=0.5)
        assert len(r.leg_fills) == 3
        assert isinstance(r.simulated_pnl, float)

    def test_three_leg_legging_risk_higher(self):
        """More legs means higher probability of at least one failing."""
        legging_events_2 = 0
        legging_events_3 = 0
        n_trials = 500

        for i in range(n_trials):
            sim2 = MonteCarloSimulator(MonteCarloSettings(seed=i))
            opp2 = _make_opportunity(n_legs=2, edge=0.05)
            plan2 = _make_plan(contracts=10, edge=0.05)
            r2 = sim2.simulate_trade(opp2, plan2, all_fill_probability=0.7)
            if r2.filled_leg_count > 0 and not r2.all_legs_filled:
                legging_events_2 += 1

            sim3 = MonteCarloSimulator(MonteCarloSettings(seed=i))
            opp3 = _make_opportunity(n_legs=3, edge=0.05)
            plan3 = TradePlan(
                kind=OpportunityKind.STRUCTURAL_BUCKET,
                execution_style=ExecutionStyle.TAKER,
                contracts=10, edge_per_contract=0.05, expected_profit=0.50,
                capital_required=5.0,
                capital_required_by_venue={"venue0": 1.67, "venue1": 1.67, "venue2": 1.66},
                legs=(
                    TradeLegPlan(venue="venue0", market_id="market0", side=Side.YES, contracts=10, limit_price=0.50),
                    TradeLegPlan(venue="venue1", market_id="market1", side=Side.NO, contracts=10, limit_price=0.50),
                    TradeLegPlan(venue="venue2", market_id="market2", side=Side.YES, contracts=10, limit_price=0.50),
                ),
            )
            r3 = sim3.simulate_trade(opp3, plan3, all_fill_probability=0.7)
            if r3.filled_leg_count > 0 and not r3.all_legs_filled:
                legging_events_3 += 1

        # 3-leg should have more legging events than 2-leg.
        assert legging_events_3 >= legging_events_2


# ---------------------------------------------------------------------------
# Tests: MonteCarloResult components add up
# ---------------------------------------------------------------------------

class TestResultConsistency:
    def test_pnl_components_add_up(self):
        """PnL components should sum to simulated_pnl."""
        for seed in range(100):
            sim = MonteCarloSimulator(MonteCarloSettings(seed=seed))
            opp = _make_opportunity()
            plan = _make_plan()
            r = sim.simulate_trade(opp, plan, all_fill_probability=0.85)

            expected = (
                r.gross_edge_pnl
                - r.slippage_cost
                - r.adverse_selection_cost
                - r.legging_loss
                - r.edge_decay_cost
                + r.resolution_pnl
            )
            assert abs(r.simulated_pnl - expected) < 1e-10, (
                f"PnL mismatch at seed={seed}: {r.simulated_pnl} != {expected}"
            )

    def test_latency_nonnegative(self):
        """Simulated latency should always be non-negative."""
        for seed in range(100):
            sim = MonteCarloSimulator(MonteCarloSettings(seed=seed))
            opp = _make_opportunity()
            plan = _make_plan()
            r = sim.simulate_trade(opp, plan, all_fill_probability=1.0)
            assert r.simulated_latency_ms >= 0.0
