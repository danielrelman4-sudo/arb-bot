"""Tests for Phase 4E: Time-to-resolution capital lock penalty."""

from __future__ import annotations

import pytest

from arb_bot.framework.capital_lock import (
    CapitalLockConfig,
    CapitalLockPenalty,
    CapitalLockResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pen(**kw) -> CapitalLockPenalty:
    return CapitalLockPenalty(CapitalLockConfig(**kw))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self) -> None:
        cfg = CapitalLockConfig()
        assert cfg.penalty_per_hour == 0.001
        assert cfg.max_penalty_fraction == 0.50
        assert cfg.free_hours == 1.0
        assert cfg.use_opportunity_cost is True
        assert cfg.hours_per_year == 8760.0
        assert cfg.max_lock_hours == 720.0

    def test_custom(self) -> None:
        cfg = CapitalLockConfig(penalty_per_hour=0.005, max_lock_hours=24.0)
        assert cfg.penalty_per_hour == 0.005
        assert cfg.max_lock_hours == 24.0

    def test_frozen(self) -> None:
        cfg = CapitalLockConfig()
        with pytest.raises(AttributeError):
            cfg.penalty_per_hour = 0.01  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Compute — blocking
# ---------------------------------------------------------------------------


class TestComputeBlocking:
    def test_exceeds_max_lock(self) -> None:
        p = _pen(max_lock_hours=48.0)
        r = p.compute(kelly_fraction=0.20, hours_to_resolution=100.0)
        assert r.blocked is True
        assert r.block_reason == "exceeds_max_lock_hours"
        assert r.adjusted_fraction == 0.0

    def test_exactly_at_max_lock_not_blocked(self) -> None:
        p = _pen(max_lock_hours=48.0)
        r = p.compute(kelly_fraction=0.20, hours_to_resolution=48.0)
        assert r.blocked is False


# ---------------------------------------------------------------------------
# Compute — basic penalty
# ---------------------------------------------------------------------------


class TestComputeBasic:
    def test_short_duration_low_penalty(self) -> None:
        p = _pen(use_opportunity_cost=False, free_hours=0.0)
        r = p.compute(kelly_fraction=0.20, hours_to_resolution=2.0)
        # penalty = 2 * 0.001 = 0.002
        assert r.penalty_fraction == pytest.approx(0.002)
        assert r.adjusted_fraction == pytest.approx(0.20 * (1.0 - 0.002))

    def test_long_duration_high_penalty(self) -> None:
        p = _pen(use_opportunity_cost=False, free_hours=0.0)
        r = p.compute(kelly_fraction=0.20, hours_to_resolution=200.0)
        # penalty = 200 * 0.001 = 0.20
        assert r.penalty_fraction == pytest.approx(0.20)
        assert r.adjusted_fraction < 0.20

    def test_free_hours_exempt(self) -> None:
        p = _pen(use_opportunity_cost=False, free_hours=5.0)
        r = p.compute(kelly_fraction=0.20, hours_to_resolution=3.0)
        # 3 < 5 → effective_hours=0, penalty=0.
        assert r.penalty_fraction == 0.0
        assert r.adjusted_fraction == pytest.approx(0.20)

    def test_free_hours_partial(self) -> None:
        p = _pen(use_opportunity_cost=False, free_hours=5.0)
        r = p.compute(kelly_fraction=0.20, hours_to_resolution=8.0)
        # effective = 3 hours, penalty = 0.003.
        assert r.penalty_fraction == pytest.approx(0.003)

    def test_penalty_capped(self) -> None:
        p = _pen(
            use_opportunity_cost=False, free_hours=0.0,
            max_penalty_fraction=0.30, max_lock_hours=10000.0,
        )
        r = p.compute(kelly_fraction=0.20, hours_to_resolution=5000.0)
        # Raw penalty = 5000*0.001 = 5.0, capped to 0.30.
        assert r.total_penalty == pytest.approx(0.30)
        assert r.adjusted_fraction == pytest.approx(0.20 * 0.70)


# ---------------------------------------------------------------------------
# Opportunity cost
# ---------------------------------------------------------------------------


class TestOpportunityCost:
    def test_adds_to_penalty(self) -> None:
        p = _pen(use_opportunity_cost=True, free_hours=0.0)
        r = p.compute(kelly_fraction=0.20, hours_to_resolution=100.0,
                      annual_opportunity_rate=0.10)
        # Base = 100 * 0.001 = 0.10
        # Opp = 0.10 * (100 / 8760) ≈ 0.00114
        assert r.opportunity_cost_penalty > 0.0
        assert r.total_penalty > r.penalty_fraction

    def test_disabled(self) -> None:
        p = _pen(use_opportunity_cost=False)
        r = p.compute(kelly_fraction=0.20, hours_to_resolution=100.0,
                      annual_opportunity_rate=0.10)
        assert r.opportunity_cost_penalty == 0.0

    def test_higher_rate_more_penalty(self) -> None:
        p = _pen(use_opportunity_cost=True, free_hours=0.0)
        low = p.compute(kelly_fraction=0.20, hours_to_resolution=100.0,
                        annual_opportunity_rate=0.05)
        high = p.compute(kelly_fraction=0.20, hours_to_resolution=100.0,
                         annual_opportunity_rate=0.20)
        assert high.opportunity_cost_penalty > low.opportunity_cost_penalty


# ---------------------------------------------------------------------------
# Duration comparison
# ---------------------------------------------------------------------------


class TestDurationComparison:
    def test_shorter_gets_more_allocation(self) -> None:
        p = _pen()
        short = p.compute(kelly_fraction=0.20, hours_to_resolution=5.0)
        long = p.compute(kelly_fraction=0.20, hours_to_resolution=200.0)
        assert short.adjusted_fraction > long.adjusted_fraction

    def test_zero_hours(self) -> None:
        p = _pen(use_opportunity_cost=True, free_hours=0.0)
        r = p.compute(kelly_fraction=0.20, hours_to_resolution=0.0)
        # No base penalty, tiny opp cost (0).
        assert r.penalty_fraction == 0.0
        assert r.adjusted_fraction == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# Record actual resolution
# ---------------------------------------------------------------------------


class TestRecordActual:
    def test_records(self) -> None:
        p = _pen()
        p.record_actual_resolution("market_a", 24.0)
        p.record_actual_resolution("market_a", 30.0)
        assert len(p._resolution_actuals["market_a"]) == 2

    def test_independent_markets(self) -> None:
        p = _pen()
        p.record_actual_resolution("a", 10.0)
        p.record_actual_resolution("b", 20.0)
        assert len(p._resolution_actuals["a"]) == 1
        assert len(p._resolution_actuals["b"]) == 1


# ---------------------------------------------------------------------------
# Average penalty
# ---------------------------------------------------------------------------


class TestAvgPenalty:
    def test_empty(self) -> None:
        p = _pen()
        assert p.avg_penalty() == 0.0

    def test_populated(self) -> None:
        p = _pen(use_opportunity_cost=False, free_hours=0.0)
        p.compute(kelly_fraction=0.20, hours_to_resolution=10.0)
        p.compute(kelly_fraction=0.20, hours_to_resolution=20.0)
        avg = p.avg_penalty()
        # 0.01 + 0.02 = 0.03 / 2 = 0.015
        assert avg == pytest.approx(0.015)


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clears_all(self) -> None:
        p = _pen()
        p.compute(kelly_fraction=0.20, hours_to_resolution=10.0)
        p.record_actual_resolution("a", 5.0)
        p.clear()
        assert p.avg_penalty() == 0.0
        assert len(p._resolution_actuals) == 0


# ---------------------------------------------------------------------------
# Config property
# ---------------------------------------------------------------------------


class TestConfigProperty:
    def test_config_accessible(self) -> None:
        cfg = CapitalLockConfig(max_lock_hours=24.0)
        p = CapitalLockPenalty(cfg)
        assert p.config.max_lock_hours == 24.0


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_portfolio_of_durations(self) -> None:
        """Different markets penalized proportionally."""
        p = _pen()
        results = {}
        for name, hours in [("quick", 2.0), ("medium", 48.0), ("long", 500.0)]:
            r = p.compute(kelly_fraction=0.20, hours_to_resolution=hours)
            results[name] = r

        assert results["quick"].adjusted_fraction > results["medium"].adjusted_fraction
        assert results["medium"].adjusted_fraction > results["long"].adjusted_fraction
        assert all(not r.blocked for r in results.values())
