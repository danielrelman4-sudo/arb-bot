"""Tests for Phase 4A: Tail-risk-aware Kelly sizing."""

from __future__ import annotations

import math

import pytest

from arb_bot.framework.kelly_sizing import (
    KellySizingResult,
    TailRiskKelly,
    TailRiskKellyConfig,
    _raw_kelly,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sizer(**kw) -> TailRiskKelly:
    return TailRiskKelly(TailRiskKellyConfig(**kw))


# Edge must be large relative to cost for positive Kelly.
# b = edge/cost, need b*p > a*q where a=failure_loss/cost, p=fill_prob, q=1-p.
# With a=1, p=0.8, q=0.2: need b > 0.25, i.e. edge > 0.25*cost.
EDGE = 0.30
COST = 0.10
FILL = 0.8


# ---------------------------------------------------------------------------
# TailRiskKellyConfig
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self) -> None:
        cfg = TailRiskKellyConfig()
        assert cfg.base_kelly_fraction == 0.25
        assert cfg.uncertainty_haircut_factor == 1.0
        assert cfg.variance_haircut_factor == 0.5
        assert cfg.min_confidence == 0.1
        assert cfg.max_model_uncertainty == 0.8
        assert cfg.lane_variance_window == 50

    def test_custom(self) -> None:
        cfg = TailRiskKellyConfig(base_kelly_fraction=0.5, min_confidence=0.2)
        assert cfg.base_kelly_fraction == 0.5
        assert cfg.min_confidence == 0.2

    def test_frozen(self) -> None:
        cfg = TailRiskKellyConfig()
        with pytest.raises(AttributeError):
            cfg.base_kelly_fraction = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Raw Kelly helper
# ---------------------------------------------------------------------------


class TestRawKelly:
    def test_positive_edge(self) -> None:
        raw = _raw_kelly(edge=EDGE, cost=COST, fill_prob=1.0)
        assert raw > 0.0

    def test_zero_edge(self) -> None:
        assert _raw_kelly(edge=0.0, cost=COST, fill_prob=1.0) == 0.0

    def test_negative_edge(self) -> None:
        assert _raw_kelly(edge=-0.01, cost=COST, fill_prob=1.0) == 0.0

    def test_zero_cost(self) -> None:
        assert _raw_kelly(edge=EDGE, cost=0.0, fill_prob=1.0) == 0.0

    def test_fill_prob_clamp(self) -> None:
        # fill_prob > 1 clamped to 1.
        r1 = _raw_kelly(edge=EDGE, cost=COST, fill_prob=1.0)
        r2 = _raw_kelly(edge=EDGE, cost=COST, fill_prob=1.5)
        assert r1 == pytest.approx(r2)

    def test_fill_prob_zero(self) -> None:
        raw = _raw_kelly(edge=EDGE, cost=COST, fill_prob=0.0)
        assert raw == 0.0

    def test_low_fill_prob_reduces(self) -> None:
        high = _raw_kelly(edge=EDGE, cost=COST, fill_prob=0.95)
        low = _raw_kelly(edge=EDGE, cost=COST, fill_prob=0.5)
        assert high > low

    def test_custom_failure_loss(self) -> None:
        # Smaller failure loss → larger fraction.
        full = _raw_kelly(edge=EDGE, cost=COST, fill_prob=FILL, failure_loss=COST)
        half = _raw_kelly(edge=EDGE, cost=COST, fill_prob=FILL, failure_loss=COST * 0.5)
        assert half >= full

    def test_result_bounded(self) -> None:
        raw = _raw_kelly(edge=0.50, cost=0.10, fill_prob=1.0)
        assert 0.0 <= raw <= 1.0


# ---------------------------------------------------------------------------
# Compute — blocking conditions
# ---------------------------------------------------------------------------


class TestComputeBlocking:
    def test_high_uncertainty_blocks(self) -> None:
        sizer = _sizer(max_model_uncertainty=0.8)
        result = sizer.compute(edge=EDGE, cost=COST, fill_prob=FILL, model_uncertainty=0.9)
        assert result.blocked is True
        assert result.block_reason == "model_uncertainty_too_high"
        assert result.adjusted_fraction == 0.0

    def test_low_confidence_blocks(self) -> None:
        sizer = _sizer(min_confidence=0.5, max_model_uncertainty=0.9)
        # uncertainty = 0.55 → confidence = 0.45 < 0.5.
        result = sizer.compute(edge=EDGE, cost=COST, fill_prob=FILL, model_uncertainty=0.55)
        assert result.blocked is True
        assert result.block_reason == "confidence_below_minimum"

    def test_exactly_at_max_uncertainty(self) -> None:
        sizer = _sizer(max_model_uncertainty=0.8)
        result = sizer.compute(edge=EDGE, cost=COST, fill_prob=FILL, model_uncertainty=0.8)
        # 0.8 is NOT > 0.8, so should not be blocked by uncertainty gate.
        assert result.block_reason != "model_uncertainty_too_high"

    def test_confidence_just_below_min(self) -> None:
        sizer = _sizer(min_confidence=0.5, max_model_uncertainty=0.9)
        # uncertainty = 0.55 → confidence = 0.45 < 0.5.
        result = sizer.compute(edge=EDGE, cost=COST, fill_prob=FILL, model_uncertainty=0.55)
        assert result.blocked is True
        assert result.block_reason == "confidence_below_minimum"

    def test_negative_edge_blocks(self) -> None:
        sizer = _sizer()
        result = sizer.compute(edge=-0.01, cost=COST, fill_prob=FILL)
        assert result.blocked is True
        assert result.block_reason == "negative_edge"

    def test_edge_too_small_relative_to_cost(self) -> None:
        sizer = _sizer()
        # edge=0.01, cost=0.50 → b=0.02, way too small.
        result = sizer.compute(edge=0.01, cost=0.50, fill_prob=FILL)
        assert result.blocked is True
        assert result.block_reason == "negative_edge"


# ---------------------------------------------------------------------------
# Compute — successful sizing
# ---------------------------------------------------------------------------


class TestComputeSuccess:
    def test_zero_uncertainty(self) -> None:
        sizer = _sizer()
        result = sizer.compute(edge=EDGE, cost=COST, fill_prob=FILL, model_uncertainty=0.0)
        assert result.blocked is False
        assert result.adjusted_fraction > 0.0
        assert result.uncertainty_haircut == 0.0
        assert result.confidence == 1.0

    def test_moderate_uncertainty_reduces(self) -> None:
        sizer = _sizer()
        no_unc = sizer.compute(edge=EDGE, cost=COST, fill_prob=FILL, model_uncertainty=0.0)
        with_unc = sizer.compute(edge=EDGE, cost=COST, fill_prob=FILL, model_uncertainty=0.3)
        assert with_unc.adjusted_fraction < no_unc.adjusted_fraction
        # Baker-McHale shrinkage: haircut is > 0 for non-zero uncertainty.
        assert with_unc.uncertainty_haircut > 0.0

    def test_fractional_kelly_cap(self) -> None:
        sizer = _sizer(base_kelly_fraction=0.25)
        result = sizer.compute(edge=0.50, cost=0.10, fill_prob=1.0, model_uncertainty=0.0)
        # Raw kelly should be high, but capped at 0.25.
        assert result.adjusted_fraction <= 0.25

    def test_fraction_bounded_0_1(self) -> None:
        sizer = _sizer()
        result = sizer.compute(edge=EDGE, cost=COST, fill_prob=FILL, model_uncertainty=0.0)
        assert 0.0 <= result.adjusted_fraction <= 1.0

    def test_raw_kelly_populated(self) -> None:
        sizer = _sizer()
        result = sizer.compute(edge=EDGE, cost=COST, fill_prob=FILL)
        assert result.raw_kelly > 0.0


# ---------------------------------------------------------------------------
# Variance haircut
# ---------------------------------------------------------------------------


class TestVarianceHaircut:
    def test_no_outcomes_no_haircut(self) -> None:
        sizer = _sizer()
        result = sizer.compute(edge=EDGE, cost=COST, fill_prob=FILL, lane="lane_a")
        assert result.variance_haircut == 0.0

    def test_low_variance_small_haircut(self) -> None:
        sizer = _sizer(variance_haircut_factor=0.5)
        # Record uniform outcomes → low variance.
        for _ in range(20):
            sizer.record_outcome("lane_a", 0.01)
        result = sizer.compute(edge=EDGE, cost=COST, fill_prob=FILL, lane="lane_a")
        assert result.variance_haircut < 0.01

    def test_high_variance_large_haircut(self) -> None:
        sizer = _sizer(variance_haircut_factor=0.5)
        # Record volatile outcomes → high variance.
        for i in range(20):
            sizer.record_outcome("lane_a", 1.0 if i % 2 == 0 else -1.0)
        result = sizer.compute(edge=EDGE, cost=COST, fill_prob=FILL, lane="lane_a")
        assert result.variance_haircut > 0.1

    def test_variance_reduces_fraction(self) -> None:
        sizer = _sizer(variance_haircut_factor=1.0)
        base = sizer.compute(edge=EDGE, cost=COST, fill_prob=FILL, lane="lane_a")
        # Add volatile outcomes.
        for i in range(20):
            sizer.record_outcome("lane_a", 1.0 if i % 2 == 0 else -1.0)
        with_var = sizer.compute(edge=EDGE, cost=COST, fill_prob=FILL, lane="lane_a")
        assert with_var.adjusted_fraction < base.adjusted_fraction

    def test_different_lanes_independent(self) -> None:
        sizer = _sizer()
        for i in range(20):
            sizer.record_outcome("volatile", 1.0 if i % 2 == 0 else -1.0)
        for _ in range(20):
            sizer.record_outcome("stable", 0.01)
        r_vol = sizer.compute(edge=EDGE, cost=COST, fill_prob=FILL, lane="volatile")
        r_stb = sizer.compute(edge=EDGE, cost=COST, fill_prob=FILL, lane="stable")
        assert r_vol.variance_haircut > r_stb.variance_haircut


# ---------------------------------------------------------------------------
# Record outcome and window
# ---------------------------------------------------------------------------


class TestRecordOutcome:
    def test_records_and_trims(self) -> None:
        sizer = _sizer(lane_variance_window=5)
        for i in range(10):
            sizer.record_outcome("lane_a", float(i))
        # Only last 5 should be kept.
        assert len(sizer._lane_outcomes["lane_a"]) == 5
        assert sizer._lane_outcomes["lane_a"][0] == 5.0

    def test_single_outcome_no_variance(self) -> None:
        sizer = _sizer()
        sizer.record_outcome("lane_a", 0.05)
        result = sizer.compute(edge=EDGE, cost=COST, fill_prob=FILL, lane="lane_a")
        assert result.variance_haircut == 0.0


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clears_all_lanes(self) -> None:
        sizer = _sizer()
        sizer.record_outcome("a", 0.1)
        sizer.record_outcome("b", 0.2)
        sizer.clear()
        assert len(sizer._lane_outcomes) == 0


# ---------------------------------------------------------------------------
# Config property
# ---------------------------------------------------------------------------


class TestConfigProperty:
    def test_config_accessible(self) -> None:
        cfg = TailRiskKellyConfig(base_kelly_fraction=0.3)
        sizer = TailRiskKelly(cfg)
        assert sizer.config.base_kelly_fraction == 0.3


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_realistic_scenario(self) -> None:
        """Simulate realistic sizing across multiple lanes."""
        sizer = _sizer(
            base_kelly_fraction=0.25,
            uncertainty_haircut_factor=1.0,
            variance_haircut_factor=0.5,
            min_confidence=0.1,
            max_model_uncertainty=0.8,
        )

        # Lane with good track record: stable outcomes.
        for _ in range(30):
            sizer.record_outcome("cross_venue", 0.02)

        # Lane with volatile track record.
        for i in range(30):
            sizer.record_outcome("parity", 0.05 if i % 2 == 0 else -0.03)

        # High-confidence opportunity on stable lane.
        r1 = sizer.compute(
            edge=EDGE, cost=COST, fill_prob=FILL,
            model_uncertainty=0.1, lane="cross_venue",
        )
        assert r1.blocked is False
        assert r1.adjusted_fraction > 0.0

        # Same opportunity but low confidence.
        r2 = sizer.compute(
            edge=EDGE, cost=COST, fill_prob=FILL,
            model_uncertainty=0.5, lane="cross_venue",
        )
        assert r2.adjusted_fraction < r1.adjusted_fraction

        # Same opportunity on volatile lane.
        r3 = sizer.compute(
            edge=EDGE, cost=COST, fill_prob=FILL,
            model_uncertainty=0.1, lane="parity",
        )
        assert r3.adjusted_fraction < r1.adjusted_fraction

        # Blocked by high uncertainty.
        r4 = sizer.compute(
            edge=EDGE, cost=COST, fill_prob=FILL,
            model_uncertainty=0.85,
        )
        assert r4.blocked is True

    def test_default_config(self) -> None:
        sizer = TailRiskKelly()
        assert sizer.config.base_kelly_fraction == 0.25
