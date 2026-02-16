"""Tests for Phase 3C: Execution-cost completeness."""

from __future__ import annotations

import pytest

from arb_bot.framework.fee_model import (
    FeeDiscrepancy,
    FeeEstimate,
    FeeModel,
    FeeModelConfig,
    FeeReconciliationReport,
    OrderType,
    VenueFeeReconciliation,
    VenueFeeSchedule,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _kalshi_schedule(**kw: float) -> VenueFeeSchedule:
    defaults = dict(
        venue="kalshi",
        taker_fee_per_contract=0.07,
        maker_fee_per_contract=-0.02,
        taker_fee_rate=0.0,
        maker_fee_rate=0.0,
        settlement_fee_per_contract=0.0,
    )
    defaults.update(kw)
    return VenueFeeSchedule(**defaults)


def _poly_schedule(**kw: float) -> VenueFeeSchedule:
    defaults = dict(
        venue="polymarket",
        taker_fee_per_contract=0.0,
        maker_fee_per_contract=0.0,
        taker_fee_rate=0.02,
        maker_fee_rate=0.0,
        settlement_fee_per_contract=0.0,
    )
    defaults.update(kw)
    return VenueFeeSchedule(**defaults)


def _model(*schedules: VenueFeeSchedule, **kw) -> FeeModel:
    cfg = FeeModelConfig(venues=tuple(schedules), **kw)
    return FeeModel(cfg)


# ---------------------------------------------------------------------------
# VenueFeeSchedule
# ---------------------------------------------------------------------------


class TestVenueFeeSchedule:
    def test_defaults(self) -> None:
        s = VenueFeeSchedule(venue="test")
        assert s.taker_fee_per_contract == 0.0
        assert s.maker_fee_per_contract == 0.0
        assert s.settlement_fee_per_contract == 0.0

    def test_custom(self) -> None:
        s = _kalshi_schedule()
        assert s.venue == "kalshi"
        assert s.taker_fee_per_contract == 0.07
        assert s.maker_fee_per_contract == -0.02


# ---------------------------------------------------------------------------
# FeeModelConfig
# ---------------------------------------------------------------------------


class TestFeeModelConfig:
    def test_defaults(self) -> None:
        cfg = FeeModelConfig()
        assert cfg.venues == ()
        assert cfg.default_taker_fee == 0.0
        assert cfg.reconciliation_tolerance == 0.005

    def test_with_venues(self) -> None:
        cfg = FeeModelConfig(venues=(_kalshi_schedule(),))
        assert len(cfg.venues) == 1


# ---------------------------------------------------------------------------
# Fee estimation — flat fees
# ---------------------------------------------------------------------------


class TestFlatFees:
    def test_taker_flat(self) -> None:
        model = _model(_kalshi_schedule(taker_fee_per_contract=0.07))
        est = model.estimate("kalshi", OrderType.TAKER, contracts=10)
        assert est.per_contract_fee == pytest.approx(0.07)
        assert est.total_fee == pytest.approx(0.70)

    def test_maker_flat(self) -> None:
        model = _model(_kalshi_schedule(maker_fee_per_contract=-0.02))
        est = model.estimate("kalshi", OrderType.MAKER, contracts=10)
        assert est.per_contract_fee == pytest.approx(-0.02)
        assert est.total_fee == pytest.approx(-0.20)

    def test_zero_contracts(self) -> None:
        model = _model(_kalshi_schedule(taker_fee_per_contract=0.07))
        est = model.estimate("kalshi", OrderType.TAKER, contracts=0)
        assert est.total_fee == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Fee estimation — proportional fees
# ---------------------------------------------------------------------------


class TestProportionalFees:
    def test_taker_proportional(self) -> None:
        model = _model(_poly_schedule(taker_fee_rate=0.02))
        est = model.estimate("polymarket", OrderType.TAKER, contracts=10, price=0.50)
        # 0.02 * 0.50 * 10 = 0.10
        assert est.total_fee == pytest.approx(0.10)
        assert est.per_contract_fee == pytest.approx(0.01)

    def test_maker_proportional(self) -> None:
        model = _model(_poly_schedule(maker_fee_rate=0.01))
        est = model.estimate("polymarket", OrderType.MAKER, contracts=5, price=0.60)
        # 0.01 * 0.60 * 5 = 0.03
        assert est.total_fee == pytest.approx(0.03)


# ---------------------------------------------------------------------------
# Fee estimation — combined flat + proportional
# ---------------------------------------------------------------------------


class TestCombinedFees:
    def test_flat_plus_proportional(self) -> None:
        schedule = VenueFeeSchedule(
            venue="test",
            taker_fee_per_contract=0.05,
            taker_fee_rate=0.01,
        )
        model = _model(schedule)
        est = model.estimate("test", OrderType.TAKER, contracts=10, price=0.50)
        # Flat: 0.05 * 10 = 0.50
        # Prop: 0.01 * 0.50 * 10 = 0.05
        # Total: 0.55
        assert est.total_fee == pytest.approx(0.55)
        assert "flat" in est.breakdown
        assert "proportional" in est.breakdown


# ---------------------------------------------------------------------------
# Fee estimation — P*(1-P) curve fees (Kalshi-style)
# ---------------------------------------------------------------------------


class TestCurveFees:
    def test_kalshi_curve_at_50_cents(self) -> None:
        """Kalshi formula: ceil(0.035 * C * P * (1-P)) rounded up to cent."""
        schedule = VenueFeeSchedule(
            venue="kalshi",
            taker_curve_coefficient=0.035,
        )
        model = _model(schedule)
        est = model.estimate("kalshi", OrderType.TAKER, contracts=10, price=0.50)
        # 0.035 * 10 * 0.50 * 0.50 = 0.0875 → ceil to cent = 0.09
        assert est.total_fee == pytest.approx(0.09)
        assert est.breakdown["curve"] == pytest.approx(0.09)

    def test_kalshi_curve_at_extreme_price(self) -> None:
        """Near 0 or 1, P*(1-P) → 0 so fee → 0."""
        schedule = VenueFeeSchedule(
            venue="kalshi",
            taker_curve_coefficient=0.035,
        )
        model = _model(schedule)
        est = model.estimate("kalshi", OrderType.TAKER, contracts=10, price=0.99)
        # 0.035 * 10 * 0.99 * 0.01 = 0.003465 → ceil to cent = 0.01
        assert est.total_fee == pytest.approx(0.01)

    def test_kalshi_curve_at_low_price(self) -> None:
        schedule = VenueFeeSchedule(
            venue="kalshi",
            taker_curve_coefficient=0.035,
        )
        model = _model(schedule)
        est = model.estimate("kalshi", OrderType.TAKER, contracts=10, price=0.05)
        # 0.035 * 10 * 0.05 * 0.95 = 0.016625 → ceil to cent = 0.02
        assert est.total_fee == pytest.approx(0.02)

    def test_curve_disabled_when_zero(self) -> None:
        schedule = VenueFeeSchedule(
            venue="test",
            taker_curve_coefficient=0.0,
        )
        model = _model(schedule)
        est = model.estimate("test", OrderType.TAKER, contracts=10, price=0.50)
        assert est.breakdown["curve"] == pytest.approx(0.0)

    def test_curve_zero_at_boundary_prices(self) -> None:
        """Price exactly 0 or 1 should produce zero curve fee."""
        schedule = VenueFeeSchedule(
            venue="test",
            taker_curve_coefficient=0.035,
        )
        model = _model(schedule)
        est0 = model.estimate("test", OrderType.TAKER, contracts=10, price=0.0)
        est1 = model.estimate("test", OrderType.TAKER, contracts=10, price=1.0)
        assert est0.breakdown["curve"] == pytest.approx(0.0)
        assert est1.breakdown["curve"] == pytest.approx(0.0)

    def test_curve_no_round_up(self) -> None:
        schedule = VenueFeeSchedule(
            venue="test",
            taker_curve_coefficient=0.035,
            curve_round_up=False,
        )
        model = _model(schedule)
        est = model.estimate("test", OrderType.TAKER, contracts=10, price=0.50)
        # 0.035 * 10 * 0.50 * 0.50 = 0.0875 (exact, no rounding)
        assert est.total_fee == pytest.approx(0.0875)

    def test_curve_maker(self) -> None:
        schedule = VenueFeeSchedule(
            venue="test",
            maker_curve_coefficient=0.02,
        )
        model = _model(schedule)
        est = model.estimate("test", OrderType.MAKER, contracts=10, price=0.50)
        # 0.02 * 10 * 0.50 * 0.50 = 0.05
        assert est.total_fee == pytest.approx(0.05)

    def test_curve_plus_flat(self) -> None:
        """Curve fee combines with flat fee."""
        schedule = VenueFeeSchedule(
            venue="kalshi",
            taker_fee_per_contract=0.01,
            taker_curve_coefficient=0.035,
        )
        model = _model(schedule)
        est = model.estimate("kalshi", OrderType.TAKER, contracts=10, price=0.50)
        # Flat: 0.01 * 10 = 0.10
        # Curve: ceil(0.035 * 10 * 0.25 * 100) / 100 = ceil(8.75)/100 = 0.09
        # Total: 0.19
        assert est.total_fee == pytest.approx(0.19)

    def test_realistic_kalshi_fee_shape(self) -> None:
        """Fee should peak at P=0.50 and decrease toward extremes."""
        schedule = VenueFeeSchedule(
            venue="kalshi",
            taker_curve_coefficient=0.035,
            curve_round_up=False,
        )
        model = _model(schedule)
        fee_50 = model.estimate("kalshi", OrderType.TAKER, 10, price=0.50).total_fee
        fee_30 = model.estimate("kalshi", OrderType.TAKER, 10, price=0.30).total_fee
        fee_10 = model.estimate("kalshi", OrderType.TAKER, 10, price=0.10).total_fee
        fee_90 = model.estimate("kalshi", OrderType.TAKER, 10, price=0.90).total_fee
        # Peak at 50.
        assert fee_50 > fee_30
        assert fee_50 > fee_10
        # Symmetric: P=0.30 same as P=0.70, P=0.10 same as P=0.90.
        assert fee_10 == pytest.approx(fee_90)


# ---------------------------------------------------------------------------
# Fee estimation — min/max caps
# ---------------------------------------------------------------------------


class TestMinMaxCaps:
    def test_min_fee_floor(self) -> None:
        schedule = VenueFeeSchedule(
            venue="test",
            taker_fee_per_contract=0.001,
            min_fee_per_order=0.10,
        )
        model = _model(schedule)
        est = model.estimate("test", OrderType.TAKER, contracts=1)
        # Raw: 0.001 < min 0.10 → floored to 0.10.
        assert est.total_fee == pytest.approx(0.10)
        assert "min_floor_applied" in est.breakdown

    def test_max_fee_cap(self) -> None:
        schedule = VenueFeeSchedule(
            venue="test",
            taker_fee_per_contract=0.10,
            max_fee_per_order=0.50,
        )
        model = _model(schedule)
        est = model.estimate("test", OrderType.TAKER, contracts=100)
        # Raw: 0.10 * 100 = 10.0 > max 0.50 → capped.
        assert est.total_fee == pytest.approx(0.50)
        assert "max_cap_applied" in est.breakdown

    def test_no_cap_when_zero(self) -> None:
        schedule = VenueFeeSchedule(
            venue="test",
            taker_fee_per_contract=0.10,
            max_fee_per_order=0.0,  # Disabled.
        )
        model = _model(schedule)
        est = model.estimate("test", OrderType.TAKER, contracts=100)
        assert est.total_fee == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Settlement fees
# ---------------------------------------------------------------------------


class TestSettlementFees:
    def test_settlement_fee(self) -> None:
        schedule = VenueFeeSchedule(
            venue="test",
            taker_fee_per_contract=0.05,
            settlement_fee_per_contract=0.01,
        )
        model = _model(schedule)
        est = model.estimate("test", OrderType.TAKER, contracts=10)
        assert est.settlement_fee == pytest.approx(0.10)
        # Settlement is separate from trading fee.
        assert est.total_fee == pytest.approx(0.50)

    def test_no_settlement(self) -> None:
        model = _model(_kalshi_schedule())
        est = model.estimate("kalshi", OrderType.TAKER, contracts=10)
        assert est.settlement_fee == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Unknown venue fallback
# ---------------------------------------------------------------------------


class TestUnknownVenue:
    def test_fallback_taker(self) -> None:
        model = _model(default_taker_fee=0.05)
        est = model.estimate("unknown", OrderType.TAKER, contracts=10)
        assert est.total_fee == pytest.approx(0.50)

    def test_fallback_maker_zero(self) -> None:
        model = _model(default_taker_fee=0.05)
        est = model.estimate("unknown", OrderType.MAKER, contracts=10)
        assert est.total_fee == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Total cost convenience
# ---------------------------------------------------------------------------


class TestTotalCost:
    def test_price_plus_fee(self) -> None:
        model = _model(_kalshi_schedule(taker_fee_per_contract=0.07))
        total = model.total_cost("kalshi", OrderType.TAKER, contracts=10, price=0.55)
        assert total == pytest.approx(0.62)

    def test_maker_rebate_reduces_cost(self) -> None:
        model = _model(_kalshi_schedule(maker_fee_per_contract=-0.02))
        total = model.total_cost("kalshi", OrderType.MAKER, contracts=10, price=0.55)
        assert total == pytest.approx(0.53)


# ---------------------------------------------------------------------------
# Get schedule
# ---------------------------------------------------------------------------


class TestGetSchedule:
    def test_existing(self) -> None:
        model = _model(_kalshi_schedule())
        assert model.get_schedule("kalshi") is not None

    def test_missing(self) -> None:
        model = _model(_kalshi_schedule())
        assert model.get_schedule("unknown") is None


# ---------------------------------------------------------------------------
# Reconciliation — recording
# ---------------------------------------------------------------------------


class TestReconciliationRecording:
    def test_record_and_count(self) -> None:
        model = _model(_kalshi_schedule(taker_fee_per_contract=0.07))
        model.record_actual("kalshi", OrderType.TAKER, 10, actual_fee=0.70)
        assert model.reconciliation_count == 1

    def test_clear(self) -> None:
        model = _model(_kalshi_schedule(taker_fee_per_contract=0.07))
        model.record_actual("kalshi", OrderType.TAKER, 10, actual_fee=0.70)
        model.clear_reconciliation()
        assert model.reconciliation_count == 0


# ---------------------------------------------------------------------------
# Reconciliation — report
# ---------------------------------------------------------------------------


class TestReconciliationReport:
    def test_not_enough_samples(self) -> None:
        model = _model(_kalshi_schedule())
        assert model.reconciliation_report(min_samples=5) is None

    def test_perfect_reconciliation(self) -> None:
        model = _model(_kalshi_schedule(taker_fee_per_contract=0.07))
        for _ in range(10):
            model.record_actual("kalshi", OrderType.TAKER, 10, actual_fee=0.70)
        report = model.reconciliation_report()
        assert report is not None
        assert report.mean_error == pytest.approx(0.0)
        assert report.mae == pytest.approx(0.0)
        assert report.is_reconciled is True

    def test_systematic_over_estimate(self) -> None:
        model = _model(_kalshi_schedule(taker_fee_per_contract=0.07))
        for _ in range(10):
            # Model estimates 0.70 but actual is 0.60.
            model.record_actual("kalshi", OrderType.TAKER, 10, actual_fee=0.60)
        report = model.reconciliation_report()
        assert report.mean_error == pytest.approx(0.10)
        assert report.is_reconciled is False

    def test_systematic_under_estimate(self) -> None:
        model = _model(_kalshi_schedule(taker_fee_per_contract=0.07))
        for _ in range(10):
            model.record_actual("kalshi", OrderType.TAKER, 10, actual_fee=0.80)
        report = model.reconciliation_report()
        assert report.mean_error == pytest.approx(-0.10)
        assert report.is_reconciled is False

    def test_discrepancy_message(self) -> None:
        model = _model(
            _kalshi_schedule(taker_fee_per_contract=0.07),
            reconciliation_tolerance=0.005,
        )
        for _ in range(5):
            model.record_actual("kalshi", OrderType.TAKER, 10, actual_fee=0.60)
        report = model.reconciliation_report()
        assert len(report.discrepancies) == 1
        assert "over" in report.discrepancies[0].message

    def test_within_tolerance(self) -> None:
        model = _model(
            _kalshi_schedule(taker_fee_per_contract=0.07),
            reconciliation_tolerance=0.20,
        )
        for _ in range(5):
            model.record_actual("kalshi", OrderType.TAKER, 10, actual_fee=0.60)
        report = model.reconciliation_report()
        assert report.is_reconciled is True


# ---------------------------------------------------------------------------
# Reconciliation — per-venue
# ---------------------------------------------------------------------------


class TestPerVenueReconciliation:
    def test_multi_venue(self) -> None:
        model = _model(
            _kalshi_schedule(taker_fee_per_contract=0.07),
            _poly_schedule(taker_fee_rate=0.02),
        )
        for _ in range(5):
            model.record_actual("kalshi", OrderType.TAKER, 10, actual_fee=0.70)
        for _ in range(3):
            model.record_actual(
                "polymarket", OrderType.TAKER, 10,
                actual_fee=0.10, price=0.50,
            )
        report = model.reconciliation_report()
        assert "kalshi" in report.by_venue
        assert "polymarket" in report.by_venue
        assert report.by_venue["kalshi"].sample_count == 5
        assert report.by_venue["polymarket"].sample_count == 3

    def test_totals(self) -> None:
        model = _model(_kalshi_schedule(taker_fee_per_contract=0.07))
        model.record_actual("kalshi", OrderType.TAKER, 10, actual_fee=0.70)
        model.record_actual("kalshi", OrderType.TAKER, 5, actual_fee=0.35)
        report = model.reconciliation_report()
        assert report.total_estimated == pytest.approx(1.05)
        assert report.total_actual == pytest.approx(1.05)


# ---------------------------------------------------------------------------
# FeeEstimate
# ---------------------------------------------------------------------------


class TestFeeEstimate:
    def test_fields(self) -> None:
        est = FeeEstimate(
            venue="kalshi",
            order_type=OrderType.TAKER,
            contracts=10,
            per_contract_fee=0.07,
            total_fee=0.70,
            settlement_fee=0.0,
            breakdown={"flat": 0.70},
        )
        assert est.venue == "kalshi"
        assert est.total_fee == 0.70


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_cross_venue_cost_comparison(self) -> None:
        """Compare total execution cost across venues."""
        model = _model(
            _kalshi_schedule(taker_fee_per_contract=0.07, settlement_fee_per_contract=0.01),
            _poly_schedule(taker_fee_rate=0.02),
        )

        # Kalshi: 10 contracts at 0.55.
        k_est = model.estimate("kalshi", OrderType.TAKER, 10, price=0.55)
        k_total = k_est.total_fee + k_est.settlement_fee
        # Flat: 0.07 * 10 = 0.70, Settlement: 0.01 * 10 = 0.10 → 0.80
        assert k_total == pytest.approx(0.80)

        # Polymarket: 10 contracts at 0.45.
        p_est = model.estimate("polymarket", OrderType.TAKER, 10, price=0.45)
        p_total = p_est.total_fee + p_est.settlement_fee
        # Prop: 0.02 * 0.45 * 10 = 0.09
        assert p_total == pytest.approx(0.09)

    def test_maker_vs_taker(self) -> None:
        """Maker should be cheaper (or negative) vs taker."""
        model = _model(_kalshi_schedule(
            taker_fee_per_contract=0.07,
            maker_fee_per_contract=-0.02,
        ))
        taker = model.estimate("kalshi", OrderType.TAKER, 10)
        maker = model.estimate("kalshi", OrderType.MAKER, 10)
        assert maker.total_fee < taker.total_fee
        assert maker.total_fee < 0  # Rebate.

    def test_config_property(self) -> None:
        cfg = FeeModelConfig(default_taker_fee=0.05)
        model = FeeModel(cfg)
        assert model.config.default_taker_fee == 0.05
