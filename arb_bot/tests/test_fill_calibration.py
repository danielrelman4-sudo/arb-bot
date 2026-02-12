"""Tests for Phase 3B: Fill and slippage calibration."""

from __future__ import annotations

import pytest

from arb_bot.fill_calibration import (
    CalibrationConfig,
    CalibrationReport,
    FillCalibrator,
    KindCalibration,
    OutcomeRecord,
    PairedSample,
    PredictionRecord,
    VenueCalibration,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TS = 1000.0


def _prediction(
    intent_id: str = "i1",
    fill_rate: float = 0.9,
    slippage: float = 0.005,
    edge: float = 0.02,
    fill_prob: float = 0.8,
    contracts: int = 10,
    venue: str = "kalshi",
    kind: str = "cross_venue",
    ts: float = TS,
) -> PredictionRecord:
    return PredictionRecord(
        intent_id=intent_id,
        timestamp=ts,
        predicted_fill_rate=fill_rate,
        predicted_slippage=slippage,
        predicted_edge_per_contract=edge,
        predicted_fill_probability=fill_prob,
        contracts=contracts,
        venue=venue,
        kind=kind,
    )


def _outcome(
    intent_id: str = "i1",
    fill_rate: float = 0.8,
    slippage: float = 0.008,
    pnl_per: float = 0.015,
    filled: int = 8,
    planned: int = 10,
    venue: str = "kalshi",
    kind: str = "cross_venue",
    ts: float = TS + 5,
) -> OutcomeRecord:
    return OutcomeRecord(
        intent_id=intent_id,
        timestamp=ts,
        realized_fill_rate=fill_rate,
        realized_slippage=slippage,
        realized_pnl_per_contract=pnl_per,
        filled_contracts=filled,
        planned_contracts=planned,
        venue=venue,
        kind=kind,
    )


def _calibrator_with_samples(
    n: int = 10,
    pred_fill: float = 0.9,
    real_fill: float = 0.8,
    pred_slip: float = 0.005,
    real_slip: float = 0.008,
    pred_edge: float = 0.02,
    real_pnl: float = 0.015,
    pred_prob: float = 0.8,
    venue: str = "kalshi",
    kind: str = "cross_venue",
    config: CalibrationConfig | None = None,
) -> FillCalibrator:
    cal = FillCalibrator(config or CalibrationConfig(min_samples=1))
    for i in range(n):
        iid = f"i{i}"
        cal.record_prediction(_prediction(
            intent_id=iid, fill_rate=pred_fill, slippage=pred_slip,
            edge=pred_edge, fill_prob=pred_prob, venue=venue, kind=kind,
        ))
        cal.record_outcome(_outcome(
            intent_id=iid, fill_rate=real_fill, slippage=real_slip,
            pnl_per=real_pnl, venue=venue, kind=kind,
        ))
    return cal


# ---------------------------------------------------------------------------
# CalibrationConfig
# ---------------------------------------------------------------------------


class TestCalibrationConfig:
    def test_defaults(self) -> None:
        cfg = CalibrationConfig()
        assert cfg.window_size == 200
        assert cfg.min_samples == 10
        assert cfg.alert_fill_rate_bias == 0.15
        assert cfg.alert_slippage_bias == 0.005
        assert cfg.alert_pnl_bias == 0.01

    def test_custom(self) -> None:
        cfg = CalibrationConfig(window_size=50, min_samples=5)
        assert cfg.window_size == 50
        assert cfg.min_samples == 5


# ---------------------------------------------------------------------------
# PredictionRecord / OutcomeRecord
# ---------------------------------------------------------------------------


class TestRecords:
    def test_prediction_fields(self) -> None:
        p = _prediction()
        assert p.intent_id == "i1"
        assert p.predicted_fill_rate == 0.9
        assert p.venue == "kalshi"

    def test_outcome_fields(self) -> None:
        o = _outcome()
        assert o.intent_id == "i1"
        assert o.realized_fill_rate == 0.8
        assert o.realized_slippage == 0.008


# ---------------------------------------------------------------------------
# FillCalibrator — recording
# ---------------------------------------------------------------------------


class TestRecording:
    def test_record_and_pair(self) -> None:
        cal = FillCalibrator()
        cal.record_prediction(_prediction(intent_id="a"))
        assert cal.pending_count == 1
        assert cal.sample_count == 0

        matched = cal.record_outcome(_outcome(intent_id="a"))
        assert matched is True
        assert cal.pending_count == 0
        assert cal.sample_count == 1

    def test_no_match(self) -> None:
        cal = FillCalibrator()
        cal.record_prediction(_prediction(intent_id="a"))
        matched = cal.record_outcome(_outcome(intent_id="b"))
        assert matched is False
        assert cal.sample_count == 0
        assert cal.pending_count == 1

    def test_duplicate_prediction_overwrites(self) -> None:
        cal = FillCalibrator()
        cal.record_prediction(_prediction(intent_id="a", fill_rate=0.5))
        cal.record_prediction(_prediction(intent_id="a", fill_rate=0.9))
        assert cal.pending_count == 1
        cal.record_outcome(_outcome(intent_id="a"))
        assert cal.sample_count == 1

    def test_window_trimming(self) -> None:
        cfg = CalibrationConfig(window_size=5, min_samples=1)
        cal = FillCalibrator(cfg)
        for i in range(10):
            iid = f"i{i}"
            cal.record_prediction(_prediction(intent_id=iid))
            cal.record_outcome(_outcome(intent_id=iid))
        assert cal.sample_count == 5

    def test_clear(self) -> None:
        cal = FillCalibrator()
        cal.record_prediction(_prediction(intent_id="a"))
        cal.record_outcome(_outcome(intent_id="a"))
        cal.record_prediction(_prediction(intent_id="b"))
        cal.clear()
        assert cal.sample_count == 0
        assert cal.pending_count == 0


# ---------------------------------------------------------------------------
# FillCalibrator — purge stale
# ---------------------------------------------------------------------------


class TestPurgeStale:
    def test_purge_old(self) -> None:
        cal = FillCalibrator()
        cal.record_prediction(_prediction(intent_id="old", ts=100.0))
        cal.record_prediction(_prediction(intent_id="new", ts=200.0))
        purged = cal.purge_stale_predictions(before_timestamp=150.0)
        assert purged == 1
        assert cal.pending_count == 1

    def test_purge_nothing(self) -> None:
        cal = FillCalibrator()
        cal.record_prediction(_prediction(intent_id="a", ts=200.0))
        purged = cal.purge_stale_predictions(before_timestamp=100.0)
        assert purged == 0
        assert cal.pending_count == 1


# ---------------------------------------------------------------------------
# Calibration report — not enough samples
# ---------------------------------------------------------------------------


class TestNotEnoughSamples:
    def test_returns_none(self) -> None:
        cal = FillCalibrator(CalibrationConfig(min_samples=10))
        for i in range(5):
            iid = f"i{i}"
            cal.record_prediction(_prediction(intent_id=iid))
            cal.record_outcome(_outcome(intent_id=iid))
        report = cal.calibration_report()
        assert report is None


# ---------------------------------------------------------------------------
# Calibration report — fill rate
# ---------------------------------------------------------------------------


class TestFillRateCalibration:
    def test_unbiased(self) -> None:
        cal = _calibrator_with_samples(
            n=20, pred_fill=0.8, real_fill=0.8,
        )
        report = cal.calibration_report()
        assert report is not None
        assert report.fill_rate_bias == pytest.approx(0.0)
        assert report.fill_rate_mae == pytest.approx(0.0)

    def test_over_prediction(self) -> None:
        cal = _calibrator_with_samples(
            n=20, pred_fill=0.9, real_fill=0.7,
        )
        report = cal.calibration_report()
        assert report is not None
        assert report.fill_rate_bias == pytest.approx(0.2)
        assert report.fill_rate_mae == pytest.approx(0.2)

    def test_under_prediction(self) -> None:
        cal = _calibrator_with_samples(
            n=20, pred_fill=0.6, real_fill=0.9,
        )
        report = cal.calibration_report()
        assert report is not None
        assert report.fill_rate_bias == pytest.approx(-0.3)


# ---------------------------------------------------------------------------
# Calibration report — slippage
# ---------------------------------------------------------------------------


class TestSlippageCalibration:
    def test_unbiased(self) -> None:
        cal = _calibrator_with_samples(
            n=20, pred_slip=0.01, real_slip=0.01,
        )
        report = cal.calibration_report()
        assert report.slippage_bias == pytest.approx(0.0)

    def test_slippage_over_prediction(self) -> None:
        cal = _calibrator_with_samples(
            n=20, pred_slip=0.015, real_slip=0.005,
        )
        report = cal.calibration_report()
        assert report.slippage_bias == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# Calibration report — PnL
# ---------------------------------------------------------------------------


class TestPnLCalibration:
    def test_unbiased(self) -> None:
        cal = _calibrator_with_samples(
            n=20, pred_edge=0.02, real_pnl=0.02,
        )
        report = cal.calibration_report()
        assert report.pnl_bias == pytest.approx(0.0)

    def test_pnl_over_prediction(self) -> None:
        cal = _calibrator_with_samples(
            n=20, pred_edge=0.03, real_pnl=0.01,
        )
        report = cal.calibration_report()
        assert report.pnl_bias == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# Brier score
# ---------------------------------------------------------------------------


class TestBrierScore:
    def test_perfect_prediction(self) -> None:
        """All fills predicted at 1.0 and realized at 1.0 → Brier = 0."""
        cal = _calibrator_with_samples(
            n=20, pred_prob=1.0, real_fill=1.0,
        )
        report = cal.calibration_report()
        assert report.fill_probability_brier == pytest.approx(0.0)

    def test_worst_prediction(self) -> None:
        """Predicted 1.0 but never fills → Brier = 1.0."""
        cal = _calibrator_with_samples(
            n=20, pred_prob=1.0, real_fill=0.5,
        )
        report = cal.calibration_report()
        assert report.fill_probability_brier == pytest.approx(1.0)

    def test_moderate_prediction(self) -> None:
        """Predicted 0.5, half fill fully → Brier should be 0.25."""
        cfg = CalibrationConfig(min_samples=1)
        cal = FillCalibrator(cfg)
        # 10 that fill fully.
        for i in range(10):
            iid = f"fill_{i}"
            cal.record_prediction(_prediction(intent_id=iid, fill_prob=0.5))
            cal.record_outcome(_outcome(intent_id=iid, fill_rate=1.0))
        # 10 that don't fill fully.
        for i in range(10):
            iid = f"miss_{i}"
            cal.record_prediction(_prediction(intent_id=iid, fill_prob=0.5))
            cal.record_outcome(_outcome(intent_id=iid, fill_rate=0.5))
        report = cal.calibration_report()
        assert report.fill_probability_brier == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------


class TestAlerts:
    def test_no_alerts_when_calibrated(self) -> None:
        cal = _calibrator_with_samples(
            n=20, pred_fill=0.8, real_fill=0.8,
            pred_slip=0.005, real_slip=0.005,
            pred_edge=0.02, real_pnl=0.02,
        )
        report = cal.calibration_report()
        assert report.is_calibrated is True
        assert len(report.alerts) == 0

    def test_fill_rate_alert(self) -> None:
        cal = _calibrator_with_samples(
            n=20, pred_fill=0.9, real_fill=0.5,
            config=CalibrationConfig(
                min_samples=1, alert_fill_rate_bias=0.15,
                alert_slippage_bias=1.0, alert_pnl_bias=1.0,
            ),
        )
        report = cal.calibration_report()
        assert report.is_calibrated is False
        assert any(a.metric == "fill_rate" for a in report.alerts)

    def test_slippage_alert(self) -> None:
        cal = _calibrator_with_samples(
            n=20, pred_slip=0.02, real_slip=0.001,
            config=CalibrationConfig(
                min_samples=1, alert_fill_rate_bias=1.0,
                alert_slippage_bias=0.005, alert_pnl_bias=1.0,
            ),
        )
        report = cal.calibration_report()
        assert any(a.metric == "slippage" for a in report.alerts)

    def test_pnl_alert(self) -> None:
        cal = _calibrator_with_samples(
            n=20, pred_edge=0.05, real_pnl=0.01,
            config=CalibrationConfig(
                min_samples=1, alert_fill_rate_bias=1.0,
                alert_slippage_bias=1.0, alert_pnl_bias=0.01,
            ),
        )
        report = cal.calibration_report()
        assert any(a.metric == "pnl" for a in report.alerts)

    def test_multiple_alerts(self) -> None:
        cal = _calibrator_with_samples(
            n=20, pred_fill=0.9, real_fill=0.5,
            pred_slip=0.02, real_slip=0.001,
            pred_edge=0.05, real_pnl=0.01,
            config=CalibrationConfig(min_samples=1),
        )
        report = cal.calibration_report()
        assert len(report.alerts) == 3

    def test_under_prediction_alert_message(self) -> None:
        cal = _calibrator_with_samples(
            n=20, pred_fill=0.5, real_fill=0.9,
            config=CalibrationConfig(
                min_samples=1, alert_fill_rate_bias=0.15,
                alert_slippage_bias=1.0, alert_pnl_bias=1.0,
            ),
        )
        report = cal.calibration_report()
        alert = [a for a in report.alerts if a.metric == "fill_rate"][0]
        assert "under" in alert.message


# ---------------------------------------------------------------------------
# Per-venue breakdown
# ---------------------------------------------------------------------------


class TestVenueBreakdown:
    def test_single_venue(self) -> None:
        cal = _calibrator_with_samples(n=10, venue="kalshi")
        report = cal.calibration_report()
        assert "kalshi" in report.by_venue
        assert report.by_venue["kalshi"].sample_count == 10

    def test_multiple_venues(self) -> None:
        cfg = CalibrationConfig(min_samples=1)
        cal = FillCalibrator(cfg)
        for i in range(5):
            iid = f"k{i}"
            cal.record_prediction(_prediction(intent_id=iid, venue="kalshi"))
            cal.record_outcome(_outcome(intent_id=iid, venue="kalshi"))
        for i in range(3):
            iid = f"p{i}"
            cal.record_prediction(_prediction(intent_id=iid, venue="polymarket"))
            cal.record_outcome(_outcome(intent_id=iid, venue="polymarket"))
        report = cal.calibration_report()
        assert report.by_venue["kalshi"].sample_count == 5
        assert report.by_venue["polymarket"].sample_count == 3


# ---------------------------------------------------------------------------
# Per-kind breakdown
# ---------------------------------------------------------------------------


class TestKindBreakdown:
    def test_single_kind(self) -> None:
        cal = _calibrator_with_samples(n=10, kind="cross_venue")
        report = cal.calibration_report()
        assert "cross_venue" in report.by_kind
        assert report.by_kind["cross_venue"].sample_count == 10

    def test_multiple_kinds(self) -> None:
        cfg = CalibrationConfig(min_samples=1)
        cal = FillCalibrator(cfg)
        for i in range(4):
            iid = f"cv{i}"
            cal.record_prediction(_prediction(intent_id=iid, kind="cross_venue"))
            cal.record_outcome(_outcome(intent_id=iid, kind="cross_venue"))
        for i in range(6):
            iid = f"iv{i}"
            cal.record_prediction(_prediction(intent_id=iid, kind="intra_venue"))
            cal.record_outcome(_outcome(intent_id=iid, kind="intra_venue"))
        report = cal.calibration_report()
        assert report.by_kind["cross_venue"].sample_count == 4
        assert report.by_kind["intra_venue"].sample_count == 6


# ---------------------------------------------------------------------------
# CalibrationReport properties
# ---------------------------------------------------------------------------


class TestReportProperties:
    def test_is_calibrated(self) -> None:
        cal = _calibrator_with_samples(
            n=10, pred_fill=0.8, real_fill=0.8,
            pred_slip=0.005, real_slip=0.005,
            pred_edge=0.02, real_pnl=0.02,
        )
        report = cal.calibration_report()
        assert report.is_calibrated is True
        assert report.has_enough_samples is True

    def test_config_property(self) -> None:
        cfg = CalibrationConfig(window_size=50)
        cal = FillCalibrator(cfg)
        assert cal.config.window_size == 50


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_full_workflow(self) -> None:
        cfg = CalibrationConfig(
            window_size=100,
            min_samples=5,
            alert_fill_rate_bias=0.1,
            alert_slippage_bias=0.003,
            alert_pnl_bias=0.005,
        )
        cal = FillCalibrator(cfg)

        # Record some well-calibrated trades.
        for i in range(5):
            iid = f"good_{i}"
            cal.record_prediction(_prediction(
                intent_id=iid, fill_rate=0.85, slippage=0.006,
                edge=0.02, fill_prob=0.8, venue="kalshi",
            ))
            cal.record_outcome(_outcome(
                intent_id=iid, fill_rate=0.83, slippage=0.007,
                pnl_per=0.019, venue="kalshi",
            ))

        # Record some miscalibrated trades.
        for i in range(5):
            iid = f"bad_{i}"
            cal.record_prediction(_prediction(
                intent_id=iid, fill_rate=0.95, slippage=0.002,
                edge=0.03, fill_prob=0.9, venue="polymarket",
            ))
            cal.record_outcome(_outcome(
                intent_id=iid, fill_rate=0.60, slippage=0.015,
                pnl_per=0.005, venue="polymarket",
            ))

        report = cal.calibration_report()
        assert report is not None
        assert report.sample_count == 10
        assert report.has_enough_samples is True
        assert len(report.by_venue) == 2
        # Polymarket should show worse calibration.
        assert abs(report.by_venue["polymarket"].fill_rate_bias) > abs(
            report.by_venue["kalshi"].fill_rate_bias
        )

    def test_mixed_match_unmatched(self) -> None:
        cfg = CalibrationConfig(min_samples=1)
        cal = FillCalibrator(cfg)

        # Prediction with no outcome.
        cal.record_prediction(_prediction(intent_id="orphan"))

        # Matched pair.
        cal.record_prediction(_prediction(intent_id="matched"))
        cal.record_outcome(_outcome(intent_id="matched"))

        # Outcome with no prediction.
        cal.record_outcome(_outcome(intent_id="no_pred"))

        assert cal.sample_count == 1
        assert cal.pending_count == 1  # "orphan" still pending.
