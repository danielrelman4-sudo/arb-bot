"""Fill and slippage calibration (Phase 3B).

Compares predicted fill/slippage estimates from the execution model
against realized outcomes from the analytics store.  Produces
calibration reports that indicate whether the model is systematically
over- or under-estimating fills and costs.

Usage::

    calibrator = FillCalibrator(config)
    calibrator.record_prediction(prediction)
    calibrator.record_outcome(outcome)
    report = calibrator.calibration_report()
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationConfig:
    """Configuration for fill calibration.

    Parameters
    ----------
    window_size:
        Maximum number of prediction/outcome pairs to keep in the
        rolling window for calibration. Default 200.
    min_samples:
        Minimum number of paired samples required before producing
        a calibration report. Default 10.
    alert_fill_rate_bias:
        Alert threshold for absolute fill-rate bias. If the mean
        (predicted - realized) fill rate exceeds this, the model is
        miscalibrated. Default 0.15.
    alert_slippage_bias:
        Alert threshold for absolute slippage bias. Default 0.005.
    alert_pnl_bias:
        Alert threshold for absolute PnL bias per contract. Default 0.01.
    """

    window_size: int = 200
    min_samples: int = 10
    alert_fill_rate_bias: float = 0.15
    alert_slippage_bias: float = 0.005
    alert_pnl_bias: float = 0.01


# ---------------------------------------------------------------------------
# Prediction / outcome records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PredictionRecord:
    """A prediction from the execution model at decision time."""

    intent_id: str
    timestamp: float
    predicted_fill_rate: float
    predicted_slippage: float
    predicted_edge_per_contract: float
    predicted_fill_probability: float
    contracts: int
    venue: str = ""
    kind: str = ""


@dataclass(frozen=True)
class OutcomeRecord:
    """A realized outcome after execution."""

    intent_id: str
    timestamp: float
    realized_fill_rate: float
    realized_slippage: float
    realized_pnl_per_contract: float
    filled_contracts: int
    planned_contracts: int
    venue: str = ""
    kind: str = ""


# ---------------------------------------------------------------------------
# Paired sample
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairedSample:
    """A matched prediction-outcome pair."""

    intent_id: str
    predicted_fill_rate: float
    realized_fill_rate: float
    predicted_slippage: float
    realized_slippage: float
    predicted_edge: float
    realized_pnl: float
    predicted_fill_probability: float
    venue: str
    kind: str


# ---------------------------------------------------------------------------
# Calibration report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationAlert:
    """A calibration alert indicating model miscalibration."""

    metric: str
    bias: float
    threshold: float
    message: str


@dataclass(frozen=True)
class CalibrationReport:
    """Summary of model calibration quality."""

    sample_count: int
    fill_rate_bias: float
    fill_rate_mae: float
    slippage_bias: float
    slippage_mae: float
    pnl_bias: float
    pnl_mae: float
    fill_probability_brier: float
    alerts: tuple[CalibrationAlert, ...]
    by_venue: Dict[str, "VenueCalibration"]
    by_kind: Dict[str, "KindCalibration"]

    @property
    def is_calibrated(self) -> bool:
        return len(self.alerts) == 0

    @property
    def has_enough_samples(self) -> bool:
        return self.sample_count > 0


@dataclass(frozen=True)
class VenueCalibration:
    """Per-venue calibration stats."""

    venue: str
    sample_count: int
    fill_rate_bias: float
    slippage_bias: float


@dataclass(frozen=True)
class KindCalibration:
    """Per-kind (strategy type) calibration stats."""

    kind: str
    sample_count: int
    fill_rate_bias: float
    slippage_bias: float


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------


class FillCalibrator:
    """Rolling calibration tracker for fill/slippage predictions.

    Records prediction-outcome pairs and computes bias and error metrics.
    """

    def __init__(self, config: CalibrationConfig | None = None) -> None:
        self._config = config or CalibrationConfig()
        self._predictions: Dict[str, PredictionRecord] = {}
        self._paired: list[PairedSample] = []

    @property
    def config(self) -> CalibrationConfig:
        return self._config

    @property
    def sample_count(self) -> int:
        return len(self._paired)

    @property
    def pending_count(self) -> int:
        return len(self._predictions)

    def record_prediction(self, prediction: PredictionRecord) -> None:
        """Record a prediction for later matching with an outcome."""
        self._predictions[prediction.intent_id] = prediction

    def record_outcome(self, outcome: OutcomeRecord) -> bool:
        """Record an outcome and pair it with its prediction.

        Returns True if a matching prediction was found.
        """
        prediction = self._predictions.pop(outcome.intent_id, None)
        if prediction is None:
            return False

        planned = max(1, outcome.planned_contracts)
        realized_pnl_per = outcome.realized_pnl_per_contract

        sample = PairedSample(
            intent_id=outcome.intent_id,
            predicted_fill_rate=prediction.predicted_fill_rate,
            realized_fill_rate=outcome.realized_fill_rate,
            predicted_slippage=prediction.predicted_slippage,
            realized_slippage=outcome.realized_slippage,
            predicted_edge=prediction.predicted_edge_per_contract,
            realized_pnl=realized_pnl_per,
            predicted_fill_probability=prediction.predicted_fill_probability,
            venue=prediction.venue or outcome.venue,
            kind=prediction.kind or outcome.kind,
        )

        self._paired.append(sample)

        # Trim to window size.
        if len(self._paired) > self._config.window_size:
            self._paired = self._paired[-self._config.window_size:]

        return True

    def calibration_report(self) -> CalibrationReport | None:
        """Compute calibration report from paired samples.

        Returns None if fewer than min_samples pairs are available.
        """
        cfg = self._config
        if len(self._paired) < cfg.min_samples:
            return None

        samples = self._paired

        # Fill rate metrics.
        fill_rate_errors = [
            s.predicted_fill_rate - s.realized_fill_rate for s in samples
        ]
        fill_rate_bias = statistics.mean(fill_rate_errors)
        fill_rate_mae = statistics.mean(abs(e) for e in fill_rate_errors)

        # Slippage metrics.
        slippage_errors = [
            s.predicted_slippage - s.realized_slippage for s in samples
        ]
        slippage_bias = statistics.mean(slippage_errors)
        slippage_mae = statistics.mean(abs(e) for e in slippage_errors)

        # PnL metrics.
        pnl_errors = [
            s.predicted_edge - s.realized_pnl for s in samples
        ]
        pnl_bias = statistics.mean(pnl_errors)
        pnl_mae = statistics.mean(abs(e) for e in pnl_errors)

        # Brier score for fill probability (binary: did it fill fully?).
        brier_scores = [
            (s.predicted_fill_probability - (1.0 if s.realized_fill_rate >= 1.0 else 0.0)) ** 2
            for s in samples
        ]
        fill_probability_brier = statistics.mean(brier_scores)

        # Alerts.
        alerts: list[CalibrationAlert] = []
        if abs(fill_rate_bias) > cfg.alert_fill_rate_bias:
            direction = "over" if fill_rate_bias > 0 else "under"
            alerts.append(CalibrationAlert(
                metric="fill_rate",
                bias=fill_rate_bias,
                threshold=cfg.alert_fill_rate_bias,
                message=f"Fill rate model {direction}-predicts by {abs(fill_rate_bias):.3f}",
            ))
        if abs(slippage_bias) > cfg.alert_slippage_bias:
            direction = "over" if slippage_bias > 0 else "under"
            alerts.append(CalibrationAlert(
                metric="slippage",
                bias=slippage_bias,
                threshold=cfg.alert_slippage_bias,
                message=f"Slippage model {direction}-predicts by {abs(slippage_bias):.4f}",
            ))
        if abs(pnl_bias) > cfg.alert_pnl_bias:
            direction = "over" if pnl_bias > 0 else "under"
            alerts.append(CalibrationAlert(
                metric="pnl",
                bias=pnl_bias,
                threshold=cfg.alert_pnl_bias,
                message=f"PnL model {direction}-predicts by {abs(pnl_bias):.4f}/contract",
            ))

        # Per-venue breakdown.
        by_venue = self._group_by_venue(samples)
        by_kind = self._group_by_kind(samples)

        return CalibrationReport(
            sample_count=len(samples),
            fill_rate_bias=fill_rate_bias,
            fill_rate_mae=fill_rate_mae,
            slippage_bias=slippage_bias,
            slippage_mae=slippage_mae,
            pnl_bias=pnl_bias,
            pnl_mae=pnl_mae,
            fill_probability_brier=fill_probability_brier,
            alerts=tuple(alerts),
            by_venue=by_venue,
            by_kind=by_kind,
        )

    def clear(self) -> None:
        """Clear all predictions and paired samples."""
        self._predictions.clear()
        self._paired.clear()

    def purge_stale_predictions(self, before_timestamp: float) -> int:
        """Remove predictions older than the given timestamp.

        Returns the number of predictions purged.
        """
        stale = [
            k for k, v in self._predictions.items()
            if v.timestamp < before_timestamp
        ]
        for k in stale:
            del self._predictions[k]
        return len(stale)

    def _group_by_venue(self, samples: list[PairedSample]) -> Dict[str, VenueCalibration]:
        groups: Dict[str, list[PairedSample]] = {}
        for s in samples:
            venue = s.venue or "unknown"
            groups.setdefault(venue, []).append(s)

        result: Dict[str, VenueCalibration] = {}
        for venue, group in groups.items():
            fr_errors = [s.predicted_fill_rate - s.realized_fill_rate for s in group]
            sl_errors = [s.predicted_slippage - s.realized_slippage for s in group]
            result[venue] = VenueCalibration(
                venue=venue,
                sample_count=len(group),
                fill_rate_bias=statistics.mean(fr_errors),
                slippage_bias=statistics.mean(sl_errors),
            )
        return result

    def _group_by_kind(self, samples: list[PairedSample]) -> Dict[str, KindCalibration]:
        groups: Dict[str, list[PairedSample]] = {}
        for s in samples:
            kind = s.kind or "unknown"
            groups.setdefault(kind, []).append(s)

        result: Dict[str, KindCalibration] = {}
        for kind, group in groups.items():
            fr_errors = [s.predicted_fill_rate - s.realized_fill_rate for s in group]
            sl_errors = [s.predicted_slippage - s.realized_slippage for s in group]
            result[kind] = KindCalibration(
                kind=kind,
                sample_count=len(group),
                fill_rate_bias=statistics.mean(fr_errors),
                slippage_bias=statistics.mean(sl_errors),
            )
        return result
