"""Tests for model calibration (Platt scaling) and probability blending."""

from __future__ import annotations

import math

import pytest

from arb_bot.crypto.calibration import ModelCalibrator, CalibrationBin
from arb_bot.crypto.edge_detector import blend_probabilities


# ── ModelCalibrator tests ──────────────────────────────────────────


class TestRecordOutcome:
    def test_record_outcome_tracks_predictions(self) -> None:
        cal = ModelCalibrator(min_samples_for_calibration=100)
        cal.record_outcome(0.7, True, timestamp=1.0, ticker="KXBTCD-T97000")
        cal.record_outcome(0.3, False, timestamp=2.0, ticker="KXBTCD-T98000")
        assert cal.num_records == 2
        assert not cal.is_calibrated


class TestBrierScore:
    def test_brier_score_perfect_predictions(self) -> None:
        """All predictions correct -> Brier score near 0."""
        cal = ModelCalibrator()
        # Predict 1.0 for outcomes that are True, 0.0 for False
        for _ in range(50):
            cal.record_outcome(1.0, True)
            cal.record_outcome(0.0, False)
        score = cal.compute_brier_score()
        assert score < 0.01

    def test_brier_score_random_predictions(self) -> None:
        """Predicting 0.5 for everything -> Brier score ~ 0.25."""
        cal = ModelCalibrator(min_samples_for_calibration=1000)
        for _ in range(100):
            cal.record_outcome(0.5, True)
            cal.record_outcome(0.5, False)
        score = cal.compute_brier_score()
        assert abs(score - 0.25) < 0.01


class TestPlattCalibration:
    def test_platt_not_calibrated_before_min_samples(self) -> None:
        cal = ModelCalibrator(min_samples_for_calibration=50)
        for i in range(49):
            cal.record_outcome(0.5, i % 2 == 0)
        assert not cal.is_calibrated

    def test_platt_calibration_after_min_samples(self) -> None:
        cal = ModelCalibrator(min_samples_for_calibration=50, recalibrate_every=1)
        # Feed well-calibrated data: high prob -> True, low prob -> False
        for i in range(60):
            if i < 30:
                cal.record_outcome(0.9, True)
            else:
                cal.record_outcome(0.1, False)
        assert cal.is_calibrated
        a, b = cal.platt_params
        # After fitting, params should have changed from defaults (1.0, 0.0)
        assert not (a == 1.0 and b == 0.0)

    def test_calibrate_identity_when_not_calibrated(self) -> None:
        """Before calibration, calibrate() should return raw prob."""
        cal = ModelCalibrator(min_samples_for_calibration=100)
        assert cal.calibrate(0.75) == 0.75
        assert cal.calibrate(0.10) == 0.10
        assert cal.calibrate(0.50) == 0.50


class TestCalibrationCurve:
    def test_calibration_curve_bins(self) -> None:
        cal = ModelCalibrator(min_samples_for_calibration=1000)
        # Create data across the probability spectrum
        for i in range(100):
            prob = i / 100.0
            outcome = prob > 0.5
            cal.record_outcome(prob, outcome)

        bins = cal.compute_calibration_curve(num_bins=5)
        assert len(bins) > 0
        for b in bins:
            assert 0.0 <= b.bin_lower < b.bin_upper <= 1.0
            assert b.count > 0
            assert 0.0 <= b.mean_predicted <= 1.0
            assert 0.0 <= b.mean_realized <= 1.0

    def test_calibration_curve_empty(self) -> None:
        cal = ModelCalibrator()
        bins = cal.compute_calibration_curve()
        assert bins == []


# ── Blend probabilities tests ─────────────────────────────────────


class TestBlendProbabilities:
    def test_blend_probabilities_high_confidence(self) -> None:
        """Low uncertainty -> model weight dominant."""
        result = blend_probabilities(
            model_prob=0.80,
            market_prob=0.50,
            model_uncertainty=0.01,  # Very tight CI
            base_model_weight=0.7,
        )
        # With low uncertainty, model should dominate
        # model_weight ~ 0.7 * exp(-5 * 0.01) ~ 0.7 * 0.951 ~ 0.666
        # blended ~ 0.666 * 0.80 + 0.334 * 0.50 ~ 0.700
        assert result > 0.65  # Closer to model's 0.80
        assert result < 0.85

    def test_blend_probabilities_low_confidence(self) -> None:
        """High uncertainty -> market weight dominant."""
        result = blend_probabilities(
            model_prob=0.80,
            market_prob=0.50,
            model_uncertainty=0.30,  # Very wide CI
            base_model_weight=0.7,
        )
        # With high uncertainty, market should dominate
        # model_weight ~ 0.7 * exp(-5 * 0.30) ~ 0.7 * 0.223 ~ 0.156
        # blended ~ 0.156 * 0.80 + 0.844 * 0.50 ~ 0.547
        assert result < 0.60  # Closer to market's 0.50
        assert result > 0.45

    def test_blend_50_50_at_moderate_uncertainty(self) -> None:
        """At moderate uncertainty with equal probs, blend stays near 0.5."""
        result = blend_probabilities(
            model_prob=0.50,
            market_prob=0.50,
            model_uncertainty=0.10,
        )
        assert abs(result - 0.50) < 0.01

    def test_blend_clamps_to_0_1(self) -> None:
        """Result should always be in [0, 1]."""
        result_low = blend_probabilities(0.0, 0.0, 0.0)
        result_high = blend_probabilities(1.0, 1.0, 0.0)
        assert result_low >= 0.0
        assert result_high <= 1.0

    def test_blend_zero_uncertainty_uses_base_weight(self) -> None:
        """At uncertainty=0, model gets exactly base_model_weight."""
        result = blend_probabilities(
            model_prob=1.0,
            market_prob=0.0,
            model_uncertainty=0.0,
            base_model_weight=0.7,
        )
        # model_weight = 0.7 * exp(0) = 0.7
        # blended = 0.7 * 1.0 + 0.3 * 0.0 = 0.7
        assert abs(result - 0.7) < 1e-10


# ── Market Brier score and Brier skill score tests ────────────────


class TestMarketBrierScore:
    def test_market_brier_score_computation(self) -> None:
        """Market Brier score uses market_implied_prob."""
        cal = ModelCalibrator(min_samples_for_calibration=1000)
        # Market predicts 0.8 for outcomes that are True -> (0.8 - 1.0)^2 = 0.04
        # Market predicts 0.2 for outcomes that are False -> (0.2 - 0.0)^2 = 0.04
        for _ in range(50):
            cal.record_outcome(0.9, True, market_implied_prob=0.8)
            cal.record_outcome(0.1, False, market_implied_prob=0.2)
        score = cal.compute_market_brier_score()
        assert abs(score - 0.04) < 0.01

    def test_brier_skill_positive_when_model_better(self) -> None:
        """Model well-calibrated, market random -> BSS > 0."""
        cal = ModelCalibrator(min_samples_for_calibration=1000)
        # Model predicts perfectly, market predicts 0.5
        for _ in range(50):
            cal.record_outcome(1.0, True, market_implied_prob=0.5)
            cal.record_outcome(0.0, False, market_implied_prob=0.5)
        bss = cal.compute_brier_skill_score()
        assert bss > 0  # Model beats market

    def test_brier_skill_negative_when_market_better(self) -> None:
        """Model miscalibrated, market accurate -> BSS < 0."""
        cal = ModelCalibrator(min_samples_for_calibration=1000)
        # Model always predicts 0.5, market predicts accurately
        for _ in range(50):
            cal.record_outcome(0.5, True, market_implied_prob=0.95)
            cal.record_outcome(0.5, False, market_implied_prob=0.05)
        bss = cal.compute_brier_skill_score()
        assert bss < 0  # Market beats model

    def test_market_brier_empty_records(self) -> None:
        """Empty calibrator returns 1.0."""
        cal = ModelCalibrator()
        assert cal.compute_market_brier_score() == 1.0

    def test_brier_skill_empty_records(self) -> None:
        """Empty calibrator returns 0.0."""
        cal = ModelCalibrator()
        assert cal.compute_brier_skill_score() == 0.0
