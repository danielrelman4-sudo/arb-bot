"""Tests for isotonic calibration (A4)."""
import pytest
import numpy as np
from arb_bot.crypto.calibration import ModelCalibrator


class TestIsotonicFit:
    def test_pava_produces_monotonic_output(self):
        """PAVA should produce monotonically non-decreasing values."""
        cal = ModelCalibrator(
            min_samples_for_calibration=50,
            method="isotonic",
            isotonic_min_samples=5,
        )
        # Record samples: overconfident model (predicts high, outcomes are mixed)
        for i in range(20):
            cal.record_outcome(
                predicted_prob=0.3 + i * 0.02, outcome=bool(i % 3 == 0),
            )
        assert cal.is_calibrated
        # Check monotonicity
        assert cal._isotonic_x is not None
        for i in range(len(cal._isotonic_y) - 1):
            assert cal._isotonic_y[i] <= cal._isotonic_y[i + 1]

    def test_isotonic_identity_for_perfect_predictions(self):
        """Perfect predictions should produce near-identity calibration."""
        cal = ModelCalibrator(method="isotonic", isotonic_min_samples=5)
        for i in range(20):
            p = i / 20.0
            outcome = p > 0.5  # crude but consistent
            cal.record_outcome(predicted_prob=p, outcome=outcome)
        # Low probs should map low, high probs should map high
        assert cal.calibrate(0.1) < 0.5
        assert cal.calibrate(0.9) > 0.5

    def test_isotonic_handles_ties(self):
        """Same predicted prob with different outcomes should average."""
        cal = ModelCalibrator(method="isotonic", isotonic_min_samples=4)
        cal.record_outcome(0.5, True)
        cal.record_outcome(0.5, True)
        cal.record_outcome(0.5, False)
        cal.record_outcome(0.5, False)
        # Should calibrate 0.5 -> ~0.5
        result = cal.calibrate(0.5)
        assert 0.3 <= result <= 0.7

    def test_isotonic_overconfident_model_compressed(self):
        """Overconfident model should have probabilities compressed toward 0.5."""
        cal = ModelCalibrator(method="isotonic", isotonic_min_samples=5)
        # Model predicts 0.8-0.9 but wins only 50%
        for i in range(20):
            cal.record_outcome(
                predicted_prob=0.8 + 0.01 * i, outcome=bool(i % 2 == 0),
            )
        # 0.85 should be calibrated lower (toward actual 50%)
        assert cal.calibrate(0.85) < 0.8

    def test_isotonic_min_samples_respected(self):
        """Should not calibrate with fewer than min samples."""
        cal = ModelCalibrator(method="isotonic", isotonic_min_samples=10)
        for i in range(5):
            cal.record_outcome(predicted_prob=0.5, outcome=True)
        assert not cal.is_calibrated
        # raw prob returned unchanged
        assert cal.calibrate(0.7) == 0.7

    def test_isotonic_recalibrates_on_schedule(self):
        """Should re-fit after recalibrate_every new outcomes."""
        cal = ModelCalibrator(
            method="isotonic", isotonic_min_samples=5, recalibrate_every=5,
        )
        for i in range(10):
            cal.record_outcome(
                predicted_prob=0.3 + i * 0.05, outcome=bool(i > 5),
            )
        assert cal.is_calibrated
        old_x = cal._isotonic_x.copy() if cal._isotonic_x is not None else None
        # Add 5 more to trigger re-fit
        for i in range(5):
            cal.record_outcome(predicted_prob=0.6, outcome=True)
        # Should have re-fitted (breakpoints may differ)
        assert cal._isotonic_x is not None

    def test_isotonic_clamps_input(self):
        """Out-of-range inputs should be clamped."""
        cal = ModelCalibrator(
            method="isotonic", isotonic_min_samples=5, recalibrate_every=5,
        )
        for i in range(10):
            cal.record_outcome(
                predicted_prob=0.2 + i * 0.06, outcome=bool(i > 4),
            )
        result_low = cal.calibrate(-0.5)
        result_high = cal.calibrate(1.5)
        assert 0.0 <= result_low <= 1.0
        assert 0.0 <= result_high <= 1.0

    def test_platt_method_still_works(self):
        """Platt method should still work when selected."""
        cal = ModelCalibrator(
            method="platt", min_samples_for_calibration=5, recalibrate_every=5,
        )
        for i in range(10):
            cal.record_outcome(predicted_prob=0.5, outcome=bool(i % 2 == 0))
        assert cal.is_calibrated
        # Should use Platt (not isotonic)
        assert cal._isotonic_x is None

    def test_calibrate_returns_raw_when_uncalibrated(self):
        """Before calibration, should return raw prob."""
        cal = ModelCalibrator(method="isotonic", isotonic_min_samples=100)
        assert cal.calibrate(0.73) == 0.73

    def test_interpolation_between_breakpoints(self):
        """Values between breakpoints should be linearly interpolated."""
        cal = ModelCalibrator(method="isotonic", isotonic_min_samples=4)
        # Create a simple mapping: low predictions -> 0, high predictions -> 1
        for _ in range(5):
            cal.record_outcome(predicted_prob=0.2, outcome=False)
        for _ in range(5):
            cal.record_outcome(predicted_prob=0.8, outcome=True)
        # Midpoint should interpolate
        mid = cal.calibrate(0.5)
        assert 0.1 < mid < 0.9  # should be somewhere in between
