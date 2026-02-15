"""Tests for OFI calibrator module."""
from __future__ import annotations

import numpy as np
import pytest

from arb_bot.crypto.ofi_calibrator import OFICalibrationResult, OFICalibrator


class TestOFICalibrator:
    """Tests for OFICalibrator."""

    def test_ofi_alpha_calibration(self) -> None:
        """Generate 200 synthetic samples where fwd_return = 0.5 * ofi + noise.
        Verify fitted alpha ~ 0.5 and R² > 0.5."""
        rng = np.random.default_rng(42)
        cal = OFICalibrator(max_samples=5000, min_samples=30, min_r_squared=0.01)

        for _ in range(200):
            ofi = rng.uniform(-1.0, 1.0)
            fwd_return = 0.5 * ofi + rng.normal(0, 0.01)
            cal.record_sample(ofi, fwd_return)

        result = cal.calibrate()
        assert abs(result.alpha - 0.5) < 0.05, f"alpha={result.alpha}, expected ~0.5"
        assert result.r_squared > 0.5, f"R²={result.r_squared}, expected > 0.5"
        assert result.n_samples == 200

    def test_ofi_calibrator_low_r2_returns_zero_alpha(self) -> None:
        """Pure noise: OFI random, return random. Alpha should be 0 (R² too low)."""
        rng = np.random.default_rng(42)
        cal = OFICalibrator(max_samples=5000, min_samples=30, min_r_squared=0.01)

        for _ in range(100):
            ofi = rng.uniform(-1.0, 1.0)
            fwd_return = rng.normal(0, 1.0)
            cal.record_sample(ofi, fwd_return)

        result = cal.calibrate()
        assert result.alpha == 0.0, f"alpha={result.alpha}, expected 0.0 for noise"
        assert result.n_samples == 100

    def test_ofi_calibrator_rolling_window(self) -> None:
        """Create calibrator with max_samples=50, add 100 samples.
        Verify only 50 are kept."""
        rng = np.random.default_rng(42)
        cal = OFICalibrator(max_samples=50, min_samples=10)

        for _ in range(100):
            cal.record_sample(rng.uniform(-1, 1), rng.normal(0, 0.1))

        assert cal.sample_count == 50

    def test_ofi_calibrator_insufficient_samples(self) -> None:
        """Add only 5 samples with min_samples=30. Alpha should be 0."""
        cal = OFICalibrator(min_samples=30)
        for i in range(5):
            cal.record_sample(float(i) * 0.1, float(i) * 0.05)

        result = cal.calibrate()
        assert result.alpha == 0.0
        assert result.r_squared == 0.0
        assert result.n_samples == 5

    def test_ofi_calibrator_zero_ofi_returns_zero(self) -> None:
        """All OFI values are 0.0. Alpha should be 0 (degenerate)."""
        rng = np.random.default_rng(42)
        cal = OFICalibrator(min_samples=10)

        for _ in range(50):
            cal.record_sample(0.0, rng.normal(0, 0.01))

        result = cal.calibrate()
        assert result.alpha == 0.0
        assert result.r_squared == 0.0
        assert result.n_samples == 50

    def test_ofi_calibrator_negative_alpha(self) -> None:
        """Generate data where fwd_return = -0.3 * ofi + noise.
        Verify alpha ~ -0.3."""
        rng = np.random.default_rng(42)
        cal = OFICalibrator(max_samples=5000, min_samples=30, min_r_squared=0.01)

        for _ in range(200):
            ofi = rng.uniform(-1.0, 1.0)
            fwd_return = -0.3 * ofi + rng.normal(0, 0.01)
            cal.record_sample(ofi, fwd_return)

        result = cal.calibrate()
        assert abs(result.alpha - (-0.3)) < 0.05, f"alpha={result.alpha}, expected ~-0.3"
        assert result.r_squared > 0.5

    def test_ofi_calibration_result_dataclass(self) -> None:
        """Verify OFICalibrationResult fields are accessible."""
        result = OFICalibrationResult(alpha=0.42, r_squared=0.85, n_samples=100)
        assert result.alpha == 0.42
        assert result.r_squared == 0.85
        assert result.n_samples == 100

    def test_sample_count_property(self) -> None:
        """Verify sample_count returns correct count."""
        cal = OFICalibrator()
        assert cal.sample_count == 0

        cal.record_sample(0.1, 0.05)
        assert cal.sample_count == 1

        cal.record_sample(0.2, 0.10)
        assert cal.sample_count == 2

        cal.record_sample(-0.5, -0.25)
        assert cal.sample_count == 3
