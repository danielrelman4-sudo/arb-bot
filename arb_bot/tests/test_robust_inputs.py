"""Tests for Phase 4D: Outlier-robust sizing inputs."""

from __future__ import annotations

import math

import pytest

from arb_bot.robust_inputs import (
    DiagnosisReport,
    RobustInputConfig,
    RobustInputFilter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _filt(**kw) -> RobustInputFilter:
    return RobustInputFilter(RobustInputConfig(**kw))


# Tightly clustered data — no outliers by IQR fencing.
CLEAN = [1.00, 1.01, 1.02, 0.99, 0.98, 1.01, 1.00, 0.99, 1.02, 1.00]
WITH_OUTLIER = CLEAN + [5.0]
# Wider-spread clean data for consecutive-reset tests.
VERY_CLEAN = [1.0] * 10


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self) -> None:
        cfg = RobustInputConfig()
        assert cfg.iqr_multiplier == 1.5
        assert cfg.mad_threshold == 3.5
        assert cfg.min_samples == 5
        assert cfg.winsorize_fraction == 0.05
        assert cfg.max_consecutive_outliers == 3

    def test_custom(self) -> None:
        cfg = RobustInputConfig(iqr_multiplier=2.0, min_samples=10)
        assert cfg.iqr_multiplier == 2.0
        assert cfg.min_samples == 10

    def test_frozen(self) -> None:
        cfg = RobustInputConfig()
        with pytest.raises(AttributeError):
            cfg.iqr_multiplier = 2.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Filter values — IQR fencing
# ---------------------------------------------------------------------------


class TestFilterValues:
    def test_clean_data_unchanged(self) -> None:
        f = _filt()
        result = f.filter_values(CLEAN)
        assert len(result) == len(CLEAN)

    def test_outlier_removed(self) -> None:
        f = _filt()
        result = f.filter_values(WITH_OUTLIER)
        assert 5.0 not in result
        assert len(result) < len(WITH_OUTLIER)

    def test_below_min_samples_passthrough(self) -> None:
        f = _filt(min_samples=20)
        result = f.filter_values(WITH_OUTLIER)
        assert len(result) == len(WITH_OUTLIER)

    def test_all_same_values(self) -> None:
        f = _filt()
        data = [1.0] * 10
        result = f.filter_values(data)
        assert len(result) == 10

    def test_empty_input(self) -> None:
        f = _filt()
        assert f.filter_values([]) == []

    def test_single_value(self) -> None:
        f = _filt(min_samples=1)
        result = f.filter_values([42.0])
        assert result == [42.0]

    def test_multiple_outliers(self) -> None:
        f = _filt()
        data = CLEAN + [10.0, -5.0]
        result = f.filter_values(data)
        assert 10.0 not in result
        assert -5.0 not in result

    def test_wider_multiplier_more_permissive(self) -> None:
        narrow = _filt(iqr_multiplier=1.0)
        wide = _filt(iqr_multiplier=3.0)
        data = CLEAN + [1.5]
        r_narrow = narrow.filter_values(data)
        r_wide = wide.filter_values(data)
        assert len(r_wide) >= len(r_narrow)


# ---------------------------------------------------------------------------
# Winsorize
# ---------------------------------------------------------------------------


class TestWinsorize:
    def test_clips_tails(self) -> None:
        f = _filt(winsorize_fraction=0.1)
        data = list(range(20))  # 0..19
        result = f.winsorize(data)
        assert min(result) >= 2.0
        assert max(result) <= 17.0

    def test_preserves_length(self) -> None:
        f = _filt()
        result = f.winsorize(CLEAN)
        assert len(result) == len(CLEAN)

    def test_below_min_samples_passthrough(self) -> None:
        f = _filt(min_samples=20)
        result = f.winsorize(WITH_OUTLIER)
        assert len(result) == len(WITH_OUTLIER)
        assert 5.0 in result

    def test_zero_fraction_no_change(self) -> None:
        f = _filt(winsorize_fraction=0.0)
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        result = f.winsorize(data)
        assert result == data


# ---------------------------------------------------------------------------
# Is outlier — MAD-based
# ---------------------------------------------------------------------------


class TestIsOutlier:
    def test_normal_value_not_outlier(self) -> None:
        f = _filt()
        assert f.is_outlier(1.0, CLEAN) is False

    def test_extreme_value_is_outlier(self) -> None:
        f = _filt()
        assert f.is_outlier(10.0, CLEAN) is True

    def test_below_min_samples(self) -> None:
        f = _filt(min_samples=20)
        assert f.is_outlier(10.0, CLEAN) is False

    def test_all_same_reference(self) -> None:
        f = _filt()
        ref = [1.0] * 10
        assert f.is_outlier(1.0, ref) is False
        assert f.is_outlier(2.0, ref) is True

    def test_lower_threshold_more_sensitive(self) -> None:
        sensitive = _filt(mad_threshold=2.0)
        lenient = _filt(mad_threshold=5.0)
        val = 1.3
        s_result = sensitive.is_outlier(val, CLEAN)
        l_result = lenient.is_outlier(val, CLEAN)
        assert l_result is False or s_result is True


# ---------------------------------------------------------------------------
# Robust mean and std
# ---------------------------------------------------------------------------


class TestRobustStats:
    def test_robust_mean_clean(self) -> None:
        f = _filt()
        mean = f.robust_mean(CLEAN)
        assert abs(mean - 1.0) < 0.05

    def test_robust_mean_resists_outlier(self) -> None:
        # Use higher winsorize fraction so small datasets clip effectively.
        f = _filt(winsorize_fraction=0.15)
        outlier_mean = f.robust_mean(WITH_OUTLIER)
        regular_mean = sum(WITH_OUTLIER) / len(WITH_OUTLIER)
        # Robust mean should be closer to center than regular mean.
        assert abs(outlier_mean - 1.0) < abs(regular_mean - 1.0)

    def test_robust_mean_empty(self) -> None:
        f = _filt()
        assert f.robust_mean([]) == 0.0

    def test_robust_std_clean(self) -> None:
        f = _filt()
        std = f.robust_std(CLEAN)
        assert std >= 0.0
        assert std < 0.5

    def test_robust_std_single(self) -> None:
        f = _filt()
        assert f.robust_std([1.0]) == 0.0

    def test_robust_std_two_values(self) -> None:
        f = _filt()
        std = f.robust_std([1.0, 3.0])
        assert std == pytest.approx(1.4826)


# ---------------------------------------------------------------------------
# Diagnose
# ---------------------------------------------------------------------------


class TestDiagnose:
    def test_clean_data(self) -> None:
        f = _filt()
        report = f.diagnose(CLEAN, "edge")
        assert report.field_name == "edge"
        assert report.sample_count == len(CLEAN)
        assert report.outlier_count == 0
        assert report.outlier_fraction == 0.0

    def test_with_outlier(self) -> None:
        f = _filt()
        report = f.diagnose(WITH_OUTLIER, "edge")
        assert report.outlier_count >= 1
        assert report.outlier_fraction > 0.0

    def test_empty_data(self) -> None:
        f = _filt()
        report = f.diagnose([], "edge")
        assert report.sample_count == 0
        assert report.outlier_count == 0

    def test_fences_populated(self) -> None:
        f = _filt()
        report = f.diagnose(CLEAN, "edge")
        assert report.lower_fence <= report.upper_fence
        assert report.q1 <= report.q3


# ---------------------------------------------------------------------------
# Consecutive outlier tracking
# ---------------------------------------------------------------------------


class TestConsecutiveOutliers:
    def test_no_alert_initially(self) -> None:
        f = _filt(max_consecutive_outliers=3)
        assert f.has_consecutive_alert("edge") is False

    def test_alert_after_threshold(self) -> None:
        f = _filt(max_consecutive_outliers=3)
        for _ in range(3):
            f.filter_values(WITH_OUTLIER, "edge")
        assert f.has_consecutive_alert("edge") is True

    def test_alert_resets_on_clean(self) -> None:
        f = _filt(max_consecutive_outliers=2)
        f.filter_values(WITH_OUTLIER, "edge")
        f.filter_values(WITH_OUTLIER, "edge")
        assert f.has_consecutive_alert("edge") is True
        # Identical values → IQR=0, no outliers → resets consecutive count.
        f.filter_values(VERY_CLEAN, "edge")
        assert f.has_consecutive_alert("edge") is False

    def test_independent_fields(self) -> None:
        f = _filt(max_consecutive_outliers=2)
        f.filter_values(WITH_OUTLIER, "edge")
        f.filter_values(WITH_OUTLIER, "edge")
        assert f.has_consecutive_alert("edge") is True
        assert f.has_consecutive_alert("cost") is False


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clears_state(self) -> None:
        f = _filt(max_consecutive_outliers=2)
        f.filter_values(WITH_OUTLIER, "edge")
        f.filter_values(WITH_OUTLIER, "edge")
        f.clear()
        assert f.has_consecutive_alert("edge") is False


# ---------------------------------------------------------------------------
# Config property
# ---------------------------------------------------------------------------


class TestConfigProperty:
    def test_config_accessible(self) -> None:
        cfg = RobustInputConfig(iqr_multiplier=2.5)
        f = RobustInputFilter(cfg)
        assert f.config.iqr_multiplier == 2.5


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_pipeline(self) -> None:
        """Full pipeline: diagnose → filter → robust stats."""
        f = _filt()
        raw = [0.03, 0.032, 0.028, 0.031, 0.029, 0.030, 0.5, 0.033]

        report = f.diagnose(raw, "edge")
        assert report.outlier_count >= 1

        cleaned = f.filter_values(raw, "edge")
        assert 0.5 not in cleaned

        # Robust mean via winsorize should clip extreme value.
        # Use higher winsorize fraction for this small dataset.
        f2 = _filt(winsorize_fraction=0.15)
        regular_mean = sum(raw) / len(raw)
        robust_mean = f2.robust_mean(raw)
        assert abs(robust_mean - 0.03) < abs(regular_mean - 0.03)

        std = f.robust_std(raw)
        assert std > 0.0
