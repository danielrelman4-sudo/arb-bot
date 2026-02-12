"""Tests for auto_tuner module (Phase 6E).

This module adjusts live trading parameters. Tests are designed to
validate that ALL safety invariants hold:
1. Parameters NEVER exceed hard bounds.
2. Adjustments NEVER exceed max_step_fraction.
3. No tuning with insufficient samples.
4. Revert on degradation.
"""

from __future__ import annotations

import pytest

from arb_bot.auto_tuner import (
    AdjustDirection,
    Adjustment,
    AutoTuner,
    AutoTunerConfig,
    TunableParam,
    TuneCycleReport,
)


def _tuner(**kw: object) -> AutoTuner:
    return AutoTuner(AutoTunerConfig(**kw))  # type: ignore[arg-type]


def _seed_kpis(tuner: AutoTuner, name: str, values: list[float], start_ts: float = 100.0) -> None:
    """Seed a KPI with a sequence of values."""
    for i, v in enumerate(values):
        tuner.record_kpi(name, v, ts=start_ts + float(i))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self) -> None:
        cfg = AutoTunerConfig()
        assert cfg.min_samples == 20
        assert cfg.max_step_fraction == 0.10
        assert cfg.revert_lookback == 5
        assert cfg.revert_degradation_threshold == 0.10
        assert cfg.tune_interval_seconds == 300.0
        assert cfg.max_params == 50

    def test_frozen(self) -> None:
        cfg = AutoTunerConfig()
        with pytest.raises(AttributeError):
            cfg.min_samples = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Define parameter
# ---------------------------------------------------------------------------


class TestDefineParam:
    def test_basic(self) -> None:
        t = _tuner()
        ok = t.define_param("edge", current=0.01, hard_min=0.003, hard_max=0.05)
        assert ok is True
        assert t.param_count() == 1
        assert t.get_value("edge") == 0.01

    def test_clamps_to_bounds(self) -> None:
        """Current value outside bounds gets clamped."""
        t = _tuner()
        t.define_param("edge", current=0.10, hard_min=0.003, hard_max=0.05)
        assert t.get_value("edge") == 0.05  # Clamped to max.

    def test_clamps_below_min(self) -> None:
        t = _tuner()
        t.define_param("edge", current=0.001, hard_min=0.003, hard_max=0.05)
        assert t.get_value("edge") == 0.003  # Clamped to min.

    def test_invalid_bounds(self) -> None:
        """min > max is rejected."""
        t = _tuner()
        ok = t.define_param("edge", current=0.01, hard_min=0.05, hard_max=0.003)
        assert ok is False

    def test_max_params(self) -> None:
        t = _tuner(max_params=2)
        t.define_param("a", current=1.0, hard_min=0.0, hard_max=2.0)
        t.define_param("b", current=1.0, hard_min=0.0, hard_max=2.0)
        ok = t.define_param("c", current=1.0, hard_min=0.0, hard_max=2.0)
        assert ok is False
        assert t.param_count() == 2

    def test_default_value_stored(self) -> None:
        t = _tuner()
        t.define_param("edge", current=0.01, hard_min=0.003, hard_max=0.05)
        param = t.get_param("edge")
        assert param is not None
        assert param.default_value == 0.01


# ---------------------------------------------------------------------------
# Record KPI
# ---------------------------------------------------------------------------


class TestRecordKPI:
    def test_basic(self) -> None:
        t = _tuner()
        t.record_kpi("win_rate", 0.62, ts=100.0)
        t.record_kpi("win_rate", 0.65, ts=101.0)
        # Verified via tune_cycle behavior below.


# ---------------------------------------------------------------------------
# Safety invariant: hard bounds NEVER exceeded
# ---------------------------------------------------------------------------


class TestHardBoundsInvariant:
    def test_increase_clamped_at_max(self) -> None:
        """Even with many cycles, parameter never exceeds hard_max."""
        t = _tuner(min_samples=3, max_step_fraction=0.50)
        t.define_param("p", current=0.04, hard_min=0.01, hard_max=0.05,
                        kpi_name="wr", kpi_direction="higher_is_better",
                        adjust_direction="increase_when_good")
        # Seed with improving KPIs (recent >= baseline).
        _seed_kpis(t, "wr", [0.5, 0.6, 0.7, 0.8, 0.9])

        for _ in range(20):
            t.tune_cycle(now=1000.0)

        assert t.get_value("p") is not None
        assert t.get_value("p") <= 0.05  # NEVER exceeds hard_max.

    def test_decrease_clamped_at_min(self) -> None:
        """Even with many cycles, parameter never goes below hard_min."""
        t = _tuner(min_samples=3, max_step_fraction=0.50)
        t.define_param("p", current=0.02, hard_min=0.01, hard_max=0.05,
                        kpi_name="wr", kpi_direction="higher_is_better",
                        adjust_direction="decrease_when_good")
        # Seed with improving KPIs.
        _seed_kpis(t, "wr", [0.5, 0.6, 0.7, 0.8, 0.9])

        for _ in range(20):
            t.tune_cycle(now=1000.0)

        assert t.get_value("p") is not None
        assert t.get_value("p") >= 0.01  # NEVER below hard_min.


# ---------------------------------------------------------------------------
# Safety invariant: max step size
# ---------------------------------------------------------------------------


class TestMaxStepInvariant:
    def test_step_bounded(self) -> None:
        """Single cycle adjustment is bounded by max_step_fraction."""
        t = _tuner(min_samples=3, max_step_fraction=0.10)
        t.define_param("p", current=1.0, hard_min=0.0, hard_max=10.0,
                        kpi_name="wr", kpi_direction="higher_is_better",
                        adjust_direction="increase_when_good")
        _seed_kpis(t, "wr", [0.5, 0.6, 0.7, 0.8, 0.9])

        report = t.tune_cycle(now=1000.0)
        if report.params_adjusted > 0:
            adj = report.adjustments[0]
            change = abs(adj.new_value - adj.old_value)
            max_allowed = abs(adj.old_value) * 0.10
            assert change <= max_allowed + 1e-10  # Float tolerance.


# ---------------------------------------------------------------------------
# Safety invariant: minimum samples
# ---------------------------------------------------------------------------


class TestMinSamplesInvariant:
    def test_skip_insufficient_samples(self) -> None:
        """No tuning when KPI has fewer observations than min_samples."""
        t = _tuner(min_samples=20)
        t.define_param("p", current=1.0, hard_min=0.0, hard_max=10.0,
                        kpi_name="wr")
        # Only 5 observations < 20 required.
        _seed_kpis(t, "wr", [0.6, 0.7, 0.8, 0.9, 1.0])

        report = t.tune_cycle(now=1000.0)
        assert report.params_adjusted == 0
        assert report.skipped_insufficient_samples == 1

    def test_tune_with_sufficient_samples(self) -> None:
        t = _tuner(min_samples=5)
        t.define_param("p", current=1.0, hard_min=0.0, hard_max=10.0,
                        kpi_name="wr", kpi_direction="higher_is_better",
                        adjust_direction="increase_when_good")
        _seed_kpis(t, "wr", [0.5, 0.6, 0.7, 0.8, 0.9])

        report = t.tune_cycle(now=1000.0)
        assert report.skipped_insufficient_samples == 0


# ---------------------------------------------------------------------------
# Tuning direction
# ---------------------------------------------------------------------------


class TestTuningDirection:
    def test_increase_when_kpi_good(self) -> None:
        """KPI is good (recent >= baseline) → param increases."""
        t = _tuner(min_samples=3, max_step_fraction=0.10, revert_lookback=3)
        t.define_param("p", current=1.0, hard_min=0.0, hard_max=10.0,
                        kpi_name="wr", kpi_direction="higher_is_better",
                        adjust_direction="increase_when_good")
        # Recent values (last 3) are >= overall mean → KPI is "good".
        _seed_kpis(t, "wr", [0.5, 0.5, 0.7, 0.8, 0.9])

        report = t.tune_cycle(now=1000.0)
        adj = next((a for a in report.adjustments if a.param_name == "p"), None)
        assert adj is not None
        assert adj.new_value > adj.old_value  # Increased.

    def test_decrease_when_kpi_bad(self) -> None:
        """KPI is bad (recent < baseline) → param decreases for 'increase_when_good'."""
        t = _tuner(min_samples=3, max_step_fraction=0.10, revert_lookback=3)
        t.define_param("p", current=1.0, hard_min=0.0, hard_max=10.0,
                        kpi_name="wr", kpi_direction="higher_is_better",
                        adjust_direction="increase_when_good")
        # Recent values (last 3) are < overall mean → KPI is "bad".
        _seed_kpis(t, "wr", [0.9, 0.8, 0.3, 0.2, 0.1])

        report = t.tune_cycle(now=1000.0)
        adj = next((a for a in report.adjustments if a.param_name == "p"), None)
        assert adj is not None
        assert adj.new_value < adj.old_value  # Decreased.

    def test_decrease_when_good_direction(self) -> None:
        """adjust_direction='decrease_when_good' inverts behavior."""
        t = _tuner(min_samples=3, max_step_fraction=0.10, revert_lookback=3)
        t.define_param("p", current=1.0, hard_min=0.0, hard_max=10.0,
                        kpi_name="wr", kpi_direction="higher_is_better",
                        adjust_direction="decrease_when_good")
        _seed_kpis(t, "wr", [0.5, 0.5, 0.7, 0.8, 0.9])

        report = t.tune_cycle(now=1000.0)
        adj = next((a for a in report.adjustments if a.param_name == "p"), None)
        assert adj is not None
        assert adj.new_value < adj.old_value  # Decreased (inverse).


# ---------------------------------------------------------------------------
# Revert on degradation
# ---------------------------------------------------------------------------


class TestRevert:
    def test_revert_when_kpi_degrades(self) -> None:
        """After tuning, if KPIs degrade beyond threshold, revert."""
        t = _tuner(
            min_samples=5,
            max_step_fraction=0.10,
            revert_lookback=3,
            revert_degradation_threshold=0.10,
        )
        t.define_param("p", current=1.0, hard_min=0.0, hard_max=10.0,
                        kpi_name="wr", kpi_direction="higher_is_better",
                        adjust_direction="increase_when_good")

        # Good KPIs → tune up.
        _seed_kpis(t, "wr", [0.5, 0.6, 0.7, 0.8, 0.9])
        report1 = t.tune_cycle(now=1000.0)
        assert report1.params_adjusted == 1
        tuned_value = t.get_value("p")
        assert tuned_value is not None
        assert tuned_value > 1.0

        # Now KPIs degrade badly — recent 3 much worse than overall mean.
        t.record_kpi("wr", 0.3, ts=200.0)
        t.record_kpi("wr", 0.2, ts=201.0)
        t.record_kpi("wr", 0.1, ts=202.0)

        report2 = t.tune_cycle(now=2000.0)
        assert report2.params_reverted == 1
        # Should have reverted to previous value (1.0).
        assert t.get_value("p") == 1.0

    def test_no_revert_without_degradation(self) -> None:
        """No revert if KPIs remain stable."""
        t = _tuner(min_samples=5, max_step_fraction=0.10, revert_lookback=3,
                    revert_degradation_threshold=0.10)
        t.define_param("p", current=1.0, hard_min=0.0, hard_max=10.0,
                        kpi_name="wr", kpi_direction="higher_is_better",
                        adjust_direction="increase_when_good")

        # Good KPIs → tune.
        _seed_kpis(t, "wr", [0.5, 0.6, 0.7, 0.8, 0.9])
        t.tune_cycle(now=1000.0)

        # KPIs stay good — more good values.
        t.record_kpi("wr", 0.85, ts=200.0)
        t.record_kpi("wr", 0.90, ts=201.0)
        t.record_kpi("wr", 0.88, ts=202.0)

        report = t.tune_cycle(now=2000.0)
        assert report.params_reverted == 0


# ---------------------------------------------------------------------------
# Reset to default
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_param(self) -> None:
        t = _tuner(min_samples=3, max_step_fraction=0.10)
        t.define_param("p", current=1.0, hard_min=0.0, hard_max=10.0,
                        kpi_name="wr", kpi_direction="higher_is_better",
                        adjust_direction="increase_when_good")
        _seed_kpis(t, "wr", [0.5, 0.6, 0.7, 0.8, 0.9])
        t.tune_cycle(now=1000.0)

        # Value should have changed.
        assert t.get_value("p") != 1.0

        # Reset.
        t.reset_param("p")
        assert t.get_value("p") == 1.0  # Back to default.


# ---------------------------------------------------------------------------
# No KPI linked
# ---------------------------------------------------------------------------


class TestNoKPI:
    def test_skip_param_without_kpi(self) -> None:
        t = _tuner(min_samples=3)
        t.define_param("p", current=1.0, hard_min=0.0, hard_max=10.0)
        _seed_kpis(t, "wr", [0.5, 0.6, 0.7])
        report = t.tune_cycle(now=1000.0)
        assert report.params_adjusted == 0


# ---------------------------------------------------------------------------
# At-bound hold
# ---------------------------------------------------------------------------


class TestAtBound:
    def test_hold_at_max(self) -> None:
        """If already at hard_max, can't increase further → hold."""
        t = _tuner(min_samples=3, max_step_fraction=0.10, revert_lookback=3)
        t.define_param("p", current=10.0, hard_min=0.0, hard_max=10.0,
                        kpi_name="wr", kpi_direction="higher_is_better",
                        adjust_direction="increase_when_good")
        _seed_kpis(t, "wr", [0.5, 0.5, 0.7, 0.8, 0.9])

        report = t.tune_cycle(now=1000.0)
        adj = next((a for a in report.adjustments if a.param_name == "p"), None)
        assert adj is not None
        assert adj.direction == AdjustDirection.HOLD
        assert adj.reason == "at_bound"
        assert t.get_value("p") == 10.0  # Unchanged.


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


class TestAuditLog:
    def test_audit_records_adjustments(self) -> None:
        t = _tuner(min_samples=3, max_step_fraction=0.10, revert_lookback=3)
        t.define_param("p", current=1.0, hard_min=0.0, hard_max=10.0,
                        kpi_name="wr", kpi_direction="higher_is_better",
                        adjust_direction="increase_when_good")
        _seed_kpis(t, "wr", [0.5, 0.5, 0.7, 0.8, 0.9])
        t.tune_cycle(now=1000.0)

        log = t.audit_log()
        assert len(log) >= 1
        assert log[0].param_name == "p"
        assert log[0].reason == "kpi_driven"

    def test_audit_limit(self) -> None:
        t = _tuner(min_samples=3, max_step_fraction=0.10, revert_lookback=3)
        t.define_param("p", current=1.0, hard_min=0.0, hard_max=10.0,
                        kpi_name="wr", kpi_direction="higher_is_better",
                        adjust_direction="increase_when_good")
        _seed_kpis(t, "wr", [0.5, 0.5, 0.7, 0.8, 0.9])
        for i in range(10):
            t.tune_cycle(now=1000.0 + i)

        log = t.audit_log(limit=3)
        assert len(log) <= 3


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


class TestMisc:
    def test_param_names(self) -> None:
        t = _tuner()
        t.define_param("a", current=1.0, hard_min=0.0, hard_max=2.0)
        t.define_param("b", current=1.0, hard_min=0.0, hard_max=2.0)
        assert set(t.param_names()) == {"a", "b"}

    def test_get_nonexistent(self) -> None:
        t = _tuner()
        assert t.get_param("missing") is None
        assert t.get_value("missing") is None

    def test_clear(self) -> None:
        t = _tuner()
        t.define_param("p", current=1.0, hard_min=0.0, hard_max=2.0)
        t.record_kpi("wr", 0.5, ts=1.0)
        t.clear()
        assert t.param_count() == 0
        assert len(t.audit_log()) == 0


# ---------------------------------------------------------------------------
# Multiple parameters
# ---------------------------------------------------------------------------


class TestMultipleParams:
    def test_independent_tuning(self) -> None:
        """Each param is tuned independently based on its own KPI."""
        t = _tuner(min_samples=3, max_step_fraction=0.10, revert_lookback=3)
        t.define_param("edge", current=0.01, hard_min=0.003, hard_max=0.05,
                        kpi_name="wr", kpi_direction="higher_is_better",
                        adjust_direction="increase_when_good")
        t.define_param("size", current=100.0, hard_min=10.0, hard_max=500.0,
                        kpi_name="pnl", kpi_direction="higher_is_better",
                        adjust_direction="increase_when_good")

        _seed_kpis(t, "wr", [0.5, 0.5, 0.7, 0.8, 0.9])
        _seed_kpis(t, "pnl", [10, 10, 15, 20, 25])

        report = t.tune_cycle(now=1000.0)
        assert report.params_evaluated == 2

    def test_mixed_sufficient_insufficient(self) -> None:
        """One param has enough KPI data, other doesn't."""
        t = _tuner(min_samples=10)
        t.define_param("a", current=1.0, hard_min=0.0, hard_max=10.0, kpi_name="wr")
        t.define_param("b", current=1.0, hard_min=0.0, hard_max=10.0, kpi_name="pnl")

        _seed_kpis(t, "wr", list(range(15)))  # 15 >= 10.
        _seed_kpis(t, "pnl", [1.0, 2.0, 3.0])  # 3 < 10.

        report = t.tune_cycle(now=1000.0)
        assert report.skipped_insufficient_samples >= 1


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_full_tuning_lifecycle(self) -> None:
        """Define → record KPIs → tune → verify bounds → degrade → revert."""
        t = _tuner(
            min_samples=5,
            max_step_fraction=0.10,
            revert_lookback=3,
            revert_degradation_threshold=0.10,
        )

        # Define edge threshold param.
        t.define_param(
            "edge_threshold",
            current=0.01,
            hard_min=0.003,
            hard_max=0.05,
            kpi_name="win_rate",
            kpi_direction="higher_is_better",
            adjust_direction="increase_when_good",
        )

        # Phase 1: Accumulate good KPIs.
        _seed_kpis(t, "win_rate", [0.55, 0.58, 0.60, 0.62, 0.65])

        # Tune — should increase edge_threshold since win_rate is good.
        report1 = t.tune_cycle(now=1000.0)
        assert report1.params_adjusted == 1
        value_after_tune = t.get_value("edge_threshold")
        assert value_after_tune is not None
        assert value_after_tune > 0.01  # Increased.
        assert value_after_tune <= 0.05  # Within bounds.

        # Phase 2: KPIs degrade.
        t.record_kpi("win_rate", 0.35, ts=300.0)
        t.record_kpi("win_rate", 0.30, ts=301.0)
        t.record_kpi("win_rate", 0.25, ts=302.0)

        # Tune — should revert.
        report2 = t.tune_cycle(now=2000.0)
        assert report2.params_reverted == 1
        assert t.get_value("edge_threshold") == 0.01  # Reverted to pre-tune.

        # Verify safety invariants held throughout.
        param = t.get_param("edge_threshold")
        assert param is not None
        assert param.current_value >= param.hard_min
        assert param.current_value <= param.hard_max

    def test_stress_many_cycles(self) -> None:
        """Run many tune cycles — bounds must ALWAYS hold."""
        t = _tuner(min_samples=3, max_step_fraction=0.20)
        t.define_param("p", current=5.0, hard_min=1.0, hard_max=10.0,
                        kpi_name="wr", kpi_direction="higher_is_better",
                        adjust_direction="increase_when_good")

        # Alternating good and bad KPIs.
        for i in range(100):
            val = 0.8 if i % 3 == 0 else 0.3
            t.record_kpi("wr", val, ts=float(i))

        for i in range(50):
            t.tune_cycle(now=1000.0 + float(i))
            val = t.get_value("p")
            assert val is not None
            assert 1.0 <= val <= 10.0, f"Bounds violated: {val}"
