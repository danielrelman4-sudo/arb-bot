"""Tests for Phase 3E: Latency SLO + auto-throttle."""

from __future__ import annotations

import pytest

from arb_bot.framework.latency_slo import (
    LatencyReport,
    LatencySLO,
    LatencyTracker,
    LatencyTrackerConfig,
    SLOEvaluation,
    SLOStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slo(
    name: str = "quote_age",
    p50: float = 0.0,
    p95: float = 0.5,
    p99: float = 1.0,
    breach_window: int = 3,
) -> LatencySLO:
    return LatencySLO(
        name=name,
        target_p50=p50,
        target_p95=p95,
        target_p99=p99,
        breach_window=breach_window,
    )


def _tracker(*slos: LatencySLO, **kw) -> LatencyTracker:
    cfg = LatencyTrackerConfig(slos=tuple(slos), **kw)
    return LatencyTracker(cfg)


# ---------------------------------------------------------------------------
# LatencySLO
# ---------------------------------------------------------------------------


class TestLatencySLO:
    def test_defaults(self) -> None:
        slo = LatencySLO(name="test")
        assert slo.target_p50 == 0.0
        assert slo.target_p95 == 0.0
        assert slo.breach_window == 3

    def test_custom(self) -> None:
        slo = _slo(p95=0.5, p99=1.0)
        assert slo.target_p95 == 0.5
        assert slo.target_p99 == 1.0


# ---------------------------------------------------------------------------
# LatencyTrackerConfig
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self) -> None:
        cfg = LatencyTrackerConfig()
        assert cfg.slos == ()
        assert cfg.window_size == 1000
        assert cfg.throttle_cooldown_evaluations == 5


# ---------------------------------------------------------------------------
# Recording samples
# ---------------------------------------------------------------------------


class TestRecording:
    def test_record_and_count(self) -> None:
        tracker = _tracker(_slo())
        tracker.record("quote_age", 0.1)
        tracker.record("quote_age", 0.2)
        assert tracker.sample_count("quote_age") == 2

    def test_unknown_metric(self) -> None:
        tracker = _tracker(_slo())
        assert tracker.sample_count("unknown") == 0

    def test_window_trimming(self) -> None:
        tracker = _tracker(_slo(), window_size=5)
        for i in range(10):
            tracker.record("quote_age", float(i))
        assert tracker.sample_count("quote_age") == 5


# ---------------------------------------------------------------------------
# Percentile computation
# ---------------------------------------------------------------------------


class TestPercentile:
    def test_single_sample(self) -> None:
        tracker = _tracker(_slo())
        tracker.record("test", 0.5)
        assert tracker.percentile("test", 50.0) == pytest.approx(0.5)
        assert tracker.percentile("test", 95.0) == pytest.approx(0.5)

    def test_no_samples(self) -> None:
        tracker = _tracker(_slo())
        assert tracker.percentile("test", 50.0) is None

    def test_uniform_samples(self) -> None:
        tracker = _tracker(_slo())
        for i in range(100):
            tracker.record("test", float(i) / 100.0)
        p50 = tracker.percentile("test", 50.0)
        p95 = tracker.percentile("test", 95.0)
        assert p50 is not None
        assert p95 is not None
        assert p50 < p95

    def test_p99_high(self) -> None:
        tracker = _tracker(_slo())
        # 99 fast, 1 slow.
        for _ in range(99):
            tracker.record("test", 0.01)
        tracker.record("test", 10.0)
        p99 = tracker.percentile("test", 99.0)
        assert p99 > 0.01


# ---------------------------------------------------------------------------
# SLO evaluation — passing
# ---------------------------------------------------------------------------


class TestSLOPassing:
    def test_all_within_target(self) -> None:
        slo = _slo(p95=0.5, p99=1.0)
        tracker = _tracker(slo)
        for _ in range(100):
            tracker.record("quote_age", 0.1)
        report = tracker.evaluate()
        assert report.all_passing
        assert not report.any_breached

    def test_insufficient_data(self) -> None:
        tracker = _tracker(_slo())
        report = tracker.evaluate()
        assert report.evaluations[0].status == SLOStatus.INSUFFICIENT_DATA


# ---------------------------------------------------------------------------
# SLO evaluation — breached
# ---------------------------------------------------------------------------


class TestSLOBreached:
    def test_p95_breach(self) -> None:
        slo = _slo(p95=0.1)
        tracker = _tracker(slo)
        for _ in range(100):
            tracker.record("quote_age", 0.5)  # All exceed p95 target.
        report = tracker.evaluate()
        assert report.any_breached
        eval_ = report.evaluations[0]
        assert eval_.status == SLOStatus.BREACHED
        assert eval_.breaches.get("p95") is True

    def test_p99_breach_only(self) -> None:
        slo = _slo(p95=1.0, p99=0.1)
        tracker = _tracker(slo)
        # Most fast, one very slow for p99.
        for _ in range(98):
            tracker.record("quote_age", 0.05)
        tracker.record("quote_age", 5.0)
        tracker.record("quote_age", 5.0)
        report = tracker.evaluate()
        eval_ = report.evaluations[0]
        assert eval_.breaches.get("p95") is False  # p95 within target.
        assert eval_.breaches.get("p99") is True

    def test_p50_breach(self) -> None:
        slo = _slo(p50=0.1)
        tracker = _tracker(slo)
        for _ in range(100):
            tracker.record("quote_age", 0.5)
        report = tracker.evaluate()
        eval_ = report.evaluations[0]
        assert eval_.breaches.get("p50") is True


# ---------------------------------------------------------------------------
# Consecutive breach tracking
# ---------------------------------------------------------------------------


class TestConsecutiveBreaches:
    def test_breach_count_increments(self) -> None:
        slo = _slo(p95=0.1, breach_window=3)
        tracker = _tracker(slo)
        for _ in range(100):
            tracker.record("quote_age", 0.5)

        report1 = tracker.evaluate()
        assert report1.evaluations[0].consecutive_breach_count == 1

        report2 = tracker.evaluate()
        assert report2.evaluations[0].consecutive_breach_count == 2

    def test_breach_count_resets_on_pass(self) -> None:
        slo = _slo(p95=0.5, breach_window=3)
        tracker = _tracker(slo)

        # Breach.
        for _ in range(100):
            tracker.record("quote_age", 1.0)
        tracker.evaluate()

        # Clear and pass.
        tracker._samples["quote_age"] = [0.1] * 100
        report = tracker.evaluate()
        assert report.evaluations[0].consecutive_breach_count == 0


# ---------------------------------------------------------------------------
# Auto-throttle
# ---------------------------------------------------------------------------


class TestAutoThrottle:
    def test_no_throttle_below_window(self) -> None:
        slo = _slo(p95=0.1, breach_window=3)
        tracker = _tracker(slo)
        for _ in range(100):
            tracker.record("quote_age", 0.5)

        # 1st and 2nd breach: below window of 3.
        tracker.evaluate()
        report = tracker.evaluate()
        assert not report.should_throttle
        assert not report.is_throttled

    def test_throttle_at_window(self) -> None:
        slo = _slo(p95=0.1, breach_window=3)
        tracker = _tracker(slo)
        for _ in range(100):
            tracker.record("quote_age", 0.5)

        tracker.evaluate()  # Breach 1.
        tracker.evaluate()  # Breach 2.
        report = tracker.evaluate()  # Breach 3 → throttle.
        assert report.should_throttle
        assert report.is_throttled
        assert len(report.throttle_reasons) == 1

    def test_throttle_stays_during_cooldown(self) -> None:
        slo = _slo(p95=0.1, breach_window=1)
        tracker = _tracker(slo, throttle_cooldown_evaluations=3)
        for _ in range(100):
            tracker.record("quote_age", 0.5)
        tracker.evaluate()  # Breach → throttle.

        # Pass but still in cooldown.
        tracker._samples["quote_age"] = [0.01] * 100
        report1 = tracker.evaluate()  # Cooldown 1.
        assert report1.is_throttled  # Still throttled.

        report2 = tracker.evaluate()  # Cooldown 2.
        assert report2.is_throttled

        report3 = tracker.evaluate()  # Cooldown 3 → exit.
        assert not report3.is_throttled

    def test_cooldown_resets_on_new_breach(self) -> None:
        slo = _slo(p95=0.1, breach_window=1)
        tracker = _tracker(slo, throttle_cooldown_evaluations=3)

        # Breach → throttle.
        for _ in range(100):
            tracker.record("quote_age", 0.5)
        tracker.evaluate()

        # One pass (cooldown progress).
        tracker._samples["quote_age"] = [0.01] * 100
        tracker.evaluate()

        # New breach resets cooldown.
        tracker._samples["quote_age"] = [0.5] * 100
        report = tracker.evaluate()
        assert report.is_throttled

    def test_not_throttled_initially(self) -> None:
        tracker = _tracker(_slo())
        assert tracker.is_throttled is False


# ---------------------------------------------------------------------------
# Multiple SLOs
# ---------------------------------------------------------------------------


class TestMultipleSLOs:
    def test_independent_tracking(self) -> None:
        slo_a = _slo(name="quote_age", p95=0.5)
        slo_b = _slo(name="exec_latency", p95=0.2)
        tracker = _tracker(slo_a, slo_b)

        for _ in range(100):
            tracker.record("quote_age", 0.1)    # Pass.
            tracker.record("exec_latency", 0.5)  # Breach.

        report = tracker.evaluate()
        evals = {e.name: e for e in report.evaluations}
        assert evals["quote_age"].status == SLOStatus.PASSING
        assert evals["exec_latency"].status == SLOStatus.BREACHED

    def test_throttle_from_any_slo(self) -> None:
        slo_a = _slo(name="a", p95=0.5, breach_window=1)
        slo_b = _slo(name="b", p95=0.5, breach_window=1)
        tracker = _tracker(slo_a, slo_b)

        for _ in range(100):
            tracker.record("a", 0.1)  # Pass.
            tracker.record("b", 1.0)  # Breach.

        report = tracker.evaluate()
        assert report.should_throttle
        # Only "b" should be in reasons.
        assert any("b" in r for r in report.throttle_reasons)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_all(self) -> None:
        tracker = _tracker(_slo(p95=0.1, breach_window=1))
        for _ in range(100):
            tracker.record("quote_age", 0.5)
        tracker.evaluate()  # Throttle.

        tracker.reset()
        assert tracker.sample_count("quote_age") == 0
        assert tracker.is_throttled is False


# ---------------------------------------------------------------------------
# Report properties
# ---------------------------------------------------------------------------


class TestReportProperties:
    def test_all_passing(self) -> None:
        tracker = _tracker(_slo(p95=1.0))
        for _ in range(100):
            tracker.record("quote_age", 0.1)
        report = tracker.evaluate()
        assert report.all_passing is True
        assert report.any_breached is False

    def test_config_property(self) -> None:
        cfg = LatencyTrackerConfig(window_size=500)
        tracker = LatencyTracker(cfg)
        assert tracker.config.window_size == 500


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_realistic_scenario(self) -> None:
        """Simulate realistic latency tracking across multiple metrics."""
        quote_slo = _slo(name="quote_age", p95=0.3, p99=1.0, breach_window=2)
        exec_slo = _slo(name="exec_latency", p95=0.5, p99=2.0, breach_window=2)
        tracker = _tracker(quote_slo, exec_slo, throttle_cooldown_evaluations=2)

        # Normal operation: all pass.
        for _ in range(100):
            tracker.record("quote_age", 0.05)
            tracker.record("exec_latency", 0.15)
        report = tracker.evaluate()
        assert report.all_passing
        assert not report.is_throttled

        # Degraded: quote age breaches.
        for _ in range(100):
            tracker.record("quote_age", 0.8)
        report = tracker.evaluate()  # Breach 1.
        assert report.any_breached
        assert not report.should_throttle  # Need 2 consecutive.

        report = tracker.evaluate()  # Breach 2 → throttle.
        assert report.should_throttle
        assert report.is_throttled

        # Recovery: quotes improve.
        tracker._samples["quote_age"] = [0.05] * 100
        report = tracker.evaluate()  # Cooldown 1.
        assert report.is_throttled  # Still in cooldown.

        report = tracker.evaluate()  # Cooldown 2 → exit.
        assert not report.is_throttled
