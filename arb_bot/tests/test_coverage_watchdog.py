"""Tests for Phase 5D: Cross/parity coverage watchdog."""

from __future__ import annotations

import pytest

from arb_bot.coverage_watchdog import (
    CoverageReport,
    CoverageWatchdog,
    CoverageWatchdogConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wd(**kw) -> CoverageWatchdog:
    return CoverageWatchdog(CoverageWatchdogConfig(**kw))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self) -> None:
        cfg = CoverageWatchdogConfig()
        assert cfg.cross_coverage_threshold == 0.80
        assert cfg.parity_coverage_threshold == 0.80
        assert cfg.alert_cooldown_seconds == 60.0
        assert cfg.repair_batch_size == 10
        assert cfg.stale_timeout_seconds == 300.0

    def test_frozen(self) -> None:
        cfg = CoverageWatchdogConfig()
        with pytest.raises(AttributeError):
            cfg.cross_coverage_threshold = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------


class TestRegister:
    def test_register_cross_pair(self) -> None:
        wd = _wd()
        wd.register_cross_pair("kalshi:BTC", "poly:BTC")
        assert wd.cross_pair_count() == 1

    def test_register_parity_rule(self) -> None:
        wd = _wd()
        wd.register_parity_rule("BTC_yes_no_sum")
        assert wd.parity_rule_count() == 1

    def test_no_duplicate_register(self) -> None:
        wd = _wd()
        wd.register_cross_pair("kalshi:BTC", "poly:BTC")
        wd.register_cross_pair("kalshi:BTC", "poly:BTC")
        assert wd.cross_pair_count() == 1

    def test_multiple_pairs(self) -> None:
        wd = _wd()
        wd.register_cross_pair("k:A", "p:A")
        wd.register_cross_pair("k:B", "p:B")
        assert wd.cross_pair_count() == 2


# ---------------------------------------------------------------------------
# Update coverage
# ---------------------------------------------------------------------------


class TestUpdateCoverage:
    def test_update_cross(self) -> None:
        wd = _wd()
        wd.register_cross_pair("k:A", "p:A")
        wd.update_cross_coverage("k:A", "p:A", covered=True, now=100.0)
        state = wd.get_cross_state("k:A", "p:A")
        assert state is not None
        assert state.covered is True
        assert state.total_checks == 1
        assert state.total_covered == 1

    def test_update_parity(self) -> None:
        wd = _wd()
        wd.register_parity_rule("rule_1")
        wd.update_parity_coverage("rule_1", covered=True, now=100.0)
        state = wd.get_parity_state("rule_1")
        assert state is not None
        assert state.covered is True

    def test_consecutive_uncovered(self) -> None:
        wd = _wd()
        wd.register_cross_pair("k:A", "p:A")
        wd.update_cross_coverage("k:A", "p:A", covered=False, now=100.0)
        wd.update_cross_coverage("k:A", "p:A", covered=False, now=101.0)
        state = wd.get_cross_state("k:A", "p:A")
        assert state is not None
        assert state.consecutive_uncovered == 2

    def test_consecutive_resets_on_covered(self) -> None:
        wd = _wd()
        wd.register_cross_pair("k:A", "p:A")
        wd.update_cross_coverage("k:A", "p:A", covered=False, now=100.0)
        wd.update_cross_coverage("k:A", "p:A", covered=True, now=101.0)
        state = wd.get_cross_state("k:A", "p:A")
        assert state is not None
        assert state.consecutive_uncovered == 0

    def test_update_nonexistent_is_noop(self) -> None:
        wd = _wd()
        wd.update_cross_coverage("nope", "nope", covered=True, now=100.0)
        # No error.


# ---------------------------------------------------------------------------
# Stale timeout
# ---------------------------------------------------------------------------


class TestStaleTimeout:
    def test_stale_data_not_covered(self) -> None:
        wd = _wd(stale_timeout_seconds=60.0)
        wd.register_cross_pair("k:A", "p:A")
        wd.update_cross_coverage("k:A", "p:A", covered=True, now=100.0)

        report = wd.report(now=200.0)  # 100s > 60s stale.
        assert report.cross_covered == 0

    def test_fresh_data_covered(self) -> None:
        wd = _wd(stale_timeout_seconds=60.0)
        wd.register_cross_pair("k:A", "p:A")
        wd.update_cross_coverage("k:A", "p:A", covered=True, now=100.0)

        report = wd.report(now=130.0)  # 30s < 60s.
        assert report.cross_covered == 1

    def test_never_updated_is_uncovered(self) -> None:
        wd = _wd()
        wd.register_cross_pair("k:A", "p:A")
        report = wd.report(now=100.0)
        assert report.cross_covered == 0


# ---------------------------------------------------------------------------
# Report — no alert
# ---------------------------------------------------------------------------


class TestReportNoAlert:
    def test_full_coverage(self) -> None:
        wd = _wd(cross_coverage_threshold=0.80)
        for i in range(5):
            wd.register_cross_pair(f"k:{i}", f"p:{i}")
            wd.update_cross_coverage(f"k:{i}", f"p:{i}", covered=True, now=100.0)

        report = wd.report(now=100.0)
        assert report.cross_coverage == pytest.approx(1.0)
        assert report.cross_alert is False
        assert report.any_alert is False

    def test_no_items(self) -> None:
        wd = _wd()
        report = wd.report(now=100.0)
        assert report.cross_coverage == 1.0  # No items = vacuously covered.
        assert report.any_alert is False

    def test_at_threshold(self) -> None:
        wd = _wd(cross_coverage_threshold=0.80)
        for i in range(10):
            wd.register_cross_pair(f"k:{i}", f"p:{i}")
        # Cover 8 out of 10 = 80%.
        for i in range(8):
            wd.update_cross_coverage(f"k:{i}", f"p:{i}", covered=True, now=100.0)

        report = wd.report(now=100.0)
        assert report.cross_coverage == pytest.approx(0.80)
        assert report.cross_alert is False


# ---------------------------------------------------------------------------
# Report — alert
# ---------------------------------------------------------------------------


class TestReportAlert:
    def test_cross_alert(self) -> None:
        wd = _wd(cross_coverage_threshold=0.80)
        for i in range(10):
            wd.register_cross_pair(f"k:{i}", f"p:{i}")
        # Only cover 5 = 50% < 80%.
        for i in range(5):
            wd.update_cross_coverage(f"k:{i}", f"p:{i}", covered=True, now=100.0)

        report = wd.report(now=100.0)
        assert report.cross_coverage == pytest.approx(0.50)
        assert report.cross_alert is True
        assert report.any_alert is True
        assert len(report.uncovered_cross_pairs) == 5

    def test_parity_alert(self) -> None:
        wd = _wd(parity_coverage_threshold=0.90)
        for i in range(10):
            wd.register_parity_rule(f"rule_{i}")
        # Cover 7 = 70% < 90%.
        for i in range(7):
            wd.update_parity_coverage(f"rule_{i}", covered=True, now=100.0)

        report = wd.report(now=100.0)
        assert report.parity_alert is True
        assert len(report.uncovered_parity_rules) == 3

    def test_both_alert(self) -> None:
        wd = _wd(cross_coverage_threshold=0.90, parity_coverage_threshold=0.90)
        wd.register_cross_pair("k:A", "p:A")
        wd.register_parity_rule("rule_1")
        # Neither covered.
        report = wd.report(now=100.0)
        assert report.cross_alert is True
        assert report.parity_alert is True
        assert report.any_alert is True


# ---------------------------------------------------------------------------
# Repair candidates
# ---------------------------------------------------------------------------


class TestRepairCandidates:
    def test_selects_uncovered(self) -> None:
        wd = _wd(repair_batch_size=5, alert_cooldown_seconds=0.0)
        for i in range(3):
            wd.register_cross_pair(f"k:{i}", f"p:{i}")
        # All uncovered.
        report = wd.report(now=100.0)
        assert len(report.repair_candidates) == 3

    def test_batch_size_limit(self) -> None:
        wd = _wd(repair_batch_size=2, alert_cooldown_seconds=0.0)
        for i in range(5):
            wd.register_cross_pair(f"k:{i}", f"p:{i}")
        report = wd.report(now=100.0)
        assert len(report.repair_candidates) <= 2

    def test_prioritizes_consecutive_uncovered(self) -> None:
        wd = _wd(repair_batch_size=10, alert_cooldown_seconds=0.0)
        wd.register_cross_pair("k:A", "p:A")
        wd.register_cross_pair("k:B", "p:B")

        # B has more consecutive uncovered.
        wd.update_cross_coverage("k:B", "p:B", covered=False, now=100.0)
        wd.update_cross_coverage("k:B", "p:B", covered=False, now=101.0)
        wd.update_cross_coverage("k:B", "p:B", covered=False, now=102.0)
        # A has 1.
        wd.update_cross_coverage("k:A", "p:A", covered=False, now=100.0)

        report = wd.report(now=103.0)
        # B should be first (highest consecutive_uncovered).
        assert report.repair_candidates[0] == "k:B|p:B"

    def test_cooldown_prevents_repeated_alerts(self) -> None:
        wd = _wd(repair_batch_size=10, alert_cooldown_seconds=60.0)
        wd.register_cross_pair("k:A", "p:A")

        r1 = wd.report(now=100.0)
        assert len(r1.repair_candidates) == 1

        # Within cooldown — should not appear.
        r2 = wd.report(now=130.0)
        assert len(r2.repair_candidates) == 0

        # After cooldown.
        r3 = wd.report(now=161.0)
        assert len(r3.repair_candidates) == 1


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clears_all(self) -> None:
        wd = _wd()
        wd.register_cross_pair("k:A", "p:A")
        wd.register_parity_rule("rule_1")
        wd.clear()
        assert wd.cross_pair_count() == 0
        assert wd.parity_rule_count() == 0


# ---------------------------------------------------------------------------
# Config property
# ---------------------------------------------------------------------------


class TestConfigProperty:
    def test_config_accessible(self) -> None:
        cfg = CoverageWatchdogConfig(cross_coverage_threshold=0.95)
        wd = CoverageWatchdog(cfg)
        assert wd.config.cross_coverage_threshold == 0.95


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_coverage_degradation_and_repair(self) -> None:
        """Coverage drops, repair brings it back."""
        wd = _wd(
            cross_coverage_threshold=0.80,
            stale_timeout_seconds=60.0,
            alert_cooldown_seconds=0.0,
        )

        # Register 10 pairs, all covered.
        for i in range(10):
            wd.register_cross_pair(f"k:{i}", f"p:{i}")
            wd.update_cross_coverage(f"k:{i}", f"p:{i}", covered=True, now=100.0)

        report = wd.report(now=100.0)
        assert report.cross_alert is False
        assert report.cross_coverage == pytest.approx(1.0)

        # 4 pairs go stale → 6/10 = 60% < 80%.
        for i in range(4):
            wd.update_cross_coverage(f"k:{i}", f"p:{i}", covered=False, now=150.0)

        report = wd.report(now=150.0)
        assert report.cross_alert is True
        assert report.cross_coverage == pytest.approx(0.60)
        assert len(report.repair_candidates) == 4

        # Repair: re-cover the 4 pairs.
        for pair_key in report.repair_candidates:
            parts = pair_key.split("|")
            wd.update_cross_coverage(parts[0], parts[1], covered=True, now=160.0)

        report2 = wd.report(now=160.0)
        assert report2.cross_alert is False
        assert report2.cross_coverage == pytest.approx(1.0)
