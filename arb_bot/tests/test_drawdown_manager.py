"""Tests for Phase 4F: Drawdown-adaptive de-risking."""

from __future__ import annotations

import pytest

from arb_bot.drawdown_manager import (
    DrawdownConfig,
    DrawdownManager,
    DrawdownResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dd(**kw) -> DrawdownManager:
    return DrawdownManager(DrawdownConfig(**kw))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self) -> None:
        cfg = DrawdownConfig()
        assert cfg.tiers == ((0.05, 0.75), (0.10, 0.50), (0.15, 0.25))
        assert cfg.shutdown_drawdown == 0.20
        assert cfg.recovery_buffer == 0.50
        assert cfg.min_equity_history == 1

    def test_custom(self) -> None:
        cfg = DrawdownConfig(shutdown_drawdown=0.30, recovery_buffer=0.25)
        assert cfg.shutdown_drawdown == 0.30
        assert cfg.recovery_buffer == 0.25

    def test_frozen(self) -> None:
        cfg = DrawdownConfig()
        with pytest.raises(AttributeError):
            cfg.shutdown_drawdown = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Update equity
# ---------------------------------------------------------------------------


class TestUpdateEquity:
    def test_hwm_tracks_peak(self) -> None:
        dd = _dd()
        dd.update_equity(1000.0)
        dd.update_equity(1100.0)
        dd.update_equity(1050.0)
        assert dd.high_water_mark == 1100.0

    def test_drawdown_pct(self) -> None:
        dd = _dd()
        dd.update_equity(1000.0)
        dd.update_equity(900.0)
        assert dd.drawdown_pct == pytest.approx(0.10)

    def test_no_drawdown_at_peak(self) -> None:
        dd = _dd()
        dd.update_equity(1000.0)
        assert dd.drawdown_pct == 0.0

    def test_zero_equity(self) -> None:
        dd = _dd()
        dd.update_equity(0.0)
        assert dd.drawdown_pct == 0.0


# ---------------------------------------------------------------------------
# Compute multiplier — no drawdown
# ---------------------------------------------------------------------------


class TestNoDrawdown:
    def test_at_hwm(self) -> None:
        dd = _dd()
        dd.update_equity(1000.0)
        r = dd.compute_multiplier()
        assert r.multiplier == 1.0
        assert r.active_tier == -1
        assert r.shutdown is False

    def test_above_previous_hwm(self) -> None:
        dd = _dd()
        dd.update_equity(1000.0)
        dd.update_equity(1100.0)
        r = dd.compute_multiplier()
        assert r.multiplier == 1.0


# ---------------------------------------------------------------------------
# Compute multiplier — tiered drawdown
# ---------------------------------------------------------------------------


class TestTieredDrawdown:
    def test_below_first_tier(self) -> None:
        dd = _dd()
        dd.update_equity(1000.0)
        dd.update_equity(960.0)  # 4% drawdown, below 5% tier.
        r = dd.compute_multiplier()
        assert r.multiplier == 1.0
        assert r.active_tier == -1

    def test_first_tier(self) -> None:
        dd = _dd()
        dd.update_equity(1000.0)
        dd.update_equity(940.0)  # 6% drawdown → tier 0 (0.75x).
        r = dd.compute_multiplier()
        assert r.multiplier == 0.75
        assert r.active_tier == 0

    def test_second_tier(self) -> None:
        dd = _dd()
        dd.update_equity(1000.0)
        dd.update_equity(880.0)  # 12% drawdown → tier 1 (0.50x).
        r = dd.compute_multiplier()
        assert r.multiplier == 0.50
        assert r.active_tier == 1

    def test_third_tier(self) -> None:
        dd = _dd()
        dd.update_equity(1000.0)
        dd.update_equity(840.0)  # 16% drawdown → tier 2 (0.25x).
        r = dd.compute_multiplier()
        assert r.multiplier == 0.25
        assert r.active_tier == 2

    def test_exactly_at_tier_boundary(self) -> None:
        dd = _dd()
        dd.update_equity(1000.0)
        dd.update_equity(950.0)  # Exactly 5% → tier 0.
        r = dd.compute_multiplier()
        assert r.multiplier == 0.75
        assert r.active_tier == 0

    def test_custom_tiers(self) -> None:
        dd = _dd(tiers=((0.03, 0.80), (0.08, 0.40)))
        dd.update_equity(1000.0)
        dd.update_equity(950.0)  # 5% → tier 0 (0.80x).
        r = dd.compute_multiplier()
        assert r.multiplier == 0.80


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


class TestShutdown:
    def test_triggers_at_threshold(self) -> None:
        dd = _dd(shutdown_drawdown=0.20)
        dd.update_equity(1000.0)
        dd.update_equity(780.0)  # 22% drawdown.
        assert dd.is_shutdown is True
        r = dd.compute_multiplier()
        assert r.multiplier == 0.0
        assert r.shutdown is True

    def test_exactly_at_shutdown(self) -> None:
        dd = _dd(shutdown_drawdown=0.20)
        dd.update_equity(1000.0)
        dd.update_equity(800.0)  # Exactly 20%.
        assert dd.is_shutdown is True

    def test_recovery_needs_buffer(self) -> None:
        dd = _dd(shutdown_drawdown=0.20, recovery_buffer=0.50)
        dd.update_equity(1000.0)
        dd.update_equity(780.0)  # 22% dd, triggers shutdown. HWM=1000, shutdown_equity=780.
        assert dd.is_shutdown is True

        # Recovery target = 780 + (1000 - 780) * 0.50 = 780 + 110 = 890.
        dd.update_equity(850.0)  # Still below 890.
        r = dd.compute_multiplier()
        assert r.shutdown is True  # Still shut down.
        assert r.recovering is True

        dd.update_equity(895.0)  # Above 890 → recovered.
        r = dd.compute_multiplier()
        assert r.shutdown is False

    def test_new_hwm_clears_shutdown(self) -> None:
        dd = _dd(shutdown_drawdown=0.20)
        dd.update_equity(1000.0)
        dd.update_equity(780.0)
        assert dd.is_shutdown is True
        dd.update_equity(1010.0)  # New HWM.
        assert dd.is_shutdown is False


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_current_equity(self) -> None:
        dd = _dd()
        dd.update_equity(1000.0)
        assert dd.current_equity == 1000.0

    def test_high_water_mark(self) -> None:
        dd = _dd()
        dd.update_equity(1000.0)
        dd.update_equity(800.0)
        assert dd.high_water_mark == 1000.0


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_resets_all(self) -> None:
        dd = _dd()
        dd.update_equity(1000.0)
        dd.update_equity(700.0)
        dd.compute_multiplier()
        dd.reset()
        assert dd.high_water_mark == 0.0
        assert dd.current_equity == 0.0
        assert dd.is_shutdown is False
        assert dd.drawdown_pct == 0.0


# ---------------------------------------------------------------------------
# Config property
# ---------------------------------------------------------------------------


class TestConfigProperty:
    def test_config_accessible(self) -> None:
        cfg = DrawdownConfig(shutdown_drawdown=0.15)
        dd = DrawdownManager(cfg)
        assert dd.config.shutdown_drawdown == 0.15


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_full_lifecycle(self) -> None:
        """Simulate equity curve with drawdown and recovery."""
        dd = _dd()

        # Start trading, equity grows.
        dd.update_equity(10_000.0)
        r = dd.compute_multiplier()
        assert r.multiplier == 1.0

        dd.update_equity(10_500.0)
        r = dd.compute_multiplier()
        assert r.multiplier == 1.0

        # Small drawdown — tier 1.
        dd.update_equity(9_900.0)  # ~5.7% from 10500.
        r = dd.compute_multiplier()
        assert r.multiplier == 0.75

        # Deeper drawdown — tier 2.
        dd.update_equity(9_200.0)  # ~12.4% from 10500.
        r = dd.compute_multiplier()
        assert r.multiplier == 0.50

        # Recovery.
        dd.update_equity(10_200.0)
        r = dd.compute_multiplier()
        assert r.multiplier >= 0.75  # Back within tier 0 range.

        # Full recovery to new HWM.
        dd.update_equity(10_600.0)
        r = dd.compute_multiplier()
        assert r.multiplier == 1.0
