"""Tests for Phase 4H: Exposure and concentration controls."""

from __future__ import annotations

import pytest

from arb_bot.exposure_manager import (
    ExposureCheckResult,
    ExposureConfig,
    ExposureManager,
    ExposureSnapshot,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mgr(**kw) -> ExposureManager:
    return ExposureManager(ExposureConfig(**kw))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self) -> None:
        cfg = ExposureConfig()
        assert cfg.max_total_exposure == 5000.0
        assert cfg.max_venue_exposure == 3000.0
        assert cfg.max_category_exposure == 2000.0
        assert cfg.max_venue_fraction == 0.60
        assert cfg.max_category_fraction == 0.40
        assert cfg.max_single_market == 1000.0
        assert cfg.max_open_positions == 20

    def test_frozen(self) -> None:
        cfg = ExposureConfig()
        with pytest.raises(AttributeError):
            cfg.max_total_exposure = 10000.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Add / remove positions
# ---------------------------------------------------------------------------


class TestPositionTracking:
    def test_add_position(self) -> None:
        m = _mgr()
        m.add_position("kalshi", "politics", capital=500.0, market_id="m1")
        snap = m.snapshot()
        assert snap.position_count == 1
        assert snap.total_exposure == 500.0

    def test_add_multiple(self) -> None:
        m = _mgr()
        m.add_position("kalshi", "politics", capital=500.0, market_id="m1")
        m.add_position("poly", "crypto", capital=300.0, market_id="m2")
        snap = m.snapshot()
        assert snap.position_count == 2
        assert snap.total_exposure == 800.0

    def test_remove_position(self) -> None:
        m = _mgr()
        m.add_position("kalshi", "politics", capital=500.0, market_id="m1")
        m.add_position("poly", "crypto", capital=300.0, market_id="m2")
        m.remove_position("m1")
        snap = m.snapshot()
        assert snap.position_count == 1
        assert snap.total_exposure == 300.0


# ---------------------------------------------------------------------------
# Check — allowed
# ---------------------------------------------------------------------------


class TestCheckAllowed:
    def test_within_all_limits(self) -> None:
        m = _mgr()
        check = m.check_new_position("kalshi", "politics", capital=100.0)
        assert check.allowed is True
        assert len(check.breached_limits) == 0

    def test_max_additional_capital(self) -> None:
        m = _mgr(max_total_exposure=1000.0)
        m.add_position("kalshi", "politics", capital=700.0)
        check = m.check_new_position("kalshi", "politics", capital=100.0)
        assert check.max_additional_capital == pytest.approx(300.0)


# ---------------------------------------------------------------------------
# Check — breaches
# ---------------------------------------------------------------------------


class TestCheckBreaches:
    def test_total_exposure(self) -> None:
        m = _mgr(max_total_exposure=1000.0)
        m.add_position("kalshi", "politics", capital=900.0)
        check = m.check_new_position("poly", "crypto", capital=200.0)
        assert check.allowed is False
        assert "max_total_exposure" in check.breached_limits

    def test_venue_exposure(self) -> None:
        m = _mgr(max_venue_exposure=500.0)
        m.add_position("kalshi", "politics", capital=400.0)
        check = m.check_new_position("kalshi", "crypto", capital=200.0)
        assert "max_venue_exposure" in check.breached_limits

    def test_category_exposure(self) -> None:
        m = _mgr(max_category_exposure=500.0)
        m.add_position("kalshi", "politics", capital=400.0)
        check = m.check_new_position("poly", "politics", capital=200.0)
        assert "max_category_exposure" in check.breached_limits

    def test_venue_fraction(self) -> None:
        m = _mgr(max_venue_fraction=0.50, max_total_exposure=10000.0)
        m.add_position("poly", "crypto", capital=100.0)
        # New position: venue kalshi gets 900/(100+900) = 90% > 50%.
        check = m.check_new_position("kalshi", "politics", capital=900.0)
        assert "max_venue_fraction" in check.breached_limits

    def test_category_fraction(self) -> None:
        m = _mgr(max_category_fraction=0.40, max_total_exposure=10000.0)
        m.add_position("kalshi", "sports", capital=100.0)
        # politics gets 600/700 = 85.7% > 40%.
        check = m.check_new_position("kalshi", "politics", capital=600.0)
        assert "max_category_fraction" in check.breached_limits

    def test_single_market(self) -> None:
        m = _mgr(max_single_market=500.0)
        m.add_position("kalshi", "politics", capital=400.0, market_id="m1")
        check = m.check_new_position("kalshi", "politics", capital=200.0, market_id="m1")
        assert "max_single_market" in check.breached_limits

    def test_max_open_positions(self) -> None:
        m = _mgr(max_open_positions=2)
        m.add_position("kalshi", "a", capital=10.0, market_id="m1")
        m.add_position("kalshi", "b", capital=10.0, market_id="m2")
        check = m.check_new_position("kalshi", "c", capital=10.0)
        assert "max_open_positions" in check.breached_limits

    def test_multiple_breaches(self) -> None:
        m = _mgr(max_total_exposure=100.0, max_venue_exposure=50.0)
        m.add_position("kalshi", "politics", capital=80.0)
        check = m.check_new_position("kalshi", "politics", capital=30.0)
        assert "max_total_exposure" in check.breached_limits
        assert "max_venue_exposure" in check.breached_limits


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


class TestSnapshot:
    def test_venue_breakdown(self) -> None:
        m = _mgr()
        m.add_position("kalshi", "politics", capital=300.0)
        m.add_position("poly", "crypto", capital=200.0)
        snap = m.snapshot()
        assert snap.venue_exposures["kalshi"] == 300.0
        assert snap.venue_exposures["poly"] == 200.0

    def test_category_breakdown(self) -> None:
        m = _mgr()
        m.add_position("kalshi", "politics", capital=300.0)
        m.add_position("poly", "politics", capital=200.0)
        snap = m.snapshot()
        assert snap.category_exposures["politics"] == 500.0

    def test_utilization(self) -> None:
        m = _mgr(max_total_exposure=1000.0)
        m.add_position("kalshi", "politics", capital=500.0)
        snap = m.snapshot()
        assert snap.utilization == pytest.approx(0.50)

    def test_empty(self) -> None:
        m = _mgr()
        snap = m.snapshot()
        assert snap.total_exposure == 0.0
        assert snap.position_count == 0
        assert snap.utilization == 0.0


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clears_all(self) -> None:
        m = _mgr()
        m.add_position("kalshi", "politics", capital=500.0)
        m.clear()
        assert m.snapshot().position_count == 0


# ---------------------------------------------------------------------------
# Config property
# ---------------------------------------------------------------------------


class TestConfigProperty:
    def test_config_accessible(self) -> None:
        cfg = ExposureConfig(max_total_exposure=10000.0)
        m = ExposureManager(cfg)
        assert m.config.max_total_exposure == 10000.0


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_portfolio_buildup(self) -> None:
        """Build up portfolio and hit various limits."""
        m = _mgr(
            max_total_exposure=3000.0,
            max_venue_exposure=2000.0,
            max_category_exposure=1500.0,
            max_venue_fraction=0.70,
            max_category_fraction=0.60,
            max_single_market=800.0,
            max_open_positions=5,
        )

        # First position — all clear (no existing positions).
        check = m.check_new_position("kalshi", "politics", 400.0, "m1")
        assert check.allowed is True
        m.add_position("kalshi", "politics", 400.0, "m1")

        # Second — different venue to maintain balance.
        check = m.check_new_position("poly", "crypto", 400.0, "m2")
        assert check.allowed is True
        m.add_position("poly", "crypto", 400.0, "m2")

        # Third — balanced addition.
        check = m.check_new_position("kalshi", "sports", 300.0, "m3")
        assert check.allowed is True
        m.add_position("kalshi", "sports", 300.0, "m3")

        # Fourth — would exceed total.
        check = m.check_new_position("poly", "weather", 2000.0, "m4")
        assert check.allowed is False
        assert "max_total_exposure" in check.breached_limits

        # Smaller amount OK.
        check = m.check_new_position("poly", "weather", 200.0, "m4")
        assert check.allowed is True
