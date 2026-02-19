"""Tests for Phase 2C: Delta/snapshot reconciliation loop."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from arb_bot.models import BinaryQuote
from arb_bot.framework.reconciliation import (
    DivergenceRecord,
    ReconciliationConfig,
    ReconciliationLoop,
    ReconciliationResult,
    ReconciliationStats,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NOW = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def _q(
    venue: str = "kalshi",
    market_id: str = "M1",
    yes: float = 0.55,
    no: float = 0.45,
    yes_size: float = 20.0,
    no_size: float = 20.0,
) -> BinaryQuote:
    return BinaryQuote(
        venue=venue,
        market_id=market_id,
        yes_buy_price=yes,
        no_buy_price=no,
        yes_buy_size=yes_size,
        no_buy_size=no_size,
        observed_at=NOW,
    )


def _cache(*quotes: BinaryQuote) -> dict[tuple[str, str], BinaryQuote]:
    """Build a stream cache dict from a list of quotes."""
    return {(q.venue, q.market_id): q for q in quotes}


# ---------------------------------------------------------------------------
# ReconciliationConfig
# ---------------------------------------------------------------------------


class TestReconciliationConfig:
    def test_defaults(self) -> None:
        cfg = ReconciliationConfig()
        assert cfg.price_divergence_threshold == 0.05
        assert cfg.size_divergence_ratio == 0.5
        assert cfg.reconciliation_interval_seconds == 60.0
        assert cfg.max_snapshot_age_seconds == 120.0
        assert cfg.divergence_count_for_resync == 2
        assert cfg.enabled is True


# ---------------------------------------------------------------------------
# ReconciliationStats
# ---------------------------------------------------------------------------


class TestReconciliationStats:
    def test_defaults(self) -> None:
        stats = ReconciliationStats()
        assert stats.total_checks == 0
        assert stats.total_divergences == 0
        assert stats.resyncs_triggered == 0
        assert stats.markets_checked == 0
        assert stats.last_check_at is None

    def test_reset(self) -> None:
        stats = ReconciliationStats(
            total_checks=5,
            total_divergences=3,
            resyncs_triggered=1,
            markets_checked=10,
            last_check_at=NOW,
        )
        stats.reset()
        assert stats.total_checks == 0
        assert stats.last_check_at is None


# ---------------------------------------------------------------------------
# ReconciliationResult
# ---------------------------------------------------------------------------


class TestReconciliationResult:
    def test_no_divergence(self) -> None:
        result = ReconciliationResult(
            divergences=(),
            markets_needing_resync=(),
            markets_checked=5,
            snapshot_count=10,
        )
        assert result.has_divergence is False
        assert result.resync_needed is False
        assert "OK" in result.summary
        assert "5" in result.summary

    def test_with_divergence(self) -> None:
        div = DivergenceRecord(
            market_key="kalshi/M1",
            field_name="yes_buy_price",
            stream_value=0.60,
            snapshot_value=0.50,
            delta=0.10,
            detected_at=NOW,
        )
        result = ReconciliationResult(
            divergences=(div,),
            markets_needing_resync=(),
            markets_checked=5,
            snapshot_count=10,
        )
        assert result.has_divergence is True
        assert result.resync_needed is False
        assert "1 divergence" in result.summary

    def test_with_resync(self) -> None:
        div = DivergenceRecord(
            market_key="kalshi/M1",
            field_name="yes_buy_price",
            stream_value=0.60,
            snapshot_value=0.50,
            delta=0.10,
            detected_at=NOW,
        )
        result = ReconciliationResult(
            divergences=(div,),
            markets_needing_resync=(("kalshi", "M1"),),
            markets_checked=5,
            snapshot_count=10,
        )
        assert result.resync_needed is True
        assert "resync needed" in result.summary
        assert "kalshi/M1" in result.summary


# ---------------------------------------------------------------------------
# DivergenceRecord
# ---------------------------------------------------------------------------


class TestDivergenceRecord:
    def test_fields(self) -> None:
        rec = DivergenceRecord(
            market_key="kalshi/M1",
            field_name="yes_buy_price",
            stream_value=0.60,
            snapshot_value=0.50,
            delta=0.10,
            detected_at=NOW,
        )
        assert rec.market_key == "kalshi/M1"
        assert rec.delta == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# Snapshot management
# ---------------------------------------------------------------------------


class TestSnapshotManagement:
    def test_update_snapshots(self) -> None:
        loop = ReconciliationLoop()
        count = loop.update_snapshots([_q(), _q(market_id="M2")], now=NOW)
        assert count == 2
        assert loop.snapshot_count() == 2

    def test_has_snapshot(self) -> None:
        loop = ReconciliationLoop()
        loop.update_snapshots([_q()], now=NOW)
        assert loop.has_snapshot("kalshi", "M1") is True
        assert loop.has_snapshot("kalshi", "M2") is False

    def test_clear_snapshot(self) -> None:
        loop = ReconciliationLoop()
        loop.update_snapshots([_q()], now=NOW)
        assert loop.clear_snapshot("kalshi", "M1") is True
        assert loop.snapshot_count() == 0

    def test_clear_nonexistent_snapshot(self) -> None:
        loop = ReconciliationLoop()
        assert loop.clear_snapshot("kalshi", "M1") is False

    def test_clear_all_snapshots(self) -> None:
        loop = ReconciliationLoop()
        loop.update_snapshots([_q(), _q(market_id="M2")], now=NOW)
        loop.clear_all_snapshots()
        assert loop.snapshot_count() == 0

    def test_update_overwrites_previous(self) -> None:
        loop = ReconciliationLoop()
        loop.update_snapshots([_q(yes=0.50)], now=NOW)
        loop.update_snapshots([_q(yes=0.60)], now=NOW)
        assert loop.snapshot_count() == 1  # Same key, just updated


# ---------------------------------------------------------------------------
# Price divergence detection
# ---------------------------------------------------------------------------


class TestPriceDivergence:
    def test_no_divergence_matching_prices(self) -> None:
        loop = ReconciliationLoop(ReconciliationConfig(price_divergence_threshold=0.05))
        loop.update_snapshots([_q(yes=0.55, no=0.45)], now=NOW)
        result = loop.check(_cache(_q(yes=0.55, no=0.45)), now=NOW)
        assert result.has_divergence is False
        assert result.markets_checked == 1

    def test_small_divergence_within_threshold(self) -> None:
        loop = ReconciliationLoop(ReconciliationConfig(price_divergence_threshold=0.05))
        loop.update_snapshots([_q(yes=0.55, no=0.45)], now=NOW)
        result = loop.check(_cache(_q(yes=0.57, no=0.43)), now=NOW)
        assert result.has_divergence is False

    def test_large_price_divergence_detected(self) -> None:
        loop = ReconciliationLoop(ReconciliationConfig(price_divergence_threshold=0.05))
        loop.update_snapshots([_q(yes=0.55, no=0.45)], now=NOW)
        result = loop.check(_cache(_q(yes=0.70, no=0.30)), now=NOW)
        assert result.has_divergence is True
        assert len(result.divergences) == 2  # Both yes and no diverged

    def test_yes_only_divergence(self) -> None:
        loop = ReconciliationLoop(ReconciliationConfig(price_divergence_threshold=0.05))
        loop.update_snapshots([_q(yes=0.55, no=0.45)], now=NOW)
        result = loop.check(_cache(_q(yes=0.65, no=0.45)), now=NOW)
        assert len(result.divergences) == 1
        assert result.divergences[0].field_name == "yes_buy_price"
        assert result.divergences[0].delta == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# Size divergence detection
# ---------------------------------------------------------------------------


class TestSizeDivergence:
    def test_no_size_divergence(self) -> None:
        loop = ReconciliationLoop(ReconciliationConfig(size_divergence_ratio=0.5))
        loop.update_snapshots([_q(yes_size=20.0, no_size=20.0)], now=NOW)
        result = loop.check(_cache(_q(yes_size=18.0, no_size=22.0)), now=NOW)
        assert result.has_divergence is False

    def test_large_size_divergence(self) -> None:
        loop = ReconciliationLoop(ReconciliationConfig(
            price_divergence_threshold=1.0,  # Disable price check
            size_divergence_ratio=0.5,
        ))
        loop.update_snapshots([_q(yes_size=100.0, no_size=100.0)], now=NOW)
        result = loop.check(_cache(_q(yes_size=10.0, no_size=100.0)), now=NOW)
        assert result.has_divergence is True
        assert len(result.divergences) == 1
        assert result.divergences[0].field_name == "yes_buy_size"

    def test_zero_size_no_division_error(self) -> None:
        loop = ReconciliationLoop(ReconciliationConfig(
            price_divergence_threshold=1.0,
            size_divergence_ratio=0.5,
        ))
        loop.update_snapshots([_q(yes_size=0.0, no_size=0.0)], now=NOW)
        result = loop.check(_cache(_q(yes_size=0.0, no_size=0.0)), now=NOW)
        assert result.has_divergence is False


# ---------------------------------------------------------------------------
# Stale snapshot skipping
# ---------------------------------------------------------------------------


class TestStaleSnapshot:
    def test_fresh_snapshot_compared(self) -> None:
        loop = ReconciliationLoop(ReconciliationConfig(max_snapshot_age_seconds=60.0))
        loop.update_snapshots([_q(yes=0.55)], now=NOW)
        # Check 30s later — snapshot still fresh.
        check_time = NOW + timedelta(seconds=30)
        result = loop.check(_cache(_q(yes=0.55)), now=check_time)
        assert result.markets_checked == 1

    def test_stale_snapshot_skipped(self) -> None:
        loop = ReconciliationLoop(ReconciliationConfig(max_snapshot_age_seconds=60.0))
        loop.update_snapshots([_q(yes=0.55)], now=NOW)
        # Check 90s later — snapshot too old.
        check_time = NOW + timedelta(seconds=90)
        result = loop.check(_cache(_q(yes=0.55)), now=check_time)
        assert result.markets_checked == 0

    def test_missing_stream_quote_skipped(self) -> None:
        loop = ReconciliationLoop()
        loop.update_snapshots([_q()], now=NOW)
        # Stream has no quote for this market.
        result = loop.check({}, now=NOW)
        assert result.markets_checked == 0


# ---------------------------------------------------------------------------
# Consecutive divergence and resync
# ---------------------------------------------------------------------------


class TestResyncTrigger:
    def test_single_divergence_no_resync(self) -> None:
        loop = ReconciliationLoop(ReconciliationConfig(
            price_divergence_threshold=0.05,
            divergence_count_for_resync=2,
        ))
        loop.update_snapshots([_q(yes=0.55)], now=NOW)
        result = loop.check(_cache(_q(yes=0.70)), now=NOW)
        assert result.has_divergence is True
        assert result.resync_needed is False

    def test_consecutive_divergence_triggers_resync(self) -> None:
        loop = ReconciliationLoop(ReconciliationConfig(
            price_divergence_threshold=0.05,
            divergence_count_for_resync=2,
        ))
        loop.update_snapshots([_q(yes=0.55)], now=NOW)
        # First check — divergent.
        loop.check(_cache(_q(yes=0.70)), now=NOW)
        # Second check — still divergent → resync.
        result = loop.check(_cache(_q(yes=0.70)), now=NOW)
        assert result.resync_needed is True
        assert ("kalshi", "M1") in result.markets_needing_resync

    def test_divergence_counter_resets_on_match(self) -> None:
        loop = ReconciliationLoop(ReconciliationConfig(
            price_divergence_threshold=0.05,
            divergence_count_for_resync=3,
        ))
        loop.update_snapshots([_q(yes=0.55)], now=NOW)
        # First check — divergent.
        loop.check(_cache(_q(yes=0.70)), now=NOW)
        # Second check — matches → reset.
        loop.check(_cache(_q(yes=0.55)), now=NOW)
        # Third check — divergent again (count=1, not 2).
        result = loop.check(_cache(_q(yes=0.70)), now=NOW)
        assert result.resync_needed is False

    def test_reset_divergence_count(self) -> None:
        loop = ReconciliationLoop(ReconciliationConfig(
            price_divergence_threshold=0.05,
            divergence_count_for_resync=2,
        ))
        loop.update_snapshots([_q(yes=0.55)], now=NOW)
        loop.check(_cache(_q(yes=0.70)), now=NOW)
        # Manually reset (e.g., after successful resync).
        loop.reset_divergence_count("kalshi", "M1")
        result = loop.check(_cache(_q(yes=0.70)), now=NOW)
        # Only count=1 now, so no resync yet.
        assert result.resync_needed is False


# ---------------------------------------------------------------------------
# is_due
# ---------------------------------------------------------------------------


class TestIsDue:
    def test_due_on_first_call(self) -> None:
        loop = ReconciliationLoop(ReconciliationConfig(
            reconciliation_interval_seconds=60.0,
        ))
        assert loop.is_due(now=NOW) is True

    def test_not_due_too_soon(self) -> None:
        loop = ReconciliationLoop(ReconciliationConfig(
            reconciliation_interval_seconds=60.0,
        ))
        loop.update_snapshots([_q()], now=NOW)
        loop.check(_cache(_q()), now=NOW)
        assert loop.is_due(now=NOW + timedelta(seconds=30)) is False

    def test_due_after_interval(self) -> None:
        loop = ReconciliationLoop(ReconciliationConfig(
            reconciliation_interval_seconds=60.0,
        ))
        loop.update_snapshots([_q()], now=NOW)
        loop.check(_cache(_q()), now=NOW)
        assert loop.is_due(now=NOW + timedelta(seconds=61)) is True

    def test_disabled(self) -> None:
        loop = ReconciliationLoop(ReconciliationConfig(enabled=False))
        assert loop.is_due(now=NOW) is False


# ---------------------------------------------------------------------------
# Stats tracking
# ---------------------------------------------------------------------------


class TestStatsTracking:
    def test_check_increments_stats(self) -> None:
        loop = ReconciliationLoop()
        loop.update_snapshots([_q()], now=NOW)
        loop.check(_cache(_q()), now=NOW)
        assert loop.stats.total_checks == 1
        assert loop.stats.markets_checked == 1
        assert loop.stats.last_check_at == NOW

    def test_divergence_increments_stats(self) -> None:
        loop = ReconciliationLoop(ReconciliationConfig(
            price_divergence_threshold=0.05,
        ))
        loop.update_snapshots([_q(yes=0.55)], now=NOW)
        loop.check(_cache(_q(yes=0.70)), now=NOW)
        assert loop.stats.total_divergences > 0

    def test_resync_increments_stats(self) -> None:
        loop = ReconciliationLoop(ReconciliationConfig(
            price_divergence_threshold=0.05,
            divergence_count_for_resync=1,
        ))
        loop.update_snapshots([_q(yes=0.55)], now=NOW)
        loop.check(_cache(_q(yes=0.70)), now=NOW)
        assert loop.stats.resyncs_triggered == 1


# ---------------------------------------------------------------------------
# Multiple markets
# ---------------------------------------------------------------------------


class TestMultipleMarkets:
    def test_mixed_markets(self) -> None:
        loop = ReconciliationLoop(ReconciliationConfig(
            price_divergence_threshold=0.05,
        ))
        loop.update_snapshots([
            _q(market_id="M1", yes=0.55, no=0.45),
            _q(market_id="M2", yes=0.60, no=0.40),
        ], now=NOW)
        result = loop.check(_cache(
            _q(market_id="M1", yes=0.55, no=0.45),  # Matches
            _q(market_id="M2", yes=0.80, no=0.20),  # Diverged
        ), now=NOW)
        assert result.markets_checked == 2
        assert result.has_divergence is True
        # Only M2 should have divergences.
        diverged_markets = {d.market_key for d in result.divergences}
        assert "kalshi/M2" in diverged_markets
        assert "kalshi/M1" not in diverged_markets

    def test_cross_venue_markets(self) -> None:
        loop = ReconciliationLoop(ReconciliationConfig(
            price_divergence_threshold=0.05,
        ))
        loop.update_snapshots([
            _q(venue="kalshi", market_id="M1", yes=0.55),
            _q(venue="polymarket", market_id="P1", yes=0.60),
        ], now=NOW)
        result = loop.check(_cache(
            _q(venue="kalshi", market_id="M1", yes=0.55),
            _q(venue="polymarket", market_id="P1", yes=0.60),
        ), now=NOW)
        assert result.markets_checked == 2
        assert result.has_divergence is False


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_full_lifecycle(self) -> None:
        """Simulate: snapshot → stream diverges → resync → re-snapshot → match."""
        loop = ReconciliationLoop(ReconciliationConfig(
            price_divergence_threshold=0.05,
            divergence_count_for_resync=2,
        ))

        # 1. Store snapshot from poll.
        loop.update_snapshots([_q(yes=0.55, no=0.45)], now=NOW)

        # 2. Stream matches — all good.
        r1 = loop.check(_cache(_q(yes=0.55, no=0.45)), now=NOW)
        assert r1.has_divergence is False

        # 3. Stream diverges.
        r2 = loop.check(_cache(_q(yes=0.70, no=0.30)), now=NOW)
        assert r2.has_divergence is True
        assert r2.resync_needed is False  # Only 1 consecutive.

        # 4. Still diverging → resync triggered.
        r3 = loop.check(_cache(_q(yes=0.70, no=0.30)), now=NOW)
        assert r3.resync_needed is True

        # 5. Resync happened — update snapshot with new poll data.
        loop.update_snapshots([_q(yes=0.70, no=0.30)], now=NOW)

        # 6. Now stream matches the new snapshot.
        r4 = loop.check(_cache(_q(yes=0.70, no=0.30)), now=NOW)
        assert r4.has_divergence is False

    def test_config_property(self) -> None:
        cfg = ReconciliationConfig(price_divergence_threshold=0.10)
        loop = ReconciliationLoop(cfg)
        assert loop.config.price_divergence_threshold == 0.10

    def test_now_defaults(self) -> None:
        loop = ReconciliationLoop()
        loop.update_snapshots([_q()])
        result = loop.check(_cache(_q()))
        assert result.markets_checked == 1
