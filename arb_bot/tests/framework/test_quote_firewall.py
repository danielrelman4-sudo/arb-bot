"""Tests for Phase 2B: Quote sanity firewall hardening."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from arb_bot.models import BinaryQuote
from arb_bot.framework.quote_firewall import FirewallConfig, FirewallStats, QuoteFirewall


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
    yes_bid: float | None = None,
    no_bid: float | None = None,
    age_seconds: float = 5.0,
) -> BinaryQuote:
    return BinaryQuote(
        venue=venue,
        market_id=market_id,
        yes_buy_price=yes,
        no_buy_price=no,
        yes_buy_size=yes_size,
        no_buy_size=no_size,
        yes_bid_price=yes_bid,
        no_bid_price=no_bid,
        observed_at=NOW - timedelta(seconds=age_seconds),
    )


# ---------------------------------------------------------------------------
# FirewallConfig
# ---------------------------------------------------------------------------


class TestFirewallConfig:
    def test_defaults(self) -> None:
        config = FirewallConfig()
        assert config.max_complement_sum_deviation == 0.15
        assert config.min_price_floor == 0.01
        assert config.max_price_ceiling == 0.99
        assert config.max_price_jump == 0.0
        assert config.reject_locked_book is True
        assert config.reject_inverted_prices is False


# ---------------------------------------------------------------------------
# Complement sum deviation
# ---------------------------------------------------------------------------


class TestComplementDeviation:
    def test_passes_normal(self) -> None:
        fw = QuoteFirewall(FirewallConfig(max_complement_sum_deviation=0.10))
        accepted, rejected = fw.filter([_q(yes=0.55, no=0.48)], now=NOW)  # sum=1.03
        assert len(accepted) == 1
        assert len(rejected) == 0

    def test_rejects_extreme(self) -> None:
        fw = QuoteFirewall(FirewallConfig(max_complement_sum_deviation=0.10))
        accepted, rejected = fw.filter([_q(yes=0.80, no=0.80)], now=NOW)  # sum=1.60
        assert len(accepted) == 0
        assert "kalshi/M1" in rejected
        assert any("complement_deviation" in r for r in rejected["kalshi/M1"])

    def test_disabled_when_zero(self) -> None:
        fw = QuoteFirewall(FirewallConfig(max_complement_sum_deviation=0.0))
        accepted, _ = fw.filter([_q(yes=0.90, no=0.90)], now=NOW)
        assert len(accepted) == 1


# ---------------------------------------------------------------------------
# Price floor
# ---------------------------------------------------------------------------


class TestPriceFloor:
    def test_passes_normal(self) -> None:
        fw = QuoteFirewall(FirewallConfig(min_price_floor=0.02))
        accepted, _ = fw.filter([_q(yes=0.55, no=0.45)], now=NOW)
        assert len(accepted) == 1

    def test_rejects_below_floor(self) -> None:
        fw = QuoteFirewall(FirewallConfig(min_price_floor=0.05))
        accepted, rejected = fw.filter([_q(yes=0.03, no=0.97)], now=NOW)
        assert len(accepted) == 0
        assert any("below_price_floor" in r for r in rejected["kalshi/M1"])

    def test_disabled_when_zero(self) -> None:
        fw = QuoteFirewall(FirewallConfig(min_price_floor=0.0, max_price_ceiling=1.0))
        accepted, _ = fw.filter([_q(yes=0.001, no=0.999)], now=NOW)
        assert len(accepted) == 1


# ---------------------------------------------------------------------------
# Price ceiling
# ---------------------------------------------------------------------------


class TestPriceCeiling:
    def test_passes_normal(self) -> None:
        fw = QuoteFirewall(FirewallConfig(max_price_ceiling=0.98))
        accepted, _ = fw.filter([_q(yes=0.55, no=0.45)], now=NOW)
        assert len(accepted) == 1

    def test_rejects_above_ceiling(self) -> None:
        fw = QuoteFirewall(FirewallConfig(max_price_ceiling=0.95))
        accepted, rejected = fw.filter([_q(yes=0.97, no=0.03)], now=NOW)
        assert len(accepted) == 0
        assert any("above_price_ceiling" in r for r in rejected["kalshi/M1"])

    def test_disabled_when_one(self) -> None:
        fw = QuoteFirewall(FirewallConfig(max_price_ceiling=1.0))
        accepted, _ = fw.filter([_q(yes=0.99, no=0.01)], now=NOW)
        assert len(accepted) == 1


# ---------------------------------------------------------------------------
# Price jump
# ---------------------------------------------------------------------------


class TestPriceJump:
    def test_no_history_no_rejection(self) -> None:
        fw = QuoteFirewall(FirewallConfig(max_price_jump=0.10))
        accepted, _ = fw.filter([_q(yes=0.55, no=0.45)], now=NOW)
        assert len(accepted) == 1

    def test_small_jump_passes(self) -> None:
        fw = QuoteFirewall(FirewallConfig(max_price_jump=0.10))
        fw.filter([_q(yes=0.55, no=0.45)], now=NOW)
        accepted, _ = fw.filter([_q(yes=0.57, no=0.43)], now=NOW)
        assert len(accepted) == 1

    def test_large_jump_rejected(self) -> None:
        fw = QuoteFirewall(FirewallConfig(max_price_jump=0.10))
        fw.filter([_q(yes=0.55, no=0.45)], now=NOW)
        accepted, rejected = fw.filter([_q(yes=0.80, no=0.20)], now=NOW)
        assert len(accepted) == 0
        assert any("price_jump" in r for r in rejected["kalshi/M1"])

    def test_disabled_when_zero(self) -> None:
        fw = QuoteFirewall(FirewallConfig(max_price_jump=0.0))
        fw.filter([_q(yes=0.10, no=0.90)], now=NOW)
        accepted, _ = fw.filter([_q(yes=0.90, no=0.10)], now=NOW)
        assert len(accepted) == 1

    def test_reset_price_history(self) -> None:
        fw = QuoteFirewall(FirewallConfig(max_price_jump=0.10))
        fw.filter([_q(yes=0.10, no=0.90)], now=NOW)
        fw.reset_price_history()
        accepted, _ = fw.filter([_q(yes=0.90, no=0.10)], now=NOW)
        assert len(accepted) == 1  # No history to compare


# ---------------------------------------------------------------------------
# Quote age
# ---------------------------------------------------------------------------


class TestQuoteAge:
    def test_disabled_by_default(self) -> None:
        fw = QuoteFirewall()
        accepted, _ = fw.filter([_q(age_seconds=9999.0)], now=NOW)
        assert len(accepted) == 1

    def test_passes_fresh(self) -> None:
        fw = QuoteFirewall(FirewallConfig(max_quote_age_seconds=30.0))
        accepted, _ = fw.filter([_q(age_seconds=10.0)], now=NOW)
        assert len(accepted) == 1

    def test_rejects_stale(self) -> None:
        fw = QuoteFirewall(FirewallConfig(max_quote_age_seconds=30.0))
        accepted, rejected = fw.filter([_q(age_seconds=60.0)], now=NOW)
        assert len(accepted) == 0
        assert any("stale" in r for r in rejected["kalshi/M1"])


# ---------------------------------------------------------------------------
# Book depth
# ---------------------------------------------------------------------------


class TestBookDepth:
    def test_disabled_by_default(self) -> None:
        fw = QuoteFirewall()
        accepted, _ = fw.filter([_q(yes_size=0.5, no_size=0.5)], now=NOW)
        assert len(accepted) == 1

    def test_passes_deep(self) -> None:
        fw = QuoteFirewall(FirewallConfig(min_book_depth_contracts=5.0))
        accepted, _ = fw.filter([_q(yes_size=10.0, no_size=10.0)], now=NOW)
        assert len(accepted) == 1

    def test_rejects_thin(self) -> None:
        fw = QuoteFirewall(FirewallConfig(min_book_depth_contracts=10.0))
        accepted, rejected = fw.filter([_q(yes_size=5.0, no_size=15.0)], now=NOW)
        assert len(accepted) == 0
        assert any("thin_book" in r for r in rejected["kalshi/M1"])


# ---------------------------------------------------------------------------
# Locked/crossed book
# ---------------------------------------------------------------------------


class TestLockedBook:
    def test_no_bids_passes(self) -> None:
        fw = QuoteFirewall(FirewallConfig(reject_locked_book=True))
        accepted, _ = fw.filter([_q(yes_bid=None, no_bid=None)], now=NOW)
        assert len(accepted) == 1

    def test_normal_bid_ask_passes(self) -> None:
        fw = QuoteFirewall(FirewallConfig(reject_locked_book=True))
        accepted, _ = fw.filter([_q(yes=0.55, yes_bid=0.53, no=0.45, no_bid=0.43)], now=NOW)
        assert len(accepted) == 1

    def test_locked_book_rejected(self) -> None:
        fw = QuoteFirewall(FirewallConfig(reject_locked_book=True))
        accepted, rejected = fw.filter(
            [_q(yes=0.55, yes_bid=0.55, no=0.45, no_bid=0.43)], now=NOW
        )
        assert len(accepted) == 0
        assert any("locked_book" in r for r in rejected["kalshi/M1"])

    def test_crossed_book_rejected(self) -> None:
        fw = QuoteFirewall(FirewallConfig(reject_locked_book=True))
        accepted, rejected = fw.filter(
            [_q(yes=0.55, yes_bid=0.57)], now=NOW
        )
        assert len(accepted) == 0
        assert any("locked_book" in r for r in rejected["kalshi/M1"])

    def test_disabled(self) -> None:
        fw = QuoteFirewall(FirewallConfig(reject_locked_book=False))
        accepted, _ = fw.filter([_q(yes=0.55, yes_bid=0.60)], now=NOW)
        assert len(accepted) == 1


# ---------------------------------------------------------------------------
# Inverted prices
# ---------------------------------------------------------------------------


class TestInvertedPrices:
    def test_disabled_by_default(self) -> None:
        fw = QuoteFirewall()
        accepted, _ = fw.filter([_q(yes=0.60, no=0.50)], now=NOW)  # sum=1.10
        assert len(accepted) == 1

    def test_enabled_rejects_inverted(self) -> None:
        fw = QuoteFirewall(FirewallConfig(reject_inverted_prices=True))
        accepted, rejected = fw.filter([_q(yes=0.60, no=0.50)], now=NOW)
        assert len(accepted) == 0
        assert any("inverted_prices" in r for r in rejected["kalshi/M1"])

    def test_enabled_passes_normal(self) -> None:
        fw = QuoteFirewall(FirewallConfig(reject_inverted_prices=True))
        accepted, _ = fw.filter([_q(yes=0.55, no=0.45)], now=NOW)  # sum=1.0
        assert len(accepted) == 1


# ---------------------------------------------------------------------------
# FirewallStats
# ---------------------------------------------------------------------------


class TestFirewallStats:
    def test_stats_tracking(self) -> None:
        fw = QuoteFirewall(FirewallConfig(min_price_floor=0.10))
        fw.filter([_q(yes=0.55, no=0.45), _q(market_id="M2", yes=0.05, no=0.95)], now=NOW)
        assert fw.stats.total_checked == 2
        assert fw.stats.total_rejected == 1
        assert fw.stats.rejections_by_reason.get("below_price_floor(yes=0.0500)") == 1

    def test_stats_reset(self) -> None:
        fw = QuoteFirewall(FirewallConfig(min_price_floor=0.10))
        fw.filter([_q(yes=0.05, no=0.95)], now=NOW)
        fw.reset_stats()
        assert fw.stats.total_checked == 0
        assert fw.stats.total_rejected == 0

    def test_stats_defaults(self) -> None:
        stats = FirewallStats()
        assert stats.total_checked == 0
        assert stats.total_rejected == 0
        assert stats.rejections_by_reason == {}


# ---------------------------------------------------------------------------
# Multiple quotes
# ---------------------------------------------------------------------------


class TestMultipleQuotes:
    def test_mixed_accept_reject(self) -> None:
        fw = QuoteFirewall(FirewallConfig(min_price_floor=0.05))
        quotes = [
            _q(market_id="M1", yes=0.55, no=0.45),  # passes
            _q(market_id="M2", yes=0.03, no=0.97),  # rejected
            _q(market_id="M3", yes=0.60, no=0.40),  # passes
        ]
        accepted, rejected = fw.filter(quotes, now=NOW)
        assert len(accepted) == 2
        assert len(rejected) == 1
        assert "kalshi/M2" in rejected

    def test_multiple_rejection_reasons(self) -> None:
        fw = QuoteFirewall(FirewallConfig(
            min_price_floor=0.10,
            max_complement_sum_deviation=0.05,
        ))
        quotes = [_q(yes=0.05, no=0.80)]  # Below floor AND complement deviation
        accepted, rejected = fw.filter(quotes, now=NOW)
        assert len(accepted) == 0
        reasons = rejected["kalshi/M1"]
        assert len(reasons) >= 2


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_all_checks_pass(self) -> None:
        fw = QuoteFirewall(FirewallConfig(
            max_complement_sum_deviation=0.10,
            min_price_floor=0.05,
            max_price_ceiling=0.95,
            min_book_depth_contracts=5.0,
            reject_locked_book=True,
        ))
        quotes = [_q(yes=0.55, no=0.45, yes_size=20.0, no_size=20.0, yes_bid=0.53)]
        accepted, rejected = fw.filter(quotes, now=NOW)
        assert len(accepted) == 1
        assert len(rejected) == 0

    def test_now_defaults_to_current(self) -> None:
        fw = QuoteFirewall()
        accepted, _ = fw.filter([_q(age_seconds=0.0)])
        assert len(accepted) == 1

    def test_config_property(self) -> None:
        config = FirewallConfig(min_price_floor=0.42)
        fw = QuoteFirewall(config)
        assert fw.config.min_price_floor == 0.42
