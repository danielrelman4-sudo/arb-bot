"""Tests for Phase 2A: Pre-trade quote and mapping risk gates."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from arb_bot.models import (
    BinaryQuote,
    ExecutionStyle,
    OpportunityKind,
    Side,
    TradeLegPlan,
    TradePlan,
)
from arb_bot.framework.quote_gate import QuoteGateChecker, QuoteGateConfig, QuoteGateResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NOW = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def _quote(
    venue: str = "kalshi",
    market_id: str = "M1",
    yes_price: float = 0.55,
    no_price: float = 0.50,
    yes_size: float = 20.0,
    no_size: float = 20.0,
    yes_bid: float | None = 0.53,
    no_bid: float | None = 0.48,
    age_seconds: float = 5.0,
    fee: float = 0.0,
) -> BinaryQuote:
    return BinaryQuote(
        venue=venue,
        market_id=market_id,
        yes_buy_price=yes_price,
        no_buy_price=no_price,
        yes_buy_size=yes_size,
        no_buy_size=no_size,
        yes_bid_price=yes_bid,
        no_bid_price=no_bid,
        fee_per_contract=fee,
        observed_at=NOW - timedelta(seconds=age_seconds),
    )


def _cross_plan(
    match_score: float | None = None,
    contracts: int = 10,
) -> TradePlan:
    meta = {}
    if match_score is not None:
        meta["match_score"] = match_score
    return TradePlan(
        kind=OpportunityKind.CROSS_VENUE,
        execution_style=ExecutionStyle.TAKER,
        legs=(
            TradeLegPlan(venue="kalshi", market_id="K1", side=Side.YES, contracts=contracts, limit_price=0.45),
            TradeLegPlan(venue="polymarket", market_id="P1", side=Side.NO, contracts=contracts, limit_price=0.50),
        ),
        contracts=contracts,
        capital_required=9.5,
        capital_required_by_venue={"kalshi": 4.5, "polymarket": 5.0},
        expected_profit=0.50,
        edge_per_contract=0.05,
        metadata=meta,
    )


def _intra_plan() -> TradePlan:
    return TradePlan(
        kind=OpportunityKind.INTRA_VENUE,
        execution_style=ExecutionStyle.TAKER,
        legs=(
            TradeLegPlan(venue="kalshi", market_id="M1", side=Side.YES, contracts=10, limit_price=0.40),
            TradeLegPlan(venue="kalshi", market_id="M1", side=Side.NO, contracts=10, limit_price=0.40),
        ),
        contracts=10,
        capital_required=8.0,
        capital_required_by_venue={"kalshi": 8.0},
        expected_profit=1.0,
        edge_per_contract=0.10,
    )


# ---------------------------------------------------------------------------
# QuoteGateConfig
# ---------------------------------------------------------------------------


class TestQuoteGateConfig:
    def test_defaults(self) -> None:
        config = QuoteGateConfig()
        assert config.max_quote_age_seconds == 0.0
        assert config.min_mapping_confidence == 0.0
        assert config.max_bid_ask_spread == 0.0
        assert config.require_both_sides_quoted is False
        assert config.max_price_vs_complement_deviation == 0.0
        assert config.min_size_contracts == 0.0


# ---------------------------------------------------------------------------
# QuoteGateResult
# ---------------------------------------------------------------------------


class TestQuoteGateResult:
    def test_passed(self) -> None:
        r = QuoteGateResult(passed=True, reason="ok")
        assert r.passed is True

    def test_failed(self) -> None:
        r = QuoteGateResult(passed=False, reason="stale", checks_run=3, checks_failed=1)
        assert r.passed is False
        assert r.checks_failed == 1


# ---------------------------------------------------------------------------
# Gate 1: Quote availability
# ---------------------------------------------------------------------------


class TestQuoteAvailability:
    def test_passes_with_all_quotes(self) -> None:
        checker = QuoteGateChecker()
        plan = _intra_plan()
        quotes = {("kalshi", "M1"): _quote()}
        result = checker.check(plan, quotes, now=NOW)
        assert result.passed is True

    def test_fails_with_missing_quote(self) -> None:
        checker = QuoteGateChecker()
        plan = _cross_plan()
        quotes = {("kalshi", "K1"): _quote(market_id="K1")}
        # Missing polymarket/P1
        result = checker.check(plan, quotes, now=NOW)
        assert result.passed is False
        assert "missing quotes" in result.reason
        assert "polymarket/P1" in result.reason

    def test_fails_with_empty_quotes(self) -> None:
        checker = QuoteGateChecker()
        plan = _cross_plan()
        result = checker.check(plan, {}, now=NOW)
        assert result.passed is False


# ---------------------------------------------------------------------------
# Gate 2: Quote freshness
# ---------------------------------------------------------------------------


class TestQuoteFreshness:
    def test_disabled_by_default(self) -> None:
        checker = QuoteGateChecker()
        plan = _intra_plan()
        quotes = {("kalshi", "M1"): _quote(age_seconds=9999.0)}
        result = checker.check(plan, quotes, now=NOW)
        assert result.passed is True  # Disabled (max_quote_age_seconds=0)

    def test_passes_fresh_quote(self) -> None:
        config = QuoteGateConfig(max_quote_age_seconds=30.0)
        checker = QuoteGateChecker(config)
        plan = _intra_plan()
        quotes = {("kalshi", "M1"): _quote(age_seconds=10.0)}
        result = checker.check(plan, quotes, now=NOW)
        assert result.passed is True

    def test_fails_stale_quote(self) -> None:
        config = QuoteGateConfig(max_quote_age_seconds=30.0)
        checker = QuoteGateChecker(config)
        plan = _intra_plan()
        quotes = {("kalshi", "M1"): _quote(age_seconds=60.0)}
        result = checker.check(plan, quotes, now=NOW)
        assert result.passed is False
        assert "stale" in result.reason

    def test_one_stale_one_fresh(self) -> None:
        config = QuoteGateConfig(max_quote_age_seconds=30.0)
        checker = QuoteGateChecker(config)
        plan = _cross_plan()
        quotes = {
            ("kalshi", "K1"): _quote(venue="kalshi", market_id="K1", age_seconds=5.0),
            ("polymarket", "P1"): _quote(venue="polymarket", market_id="P1", age_seconds=60.0),
        }
        result = checker.check(plan, quotes, now=NOW)
        assert result.passed is False
        assert "polymarket/P1" in result.reason


# ---------------------------------------------------------------------------
# Gate 3: Mapping confidence
# ---------------------------------------------------------------------------


class TestMappingConfidence:
    def test_disabled_by_default(self) -> None:
        checker = QuoteGateChecker()
        plan = _cross_plan(match_score=0.1)
        quotes = {
            ("kalshi", "K1"): _quote(venue="kalshi", market_id="K1"),
            ("polymarket", "P1"): _quote(venue="polymarket", market_id="P1"),
        }
        result = checker.check(plan, quotes, now=NOW)
        assert result.passed is True

    def test_passes_high_confidence(self) -> None:
        config = QuoteGateConfig(min_mapping_confidence=0.8)
        checker = QuoteGateChecker(config)
        plan = _cross_plan(match_score=0.95)
        quotes = {
            ("kalshi", "K1"): _quote(venue="kalshi", market_id="K1"),
            ("polymarket", "P1"): _quote(venue="polymarket", market_id="P1"),
        }
        result = checker.check(plan, quotes, now=NOW)
        assert result.passed is True

    def test_fails_low_confidence(self) -> None:
        config = QuoteGateConfig(min_mapping_confidence=0.8)
        checker = QuoteGateChecker(config)
        plan = _cross_plan(match_score=0.65)
        quotes = {
            ("kalshi", "K1"): _quote(venue="kalshi", market_id="K1"),
            ("polymarket", "P1"): _quote(venue="polymarket", market_id="P1"),
        }
        result = checker.check(plan, quotes, now=NOW)
        assert result.passed is False
        assert "mapping confidence" in result.reason

    def test_no_match_score_skipped(self) -> None:
        """Plans without match_score in metadata skip this gate."""
        config = QuoteGateConfig(min_mapping_confidence=0.8)
        checker = QuoteGateChecker(config)
        plan = _cross_plan(match_score=None)
        quotes = {
            ("kalshi", "K1"): _quote(venue="kalshi", market_id="K1"),
            ("polymarket", "P1"): _quote(venue="polymarket", market_id="P1"),
        }
        result = checker.check(plan, quotes, now=NOW)
        assert result.passed is True


# ---------------------------------------------------------------------------
# Gate 4: Bid-ask spread
# ---------------------------------------------------------------------------


class TestBidAskSpread:
    def test_disabled_by_default(self) -> None:
        checker = QuoteGateChecker()
        plan = _intra_plan()
        quotes = {("kalshi", "M1"): _quote(yes_price=0.90, yes_bid=0.10)}
        result = checker.check(plan, quotes, now=NOW)
        assert result.passed is True

    def test_passes_tight_spread(self) -> None:
        config = QuoteGateConfig(max_bid_ask_spread=0.05)
        checker = QuoteGateChecker(config)
        plan = _intra_plan()
        quotes = {("kalshi", "M1"): _quote(yes_price=0.55, yes_bid=0.53, no_price=0.50, no_bid=0.48)}
        result = checker.check(plan, quotes, now=NOW)
        assert result.passed is True

    def test_fails_wide_spread(self) -> None:
        config = QuoteGateConfig(max_bid_ask_spread=0.05)
        checker = QuoteGateChecker(config)
        plan = _intra_plan()
        quotes = {("kalshi", "M1"): _quote(yes_price=0.60, yes_bid=0.40)}  # spread = 0.20
        result = checker.check(plan, quotes, now=NOW)
        assert result.passed is False
        assert "wide spread" in result.reason

    def test_no_bid_skipped(self) -> None:
        """If no bid price, spread check is skipped for that side."""
        config = QuoteGateConfig(max_bid_ask_spread=0.05)
        checker = QuoteGateChecker(config)
        plan = _intra_plan()
        quotes = {("kalshi", "M1"): _quote(yes_bid=None, no_bid=None)}
        result = checker.check(plan, quotes, now=NOW)
        assert result.passed is True


# ---------------------------------------------------------------------------
# Gate 5: Both sides quoted
# ---------------------------------------------------------------------------


class TestBothSidesQuoted:
    def test_disabled_by_default(self) -> None:
        checker = QuoteGateChecker()
        plan = _intra_plan()
        quotes = {("kalshi", "M1"): _quote(yes_bid=None, no_bid=None)}
        result = checker.check(plan, quotes, now=NOW)
        assert result.passed is True

    def test_passes_with_bids(self) -> None:
        config = QuoteGateConfig(require_both_sides_quoted=True)
        checker = QuoteGateChecker(config)
        plan = _intra_plan()
        quotes = {("kalshi", "M1"): _quote(yes_bid=0.53, no_bid=0.48)}
        result = checker.check(plan, quotes, now=NOW)
        assert result.passed is True

    def test_fails_missing_yes_bid(self) -> None:
        config = QuoteGateConfig(require_both_sides_quoted=True)
        checker = QuoteGateChecker(config)
        # Plan has a YES leg
        plan = TradePlan(
            kind=OpportunityKind.INTRA_VENUE,
            execution_style=ExecutionStyle.TAKER,
            legs=(
                TradeLegPlan(venue="kalshi", market_id="M1", side=Side.YES, contracts=10, limit_price=0.40),
            ),
            contracts=10,
            capital_required=4.0,
            capital_required_by_venue={"kalshi": 4.0},
            expected_profit=0.5,
            edge_per_contract=0.05,
        )
        quotes = {("kalshi", "M1"): _quote(yes_bid=None)}
        result = checker.check(plan, quotes, now=NOW)
        assert result.passed is False
        assert "no bid" in result.reason


# ---------------------------------------------------------------------------
# Gate 6: Price vs complement deviation
# ---------------------------------------------------------------------------


class TestPriceComplementDeviation:
    def test_disabled_by_default(self) -> None:
        checker = QuoteGateChecker()
        plan = _intra_plan()
        quotes = {("kalshi", "M1"): _quote(yes_price=0.90, no_price=0.90)}  # sum = 1.80
        result = checker.check(plan, quotes, now=NOW)
        assert result.passed is True

    def test_passes_normal_complement(self) -> None:
        config = QuoteGateConfig(max_price_vs_complement_deviation=0.05)
        checker = QuoteGateChecker(config)
        plan = _intra_plan()
        quotes = {("kalshi", "M1"): _quote(yes_price=0.55, no_price=0.48)}  # sum = 1.03
        result = checker.check(plan, quotes, now=NOW)
        assert result.passed is True

    def test_fails_extreme_deviation(self) -> None:
        config = QuoteGateConfig(max_price_vs_complement_deviation=0.05)
        checker = QuoteGateChecker(config)
        plan = _intra_plan()
        quotes = {("kalshi", "M1"): _quote(yes_price=0.80, no_price=0.80)}  # sum = 1.60, dev = 0.60
        result = checker.check(plan, quotes, now=NOW)
        assert result.passed is False
        assert "complement deviation" in result.reason


# ---------------------------------------------------------------------------
# Gate 7: Minimum size
# ---------------------------------------------------------------------------


class TestMinSize:
    def test_disabled_by_default(self) -> None:
        checker = QuoteGateChecker()
        plan = _intra_plan()
        quotes = {("kalshi", "M1"): _quote(yes_size=0.5, no_size=0.5)}
        result = checker.check(plan, quotes, now=NOW)
        assert result.passed is True

    def test_passes_sufficient_size(self) -> None:
        config = QuoteGateConfig(min_size_contracts=5.0)
        checker = QuoteGateChecker(config)
        plan = _intra_plan()
        quotes = {("kalshi", "M1"): _quote(yes_size=20.0, no_size=20.0)}
        result = checker.check(plan, quotes, now=NOW)
        assert result.passed is True

    def test_fails_thin_book(self) -> None:
        config = QuoteGateConfig(min_size_contracts=10.0)
        checker = QuoteGateChecker(config)
        plan = _intra_plan()
        quotes = {("kalshi", "M1"): _quote(yes_size=5.0, no_size=15.0)}
        result = checker.check(plan, quotes, now=NOW)
        assert result.passed is False
        assert "thin book" in result.reason


# ---------------------------------------------------------------------------
# Multiple gates
# ---------------------------------------------------------------------------


class TestMultipleGates:
    def test_multiple_failures(self) -> None:
        config = QuoteGateConfig(
            max_quote_age_seconds=10.0,
            min_size_contracts=15.0,
        )
        checker = QuoteGateChecker(config)
        plan = _intra_plan()
        quotes = {("kalshi", "M1"): _quote(age_seconds=30.0, yes_size=5.0, no_size=5.0)}
        result = checker.check(plan, quotes, now=NOW)
        assert result.passed is False
        assert result.checks_failed >= 2
        assert "stale" in result.reason
        assert "thin book" in result.reason

    def test_all_gates_enabled_pass(self) -> None:
        config = QuoteGateConfig(
            max_quote_age_seconds=30.0,
            min_mapping_confidence=0.5,
            max_bid_ask_spread=0.10,
            require_both_sides_quoted=True,
            max_price_vs_complement_deviation=0.10,
            min_size_contracts=5.0,
        )
        checker = QuoteGateChecker(config)
        plan = _cross_plan(match_score=0.9)
        quotes = {
            ("kalshi", "K1"): _quote(venue="kalshi", market_id="K1", age_seconds=5.0),
            ("polymarket", "P1"): _quote(venue="polymarket", market_id="P1", age_seconds=5.0),
        }
        result = checker.check(plan, quotes, now=NOW)
        assert result.passed is True

    def test_defaults_pass_everything(self) -> None:
        """With default config (all gates disabled), everything passes."""
        checker = QuoteGateChecker()
        plan = _intra_plan()
        quotes = {("kalshi", "M1"): _quote()}
        result = checker.check(plan, quotes, now=NOW)
        assert result.passed is True


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_config_property(self) -> None:
        config = QuoteGateConfig(max_quote_age_seconds=42.0)
        checker = QuoteGateChecker(config)
        assert checker.config.max_quote_age_seconds == 42.0

    def test_now_defaults_to_current_time(self) -> None:
        checker = QuoteGateChecker()
        plan = _intra_plan()
        quotes = {("kalshi", "M1"): _quote(age_seconds=0.0)}
        # This should not crash — now auto-computed
        result = checker.check(plan, quotes)
        assert result.passed is True
