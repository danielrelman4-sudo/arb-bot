"""Tests for crypto edge detector."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from arb_bot.crypto.edge_detector import (
    CryptoEdge,
    EdgeDetector,
    compute_implied_probability,
)
from arb_bot.crypto.market_scanner import CryptoMarket, CryptoMarketMeta, CryptoMarketQuote
from arb_bot.crypto.price_model import ProbabilityEstimate


def _make_meta(direction: str = "above", strike: float = 97000.0) -> CryptoMarketMeta:
    return CryptoMarketMeta(
        underlying="BTC",
        interval="daily",
        expiry=datetime(2026, 2, 14, tzinfo=timezone.utc),
        strike=strike,
        direction=direction,
        series_ticker="KXBTCD",
    )


def _make_quote(
    ticker: str = "KXBTCD-26FEB14-T97000",
    yes_price: float = 0.55,
    no_price: float = 0.50,
    tte_minutes: float = 10.0,
    direction: str = "above",
) -> CryptoMarketQuote:
    meta = _make_meta(direction=direction)
    market = CryptoMarket(ticker=ticker, meta=meta)
    implied = compute_implied_probability(yes_price, no_price)
    return CryptoMarketQuote(
        market=market,
        yes_buy_price=yes_price,
        no_buy_price=no_price,
        yes_buy_size=100.0,
        no_buy_size=100.0,
        yes_bid_price=yes_price - 0.02,
        no_bid_price=no_price - 0.02,
        time_to_expiry_minutes=tte_minutes,
        implied_probability=implied,
    )


class TestComputeImpliedProbability:
    def test_fair_market(self) -> None:
        # YES = 0.50, NO = 0.50 → implied = 0.50
        assert abs(compute_implied_probability(0.50, 0.50) - 0.50) < 1e-10

    def test_yes_favored(self) -> None:
        # YES = 0.70, NO = 0.35 → implied = 0.5*(0.70 + 0.65) = 0.675
        assert abs(compute_implied_probability(0.70, 0.35) - 0.675) < 1e-10

    def test_overround(self) -> None:
        # YES = 0.55, NO = 0.50 → implied = 0.5*(0.55 + 0.50) = 0.525
        assert abs(compute_implied_probability(0.55, 0.50) - 0.525) < 1e-10


class TestEdgeDetector:
    def test_positive_edge_buy_yes(self) -> None:
        """Model says prob higher than market → buy YES."""
        detector = EdgeDetector(min_edge_pct=0.01, min_edge_cents=0.01, min_model_market_divergence=0.0)
        quote = _make_quote(yes_price=0.50, no_price=0.50)  # implied ~0.50
        model = ProbabilityEstimate(0.60, 0.55, 0.65, 0.05, 1000)
        edges = detector.detect_edges(
            [quote],
            {quote.market.ticker: model},
        )
        assert len(edges) == 1
        assert edges[0].side == "yes"
        assert edges[0].edge > 0

    def test_negative_edge_buy_no(self) -> None:
        """Model says prob lower than market → buy NO."""
        detector = EdgeDetector(min_edge_pct=0.01, min_edge_cents=0.01, min_model_market_divergence=0.0)
        quote = _make_quote(yes_price=0.60, no_price=0.45)  # implied ~0.575
        model = ProbabilityEstimate(0.45, 0.40, 0.50, 0.05, 1000)
        edges = detector.detect_edges(
            [quote],
            {quote.market.ticker: model},
        )
        assert len(edges) == 1
        assert edges[0].side == "no"
        assert edges[0].edge < 0

    def test_edge_below_threshold_filtered(self) -> None:
        detector = EdgeDetector(min_edge_pct=0.10, min_model_market_divergence=0.0)
        quote = _make_quote(yes_price=0.50, no_price=0.50)  # implied ~0.50
        model = ProbabilityEstimate(0.55, 0.50, 0.60, 0.05, 1000)  # 5% edge
        edges = detector.detect_edges([quote], {quote.market.ticker: model})
        assert len(edges) == 0

    def test_uncertainty_too_high_filtered(self) -> None:
        detector = EdgeDetector(
            min_edge_pct=0.01,
            min_edge_cents=0.01,
            max_model_uncertainty=0.05,
            min_model_market_divergence=0.0,
        )
        quote = _make_quote(yes_price=0.50, no_price=0.50)
        model = ProbabilityEstimate(0.60, 0.40, 0.80, 0.20, 100)  # high uncertainty
        edges = detector.detect_edges([quote], {quote.market.ticker: model})
        assert len(edges) == 0

    def test_no_model_for_market_skipped(self) -> None:
        detector = EdgeDetector(min_edge_pct=0.01, min_edge_cents=0.01, min_model_market_divergence=0.0)
        quote = _make_quote()
        edges = detector.detect_edges([quote], {})
        assert len(edges) == 0

    def test_sorted_by_absolute_edge(self) -> None:
        detector = EdgeDetector(min_edge_pct=0.01, min_edge_cents=0.01, min_model_market_divergence=0.0)
        q1 = _make_quote(ticker="KXBTCD-26FEB14-T97000", yes_price=0.50, no_price=0.50)
        q2 = _make_quote(ticker="KXBTCD-26FEB14-T98000", yes_price=0.50, no_price=0.50)

        m1 = ProbabilityEstimate(0.55, 0.50, 0.60, 0.05, 1000)  # 5% edge
        m2 = ProbabilityEstimate(0.65, 0.60, 0.70, 0.05, 1000)  # 15% edge

        edges = detector.detect_edges(
            [q1, q2],
            {q1.market.ticker: m1, q2.market.ticker: m2},
        )
        assert len(edges) == 2
        assert abs(edges[0].edge) >= abs(edges[1].edge)

    def test_zero_edge_skipped(self) -> None:
        detector = EdgeDetector(min_edge_pct=0.01, min_edge_cents=0.01, min_model_market_divergence=0.0)
        quote = _make_quote(yes_price=0.50, no_price=0.50)  # implied 0.50
        model = ProbabilityEstimate(0.50, 0.45, 0.55, 0.05, 1000)  # exact match
        edges = detector.detect_edges([quote], {quote.market.ticker: model})
        assert len(edges) == 0

    def test_edge_fields_populated(self) -> None:
        detector = EdgeDetector(min_edge_pct=0.01, min_edge_cents=0.01, min_model_market_divergence=0.0)
        quote = _make_quote(
            ticker="KXBTCD-26FEB14-T97000",
            yes_price=0.50,
            no_price=0.50,
            tte_minutes=8.5,
        )
        model = ProbabilityEstimate(0.60, 0.55, 0.65, 0.05, 1000)
        edges = detector.detect_edges([quote], {quote.market.ticker: model})
        assert len(edges) == 1
        e = edges[0]
        assert e.market.ticker == "KXBTCD-26FEB14-T97000"
        assert e.model_uncertainty == 0.05
        assert abs(e.time_to_expiry_minutes - 8.5) < 1e-10
        assert e.yes_buy_price == 0.50
        assert e.no_buy_price == 0.50

    def test_edge_cents_filter(self) -> None:
        detector = EdgeDetector(min_edge_pct=0.01, min_edge_cents=0.05, min_model_market_divergence=0.0)
        quote = _make_quote(yes_price=0.50, no_price=0.50)
        model = ProbabilityEstimate(0.53, 0.48, 0.58, 0.05, 1000)  # 3% = $0.03
        edges = detector.detect_edges([quote], {quote.market.ticker: model})
        assert len(edges) == 0  # 0.03 < 0.05 min_edge_cents

    def test_multiple_markets(self) -> None:
        detector = EdgeDetector(min_edge_pct=0.01, min_edge_cents=0.01, min_model_market_divergence=0.0)
        quotes = [
            _make_quote(ticker="KXBTCD-26FEB14-T97000", yes_price=0.50, no_price=0.50),
            _make_quote(ticker="KXBTCD-26FEB14-T98000", yes_price=0.40, no_price=0.60),
            _make_quote(ticker="KXBTCD-26FEB14-T99000", yes_price=0.30, no_price=0.70),
        ]
        model_probs = {
            quotes[0].market.ticker: ProbabilityEstimate(0.60, 0.55, 0.65, 0.05, 1000),
            quotes[1].market.ticker: ProbabilityEstimate(0.50, 0.45, 0.55, 0.05, 1000),
            # quotes[2] has no model — should be skipped
        }
        edges = detector.detect_edges(quotes, model_probs)
        assert len(edges) == 2


class TestDailyMinEdge:
    def test_daily_market_4pct_edge_filtered_with_6pct_threshold(self) -> None:
        """Daily above/below markets with 4% edge should be rejected at 6% min."""
        detector = EdgeDetector(
            min_edge_pct=0.01,
            min_edge_pct_daily=0.06,
            min_edge_cents=0.01,
            use_blending=False,
            min_model_market_divergence=0.0,
        )
        # above direction + tte > 30 min => daily threshold applies
        quote = _make_quote(
            ticker="KXBTCD-26FEB14-T97000",
            yes_price=0.50,
            no_price=0.50,
            tte_minutes=120.0,
            direction="above",
        )
        # Model says 0.54 => ~4% edge vs 0.50 cost => below 6% daily threshold
        model = ProbabilityEstimate(0.54, 0.49, 0.59, 0.05, 1000)
        edges = detector.detect_edges([quote], {quote.market.ticker: model})
        assert len(edges) == 0

    def test_daily_market_7pct_edge_passes(self) -> None:
        """Daily above/below markets with 7% edge should pass at 6% min."""
        detector = EdgeDetector(
            min_edge_pct=0.01,
            min_edge_pct_daily=0.06,
            min_edge_cents=0.01,
            use_blending=False,
            min_model_market_divergence=0.0,
        )
        quote = _make_quote(
            ticker="KXBTCD-26FEB14-T97000",
            yes_price=0.50,
            no_price=0.50,
            tte_minutes=120.0,
            direction="above",
        )
        # Model says 0.57 => 7% edge vs 0.50 cost => above 6% daily threshold
        model = ProbabilityEstimate(0.57, 0.52, 0.62, 0.05, 1000)
        edges = detector.detect_edges([quote], {quote.market.ticker: model})
        assert len(edges) == 1
        assert edges[0].side == "yes"

    def test_short_term_above_uses_base_threshold(self) -> None:
        """above/below with tte <= 30 min uses base threshold, not daily."""
        detector = EdgeDetector(
            min_edge_pct=0.01,
            min_edge_pct_daily=0.06,
            min_edge_cents=0.01,
            use_blending=False,
            min_model_market_divergence=0.0,
        )
        # tte=10 minutes => not daily, uses base min_edge_pct=0.01
        quote = _make_quote(
            ticker="KXBTCD-26FEB14-T97000",
            yes_price=0.50,
            no_price=0.50,
            tte_minutes=10.0,
            direction="above",
        )
        # 4% edge passes base threshold of 1%
        model = ProbabilityEstimate(0.54, 0.49, 0.59, 0.05, 1000)
        edges = detector.detect_edges([quote], {quote.market.ticker: model})
        assert len(edges) == 1
