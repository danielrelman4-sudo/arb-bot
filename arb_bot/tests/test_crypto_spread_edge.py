"""Tests for spread-adjusted edge calculation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from arb_bot.crypto.edge_detector import CryptoEdge, EdgeDetector, compute_implied_probability
from arb_bot.crypto.market_scanner import CryptoMarket, CryptoMarketMeta, CryptoMarketQuote
from arb_bot.crypto.price_model import ProbabilityEstimate


def _make_quote(
    yes_buy: float = 0.40, no_buy: float = 0.60,
    yes_bid: float | None = 0.38, no_bid: float | None = 0.58,
    tte: float = 10.0,
) -> CryptoMarketQuote:
    now = datetime.now(timezone.utc)
    meta = CryptoMarketMeta(
        underlying="BTC", interval="daily",
        expiry=now + timedelta(minutes=tte),
        strike=70000.0, direction="above",
        series_ticker="KXBTCD",
    )
    market = CryptoMarket(ticker="KXBTCD-TEST-T70000", meta=meta)
    y_mid = (yes_bid + yes_buy) / 2.0 if yes_bid else yes_buy
    n_mid = (no_bid + no_buy) / 2.0 if no_bid else no_buy
    implied = compute_implied_probability(y_mid, n_mid)
    return CryptoMarketQuote(
        market=market,
        yes_buy_price=yes_buy, no_buy_price=no_buy,
        yes_buy_size=100, no_buy_size=100,
        yes_bid_price=yes_bid, no_bid_price=no_bid,
        time_to_expiry_minutes=tte, implied_probability=implied,
    )


class TestSpreadCostField:
    """CryptoEdge carries spread cost."""

    def test_edge_has_spread_cost_attribute(self):
        """CryptoEdge dataclass should have spread_cost field."""
        now = datetime.now(timezone.utc)
        meta = CryptoMarketMeta(
            underlying="BTC", interval="daily",
            expiry=now + timedelta(minutes=10),
            strike=70000.0, direction="above",
            series_ticker="KXBTCD",
        )
        market = CryptoMarket(ticker="TEST", meta=meta)
        prob = ProbabilityEstimate(0.5, 0.4, 0.6, 0.1, 1000)
        edge = CryptoEdge(
            market=market, model_prob=prob,
            market_implied_prob=0.4, edge=0.1,
            edge_cents=0.10, side="yes",
            recommended_price=0.40, model_uncertainty=0.05,
            time_to_expiry_minutes=10.0,
            yes_buy_price=0.40, no_buy_price=0.60,
            blended_probability=0.5, spread_cost=0.02,
        )
        assert edge.spread_cost == 0.02

    def test_spread_cost_defaults_to_zero(self):
        """spread_cost should default to 0.0."""
        now = datetime.now(timezone.utc)
        meta = CryptoMarketMeta(
            underlying="BTC", interval="daily",
            expiry=now + timedelta(minutes=10),
            strike=70000.0, direction="above",
            series_ticker="KXBTCD",
        )
        market = CryptoMarket(ticker="TEST", meta=meta)
        prob = ProbabilityEstimate(0.5, 0.4, 0.6, 0.1, 1000)
        edge = CryptoEdge(
            market=market, model_prob=prob,
            market_implied_prob=0.4, edge=0.1,
            edge_cents=0.10, side="yes",
            recommended_price=0.40, model_uncertainty=0.05,
            time_to_expiry_minutes=10.0,
            yes_buy_price=0.40, no_buy_price=0.60,
        )
        assert edge.spread_cost == 0.0


class TestSpreadComputation:
    """EdgeDetector computes spread cost for detected edges."""

    def test_spread_cost_from_yes_side(self):
        """When buying YES, spread = (yes_ask - yes_bid) / 2."""
        det = EdgeDetector(
            min_edge_pct=0.01, min_edge_cents=0.01,
            max_model_uncertainty=0.50, use_blending=False,
        )
        # YES ask = 0.40, YES bid = 0.36 -> spread = 0.04, half = 0.02
        quote = _make_quote(yes_buy=0.40, yes_bid=0.36, no_buy=0.60, no_bid=0.58)
        model_probs = {
            quote.market.ticker: ProbabilityEstimate(
                probability=0.55, ci_lower=0.50, ci_upper=0.60,
                uncertainty=0.05, num_paths=1000,
            ),
        }
        edges = det.detect_edges([quote], model_probs)
        assert len(edges) >= 1
        edge = edges[0]
        assert edge.side == "yes"
        assert edge.spread_cost == pytest.approx(0.02)

    def test_spread_cost_from_no_side(self):
        """When buying NO, spread = (no_ask - no_bid) / 2."""
        det = EdgeDetector(
            min_edge_pct=0.01, min_edge_cents=0.01,
            max_model_uncertainty=0.50, use_blending=False,
        )
        # Model says prob=0.30, market=0.40 -> edge on NO side
        # NO ask = 0.60, NO bid = 0.54 -> spread = 0.06, half = 0.03
        quote = _make_quote(yes_buy=0.40, yes_bid=0.38, no_buy=0.60, no_bid=0.54)
        model_probs = {
            quote.market.ticker: ProbabilityEstimate(
                probability=0.30, ci_lower=0.25, ci_upper=0.35,
                uncertainty=0.05, num_paths=1000,
            ),
        }
        edges = det.detect_edges([quote], model_probs)
        assert len(edges) >= 1
        edge = edges[0]
        assert edge.side == "no"
        assert edge.spread_cost == pytest.approx(0.03)

    def test_no_bid_falls_back_to_ask(self):
        """When no bid price, spread_cost should be 0."""
        det = EdgeDetector(
            min_edge_pct=0.01, min_edge_cents=0.01,
            max_model_uncertainty=0.50, use_blending=False,
        )
        quote = _make_quote(yes_buy=0.40, yes_bid=None, no_buy=0.60, no_bid=None)
        model_probs = {
            quote.market.ticker: ProbabilityEstimate(
                probability=0.55, ci_lower=0.50, ci_upper=0.60,
                uncertainty=0.05, num_paths=1000,
            ),
        }
        edges = det.detect_edges([quote], model_probs)
        if edges:
            assert edges[0].spread_cost == 0.0

    def test_wide_spread_larger_cost(self):
        """Wider spread should produce larger spread_cost."""
        det = EdgeDetector(
            min_edge_pct=0.01, min_edge_cents=0.01,
            max_model_uncertainty=0.50, use_blending=False,
        )
        # Tight: ask=0.40, bid=0.39 -> half_spread=0.005
        tight = _make_quote(yes_buy=0.40, yes_bid=0.39, no_buy=0.60, no_bid=0.59)
        # Wide: ask=0.40, bid=0.30 -> half_spread=0.05
        wide = _make_quote(yes_buy=0.40, yes_bid=0.30, no_buy=0.60, no_bid=0.50)

        model = ProbabilityEstimate(
            probability=0.55, ci_lower=0.50, ci_upper=0.60,
            uncertainty=0.05, num_paths=1000,
        )

        edges_tight = det.detect_edges([tight], {tight.market.ticker: model})
        edges_wide = det.detect_edges([wide], {wide.market.ticker: model})

        if edges_tight and edges_wide:
            assert edges_wide[0].spread_cost > edges_tight[0].spread_cost
