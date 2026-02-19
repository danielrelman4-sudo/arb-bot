"""Tests for dependency_discovery module (Phase 6C)."""

from __future__ import annotations

import pytest

from arb_bot.framework.dependency_discovery import (
    CandidateLink,
    DependencyDiscovery,
    DependencyDiscoveryConfig,
    DiscoveryReport,
    MarketInfo,
    _jaccard_similarity,
    _pearson_correlation,
    _tokenize,
)


def _disco(**kw: object) -> DependencyDiscovery:
    return DependencyDiscovery(DependencyDiscoveryConfig(**kw))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self) -> None:
        cfg = DependencyDiscoveryConfig()
        assert cfg.text_similarity_threshold == 0.40
        assert cfg.comovement_threshold == 0.70
        assert cfg.min_price_observations == 5
        assert cfg.max_candidates == 50
        assert cfg.combined_score_threshold == 0.30
        assert cfg.text_weight == 0.5
        assert cfg.comovement_weight == 0.5

    def test_frozen(self) -> None:
        cfg = DependencyDiscoveryConfig()
        with pytest.raises(AttributeError):
            cfg.text_weight = 1.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_basic(self) -> None:
        tokens = _tokenize("Will BTC exceed 50K by March?")
        assert "will" in tokens
        assert "btc" in tokens
        assert "50k" in tokens
        assert "march" in tokens

    def test_removes_short(self) -> None:
        tokens = _tokenize("a I to be or")
        assert "a" not in tokens
        assert "i" not in tokens  # len < 2
        assert "to" in tokens
        assert "be" in tokens
        assert "or" in tokens

    def test_normalizes_punctuation(self) -> None:
        tokens = _tokenize("hello, world! foo-bar")
        assert "hello" in tokens
        assert "world" in tokens

    def test_empty(self) -> None:
        assert _tokenize("") == set()


# ---------------------------------------------------------------------------
# Jaccard similarity
# ---------------------------------------------------------------------------


class TestJaccard:
    def test_identical(self) -> None:
        s = {"a", "b", "c"}
        assert _jaccard_similarity(s, s) == 1.0

    def test_disjoint(self) -> None:
        assert _jaccard_similarity({"a", "b"}, {"c", "d"}) == 0.0

    def test_partial(self) -> None:
        sim = _jaccard_similarity({"a", "b", "c"}, {"b", "c", "d"})
        assert abs(sim - 0.5) < 0.01  # 2/4 = 0.5.

    def test_empty(self) -> None:
        assert _jaccard_similarity(set(), set()) == 0.0


# ---------------------------------------------------------------------------
# Pearson correlation
# ---------------------------------------------------------------------------


class TestPearson:
    def test_perfect_positive(self) -> None:
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [2.0, 4.0, 6.0, 8.0, 10.0]
        assert abs(_pearson_correlation(xs, ys) - 1.0) < 0.001

    def test_perfect_negative(self) -> None:
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [10.0, 8.0, 6.0, 4.0, 2.0]
        assert abs(_pearson_correlation(xs, ys) - (-1.0)) < 0.001

    def test_no_correlation(self) -> None:
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [3.0, 1.0, 4.0, 1.0, 5.0]
        corr = _pearson_correlation(xs, ys)
        assert abs(corr) < 0.5  # Low correlation.

    def test_constant_series(self) -> None:
        xs = [5.0, 5.0, 5.0]
        ys = [1.0, 2.0, 3.0]
        assert _pearson_correlation(xs, ys) == 0.0  # Zero variance in xs.

    def test_both_constant_series(self) -> None:
        """Both series constant → both denominators zero → 0/0 → must return 0.0.

        This is the edge case the NaN/Inf guard protects against. Without
        the guard, denom_x * denom_y = 0 * 0 = 0, causing ZeroDivisionError
        or NaN depending on the numerator.
        """
        xs = [3.0, 3.0, 3.0]
        ys = [7.0, 7.0, 7.0]
        assert _pearson_correlation(xs, ys) == 0.0

    def test_too_few_points(self) -> None:
        assert _pearson_correlation([1.0], [2.0]) == 0.0


# ---------------------------------------------------------------------------
# Add market
# ---------------------------------------------------------------------------


class TestAddMarket:
    def test_basic(self) -> None:
        d = _disco()
        d.add_market("m1", "Will BTC exceed 50K?", venue="kalshi")
        assert d.market_count() == 1
        info = d.get_market("m1")
        assert info is not None
        assert info.venue == "kalshi"
        assert "btc" in info.tokens

    def test_with_metadata(self) -> None:
        d = _disco()
        d.add_market("m1", "test", metadata={"category": "crypto"})
        info = d.get_market("m1")
        assert info is not None
        assert info.metadata["category"] == "crypto"


# ---------------------------------------------------------------------------
# Record price
# ---------------------------------------------------------------------------


class TestRecordPrice:
    def test_basic(self) -> None:
        d = _disco()
        d.add_market("m1", "test")
        ok = d.record_price("m1", 0.55, ts=100.0)
        assert ok is True
        info = d.get_market("m1")
        assert info is not None
        assert info.prices == [0.55]

    def test_nonexistent(self) -> None:
        d = _disco()
        assert d.record_price("missing", 0.5, ts=1.0) is False

    def test_history_limit(self) -> None:
        d = _disco(price_history_limit=5)
        d.add_market("m1", "test")
        for i in range(10):
            d.record_price("m1", float(i) / 10, ts=float(i))
        info = d.get_market("m1")
        assert info is not None
        assert len(info.prices) == 5


# ---------------------------------------------------------------------------
# Text similarity
# ---------------------------------------------------------------------------


class TestTextSimilarity:
    def test_similar_markets(self) -> None:
        d = _disco()
        d.add_market("m1", "Will BTC exceed 50K by March?", venue="kalshi")
        d.add_market("m2", "Bitcoin exceed 50K end of March", venue="poly")
        sim = d.text_similarity("m1", "m2")
        assert sim > 0.2  # Shared tokens: exceed, 50k, march.

    def test_unrelated_markets(self) -> None:
        d = _disco()
        d.add_market("m1", "Will it rain in NYC tomorrow?")
        d.add_market("m2", "Bitcoin price prediction crypto")
        sim = d.text_similarity("m1", "m2")
        assert sim < 0.1  # Very few shared tokens.

    def test_nonexistent(self) -> None:
        d = _disco()
        assert d.text_similarity("a", "b") == 0.0


# ---------------------------------------------------------------------------
# Co-movement
# ---------------------------------------------------------------------------


class TestComovement:
    def test_correlated(self) -> None:
        d = _disco(min_price_observations=3)
        d.add_market("m1", "BTC 50K")
        d.add_market("m2", "BTC 50K poly")
        for i in range(5):
            d.record_price("m1", 0.50 + i * 0.01, ts=float(i))
            d.record_price("m2", 0.48 + i * 0.01, ts=float(i))
        comove = d.comovement("m1", "m2")
        assert comove > 0.9  # Highly correlated.

    def test_insufficient_data(self) -> None:
        d = _disco(min_price_observations=5)
        d.add_market("m1", "BTC")
        d.add_market("m2", "BTC poly")
        d.record_price("m1", 0.5, ts=1.0)
        d.record_price("m2", 0.5, ts=1.0)
        assert d.comovement("m1", "m2") == 0.0


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


class TestDiscover:
    def test_finds_text_candidate(self) -> None:
        d = _disco(combined_score_threshold=0.15, text_weight=1.0, comovement_weight=0.0)
        d.add_market("m1", "Will BTC exceed 50K by March?", venue="kalshi")
        d.add_market("m2", "Will BTC go above 50K in March?", venue="poly")
        report = d.discover()
        assert report.candidates_found >= 1
        assert report.candidates[0].market_a == "m1"
        assert report.candidates[0].market_b == "m2"
        assert report.candidates[0].cross_venue is True

    def test_finds_comovement_candidate(self) -> None:
        d = _disco(
            combined_score_threshold=0.15,
            text_weight=0.0,
            comovement_weight=1.0,
            min_price_observations=3,
        )
        d.add_market("m1", "alpha", venue="kalshi")
        d.add_market("m2", "beta", venue="poly")
        for i in range(5):
            d.record_price("m1", 0.50 + i * 0.02, ts=float(i))
            d.record_price("m2", 0.48 + i * 0.02, ts=float(i))
        report = d.discover()
        assert report.candidates_found >= 1
        assert report.candidates[0].comovement > 0.9

    def test_no_candidates(self) -> None:
        d = _disco(combined_score_threshold=0.99)
        d.add_market("m1", "rain NYC", venue="kalshi")
        d.add_market("m2", "bitcoin crypto", venue="poly")
        report = d.discover()
        assert report.candidates_found == 0

    def test_max_candidates(self) -> None:
        d = _disco(combined_score_threshold=0.01, max_candidates=2, text_weight=1.0, comovement_weight=0.0)
        # Add markets with shared tokens.
        for i in range(5):
            d.add_market(f"m{i}", f"common shared terms market {i}")
        report = d.discover()
        assert report.candidates_found <= 2

    def test_same_venue_not_cross(self) -> None:
        d = _disco(combined_score_threshold=0.1, text_weight=1.0, comovement_weight=0.0)
        d.add_market("m1", "BTC 50K March", venue="kalshi")
        d.add_market("m2", "BTC 50K March end", venue="kalshi")
        report = d.discover()
        if report.candidates_found > 0:
            assert report.candidates[0].cross_venue is False

    def test_pairs_evaluated(self) -> None:
        d = _disco(combined_score_threshold=0.99)
        d.add_market("a", "x")
        d.add_market("b", "y")
        d.add_market("c", "z")
        report = d.discover()
        assert report.pairs_evaluated == 3  # C(3,2) = 3.
        assert report.total_markets == 3

    def test_sorted_by_score(self) -> None:
        d = _disco(combined_score_threshold=0.01, text_weight=1.0, comovement_weight=0.0)
        d.add_market("m1", "alpha beta gamma")
        d.add_market("m2", "alpha beta delta")  # 2/4 shared.
        d.add_market("m3", "alpha beta gamma delta")  # 3/4 shared with m1.
        report = d.discover()
        if report.candidates_found >= 2:
            assert report.candidates[0].combined_score >= report.candidates[1].combined_score


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


class TestMisc:
    def test_clear(self) -> None:
        d = _disco()
        d.add_market("m1", "test")
        d.clear()
        assert d.market_count() == 0


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_cross_venue_discovery(self) -> None:
        """Full workflow: add markets, record prices, discover links."""
        d = _disco(
            combined_score_threshold=0.20,
            min_price_observations=3,
            text_weight=0.5,
            comovement_weight=0.5,
        )

        # Two markets about the same underlying on different venues.
        d.add_market("k1", "Will Bitcoin exceed 50000 by end of March 2024?", venue="kalshi")
        d.add_market("p1", "Bitcoin above 50000 March 2024 deadline", venue="poly")

        # Unrelated market.
        d.add_market("k2", "Will it rain in New York tomorrow?", venue="kalshi")

        # Record correlated prices for BTC markets.
        for i in range(6):
            d.record_price("k1", 0.50 + i * 0.02, ts=float(i))
            d.record_price("p1", 0.48 + i * 0.02, ts=float(i))
            d.record_price("k2", 0.30 + (i % 2) * 0.05, ts=float(i))

        report = d.discover()
        assert report.total_markets == 3
        assert report.pairs_evaluated == 3  # C(3,2).

        # The BTC pair should be the top candidate.
        assert report.candidates_found >= 1
        top = report.candidates[0]
        assert {top.market_a, top.market_b} == {"k1", "p1"}
        assert top.cross_venue is True
        assert top.text_similarity > 0.1
        assert top.comovement > 0.5
