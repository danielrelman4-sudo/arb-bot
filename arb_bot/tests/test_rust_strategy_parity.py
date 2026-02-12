"""Parity tests: Rust find_opportunities vs Python ArbitrageFinder (Phase 7B-2).

Runs the same quote sets through both implementations and asserts
opportunity counts, edge values, and leg details match.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List

import pytest

try:
    import arb_engine_rs  # type: ignore[import-untyped]
except ImportError:
    pytest.skip("arb_engine_rs not installed", allow_module_level=True)

from arb_bot.models import (
    ArbitrageOpportunity,
    BinaryQuote,
    ExecutionStyle,
    OpportunityKind,
    Side,
)
from arb_bot.strategy import ArbitrageFinder

TOLERANCE = 1e-10


# ---------------------------------------------------------------------------
# Helpers: serialize Python objects to Rust JSON format
# ---------------------------------------------------------------------------


def _quote_to_dict(q: BinaryQuote) -> Dict[str, Any]:
    """Serialize a BinaryQuote for the Rust find_opportunities input."""
    # Stringify all metadata values for Rust (HashMap<String, String>).
    meta = {str(k): str(v) for k, v in q.metadata.items()}
    return {
        "venue": q.venue,
        "market_id": q.market_id,
        "yes_buy_price": q.yes_buy_price,
        "no_buy_price": q.no_buy_price,
        "yes_buy_size": q.yes_buy_size,
        "no_buy_size": q.no_buy_size,
        "yes_bid_price": q.yes_bid_price,
        "no_bid_price": q.no_bid_price,
        "yes_bid_size": q.yes_bid_size,
        "no_bid_size": q.no_bid_size,
        "yes_maker_buy_price": q.yes_maker_buy_price,
        "no_maker_buy_price": q.no_maker_buy_price,
        "yes_maker_buy_size": q.yes_maker_buy_size,
        "no_maker_buy_size": q.no_maker_buy_size,
        "fee_per_contract": q.fee_per_contract,
        "observed_at": q.observed_at.isoformat(),
        "market_text": q.market_text,
        "metadata": meta,
    }


def _config_dict(
    min_edge: float = 0.01,
    enable_cross: bool = False,
    match_score: float = 0.62,
    mapping_required: bool = False,
    fuzzy_fallback: bool = True,
    maker: bool = False,
    structural: bool = False,
    kinds: List[str] | None = None,
    override_edge: float | None = None,
) -> Dict[str, Any]:
    return {
        "min_net_edge_per_contract": min_edge,
        "enable_cross_venue": enable_cross,
        "cross_venue_min_match_score": match_score,
        "cross_venue_mapping_required": mapping_required,
        "enable_fuzzy_cross_venue_fallback": fuzzy_fallback,
        "enable_maker_estimates": maker,
        "enable_structural_arb": structural,
        "selected_kinds": kinds,
        "min_net_edge_override": override_edge,
    }


def _empty_rules() -> Dict[str, Any]:
    return {"buckets": [], "event_trees": [], "parity_checks": []}


def _run_rust(
    quotes: List[BinaryQuote],
    config: Dict[str, Any],
    rules: Dict[str, Any] | None = None,
    mappings: List[Dict[str, Any]] | None = None,
    enabled_ids: List[str] | None = None,
) -> List[Dict[str, Any]]:
    quotes_json = json.dumps([_quote_to_dict(q) for q in quotes])
    config_json = json.dumps(config)
    rules_json = json.dumps(rules or _empty_rules())
    mappings_json = json.dumps(mappings or [])
    enabled_json = json.dumps(enabled_ids or [])

    result = arb_engine_rs.find_opportunities(
        quotes_json, config_json, rules_json, mappings_json, enabled_json,
    )
    return json.loads(result)


def _close(a: float, b: float) -> bool:
    return abs(a - b) < TOLERANCE


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIntraVenueParity:
    """Compare intra-venue detection."""

    def test_basic_arb(self) -> None:
        quotes = [
            BinaryQuote(
                venue="kalshi", market_id="M1",
                yes_buy_price=0.40, no_buy_price=0.40,
                yes_buy_size=100, no_buy_size=100,
                metadata={"title": "Will it rain"},
            ),
        ]

        # Python
        finder = ArbitrageFinder(
            min_net_edge_per_contract=0.01,
            enable_cross_venue=False,
            enable_maker_estimates=False,
        )
        py_opps = finder.find(quotes)

        # Rust
        config = _config_dict(min_edge=0.01)
        rs_opps = _run_rust(quotes, config)

        assert len(py_opps) == len(rs_opps), f"count: py={len(py_opps)} rs={len(rs_opps)}"
        assert len(py_opps) == 1

        py = py_opps[0]
        rs = rs_opps[0]

        assert py.kind.value == rs["kind"]
        assert py.execution_style.value == rs["execution_style"]
        assert _close(py.net_edge_per_contract, rs["net_edge_per_contract"])
        assert _close(py.gross_edge_per_contract, rs["gross_edge_per_contract"])
        assert len(py.legs) == len(rs["legs"])

    def test_no_arb(self) -> None:
        quotes = [
            BinaryQuote(
                venue="kalshi", market_id="M1",
                yes_buy_price=0.55, no_buy_price=0.55,
                yes_buy_size=100, no_buy_size=100,
                metadata={"title": "test"},
            ),
        ]

        finder = ArbitrageFinder(
            min_net_edge_per_contract=0.01,
            enable_cross_venue=False,
            enable_maker_estimates=False,
        )
        py_opps = finder.find(quotes)

        config = _config_dict(min_edge=0.01)
        rs_opps = _run_rust(quotes, config)

        assert len(py_opps) == 0
        assert len(rs_opps) == 0

    def test_multiple_quotes(self) -> None:
        quotes = [
            BinaryQuote(
                venue="kalshi", market_id="M1",
                yes_buy_price=0.47, no_buy_price=0.48,
                yes_buy_size=100, no_buy_size=100,
                metadata={"title": "Will it rain tomorrow"},
            ),
            BinaryQuote(
                venue="kalshi", market_id="M2",
                yes_buy_price=0.60, no_buy_price=0.45,
                yes_buy_size=100, no_buy_size=100,
                metadata={"title": "Will the team win"},
            ),
        ]

        finder = ArbitrageFinder(
            min_net_edge_per_contract=0.01,
            enable_cross_venue=False,
            enable_maker_estimates=False,
        )
        py_opps = finder.find(quotes)

        config = _config_dict(min_edge=0.01)
        rs_opps = _run_rust(quotes, config)

        assert len(py_opps) == len(rs_opps)
        for py, rs in zip(py_opps, rs_opps):
            assert py.kind.value == rs["kind"]
            assert _close(py.net_edge_per_contract, rs["net_edge_per_contract"])

    def test_with_fees(self) -> None:
        quotes = [
            BinaryQuote(
                venue="kalshi", market_id="M1",
                yes_buy_price=0.40, no_buy_price=0.40,
                yes_buy_size=100, no_buy_size=100,
                fee_per_contract=0.05,
                metadata={"title": "test"},
            ),
        ]

        finder = ArbitrageFinder(
            min_net_edge_per_contract=0.01,
            enable_cross_venue=False,
            enable_maker_estimates=False,
        )
        py_opps = finder.find(quotes)

        config = _config_dict(min_edge=0.01)
        rs_opps = _run_rust(quotes, config)

        assert len(py_opps) == len(rs_opps)
        if py_opps:
            assert _close(py_opps[0].fee_per_contract, rs_opps[0]["fee_per_contract"])
            assert _close(py_opps[0].net_edge_per_contract, rs_opps[0]["net_edge_per_contract"])


class TestCrossVenueFuzzyParity:
    """Compare fuzzy cross-venue detection."""

    def test_basic_cross_venue(self) -> None:
        quotes = [
            BinaryQuote(
                venue="kalshi", market_id="K1",
                yes_buy_price=0.44, no_buy_price=0.57,
                yes_buy_size=100, no_buy_size=100,
                metadata={"title": "Will BTC be above 100k on Dec 31"},
            ),
            BinaryQuote(
                venue="polymarket", market_id="P1",
                yes_buy_price=0.56, no_buy_price=0.50,
                yes_buy_size=100, no_buy_size=100,
                metadata={"question": "Bitcoin above 100k by Dec 31"},
            ),
        ]

        finder = ArbitrageFinder(
            min_net_edge_per_contract=0.01,
            enable_cross_venue=True,
            cross_venue_min_match_score=0.2,
            enable_maker_estimates=False,
        )
        py_opps = finder.find(quotes)
        py_cross = [o for o in py_opps if o.kind is OpportunityKind.CROSS_VENUE]

        config = _config_dict(
            min_edge=0.01, enable_cross=True, match_score=0.2,
        )
        rs_opps = _run_rust(quotes, config)
        rs_cross = [o for o in rs_opps if o["kind"] == "cross_venue"]

        assert len(py_cross) == len(rs_cross), f"cross count: py={len(py_cross)} rs={len(rs_cross)}"
        if py_cross:
            py_best = max(py_cross, key=lambda o: o.net_edge_per_contract)
            rs_best = max(rs_cross, key=lambda o: o["net_edge_per_contract"])
            assert _close(py_best.net_edge_per_contract, rs_best["net_edge_per_contract"])

    def test_same_venue_not_matched(self) -> None:
        """Fuzzy should not match quotes from the same venue."""
        quotes = [
            BinaryQuote(
                venue="kalshi", market_id="K1",
                yes_buy_price=0.40, no_buy_price=0.40,
                yes_buy_size=100, no_buy_size=100,
                metadata={"title": "BTC above 50K March"},
            ),
            BinaryQuote(
                venue="kalshi", market_id="K2",
                yes_buy_price=0.40, no_buy_price=0.40,
                yes_buy_size=100, no_buy_size=100,
                metadata={"title": "BTC above 50K March end"},
            ),
        ]

        finder = ArbitrageFinder(
            min_net_edge_per_contract=0.01,
            enable_cross_venue=True,
            cross_venue_min_match_score=0.2,
            enable_maker_estimates=False,
        )
        py_cross = [o for o in finder.find(quotes) if o.kind is OpportunityKind.CROSS_VENUE]

        config = _config_dict(min_edge=0.01, enable_cross=True, match_score=0.2)
        rs_cross = [o for o in _run_rust(quotes, config) if o["kind"] == "cross_venue"]

        assert len(py_cross) == 0
        assert len(rs_cross) == 0


class TestStructuralParity:
    """Compare structural detection using in-memory rules."""

    def _make_structural_quotes(self) -> List[BinaryQuote]:
        return [
            BinaryQuote(
                venue="kalshi", market_id="m1",
                yes_buy_price=0.30, no_buy_price=0.70,
                yes_buy_size=100, no_buy_size=100,
                metadata={"title": "outcome A"},
            ),
            BinaryQuote(
                venue="kalshi", market_id="m2",
                yes_buy_price=0.30, no_buy_price=0.70,
                yes_buy_size=100, no_buy_size=100,
                metadata={"title": "outcome B"},
            ),
            BinaryQuote(
                venue="kalshi", market_id="m3",
                yes_buy_price=0.30, no_buy_price=0.70,
                yes_buy_size=100, no_buy_size=100,
                metadata={"title": "outcome C"},
            ),
        ]

    def test_bucket_detection(self) -> None:
        quotes = self._make_structural_quotes()

        rules = {
            "buckets": [{
                "group_id": "bucket1",
                "legs": [
                    {"venue": "kalshi", "market_id": "m1", "side": "yes"},
                    {"venue": "kalshi", "market_id": "m2", "side": "yes"},
                    {"venue": "kalshi", "market_id": "m3", "side": "yes"},
                ],
                "payout_per_contract": 1.0,
            }],
            "event_trees": [],
            "parity_checks": [],
        }

        config = _config_dict(min_edge=0.01, structural=True)
        rs_opps = _run_rust(quotes, config, rules=rules)
        rs_bucket = [o for o in rs_opps if o["kind"] == "structural_bucket"]

        assert len(rs_bucket) == 1
        # 3 legs at 0.30 each = 0.90 cost, payout 1.0, edge = 0.10
        assert _close(rs_bucket[0]["net_edge_per_contract"], 0.10)

    def test_event_tree_detection(self) -> None:
        parent = BinaryQuote(
            venue="kalshi", market_id="parent",
            yes_buy_price=0.60, no_buy_price=0.45,
            yes_buy_size=100, no_buy_size=100,
            metadata={"title": "parent event"},
        )
        child1 = BinaryQuote(
            venue="kalshi", market_id="child1",
            yes_buy_price=0.25, no_buy_price=0.80,
            yes_buy_size=100, no_buy_size=100,
            metadata={"title": "child 1"},
        )
        child2 = BinaryQuote(
            venue="kalshi", market_id="child2",
            yes_buy_price=0.25, no_buy_price=0.80,
            yes_buy_size=100, no_buy_size=100,
            metadata={"title": "child 2"},
        )
        quotes = [parent, child1, child2]

        rules = {
            "buckets": [],
            "event_trees": [{
                "group_id": "tree1",
                "parent": {"venue": "kalshi", "market_id": "parent", "side": "yes"},
                "children": [
                    {"venue": "kalshi", "market_id": "child1", "side": "yes"},
                    {"venue": "kalshi", "market_id": "child2", "side": "yes"},
                ],
            }],
            "parity_checks": [],
        }

        config = _config_dict(min_edge=0.001, structural=True)
        rs_opps = _run_rust(quotes, config, rules=rules)
        rs_trees = [o for o in rs_opps if o["kind"] == "structural_event_tree"]

        # Should find up to 2 baskets (parent_no_children_yes, parent_yes_children_no)
        assert len(rs_trees) >= 1

    def test_parity_detection(self) -> None:
        left = BinaryQuote(
            venue="kalshi", market_id="left",
            yes_buy_price=0.40, no_buy_price=0.40,
            yes_buy_size=100, no_buy_size=100,
            metadata={"title": "left market"},
        )
        right = BinaryQuote(
            venue="polymarket", market_id="right",
            yes_buy_price=0.40, no_buy_price=0.40,
            yes_buy_size=100, no_buy_size=100,
            metadata={"title": "right market"},
        )
        quotes = [left, right]

        rules = {
            "buckets": [],
            "event_trees": [],
            "parity_checks": [{
                "group_id": "parity1",
                "left": {"venue": "kalshi", "market_id": "left", "side": "yes"},
                "right": {"venue": "polymarket", "market_id": "right", "side": "yes"},
                "relationship": "equivalent",
            }],
        }

        config = _config_dict(min_edge=0.01, structural=True)
        rs_opps = _run_rust(quotes, config, rules=rules)
        rs_parity = [o for o in rs_opps if o["kind"] == "structural_parity"]

        # Equivalent: (left_YES, flip(right)=right_NO) and (flip(left)=left_NO, right_YES)
        # With prices 0.40 + 0.40 = 0.80, edge = 0.20
        assert len(rs_parity) == 2
        for opp in rs_parity:
            assert _close(opp["net_edge_per_contract"], 0.20)


class TestMakerEstimateParity:
    """Test maker estimate style detection."""

    def test_maker_style_doubles_results(self) -> None:
        quotes = [
            BinaryQuote(
                venue="kalshi", market_id="M1",
                yes_buy_price=0.40, no_buy_price=0.40,
                yes_buy_size=100, no_buy_size=100,
                yes_maker_buy_price=0.38, no_maker_buy_price=0.38,
                yes_maker_buy_size=50, no_maker_buy_size=50,
                metadata={"title": "test maker"},
            ),
        ]

        # Without maker
        finder_no_maker = ArbitrageFinder(
            min_net_edge_per_contract=0.01,
            enable_cross_venue=False,
            enable_maker_estimates=False,
        )
        py_no_maker = finder_no_maker.find(quotes)

        config_no_maker = _config_dict(min_edge=0.01)
        rs_no_maker = _run_rust(quotes, config_no_maker)

        # With maker
        finder_maker = ArbitrageFinder(
            min_net_edge_per_contract=0.01,
            enable_cross_venue=False,
            enable_maker_estimates=True,
        )
        py_maker = finder_maker.find(quotes)

        config_maker = _config_dict(min_edge=0.01, maker=True)
        rs_maker = _run_rust(quotes, config_maker)

        # Should have more with maker enabled
        assert len(py_no_maker) == len(rs_no_maker)
        assert len(py_maker) == len(rs_maker)
        assert len(py_maker) > len(py_no_maker)


class TestEdgeValueParity:
    """Detailed edge value comparison across scenarios."""

    SCENARIOS = [
        # (yes_buy, no_buy, fee, expected_net_edge)
        (0.40, 0.40, 0.00, 0.20),
        (0.45, 0.45, 0.00, 0.10),
        (0.49, 0.49, 0.00, 0.02),
        (0.40, 0.40, 0.05, 0.15),
        (0.30, 0.50, 0.00, 0.20),
        (0.50, 0.30, 0.00, 0.20),
    ]

    @pytest.mark.parametrize("yes_buy,no_buy,fee,expected_edge", SCENARIOS)
    def test_edge_values(
        self, yes_buy: float, no_buy: float, fee: float, expected_edge: float,
    ) -> None:
        quotes = [
            BinaryQuote(
                venue="kalshi", market_id="M1",
                yes_buy_price=yes_buy, no_buy_price=no_buy,
                yes_buy_size=100, no_buy_size=100,
                fee_per_contract=fee,
                metadata={"title": "test"},
            ),
        ]

        finder = ArbitrageFinder(
            min_net_edge_per_contract=0.001,
            enable_cross_venue=False,
            enable_maker_estimates=False,
        )
        py_opps = finder.find(quotes)

        config = _config_dict(min_edge=0.001)
        rs_opps = _run_rust(quotes, config)

        assert len(py_opps) == len(rs_opps)
        if py_opps:
            assert _close(py_opps[0].net_edge_per_contract, expected_edge)
            assert _close(rs_opps[0]["net_edge_per_contract"], expected_edge)
            assert _close(py_opps[0].net_edge_per_contract, rs_opps[0]["net_edge_per_contract"])


class TestSortingParity:
    """Ensure sorting order matches."""

    def test_sorting_by_net_edge(self) -> None:
        quotes = [
            BinaryQuote(
                venue="kalshi", market_id="M1",
                yes_buy_price=0.40, no_buy_price=0.40,
                yes_buy_size=100, no_buy_size=100,
                metadata={"title": "high edge"},
            ),
            BinaryQuote(
                venue="kalshi", market_id="M2",
                yes_buy_price=0.45, no_buy_price=0.45,
                yes_buy_size=100, no_buy_size=100,
                metadata={"title": "lower edge"},
            ),
        ]

        finder = ArbitrageFinder(
            min_net_edge_per_contract=0.01,
            enable_cross_venue=False,
            enable_maker_estimates=False,
        )
        py_opps = finder.find(quotes)

        config = _config_dict(min_edge=0.01)
        rs_opps = _run_rust(quotes, config)

        assert len(py_opps) == len(rs_opps) == 2

        # Both should be sorted net_edge DESC
        assert py_opps[0].net_edge_per_contract >= py_opps[1].net_edge_per_contract
        assert rs_opps[0]["net_edge_per_contract"] >= rs_opps[1]["net_edge_per_contract"]

        # And the actual values should match
        for py, rs in zip(py_opps, rs_opps):
            assert _close(py.net_edge_per_contract, rs["net_edge_per_contract"])


class TestPerformanceStrategy:
    """Benchmark Rust vs Python strategy performance."""

    def test_intra_venue_perf(self) -> None:
        import time

        quotes = [
            BinaryQuote(
                venue="kalshi", market_id=f"M{i}",
                yes_buy_price=0.40 + i * 0.001, no_buy_price=0.40 + i * 0.001,
                yes_buy_size=100, no_buy_size=100,
                metadata={"title": f"market {i}"},
            )
            for i in range(200)
        ]

        finder = ArbitrageFinder(
            min_net_edge_per_contract=0.01,
            enable_cross_venue=False,
            enable_maker_estimates=False,
        )

        start = time.perf_counter()
        py_opps = finder.find(quotes)
        py_time = time.perf_counter() - start

        config = _config_dict(min_edge=0.01)
        start = time.perf_counter()
        rs_opps = _run_rust(quotes, config)
        rs_time = time.perf_counter() - start

        assert len(py_opps) == len(rs_opps)
        print(f"\nIntra-venue 200 quotes: Python={py_time:.4f}s  Rust={rs_time:.4f}s  "
              f"speedup={py_time / max(rs_time, 1e-9):.1f}x")

    def test_fuzzy_cross_venue_perf(self) -> None:
        import time

        quotes = []
        for i in range(50):
            quotes.append(BinaryQuote(
                venue="kalshi", market_id=f"K{i}",
                yes_buy_price=0.40 + i * 0.002, no_buy_price=0.60 - i * 0.002,
                yes_buy_size=100, no_buy_size=100,
                metadata={"title": f"Will event {i} happen by March 2024"},
            ))
            quotes.append(BinaryQuote(
                venue="polymarket", market_id=f"P{i}",
                yes_buy_price=0.42 + i * 0.002, no_buy_price=0.58 - i * 0.002,
                yes_buy_size=100, no_buy_size=100,
                metadata={"question": f"Event {i} happening March 2024"},
            ))

        finder = ArbitrageFinder(
            min_net_edge_per_contract=0.01,
            enable_cross_venue=True,
            cross_venue_min_match_score=0.3,
            enable_maker_estimates=False,
        )

        start = time.perf_counter()
        py_opps = finder.find(quotes)
        py_time = time.perf_counter() - start

        config = _config_dict(min_edge=0.01, enable_cross=True, match_score=0.3)
        start = time.perf_counter()
        rs_opps = _run_rust(quotes, config)
        rs_time = time.perf_counter() - start

        print(f"\nFuzzy cross-venue 100 quotes: Python={py_time:.4f}s  Rust={rs_time:.4f}s  "
              f"speedup={py_time / max(rs_time, 1e-9):.1f}x")
