import json
from pathlib import Path

from arb_bot.models import BinaryQuote, ExecutionStyle, OpportunityKind, Side
from arb_bot.strategy import ArbitrageFinder


def test_finds_intra_venue_yes_no_arbitrage() -> None:
    finder = ArbitrageFinder(
        min_net_edge_per_contract=0.01,
        enable_cross_venue=False,
        enable_maker_estimates=False,
    )

    quotes = [
        BinaryQuote(
            venue="kalshi",
            market_id="M1",
            yes_buy_price=0.47,
            no_buy_price=0.48,
            yes_buy_size=100,
            no_buy_size=100,
            fee_per_contract=0.0,
            metadata={"title": "Will it rain tomorrow"},
        ),
        BinaryQuote(
            venue="kalshi",
            market_id="M2",
            yes_buy_price=0.60,
            no_buy_price=0.45,
            yes_buy_size=100,
            no_buy_size=100,
            fee_per_contract=0.0,
            metadata={"title": "Will the team win"},
        ),
    ]

    opportunities = finder.find(quotes)
    assert len(opportunities) == 1
    assert opportunities[0].kind is OpportunityKind.INTRA_VENUE
    assert opportunities[0].execution_style is ExecutionStyle.TAKER
    assert opportunities[0].legs[0].market_id == "M1"


def test_finds_cross_venue_arbitrage_when_markets_match() -> None:
    finder = ArbitrageFinder(
        min_net_edge_per_contract=0.01,
        enable_cross_venue=True,
        cross_venue_min_match_score=0.4,
        enable_maker_estimates=False,
    )

    quotes = [
        BinaryQuote(
            venue="kalshi",
            market_id="K1",
            yes_buy_price=0.44,
            no_buy_price=0.57,
            yes_buy_size=100,
            no_buy_size=100,
            fee_per_contract=0.0,
            metadata={"title": "Will BTC be above 100k on Dec 31"},
        ),
        BinaryQuote(
            venue="polymarket",
            market_id="P1",
            yes_buy_price=0.56,
            no_buy_price=0.50,
            yes_buy_size=100,
            no_buy_size=100,
            fee_per_contract=0.0,
            metadata={"question": "Bitcoin above 100k by Dec 31"},
        ),
    ]

    opportunities = finder.find(quotes)
    cross = [opp for opp in opportunities if opp.kind is OpportunityKind.CROSS_VENUE]
    assert cross
    best = max(cross, key=lambda opp: opp.net_edge_per_contract)
    assert best.net_edge_per_contract > 0.01
    assert {leg.venue for leg in best.legs} == {"kalshi", "polymarket"}
    assert {leg.side for leg in best.legs} == {Side.YES, Side.NO}


def test_uses_mapping_file_for_cross_venue_pairing(tmp_path: Path) -> None:
    mapping_path = tmp_path / "map.csv"
    mapping_path.write_text(
        "group_id,kalshi_market_id,polymarket_market_id\n"
        "g1,K_MATCH,P_MATCH\n",
        encoding="utf-8",
    )

    finder = ArbitrageFinder(
        min_net_edge_per_contract=0.01,
        enable_cross_venue=True,
        cross_venue_mapping_path=str(mapping_path),
        cross_venue_mapping_required=True,
        enable_fuzzy_cross_venue_fallback=False,
        enable_maker_estimates=False,
    )

    quotes = [
        BinaryQuote(
            venue="kalshi",
            market_id="K_MATCH",
            yes_buy_price=0.45,
            no_buy_price=0.60,
            yes_buy_size=100,
            no_buy_size=100,
            fee_per_contract=0.0,
            metadata={"title": "Some kalshi market"},
        ),
        BinaryQuote(
            venue="polymarket",
            market_id="P_MATCH",
            yes_buy_price=0.52,
            no_buy_price=0.50,
            yes_buy_size=100,
            no_buy_size=100,
            fee_per_contract=0.0,
            metadata={"question": "Different text but mapped"},
        ),
    ]

    opportunities = finder.find(quotes)
    cross = [opp for opp in opportunities if opp.kind is OpportunityKind.CROSS_VENUE]
    assert cross
    assert any(opp.match_key.startswith("g1") for opp in cross)


def test_maker_estimate_and_near_detection() -> None:
    finder = ArbitrageFinder(
        min_net_edge_per_contract=0.0,
        enable_cross_venue=False,
        enable_maker_estimates=True,
    )

    quote = BinaryQuote(
        venue="kalshi",
        market_id="M3",
        yes_buy_price=0.519,
        no_buy_price=0.50,
        yes_buy_size=100,
        no_buy_size=100,
        yes_maker_buy_price=0.50,
        no_maker_buy_price=0.49,
        yes_maker_buy_size=100,
        no_maker_buy_size=100,
        fee_per_contract=0.0,
        metadata={"title": "maker test market"},
    )

    opportunities = finder.find([quote], min_net_edge_override=-0.02)
    styles = {opp.execution_style for opp in opportunities}
    assert ExecutionStyle.TAKER in styles
    assert ExecutionStyle.MAKER_ESTIMATE in styles

    near = finder.find_near([quote], max_total_cost=1.02)
    assert near


def test_structural_rules_generate_bucket_event_tree_and_parity_opportunities(tmp_path: Path) -> None:
    rules_path = tmp_path / "structural.json"
    rules_path.write_text(
        json.dumps(
            {
                "mutually_exclusive_buckets": [
                    {
                        "group_id": "bucket1",
                        "payout_per_contract": 1.0,
                        "legs": [
                            {"venue": "kalshi", "market_id": "B1", "side": "yes"},
                            {"venue": "kalshi", "market_id": "B2", "side": "yes"},
                            {"venue": "kalshi", "market_id": "B3", "side": "yes"},
                        ],
                    }
                ],
                "event_trees": [
                    {
                        "group_id": "tree1",
                        "parent": {"venue": "kalshi", "market_id": "P", "side": "yes"},
                        "children": [
                            {"venue": "kalshi", "market_id": "C1", "side": "yes"},
                            {"venue": "kalshi", "market_id": "C2", "side": "yes"},
                        ],
                    }
                ],
                "cross_market_parity_checks": [
                    {
                        "group_id": "parity1",
                        "relationship": "equivalent",
                        "left": {"venue": "kalshi", "market_id": "L", "side": "yes"},
                        "right": {"venue": "polymarket", "market_id": "R", "side": "yes"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    finder = ArbitrageFinder(
        min_net_edge_per_contract=0.01,
        enable_cross_venue=False,
        enable_structural_arb=True,
        structural_rules_path=str(rules_path),
        enable_maker_estimates=False,
    )

    quotes = [
        BinaryQuote("kalshi", "B1", 0.30, 0.70, 100, 100, fee_per_contract=0.0),
        BinaryQuote("kalshi", "B2", 0.30, 0.70, 100, 100, fee_per_contract=0.0),
        BinaryQuote("kalshi", "B3", 0.30, 0.70, 100, 100, fee_per_contract=0.0),
        BinaryQuote("kalshi", "P", 0.62, 0.40, 100, 100, fee_per_contract=0.0),
        BinaryQuote("kalshi", "C1", 0.30, 0.70, 100, 100, fee_per_contract=0.0),
        BinaryQuote("kalshi", "C2", 0.25, 0.75, 100, 100, fee_per_contract=0.0),
        BinaryQuote("kalshi", "L", 0.48, 0.52, 100, 100, fee_per_contract=0.0),
        BinaryQuote("polymarket", "R", 0.53, 0.47, 100, 100, fee_per_contract=0.0),
    ]

    opportunities = finder.find(quotes)
    kinds = {opp.kind for opp in opportunities}

    assert OpportunityKind.STRUCTURAL_BUCKET in kinds
    assert OpportunityKind.STRUCTURAL_EVENT_TREE in kinds
    assert OpportunityKind.STRUCTURAL_PARITY in kinds

    bucket = next(opp for opp in opportunities if opp.kind is OpportunityKind.STRUCTURAL_BUCKET)
    assert len(bucket.legs) == 3
    assert bucket.payout_per_contract == 1.0


def test_find_by_kind_filters_requested_lane_only() -> None:
    finder = ArbitrageFinder(
        min_net_edge_per_contract=0.01,
        enable_cross_venue=True,
        cross_venue_min_match_score=0.4,
        enable_maker_estimates=False,
    )

    quotes = [
        BinaryQuote(
            venue="kalshi",
            market_id="K1",
            yes_buy_price=0.44,
            no_buy_price=0.48,
            yes_buy_size=100,
            no_buy_size=100,
            fee_per_contract=0.0,
            metadata={"title": "Will BTC be above 100k on Dec 31"},
        ),
        BinaryQuote(
            venue="polymarket",
            market_id="P1",
            yes_buy_price=0.56,
            no_buy_price=0.50,
            yes_buy_size=100,
            no_buy_size=100,
            fee_per_contract=0.0,
            metadata={"question": "Bitcoin above 100k by Dec 31"},
        ),
    ]

    intra_only = finder.find_by_kind(quotes, kinds={OpportunityKind.INTRA_VENUE})
    assert intra_only
    assert all(opp.kind is OpportunityKind.INTRA_VENUE for opp in intra_only)

    cross_only = finder.find_by_kind(quotes, kinds={OpportunityKind.CROSS_VENUE})
    assert cross_only
    assert all(opp.kind is OpportunityKind.CROSS_VENUE for opp in cross_only)


def test_tighter_price_sum_for_heuristic_buckets(tmp_path: Path) -> None:
    """Heuristic bucket with price_sum=0.82 is rejected, but exchange-confirmed is accepted.

    _BUCKET_PRICE_SUM_MIN_HEURISTIC is 0.85, so 0.82 fails for heuristic.
    _BUCKET_PRICE_SUM_MIN_EXCHANGE is 0.70, so 0.82 passes for exchange_api.
    """
    # Setup: three legs summing to 0.82 total (0.24 + 0.28 + 0.30 = 0.82)
    # Net edge = 1.0 - 0.82 = 0.18 per contract

    # Case 1: heuristic-only bucket -> rejected by tighter band (0.85 min)
    heuristic_rules = {
        "mutually_exclusive_buckets": [
            {
                "group_id": "heuristic_bucket",
                "payout_per_contract": 1.0,
                "exclusivity_source": "heuristic",
                "legs": [
                    {"venue": "kalshi", "market_id": "H1", "side": "yes"},
                    {"venue": "kalshi", "market_id": "H2", "side": "yes"},
                    {"venue": "kalshi", "market_id": "H3", "side": "yes"},
                ],
            }
        ],
        "event_trees": [],
        "cross_market_parity_checks": [],
    }
    rules_path_h = tmp_path / "heuristic.json"
    rules_path_h.write_text(json.dumps(heuristic_rules), encoding="utf-8")

    finder_h = ArbitrageFinder(
        min_net_edge_per_contract=0.01,
        enable_cross_venue=False,
        enable_structural_arb=True,
        structural_rules_path=str(rules_path_h),
        enable_maker_estimates=False,
    )

    quotes = [
        BinaryQuote("kalshi", "H1", 0.24, 0.76, 100, 100, fee_per_contract=0.0),
        BinaryQuote("kalshi", "H2", 0.28, 0.72, 100, 100, fee_per_contract=0.0),
        BinaryQuote("kalshi", "H3", 0.30, 0.70, 100, 100, fee_per_contract=0.0),
    ]

    opps_h = finder_h.find(quotes)
    bucket_opps_h = [o for o in opps_h if o.kind is OpportunityKind.STRUCTURAL_BUCKET]
    assert len(bucket_opps_h) == 0, "heuristic bucket at price_sum=0.82 should be rejected"

    # Case 2: exchange-confirmed bucket -> accepted by wider band (0.70 min)
    exchange_rules = {
        "mutually_exclusive_buckets": [
            {
                "group_id": "exchange_bucket",
                "payout_per_contract": 1.0,
                "exclusivity_source": "exchange_api",
                "legs": [
                    {"venue": "kalshi", "market_id": "H1", "side": "yes"},
                    {"venue": "kalshi", "market_id": "H2", "side": "yes"},
                    {"venue": "kalshi", "market_id": "H3", "side": "yes"},
                ],
            }
        ],
        "event_trees": [],
        "cross_market_parity_checks": [],
    }
    rules_path_e = tmp_path / "exchange.json"
    rules_path_e.write_text(json.dumps(exchange_rules), encoding="utf-8")

    finder_e = ArbitrageFinder(
        min_net_edge_per_contract=0.01,
        enable_cross_venue=False,
        enable_structural_arb=True,
        structural_rules_path=str(rules_path_e),
        enable_maker_estimates=False,
    )

    opps_e = finder_e.find(quotes)
    bucket_opps_e = [o for o in opps_e if o.kind is OpportunityKind.STRUCTURAL_BUCKET]
    assert len(bucket_opps_e) > 0, "exchange bucket at price_sum=0.82 should be accepted"
    assert bucket_opps_e[0].metadata.get("exclusivity_source") == "exchange_api"


# ---------------------------------------------------------------------------
# Tests: max_bucket_legs filter
# ---------------------------------------------------------------------------


def test_max_bucket_legs_filters_large_buckets(tmp_path: Path) -> None:
    """Buckets with more legs than max_bucket_legs are filtered out."""
    rules = {
        "mutually_exclusive_buckets": [
            {
                "group_id": "three_leg",
                "payout_per_contract": 1.0,
                "exclusivity_source": "exchange_api",
                "legs": [
                    {"venue": "kalshi", "market_id": "A", "side": "yes"},
                    {"venue": "kalshi", "market_id": "B", "side": "yes"},
                    {"venue": "kalshi", "market_id": "C", "side": "yes"},
                ],
            },
            {
                "group_id": "five_leg",
                "payout_per_contract": 1.0,
                "exclusivity_source": "exchange_api",
                "legs": [
                    {"venue": "kalshi", "market_id": "D", "side": "yes"},
                    {"venue": "kalshi", "market_id": "E", "side": "yes"},
                    {"venue": "kalshi", "market_id": "F", "side": "yes"},
                    {"venue": "kalshi", "market_id": "G", "side": "yes"},
                    {"venue": "kalshi", "market_id": "H", "side": "yes"},
                ],
            },
        ],
        "event_trees": [],
        "cross_market_parity_checks": [],
    }
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(json.dumps(rules), encoding="utf-8")

    # max_bucket_legs=4: 3-leg bucket passes, 5-leg bucket filtered
    finder = ArbitrageFinder(
        min_net_edge_per_contract=0.01,
        enable_cross_venue=False,
        enable_structural_arb=True,
        structural_rules_path=str(rules_path),
        enable_maker_estimates=False,
        max_bucket_legs=4,
    )

    quotes = [
        BinaryQuote("kalshi", "A", 0.30, 0.70, 100, 100, fee_per_contract=0.0),
        BinaryQuote("kalshi", "B", 0.30, 0.70, 100, 100, fee_per_contract=0.0),
        BinaryQuote("kalshi", "C", 0.30, 0.70, 100, 100, fee_per_contract=0.0),
        BinaryQuote("kalshi", "D", 0.18, 0.82, 100, 100, fee_per_contract=0.0),
        BinaryQuote("kalshi", "E", 0.18, 0.82, 100, 100, fee_per_contract=0.0),
        BinaryQuote("kalshi", "F", 0.18, 0.82, 100, 100, fee_per_contract=0.0),
        BinaryQuote("kalshi", "G", 0.18, 0.82, 100, 100, fee_per_contract=0.0),
        BinaryQuote("kalshi", "H", 0.18, 0.82, 100, 100, fee_per_contract=0.0),
    ]

    opps = finder.find(quotes)
    bucket_opps = [o for o in opps if o.kind is OpportunityKind.STRUCTURAL_BUCKET]

    # Only the 3-leg bucket should pass
    assert len(bucket_opps) == 1
    assert bucket_opps[0].metadata.get("bucket_group_id") == "three_leg"


def test_max_bucket_legs_zero_means_unlimited(tmp_path: Path) -> None:
    """max_bucket_legs=0 means no filtering (unlimited)."""
    rules = {
        "mutually_exclusive_buckets": [
            {
                "group_id": "five_leg",
                "payout_per_contract": 1.0,
                "exclusivity_source": "exchange_api",
                "legs": [
                    {"venue": "kalshi", "market_id": "A", "side": "yes"},
                    {"venue": "kalshi", "market_id": "B", "side": "yes"},
                    {"venue": "kalshi", "market_id": "C", "side": "yes"},
                    {"venue": "kalshi", "market_id": "D", "side": "yes"},
                    {"venue": "kalshi", "market_id": "E", "side": "yes"},
                ],
            },
        ],
        "event_trees": [],
        "cross_market_parity_checks": [],
    }
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(json.dumps(rules), encoding="utf-8")

    finder = ArbitrageFinder(
        min_net_edge_per_contract=0.01,
        enable_cross_venue=False,
        enable_structural_arb=True,
        structural_rules_path=str(rules_path),
        enable_maker_estimates=False,
        max_bucket_legs=0,  # unlimited
    )

    quotes = [
        BinaryQuote("kalshi", "A", 0.18, 0.82, 100, 100, fee_per_contract=0.0),
        BinaryQuote("kalshi", "B", 0.18, 0.82, 100, 100, fee_per_contract=0.0),
        BinaryQuote("kalshi", "C", 0.18, 0.82, 100, 100, fee_per_contract=0.0),
        BinaryQuote("kalshi", "D", 0.18, 0.82, 100, 100, fee_per_contract=0.0),
        BinaryQuote("kalshi", "E", 0.18, 0.82, 100, 100, fee_per_contract=0.0),
    ]

    opps = finder.find(quotes)
    bucket_opps = [o for o in opps if o.kind is OpportunityKind.STRUCTURAL_BUCKET]
    # 5-leg bucket should pass with max_bucket_legs=0
    assert len(bucket_opps) == 1


def test_max_bucket_legs_exact_boundary(tmp_path: Path) -> None:
    """Bucket with exactly max_bucket_legs legs should pass."""
    rules = {
        "mutually_exclusive_buckets": [
            {
                "group_id": "four_leg",
                "payout_per_contract": 1.0,
                "exclusivity_source": "exchange_api",
                "legs": [
                    {"venue": "kalshi", "market_id": "A", "side": "yes"},
                    {"venue": "kalshi", "market_id": "B", "side": "yes"},
                    {"venue": "kalshi", "market_id": "C", "side": "yes"},
                    {"venue": "kalshi", "market_id": "D", "side": "yes"},
                ],
            },
        ],
        "event_trees": [],
        "cross_market_parity_checks": [],
    }
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(json.dumps(rules), encoding="utf-8")

    finder = ArbitrageFinder(
        min_net_edge_per_contract=0.01,
        enable_cross_venue=False,
        enable_structural_arb=True,
        structural_rules_path=str(rules_path),
        enable_maker_estimates=False,
        max_bucket_legs=4,
    )

    quotes = [
        BinaryQuote("kalshi", "A", 0.23, 0.77, 100, 100, fee_per_contract=0.0),
        BinaryQuote("kalshi", "B", 0.23, 0.77, 100, 100, fee_per_contract=0.0),
        BinaryQuote("kalshi", "C", 0.23, 0.77, 100, 100, fee_per_contract=0.0),
        BinaryQuote("kalshi", "D", 0.23, 0.77, 100, 100, fee_per_contract=0.0),
    ]

    opps = finder.find(quotes)
    bucket_opps = [o for o in opps if o.kind is OpportunityKind.STRUCTURAL_BUCKET]
    # 4-leg bucket with max=4 should pass (boundary)
    assert len(bucket_opps) == 1
