import json
from datetime import datetime, timezone
from pathlib import Path

from arb_bot.correlation import CorrelationConstraintModel
from arb_bot.models import ArbitrageOpportunity, ExecutionStyle, OpportunityKind, OpportunityLeg, Side


def _build_opportunity(
    *,
    kind: OpportunityKind,
    legs: tuple[OpportunityLeg, ...],
    match_key: str,
    match_score: float = 1.0,
) -> ArbitrageOpportunity:
    payout = 1.0
    total_cost = sum(leg.buy_price for leg in legs)
    gross = payout - total_cost
    return ArbitrageOpportunity(
        kind=kind,
        execution_style=ExecutionStyle.TAKER,
        legs=legs,
        gross_edge_per_contract=gross,
        net_edge_per_contract=gross,
        fee_per_contract=0.0,
        observed_at=datetime.now(timezone.utc),
        match_key=match_key,
        match_score=match_score,
        payout_per_contract=payout,
    )


def test_constraint_model_preserves_single_market_yes_no_payout() -> None:
    model = CorrelationConstraintModel(
        structural_rules_path=None,
        enabled=True,
        max_bruteforce_markets=10,
    )

    opp = _build_opportunity(
        kind=OpportunityKind.INTRA_VENUE,
        legs=(
            OpportunityLeg("kalshi", "m1", Side.YES, 0.45, 100),
            OpportunityLeg("kalshi", "m1", Side.NO, 0.50, 100),
        ),
        match_key="kalshi:m1",
    )

    assessment = model.assess(opp)
    assert assessment.min_payout_per_contract == 1.0
    assert assessment.adjusted_payout_per_contract == 1.0
    assert assessment.residual_net_edge_per_contract == opp.net_edge_per_contract


def test_constraint_model_penalizes_uncovered_event_tree_combo(tmp_path: Path) -> None:
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(
        json.dumps(
            {
                "event_trees": [
                    {
                        "group_id": "tree",
                        "parent": {"venue": "kalshi", "market_id": "parent", "side": "yes"},
                        "children": [
                            {"venue": "kalshi", "market_id": "child_a", "side": "yes"},
                            {"venue": "kalshi", "market_id": "child_b", "side": "yes"},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    model = CorrelationConstraintModel(
        structural_rules_path=str(rules_path),
        enabled=True,
        max_bruteforce_markets=10,
    )

    # child_a YES + parent NO has a zero-payout state (parent YES, child_b YES).
    opp = _build_opportunity(
        kind=OpportunityKind.INTRA_VENUE,
        legs=(
            OpportunityLeg("kalshi", "child_a", Side.YES, 0.05, 100),
            OpportunityLeg("kalshi", "parent", Side.NO, 0.90, 100),
        ),
        match_key="tree:test",
        match_score=1.0,
    )

    assessment = model.assess(opp)
    assert assessment.min_payout_per_contract == 0.0
    assert assessment.adjusted_payout_per_contract == 0.0
    assert assessment.residual_net_edge_per_contract < 0.0


def test_cross_venue_equivalence_assumption_depends_on_match_score() -> None:
    model = CorrelationConstraintModel(
        structural_rules_path=None,
        enabled=True,
        max_bruteforce_markets=10,
        cross_venue_equivalence_min_match_score=0.9,
    )

    strong_match = _build_opportunity(
        kind=OpportunityKind.CROSS_VENUE,
        legs=(
            OpportunityLeg("kalshi", "same_event_k", Side.YES, 0.40, 100),
            OpportunityLeg("polymarket", "same_event_p", Side.NO, 0.55, 100),
        ),
        match_key="same-event",
        match_score=1.0,
    )
    weak_match = _build_opportunity(
        kind=OpportunityKind.CROSS_VENUE,
        legs=(
            OpportunityLeg("kalshi", "unrelated_k", Side.YES, 0.40, 100),
            OpportunityLeg("polymarket", "unrelated_p", Side.NO, 0.55, 100),
        ),
        match_key="weak-event",
        match_score=0.3,
    )

    strong_assessment = model.assess(strong_match)
    weak_assessment = model.assess(weak_match)

    assert strong_assessment.min_payout_per_contract == 1.0
    assert weak_assessment.min_payout_per_contract == 0.0


def test_structural_bucket_can_be_priced_as_non_exhaustive(tmp_path: Path) -> None:
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(
        json.dumps(
            {
                "mutually_exclusive_buckets": [
                    {
                        "group_id": "bucket",
                        "payout_per_contract": 1.0,
                        "legs": [
                            {"venue": "kalshi", "market_id": "a", "side": "yes"},
                            {"venue": "kalshi", "market_id": "b", "side": "yes"},
                            {"venue": "kalshi", "market_id": "c", "side": "yes"},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    model_exhaustive = CorrelationConstraintModel(
        structural_rules_path=str(rules_path),
        enabled=True,
        max_bruteforce_markets=10,
        assume_structural_buckets_exhaustive=True,
    )
    model_non_exhaustive = CorrelationConstraintModel(
        structural_rules_path=str(rules_path),
        enabled=True,
        max_bruteforce_markets=10,
        assume_structural_buckets_exhaustive=False,
    )

    opp = _build_opportunity(
        kind=OpportunityKind.STRUCTURAL_BUCKET,
        legs=(
            OpportunityLeg("kalshi", "a", Side.YES, 0.10, 100),
            OpportunityLeg("kalshi", "b", Side.YES, 0.10, 100),
            OpportunityLeg("kalshi", "c", Side.YES, 0.10, 100),
        ),
        match_key="bucket",
    )

    exhaustive_assessment = model_exhaustive.assess(opp)
    non_exhaustive_assessment = model_non_exhaustive.assess(opp)

    assert exhaustive_assessment.min_payout_per_contract == 1.0
    assert non_exhaustive_assessment.min_payout_per_contract == 0.0
    assert non_exhaustive_assessment.residual_net_edge_per_contract < 0.0
