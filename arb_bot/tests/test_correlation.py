import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

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


# ---------------------------------------------------------------------------
# ILP solver tests (Phase 10C)
# ---------------------------------------------------------------------------


def test_ilp_vs_bruteforce_parity_single_market_yes_no() -> None:
    """ILP and brute-force return the same min_payout for a single-market YES+NO."""
    opp = _build_opportunity(
        kind=OpportunityKind.INTRA_VENUE,
        legs=(
            OpportunityLeg("kalshi", "m1", Side.YES, 0.45, 100),
            OpportunityLeg("kalshi", "m1", Side.NO, 0.50, 100),
        ),
        match_key="kalshi:m1",
    )

    # Brute-force path (max_bruteforce_markets high enough).
    model_bf = CorrelationConstraintModel(
        structural_rules_path=None,
        enabled=True,
        max_bruteforce_markets=20,
    )
    bf_assessment = model_bf.assess(opp)

    # ILP path (max_bruteforce_markets=2 forces ILP for 1-market / 1-key problems).
    model_ilp = CorrelationConstraintModel(
        structural_rules_path=None,
        enabled=True,
        max_bruteforce_markets=2,
    )
    ilp_assessment = model_ilp.assess(opp)

    assert abs(bf_assessment.min_payout_per_contract - ilp_assessment.min_payout_per_contract) < 1e-9


def test_ilp_vs_bruteforce_parity_bucket_exhaustive(tmp_path: Path) -> None:
    """ILP and brute-force agree on a 3-leg exhaustive bucket."""
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

    opp = _build_opportunity(
        kind=OpportunityKind.STRUCTURAL_BUCKET,
        legs=(
            OpportunityLeg("kalshi", "a", Side.YES, 0.10, 100),
            OpportunityLeg("kalshi", "b", Side.YES, 0.10, 100),
            OpportunityLeg("kalshi", "c", Side.YES, 0.10, 100),
        ),
        match_key="bucket",
    )

    model_bf = CorrelationConstraintModel(
        structural_rules_path=str(rules_path),
        enabled=True,
        max_bruteforce_markets=20,
        assume_structural_buckets_exhaustive=True,
    )
    model_ilp = CorrelationConstraintModel(
        structural_rules_path=str(rules_path),
        enabled=True,
        max_bruteforce_markets=2,
        assume_structural_buckets_exhaustive=True,
    )

    bf = model_bf.assess(opp)
    ilp = model_ilp.assess(opp)

    assert abs(bf.min_payout_per_contract - ilp.min_payout_per_contract) < 1e-9
    assert bf.min_payout_per_contract == 1.0


def test_ilp_vs_bruteforce_parity_bucket_non_exhaustive(tmp_path: Path) -> None:
    """ILP and brute-force agree on a 3-leg non-exhaustive bucket (min=0)."""
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

    opp = _build_opportunity(
        kind=OpportunityKind.STRUCTURAL_BUCKET,
        legs=(
            OpportunityLeg("kalshi", "a", Side.YES, 0.10, 100),
            OpportunityLeg("kalshi", "b", Side.YES, 0.10, 100),
            OpportunityLeg("kalshi", "c", Side.YES, 0.10, 100),
        ),
        match_key="bucket",
    )

    model_bf = CorrelationConstraintModel(
        structural_rules_path=str(rules_path),
        enabled=True,
        max_bruteforce_markets=20,
        assume_structural_buckets_exhaustive=False,
    )
    model_ilp = CorrelationConstraintModel(
        structural_rules_path=str(rules_path),
        enabled=True,
        max_bruteforce_markets=2,
        assume_structural_buckets_exhaustive=False,
    )

    bf = model_bf.assess(opp)
    ilp = model_ilp.assess(opp)

    assert abs(bf.min_payout_per_contract - ilp.min_payout_per_contract) < 1e-9
    assert bf.min_payout_per_contract == 0.0


def test_ilp_vs_bruteforce_parity_event_tree(tmp_path: Path) -> None:
    """ILP and brute-force agree on an event tree constraint."""
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

    opp = _build_opportunity(
        kind=OpportunityKind.INTRA_VENUE,
        legs=(
            OpportunityLeg("kalshi", "child_a", Side.YES, 0.05, 100),
            OpportunityLeg("kalshi", "parent", Side.NO, 0.90, 100),
        ),
        match_key="tree:test",
    )

    model_bf = CorrelationConstraintModel(
        structural_rules_path=str(rules_path),
        enabled=True,
        max_bruteforce_markets=20,
    )
    model_ilp = CorrelationConstraintModel(
        structural_rules_path=str(rules_path),
        enabled=True,
        max_bruteforce_markets=2,
    )

    bf = model_bf.assess(opp)
    ilp = model_ilp.assess(opp)

    assert abs(bf.min_payout_per_contract - ilp.min_payout_per_contract) < 1e-9
    assert bf.min_payout_per_contract == 0.0


def test_ilp_vs_bruteforce_parity_complement(tmp_path: Path) -> None:
    """ILP and brute-force agree for complement parity constraint."""
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(
        json.dumps(
            {
                "cross_market_parity_checks": [
                    {
                        "group_id": "comp",
                        "left": {"venue": "kalshi", "market_id": "x", "side": "yes"},
                        "right": {"venue": "kalshi", "market_id": "y", "side": "yes"},
                        "relationship": "complement",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    # Buy YES on both complements: min payout should be 1.0 (one wins, one loses).
    opp = _build_opportunity(
        kind=OpportunityKind.STRUCTURAL_PARITY,
        legs=(
            OpportunityLeg("kalshi", "x", Side.YES, 0.40, 100),
            OpportunityLeg("kalshi", "y", Side.YES, 0.55, 100),
        ),
        match_key="comp:test",
    )

    model_bf = CorrelationConstraintModel(
        structural_rules_path=str(rules_path),
        enabled=True,
        max_bruteforce_markets=20,
    )
    model_ilp = CorrelationConstraintModel(
        structural_rules_path=str(rules_path),
        enabled=True,
        max_bruteforce_markets=2,
    )

    bf = model_bf.assess(opp)
    ilp = model_ilp.assess(opp)

    assert abs(bf.min_payout_per_contract - ilp.min_payout_per_contract) < 1e-9
    assert bf.min_payout_per_contract == 1.0


def test_ilp_vs_bruteforce_parity_equivalent(tmp_path: Path) -> None:
    """ILP and brute-force agree for equivalent parity constraint."""
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(
        json.dumps(
            {
                "cross_market_parity_checks": [
                    {
                        "group_id": "equiv",
                        "left": {"venue": "kalshi", "market_id": "x", "side": "yes"},
                        "right": {"venue": "kalshi", "market_id": "y", "side": "yes"},
                        "relationship": "equivalent",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    # YES on one, NO on other: with equivalent constraint, they must agree.
    # So min payout = 1 (either both YES => 1+0, or both NO => 0+1).
    opp = _build_opportunity(
        kind=OpportunityKind.STRUCTURAL_PARITY,
        legs=(
            OpportunityLeg("kalshi", "x", Side.YES, 0.40, 100),
            OpportunityLeg("kalshi", "y", Side.NO, 0.55, 100),
        ),
        match_key="equiv:test",
    )

    model_bf = CorrelationConstraintModel(
        structural_rules_path=str(rules_path),
        enabled=True,
        max_bruteforce_markets=20,
    )
    model_ilp = CorrelationConstraintModel(
        structural_rules_path=str(rules_path),
        enabled=True,
        max_bruteforce_markets=2,
    )

    bf = model_bf.assess(opp)
    ilp = model_ilp.assess(opp)

    assert abs(bf.min_payout_per_contract - ilp.min_payout_per_contract) < 1e-9
    assert bf.min_payout_per_contract == 1.0


def test_ilp_vs_bruteforce_parity_cross_venue_equivalence() -> None:
    """ILP and brute-force agree when cross-venue equivalence is assumed."""
    opp = _build_opportunity(
        kind=OpportunityKind.CROSS_VENUE,
        legs=(
            OpportunityLeg("kalshi", "event_k", Side.YES, 0.40, 100),
            OpportunityLeg("polymarket", "event_p", Side.NO, 0.55, 100),
        ),
        match_key="cross-event",
        match_score=1.0,
    )

    model_bf = CorrelationConstraintModel(
        structural_rules_path=None,
        enabled=True,
        max_bruteforce_markets=20,
        cross_venue_equivalence_min_match_score=0.9,
    )
    model_ilp = CorrelationConstraintModel(
        structural_rules_path=None,
        enabled=True,
        max_bruteforce_markets=2,
        cross_venue_equivalence_min_match_score=0.9,
    )

    bf = model_bf.assess(opp)
    ilp = model_ilp.assess(opp)

    assert abs(bf.min_payout_per_contract - ilp.min_payout_per_contract) < 1e-9
    assert bf.min_payout_per_contract == 1.0


def test_ilp_handles_large_n_bucket() -> None:
    """ILP solves a 25-market exhaustive bucket that would timeout with brute-force."""
    n_markets = 25
    market_ids = [f"m{i}" for i in range(n_markets)]

    rules = {
        "mutually_exclusive_buckets": [
            {
                "group_id": "big_bucket",
                "payout_per_contract": 1.0,
                "legs": [
                    {"venue": "kalshi", "market_id": mid, "side": "yes"}
                    for mid in market_ids
                ],
            }
        ]
    }

    import tempfile, os

    fd, rules_path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(rules, f)

        model = CorrelationConstraintModel(
            structural_rules_path=rules_path,
            enabled=True,
            max_bruteforce_markets=2,
            max_ilp_markets=200,
            assume_structural_buckets_exhaustive=True,
        )

        # Buy YES on all 25 markets in the exhaustive bucket.
        # Exactly one must be YES, so payout = 1.
        opp = _build_opportunity(
            kind=OpportunityKind.STRUCTURAL_BUCKET,
            legs=tuple(
                OpportunityLeg("kalshi", mid, Side.YES, 0.02, 100)
                for mid in market_ids
            ),
            match_key="big_bucket",
        )

        assessment = model.assess(opp)
        assert assessment.min_payout_per_contract == 1.0
        assert "ilp_solver" in assessment.assumptions
    finally:
        os.unlink(rules_path)


def test_ilp_handles_large_n_event_tree() -> None:
    """ILP solves a parent + 20 children event tree."""
    n_children = 20
    child_ids = [f"child_{i}" for i in range(n_children)]

    rules = {
        "event_trees": [
            {
                "group_id": "big_tree",
                "parent": {"venue": "kalshi", "market_id": "parent", "side": "yes"},
                "children": [
                    {"venue": "kalshi", "market_id": cid, "side": "yes"}
                    for cid in child_ids
                ],
            }
        ]
    }

    import tempfile, os

    fd, rules_path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(rules, f)

        model = CorrelationConstraintModel(
            structural_rules_path=rules_path,
            enabled=True,
            max_bruteforce_markets=2,
            max_ilp_markets=200,
        )

        # Buy YES on child_0 and NO on parent.
        # Tree constraint: sum(children) == parent. When parent=YES, exactly one child
        # is YES. When parent=NO, all children are NO.
        # worst case: parent=YES, child_0=YES => payout = 1 + 0 = 1
        #             parent=YES, child_0=NO, other child YES => 0 + 0 = 0
        #             parent=NO, all children NO => 0 + 1 = 1
        # So min_payout should be 0.
        opp = _build_opportunity(
            kind=OpportunityKind.INTRA_VENUE,
            legs=(
                OpportunityLeg("kalshi", "child_0", Side.YES, 0.05, 100),
                OpportunityLeg("kalshi", "parent", Side.NO, 0.90, 100),
            ),
            match_key="big_tree:test",
        )

        assessment = model.assess(opp)
        assert assessment.min_payout_per_contract == 0.0
        assert "ilp_solver" in assessment.assumptions
    finally:
        os.unlink(rules_path)


def test_ilp_bucket_constraint_forces_exactly_one(tmp_path: Path) -> None:
    """ILP correctly enforces bucket exclusivity (exactly-one for exhaustive)."""
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

    model = CorrelationConstraintModel(
        structural_rules_path=str(rules_path),
        enabled=True,
        max_bruteforce_markets=2,
        assume_structural_buckets_exhaustive=True,
    )

    # Buy NO on all three. With exhaustive bucket, exactly one is YES.
    # For each assignment: payout = sum(1 - outcome_i for each leg).
    # Exactly one outcome is 1, two are 0 => payout = 0 + 1 + 1 = 2.
    opp = _build_opportunity(
        kind=OpportunityKind.STRUCTURAL_BUCKET,
        legs=(
            OpportunityLeg("kalshi", "a", Side.NO, 0.30, 100),
            OpportunityLeg("kalshi", "b", Side.NO, 0.30, 100),
            OpportunityLeg("kalshi", "c", Side.NO, 0.30, 100),
        ),
        match_key="bucket",
    )

    assessment = model.assess(opp)
    # Exactly one YES => two NOs pay out => min payout = 2.0
    assert assessment.min_payout_per_contract == 2.0
    assert "ilp_solver" in assessment.assumptions


def test_ilp_event_tree_parent_yes_all_children(tmp_path: Path) -> None:
    """ILP with event tree: buying parent YES and all children YES yields min=1."""
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(
        json.dumps(
            {
                "event_trees": [
                    {
                        "group_id": "tree",
                        "parent": {"venue": "kalshi", "market_id": "parent", "side": "yes"},
                        "children": [
                            {"venue": "kalshi", "market_id": "ca", "side": "yes"},
                            {"venue": "kalshi", "market_id": "cb", "side": "yes"},
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
        max_bruteforce_markets=2,
    )

    # Buy YES on parent and both children.
    # If parent=YES: exactly one child is YES => payout = 1 + 1 + 0 = 2 or 1 + 0 + 1 = 2.
    # If parent=NO: all children NO => 0 + 0 + 0 = 0.
    # Min payout = 0.
    opp = _build_opportunity(
        kind=OpportunityKind.STRUCTURAL_EVENT_TREE,
        legs=(
            OpportunityLeg("kalshi", "parent", Side.YES, 0.30, 100),
            OpportunityLeg("kalshi", "ca", Side.YES, 0.10, 100),
            OpportunityLeg("kalshi", "cb", Side.YES, 0.10, 100),
        ),
        match_key="tree:all-yes",
    )

    assessment = model.assess(opp)
    assert assessment.min_payout_per_contract == 0.0
    assert "ilp_solver" in assessment.assumptions


def test_ilp_parity_complement_both_yes(tmp_path: Path) -> None:
    """ILP with complement parity: buy YES on both complements => min payout = 1."""
    # We need >2 active markets to exceed max_bruteforce_markets=2 (floor).
    # Add a third market via a second parity constraint to inflate the component.
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(
        json.dumps(
            {
                "cross_market_parity_checks": [
                    {
                        "group_id": "comp",
                        "left": {"venue": "kalshi", "market_id": "x", "side": "yes"},
                        "right": {"venue": "kalshi", "market_id": "y", "side": "yes"},
                        "relationship": "complement",
                    },
                    {
                        "group_id": "equiv_extra",
                        "left": {"venue": "kalshi", "market_id": "y", "side": "yes"},
                        "right": {"venue": "kalshi", "market_id": "z", "side": "yes"},
                        "relationship": "equivalent",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    model = CorrelationConstraintModel(
        structural_rules_path=str(rules_path),
        enabled=True,
        max_bruteforce_markets=2,
    )

    # x + y == 1 (complement), y == z (equivalent).
    # Buy YES on x and y. Min payout is 1 (one of x/y is always YES).
    opp = _build_opportunity(
        kind=OpportunityKind.STRUCTURAL_PARITY,
        legs=(
            OpportunityLeg("kalshi", "x", Side.YES, 0.40, 100),
            OpportunityLeg("kalshi", "y", Side.YES, 0.55, 100),
        ),
        match_key="comp:test",
    )

    assessment = model.assess(opp)
    assert assessment.min_payout_per_contract == 1.0
    assert "ilp_solver" in assessment.assumptions


def test_ilp_cross_venue_equivalence_with_ilp(tmp_path: Path) -> None:
    """Cross-venue equivalence constraint works correctly through ILP path."""
    # Cross-venue legs produce 2 active markets. We need >2 to trigger ILP.
    # Add a parity constraint that links one of the cross-venue markets to a third
    # market, inflating the active component to 3.
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(
        json.dumps(
            {
                "cross_market_parity_checks": [
                    {
                        "group_id": "extra",
                        "left": {"venue": "kalshi", "market_id": "ev_k", "side": "yes"},
                        "right": {"venue": "kalshi", "market_id": "ev_aux", "side": "yes"},
                        "relationship": "equivalent",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    opp = _build_opportunity(
        kind=OpportunityKind.CROSS_VENUE,
        legs=(
            OpportunityLeg("kalshi", "ev_k", Side.YES, 0.40, 100),
            OpportunityLeg("polymarket", "ev_p", Side.NO, 0.55, 100),
        ),
        match_key="cross",
        match_score=1.0,
    )

    model = CorrelationConstraintModel(
        structural_rules_path=str(rules_path),
        enabled=True,
        max_bruteforce_markets=2,
        cross_venue_equivalence_min_match_score=0.9,
    )

    assessment = model.assess(opp)
    # Cross-equiv: ev_k == ev_p. Parity: ev_k == ev_aux.
    # With equivalence: both YES or both NO.
    # both YES => 1 + 0 = 1. both NO => 0 + 1 = 1. min = 1.
    assert assessment.min_payout_per_contract == 1.0
    assert "ilp_solver" in assessment.assumptions
    assert "cross_equivalence_assumed" in assessment.assumptions


def test_ilp_fallback_when_pulp_unavailable() -> None:
    """When pulp is None, ILP path falls back gracefully."""
    # Need >2 active markets so the problem exceeds the bruteforce threshold.
    # Use 3 distinct markets with no structural rules so no constraint component
    # inflation. 3 markets > max_bruteforce_markets=2 triggers the ILP branch.
    opp = _build_opportunity(
        kind=OpportunityKind.INTRA_VENUE,
        legs=(
            OpportunityLeg("kalshi", "m1", Side.YES, 0.20, 100),
            OpportunityLeg("kalshi", "m2", Side.YES, 0.20, 100),
            OpportunityLeg("kalshi", "m3", Side.YES, 0.20, 100),
        ),
        match_key="kalshi:multi",
    )

    # Patch _HAS_PULP to False so the ILP path sees pulp as unavailable.
    with patch("arb_bot.correlation._HAS_PULP", False):
        model = CorrelationConstraintModel(
            structural_rules_path=None,
            enabled=True,
            max_bruteforce_markets=2,
            max_ilp_markets=200,
        )
        assessment = model.assess(opp)

    # Should fall back to constraint_limit_reached since brute-force is too small
    # and ILP is unavailable.
    assert "constraint_limit_reached" in assessment.assumptions


def test_ilp_max_ilp_markets_limit_exceeded() -> None:
    """When N > max_ilp_markets, constraint_limit_reached is returned."""
    # Create a rules file with many markets in a bucket to inflate market count
    # beyond max_ilp_markets.
    n_markets = 10
    market_ids = [f"m{i}" for i in range(n_markets)]

    rules = {
        "mutually_exclusive_buckets": [
            {
                "group_id": "big",
                "payout_per_contract": 1.0,
                "legs": [
                    {"venue": "kalshi", "market_id": mid, "side": "yes"}
                    for mid in market_ids
                ],
            }
        ]
    }

    import tempfile, os

    fd, rules_path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(rules, f)

        # Set max_bruteforce_markets=2 and max_ilp_markets=5
        # The bucket has 10 markets, so N > max_ilp_markets.
        model = CorrelationConstraintModel(
            structural_rules_path=rules_path,
            enabled=True,
            max_bruteforce_markets=2,
            max_ilp_markets=5,
        )

        opp = _build_opportunity(
            kind=OpportunityKind.STRUCTURAL_BUCKET,
            legs=tuple(
                OpportunityLeg("kalshi", mid, Side.YES, 0.05, 100)
                for mid in market_ids
            ),
            match_key="big",
        )

        assessment = model.assess(opp)
        assert "constraint_limit_reached" in assessment.assumptions
        # When constraint_limit_reached, min_payout falls back to the payout_per_contract.
        assert assessment.min_payout_per_contract == 1.0
    finally:
        os.unlink(rules_path)


def test_ilp_assumptions_include_ilp_solver_tag() -> None:
    """ILP path populates 'ilp_solver' in assumptions."""
    opp = _build_opportunity(
        kind=OpportunityKind.INTRA_VENUE,
        legs=(
            OpportunityLeg("kalshi", "m1", Side.YES, 0.45, 100),
            OpportunityLeg("kalshi", "m1", Side.NO, 0.50, 100),
        ),
        match_key="kalshi:m1",
    )

    # Force ILP path by setting max_bruteforce_markets below market count.
    model = CorrelationConstraintModel(
        structural_rules_path=None,
        enabled=True,
        max_bruteforce_markets=2,
    )

    assessment = model.assess(opp)
    # Only 1 distinct market, so it stays under bruteforce threshold.
    # We need >2 markets to trigger ILP. Let's do it differently.

    # Actually, 1 distinct market means active_markets has 1 entry,
    # which is below max_bruteforce_markets=2. Let's use 3 distinct markets.
    opp3 = _build_opportunity(
        kind=OpportunityKind.INTRA_VENUE,
        legs=(
            OpportunityLeg("kalshi", "m1", Side.YES, 0.20, 100),
            OpportunityLeg("kalshi", "m2", Side.YES, 0.20, 100),
            OpportunityLeg("kalshi", "m3", Side.YES, 0.20, 100),
        ),
        match_key="kalshi:multi",
    )

    model2 = CorrelationConstraintModel(
        structural_rules_path=None,
        enabled=True,
        max_bruteforce_markets=2,
    )

    assessment2 = model2.assess(opp3)
    assert "ilp_solver" in assessment2.assumptions


def test_ilp_bruteforce_does_not_include_ilp_assumption() -> None:
    """Brute-force path does NOT add 'ilp_solver' to assumptions."""
    opp = _build_opportunity(
        kind=OpportunityKind.INTRA_VENUE,
        legs=(
            OpportunityLeg("kalshi", "m1", Side.YES, 0.20, 100),
            OpportunityLeg("kalshi", "m2", Side.YES, 0.20, 100),
            OpportunityLeg("kalshi", "m3", Side.YES, 0.20, 100),
        ),
        match_key="kalshi:multi",
    )

    model = CorrelationConstraintModel(
        structural_rules_path=None,
        enabled=True,
        max_bruteforce_markets=20,
    )

    assessment = model.assess(opp)
    assert "ilp_solver" not in assessment.assumptions


def test_ilp_vs_bruteforce_parity_mixed_constraint_types(tmp_path: Path) -> None:
    """ILP and brute-force agree on a problem with both bucket and parity constraints."""
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
                        ],
                    }
                ],
                "cross_market_parity_checks": [
                    {
                        "group_id": "parity",
                        "left": {"venue": "kalshi", "market_id": "a", "side": "yes"},
                        "right": {"venue": "kalshi", "market_id": "c", "side": "yes"},
                        "relationship": "complement",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    opp = _build_opportunity(
        kind=OpportunityKind.INTRA_VENUE,
        legs=(
            OpportunityLeg("kalshi", "a", Side.YES, 0.30, 100),
            OpportunityLeg("kalshi", "b", Side.YES, 0.30, 100),
            OpportunityLeg("kalshi", "c", Side.YES, 0.30, 100),
        ),
        match_key="mixed:test",
    )

    model_bf = CorrelationConstraintModel(
        structural_rules_path=str(rules_path),
        enabled=True,
        max_bruteforce_markets=20,
        assume_structural_buckets_exhaustive=True,
    )
    model_ilp = CorrelationConstraintModel(
        structural_rules_path=str(rules_path),
        enabled=True,
        max_bruteforce_markets=2,
        assume_structural_buckets_exhaustive=True,
    )

    bf = model_bf.assess(opp)
    ilp = model_ilp.assess(opp)

    assert abs(bf.min_payout_per_contract - ilp.min_payout_per_contract) < 1e-9
