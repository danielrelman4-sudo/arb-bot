from __future__ import annotations

import json

from arb_bot.structural_rules import load_structural_rules


def test_load_structural_rules_drops_event_tree_with_non_yes_parent(tmp_path) -> None:
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(
        json.dumps(
            {
                "mutually_exclusive_buckets": [],
                "event_trees": [
                    {
                        "group_id": "bad-parent-side",
                        "parent": {"venue": "kalshi", "market_id": "P", "side": "no"},
                        "children": [
                            {"venue": "kalshi", "market_id": "C1", "side": "yes"},
                            {"venue": "kalshi", "market_id": "C2", "side": "yes"},
                        ],
                    }
                ],
                "cross_market_parity_checks": [],
            }
        ),
        encoding="utf-8",
    )

    rules = load_structural_rules(str(rules_path))
    assert len(rules.event_trees) == 0


def test_load_structural_rules_drops_event_tree_contained_in_bucket(tmp_path) -> None:
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(
        json.dumps(
            {
                "mutually_exclusive_buckets": [
                    {
                        "group_id": "bucket",
                        "payout_per_contract": 1.0,
                        "legs": [
                            {"venue": "kalshi", "market_id": "P", "side": "yes"},
                            {"venue": "kalshi", "market_id": "C1", "side": "yes"},
                            {"venue": "kalshi", "market_id": "C2", "side": "yes"},
                        ],
                    }
                ],
                "event_trees": [
                    {
                        "group_id": "conflicting-tree",
                        "parent": {"venue": "kalshi", "market_id": "P", "side": "yes"},
                        "children": [
                            {"venue": "kalshi", "market_id": "C1", "side": "yes"},
                            {"venue": "kalshi", "market_id": "C2", "side": "yes"},
                        ],
                    }
                ],
                "cross_market_parity_checks": [],
            }
        ),
        encoding="utf-8",
    )

    rules = load_structural_rules(str(rules_path))
    assert len(rules.event_trees) == 0


def test_load_structural_rules_keeps_non_conflicting_event_tree(tmp_path) -> None:
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(
        json.dumps(
            {
                "mutually_exclusive_buckets": [
                    {
                        "group_id": "bucket",
                        "payout_per_contract": 1.0,
                        "legs": [
                            {"venue": "kalshi", "market_id": "A", "side": "yes"},
                            {"venue": "kalshi", "market_id": "B", "side": "yes"},
                        ],
                    }
                ],
                "event_trees": [
                    {
                        "group_id": "valid-tree",
                        "parent": {"venue": "kalshi", "market_id": "PARENT", "side": "yes"},
                        "children": [
                            {"venue": "kalshi", "market_id": "CHILD1", "side": "yes"},
                            {"venue": "kalshi", "market_id": "CHILD2", "side": "yes"},
                        ],
                    }
                ],
                "cross_market_parity_checks": [],
            }
        ),
        encoding="utf-8",
    )

    rules = load_structural_rules(str(rules_path))
    assert len(rules.event_trees) == 1
    assert rules.event_trees[0].group_id == "valid-tree"


def test_load_structural_rules_dedupes_identical_bucket_leg_sets(tmp_path) -> None:
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(
        json.dumps(
            {
                "mutually_exclusive_buckets": [
                    {
                        "group_id": "bucket-a",
                        "payout_per_contract": 1.0,
                        "legs": [
                            {"venue": "kalshi", "market_id": "A", "side": "yes"},
                            {"venue": "polymarket", "market_id": "B", "side": "yes"},
                        ],
                    },
                    {
                        "group_id": "bucket-b",
                        "payout_per_contract": 1.0,
                        "legs": [
                            {"venue": "polymarket", "market_id": "B", "side": "yes"},
                            {"venue": "kalshi", "market_id": "A", "side": "yes"},
                        ],
                    },
                ],
                "event_trees": [],
                "cross_market_parity_checks": [],
            }
        ),
        encoding="utf-8",
    )

    rules = load_structural_rules(str(rules_path))
    assert len(rules.buckets) == 1
