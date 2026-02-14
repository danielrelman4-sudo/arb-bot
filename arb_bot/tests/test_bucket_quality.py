import csv
import json
from pathlib import Path

from arb_bot.bucket_quality import BucketQualityModel
from arb_bot.models import BinaryQuote, OpportunityKind
from arb_bot.strategy import ArbitrageFinder


def _write_bucket_history(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "kind",
        "match_key",
        "action",
        "detected_edge_per_contract",
        "fill_probability",
        "expected_realized_profit",
        "realized_profit",
        "expected_slippage_per_contract",
        "execution_latency_ms",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_bucket_quality_model_prefers_high_signal_and_reserves_exploration(tmp_path: Path) -> None:
    history_path = tmp_path / "paper_history.csv"
    rows: list[dict[str, object]] = []

    for _ in range(10):
        rows.append(
            {
                "kind": "structural_bucket",
                "match_key": "B_GOOD",
                "action": "dry_run",
                "detected_edge_per_contract": 0.015,
                "fill_probability": 0.9,
                "expected_realized_profit": 0.03,
                "realized_profit": "",
                "expected_slippage_per_contract": 0.001,
                "execution_latency_ms": 150,
            }
        )
    for _ in range(10):
        rows.append(
            {
                "kind": "structural_bucket",
                "match_key": "B_BAD",
                "action": "dry_run",
                "detected_edge_per_contract": 0.004,
                "fill_probability": 0.6,
                "expected_realized_profit": -0.02,
                "realized_profit": "",
                "expected_slippage_per_contract": 0.01,
                "execution_latency_ms": 500,
            }
        )

    _write_bucket_history(history_path, rows)

    model = BucketQualityModel(
        bucket_leg_counts={"B_GOOD": 3, "B_BAD": 3, "B_NEW": 3},
        enabled=True,
        history_glob=str(history_path),
        history_max_files=1,
        min_observations=5,
        max_active_buckets=2,
        explore_fraction=0.5,
        min_score=0.0,
        live_update_interval=1000,
    )

    active = model.active_bucket_ids
    assert "B_GOOD" in active
    assert "B_BAD" not in active
    assert "B_NEW" in active


def test_strategy_filters_structural_buckets_via_quality_model(tmp_path: Path) -> None:
    rules_path = tmp_path / "structural_rules.json"
    rules_path.write_text(
        json.dumps(
            {
                "mutually_exclusive_buckets": [
                    {
                        "group_id": "bucket_good",
                        "payout_per_contract": 1.0,
                        "legs": [
                            {"venue": "kalshi", "market_id": "G1", "side": "yes"},
                            {"venue": "kalshi", "market_id": "G2", "side": "yes"},
                        ],
                    },
                    {
                        "group_id": "bucket_bad",
                        "payout_per_contract": 1.0,
                        "legs": [
                            {"venue": "kalshi", "market_id": "B1", "side": "yes"},
                            {"venue": "kalshi", "market_id": "B2", "side": "yes"},
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    history_path = tmp_path / "paper_history.csv"
    rows = [
        {
            "kind": "structural_bucket",
            "match_key": "bucket_good",
            "action": "dry_run",
            "detected_edge_per_contract": 0.02,
            "fill_probability": 0.9,
            "expected_realized_profit": 0.04,
            "realized_profit": "",
            "expected_slippage_per_contract": 0.001,
            "execution_latency_ms": 120,
        },
        {
            "kind": "structural_bucket",
            "match_key": "bucket_bad",
            "action": "dry_run",
            "detected_edge_per_contract": 0.005,
            "fill_probability": 0.5,
            "expected_realized_profit": -0.02,
            "realized_profit": "",
            "expected_slippage_per_contract": 0.02,
            "execution_latency_ms": 800,
        },
    ]
    _write_bucket_history(history_path, rows)

    finder = ArbitrageFinder(
        min_net_edge_per_contract=0.001,
        enable_cross_venue=False,
        enable_structural_arb=True,
        structural_rules_path=str(rules_path),
        enable_maker_estimates=False,
        enable_bucket_quality_model=True,
        bucket_quality_history_glob=str(history_path),
        bucket_quality_history_max_files=1,
        bucket_quality_min_observations=1,
        bucket_quality_max_active_buckets=1,
        bucket_quality_explore_fraction=0.0,
        bucket_quality_min_score=-1.0,
        bucket_quality_live_update_interval=1000,
        bucket_quality_use_thompson_sampling=False,  # deterministic for test
    )

    quotes = [
        BinaryQuote("kalshi", "G1", 0.49, 0.51, 100, 100, fee_per_contract=0.0),
        BinaryQuote("kalshi", "G2", 0.49, 0.51, 100, 100, fee_per_contract=0.0),
        BinaryQuote("kalshi", "B1", 0.49, 0.51, 100, 100, fee_per_contract=0.0),
        BinaryQuote("kalshi", "B2", 0.49, 0.51, 100, 100, fee_per_contract=0.0),
    ]
    opportunities = finder.find(quotes)
    bucket_opps = [opp for opp in opportunities if opp.kind is OpportunityKind.STRUCTURAL_BUCKET]

    assert bucket_opps
    assert all(opp.match_key == "bucket_good" for opp in bucket_opps)


def test_bucket_quality_without_history_keeps_full_universe_active() -> None:
    model = BucketQualityModel(
        bucket_leg_counts={"B1": 3, "B2": 4, "B3": 5},
        enabled=True,
        history_glob="",
        history_max_files=1,
        min_observations=5,
        max_active_buckets=1,
        explore_fraction=0.5,
        min_score=0.0,
        live_update_interval=1000,
    )
    assert model.active_bucket_ids == {"B1", "B2", "B3"}


# ── consecutive failure limit tests ──────────────────────────────


def _observe(model: BucketQualityModel, group_id: str, action: str) -> None:
    """Helper to record an observation with minimal required fields."""
    model.observe_decision(
        group_id=group_id,
        action=action,
        detected_edge_per_contract=0.02,
        fill_probability=0.8,
        expected_realized_profit=0.01,
        realized_profit=0.01 if action == "filled" else None,
        expected_slippage_per_contract=0.001,
        execution_latency_ms=100.0,
    )


def test_consecutive_failures_disable_bucket() -> None:
    """After max_consecutive_bucket_failures execution failures, bucket is disabled."""
    model = BucketQualityModel(
        bucket_leg_counts={"B1": 3, "B2": 3},
        enabled=True,
        history_glob="",
        max_consecutive_bucket_failures=3,
        live_update_interval=10000,
    )
    assert model.should_enable_bucket("B1") is True

    _observe(model, "B1", "execution_failed")
    _observe(model, "B1", "execution_failed")
    assert model.should_enable_bucket("B1") is True  # 2 < 3

    _observe(model, "B1", "execution_failed")
    assert model.should_enable_bucket("B1") is False  # 3 >= 3

    # B2 should be unaffected
    assert model.should_enable_bucket("B2") is True


def test_consecutive_failures_reset_on_fill() -> None:
    """A successful fill resets the consecutive failure counter."""
    model = BucketQualityModel(
        bucket_leg_counts={"B1": 3},
        enabled=True,
        history_glob="",
        max_consecutive_bucket_failures=3,
        live_update_interval=10000,
    )
    _observe(model, "B1", "execution_failed")
    _observe(model, "B1", "execution_failed")
    assert model.should_enable_bucket("B1") is True  # 2 < 3

    _observe(model, "B1", "filled")
    assert model.should_enable_bucket("B1") is True  # reset to 0

    # Can fail again up to limit
    _observe(model, "B1", "execution_failed")
    _observe(model, "B1", "execution_failed")
    _observe(model, "B1", "execution_failed")
    assert model.should_enable_bucket("B1") is False


def test_consecutive_failures_skips_dont_count() -> None:
    """Skipped opportunities should not increment the failure counter."""
    model = BucketQualityModel(
        bucket_leg_counts={"B1": 3},
        enabled=True,
        history_glob="",
        max_consecutive_bucket_failures=2,
        live_update_interval=10000,
    )
    _observe(model, "B1", "execution_failed")
    _observe(model, "B1", "skipped")  # should not increment
    assert model.should_enable_bucket("B1") is True  # still at 1

    _observe(model, "B1", "execution_failed")
    assert model.should_enable_bucket("B1") is False  # now at 2


def test_consecutive_failures_zero_means_unlimited() -> None:
    """max_consecutive_bucket_failures=0 means no limit."""
    model = BucketQualityModel(
        bucket_leg_counts={"B1": 3},
        enabled=True,
        history_glob="",
        max_consecutive_bucket_failures=0,
        live_update_interval=10000,
    )
    for _ in range(20):
        _observe(model, "B1", "execution_failed")
    assert model.should_enable_bucket("B1") is True


def test_consecutive_failures_excluded_from_thompson_select() -> None:
    """Failure-limited buckets should not appear in Thompson sampling active set."""
    model = BucketQualityModel(
        bucket_leg_counts={"B1": 3, "B2": 3},
        enabled=True,
        history_glob="",
        max_consecutive_bucket_failures=2,
        max_active_buckets=0,  # unlimited capacity
        live_update_interval=10000,
        use_thompson_sampling=True,
    )
    # Give B1 some history so it has an accumulator
    _observe(model, "B1", "execution_failed")
    _observe(model, "B1", "execution_failed")

    # Force recompute
    model._recompute_active_set()

    # B1 should not be in active set
    assert "B1" not in model.active_bucket_ids
    assert "B2" in model.active_bucket_ids


def test_consecutive_failures_excluded_from_deterministic_select() -> None:
    """Failure-limited buckets should not appear in deterministic selection."""
    model = BucketQualityModel(
        bucket_leg_counts={"B1": 3, "B2": 3},
        enabled=True,
        history_glob="",
        max_consecutive_bucket_failures=2,
        max_active_buckets=0,
        live_update_interval=10000,
        use_thompson_sampling=False,
    )
    _observe(model, "B1", "execution_failed")
    _observe(model, "B1", "execution_failed")

    model._recompute_active_set()

    assert "B1" not in model.active_bucket_ids
    assert "B2" in model.active_bucket_ids


def test_consecutive_failures_settled_resets_counter() -> None:
    """A settled action (positive outcome) should also reset failures."""
    model = BucketQualityModel(
        bucket_leg_counts={"B1": 3},
        enabled=True,
        history_glob="",
        max_consecutive_bucket_failures=2,
        live_update_interval=10000,
    )
    _observe(model, "B1", "execution_failed")

    # Settled also resets (same as filled — successful trade)
    model.observe_decision(
        group_id="B1",
        action="settled",
        detected_edge_per_contract=0.02,
        fill_probability=0.8,
        expected_realized_profit=0.01,
        realized_profit=0.01,
        expected_slippage_per_contract=0.001,
        execution_latency_ms=100.0,
    )
    assert model.should_enable_bucket("B1") is True

    _observe(model, "B1", "execution_failed")
    assert model.should_enable_bucket("B1") is True  # only 1 failure since reset
