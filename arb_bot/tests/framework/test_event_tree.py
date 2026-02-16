"""Tests for event_tree module (Phase 6B)."""

from __future__ import annotations

import pytest

from arb_bot.framework.event_tree import (
    BasketOpportunity,
    ChildOutcome,
    EventNode,
    EventTree,
    EventTreeConfig,
    TreeCoverageReport,
    TreeValidation,
    ValidationStatus,
)


def _tree(**kw: object) -> EventTree:
    return EventTree(EventTreeConfig(**kw))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self) -> None:
        cfg = EventTreeConfig()
        assert cfg.sum_tolerance == 0.05
        assert cfg.expected_sum == 1.0
        assert cfg.min_children == 2
        assert cfg.price_stale_seconds == 300.0
        assert cfg.max_events == 5000

    def test_frozen(self) -> None:
        cfg = EventTreeConfig()
        with pytest.raises(AttributeError):
            cfg.sum_tolerance = 0.1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Add event
# ---------------------------------------------------------------------------


class TestAddEvent:
    def test_basic(self) -> None:
        t = _tree()
        t.add_event("election", children=["dem", "rep", "other"], now=100.0)
        assert t.event_count() == 1
        node = t.get_event("election")
        assert node is not None
        assert len(node.children) == 3

    def test_custom_payouts(self) -> None:
        t = _tree()
        t.add_event("e1", children=["a", "b"], payouts={"a": 2.0, "b": 0.5}, now=1.0)
        node = t.get_event("e1")
        assert node is not None
        assert node.children["a"].payout == 2.0
        assert node.children["b"].payout == 0.5

    def test_default_payout_is_1(self) -> None:
        t = _tree()
        t.add_event("e1", children=["a", "b"], now=1.0)
        node = t.get_event("e1")
        assert node is not None
        assert node.children["a"].payout == 1.0

    def test_metadata(self) -> None:
        t = _tree()
        t.add_event("e1", children=["a", "b"], metadata={"category": "politics"}, now=1.0)
        node = t.get_event("e1")
        assert node is not None
        assert node.metadata["category"] == "politics"

    def test_overwrite_event(self) -> None:
        t = _tree()
        t.add_event("e1", children=["a", "b"], now=1.0)
        t.add_event("e1", children=["x", "y", "z"], now=2.0)
        assert t.event_count() == 1
        node = t.get_event("e1")
        assert node is not None
        assert len(node.children) == 3
        assert "x" in node.children


# ---------------------------------------------------------------------------
# Set price
# ---------------------------------------------------------------------------


class TestSetPrice:
    def test_set_price(self) -> None:
        t = _tree()
        t.add_event("e1", children=["a", "b"], now=100.0)
        ok = t.set_price("e1", "a", 0.55, now=101.0)
        assert ok is True
        node = t.get_event("e1")
        assert node is not None
        assert node.children["a"].price == 0.55
        assert node.children["a"].price_updated_at == 101.0

    def test_set_price_nonexistent_event(self) -> None:
        t = _tree()
        assert t.set_price("missing", "a", 0.5, now=1.0) is False

    def test_set_price_nonexistent_outcome(self) -> None:
        t = _tree()
        t.add_event("e1", children=["a"], now=1.0)
        assert t.set_price("e1", "missing", 0.5, now=1.0) is False

    def test_set_price_with_venue(self) -> None:
        t = _tree()
        t.add_event("e1", children=["a"], now=1.0)
        t.set_price("e1", "a", 0.5, venue="kalshi", now=2.0)
        node = t.get_event("e1")
        assert node is not None
        assert node.children["a"].venue == "kalshi"


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------


class TestValidate:
    def test_valid_sum(self) -> None:
        t = _tree()
        t.add_event("e1", children=["a", "b", "c"], now=100.0)
        t.set_price("e1", "a", 0.50, now=100.0)
        t.set_price("e1", "b", 0.30, now=100.0)
        t.set_price("e1", "c", 0.20, now=100.0)
        v = t.validate("e1", now=100.0)
        assert v.status == ValidationStatus.VALID
        assert abs(v.price_sum - 1.0) < 0.001
        assert v.child_count == 3

    def test_valid_within_tolerance(self) -> None:
        t = _tree(sum_tolerance=0.05)
        t.add_event("e1", children=["a", "b"], now=100.0)
        t.set_price("e1", "a", 0.52, now=100.0)
        t.set_price("e1", "b", 0.50, now=100.0)
        v = t.validate("e1", now=100.0)
        assert v.status == ValidationStatus.VALID  # Sum=1.02, deviation=0.02 < 0.05.

    def test_sum_violation(self) -> None:
        t = _tree(sum_tolerance=0.05)
        t.add_event("e1", children=["a", "b"], now=100.0)
        t.set_price("e1", "a", 0.60, now=100.0)
        t.set_price("e1", "b", 0.50, now=100.0)
        v = t.validate("e1", now=100.0)
        assert v.status == ValidationStatus.SUM_VIOLATION
        assert abs(v.price_sum - 1.10) < 0.001
        assert v.deviation > 0.05

    def test_missing_prices(self) -> None:
        t = _tree()
        t.add_event("e1", children=["a", "b", "c"], now=100.0)
        t.set_price("e1", "a", 0.50, now=100.0)
        # b and c have no prices.
        v = t.validate("e1", now=100.0)
        assert v.status == ValidationStatus.MISSING_PRICES
        assert set(v.missing_prices) == {"b", "c"}

    def test_stale_prices(self) -> None:
        t = _tree(price_stale_seconds=60.0)
        t.add_event("e1", children=["a", "b"], now=100.0)
        t.set_price("e1", "a", 0.50, now=100.0)
        t.set_price("e1", "b", 0.50, now=100.0)
        v = t.validate("e1", now=200.0)  # 100s > 60s stale.
        assert v.status == ValidationStatus.STALE_PRICES
        assert len(v.stale_prices) == 2

    def test_not_found(self) -> None:
        t = _tree()
        v = t.validate("missing", now=100.0)
        assert v.status == ValidationStatus.NOT_FOUND

    def test_no_children(self) -> None:
        t = _tree(min_children=2)
        t.add_event("e1", children=["a"], now=100.0)
        v = t.validate("e1", now=100.0)
        assert v.status == ValidationStatus.NO_CHILDREN

    def test_zero_price_without_update_is_missing(self) -> None:
        """A child with no set_price() call is missing, regardless of price value."""
        t = _tree()
        t.add_event("e1", children=["a", "b"], now=100.0)
        t.set_price("e1", "a", 0.5, now=100.0)
        v = t.validate("e1", now=100.0)
        assert v.status == ValidationStatus.MISSING_PRICES

    def test_explicit_zero_price_is_not_missing(self) -> None:
        """A legitimate 0.0 price that was explicitly set is NOT missing.

        This is the critical fix — the old sentinel-check approach would
        have falsely flagged a 0.0 price as missing. The new price_set
        flag ensures we only flag outcomes where set_price() was never
        called.
        """
        t = _tree(sum_tolerance=1.0)  # Loose tolerance for this test.
        t.add_event("e1", children=["a", "b"], now=100.0)
        t.set_price("e1", "a", 0.0, now=100.0)   # Legitimate zero.
        t.set_price("e1", "b", 1.0, now=100.0)
        v = t.validate("e1", now=100.0)
        # Should NOT be MISSING_PRICES — both prices were explicitly set.
        assert v.status == ValidationStatus.VALID
        assert v.missing_prices == ()
        assert abs(v.price_sum - 1.0) < 0.001


# ---------------------------------------------------------------------------
# Find opportunities
# ---------------------------------------------------------------------------


class TestFindOpportunities:
    def test_overpriced_basket(self) -> None:
        t = _tree(sum_tolerance=0.05)
        t.add_event("e1", children=["a", "b"], now=100.0)
        t.set_price("e1", "a", 0.60, now=100.0)
        t.set_price("e1", "b", 0.50, now=100.0)
        opps = t.find_opportunities(min_edge=0.05, now=100.0)
        assert len(opps) == 1
        assert opps[0].event_id == "e1"
        assert opps[0].edge > 0  # Overpriced.

    def test_underpriced_basket(self) -> None:
        t = _tree(sum_tolerance=0.05)
        t.add_event("e1", children=["a", "b"], now=100.0)
        t.set_price("e1", "a", 0.40, now=100.0)
        t.set_price("e1", "b", 0.50, now=100.0)
        opps = t.find_opportunities(min_edge=0.05, now=100.0)
        assert len(opps) == 1
        assert opps[0].edge < 0  # Underpriced.

    def test_no_opportunity_within_tolerance(self) -> None:
        t = _tree(sum_tolerance=0.05)
        t.add_event("e1", children=["a", "b"], now=100.0)
        t.set_price("e1", "a", 0.51, now=100.0)
        t.set_price("e1", "b", 0.50, now=100.0)
        opps = t.find_opportunities(min_edge=0.05, now=100.0)
        assert len(opps) == 0  # Edge=0.01 < min_edge=0.05.

    def test_multiple_opportunities_sorted(self) -> None:
        t = _tree(sum_tolerance=0.01)
        t.add_event("e1", children=["a", "b"], now=100.0)
        t.set_price("e1", "a", 0.60, now=100.0)
        t.set_price("e1", "b", 0.50, now=100.0)  # edge=0.10

        t.add_event("e2", children=["x", "y"], now=100.0)
        t.set_price("e2", "x", 0.55, now=100.0)
        t.set_price("e2", "y", 0.50, now=100.0)  # edge=0.05

        opps = t.find_opportunities(min_edge=0.02, now=100.0)
        assert len(opps) == 2
        assert abs(opps[0].edge) >= abs(opps[1].edge)  # Sorted by |edge|.

    def test_skips_missing_prices(self) -> None:
        t = _tree()
        t.add_event("e1", children=["a", "b"], now=100.0)
        t.set_price("e1", "a", 0.60, now=100.0)
        # b has no price → missing.
        opps = t.find_opportunities(min_edge=0.01, now=100.0)
        assert len(opps) == 0

    def test_opportunity_child_prices(self) -> None:
        t = _tree(sum_tolerance=0.01)
        t.add_event("e1", children=["a", "b"], now=100.0)
        t.set_price("e1", "a", 0.60, now=100.0)
        t.set_price("e1", "b", 0.50, now=100.0)
        opps = t.find_opportunities(min_edge=0.05, now=100.0)
        assert opps[0].child_prices == {"a": 0.60, "b": 0.50}


# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------


class TestCoverageReport:
    def test_empty(self) -> None:
        t = _tree()
        r = t.coverage_report(now=100.0)
        assert r.total_events == 0
        assert r.valid_count == 0

    def test_mixed_statuses(self) -> None:
        t = _tree(sum_tolerance=0.05, price_stale_seconds=60.0)

        # Valid event.
        t.add_event("valid", children=["a", "b"], now=100.0)
        t.set_price("valid", "a", 0.50, now=100.0)
        t.set_price("valid", "b", 0.50, now=100.0)

        # Missing prices.
        t.add_event("missing", children=["x", "y"], now=100.0)
        t.set_price("missing", "x", 0.50, now=100.0)

        # Violation.
        t.add_event("violation", children=["m", "n"], now=100.0)
        t.set_price("violation", "m", 0.70, now=100.0)
        t.set_price("violation", "n", 0.50, now=100.0)

        r = t.coverage_report(now=100.0)
        assert r.total_events == 3
        assert r.valid_count == 1
        assert r.missing_count == 1
        assert r.violation_count == 1


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


class TestMisc:
    def test_event_ids(self) -> None:
        t = _tree()
        t.add_event("a", children=["x", "y"], now=1.0)
        t.add_event("b", children=["x", "y"], now=1.0)
        assert set(t.event_ids()) == {"a", "b"}

    def test_remove_event(self) -> None:
        t = _tree()
        t.add_event("a", children=["x", "y"], now=1.0)
        t.remove_event("a")
        assert t.event_count() == 0

    def test_clear(self) -> None:
        t = _tree()
        t.add_event("a", children=["x", "y"], now=1.0)
        t.add_event("b", children=["x", "y"], now=1.0)
        t.clear()
        assert t.event_count() == 0

    def test_set_payout(self) -> None:
        t = _tree()
        t.add_event("e1", children=["a"], now=1.0)
        ok = t.set_payout("e1", "a", 2.5)
        assert ok is True
        node = t.get_event("e1")
        assert node is not None
        assert node.children["a"].payout == 2.5

    def test_set_payout_nonexistent(self) -> None:
        t = _tree()
        assert t.set_payout("missing", "a", 1.0) is False


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_election_basket(self) -> None:
        """Full election event tree lifecycle."""
        t = _tree(sum_tolerance=0.03)
        t.add_event(
            "2024_election",
            children=["dem_win", "rep_win", "other"],
            metadata={"category": "politics", "year": 2024},
            now=100.0,
        )

        # Set prices.
        t.set_price("2024_election", "dem_win", 0.55, venue="kalshi", now=100.0)
        t.set_price("2024_election", "rep_win", 0.40, venue="kalshi", now=100.0)
        t.set_price("2024_election", "other", 0.06, venue="kalshi", now=100.0)

        # Validate — sum=1.01, within 0.03 tolerance.
        v = t.validate("2024_election", now=100.0)
        assert v.status == ValidationStatus.VALID
        assert v.child_count == 3
        assert abs(v.price_sum - 1.01) < 0.001

        # Now prices move to create opportunity.
        t.set_price("2024_election", "dem_win", 0.58, now=101.0)
        t.set_price("2024_election", "rep_win", 0.42, now=101.0)
        t.set_price("2024_election", "other", 0.06, now=101.0)

        # Sum=1.06, deviation=0.06 > 0.03 → violation + opportunity.
        v = t.validate("2024_election", now=101.0)
        assert v.status == ValidationStatus.SUM_VIOLATION

        opps = t.find_opportunities(min_edge=0.03, now=101.0)
        assert len(opps) == 1
        assert opps[0].event_id == "2024_election"
        assert opps[0].edge > 0  # Overpriced basket.

        # Coverage report.
        r = t.coverage_report(now=101.0)
        assert r.total_events == 1
        assert r.violation_count == 1
        assert len(r.opportunities) == 1
