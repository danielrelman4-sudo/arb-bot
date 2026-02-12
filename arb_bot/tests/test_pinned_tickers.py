"""Tests for Phase 8B: Parity ticker priority pinning.

Validates that structural rule tickers (parity, bucket, event-tree) are always
included in Kalshi REST refresh cycles regardless of cursor rotation, eliminating
the coverage oscillation where parity rules alternated between 10/12 and 0/12.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from typing import Any

import pytest

from arb_bot.config import (
    KalshiSettings,
    _dedupe_in_order,
    _derive_kalshi_stream_priority_tickers,
    _read_kalshi_tickers_from_structural_rules,
)
from arb_bot.exchanges.kalshi import KalshiAdapter


def _make_settings(
    priority_tickers: list[str] | None = None,
    pinned_tickers: list[str] | None = None,
    market_limit: int = 40,
) -> KalshiSettings:
    return KalshiSettings(
        stream_priority_tickers=priority_tickers or [],
        stream_pinned_tickers=pinned_tickers or [],
        market_limit=market_limit,
    )


def _make_adapter(settings: KalshiSettings) -> KalshiAdapter:
    """Create a KalshiAdapter without full init (skip HTTP client)."""
    adapter = KalshiAdapter.__new__(KalshiAdapter)
    adapter._settings = settings
    adapter._priority_refresh_cursor = 0
    return adapter


# ===================================================================
# Tests for _select_priority_refresh_tickers with pinning
# ===================================================================


class TestSelectPriorityRefreshWithPinning:
    """Tests for pinned tickers in the REST priority refresh rotation."""

    def test_pinned_tickers_always_included(self) -> None:
        """Pinned tickers should appear in every refresh regardless of cursor."""
        pinned = ["PARITY-A", "PARITY-B"]
        universe = [f"MKT-{i}" for i in range(100)]
        settings = _make_settings(priority_tickers=universe, pinned_tickers=pinned)
        adapter = _make_adapter(settings)

        # Run 5 refresh cycles; pinned tickers should appear every time.
        for cycle in range(5):
            selected = adapter._select_priority_refresh_tickers(universe, limit=10)
            assert "PARITY-A" in selected, f"PARITY-A missing in cycle {cycle}"
            assert "PARITY-B" in selected, f"PARITY-B missing in cycle {cycle}"
            assert len(selected) == 10

    def test_pinned_tickers_at_front(self) -> None:
        """Pinned tickers should be at the beginning of the selected list."""
        pinned = ["PIN-1", "PIN-2"]
        universe = [f"MKT-{i}" for i in range(50)]
        settings = _make_settings(priority_tickers=universe, pinned_tickers=pinned)
        adapter = _make_adapter(settings)

        selected = adapter._select_priority_refresh_tickers(universe, limit=10)
        assert selected[0] == "PIN-1"
        assert selected[1] == "PIN-2"

    def test_pinned_tickers_not_duplicated(self) -> None:
        """Pinned tickers that also appear in universe should not be duplicated."""
        pinned = ["MKT-0", "MKT-5"]
        universe = [f"MKT-{i}" for i in range(20)]
        settings = _make_settings(priority_tickers=universe, pinned_tickers=pinned)
        adapter = _make_adapter(settings)

        selected = adapter._select_priority_refresh_tickers(universe, limit=10)
        assert selected.count("MKT-0") == 1
        assert selected.count("MKT-5") == 1
        assert len(selected) == 10

    def test_cursor_advances_around_pinned(self) -> None:
        """Cursor should still rotate through non-pinned universe slots."""
        pinned = ["PIN-A"]
        universe = ["U-0", "U-1", "U-2", "U-3", "U-4"]
        settings = _make_settings(priority_tickers=universe, pinned_tickers=pinned)
        adapter = _make_adapter(settings)

        # With limit=3: 1 pinned + 2 rotating from universe.
        cycle1 = adapter._select_priority_refresh_tickers(universe, limit=3)
        assert cycle1[0] == "PIN-A"
        # Remaining 2 slots filled from universe starting at cursor=0
        rotating1 = [t for t in cycle1 if t != "PIN-A"]

        cycle2 = adapter._select_priority_refresh_tickers(universe, limit=3)
        assert cycle2[0] == "PIN-A"
        rotating2 = [t for t in cycle2 if t != "PIN-A"]

        # Rotating portions should be different (cursor advanced)
        assert rotating1 != rotating2 or len(universe) <= 3

    def test_no_pinned_tickers_falls_back_to_rotation(self) -> None:
        """With no pinned tickers, behaviour should be pure cursor rotation."""
        universe = [f"MKT-{i}" for i in range(10)]
        settings = _make_settings(priority_tickers=universe, pinned_tickers=[])
        adapter = _make_adapter(settings)

        selected = adapter._select_priority_refresh_tickers(universe, limit=5)
        assert len(selected) == 5
        # First cycle should start from cursor=0
        assert selected == universe[:5]

    def test_pinned_overflow_capped_by_limit(self) -> None:
        """If pinned tickers exceed limit, only first `limit` pinned are returned."""
        pinned = [f"PIN-{i}" for i in range(15)]
        universe = [f"MKT-{i}" for i in range(50)]
        settings = _make_settings(priority_tickers=universe, pinned_tickers=pinned)
        adapter = _make_adapter(settings)

        selected = adapter._select_priority_refresh_tickers(universe, limit=10)
        assert len(selected) == 10
        assert all(t.startswith("PIN-") for t in selected)

    def test_coverage_stability_across_many_cycles(self) -> None:
        """Pinned tickers appear in every one of 20 consecutive cycles."""
        pinned = ["PARITY-1", "PARITY-2", "PARITY-3"]
        universe = [f"MKT-{i}" for i in range(200)]
        settings = _make_settings(priority_tickers=universe, pinned_tickers=pinned)
        adapter = _make_adapter(settings)

        for cycle in range(20):
            selected = adapter._select_priority_refresh_tickers(universe, limit=20)
            for pin in pinned:
                assert pin in selected, f"{pin} missing in cycle {cycle}"


# ===================================================================
# Tests for _stream_subscription_tickers with pinning
# ===================================================================


class TestStreamSubscriptionWithPinning:
    """Tests for pinned tickers in WebSocket stream subscription ordering."""

    def test_pinned_tickers_first_in_subscription(self) -> None:
        """Pinned tickers should appear before priority tickers in subscription."""
        pinned = ["PIN-A", "PIN-B"]
        priority = [f"PRI-{i}" for i in range(10)]
        settings = _make_settings(
            priority_tickers=priority,
            pinned_tickers=pinned,
            market_limit=20,
        )
        adapter = _make_adapter(settings)

        ordered = adapter._stream_subscription_tickers({})
        assert ordered[0] == "PIN-A"
        assert ordered[1] == "PIN-B"
        # Priority tickers follow
        assert "PRI-0" in ordered

    def test_pinned_survive_market_limit_truncation(self) -> None:
        """Even with tight market_limit, pinned tickers survive because they're first."""
        pinned = ["PIN-A", "PIN-B", "PIN-C"]
        priority = [f"PRI-{i}" for i in range(100)]
        settings = _make_settings(
            priority_tickers=priority,
            pinned_tickers=pinned,
            market_limit=5,
        )
        adapter = _make_adapter(settings)

        ordered = adapter._stream_subscription_tickers({})
        assert len(ordered) == 5
        assert "PIN-A" in ordered
        assert "PIN-B" in ordered
        assert "PIN-C" in ordered

    def test_pinned_not_duplicated_with_priority(self) -> None:
        """If a ticker is both pinned and priority, it appears only once."""
        pinned = ["SHARED-1", "ONLY-PIN"]
        priority = ["SHARED-1", "ONLY-PRI"]
        settings = _make_settings(
            priority_tickers=priority,
            pinned_tickers=pinned,
            market_limit=10,
        )
        adapter = _make_adapter(settings)

        ordered = adapter._stream_subscription_tickers({})
        assert ordered.count("SHARED-1") == 1
        assert "ONLY-PIN" in ordered
        assert "ONLY-PRI" in ordered


# ===================================================================
# Tests for structural rules ticker extraction
# ===================================================================


class TestStructuralTickerExtraction:
    """Tests for _read_kalshi_tickers_from_structural_rules pinning source."""

    def test_extracts_parity_tickers(self, tmp_path: Any) -> None:
        """Parity rule left/right tickers should be extracted."""
        rules = {
            "mutually_exclusive_buckets": [],
            "event_trees": [],
            "cross_market_parity_checks": [
                {
                    "group_id": "test_pair",
                    "relationship": "equivalent",
                    "left": {"venue": "kalshi", "market_id": "PARITY-L"},
                    "right": {"venue": "kalshi", "market_id": "PARITY-R"},
                },
            ],
        }
        path = tmp_path / "rules.json"
        path.write_text(json.dumps(rules))

        tickers = _read_kalshi_tickers_from_structural_rules(str(path))
        assert "PARITY-L" in tickers
        assert "PARITY-R" in tickers

    def test_extracts_bucket_tickers(self, tmp_path: Any) -> None:
        rules = {
            "mutually_exclusive_buckets": [
                {
                    "group_id": "bucket1",
                    "legs": [
                        {"venue": "kalshi", "market_id": "BKT-A"},
                        {"venue": "kalshi", "market_id": "BKT-B"},
                        {"venue": "kalshi", "market_id": "BKT-C"},
                    ],
                },
            ],
            "event_trees": [],
            "cross_market_parity_checks": [],
        }
        path = tmp_path / "rules.json"
        path.write_text(json.dumps(rules))

        tickers = _read_kalshi_tickers_from_structural_rules(str(path))
        assert "BKT-A" in tickers
        assert "BKT-B" in tickers
        assert "BKT-C" in tickers

    def test_extracts_event_tree_tickers(self, tmp_path: Any) -> None:
        rules = {
            "mutually_exclusive_buckets": [],
            "event_trees": [
                {
                    "group_id": "tree1",
                    "parent": {"venue": "kalshi", "market_id": "PARENT"},
                    "children": [
                        {"venue": "kalshi", "market_id": "CHILD-1"},
                        {"venue": "kalshi", "market_id": "CHILD-2"},
                    ],
                },
            ],
            "cross_market_parity_checks": [],
        }
        path = tmp_path / "rules.json"
        path.write_text(json.dumps(rules))

        tickers = _read_kalshi_tickers_from_structural_rules(str(path))
        assert "PARENT" in tickers
        assert "CHILD-1" in tickers
        assert "CHILD-2" in tickers

    def test_ignores_non_kalshi_venues(self, tmp_path: Any) -> None:
        rules = {
            "mutually_exclusive_buckets": [],
            "event_trees": [],
            "cross_market_parity_checks": [
                {
                    "group_id": "cross_pair",
                    "left": {"venue": "kalshi", "market_id": "K-TICKER"},
                    "right": {"venue": "polymarket", "market_id": "PM-TICKER"},
                },
            ],
        }
        path = tmp_path / "rules.json"
        path.write_text(json.dumps(rules))

        tickers = _read_kalshi_tickers_from_structural_rules(str(path))
        assert "K-TICKER" in tickers
        assert "PM-TICKER" not in tickers

    def test_missing_file_returns_empty(self) -> None:
        tickers = _read_kalshi_tickers_from_structural_rules("/nonexistent/path.json")
        assert tickers == []

    def test_none_path_returns_empty(self) -> None:
        tickers = _read_kalshi_tickers_from_structural_rules(None)
        assert tickers == []


# ===================================================================
# Tests for config pinning integration
# ===================================================================


class TestConfigPinning:
    """Tests for pinned ticker config integration."""

    def test_kalshi_settings_has_pinned_tickers(self) -> None:
        settings = KalshiSettings(stream_pinned_tickers=["A", "B"])
        assert settings.stream_pinned_tickers == ["A", "B"]

    def test_kalshi_settings_default_empty_pinned(self) -> None:
        settings = KalshiSettings()
        assert settings.stream_pinned_tickers == []

    def test_dedupe_in_order_preserves_first_occurrence(self) -> None:
        result = _dedupe_in_order(["A", "B", "A", "C", "B"])
        assert result == ["A", "B", "C"]
