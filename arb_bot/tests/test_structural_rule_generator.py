from __future__ import annotations

import asyncio

from arb_bot.structural_rule_generator import (
    GenerationSettings,
    MarketSnapshot,
    _classify_bucket_exclusivity,
    _looks_like_numeric_thresholds,
    _looks_like_temporal_suffixes,
    build_generation_diagnostics,
    fetch_kalshi_markets_all_events,
    fetch_polymarket_markets_all_events,
    fetch_polymarket_markets_for_events,
    generate_parity_rules_from_cross_mapping_rows,
    generate_structural_rules_payload,
    merge_with_existing_rules,
    snapshot_from_market,
    validate_structural_rules_payload,
)


def _kalshi_market(event_ticker: str, suffix: str, outcome: str, *, liquidity: float = 100.0) -> dict:
    return {
        "venue": "kalshi",
        "event_ticker": event_ticker,
        "ticker": f"{event_ticker}-{suffix}",
        "title": "Who will win the election?",
        "subtitle": outcome,
        "status": "open",
        "liquidity": liquidity,
    }


def _polymarket_market(event_slug: str, market_id: str, outcome: str, *, liquidity: float = 100.0) -> dict:
    return {
        "venue": "polymarket",
        "event_slug": event_slug,
        "event_title": "Who will win the election?",
        "id": market_id,
        "title": f"{outcome} wins",
        "outcome": outcome,
        "active": True,
        "closed": False,
        "liquidity": liquidity,
    }


def test_snapshot_from_market_supports_polymarket() -> None:
    snapshot = snapshot_from_market(
        {
            "venue": "polymarket",
            "event_slug": "dem-nom-2028",
            "event_title": "Who will be nominee?",
            "id": "0xabc",
            "title": "Alice",
            "outcome": "Alice",
            "active": True,
            "closed": False,
        }
    )
    assert snapshot is not None
    assert snapshot.venue == "polymarket"
    assert snapshot.event_key == "dem-nom-2028"
    assert snapshot.market_id == "0xabc"


def test_generate_structural_rules_for_kalshi_and_polymarket() -> None:
    markets = [
        _kalshi_market("KXEVENT", "A", "Alice", liquidity=100.0),
        _kalshi_market("KXEVENT", "B", "Bob", liquidity=95.0),
        _kalshi_market("KXEVENT", "C", "Carol", liquidity=90.0),
        _polymarket_market("pm-event", "0x1", "Alice", liquidity=100.0),
        _polymarket_market("pm-event", "0x2", "Bob", liquidity=95.0),
        _polymarket_market("pm-event", "0x3", "Carol", liquidity=90.0),
    ]

    payload = generate_structural_rules_payload(
        markets,
        settings=GenerationSettings(
            min_bucket_size=3,
            min_shared_title_ratio=0.8,
            min_event_coverage_ratio=0.8,
            create_event_trees=True,
        ),
    )

    buckets = payload["mutually_exclusive_buckets"]
    trees = payload["event_trees"]

    assert len(buckets) == 2
    assert len(trees) == 0

    bucket_venues = {bucket["legs"][0]["venue"] for bucket in buckets}
    assert bucket_venues == {"kalshi", "polymarket"}

    assert trees == []


def test_validate_flags_low_event_coverage() -> None:
    markets = [
        _kalshi_market("KXEVENT", "A", "Alice"),
        _kalshi_market("KXEVENT", "B", "Bob"),
        _kalshi_market("KXEVENT", "C", "Carol"),
        _kalshi_market("KXEVENT", "D", "Dave"),
        _kalshi_market("KXEVENT", "E", "Eve"),
    ]
    payload = {
        "mutually_exclusive_buckets": [
            {
                "group_id": "too_small_bucket",
                "payout_per_contract": 1.0,
                "legs": [
                    {"venue": "kalshi", "market_id": "KXEVENT-A", "side": "yes"},
                    {"venue": "kalshi", "market_id": "KXEVENT-B", "side": "yes"},
                ],
            }
        ],
        "event_trees": [],
        "cross_market_parity_checks": [],
    }

    issues = validate_structural_rules_payload(payload, markets=markets, min_event_coverage_ratio=0.8)
    assert any("low event coverage" in issue for issue in issues)


def test_merge_preserves_existing_parity_rules() -> None:
    existing = {
        "mutually_exclusive_buckets": [],
        "event_trees": [],
        "cross_market_parity_checks": [
            {
                "group_id": "eq_1",
                "relationship": "equivalent",
                "left": {"venue": "kalshi", "market_id": "A", "side": "yes"},
                "right": {"venue": "polymarket", "market_id": "B", "side": "yes"},
            }
        ],
    }
    generated = {
        "mutually_exclusive_buckets": [
            {
                "group_id": "kalshi_event_exclusive",
                "payout_per_contract": 1.0,
                "legs": [
                    {"venue": "kalshi", "market_id": "A", "side": "yes"},
                    {"venue": "kalshi", "market_id": "C", "side": "yes"},
                ],
            }
        ],
        "event_trees": [],
        "cross_market_parity_checks": [],
    }

    merged = merge_with_existing_rules(existing_payload=existing, generated_payload=generated)
    assert len(merged["cross_market_parity_checks"]) == 1
    assert merged["cross_market_parity_checks"][0]["group_id"] == "eq_1"


def test_generate_parity_rules_from_cross_mapping_rows() -> None:
    rows = [
        {
            "group_id": "fed-cut-sept",
            "kalshi_market_id": "KXFED-YES",
            "polymarket_market_id": "0xabc",
        },
        {
            "group_id": "fed-cut-sept",
            "kalshi_market_id": "KXFED-YES",
            "polymarket_market_id": "0xabc",
        },
        {
            "group_id": "missing-polymarket",
            "kalshi_market_id": "KXFED-NO",
            "polymarket_market_id": "",
        },
    ]

    parity, skip_reasons = generate_parity_rules_from_cross_mapping_rows(rows, relationship="equivalent")
    assert len(parity) == 1
    assert parity[0]["group_id"] == "fed_cut_sept_parity"
    assert parity[0]["left"]["market_id"] == "KXFED-YES"
    assert parity[0]["right"]["market_id"] == "0xabc"
    assert skip_reasons["duplicate_market_pair"] == 1
    assert skip_reasons["missing_direct_market_ids"] == 1


def test_merge_includes_generated_parity_rules() -> None:
    existing = {
        "mutually_exclusive_buckets": [],
        "event_trees": [],
        "cross_market_parity_checks": [
            {
                "group_id": "eq_1",
                "relationship": "equivalent",
                "left": {"venue": "kalshi", "market_id": "A", "side": "yes"},
                "right": {"venue": "polymarket", "market_id": "B", "side": "yes"},
            }
        ],
    }
    generated = {
        "mutually_exclusive_buckets": [],
        "event_trees": [],
        "cross_market_parity_checks": [
            {
                "group_id": "eq_2",
                "relationship": "equivalent",
                "left": {"venue": "kalshi", "market_id": "C", "side": "yes"},
                "right": {"venue": "polymarket", "market_id": "D", "side": "yes"},
            }
        ],
    }

    merged = merge_with_existing_rules(existing_payload=existing, generated_payload=generated)
    assert len(merged["cross_market_parity_checks"]) == 2
    assert {item["group_id"] for item in merged["cross_market_parity_checks"]} == {"eq_1", "eq_2"}


def test_merge_generated_parity_overrides_existing_group_id() -> None:
    existing = {
        "mutually_exclusive_buckets": [],
        "event_trees": [],
        "cross_market_parity_checks": [
            {
                "group_id": "eq_1",
                "relationship": "equivalent",
                "left": {"venue": "kalshi", "market_id": "A", "side": "yes"},
                "right": {"venue": "polymarket", "market_id": "B", "side": "yes"},
            }
        ],
    }
    generated = {
        "mutually_exclusive_buckets": [],
        "event_trees": [],
        "cross_market_parity_checks": [
            {
                "group_id": "eq_1",
                "relationship": "complement",
                "left": {"venue": "kalshi", "market_id": "A", "side": "yes"},
                "right": {"venue": "polymarket", "market_id": "B", "side": "yes"},
            }
        ],
    }

    merged = merge_with_existing_rules(existing_payload=existing, generated_payload=generated)
    assert len(merged["cross_market_parity_checks"]) == 1
    assert merged["cross_market_parity_checks"][0]["group_id"] == "eq_1"
    assert merged["cross_market_parity_checks"][0]["relationship"] == "complement"


def test_generate_falls_back_to_all_markets_when_titles_do_not_cluster() -> None:
    markets = [
        _kalshi_market("KXEVENT", "A", "Alice"),
        _kalshi_market("KXEVENT", "B", "Bob"),
        _kalshi_market("KXEVENT", "C", "Carol"),
    ]
    markets[0]["title"] = "Will Alice win?"
    markets[1]["title"] = "Will Bob win?"
    markets[2]["title"] = "Will Carol win?"

    payload = generate_structural_rules_payload(
        markets,
        settings=GenerationSettings(
            min_bucket_size=3,
            min_shared_title_ratio=0.8,
            min_event_coverage_ratio=0.8,
            create_event_trees=True,
        ),
    )
    assert len(payload["mutually_exclusive_buckets"]) == 1
    assert len(payload["event_trees"]) == 0


def test_generate_event_tree_from_explicit_parent_child_links() -> None:
    markets = [
        _kalshi_market("KXEVENT", "PARENT", "Parent", liquidity=150.0),
        _kalshi_market("KXEVENT", "CHILD_A", "Child A", liquidity=100.0),
        _kalshi_market("KXEVENT", "CHILD_B", "Child B", liquidity=95.0),
    ]
    markets[0]["child_market_ids"] = ["KXEVENT-CHILD_A", "KXEVENT-CHILD_B"]
    markets[1]["parent_market_id"] = "KXEVENT-PARENT"
    markets[2]["parent_market_id"] = "KXEVENT-PARENT"

    payload = generate_structural_rules_payload(
        markets,
        settings=GenerationSettings(
            min_bucket_size=3,
            min_shared_title_ratio=0.8,
            min_event_coverage_ratio=0.8,
            create_event_trees=True,
        ),
    )

    trees = payload["event_trees"]
    assert len(trees) == 1
    tree = trees[0]
    assert tree["parent"]["market_id"] == "KXEVENT-PARENT"
    assert tree["parent"]["side"] == "yes"
    assert {child["market_id"] for child in tree["children"]} == {"KXEVENT-CHILD_A", "KXEVENT-CHILD_B"}
    assert {child["side"] for child in tree["children"]} == {"yes"}


def test_generation_diagnostics_reports_skip_reasons() -> None:
    markets = [
        _kalshi_market("KXEVENT", "A", "Alice"),
        _kalshi_market("KXEVENT", "B", "Bob"),
        _kalshi_market("KXEVENT", "C", "Carol"),
    ]
    diagnostics = build_generation_diagnostics(
        markets,
        settings=GenerationSettings(max_markets_per_event=2),
    )
    assert len(diagnostics) == 1
    assert diagnostics[0].skip_reason == "exceeds_max_markets_per_event"


def test_generation_diagnostics_includes_missing_requested_selector() -> None:
    markets = [
        _kalshi_market("KXEVENT", "A", "Alice"),
        _kalshi_market("KXEVENT", "B", "Bob"),
        _kalshi_market("KXEVENT", "C", "Carol"),
    ]
    diagnostics = build_generation_diagnostics(
        markets,
        settings=GenerationSettings(),
        include_event_keys={"kalshi:KXEVENT", "polymarket:missing-event"},
    )
    missing = [item for item in diagnostics if item.skip_reason == "no_markets_for_selector"]
    assert missing
    assert missing[0].venue == "polymarket"
    assert missing[0].event_key == "missing-event"


def test_fetch_polymarket_markets_resolves_slug_endpoint(monkeypatch) -> None:
    calls: list[tuple[str, dict | None]] = []

    class _MockResponse:
        def __init__(self, payload: dict | list, status_code: int = 200) -> None:
            self._payload = payload
            self.status_code = status_code

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(f"status={self.status_code}")

        def json(self):
            return self._payload

    class _MockAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, path: str, params: dict | None = None):
            calls.append((path, params))
            if path == "/events/slug/demo-event":
                return _MockResponse(
                    {
                        "slug": "demo-event",
                        "title": "Demo Event",
                        "markets": [
                            {
                                "id": "0xabc",
                                "question": "Alice wins",
                                "active": True,
                                "closed": False,
                            }
                        ],
                    }
                )
            return _MockResponse({}, status_code=404)

    monkeypatch.setattr("arb_bot.structural_rule_generator.httpx.AsyncClient", _MockAsyncClient)

    markets = asyncio.run(
        fetch_polymarket_markets_for_events(
            gamma_base_url="https://gamma-api.polymarket.com",
            event_slugs=["demo-event"],
        )
    )

    assert calls
    assert calls[0][0] == "/events/slug/demo-event"
    assert len(markets) == 1
    assert markets[0]["venue"] == "polymarket"
    assert markets[0]["event_slug"] == "demo-event"


def test_fetch_polymarket_markets_all_events_parses_nested_markets(monkeypatch) -> None:
    calls: list[tuple[str, dict | None]] = []

    class _MockResponse:
        def __init__(self, payload: dict | list, status_code: int = 200) -> None:
            self._payload = payload
            self.status_code = status_code

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(f"status={self.status_code}")

        def json(self):
            return self._payload

    class _MockAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, path: str, params: dict | None = None):
            calls.append((path, params))
            if path == "/events" and params and params.get("offset") == 0:
                return _MockResponse(
                    [
                        {
                            "slug": "event-a",
                            "title": "Event A",
                            "markets": [
                                {"id": "0x1", "question": "A1"},
                                {"id": "0x2", "question": "A2"},
                            ],
                        },
                        {
                            "slug": "event-b",
                            "title": "Event B",
                            "markets": [
                                {"id": "0x2", "question": "dupe"},
                                {"id": "0x3", "question": "B1"},
                            ],
                        },
                    ]
                )
            if path == "/events" and params and params.get("offset") == 2:
                return _MockResponse([])
            return _MockResponse({}, status_code=404)

    monkeypatch.setattr("arb_bot.structural_rule_generator.httpx.AsyncClient", _MockAsyncClient)

    markets = asyncio.run(
        fetch_polymarket_markets_all_events(
            gamma_base_url="https://gamma-api.polymarket.com",
            max_pages=2,
            page_size=2,
        )
    )

    assert calls
    assert calls[0][0] == "/events"
    assert len(markets) == 3
    assert {market["id"] for market in markets} == {"0x1", "0x2", "0x3"}
    assert all(market["venue"] == "polymarket" for market in markets)


def test_fetch_kalshi_markets_all_events_parses_nested_markets(monkeypatch) -> None:
    calls: list[tuple[str, dict | None]] = []

    class _MockResponse:
        def __init__(self, payload: dict | list, status_code: int = 200) -> None:
            self._payload = payload
            self.status_code = status_code

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(f"status={self.status_code}")

        def json(self):
            return self._payload

    class _MockAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, path: str, params: dict | None = None):
            calls.append((path, params))
            if path == "/events":
                assert params is not None
                if params.get("cursor") is None:
                    return _MockResponse(
                        {
                            "events": [
                                {
                                    "event_ticker": "KX-1",
                                    "title": "Event 1",
                                    "markets": [
                                        {"ticker": "KX-1-A", "subtitle": "A"},
                                        {"ticker": "KX-1-B", "subtitle": "B"},
                                    ],
                                }
                            ],
                            "cursor": "next",
                        }
                    )
                if params.get("cursor") == "next":
                    return _MockResponse({"events": []})
            return _MockResponse({}, status_code=404)

    monkeypatch.setattr("arb_bot.structural_rule_generator.httpx.AsyncClient", _MockAsyncClient)

    markets = asyncio.run(
        fetch_kalshi_markets_all_events(
            api_base_url="https://api.elections.kalshi.com/trade-api/v2",
            max_pages=3,
            page_size=100,
        )
    )

    assert calls
    assert calls[0][0] == "/events"
    assert len(markets) == 2
    assert {market["ticker"] for market in markets} == {"KX-1-A", "KX-1-B"}
    assert all(market["venue"] == "kalshi" for market in markets)
    assert all(market["event_ticker"] == "KX-1" for market in markets)


def test_fetch_kalshi_markets_all_events_falls_back_to_event_details(monkeypatch) -> None:
    calls: list[tuple[str, dict | None]] = []

    class _MockResponse:
        def __init__(self, payload: dict | list, status_code: int = 200) -> None:
            self._payload = payload
            self.status_code = status_code

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(f"status={self.status_code}")

        def json(self):
            return self._payload

    class _MockAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, path: str, params: dict | None = None):
            calls.append((path, params))
            if path == "/events":
                return _MockResponse({"events": [{"event_ticker": "KX-2", "title": "Event 2"}]})
            if path == "/events/KX-2":
                return _MockResponse(
                    {
                        "event_ticker": "KX-2",
                        "title": "Event 2",
                        "markets": [
                            {"ticker": "KX-2-A", "subtitle": "A"},
                            {"ticker": "KX-2-B", "subtitle": "B"},
                        ],
                    }
                )
            return _MockResponse({}, status_code=404)

    monkeypatch.setattr("arb_bot.structural_rule_generator.httpx.AsyncClient", _MockAsyncClient)

    markets = asyncio.run(
        fetch_kalshi_markets_all_events(
            api_base_url="https://api.elections.kalshi.com/trade-api/v2",
            max_pages=1,
            page_size=100,
            event_fetch_concurrency=2,
        )
    )

    assert len(markets) == 2
    assert any(path == "/events/KX-2" for path, _ in calls)
    assert {market["ticker"] for market in markets} == {"KX-2-A", "KX-2-B"}


# ---------------------------------------------------------------------------
# Mutual-exclusivity classifier tests
# ---------------------------------------------------------------------------


def _make_snapshot(market_id: str, outcome: str, title: str = "Same title") -> MarketSnapshot:
    return MarketSnapshot(
        venue="kalshi",
        event_key="KXTEST",
        market_id=market_id,
        title=title,
        outcome=outcome,
        status="open",
        liquidity=100.0,
    )


class TestLooksLikeNumericThresholds:
    def test_non_uniform_monotonic_values(self) -> None:
        # KXAGENCIES pattern: 5, 10, 20, 30, 50, 100, 200
        assert _looks_like_numeric_thresholds(["5", "10", "20", "30", "50", "100", "200"]) is True

    def test_non_uniform_decimals(self) -> None:
        # Gas price brackets: 3.80, 3.90, 4.00, 4.50, 5.00
        assert _looks_like_numeric_thresholds(["3.80", "3.90", "4.00", "4.50", "5.00"]) is True

    def test_uniform_spacing_returns_false(self) -> None:
        # Could be legitimate partitioned ranges: 0-5, 5-10, 10-15
        assert _looks_like_numeric_thresholds(["5", "10", "15", "20", "25"]) is False

    def test_named_outcomes_returns_false(self) -> None:
        assert _looks_like_numeric_thresholds(["alice", "bob", "carol"]) is False

    def test_mixed_text_and_numbers_returns_false(self) -> None:
        assert _looks_like_numeric_thresholds(["alice", "10", "carol"]) is False

    def test_two_outcomes_returns_false(self) -> None:
        # Too few to classify
        assert _looks_like_numeric_thresholds(["5", "10"]) is False

    def test_unsorted_non_uniform_detected(self) -> None:
        # Even when outcomes arrive in non-numeric order, we sort and detect
        assert _looks_like_numeric_thresholds(["10", "100", "20", "200", "30", "5", "50"]) is True

    def test_decreasing_non_uniform(self) -> None:
        # Reverse order — still detected after sorting
        assert _looks_like_numeric_thresholds(["200", "100", "50", "20", "10"]) is True


class TestLooksLikeTemporalSuffixes:
    def test_month_suffixes(self) -> None:
        # KXACAHSAFSA pattern: -27, -MAR26, -MAY26
        assert _looks_like_temporal_suffixes([
            "KXACAHSAFSA-MAR26", "KXACAHSAFSA-MAY26", "KXACAHSAFSA-JUL26"
        ]) is True

    def test_quarter_suffixes(self) -> None:
        assert _looks_like_temporal_suffixes([
            "KXFOO-26Q1", "KXFOO-26Q2", "KXFOO-26Q3", "KXFOO-26Q4"
        ]) is True

    def test_non_temporal_suffixes(self) -> None:
        assert _looks_like_temporal_suffixes([
            "KXEVENT-ALICE", "KXEVENT-BOB", "KXEVENT-CAROL"
        ]) is False

    def test_too_few_ids(self) -> None:
        assert _looks_like_temporal_suffixes(["KXFOO-MAR26"]) is False


class TestClassifyBucketExclusivity:
    def test_legitimate_winner_bucket_passes(self) -> None:
        """True mutually exclusive: Who will win the election?"""
        candidates = [
            _make_snapshot("KXEVENT-A", "Alice", "Who will win?"),
            _make_snapshot("KXEVENT-B", "Bob", "Who will win?"),
            _make_snapshot("KXEVENT-C", "Carol", "Who will win?"),
        ]
        outcomes = ["alice", "bob", "carol"]
        assert _classify_bucket_exclusivity(candidates, outcomes) is None

    def test_temporal_variant_rejected(self) -> None:
        """KXACAHSAFSA: temporal variants — not mutually exclusive."""
        candidates = [
            _make_snapshot("KXACAHSAFSA-27", "2027"),
            _make_snapshot("KXACAHSAFSA-MAR26", "March 2026"),
            _make_snapshot("KXACAHSAFSA-MAY26", "May 2026"),
        ]
        outcomes = ["2027", "march 2026", "may 2026"]
        result = _classify_bucket_exclusivity(candidates, outcomes)
        assert result is not None
        assert "temporal" in result

    def test_numeric_threshold_rejected(self) -> None:
        """KXAGENCIES: numeric thresholds — not mutually exclusive."""
        candidates = [
            _make_snapshot("KXAGENCIES-26-5", "5"),
            _make_snapshot("KXAGENCIES-26-10", "10"),
            _make_snapshot("KXAGENCIES-26-20", "20"),
            _make_snapshot("KXAGENCIES-26-30", "30"),
            _make_snapshot("KXAGENCIES-26-50", "50"),
            _make_snapshot("KXAGENCIES-26-100", "100"),
            _make_snapshot("KXAGENCIES-26-200", "200"),
        ]
        outcomes = ["5", "10", "20", "30", "50", "100", "200"]
        result = _classify_bucket_exclusivity(candidates, outcomes)
        assert result is not None
        assert "numeric_threshold" in result

    def test_independent_entity_rejected(self) -> None:
        """Markets about which countries/companies will do X — not exclusive."""
        candidates = [
            _make_snapshot("KXFOO-AMAZ", "Amazon", "Which companies will report?"),
            _make_snapshot("KXFOO-GOOG", "Google", "Which companies will report?"),
            _make_snapshot("KXFOO-META", "Meta", "Which companies will report?"),
        ]
        outcomes = ["amazon", "google", "meta"]
        result = _classify_bucket_exclusivity(candidates, outcomes)
        assert result is not None
        assert "independent" in result

    def test_threshold_language_in_outcome_rejected(self) -> None:
        """Outcomes containing 'above', 'below', etc."""
        candidates = [
            _make_snapshot("KXFOO-A", "Above $5B"),
            _make_snapshot("KXFOO-B", "Above $10B"),
            _make_snapshot("KXFOO-C", "Above $20B"),
        ]
        outcomes = ["above 5b", "above 10b", "above 20b"]
        result = _classify_bucket_exclusivity(candidates, outcomes)
        assert result is not None
        assert "threshold" in result

    def test_temporal_language_in_title_rejected(self) -> None:
        """Titles containing 'by March', 'before Q2'."""
        candidates = [
            _make_snapshot("KXFOO-A", "March 2026", "Will X happen by March 2026?"),
            _make_snapshot("KXFOO-B", "June 2026", "Will X happen by June 2026?"),
            _make_snapshot("KXFOO-C", "Dec 2026", "Will X happen by December 2026?"),
        ]
        outcomes = ["march 2026", "june 2026", "dec 2026"]
        result = _classify_bucket_exclusivity(candidates, outcomes)
        assert result is not None
        assert "temporal" in result

    def test_dollar_threshold_in_outcome_rejected(self) -> None:
        """Outcomes with dollar amounts like $100M."""
        candidates = [
            _make_snapshot("KXFOO-A", "$100M"),
            _make_snapshot("KXFOO-B", "$500M"),
            _make_snapshot("KXFOO-C", "$1B"),
        ]
        outcomes = ["$100m", "$500m", "$1b"]
        result = _classify_bucket_exclusivity(candidates, outcomes)
        assert result is not None


class TestGenerateRejectsNonExclusiveBuckets:
    """Integration tests: verify generate_structural_rules_payload skips bad buckets."""

    def test_numeric_thresholds_not_emitted(self) -> None:
        """An event with numeric threshold outcomes should NOT produce a bucket."""
        markets = [
            _kalshi_market("KXAGENCIES", "26-5", "5"),
            _kalshi_market("KXAGENCIES", "26-10", "10"),
            _kalshi_market("KXAGENCIES", "26-20", "20"),
            _kalshi_market("KXAGENCIES", "26-30", "30"),
            _kalshi_market("KXAGENCIES", "26-50", "50"),
            _kalshi_market("KXAGENCIES", "26-100", "100"),
            _kalshi_market("KXAGENCIES", "26-200", "200"),
        ]
        payload = generate_structural_rules_payload(markets)
        assert len(payload["mutually_exclusive_buckets"]) == 0

    def test_temporal_variants_not_emitted(self) -> None:
        """An event with temporal market IDs should NOT produce a bucket."""
        markets = [
            _kalshi_market("KXACAHSAFSA", "27", "2027"),
            _kalshi_market("KXACAHSAFSA", "MAR26", "March 2026"),
            _kalshi_market("KXACAHSAFSA", "MAY26", "May 2026"),
        ]
        payload = generate_structural_rules_payload(markets)
        assert len(payload["mutually_exclusive_buckets"]) == 0

    def test_legitimate_winner_still_emitted(self) -> None:
        """A true who-will-win event should still produce a bucket."""
        markets = [
            _kalshi_market("KXELECTION", "ALICE", "Alice"),
            _kalshi_market("KXELECTION", "BOB", "Bob"),
            _kalshi_market("KXELECTION", "CAROL", "Carol"),
        ]
        payload = generate_structural_rules_payload(markets)
        assert len(payload["mutually_exclusive_buckets"]) == 1

    def test_diagnostics_reports_exclusivity_rejection(self) -> None:
        """Diagnostics should report the exclusivity rejection reason."""
        markets = [
            _kalshi_market("KXAGENCIES", "26-5", "5"),
            _kalshi_market("KXAGENCIES", "26-10", "10"),
            _kalshi_market("KXAGENCIES", "26-20", "20"),
            _kalshi_market("KXAGENCIES", "26-30", "30"),
            _kalshi_market("KXAGENCIES", "26-50", "50"),
            _kalshi_market("KXAGENCIES", "26-100", "100"),
            _kalshi_market("KXAGENCIES", "26-200", "200"),
        ]
        diagnostics = build_generation_diagnostics(markets)
        assert len(diagnostics) == 1
        assert diagnostics[0].bucket_emitted is False
        assert diagnostics[0].skip_reason is not None
        assert "not_mutually_exclusive" in diagnostics[0].skip_reason


# ---------------------------------------------------------------------------
# Exchange-authoritative mutual exclusivity tests
# ---------------------------------------------------------------------------


class TestExchangeMutuallyExclusiveField:
    """Tests for the exchange_mutually_exclusive field on MarketSnapshot."""

    def test_snapshot_kalshi_with_mutually_exclusive_true(self) -> None:
        """Kalshi market with mutually_exclusive=True populates field."""
        raw = {
            "venue": "kalshi",
            "event_ticker": "KXNEWPOPE",
            "ticker": "KXNEWPOPE-70-YES",
            "title": "Who will be the next pope?",
            "subtitle": "Cardinal A",
            "status": "open",
            "liquidity": 100.0,
            "mutually_exclusive": True,
        }
        snapshot = snapshot_from_market(raw)
        assert snapshot is not None
        assert snapshot.exchange_mutually_exclusive is True

    def test_snapshot_kalshi_with_mutually_exclusive_false(self) -> None:
        """Kalshi market with mutually_exclusive=False populates field."""
        raw = {
            "venue": "kalshi",
            "event_ticker": "KXAGENCIES",
            "ticker": "KXAGENCIES-26-5",
            "title": "How many agencies?",
            "subtitle": "5",
            "status": "open",
            "liquidity": 100.0,
            "mutually_exclusive": False,
        }
        snapshot = snapshot_from_market(raw)
        assert snapshot is not None
        assert snapshot.exchange_mutually_exclusive is False

    def test_snapshot_kalshi_without_mutually_exclusive(self) -> None:
        """Kalshi market without the field defaults to None."""
        raw = {
            "venue": "kalshi",
            "event_ticker": "KXFOO",
            "ticker": "KXFOO-A",
            "title": "Test",
            "subtitle": "A",
            "status": "open",
            "liquidity": 100.0,
        }
        snapshot = snapshot_from_market(raw)
        assert snapshot is not None
        assert snapshot.exchange_mutually_exclusive is None

    def test_snapshot_polymarket_with_neg_risk_true(self) -> None:
        """Polymarket market with negRisk=True populates field."""
        raw = {
            "venue": "polymarket",
            "event_slug": "next-pope",
            "id": "0xabc",
            "title": "Cardinal A",
            "outcome": "Cardinal A",
            "active": True,
            "closed": False,
            "negRisk": True,
        }
        snapshot = snapshot_from_market(raw)
        assert snapshot is not None
        assert snapshot.exchange_mutually_exclusive is True

    def test_snapshot_polymarket_with_neg_risk_false(self) -> None:
        """Polymarket market with negRisk=False populates field."""
        raw = {
            "venue": "polymarket",
            "event_slug": "some-event",
            "id": "0xdef",
            "title": "Will X happen?",
            "outcome": "Yes",
            "active": True,
            "closed": False,
            "negRisk": False,
        }
        snapshot = snapshot_from_market(raw)
        assert snapshot is not None
        assert snapshot.exchange_mutually_exclusive is False

    def test_snapshot_polymarket_without_neg_risk(self) -> None:
        """Polymarket market without negRisk defaults to None."""
        raw = {
            "venue": "polymarket",
            "event_slug": "some-event",
            "id": "0xghi",
            "title": "Will X happen?",
            "outcome": "Yes",
            "active": True,
            "closed": False,
        }
        snapshot = snapshot_from_market(raw)
        assert snapshot is not None
        assert snapshot.exchange_mutually_exclusive is None


def _kalshi_market_with_me(
    event_ticker: str,
    suffix: str,
    outcome: str,
    *,
    mutually_exclusive: bool | None = None,
    liquidity: float = 100.0,
) -> dict:
    """Helper: Kalshi market dict with optional mutually_exclusive field."""
    row = {
        "venue": "kalshi",
        "event_ticker": event_ticker,
        "ticker": f"{event_ticker}-{suffix}",
        "title": "Who will win the election?",
        "subtitle": outcome,
        "status": "open",
        "liquidity": liquidity,
    }
    if mutually_exclusive is not None:
        row["mutually_exclusive"] = mutually_exclusive
    return row


class TestExchangeAuthorityInGeneration:
    """Tests for tiered exclusivity check in generate_structural_rules_payload."""

    def test_exchange_confirmed_bucket_bypasses_heuristics(self) -> None:
        """ME=True bucket that would fail numeric threshold check still generates.

        Without exchange confirmation, these numeric outcomes would be
        rejected by _looks_like_numeric_thresholds().  With ME=True, the
        heuristic is skipped.
        """
        markets = [
            _kalshi_market_with_me("KXTEST", "5", "5", mutually_exclusive=True),
            _kalshi_market_with_me("KXTEST", "10", "10", mutually_exclusive=True),
            _kalshi_market_with_me("KXTEST", "20", "20", mutually_exclusive=True),
            _kalshi_market_with_me("KXTEST", "50", "50", mutually_exclusive=True),
            _kalshi_market_with_me("KXTEST", "100", "100", mutually_exclusive=True),
        ]
        payload = generate_structural_rules_payload(markets)
        buckets = payload["mutually_exclusive_buckets"]
        assert len(buckets) == 1
        assert buckets[0]["exclusivity_source"] == "exchange_api"

    def test_exchange_denied_bucket_rejected(self) -> None:
        """ME=False bucket is rejected regardless of heuristic outcome.

        Even though Alice/Bob/Carol looks like a legit winner bucket,
        the exchange explicitly says NOT mutually exclusive — hard reject.
        """
        markets = [
            _kalshi_market_with_me("KXEVENT", "A", "Alice", mutually_exclusive=False),
            _kalshi_market_with_me("KXEVENT", "B", "Bob", mutually_exclusive=False),
            _kalshi_market_with_me("KXEVENT", "C", "Carol", mutually_exclusive=False),
        ]
        payload = generate_structural_rules_payload(markets)
        assert len(payload["mutually_exclusive_buckets"]) == 0

    def test_heuristic_fallback_when_no_exchange_field(self) -> None:
        """None field falls through to existing heuristic checks.

        Alice/Bob/Carol with no exchange field passes heuristics and emits.
        """
        markets = [
            _kalshi_market_with_me("KXEVENT", "A", "Alice", mutually_exclusive=None),
            _kalshi_market_with_me("KXEVENT", "B", "Bob", mutually_exclusive=None),
            _kalshi_market_with_me("KXEVENT", "C", "Carol", mutually_exclusive=None),
        ]
        payload = generate_structural_rules_payload(markets)
        buckets = payload["mutually_exclusive_buckets"]
        assert len(buckets) == 1
        assert buckets[0]["exclusivity_source"] == "heuristic"

    def test_heuristic_still_rejects_when_no_exchange_field(self) -> None:
        """Numeric thresholds without exchange flag still rejected by heuristic."""
        markets = [
            _kalshi_market_with_me("KXTEST", "5", "5"),
            _kalshi_market_with_me("KXTEST", "10", "10"),
            _kalshi_market_with_me("KXTEST", "20", "20"),
            _kalshi_market_with_me("KXTEST", "50", "50"),
            _kalshi_market_with_me("KXTEST", "100", "100"),
        ]
        payload = generate_structural_rules_payload(markets)
        assert len(payload["mutually_exclusive_buckets"]) == 0

    def test_exclusivity_source_in_generated_payload(self) -> None:
        """Verify exclusivity_source field is present in emitted bucket JSON."""
        markets = [
            _kalshi_market_with_me("KXEVENT", "A", "Alice", mutually_exclusive=True),
            _kalshi_market_with_me("KXEVENT", "B", "Bob", mutually_exclusive=True),
            _kalshi_market_with_me("KXEVENT", "C", "Carol", mutually_exclusive=True),
        ]
        payload = generate_structural_rules_payload(markets)
        buckets = payload["mutually_exclusive_buckets"]
        assert len(buckets) == 1
        assert "exclusivity_source" in buckets[0]
        assert buckets[0]["exclusivity_source"] == "exchange_api"

    def test_diagnostics_exchange_denied_reports_reason(self) -> None:
        """Diagnostics should report exchange_denied for ME=False events."""
        markets = [
            _kalshi_market_with_me("KXEVENT", "A", "Alice", mutually_exclusive=False),
            _kalshi_market_with_me("KXEVENT", "B", "Bob", mutually_exclusive=False),
            _kalshi_market_with_me("KXEVENT", "C", "Carol", mutually_exclusive=False),
        ]
        diagnostics = build_generation_diagnostics(markets)
        assert len(diagnostics) == 1
        assert diagnostics[0].bucket_emitted is False
        assert diagnostics[0].skip_reason is not None
        assert "exchange_denied" in diagnostics[0].skip_reason
