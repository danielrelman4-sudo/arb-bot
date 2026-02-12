from __future__ import annotations

import asyncio

from arb_bot.structural_rule_generator import (
    GenerationSettings,
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
