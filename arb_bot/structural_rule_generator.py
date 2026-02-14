from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import httpx


_OPEN_STATUSES = {"", "open", "active", "trading"}


@dataclass(frozen=True)
class GenerationSettings:
    min_bucket_size: int = 3
    min_shared_title_ratio: float = 0.8
    min_event_coverage_ratio: float = 0.8
    max_markets_per_event: int = 60
    create_event_trees: bool = True


@dataclass(frozen=True)
class MarketSnapshot:
    venue: str
    event_key: str
    market_id: str
    title: str
    outcome: str
    status: str
    liquidity: float
    parent_market_ids: tuple[str, ...] = ()
    child_market_ids: tuple[str, ...] = ()
    exchange_mutually_exclusive: bool | None = None


@dataclass(frozen=True)
class EventGroup:
    venue: str
    event_key: str


@dataclass(frozen=True)
class GroupGenerationDiagnostics:
    venue: str
    event_key: str
    total_open_markets: int
    dominant_title_ratio: float
    candidate_markets: int
    coverage_ratio: float
    unique_outcomes: int
    bucket_emitted: bool
    event_tree_emitted: bool
    skip_reason: str | None = None
    event_tree_skip_reason: str | None = None


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _normalize_text(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return " ".join(cleaned.split())


def _slugify(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", text.lower())
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "group"


def _normalize_event_key(venue: str, event_key: str) -> str:
    value = event_key.strip()
    if venue == "kalshi":
        return value.upper()
    return value.lower()


def _event_selector_key(venue: str, event_key: str) -> str:
    return f"{venue}:{_normalize_event_key(venue, event_key)}".lower()


def _load_cross_mapping_rows(path: str | None) -> list[dict[str, str]]:
    if not path:
        return []

    mapping_path = Path(path)
    if not mapping_path.exists():
        return []

    rows: list[dict[str, str]] = []
    with mapping_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if isinstance(row, dict):
                rows.append({str(key): str(value or "") for key, value in row.items()})
    return rows


def _unique_group_id(base: str, seen: set[str]) -> str:
    if base not in seen:
        seen.add(base)
        return base
    index = 2
    while True:
        candidate = f"{base}_{index}"
        if candidate not in seen:
            seen.add(candidate)
            return candidate
        index += 1


def _mapping_market_id(row: dict[str, str], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def generate_parity_rules_from_cross_mapping_rows(
    rows: Iterable[dict[str, str]],
    *,
    relationship: str = "equivalent",
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    relation = relationship.strip().lower()
    if relation not in {"equivalent", "complement"}:
        relation = "equivalent"

    parity_rules: list[dict[str, Any]] = []
    skip_reasons: Counter[str] = Counter()
    seen_pairs: set[tuple[str, str]] = set()
    seen_group_ids: set[str] = set()

    for idx, row in enumerate(rows, start=1):
        kalshi_market_id = _mapping_market_id(
            row,
            keys=("kalshi_market_id", "kalshi_ticker"),
        )
        polymarket_market_id = _mapping_market_id(
            row,
            keys=("polymarket_market_id", "polymarket_condition_id"),
        )
        if not kalshi_market_id or not polymarket_market_id:
            skip_reasons["missing_direct_market_ids"] += 1
            continue

        pair_key = (kalshi_market_id.lower(), polymarket_market_id.lower())
        if pair_key in seen_pairs:
            skip_reasons["duplicate_market_pair"] += 1
            continue
        seen_pairs.add(pair_key)

        base_group = str(row.get("group_id") or f"mapping_{idx}").strip()
        if not base_group:
            base_group = f"mapping_{idx}"
        group_id = _unique_group_id(f"{_slugify(base_group)}_parity", seen_group_ids)

        parity_rules.append(
            {
                "group_id": group_id,
                "relationship": relation,
                "left": {
                    "venue": "kalshi",
                    "market_id": kalshi_market_id,
                    "side": "yes",
                },
                "right": {
                    "venue": "polymarket",
                    "market_id": polymarket_market_id,
                    "side": "yes",
                },
            }
        )

    return parity_rules, dict(skip_reasons)


def _normalize_market_id(value: Any) -> str:
    return str(value or "").strip()


def _parse_json_array_like(text: str) -> Any:
    raw = text.strip()
    if not raw:
        return raw
    if raw.startswith("[") or raw.startswith("{"):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


def _extract_market_id(value: Any) -> str | None:
    if isinstance(value, dict):
        for key in (
            "market_id",
            "marketId",
            "conditionId",
            "condition_id",
            "ticker",
            "id",
            "slug",
        ):
            market_id = _normalize_market_id(value.get(key))
            if market_id:
                return market_id
        return None

    if isinstance(value, str):
        parsed = _parse_json_array_like(value)
        if parsed is not value:
            nested = _extract_market_ids(parsed)
            if nested:
                return nested[0]
            return None
        market_id = _normalize_market_id(value)
        return market_id or None

    if isinstance(value, (int, float)):
        market_id = _normalize_market_id(value)
        return market_id or None

    return None


def _extract_market_ids(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        parsed = _parse_json_array_like(value)
        if parsed is not value:
            return _extract_market_ids(parsed)
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if len(parts) > 1:
            return parts
        single = _extract_market_id(value)
        return [single] if single else []

    if isinstance(value, (list, tuple, set)):
        ids: list[str] = []
        for item in value:
            item_ids = _extract_market_ids(item)
            ids.extend(item_ids)
        return ids

    if isinstance(value, dict):
        direct = _extract_market_id(value)
        if direct:
            return [direct]
        for key in ("markets", "children", "child_markets", "childMarkets", "market_ids", "marketIds"):
            nested = value.get(key)
            if nested is not None:
                return _extract_market_ids(nested)
        return []

    single = _extract_market_id(value)
    return [single] if single else []


def _dedupe_ids(ids: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in ids:
        market_id = _normalize_market_id(raw)
        if not market_id:
            continue
        token = market_id.lower()
        if token in seen:
            continue
        seen.add(token)
        ordered.append(market_id)
    return tuple(ordered)


def _extract_parent_market_ids(raw: dict[str, Any]) -> tuple[str, ...]:
    ids: list[str] = []
    for key in (
        "parent_market_id",
        "parentMarketId",
        "parent_market_ticker",
        "parentMarketTicker",
        "parent_condition_id",
        "parentConditionId",
        "parent_id",
        "parentId",
        "parent_slug",
        "parentSlug",
    ):
        ids.extend(_extract_market_ids(raw.get(key)))

    for key in ("parent", "parent_market", "parentMarket"):
        parent_value = raw.get(key)
        if parent_value is not None:
            ids.extend(_extract_market_ids(parent_value))

    return _dedupe_ids(ids)


def _extract_child_market_ids(raw: dict[str, Any]) -> tuple[str, ...]:
    ids: list[str] = []
    for key in (
        "child_market_ids",
        "childMarketIds",
        "child_markets",
        "childMarkets",
        "children",
        "related_market_ids",
        "relatedMarketIds",
        "related_markets",
        "relatedMarkets",
    ):
        ids.extend(_extract_market_ids(raw.get(key)))

    return _dedupe_ids(ids)


def _derive_kalshi_outcome(event_ticker: str, market_id: str) -> str:
    upper_event = event_ticker.upper()
    upper_market = market_id.upper()
    prefix = f"{upper_event}-"
    if upper_event and upper_market.startswith(prefix):
        return market_id[len(prefix) :].strip("-_ ").lower()
    return market_id.lower()


def _infer_venue(raw: dict[str, Any]) -> str:
    explicit = str(raw.get("venue") or raw.get("exchange") or "").strip().lower()
    if explicit in {"kalshi", "polymarket"}:
        return explicit

    if any(key in raw for key in ("event_ticker", "eventTicker", "ticker", "market_ticker", "marketTicker")):
        return "kalshi"

    if any(key in raw for key in ("conditionId", "clobTokenIds", "question", "event_slug", "eventSlug")):
        return "polymarket"

    return ""


def snapshot_from_market(raw: dict[str, Any]) -> MarketSnapshot | None:
    venue = _infer_venue(raw)
    if venue == "kalshi":
        event_key = _normalize_event_key(
            venue,
            str(raw.get("event_ticker") or raw.get("eventTicker") or "").strip(),
        )
        market_id = str(
            raw.get("ticker")
            or raw.get("market_ticker")
            or raw.get("marketTicker")
            or raw.get("market_id")
            or raw.get("id")
            or ""
        ).strip()
        if not event_key or not market_id:
            return None

        title = str(raw.get("title") or raw.get("question") or raw.get("name") or "").strip()
        outcome = str(
            raw.get("subtitle")
            or raw.get("yes_sub_title")
            or raw.get("yesSubtitle")
            or raw.get("outcome")
            or ""
        ).strip()
        if not outcome:
            outcome = _derive_kalshi_outcome(event_key, market_id)

        status = str(raw.get("status") or "").strip().lower()
        liquidity = _as_float(
            raw.get("liquidity")
            or raw.get("liquidity_dollars")
            or raw.get("open_interest")
            or raw.get("volume_24h")
            or raw.get("volume")
            or 0.0
        )
        # Kalshi's /events endpoint provides event-level `mutually_exclusive`
        # boolean. When events are fetched with nested markets, callers
        # inject this field into each market dict (see _append_kalshi_event_markets).
        me_raw = raw.get("mutually_exclusive")
        exchange_me: bool | None = None
        if isinstance(me_raw, bool):
            exchange_me = me_raw

        return MarketSnapshot(
            venue=venue,
            event_key=event_key,
            market_id=market_id,
            title=title,
            outcome=outcome,
            status=status,
            liquidity=liquidity,
            parent_market_ids=_extract_parent_market_ids(raw),
            child_market_ids=_extract_child_market_ids(raw),
            exchange_mutually_exclusive=exchange_me,
        )

    if venue == "polymarket":
        event_key = _normalize_event_key(
            venue,
            str(
                raw.get("event_slug")
                or raw.get("eventSlug")
                or raw.get("event_ticker")
                or raw.get("eventTicker")
                or ""
            ).strip(),
        )
        market_id = str(raw.get("conditionId") or raw.get("id") or raw.get("slug") or "").strip()
        if not event_key or not market_id:
            return None

        title = str(raw.get("event_title") or raw.get("eventTitle") or raw.get("title") or raw.get("question") or "").strip()
        outcome = str(raw.get("outcome") or raw.get("title") or raw.get("question") or raw.get("name") or "").strip()

        # Gamma commonly exposes active/closed booleans instead of status text.
        active = raw.get("active")
        closed = raw.get("closed")
        if isinstance(closed, bool) and closed:
            status = "closed"
        elif isinstance(active, bool) and not active:
            status = "closed"
        else:
            status = str(raw.get("status") or "open").strip().lower()

        liquidity = _as_float(
            raw.get("liquidity")
            or raw.get("liquidityNum")
            or raw.get("liquidityClob")
            or raw.get("volume24hr")
            or raw.get("volume24h")
            or raw.get("volume")
            or 0.0
        )
        # Polymarket's Gamma API provides `negRisk` on events, indicating
        # that outcomes are structurally guaranteed mutually exclusive by
        # the NegRisk smart contract adapter.
        # Note: can't use `or` chain because False is a valid signal.
        _sentinel = object()
        neg_risk = _sentinel
        for _nr_key in ("negRisk", "neg_risk", "enableNegRisk"):
            val = raw.get(_nr_key, _sentinel)
            if val is not _sentinel:
                neg_risk = val
                break
        poly_me: bool | None = None
        if neg_risk is not _sentinel:
            if isinstance(neg_risk, bool):
                poly_me = neg_risk
            elif neg_risk is not None:
                # Handle string "true"/"false" from some API responses
                poly_me = str(neg_risk).lower() in {"true", "1", "yes"}

        return MarketSnapshot(
            venue=venue,
            event_key=event_key,
            market_id=market_id,
            title=title,
            outcome=outcome,
            status=status,
            liquidity=liquidity,
            parent_market_ids=_extract_parent_market_ids(raw),
            child_market_ids=_extract_child_market_ids(raw),
            exchange_mutually_exclusive=poly_me,
        )

    return None


def _group_open_markets_by_event(
    snapshots: Iterable[MarketSnapshot],
    include_event_keys: set[str] | None = None,
) -> dict[EventGroup, list[MarketSnapshot]]:
    grouped: dict[EventGroup, list[MarketSnapshot]] = defaultdict(list)
    include = {
        value.strip().lower()
        for value in (include_event_keys or set())
        if value.strip()
    }

    for snapshot in snapshots:
        selector = _event_selector_key(snapshot.venue, snapshot.event_key)
        if include and selector not in include:
            continue
        if snapshot.status not in _OPEN_STATUSES:
            continue

        key = EventGroup(venue=snapshot.venue, event_key=snapshot.event_key)
        grouped[key].append(snapshot)

    for key in list(grouped.keys()):
        grouped[key].sort(key=lambda item: item.market_id)

    return grouped


def _dominant_title(markets: list[MarketSnapshot]) -> tuple[str, int]:
    counts = Counter(_normalize_text(item.title) for item in markets if item.title.strip())
    if not counts:
        return "", 0
    title, count = counts.most_common(1)[0]
    return title, count


def _bucket_group_id(group: EventGroup) -> str:
    return f"{group.venue}_{_slugify(group.event_key)}_exclusive"


# ---------------------------------------------------------------------------
# Market-type classifier: reject non-mutually-exclusive buckets
# ---------------------------------------------------------------------------

# Patterns indicating cumulative thresholds (multiple can resolve YES)
_THRESHOLD_PATTERNS: list[re.Pattern[str]] = [
    # "above/below/over/under X", "at least X", "or more", "or fewer"
    re.compile(r"\b(above|below|over|under|at least|or more|or fewer|or less|or higher|or lower|no more than|no less than|no fewer than)\b", re.IGNORECASE),
    # Dollar/price thresholds: "$5B", "$100M", "$1,000"
    re.compile(r"\$[\d,]+\.?\d*\s*[bmkt]?\b", re.IGNORECASE),
    # "higher than", "lower than", "greater than", "less than"
    re.compile(r"\b(higher|lower|greater|less|more|fewer)\s+than\b", re.IGNORECASE),
    # "+X%" or "-X%"
    re.compile(r"[+-]\d+(\.\d+)?%"),
]

# Patterns indicating temporal cutoffs (nested — if true by March, also true by December)
_TEMPORAL_PATTERNS: list[re.Pattern[str]] = [
    # "by [date]", "before [date]", "by end of"
    re.compile(r"\b(by|before)\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|q[1-4]|end of|year)", re.IGNORECASE),
    # "by March 31", "by 2027", "by December"
    re.compile(r"\bby\s+\d{4}\b", re.IGNORECASE),
]

# Patterns indicating independent per-entity markets (multiple can be true)
_INDEPENDENT_ENTITY_PATTERNS: list[re.Pattern[str]] = [
    # "Will [Company/Country] X?" where each entity is independent
    re.compile(r"\bwhich\s+(countries|companies|teams|players|states|cities)\b", re.IGNORECASE),
]


def _looks_like_numeric_thresholds(outcomes: list[str]) -> bool:
    """Detect if outcomes are numeric values suggesting threshold/range brackets.

    Outcomes may arrive in any order (e.g. sorted by market_id, not numerically),
    so we sort them numerically first before checking for threshold patterns.

    Examples that should return True:
    - ["5", "10", "20", "30", "50", "100", "200"]  (KXAGENCIES)
    - ["3.80", "3.90", "4.00", "4.10", "4.50"]  (gas price brackets)
    - ["10", "100", "20", "200", "30", "5", "50"]  (same as above, unsorted)

    Examples that should return False:
    - ["alice", "bob", "carol"]  (named candidates)
    """
    if len(outcomes) < 3:
        return False

    numeric_values: list[float] = []
    for outcome in outcomes:
        # Strip common prefixes/suffixes
        cleaned = re.sub(r"[,$%+]", "", outcome.strip())
        try:
            numeric_values.append(float(cleaned))
        except ValueError:
            return False

    if len(numeric_values) != len(outcomes):
        return False

    # Sort numerically — outcomes often arrive in market_id sort order
    numeric_values.sort()

    # All values must be distinct
    if len(set(numeric_values)) != len(numeric_values):
        return False

    # Check spacing pattern
    diffs = [numeric_values[i + 1] - numeric_values[i] for i in range(len(numeric_values) - 1)]

    # Check for non-uniform spacing (strong signal for thresholds like 5,10,20,50,100,200)
    # Uniform spacing might be legitimate partitioned ranges ("0-5", "5-10", "10-15")
    if len(set(round(d, 6) for d in diffs)) == 1:
        # Uniformly spaced — might be partitioned ranges (legit exclusive bucket)
        return False

    # Non-uniform spacing with all-numeric values = almost certainly thresholds
    return True


def _looks_like_temporal_suffixes(market_ids: list[str]) -> bool:
    """Detect if market ID suffixes suggest temporal variants of the same question.

    Examples that should return True:
    - ["KXFOO-MAR26", "KXFOO-MAY26", "KXFOO-27"]
    - ["KXFOO-26Q1", "KXFOO-26Q2", "KXFOO-26Q3"]
    """
    # Extract the varying suffix portion from market IDs
    if len(market_ids) < 2:
        return False

    # Month abbreviation patterns in suffixes
    month_pattern = re.compile(r"(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\d{2}", re.IGNORECASE)
    quarter_pattern = re.compile(r"\d{2}Q[1-4]", re.IGNORECASE)

    month_count = sum(1 for mid in market_ids if month_pattern.search(mid))
    quarter_count = sum(1 for mid in market_ids if quarter_pattern.search(mid))

    # If majority of IDs contain month or quarter patterns, likely temporal
    total = len(market_ids)
    if month_count >= total * 0.5:
        return True
    if quarter_count >= total * 0.5:
        return True

    return False


def _classify_bucket_exclusivity(
    candidate: list[MarketSnapshot],
    outcomes: list[str],
) -> str | None:
    """Return a rejection reason if the candidate set is NOT mutually exclusive.

    Returns None if the bucket looks legitimate (truly mutually exclusive).
    Returns a string description of why it was rejected.
    """
    market_ids = [item.market_id for item in candidate]
    titles = [item.title for item in candidate]

    # Check 1: Temporal suffix detection on market IDs
    if _looks_like_temporal_suffixes(market_ids):
        return "temporal_variant_markets"

    # Check 2: Numeric threshold detection on outcomes
    if _looks_like_numeric_thresholds(outcomes):
        return "numeric_threshold_markets"

    # Check 3: Pattern matching on outcome text
    for outcome in outcomes:
        for pattern in _THRESHOLD_PATTERNS:
            if pattern.search(outcome):
                return f"threshold_language_in_outcome"

    # Check 4: Pattern matching on title text for temporal cutoffs
    for title in titles:
        for pattern in _TEMPORAL_PATTERNS:
            if pattern.search(title):
                return "temporal_language_in_title"

    # Check 5: Pattern matching on title for independent entities
    for title in titles:
        for pattern in _INDEPENDENT_ENTITY_PATTERNS:
            if pattern.search(title):
                return "independent_entity_language_in_title"

    return None


def _event_tree_group_id_from_parent(parent: MarketSnapshot) -> str:
    return f"{parent.venue}_{_slugify(parent.event_key)}_tree_{_slugify(parent.market_id)}"


def _generate_explicit_event_trees(
    snapshots: list[MarketSnapshot],
    *,
    settings: GenerationSettings,
) -> tuple[list[dict[str, Any]], set[str]]:
    if not settings.create_event_trees:
        return [], set()

    open_snapshots = [item for item in snapshots if item.status in _OPEN_STATUSES]
    by_market_key: dict[tuple[str, str], MarketSnapshot] = {
        (item.venue, item.market_id.lower()): item for item in open_snapshots
    }
    parent_to_children: dict[tuple[str, str], set[str]] = defaultdict(set)

    for item in open_snapshots:
        item_key = (item.venue, item.market_id.lower())
        for child_id in item.child_market_ids:
            child = _normalize_market_id(child_id)
            if not child:
                continue
            parent_to_children[item_key].add(child.lower())
        for parent_id in item.parent_market_ids:
            parent = _normalize_market_id(parent_id)
            if not parent:
                continue
            parent_to_children[(item.venue, parent.lower())].add(item.market_id.lower())

    trees: list[dict[str, Any]] = []
    emitted_events: set[str] = set()
    seen_signatures: set[tuple[str, str, tuple[str, ...]]] = set()

    for (venue, parent_id_lc), child_ids_lc in sorted(parent_to_children.items(), key=lambda item: item[0]):
        parent = by_market_key.get((venue, parent_id_lc))
        if parent is None:
            continue

        children: list[MarketSnapshot] = []
        seen_children: set[str] = set()
        for child_id_lc in sorted(child_ids_lc):
            if child_id_lc == parent_id_lc:
                continue
            child = by_market_key.get((venue, child_id_lc))
            if child is None:
                continue
            if child.event_key != parent.event_key:
                continue
            child_token = child.market_id.lower()
            if child_token in seen_children:
                continue
            seen_children.add(child_token)
            children.append(child)

        if len(children) < 2:
            continue

        signature = (
            venue,
            parent.market_id.lower(),
            tuple(child.market_id.lower() for child in children),
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        group_id = _event_tree_group_id_from_parent(parent)
        tree = {
            "group_id": group_id,
            "parent": {
                "venue": parent.venue,
                "market_id": parent.market_id,
                "side": "yes",
            },
            "children": [
                {
                    "venue": child.venue,
                    "market_id": child.market_id,
                    "side": "yes",
                }
                for child in children
            ],
        }
        trees.append(tree)
        emitted_events.add(_event_selector_key(parent.venue, parent.event_key))

    return trees, emitted_events


def _candidate_markets_for_group(
    event_markets: list[MarketSnapshot],
    *,
    settings: GenerationSettings,
) -> tuple[list[MarketSnapshot], float]:
    """Returns candidate markets plus dominant-title ratio.

    Prefer a dominant-title subset when it is strong enough; otherwise fall back
    to full-event markets (still guarded later by coverage/outcome checks).
    """
    dominant_title, dominant_count = _dominant_title(event_markets)
    dominant_ratio = 0.0
    if dominant_title:
        dominant_ratio = dominant_count / max(1, len(event_markets))
        by_title = [item for item in event_markets if _normalize_text(item.title) == dominant_title]
        if dominant_ratio >= settings.min_shared_title_ratio and len(by_title) >= max(2, settings.min_bucket_size):
            return by_title, dominant_ratio

    return list(event_markets), dominant_ratio


def build_generation_diagnostics(
    markets: Iterable[dict[str, Any]],
    *,
    settings: GenerationSettings | None = None,
    include_event_keys: set[str] | None = None,
) -> list[GroupGenerationDiagnostics]:
    cfg = settings or GenerationSettings()
    snapshots = [snapshot for raw in markets if (snapshot := snapshot_from_market(raw)) is not None]
    grouped = _group_open_markets_by_event(snapshots, include_event_keys=include_event_keys)
    _event_trees, emitted_event_tree_events = _generate_explicit_event_trees(snapshots, settings=cfg)

    diagnostics: list[GroupGenerationDiagnostics] = []
    all_selector_counts: Counter[str] = Counter(
        _event_selector_key(snapshot.venue, snapshot.event_key)
        for snapshot in snapshots
    )
    open_selector_counts: Counter[str] = Counter(
        _event_selector_key(group.venue, group.event_key)
        for group in grouped.keys()
    )
    for group, event_markets in sorted(grouped.items(), key=lambda item: (item[0].venue, item[0].event_key)):
        total = len(event_markets)
        min_size = max(2, cfg.min_bucket_size)

        dominant_title = ""
        dominant_count = 0
        dominant_ratio = 0.0
        candidate_markets = 0
        coverage_ratio = 0.0
        unique_outcomes = 0
        bucket_emitted = False
        event_tree_emitted = False
        skip_reason: str | None = None
        event_tree_skip_reason: str | None = None

        if total < min_size:
            skip_reason = "too_few_markets"
        elif cfg.max_markets_per_event > 0 and total > cfg.max_markets_per_event:
            skip_reason = "exceeds_max_markets_per_event"
        else:
            dominant_title, dominant_count = _dominant_title(event_markets)
            candidate, dominant_ratio = _candidate_markets_for_group(event_markets, settings=cfg)
            candidate_markets = len(candidate)
            if candidate_markets < min_size:
                skip_reason = "candidate_markets_below_min_bucket_size"
            else:
                coverage_ratio = candidate_markets / max(1, total)
                if coverage_ratio < cfg.min_event_coverage_ratio:
                    skip_reason = "coverage_ratio_below_threshold"
                else:
                    outcomes = [_normalize_text(item.outcome) for item in candidate]
                    unique_outcomes = len(set(outcomes))
                    if any(not outcome for outcome in outcomes):
                        skip_reason = "empty_outcome"
                    elif unique_outcomes != candidate_markets:
                        skip_reason = "duplicate_outcomes"
                    else:
                        # Tiered exclusivity check (mirrors generate logic).
                        exchange_flags = [m.exchange_mutually_exclusive for m in candidate]
                        any_denied = any(f is False for f in exchange_flags)
                        all_confirmed = all(f is True for f in exchange_flags)

                        if any_denied:
                            skip_reason = "not_mutually_exclusive:exchange_denied"

                        # Always check temporal/threshold patterns regardless
                        # of exchange flag — exchanges misclassify nested
                        # temporal intervals as mutually exclusive.
                        if skip_reason is None:
                            diag_market_ids = [m.market_id for m in candidate]
                            diag_titles = [m.title for m in candidate]
                            if _looks_like_temporal_suffixes(diag_market_ids):
                                skip_reason = "not_mutually_exclusive:temporal_variant_markets"
                            elif _looks_like_numeric_thresholds(outcomes):
                                skip_reason = "not_mutually_exclusive:numeric_threshold_markets"
                            elif any(
                                pattern.search(title)
                                for title in diag_titles
                                for pattern in _TEMPORAL_PATTERNS
                            ):
                                skip_reason = "not_mutually_exclusive:temporal_language_in_title"

                        if skip_reason is None and not all_confirmed:
                            rejection = _classify_bucket_exclusivity(candidate, outcomes)
                            if rejection is not None:
                                skip_reason = f"not_mutually_exclusive:{rejection}"

                        if skip_reason is None:
                            bucket_emitted = True
                            selector = _event_selector_key(group.venue, group.event_key)
                            event_tree_emitted = selector in emitted_event_tree_events
                            if cfg.create_event_trees and not event_tree_emitted:
                                event_tree_skip_reason = "no_explicit_parent_child_structure"
                            elif not cfg.create_event_trees:
                                event_tree_skip_reason = "event_tree_generation_disabled"

        diagnostics.append(
            GroupGenerationDiagnostics(
                venue=group.venue,
                event_key=group.event_key,
                total_open_markets=total,
                dominant_title_ratio=dominant_ratio,
                candidate_markets=candidate_markets,
                coverage_ratio=coverage_ratio,
                unique_outcomes=unique_outcomes,
                bucket_emitted=bucket_emitted,
                event_tree_emitted=event_tree_emitted,
                skip_reason=skip_reason,
                event_tree_skip_reason=event_tree_skip_reason,
            )
        )

    requested = {
        value.strip().lower()
        for value in (include_event_keys or set())
        if value.strip()
    }
    for selector in sorted(requested):
        if selector in open_selector_counts:
            continue
        venue, _, event_key = selector.partition(":")
        total_seen = int(all_selector_counts.get(selector, 0))
        skip_reason = "no_open_markets_for_selector" if total_seen > 0 else "no_markets_for_selector"
        diagnostics.append(
            GroupGenerationDiagnostics(
                venue=venue or "unknown",
                event_key=event_key or "unknown",
                total_open_markets=0,
                dominant_title_ratio=0.0,
                candidate_markets=0,
                coverage_ratio=0.0,
                unique_outcomes=0,
                bucket_emitted=False,
                event_tree_emitted=False,
                skip_reason=skip_reason,
                event_tree_skip_reason=None,
            )
        )

    return diagnostics


def generate_structural_rules_payload(
    markets: Iterable[dict[str, Any]],
    *,
    settings: GenerationSettings | None = None,
    include_event_keys: set[str] | None = None,
    parity_rules: list[dict[str, Any]] | None = None,
    existing_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = settings or GenerationSettings()
    snapshots = [snapshot for raw in markets if (snapshot := snapshot_from_market(raw)) is not None]
    grouped = _group_open_markets_by_event(snapshots, include_event_keys=include_event_keys)

    buckets: list[dict[str, Any]] = []
    event_trees, _emitted_events = _generate_explicit_event_trees(snapshots, settings=cfg)

    for group, event_markets in sorted(grouped.items(), key=lambda item: (item[0].venue, item[0].event_key)):
        if len(event_markets) < max(2, cfg.min_bucket_size):
            continue
        if cfg.max_markets_per_event > 0 and len(event_markets) > cfg.max_markets_per_event:
            continue

        candidate, _dominant_ratio = _candidate_markets_for_group(event_markets, settings=cfg)
        if len(candidate) < max(2, cfg.min_bucket_size):
            continue

        coverage_ratio = len(candidate) / max(1, len(event_markets))
        if coverage_ratio < cfg.min_event_coverage_ratio:
            continue

        outcomes = [_normalize_text(item.outcome) for item in candidate]
        if any(not outcome for outcome in outcomes):
            continue
        if len(set(outcomes)) != len(outcomes):
            continue

        # --- Tiered mutual exclusivity check ---
        # Priority 1: exchange-authoritative flag (Kalshi mutually_exclusive,
        # Polymarket negRisk).  If any market explicitly denies exclusivity,
        # hard reject.  If all confirm, trust the exchange for most checks.
        exchange_flags = [m.exchange_mutually_exclusive for m in candidate]
        any_denied = any(f is False for f in exchange_flags)
        all_confirmed = all(f is True for f in exchange_flags)

        if any_denied:
            continue

        # Priority 1.5: ALWAYS run temporal/threshold structural checks,
        # even when exchange confirms exclusivity.  Exchanges (especially
        # Kalshi) mark entire events as mutually_exclusive even when
        # sub-markets are nested temporal intervals ("by March" ⊂ "by April")
        # or cumulative thresholds — these can NEVER be mutually exclusive
        # regardless of exchange flag.
        market_ids = [m.market_id for m in candidate]
        titles = [m.title for m in candidate]
        if _looks_like_temporal_suffixes(market_ids):
            continue
        if _looks_like_numeric_thresholds(outcomes):
            continue
        if any(
            pattern.search(title)
            for title in titles
            for pattern in _TEMPORAL_PATTERNS
        ):
            continue

        exclusivity_source = "exchange_api" if all_confirmed else "heuristic"

        # Priority 2: remaining heuristic text-pattern checks (only when no
        # exchange confirmation is available).
        if not all_confirmed:
            rejection = _classify_bucket_exclusivity(candidate, outcomes)
            if rejection is not None:
                continue

        candidate.sort(key=lambda item: item.market_id)
        bucket = {
            "group_id": _bucket_group_id(group),
            "payout_per_contract": 1.0,
            "exclusivity_source": exclusivity_source,
            "legs": [
                {
                    "venue": group.venue,
                    "market_id": item.market_id,
                    "side": "yes",
                }
                for item in candidate
            ],
        }
        buckets.append(bucket)

    payload = {
        "mutually_exclusive_buckets": buckets,
        "event_trees": event_trees,
        "cross_market_parity_checks": list(parity_rules or []),
    }

    if existing_payload is not None:
        payload = merge_with_existing_rules(existing_payload=existing_payload, generated_payload=payload)

    return payload


def merge_with_existing_rules(
    *,
    existing_payload: dict[str, Any],
    generated_payload: dict[str, Any],
) -> dict[str, Any]:
    existing_buckets = _list_dict(existing_payload.get("mutually_exclusive_buckets"))
    existing_event_trees = _list_dict(existing_payload.get("event_trees"))
    existing_parity = _list_dict(existing_payload.get("cross_market_parity_checks"))

    generated_buckets = _list_dict(generated_payload.get("mutually_exclusive_buckets"))
    generated_event_trees = _list_dict(generated_payload.get("event_trees"))
    generated_parity = _list_dict(generated_payload.get("cross_market_parity_checks"))

    merged = {
        "mutually_exclusive_buckets": _merge_by_group_id(existing_buckets, generated_buckets),
        "event_trees": _merge_by_group_id(existing_event_trees, generated_event_trees),
        "cross_market_parity_checks": _merge_by_group_id(existing_parity, generated_parity),
    }
    return merged


def _list_dict(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _merge_by_group_id(
    existing_items: list[dict[str, Any]],
    generated_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    for item in existing_items + generated_items:
        group_id = str(item.get("group_id") or "").strip()
        if not group_id:
            continue
        if group_id not in index:
            order.append(group_id)
        index[group_id] = item

    return [index[group_id] for group_id in order]


def validate_structural_rules_payload(
    payload: dict[str, Any],
    *,
    markets: Iterable[dict[str, Any]] | None = None,
    min_event_coverage_ratio: float = 0.8,
) -> list[str]:
    issues: list[str] = []
    buckets = _list_dict(payload.get("mutually_exclusive_buckets"))
    event_trees = _list_dict(payload.get("event_trees"))

    market_to_event: dict[tuple[str, str], EventGroup] = {}
    event_sizes: dict[EventGroup, int] = {}
    if markets is not None:
        snapshots = [snapshot for raw in markets if (snapshot := snapshot_from_market(raw)) is not None]
        grouped = _group_open_markets_by_event(snapshots)
        event_sizes = {group: len(items) for group, items in grouped.items()}
        for group, items in grouped.items():
            for item in items:
                market_to_event[(item.venue, item.market_id)] = group

    for idx, bucket in enumerate(buckets, start=1):
        group_id = str(bucket.get("group_id") or f"bucket_{idx}")
        legs = _list_dict(bucket.get("legs"))
        if len(legs) < 2:
            issues.append(f"{group_id}: has fewer than 2 legs")
            continue

        keys: list[tuple[str, str, str]] = []
        invalid_side = False
        for leg in legs:
            market_id = str(leg.get("market_id") or "").strip()
            venue = str(leg.get("venue") or "").strip().lower()
            side = str(leg.get("side") or "").strip().lower()
            if not market_id or not venue:
                issues.append(f"{group_id}: contains leg with missing venue/market_id")
                continue
            if side not in {"yes", "no"}:
                invalid_side = True
            keys.append((venue, market_id, side))
        if invalid_side:
            issues.append(f"{group_id}: contains invalid side value")

        if len(set(keys)) != len(keys):
            issues.append(f"{group_id}: contains duplicate legs")

        if market_to_event:
            bucket_events = {
                market_to_event[(venue, market_id)]
                for venue, market_id, _ in keys
                if (venue, market_id) in market_to_event
            }
            if len(bucket_events) == 1:
                event_group = next(iter(bucket_events))
                event_size = event_sizes.get(event_group, 0)
                if event_size > 0:
                    coverage = len({(venue, market_id) for venue, market_id, _ in keys}) / event_size
                    if coverage < min_event_coverage_ratio:
                        issues.append(
                            f"{group_id}: low event coverage {coverage:.2f} for "
                            f"{event_group.venue}:{event_group.event_key} "
                            f"(min={min_event_coverage_ratio:.2f})"
                        )

    for idx, rule in enumerate(event_trees, start=1):
        group_id = str(rule.get("group_id") or f"event_tree_{idx}")
        parent = rule.get("parent")
        children = _list_dict(rule.get("children"))
        if not isinstance(parent, dict):
            issues.append(f"{group_id}: missing parent leg")
            continue
        if len(children) < 2:
            issues.append(f"{group_id}: has fewer than 2 children")
            continue
        parent_market = str(parent.get("market_id") or "").strip()
        parent_venue = str(parent.get("venue") or "").strip().lower()
        child_keys = {
            (str(item.get("venue") or "").strip().lower(), str(item.get("market_id") or "").strip())
            for item in children
        }
        if parent_market and parent_venue and (parent_venue, parent_market) in child_keys:
            issues.append(f"{group_id}: parent also appears as child")

    return issues


def load_markets_from_json(path: str) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        markets = payload.get("markets")
        if isinstance(markets, list):
            return [item for item in markets if isinstance(item, dict)]
    raise ValueError("input JSON must be a list[dict] or an object containing `markets`")


async def fetch_kalshi_markets_for_events(
    *,
    api_base_url: str,
    event_tickers: list[str],
    timeout_seconds: float = 10.0,
) -> list[dict[str, Any]]:
    base_url = api_base_url.rstrip("/")
    event_ids = [item.strip().upper() for item in event_tickers if item.strip()]
    if not event_ids:
        return []

    markets: list[dict[str, Any]] = []
    seen_market_ids: set[str] = set()
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout_seconds) as client:
        for event_ticker in event_ids:
            payload = await _fetch_json_with_retry(client, f"/events/{event_ticker}")
            if not isinstance(payload, dict):
                continue
            event_title = str(payload.get("title") or payload.get("name") or payload.get("question") or "").strip()
            event_markets = payload.get("markets")
            if not isinstance(event_markets, list):
                continue
            # Extract event-level mutually_exclusive flag from Kalshi response.
            me_raw = payload.get("mutually_exclusive")
            me_flag: bool | None = me_raw if isinstance(me_raw, bool) else None
            _append_kalshi_event_markets(
                destination=markets,
                seen_market_ids=seen_market_ids,
                event_ticker=event_ticker,
                event_title=event_title,
                event_markets=event_markets,
                mutually_exclusive=me_flag,
            )
    return markets


async def _fetch_json_with_retry(
    client: httpx.AsyncClient,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    max_attempts: int = 5,
    base_delay_seconds: float = 0.25,
) -> Any:
    last_exc: Exception | None = None
    for attempt in range(max(1, max_attempts)):
        try:
            response = await client.get(path, params=params)
            if response.status_code == 429 and attempt + 1 < max_attempts:
                delay = base_delay_seconds * (2 ** attempt)
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = max(delay, float(retry_after))
                    except ValueError:
                        pass
                await asyncio.sleep(delay)
                continue
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            last_exc = exc
            raise
        except Exception as exc:
            last_exc = exc
            if attempt + 1 < max_attempts:
                await asyncio.sleep(base_delay_seconds * (2 ** attempt))
                continue
            raise

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"request failed for path={path}")


def _append_kalshi_event_markets(
    *,
    destination: list[dict[str, Any]],
    seen_market_ids: set[str],
    event_ticker: str,
    event_title: str,
    event_markets: list[Any],
    mutually_exclusive: bool | None = None,
) -> None:
    ticker = event_ticker.strip().upper()
    title = event_title.strip()

    for market in event_markets:
        if not isinstance(market, dict):
            continue

        market_id = str(
            market.get("ticker")
            or market.get("market_ticker")
            or market.get("marketTicker")
            or market.get("id")
            or ""
        ).strip()
        if market_id and market_id in seen_market_ids:
            continue
        if market_id:
            seen_market_ids.add(market_id)

        row = dict(market)
        row.setdefault("venue", "kalshi")
        if ticker:
            row.setdefault("event_ticker", ticker)
        if title:
            row.setdefault("event_title", title)
        # Inject event-level mutual exclusivity flag into each market dict
        # so snapshot_from_market() can read it.  Kalshi only exposes this
        # at the event level, not on individual markets.
        if mutually_exclusive is not None and "mutually_exclusive" not in row:
            row["mutually_exclusive"] = mutually_exclusive
        destination.append(row)


async def fetch_kalshi_markets_all_events(
    *,
    api_base_url: str,
    timeout_seconds: float = 10.0,
    max_pages: int = 20,
    page_size: int = 200,
    event_fetch_concurrency: int = 8,
) -> list[dict[str, Any]]:
    base_url = api_base_url.rstrip("/")
    if max_pages <= 0:
        return []

    headers = {
        "Accept": "application/json",
        "User-Agent": "arb-bot/1.0",
    }
    markets: list[dict[str, Any]] = []
    seen_market_ids: set[str] = set()
    events_to_expand: list[str] = []
    seen_event_tickers: set[str] = set()

    async with httpx.AsyncClient(base_url=base_url, timeout=timeout_seconds, headers=headers) as client:
        cursor: str | None = None
        for _ in range(max_pages):
            params: dict[str, Any] = {
                "status": "open",
                "limit": max(1, page_size),
            }
            if cursor:
                params["cursor"] = cursor

            payload = await _fetch_json_with_retry(client, "/events", params=params)

            events: list[dict[str, Any]] = []
            if isinstance(payload, list):
                events = [event for event in payload if isinstance(event, dict)]
                cursor = None
            elif isinstance(payload, dict):
                raw_events = payload.get("events")
                if isinstance(raw_events, list):
                    events = [event for event in raw_events if isinstance(event, dict)]
                cursor = str(payload.get("cursor") or "").strip() or None

            if not events:
                break

            for event in events:
                event_ticker = str(
                    event.get("event_ticker")
                    or event.get("eventTicker")
                    or event.get("ticker")
                    or ""
                ).strip().upper()
                if not event_ticker or event_ticker in seen_event_tickers:
                    continue
                seen_event_tickers.add(event_ticker)

                event_title = str(event.get("title") or event.get("name") or event.get("question") or "").strip()
                event_markets = event.get("markets")
                me_raw = event.get("mutually_exclusive")
                me_flag: bool | None = me_raw if isinstance(me_raw, bool) else None
                if isinstance(event_markets, list) and event_markets:
                    _append_kalshi_event_markets(
                        destination=markets,
                        seen_market_ids=seen_market_ids,
                        event_ticker=event_ticker,
                        event_title=event_title,
                        event_markets=event_markets,
                        mutually_exclusive=me_flag,
                    )
                else:
                    events_to_expand.append(event_ticker)

            if not cursor:
                break

        if events_to_expand:
            semaphore = asyncio.Semaphore(max(1, event_fetch_concurrency))

            async def _fetch_event_detail(event_ticker: str) -> tuple[str, str, list[Any], bool | None]:
                async with semaphore:
                    payload = await _fetch_json_with_retry(client, f"/events/{event_ticker}")
                    if not isinstance(payload, dict):
                        return event_ticker, "", [], None
                    event_title = str(payload.get("title") or payload.get("name") or payload.get("question") or "").strip()
                    me_raw = payload.get("mutually_exclusive")
                    me_flag: bool | None = me_raw if isinstance(me_raw, bool) else None
                    event_markets = payload.get("markets")
                    if not isinstance(event_markets, list):
                        return event_ticker, event_title, [], me_flag
                    return event_ticker, event_title, event_markets, me_flag

            tasks = [_fetch_event_detail(event_ticker) for event_ticker in events_to_expand]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for event_ticker, result in zip(events_to_expand, results):
                if isinstance(result, Exception):
                    continue
                _, event_title, event_markets, me_flag = result
                _append_kalshi_event_markets(
                    destination=markets,
                    seen_market_ids=seen_market_ids,
                    event_ticker=event_ticker,
                    event_title=event_title,
                    event_markets=event_markets,
                    mutually_exclusive=me_flag,
                )

    return markets


async def fetch_polymarket_markets_for_events(
    *,
    gamma_base_url: str,
    event_slugs: list[str],
    timeout_seconds: float = 10.0,
    page_size: int = 100,
) -> list[dict[str, Any]]:
    base_url = gamma_base_url.rstrip("/")
    wanted = {item.strip().lower() for item in event_slugs if item.strip()}
    if not wanted:
        return []

    found_events: dict[str, dict[str, Any]] = {}
    headers = {
        "Accept": "application/json",
        "User-Agent": "arb-bot/1.0",
    }
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout_seconds, headers=headers) as client:
        # Resolve explicit event slugs directly first to avoid heavy global pagination.
        for slug in sorted(wanted):
            event: dict[str, Any] | None = None

            # Preferred endpoint from Polymarket docs.
            try:
                response = await client.get(f"/events/slug/{slug}")
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, dict):
                    event = payload.get("event") if isinstance(payload.get("event"), dict) else payload
            except Exception:
                event = None

            # Compatibility fallback for APIs that return arrays on /events?slug=...
            if event is None:
                try:
                    response = await client.get(
                        "/events",
                        params={
                            "slug": slug,
                            "limit": max(1, page_size),
                            "offset": 0,
                        },
                    )
                    response.raise_for_status()
                    payload = response.json()
                    if isinstance(payload, list):
                        for item in payload:
                            if isinstance(item, dict) and str(item.get("slug") or "").strip().lower() == slug:
                                event = item
                                break
                    elif isinstance(payload, dict):
                        raw_events = payload.get("events")
                        if isinstance(raw_events, list):
                            for item in raw_events:
                                if isinstance(item, dict) and str(item.get("slug") or "").strip().lower() == slug:
                                    event = item
                                    break
                except Exception:
                    event = None

            # Legacy fallback kept for compatibility.
            if event is None:
                try:
                    response = await client.get(f"/events/{slug}")
                    response.raise_for_status()
                    payload = response.json()
                    if isinstance(payload, dict):
                        event = payload.get("event") if isinstance(payload.get("event"), dict) else payload
                except Exception:
                    event = None

            if isinstance(event, dict):
                found_events[slug] = event

    markets: list[dict[str, Any]] = []
    seen_market_ids: set[str] = set()
    for slug in sorted(found_events.keys()):
        event = found_events[slug]
        event_title = str(event.get("title") or event.get("question") or "").strip()
        # Polymarket event-level negRisk flag — inject into each market dict.
        neg_risk = event.get("negRisk") or event.get("neg_risk") or event.get("enableNegRisk")
        event_markets = event.get("markets")
        if not isinstance(event_markets, list):
            continue
        for market in event_markets:
            if isinstance(market, dict):
                market_id = str(market.get("conditionId") or market.get("id") or market.get("slug") or "").strip()
                if market_id and market_id in seen_market_ids:
                    continue
                if market_id:
                    seen_market_ids.add(market_id)
                row = dict(market)
                row.setdefault("venue", "polymarket")
                row.setdefault("event_slug", slug)
                row.setdefault("event_title", event_title)
                if neg_risk is not None and "negRisk" not in row and "neg_risk" not in row:
                    row["negRisk"] = neg_risk
                markets.append(row)
    return markets


async def fetch_polymarket_markets_all_events(
    *,
    gamma_base_url: str,
    timeout_seconds: float = 10.0,
    max_pages: int = 20,
    page_size: int = 100,
) -> list[dict[str, Any]]:
    base_url = gamma_base_url.rstrip("/")
    if max_pages <= 0:
        return []

    headers = {
        "Accept": "application/json",
        "User-Agent": "arb-bot/1.0",
    }
    markets: list[dict[str, Any]] = []
    seen_market_ids: set[str] = set()

    async with httpx.AsyncClient(base_url=base_url, timeout=timeout_seconds, headers=headers) as client:
        for page in range(max_pages):
            params = {
                "active": "true",
                "closed": "false",
                "limit": max(1, page_size),
                "offset": page * max(1, page_size),
            }
            response = await client.get("/events", params=params)
            response.raise_for_status()
            payload = response.json()

            events: list[dict[str, Any]] = []
            if isinstance(payload, list):
                events = [event for event in payload if isinstance(event, dict)]
            elif isinstance(payload, dict):
                raw_events = payload.get("events")
                if isinstance(raw_events, list):
                    events = [event for event in raw_events if isinstance(event, dict)]

            if not events:
                break

            for event in events:
                slug = str(event.get("slug") or "").strip().lower()
                title = str(event.get("title") or event.get("question") or "").strip()
                neg_risk = event.get("negRisk") or event.get("neg_risk") or event.get("enableNegRisk")
                event_markets = event.get("markets")
                if not isinstance(event_markets, list):
                    continue
                for market in event_markets:
                    if not isinstance(market, dict):
                        continue
                    market_id = str(market.get("conditionId") or market.get("id") or market.get("slug") or "").strip()
                    if market_id and market_id in seen_market_ids:
                        continue
                    if market_id:
                        seen_market_ids.add(market_id)
                    row = dict(market)
                    row.setdefault("venue", "polymarket")
                    if slug:
                        row.setdefault("event_slug", slug)
                    if title:
                        row.setdefault("event_title", title)
                    if neg_risk is not None and "negRisk" not in row and "neg_risk" not in row:
                        row["negRisk"] = neg_risk
                    markets.append(row)

            if len(events) < max(1, page_size):
                break

    return markets


def _parse_csv_set(raw: str | None, *, upper: bool) -> set[str]:
    if raw is None or not raw.strip():
        return set()
    values = {value.strip() for value in raw.split(",") if value.strip()}
    if upper:
        return {value.upper() for value in values}
    return {value.lower() for value in values}


def _build_include_event_keys(kalshi_events: set[str], polymarket_events: set[str]) -> set[str]:
    keys = set()
    for event in kalshi_events:
        keys.add(_event_selector_key("kalshi", event))
    for event in polymarket_events:
        keys.add(_event_selector_key("polymarket", event))
    return keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate structural rules from Kalshi and Polymarket market metadata.")
    parser.add_argument(
        "--input-markets-json",
        type=str,
        default=None,
        help="Optional JSON path containing market rows. If omitted, fetches from APIs by event selectors.",
    )
    parser.add_argument(
        "--kalshi-event-tickers",
        type=str,
        default="",
        help="Comma-separated Kalshi event tickers to fetch.",
    )
    parser.add_argument(
        "--kalshi-all-events",
        action="store_true",
        help="Fetch all open Kalshi events and parse nested event markets.",
    )
    parser.add_argument(
        "--kalshi-max-pages",
        type=int,
        default=20,
        help="Max pages to crawl from Kalshi /events when --kalshi-all-events is enabled.",
    )
    parser.add_argument(
        "--kalshi-page-size",
        type=int,
        default=200,
        help="Page size for Kalshi /events crawling.",
    )
    parser.add_argument(
        "--kalshi-event-fetch-concurrency",
        type=int,
        default=8,
        help="Concurrency for Kalshi per-event detail fetches when /events pages do not include nested markets.",
    )
    parser.add_argument(
        "--polymarket-event-slugs",
        type=str,
        default="",
        help="Comma-separated Polymarket event slugs to fetch.",
    )
    parser.add_argument(
        "--polymarket-all-events",
        action="store_true",
        help="Fetch all active/open Polymarket events and parse nested markets.",
    )
    parser.add_argument(
        "--polymarket-max-pages",
        type=int,
        default=20,
        help="Max pages to crawl from Polymarket /events when --polymarket-all-events is enabled.",
    )
    parser.add_argument(
        "--polymarket-page-size",
        type=int,
        default=100,
        help="Page size for Polymarket /events crawling.",
    )
    parser.add_argument(
        "--kalshi-api-base-url",
        type=str,
        default="https://api.elections.kalshi.com/trade-api/v2",
        help="Kalshi REST API base URL.",
    )
    parser.add_argument(
        "--polymarket-gamma-base-url",
        type=str,
        default="https://gamma-api.polymarket.com",
        help="Polymarket Gamma API base URL.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="arb_bot/config/structural_rules.generated.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--diagnostics-output",
        type=str,
        default=None,
        help="Optional JSON path to write per-event generation diagnostics.",
    )
    parser.add_argument(
        "--existing-rules",
        type=str,
        default=None,
        help="Optional existing structural-rules JSON. When provided, merges buckets/trees/parity by group_id.",
    )
    parser.add_argument(
        "--cross-mapping-path",
        type=str,
        default=None,
        help="Optional cross-venue CSV mapping path used to auto-generate parity checks.",
    )
    parser.add_argument(
        "--parity-relationship",
        type=str,
        default="equivalent",
        help="Parity relationship for generated checks: equivalent or complement.",
    )
    parser.add_argument("--min-bucket-size", type=int, default=3)
    parser.add_argument("--min-title-ratio", type=float, default=0.8)
    parser.add_argument("--min-coverage-ratio", type=float, default=0.8)
    parser.add_argument("--max-markets-per-event", type=int, default=60)
    parser.add_argument(
        "--no-event-trees",
        action="store_true",
        help="Disable generation of event-tree rules from generated buckets.",
    )
    return parser.parse_args()


async def _main() -> int:
    args = parse_args()

    kalshi_events = _parse_csv_set(args.kalshi_event_tickers, upper=True)
    polymarket_events = _parse_csv_set(args.polymarket_event_slugs, upper=False)

    markets: list[dict[str, Any]]
    if args.input_markets_json:
        markets = load_markets_from_json(args.input_markets_json)
    else:
        wants_kalshi = bool(kalshi_events) or bool(args.kalshi_all_events)
        wants_polymarket = bool(polymarket_events) or bool(args.polymarket_all_events)
        if not wants_kalshi and not wants_polymarket:
            raise ValueError(
                "provide --input-markets-json or one/both of --kalshi-event-tickers, --kalshi-all-events, --polymarket-event-slugs, --polymarket-all-events"
            )

        if args.kalshi_all_events:
            kalshi_task = fetch_kalshi_markets_all_events(
                api_base_url=args.kalshi_api_base_url,
                max_pages=max(1, args.kalshi_max_pages),
                page_size=max(1, args.kalshi_page_size),
                event_fetch_concurrency=max(1, args.kalshi_event_fetch_concurrency),
            )
        else:
            kalshi_task = fetch_kalshi_markets_for_events(
                api_base_url=args.kalshi_api_base_url,
                event_tickers=sorted(kalshi_events),
            )
        if args.polymarket_all_events:
            polymarket_task = fetch_polymarket_markets_all_events(
                gamma_base_url=args.polymarket_gamma_base_url,
                max_pages=max(1, args.polymarket_max_pages),
                page_size=max(1, args.polymarket_page_size),
            )
        else:
            polymarket_task = fetch_polymarket_markets_for_events(
                gamma_base_url=args.polymarket_gamma_base_url,
                event_slugs=sorted(polymarket_events),
                page_size=max(1, args.polymarket_page_size),
            )
        kalshi_markets, polymarket_markets = await asyncio.gather(kalshi_task, polymarket_task)
        if wants_kalshi and not kalshi_markets:
            raise ValueError(
                "kalshi fetch returned 0 markets; check connectivity, API endpoint access, and selector inputs"
            )
        if wants_polymarket and not polymarket_markets:
            raise ValueError(
                "polymarket fetch returned 0 markets; check connectivity, gamma endpoint access, and selector inputs"
            )
        markets = [*kalshi_markets, *polymarket_markets]
        print(
            "fetched markets: kalshi=%d polymarket=%d total=%d"
            % (len(kalshi_markets), len(polymarket_markets), len(markets))
        )
        if not markets:
            raise ValueError("no markets fetched for requested selectors; check event ids/slugs and connectivity")

    existing_payload: dict[str, Any] | None = None
    existing_path = args.existing_rules
    if existing_path:
        existing_payload = json.loads(Path(existing_path).read_text(encoding="utf-8"))

    mapping_rows = _load_cross_mapping_rows(args.cross_mapping_path)
    generated_parity, parity_skip_counts = generate_parity_rules_from_cross_mapping_rows(
        mapping_rows,
        relationship=args.parity_relationship,
    )

    settings = GenerationSettings(
        min_bucket_size=max(2, args.min_bucket_size),
        min_shared_title_ratio=max(0.0, min(1.0, args.min_title_ratio)),
        min_event_coverage_ratio=max(0.0, min(1.0, args.min_coverage_ratio)),
        max_markets_per_event=max(0, args.max_markets_per_event),
        create_event_trees=not args.no_event_trees,
    )
    include_event_keys = None if (args.polymarket_all_events or args.kalshi_all_events) else (_build_include_event_keys(kalshi_events, polymarket_events) or None)

    payload = generate_structural_rules_payload(
        markets,
        settings=settings,
        include_event_keys=include_event_keys,
        parity_rules=generated_parity,
        existing_payload=existing_payload,
    )
    diagnostics = build_generation_diagnostics(
        markets,
        settings=settings,
        include_event_keys=include_event_keys,
    )
    issues = validate_structural_rules_payload(
        payload,
        markets=markets,
        min_event_coverage_ratio=settings.min_event_coverage_ratio,
    )
    if issues:
        print("validation issues detected:")
        for issue in issues:
            print(f"- {issue}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.diagnostics_output:
        diag_path = Path(args.diagnostics_output)
        diag_path.parent.mkdir(parents=True, exist_ok=True)
        diag_payload = [
            {
                "venue": item.venue,
                "event_key": item.event_key,
                "total_open_markets": item.total_open_markets,
                "dominant_title_ratio": item.dominant_title_ratio,
                "candidate_markets": item.candidate_markets,
                "coverage_ratio": item.coverage_ratio,
                "unique_outcomes": item.unique_outcomes,
                "bucket_emitted": item.bucket_emitted,
                "event_tree_emitted": item.event_tree_emitted,
                "skip_reason": item.skip_reason,
                "event_tree_skip_reason": item.event_tree_skip_reason,
            }
            for item in diagnostics
        ]
        diag_path.write_text(json.dumps(diag_payload, indent=2), encoding="utf-8")

    skip_counts = Counter(
        item.skip_reason or "emitted"
        for item in diagnostics
    )

    print(f"generated structural rules -> {output_path}")
    print(
        "counts: buckets=%d event_trees=%d parity=%d"
        % (
            len(_list_dict(payload.get("mutually_exclusive_buckets"))),
            len(_list_dict(payload.get("event_trees"))),
            len(_list_dict(payload.get("cross_market_parity_checks"))),
        )
    )
    print("generation reasons: " + ", ".join(f"{reason}={count}" for reason, count in sorted(skip_counts.items())))
    if args.cross_mapping_path:
        print(
            "parity mapping: rows=%d generated=%d path=%s"
            % (len(mapping_rows), len(generated_parity), args.cross_mapping_path)
        )
        if parity_skip_counts:
            print(
                "parity reasons: "
                + ", ".join(f"{reason}={count}" for reason, count in sorted(parity_skip_counts.items()))
            )
    if args.diagnostics_output:
        print(f"diagnostics -> {args.diagnostics_output}")
    return 0


def main() -> int:
    return asyncio.run(_main())


if __name__ == "__main__":
    raise SystemExit(main())
