from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from arb_bot.models import Side

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MarketLegRef:
    venue: str
    market_id: str
    side: Side

    @property
    def key(self) -> tuple[str, str]:
        return self.venue.lower(), self.market_id.lower()


@dataclass(frozen=True)
class ExclusiveBucketRule:
    group_id: str
    legs: tuple[MarketLegRef, ...]
    payout_per_contract: float = 1.0
    exclusivity_source: str = "heuristic"  # "exchange_api" or "heuristic"


@dataclass(frozen=True)
class EventTreeRule:
    group_id: str
    parent: MarketLegRef
    children: tuple[MarketLegRef, ...]


@dataclass(frozen=True)
class ParityRule:
    group_id: str
    left: MarketLegRef
    right: MarketLegRef
    relationship: str = "equivalent"


@dataclass(frozen=True)
class StructuralRuleSet:
    buckets: tuple[ExclusiveBucketRule, ...] = field(default_factory=tuple)
    event_trees: tuple[EventTreeRule, ...] = field(default_factory=tuple)
    parity_checks: tuple[ParityRule, ...] = field(default_factory=tuple)

    @property
    def is_empty(self) -> bool:
        return not (self.buckets or self.event_trees or self.parity_checks)


def load_structural_rules(path: str | None) -> StructuralRuleSet:
    if not path:
        return StructuralRuleSet()

    rules_path = Path(path)
    if not rules_path.exists():
        LOGGER.warning("structural rules file not found: %s", path)
        return StructuralRuleSet()

    try:
        payload = json.loads(rules_path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.warning("failed to parse structural rules %s: %s", path, exc)
        return StructuralRuleSet()

    if not isinstance(payload, dict):
        LOGGER.warning("structural rules must be a JSON object: %s", path)
        return StructuralRuleSet()

    buckets = _parse_buckets(payload.get("mutually_exclusive_buckets"))
    event_trees = _parse_event_trees(payload.get("event_trees"))
    event_trees = _filter_event_trees_conflicting_with_buckets(event_trees, buckets)
    parity_checks = _parse_parity_checks(payload.get("cross_market_parity_checks"))
    return StructuralRuleSet(
        buckets=tuple(buckets),
        event_trees=tuple(event_trees),
        parity_checks=tuple(parity_checks),
    )


def _parse_buckets(raw: Any) -> list[ExclusiveBucketRule]:
    if not isinstance(raw, list):
        return []

    parsed: list[ExclusiveBucketRule] = []
    seen_signatures: set[tuple[tuple[str, str, str], ...]] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        group_id = str(item.get("group_id") or f"bucket_{index}").strip()
        legs = _parse_leg_refs(item.get("legs"))
        if len(legs) < 2:
            continue
        signature = tuple(
            sorted((leg.venue.lower(), leg.market_id.lower(), leg.side.value) for leg in legs)
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        payout = _to_float(item.get("payout_per_contract"), default=1.0)
        exclusivity_source = str(item.get("exclusivity_source") or "heuristic").strip().lower()
        if exclusivity_source not in {"exchange_api", "heuristic"}:
            exclusivity_source = "heuristic"
        parsed.append(
            ExclusiveBucketRule(
                group_id=group_id,
                legs=tuple(legs),
                payout_per_contract=max(0.0, payout),
                exclusivity_source=exclusivity_source,
            )
        )
    return parsed


def _parse_event_trees(raw: Any) -> list[EventTreeRule]:
    if not isinstance(raw, list):
        return []

    parsed: list[EventTreeRule] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        group_id = str(item.get("group_id") or f"event_tree_{index}").strip()
        parent = _parse_leg_ref(item.get("parent"))
        children = _parse_leg_refs(item.get("children"))
        if parent is None or parent.side is not Side.YES:
            continue

        deduped_children: list[MarketLegRef] = []
        seen_children: set[tuple[str, str]] = set()
        for child in children:
            if child.side is not Side.YES:
                continue
            if child.key == parent.key:
                continue
            if child.key in seen_children:
                continue
            seen_children.add(child.key)
            deduped_children.append(child)

        if len(deduped_children) < 2:
            continue
        parsed.append(EventTreeRule(group_id=group_id, parent=parent, children=tuple(deduped_children)))
    return parsed


def _parse_parity_checks(raw: Any) -> list[ParityRule]:
    if not isinstance(raw, list):
        return []

    parsed: list[ParityRule] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        group_id = str(item.get("group_id") or f"parity_{index}").strip()
        left = _parse_leg_ref(item.get("left"))
        right = _parse_leg_ref(item.get("right"))
        if left is None or right is None:
            continue
        relationship = str(item.get("relationship") or "equivalent").strip().lower()
        if relationship not in {"equivalent", "complement"}:
            relationship = "equivalent"
        parsed.append(
            ParityRule(
                group_id=group_id,
                left=left,
                right=right,
                relationship=relationship,
            )
        )
    return parsed


def _filter_event_trees_conflicting_with_buckets(
    event_trees: list[EventTreeRule],
    buckets: list[ExclusiveBucketRule],
) -> list[EventTreeRule]:
    if not event_trees or not buckets:
        return event_trees

    bucket_market_sets = [{leg.key for leg in bucket.legs} for bucket in buckets]
    filtered: list[EventTreeRule] = []
    dropped = 0

    for tree in event_trees:
        tree_markets = {tree.parent.key, *(child.key for child in tree.children)}
        conflicts = any(tree_markets.issubset(bucket_markets) for bucket_markets in bucket_market_sets)
        if conflicts:
            dropped += 1
            continue
        filtered.append(tree)

    if dropped > 0:
        LOGGER.warning(
            "dropped %d event_tree rules that were fully contained in mutually exclusive buckets",
            dropped,
        )
    return filtered


def _parse_leg_refs(raw: Any) -> list[MarketLegRef]:
    if not isinstance(raw, list):
        return []

    refs: list[MarketLegRef] = []
    for item in raw:
        ref = _parse_leg_ref(item)
        if ref is not None:
            refs.append(ref)
    return refs


def _parse_leg_ref(raw: Any) -> MarketLegRef | None:
    if not isinstance(raw, dict):
        return None
    venue = str(raw.get("venue") or "").strip().lower()
    market_id = str(raw.get("market_id") or "").strip()
    side_raw = str(raw.get("side") or "").strip().lower()
    if not venue or not market_id or side_raw not in {"yes", "no"}:
        return None
    return MarketLegRef(venue=venue, market_id=market_id, side=Side(side_raw))


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
