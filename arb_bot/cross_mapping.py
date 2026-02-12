from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class VenueRef:
    key: str
    value: str


@dataclass(frozen=True)
class CrossVenueMapping:
    """A mapping between equivalent markets across 2 or 3 venues.

    The ``kalshi`` and ``polymarket`` fields are required for backward
    compatibility.  ``forecastex`` is optional — when present the mapping
    represents a 3-venue group and the system can generate cross-venue
    pairs for all three venue combinations (K↔P, K↔F, P↔F).
    """

    group_id: str
    kalshi: VenueRef
    polymarket: VenueRef
    forecastex: VenueRef | None = None


@dataclass(frozen=True)
class VenuePair:
    """A directional pair of venue refs extracted from a mapping.

    Used by the strategy layer to iterate all tradeable cross-venue
    combinations from a single mapping row.
    """

    left_venue: str
    left_ref: VenueRef
    right_venue: str
    right_ref: VenueRef
    group_id: str


def venue_pairs(mapping: CrossVenueMapping) -> list[VenuePair]:
    """Generate all tradeable venue pairs from a mapping.

    Only includes pairs where both venues have a non-empty ref value.
    A 2-venue mapping (K+P) produces 1 pair.
    A 3-venue mapping (K+P+F) produces up to 3 pairs (K↔P, K↔F, P↔F).
    """
    refs: list[tuple[str, VenueRef]] = []
    for venue, ref in [
        ("kalshi", mapping.kalshi),
        ("polymarket", mapping.polymarket),
    ]:
        if ref.value.strip():
            refs.append((venue, ref))
    if mapping.forecastex is not None and mapping.forecastex.value.strip():
        refs.append(("forecastex", mapping.forecastex))

    pairs: list[VenuePair] = []
    for i in range(len(refs)):
        for j in range(i + 1, len(refs)):
            left_venue, left_ref = refs[i]
            right_venue, right_ref = refs[j]
            pairs.append(
                VenuePair(
                    left_venue=left_venue,
                    left_ref=left_ref,
                    right_venue=right_venue,
                    right_ref=right_ref,
                    group_id=mapping.group_id,
                )
            )
    return pairs


def all_venue_refs(mapping: CrossVenueMapping) -> Iterator[tuple[str, VenueRef]]:
    """Yield (venue_name, ref) for every venue with a non-empty ref."""
    if mapping.kalshi.value.strip():
        yield ("kalshi", mapping.kalshi)
    if mapping.polymarket.value.strip():
        yield ("polymarket", mapping.polymarket)
    if mapping.forecastex is not None and mapping.forecastex.value.strip():
        yield ("forecastex", mapping.forecastex)


def load_cross_venue_mappings(path: str | None) -> list[CrossVenueMapping]:
    if not path:
        return []

    file_path = Path(path)
    if not file_path.exists():
        return []

    mappings: list[CrossVenueMapping] = []
    with file_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader, start=1):
            mapping = _row_to_mapping(row, idx)
            if mapping is not None:
                mappings.append(mapping)
    return mappings


def _row_to_mapping(row: dict[str, str], idx: int) -> CrossVenueMapping | None:
    group_id = (row.get("group_id") or row.get("id") or f"map_{idx}").strip()

    kalshi = _pick_ref(
        row,
        keys=[
            "kalshi_market_id",
            "kalshi_ticker",
            "kalshi_event_ticker",
        ],
    )
    polymarket = _pick_ref(
        row,
        keys=[
            "polymarket_market_id",
            "polymarket_condition_id",
            "polymarket_slug",
        ],
    )

    forecastex = _pick_ref(
        row,
        keys=[
            "forecastex_market_id",
            "forecastex_symbol",
            "forecastex_contract_id",
        ],
    )

    # At least two venues must be present for a valid mapping.
    present_count = sum(1 for ref in (kalshi, polymarket, forecastex) if ref is not None)
    if present_count < 2:
        return None

    # For backward compatibility: if kalshi or polymarket is missing but
    # forecastex is present, create a synthetic empty ref for the missing
    # venue so the dataclass stays consistent.  The venue_pairs() helper
    # only generates pairs for non-None refs so no spurious pairs appear.
    if kalshi is None:
        kalshi = VenueRef(key="kalshi_market_id", value="")
    if polymarket is None:
        polymarket = VenueRef(key="polymarket_market_id", value="")

    return CrossVenueMapping(
        group_id=group_id,
        kalshi=kalshi,
        polymarket=polymarket,
        forecastex=forecastex,
    )


def _pick_ref(row: dict[str, str], keys: list[str]) -> VenueRef | None:
    for key in keys:
        value = (row.get(key) or "").strip()
        if value:
            return VenueRef(key=key, value=value)
    return None
