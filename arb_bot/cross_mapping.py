from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VenueRef:
    key: str
    value: str


@dataclass(frozen=True)
class CrossVenueMapping:
    group_id: str
    kalshi: VenueRef
    polymarket: VenueRef


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

    if kalshi is None or polymarket is None:
        return None

    return CrossVenueMapping(group_id=group_id, kalshi=kalshi, polymarket=polymarket)


def _pick_ref(row: dict[str, str], keys: list[str]) -> VenueRef | None:
    for key in keys:
        value = (row.get(key) or "").strip()
        if value:
            return VenueRef(key=key, value=value)
    return None
