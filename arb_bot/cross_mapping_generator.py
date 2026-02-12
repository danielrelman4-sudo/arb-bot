from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from arb_bot.structural_rule_generator import (
    fetch_kalshi_markets_all_events,
    fetch_kalshi_markets_for_events,
    fetch_polymarket_markets_all_events,
    fetch_polymarket_markets_for_events,
    load_markets_from_json,
)

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "will",
    "with",
    "market",
    "markets",
}


@dataclass(frozen=True)
class CandidateMarket:
    venue: str
    market_id: str
    text: str
    tokens: frozenset[str]
    liquidity: float
    volume: float


@dataclass(frozen=True)
class CandidateMatch:
    kalshi_market_id: str
    polymarket_market_id: str
    score: float
    shared_tokens: int
    kalshi_text: str
    polymarket_text: str


@dataclass(frozen=True)
class MappingDiagnostics:
    kalshi_candidates: int
    polymarket_candidates: int
    mappings_emitted: int
    skip_reasons: dict[str, int]
    unmatched_kalshi: int
    unmatched_polymarket: int


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _normalize_text(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return " ".join(cleaned.split())


def _tokenize(text: str) -> frozenset[str]:
    tokens = set()
    normalized = _normalize_text(text)
    for token in normalized.split():
        if token in _STOPWORDS:
            continue
        if len(token) <= 1 and not token.isdigit():
            continue
        tokens.add(token)
    return frozenset(tokens)


def _slugify(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", text.lower())
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "mapped_market"


def _derive_kalshi_outcome(event_ticker: str, market_id: str) -> str:
    upper_event = event_ticker.upper()
    upper_market = market_id.upper()
    prefix = f"{upper_event}-"
    if upper_event and upper_market.startswith(prefix):
        return market_id[len(prefix) :].strip("-_ ").lower()
    return ""


def _build_kalshi_text(market: dict[str, Any]) -> str:
    market_id = str(market.get("ticker") or market.get("market_ticker") or market.get("marketTicker") or "").strip()
    event_ticker = str(market.get("event_ticker") or market.get("eventTicker") or "").strip()
    title = str(
        market.get("title")
        or market.get("question")
        or market.get("name")
        or market.get("event_title")
        or ""
    ).strip()
    outcome = str(
        market.get("subtitle")
        or market.get("yes_sub_title")
        or market.get("yesSubtitle")
        or market.get("outcome")
        or ""
    ).strip()
    if not outcome and market_id:
        outcome = _derive_kalshi_outcome(event_ticker, market_id)

    text_parts = [part for part in (title, outcome) if part]
    return " ".join(text_parts).strip()


def _build_polymarket_text(market: dict[str, Any]) -> str:
    question = str(market.get("question") or market.get("title") or market.get("name") or "").strip()
    event_title = str(market.get("event_title") or market.get("eventTitle") or "").strip()

    if question and event_title:
        q = _normalize_text(question)
        e = _normalize_text(event_title)
        if q and e and q in e:
            return event_title
        if q and e and e in q:
            return question
        return f"{question} {event_title}".strip()
    return question or event_title


def _kalshi_candidate(market: dict[str, Any]) -> CandidateMarket | None:
    market_id = str(
        market.get("ticker")
        or market.get("market_ticker")
        or market.get("marketTicker")
        or market.get("market_id")
        or market.get("id")
        or ""
    ).strip()
    if not market_id:
        return None
    text = _build_kalshi_text(market)
    tokens = _tokenize(text)
    if len(tokens) < 2:
        return None
    liquidity = _as_float(
        market.get("liquidity")
        or market.get("liquidity_dollars")
        or market.get("open_interest")
        or 0.0
    )
    volume = _as_float(market.get("volume_24h") or market.get("volume") or 0.0)
    return CandidateMarket(
        venue="kalshi",
        market_id=market_id,
        text=text,
        tokens=tokens,
        liquidity=liquidity,
        volume=volume,
    )


def _polymarket_candidate(market: dict[str, Any]) -> CandidateMarket | None:
    market_id = str(market.get("conditionId") or market.get("id") or market.get("slug") or "").strip()
    if not market_id:
        return None
    text = _build_polymarket_text(market)
    tokens = _tokenize(text)
    if len(tokens) < 2:
        return None
    liquidity = _as_float(
        market.get("liquidity")
        or market.get("liquidityNum")
        or market.get("liquidityClob")
        or 0.0
    )
    volume = _as_float(
        market.get("volume24hr")
        or market.get("volume24h")
        or market.get("volume24")
        or market.get("volume")
        or 0.0
    )
    return CandidateMarket(
        venue="polymarket",
        market_id=market_id,
        text=text,
        tokens=tokens,
        liquidity=liquidity,
        volume=volume,
    )


def _build_candidates(markets: list[dict[str, Any]]) -> tuple[list[CandidateMarket], list[CandidateMarket]]:
    kalshi: list[CandidateMarket] = []
    polymarket: list[CandidateMarket] = []
    seen_k: set[str] = set()
    seen_p: set[str] = set()

    for market in markets:
        venue = str(market.get("venue") or market.get("exchange") or "").strip().lower()
        if venue == "kalshi" or "event_ticker" in market or "market_ticker" in market or "ticker" in market:
            candidate = _kalshi_candidate(market)
            if candidate is None or candidate.market_id in seen_k:
                continue
            seen_k.add(candidate.market_id)
            kalshi.append(candidate)
            continue

        if venue == "polymarket" or "conditionId" in market or "clobTokenIds" in market or "event_slug" in market:
            candidate = _polymarket_candidate(market)
            if candidate is None or candidate.market_id in seen_p:
                continue
            seen_p.add(candidate.market_id)
            polymarket.append(candidate)

    return kalshi, polymarket


def _similarity(left_tokens: frozenset[str], right_tokens: frozenset[str]) -> tuple[float, int]:
    if not left_tokens or not right_tokens:
        return 0.0, 0
    shared = left_tokens & right_tokens
    if not shared:
        return 0.0, 0
    intersection = len(shared)
    union = len(left_tokens | right_tokens)
    jaccard = intersection / max(1, union)
    containment = intersection / max(1, min(len(left_tokens), len(right_tokens)))
    score = (0.7 * jaccard) + (0.3 * containment)
    return score, intersection


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


def generate_cross_venue_mapping_rows(
    markets: list[dict[str, Any]],
    *,
    min_match_score: float = 0.62,
    min_shared_tokens: int = 2,
    min_score_gap: float = 0.03,
    max_token_share: float = 0.02,
    max_candidate_pool: int = 2000,
) -> tuple[list[dict[str, str]], MappingDiagnostics]:
    kalshi_candidates, polymarket_candidates = _build_candidates(markets)

    skip_reasons: Counter[str] = Counter()
    if not kalshi_candidates or not polymarket_candidates:
        diagnostics = MappingDiagnostics(
            kalshi_candidates=len(kalshi_candidates),
            polymarket_candidates=len(polymarket_candidates),
            mappings_emitted=0,
            skip_reasons=dict(skip_reasons),
            unmatched_kalshi=len(kalshi_candidates),
            unmatched_polymarket=len(polymarket_candidates),
        )
        return [], diagnostics

    token_freq: Counter[str] = Counter()
    token_to_pm_indexes: dict[str, list[int]] = defaultdict(list)
    for idx, candidate in enumerate(polymarket_candidates):
        for token in candidate.tokens:
            token_freq[token] += 1
            token_to_pm_indexes[token].append(idx)

    max_df = max(2, int(len(polymarket_candidates) * max(0.0001, max_token_share)))
    provisional_matches: list[CandidateMatch] = []

    for kalshi in kalshi_candidates:
        indexed_tokens = [
            token
            for token in kalshi.tokens
            if token_freq.get(token, 0) > 0 and token_freq.get(token, 0) <= max_df
        ]
        if not indexed_tokens:
            skip_reasons["no_indexable_tokens"] += 1
            continue

        candidate_indexes: set[int] = set()
        for token in sorted(indexed_tokens, key=lambda value: token_freq.get(value, 0)):
            candidate_indexes.update(token_to_pm_indexes.get(token, []))
            if len(candidate_indexes) >= max_candidate_pool:
                break

        if not candidate_indexes:
            skip_reasons["no_candidate_pool"] += 1
            continue

        scored: list[tuple[float, int, CandidateMarket]] = []
        for pm_index in candidate_indexes:
            polymarket = polymarket_candidates[pm_index]
            score, shared_tokens = _similarity(kalshi.tokens, polymarket.tokens)
            if shared_tokens < min_shared_tokens:
                continue
            if score < min_match_score:
                continue
            scored.append((score, shared_tokens, polymarket))

        if not scored:
            skip_reasons["below_threshold"] += 1
            continue

        scored.sort(
            key=lambda item: (
                item[0],
                item[1],
                item[2].liquidity,
                item[2].volume,
            ),
            reverse=True,
        )
        best_score, best_shared_tokens, best_pm = scored[0]
        if len(scored) > 1 and (best_score - scored[1][0]) < min_score_gap:
            skip_reasons["ambiguous_top_match"] += 1
            continue

        provisional_matches.append(
            CandidateMatch(
                kalshi_market_id=kalshi.market_id,
                polymarket_market_id=best_pm.market_id,
                score=best_score,
                shared_tokens=best_shared_tokens,
                kalshi_text=kalshi.text,
                polymarket_text=best_pm.text,
            )
        )

    provisional_matches.sort(
        key=lambda item: (item.score, item.shared_tokens),
        reverse=True,
    )

    assigned_kalshi: set[str] = set()
    assigned_polymarket: set[str] = set()
    rows: list[dict[str, str]] = []
    seen_group_ids: set[str] = set()

    for match in provisional_matches:
        if match.kalshi_market_id in assigned_kalshi:
            continue
        if match.polymarket_market_id in assigned_polymarket:
            continue

        assigned_kalshi.add(match.kalshi_market_id)
        assigned_polymarket.add(match.polymarket_market_id)

        base_group_id = _slugify(match.kalshi_text)
        group_id = _unique_group_id(base_group_id, seen_group_ids)

        rows.append(
            {
                "group_id": group_id,
                "kalshi_market_id": match.kalshi_market_id,
                "polymarket_market_id": match.polymarket_market_id,
            }
        )

    diagnostics = MappingDiagnostics(
        kalshi_candidates=len(kalshi_candidates),
        polymarket_candidates=len(polymarket_candidates),
        mappings_emitted=len(rows),
        skip_reasons=dict(skip_reasons),
        unmatched_kalshi=max(0, len(kalshi_candidates) - len(assigned_kalshi)),
        unmatched_polymarket=max(0, len(polymarket_candidates) - len(assigned_polymarket)),
    )
    return rows, diagnostics


def _parse_csv_set(raw: str | None, *, upper: bool) -> set[str]:
    if raw is None or not raw.strip():
        return set()
    values = {value.strip() for value in raw.split(",") if value.strip()}
    if upper:
        return {value.upper() for value in values}
    return {value.lower() for value in values}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate cross-venue Kalshi<->Polymarket market mapping CSV from live event books.",
    )
    parser.add_argument(
        "--input-markets-json",
        type=str,
        default=None,
        help="Optional input JSON containing mixed venue market rows.",
    )
    parser.add_argument(
        "--kalshi-event-tickers",
        type=str,
        default="",
        help="Optional comma-separated Kalshi event tickers (when not crawling all Kalshi events).",
    )
    parser.add_argument(
        "--polymarket-event-slugs",
        type=str,
        default="",
        help="Optional comma-separated Polymarket event slugs (when not crawling all Polymarket events).",
    )
    parser.add_argument(
        "--kalshi-all-events",
        action="store_true",
        help="Crawl all open Kalshi events.",
    )
    parser.add_argument(
        "--polymarket-all-events",
        action="store_true",
        help="Crawl all open Polymarket events.",
    )
    parser.add_argument("--kalshi-max-pages", type=int, default=20)
    parser.add_argument("--kalshi-page-size", type=int, default=200)
    parser.add_argument("--kalshi-event-fetch-concurrency", type=int, default=8)
    parser.add_argument("--polymarket-max-pages", type=int, default=20)
    parser.add_argument("--polymarket-page-size", type=int, default=100)
    parser.add_argument(
        "--kalshi-api-base-url",
        type=str,
        default="https://api.elections.kalshi.com/trade-api/v2",
    )
    parser.add_argument(
        "--polymarket-gamma-base-url",
        type=str,
        default="https://gamma-api.polymarket.com",
    )
    parser.add_argument("--min-match-score", type=float, default=0.62)
    parser.add_argument("--min-shared-tokens", type=int, default=2)
    parser.add_argument("--min-score-gap", type=float, default=0.03)
    parser.add_argument("--max-token-share", type=float, default=0.02)
    parser.add_argument("--max-candidate-pool", type=int, default=2000)
    parser.add_argument(
        "--output",
        type=str,
        default="arb_bot/config/cross_venue_map.generated.csv",
    )
    parser.add_argument(
        "--diagnostics-output",
        type=str,
        default=None,
        help="Optional JSON path for diagnostics summary.",
    )
    return parser.parse_args()


async def _main() -> int:
    args = parse_args()
    kalshi_events = _parse_csv_set(args.kalshi_event_tickers, upper=True)
    polymarket_events = _parse_csv_set(args.polymarket_event_slugs, upper=False)

    markets: list[dict[str, Any]] = []
    if args.input_markets_json:
        markets = load_markets_from_json(args.input_markets_json)
    else:
        if not args.kalshi_all_events and not kalshi_events and not args.polymarket_all_events and not polymarket_events:
            args.kalshi_all_events = True
            args.polymarket_all_events = True

        wants_kalshi = bool(args.kalshi_all_events) or bool(kalshi_events)
        wants_polymarket = bool(args.polymarket_all_events) or bool(polymarket_events)
        if not wants_kalshi and not wants_polymarket:
            raise ValueError(
                "no markets requested: use --kalshi-all-events/--polymarket-all-events or explicit selector lists"
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
        print(
            "fetched markets: kalshi=%d polymarket=%d total=%d"
            % (len(kalshi_markets), len(polymarket_markets), len(kalshi_markets) + len(polymarket_markets))
        )
        if wants_kalshi and not kalshi_markets:
            raise ValueError("kalshi fetch returned 0 markets")
        if wants_polymarket and not polymarket_markets:
            raise ValueError("polymarket fetch returned 0 markets")
        markets = [*kalshi_markets, *polymarket_markets]

    rows, diagnostics = generate_cross_venue_mapping_rows(
        markets,
        min_match_score=max(0.0, min(1.0, args.min_match_score)),
        min_shared_tokens=max(1, args.min_shared_tokens),
        min_score_gap=max(0.0, args.min_score_gap),
        max_token_share=max(0.0001, min(1.0, args.max_token_share)),
        max_candidate_pool=max(50, args.max_candidate_pool),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["group_id", "kalshi_market_id", "polymarket_market_id"],
        )
        writer.writeheader()
        writer.writerows(rows)

    if args.diagnostics_output:
        diag_path = Path(args.diagnostics_output)
        diag_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "kalshi_candidates": diagnostics.kalshi_candidates,
            "polymarket_candidates": diagnostics.polymarket_candidates,
            "mappings_emitted": diagnostics.mappings_emitted,
            "unmatched_kalshi": diagnostics.unmatched_kalshi,
            "unmatched_polymarket": diagnostics.unmatched_polymarket,
            "skip_reasons": diagnostics.skip_reasons,
        }
        diag_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"generated cross-venue mapping -> {output_path}")
    print(
        "counts: mappings=%d kalshi_candidates=%d polymarket_candidates=%d unmatched_kalshi=%d unmatched_polymarket=%d"
        % (
            diagnostics.mappings_emitted,
            diagnostics.kalshi_candidates,
            diagnostics.polymarket_candidates,
            diagnostics.unmatched_kalshi,
            diagnostics.unmatched_polymarket,
        )
    )
    print(
        "skip_reasons: "
        + ", ".join(f"{key}={value}" for key, value in sorted(diagnostics.skip_reasons.items()))
    )
    if args.diagnostics_output:
        print(f"diagnostics -> {args.diagnostics_output}")
    return 0


def main() -> int:
    return asyncio.run(_main())


if __name__ == "__main__":
    raise SystemExit(main())
