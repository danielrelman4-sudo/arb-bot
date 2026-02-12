from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import logging
import math
import random
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

try:
    import websockets
except ImportError:  # pragma: no cover - optional for stream mode
    websockets = None

from arb_bot.config import PolymarketSettings
from arb_bot.binary_math import build_quote_diagnostics, choose_effective_buy_price
from arb_bot.models import BinaryQuote, LegExecutionResult, OrderState, OrderStatus, PairExecutionResult, Side, TradeLegPlan, TradePlan

from .base import ExchangeAdapter

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _BinaryTokenPair:
    market_id: str
    question: str
    yes_token_id: str
    no_token_id: str
    slug: str
    liquidity: float
    volume_24h: float
    volume_total: float
    resolution_ts: float | None
    summary_yes_bid: float | None
    summary_yes_ask: float | None
    summary_yes_mark: float | None
    summary_no_mark: float | None


class PolymarketAdapter(ExchangeAdapter):
    venue = "polymarket"

    def __init__(self, settings: PolymarketSettings, timeout_seconds: float = 10.0) -> None:
        self._settings = settings
        self._gamma = httpx.AsyncClient(
            base_url=settings.gamma_base_url.rstrip("/"),
            timeout=timeout_seconds,
        )
        self._clob = httpx.AsyncClient(
            base_url=settings.clob_base_url.rstrip("/"),
            timeout=timeout_seconds,
        )

        self._live_client: Any = None
        self._order_args_cls: Any = None
        self._order_type_cls: Any = None
        self._buy_constant: Any = "BUY"
        self._live_error: str | None = None
        self._events_fallback_cache_markets: list[dict[str, Any]] = []
        self._events_fallback_cache_ts: float = 0.0
        self._events_fallback_cache_ttl_seconds = 120.0

        self._initialize_live_client()

    def supports_streaming(self) -> bool:
        return bool(self._settings.enable_stream and websockets is not None)

    async def stream_quotes(self) -> AsyncIterator[BinaryQuote]:
        if not self.supports_streaming():
            return

        token_pairs = await self._fetch_market_pairs()
        if not token_pairs:
            return

        token_to_pair: dict[str, tuple[_BinaryTokenPair, Side]] = {}
        for pair in token_pairs:
            token_to_pair[pair.yes_token_id] = (pair, Side.YES)
            token_to_pair[pair.no_token_id] = (pair, Side.NO)

        asset_ids = list(token_to_pair.keys())
        ws_url = self._settings.ws_base_url.rstrip("/") + "/market"

        state: dict[str, dict[str, float | None]] = {}

        subscribe_payload = {
            "type": "MARKET",
            "assets_ids": asset_ids,
            "auth": {},
            "custom_feature_enabled": self._settings.ws_custom_feature_enabled,
        }

        async with websockets.connect(ws_url, ping_interval=20, ping_timeout=20, max_size=None) as socket:
            await socket.send(json.dumps(subscribe_payload))

            while True:
                raw = await socket.recv()
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")

                if raw == "PING":
                    await socket.send("PONG")
                    continue

                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                events = payload if isinstance(payload, list) else [payload]
                for event in events:
                    if not isinstance(event, dict):
                        continue

                    event_type = str(event.get("event_type") or "").lower()
                    if event_type not in {"book", "best_bid_ask", "price_change"}:
                        continue

                    asset_id = str(event.get("asset_id") or "").strip()
                    if not asset_id:
                        continue

                    book_state = self._extract_stream_book_state(event)
                    if book_state is None:
                        continue

                    state[asset_id] = book_state

                    pair_info = token_to_pair.get(asset_id)
                    if pair_info is None:
                        continue

                    pair, _ = pair_info
                    yes_state = state.get(pair.yes_token_id)
                    no_state = state.get(pair.no_token_id)
                    if yes_state is None or no_state is None:
                        continue

                    quote = self._quote_from_stream_pair(pair, yes_state, no_state)
                    if quote is not None:
                        yield quote

    async def fetch_quotes(self) -> list[BinaryQuote]:
        token_pairs = await self._fetch_market_pairs()
        if not token_pairs:
            return []

        # In stream mode, a fast summary snapshot improves early lane coverage
        # without forcing thousands of /book calls before the websocket warms up.
        if self._settings.enable_stream:
            summary_quotes: list[BinaryQuote] = []
            for pair in token_pairs:
                quote = self._quote_from_pair_summary(pair)
                if quote is not None:
                    summary_quotes.append(quote)
            if summary_quotes:
                return summary_quotes

        semaphore = asyncio.Semaphore(max(1, self._settings.max_orderbook_concurrency))

        async def _run(pair: _BinaryTokenPair) -> BinaryQuote | None:
            async with semaphore:
                return await self._quote_market_pair(pair)

        tasks = [_run(pair) for pair in token_pairs]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        quotes: list[BinaryQuote] = []
        for pair, result in zip(token_pairs, results):
            if isinstance(result, Exception):
                LOGGER.warning("polymarket quote failed for %s: %s", pair.market_id, result)
                continue
            if result is not None:
                quotes.append(result)
        return quotes

    async def _fetch_market_pairs(self) -> list[_BinaryTokenPair]:
        wanted = {value.strip() for value in self._settings.market_ids if value.strip()}
        priority_ids = {value.strip().lower() for value in self._settings.priority_market_ids if value.strip()}
        markets = await self._fetch_markets_payload(priority_ids=priority_ids)
        if not markets:
            return []

        markets = sorted(markets, key=self._market_priority_score, reverse=True)

        pairs: list[_BinaryTokenPair] = []
        seen_market_ids: set[str] = set()

        def _collect(pass_priority_only: bool) -> None:
            if len(pairs) >= self._settings.market_limit:
                return
            for market in markets:
                pair = self._parse_binary_pair(market)
                if pair is None:
                    continue
                market_key = pair.market_id.strip().lower()
                is_priority = market_key in priority_ids
                if pass_priority_only and not is_priority:
                    continue
                if not pass_priority_only and is_priority:
                    continue

                if pair.market_id in seen_market_ids:
                    continue

                if wanted and pair.market_id not in wanted and pair.slug not in wanted:
                    continue

                if pair.liquidity < self._settings.min_liquidity:
                    continue

                pairs.append(pair)
                seen_market_ids.add(pair.market_id)
                if len(pairs) >= self._settings.market_limit:
                    break

        # Priority pass: ensure mapped cross/parity markets are included first,
        # then fill remaining capacity from the full universe.
        _collect(pass_priority_only=True)
        _collect(pass_priority_only=False)

        if priority_ids:
            covered = sum(1 for pair in pairs if pair.market_id.strip().lower() in priority_ids)
            LOGGER.info(
                "polymarket priority coverage in pair universe: covered=%d total=%d",
                covered,
                len(priority_ids),
            )

        return pairs

    def _market_priority_score(self, market: dict[str, Any]) -> float:
        liquidity = self._to_size(
            market.get("liquidity")
            or market.get("liquidityNum")
            or market.get("liquidityClob")
            or 0.0
        )
        volume = self._to_size(
            market.get("volume24hr")
            or market.get("volume24h")
            or market.get("volume24")
            or market.get("volume")
            or 0.0
        )

        spread = 0.0
        yes_bid = self._to_price(market.get("bestBid") or market.get("best_bid"))
        yes_ask = self._to_price(market.get("bestAsk") or market.get("best_ask"))
        if yes_bid is not None and yes_ask is not None:
            spread = max(0.0, yes_ask - yes_bid)

        return math.log1p(liquidity) + math.log1p(volume) - spread

    async def _fetch_markets_payload(self, priority_ids: set[str] | None = None) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        wanted_priority = {value.strip().lower() for value in (priority_ids or set()) if value.strip()}
        seen_priority: set[str] = set()

        for page in range(max(1, self._settings.market_scan_pages)):
            params = {
                "active": "true",
                "closed": "false",
                "limit": self._settings.market_page_size,
                "offset": page * self._settings.market_page_size,
            }

            try:
                response = await self._gamma.get("/markets", params=params)
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, list):
                    for item in payload:
                        if not isinstance(item, dict):
                            continue
                        market_id = self._market_identifier(item)
                        if not market_id or market_id in seen_ids:
                            continue
                        seen_ids.add(market_id)
                        market_key = market_id.lower()
                        if market_key in wanted_priority:
                            seen_priority.add(market_key)
                        merged.append(item)
            except Exception as exc:
                LOGGER.debug("gamma /markets page %d failed: %s", page, exc)
                break

            if len(merged) >= self._settings.market_limit * 3:
                break

        if wanted_priority:
            missing_priority = wanted_priority - seen_priority
            if missing_priority:
                priority_markets = await self._fetch_priority_markets_from_events(missing_priority)
                for item in priority_markets:
                    market_id = self._market_identifier(item)
                    if not market_id or market_id in seen_ids:
                        continue
                    seen_ids.add(market_id)
                    merged.append(item)
                    market_key = market_id.lower()
                    if market_key in wanted_priority:
                        seen_priority.add(market_key)
                LOGGER.info(
                    "polymarket priority backfill from /events matched=%d missing=%d",
                    len(seen_priority),
                    len(wanted_priority - seen_priority),
                )

        viable_pairs = self._count_viable_pairs(merged)
        if viable_pairs < self._settings.min_pairs_before_event_fallback:
            LOGGER.debug(
                "polymarket /markets yielded %d viable pairs (< %d), enriching with /events fallback",
                viable_pairs,
                self._settings.min_pairs_before_event_fallback,
            )
            fallback = await self._fetch_markets_from_events_fallback_cached()
            for item in fallback:
                market_id = self._market_identifier(item)
                if not market_id or market_id in seen_ids:
                    continue
                seen_ids.add(market_id)
                merged.append(item)

        return merged

    async def _fetch_priority_markets_from_events(self, priority_ids: set[str]) -> list[dict[str, Any]]:
        if not priority_ids:
            return []

        targets = {value.strip().lower() for value in priority_ids if value.strip()}
        if not targets:
            return []

        max_pages = max(1, int(self._settings.priority_backfill_scan_pages))
        matched: list[dict[str, Any]] = []
        seen_targets: set[str] = set()

        for page in range(max_pages):
            params = {
                "active": "true",
                "closed": "false",
                "limit": self._settings.market_page_size,
                "offset": page * self._settings.market_page_size,
            }
            try:
                response = await self._gamma.get("/events", params=params)
                response.raise_for_status()
                payload = response.json()
            except Exception as exc:
                LOGGER.debug("gamma /events priority backfill page %d failed: %s", page, exc)
                break

            page_events: list[dict[str, Any]] = []
            if isinstance(payload, list):
                page_events = [item for item in payload if isinstance(item, dict)]
            elif isinstance(payload, dict):
                events = payload.get("events")
                if isinstance(events, list):
                    page_events = [item for item in events if isinstance(item, dict)]

            if not page_events:
                break

            for event in page_events:
                event_markets = event.get("markets")
                if not isinstance(event_markets, list):
                    continue
                for market in event_markets:
                    if not isinstance(market, dict):
                        continue
                    market_id = self._market_identifier(market).lower()
                    if not market_id or market_id not in targets or market_id in seen_targets:
                        continue
                    seen_targets.add(market_id)
                    matched.append(market)

            if seen_targets >= targets:
                break

        return matched

    async def _fetch_markets_from_events_fallback_cached(self) -> list[dict[str, Any]]:
        now = time.time()
        age_seconds = now - self._events_fallback_cache_ts
        if self._events_fallback_cache_markets and age_seconds < self._events_fallback_cache_ttl_seconds:
            return list(self._events_fallback_cache_markets)

        fresh = await self._fetch_markets_from_events_fallback()
        if fresh:
            self._events_fallback_cache_markets = list(fresh)
            self._events_fallback_cache_ts = now
        return fresh

    async def _fetch_markets_from_events_fallback(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for page in range(max(1, self._settings.market_scan_pages)):
            params = {
                "active": "true",
                "closed": "false",
                "limit": self._settings.market_page_size,
                "offset": page * self._settings.market_page_size,
            }
            try:
                response = await self._gamma.get("/events", params=params)
                response.raise_for_status()
                payload = response.json()
            except Exception as exc:
                LOGGER.debug("gamma /events page %d failed: %s", page, exc)
                break

            page_events = []
            if isinstance(payload, list):
                page_events = payload
            elif isinstance(payload, dict):
                raw = payload.get("events")
                if isinstance(raw, list):
                    page_events = raw

            if not page_events:
                break

            events.extend([event for event in page_events if isinstance(event, dict)])

        markets: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for event in events:
            event_markets = event.get("markets") if isinstance(event, dict) else None
            if isinstance(event_markets, list):
                for market in event_markets:
                    if not isinstance(market, dict):
                        continue
                    market_id = self._market_identifier(market)
                    if not market_id or market_id in seen_ids:
                        continue
                    seen_ids.add(market_id)
                    markets.append(market)
        return markets

    def _count_viable_pairs(self, markets: list[dict[str, Any]]) -> int:
        count = 0
        for market in markets:
            pair = self._parse_binary_pair(market)
            if pair is None:
                continue
            if pair.liquidity < self._settings.min_liquidity:
                continue
            count += 1
        return count

    @staticmethod
    def _market_identifier(market: dict[str, Any]) -> str:
        return str(
            market.get("conditionId")
            or market.get("condition_id")
            or market.get("market_id")
            or market.get("id")
            or market.get("slug")
            or ""
        ).strip()

    def _parse_binary_pair(self, market: dict[str, Any]) -> _BinaryTokenPair | None:
        outcomes = self._parse_json_array(market.get("outcomes"))
        token_ids = self._parse_json_array(market.get("clobTokenIds"))
        outcome_prices = self._parse_json_array(market.get("outcomePrices"))

        if len(outcomes) != 2 or len(token_ids) != 2:
            return None

        normalized_tokens = [str(value).strip() for value in token_ids]
        if not normalized_tokens[0] or not normalized_tokens[1]:
            return None
        if normalized_tokens[0] == normalized_tokens[1]:
            return None

        outcome_labels = [str(value).strip().lower() for value in outcomes]
        indices = self._resolve_yes_no_indices(
            outcome_labels,
            strict=self._settings.require_strict_yes_no_labels,
        )
        if indices is None:
            return None
        yes_idx, no_idx = indices

        market_id = self._market_identifier(market)
        if not market_id:
            return None

        return _BinaryTokenPair(
            market_id=market_id,
            question=str(market.get("question") or market.get("title") or ""),
            yes_token_id=normalized_tokens[yes_idx],
            no_token_id=normalized_tokens[no_idx],
            slug=str(market.get("slug") or "").strip(),
            liquidity=self._to_size(
                market.get("liquidity")
                or market.get("liquidityNum")
                or market.get("liquidityClob")
                or 0.0
            ),
            volume_24h=self._to_size(
                market.get("volume24hr")
                or market.get("volume24h")
                or market.get("volume24")
                or 0.0
            ),
            volume_total=self._to_size(market.get("volume") or market.get("totalVolume") or 0.0),
            resolution_ts=self._coerce_timestamp(
                market.get("endDate")
                or market.get("endTime")
                or market.get("resolveDate")
                or market.get("resolutionDate")
                or market.get("closedTime")
                or market.get("expiresAt")
            ),
            summary_yes_bid=self._to_price(market.get("bestBid") or market.get("best_bid")),
            summary_yes_ask=self._to_price(market.get("bestAsk") or market.get("best_ask")),
            summary_yes_mark=self._to_price(
                outcome_prices[yes_idx] if len(outcome_prices) == 2 else None
            ),
            summary_no_mark=self._to_price(
                outcome_prices[no_idx] if len(outcome_prices) == 2 else None
            ),
        )

    def _quote_from_pair_summary(self, pair: _BinaryTokenPair) -> BinaryQuote | None:
        yes_bid = pair.summary_yes_bid
        yes_ask = pair.summary_yes_ask if pair.summary_yes_ask is not None else pair.summary_yes_mark
        if yes_bid is None and pair.summary_yes_mark is not None:
            yes_bid = pair.summary_yes_mark

        if pair.summary_no_mark is not None:
            no_ask = pair.summary_no_mark
            no_bid = pair.summary_no_mark
        else:
            no_ask = 1.0 - yes_bid if yes_bid is not None else None
            no_bid = 1.0 - yes_ask if yes_ask is not None else None

        if no_ask is None and no_bid is not None:
            no_ask = no_bid
        if no_bid is None and no_ask is not None:
            no_bid = no_ask
        if yes_ask is None and no_bid is not None:
            yes_ask = 1.0 - no_bid
        if yes_bid is None and no_ask is not None:
            yes_bid = 1.0 - no_ask

        if yes_ask is None or no_ask is None:
            return None

        depth_size = max(1.0, self._settings.quote_depth_contracts)
        yes_effective = choose_effective_buy_price(
            side="yes",
            direct_ask_price=yes_ask,
            direct_ask_size=depth_size,
            opposite_bid_price=no_bid,
            opposite_bid_size=depth_size,
        )
        no_effective = choose_effective_buy_price(
            side="no",
            direct_ask_price=no_ask,
            direct_ask_size=depth_size,
            opposite_bid_price=yes_bid,
            opposite_bid_size=depth_size,
        )
        if yes_effective is None or no_effective is None:
            return None

        yes_maker = self._estimate_maker_buy_price(yes_bid, yes_effective.price)
        no_maker = self._estimate_maker_buy_price(no_bid, no_effective.price)

        metadata = {
            "question": pair.question,
            "canonical_text": pair.question,
            "slug": pair.slug,
            "yes_token_id": pair.yes_token_id,
            "no_token_id": pair.no_token_id,
            "liquidity": pair.liquidity,
            "volume_24h": pair.volume_24h,
            "volume": pair.volume_total,
            "resolution_ts": pair.resolution_ts,
            "yes_bid_price": yes_bid,
            "no_bid_price": no_bid,
            "quote_source": "summary_bootstrap",
        }
        metadata.update(
            self._binary_quote_metadata(
                yes_buy_price=yes_effective.price,
                no_buy_price=no_effective.price,
                yes_bid_price=yes_bid,
                no_bid_price=no_bid,
                yes_buy_source=yes_effective.source,
                no_buy_source=no_effective.source,
            )
        )

        return BinaryQuote(
            venue=self.venue,
            market_id=pair.market_id,
            yes_buy_price=yes_effective.price,
            no_buy_price=no_effective.price,
            yes_buy_size=yes_effective.size,
            no_buy_size=no_effective.size,
            yes_bid_price=yes_bid,
            no_bid_price=no_bid,
            yes_bid_size=depth_size,
            no_bid_size=depth_size,
            yes_maker_buy_price=yes_maker,
            no_maker_buy_price=no_maker,
            yes_maker_buy_size=depth_size,
            no_maker_buy_size=depth_size,
            fee_per_contract=self._settings.taker_fee_per_contract,
            metadata=metadata,
        )

    async def _quote_market_pair(self, pair: _BinaryTokenPair) -> BinaryQuote | None:
        yes_book_task = self._fetch_top_book(pair.yes_token_id)
        no_book_task = self._fetch_top_book(pair.no_token_id)

        yes_book, no_book = await asyncio.gather(yes_book_task, no_book_task)

        yes_ask, yes_ask_size, yes_bid, yes_bid_size, yes_book_ts = yes_book
        no_ask, no_ask_size, no_bid, no_bid_size, no_book_ts = no_book

        if not self._pair_is_fresh(
            yes_book_ts,
            no_book_ts,
            self._settings.max_pair_staleness_seconds,
        ):
            return None

        yes_effective = choose_effective_buy_price(
            side="yes",
            direct_ask_price=yes_ask,
            direct_ask_size=yes_ask_size,
            opposite_bid_price=no_bid,
            opposite_bid_size=no_bid_size,
        )
        no_effective = choose_effective_buy_price(
            side="no",
            direct_ask_price=no_ask,
            direct_ask_size=no_ask_size,
            opposite_bid_price=yes_bid,
            opposite_bid_size=yes_bid_size,
        )
        if yes_effective is None or no_effective is None:
            return None

        yes_maker = self._estimate_maker_buy_price(yes_bid, yes_effective.price)
        no_maker = self._estimate_maker_buy_price(no_bid, no_effective.price)

        metadata = {
            "question": pair.question,
            "canonical_text": pair.question,
            "slug": pair.slug,
            "yes_token_id": pair.yes_token_id,
            "no_token_id": pair.no_token_id,
            "liquidity": pair.liquidity,
            "volume_24h": pair.volume_24h,
            "volume": pair.volume_total,
            "resolution_ts": pair.resolution_ts,
            "yes_bid_price": yes_bid,
            "no_bid_price": no_bid,
        }
        metadata.update(
            self._binary_quote_metadata(
                yes_buy_price=yes_effective.price,
                no_buy_price=no_effective.price,
                yes_bid_price=yes_bid,
                no_bid_price=no_bid,
                yes_buy_source=yes_effective.source,
                no_buy_source=no_effective.source,
            )
        )

        return BinaryQuote(
            venue=self.venue,
            market_id=pair.market_id,
            yes_buy_price=yes_effective.price,
            no_buy_price=no_effective.price,
            yes_buy_size=yes_effective.size,
            no_buy_size=no_effective.size,
            yes_bid_price=yes_bid,
            no_bid_price=no_bid,
            yes_bid_size=yes_bid_size,
            no_bid_size=no_bid_size,
            yes_maker_buy_price=yes_maker,
            no_maker_buy_price=no_maker,
            yes_maker_buy_size=yes_bid_size,
            no_maker_buy_size=no_bid_size,
            fee_per_contract=self._settings.taker_fee_per_contract,
            metadata=metadata,
        )

    async def _fetch_top_book(self, token_id: str) -> tuple[float | None, float, float | None, float, float]:
        attempts = max(1, self._settings.book_retry_attempts)
        base_delay = max(0.0, self._settings.book_retry_base_delay_seconds)
        max_delay = max(base_delay, self._settings.book_retry_max_delay_seconds)

        for attempt in range(attempts):
            response = await self._clob.get("/book", params={"token_id": token_id})

            if response.status_code == 429:
                if attempt >= attempts - 1:
                    LOGGER.debug("polymarket /book rate-limited after retries for token %s", token_id)
                    return None, 0.0, None, 0.0, time.time()

                retry_after_raw = response.headers.get("Retry-After")
                retry_after_seconds: float | None = None
                if retry_after_raw:
                    try:
                        retry_after_seconds = max(0.0, float(retry_after_raw))
                    except ValueError:
                        retry_after_seconds = None

                if retry_after_seconds is None:
                    exp_backoff = base_delay * (2**attempt)
                    jitter = random.uniform(0.0, base_delay) if base_delay > 0 else 0.0
                    retry_after_seconds = min(max_delay, exp_backoff + jitter)

                await asyncio.sleep(retry_after_seconds)
                continue

            response.raise_for_status()

            payload = response.json()
            asks = payload.get("asks") if isinstance(payload, dict) else None
            bids = payload.get("bids") if isinstance(payload, dict) else None

            depth_target = max(0.0, self._settings.quote_depth_contracts)
            ask_price, ask_size = self._extract_depth_level(asks, side="ask", target_size=depth_target)
            bid_price, bid_size = self._extract_depth_level(bids, side="bid", target_size=depth_target)
            fetched_at = time.time()
            return ask_price, ask_size, bid_price, bid_size, fetched_at

        return None, 0.0, None, 0.0, time.time()

    async def place_pair_order(self, plan: TradePlan) -> PairExecutionResult:
        yes_plan = next((leg for leg in plan.legs if leg.side is Side.YES), None)
        no_plan = next((leg for leg in plan.legs if leg.side is Side.NO), None)

        if yes_plan is None or no_plan is None:
            return PairExecutionResult(
                venue=plan.venue or self.venue,
                market_id=plan.market_id,
                yes_leg=LegExecutionResult(False, None, plan.contracts, 0, None),
                no_leg=LegExecutionResult(False, None, plan.contracts, 0, None),
                error="plan missing yes/no legs",
            )

        yes_task = self.place_single_order(yes_plan)
        no_task = self.place_single_order(no_plan)
        yes_leg, no_leg = await asyncio.gather(yes_task, no_task)

        error = None
        if not yes_leg.success or not no_leg.success:
            error = "one or more legs failed"

        return PairExecutionResult(
            venue=yes_plan.venue,
            market_id=yes_plan.market_id,
            yes_leg=yes_leg,
            no_leg=no_leg,
            error=error,
        )

    async def place_single_order(self, leg: TradeLegPlan) -> LegExecutionResult:
        if self._live_client is None:
            return LegExecutionResult(
                success=False,
                order_id=None,
                requested_contracts=leg.contracts,
                filled_contracts=0,
                average_price=None,
                raw={"error": self._live_error or "polymarket live client unavailable"},
            )

        token_id = self._token_id_for_leg(leg)
        if token_id is None:
            return LegExecutionResult(
                success=False,
                order_id=None,
                requested_contracts=leg.contracts,
                filled_contracts=0,
                average_price=None,
                raw={"error": "missing token id for leg"},
            )

        try:
            result = await asyncio.to_thread(
                self._submit_buy_order,
                token_id,
                leg.limit_price,
                leg.contracts,
            )
        except Exception as exc:
            return LegExecutionResult(
                success=False,
                order_id=None,
                requested_contracts=leg.contracts,
                filled_contracts=0,
                average_price=None,
                raw={"error": str(exc)},
            )

        return self._to_leg_result(result, leg.contracts)

    async def cancel_order(self, order_id: str) -> bool:
        if self._live_client is None:
            return False
        try:
            result = await asyncio.to_thread(self._live_client.cancel, order_id)
            if isinstance(result, dict):
                return not result.get("not_canceled", False)
            return True
        except Exception as exc:
            LOGGER.warning("polymarket cancel_order failed for %s: %s", order_id, exc)
            return False

    async def get_order_status(self, order_id: str) -> OrderStatus | None:
        if self._live_client is None:
            return None
        try:
            result = await asyncio.to_thread(self._live_client.get_order, order_id)
        except Exception as exc:
            LOGGER.warning("polymarket get_order_status failed for %s: %s", order_id, exc)
            return None

        if not isinstance(result, dict):
            return None

        status_str = str(result.get("status", "")).lower()
        filled = 0
        remaining = 0
        try:
            size_matched = float(result.get("size_matched", 0) or 0)
            original_size = float(result.get("original_size", 0) or result.get("size", 0) or 0)
            filled = int(size_matched)
            remaining = max(0, int(original_size) - filled)
        except (TypeError, ValueError):
            pass

        avg_price = result.get("associate_trades", {}).get("avg_price") or result.get("price")
        if avg_price is not None:
            try:
                avg_price = float(avg_price)
            except (TypeError, ValueError):
                avg_price = None

        state_map = {
            "live": OrderState.OPEN,
            "matched": OrderState.FILLED,
            "canceled": OrderState.CANCELLED,
            "cancelled": OrderState.CANCELLED,
            "expired": OrderState.EXPIRED,
        }
        state = state_map.get(status_str, OrderState.UNKNOWN)
        if filled > 0 and remaining > 0 and state not in (OrderState.CANCELLED, OrderState.EXPIRED):
            state = OrderState.PARTIALLY_FILLED

        return OrderStatus(
            order_id=order_id,
            state=state,
            filled_contracts=filled,
            remaining_contracts=remaining,
            average_price=avg_price,
            raw=result,
        )

    def _token_id_for_leg(self, leg: TradeLegPlan) -> str | None:
        if leg.side is Side.YES:
            token = leg.metadata.get("yes_token_id")
        else:
            token = leg.metadata.get("no_token_id")

        if token is None:
            return None
        token_str = str(token).strip()
        return token_str or None

    def _submit_buy_order(self, token_id: str, price: float, contracts: int) -> Any:
        assert self._live_client is not None
        assert self._order_args_cls is not None

        args = self._order_args_cls(
            token_id=token_id,
            price=float(price),
            size=float(contracts),
            side=self._buy_constant,
        )
        signed = self._live_client.create_order(args)

        if self._order_type_cls is not None:
            if hasattr(self._order_type_cls, "FOK"):
                return self._live_client.post_order(signed, self._order_type_cls.FOK)
            if hasattr(self._order_type_cls, "GTC"):
                return self._live_client.post_order(signed, self._order_type_cls.GTC)

        return self._live_client.post_order(signed)

    def _initialize_live_client(self) -> None:
        if not self._settings.private_key:
            self._live_error = "POLYMARKET_PRIVATE_KEY missing"
            return

        try:
            client_mod = importlib.import_module("py_clob_client.client")
            types_mod = importlib.import_module("py_clob_client.clob_types")
            constants_mod = importlib.import_module("py_clob_client.order_builder.constants")

            clob_client_cls = getattr(client_mod, "ClobClient")
            self._order_args_cls = getattr(types_mod, "OrderArgs")
            self._order_type_cls = getattr(types_mod, "OrderType", None)
            self._buy_constant = getattr(constants_mod, "BUY", "BUY")

            kwargs = {
                "key": self._settings.private_key,
                "chain_id": self._settings.chain_id,
            }
            if self._settings.funder:
                kwargs["funder"] = self._settings.funder

            signature = inspect.signature(clob_client_cls)
            if "host" in signature.parameters:
                kwargs["host"] = self._settings.clob_base_url

            try:
                self._live_client = clob_client_cls(**kwargs)
            except TypeError:
                kwargs.pop("funder", None)
                self._live_client = clob_client_cls(**kwargs)

            if self._settings.api_key and self._settings.api_secret and self._settings.api_passphrase:
                creds = {
                    "key": self._settings.api_key,
                    "secret": self._settings.api_secret,
                    "passphrase": self._settings.api_passphrase,
                }
                try:
                    self._live_client.set_api_creds(creds)
                except Exception:
                    LOGGER.warning("could not set explicit polymarket api creds; trying derived creds")
                    self._live_client.set_api_creds(self._live_client.create_or_derive_api_creds())
            else:
                self._live_client.set_api_creds(self._live_client.create_or_derive_api_creds())
        except Exception as exc:
            self._live_client = None
            self._live_error = str(exc)

    def _quote_from_stream_pair(
        self,
        pair: _BinaryTokenPair,
        yes_state: dict[str, float | None],
        no_state: dict[str, float | None],
    ) -> BinaryQuote | None:
        if not self._pair_is_fresh(
            yes_state.get("updated_at"),
            no_state.get("updated_at"),
            self._settings.max_pair_staleness_seconds,
        ):
            return None

        yes_ask = yes_state.get("ask")
        no_ask = no_state.get("ask")
        yes_bid = yes_state.get("bid")
        no_bid = no_state.get("bid")

        yes_effective = choose_effective_buy_price(
            side="yes",
            direct_ask_price=yes_ask,
            direct_ask_size=self._to_size(yes_state.get("ask_size", 0.0)),
            opposite_bid_price=no_bid,
            opposite_bid_size=self._to_size(no_state.get("bid_size", 0.0)),
        )
        no_effective = choose_effective_buy_price(
            side="no",
            direct_ask_price=no_ask,
            direct_ask_size=self._to_size(no_state.get("ask_size", 0.0)),
            opposite_bid_price=yes_bid,
            opposite_bid_size=self._to_size(yes_state.get("bid_size", 0.0)),
        )
        if yes_effective is None or no_effective is None:
            return None

        yes_maker = self._estimate_maker_buy_price(yes_bid, yes_effective.price)
        no_maker = self._estimate_maker_buy_price(no_bid, no_effective.price)

        metadata = {
            "question": pair.question,
            "canonical_text": pair.question,
            "slug": pair.slug,
            "yes_token_id": pair.yes_token_id,
            "no_token_id": pair.no_token_id,
            "liquidity": pair.liquidity,
            "volume_24h": pair.volume_24h,
            "volume": pair.volume_total,
            "resolution_ts": pair.resolution_ts,
            "yes_bid_price": yes_bid,
            "no_bid_price": no_bid,
        }
        metadata.update(
            self._binary_quote_metadata(
                yes_buy_price=yes_effective.price,
                no_buy_price=no_effective.price,
                yes_bid_price=yes_bid,
                no_bid_price=no_bid,
                yes_buy_source=yes_effective.source,
                no_buy_source=no_effective.source,
            )
        )

        return BinaryQuote(
            venue=self.venue,
            market_id=pair.market_id,
            yes_buy_price=yes_effective.price,
            no_buy_price=no_effective.price,
            yes_buy_size=yes_effective.size,
            no_buy_size=no_effective.size,
            yes_bid_price=yes_bid,
            no_bid_price=no_bid,
            yes_bid_size=self._to_size(yes_state.get("bid_size", 0.0)),
            no_bid_size=self._to_size(no_state.get("bid_size", 0.0)),
            yes_maker_buy_price=yes_maker,
            no_maker_buy_price=no_maker,
            yes_maker_buy_size=self._to_size(yes_state.get("bid_size", 0.0)),
            no_maker_buy_size=self._to_size(no_state.get("bid_size", 0.0)),
            fee_per_contract=self._settings.taker_fee_per_contract,
            metadata=metadata,
        )

    def _extract_stream_book_state(self, event: dict[str, Any]) -> dict[str, float | None] | None:
        event_ts = self._extract_event_timestamp(event)
        if "best_bid" in event and "best_ask" in event:
            bid = self._to_price(event.get("best_bid"))
            ask = self._to_price(event.get("best_ask"))
            if bid is None and ask is None:
                return None
            return {
                "bid": bid,
                "ask": ask,
                "bid_size": self._to_size(event.get("best_bid_size") or 0.0),
                "ask_size": self._to_size(event.get("best_ask_size") or 0.0),
                "updated_at": event_ts,
            }

        bids = event.get("bids") or event.get("buys")
        asks = event.get("asks") or event.get("sells")
        depth_target = max(0.0, self._settings.quote_depth_contracts)
        ask_price, ask_size = self._extract_depth_level(asks, side="ask", target_size=depth_target)
        bid_price, bid_size = self._extract_depth_level(bids, side="bid", target_size=depth_target)
        if ask_price is None and bid_price is None:
            return None

        return {
            "bid": bid_price,
            "ask": ask_price,
            "bid_size": bid_size,
            "ask_size": ask_size,
            "updated_at": event_ts,
        }

    def _estimate_maker_buy_price(self, bid: float | None, taker_ask: float | None) -> float | None:
        if bid is None or taker_ask is None:
            return None

        candidate = bid + self._settings.maker_tick_size
        floor = max(
            0.0,
            taker_ask - (self._settings.maker_tick_size * self._settings.maker_aggressiveness_ticks),
        )
        candidate = max(candidate, floor)
        candidate = min(candidate, taker_ask)

        if candidate < 0 or candidate > 1:
            return None
        return candidate

    @staticmethod
    def _binary_quote_metadata(
        yes_buy_price: float,
        no_buy_price: float,
        yes_bid_price: float | None,
        no_bid_price: float | None,
        yes_buy_source: str,
        no_buy_source: str,
    ) -> dict[str, float | str | None]:
        diagnostics = build_quote_diagnostics(
            yes_buy_price=yes_buy_price,
            no_buy_price=no_buy_price,
            yes_bid_price=yes_bid_price,
            no_bid_price=no_bid_price,
        )
        return {
            "yes_buy_source": yes_buy_source,
            "no_buy_source": no_buy_source,
            "ask_implied_probability": diagnostics.ask_implied_probability,
            "ask_edge_per_side": diagnostics.ask_edge_per_side,
            "bid_implied_probability": diagnostics.bid_implied_probability,
            "bid_edge_per_side": diagnostics.bid_edge_per_side,
            "midpoint_consistency_gap": diagnostics.midpoint_consistency_gap,
            "yes_spread": diagnostics.yes_spread,
            "no_spread": diagnostics.no_spread,
            "spread_asymmetry": diagnostics.spread_asymmetry,
        }

    @staticmethod
    def _extract_best_level(
        levels: Any,
        side: str,
    ) -> tuple[float | None, float]:
        if not isinstance(levels, list) or not levels:
            return None, 0.0

        best_price: float | None = None
        best_size = 0.0

        for level in levels:
            if not isinstance(level, dict):
                continue
            price = PolymarketAdapter._to_price(level.get("price"))
            if price is None:
                continue
            size = PolymarketAdapter._to_size(level.get("size") or level.get("amount") or level.get("quantity"))

            if best_price is None:
                best_price, best_size = price, size
                continue

            if side == "ask":
                if price < best_price:
                    best_price, best_size = price, size
            else:
                if price > best_price:
                    best_price, best_size = price, size

        if best_price is None:
            return None, 0.0

        return best_price, best_size

    @staticmethod
    def _extract_depth_level(
        levels: Any,
        side: str,
        target_size: float,
    ) -> tuple[float | None, float]:
        if not isinstance(levels, list) or not levels:
            return None, 0.0

        parsed: list[tuple[float, float]] = []
        for level in levels:
            if not isinstance(level, dict):
                continue
            price = PolymarketAdapter._to_price(level.get("price"))
            if price is None:
                continue
            size = PolymarketAdapter._to_size(level.get("size") or level.get("amount") or level.get("quantity"))
            if size <= 0:
                continue
            parsed.append((price, size))

        if not parsed:
            return None, 0.0

        if side == "ask":
            parsed.sort(key=lambda item: item[0])
        else:
            parsed.sort(key=lambda item: item[0], reverse=True)

        required_size = max(0.0, target_size)
        if required_size <= 0:
            top_price, top_size = parsed[0]
            return top_price, top_size

        filled = 0.0
        notional = 0.0
        remaining = required_size

        for price, size in parsed:
            take = min(size, remaining)
            if take <= 0:
                continue
            filled += take
            notional += price * take
            remaining -= take
            if remaining <= 1e-9:
                break

        if filled <= 0:
            return None, 0.0

        return notional / filled, filled

    @staticmethod
    def _resolve_yes_no_indices(
        outcome_labels: list[str],
        strict: bool,
    ) -> tuple[int, int] | None:
        if len(outcome_labels) != 2:
            return None

        labels = [label.strip().lower() for label in outcome_labels]
        if labels[0] == labels[1]:
            return None

        if strict:
            if set(labels) != {"yes", "no"}:
                return None
            return labels.index("yes"), labels.index("no")

        yes_aliases = {"yes", "true", "1", "up"}
        no_aliases = {"no", "false", "0", "down"}

        first, second = labels
        if first in yes_aliases and second in no_aliases:
            return 0, 1
        if first in no_aliases and second in yes_aliases:
            return 1, 0
        if set(labels) == {"yes", "no"}:
            return labels.index("yes"), labels.index("no")
        return 0, 1

    @staticmethod
    def _pair_is_fresh(
        left_updated_at: float | None,
        right_updated_at: float | None,
        max_skew_seconds: float,
    ) -> bool:
        if left_updated_at is None or right_updated_at is None:
            return True
        threshold = max(0.0, max_skew_seconds)
        return abs(left_updated_at - right_updated_at) <= threshold

    @staticmethod
    def _extract_event_timestamp(event: dict[str, Any]) -> float:
        candidates = (
            event.get("timestamp"),
            event.get("updated_at"),
            event.get("time"),
            event.get("ts"),
            event.get("event_time"),
        )
        for value in candidates:
            parsed = PolymarketAdapter._coerce_timestamp(value)
            if parsed is not None:
                return parsed
        return time.time()

    @staticmethod
    def _coerce_timestamp(value: Any) -> float | None:
        if value is None:
            return None

        if isinstance(value, (int, float)):
            numeric = float(value)
            if numeric > 1e18:
                numeric /= 1_000_000_000.0
            elif numeric > 1e15:
                numeric /= 1_000_000.0
            elif numeric > 1e12:
                numeric /= 1_000.0
            if numeric <= 0:
                return None
            return numeric

        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None

            try:
                numeric = float(text)
                return PolymarketAdapter._coerce_timestamp(numeric)
            except ValueError:
                pass

            try:
                parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError:
                return None
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.timestamp()

        return None

    @staticmethod
    def _parse_json_array(value: Any) -> list[Any]:
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return []
            return parsed if isinstance(parsed, list) else []
        return []

    @staticmethod
    def _to_price(value: Any) -> float | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if numeric > 1:
            numeric /= 100.0
        if numeric < 0:
            return None
        return min(1.0, numeric)

    @staticmethod
    def _to_size(value: Any) -> float:
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _to_leg_result(result: Any, requested_contracts: int) -> LegExecutionResult:
        if isinstance(result, Exception):
            return LegExecutionResult(
                success=False,
                order_id=None,
                requested_contracts=requested_contracts,
                filled_contracts=0,
                average_price=None,
                raw={"error": str(result)},
            )

        raw = result if isinstance(result, dict) else {"data": str(result)}
        order_id = raw.get("orderID") or raw.get("order_id") or raw.get("id")

        filled = raw.get("filled") or raw.get("matched") or raw.get("size_matched") or requested_contracts
        try:
            filled_contracts = int(float(filled))
        except (TypeError, ValueError):
            filled_contracts = 0

        avg_price = raw.get("avgPrice") or raw.get("average_price")
        try:
            avg_price = float(avg_price) if avg_price is not None else None
        except (TypeError, ValueError):
            avg_price = None

        success = bool(order_id)

        return LegExecutionResult(
            success=success,
            order_id=str(order_id) if order_id else None,
            requested_contracts=requested_contracts,
            filled_contracts=filled_contracts,
            average_price=avg_price,
            raw=raw,
        )

    async def aclose(self) -> None:
        await self._gamma.aclose()
        await self._clob.aclose()
