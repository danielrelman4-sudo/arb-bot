from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import os
import socket
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import suppress
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import urlparse

import httpx
try:
    import websockets
except ImportError:  # pragma: no cover - optional for stream mode
    websockets = None

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
except ImportError:  # pragma: no cover - optional until live mode is enabled
    hashes = None
    serialization = None
    padding = None

from arb_bot.config import KalshiSettings
from arb_bot.binary_math import build_quote_diagnostics, choose_effective_buy_price
from arb_bot.models import BinaryQuote, LegExecutionResult, OrderState, OrderStatus, PairExecutionResult, Side, TradeLegPlan, TradePlan

from .base import ExchangeAdapter

LOGGER = logging.getLogger(__name__)


class KalshiAdapter(ExchangeAdapter):
    venue = "kalshi"

    def __init__(self, settings: KalshiSettings, timeout_seconds: float = 10.0) -> None:
        self._settings = settings
        self._timeout_seconds = timeout_seconds
        self._api_failover_lock = asyncio.Lock()
        self._ws_failover_lock = asyncio.Lock()
        self._api_base_urls = self._build_kalshi_host_candidates(settings.api_base_url)
        self._ws_urls = self._build_kalshi_ws_candidates(settings.ws_url)
        self._api_base_index = 0
        self._ws_index = 0
        self._client = httpx.AsyncClient(
            base_url=self._active_api_base_url(),
            timeout=timeout_seconds,
        )
        self._private_key = self._load_private_key()
        self._discovery_cache_markets: list[dict[str, Any]] = []
        self._discovery_cache_ts: float = 0.0
        self._discovery_cache_ttl_seconds = 300.0
        self._events_429_consecutive = 0
        self._events_circuit_open_until_ts = 0.0
        self._stream_request_seq = 0
        self._last_api_failover_ts = 0.0
        self._last_ws_failover_ts = 0.0
        self._failover_min_interval_seconds = 2.0
        self._priority_refresh_cursor = 0
        self._request_base_interval_seconds = max(0.0, float(self._settings.request_pause_seconds))
        self._request_min_interval_seconds = max(
            0.01,
            self._request_base_interval_seconds * 0.25 if self._request_base_interval_seconds > 0 else 0.01,
        )
        self._request_max_interval_seconds = max(2.0, self._request_base_interval_seconds * 12.0)
        self._request_throttle_locks: dict[str, asyncio.Lock] = {}
        self._endpoint_next_request_ts: dict[str, float] = {}
        self._endpoint_interval_seconds: dict[str, float] = {}
        self._endpoint_success_streak: dict[str, int] = {}
        self._dynamic_priority_refresh_limit = max(1, int(self._settings.stream_priority_refresh_limit))
        self._dynamic_priority_limit_success_streak = 0

    async def fetch_quotes(self) -> list[BinaryQuote]:
        return await self._fetch_quotes_impl(force_full_scan=False)

    async def fetch_quotes_full_scan(self) -> list[BinaryQuote]:
        return await self._fetch_quotes_impl(force_full_scan=True)

    async def _fetch_quotes_impl(self, force_full_scan: bool) -> list[BinaryQuote]:
        if self._settings.market_tickers:
            return await self._fetch_ticker_quotes(self._settings.market_tickers)
        if (
            not force_full_scan
            and self._settings.enable_stream
            and self._settings.stream_priority_tickers
        ):
            # Keep stream-mode REST refresh bounded to avoid 429 storms on expanded universes.
            priority_limit = max(
                1,
                min(self._dynamic_priority_refresh_limit, max(1, self._settings.market_limit)),
            )
            priority_universe = self._dedupe_tickers(self._settings.stream_priority_tickers)
            priority_tickers = self._select_priority_refresh_tickers(priority_universe, priority_limit)
            if priority_tickers:
                return await self._fetch_ticker_quotes(priority_tickers)

        summaries = await self._fetch_market_summaries_paged()
        if not summaries:
            summaries = await self._fetch_market_summaries_from_events_cached()
        if not summaries:
            return []

        if self._settings.use_orderbook_quotes:
            quotes = await self._fetch_quotes_from_market_summaries(summaries)
        else:
            quotes = []
            for market in summaries:
                quote = self._quote_from_market_summary(market)
                if quote is not None and self._quote_passes_filters(quote, market):
                    quotes.append(quote)

        return quotes[: self._settings.market_limit]

    def supports_streaming(self) -> bool:
        return bool(self._settings.enable_stream and websockets is not None and self._has_stream_auth())

    async def stream_quotes(self) -> AsyncIterator[BinaryQuote]:
        if not self.supports_streaming():
            return

        bootstrap_timeout_seconds = max(2.0, self._timeout_seconds * 0.8)
        try:
            market_cache = await asyncio.wait_for(
                self._build_stream_market_cache(),
                timeout=bootstrap_timeout_seconds,
            )
        except asyncio.TimeoutError:
            LOGGER.warning(
                "kalshi stream bootstrap cache timed out after %.1fs; continuing with empty cache",
                bootstrap_timeout_seconds,
            )
            market_cache = {}
        except Exception as exc:
            # Discovery endpoints can be rate-limited; websocket streaming should
            # still proceed so we can recover from live ticker flow.
            LOGGER.warning("kalshi stream bootstrap cache failed; continuing with empty cache: %s", exc)
            market_cache = {}
        market_lookup = self._build_stream_market_lookup(market_cache)
        subscription_tickers = self._stream_subscription_tickers(market_cache)
        shards = self._build_stream_subscription_shards(subscription_tickers)
        queue: asyncio.Queue[BinaryQuote] = asyncio.Queue()
        shard_tasks: list[asyncio.Task[None]] = []

        seeded = 0
        for market in market_cache.values():
            quote = self._quote_from_market_summary(market)
            if quote is None:
                continue
            if not self._quote_passes_filters(quote, market):
                continue
            queue.put_nowait(quote)
            seeded += 1
        if seeded > 0:
            LOGGER.info("kalshi stream seeded %d quotes from bootstrap cache", seeded)

        for shard_index, shard_tickers in enumerate(shards, start=1):
            shard_tasks.append(
                asyncio.create_task(
                    self._run_stream_shard(
                        shard_index=shard_index,
                        shard_tickers=shard_tickers,
                        market_cache=market_cache,
                        market_lookup=market_lookup,
                        output_queue=queue,
                    )
                )
            )

        try:
            while True:
                quote = await queue.get()
                yield quote
        finally:
            for task in shard_tasks:
                task.cancel()
            await asyncio.gather(*shard_tasks, return_exceptions=True)

    async def _run_stream_shard(
        self,
        shard_index: int,
        shard_tickers: tuple[str, ...],
        market_cache: dict[str, dict[str, Any]],
        market_lookup: dict[str, str],
        output_queue: asyncio.Queue[BinaryQuote],
    ) -> None:
        reconnect_delay = max(0.5, self._settings.stream_reconnect_delay_seconds)
        ping_interval = max(5.0, self._settings.stream_ping_interval_seconds)

        while True:
            ws_url = self._active_ws_url()
            try:
                ws_headers = self._ws_auth_headers(ws_url)
                connect_kwargs: dict[str, Any] = {
                    "uri": ws_url,
                    "ping_interval": ping_interval,
                    "ping_timeout": ping_interval,
                    "max_size": None,
                }
                if ws_headers:
                    connect_kwargs["additional_headers"] = ws_headers

                try:
                    socket_ctx = websockets.connect(**connect_kwargs)
                except TypeError:
                    connect_kwargs.pop("additional_headers", None)
                    if ws_headers:
                        connect_kwargs["extra_headers"] = ws_headers
                    socket_ctx = websockets.connect(**connect_kwargs)

                async with socket_ctx as socket:
                    subscribed = await self._subscribe_shard(
                        socket=socket,
                        ws_url=ws_url,
                        shard_index=shard_index,
                        shard_tickers=shard_tickers,
                        market_cache=market_cache,
                        market_lookup=market_lookup,
                        output_queue=output_queue,
                    )
                    if not subscribed:
                        raise RuntimeError(
                            f"kalshi shard subscription failed shard={shard_index} tickers={len(shard_tickers)}"
                        )

                    while True:
                        raw = await socket.recv()
                        if isinstance(raw, bytes):
                            raw = raw.decode("utf-8")

                        if raw == "PING":
                            await socket.send("PONG")
                            continue

                        frames = self._parse_stream_frames(raw)
                        await self._emit_quotes_from_frames(
                            frames=frames,
                            market_cache=market_cache,
                            market_lookup=market_lookup,
                            output_queue=output_queue,
                        )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - network dependent
                LOGGER.warning(
                    "kalshi stream failed shard=%d tickers=%d ws_url=%s: %s",
                    shard_index,
                    len(shard_tickers),
                    ws_url,
                    exc,
                )
                await self._rotate_ws_url_if_needed(exc)
                await asyncio.sleep(reconnect_delay)

    async def _subscribe_shard(
        self,
        socket: Any,
        ws_url: str,
        shard_index: int,
        shard_tickers: tuple[str, ...],
        market_cache: dict[str, dict[str, Any]],
        market_lookup: dict[str, str],
        output_queue: asyncio.Queue[BinaryQuote],
    ) -> bool:
        subscription_batch_size = max(1, int(self._settings.stream_subscription_batch_size))
        batches: list[list[str]]
        if shard_tickers:
            batches = [
                list(shard_tickers[index : index + subscription_batch_size])
                for index in range(0, len(shard_tickers), subscription_batch_size)
            ]
        else:
            batches = [[]]

        total_batches = len(batches)
        for batch_index, batch in enumerate(batches, start=1):
            ok = await self._subscribe_batch_with_retry(
                socket=socket,
                ws_url=ws_url,
                shard_index=shard_index,
                batch_index=batch_index,
                total_batches=total_batches,
                batch=batch,
                market_cache=market_cache,
                market_lookup=market_lookup,
                output_queue=output_queue,
            )
            if not ok:
                return False
        return True

    async def _subscribe_batch_with_retry(
        self,
        socket: Any,
        ws_url: str,
        shard_index: int,
        batch_index: int,
        total_batches: int,
        batch: list[str],
        market_cache: dict[str, dict[str, Any]],
        market_lookup: dict[str, str],
        output_queue: asyncio.Queue[BinaryQuote],
    ) -> bool:
        max_attempts = max(1, int(self._settings.stream_subscription_retry_attempts))
        ack_timeout = max(0.5, float(self._settings.stream_subscription_ack_timeout_seconds))

        for attempt in range(1, max_attempts + 1):
            request_id = self._next_stream_request_id()
            payload: dict[str, Any] = {
                "id": request_id,
                "cmd": "subscribe",
                "params": {"channels": ["ticker"]},
            }
            if batch:
                payload["params"]["market_tickers"] = batch

            await socket.send(json.dumps(payload))

            signal = await self._await_subscribe_signal(
                socket=socket,
                request_id=request_id,
                timeout_seconds=ack_timeout,
                market_cache=market_cache,
                market_lookup=market_lookup,
                output_queue=output_queue,
            )
            if signal:
                LOGGER.info(
                    "kalshi stream subscribed shard=%d batch=%d/%d size=%d attempt=%d ws_url=%s",
                    shard_index,
                    batch_index,
                    total_batches,
                    len(batch),
                    attempt,
                    ws_url,
                )
                return True

            await asyncio.sleep(min(1.0, 0.1 * attempt))

        if len(batch) > 1:
            mid = len(batch) // 2
            left = batch[:mid]
            right = batch[mid:]
            LOGGER.warning(
                "kalshi stream splitting subscribe batch shard=%d batch=%d/%d size=%d after retries",
                shard_index,
                batch_index,
                total_batches,
                len(batch),
            )
            left_ok = await self._subscribe_batch_with_retry(
                socket=socket,
                ws_url=ws_url,
                shard_index=shard_index,
                batch_index=batch_index,
                total_batches=total_batches,
                batch=left,
                market_cache=market_cache,
                market_lookup=market_lookup,
                output_queue=output_queue,
            )
            if not left_ok:
                return False
            return await self._subscribe_batch_with_retry(
                socket=socket,
                ws_url=ws_url,
                shard_index=shard_index,
                batch_index=batch_index,
                total_batches=total_batches,
                batch=right,
                market_cache=market_cache,
                market_lookup=market_lookup,
                output_queue=output_queue,
            )

        LOGGER.warning(
            "kalshi stream failed to subscribe shard=%d batch=%d/%d size=%d ws_url=%s",
            shard_index,
            batch_index,
            total_batches,
            len(batch),
            ws_url,
        )
        return False

    async def _await_subscribe_signal(
        self,
        socket: Any,
        request_id: int,
        timeout_seconds: float,
        market_cache: dict[str, dict[str, Any]],
        market_lookup: dict[str, str],
        output_queue: asyncio.Queue[BinaryQuote],
    ) -> bool:
        deadline = time.monotonic() + timeout_seconds
        saw_ticker = False

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return saw_ticker

            try:
                raw = await asyncio.wait_for(socket.recv(), timeout=remaining)
            except asyncio.TimeoutError:
                return saw_ticker

            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            if raw == "PING":
                await socket.send("PONG")
                continue

            frames = self._parse_stream_frames(raw)
            for frame in frames:
                if self._is_subscribe_error_frame(frame, request_id):
                    LOGGER.warning("kalshi stream subscribe error frame: %s", frame)
                    return False
                if self._is_subscribe_success_frame(frame, request_id):
                    return True

            if await self._emit_quotes_from_frames(
                frames=frames,
                market_cache=market_cache,
                market_lookup=market_lookup,
                output_queue=output_queue,
            ):
                saw_ticker = True

    def _build_stream_subscription_shards(self, tickers: list[str]) -> list[tuple[str, ...]]:
        if not tickers:
            return [tuple()]
        max_per_socket = max(1, int(self._settings.stream_max_tickers_per_socket))
        return [
            tuple(tickers[index : index + max_per_socket])
            for index in range(0, len(tickers), max_per_socket)
        ]

    def _has_stream_auth(self) -> bool:
        return bool(self._private_key is not None and self._settings.key_id)

    async def _build_stream_market_cache(self) -> dict[str, dict[str, Any]]:
        if self._settings.market_tickers:
            return await self._fetch_market_details_for_tickers(self._settings.market_tickers)

        summaries: list[dict[str, Any]] = []
        try:
            bootstrap_pages = max(0, int(self._settings.stream_bootstrap_scan_pages))
            if bootstrap_pages > 0:
                summaries = await self._fetch_market_summaries_paged(max_pages_override=bootstrap_pages)
        except Exception as exc:
            LOGGER.warning("kalshi stream market bootstrap via /markets failed: %s", exc)
        if not summaries:
            try:
                summaries = await self._fetch_market_summaries_from_events_cached()
            except Exception as exc:
                LOGGER.warning("kalshi stream market bootstrap via /events failed: %s", exc)
                summaries = []

        cache: dict[str, dict[str, Any]] = {}
        for market in summaries:
            ticker = str(market.get("ticker") or market.get("market_ticker") or "").strip()
            if not ticker:
                continue
            cache[ticker] = dict(market)

        enrich_limit = max(0, int(self._settings.stream_bootstrap_enrich_limit))
        if enrich_limit > 0:
            missing_priority = [
                ticker
                for ticker in self._settings.stream_priority_tickers
                if ticker and ticker not in cache
            ]
            if missing_priority:
                details = await self._fetch_market_details_for_tickers(missing_priority[:enrich_limit])
                if details:
                    cache.update(details)
        return cache

    async def _fetch_market_details_for_tickers(self, tickers: Sequence[str]) -> dict[str, dict[str, Any]]:
        semaphore = asyncio.Semaphore(max(1, self._settings.max_orderbook_concurrency))

        async def _run(ticker: str) -> tuple[str, dict[str, Any]]:
            ticker = ticker.strip()
            if not ticker:
                return "", {}
            async with semaphore:
                backoff = max(0.1, self._settings.request_pause_seconds)
                for _ in range(3):
                    try:
                        response = await self._get(f"/markets/{ticker}")
                        if response.status_code == 429:
                            await asyncio.sleep(backoff)
                            backoff = min(2.0, backoff * 2)
                            continue
                        response.raise_for_status()
                        payload = response.json()
                        market = payload.get("market", payload) if isinstance(payload, dict) else None
                        if isinstance(market, dict):
                            return ticker, market
                        break
                    except Exception:
                        await asyncio.sleep(backoff)
                        backoff = min(2.0, backoff * 2)
                return ticker, {"ticker": ticker}
            return ticker, {"ticker": ticker}

        tasks = [_run(ticker) for ticker in tickers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        cache: dict[str, dict[str, Any]] = {}
        for result in results:
            if isinstance(result, Exception):
                continue
            ticker, market = result
            if ticker:
                cache[ticker] = market
        return cache

    def _stream_subscription_tickers(self, market_cache: dict[str, dict[str, Any]]) -> list[str]:
        if self._settings.market_tickers:
            base = [ticker.strip() for ticker in self._settings.market_tickers if ticker.strip()]
        else:
            base = list(market_cache.keys())

        if not base and self._settings.stream_priority_tickers:
            base = [ticker.strip() for ticker in self._settings.stream_priority_tickers if ticker.strip()]

        ordered: list[str] = []
        seen: set[str] = set()

        # Phase 8B: Pinned tickers (structural rules) get highest subscription priority.
        for ticker in self._settings.stream_pinned_tickers:
            value = ticker.strip()
            if not value or value in seen:
                continue
            seen.add(value)
            ordered.append(value)

        for ticker in self._settings.stream_priority_tickers:
            value = ticker.strip()
            if not value or value in seen:
                continue
            seen.add(value)
            ordered.append(value)

        for ticker in base:
            value = ticker.strip()
            if not value or value in seen:
                continue
            seen.add(value)
            ordered.append(value)

        if self._settings.market_limit > 0:
            return ordered[: self._settings.market_limit]
        return ordered

    def _parse_stream_frames(self, raw: str) -> list[dict[str, Any]]:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return []
        frames = payload if isinstance(payload, list) else [payload]
        return [frame for frame in frames if isinstance(frame, dict)]

    async def _emit_quotes_from_frames(
        self,
        frames: list[dict[str, Any]],
        market_cache: dict[str, dict[str, Any]],
        market_lookup: dict[str, str],
        output_queue: asyncio.Queue[BinaryQuote],
    ) -> bool:
        emitted = False
        for message in self._extract_ticker_messages_from_frames(frames):
            quote = self._quote_from_ticker_message(message, market_cache, market_lookup)
            if quote is None:
                continue
            emitted = True
            await output_queue.put(quote)
        return emitted

    @staticmethod
    def _frame_request_id(frame: dict[str, Any]) -> int | None:
        raw = frame.get("id")
        if raw is None:
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    def _is_subscribe_success_frame(self, frame: dict[str, Any], request_id: int) -> bool:
        frame_type = str(frame.get("type") or frame.get("event") or "").strip().lower()
        frame_id = self._frame_request_id(frame)
        if frame_id is not None and frame_id != request_id:
            return False
        if frame_type in {"subscribed", "subscription_ack", "ok", "success"}:
            return True
        if frame.get("ok") is True and (frame_id == request_id or frame_id is None):
            return True
        return False

    def _is_subscribe_error_frame(self, frame: dict[str, Any], request_id: int) -> bool:
        frame_type = str(frame.get("type") or frame.get("event") or "").strip().lower()
        frame_id = self._frame_request_id(frame)
        if frame_id is not None and frame_id != request_id:
            return False
        if frame_type in {"error", "failed", "reject", "rejected"}:
            return True
        if frame.get("error"):
            return True
        return False

    def _next_stream_request_id(self) -> int:
        self._stream_request_seq += 1
        return self._stream_request_seq

    def _extract_ticker_messages(self, raw: str) -> list[dict[str, Any]]:
        frames = self._parse_stream_frames(raw)
        return self._extract_ticker_messages_from_frames(frames)

    def _extract_ticker_messages_from_frames(self, frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for frame in frames:
            frame_type = str(frame.get("type") or frame.get("channel") or "").lower()
            if frame_type == "ticker":
                message = frame.get("msg")
                if isinstance(message, dict):
                    messages.append(message)
                elif isinstance(message, list):
                    for item in message:
                        if isinstance(item, dict):
                            messages.append(item)
                elif isinstance(message, str):
                    with suppress(json.JSONDecodeError, TypeError, ValueError):
                        nested = json.loads(message)
                        if isinstance(nested, dict):
                            messages.append(nested)
                        elif isinstance(nested, list):
                            messages.extend(item for item in nested if isinstance(item, dict))
                else:
                    messages.append(frame)
                continue

            if frame_type in {"subscribed", "unsubscribed", "pong", "heartbeat"}:
                continue

            if frame_type == "error":
                LOGGER.warning("kalshi stream error frame: %s", frame)
                continue

            # Some gateways deliver the ticker payload without an explicit wrapper.
            if (
                ("market_ticker" in frame or "ticker" in frame or "market_id" in frame or "marketId" in frame)
                and any(
                    key in frame
                    for key in (
                        "yes_bid",
                        "yes_ask",
                        "no_bid",
                        "no_ask",
                        "yes_bid_dollars",
                        "yes_ask_dollars",
                        "no_bid_dollars",
                        "no_ask_dollars",
                        "yes_bid_fp",
                        "yes_ask_fp",
                        "no_bid_fp",
                        "no_ask_fp",
                    )
                )
            ):
                messages.append(frame)

        return messages

    @staticmethod
    def _lookup_key(value: Any) -> str:
        return str(value or "").strip().lower()

    def _register_market_lookup_entry(
        self,
        market_lookup: dict[str, str],
        ticker: str,
        market: dict[str, Any] | None = None,
        aliases: Sequence[Any] = (),
    ) -> None:
        canonical = ticker.strip()
        if not canonical:
            return

        keys: list[Any] = [canonical]
        if market is not None:
            keys.extend(
                (
                    market.get("ticker"),
                    market.get("market_ticker"),
                    market.get("marketTicker"),
                    market.get("market_id"),
                    market.get("marketId"),
                    market.get("id"),
                )
            )
        keys.extend(aliases)

        for raw_key in keys:
            key = self._lookup_key(raw_key)
            if not key:
                continue
            market_lookup[key] = canonical

    def _refresh_stream_market_lookup(
        self,
        market_lookup: dict[str, str],
        market_cache: dict[str, dict[str, Any]],
    ) -> None:
        for ticker, market in market_cache.items():
            self._register_market_lookup_entry(market_lookup, ticker, market)

    def _build_stream_market_lookup(self, market_cache: dict[str, dict[str, Any]]) -> dict[str, str]:
        lookup: dict[str, str] = {}
        self._refresh_stream_market_lookup(lookup, market_cache)
        return lookup

    def _resolve_stream_ticker(
        self,
        message: dict[str, Any],
        market_lookup: dict[str, str],
    ) -> str:
        direct_candidates = (
            message.get("market_ticker"),
            message.get("marketTicker"),
            message.get("ticker"),
        )
        for raw in direct_candidates:
            key = self._lookup_key(raw)
            if not key:
                continue
            resolved = market_lookup.get(key)
            if resolved:
                return resolved
            return str(raw).strip()

        id_candidates = (
            message.get("market_id"),
            message.get("marketId"),
            message.get("id"),
        )
        for raw in id_candidates:
            key = self._lookup_key(raw)
            if not key:
                continue
            resolved = market_lookup.get(key)
            if resolved:
                return resolved

        return ""

    @staticmethod
    def _normalize_stream_price(value: Any) -> float | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if numeric < 0:
            return None
        for scale in (1.0, 100.0, 10_000.0, 1_000_000.0, 1_000_000_000.0):
            candidate = numeric / scale
            if 0.0 <= candidate <= 1.0:
                return candidate
        return None

    def _quote_from_ticker_message(
        self,
        message: dict[str, Any],
        market_cache: dict[str, dict[str, Any]],
        market_lookup: dict[str, str] | None = None,
    ) -> BinaryQuote | None:
        lookup = market_lookup or {}
        ticker = self._resolve_stream_ticker(message, lookup)
        if not ticker:
            return None

        market = dict(market_cache.get(ticker, {"ticker": ticker}))
        market.setdefault("ticker", ticker)
        market.setdefault("market_ticker", ticker)
        stream_market_id = message.get("market_id") or message.get("marketId") or message.get("id")
        if stream_market_id is not None:
            market["market_id"] = stream_market_id

        for src_key, dst_key in (
            ("yes_bid", "yes_bid"),
            ("yesBid", "yes_bid"),
            ("yes_bid_dollars", "yes_bid"),
            ("yesBidDollars", "yes_bid"),
            ("yes_bid_fp", "yes_bid"),
            ("yesBidFp", "yes_bid"),
            ("yes_ask", "yes_ask"),
            ("yesAsk", "yes_ask"),
            ("yes_ask_dollars", "yes_ask"),
            ("yesAskDollars", "yes_ask"),
            ("yes_ask_fp", "yes_ask"),
            ("yesAskFp", "yes_ask"),
            ("no_bid", "no_bid"),
            ("noBid", "no_bid"),
            ("no_bid_dollars", "no_bid"),
            ("noBidDollars", "no_bid"),
            ("no_bid_fp", "no_bid"),
            ("noBidFp", "no_bid"),
            ("no_ask", "no_ask"),
            ("noAsk", "no_ask"),
            ("no_ask_dollars", "no_ask"),
            ("noAskDollars", "no_ask"),
            ("no_ask_fp", "no_ask"),
            ("noAskFp", "no_ask"),
            ("yes_bid_size", "yes_bid_size"),
            ("yesBidSize", "yes_bid_size"),
            ("yes_bid_quantity", "yes_bid_size"),
            ("yesBidQuantity", "yes_bid_size"),
            ("yes_ask_size", "yes_ask_size"),
            ("yesAskSize", "yes_ask_size"),
            ("yes_ask_quantity", "yes_ask_size"),
            ("yesAskQuantity", "yes_ask_size"),
            ("no_bid_size", "no_bid_size"),
            ("noBidSize", "no_bid_size"),
            ("no_bid_quantity", "no_bid_size"),
            ("noBidQuantity", "no_bid_size"),
            ("no_ask_size", "no_ask_size"),
            ("noAskSize", "no_ask_size"),
            ("no_ask_quantity", "no_ask_size"),
            ("noAskQuantity", "no_ask_size"),
            ("volume", "volume"),
            ("volume_24h", "volume_24h"),
            ("liquidity", "liquidity"),
            ("liquidity_dollars", "liquidity_dollars"),
            ("open_interest", "open_interest"),
            ("event_ticker", "event_ticker"),
            ("title", "title"),
            ("subtitle", "subtitle"),
            ("close_time", "close_time"),
            ("end_time", "end_time"),
        ):
            value = message.get(src_key)
            if value is not None:
                if src_key.endswith("_fp") or src_key.endswith("Fp"):
                    normalized = self._normalize_stream_price(value)
                    if normalized is not None:
                        market[dst_key] = normalized
                    continue
                market[dst_key] = value

        market_cache[ticker] = market
        self._register_market_lookup_entry(
            lookup,
            ticker,
            market=market,
            aliases=(
                message.get("market_ticker"),
                message.get("marketTicker"),
                message.get("ticker"),
                message.get("market_id"),
                message.get("marketId"),
                message.get("id"),
            ),
        )
        quote = self._quote_from_market_summary(market)
        if quote is None:
            return None
        if not self._quote_passes_filters(quote, market):
            return None
        return quote

    def _ws_auth_headers(self, ws_url: str) -> dict[str, str]:
        if self._private_key is None or not self._settings.key_id:
            return {}
        if hashes is None or padding is None:
            return {}

        path = self._canonical_ws_signing_path(ws_url)
        ts_ms = str(int(time.time() * 1000))
        payload = f"{ts_ms}GET{path}".encode("utf-8")
        signature = self._private_key.sign(
            payload,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self._settings.key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts_ms,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode("utf-8"),
        }

    @staticmethod
    def _canonical_ws_signing_path(ws_url: str) -> str:
        parsed = urlparse(ws_url)
        raw = (parsed.path or "").strip()
        if not raw or raw == "/":
            return "/trade-api/ws/v2"
        if not raw.startswith("/"):
            raw = "/" + raw
        if raw.startswith("/ws/"):
            return f"/trade-api{raw}"
        return raw

    def _events_circuit_open(self) -> bool:
        return time.time() < self._events_circuit_open_until_ts

    def _events_circuit_remaining_seconds(self) -> float:
        return max(0.0, self._events_circuit_open_until_ts - time.time())

    def _record_event_429(self, endpoint: str) -> None:
        self._events_429_consecutive += 1
        threshold = max(1, int(self._settings.events_429_circuit_threshold))
        if self._events_429_consecutive < threshold:
            return

        cooldown = max(1.0, float(self._settings.events_429_circuit_cooldown_seconds))
        self._events_circuit_open_until_ts = time.time() + cooldown
        self._events_429_consecutive = 0
        LOGGER.warning(
            "kalshi events circuit opened endpoint=%s cooldown=%.1fs",
            endpoint,
            cooldown,
        )

    def _record_event_success(self) -> None:
        self._events_429_consecutive = 0

    async def _fetch_market_summaries_from_events_cached(self) -> list[dict[str, Any]]:
        now = time.time()
        if self._discovery_cache_markets and (now - self._discovery_cache_ts) < self._discovery_cache_ttl_seconds:
            return list(self._discovery_cache_markets)

        fresh = await self._fetch_market_summaries_from_events()
        if fresh:
            self._discovery_cache_markets = list(fresh)
            self._discovery_cache_ts = now
        return fresh

    async def _fetch_market_summaries_from_events(self) -> list[dict[str, Any]]:
        include_events = {value.strip().upper() for value in self._settings.include_event_tickers if value.strip()}
        excluded_prefixes = tuple(prefix.strip().upper() for prefix in self._settings.exclude_ticker_prefixes if prefix.strip())

        event_tickers: list[str] = []
        if include_events:
            event_tickers = sorted(include_events)
        else:
            event_tickers = await self._fetch_open_event_tickers()

        if not event_tickers:
            return []

        semaphore = asyncio.Semaphore(max(1, self._settings.max_orderbook_concurrency))

        async def _run(event_ticker: str) -> list[dict[str, Any]]:
            async with semaphore:
                return await self._fetch_event_markets(event_ticker)

        tasks = [_run(event_ticker) for event_ticker in event_tickers]
        details = await asyncio.gather(*tasks, return_exceptions=True)

        collected: list[dict[str, Any]] = []
        for event_ticker, result in zip(event_tickers, details):
            if isinstance(result, Exception):
                LOGGER.debug("kalshi event fetch failed for %s: %s", event_ticker, result)
                continue

            for market in result:
                ticker = str(market.get("ticker") or market.get("market_ticker") or "").strip()
                if not ticker:
                    continue

                ticker_upper = ticker.upper()
                if excluded_prefixes and ticker_upper.startswith(excluded_prefixes):
                    continue

                liquidity = self._normalize_size(
                    market.get("liquidity")
                    or market.get("liquidity_dollars")
                    or market.get("open_interest")
                    or 0.0
                )
                if liquidity < self._settings.min_liquidity:
                    continue

                # Ensure event_ticker is present on market metadata for downstream filters/mapping.
                if not market.get("event_ticker"):
                    market["event_ticker"] = event_ticker
                collected.append(market)

        collected.sort(key=self._market_priority_score, reverse=True)
        return collected[: self._settings.market_limit]

    async def _fetch_open_event_tickers(self) -> list[str]:
        if self._events_circuit_open():
            LOGGER.warning(
                "kalshi events discovery skipped; circuit open for %.1fs",
                self._events_circuit_remaining_seconds(),
            )
            return []

        cursor: str | None = None
        tickers: list[str] = []
        seen: set[str] = set()

        for _ in range(max(1, self._settings.market_scan_pages)):
            params: dict[str, Any] = {"status": "open", "limit": self._settings.market_page_size}
            if cursor:
                params["cursor"] = cursor

            response = await self._get("/events", params=params)
            if response.status_code == 429:
                self._record_event_429("/events")
                await asyncio.sleep(max(0.1, self._settings.request_pause_seconds))
                if self._events_circuit_open():
                    break
                response = await self._get("/events", params=params)
            if response.status_code == 429:
                self._record_event_429("/events")
                if self._events_circuit_open():
                    break
                continue
            self._record_event_success()
            response.raise_for_status()
            payload = response.json()
            events = payload.get("events", payload if isinstance(payload, list) else [])

            for event in events:
                if not isinstance(event, dict):
                    continue
                ticker = str(event.get("event_ticker") or "").strip().upper()
                if not ticker or ticker in seen:
                    continue
                seen.add(ticker)
                tickers.append(ticker)

            cursor = payload.get("cursor") if isinstance(payload, dict) else None
            if not cursor:
                break
            if self._settings.request_pause_seconds > 0:
                await asyncio.sleep(self._settings.request_pause_seconds)

        return tickers

    async def _fetch_event_markets(self, event_ticker: str) -> list[dict[str, Any]]:
        if self._events_circuit_open():
            return []

        response = await self._get(f"/events/{event_ticker}")
        if response.status_code == 429:
            self._record_event_429(f"/events/{event_ticker}")
            await asyncio.sleep(max(0.1, self._settings.request_pause_seconds))
            if self._events_circuit_open():
                return []
            response = await self._get(f"/events/{event_ticker}")
        if response.status_code == 429:
            self._record_event_429(f"/events/{event_ticker}")
            return []
        self._record_event_success()
        response.raise_for_status()

        payload = response.json()
        if not isinstance(payload, dict):
            return []
        markets = payload.get("markets")
        if not isinstance(markets, list):
            return []

        parsed: list[dict[str, Any]] = []
        for market in markets:
            if isinstance(market, dict):
                parsed.append(market)
        return parsed

    async def _fetch_market_summaries_paged(self, max_pages_override: int | None = None) -> list[dict[str, Any]]:
        include_events = {value.strip().upper() for value in self._settings.include_event_tickers if value.strip()}
        excluded_prefixes = tuple(prefix.strip().upper() for prefix in self._settings.exclude_ticker_prefixes if prefix.strip())

        cursor: str | None = None
        collected: list[dict[str, Any]] = []

        max_pages = max_pages_override if max_pages_override is not None else self._settings.market_scan_pages
        for _ in range(max(1, int(max_pages))):
            payload = await self._get_markets_page(cursor)
            markets = payload.get("markets", payload if isinstance(payload, list) else [])

            for market in markets:
                if not isinstance(market, dict):
                    continue
                ticker = str(market.get("ticker") or market.get("market_ticker") or "").strip()
                if not ticker:
                    continue

                ticker_upper = ticker.upper()
                if excluded_prefixes and ticker_upper.startswith(excluded_prefixes):
                    continue

                event_ticker = str(market.get("event_ticker") or "").strip().upper()
                if include_events and event_ticker not in include_events:
                    continue

                liquidity = self._normalize_size(
                    market.get("liquidity")
                    or market.get("liquidity_dollars")
                    or market.get("open_interest")
                    or 0.0
                )
                if liquidity < self._settings.min_liquidity:
                    continue

                collected.append(market)

            cursor = payload.get("cursor")
            if not cursor:
                break

            if self._settings.request_pause_seconds > 0:
                await asyncio.sleep(self._settings.request_pause_seconds)

        collected.sort(key=self._market_priority_score, reverse=True)
        return collected[: self._settings.market_limit]

    async def _get_markets_page(self, cursor: str | None) -> dict[str, Any]:
        params: dict[str, Any] = {"status": "open", "limit": self._settings.market_page_size}
        if cursor:
            params["cursor"] = cursor

        delay = max(0.1, self._settings.request_pause_seconds)
        for attempt in range(4):
            response = await self._get("/markets", params=params)
            if response.status_code == 429:
                await asyncio.sleep(delay)
                delay *= 2
                continue

            response.raise_for_status()
            payload = response.json()
            return payload if isinstance(payload, dict) else {"markets": payload}

        LOGGER.warning("kalshi /markets rate-limited repeatedly; returning empty page")
        return {"markets": []}

    def _market_priority_score(self, market: dict[str, Any]) -> float:
        liquidity = self._normalize_size(
            market.get("liquidity")
            or market.get("liquidity_dollars")
            or market.get("open_interest")
            or 0.0
        )
        volume = self._normalize_size(market.get("volume_24h") or market.get("volume") or 0.0)

        yes_ask = self._normalize_price(market.get("yes_ask"))
        yes_bid = self._normalize_price(market.get("yes_bid"))
        no_ask = self._normalize_price(market.get("no_ask"))
        no_bid = self._normalize_price(market.get("no_bid"))
        if yes_ask is None:
            yes_ask = self._normalize_price(market.get("yes_ask_dollars"))
        if yes_bid is None:
            yes_bid = self._normalize_price(market.get("yes_bid_dollars"))
        if no_ask is None:
            no_ask = self._normalize_price(market.get("no_ask_dollars"))
        if no_bid is None:
            no_bid = self._normalize_price(market.get("no_bid_dollars"))
        spread = 0.0
        if yes_ask is not None and yes_bid is not None:
            spread += max(0.0, yes_ask - yes_bid)
        if no_ask is not None and no_bid is not None:
            spread += max(0.0, no_ask - no_bid)

        return math.log1p(liquidity) + math.log1p(volume) - spread

    async def _fetch_quotes_from_market_summaries(self, summaries: Sequence[dict[str, Any]]) -> list[BinaryQuote]:
        semaphore = asyncio.Semaphore(max(1, self._settings.max_orderbook_concurrency))

        async def _run(summary: dict[str, Any]) -> BinaryQuote | None:
            async with semaphore:
                return await self._fetch_quote_for_summary(summary)

        tasks = [_run(summary) for summary in summaries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        quotes: list[BinaryQuote] = []
        for summary, result in zip(summaries, results):
            if isinstance(result, Exception):
                ticker = summary.get("ticker") or summary.get("market_ticker") or "unknown"
                LOGGER.warning("kalshi orderbook fetch failed for %s: %s", ticker, result)
                fallback = self._quote_from_market_summary(summary)
                if fallback is not None and self._quote_passes_filters(fallback, summary):
                    quotes.append(fallback)
                continue

            if result is not None and self._quote_passes_filters(result, summary):
                quotes.append(result)

        return quotes

    async def _fetch_quote_for_summary(self, market_summary: dict[str, Any]) -> BinaryQuote | None:
        ticker = str(market_summary.get("ticker") or market_summary.get("market_ticker") or "").strip()
        if not ticker:
            return None

        try:
            book_res = await self._get(f"/markets/{ticker}/orderbook")
            if book_res.status_code == 429:
                await asyncio.sleep(max(0.1, self._settings.request_pause_seconds))
                book_res = await self._get(f"/markets/{ticker}/orderbook")
            book_res.raise_for_status()
            raw_book = book_res.json()
            book_payload = raw_book.get("orderbook", raw_book)
        except Exception as exc:
            LOGGER.debug("kalshi orderbook fetch failed for %s: %s", ticker, exc)
            return self._quote_from_market_summary(market_summary)

        quote_from_book = self._quote_from_orderbook(ticker, book_payload, market_summary)
        if quote_from_book is not None:
            return quote_from_book

        return self._quote_from_market_summary(market_summary)

    async def _fetch_ticker_quotes(self, tickers: Sequence[str]) -> list[BinaryQuote]:
        if not tickers:
            return []
        semaphore = asyncio.Semaphore(max(1, self._settings.max_orderbook_concurrency))

        async def _run(ticker: str) -> BinaryQuote | None:
            async with semaphore:
                return await self._fetch_single_ticker_quote(ticker)

        tasks = [_run(ticker) for ticker in tickers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        quotes: list[BinaryQuote] = []
        throttled_failures = 0
        successful = 0
        for ticker, result in zip(tickers, results):
            if isinstance(result, Exception):
                LOGGER.warning("kalshi quote failed for %s: %s", ticker, result)
                if "429" in str(result):
                    throttled_failures += 1
                continue
            if result is not None and self._quote_passes_filters(result):
                quotes.append(result)
                successful += 1
        self._rebalance_priority_refresh_limit(
            attempted=len(tickers),
            successful=successful,
            throttled=throttled_failures,
        )
        return quotes

    async def _fetch_single_ticker_quote(self, ticker: str) -> BinaryQuote | None:
        market_res: Any = None
        book_res: Any = None

        try:
            market_res = await self._get(f"/markets/{ticker}")
        except Exception as exc:
            market_res = exc

        if self._settings.use_orderbook_quotes:
            try:
                book_res = await self._get(f"/markets/{ticker}/orderbook")
            except Exception as exc:
                book_res = exc

        market_payload: dict[str, Any] | None = None
        book_payload: dict[str, Any] | None = None

        if isinstance(market_res, Exception):
            LOGGER.debug("kalshi market fetch failed for %s: %s", ticker, market_res)
        elif isinstance(market_res, httpx.Response):
            market_res.raise_for_status()
            raw_market = market_res.json()
            market_payload = raw_market.get("market", raw_market)
        else:
            LOGGER.debug(
                "kalshi market fetch returned invalid response type for %s: %s",
                ticker,
                type(market_res).__name__,
            )

        if self._settings.use_orderbook_quotes:
            if isinstance(book_res, Exception):
                LOGGER.debug("kalshi orderbook fetch failed for %s: %s", ticker, book_res)
            elif isinstance(book_res, httpx.Response):
                book_res.raise_for_status()
                raw_book = book_res.json()
                book_payload = raw_book.get("orderbook", raw_book)
            else:
                LOGGER.debug(
                    "kalshi orderbook fetch returned invalid response type for %s: %s",
                    ticker,
                    type(book_res).__name__,
                )

        if self._settings.use_orderbook_quotes:
            quote_from_book = self._quote_from_orderbook(ticker, book_payload, market_payload)
            if quote_from_book is not None:
                return quote_from_book

        if market_payload is None:
            return None

        return self._quote_from_market_summary(market_payload)

    @staticmethod
    def _dedupe_tickers(values: Sequence[str]) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        for raw in values:
            ticker = str(raw).strip()
            if not ticker or ticker in seen:
                continue
            seen.add(ticker)
            ordered.append(ticker)
        return ordered

    def _select_priority_refresh_tickers(self, universe: Sequence[str], limit: int) -> list[str]:
        size = len(universe)
        if size == 0:
            return []
        count = max(1, min(limit, size))
        if count >= size:
            return list(universe)

        # Phase 8B: Always include pinned tickers (structural rules) first,
        # then fill remaining slots from the rotating cursor window.
        pinned = self._settings.stream_pinned_tickers
        selected: list[str] = []
        seen: set[str] = set()

        for ticker in pinned:
            value = ticker.strip()
            if value and value not in seen:
                selected.append(value)
                seen.add(value)
            if len(selected) >= count:
                break

        remaining = count - len(selected)
        if remaining > 0:
            start = self._priority_refresh_cursor % size
            scanned = 0
            added = 0
            while added < remaining and scanned < size:
                candidate = universe[(start + scanned) % size]
                scanned += 1
                if candidate not in seen:
                    selected.append(candidate)
                    seen.add(candidate)
                    added += 1
            self._priority_refresh_cursor = (start + scanned) % size
        else:
            # All slots taken by pinned tickers — still advance cursor.
            self._priority_refresh_cursor = (self._priority_refresh_cursor + 1) % max(1, size)

        return selected

    def _rebalance_priority_refresh_limit(self, attempted: int, successful: int, throttled: int) -> None:
        if attempted <= 0 or not self._settings.stream_priority_tickers:
            return

        max_limit = max(
            1,
            min(
                int(self._settings.market_limit),
                len(self._settings.stream_priority_tickers),
            ),
        )
        min_limit = max(10, min(max_limit, 40))
        current = self._dynamic_priority_refresh_limit

        if throttled > 0:
            self._dynamic_priority_limit_success_streak = 0
            reduced = max(min_limit, int(math.ceil(current * 0.75)))
            if reduced < current:
                self._dynamic_priority_refresh_limit = reduced
                LOGGER.info(
                    "kalshi adaptive refresh limit reduced=%d (from=%d throttled=%d attempted=%d)",
                    reduced,
                    current,
                    throttled,
                    attempted,
                )
            return

        if successful >= max(1, int(math.ceil(attempted * 0.9))):
            self._dynamic_priority_limit_success_streak += 1
            if self._dynamic_priority_limit_success_streak >= 3 and current < max_limit:
                step = max(5, current // 10)
                expanded = min(max_limit, current + step)
                if expanded > current:
                    self._dynamic_priority_refresh_limit = expanded
                    LOGGER.info(
                        "kalshi adaptive refresh limit increased=%d (from=%d successful=%d attempted=%d)",
                        expanded,
                        current,
                        successful,
                        attempted,
                    )
                self._dynamic_priority_limit_success_streak = 0
        else:
            self._dynamic_priority_limit_success_streak = 0

    def _quote_from_market_summary(self, market: dict[str, Any]) -> BinaryQuote | None:
        market_id = str(market.get("ticker") or market.get("market_ticker") or "").strip()
        if not market_id:
            return None

        yes_bid = self._normalize_price(market.get("yes_bid"))
        no_bid = self._normalize_price(market.get("no_bid"))
        yes_ask = self._normalize_price(market.get("yes_ask"))
        no_ask = self._normalize_price(market.get("no_ask"))
        if yes_bid is None:
            yes_bid = self._normalize_price(market.get("yes_bid_dollars"))
        if no_bid is None:
            no_bid = self._normalize_price(market.get("no_bid_dollars"))
        if yes_ask is None:
            yes_ask = self._normalize_price(market.get("yes_ask_dollars"))
        if no_ask is None:
            no_ask = self._normalize_price(market.get("no_ask_dollars"))

        if yes_ask is None and no_bid is not None:
            yes_ask = 1.0 - no_bid
        if no_ask is None and yes_bid is not None:
            no_ask = 1.0 - yes_bid

        if yes_ask is None or no_ask is None:
            return None

        yes_bid_size = self._normalize_size(market.get("yes_bid_size") or 0.0)
        no_bid_size = self._normalize_size(market.get("no_bid_size") or 0.0)

        yes_ask_size = self._normalize_size(
            market.get("yes_ask_size")
            or market.get("liquidity")
            or market.get("open_interest")
            or 0.0
        )
        no_ask_size = self._normalize_size(
            market.get("no_ask_size")
            or market.get("liquidity")
            or market.get("open_interest")
            or 0.0
        )

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
        diagnostics = build_quote_diagnostics(
            yes_buy_price=yes_effective.price,
            no_buy_price=no_effective.price,
            yes_bid_price=yes_bid,
            no_bid_price=no_bid,
        )

        title = str(market.get("title") or market.get("subtitle") or "")
        resolution_ts = self._extract_resolution_ts(market)

        return BinaryQuote(
            venue=self.venue,
            market_id=market_id,
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
            metadata={
                "title": title,
                "canonical_text": title,
                "event_ticker": market.get("event_ticker"),
                "resolution_ts": resolution_ts,
                "volume_24h": self._normalize_size(market.get("volume_24h") or market.get("volume") or 0.0),
                "volume": self._normalize_size(market.get("volume") or market.get("volume_24h") or 0.0),
                "liquidity": self._normalize_size(
                    market.get("liquidity")
                    or market.get("liquidity_dollars")
                    or market.get("open_interest")
                    or 0.0
                ),
                "liquidity_dollars": self._normalize_size(market.get("liquidity_dollars") or 0.0),
                "open_interest": self._normalize_size(market.get("open_interest") or 0.0),
                "yes_bid_price": yes_bid,
                "no_bid_price": no_bid,
                "yes_buy_source": yes_effective.source,
                "no_buy_source": no_effective.source,
                "ask_implied_probability": diagnostics.ask_implied_probability,
                "ask_edge_per_side": diagnostics.ask_edge_per_side,
                "bid_implied_probability": diagnostics.bid_implied_probability,
                "bid_edge_per_side": diagnostics.bid_edge_per_side,
                "midpoint_consistency_gap": diagnostics.midpoint_consistency_gap,
                "yes_spread": diagnostics.yes_spread,
                "no_spread": diagnostics.no_spread,
                "spread_asymmetry": diagnostics.spread_asymmetry,
            },
        )

    def _quote_from_orderbook(
        self,
        ticker: str,
        orderbook: dict[str, Any] | None,
        market: dict[str, Any] | None,
    ) -> BinaryQuote | None:
        if not orderbook:
            return None

        yes_bids = self._extract_levels(orderbook.get("yes"))
        no_bids = self._extract_levels(orderbook.get("no"))

        best_yes_bid_price, best_yes_bid_size = yes_bids[0] if yes_bids else (None, 0.0)
        best_no_bid_price, best_no_bid_size = no_bids[0] if no_bids else (None, 0.0)

        summary_yes_ask = self._normalize_price(market.get("yes_ask")) if market else None
        summary_no_ask = self._normalize_price(market.get("no_ask")) if market else None
        if summary_yes_ask is None and market:
            summary_yes_ask = self._normalize_price(market.get("yes_ask_dollars"))
        if summary_no_ask is None and market:
            summary_no_ask = self._normalize_price(market.get("no_ask_dollars"))
        summary_yes_size = self._normalize_size(market.get("yes_ask_size") if market else 0.0)
        summary_no_size = self._normalize_size(market.get("no_ask_size") if market else 0.0)

        yes_effective = choose_effective_buy_price(
            side="yes",
            direct_ask_price=summary_yes_ask,
            direct_ask_size=summary_yes_size,
            opposite_bid_price=best_no_bid_price,
            opposite_bid_size=best_no_bid_size,
        )
        no_effective = choose_effective_buy_price(
            side="no",
            direct_ask_price=summary_no_ask,
            direct_ask_size=summary_no_size,
            opposite_bid_price=best_yes_bid_price,
            opposite_bid_size=best_yes_bid_size,
        )
        if yes_effective is None or no_effective is None:
            return None

        market_title = ""
        event_ticker = None
        resolution_ts = None
        if market:
            market_title = str(market.get("title") or market.get("subtitle") or "")
            event_ticker = market.get("event_ticker")
            resolution_ts = self._extract_resolution_ts(market)

        yes_maker = self._estimate_maker_buy_price(best_yes_bid_price, yes_effective.price)
        no_maker = self._estimate_maker_buy_price(best_no_bid_price, no_effective.price)
        diagnostics = build_quote_diagnostics(
            yes_buy_price=yes_effective.price,
            no_buy_price=no_effective.price,
            yes_bid_price=best_yes_bid_price,
            no_bid_price=best_no_bid_price,
        )

        return BinaryQuote(
            venue=self.venue,
            market_id=ticker,
            yes_buy_price=yes_effective.price,
            no_buy_price=no_effective.price,
            yes_buy_size=yes_effective.size,
            no_buy_size=no_effective.size,
            yes_bid_price=best_yes_bid_price,
            no_bid_price=best_no_bid_price,
            yes_bid_size=best_yes_bid_size,
            no_bid_size=best_no_bid_size,
            yes_maker_buy_price=yes_maker,
            no_maker_buy_price=no_maker,
            yes_maker_buy_size=best_yes_bid_size,
            no_maker_buy_size=best_no_bid_size,
            fee_per_contract=self._settings.taker_fee_per_contract,
            metadata={
                "title": market_title,
                "canonical_text": market_title,
                "event_ticker": event_ticker,
                "resolution_ts": resolution_ts,
                "volume_24h": self._normalize_size((market or {}).get("volume_24h") if market else 0.0),
                "volume": self._normalize_size((market or {}).get("volume") if market else 0.0),
                "liquidity": self._normalize_size(
                    ((market or {}).get("liquidity") if market else 0.0)
                    or ((market or {}).get("liquidity_dollars") if market else 0.0)
                    or ((market or {}).get("open_interest") if market else 0.0)
                ),
                "liquidity_dollars": self._normalize_size((market or {}).get("liquidity_dollars") if market else 0.0),
                "open_interest": self._normalize_size((market or {}).get("open_interest") if market else 0.0),
                "yes_bid_price": best_yes_bid_price,
                "no_bid_price": best_no_bid_price,
                "yes_buy_source": yes_effective.source,
                "no_buy_source": no_effective.source,
                "ask_implied_probability": diagnostics.ask_implied_probability,
                "ask_edge_per_side": diagnostics.ask_edge_per_side,
                "bid_implied_probability": diagnostics.bid_implied_probability,
                "bid_edge_per_side": diagnostics.bid_edge_per_side,
                "midpoint_consistency_gap": diagnostics.midpoint_consistency_gap,
                "yes_spread": diagnostics.yes_spread,
                "no_spread": diagnostics.no_spread,
                "spread_asymmetry": diagnostics.spread_asymmetry,
            },
        )

    def _quote_passes_filters(self, quote: BinaryQuote, market: dict[str, Any] | None = None) -> bool:
        if quote.yes_buy_size <= 0 or quote.no_buy_size <= 0:
            return False

        if self._settings.require_nondegenerate_quotes:
            if self._is_degenerate(quote.yes_buy_price, quote.no_buy_price):
                return False

        if market is not None:
            liquidity = self._normalize_size(
                market.get("liquidity")
                or market.get("liquidity_dollars")
                or market.get("open_interest")
                or 0.0
            )
            if liquidity < self._settings.min_liquidity:
                return False

        return True

    @staticmethod
    def _is_degenerate(yes_ask: float, no_ask: float) -> bool:
        epsilon = 1e-9
        yes_hard = abs(yes_ask - 0.0) <= epsilon or abs(yes_ask - 1.0) <= epsilon
        no_hard = abs(no_ask - 0.0) <= epsilon or abs(no_ask - 1.0) <= epsilon
        return yes_hard and no_hard

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

        if candidate < 0:
            return None
        if candidate > 1:
            return None

        return candidate

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
        if self._private_key is None or not self._settings.key_id:
            return LegExecutionResult(False, None, leg.contracts, 0, None, {"error": "missing kalshi credentials"})

        payload = self._build_order_payload(
            market_id=leg.market_id,
            side=leg.side,
            contracts=leg.contracts,
            limit_price=leg.limit_price,
        )

        result = await self._private_request("POST", "/portfolio/orders", json=payload)
        return self._to_leg_result(result, leg.contracts)

    async def cancel_order(self, order_id: str) -> bool:
        if self._private_key is None or not self._settings.key_id:
            return False
        try:
            await self._private_request("DELETE", f"/portfolio/orders/{order_id}")
            return True
        except Exception as exc:
            LOGGER.warning("kalshi cancel_order failed for %s: %s", order_id, exc)
            return False

    async def get_order_status(self, order_id: str) -> OrderStatus | None:
        if self._private_key is None or not self._settings.key_id:
            return None
        try:
            payload = await self._private_request("GET", f"/portfolio/orders/{order_id}")
        except Exception as exc:
            LOGGER.warning("kalshi get_order_status failed for %s: %s", order_id, exc)
            return None

        order = payload.get("order") if isinstance(payload.get("order"), dict) else payload
        status_str = str(order.get("status", "")).lower()
        filled = int(order.get("filled_count", 0) or 0)
        remaining = int(order.get("remaining_count", 0) or 0)

        avg_price = order.get("average_price")
        if avg_price is not None:
            try:
                avg_price = float(avg_price)
                if avg_price > 1:
                    avg_price /= 100.0
            except (TypeError, ValueError):
                avg_price = None

        state_map = {
            "resting": OrderState.OPEN,
            "pending": OrderState.OPEN,
            "executed": OrderState.FILLED,
            "canceled": OrderState.CANCELLED,
            "cancelled": OrderState.CANCELLED,
            "expired": OrderState.EXPIRED,
        }
        state = state_map.get(status_str, OrderState.UNKNOWN)
        if state is OrderState.UNKNOWN and filled > 0 and remaining > 0:
            state = OrderState.PARTIALLY_FILLED

        return OrderStatus(
            order_id=order_id,
            state=state,
            filled_contracts=filled,
            remaining_contracts=remaining,
            average_price=avg_price,
            raw=payload,
        )

    async def get_available_cash(self) -> float | None:
        if self._private_key is None or not self._settings.key_id:
            return None

        try:
            payload = await self._private_request("GET", "/portfolio/balance")
        except Exception as exc:
            LOGGER.warning("kalshi balance fetch failed: %s", exc)
            return None

        for key in ("balance", "cash_balance", "available_cash"):
            value = payload.get(key)
            if value is not None:
                return float(value)

        nested = payload.get("portfolio_balance")
        if isinstance(nested, dict):
            for key in ("balance", "cash_balance", "available_cash"):
                value = nested.get(key)
                if value is not None:
                    return float(value)

        return None

    async def _private_request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        headers = self._auth_headers(method, path)
        response = await self._client.request(method, path, headers=headers, **kwargs)
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else {"data": payload}

    def _build_order_payload(
        self,
        market_id: str,
        side: Side,
        contracts: int,
        limit_price: float,
    ) -> dict[str, Any]:
        price_cents = int(round(limit_price * 100))
        payload = {
            "ticker": market_id,
            "action": "buy",
            "side": side.value,
            "count": contracts,
            "type": "limit",
            "expiration_ts": int((time.time() + 10) * 1000),
            "client_order_id": str(uuid.uuid4()),
        }
        if side is Side.YES:
            payload["yes_price"] = price_cents
        else:
            payload["no_price"] = price_cents
        return payload

    def _auth_headers(self, method: str, path: str) -> dict[str, str]:
        assert self._private_key is not None
        assert self._settings.key_id
        if hashes is None or padding is None:
            raise RuntimeError("cryptography dependency missing for kalshi auth")

        ts_ms = str(int(time.time() * 1000))
        canonical_path = self._canonical_signing_path(path)
        payload = f"{ts_ms}{method.upper()}{canonical_path}".encode("utf-8")
        signature = self._private_key.sign(
            payload,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self._settings.key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts_ms,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode("utf-8"),
        }

    @staticmethod
    def _canonical_signing_path(path: str) -> str:
        if path.startswith("/trade-api/"):
            return path
        return f"/trade-api/v2{path if path.startswith('/') else '/' + path}"

    def _load_private_key(self):
        if serialization is None:
            return None
        pem_text = self._settings.private_key_pem
        if not pem_text and self._settings.private_key_path:
            pem_path = Path(self._settings.private_key_path)
            if pem_path.exists():
                pem_text = pem_path.read_text(encoding="utf-8")

        if not pem_text:
            return None

        return serialization.load_pem_private_key(
            pem_text.encode("utf-8"),
            password=None,
        )

    @staticmethod
    def _normalize_price(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            numeric = float(value)
            if numeric < 0:
                return None
            if numeric <= 100:
                return max(0.0, min(1.0, numeric / 100.0))
            for scale in (10_000.0, 1_000_000.0, 1_000_000_000.0):
                candidate = numeric / scale
                if 0.0 <= candidate <= 1.0:
                    return candidate
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if numeric < 0:
            return None
        if numeric <= 1:
            return max(0.0, min(1.0, numeric))
        for scale in (100.0, 10_000.0, 1_000_000.0, 1_000_000_000.0):
            candidate = numeric / scale
            if 0.0 <= candidate <= 1.0:
                return candidate
        return None

    @staticmethod
    def _normalize_size(value: Any) -> float:
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return 0.0

    def _extract_resolution_ts(self, market: dict[str, Any]) -> float | None:
        candidates: list[Any] = []
        for key in (
            "close_time",
            "closeTime",
            "end_time",
            "endTime",
            "close_date",
            "closeDate",
            "expiration_time",
            "expirationTime",
            "expiration_ts",
            "expirationTs",
            "resolution_time",
            "resolve_time",
            "resolve_date",
            "settlement_time",
            "settlementTime",
        ):
            if key in market:
                candidates.append(market.get(key))

        rules = market.get("rules")
        if isinstance(rules, dict):
            for key in (
                "close_time",
                "end_time",
                "expiration_time",
                "resolution_time",
                "resolve_time",
            ):
                if key in rules:
                    candidates.append(rules.get(key))

        for value in candidates:
            parsed = self._coerce_timestamp(value)
            if parsed is not None:
                return parsed
        return None

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
                return KalshiAdapter._coerce_timestamp(numeric)
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

    @classmethod
    def _extract_levels(cls, levels: Any) -> list[tuple[float, float]]:
        parsed: list[tuple[float, float]] = []
        if not isinstance(levels, Iterable):
            return parsed

        for entry in levels:
            price: float | None = None
            size: float | None = None
            if isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)) and len(entry) >= 2:
                price = cls._normalize_price(entry[0])
                size = cls._normalize_size(entry[1])
            elif isinstance(entry, dict):
                price = cls._normalize_price(entry.get("price"))
                size = cls._normalize_size(
                    entry.get("size")
                    or entry.get("quantity")
                    or entry.get("count")
                )

            if price is None or size is None or size <= 0:
                continue
            parsed.append((price, size))

        parsed.sort(key=lambda row: row[0], reverse=True)
        return parsed

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

        raw = result if isinstance(result, dict) else {"data": result}
        order = raw.get("order") if isinstance(raw.get("order"), dict) else raw

        order_id = order.get("order_id") or order.get("id")
        filled = order.get("filled_count") or order.get("filled") or requested_contracts

        avg_price = order.get("average_price")
        if avg_price is not None:
            try:
                avg_price = float(avg_price)
                if avg_price > 1:
                    avg_price /= 100.0
            except (TypeError, ValueError):
                avg_price = None

        try:
            filled_contracts = int(filled)
        except (TypeError, ValueError):
            filled_contracts = 0

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
        await self._client.aclose()

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> httpx.Response:
        attempts = max(1, len(self._api_base_urls))
        retry_budget_429 = 2
        retry_count_429 = 0
        last_exc: httpx.RequestError | None = None

        while True:
            should_retry_for_429 = False
            for attempt in range(attempts):
                try:
                    await self._throttle_request_rate(path)
                    response = await self._client.get(path, params=params)
                    if response is None:
                        raise RuntimeError(f"kalshi client returned no response for path={path}")
                    self._record_request_response(path, response)
                    if response.status_code == 429 and retry_count_429 < retry_budget_429:
                        retry_count_429 += 1
                        retry_after_seconds = self._retry_after_seconds(response)
                        if retry_after_seconds > 0:
                            await asyncio.sleep(retry_after_seconds)
                        should_retry_for_429 = True
                        break
                    return response
                except httpx.RequestError as exc:
                    last_exc = exc
                    self._record_request_error(path)
                    if not self._should_rotate_for_error(exc):
                        raise
                    if attempt >= attempts - 1:
                        raise
                    await self._rotate_api_base_url(exc)

            if should_retry_for_429:
                continue
            break

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("kalshi request failed without exception")

    async def _rotate_api_base_url(self, error: Exception) -> None:
        if len(self._api_base_urls) <= 1:
            return

        async with self._api_failover_lock:
            now_ts = time.time()
            if (now_ts - self._last_api_failover_ts) < self._failover_min_interval_seconds:
                return
            current_index = self._api_base_index
            next_index = (current_index + 1) % len(self._api_base_urls)
            if next_index == current_index:
                return

            old_url = self._api_base_urls[current_index]
            new_url = self._api_base_urls[next_index]
            old_client = self._client
            self._client = httpx.AsyncClient(
                base_url=new_url,
                timeout=self._timeout_seconds,
            )
            self._api_base_index = next_index
            self._last_api_failover_ts = now_ts
            LOGGER.warning("kalshi api failover: %s -> %s (error=%s)", old_url, new_url, error)
            with suppress(Exception):
                await old_client.aclose()

    async def _rotate_ws_url_if_needed(self, error: Exception) -> None:
        if len(self._ws_urls) <= 1:
            return
        if not self._should_rotate_for_error(error):
            return

        async with self._ws_failover_lock:
            now_ts = time.time()
            if (now_ts - self._last_ws_failover_ts) < self._failover_min_interval_seconds:
                return
            current_index = self._ws_index
            next_index = (current_index + 1) % len(self._ws_urls)
            if next_index == current_index:
                return
            old_url = self._ws_urls[current_index]
            new_url = self._ws_urls[next_index]
            self._ws_index = next_index
            self._last_ws_failover_ts = now_ts
            LOGGER.warning("kalshi ws failover: %s -> %s (error=%s)", old_url, new_url, error)

    def _active_api_base_url(self) -> str:
        return self._api_base_urls[self._api_base_index]

    def _active_ws_url(self) -> str:
        return self._ws_urls[self._ws_index]

    @staticmethod
    def _should_rotate_for_error(error: Exception) -> bool:
        text = str(error).lower()
        return (
            "nodename nor servname provided" in text
            or "name or service not known" in text
            or "temporary failure in name resolution" in text
            or "failed to resolve" in text
            or "getaddrinfo" in text
        )

    @staticmethod
    def _build_kalshi_host_candidates(url: str) -> list[str]:
        base = url.rstrip("/")
        parsed = urlparse(base)
        hostname = (parsed.hostname or "").lower()
        allow_host_failover = (
            str(os.getenv("KALSHI_ALLOW_HOST_FAILOVER") or "").strip().lower()
            in {"1", "true", "yes", "on"}
        )

        if hostname in {"api.kalshi.com", "api.elections.kalshi.com"}:
            candidates = [KalshiAdapter._replace_hostname(base, "api.elections.kalshi.com")]
            if allow_host_failover:
                candidates.append(KalshiAdapter._replace_hostname(base, "api.kalshi.com"))
            return KalshiAdapter._prefer_resolvable_urls(candidates)
        return [base]

    @staticmethod
    def _build_kalshi_ws_candidates(url: str) -> list[str]:
        base = url.rstrip("/")
        candidates: list[str] = []
        parsed = urlparse(base)
        if not parsed.scheme or not parsed.netloc:
            return [base]
        allow_host_failover = (
            str(os.getenv("KALSHI_ALLOW_HOST_FAILOVER") or "").strip().lower()
            in {"1", "true", "yes", "on"}
        )

        path_variants = [parsed.path or "/trade-api/ws/v2"]
        for path in ("/trade-api/ws/v2",):
            if path not in path_variants:
                path_variants.append(path)

        parsed_host = (parsed.hostname or "").lower()
        host_candidates: list[str]
        known_kalshi_host = parsed_host in {"api.kalshi.com", "api.elections.kalshi.com"}
        if parsed_host in {"api.kalshi.com", "api.elections.kalshi.com"}:
            host_candidates = ["api.elections.kalshi.com"]
            if allow_host_failover:
                host_candidates.append("api.kalshi.com")
        else:
            host_candidates = [parsed.hostname or ""]

        for hostname in host_candidates:
            if not hostname:
                continue
            for path in path_variants:
                rebuilt = KalshiAdapter._replace_hostname_and_path(base, hostname, path)
                if rebuilt and rebuilt not in candidates:
                    candidates.append(rebuilt)

        if not known_kalshi_host and base not in candidates:
            candidates.insert(0, base)
        return KalshiAdapter._prefer_resolvable_urls(candidates)

    @staticmethod
    def _hostname_resolves(hostname: str) -> bool:
        try:
            socket.getaddrinfo(hostname, None)
            return True
        except OSError:
            return False

    @staticmethod
    def _prefer_resolvable_urls(candidates: list[str]) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        resolvable: list[str] = []
        unresolved: list[str] = []
        for candidate in candidates:
            value = candidate.rstrip("/")
            if not value or value in seen:
                continue
            seen.add(value)
            host = (urlparse(value).hostname or "").strip().lower()
            if host and KalshiAdapter._hostname_resolves(host):
                resolvable.append(value)
            else:
                unresolved.append(value)
        ordered.extend(resolvable)
        if not ordered:
            ordered.extend(unresolved)
        return ordered

    async def _throttle_request_rate(self, path: str) -> None:
        endpoint = self._request_endpoint_key(path)
        min_interval = self._endpoint_interval_seconds.get(endpoint, self._request_base_interval_seconds)
        if min_interval <= 0.0:
            return
        lock = self._request_throttle_locks.setdefault(endpoint, asyncio.Lock())
        async with lock:
            now = time.monotonic()
            wait = self._endpoint_next_request_ts.get(endpoint, 0.0) - now
            if wait > 0:
                await asyncio.sleep(wait)
                now = time.monotonic()
            self._endpoint_next_request_ts[endpoint] = now + min_interval

    def _record_request_error(self, path: str) -> None:
        endpoint = self._request_endpoint_key(path)
        current = self._endpoint_interval_seconds.get(endpoint, self._request_base_interval_seconds)
        increased = min(
            self._request_max_interval_seconds,
            max(self._request_min_interval_seconds, current * 1.15),
        )
        self._endpoint_interval_seconds[endpoint] = increased
        self._endpoint_success_streak[endpoint] = 0

    def _record_request_response(self, path: str, response: httpx.Response) -> None:
        endpoint = self._request_endpoint_key(path)
        current = self._endpoint_interval_seconds.get(endpoint, self._request_base_interval_seconds)
        now = time.monotonic()

        if response.status_code == 429:
            retry_after_seconds = self._retry_after_seconds(response)
            increased = max(
                current * 1.75,
                self._request_base_interval_seconds if self._request_base_interval_seconds > 0 else 0.05,
                retry_after_seconds,
            )
            current = min(self._request_max_interval_seconds, increased)
            self._endpoint_success_streak[endpoint] = 0
        elif response.status_code >= 500:
            current = min(
                self._request_max_interval_seconds,
                max(self._request_min_interval_seconds, current * 1.25),
            )
            self._endpoint_success_streak[endpoint] = 0
        else:
            streak = self._endpoint_success_streak.get(endpoint, 0) + 1
            if streak >= 8:
                current = max(self._request_min_interval_seconds, current * 0.9)
                streak = 0
            self._endpoint_success_streak[endpoint] = streak

        self._endpoint_interval_seconds[endpoint] = current
        self._endpoint_next_request_ts[endpoint] = now + current

    def _request_endpoint_key(self, path: str) -> str:
        normalized = (path or "").strip().lower()
        if normalized.startswith("/markets/") and normalized.endswith("/orderbook"):
            return "market_orderbook"
        if normalized.startswith("/markets/"):
            return "market_detail"
        if normalized.startswith("/markets"):
            return "markets"
        if normalized.startswith("/events/"):
            return "event_detail"
        if normalized.startswith("/events"):
            return "events"
        return "default"

    @staticmethod
    def _retry_after_seconds(response: httpx.Response) -> float:
        raw = response.headers.get("Retry-After")
        if not raw:
            return 0.0

        try:
            numeric = float(raw)
            return max(0.0, numeric)
        except (TypeError, ValueError):
            pass

        try:
            dt = parsedate_to_datetime(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            return max(0.0, (dt - now).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return 0.0

    @staticmethod
    def _replace_hostname(url: str, hostname: str) -> str:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return url

        userinfo = ""
        if parsed.username:
            userinfo = parsed.username
            if parsed.password:
                userinfo += f":{parsed.password}"
            userinfo += "@"

        port = f":{parsed.port}" if parsed.port else ""
        netloc = f"{userinfo}{hostname}{port}"
        return parsed._replace(netloc=netloc).geturl().rstrip("/")

    @staticmethod
    def _replace_hostname_and_path(url: str, hostname: str, path: str) -> str:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return url
        normalized_path = path if path.startswith("/") else f"/{path}"

        userinfo = ""
        if parsed.username:
            userinfo = parsed.username
            if parsed.password:
                userinfo += f":{parsed.password}"
            userinfo += "@"

        port = f":{parsed.port}" if parsed.port else ""
        netloc = f"{userinfo}{hostname}{port}"
        return parsed._replace(netloc=netloc, path=normalized_path).geturl().rstrip("/")
