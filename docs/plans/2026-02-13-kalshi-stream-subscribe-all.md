# Kalshi Stream: Subscribe-to-All Architecture

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite Kalshi WebSocket streaming to subscribe to ALL markets without pre-filtering, eliminating the bootstrap timeout that kills cross-venue coverage.

**Architecture:** Replace the current "REST bootstrap → filter tickers → shard → subscribe specific" pipeline with "connect immediately → subscribe ALL on ticker channel → build market cache lazily from stream data + one background REST enrichment pass." Single WebSocket connection, no sharding needed.

**Tech Stack:** Python 3.9, asyncio, websockets library, existing BinaryQuote model

---

## Context

**Current flow (broken):**
1. `_build_stream_market_cache()` — REST calls to `/markets` (3 pages) + optional ticker enrichment → 8-20s, frequently times out
2. `_stream_subscription_tickers()` — Builds prioritized ticker list from cache (empty if bootstrap failed)
3. `_build_stream_subscription_shards()` — Splits into 250-ticker shards across N sockets
4. Subscribe each shard → ticker-specific messages only

**Problems:**
- Bootstrap timeout → empty cache → empty subscriptions → no data
- Even with partial cache, only covers ~300/1690 mapping tickers
- Individual ticker REST fallback = 600 HTTP calls/cycle
- Multiple WebSocket connections for sharding = unnecessary complexity

**New flow:**
1. Connect WebSocket immediately (no REST bootstrap)
2. Send single subscribe: `{"channels": ["ticker"]}` (no `market_tickers` → receives ALL)
3. Build market cache lazily as ticker messages arrive
4. Kick off one background REST task to enrich metadata (event_ticker, title) for cross-venue matching
5. Single socket, no sharding

**Key Kalshi API fact (confirmed via official docs):** Omitting `market_tickers` from the subscribe params subscribes to ALL markets on that channel. The ticker channel is event-driven (only fires on change), not a firehose.

---

### Task 1: Add `stream_subscribe_all` Config Flag

**Files:**
- Modify: `arb_bot/config.py:527-563` (KalshiSettings dataclass)
- Modify: `arb_bot/config.py:1154-1172` (env var loading)
- Test: `arb_bot/tests/test_kalshi_stream_subscribe_all.py`

**Step 1: Write the failing test**

```python
# arb_bot/tests/test_kalshi_stream_subscribe_all.py

"""Tests for Kalshi subscribe-all streaming mode."""
from __future__ import annotations

import os
from unittest import mock

from arb_bot.config import KalshiSettings, load_settings


def test_kalshi_settings_subscribe_all_default_false():
    """Default is False to preserve backward compat."""
    settings = KalshiSettings()
    assert settings.stream_subscribe_all is False


def test_kalshi_settings_subscribe_all_from_env():
    """Env var KALSHI_STREAM_SUBSCRIBE_ALL=true enables it."""
    with mock.patch.dict(os.environ, {"KALSHI_STREAM_SUBSCRIBE_ALL": "true"}, clear=False):
        app = load_settings()
        assert app.kalshi.stream_subscribe_all is True


def test_kalshi_settings_subscribe_all_env_false():
    with mock.patch.dict(os.environ, {"KALSHI_STREAM_SUBSCRIBE_ALL": "false"}, clear=False):
        app = load_settings()
        assert app.kalshi.stream_subscribe_all is False
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest arb_bot/tests/test_kalshi_stream_subscribe_all.py -v`
Expected: FAIL with `AttributeError: ... has no attribute 'stream_subscribe_all'`

**Step 3: Write minimal implementation**

In `arb_bot/config.py` KalshiSettings dataclass (after `stream_bootstrap_enrich_limit`):

```python
    stream_subscribe_all: bool = False
```

In `arb_bot/config.py` load_settings() Kalshi construction (after `stream_bootstrap_enrich_limit` line):

```python
            stream_subscribe_all=_as_bool(os.getenv("KALSHI_STREAM_SUBSCRIBE_ALL"), False),
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest arb_bot/tests/test_kalshi_stream_subscribe_all.py -v`
Expected: 3 PASSED

**Step 5: Commit**

```bash
git add arb_bot/config.py arb_bot/tests/test_kalshi_stream_subscribe_all.py
git commit -m "feat: add KALSHI_STREAM_SUBSCRIBE_ALL config flag"
```

---

### Task 2: Implement `_stream_quotes_subscribe_all()` Core Method

**Files:**
- Modify: `arb_bot/exchanges/kalshi.py` (add new method after `stream_quotes`)
- Test: `arb_bot/tests/test_kalshi_stream_subscribe_all.py`

**Step 1: Write the failing tests**

Append to `arb_bot/tests/test_kalshi_stream_subscribe_all.py`:

```python
import asyncio
import json
import types
from dataclasses import dataclass, field, replace
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch

from arb_bot.exchanges.kalshi import KalshiAdapter
from arb_bot.models import BinaryQuote


def _make_settings(**overrides) -> KalshiSettings:
    """Create KalshiSettings with sensible test defaults."""
    defaults = dict(
        enabled=True,
        api_base_url="https://api.elections.kalshi.com/trade-api/v2",
        ws_url="wss://api.elections.kalshi.com/trade-api/ws/v2",
        enable_stream=True,
        stream_subscribe_all=True,
        stream_reconnect_delay_seconds=0.01,
        stream_ping_interval_seconds=20.0,
        stream_subscription_batch_size=200,
        stream_subscription_retry_attempts=1,
        stream_subscription_ack_timeout_seconds=0.5,
        stream_max_tickers_per_socket=250,
        stream_priority_tickers=[],
        stream_pinned_tickers=[],
        stream_priority_refresh_limit=300,
        stream_allow_rest_topup=False,
        stream_bootstrap_scan_pages=0,
        stream_bootstrap_enrich_limit=0,
        market_tickers=[],
        market_limit=400,
        use_orderbook_quotes=False,
        max_orderbook_concurrency=4,
        market_scan_pages=1,
        market_page_size=100,
        request_pause_seconds=0.0,
        min_liquidity=0.0,
        exclude_ticker_prefixes=[],
        include_event_tickers=[],
        require_nondegenerate_quotes=False,
        maker_tick_size=0.01,
        maker_aggressiveness_ticks=2,
        taker_fee_per_contract=0.0,
        key_id="test-key",
        private_key_path=None,
        private_key_pem=None,
        events_429_circuit_threshold=6,
        events_429_circuit_cooldown_seconds=180.0,
    )
    defaults.update(overrides)
    return KalshiSettings(**defaults)


def _make_adapter(settings: KalshiSettings | None = None) -> KalshiAdapter:
    """Create adapter with mocked HTTP client."""
    adapter = KalshiAdapter.__new__(KalshiAdapter)
    adapter._settings = settings or _make_settings()
    adapter._timeout_seconds = 10.0
    adapter._client = AsyncMock()
    adapter._private_key = None
    adapter._discovery_cache_markets = []
    adapter._discovery_cache_ts = 0.0
    adapter._discovery_cache_ttl_seconds = 300.0
    adapter._events_429_consecutive = 0
    adapter._events_circuit_open_until_ts = 0.0
    adapter._stream_request_seq = 0
    adapter._last_api_failover_ts = 0.0
    adapter._last_ws_failover_ts = 0.0
    adapter._failover_min_interval_seconds = 2.0
    adapter._priority_refresh_cursor = 0
    adapter._request_base_interval_seconds = 0.0
    adapter._request_min_interval_seconds = 0.01
    adapter._request_max_interval_seconds = 2.0
    adapter._request_throttle_locks = {}
    adapter._endpoint_next_request_ts = {}
    adapter._endpoint_interval_seconds = {}
    adapter._endpoint_success_streak = {}
    adapter._dynamic_priority_refresh_limit = 300
    adapter._dynamic_priority_limit_success_streak = 0
    adapter._stream_active = False
    return adapter


class FakeWebSocket:
    """Fake WebSocket that yields pre-configured messages then blocks."""

    def __init__(self, messages: list[str]):
        self._messages = list(messages)
        self._sent: list[str] = []
        self._recv_index = 0

    async def recv(self) -> str:
        if self._recv_index < len(self._messages):
            msg = self._messages[self._recv_index]
            self._recv_index += 1
            return msg
        # Block forever (simulate idle connection)
        await asyncio.sleep(999)
        return ""

    async def send(self, data: str) -> None:
        self._sent.append(data)

    async def close(self) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


def test_subscribe_all_sends_no_market_tickers():
    """When stream_subscribe_all=True, subscribe command omits market_tickers."""
    adapter = _make_adapter()

    # Build a subscribe ack + one ticker message
    ack = json.dumps({"id": 1, "type": "subscribed", "msg": {}})
    ticker_msg = json.dumps({
        "type": "ticker",
        "msg": {
            "market_ticker": "KXTEST-26-YES",
            "yes_bid": 0.45,
            "yes_ask": 0.55,
            "no_bid": 0.40,
            "no_ask": 0.60,
        },
    })
    ws = FakeWebSocket([ack, ticker_msg])

    quotes: list[BinaryQuote] = []

    async def _run():
        with patch("arb_bot.exchanges.kalshi.websockets") as mock_ws:
            mock_ws.connect.return_value = ws

            # Collect quotes with timeout
            async def _collect():
                async for quote in adapter._stream_quotes_subscribe_all():
                    quotes.append(quote)
                    if len(quotes) >= 1:
                        break

            await asyncio.wait_for(_collect(), timeout=5.0)

    asyncio.get_event_loop().run_until_complete(_run())

    # Verify subscribe message was sent WITHOUT market_tickers
    assert len(ws._sent) >= 1
    sub_msg = json.loads(ws._sent[0])
    assert sub_msg["cmd"] == "subscribe"
    assert "channels" in sub_msg["params"]
    assert "ticker" in sub_msg["params"]["channels"]
    assert "market_tickers" not in sub_msg["params"]

    # Verify we got a quote
    assert len(quotes) == 1
    assert quotes[0].market_id == "KXTEST-26-YES"


def test_subscribe_all_builds_cache_lazily():
    """Market cache should be populated from stream data, not REST bootstrap."""
    adapter = _make_adapter()

    ticker1 = json.dumps({
        "type": "ticker",
        "msg": {
            "market_ticker": "KXFOO-26-BAR",
            "yes_bid": 0.30,
            "yes_ask": 0.40,
            "event_ticker": "KXFOO",
        },
    })
    ticker2 = json.dumps({
        "type": "ticker",
        "msg": {
            "market_ticker": "KXBAZ-26-QUX",
            "yes_bid": 0.60,
            "yes_ask": 0.70,
            "event_ticker": "KXBAZ",
        },
    })
    ack = json.dumps({"id": 1, "type": "subscribed", "msg": {}})
    ws = FakeWebSocket([ack, ticker1, ticker2])

    quotes: list[BinaryQuote] = []

    async def _run():
        with patch("arb_bot.exchanges.kalshi.websockets") as mock_ws:
            mock_ws.connect.return_value = ws
            async def _collect():
                async for quote in adapter._stream_quotes_subscribe_all():
                    quotes.append(quote)
                    if len(quotes) >= 2:
                        break
            await asyncio.wait_for(_collect(), timeout=5.0)

    asyncio.get_event_loop().run_until_complete(_run())

    assert len(quotes) == 2
    tickers = {q.market_id for q in quotes}
    assert tickers == {"KXFOO-26-BAR", "KXBAZ-26-QUX"}


def test_subscribe_all_reconnects_on_failure():
    """On WebSocket error, should reconnect with backoff."""
    adapter = _make_adapter(_make_settings(stream_reconnect_delay_seconds=0.01))

    call_count = 0
    ack = json.dumps({"id": 1, "type": "subscribed", "msg": {}})
    ticker_msg = json.dumps({
        "type": "ticker",
        "msg": {
            "market_ticker": "KXRECONN-26-OK",
            "yes_bid": 0.50,
            "yes_ask": 0.55,
        },
    })

    class FailOnceWebSocket(FakeWebSocket):
        def __init__(self):
            super().__init__([ack, ticker_msg])
            self._attempt = 0

        async def recv(self) -> str:
            if self._attempt == 0 and self._recv_index == 0:
                self._attempt += 1
                raise ConnectionError("simulated disconnect")
            return await super().recv()

    ws = FailOnceWebSocket()
    quotes: list[BinaryQuote] = []

    async def _run():
        with patch("arb_bot.exchanges.kalshi.websockets") as mock_ws:
            mock_ws.connect.return_value = ws
            async def _collect():
                async for quote in adapter._stream_quotes_subscribe_all():
                    quotes.append(quote)
                    if len(quotes) >= 1:
                        break
            await asyncio.wait_for(_collect(), timeout=5.0)

    asyncio.get_event_loop().run_until_complete(_run())
    assert len(quotes) >= 1
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest arb_bot/tests/test_kalshi_stream_subscribe_all.py::test_subscribe_all_sends_no_market_tickers -v`
Expected: FAIL with `AttributeError: 'KalshiAdapter' has no attribute '_stream_quotes_subscribe_all'`

**Step 3: Write the implementation**

Add new method to `arb_bot/exchanges/kalshi.py` after the existing `stream_quotes()` method (after line 207):

```python
    async def _stream_quotes_subscribe_all(self) -> AsyncIterator[BinaryQuote]:
        """Stream ALL Kalshi ticker updates without pre-filtering.

        Unlike the sharded approach, this subscribes to the ``ticker`` channel
        with no ``market_tickers`` filter, receiving updates for every active
        market.  Market metadata is built lazily from incoming messages plus
        a single background REST enrichment pass.

        Advantages:
        - No REST bootstrap required before streaming starts
        - Single WebSocket connection (no sharding)
        - Automatic coverage of new markets
        - Eliminates bootstrap timeout problem entirely
        """
        market_cache: dict[str, dict[str, Any]] = {}
        market_lookup: dict[str, str] = {}
        queue: asyncio.Queue[BinaryQuote] = asyncio.Queue()
        reconnect_delay = max(0.5, self._settings.stream_reconnect_delay_seconds)
        ping_interval = max(5.0, self._settings.stream_ping_interval_seconds)

        # Kick off background REST enrichment (non-blocking).
        enrich_task: asyncio.Task[None] | None = None

        async def _background_enrich() -> None:
            """Fetch market metadata via REST to populate event_ticker, title, etc."""
            try:
                await asyncio.sleep(2.0)  # Let stream warm up first.
                summaries = await self._fetch_market_summaries_paged()
                if summaries:
                    for market in summaries:
                        ticker = str(market.get("ticker") or market.get("market_ticker") or "").strip()
                        if ticker and ticker not in market_cache:
                            market_cache[ticker] = dict(market)
                    LOGGER.info(
                        "kalshi subscribe-all background enrichment loaded %d market summaries",
                        len(summaries),
                    )
                # Also backfill priority tickers via event fetches.
                summaries = await self._backfill_priority_tickers_into_summaries(summaries or [])
                for market in summaries:
                    ticker = str(market.get("ticker") or market.get("market_ticker") or "").strip()
                    if ticker and ticker not in market_cache:
                        market_cache[ticker] = dict(market)
                LOGGER.info(
                    "kalshi subscribe-all background enrichment total cache: %d markets",
                    len(market_cache),
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                LOGGER.warning("kalshi subscribe-all background enrichment failed: %s", exc)

        async def _ws_loop() -> None:
            """Single WebSocket connection with reconnection."""
            nonlocal enrich_task
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
                        # Subscribe to ALL markets on ticker channel.
                        request_id = self._next_stream_request_id()
                        subscribe_payload = json.dumps({
                            "id": request_id,
                            "cmd": "subscribe",
                            "params": {"channels": ["ticker"]},
                        })
                        await socket.send(subscribe_payload)
                        LOGGER.info(
                            "kalshi subscribe-all connected and subscribed to ALL tickers ws_url=%s",
                            ws_url,
                        )

                        # Start background enrichment on first successful connection.
                        if enrich_task is None or enrich_task.done():
                            enrich_task = asyncio.create_task(_background_enrich())

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
                                output_queue=queue,
                            )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    LOGGER.warning(
                        "kalshi subscribe-all connection failed ws_url=%s: %s",
                        ws_url,
                        exc,
                    )
                    await self._rotate_ws_url_if_needed(exc)
                    await asyncio.sleep(reconnect_delay)

        ws_task = asyncio.create_task(_ws_loop())

        try:
            while True:
                quote = await queue.get()
                yield quote
        finally:
            ws_task.cancel()
            if enrich_task is not None:
                enrich_task.cancel()
            await asyncio.gather(ws_task, return_exceptions=True)
            if enrich_task is not None:
                await asyncio.gather(enrich_task, return_exceptions=True)
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest arb_bot/tests/test_kalshi_stream_subscribe_all.py -v`
Expected: 6 PASSED (3 from Task 1 + 3 new)

**Step 5: Commit**

```bash
git add arb_bot/exchanges/kalshi.py arb_bot/tests/test_kalshi_stream_subscribe_all.py
git commit -m "feat: implement _stream_quotes_subscribe_all with lazy cache"
```

---

### Task 3: Wire `stream_quotes()` to Use Subscribe-All When Enabled

**Files:**
- Modify: `arb_bot/exchanges/kalshi.py:139-207` (stream_quotes method)
- Test: `arb_bot/tests/test_kalshi_stream_subscribe_all.py`

**Step 1: Write the failing test**

Append to test file:

```python
def test_stream_quotes_delegates_to_subscribe_all():
    """stream_quotes() should use subscribe-all path when flag is True."""
    adapter = _make_adapter(_make_settings(stream_subscribe_all=True))
    # Monkey-patch _has_stream_auth to return True
    adapter._has_stream_auth = lambda: True

    ack = json.dumps({"id": 1, "type": "subscribed", "msg": {}})
    ticker_msg = json.dumps({
        "type": "ticker",
        "msg": {
            "market_ticker": "KXDELEGATE-26-OK",
            "yes_bid": 0.50,
            "yes_ask": 0.55,
        },
    })
    ws = FakeWebSocket([ack, ticker_msg])
    quotes: list[BinaryQuote] = []

    async def _run():
        with patch("arb_bot.exchanges.kalshi.websockets") as mock_ws:
            mock_ws.connect.return_value = ws
            async def _collect():
                async for quote in adapter.stream_quotes():
                    quotes.append(quote)
                    if len(quotes) >= 1:
                        break
            await asyncio.wait_for(_collect(), timeout=5.0)

    asyncio.get_event_loop().run_until_complete(_run())

    # Should use subscribe-all (no market_tickers in subscribe msg)
    assert len(quotes) == 1
    sub_msg = json.loads(ws._sent[0])
    assert "market_tickers" not in sub_msg["params"]


def test_stream_quotes_uses_legacy_when_flag_false():
    """stream_quotes() should use old sharded path when flag is False."""
    adapter = _make_adapter(_make_settings(
        stream_subscribe_all=False,
        stream_bootstrap_scan_pages=0,
    ))
    adapter._has_stream_auth = lambda: True

    # Legacy path will try bootstrap + shard, should still work with empty shards
    ack = json.dumps({"id": 1, "type": "subscribed", "msg": {}})
    ticker_msg = json.dumps({
        "type": "ticker",
        "msg": {
            "market_ticker": "KXLEGACY-26-OK",
            "yes_bid": 0.50,
            "yes_ask": 0.55,
        },
    })
    ws = FakeWebSocket([ack, ticker_msg])
    quotes: list[BinaryQuote] = []

    async def _run():
        with patch("arb_bot.exchanges.kalshi.websockets") as mock_ws:
            mock_ws.connect.return_value = ws
            async def _collect():
                async for quote in adapter.stream_quotes():
                    quotes.append(quote)
                    if len(quotes) >= 1:
                        break
            await asyncio.wait_for(_collect(), timeout=5.0)

    asyncio.get_event_loop().run_until_complete(_run())
    assert len(quotes) == 1
```

**Step 2: Run to verify failure**

Run: `python3 -m pytest arb_bot/tests/test_kalshi_stream_subscribe_all.py::test_stream_quotes_delegates_to_subscribe_all -v`
Expected: FAIL (stream_quotes doesn't yet delegate)

**Step 3: Modify `stream_quotes()` to conditionally delegate**

Replace the first 5 lines of `stream_quotes()` (lines 139-143) with:

```python
    async def stream_quotes(self) -> AsyncIterator[BinaryQuote]:
        if not self.supports_streaming():
            return

        self._stream_active = True

        # Use subscribe-all mode when enabled — skips REST bootstrap entirely.
        if self._settings.stream_subscribe_all:
            async for quote in self._stream_quotes_subscribe_all():
                yield quote
            return

        # Legacy sharded path below...
        bootstrap_timeout_seconds = max(5.0, self._timeout_seconds * 2.0)
```

**Step 4: Run all tests**

Run: `python3 -m pytest arb_bot/tests/test_kalshi_stream_subscribe_all.py -v`
Expected: 8 PASSED

Run: `python3 -m pytest arb_bot/tests/ -x -q`
Expected: 2242+ PASSED

**Step 5: Commit**

```bash
git add arb_bot/exchanges/kalshi.py arb_bot/tests/test_kalshi_stream_subscribe_all.py
git commit -m "feat: wire stream_quotes to delegate to subscribe-all when enabled"
```

---

### Task 4: Add Background REST Enrichment Tests

**Files:**
- Test: `arb_bot/tests/test_kalshi_stream_subscribe_all.py`

**Step 1: Write enrichment tests**

Append to test file:

```python
def test_subscribe_all_background_enrichment_populates_cache():
    """Background REST should enrich market_cache with event_ticker, title."""
    adapter = _make_adapter(_make_settings(
        stream_subscribe_all=True,
        market_scan_pages=1,
        market_page_size=100,
        stream_priority_tickers=["KXENRICH-26-YES"],
    ))

    # Mock the REST paged fetcher to return market summaries
    rest_summaries = [
        {"ticker": "KXENRICH-26-YES", "event_ticker": "KXENRICH", "title": "Test Enriched", "yes_bid": 0.50, "yes_ask": 0.55},
    ]

    async def _fake_fetch_paged(*args, **kwargs):
        return rest_summaries

    adapter._fetch_market_summaries_paged = _fake_fetch_paged

    async def _fake_backfill(summaries):
        return summaries

    adapter._backfill_priority_tickers_into_summaries = _fake_backfill

    ack = json.dumps({"id": 1, "type": "subscribed", "msg": {}})
    ticker_msg = json.dumps({
        "type": "ticker",
        "msg": {
            "market_ticker": "KXENRICH-26-YES",
            "yes_bid": 0.60,
            "yes_ask": 0.65,
        },
    })
    ws = FakeWebSocket([ack, ticker_msg])
    quotes: list[BinaryQuote] = []

    async def _run():
        with patch("arb_bot.exchanges.kalshi.websockets") as mock_ws:
            mock_ws.connect.return_value = ws
            async def _collect():
                async for quote in adapter._stream_quotes_subscribe_all():
                    quotes.append(quote)
                    if len(quotes) >= 1:
                        break
            await asyncio.wait_for(_collect(), timeout=5.0)

    asyncio.get_event_loop().run_until_complete(_run())
    assert len(quotes) >= 1
```

**Step 2: Run test**

Run: `python3 -m pytest arb_bot/tests/test_kalshi_stream_subscribe_all.py -v`
Expected: 9 PASSED

**Step 3: Commit**

```bash
git add arb_bot/tests/test_kalshi_stream_subscribe_all.py
git commit -m "test: add background enrichment tests for subscribe-all"
```

---

### Task 5: Set KALSHI_STREAM_SUBSCRIBE_ALL=true in .env and Run Integration Test

**Files:**
- Modify: `arb_bot/.env`

**Step 1: Update .env**

Add to `arb_bot/.env`:

```
KALSHI_STREAM_SUBSCRIBE_ALL=true
```

**Step 2: Run full test suite**

Run: `python3 -m pytest arb_bot/tests/ -x -q`
Expected: 2242+ PASSED (no regressions)

**Step 3: Integration test — start paper sim in stream mode**

```bash
cd /Users/danielelman/Documents/New\ project/arb_bot/.claude/worktrees/focused-mclean
nohup python3 -m arb_bot.main --paper-minutes 10 --stream --paper-output arb_bot/output/paper_stream_subscribeall_test.csv > arb_bot/output/paper_stream_subscribeall_test.log 2>&1 &
echo "PID: $!"
```

Wait 3 minutes, then check log:

```bash
grep -E "subscribe-all|cross_pairs|Found" arb_bot/output/paper_stream_subscribeall_test.log
```

Expected:
- `kalshi subscribe-all connected and subscribed to ALL tickers`
- `cross_pairs=50+/1690` (non-zero cross-venue coverage)
- No bootstrap timeout warnings

**Step 4: Commit .env change**

```bash
git add arb_bot/.env
git commit -m "feat: enable subscribe-all streaming mode in .env"
```

---

### Task 6: Run Full Regression + Overnight Stream Sim

**Step 1: Full regression**

Run: `python3 -m pytest arb_bot/tests/ -x -q`
Expected: All tests pass

**Step 2: Start overnight stream-mode paper sim**

```bash
kill $(pgrep -f "arb_bot.main") 2>/dev/null
nohup python3 -m arb_bot.main --paper-minutes 480 --stream --paper-output arb_bot/output/paper_overnight_stream_v4_20260213.csv > arb_bot/output/paper_overnight_stream_v4_20260213.log 2>&1 &
echo "PID: $!"
```

**Step 3: Monitor first 5 minutes**

```bash
sleep 300 && grep -E "subscribe-all|cross_pairs|Found|decision breakdown" arb_bot/output/paper_overnight_stream_v4_20260213.log | head -20
```

Expected:
- `subscribe-all connected` — immediate connection, no bootstrap delay
- `cross_pairs > 0` — cross-venue coverage working
- Opportunities detected across all lanes
- Trade decisions happening

**Step 4: Final commit**

```bash
git add -A
git commit -m "feat: Kalshi subscribe-all streaming — eliminates bootstrap timeout, enables cross-venue coverage"
```

---

## Summary of Changes

| File | Change |
|------|--------|
| `arb_bot/config.py` | Add `stream_subscribe_all: bool = False` to KalshiSettings + env var loading |
| `arb_bot/exchanges/kalshi.py` | Add `_stream_quotes_subscribe_all()` method; modify `stream_quotes()` to delegate |
| `arb_bot/.env` | Set `KALSHI_STREAM_SUBSCRIBE_ALL=true` |
| `arb_bot/tests/test_kalshi_stream_subscribe_all.py` | New test file with 9+ tests |

**Total estimated implementation time:** 30-45 minutes

**Risk:** Low — the old sharded path is preserved as fallback when `stream_subscribe_all=False`. Can revert by setting env var to false.
