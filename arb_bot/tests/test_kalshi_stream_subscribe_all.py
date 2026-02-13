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


import asyncio
import json
from unittest.mock import AsyncMock, patch

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


def _make_adapter(settings=None):
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
    adapter._ws_urls = [adapter._settings.ws_url]
    adapter._ws_index = 0
    adapter._api_base_urls = [adapter._settings.api_base_url]
    adapter._api_base_index = 0
    adapter._api_failover_lock = asyncio.Lock()
    adapter._ws_failover_lock = asyncio.Lock()
    return adapter


class FakeWebSocket:
    """Fake WebSocket that yields pre-configured messages then blocks."""
    def __init__(self, messages):
        self._messages = list(messages)
        self._sent = []
        self._recv_index = 0
    async def recv(self):
        if self._recv_index < len(self._messages):
            msg = self._messages[self._recv_index]
            self._recv_index += 1
            return msg
        await asyncio.sleep(999)
        return ""
    async def send(self, data):
        self._sent.append(data)
    async def close(self):
        pass
    async def __aenter__(self):
        return self
    async def __aexit__(self, *args):
        pass


def test_subscribe_all_sends_no_market_tickers():
    """When stream_subscribe_all=True, subscribe command omits market_tickers."""
    adapter = _make_adapter()
    ack = json.dumps({"id": 1, "type": "subscribed", "msg": {}})
    ticker_msg = json.dumps({
        "type": "ticker",
        "msg": {
            "market_ticker": "KXTEST-26-YES",
            "yes_bid": 0.45,
            "yes_ask": 0.55,
            "no_bid": 0.40,
            "no_ask": 0.60,
            "open_interest": 100,
        },
    })
    ws = FakeWebSocket([ack, ticker_msg])
    quotes = []

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
    assert len(ws._sent) >= 1
    sub_msg = json.loads(ws._sent[0])
    assert sub_msg["cmd"] == "subscribe"
    assert "ticker" in sub_msg["params"]["channels"]
    assert "market_tickers" not in sub_msg["params"]
    assert len(quotes) == 1
    assert quotes[0].market_id == "KXTEST-26-YES"


def test_subscribe_all_builds_cache_lazily():
    """Market cache populated from stream data, not REST bootstrap."""
    adapter = _make_adapter()
    ticker1 = json.dumps({"type": "ticker", "msg": {"market_ticker": "KXFOO-26-BAR", "yes_bid": 0.30, "yes_ask": 0.40, "event_ticker": "KXFOO", "open_interest": 50}})
    ticker2 = json.dumps({"type": "ticker", "msg": {"market_ticker": "KXBAZ-26-QUX", "yes_bid": 0.60, "yes_ask": 0.70, "event_ticker": "KXBAZ", "open_interest": 50}})
    ack = json.dumps({"id": 1, "type": "subscribed", "msg": {}})
    ws = FakeWebSocket([ack, ticker1, ticker2])
    quotes = []

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
    assert {q.market_id for q in quotes} == {"KXFOO-26-BAR", "KXBAZ-26-QUX"}


def test_subscribe_all_reconnects_on_failure():
    """On WebSocket error, should reconnect with backoff."""
    adapter = _make_adapter(_make_settings(stream_reconnect_delay_seconds=0.01))
    ack = json.dumps({"id": 1, "type": "subscribed", "msg": {}})
    ticker_msg = json.dumps({"type": "ticker", "msg": {"market_ticker": "KXRECONN-26-OK", "yes_bid": 0.50, "yes_ask": 0.55, "open_interest": 50}})

    class FailOnceWebSocket(FakeWebSocket):
        def __init__(self):
            super().__init__([ack, ticker_msg])
            self._attempt = 0
        async def recv(self):
            if self._attempt == 0 and self._recv_index == 0:
                self._attempt += 1
                raise ConnectionError("simulated disconnect")
            return await super().recv()

    ws = FailOnceWebSocket()
    quotes = []

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


def test_stream_quotes_delegates_to_subscribe_all():
    """stream_quotes() should use subscribe-all path when flag is True."""
    adapter = _make_adapter(_make_settings(stream_subscribe_all=True))
    adapter._has_stream_auth = lambda: True

    ack = json.dumps({"id": 1, "type": "subscribed", "msg": {}})
    ticker_msg = json.dumps({"type": "ticker", "msg": {"market_ticker": "KXDELEGATE-26-OK", "yes_bid": 0.50, "yes_ask": 0.55, "open_interest": 50}})
    ws = FakeWebSocket([ack, ticker_msg])
    quotes = []

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
    sub_msg = json.loads(ws._sent[0])
    assert "market_tickers" not in sub_msg["params"]


def test_stream_quotes_uses_legacy_when_flag_false():
    """stream_quotes() should use old sharded path when flag is False."""
    adapter = _make_adapter(_make_settings(stream_subscribe_all=False, stream_bootstrap_scan_pages=0))
    adapter._has_stream_auth = lambda: True

    ack = json.dumps({"id": 1, "type": "subscribed", "msg": {}})
    ticker_msg = json.dumps({"type": "ticker", "msg": {"market_ticker": "KXLEGACY-26-OK", "yes_bid": 0.50, "yes_ask": 0.55, "open_interest": 50}})
    ws = FakeWebSocket([ack, ticker_msg])
    quotes = []

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
            "no_bid": 0.30,
            "no_ask": 0.40,
            "open_interest": 100,
        },
    })
    ws = FakeWebSocket([ack, ticker_msg])
    quotes = []

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


def test_fetch_quotes_uses_paged_scan_when_subscribe_all_active():
    """When subscribe_all=True and stream_active, fetch_quotes uses paged scan not individual tickers."""
    adapter = _make_adapter(_make_settings(
        stream_subscribe_all=True,
        stream_priority_tickers=["KXTICKER-26-YES", "KXTICKER-26-NO"],
    ))
    adapter._stream_active = True  # Simulate active stream

    paged_called = []
    ticker_called = []

    async def _fake_paged(*args, **kwargs):
        paged_called.append(True)
        return [
            {"ticker": "KXTICKER-26-YES", "yes_bid": 0.50, "yes_ask": 0.55,
             "no_bid": 0.40, "no_ask": 0.50, "open_interest": 100},
        ]

    async def _fake_ticker_quotes(tickers):
        ticker_called.append(tickers)
        return []

    async def _fake_backfill(summaries):
        return summaries

    adapter._fetch_market_summaries_paged = _fake_paged
    adapter._fetch_ticker_quotes = _fake_ticker_quotes
    adapter._backfill_priority_tickers_into_summaries = _fake_backfill

    async def _run():
        return await adapter._fetch_quotes_impl(force_full_scan=False)

    asyncio.get_event_loop().run_until_complete(_run())

    # Should use paged scan, NOT individual ticker fetches
    assert len(paged_called) == 1, "Expected paged scan to be called"
    assert len(ticker_called) == 0, "Individual ticker fetch should NOT be called in subscribe-all mode"


def test_ticker_message_preserves_rest_cached_no_bid():
    """Ticker messages lacking no_bid/no_ask should preserve REST-cached values.

    Background REST enrichment populates market_cache with real no_bid/no_ask
    from the /markets endpoint.  When a subsequent ticker message arrives (which
    only contains yes_bid/yes_ask), the REST-sourced NO-side prices must survive
    in the merged market dict, giving choose_effective_buy_price both the
    direct-ask and opposite-bid-transform paths.

    This test uses a delayed FakeWebSocket that delivers the ticker message
    AFTER background enrichment has populated the cache (enrichment sleeps 2s,
    ticker arrives at ~3s), ensuring the correct ordering.
    """
    adapter = _make_adapter(_make_settings(
        stream_subscribe_all=True,
        stream_priority_tickers=["KXPRES-26-YES"],
    ))

    # REST enrichment returns a market with real no_bid / no_ask.
    rest_summaries = [
        {
            "ticker": "KXPRES-26-YES",
            "event_ticker": "KXPRES",
            "title": "Presidential Test",
            "yes_bid": 0.50,
            "yes_ask": 0.55,
            "no_bid": 0.42,
            "no_ask": 0.50,
            "open_interest": 200,
        },
    ]

    async def _fake_fetch_paged(*args, **kwargs):
        return rest_summaries

    async def _fake_backfill(summaries):
        return summaries

    adapter._fetch_market_summaries_paged = _fake_fetch_paged
    adapter._backfill_priority_tickers_into_summaries = _fake_backfill

    # Ticker message only has yes_bid/yes_ask (no no_bid/no_ask), matching
    # real Kalshi ticker channel behaviour.
    ack = json.dumps({"id": 1, "type": "subscribed", "msg": {}})
    ticker_msg = json.dumps({
        "type": "ticker",
        "msg": {
            "market_ticker": "KXPRES-26-YES",
            "yes_bid": 0.52,
            "yes_ask": 0.57,
            "open_interest": 210,
        },
    })

    class DelayedWebSocket(FakeWebSocket):
        """Delivers ack immediately, then delays ticker message until after enrichment."""
        def __init__(self):
            super().__init__([ack])
            self._ticker_delivered = False
        async def recv(self):
            if self._recv_index < len(self._messages):
                # Deliver ack immediately.
                return await super().recv()
            if not self._ticker_delivered:
                # Wait long enough for background enrichment to complete.
                # Enrichment sleeps 2s then processes; we wait 3s to be safe.
                await asyncio.sleep(3.0)
                self._ticker_delivered = True
                return ticker_msg
            # Block indefinitely after ticker delivered.
            await asyncio.sleep(999)
            return ""

    ws = DelayedWebSocket()
    quotes: list[BinaryQuote] = []

    async def _run():
        with patch("arb_bot.exchanges.kalshi.websockets") as mock_ws:
            mock_ws.connect.return_value = ws
            async def _collect():
                async for quote in adapter._stream_quotes_subscribe_all():
                    quotes.append(quote)
                    # Wait for the REST enrichment quote(s) plus the ticker quote.
                    # REST enrichment emits 1 quote, then ticker emits 1 more.
                    if len(quotes) >= 2:
                        break
            await asyncio.wait_for(_collect(), timeout=10.0)

    asyncio.get_event_loop().run_until_complete(_run())

    # We expect at least 2 quotes: one from REST enrichment, one from ticker.
    assert len(quotes) >= 2, f"Expected >=2 quotes, got {len(quotes)}"

    # The first quote should be from REST enrichment (yes_bid=0.50).
    rest_quote = quotes[0]
    assert rest_quote.market_id == "KXPRES-26-YES"
    assert abs(rest_quote.yes_bid_price - 0.50) < 0.01

    # The second (last) quote should be from the ticker message, with:
    # - yes_bid updated to 0.52 (from ticker)
    # - no_bid preserved at 0.42 (from REST cache, since ticker didn't include it)
    ticker_quote = quotes[-1]
    assert ticker_quote.market_id == "KXPRES-26-YES"

    assert ticker_quote.no_bid_price is not None, (
        "no_bid_price should be preserved from REST cache, not None"
    )
    assert abs(ticker_quote.no_bid_price - 0.42) < 0.01, (
        f"Expected no_bid_price ~0.42 from REST cache, got {ticker_quote.no_bid_price}"
    )

    # Verify yes_bid was updated by the ticker message (not stale REST value).
    assert ticker_quote.yes_bid_price is not None
    assert abs(ticker_quote.yes_bid_price - 0.52) < 0.01, (
        f"Expected yes_bid_price ~0.52 from ticker update, got {ticker_quote.yes_bid_price}"
    )


def test_kalshi_single_book_invariant_yes_ask_plus_no_ask_gte_one():
    """On Kalshi (single matching engine), yes_ask + no_ask >= 1.0 always.

    Kalshi's matching engine treats YES and NO as complements of the same
    order book, enforcing yes_best_bid + no_best_bid <= 1.0.  Since:

        yes_ask = 1.0 - no_best_bid
        no_ask  = 1.0 - yes_best_bid

    then: yes_ask + no_ask = 2.0 - (yes_best_bid + no_best_bid) >= 1.0.

    This means intra-venue arb (total < 1.0) is structurally impossible.
    This test verifies the invariant using _quote_from_market_summary directly.
    """
    adapter = _make_adapter()

    # Test a range of valid Kalshi orderbook states where
    # yes_best_bid + no_best_bid <= 1.0 (exchange invariant).
    test_cases = [
        # (yes_bid, yes_ask, no_bid, no_ask, description)
        (0.50, 0.55, 0.42, 0.50, "typical spread market"),
        (0.01, 0.05, 0.01, 0.99, "extreme low YES"),
        (0.95, 0.99, 0.01, 0.05, "extreme high YES"),
        (0.50, 0.52, 0.48, 0.50, "tight spread"),
        (0.30, 0.40, 0.55, 0.70, "wide spread low YES"),
        (0.10, 0.20, 0.10, 0.90, "low liquidity both sides"),
        (0.50, 0.51, 0.49, 0.50, "near-balanced penny spread"),
    ]

    for yes_bid, yes_ask, no_bid, no_ask, desc in test_cases:
        # Verify exchange invariant holds in test data.
        assert yes_bid + no_bid <= 1.0 + 1e-9, (
            f"Test data violates exchange invariant: {desc}"
        )

        market = {
            "ticker": f"KXTEST-INV-{desc.replace(' ', '_').upper()[:10]}",
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "no_bid": no_bid,
            "no_ask": no_ask,
            "open_interest": 100,
        }
        quote = adapter._quote_from_market_summary(market)
        assert quote is not None, f"Quote should not be None for: {desc}"

        total = quote.yes_buy_price + quote.no_buy_price
        assert total >= 1.0 - 1e-9, (
            f"Kalshi invariant violated for {desc}: "
            f"yes_buy={quote.yes_buy_price:.4f} + no_buy={quote.no_buy_price:.4f} "
            f"= {total:.4f} < 1.0"
        )


def test_kalshi_single_book_derived_prices_also_satisfy_invariant():
    """Even with derived (complement) prices, yes_ask + no_ask >= 1.0.

    When no_bid and no_ask are absent (as in ticker-only data before REST
    enrichment), prices are derived as complements:
        no_ask = 1.0 - yes_bid
    This also guarantees the invariant: yes_ask + (1 - yes_bid) = yes_ask - yes_bid + 1.0 >= 1.0
    since yes_ask >= yes_bid (ask >= bid by definition).
    """
    adapter = _make_adapter()

    test_cases = [
        (0.30, 0.40, "wide spread"),
        (0.50, 0.51, "penny spread"),
        (0.01, 0.99, "maximum spread"),
        (0.50, 0.50, "zero spread"),
        (0.75, 0.80, "high YES"),
    ]

    for yes_bid, yes_ask, desc in test_cases:
        market = {
            "ticker": f"KXTEST-DER-{desc.replace(' ', '_').upper()[:10]}",
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            # no_bid and no_ask intentionally absent — will be derived.
            "open_interest": 100,
        }
        quote = adapter._quote_from_market_summary(market)
        assert quote is not None, f"Quote should not be None for: {desc}"

        total = quote.yes_buy_price + quote.no_buy_price
        assert total >= 1.0 - 1e-9, (
            f"Derived-price invariant violated for {desc}: "
            f"yes_buy={quote.yes_buy_price:.4f} + no_buy={quote.no_buy_price:.4f} "
            f"= {total:.4f} < 1.0"
        )
