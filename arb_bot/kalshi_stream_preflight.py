from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import replace
from typing import Any

from arb_bot.config import load_settings
from arb_bot.exchanges.kalshi import KalshiAdapter
from arb_bot.logging_setup import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kalshi REST + websocket preflight diagnostic.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=45.0,
        help="Max time to wait for first websocket quote.",
    )
    parser.add_argument(
        "--rest-only",
        action="store_true",
        help="Run only REST quote preflight.",
    )
    parser.add_argument(
        "--sample-tickers",
        type=int,
        default=40,
        help="How many tickers to use for websocket subscription preflight.",
    )
    return parser.parse_args()


async def _first_stream_quote(adapter: KalshiAdapter):
    async for quote in adapter.stream_quotes():
        return quote
    return None


def _extract_markets(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        raw = payload.get("markets")
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


async def _run() -> int:
    args = parse_args()
    base_settings = load_settings()
    # Keep preflight lightweight and deterministic even when the runtime profile
    # is full-universe. This is a health check, not a full scan.
    kalshi_settings = replace(
        base_settings.kalshi,
        market_limit=min(max(10, args.sample_tickers), 120),
        market_scan_pages=1,
        market_page_size=100,
        max_orderbook_concurrency=2,
        request_pause_seconds=max(0.2, base_settings.kalshi.request_pause_seconds),
        use_orderbook_quotes=False,
    )
    settings = replace(base_settings, stream_mode=True, kalshi=kalshi_settings)
    configure_logging(settings.log_level)

    logger = logging.getLogger("arb_bot.kalshi_preflight")
    adapter = KalshiAdapter(settings.kalshi)

    logger.info(
        "kalshi preflight api=%s ws=%s stream_enabled=%s supports_streaming=%s key_id_present=%s private_key_present=%s priority_tickers=%d",
        settings.kalshi.api_base_url,
        settings.kalshi.ws_url,
        settings.kalshi.enable_stream,
        adapter.supports_streaming(),
        bool(settings.kalshi.key_id),
        bool(getattr(adapter, "_private_key", None)),
        len(settings.kalshi.stream_priority_tickers),
    )

    try:
        tickers: list[str] = []
        try:
            response = await adapter._get("/markets", params={"status": "open", "limit": settings.kalshi.market_page_size})
            response.raise_for_status()
            markets = _extract_markets(response.json())
            tickers = [
                str(item.get("ticker") or item.get("market_ticker") or "").strip()
                for item in markets
                if str(item.get("ticker") or item.get("market_ticker") or "").strip()
            ]
            logger.info("kalshi REST preflight markets_page_count=%d", len(tickers))
        except Exception as exc:
            logger.warning("kalshi REST preflight failed (continuing to websocket check): %s", exc)

        if args.rest_only:
            return 0 if tickers else 2

        if not tickers and settings.kalshi.market_tickers:
            tickers = [ticker.strip() for ticker in settings.kalshi.market_tickers if ticker.strip()]
        stream_tickers = tickers[: max(1, args.sample_tickers)] if tickers else []
        if not stream_tickers and settings.kalshi.stream_priority_tickers:
            stream_tickers = settings.kalshi.stream_priority_tickers[: max(1, args.sample_tickers)]
        stream_settings = replace(
            settings.kalshi,
            market_tickers=stream_tickers,
            stream_priority_tickers=stream_tickers,
            market_limit=max(1, len(stream_tickers)),
            market_scan_pages=1,
        )
        await adapter.aclose()
        adapter = KalshiAdapter(stream_settings)
        logger.info("kalshi websocket preflight subscribing tickers=%d", len(stream_tickers))

        if not adapter.supports_streaming():
            logger.error("kalshi websocket preflight skipped: supports_streaming=false")
            return 3

        try:
            quote = await asyncio.wait_for(_first_stream_quote(adapter), timeout=max(1.0, args.timeout_seconds))
        except asyncio.TimeoutError:
            logger.error("kalshi websocket preflight timed out waiting for first quote")
            return 4
        except Exception as exc:
            logger.error("kalshi websocket preflight failed: %s", exc)
            return 5

        if quote is None:
            logger.error("kalshi websocket preflight ended without yielding quote")
            return 6

        logger.info(
            "kalshi websocket first quote market=%s yes=%.4f no=%.4f",
            quote.market_id,
            quote.yes_buy_price,
            quote.no_buy_price,
        )
        return 0
    finally:
        await adapter.aclose()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run()))
