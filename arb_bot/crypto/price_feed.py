"""Real-time crypto price feed via Binance WebSocket.

Streams trade data for volatility estimation and model input.
Falls back to REST klines for historical bootstrapping when no
WebSocket data is available yet.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

LOGGER = logging.getLogger(__name__)

# Maximum ticks to keep in memory per symbol (60 min at ~1 tick/sec = 3600).
_MAX_TICK_HISTORY = 7200


@dataclass(frozen=True)
class PriceTick:
    """A single observed trade price."""

    symbol: str
    price: float
    timestamp: float  # unix seconds
    volume: float = 0.0
    is_buyer_maker: Optional[bool] = None  # True=sell pressure, False=buy pressure, None=unknown


class PriceFeed:
    """Real-time crypto price feed via Binance WebSocket.

    Parameters
    ----------
    ws_url:
        Binance WebSocket endpoint (combined stream).
    symbols:
        Lowercase Binance symbol names, e.g. ``["btcusdt", "ethusdt"]``.
    snapshot_url:
        Binance REST klines endpoint for historical bootstrap.
    history_minutes:
        How many minutes of 1m-kline history to load at startup.
    """

    def __init__(
        self,
        ws_url: str = "wss://stream.binance.com:9443/ws",
        symbols: list[str] | None = None,
        snapshot_url: str = "https://api.binance.com/api/v3/klines",
        history_minutes: int = 60,
    ) -> None:
        self._ws_url = ws_url
        self._symbols = [s.lower() for s in (symbols or ["btcusdt", "ethusdt"])]
        self._snapshot_url = snapshot_url
        self._history_minutes = history_minutes

        self._ticks: Dict[str, Deque[PriceTick]] = {
            s: deque(maxlen=_MAX_TICK_HISTORY) for s in self._symbols
        }
        self._buy_sells: Dict[str, Deque[tuple]] = {
            s: deque(maxlen=_MAX_TICK_HISTORY) for s in self._symbols
        }  # tuples of (timestamp, volume, is_buy)
        self._current_price: Dict[str, float] = {}
        self._ws: Any = None  # websockets connection
        self._running = False
        self._ws_task: Optional[asyncio.Task[None]] = None

    # ── public API ─────────────────────────────────────────────────

    async def start(self) -> None:
        """Connect to Binance WS and begin receiving trades."""
        if self._running:
            return
        self._running = True
        self._ws_task = asyncio.create_task(self._ws_loop())
        LOGGER.info("PriceFeed: started for %s", self._symbols)

    async def stop(self) -> None:
        """Disconnect WebSocket and stop background task."""
        self._running = False
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        if self._ws_task is not None:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except (asyncio.CancelledError, Exception):
                pass
            self._ws_task = None
        LOGGER.info("PriceFeed: stopped")

    def get_current_price(self, symbol: str) -> float | None:
        """Latest price for *symbol* (e.g. ``'btcusdt'``)."""
        return self._current_price.get(symbol.lower())

    def get_price_history(self, symbol: str, minutes: int = 0) -> list[PriceTick]:
        """Recent ticks for *symbol*, optionally limited to last *minutes*."""
        sym = symbol.lower()
        ticks = self._ticks.get(sym)
        if ticks is None:
            return []
        if minutes <= 0:
            return list(ticks)
        cutoff = time.time() - minutes * 60
        return [t for t in ticks if t.timestamp >= cutoff]

    def get_returns(
        self,
        symbol: str,
        interval_seconds: int = 60,
        window_minutes: int = 0,
    ) -> list[float]:
        """Compute log returns at *interval_seconds* spacing.

        If *window_minutes* > 0, only use ticks from the last N minutes.
        Returns a list of ``ln(P[t]/P[t-1])`` values.
        """
        ticks = self.get_price_history(symbol, minutes=window_minutes)
        if len(ticks) < 2:
            return []

        # Bucket ticks into intervals and pick the last price per bucket.
        if not ticks:
            return []
        t0 = ticks[0].timestamp
        buckets: Dict[int, float] = {}
        for tick in ticks:
            bucket_idx = int((tick.timestamp - t0) / interval_seconds)
            buckets[bucket_idx] = tick.price

        sorted_keys = sorted(buckets.keys())
        if len(sorted_keys) < 2:
            return []

        returns: list[float] = []
        for i in range(1, len(sorted_keys)):
            prev_price = buckets[sorted_keys[i - 1]]
            curr_price = buckets[sorted_keys[i]]
            if prev_price > 0 and curr_price > 0:
                returns.append(math.log(curr_price / prev_price))
        return returns

    def get_buy_sell_volume(self, symbol: str, window_seconds: int = 300) -> tuple:
        """Return (buy_volume, sell_volume) over the last window_seconds."""
        sym = symbol.lower()
        dq = self._buy_sells.get(sym)
        if not dq:
            return (0.0, 0.0)
        cutoff = time.time() - window_seconds
        buy_vol = 0.0
        sell_vol = 0.0
        for ts, vol, is_buy in dq:
            if ts >= cutoff:
                if is_buy:
                    buy_vol += vol
                else:
                    sell_vol += vol
        return (buy_vol, sell_vol)

    def get_ofi(self, symbol: str, window_seconds: int = 300) -> float:
        """Order Flow Imbalance: (buy_vol - sell_vol) / (buy_vol + sell_vol).

        Returns a value in [-1, +1]. Positive = net buy pressure.
        Returns 0.0 if no data available.
        """
        buy_vol, sell_vol = self.get_buy_sell_volume(symbol, window_seconds)
        total = buy_vol + sell_vol
        if total <= 0:
            return 0.0
        return (buy_vol - sell_vol) / total

    def get_volume_flow_rate(self, symbol: str, window_seconds: int = 300) -> float:
        """Average volume per minute over the last window_seconds.

        Uses the actual data span (first tick to now) instead of the full
        window duration to avoid under-reporting when data covers less
        than the requested window.

        Returns 0.0 if insufficient data.
        """
        sym = symbol.lower()
        ticks = self._ticks.get(sym)
        if not ticks:
            return 0.0
        now = time.time()
        cutoff = now - window_seconds
        window_ticks = [t for t in ticks if t.timestamp >= cutoff]
        if not window_ticks:
            return 0.0
        total_vol = sum(t.volume for t in window_ticks)
        # Use actual data span: first qualifying tick to now
        actual_span_seconds = now - window_ticks[0].timestamp
        if actual_span_seconds < 1.0:
            # Less than 1 second of data — not meaningful
            return 0.0
        minutes = actual_span_seconds / 60.0
        return total_vol / minutes

    async def load_historical(self, symbol: str) -> None:
        """Bootstrap tick history from Binance REST klines.

        Loads 1-minute candles for the last ``history_minutes`` and
        converts each close to a ``PriceTick``.
        """
        sym = symbol.lower()
        url = (
            f"{self._snapshot_url}"
            f"?symbol={sym.upper()}"
            f"&interval=1m"
            f"&limit={self._history_minutes}"
        )
        LOGGER.info("PriceFeed: loading %d minutes of history for %s", self._history_minutes, sym)
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        LOGGER.warning("PriceFeed: klines request failed: %s", resp.status)
                        return
                    data = await resp.json()
        except ImportError:
            LOGGER.warning("PriceFeed: aiohttp not installed, skipping historical load")
            return
        except Exception as exc:
            LOGGER.warning("PriceFeed: historical load failed: %s", exc)
            return

        if not isinstance(data, list):
            return

        dq = self._ticks.setdefault(sym, deque(maxlen=_MAX_TICK_HISTORY))
        for candle in data:
            # Binance kline format: [open_time, open, high, low, close, volume, ...]
            if not isinstance(candle, list) or len(candle) < 6:
                continue
            try:
                ts = float(candle[0]) / 1000.0  # ms → sec
                close = float(candle[4])
                vol = float(candle[5])
            except (ValueError, TypeError):
                continue
            dq.append(PriceTick(symbol=sym, price=close, timestamp=ts, volume=vol))
            self._current_price[sym] = close

        LOGGER.info("PriceFeed: loaded %d klines for %s (latest=%.2f)",
                     len(data), sym, self._current_price.get(sym, 0.0))

    async def load_historical_trades(self, symbol: str, limit: int = 1000) -> int:
        """Bootstrap buy/sell volume from Binance aggTrades REST.

        Loads recent aggregate trades to populate OFI data for initial
        calibration. Returns the number of trades loaded.

        Parameters
        ----------
        symbol:
            Binance symbol (e.g., 'btcusdt').
        limit:
            Number of recent trades to fetch (max 1000 per Binance API).
        """
        sym = symbol.lower()
        url = (
            f"https://api.binance.com/api/v3/aggTrades"
            f"?symbol={sym.upper()}"
            f"&limit={min(limit, 1000)}"
        )
        LOGGER.info("PriceFeed: loading historical trades for %s", sym)
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        LOGGER.warning("PriceFeed: aggTrades request failed: %s", resp.status)
                        return 0
                    data = await resp.json()
        except ImportError:
            LOGGER.warning("PriceFeed: aiohttp not installed, skipping historical trades")
            return 0
        except Exception as exc:
            LOGGER.warning("PriceFeed: historical trades load failed: %s", exc)
            return 0

        if not isinstance(data, list):
            return 0

        count = 0
        dq = self._ticks.setdefault(sym, deque(maxlen=_MAX_TICK_HISTORY))
        dq_bs = self._buy_sells.setdefault(sym, deque(maxlen=_MAX_TICK_HISTORY))
        for trade in data:
            # Binance aggTrades format:
            # {"a":id, "p":"price", "q":"qty", "f":first_id, "l":last_id, "T":timestamp, "m":is_buyer_maker}
            try:
                ts = float(trade["T"]) / 1000.0
                price = float(trade["p"])
                qty = float(trade["q"])
                is_buyer_maker = trade.get("m")
            except (KeyError, ValueError, TypeError):
                continue

            tick = PriceTick(symbol=sym, price=price, timestamp=ts, volume=qty, is_buyer_maker=is_buyer_maker)
            dq.append(tick)
            self._current_price[sym] = price

            if isinstance(is_buyer_maker, bool):
                is_buy = not is_buyer_maker
                dq_bs.append((ts, qty, is_buy))

            count += 1

        LOGGER.info("PriceFeed: loaded %d historical trades for %s", count, sym)
        return count

    def inject_tick(self, tick: PriceTick) -> None:
        """Inject a tick manually (useful for testing or alternative feeds)."""
        sym = tick.symbol.lower()
        dq = self._ticks.setdefault(sym, deque(maxlen=_MAX_TICK_HISTORY))
        dq.append(tick)
        self._current_price[sym] = tick.price
        if tick.is_buyer_maker is not None:
            is_buy = not tick.is_buyer_maker
            dq_bs = self._buy_sells.setdefault(sym, deque(maxlen=_MAX_TICK_HISTORY))
            dq_bs.append((tick.timestamp, tick.volume, is_buy))

    # ── internal ───────────────────────────────────────────────────

    async def _ws_loop(self) -> None:
        """Reconnecting WebSocket loop."""
        while self._running:
            try:
                await self._connect_and_stream()
            except asyncio.CancelledError:
                return
            except Exception as exc:
                LOGGER.warning("PriceFeed: WS error: %s, reconnecting in 5s", exc)
                await asyncio.sleep(5)

    async def _connect_and_stream(self) -> None:
        """Connect to Binance combined streams and process messages."""
        try:
            import websockets  # type: ignore[import-untyped]
        except ImportError:
            LOGGER.warning("PriceFeed: websockets not installed, WS disabled")
            return

        streams = "/".join(f"{s}@trade" for s in self._symbols)
        url = f"{self._ws_url}/{streams}"
        LOGGER.info("PriceFeed: connecting to %s", url)

        async with websockets.connect(url) as ws:
            self._ws = ws
            LOGGER.info("PriceFeed: connected")
            while self._running:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
                except asyncio.TimeoutError:
                    # Send ping to keep alive
                    await ws.ping()
                    continue
                self._handle_message(raw)

    def _handle_message(self, raw: str | bytes) -> None:
        """Parse a Binance trade message and record the tick."""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        # Binance trade format: {"e":"trade","s":"BTCUSDT","p":"97500.00","q":"0.001","T":...}
        event_type = msg.get("e")
        if event_type != "trade":
            return

        sym = (msg.get("s") or "").lower()
        try:
            price = float(msg["p"])
            qty = float(msg["q"])
            ts = float(msg["T"]) / 1000.0  # ms → sec
        except (KeyError, ValueError, TypeError):
            return

        is_buyer_maker = msg.get("m")  # bool or None
        tick = PriceTick(symbol=sym, price=price, timestamp=ts, volume=qty, is_buyer_maker=is_buyer_maker)
        dq = self._ticks.setdefault(sym, deque(maxlen=_MAX_TICK_HISTORY))
        dq.append(tick)
        self._current_price[sym] = price
        # Track buy/sell volume
        if isinstance(is_buyer_maker, bool):
            is_buy = not is_buyer_maker  # buyer is taker = buy pressure
            dq_bs = self._buy_sells.setdefault(sym, deque(maxlen=_MAX_TICK_HISTORY))
            dq_bs.append((ts, qty, is_buy))
