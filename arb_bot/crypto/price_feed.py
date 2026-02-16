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

# Binance symbol <-> OKX spot instrument mapping
_BINANCE_TO_OKX_SPOT: dict[str, str] = {
    "btcusdt": "BTC-USDT",
    "ethusdt": "ETH-USDT",
    "solusdt": "SOL-USDT",
}
_OKX_TO_BINANCE_SPOT: dict[str, str] = {v: k for k, v in _BINANCE_TO_OKX_SPOT.items()}


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
        ws_url: str = "wss://ws.okx.com:8443/ws/v5/public",
        symbols: list[str] | None = None,
        snapshot_url: str = "https://www.okx.com/api/v5/market/candles",
        history_minutes: int = 60,
    ) -> None:
        self._ws_url = ws_url
        self._symbols = [s.lower() for s in (symbols or ["btcusdt", "ethusdt"])]
        self._snapshot_url = snapshot_url
        self._history_minutes = history_minutes

        self._symbols_set: set = {s.lower() for s in self._symbols}
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
        self._vpin_calculators: Dict[str, Any] = {}  # symbol -> VPINCalculator

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

    def get_price_at_time(self, symbol: str, target_timestamp: float) -> float | None:
        """Return the price of the tick closest to *target_timestamp*.

        Searches through stored ticks for *symbol* and returns the price
        of the tick whose timestamp is nearest to the target.  Returns
        ``None`` if no ticks exist for the symbol.
        """
        sym = symbol.lower()
        ticks = self._ticks.get(sym)
        if not ticks:
            return None

        best_tick: PriceTick | None = None
        best_diff = float("inf")
        for tick in ticks:
            diff = abs(tick.timestamp - target_timestamp)
            if diff < best_diff:
                best_diff = diff
                best_tick = tick

        return best_tick.price if best_tick is not None else None

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

    def get_ofi_multiscale(
        self,
        symbol: str,
        windows: list[int] | None = None,
    ) -> dict[int, float]:
        """Order flow imbalance computed at multiple time windows.

        Parameters
        ----------
        symbol:
            Lowercase symbol (e.g. ``"btcusdt"``).
        windows:
            List of window durations in seconds. Defaults to
            ``[30, 60, 120, 300]``.

        Returns
        -------
        dict mapping window_seconds -> OFI value in [-1, 1].
        """
        if windows is None:
            windows = [30, 60, 120, 300]
        return {w: self.get_ofi(symbol, window_seconds=w) for w in windows}

    def get_aggressor_ratio(
        self,
        symbol: str,
        window_seconds: int = 300,
    ) -> float:
        """Fraction of volume that was buy-initiated.

        Returns 0.5 if no data.
        """
        buy_vol, sell_vol = self.get_buy_sell_volume(symbol, window_seconds)
        total = buy_vol + sell_vol
        if total <= 0:
            return 0.5
        return buy_vol / total

    def get_volume_acceleration(
        self,
        symbol: str,
        short_window: int = 30,
        long_window: int = 300,
    ) -> float:
        """Ratio of recent volume rate to long-term volume rate.

        > 1.0 means recent activity is above normal, < 1.0 means below.
        Returns 1.0 if insufficient data.
        """
        sym = symbol.lower()
        ticks = self._ticks.get(sym)
        if not ticks:
            return 1.0

        now = time.time()
        short_cutoff = now - short_window
        long_cutoff = now - long_window

        short_vol = 0.0
        long_vol = 0.0
        for t in ticks:
            if t.timestamp >= long_cutoff:
                long_vol += t.volume
            if t.timestamp >= short_cutoff:
                short_vol += t.volume

        if long_vol <= 0 or long_window <= 0 or short_window <= 0:
            return 1.0

        short_rate = short_vol / short_window
        long_rate = long_vol / long_window

        if long_rate <= 0:
            return 1.0
        return short_rate / long_rate

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

    def get_total_volume(self, symbol: str, window_seconds: int = 300) -> float:
        """Sum of all tick volumes in the last *window_seconds*.

        Returns 0.0 if no data available.
        """
        sym = symbol.lower()
        ticks = self._ticks.get(sym)
        if not ticks:
            return 0.0
        cutoff = time.time() - window_seconds
        return sum(t.volume for t in ticks if t.timestamp >= cutoff)

    def project_volume(
        self,
        symbol: str,
        horizon_minutes: float,
        rate_window_seconds: int = 300,
    ) -> float | None:
        """Project total expected volume from now until *horizon_minutes*.

        Uses a simple linear forecast: ``current_rate × horizon_minutes``.
        Returns ``None`` if no volume rate is available (caller should
        fall back to clock-time).

        Parameters
        ----------
        symbol:
            Binance symbol (e.g., ``'btcusdt'``).
        horizon_minutes:
            Minutes from now to contract expiry.
        rate_window_seconds:
            Window for estimating the current volume rate.
        """
        rate = self.get_volume_flow_rate(symbol, rate_window_seconds)
        if rate <= 0:
            return None
        return rate * horizon_minutes


    async def load_historical(self, symbol: str) -> None:
        """Bootstrap tick history from OKX REST candles.

        Loads 1-minute candles for the last ``history_minutes`` and
        converts each close to a ``PriceTick``.
        """
        sym = symbol.lower()
        okx_inst = _BINANCE_TO_OKX_SPOT.get(sym)
        if not okx_inst:
            LOGGER.warning("PriceFeed: no OKX mapping for %s, skipping historical load", sym)
            return

        LOGGER.info("PriceFeed: loading %d minutes of history for %s from OKX", self._history_minutes, sym)

        all_candles: list[list] = []
        after: str | None = None
        remaining = min(self._history_minutes, 300)

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                while remaining > 0:
                    batch = min(remaining, 100)
                    url = (
                        f"{self._snapshot_url}"
                        f"?instId={okx_inst}"
                        f"&bar=1m"
                        f"&limit={batch}"
                    )
                    if after:
                        url += f"&after={after}"

                    async with session.get(url) as resp:
                        if resp.status != 200:
                            LOGGER.warning("PriceFeed: OKX candles request failed: %s", resp.status)
                            break
                        body = await resp.json()

                    data = body.get("data", [])
                    if not data:
                        break
                    all_candles.extend(data)
                    after = data[-1][0]  # Oldest timestamp for pagination
                    remaining -= len(data)
                    if len(data) < batch:
                        break

        except ImportError:
            LOGGER.warning("PriceFeed: aiohttp not installed, skipping historical load")
            return
        except Exception as exc:
            LOGGER.warning("PriceFeed: historical load failed: %s", exc)
            return

        if not all_candles:
            return

        # OKX returns newest-first; reverse for chronological order
        all_candles.reverse()

        dq = self._ticks.setdefault(sym, deque(maxlen=_MAX_TICK_HISTORY))
        for candle in all_candles:
            # OKX candle format: [ts_ms, open, high, low, close, vol, volCcy, ...]
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

        LOGGER.info("PriceFeed: loaded %d candles for %s from OKX (latest=%.2f)",
                     len(all_candles), sym, self._current_price.get(sym, 0.0))

    async def load_historical_trades(self, symbol: str, limit: int = 1000) -> int:
        """Bootstrap buy/sell volume from OKX trades REST.

        Loads recent trades to populate OFI data for initial
        calibration. Returns the number of trades loaded.

        Parameters
        ----------
        symbol:
            Binance-style symbol (e.g., 'btcusdt').
        limit:
            Number of recent trades to fetch (max 100 per OKX request,
            will paginate if more are needed).
        """
        sym = symbol.lower()
        okx_inst = _BINANCE_TO_OKX_SPOT.get(sym)
        if not okx_inst:
            LOGGER.warning("PriceFeed: no OKX mapping for %s, skipping historical trades", sym)
            return 0

        LOGGER.info("PriceFeed: loading historical trades for %s from OKX (%s)", sym, okx_inst)
        all_trades: list[dict] = []
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                remaining = min(limit, 500)  # Reasonable cap
                after: str | None = None
                while remaining > 0:
                    batch = min(remaining, 100)
                    url = f"https://www.okx.com/api/v5/market/trades?instId={okx_inst}&limit={batch}"
                    if after:
                        url += f"&after={after}"
                    async with session.get(url) as resp:
                        if resp.status != 200:
                            LOGGER.warning("PriceFeed: OKX trades request failed: %s", resp.status)
                            break
                        body = await resp.json()
                    data = body.get("data", [])
                    if not data:
                        break
                    all_trades.extend(data)
                    after = data[-1].get("tradeId", "")
                    remaining -= len(data)
                    if len(data) < batch:
                        break
        except ImportError:
            LOGGER.warning("PriceFeed: aiohttp not installed, skipping historical trades")
            return 0
        except Exception as exc:
            LOGGER.warning("PriceFeed: historical trades load failed: %s", exc)
            return 0

        if not all_trades:
            return 0

        count = 0
        dq = self._ticks.setdefault(sym, deque(maxlen=_MAX_TICK_HISTORY))
        dq_bs = self._buy_sells.setdefault(sym, deque(maxlen=_MAX_TICK_HISTORY))
        for trade in all_trades:
            # OKX trade format: {"instId":"BTC-USDT","tradeId":"...",
            #   "px":"97500","sz":"0.001","side":"buy","ts":"1700000000000"}
            try:
                ts = float(trade["ts"]) / 1000.0
                price = float(trade["px"])
                qty = float(trade["sz"])
                side = trade.get("side", "").lower()
            except (KeyError, ValueError, TypeError):
                continue

            # OKX side: "buy" = taker bought → is_buyer_maker=False
            #           "sell" = taker sold → is_buyer_maker=True
            is_buyer_maker = side == "sell"
            tick = PriceTick(symbol=sym, price=price, timestamp=ts, volume=qty, is_buyer_maker=is_buyer_maker)
            dq.append(tick)
            self._current_price[sym] = price

            is_buy = not is_buyer_maker
            dq_bs.append((ts, qty, is_buy))
            count += 1

        LOGGER.info("PriceFeed: loaded %d historical trades for %s from OKX", count, sym)
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
            # Forward to VPIN calculator if registered
            vpin_calc = self._vpin_calculators.get(sym)
            if vpin_calc is not None:
                vpin_calc.process_trade(
                    tick.price, tick.volume, is_buy, tick.timestamp,
                )

    def register_vpin(self, symbol: str, calculator: Any) -> None:
        """Register a VPINCalculator for the given symbol."""
        self._vpin_calculators[symbol.lower()] = calculator

    def get_vpin_calculator(self, symbol: str) -> Any:
        """Get the VPINCalculator for the given symbol, or None."""
        return self._vpin_calculators.get(symbol.lower())

    # ── aggTrade WebSocket ──────────────────────────────────────────

    def _handle_agg_trade_message(self, msg: dict) -> None:
        """Parse a Binance aggTrade WebSocket message and inject as tick."""
        try:
            symbol = msg.get("s", "").lower()
            if not symbol or symbol not in self._symbols_set:
                return
            price = float(msg["p"])
            qty = float(msg["q"])
            ts = float(msg["T"]) / 1000.0
            is_buyer_maker = msg.get("m")
        except (KeyError, ValueError, TypeError):
            return

        tick = PriceTick(
            symbol=symbol, price=price, timestamp=ts,
            volume=qty, is_buyer_maker=is_buyer_maker,
        )
        self.inject_tick(tick)

    def _handle_okx_trade_message(self, msg: dict) -> None:
        """Parse an OKX trade WebSocket message and inject as tick.

        Expected format::

            {
                "data": [
                    {
                        "instId": "BTC-USDT",
                        "px": "97500.00",
                        "sz": "0.15",
                        "ts": "1700000000000",
                        "side": "buy"   // or "sell"
                    }
                ]
            }
        """
        data_list = msg.get("data")
        if not data_list:
            return
        for entry in data_list:
            try:
                inst_id = entry.get("instId", "")
                binance_sym = _OKX_TO_BINANCE_SPOT.get(inst_id)
                if not binance_sym or binance_sym not in self._symbols_set:
                    continue
                price = float(entry["px"])
                qty = float(entry["sz"])
                ts = float(entry["ts"]) / 1000.0  # ms -> sec
                side = entry.get("side", "").lower()
                # OKX side: "buy" = taker bought -> is_buyer_maker=False
                #           "sell" = taker sold -> is_buyer_maker=True
                is_buyer_maker = side == "sell"
            except (KeyError, ValueError, TypeError):
                continue

            tick = PriceTick(
                symbol=binance_sym,
                price=price,
                timestamp=ts,
                volume=qty,
                is_buyer_maker=is_buyer_maker,
            )
            self.inject_tick(tick)

    async def connect_agg_trades_ws(self) -> None:
        """Connect to OKX public trades WebSocket for all symbols.

        Uses OKX instead of Binance US because Binance US has near-zero
        trade volume, causing VPIN to go stale.  Reconnects automatically
        on disconnection with timeout and ping keepalive.
        """
        try:
            import websockets  # type: ignore[import-untyped]
        except ImportError:
            LOGGER.warning("PriceFeed: websockets not installed, aggTrades WS disabled")
            return

        # Map Binance symbols to OKX instrument IDs
        okx_instruments = []
        for sym in sorted(self._symbols_set):
            okx_id = _BINANCE_TO_OKX_SPOT.get(sym)
            if okx_id:
                okx_instruments.append(okx_id)
        if not okx_instruments:
            LOGGER.warning("PriceFeed: no OKX instrument mappings for %s", self._symbols_set)
            return

        url = "wss://ws.okx.com:8443/ws/v5/public"
        LOGGER.info("PriceFeed: connecting OKX trades WS for %s", okx_instruments)

        # Ensure _running is True so the reconnect loop works.
        # connect_agg_trades_ws() can be called as a standalone task without start().
        self._running = True

        while self._running:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                    # Subscribe to trades for all instruments
                    sub_msg = {
                        "op": "subscribe",
                        "args": [{"channel": "trades", "instId": inst} for inst in okx_instruments],
                    }
                    await ws.send(json.dumps(sub_msg))
                    LOGGER.info("PriceFeed: OKX trades WS connected, subscribed to %s", okx_instruments)

                    while self._running:
                        try:
                            raw_msg = await asyncio.wait_for(ws.recv(), timeout=30.0)
                        except asyncio.TimeoutError:
                            # No message in 30s — send ping to verify connection
                            await ws.ping()
                            LOGGER.debug("PriceFeed: OKX trades WS ping (no message in 30s)")
                            continue
                        data = json.loads(raw_msg)
                        # OKX sends {"event":"subscribe",...} for confirmations
                        if "data" in data:
                            self._handle_okx_trade_message(data)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                LOGGER.warning("PriceFeed: OKX trades WS error: %s, reconnecting in 5s", exc)
                await asyncio.sleep(5)

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
        """Connect to OKX public trades WebSocket and process messages."""
        try:
            import websockets  # type: ignore[import-untyped]
        except ImportError:
            LOGGER.warning("PriceFeed: websockets not installed, WS disabled")
            return

        # Map Binance symbols to OKX instruments
        okx_instruments = []
        for sym in self._symbols:
            okx_id = _BINANCE_TO_OKX_SPOT.get(sym)
            if okx_id:
                okx_instruments.append(okx_id)
        if not okx_instruments:
            LOGGER.warning("PriceFeed: no OKX mappings for %s", self._symbols)
            return

        url = self._ws_url
        LOGGER.info("PriceFeed: connecting to OKX WS %s for %s", url, okx_instruments)

        async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
            self._ws = ws
            # Subscribe to trades
            sub_msg = {
                "op": "subscribe",
                "args": [{"channel": "trades", "instId": inst} for inst in okx_instruments],
            }
            await ws.send(json.dumps(sub_msg))
            LOGGER.info("PriceFeed: OKX WS connected, subscribed to %s", okx_instruments)

            while self._running:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
                except asyncio.TimeoutError:
                    # Send ping to keep alive
                    await ws.ping()
                    continue
                data = json.loads(raw)
                if "data" in data:
                    self._handle_okx_trade_message(data)

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
