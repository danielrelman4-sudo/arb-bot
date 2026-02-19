"""Perpetual funding rate tracker (OKX).

Polls OKX public API for funding rates and computes directional
signals based on extreme positioning. Extreme positive funding indicates
crowded longs vulnerable to liquidation cascades; extreme negative
indicates crowded shorts vulnerable to short squeezes.
"""

from __future__ import annotations

import asyncio
import time
import logging
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

LOGGER = logging.getLogger(__name__)

# Funding rate is expressed per 8-hour period
_FUNDING_PERIODS_PER_DAY = 3  # Every 8 hours

# Map Binance-style symbols to OKX perpetual swap instrument IDs
_BINANCE_TO_OKX_SWAP = {
    "BTCUSDT": "BTC-USDT-SWAP",
    "ETHUSDT": "ETH-USDT-SWAP",
    "SOLUSDT": "SOL-USDT-SWAP",
}


@dataclass(frozen=True)
class FundingSignal:
    """Directional signal derived from funding rate analysis."""
    rate: float                 # Current funding rate
    rolling_avg_8h: float       # 8-hour rolling average
    rate_of_change: float       # Rate of change over lookback
    is_extreme_long: bool       # Crowded longs detected
    is_extreme_short: bool      # Crowded shorts detected
    drift_adjustment: float     # Suggested drift modifier


class FundingRateTracker:
    """Tracks perpetual funding rates via OKX and generates signals.

    Parameters
    ----------
    symbols:
        Symbols to track (e.g. ["BTCUSDT", "ETHUSDT"]).
    api_url:
        OKX public funding-rate endpoint.
    poll_interval_seconds:
        How often to poll (default 300 = 5 min).
    extreme_threshold:
        Funding rate above this is considered extreme (default 0.0005 = 0.05%).
    max_history:
        Maximum number of historical rate samples to keep per symbol.
    drift_scale:
        Scaling factor for drift adjustment from funding signal.
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        api_url: str = "https://www.okx.com/api/v5/public/funding-rate",
        poll_interval_seconds: int = 300,
        extreme_threshold: float = 0.0005,
        max_history: int = 500,
        drift_scale: float = 2.0,
    ) -> None:
        self._symbols = [s.upper() for s in (symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT"])]
        self._api_url = api_url
        self._poll_interval = poll_interval_seconds
        self._extreme_threshold = extreme_threshold
        self._max_history = max_history
        self._drift_scale = drift_scale

        # {symbol: deque of (timestamp, rate)}
        self._history: Dict[str, Deque[Tuple[float, float]]] = {
            s: deque(maxlen=max_history) for s in self._symbols
        }
        self._current_rate: Dict[str, float] = {}
        self._task: Optional[asyncio.Task] = None
        self._client = None

    async def start(self, client) -> None:
        """Start periodic polling. Pass an httpx.AsyncClient."""
        self._client = client
        self._task = asyncio.create_task(self._poll_loop())
        LOGGER.info("FundingRateTracker: started polling %d symbols every %ds",
                     len(self._symbols), self._poll_interval)

    async def stop(self) -> None:
        """Stop polling."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        LOGGER.info("FundingRateTracker: stopped")

    async def _poll_loop(self) -> None:
        """Periodically fetch funding rates."""
        while True:
            try:
                await self._poll_current()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                LOGGER.warning("FundingRateTracker: poll error: %s", exc)
            await asyncio.sleep(self._poll_interval)

    async def _poll_current(self) -> None:
        """Fetch current funding rate from OKX for all symbols."""
        if self._client is None:
            return

        now = time.time()
        for symbol in self._symbols:
            okx_symbol = _BINANCE_TO_OKX_SWAP.get(symbol)
            if okx_symbol is None:
                LOGGER.debug("FundingRateTracker: no OKX mapping for %s", symbol)
                continue
            try:
                resp = await self._client.get(
                    self._api_url,
                    params={"instId": okx_symbol},
                )
                if resp.status_code != 200:
                    LOGGER.debug("FundingRateTracker: %s HTTP %d", symbol, resp.status_code)
                    continue
                body = resp.json()
                rate = float(body["data"][0]["fundingRate"])
                self._current_rate[symbol] = rate
                self._history[symbol].append((now, rate))
                LOGGER.debug("FundingRateTracker: %s rate=%.6f", symbol, rate)
            except Exception as exc:
                LOGGER.debug("FundingRateTracker: %s fetch error: %s", symbol, exc)

    def inject_rate(self, symbol: str, rate: float, timestamp: Optional[float] = None) -> None:
        """Manually inject a funding rate (for testing)."""
        symbol = symbol.upper()
        ts = timestamp if timestamp is not None else time.time()
        if symbol not in self._history:
            self._history[symbol] = deque(maxlen=self._max_history)
        self._history[symbol].append((ts, rate))
        self._current_rate[symbol] = rate

    def get_current_rate(self, symbol: str) -> Optional[float]:
        """Get the latest funding rate for a symbol."""
        return self._current_rate.get(symbol.upper())

    def get_rolling_avg(self, symbol: str, window_hours: float = 8.0) -> Optional[float]:
        """Get rolling average funding rate over the specified window."""
        symbol = symbol.upper()
        history = self._history.get(symbol)
        if not history:
            return None

        now = time.time()
        cutoff = now - window_hours * 3600
        rates = [r for ts, r in history if ts >= cutoff]
        if not rates:
            return None
        return sum(rates) / len(rates)

    def get_rate_of_change(self, symbol: str, window_hours: float = 8.0) -> Optional[float]:
        """Get rate of change: current rate minus oldest rate in window."""
        symbol = symbol.upper()
        history = self._history.get(symbol)
        if not history or len(history) < 2:
            return None

        now = time.time()
        cutoff = now - window_hours * 3600
        window_rates = [(ts, r) for ts, r in history if ts >= cutoff]
        if len(window_rates) < 2:
            return None

        oldest_rate = window_rates[0][1]
        newest_rate = window_rates[-1][1]
        return newest_rate - oldest_rate

    def get_funding_signal(self, symbol: str) -> Optional[FundingSignal]:
        """Compute comprehensive funding signal for trading decisions.

        Extreme positive funding -> crowded longs -> negative drift (expect pullback).
        Extreme negative funding -> crowded shorts -> positive drift (expect squeeze).
        """
        symbol = symbol.upper()
        rate = self.get_current_rate(symbol)
        if rate is None:
            return None

        rolling = self.get_rolling_avg(symbol, window_hours=8.0)
        if rolling is None:
            rolling = rate

        roc = self.get_rate_of_change(symbol, window_hours=8.0)
        if roc is None:
            roc = 0.0

        is_extreme_long = rate > self._extreme_threshold
        is_extreme_short = rate < -self._extreme_threshold

        # Drift adjustment: funding acts as a mean-reversion signal
        # High positive funding -> expect downward reversion
        # High negative funding -> expect upward reversion
        if is_extreme_long:
            # Crowded longs: negative drift (expect liquidation cascade)
            drift_adjustment = -abs(rate) * self._drift_scale * _FUNDING_PERIODS_PER_DAY
        elif is_extreme_short:
            # Crowded shorts: positive drift (expect short squeeze)
            drift_adjustment = abs(rate) * self._drift_scale * _FUNDING_PERIODS_PER_DAY
        else:
            # Normal range: mild mean-reversion
            drift_adjustment = -rate * self._drift_scale * _FUNDING_PERIODS_PER_DAY * 0.2

        return FundingSignal(
            rate=rate,
            rolling_avg_8h=rolling,
            rate_of_change=roc,
            is_extreme_long=is_extreme_long,
            is_extreme_short=is_extreme_short,
            drift_adjustment=drift_adjustment,
        )
