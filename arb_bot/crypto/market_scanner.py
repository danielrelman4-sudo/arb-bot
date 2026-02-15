"""Kalshi crypto market discovery and ticker parsing.

Discovers KXBTC*, KXETH*, KXSOL* markets from Kalshi, parses their
ticker structure, and enriches with orderbook quotes.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger(__name__)

# ── Ticker format patterns ─────────────────────────────────────────
#
# Daily above/below:   KXBTCD-26FEB14-T97500   (T=above/top, B=below/bottom)
# 15-min up/down:      KXBTC15M-26FEB14-U12    (U=up, D=down, number=interval)
# 1-hour up/down:      KXBTC1H-26FEB14-U12     (U=up, D=down, number=hour)
# Daily above/below:   KXETHD-26FEB14-T3200
# 15-min up/down:      KXETH15M-26FEB14-U12

_SERIES_RE = re.compile(
    r"^(KX(?:BTC|ETH|SOL)(?:15M|1H|D)?)"  # series ticker
    r"-(\d{2}[A-Z]{3}\d{2})"                # date (26FEB14)
    r"-([TBUD])(\d+)$"                       # direction + strike/interval
)

_MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

_DIRECTION_MAP = {
    "T": "above",
    "B": "below",
    "U": "up",
    "D": "down",
}

# Maps series prefix to underlying and interval
_SERIES_PARSE: Dict[str, tuple[str, str]] = {}


def _parse_series(series: str) -> tuple[str, str]:
    """Extract (underlying, interval) from series ticker prefix."""
    s = series.upper()
    # KXBTC15M → BTC, 15min
    # KXBTCD → BTC, daily
    # KXBTC1H → BTC, 1hour
    # KXETH15M → ETH, 15min
    for prefix in ("KXBTC", "KXETH", "KXSOL"):
        if s.startswith(prefix):
            suffix = s[len(prefix):]
            underlying = prefix[2:]  # Strip "KX"
            if suffix == "15M":
                return underlying, "15min"
            elif suffix == "1H":
                return underlying, "1hour"
            elif suffix == "D":
                return underlying, "daily"
            elif suffix == "":
                return underlying, "daily"
            else:
                return underlying, suffix.lower()
    return s, "unknown"


def _parse_date(date_str: str) -> datetime | None:
    """Parse Kalshi date format '26FEB14' → datetime(2026, 2, 14)."""
    if len(date_str) != 7:
        return None
    try:
        year_short = int(date_str[:2])
        month_str = date_str[2:5]
        day = int(date_str[5:7])
        month = _MONTH_MAP.get(month_str)
        if month is None:
            return None
        year = 2000 + year_short
        return datetime(year, month, day, tzinfo=timezone.utc)
    except (ValueError, KeyError):
        return None


@dataclass(frozen=True)
class CryptoMarketMeta:
    """Parsed metadata from a Kalshi crypto ticker."""

    underlying: str      # "BTC", "ETH", "SOL"
    interval: str        # "15min", "1hour", "daily"
    expiry: datetime     # When this contract settles
    strike: float | None  # Strike price (for above/below) or None
    direction: str       # "up", "down", "above", "below"
    series_ticker: str   # e.g. "KXBTCD", "KXBTC15M"
    interval_index: int | None = None  # For up/down: which interval (e.g. 12)
    interval_start_time: datetime | None = None  # For up/down: start of the interval


@dataclass(frozen=True)
class CryptoMarket:
    """A Kalshi crypto market with parsed metadata."""

    ticker: str
    meta: CryptoMarketMeta


@dataclass(frozen=True)
class CryptoMarketQuote:
    """A crypto market enriched with live quote data."""

    market: CryptoMarket
    yes_buy_price: float
    no_buy_price: float
    yes_buy_size: float
    no_buy_size: float
    yes_bid_price: float | None
    no_bid_price: float | None
    time_to_expiry_minutes: float
    implied_probability: float


def parse_ticker(ticker: str) -> CryptoMarketMeta | None:
    """Parse a Kalshi crypto ticker into structured metadata.

    Examples
    --------
    >>> parse_ticker("KXBTCD-26FEB14-T97500")
    CryptoMarketMeta(underlying='BTC', interval='daily', ..., strike=97500.0, direction='above')

    >>> parse_ticker("KXBTC15M-26FEB14-U12")
    CryptoMarketMeta(underlying='BTC', interval='15min', ..., strike=None, direction='up', interval_index=12)
    """
    m = _SERIES_RE.match(ticker.upper())
    if m is None:
        return None

    series = m.group(1)
    date_str = m.group(2)
    dir_char = m.group(3)
    number_str = m.group(4)

    underlying, interval = _parse_series(series)
    expiry = _parse_date(date_str)
    if expiry is None:
        return None

    direction = _DIRECTION_MAP.get(dir_char, dir_char.lower())

    # For above/below: number is the strike price
    # For up/down: number is the interval index
    if direction in ("above", "below"):
        strike = float(number_str)
        interval_index = None
    else:
        strike = None
        interval_index = int(number_str)

    return CryptoMarketMeta(
        underlying=underlying,
        interval=interval,
        expiry=expiry,
        strike=strike,
        direction=direction,
        series_ticker=series,
        interval_index=interval_index,
    )


class MarketScanner:
    """Discovers and filters Kalshi crypto binary markets.

    Parameters
    ----------
    symbols:
        Kalshi series prefixes to scan, e.g. ``["KXBTC", "KXETH"]``.
    min_minutes_to_expiry:
        Skip markets expiring sooner than this.
    max_minutes_to_expiry:
        Skip markets expiring later than this.
    min_book_depth:
        Minimum contracts at best bid/ask to consider liquid.
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        min_minutes_to_expiry: int = 2,
        max_minutes_to_expiry: int = 14,
        min_book_depth: int = 5,
    ) -> None:
        self._symbols = [s.upper() for s in (symbols or ["KXBTC", "KXETH"])]
        self._min_tte = min_minutes_to_expiry
        self._max_tte = max_minutes_to_expiry
        self._min_depth = min_book_depth

    def filter_markets(
        self,
        markets: list[CryptoMarket],
        now: datetime | None = None,
    ) -> list[CryptoMarket]:
        """Filter markets by time-to-expiry and symbol match."""
        now = now or datetime.now(timezone.utc)
        result: list[CryptoMarket] = []
        for mkt in markets:
            # Check symbol match
            if not any(mkt.meta.series_ticker.startswith(s) for s in self._symbols):
                continue
            # Check time to expiry
            tte_minutes = (mkt.meta.expiry - now).total_seconds() / 60.0
            if tte_minutes < self._min_tte or tte_minutes > self._max_tte:
                continue
            result.append(mkt)
        return result

    def parse_markets_from_tickers(self, tickers: list[str]) -> list[CryptoMarket]:
        """Parse a list of ticker strings into CryptoMarket objects."""
        markets: list[CryptoMarket] = []
        for ticker in tickers:
            meta = parse_ticker(ticker)
            if meta is not None:
                markets.append(CryptoMarket(ticker=ticker, meta=meta))
        return markets

    def compute_time_to_expiry(
        self,
        market: CryptoMarket,
        now: datetime | None = None,
    ) -> float:
        """Minutes until market expires."""
        now = now or datetime.now(timezone.utc)
        return (market.meta.expiry - now).total_seconds() / 60.0
