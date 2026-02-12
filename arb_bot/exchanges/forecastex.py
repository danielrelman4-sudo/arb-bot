"""ForecastEx / Interactive Brokers exchange adapter (Phase 9A).

Routes ForecastEx binary event contracts via the IBKR TWS API.  ForecastEx
contracts are modelled as options (secType='OPT', exchange='FORECASTX') where
YES maps to a Call (right='C') and NO maps to a Put (right='P').  Each
contract pays $1.00 USD at expiry.

Key constraint: ForecastEx instruments **cannot be sold** — to exit a position
the trader buys the opposing contract, and IBKR automatically nets the pair.

This adapter uses the ``ib_async`` library (successor to ``ib_insync``) for
async connectivity to TWS or IB Gateway.  If the library is not installed, the
adapter degrades gracefully and raises on any live call.

Fee structure: $0.01 per YES/NO pair ($0.005 per individual contract).
Order types: DAY, GTC, IOC only.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from arb_bot.models import (
    BinaryQuote,
    LegExecutionResult,
    OrderState,
    OrderStatus,
    PairExecutionResult,
    Side,
    TradeLegPlan,
    TradePlan,
)

from .base import ExchangeAdapter

# ---------------------------------------------------------------------------
# Optional ib_async import
# ---------------------------------------------------------------------------
try:
    import ib_async as ibasync  # type: ignore[import-untyped]

    _HAS_IB_ASYNC = True
except ImportError:
    ibasync = None  # type: ignore[assignment]
    _HAS_IB_ASYNC = False


LOGGER = logging.getLogger(__name__)

# Default fee per individual contract ($0.005 = half of $0.01/pair)
_DEFAULT_FEE_PER_CONTRACT = 0.005

# ForecastEx payout: $1.00 per contract at expiry
_FORECASTEX_PAYOUT = 1.0

# CME event contracts payout: $100.00 per contract
_CME_EVENT_PAYOUT = 100.0


# ---------------------------------------------------------------------------
# IBKR Rate Limiter — prevents 10-minute penalty box
# ---------------------------------------------------------------------------
#
# IBKR enforces:
#   - Web API: 10 req/s global limit, 10-min penalty box on violation
#   - TWS API: Pacing rules for reqContractDetails (~50/s) and
#     reqMktData (~50 concurrent subscriptions)
#
# We use a conservative token-bucket approach targeting well below the
# limit.  Default: 6 req/s (60% of 10) with bursts up to 8.
#


class IBKRRateLimiter:
    """Token-bucket rate limiter for IBKR API calls.

    Conservative by default to stay well below penalty thresholds.
    Thread-safe via asyncio.Lock.

    Parameters
    ----------
    max_per_second:
        Sustained rate limit (tokens refilled per second).  Default 6.0
        (60% of IBKR's hard 10 req/s limit).
    burst_size:
        Maximum burst size (token bucket capacity).  Default 8.
    """

    def __init__(
        self,
        max_per_second: float = 6.0,
        burst_size: int = 8,
    ) -> None:
        self._max_per_second = max(0.1, max_per_second)
        self._burst_size = max(1, burst_size)
        self._tokens = float(self._burst_size)
        self._last_refill_ts = time.monotonic()
        self._lock = asyncio.Lock()
        self._total_requests = 0
        self._total_waits = 0

    async def acquire(self) -> float:
        """Acquire one token, waiting if necessary.

        Returns the wait time in seconds (0.0 if no wait needed).
        """
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill_ts
            self._last_refill_ts = now

            # Refill tokens
            self._tokens = min(
                float(self._burst_size),
                self._tokens + elapsed * self._max_per_second,
            )

            if self._tokens >= 1.0:
                self._tokens -= 1.0
                self._total_requests += 1
                return 0.0

            # Calculate wait time for next token
            deficit = 1.0 - self._tokens
            wait_seconds = deficit / self._max_per_second
            self._total_waits += 1

        # Wait outside the lock so other coroutines can proceed
        await asyncio.sleep(wait_seconds)

        async with self._lock:
            self._tokens = max(0.0, self._tokens - 1.0 + wait_seconds * self._max_per_second)
            self._total_requests += 1
            return wait_seconds

    @property
    def total_requests(self) -> int:
        """Total number of requests that passed through the limiter."""
        return self._total_requests

    @property
    def total_waits(self) -> int:
        """Number of times a request had to wait for a token."""
        return self._total_waits


# ---------------------------------------------------------------------------
# Settings dataclass — kept here to stay near the adapter (also registered
# in config.py).
# ---------------------------------------------------------------------------

# NOTE: ForecastExSettings is imported from config.py.  See config.py for the
# canonical dataclass definition and env-var loading.  We import it here to
# keep the adapter self-contained for type hints.
from arb_bot.config import ForecastExSettings  # noqa: E402


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class ForecastExAdapter(ExchangeAdapter):
    """ExchangeAdapter implementation for ForecastEx via IBKR TWS API."""

    venue = "forecastex"

    def __init__(
        self,
        settings: ForecastExSettings,
        timeout_seconds: float = 10.0,
    ) -> None:
        self._settings = settings
        self._timeout_seconds = timeout_seconds

        # IB connection object — lazily created via connect()
        self._ib: Any = None
        self._connected = False

        # IBKR rate limiter — prevents 10-minute penalty box.
        # Conservative default: 6 req/s sustained, 8 burst (IBKR limit = 10 req/s).
        self._rate_limiter = IBKRRateLimiter(
            max_per_second=6.0,
            burst_size=8,
        )

        # Contract cache: symbol -> contract details
        self._contract_cache: Dict[str, Any] = {}
        self._contract_cache_ts: float = 0.0
        self._contract_cache_ttl_seconds: float = 300.0

        # Quote cache for deduplication during streaming
        self._last_quotes: Dict[str, BinaryQuote] = {}

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Connect to TWS / IB Gateway.  Returns True on success."""
        if not _HAS_IB_ASYNC:
            LOGGER.error("ib_async is not installed — cannot connect to IBKR")
            return False

        if self._connected and self._ib is not None:
            try:
                if self._ib.isConnected():
                    return True
            except Exception:
                pass

        try:
            ib = ibasync.IB()
            await ib.connectAsync(
                host=self._settings.host,
                port=self._settings.port,
                clientId=self._settings.client_id,
                timeout=self._timeout_seconds,
            )
            self._ib = ib
            self._connected = True
            LOGGER.info(
                "Connected to IBKR TWS at %s:%d (client_id=%d)",
                self._settings.host,
                self._settings.port,
                self._settings.client_id,
            )
            return True
        except Exception as exc:
            LOGGER.error("Failed to connect to IBKR TWS: %s", exc)
            self._connected = False
            return False

    async def aclose(self) -> None:
        """Disconnect from TWS / IB Gateway."""
        if self._ib is not None:
            try:
                self._ib.disconnect()
            except Exception:
                pass
            self._ib = None
            self._connected = False
            LOGGER.info("Disconnected from IBKR TWS")

    # ------------------------------------------------------------------
    # Contract helpers
    # ------------------------------------------------------------------

    def _make_contract(
        self,
        symbol: str,
        right: str,
        last_trade_date: str = "",
        strike: float = 0.0,
    ) -> Any:
        """Create an IBKR Contract object for a ForecastEx event.

        Parameters
        ----------
        symbol:
            ForecastEx event symbol (e.g. 'FED25BP50').
        right:
            'C' for YES, 'P' for NO.
        last_trade_date:
            Expiry date in YYYYMMDD format.
        strike:
            Strike price for the contract.

        Returns
        -------
        An ``ib_async.Contract`` object, or a plain dict if ib_async is
        not available (for testing).
        """
        if not _HAS_IB_ASYNC:
            return {
                "secType": "OPT",
                "symbol": symbol,
                "exchange": "FORECASTX",
                "currency": "USD",
                "right": right,
                "lastTradeDateOrContractMonth": last_trade_date,
                "strike": strike,
            }

        return ibasync.Contract(
            secType="OPT",
            symbol=symbol,
            exchange="FORECASTX",
            currency="USD",
            right=right,
            lastTradeDateOrContractMonth=last_trade_date,
            strike=strike,
        )

    @staticmethod
    def side_to_right(side: Side) -> str:
        """Map arb_bot Side to ForecastEx option right."""
        return "C" if side == Side.YES else "P"

    @staticmethod
    def right_to_side(right: str) -> Side:
        """Map ForecastEx option right to arb_bot Side."""
        return Side.YES if right.upper() == "C" else Side.NO

    # ------------------------------------------------------------------
    # Quote discovery and fetching
    # ------------------------------------------------------------------

    async def _discover_contracts(self) -> List[Any]:
        """Discover available ForecastEx event contracts.

        Uses reqContractDetails with a partial Contract specification to
        enumerate available events.  Results are cached for
        ``_contract_cache_ttl_seconds``.
        """
        if not self._ib or not self._connected:
            LOGGER.warning("ForecastEx: not connected, cannot discover contracts")
            return []

        now = time.monotonic()
        if (
            self._contract_cache
            and (now - self._contract_cache_ts) < self._contract_cache_ttl_seconds
        ):
            return list(self._contract_cache.values())

        try:
            # Rate-limit the discovery request
            await self._rate_limiter.acquire()

            # Request all available ForecastEx option contracts
            partial = ibasync.Contract(
                secType="OPT",
                exchange="FORECASTX",
                currency="USD",
            )
            details_list = await self._ib.reqContractDetailsAsync(partial)
            contracts = []
            for detail in (details_list or []):
                contract = detail.contract
                key = f"{contract.symbol}_{contract.right}_{contract.lastTradeDateOrContractMonth}"
                self._contract_cache[key] = detail
                contracts.append(detail)

            self._contract_cache_ts = now
            LOGGER.info(
                "ForecastEx: discovered %d contract details", len(contracts)
            )
            return contracts
        except Exception as exc:
            LOGGER.error("ForecastEx: contract discovery failed: %s", exc)
            return list(self._contract_cache.values())

    def _normalize_quote(
        self,
        symbol: str,
        yes_bid: float,
        yes_ask: float,
        no_bid: float,
        no_ask: float,
        yes_bid_size: float = 0.0,
        yes_ask_size: float = 0.0,
        no_bid_size: float = 0.0,
        no_ask_size: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BinaryQuote:
        """Normalize raw ForecastEx YES/NO quotes to a BinaryQuote.

        ForecastEx prices are in USD [0, 1.00].  The 'buy' price is the
        ask (price to buy).  Bid is the price at which you could sell —
        but since ForecastEx doesn't allow selling, bid is informational
        only (used for maker estimates and spread analysis).
        """
        return BinaryQuote(
            venue=self.venue,
            market_id=symbol,
            yes_buy_price=yes_ask,
            no_buy_price=no_ask,
            yes_buy_size=yes_ask_size,
            no_buy_size=no_ask_size,
            yes_bid_price=yes_bid if yes_bid > 0 else None,
            no_bid_price=no_bid if no_bid > 0 else None,
            yes_bid_size=yes_bid_size,
            no_bid_size=no_bid_size,
            fee_per_contract=_DEFAULT_FEE_PER_CONTRACT,
            observed_at=datetime.now(timezone.utc),
            metadata=metadata or {},
        )

    async def fetch_quotes(self) -> list[BinaryQuote]:
        """Fetch current quotes for all known ForecastEx contracts.

        Groups Call/Put contracts by symbol and returns one BinaryQuote
        per symbol with YES (Call) and NO (Put) sides populated.
        """
        if not self._ib or not self._connected:
            LOGGER.warning("ForecastEx: not connected, returning empty quotes")
            return []

        contracts_details = await self._discover_contracts()
        if not contracts_details:
            return []

        # Group by symbol
        by_symbol: Dict[str, Dict[str, Any]] = {}
        for detail in contracts_details:
            contract = detail.contract
            symbol = contract.symbol
            if symbol not in by_symbol:
                by_symbol[symbol] = {"C": None, "P": None, "detail": detail}
            by_symbol[symbol][contract.right] = contract

        # Apply market limit
        symbols = list(by_symbol.keys())
        if self._settings.market_limit > 0:
            symbols = symbols[: self._settings.market_limit]

        quotes: list[BinaryQuote] = []
        for symbol in symbols:
            info = by_symbol[symbol]
            yes_contract = info.get("C")
            no_contract = info.get("P")

            if not yes_contract and not no_contract:
                continue

            # Fetch market data snapshots
            yes_bid = yes_ask = yes_bid_size = yes_ask_size = 0.0
            no_bid = no_ask = no_bid_size = no_ask_size = 0.0

            try:
                if yes_contract:
                    ticker_yes = await self._request_market_data(yes_contract)
                    if ticker_yes:
                        yes_bid = _safe_float(ticker_yes.bid)
                        yes_ask = _safe_float(ticker_yes.ask)
                        yes_bid_size = _safe_float(ticker_yes.bidSize)
                        yes_ask_size = _safe_float(ticker_yes.askSize)

                if no_contract:
                    ticker_no = await self._request_market_data(no_contract)
                    if ticker_no:
                        no_bid = _safe_float(ticker_no.bid)
                        no_ask = _safe_float(ticker_no.ask)
                        no_bid_size = _safe_float(ticker_no.bidSize)
                        no_ask_size = _safe_float(ticker_no.askSize)
            except Exception as exc:
                LOGGER.warning("ForecastEx: quote fetch for %s failed: %s", symbol, exc)
                continue

            # Skip contracts with no meaningful quotes
            if yes_ask <= 0 and no_ask <= 0:
                continue

            detail_obj = info.get("detail")
            meta: Dict[str, Any] = {"source": "forecastex"}
            if detail_obj:
                contract_obj = detail_obj.contract
                meta["last_trade_date"] = getattr(
                    contract_obj, "lastTradeDateOrContractMonth", ""
                )
                meta["long_name"] = getattr(detail_obj, "longName", "")
                meta["canonical_text"] = getattr(detail_obj, "longName", symbol)

            quote = self._normalize_quote(
                symbol=symbol,
                yes_bid=yes_bid,
                yes_ask=yes_ask,
                no_bid=no_bid,
                no_ask=no_ask,
                yes_bid_size=yes_bid_size,
                yes_ask_size=yes_ask_size,
                no_bid_size=no_bid_size,
                no_ask_size=no_ask_size,
                metadata=meta,
            )
            quotes.append(quote)

        LOGGER.info("ForecastEx: fetched %d quotes", len(quotes))
        return quotes

    async def _request_market_data(self, contract: Any) -> Any:
        """Request a snapshot of market data for a single contract."""
        if not self._ib:
            return None
        try:
            await self._rate_limiter.acquire()
            ticker = await self._ib.reqTickersAsync(contract)
            return ticker[0] if ticker else None
        except Exception as exc:
            LOGGER.warning(
                "ForecastEx: reqTickers failed for %s: %s",
                getattr(contract, "symbol", "?"),
                exc,
            )
            return None

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def supports_streaming(self) -> bool:
        """ForecastEx supports real-time streaming via TWS reqMktData."""
        return bool(
            self._settings.enable_stream
            and _HAS_IB_ASYNC
            and self._connected
        )

    async def stream_quotes(self) -> AsyncIterator[BinaryQuote]:
        """Stream real-time quotes from TWS market data subscriptions.

        Uses reqMktData for live streaming — TWS pushes price updates
        as they happen.  Yields BinaryQuote objects on each update.
        """
        if not self.supports_streaming() or not self._ib:
            return

        contracts_details = await self._discover_contracts()
        if not contracts_details:
            return

        # Group by symbol → subscribe to both YES and NO
        by_symbol: Dict[str, Dict[str, Any]] = {}
        for detail in contracts_details:
            contract = detail.contract
            symbol = contract.symbol
            if symbol not in by_symbol:
                by_symbol[symbol] = {"C": None, "P": None}
            by_symbol[symbol][contract.right] = contract

        # Subscribe to market data for each contract
        subscribed_tickers: Dict[str, Dict[str, Any]] = {}
        symbols = list(by_symbol.keys())
        if self._settings.market_limit > 0:
            symbols = symbols[: self._settings.market_limit]

        for symbol in symbols:
            info = by_symbol[symbol]
            if symbol not in subscribed_tickers:
                subscribed_tickers[symbol] = {"yes": None, "no": None}

            for right_key, side_key in [("C", "yes"), ("P", "no")]:
                contract = info.get(right_key)
                if contract:
                    try:
                        self._ib.reqMktData(contract)
                        subscribed_tickers[symbol][side_key] = contract
                    except Exception as exc:
                        LOGGER.warning(
                            "ForecastEx: stream subscribe failed for %s %s: %s",
                            symbol, right_key, exc,
                        )

        LOGGER.info("ForecastEx: streaming %d symbols", len(subscribed_tickers))

        try:
            while self._connected and self._ib and self._ib.isConnected():
                # Wait for TWS to deliver pending updates
                await self._ib.updateEvent
                pending_tickers = self._ib.pendingTickers()

                for ticker in pending_tickers:
                    contract = ticker.contract
                    if not contract or getattr(contract, "exchange", "") != "FORECASTX":
                        continue

                    symbol = contract.symbol
                    right = contract.right

                    # Update the appropriate side's data
                    if symbol not in self._last_quotes:
                        self._last_quotes[symbol] = self._normalize_quote(
                            symbol=symbol,
                            yes_bid=0.0, yes_ask=0.0,
                            no_bid=0.0, no_ask=0.0,
                        )

                    prev = self._last_quotes[symbol]
                    if right == "C":
                        updated = BinaryQuote(
                            venue=self.venue,
                            market_id=symbol,
                            yes_buy_price=_safe_float(ticker.ask) or prev.yes_buy_price,
                            no_buy_price=prev.no_buy_price,
                            yes_buy_size=_safe_float(ticker.askSize) or prev.yes_buy_size,
                            no_buy_size=prev.no_buy_size,
                            yes_bid_price=_safe_float(ticker.bid) or prev.yes_bid_price,
                            no_bid_price=prev.no_bid_price,
                            yes_bid_size=_safe_float(ticker.bidSize) or prev.yes_bid_size,
                            no_bid_size=prev.no_bid_size,
                            fee_per_contract=_DEFAULT_FEE_PER_CONTRACT,
                            observed_at=datetime.now(timezone.utc),
                            metadata=prev.metadata,
                        )
                    else:
                        updated = BinaryQuote(
                            venue=self.venue,
                            market_id=symbol,
                            yes_buy_price=prev.yes_buy_price,
                            no_buy_price=_safe_float(ticker.ask) or prev.no_buy_price,
                            yes_buy_size=prev.yes_buy_size,
                            no_buy_size=_safe_float(ticker.askSize) or prev.no_buy_size,
                            yes_bid_price=prev.yes_bid_price,
                            no_bid_price=_safe_float(ticker.bid) or prev.no_bid_price,
                            yes_bid_size=prev.yes_bid_size,
                            no_bid_size=_safe_float(ticker.bidSize) or prev.no_bid_size,
                            fee_per_contract=_DEFAULT_FEE_PER_CONTRACT,
                            observed_at=datetime.now(timezone.utc),
                            metadata=prev.metadata,
                        )

                    self._last_quotes[symbol] = updated
                    yield updated

        except asyncio.CancelledError:
            LOGGER.info("ForecastEx: stream cancelled")
        except Exception as exc:
            LOGGER.error("ForecastEx: stream error: %s", exc)
        finally:
            # Cancel market data subscriptions
            for symbol, subs in subscribed_tickers.items():
                for side_key in ("yes", "no"):
                    contract = subs.get(side_key)
                    if contract and self._ib:
                        try:
                            self._ib.cancelMktData(contract)
                        except Exception:
                            pass
            LOGGER.info("ForecastEx: stream shutdown")

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    async def place_single_order(self, leg: TradeLegPlan) -> LegExecutionResult:
        """Place a single order on ForecastEx.

        ForecastEx only supports buying — if the bot wants to 'sell' a
        position, it must buy the opposing contract (IBKR nets
        automatically).  This method always generates a BUY order for
        the requested side.
        """
        if not self._ib or not self._connected:
            return LegExecutionResult(
                success=False,
                order_id=None,
                requested_contracts=leg.contracts,
                filled_contracts=0,
                average_price=None,
                raw={"error": "not connected"},
            )

        right = self.side_to_right(leg.side)
        contract = self._make_contract(
            symbol=leg.market_id,
            right=right,
        )

        # Rate-limit before qualifying + placing
        await self._rate_limiter.acquire()

        # Qualify the contract (resolve conId etc.)
        try:
            qualified = await self._ib.qualifyContractsAsync(contract)
            if qualified:
                contract = qualified[0]
        except Exception as exc:
            LOGGER.warning("ForecastEx: contract qualification failed: %s", exc)

        # Create a limit order — ForecastEx only allows BUY
        order = ibasync.LimitOrder(
            action="BUY",
            totalQuantity=leg.contracts,
            lmtPrice=leg.limit_price,
            tif=self._settings.default_tif,
        )

        try:
            trade = self._ib.placeOrder(contract, order)

            # Wait for fill with timeout
            timeout = self._timeout_seconds
            start = time.monotonic()
            while (time.monotonic() - start) < timeout:
                if trade.isDone():
                    break
                await asyncio.sleep(0.1)

            filled = trade.orderStatus.filled or 0
            avg_price = trade.orderStatus.avgFillPrice or 0.0
            order_id = str(trade.order.orderId) if trade.order else None

            return LegExecutionResult(
                success=filled > 0,
                order_id=order_id,
                requested_contracts=leg.contracts,
                filled_contracts=int(filled),
                average_price=avg_price if filled > 0 else None,
                raw={
                    "status": trade.orderStatus.status,
                    "remaining": trade.orderStatus.remaining,
                },
            )
        except Exception as exc:
            LOGGER.error("ForecastEx: order placement failed: %s", exc)
            return LegExecutionResult(
                success=False,
                order_id=None,
                requested_contracts=leg.contracts,
                filled_contracts=0,
                average_price=None,
                raw={"error": str(exc)},
            )

    async def place_pair_order(self, plan: TradePlan) -> PairExecutionResult:
        """Place both legs of a pair trade sequentially.

        ForecastEx legs are always BUY orders.  Sequential execution
        matches the safety model from Phase 0B.

        PairExecutionResult requires `venue`, `market_id`, `yes_leg`,
        `no_leg` fields.  The adapter maps TradePlan legs to YES/NO
        based on side.
        """
        empty_leg = LegExecutionResult(
            success=False,
            order_id=None,
            requested_contracts=plan.contracts,
            filled_contracts=0,
            average_price=None,
        )

        yes_result = empty_leg
        no_result = empty_leg
        error: str | None = None

        for leg_plan in plan.legs:
            result = await self.place_single_order(leg_plan)
            if leg_plan.side == Side.YES:
                yes_result = result
            else:
                no_result = result
            if not result.success:
                error = f"leg {leg_plan.side.value} failed"
                break

        # Derive venue and market_id from the plan
        venue = "forecastex"
        market_id = plan.legs[0].market_id if plan.legs else ""

        return PairExecutionResult(
            venue=venue,
            market_id=market_id,
            yes_leg=yes_result,
            no_leg=no_result,
            error=error,
        )

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order by order ID."""
        if not self._ib or not self._connected:
            return False

        try:
            open_trades = self._ib.openTrades()
            for trade in open_trades:
                if str(trade.order.orderId) == order_id:
                    self._ib.cancelOrder(trade.order)
                    return True
            LOGGER.warning("ForecastEx: order %s not found in open trades", order_id)
            return False
        except Exception as exc:
            LOGGER.error("ForecastEx: cancel_order failed: %s", exc)
            return False

    async def get_order_status(self, order_id: str) -> OrderStatus | None:
        """Get status of an order by order ID."""
        if not self._ib or not self._connected:
            return None

        try:
            open_trades = self._ib.openTrades()
            for trade in open_trades:
                if str(trade.order.orderId) == order_id:
                    status = trade.orderStatus
                    return OrderStatus(
                        order_id=order_id,
                        state=_map_ib_order_state(status.status),
                        filled_contracts=int(status.filled or 0),
                        remaining_contracts=int(status.remaining or 0),
                        average_price=status.avgFillPrice or None,
                        raw={"ib_status": status.status},
                    )
            return None
        except Exception as exc:
            LOGGER.error("ForecastEx: get_order_status failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Account / cash
    # ------------------------------------------------------------------

    async def get_available_cash(self) -> float | None:
        """Fetch available cash from the IBKR account."""
        if not self._ib or not self._connected:
            return None

        try:
            account_values = self._ib.accountValues()
            for av in account_values:
                if av.tag == "AvailableFunds" and av.currency == "USD":
                    return float(av.value)
            return None
        except Exception as exc:
            LOGGER.error("ForecastEx: get_available_cash failed: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(value: Any) -> float:
    """Convert a TWS ticker value to float, handling nan/None/-1 sentinel."""
    if value is None:
        return 0.0
    try:
        f = float(value)
        if f != f or f < 0:  # nan or negative sentinel
            return 0.0
        return f
    except (TypeError, ValueError):
        return 0.0


def _map_ib_order_state(ib_status: str) -> OrderState:
    """Map IBKR order status string to our OrderState enum."""
    ib_status_lower = (ib_status or "").lower().strip()
    mapping = {
        "submitted": OrderState.OPEN,
        "presubmitted": OrderState.OPEN,
        "pendingsubmit": OrderState.OPEN,
        "pendingcancel": OrderState.OPEN,
        "filled": OrderState.FILLED,
        "cancelled": OrderState.CANCELLED,
        "canceled": OrderState.CANCELLED,
        "inactive": OrderState.EXPIRED,
        "apicancelled": OrderState.CANCELLED,
    }
    return mapping.get(ib_status_lower, OrderState.UNKNOWN)
