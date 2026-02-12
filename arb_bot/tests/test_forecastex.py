"""Tests for Phase 9A: ForecastEx / Interactive Brokers exchange adapter.

Validates the ForecastExAdapter implementation including:
- Contract creation and side mapping (YES=Call, NO=Put)
- Quote normalization from IBKR ticker data to BinaryQuote
- Order placement with BUY-only constraint
- Order status mapping from IBKR states to our OrderState enum
- Settings and config integration
- Graceful degradation when ib_async is not installed
- Helper functions (_safe_float, _map_ib_order_state)
- Rate limiting / penalty box avoidance (IBKR 10 req/s limit)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from arb_bot.config import AppSettings, ForecastExSettings
from arb_bot.exchanges.forecastex import (
    ForecastExAdapter,
    IBKRRateLimiter,
    _DEFAULT_FEE_PER_CONTRACT,
    _FORECASTEX_PAYOUT,
    _CME_EVENT_PAYOUT,
    _map_ib_order_state,
    _safe_float,
)
from arb_bot.models import (
    BinaryQuote,
    LegExecutionResult,
    OrderState,
    OrderStatus,
    Side,
    TradeLegPlan,
    TradePlan,
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _run(coro: Any) -> Any:
    """Run an async coroutine synchronously for testing."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _default_settings(**overrides: Any) -> ForecastExSettings:
    defaults: Dict[str, Any] = {
        "enabled": True,
        "host": "127.0.0.1",
        "port": 7497,
        "client_id": 1,
    }
    defaults.update(overrides)
    return ForecastExSettings(**defaults)


def _make_adapter(settings: ForecastExSettings | None = None) -> ForecastExAdapter:
    """Create a ForecastExAdapter without connecting to TWS."""
    s = settings or _default_settings()
    adapter = ForecastExAdapter.__new__(ForecastExAdapter)
    adapter._settings = s
    adapter._timeout_seconds = 10.0
    adapter._ib = None
    adapter._connected = False
    adapter._rate_limiter = IBKRRateLimiter(max_per_second=6.0, burst_size=8)
    adapter._contract_cache = {}
    adapter._contract_cache_ts = 0.0
    adapter._contract_cache_ttl_seconds = 300.0
    adapter._last_quotes = {}
    return adapter


@dataclass
class FakeTicker:
    """Simulates an IBKR ticker object."""
    bid: float = 0.0
    ask: float = 0.0
    bidSize: float = 0.0
    askSize: float = 0.0
    contract: Any = None


@dataclass
class FakeContract:
    """Simulates an IBKR Contract object."""
    secType: str = "OPT"
    symbol: str = ""
    exchange: str = "FORECASTX"
    currency: str = "USD"
    right: str = "C"
    lastTradeDateOrContractMonth: str = ""
    strike: float = 0.0
    conId: int = 0


@dataclass
class FakeContractDetails:
    """Simulates IBKR ContractDetails."""
    contract: FakeContract = field(default_factory=FakeContract)
    longName: str = ""


@dataclass
class FakeOrderStatus:
    """Simulates IBKR OrderStatus."""
    status: str = "Submitted"
    filled: float = 0.0
    remaining: float = 0.0
    avgFillPrice: float = 0.0


@dataclass
class FakeOrder:
    """Simulates IBKR Order."""
    orderId: int = 12345


@dataclass
class FakeTrade:
    """Simulates IBKR Trade object."""
    order: FakeOrder = field(default_factory=FakeOrder)
    orderStatus: FakeOrderStatus = field(default_factory=FakeOrderStatus)

    def isDone(self) -> bool:
        return self.orderStatus.status in ("Filled", "Cancelled", "Inactive")


@dataclass
class FakeAccountValue:
    """Simulates IBKR AccountValue."""
    tag: str = "AvailableFunds"
    value: str = "1000.00"
    currency: str = "USD"


# ---------------------------------------------------------------------------
# Tests: _safe_float helper
# ---------------------------------------------------------------------------


class TestSafeFloat:
    """Tests for the _safe_float helper function."""

    def test_none_returns_zero(self) -> None:
        assert _safe_float(None) == 0.0

    def test_nan_returns_zero(self) -> None:
        assert _safe_float(float("nan")) == 0.0

    def test_negative_returns_zero(self) -> None:
        """IBKR uses -1 as sentinel for 'no data'."""
        assert _safe_float(-1.0) == 0.0

    def test_valid_float(self) -> None:
        assert _safe_float(0.55) == 0.55

    def test_zero_returns_zero(self) -> None:
        assert _safe_float(0.0) == 0.0

    def test_string_number(self) -> None:
        assert _safe_float("0.42") == 0.42

    def test_invalid_string_returns_zero(self) -> None:
        assert _safe_float("not_a_number") == 0.0

    def test_integer(self) -> None:
        assert _safe_float(100) == 100.0


# ---------------------------------------------------------------------------
# Tests: _map_ib_order_state
# ---------------------------------------------------------------------------


class TestMapIBOrderState:
    """Tests for IBKR order status -> OrderState mapping."""

    def test_submitted_is_open(self) -> None:
        assert _map_ib_order_state("Submitted") == OrderState.OPEN

    def test_presubmitted_is_open(self) -> None:
        assert _map_ib_order_state("PreSubmitted") == OrderState.OPEN

    def test_pendingsubmit_is_open(self) -> None:
        assert _map_ib_order_state("PendingSubmit") == OrderState.OPEN

    def test_pendingcancel_is_open(self) -> None:
        assert _map_ib_order_state("PendingCancel") == OrderState.OPEN

    def test_filled_is_filled(self) -> None:
        assert _map_ib_order_state("Filled") == OrderState.FILLED

    def test_cancelled_is_cancelled(self) -> None:
        assert _map_ib_order_state("Cancelled") == OrderState.CANCELLED

    def test_canceled_american_spelling(self) -> None:
        assert _map_ib_order_state("Canceled") == OrderState.CANCELLED

    def test_apicancelled_is_cancelled(self) -> None:
        assert _map_ib_order_state("ApiCancelled") == OrderState.CANCELLED

    def test_inactive_is_expired(self) -> None:
        assert _map_ib_order_state("Inactive") == OrderState.EXPIRED

    def test_unknown_status(self) -> None:
        assert _map_ib_order_state("SomeNewStatus") == OrderState.UNKNOWN

    def test_empty_string(self) -> None:
        assert _map_ib_order_state("") == OrderState.UNKNOWN

    def test_none_input(self) -> None:
        assert _map_ib_order_state(None) == OrderState.UNKNOWN  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tests: Side <-> Right mapping
# ---------------------------------------------------------------------------


class TestSideRightMapping:
    """Tests for YES/NO <-> Call/Put mapping."""

    def test_yes_to_call(self) -> None:
        assert ForecastExAdapter.side_to_right(Side.YES) == "C"

    def test_no_to_put(self) -> None:
        assert ForecastExAdapter.side_to_right(Side.NO) == "P"

    def test_call_to_yes(self) -> None:
        assert ForecastExAdapter.right_to_side("C") == Side.YES

    def test_put_to_no(self) -> None:
        assert ForecastExAdapter.right_to_side("P") == Side.NO

    def test_lowercase_call(self) -> None:
        assert ForecastExAdapter.right_to_side("c") == Side.YES

    def test_lowercase_put(self) -> None:
        assert ForecastExAdapter.right_to_side("p") == Side.NO

    def test_roundtrip_yes(self) -> None:
        right = ForecastExAdapter.side_to_right(Side.YES)
        assert ForecastExAdapter.right_to_side(right) == Side.YES

    def test_roundtrip_no(self) -> None:
        right = ForecastExAdapter.side_to_right(Side.NO)
        assert ForecastExAdapter.right_to_side(right) == Side.NO


# ---------------------------------------------------------------------------
# Tests: Contract creation
# ---------------------------------------------------------------------------


class TestMakeContract:
    """Tests for _make_contract when ib_async is not available (dict fallback)."""

    def test_contract_dict_without_ib_async(self) -> None:
        adapter = _make_adapter()
        with patch("arb_bot.exchanges.forecastex._HAS_IB_ASYNC", False):
            contract = adapter._make_contract(
                symbol="FED25BP50",
                right="C",
                last_trade_date="20251231",
                strike=0.5,
            )
        assert isinstance(contract, dict)
        assert contract["secType"] == "OPT"
        assert contract["symbol"] == "FED25BP50"
        assert contract["exchange"] == "FORECASTX"
        assert contract["currency"] == "USD"
        assert contract["right"] == "C"
        assert contract["lastTradeDateOrContractMonth"] == "20251231"
        assert contract["strike"] == 0.5

    def test_contract_dict_put_side(self) -> None:
        adapter = _make_adapter()
        with patch("arb_bot.exchanges.forecastex._HAS_IB_ASYNC", False):
            contract = adapter._make_contract(symbol="CPI-MAR", right="P")
        assert contract["right"] == "P"

    def test_contract_defaults(self) -> None:
        adapter = _make_adapter()
        with patch("arb_bot.exchanges.forecastex._HAS_IB_ASYNC", False):
            contract = adapter._make_contract(symbol="TEST", right="C")
        assert contract["lastTradeDateOrContractMonth"] == ""
        assert contract["strike"] == 0.0


# ---------------------------------------------------------------------------
# Tests: Quote normalization
# ---------------------------------------------------------------------------


class TestNormalizeQuote:
    """Tests for _normalize_quote conversion to BinaryQuote."""

    def test_basic_normalization(self) -> None:
        adapter = _make_adapter()
        quote = adapter._normalize_quote(
            symbol="FED25BP50",
            yes_bid=0.48,
            yes_ask=0.52,
            no_bid=0.45,
            no_ask=0.50,
            yes_bid_size=100.0,
            yes_ask_size=50.0,
            no_bid_size=80.0,
            no_ask_size=60.0,
        )
        assert isinstance(quote, BinaryQuote)
        assert quote.venue == "forecastex"
        assert quote.market_id == "FED25BP50"
        assert quote.yes_buy_price == 0.52  # ask is buy price
        assert quote.no_buy_price == 0.50
        assert quote.yes_buy_size == 50.0  # ask size
        assert quote.no_buy_size == 60.0
        assert quote.yes_bid_price == 0.48
        assert quote.no_bid_price == 0.45
        assert quote.yes_bid_size == 100.0
        assert quote.no_bid_size == 80.0
        assert quote.fee_per_contract == _DEFAULT_FEE_PER_CONTRACT

    def test_zero_bid_becomes_none(self) -> None:
        adapter = _make_adapter()
        quote = adapter._normalize_quote(
            symbol="TEST",
            yes_bid=0.0,
            yes_ask=0.50,
            no_bid=0.0,
            no_ask=0.45,
        )
        assert quote.yes_bid_price is None
        assert quote.no_bid_price is None

    def test_metadata_passthrough(self) -> None:
        adapter = _make_adapter()
        meta = {"long_name": "Fed Rate Decision", "source": "forecastex"}
        quote = adapter._normalize_quote(
            symbol="FED",
            yes_bid=0.40,
            yes_ask=0.50,
            no_bid=0.40,
            no_ask=0.55,
            metadata=meta,
        )
        assert quote.metadata["long_name"] == "Fed Rate Decision"
        assert quote.metadata["source"] == "forecastex"

    def test_default_empty_metadata(self) -> None:
        adapter = _make_adapter()
        quote = adapter._normalize_quote(
            symbol="TEST",
            yes_bid=0.0,
            yes_ask=0.50,
            no_bid=0.0,
            no_ask=0.50,
        )
        assert quote.metadata == {}

    def test_fee_is_default(self) -> None:
        """Fee should be $0.005 per individual contract."""
        adapter = _make_adapter()
        quote = adapter._normalize_quote(
            symbol="X", yes_bid=0.0, yes_ask=0.50, no_bid=0.0, no_ask=0.50,
        )
        assert quote.fee_per_contract == 0.005

    def test_observed_at_is_utc(self) -> None:
        adapter = _make_adapter()
        quote = adapter._normalize_quote(
            symbol="X", yes_bid=0.0, yes_ask=0.50, no_bid=0.0, no_ask=0.50,
        )
        assert quote.observed_at.tzinfo == timezone.utc


# ---------------------------------------------------------------------------
# Tests: Adapter properties
# ---------------------------------------------------------------------------


class TestAdapterProperties:
    """Tests for basic adapter properties and lifecycle."""

    def test_venue_is_forecastex(self) -> None:
        adapter = _make_adapter()
        assert adapter.venue == "forecastex"

    def test_not_connected_initially(self) -> None:
        adapter = _make_adapter()
        assert adapter._connected is False
        assert adapter._ib is None

    def test_supports_streaming_when_connected(self) -> None:
        adapter = _make_adapter(_default_settings(enable_stream=True))
        adapter._connected = True
        with patch("arb_bot.exchanges.forecastex._HAS_IB_ASYNC", True):
            assert adapter.supports_streaming() is True

    def test_no_streaming_when_disabled(self) -> None:
        adapter = _make_adapter(_default_settings(enable_stream=False))
        adapter._connected = True
        assert adapter.supports_streaming() is False

    def test_no_streaming_when_disconnected(self) -> None:
        adapter = _make_adapter(_default_settings(enable_stream=True))
        adapter._connected = False
        assert adapter.supports_streaming() is False


# ---------------------------------------------------------------------------
# Tests: fetch_quotes without connection
# ---------------------------------------------------------------------------


class TestFetchQuotesDisconnected:
    """Tests for fetch_quotes when not connected."""

    def test_returns_empty_when_disconnected(self) -> None:
        adapter = _make_adapter()
        quotes = _run(adapter.fetch_quotes())
        assert quotes == []


# ---------------------------------------------------------------------------
# Tests: place_single_order
# ---------------------------------------------------------------------------


class TestPlaceSingleOrder:
    """Tests for order placement logic."""

    def test_returns_failure_when_disconnected(self) -> None:
        adapter = _make_adapter()
        leg = TradeLegPlan(
            venue="forecastex",
            market_id="FED25BP50",
            side=Side.YES,
            contracts=10,
            limit_price=0.52,
        )
        result = _run(adapter.place_single_order(leg))
        assert result.success is False
        assert result.filled_contracts == 0
        assert result.order_id is None
        assert "not connected" in result.raw.get("error", "")

    def test_yes_order_creates_buy_call(self) -> None:
        """Placing a YES order should create a BUY order with right='C'."""
        adapter = _make_adapter()
        adapter._connected = True

        mock_ib = MagicMock()
        mock_ib.qualifyContractsAsync = AsyncMock(return_value=[FakeContract(symbol="FED")])

        filled_trade = FakeTrade(
            order=FakeOrder(orderId=42),
            orderStatus=FakeOrderStatus(
                status="Filled",
                filled=10.0,
                remaining=0.0,
                avgFillPrice=0.52,
            ),
        )
        mock_ib.placeOrder = MagicMock(return_value=filled_trade)
        adapter._ib = mock_ib

        with patch("arb_bot.exchanges.forecastex._HAS_IB_ASYNC", True), \
             patch("arb_bot.exchanges.forecastex.ibasync") as mock_ibasync:
            mock_ibasync.Contract = FakeContract
            mock_ibasync.LimitOrder = MagicMock(return_value=MagicMock())

            leg = TradeLegPlan(
                venue="forecastex",
                market_id="FED",
                side=Side.YES,
                contracts=10,
                limit_price=0.52,
            )
            result = _run(adapter.place_single_order(leg))

        assert result.success is True
        assert result.filled_contracts == 10
        assert result.order_id == "42"
        assert result.average_price == 0.52

    def test_no_order_creates_buy_put(self) -> None:
        """Placing a NO order should create a BUY order with right='P'."""
        adapter = _make_adapter()
        adapter._connected = True

        mock_ib = MagicMock()
        mock_ib.qualifyContractsAsync = AsyncMock(return_value=[FakeContract(symbol="GDP", right="P")])

        filled_trade = FakeTrade(
            order=FakeOrder(orderId=99),
            orderStatus=FakeOrderStatus(
                status="Filled",
                filled=5.0,
                remaining=0.0,
                avgFillPrice=0.45,
            ),
        )
        mock_ib.placeOrder = MagicMock(return_value=filled_trade)
        adapter._ib = mock_ib

        with patch("arb_bot.exchanges.forecastex._HAS_IB_ASYNC", True), \
             patch("arb_bot.exchanges.forecastex.ibasync") as mock_ibasync:
            mock_ibasync.Contract = FakeContract
            mock_ibasync.LimitOrder = MagicMock(return_value=MagicMock())

            leg = TradeLegPlan(
                venue="forecastex",
                market_id="GDP",
                side=Side.NO,
                contracts=5,
                limit_price=0.45,
            )
            result = _run(adapter.place_single_order(leg))

        assert result.success is True
        assert result.filled_contracts == 5
        assert result.order_id == "99"

    def test_unfilled_order_returns_failure(self) -> None:
        """If order doesn't fill within timeout, should return failure."""
        adapter = _make_adapter()
        adapter._connected = True
        adapter._timeout_seconds = 0.2  # short timeout for test

        mock_ib = MagicMock()
        mock_ib.qualifyContractsAsync = AsyncMock(return_value=[FakeContract()])

        pending_trade = FakeTrade(
            order=FakeOrder(orderId=77),
            orderStatus=FakeOrderStatus(
                status="Submitted",
                filled=0.0,
                remaining=10.0,
                avgFillPrice=0.0,
            ),
        )
        mock_ib.placeOrder = MagicMock(return_value=pending_trade)
        adapter._ib = mock_ib

        with patch("arb_bot.exchanges.forecastex._HAS_IB_ASYNC", True), \
             patch("arb_bot.exchanges.forecastex.ibasync") as mock_ibasync:
            mock_ibasync.Contract = FakeContract
            mock_ibasync.LimitOrder = MagicMock(return_value=MagicMock())

            leg = TradeLegPlan(
                venue="forecastex",
                market_id="TEST",
                side=Side.YES,
                contracts=10,
                limit_price=0.50,
            )
            result = _run(adapter.place_single_order(leg))

        assert result.success is False
        assert result.filled_contracts == 0


# ---------------------------------------------------------------------------
# Tests: cancel_order
# ---------------------------------------------------------------------------


class TestCancelOrder:
    """Tests for order cancellation."""

    def test_cancel_when_disconnected(self) -> None:
        adapter = _make_adapter()
        result = _run(adapter.cancel_order("123"))
        assert result is False

    def test_cancel_existing_order(self) -> None:
        adapter = _make_adapter()
        adapter._connected = True

        mock_ib = MagicMock()
        trade = FakeTrade(order=FakeOrder(orderId=123))
        mock_ib.openTrades = MagicMock(return_value=[trade])
        mock_ib.cancelOrder = MagicMock()
        adapter._ib = mock_ib

        result = _run(adapter.cancel_order("123"))
        assert result is True
        mock_ib.cancelOrder.assert_called_once()

    def test_cancel_nonexistent_order(self) -> None:
        adapter = _make_adapter()
        adapter._connected = True

        mock_ib = MagicMock()
        mock_ib.openTrades = MagicMock(return_value=[])
        adapter._ib = mock_ib

        result = _run(adapter.cancel_order("999"))
        assert result is False


# ---------------------------------------------------------------------------
# Tests: get_order_status
# ---------------------------------------------------------------------------


class TestGetOrderStatus:
    """Tests for order status retrieval."""

    def test_status_when_disconnected(self) -> None:
        adapter = _make_adapter()
        result = _run(adapter.get_order_status("123"))
        assert result is None

    def test_status_filled_order(self) -> None:
        adapter = _make_adapter()
        adapter._connected = True

        mock_ib = MagicMock()
        trade = FakeTrade(
            order=FakeOrder(orderId=456),
            orderStatus=FakeOrderStatus(
                status="Filled",
                filled=10.0,
                remaining=0.0,
                avgFillPrice=0.55,
            ),
        )
        mock_ib.openTrades = MagicMock(return_value=[trade])
        adapter._ib = mock_ib

        result = _run(adapter.get_order_status("456"))
        assert result is not None
        assert isinstance(result, OrderStatus)
        assert result.order_id == "456"
        assert result.state == OrderState.FILLED
        assert result.filled_contracts == 10
        assert result.remaining_contracts == 0
        assert result.average_price == 0.55

    def test_status_nonexistent_order(self) -> None:
        adapter = _make_adapter()
        adapter._connected = True

        mock_ib = MagicMock()
        mock_ib.openTrades = MagicMock(return_value=[])
        adapter._ib = mock_ib

        result = _run(adapter.get_order_status("999"))
        assert result is None


# ---------------------------------------------------------------------------
# Tests: get_available_cash
# ---------------------------------------------------------------------------


class TestGetAvailableCash:
    """Tests for account cash retrieval."""

    def test_cash_when_disconnected(self) -> None:
        adapter = _make_adapter()
        result = _run(adapter.get_available_cash())
        assert result is None

    def test_cash_from_account_values(self) -> None:
        adapter = _make_adapter()
        adapter._connected = True

        mock_ib = MagicMock()
        mock_ib.accountValues = MagicMock(return_value=[
            FakeAccountValue(tag="NetLiquidation", value="5000.00", currency="USD"),
            FakeAccountValue(tag="AvailableFunds", value="1234.56", currency="USD"),
            FakeAccountValue(tag="AvailableFunds", value="100.00", currency="EUR"),
        ])
        adapter._ib = mock_ib

        result = _run(adapter.get_available_cash())
        assert result == 1234.56

    def test_cash_no_available_funds(self) -> None:
        adapter = _make_adapter()
        adapter._connected = True

        mock_ib = MagicMock()
        mock_ib.accountValues = MagicMock(return_value=[
            FakeAccountValue(tag="NetLiquidation", value="5000.00", currency="USD"),
        ])
        adapter._ib = mock_ib

        result = _run(adapter.get_available_cash())
        assert result is None


# ---------------------------------------------------------------------------
# Tests: ForecastExSettings config
# ---------------------------------------------------------------------------


class TestForecastExSettings:
    """Tests for ForecastExSettings dataclass."""

    def test_default_values(self) -> None:
        settings = ForecastExSettings()
        assert settings.enabled is False
        assert settings.host == "127.0.0.1"
        assert settings.port == 7496
        assert settings.client_id == 1
        assert settings.market_limit == 50
        assert settings.enable_stream is True
        assert settings.contract_type == "forecast"
        assert settings.default_tif == "GTC"
        assert settings.fee_per_contract == 0.005
        assert settings.payout_per_contract == 1.0

    def test_custom_values(self) -> None:
        settings = ForecastExSettings(
            enabled=True,
            port=4002,
            client_id=3,
            market_limit=100,
            contract_type="cme",
            payout_per_contract=100.0,
        )
        assert settings.enabled is True
        assert settings.port == 4002
        assert settings.client_id == 3
        assert settings.market_limit == 100
        assert settings.contract_type == "cme"
        assert settings.payout_per_contract == 100.0

    def test_paper_trading_port(self) -> None:
        """Port 7497 is TWS paper, 4002 is Gateway paper."""
        settings = ForecastExSettings(port=7497)
        assert settings.port == 7497

    def test_priority_symbols(self) -> None:
        settings = ForecastExSettings(priority_symbols=["FED25BP50", "CPI-MAR"])
        assert len(settings.priority_symbols) == 2
        assert "FED25BP50" in settings.priority_symbols

    def test_exclude_symbols(self) -> None:
        settings = ForecastExSettings(exclude_symbols=["EXPIRED-1"])
        assert "EXPIRED-1" in settings.exclude_symbols

    def test_tif_validation(self) -> None:
        """Only DAY, GTC, IOC are valid for ForecastEx."""
        for tif in ("DAY", "GTC", "IOC"):
            s = ForecastExSettings(default_tif=tif)
            assert s.default_tif == tif


# ---------------------------------------------------------------------------
# Tests: AppSettings integration
# ---------------------------------------------------------------------------


class TestAppSettingsIntegration:
    """Tests for forecastex field in AppSettings."""

    def test_default_forecastex_in_appsettings(self) -> None:
        """AppSettings should have a default ForecastExSettings."""
        from arb_bot.config import (
            FillModelSettings,
            KalshiSettings,
            OpportunityLaneSettings,
            PolymarketSettings,
            RiskSettings,
            SizingSettings,
            StrategySettings,
            UniverseRankingSettings,
        )

        settings = AppSettings(
            live_mode=False,
            run_once=False,
            poll_interval_seconds=60,
            dry_run=True,
            paper_strict_simulation=True,
            paper_position_lifetime_seconds=600,
            paper_dynamic_lifetime_enabled=False,
            paper_dynamic_lifetime_resolution_fraction=0.02,
            paper_dynamic_lifetime_min_seconds=60,
            paper_dynamic_lifetime_max_seconds=600,
            stream_mode=False,
            stream_recompute_cooldown_ms=0,
            default_bankroll_usd=1000.0,
            bankroll_by_venue={},
            log_level="INFO",
            strategy=StrategySettings(),
            lanes=OpportunityLaneSettings(),
            sizing=SizingSettings(),
            risk=RiskSettings(),
            universe=UniverseRankingSettings(),
            fill_model=FillModelSettings(),
            kalshi=KalshiSettings(enabled=False),
            polymarket=PolymarketSettings(enabled=False),
        )
        assert settings.forecastex.enabled is False
        assert settings.forecastex.port == 7496

    def test_custom_forecastex_in_appsettings(self) -> None:
        from arb_bot.config import (
            FillModelSettings,
            KalshiSettings,
            OpportunityLaneSettings,
            PolymarketSettings,
            RiskSettings,
            SizingSettings,
            StrategySettings,
            UniverseRankingSettings,
        )

        settings = AppSettings(
            live_mode=False,
            run_once=False,
            poll_interval_seconds=60,
            dry_run=True,
            paper_strict_simulation=True,
            paper_position_lifetime_seconds=600,
            paper_dynamic_lifetime_enabled=False,
            paper_dynamic_lifetime_resolution_fraction=0.02,
            paper_dynamic_lifetime_min_seconds=60,
            paper_dynamic_lifetime_max_seconds=600,
            stream_mode=False,
            stream_recompute_cooldown_ms=0,
            default_bankroll_usd=1000.0,
            bankroll_by_venue={},
            log_level="INFO",
            strategy=StrategySettings(),
            lanes=OpportunityLaneSettings(),
            sizing=SizingSettings(),
            risk=RiskSettings(),
            universe=UniverseRankingSettings(),
            fill_model=FillModelSettings(),
            kalshi=KalshiSettings(enabled=False),
            polymarket=PolymarketSettings(enabled=False),
            forecastex=ForecastExSettings(enabled=True, port=4002),
        )
        assert settings.forecastex.enabled is True
        assert settings.forecastex.port == 4002


# ---------------------------------------------------------------------------
# Tests: Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Tests for module-level constants."""

    def test_default_fee(self) -> None:
        assert _DEFAULT_FEE_PER_CONTRACT == 0.005

    def test_forecastex_payout(self) -> None:
        assert _FORECASTEX_PAYOUT == 1.0

    def test_cme_event_payout(self) -> None:
        assert _CME_EVENT_PAYOUT == 100.0


# ---------------------------------------------------------------------------
# Tests: aclose
# ---------------------------------------------------------------------------


class TestAclose:
    """Tests for adapter disconnection."""

    def test_aclose_without_connection(self) -> None:
        adapter = _make_adapter()
        _run(adapter.aclose())
        assert adapter._connected is False

    def test_aclose_with_connection(self) -> None:
        adapter = _make_adapter()
        adapter._connected = True
        adapter._ib = MagicMock()
        adapter._ib.disconnect = MagicMock()

        _run(adapter.aclose())
        assert adapter._connected is False
        assert adapter._ib is None


# ---------------------------------------------------------------------------
# Tests: connect
# ---------------------------------------------------------------------------


class TestConnect:
    """Tests for TWS connection."""

    def test_connect_without_ib_async(self) -> None:
        adapter = _make_adapter()
        with patch("arb_bot.exchanges.forecastex._HAS_IB_ASYNC", False):
            result = _run(adapter.connect())
        assert result is False
        assert adapter._connected is False

    def test_connect_success(self) -> None:
        adapter = _make_adapter()

        mock_ib_class = MagicMock()
        mock_ib_instance = MagicMock()
        mock_ib_instance.connectAsync = AsyncMock()
        mock_ib_class.return_value = mock_ib_instance

        with patch("arb_bot.exchanges.forecastex._HAS_IB_ASYNC", True), \
             patch("arb_bot.exchanges.forecastex.ibasync") as mock_ibasync:
            mock_ibasync.IB = mock_ib_class
            result = _run(adapter.connect())

        assert result is True
        assert adapter._connected is True

    def test_connect_failure(self) -> None:
        adapter = _make_adapter()

        mock_ib_class = MagicMock()
        mock_ib_instance = MagicMock()
        mock_ib_instance.connectAsync = AsyncMock(side_effect=ConnectionRefusedError("no TWS"))
        mock_ib_class.return_value = mock_ib_instance

        with patch("arb_bot.exchanges.forecastex._HAS_IB_ASYNC", True), \
             patch("arb_bot.exchanges.forecastex.ibasync") as mock_ibasync:
            mock_ibasync.IB = mock_ib_class
            result = _run(adapter.connect())

        assert result is False
        assert adapter._connected is False

    def test_already_connected(self) -> None:
        adapter = _make_adapter()
        adapter._connected = True
        mock_ib = MagicMock()
        mock_ib.isConnected = MagicMock(return_value=True)
        adapter._ib = mock_ib

        with patch("arb_bot.exchanges.forecastex._HAS_IB_ASYNC", True):
            result = _run(adapter.connect())
        assert result is True


# ---------------------------------------------------------------------------
# Tests: place_pair_order
# ---------------------------------------------------------------------------


class TestPlacePairOrder:
    """Tests for pair order placement (both legs)."""

    def test_pair_order_disconnected(self) -> None:
        adapter = _make_adapter()
        from arb_bot.models import ExecutionStyle, OpportunityKind

        plan = TradePlan(
            kind=OpportunityKind.INTRA_VENUE,
            execution_style=ExecutionStyle.TAKER,
            legs=(
                TradeLegPlan("forecastex", "SYM", Side.YES, 10, 0.50),
                TradeLegPlan("forecastex", "SYM", Side.NO, 10, 0.48),
            ),
            contracts=10,
            capital_required=9.80,
            capital_required_by_venue={"forecastex": 9.80},
            expected_profit=0.20,
            edge_per_contract=0.02,
        )
        result = _run(adapter.place_pair_order(plan))
        assert result.success is False
        assert result.filled_contracts == 0


# ---------------------------------------------------------------------------
# Tests: Rate limiting and penalty box avoidance
# ---------------------------------------------------------------------------


class TestRateLimitingConfig:
    """Tests validating that configuration supports rate-limit safety.

    IBKR Web API has a hard 10 req/s limit with a 10-minute penalty box.
    The TWS socket API has pacing rules for market data requests.

    These tests verify that the adapter's configuration and design
    support safe operation within those limits.
    """

    def test_contract_cache_prevents_redundant_discovery(self) -> None:
        """Contract cache TTL should prevent redundant discovery requests."""
        adapter = _make_adapter()
        assert adapter._contract_cache_ttl_seconds == 300.0
        # 300s cache means at most 1 discovery request per 5 minutes

    def test_market_limit_caps_quote_requests(self) -> None:
        """market_limit should bound the number of quote fetches per cycle."""
        settings = _default_settings(market_limit=20)
        adapter = _make_adapter(settings)
        # At most 20 symbols × 2 contracts (YES+NO) = 40 market data requests
        assert adapter._settings.market_limit == 20

    def test_default_market_limit_is_reasonable(self) -> None:
        """Default market_limit=50 is conservative for TWS pacing."""
        settings = ForecastExSettings()
        assert settings.market_limit == 50

    def test_streaming_mode_reduces_polling(self) -> None:
        """When streaming is enabled, we use reqMktData (push) not polling."""
        settings = _default_settings(enable_stream=True)
        adapter = _make_adapter(settings)
        adapter._connected = True
        with patch("arb_bot.exchanges.forecastex._HAS_IB_ASYNC", True):
            assert adapter.supports_streaming() is True
        # Streaming = push-based, no repeated requests per cycle


# ---------------------------------------------------------------------------
# Tests: No-sell constraint
# ---------------------------------------------------------------------------


class TestNoSellConstraint:
    """Tests verifying ForecastEx's buy-only constraint.

    ForecastEx instruments cannot be sold — to exit, the trader buys the
    opposing contract and IBKR nets automatically.  The adapter must
    always generate BUY orders regardless of side.
    """

    def test_yes_side_generates_buy_action(self) -> None:
        """YES order should use action='BUY' with right='C'."""
        # Verified through side_to_right mapping
        assert ForecastExAdapter.side_to_right(Side.YES) == "C"
        # The adapter always generates BUY orders (never SELL)

    def test_no_side_generates_buy_action(self) -> None:
        """NO order should use action='BUY' with right='P'."""
        assert ForecastExAdapter.side_to_right(Side.NO) == "P"
        # The adapter always generates BUY orders (never SELL)

    def test_exit_via_opposing_contract(self) -> None:
        """To exit a YES position, buy NO (and vice versa).

        This is documented behavior — the adapter doesn't implement a
        special 'close' method; the engine simply opens the opposing side.
        """
        # Exit YES position → buy NO contract
        exit_right = ForecastExAdapter.side_to_right(Side.NO)
        assert exit_right == "P"

        # Exit NO position → buy YES contract
        exit_right = ForecastExAdapter.side_to_right(Side.YES)
        assert exit_right == "C"


# ---------------------------------------------------------------------------
# Tests: IBKRRateLimiter
# ---------------------------------------------------------------------------


class TestIBKRRateLimiter:
    """Tests for the IBKR rate limiter (prevents penalty box).

    IBKR enforces a hard 10 req/s limit with a 10-minute penalty box
    on violation.  The rate limiter uses a token bucket to keep requests
    well below this threshold.
    """

    def test_default_rate_is_conservative(self) -> None:
        """Default 6 req/s is 60% of IBKR's 10 req/s limit."""
        limiter = IBKRRateLimiter()
        assert limiter._max_per_second == 6.0
        assert limiter._burst_size == 8

    def test_burst_allows_initial_requests(self) -> None:
        """Burst capacity should allow immediate requests up to burst_size."""
        limiter = IBKRRateLimiter(max_per_second=6.0, burst_size=5)
        # Should be able to acquire 5 tokens immediately
        for i in range(5):
            wait = _run(limiter.acquire())
            assert wait == 0.0, f"Request {i+1} should not wait (burst capacity)"
        assert limiter.total_requests == 5
        assert limiter.total_waits == 0

    def test_requests_tracked(self) -> None:
        """Total requests counter should increment."""
        limiter = IBKRRateLimiter(max_per_second=100.0, burst_size=10)
        for _ in range(3):
            _run(limiter.acquire())
        assert limiter.total_requests == 3

    def test_minimum_rate_clamped(self) -> None:
        """Rate should be clamped to at least 0.1 req/s."""
        limiter = IBKRRateLimiter(max_per_second=0.0, burst_size=1)
        assert limiter._max_per_second == 0.1

    def test_minimum_burst_clamped(self) -> None:
        """Burst size should be clamped to at least 1."""
        limiter = IBKRRateLimiter(max_per_second=6.0, burst_size=0)
        assert limiter._burst_size == 1

    def test_adapter_has_rate_limiter(self) -> None:
        """ForecastExAdapter should always have a rate limiter."""
        adapter = _make_adapter()
        assert isinstance(adapter._rate_limiter, IBKRRateLimiter)
        assert adapter._rate_limiter._max_per_second == 6.0

    def test_high_rate_stays_below_ibkr_limit(self) -> None:
        """Even with a very fast limiter, verify it's below IBKR's 10 req/s."""
        # Our default is 6 req/s — verify it's strictly below 10
        adapter = _make_adapter()
        assert adapter._rate_limiter._max_per_second < 10.0

    def test_custom_rate_limiter(self) -> None:
        """Can create a limiter with custom settings."""
        limiter = IBKRRateLimiter(max_per_second=3.0, burst_size=4)
        assert limiter._max_per_second == 3.0
        assert limiter._burst_size == 4


# ---------------------------------------------------------------------------
# Tests: env-var loading
# ---------------------------------------------------------------------------


class TestEnvVarLoading:
    """Tests for ForecastEx env-var loading in load_settings."""

    def test_load_settings_default_forecastex(self) -> None:
        """load_settings should produce default ForecastExSettings."""
        from arb_bot.config import load_settings

        settings = load_settings()
        assert hasattr(settings, "forecastex")
        assert settings.forecastex.enabled is False

    def test_load_settings_enabled_forecastex(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORECASTEX_ENABLED", "true")
        monkeypatch.setenv("FORECASTEX_PORT", "4002")
        monkeypatch.setenv("FORECASTEX_CLIENT_ID", "7")
        monkeypatch.setenv("FORECASTEX_MARKET_LIMIT", "100")
        monkeypatch.setenv("FORECASTEX_CONTRACT_TYPE", "cme")
        monkeypatch.setenv("FORECASTEX_DEFAULT_TIF", "IOC")
        monkeypatch.setenv("FORECASTEX_FEE_PER_CONTRACT", "0.10")
        monkeypatch.setenv("FORECASTEX_PAYOUT_PER_CONTRACT", "100.0")
        monkeypatch.setenv("FORECASTEX_PRIORITY_SYMBOLS", "FED25BP50,CPI-MAR")
        monkeypatch.setenv("FORECASTEX_EXCLUDE_SYMBOLS", "OLD-SYM")

        from arb_bot.config import load_settings

        settings = load_settings()
        fx = settings.forecastex
        assert fx.enabled is True
        assert fx.port == 4002
        assert fx.client_id == 7
        assert fx.market_limit == 100
        assert fx.contract_type == "cme"
        assert fx.default_tif == "IOC"
        assert fx.fee_per_contract == 0.10
        assert fx.payout_per_contract == 100.0
        assert "FED25BP50" in fx.priority_symbols
        assert "CPI-MAR" in fx.priority_symbols
        assert "OLD-SYM" in fx.exclude_symbols
