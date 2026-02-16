"""Tests for Phase 1C: Endpoint-specific circuit breakers."""

from __future__ import annotations

import time

import pytest

from arb_bot.framework.circuit_breaker import CircuitState
from arb_bot.framework.endpoint_breaker import (
    EndpointBreakerConfig,
    EndpointCircuitBreaker,
    EndpointBreakerRegistry,
    KALSHI_ENDPOINT_CONFIGS,
    POLYMARKET_ENDPOINT_CONFIGS,
    create_default_registry,
)


# ---------------------------------------------------------------------------
# EndpointCircuitBreaker
# ---------------------------------------------------------------------------


class TestEndpointCircuitBreaker:
    def test_basic_properties(self) -> None:
        config = EndpointBreakerConfig(venue="kalshi", endpoint="orderbook")
        eb = EndpointCircuitBreaker(config)
        assert eb.venue == "kalshi"
        assert eb.endpoint == "orderbook"
        assert eb.name == "kalshi/orderbook"
        assert eb.state == CircuitState.CLOSED
        assert eb.allows_request() is True

    def test_failure_opens_breaker(self) -> None:
        config = EndpointBreakerConfig(
            venue="kalshi", endpoint="markets", failure_threshold=2,
        )
        eb = EndpointCircuitBreaker(config)
        eb.record_failure()
        assert eb.state == CircuitState.CLOSED
        eb.record_failure()
        assert eb.is_open is True

    def test_success_resets_failures(self) -> None:
        config = EndpointBreakerConfig(
            venue="kalshi", endpoint="markets", failure_threshold=3,
        )
        eb = EndpointCircuitBreaker(config)
        eb.record_failure()
        eb.record_failure()
        eb.record_success()
        eb.record_failure()
        assert eb.state == CircuitState.CLOSED

    def test_429_filtered_when_disabled(self) -> None:
        config = EndpointBreakerConfig(
            venue="kalshi", endpoint="markets",
            failure_threshold=1,
            count_429_as_failure=False,
        )
        eb = EndpointCircuitBreaker(config)
        eb.record_failure(status_code=429)
        assert eb.state == CircuitState.CLOSED  # 429 was filtered out

    def test_429_counted_when_enabled(self) -> None:
        config = EndpointBreakerConfig(
            venue="kalshi", endpoint="markets",
            failure_threshold=1,
            count_429_as_failure=True,
        )
        eb = EndpointCircuitBreaker(config)
        eb.record_failure(status_code=429)
        assert eb.is_open is True

    def test_5xx_filtered_when_disabled(self) -> None:
        config = EndpointBreakerConfig(
            venue="kalshi", endpoint="markets",
            failure_threshold=1,
            count_5xx_as_failure=False,
        )
        eb = EndpointCircuitBreaker(config)
        eb.record_failure(status_code=500)
        assert eb.state == CircuitState.CLOSED
        eb.record_failure(status_code=503)
        assert eb.state == CircuitState.CLOSED

    def test_5xx_counted_when_enabled(self) -> None:
        config = EndpointBreakerConfig(
            venue="kalshi", endpoint="markets",
            failure_threshold=1,
            count_5xx_as_failure=True,
        )
        eb = EndpointCircuitBreaker(config)
        eb.record_failure(status_code=502)
        assert eb.is_open is True

    def test_timeout_filtered_when_disabled(self) -> None:
        config = EndpointBreakerConfig(
            venue="kalshi", endpoint="markets",
            failure_threshold=1,
            count_timeout_as_failure=False,
        )
        eb = EndpointCircuitBreaker(config)
        eb.record_failure(is_timeout=True)
        assert eb.state == CircuitState.CLOSED

    def test_timeout_counted_when_enabled(self) -> None:
        config = EndpointBreakerConfig(
            venue="kalshi", endpoint="markets",
            failure_threshold=1,
            count_timeout_as_failure=True,
        )
        eb = EndpointCircuitBreaker(config)
        eb.record_failure(is_timeout=True)
        assert eb.is_open is True

    def test_non_http_failure_always_counted(self) -> None:
        """Failure with no status_code and no timeout flag always counts."""
        config = EndpointBreakerConfig(
            venue="kalshi", endpoint="markets",
            failure_threshold=1,
            count_429_as_failure=False,
            count_5xx_as_failure=False,
            count_timeout_as_failure=False,
        )
        eb = EndpointCircuitBreaker(config)
        eb.record_failure()  # No status_code, no timeout
        assert eb.is_open is True

    def test_4xx_non_429_always_counted(self) -> None:
        """Non-429 4xx errors are always counted as failures."""
        config = EndpointBreakerConfig(
            venue="kalshi", endpoint="markets",
            failure_threshold=1,
            count_429_as_failure=False,
        )
        eb = EndpointCircuitBreaker(config)
        eb.record_failure(status_code=400)
        assert eb.is_open is True

    def test_reset(self) -> None:
        config = EndpointBreakerConfig(
            venue="kalshi", endpoint="markets", failure_threshold=1,
        )
        eb = EndpointCircuitBreaker(config)
        eb.record_failure()
        assert eb.is_open is True
        eb.reset()
        assert eb.state == CircuitState.CLOSED

    def test_snapshot(self) -> None:
        config = EndpointBreakerConfig(venue="poly", endpoint="book")
        eb = EndpointCircuitBreaker(config)
        snap = eb.snapshot()
        assert snap.name == "poly/book"
        assert snap.state == CircuitState.CLOSED

    def test_jittered_recovery(self) -> None:
        """Recovery timeout should include jitter."""
        config = EndpointBreakerConfig(
            venue="kalshi", endpoint="events",
            recovery_timeout_seconds=30.0,
            recovery_jitter_seconds=10.0,
        )
        eb = EndpointCircuitBreaker(config)
        # The actual recovery timeout should be between 30 and 40.
        assert 30.0 <= eb._jittered_recovery <= 40.0


# ---------------------------------------------------------------------------
# EndpointBreakerRegistry
# ---------------------------------------------------------------------------


class TestEndpointBreakerRegistry:
    def test_register_and_get(self) -> None:
        reg = EndpointBreakerRegistry()
        config = EndpointBreakerConfig(venue="kalshi", endpoint="markets")
        eb = reg.register(config)
        assert reg.get("kalshi", "markets") is eb
        assert reg.get("kalshi", "nonexistent") is None

    def test_get_or_register(self) -> None:
        reg = EndpointBreakerRegistry()
        config = EndpointBreakerConfig(venue="kalshi", endpoint="events")
        eb1 = reg.get_or_register(config)
        eb2 = reg.get_or_register(config)
        assert eb1 is eb2

    def test_allows_request_unknown_endpoint(self) -> None:
        """Unknown endpoints are allowed by default."""
        reg = EndpointBreakerRegistry()
        assert reg.allows_request("kalshi", "unknown") is True

    def test_allows_request_open_breaker(self) -> None:
        reg = EndpointBreakerRegistry()
        config = EndpointBreakerConfig(
            venue="kalshi", endpoint="markets", failure_threshold=1,
        )
        eb = reg.register(config)
        eb.record_failure()
        assert reg.allows_request("kalshi", "markets") is False

    def test_venue_breakers(self) -> None:
        reg = EndpointBreakerRegistry()
        reg.register(EndpointBreakerConfig(venue="kalshi", endpoint="markets"))
        reg.register(EndpointBreakerConfig(venue="kalshi", endpoint="events"))
        reg.register(EndpointBreakerConfig(venue="polymarket", endpoint="book"))
        kalshi_breakers = reg.venue_breakers("kalshi")
        assert len(kalshi_breakers) == 2
        poly_breakers = reg.venue_breakers("polymarket")
        assert len(poly_breakers) == 1

    def test_venue_any_open(self) -> None:
        reg = EndpointBreakerRegistry()
        eb1 = reg.register(EndpointBreakerConfig(
            venue="kalshi", endpoint="markets", failure_threshold=1,
        ))
        reg.register(EndpointBreakerConfig(venue="kalshi", endpoint="events"))
        assert reg.venue_any_open("kalshi") is False
        eb1.record_failure()
        assert reg.venue_any_open("kalshi") is True

    def test_venue_all_open(self) -> None:
        reg = EndpointBreakerRegistry()
        eb1 = reg.register(EndpointBreakerConfig(
            venue="kalshi", endpoint="markets", failure_threshold=1,
        ))
        eb2 = reg.register(EndpointBreakerConfig(
            venue="kalshi", endpoint="events", failure_threshold=1,
        ))
        assert reg.venue_all_open("kalshi") is False
        eb1.record_failure()
        assert reg.venue_all_open("kalshi") is False
        eb2.record_failure()
        assert reg.venue_all_open("kalshi") is True

    def test_venue_all_open_empty(self) -> None:
        reg = EndpointBreakerRegistry()
        assert reg.venue_all_open("kalshi") is False

    def test_any_open(self) -> None:
        reg = EndpointBreakerRegistry()
        reg.register(EndpointBreakerConfig(venue="kalshi", endpoint="markets"))
        assert reg.any_open() is False

    def test_check_all(self) -> None:
        reg = EndpointBreakerRegistry()
        eb = reg.register(EndpointBreakerConfig(
            venue="kalshi", endpoint="markets", failure_threshold=1,
        ))
        allowed, reason = reg.check_all()
        assert allowed is True
        eb.record_failure()
        allowed, reason = reg.check_all()
        assert allowed is False
        assert "kalshi/markets" in reason

    def test_snapshots(self) -> None:
        reg = EndpointBreakerRegistry()
        reg.register(EndpointBreakerConfig(venue="kalshi", endpoint="markets"))
        reg.register(EndpointBreakerConfig(venue="polymarket", endpoint="book"))
        snaps = reg.snapshots()
        assert len(snaps) == 2

    def test_reset_all(self) -> None:
        reg = EndpointBreakerRegistry()
        eb1 = reg.register(EndpointBreakerConfig(
            venue="kalshi", endpoint="markets", failure_threshold=1,
        ))
        eb2 = reg.register(EndpointBreakerConfig(
            venue="polymarket", endpoint="book", failure_threshold=1,
        ))
        eb1.record_failure()
        eb2.record_failure()
        assert reg.any_open() is True
        reg.reset_all()
        assert reg.any_open() is False

    def test_reset_venue(self) -> None:
        reg = EndpointBreakerRegistry()
        eb1 = reg.register(EndpointBreakerConfig(
            venue="kalshi", endpoint="markets", failure_threshold=1,
        ))
        eb2 = reg.register(EndpointBreakerConfig(
            venue="polymarket", endpoint="book", failure_threshold=1,
        ))
        eb1.record_failure()
        eb2.record_failure()
        reg.reset_venue("kalshi")
        assert not eb1.is_open
        assert eb2.is_open  # polymarket still open

    def test_all_breakers(self) -> None:
        reg = EndpointBreakerRegistry()
        reg.register(EndpointBreakerConfig(venue="kalshi", endpoint="markets"))
        reg.register(EndpointBreakerConfig(venue="kalshi", endpoint="events"))
        assert len(reg.all_breakers) == 2


# ---------------------------------------------------------------------------
# Default registry
# ---------------------------------------------------------------------------


class TestDefaultRegistry:
    def test_create_default_registry(self) -> None:
        reg = create_default_registry()
        # Should have breakers for all known endpoints
        total = len(KALSHI_ENDPOINT_CONFIGS) + len(POLYMARKET_ENDPOINT_CONFIGS)
        assert len(reg.all_breakers) == total

    def test_default_registry_kalshi_endpoints(self) -> None:
        reg = create_default_registry()
        assert reg.get("kalshi", "markets") is not None
        assert reg.get("kalshi", "orderbook") is not None
        assert reg.get("kalshi", "events") is not None
        assert reg.get("kalshi", "orders") is not None

    def test_default_registry_polymarket_endpoints(self) -> None:
        reg = create_default_registry()
        assert reg.get("polymarket", "markets") is not None
        assert reg.get("polymarket", "book") is not None
        assert reg.get("polymarket", "orders") is not None


# ---------------------------------------------------------------------------
# EndpointBreakerConfig
# ---------------------------------------------------------------------------


class TestEndpointBreakerConfig:
    def test_defaults(self) -> None:
        config = EndpointBreakerConfig(venue="x", endpoint="y")
        assert config.failure_threshold == 5
        assert config.recovery_timeout_seconds == 30.0
        assert config.recovery_jitter_seconds == 10.0
        assert config.success_threshold == 3
        assert config.count_429_as_failure is True
        assert config.count_5xx_as_failure is True
        assert config.count_timeout_as_failure is True
