"""Endpoint-specific circuit breakers with jittered probe recovery (Phase 1C).

Each venue/endpoint combination gets its own circuit breaker. When an
endpoint starts failing (429s, 5xx, timeouts), its breaker opens
independently — other endpoints on the same venue continue operating.

Recovery probes use random jitter to avoid thundering herd when
multiple breakers recover simultaneously.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any

from arb_bot.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    CircuitBreakerSnapshot,
    CircuitState,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EndpointBreakerConfig:
    """Configuration for an endpoint-specific circuit breaker."""

    venue: str
    endpoint: str  # Logical name, e.g. "markets", "orderbook", "events", "book"
    failure_threshold: int = 5
    recovery_timeout_seconds: float = 30.0
    recovery_jitter_seconds: float = 10.0  # Random jitter added to recovery timeout
    success_threshold: int = 3
    count_429_as_failure: bool = True
    count_5xx_as_failure: bool = True
    count_timeout_as_failure: bool = True


class EndpointCircuitBreaker:
    """Circuit breaker for a specific venue/endpoint pair.

    Wraps a CircuitBreaker with endpoint-aware naming and jittered recovery.
    The jitter prevents multiple breakers from probing simultaneously after
    a venue-wide outage.
    """

    def __init__(self, config: EndpointBreakerConfig) -> None:
        self._config = config
        self._breaker_name = f"{config.venue}/{config.endpoint}"
        self._jittered_recovery = (
            config.recovery_timeout_seconds
            + random.uniform(0, config.recovery_jitter_seconds)
        )
        self._breaker = CircuitBreaker(CircuitBreakerConfig(
            name=self._breaker_name,
            failure_threshold=config.failure_threshold,
            recovery_timeout_seconds=self._jittered_recovery,
            success_threshold=config.success_threshold,
        ))

    @property
    def venue(self) -> str:
        return self._config.venue

    @property
    def endpoint(self) -> str:
        return self._config.endpoint

    @property
    def name(self) -> str:
        return self._breaker_name

    @property
    def state(self) -> CircuitState:
        return self._breaker.state

    @property
    def is_open(self) -> bool:
        return self._breaker.is_open

    def allows_request(self) -> bool:
        return self._breaker.allows_request()

    def record_success(self) -> None:
        self._breaker.record_success()

    def record_failure(self, status_code: int | None = None, is_timeout: bool = False) -> None:
        """Record a failure, optionally filtering by error type.

        Parameters
        ----------
        status_code:
            HTTP status code (e.g. 429, 500). None for non-HTTP errors.
        is_timeout:
            True if the failure was a timeout.
        """
        if status_code == 429 and not self._config.count_429_as_failure:
            return
        if status_code is not None and status_code >= 500 and not self._config.count_5xx_as_failure:
            return
        if is_timeout and not self._config.count_timeout_as_failure:
            return
        self._breaker.record_failure()

    def reset(self) -> None:
        self._breaker.reset()

    def snapshot(self) -> CircuitBreakerSnapshot:
        return self._breaker.snapshot()


class EndpointBreakerRegistry:
    """Registry of endpoint-specific circuit breakers.

    Organizes breakers by venue and endpoint for easy lookup. Provides
    venue-level and global aggregate checks.
    """

    def __init__(self) -> None:
        self._breakers: dict[str, EndpointCircuitBreaker] = {}  # key = "venue/endpoint"
        self._cb_registry = CircuitBreakerRegistry()

    def register(self, config: EndpointBreakerConfig) -> EndpointCircuitBreaker:
        """Create and register a new endpoint breaker."""
        breaker = EndpointCircuitBreaker(config)
        key = breaker.name
        self._breakers[key] = breaker
        self._cb_registry.register(breaker._breaker)
        return breaker

    def get(self, venue: str, endpoint: str) -> EndpointCircuitBreaker | None:
        return self._breakers.get(f"{venue}/{endpoint}")

    def get_or_register(self, config: EndpointBreakerConfig) -> EndpointCircuitBreaker:
        key = f"{config.venue}/{config.endpoint}"
        if key in self._breakers:
            return self._breakers[key]
        return self.register(config)

    def allows_request(self, venue: str, endpoint: str) -> bool:
        """Check if a specific endpoint allows requests.

        If no breaker is registered for the endpoint, returns True (open by default).
        """
        breaker = self.get(venue, endpoint)
        if breaker is None:
            return True
        return breaker.allows_request()

    def venue_breakers(self, venue: str) -> list[EndpointCircuitBreaker]:
        """Get all breakers for a specific venue."""
        return [b for b in self._breakers.values() if b.venue == venue]

    def venue_any_open(self, venue: str) -> bool:
        """True if any breaker for the given venue is open."""
        return any(b.is_open for b in self.venue_breakers(venue))

    def venue_all_open(self, venue: str) -> bool:
        """True if ALL breakers for the given venue are open (venue down)."""
        breakers = self.venue_breakers(venue)
        if not breakers:
            return False
        return all(b.is_open for b in breakers)

    def any_open(self) -> bool:
        return self._cb_registry.any_open()

    def check_all(self) -> tuple[bool, str]:
        return self._cb_registry.check_all()

    def snapshots(self) -> list[CircuitBreakerSnapshot]:
        return self._cb_registry.snapshots()

    def reset_all(self) -> None:
        for b in self._breakers.values():
            b.reset()

    def reset_venue(self, venue: str) -> None:
        for b in self.venue_breakers(venue):
            b.reset()

    @property
    def all_breakers(self) -> list[EndpointCircuitBreaker]:
        return list(self._breakers.values())


# ---------------------------------------------------------------------------
# Default endpoint configurations for known venues
# ---------------------------------------------------------------------------

KALSHI_ENDPOINT_CONFIGS = [
    EndpointBreakerConfig(venue="kalshi", endpoint="markets", failure_threshold=5, recovery_timeout_seconds=30.0),
    EndpointBreakerConfig(venue="kalshi", endpoint="market_detail", failure_threshold=5, recovery_timeout_seconds=30.0),
    EndpointBreakerConfig(venue="kalshi", endpoint="orderbook", failure_threshold=5, recovery_timeout_seconds=20.0),
    EndpointBreakerConfig(venue="kalshi", endpoint="events", failure_threshold=4, recovery_timeout_seconds=60.0),
    EndpointBreakerConfig(venue="kalshi", endpoint="event_detail", failure_threshold=4, recovery_timeout_seconds=60.0),
    EndpointBreakerConfig(venue="kalshi", endpoint="orders", failure_threshold=3, recovery_timeout_seconds=30.0),
    EndpointBreakerConfig(venue="kalshi", endpoint="balance", failure_threshold=5, recovery_timeout_seconds=30.0),
]

POLYMARKET_ENDPOINT_CONFIGS = [
    EndpointBreakerConfig(venue="polymarket", endpoint="markets", failure_threshold=5, recovery_timeout_seconds=30.0),
    EndpointBreakerConfig(venue="polymarket", endpoint="events", failure_threshold=5, recovery_timeout_seconds=30.0),
    EndpointBreakerConfig(venue="polymarket", endpoint="book", failure_threshold=5, recovery_timeout_seconds=20.0),
    EndpointBreakerConfig(venue="polymarket", endpoint="orders", failure_threshold=3, recovery_timeout_seconds=30.0),
]


def create_default_registry() -> EndpointBreakerRegistry:
    """Create a registry with default endpoint breakers for all known venues."""
    registry = EndpointBreakerRegistry()
    for config in KALSHI_ENDPOINT_CONFIGS + POLYMARKET_ENDPOINT_CONFIGS:
        registry.register(config)
    return registry
