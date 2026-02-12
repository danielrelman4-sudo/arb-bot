"""Adaptive per-endpoint API rate governor (Phase 5A).

Provides AIMD (Additive Increase / Multiplicative Decrease) rate
control with token bucket smoothing. Each venue/endpoint pair has
independent rate state. Parses Retry-After headers and adjusts
automatically.

Usage::

    gov = RateGovernor(config)
    # Before each request:
    await gov.acquire("kalshi", "market_orderbook")
    # After each response:
    gov.record_response("kalshi", "market_orderbook", status_code=200)
    # On 429:
    gov.record_response("kalshi", "market_orderbook", status_code=429, retry_after=2.0)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RateGovernorConfig:
    """Configuration for the rate governor.

    Parameters
    ----------
    base_interval:
        Default minimum interval between requests (seconds).
        Default 0.10.
    min_interval:
        Absolute minimum interval floor. Default 0.01.
    max_interval:
        Absolute maximum interval ceiling. Default 10.0.
    aimd_increase_factor:
        Multiplicative decrease on success (0.9 = shrink by 10%).
        Applied after `success_streak_threshold` consecutive successes.
        Default 0.95.
    aimd_429_factor:
        Multiplicative increase on 429 (1.75 = 75% slower).
        Default 1.75.
    aimd_5xx_factor:
        Multiplicative increase on 5xx (1.25 = 25% slower).
        Default 1.25.
    aimd_error_factor:
        Multiplicative increase on other errors. Default 1.15.
    success_streak_threshold:
        Consecutive successes required before reducing interval.
        Default 8.
    bucket_capacity:
        Token bucket capacity (burst size). Default 5.
    bucket_refill_rate:
        Tokens per second refilled. If 0, uses interval-based
        rate (1/interval). Default 0 (auto).
    retry_after_max:
        Maximum Retry-After seconds to honor. Default 120.
    """

    base_interval: float = 0.10
    min_interval: float = 0.01
    max_interval: float = 10.0
    aimd_increase_factor: float = 0.95
    aimd_429_factor: float = 1.75
    aimd_5xx_factor: float = 1.25
    aimd_error_factor: float = 1.15
    success_streak_threshold: int = 8
    bucket_capacity: int = 5
    bucket_refill_rate: float = 0.0
    retry_after_max: float = 120.0


# ---------------------------------------------------------------------------
# Endpoint state
# ---------------------------------------------------------------------------


@dataclass
class EndpointState:
    """Rate state for a single venue/endpoint pair."""

    venue: str
    endpoint: str
    interval: float
    success_streak: int = 0
    total_requests: int = 0
    total_429s: int = 0
    total_5xxs: int = 0
    total_errors: int = 0
    # Token bucket state.
    tokens: float = 0.0
    last_refill_time: float = 0.0
    # Throttle state.
    next_request_time: float = 0.0
    last_retry_after: float = 0.0


# ---------------------------------------------------------------------------
# Governor snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GovernorSnapshot:
    """Snapshot of governor state for monitoring."""

    endpoint_count: int
    total_requests: int
    total_429s: int
    total_5xxs: int
    busiest_endpoint: str
    slowest_interval: float
    fastest_interval: float


# ---------------------------------------------------------------------------
# Rate governor
# ---------------------------------------------------------------------------


class RateGovernor:
    """Adaptive per-endpoint API rate governor.

    Implements AIMD rate control with optional token bucket
    smoothing. Each venue/endpoint pair has independent state.
    """

    def __init__(self, config: RateGovernorConfig | None = None) -> None:
        self._config = config or RateGovernorConfig()
        self._endpoints: Dict[str, EndpointState] = {}

    @property
    def config(self) -> RateGovernorConfig:
        return self._config

    def _key(self, venue: str, endpoint: str) -> str:
        return f"{venue}:{endpoint}"

    def _get_state(self, venue: str, endpoint: str) -> EndpointState:
        """Get or create endpoint state."""
        key = self._key(venue, endpoint)
        if key not in self._endpoints:
            cfg = self._config
            state = EndpointState(
                venue=venue,
                endpoint=endpoint,
                interval=cfg.base_interval,
                tokens=float(cfg.bucket_capacity),
                last_refill_time=time.monotonic(),
            )
            self._endpoints[key] = state
        return self._endpoints[key]

    def acquire_wait_time(
        self,
        venue: str,
        endpoint: str,
        now: float | None = None,
    ) -> float:
        """Compute wait time before next request is allowed.

        Returns 0.0 if a request can proceed immediately.
        """
        if now is None:
            now = time.monotonic()
        state = self._get_state(venue, endpoint)
        cfg = self._config

        # Refill token bucket.
        if cfg.bucket_capacity > 0:
            elapsed = now - state.last_refill_time
            refill_rate = cfg.bucket_refill_rate
            if refill_rate <= 0 and state.interval > 0:
                refill_rate = 1.0 / state.interval
            tokens_to_add = elapsed * refill_rate
            state.tokens = min(float(cfg.bucket_capacity), state.tokens + tokens_to_add)
            state.last_refill_time = now

            if state.tokens >= 1.0:
                # Token available — check interval too.
                wait = max(0.0, state.next_request_time - now)
                return wait
            else:
                # No tokens — wait for refill.
                if refill_rate > 0:
                    token_wait = (1.0 - state.tokens) / refill_rate
                else:
                    token_wait = state.interval
                interval_wait = max(0.0, state.next_request_time - now)
                return max(token_wait, interval_wait)
        else:
            # No token bucket — pure interval.
            return max(0.0, state.next_request_time - now)

    def consume(
        self,
        venue: str,
        endpoint: str,
        now: float | None = None,
    ) -> None:
        """Mark a request as being sent. Call after waiting."""
        if now is None:
            now = time.monotonic()
        state = self._get_state(venue, endpoint)
        state.total_requests += 1

        if self._config.bucket_capacity > 0 and state.tokens >= 1.0:
            state.tokens -= 1.0

        state.next_request_time = now + state.interval

    def record_response(
        self,
        venue: str,
        endpoint: str,
        status_code: int,
        retry_after: float = 0.0,
    ) -> None:
        """Record a response and adjust rate.

        Parameters
        ----------
        venue:
            Venue identifier.
        endpoint:
            Endpoint identifier.
        status_code:
            HTTP status code.
        retry_after:
            Parsed Retry-After value in seconds. Default 0.
        """
        state = self._get_state(venue, endpoint)
        cfg = self._config

        if status_code == 429:
            state.total_429s += 1
            state.success_streak = 0
            ra = min(retry_after, cfg.retry_after_max)
            state.last_retry_after = ra
            new_interval = max(
                state.interval * cfg.aimd_429_factor,
                cfg.base_interval,
                ra,
            )
            state.interval = min(cfg.max_interval, new_interval)

        elif status_code >= 500:
            state.total_5xxs += 1
            state.success_streak = 0
            new_interval = state.interval * cfg.aimd_5xx_factor
            state.interval = min(cfg.max_interval, max(cfg.min_interval, new_interval))

        elif status_code >= 400:
            state.total_errors += 1
            state.success_streak = 0
            new_interval = state.interval * cfg.aimd_error_factor
            state.interval = min(cfg.max_interval, max(cfg.min_interval, new_interval))

        else:
            # Success.
            state.success_streak += 1
            if state.success_streak >= cfg.success_streak_threshold:
                new_interval = state.interval * cfg.aimd_increase_factor
                state.interval = max(cfg.min_interval, new_interval)
                state.success_streak = 0

    def record_error(self, venue: str, endpoint: str) -> None:
        """Record a non-HTTP error (timeout, connection error)."""
        state = self._get_state(venue, endpoint)
        cfg = self._config
        state.total_errors += 1
        state.success_streak = 0
        new_interval = state.interval * cfg.aimd_error_factor
        state.interval = min(cfg.max_interval, max(cfg.min_interval, new_interval))

    def get_state(self, venue: str, endpoint: str) -> EndpointState | None:
        """Get current state for a venue/endpoint pair."""
        key = self._key(venue, endpoint)
        return self._endpoints.get(key)

    def get_interval(self, venue: str, endpoint: str) -> float:
        """Get current interval for a venue/endpoint pair."""
        state = self.get_state(venue, endpoint)
        if state is None:
            return self._config.base_interval
        return state.interval

    def snapshot(self) -> GovernorSnapshot:
        """Get a snapshot of the governor state."""
        if not self._endpoints:
            return GovernorSnapshot(
                endpoint_count=0, total_requests=0,
                total_429s=0, total_5xxs=0,
                busiest_endpoint="", slowest_interval=0.0,
                fastest_interval=0.0,
            )

        total_reqs = sum(s.total_requests for s in self._endpoints.values())
        total_429s = sum(s.total_429s for s in self._endpoints.values())
        total_5xxs = sum(s.total_5xxs for s in self._endpoints.values())

        busiest = max(self._endpoints.values(), key=lambda s: s.total_requests)
        slowest = max(s.interval for s in self._endpoints.values())
        fastest = min(s.interval for s in self._endpoints.values())

        return GovernorSnapshot(
            endpoint_count=len(self._endpoints),
            total_requests=total_reqs,
            total_429s=total_429s,
            total_5xxs=total_5xxs,
            busiest_endpoint=self._key(busiest.venue, busiest.endpoint),
            slowest_interval=slowest,
            fastest_interval=fastest,
        )

    def clear(self) -> None:
        """Reset all state."""
        self._endpoints.clear()
