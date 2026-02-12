"""Circuit breakers and auto-pause for data health (Phase 1B).

Provides a CircuitBreaker that tracks error events and transitions
between CLOSED (normal) → OPEN (halted) → HALF_OPEN (probing) states.
Recovery requires sustained healthy probes before returning to CLOSED.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

LOGGER = logging.getLogger(__name__)


class CircuitState(str, Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Halted — too many failures
    HALF_OPEN = "half_open"  # Probing — testing if recovery is real


@dataclass
class CircuitBreakerConfig:
    """Configuration for a single circuit breaker."""

    name: str = "default"
    failure_threshold: int = 5
    recovery_timeout_seconds: float = 30.0
    success_threshold: int = 3  # Successes in HALF_OPEN before closing
    half_open_max_requests: int = 1  # Max concurrent requests in HALF_OPEN


@dataclass
class CircuitBreakerSnapshot:
    """Read-only snapshot of circuit breaker state."""

    name: str
    state: CircuitState
    failure_count: int
    success_count: int
    last_failure_ts: float
    last_state_change_ts: float
    total_trips: int


class CircuitBreaker:
    """Standard circuit breaker with CLOSED → OPEN → HALF_OPEN → CLOSED flow.

    In CLOSED state: failures increment counter. When failure_threshold is
    reached, transitions to OPEN.

    In OPEN state: all requests are rejected. After recovery_timeout_seconds,
    transitions to HALF_OPEN.

    In HALF_OPEN state: a limited number of requests are allowed through.
    If success_threshold consecutive successes occur, transitions to CLOSED.
    Any failure transitions back to OPEN.
    """

    def __init__(self, config: CircuitBreakerConfig | None = None) -> None:
        self._config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0  # For HALF_OPEN recovery tracking
        self._last_failure_ts: float = 0.0
        self._last_state_change_ts: float = time.monotonic()
        self._total_trips = 0

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def state(self) -> CircuitState:
        self._maybe_transition_to_half_open()
        return self._state

    @property
    def is_open(self) -> bool:
        """True if the circuit is open (halted) or half-open (probing)."""
        return self.state in (CircuitState.OPEN, CircuitState.HALF_OPEN)

    @property
    def is_closed(self) -> bool:
        return self.state == CircuitState.CLOSED

    def allows_request(self) -> bool:
        """Check if a request should be allowed through."""
        current = self.state  # triggers OPEN → HALF_OPEN check
        if current == CircuitState.CLOSED:
            return True
        if current == CircuitState.HALF_OPEN:
            return True  # Let probe requests through
        return False  # OPEN — reject

    def record_success(self) -> None:
        """Record a successful operation."""
        current = self.state
        if current == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self._config.success_threshold:
                self._transition_to(CircuitState.CLOSED)
                LOGGER.info(
                    "Circuit breaker '%s' CLOSED after %d consecutive successes",
                    self._config.name,
                    self._config.success_threshold,
                )
        elif current == CircuitState.CLOSED:
            # Reset failure count on success in closed state.
            self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed operation."""
        self._last_failure_ts = time.monotonic()
        current = self.state

        if current == CircuitState.HALF_OPEN:
            # Any failure in half-open goes back to open.
            self._transition_to(CircuitState.OPEN)
            LOGGER.warning(
                "Circuit breaker '%s' re-OPENED (failure during half-open probe)",
                self._config.name,
            )
        elif current == CircuitState.CLOSED:
            self._failure_count += 1
            if self._failure_count >= self._config.failure_threshold:
                self._transition_to(CircuitState.OPEN)
                self._total_trips += 1
                LOGGER.warning(
                    "Circuit breaker '%s' OPENED after %d failures (trip #%d)",
                    self._config.name,
                    self._failure_count,
                    self._total_trips,
                )
        # In OPEN state, failures are ignored (already halted).

    def reset(self) -> None:
        """Force-reset to CLOSED state (operator override)."""
        self._transition_to(CircuitState.CLOSED)
        LOGGER.info("Circuit breaker '%s' force-reset to CLOSED", self._config.name)

    def snapshot(self) -> CircuitBreakerSnapshot:
        return CircuitBreakerSnapshot(
            name=self._config.name,
            state=self.state,
            failure_count=self._failure_count,
            success_count=self._success_count,
            last_failure_ts=self._last_failure_ts,
            last_state_change_ts=self._last_state_change_ts,
            total_trips=self._total_trips,
        )

    def _maybe_transition_to_half_open(self) -> None:
        if self._state != CircuitState.OPEN:
            return
        elapsed = time.monotonic() - self._last_state_change_ts
        if elapsed >= self._config.recovery_timeout_seconds:
            self._transition_to(CircuitState.HALF_OPEN)
            LOGGER.info(
                "Circuit breaker '%s' transitioning to HALF_OPEN after %.1fs",
                self._config.name,
                elapsed,
            )

    def _transition_to(self, new_state: CircuitState) -> None:
        self._state = new_state
        self._last_state_change_ts = time.monotonic()
        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._success_count = 0
        elif new_state == CircuitState.OPEN:
            self._success_count = 0


class CircuitBreakerRegistry:
    """Registry of named circuit breakers for centralized management.

    Allows the engine to register breakers for different subsystems
    (data health, venue connectivity, etc.) and check them all at once.
    """

    def __init__(self) -> None:
        self._breakers: dict[str, CircuitBreaker] = {}

    def register(self, breaker: CircuitBreaker) -> None:
        self._breakers[breaker.name] = breaker

    def get(self, name: str) -> CircuitBreaker | None:
        return self._breakers.get(name)

    def get_or_create(self, config: CircuitBreakerConfig) -> CircuitBreaker:
        if config.name in self._breakers:
            return self._breakers[config.name]
        breaker = CircuitBreaker(config)
        self._breakers[config.name] = breaker
        return breaker

    @property
    def all_breakers(self) -> list[CircuitBreaker]:
        return list(self._breakers.values())

    def any_open(self) -> bool:
        """True if any registered circuit breaker is open or half-open."""
        return any(b.is_open for b in self._breakers.values())

    def open_breakers(self) -> list[CircuitBreaker]:
        """Return all breakers currently in OPEN or HALF_OPEN state."""
        return [b for b in self._breakers.values() if b.is_open]

    def check_all(self) -> tuple[bool, str]:
        """Check if all breakers allow requests.

        Returns (allowed, reason).
        """
        open_names = [b.name for b in self._breakers.values() if not b.allows_request()]
        if open_names:
            return False, f"circuit breaker(s) open: {', '.join(open_names)}"
        return True, "ok"

    def snapshots(self) -> list[CircuitBreakerSnapshot]:
        return [b.snapshot() for b in self._breakers.values()]

    def reset_all(self) -> None:
        for b in self._breakers.values():
            b.reset()
