"""Operational failure handling playbooks (Phase 5K).

Codified runbooks for 429 storms, stream disconnect loops,
venue outages, and other operational failures. Each playbook
defines triggers, actions, and recovery criteria.

Usage::

    mgr = PlaybookManager(config)
    mgr.register_playbook(Playbook(...))
    events = mgr.evaluate(metrics)
    for event in events:
        # Execute playbook actions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Playbook status
# ---------------------------------------------------------------------------


class PlaybookStatus(str, Enum):
    """Status of a playbook."""

    IDLE = "idle"
    TRIGGERED = "triggered"
    ACTIVE = "active"
    RECOVERING = "recovering"
    COOLDOWN = "cooldown"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlaybookManagerConfig:
    """Configuration for playbook manager.

    Parameters
    ----------
    evaluation_interval:
        Minimum seconds between evaluations. Default 5.
    max_active_playbooks:
        Maximum concurrent active playbooks. Default 3.
    default_cooldown:
        Default cooldown after playbook completes (seconds).
        Default 300.
    """

    evaluation_interval: float = 5.0
    max_active_playbooks: int = 3
    default_cooldown: float = 300.0


# ---------------------------------------------------------------------------
# Playbook definition
# ---------------------------------------------------------------------------


@dataclass
class Playbook:
    """A failure handling playbook definition.

    Parameters
    ----------
    name:
        Playbook name (e.g., "rate_limit_storm").
    trigger_fn:
        Callable(metrics) -> bool. Returns True if playbook should trigger.
    actions:
        List of action names to execute when triggered.
    recovery_fn:
        Callable(metrics) -> bool. Returns True when recovery criteria met.
    severity:
        1 = low, 2 = medium, 3 = high. Default 2.
    cooldown:
        Cooldown period after completion (seconds). Default 300.
    max_duration:
        Maximum time playbook can be active before auto-recovery
        (seconds). Default 600.
    """

    name: str
    trigger_fn: Callable[[Dict[str, Any]], bool]
    actions: List[str]
    recovery_fn: Callable[[Dict[str, Any]], bool]
    severity: int = 2
    cooldown: float = 300.0
    max_duration: float = 600.0


# ---------------------------------------------------------------------------
# Playbook state
# ---------------------------------------------------------------------------


@dataclass
class PlaybookState:
    """Runtime state for a playbook."""

    playbook: Playbook
    status: PlaybookStatus = PlaybookStatus.IDLE
    triggered_at: float = 0.0
    activated_at: float = 0.0
    completed_at: float = 0.0
    trigger_count: int = 0
    last_evaluation: float = 0.0


# ---------------------------------------------------------------------------
# Playbook event
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlaybookEvent:
    """Event from playbook evaluation."""

    playbook_name: str
    event_type: str  # "triggered", "recovered", "expired"
    actions: Tuple[str, ...]
    severity: int
    timestamp: float


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class PlaybookManager:
    """Manages operational failure playbooks.

    Evaluates triggers against current metrics, activates playbooks
    when conditions are met, and monitors recovery criteria.
    """

    def __init__(self, config: PlaybookManagerConfig | None = None) -> None:
        self._config = config or PlaybookManagerConfig()
        self._playbooks: Dict[str, PlaybookState] = {}
        self._last_evaluation: float = 0.0

    @property
    def config(self) -> PlaybookManagerConfig:
        return self._config

    def register_playbook(self, playbook: Playbook) -> None:
        """Register a failure handling playbook."""
        self._playbooks[playbook.name] = PlaybookState(playbook=playbook)

    def unregister_playbook(self, name: str) -> None:
        """Remove a playbook."""
        self._playbooks.pop(name, None)

    def evaluate(
        self,
        metrics: Dict[str, Any],
        now: float | None = None,
    ) -> List[PlaybookEvent]:
        """Evaluate all playbooks against current metrics.

        Returns list of events (triggers, recoveries, expirations).
        """
        if now is None:
            now = time.monotonic()

        events: List[PlaybookEvent] = []
        active_count = sum(
            1 for s in self._playbooks.values()
            if s.status in (PlaybookStatus.TRIGGERED, PlaybookStatus.ACTIVE)
        )

        for state in self._playbooks.values():
            pb = state.playbook
            state.last_evaluation = now

            if state.status == PlaybookStatus.IDLE:
                # Check trigger.
                try:
                    if pb.trigger_fn(metrics):
                        if active_count < self._config.max_active_playbooks:
                            state.status = PlaybookStatus.TRIGGERED
                            state.triggered_at = now
                            state.activated_at = now
                            state.trigger_count += 1
                            active_count += 1
                            events.append(PlaybookEvent(
                                playbook_name=pb.name,
                                event_type="triggered",
                                actions=tuple(pb.actions),
                                severity=pb.severity,
                                timestamp=now,
                            ))
                except Exception:
                    pass

            elif state.status in (PlaybookStatus.TRIGGERED, PlaybookStatus.ACTIVE):
                state.status = PlaybookStatus.ACTIVE

                # Check max duration.
                if now - state.activated_at > pb.max_duration:
                    state.status = PlaybookStatus.COOLDOWN
                    state.completed_at = now
                    events.append(PlaybookEvent(
                        playbook_name=pb.name,
                        event_type="expired",
                        actions=(),
                        severity=pb.severity,
                        timestamp=now,
                    ))
                    continue

                # Check recovery.
                try:
                    if pb.recovery_fn(metrics):
                        state.status = PlaybookStatus.RECOVERING
                        state.completed_at = now
                        events.append(PlaybookEvent(
                            playbook_name=pb.name,
                            event_type="recovered",
                            actions=(),
                            severity=pb.severity,
                            timestamp=now,
                        ))
                except Exception:
                    pass

            elif state.status == PlaybookStatus.RECOVERING:
                state.status = PlaybookStatus.COOLDOWN

            elif state.status == PlaybookStatus.COOLDOWN:
                if now - state.completed_at >= pb.cooldown:
                    state.status = PlaybookStatus.IDLE

        self._last_evaluation = now
        return events

    def get_state(self, name: str) -> PlaybookState | None:
        """Get state for a playbook."""
        return self._playbooks.get(name)

    def active_playbooks(self) -> List[str]:
        """List active playbook names."""
        return [
            name for name, state in self._playbooks.items()
            if state.status in (PlaybookStatus.TRIGGERED, PlaybookStatus.ACTIVE)
        ]

    def playbook_count(self) -> int:
        """Total registered playbooks."""
        return len(self._playbooks)

    def clear(self) -> None:
        """Clear all playbooks."""
        self._playbooks.clear()
