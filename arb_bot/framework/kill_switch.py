"""Operational risk controls: kill switches, daily loss cap, failure tracking (Phase 1A).

Provides a KillSwitchManager that checks multiple halt conditions before
each trade cycle. Designed to be called from RiskManager.precheck() or
directly from the engine loop.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

# Sentinel file — presence means "halt all trading".
DEFAULT_KILL_SWITCH_FILE = ".kill_switch"


@dataclass
class DailyLossTracker:
    """Tracks cumulative realized PnL within a UTC calendar day."""

    daily_loss_cap_usd: float = 0.0  # 0 = disabled
    _realized_pnl: float = 0.0
    _day_key: str = ""

    def _current_day_key(self) -> str:
        return time.strftime("%Y-%m-%d", time.gmtime())

    def _maybe_reset(self) -> None:
        key = self._current_day_key()
        if key != self._day_key:
            self._realized_pnl = 0.0
            self._day_key = key

    def record_pnl(self, amount: float) -> None:
        """Record a realized PnL event (positive = profit, negative = loss)."""
        self._maybe_reset()
        self._realized_pnl += amount

    @property
    def realized_pnl(self) -> float:
        self._maybe_reset()
        return self._realized_pnl

    def is_breached(self) -> bool:
        if self.daily_loss_cap_usd <= 0:
            return False
        self._maybe_reset()
        return self._realized_pnl <= -self.daily_loss_cap_usd


@dataclass
class ConsecutiveFailureTracker:
    """Tracks consecutive trade failures and halts after threshold."""

    max_consecutive_failures: int = 0  # 0 = disabled
    _count: int = 0

    def record_success(self) -> None:
        self._count = 0

    def record_failure(self) -> None:
        self._count += 1

    @property
    def count(self) -> int:
        return self._count

    def is_breached(self) -> bool:
        if self.max_consecutive_failures <= 0:
            return False
        return self._count >= self.max_consecutive_failures

    def reset(self) -> None:
        self._count = 0


@dataclass
class KillSwitchState:
    """Immutable snapshot of all kill switch checks."""

    halted: bool
    reason: str
    global_file_kill: bool = False
    global_env_kill: bool = False
    venue_kills: dict[str, bool] = field(default_factory=dict)
    daily_loss_breached: bool = False
    consecutive_failures_breached: bool = False
    safe_mode_active: bool = False


class KillSwitchManager:
    """Centralized operational risk control manager.

    Parameters
    ----------
    kill_switch_file:
        Path to sentinel file. If it exists, global halt is triggered.
    kill_switch_env_var:
        Env var name. If set to truthy value, global halt is triggered.
    venue_kill_switch_env_prefix:
        Env var prefix for per-venue kills.
        E.g. prefix "ARB_KILL_" → "ARB_KILL_KALSHI=1" halts kalshi.
    daily_loss_cap_usd:
        Max daily realized loss (positive number). 0 = disabled.
    max_consecutive_failures:
        Halt after this many consecutive trade failures. 0 = disabled.
    canary_mode:
        If True, bot is in canary (reduced activity) mode.
    canary_max_dollars_per_trade:
        Override max dollars per trade in canary mode.
    canary_max_contracts_per_trade:
        Override max contracts per trade in canary mode.
    """

    def __init__(
        self,
        kill_switch_file: str | Path = DEFAULT_KILL_SWITCH_FILE,
        kill_switch_env_var: str = "ARB_KILL_SWITCH",
        venue_kill_switch_env_prefix: str = "ARB_KILL_",
        daily_loss_cap_usd: float = 0.0,
        max_consecutive_failures: int = 0,
        canary_mode: bool = False,
        canary_max_dollars_per_trade: float = 10.0,
        canary_max_contracts_per_trade: int = 25,
    ) -> None:
        self._kill_switch_file = Path(kill_switch_file)
        self._kill_switch_env_var = kill_switch_env_var
        self._venue_kill_env_prefix = venue_kill_switch_env_prefix
        self._canary_mode = canary_mode
        self._canary_max_dollars_per_trade = canary_max_dollars_per_trade
        self._canary_max_contracts_per_trade = canary_max_contracts_per_trade
        self._safe_mode = False
        self._safe_mode_reason = ""

        self.daily_loss = DailyLossTracker(daily_loss_cap_usd=daily_loss_cap_usd)
        self.failures = ConsecutiveFailureTracker(
            max_consecutive_failures=max_consecutive_failures,
        )

    # ------------------------------------------------------------------
    # Safe mode (from crash recovery)
    # ------------------------------------------------------------------

    def activate_safe_mode(self, reason: str) -> None:
        self._safe_mode = True
        self._safe_mode_reason = reason
        LOGGER.warning("Safe mode activated: %s", reason)

    def deactivate_safe_mode(self) -> None:
        if self._safe_mode:
            LOGGER.info("Safe mode deactivated by operator")
        self._safe_mode = False
        self._safe_mode_reason = ""

    @property
    def safe_mode(self) -> bool:
        return self._safe_mode

    # ------------------------------------------------------------------
    # Canary mode
    # ------------------------------------------------------------------

    @property
    def canary_mode(self) -> bool:
        return self._canary_mode

    @property
    def canary_max_dollars_per_trade(self) -> float:
        return self._canary_max_dollars_per_trade

    @property
    def canary_max_contracts_per_trade(self) -> int:
        return self._canary_max_contracts_per_trade

    # ------------------------------------------------------------------
    # Core check
    # ------------------------------------------------------------------

    def check(self, venues: set[str] | None = None) -> KillSwitchState:
        """Run all kill switch checks and return the combined state.

        Parameters
        ----------
        venues:
            Set of venues involved in the pending trade. If None, only
            global checks are performed.
        """
        reasons: list[str] = []

        # 1. Global file kill switch
        global_file = self._kill_switch_file.exists()
        if global_file:
            reasons.append(f"kill switch file exists ({self._kill_switch_file})")

        # 2. Global env kill switch
        global_env = _is_truthy(os.environ.get(self._kill_switch_env_var))
        if global_env:
            reasons.append(f"kill switch env var {self._kill_switch_env_var} is set")

        # 3. Per-venue kill switches
        venue_kills: dict[str, bool] = {}
        for venue in (venues or set()):
            env_var = f"{self._venue_kill_env_prefix}{venue.upper()}"
            killed = _is_truthy(os.environ.get(env_var))
            venue_kills[venue] = killed
            if killed:
                reasons.append(f"venue kill switch active ({venue}, {env_var})")

        # 4. Daily loss cap
        daily_breached = self.daily_loss.is_breached()
        if daily_breached:
            reasons.append(
                f"daily loss cap breached (realized={self.daily_loss.realized_pnl:.2f}, "
                f"cap={self.daily_loss.daily_loss_cap_usd:.2f})"
            )

        # 5. Consecutive failures
        failures_breached = self.failures.is_breached()
        if failures_breached:
            reasons.append(
                f"consecutive failures breached ({self.failures.count} >= "
                f"{self.failures.max_consecutive_failures})"
            )

        # 6. Safe mode
        safe_mode = self._safe_mode
        if safe_mode:
            reasons.append(f"safe mode active ({self._safe_mode_reason})")

        halted = bool(reasons)
        reason = "; ".join(reasons) if reasons else "ok"

        if halted:
            LOGGER.warning("Kill switch HALT: %s", reason)

        return KillSwitchState(
            halted=halted,
            reason=reason,
            global_file_kill=global_file,
            global_env_kill=global_env,
            venue_kills=venue_kills,
            daily_loss_breached=daily_breached,
            consecutive_failures_breached=failures_breached,
            safe_mode_active=safe_mode,
        )

    def is_halted(self, venues: set[str] | None = None) -> bool:
        """Quick boolean check — is trading halted?"""
        return self.check(venues).halted


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}
