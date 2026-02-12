"""Delta/snapshot reconciliation loop (Phase 2C).

Periodically compares stream-derived quote state against REST poll
snapshots to detect and correct drift. The reconciler maintains a
per-market snapshot store from the latest REST poll, compares incoming
stream quotes against it, and flags divergence when price or size
differences exceed configurable thresholds.

When divergence is detected, the reconciler can force a resync for
the affected market(s) by requesting a targeted REST poll. This
catches issues like missed WebSocket messages, stale stream state,
and silent data corruption.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from arb_bot.models import BinaryQuote

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReconciliationConfig:
    """Configuration for delta/snapshot reconciliation.

    Parameters
    ----------
    price_divergence_threshold:
        Max absolute price difference (yes or no) between stream and
        snapshot before flagging divergence. Default 0.05.
    size_divergence_ratio:
        Max relative size difference between stream and snapshot before
        flagging divergence. Computed as |stream-snap|/max(stream,snap).
        Default 0.5 (50%).
    reconciliation_interval_seconds:
        How often (in seconds) to run a reconciliation check. Default 60.0.
    max_snapshot_age_seconds:
        Discard snapshot if older than this. A stale snapshot shouldn't
        be compared against fresh stream data. Default 120.0.
    divergence_count_for_resync:
        Number of consecutive divergent checks for a market before
        requesting a targeted resync. Default 2.
    enabled:
        Master enable/disable. Default True.
    """

    price_divergence_threshold: float = 0.05
    size_divergence_ratio: float = 0.5
    reconciliation_interval_seconds: float = 60.0
    max_snapshot_age_seconds: float = 120.0
    divergence_count_for_resync: int = 2
    enabled: bool = True


@dataclass(frozen=True)
class DivergenceRecord:
    """Record of a single divergence detection."""

    market_key: str
    field_name: str
    stream_value: float
    snapshot_value: float
    delta: float
    detected_at: datetime


@dataclass
class ReconciliationStats:
    """Tracks reconciliation statistics."""

    total_checks: int = 0
    total_divergences: int = 0
    resyncs_triggered: int = 0
    markets_checked: int = 0
    last_check_at: Optional[datetime] = None

    def reset(self) -> None:
        self.total_checks = 0
        self.total_divergences = 0
        self.resyncs_triggered = 0
        self.markets_checked = 0
        self.last_check_at = None


@dataclass
class _MarketSnapshot:
    """Internal: snapshot of a market from REST poll."""

    quote: BinaryQuote
    stored_at: datetime
    consecutive_divergences: int = 0


class ReconciliationLoop:
    """Compares stream-derived quotes against REST poll snapshots.

    Usage::

        reconciler = ReconciliationLoop(config)

        # After a REST poll, store the snapshots:
        reconciler.update_snapshots(poll_quotes)

        # Periodically, or after stream updates:
        result = reconciler.check(stream_cache, now=now)
        if result.markets_needing_resync:
            # Trigger targeted REST poll for those markets
            ...
    """

    def __init__(self, config: ReconciliationConfig | None = None) -> None:
        self._config = config or ReconciliationConfig()
        self._snapshots: Dict[tuple[str, str], _MarketSnapshot] = {}
        self._stats = ReconciliationStats()

    @property
    def config(self) -> ReconciliationConfig:
        return self._config

    @property
    def stats(self) -> ReconciliationStats:
        return self._stats

    def update_snapshots(
        self,
        quotes: list[BinaryQuote],
        now: datetime | None = None,
    ) -> int:
        """Store REST poll quotes as the ground-truth snapshot.

        Returns the number of snapshots updated.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        count = 0
        for quote in quotes:
            key = (quote.venue, quote.market_id)
            self._snapshots[key] = _MarketSnapshot(
                quote=quote,
                stored_at=now,
                consecutive_divergences=0,
            )
            count += 1
        return count

    def check(
        self,
        stream_quotes: Dict[tuple[str, str], BinaryQuote],
        now: datetime | None = None,
    ) -> ReconciliationResult:
        """Compare stream quotes against stored snapshots.

        Parameters
        ----------
        stream_quotes:
            Dict of current stream-derived quotes, keyed by (venue, market_id).
        now:
            Current time. Defaults to utcnow.

        Returns
        -------
        ReconciliationResult with divergences and markets needing resync.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        self._stats.total_checks += 1
        self._stats.last_check_at = now

        divergences: list[DivergenceRecord] = []
        markets_needing_resync: list[tuple[str, str]] = []
        markets_checked = 0

        for key, snap in list(self._snapshots.items()):
            stream_quote = stream_quotes.get(key)
            if stream_quote is None:
                continue

            # Skip stale snapshots.
            snap_age = (now - snap.stored_at).total_seconds()
            if snap_age > self._config.max_snapshot_age_seconds:
                continue

            markets_checked += 1
            market_divergences = self._compare(
                stream_quote, snap.quote, now
            )

            if market_divergences:
                divergences.extend(market_divergences)
                snap.consecutive_divergences += 1
                self._stats.total_divergences += len(market_divergences)

                market_key = f"{key[0]}/{key[1]}"
                LOGGER.warning(
                    "Reconciliation divergence for %s "
                    "(consecutive=%d): %s",
                    market_key,
                    snap.consecutive_divergences,
                    "; ".join(
                        f"{d.field_name}: stream={d.stream_value:.4f} "
                        f"snap={d.snapshot_value:.4f} delta={d.delta:.4f}"
                        for d in market_divergences
                    ),
                )

                if snap.consecutive_divergences >= self._config.divergence_count_for_resync:
                    markets_needing_resync.append(key)
                    self._stats.resyncs_triggered += 1
                    LOGGER.warning(
                        "Requesting resync for %s after %d consecutive divergences",
                        market_key,
                        snap.consecutive_divergences,
                    )
            else:
                # Reset consecutive divergence counter on match.
                snap.consecutive_divergences = 0

        self._stats.markets_checked += markets_checked

        return ReconciliationResult(
            divergences=tuple(divergences),
            markets_needing_resync=tuple(markets_needing_resync),
            markets_checked=markets_checked,
            snapshot_count=len(self._snapshots),
        )

    def _compare(
        self,
        stream: BinaryQuote,
        snapshot: BinaryQuote,
        now: datetime,
    ) -> list[DivergenceRecord]:
        """Compare a stream quote against a snapshot quote."""
        divergences: list[DivergenceRecord] = []
        market_key = f"{stream.venue}/{stream.market_id}"
        cfg = self._config

        # Price divergence checks.
        for field_name, stream_val, snap_val in [
            ("yes_buy_price", stream.yes_buy_price, snapshot.yes_buy_price),
            ("no_buy_price", stream.no_buy_price, snapshot.no_buy_price),
        ]:
            delta = abs(stream_val - snap_val)
            if delta > cfg.price_divergence_threshold:
                divergences.append(DivergenceRecord(
                    market_key=market_key,
                    field_name=field_name,
                    stream_value=stream_val,
                    snapshot_value=snap_val,
                    delta=delta,
                    detected_at=now,
                ))

        # Size divergence checks.
        for field_name, stream_val, snap_val in [
            ("yes_buy_size", stream.yes_buy_size, snapshot.yes_buy_size),
            ("no_buy_size", stream.no_buy_size, snapshot.no_buy_size),
        ]:
            denominator = max(stream_val, snap_val)
            if denominator > 0:
                ratio = abs(stream_val - snap_val) / denominator
                if ratio > cfg.size_divergence_ratio:
                    divergences.append(DivergenceRecord(
                        market_key=market_key,
                        field_name=field_name,
                        stream_value=stream_val,
                        snapshot_value=snap_val,
                        delta=ratio,
                        detected_at=now,
                    ))

        return divergences

    def clear_snapshot(self, venue: str, market_id: str) -> bool:
        """Remove a specific market snapshot. Returns True if found."""
        return self._snapshots.pop((venue, market_id), None) is not None

    def clear_all_snapshots(self) -> None:
        """Remove all snapshots (e.g., after reconnect)."""
        self._snapshots.clear()

    def snapshot_count(self) -> int:
        return len(self._snapshots)

    def has_snapshot(self, venue: str, market_id: str) -> bool:
        return (venue, market_id) in self._snapshots

    def is_due(self, now: datetime | None = None) -> bool:
        """Check if a reconciliation check is due based on interval."""
        if not self._config.enabled:
            return False
        if now is None:
            now = datetime.now(timezone.utc)
        if self._stats.last_check_at is None:
            return True
        elapsed = (now - self._stats.last_check_at).total_seconds()
        return elapsed >= self._config.reconciliation_interval_seconds

    def reset_divergence_count(self, venue: str, market_id: str) -> None:
        """Reset consecutive divergence counter after a successful resync."""
        snap = self._snapshots.get((venue, market_id))
        if snap is not None:
            snap.consecutive_divergences = 0


@dataclass(frozen=True)
class ReconciliationResult:
    """Result of a reconciliation check."""

    divergences: tuple[DivergenceRecord, ...]
    markets_needing_resync: tuple[tuple[str, str], ...]
    markets_checked: int
    snapshot_count: int

    @property
    def has_divergence(self) -> bool:
        return len(self.divergences) > 0

    @property
    def resync_needed(self) -> bool:
        return len(self.markets_needing_resync) > 0

    @property
    def summary(self) -> str:
        if not self.has_divergence:
            return f"OK: {self.markets_checked} markets checked, no divergence"
        resync_part = ""
        if self.resync_needed:
            keys = [f"{v}/{m}" for v, m in self.markets_needing_resync]
            resync_part = f", resync needed: {', '.join(keys)}"
        return (
            f"{len(self.divergences)} divergence(s) in "
            f"{self.markets_checked} markets{resync_part}"
        )
