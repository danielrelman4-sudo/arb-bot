"""Tests for Phase 5K: Operational failure handling playbooks."""

from __future__ import annotations

import pytest

from arb_bot.framework.failure_playbook import (
    Playbook,
    PlaybookEvent,
    PlaybookManager,
    PlaybookManagerConfig,
    PlaybookStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mgr(**kw) -> PlaybookManager:
    return PlaybookManager(PlaybookManagerConfig(**kw))


def _rate_limit_playbook(
    threshold: int = 10,
    cooldown: float = 60.0,
    max_duration: float = 300.0,
) -> Playbook:
    return Playbook(
        name="rate_limit_storm",
        trigger_fn=lambda m: m.get("total_429s", 0) >= threshold,
        actions=["pause_polling", "increase_intervals"],
        recovery_fn=lambda m: m.get("total_429s", 0) == 0,
        severity=3,
        cooldown=cooldown,
        max_duration=max_duration,
    )


def _disconnect_playbook() -> Playbook:
    return Playbook(
        name="stream_disconnect",
        trigger_fn=lambda m: m.get("disconnects", 0) >= 3,
        actions=["switch_to_poll", "notify"],
        recovery_fn=lambda m: m.get("stream_connected", False),
        severity=2,
        cooldown=120.0,
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self) -> None:
        cfg = PlaybookManagerConfig()
        assert cfg.evaluation_interval == 5.0
        assert cfg.max_active_playbooks == 3
        assert cfg.default_cooldown == 300.0

    def test_frozen(self) -> None:
        cfg = PlaybookManagerConfig()
        with pytest.raises(AttributeError):
            cfg.max_active_playbooks = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------


class TestRegister:
    def test_register(self) -> None:
        mgr = _mgr()
        mgr.register_playbook(_rate_limit_playbook())
        assert mgr.playbook_count() == 1

    def test_unregister(self) -> None:
        mgr = _mgr()
        mgr.register_playbook(_rate_limit_playbook())
        mgr.unregister_playbook("rate_limit_storm")
        assert mgr.playbook_count() == 0


# ---------------------------------------------------------------------------
# Trigger
# ---------------------------------------------------------------------------


class TestTrigger:
    def test_triggers_when_condition_met(self) -> None:
        mgr = _mgr()
        mgr.register_playbook(_rate_limit_playbook(threshold=5))
        events = mgr.evaluate({"total_429s": 10}, now=100.0)
        assert len(events) == 1
        assert events[0].event_type == "triggered"
        assert events[0].playbook_name == "rate_limit_storm"
        assert "pause_polling" in events[0].actions

    def test_no_trigger_below_threshold(self) -> None:
        mgr = _mgr()
        mgr.register_playbook(_rate_limit_playbook(threshold=10))
        events = mgr.evaluate({"total_429s": 5}, now=100.0)
        assert len(events) == 0

    def test_trigger_sets_status(self) -> None:
        mgr = _mgr()
        mgr.register_playbook(_rate_limit_playbook(threshold=5))
        mgr.evaluate({"total_429s": 10}, now=100.0)
        state = mgr.get_state("rate_limit_storm")
        assert state is not None
        assert state.status == PlaybookStatus.TRIGGERED

    def test_trigger_count_increments(self) -> None:
        mgr = _mgr()
        pb = _rate_limit_playbook(threshold=5, cooldown=0.0)
        mgr.register_playbook(pb)
        mgr.evaluate({"total_429s": 10}, now=100.0)
        # Recover.
        mgr.evaluate({"total_429s": 0}, now=101.0)
        mgr.evaluate({"total_429s": 0}, now=102.0)  # cooldown → idle.
        mgr.evaluate({"total_429s": 0}, now=103.0)  # Now idle.
        # Re-trigger.
        mgr.evaluate({"total_429s": 10}, now=104.0)
        state = mgr.get_state("rate_limit_storm")
        assert state is not None
        assert state.trigger_count == 2


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------


class TestRecovery:
    def test_recovers_when_condition_met(self) -> None:
        mgr = _mgr()
        mgr.register_playbook(_rate_limit_playbook(threshold=5))
        mgr.evaluate({"total_429s": 10}, now=100.0)
        events = mgr.evaluate({"total_429s": 0}, now=110.0)
        assert any(e.event_type == "recovered" for e in events)

    def test_stays_active_until_recovery(self) -> None:
        mgr = _mgr()
        mgr.register_playbook(_rate_limit_playbook(threshold=5))
        mgr.evaluate({"total_429s": 10}, now=100.0)
        mgr.evaluate({"total_429s": 8}, now=110.0)  # Still 429s.
        state = mgr.get_state("rate_limit_storm")
        assert state is not None
        assert state.status == PlaybookStatus.ACTIVE


# ---------------------------------------------------------------------------
# Expiration (max duration)
# ---------------------------------------------------------------------------


class TestExpiration:
    def test_expires_after_max_duration(self) -> None:
        mgr = _mgr()
        mgr.register_playbook(
            _rate_limit_playbook(threshold=5, max_duration=60.0)
        )
        mgr.evaluate({"total_429s": 10}, now=100.0)
        events = mgr.evaluate({"total_429s": 10}, now=200.0)  # 100s > 60s.
        assert any(e.event_type == "expired" for e in events)


# ---------------------------------------------------------------------------
# Cooldown
# ---------------------------------------------------------------------------


class TestCooldown:
    def test_enters_cooldown_after_recovery(self) -> None:
        mgr = _mgr()
        mgr.register_playbook(_rate_limit_playbook(threshold=5, cooldown=60.0))
        mgr.evaluate({"total_429s": 10}, now=100.0)
        mgr.evaluate({"total_429s": 0}, now=110.0)  # Recover.
        mgr.evaluate({"total_429s": 0}, now=111.0)  # Recovering → cooldown.
        state = mgr.get_state("rate_limit_storm")
        assert state is not None
        assert state.status == PlaybookStatus.COOLDOWN

    def test_returns_idle_after_cooldown(self) -> None:
        mgr = _mgr()
        mgr.register_playbook(_rate_limit_playbook(threshold=5, cooldown=60.0))
        mgr.evaluate({"total_429s": 10}, now=100.0)
        mgr.evaluate({"total_429s": 0}, now=110.0)  # Recover.
        mgr.evaluate({"total_429s": 0}, now=111.0)  # → cooldown.
        mgr.evaluate({"total_429s": 0}, now=200.0)  # 90s > 60s cooldown.
        state = mgr.get_state("rate_limit_storm")
        assert state is not None
        assert state.status == PlaybookStatus.IDLE

    def test_no_retrigger_during_cooldown(self) -> None:
        mgr = _mgr()
        mgr.register_playbook(_rate_limit_playbook(threshold=5, cooldown=60.0))
        mgr.evaluate({"total_429s": 10}, now=100.0)
        mgr.evaluate({"total_429s": 0}, now=110.0)
        mgr.evaluate({"total_429s": 0}, now=111.0)
        # Still in cooldown — should not trigger.
        events = mgr.evaluate({"total_429s": 10}, now=120.0)
        triggered = [e for e in events if e.event_type == "triggered"]
        assert len(triggered) == 0


# ---------------------------------------------------------------------------
# Max active playbooks
# ---------------------------------------------------------------------------


class TestMaxActive:
    def test_limits_concurrent(self) -> None:
        mgr = _mgr(max_active_playbooks=1)
        mgr.register_playbook(_rate_limit_playbook(threshold=5))
        mgr.register_playbook(_disconnect_playbook())
        events = mgr.evaluate(
            {"total_429s": 10, "disconnects": 5},
            now=100.0,
        )
        triggered = [e for e in events if e.event_type == "triggered"]
        assert len(triggered) == 1  # Only 1 allowed.


# ---------------------------------------------------------------------------
# Active playbooks
# ---------------------------------------------------------------------------


class TestActivePlaybooks:
    def test_lists_active(self) -> None:
        mgr = _mgr()
        mgr.register_playbook(_rate_limit_playbook(threshold=5))
        mgr.evaluate({"total_429s": 10}, now=100.0)
        assert "rate_limit_storm" in mgr.active_playbooks()

    def test_empty_when_idle(self) -> None:
        mgr = _mgr()
        mgr.register_playbook(_rate_limit_playbook(threshold=5))
        assert mgr.active_playbooks() == []


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_trigger_fn_exception(self) -> None:
        def bad_trigger(m):
            raise RuntimeError("oops")

        pb = Playbook(
            name="bad",
            trigger_fn=bad_trigger,
            actions=["noop"],
            recovery_fn=lambda m: True,
        )
        mgr = _mgr()
        mgr.register_playbook(pb)
        events = mgr.evaluate({}, now=100.0)
        assert len(events) == 0  # No crash.


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clears_all(self) -> None:
        mgr = _mgr()
        mgr.register_playbook(_rate_limit_playbook())
        mgr.clear()
        assert mgr.playbook_count() == 0


# ---------------------------------------------------------------------------
# Config property
# ---------------------------------------------------------------------------


class TestConfigProperty:
    def test_config_accessible(self) -> None:
        cfg = PlaybookManagerConfig(max_active_playbooks=5)
        mgr = PlaybookManager(cfg)
        assert mgr.config.max_active_playbooks == 5


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_429_storm_lifecycle(self) -> None:
        """Full 429 storm: trigger → active → recover → cooldown → idle."""
        mgr = _mgr()
        mgr.register_playbook(
            _rate_limit_playbook(threshold=5, cooldown=30.0, max_duration=120.0)
        )

        # Trigger.
        events = mgr.evaluate({"total_429s": 10}, now=100.0)
        assert len(events) == 1
        assert events[0].event_type == "triggered"

        # Still active.
        events = mgr.evaluate({"total_429s": 8}, now=110.0)
        assert mgr.active_playbooks() == ["rate_limit_storm"]

        # Recovery.
        events = mgr.evaluate({"total_429s": 0}, now=120.0)
        assert any(e.event_type == "recovered" for e in events)

        # Recovering → cooldown.
        mgr.evaluate({"total_429s": 0}, now=121.0)
        state = mgr.get_state("rate_limit_storm")
        assert state is not None
        assert state.status == PlaybookStatus.COOLDOWN

        # Cooldown expires.
        mgr.evaluate({"total_429s": 0}, now=160.0)
        assert state.status == PlaybookStatus.IDLE
