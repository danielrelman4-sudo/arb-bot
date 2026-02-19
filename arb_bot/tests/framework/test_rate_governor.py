"""Tests for Phase 5A: Adaptive per-endpoint API rate governor."""

from __future__ import annotations

import pytest

from arb_bot.framework.rate_governor import (
    EndpointState,
    GovernorSnapshot,
    RateGovernor,
    RateGovernorConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gov(**kw) -> RateGovernor:
    return RateGovernor(RateGovernorConfig(**kw))


VENUE = "kalshi"
EP = "market_orderbook"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self) -> None:
        cfg = RateGovernorConfig()
        assert cfg.base_interval == 0.10
        assert cfg.min_interval == 0.01
        assert cfg.max_interval == 10.0
        assert cfg.aimd_increase_factor == 0.95
        assert cfg.aimd_429_factor == 1.75
        assert cfg.aimd_5xx_factor == 1.25
        assert cfg.aimd_error_factor == 1.15
        assert cfg.success_streak_threshold == 8
        assert cfg.bucket_capacity == 5
        assert cfg.bucket_refill_rate == 0.0
        assert cfg.retry_after_max == 120.0

    def test_frozen(self) -> None:
        cfg = RateGovernorConfig()
        with pytest.raises(AttributeError):
            cfg.base_interval = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


class TestInitialState:
    def test_no_state_initially(self) -> None:
        gov = _gov()
        assert gov.get_state(VENUE, EP) is None

    def test_get_interval_default(self) -> None:
        gov = _gov(base_interval=0.20)
        assert gov.get_interval(VENUE, EP) == 0.20

    def test_state_created_on_acquire(self) -> None:
        gov = _gov()
        gov.acquire_wait_time(VENUE, EP, now=100.0)
        state = gov.get_state(VENUE, EP)
        assert state is not None
        assert state.venue == VENUE
        assert state.endpoint == EP

    def test_state_created_on_consume(self) -> None:
        gov = _gov()
        gov.consume(VENUE, EP, now=100.0)
        state = gov.get_state(VENUE, EP)
        assert state is not None
        assert state.total_requests == 1


# ---------------------------------------------------------------------------
# Acquire wait time
# ---------------------------------------------------------------------------


class TestAcquireWaitTime:
    def test_first_request_immediate(self) -> None:
        gov = _gov(bucket_capacity=5)
        wait = gov.acquire_wait_time(VENUE, EP, now=100.0)
        assert wait == 0.0

    def test_wait_after_consume(self) -> None:
        gov = _gov(base_interval=1.0, bucket_capacity=0)
        gov.consume(VENUE, EP, now=100.0)
        wait = gov.acquire_wait_time(VENUE, EP, now=100.5)
        assert wait == pytest.approx(0.5, abs=0.01)

    def test_no_wait_after_interval(self) -> None:
        gov = _gov(base_interval=1.0, bucket_capacity=0)
        gov.consume(VENUE, EP, now=100.0)
        wait = gov.acquire_wait_time(VENUE, EP, now=101.5)
        assert wait == 0.0

    def test_token_bucket_burst(self) -> None:
        """Multiple requests allowed via token bucket burst."""
        gov = _gov(base_interval=1.0, bucket_capacity=3)
        # First request — tokens available.
        gov.acquire_wait_time(VENUE, EP, now=100.0)
        gov.consume(VENUE, EP, now=100.0)
        # Second request — still have tokens.
        wait = gov.acquire_wait_time(VENUE, EP, now=100.0)
        assert wait <= 1.0  # Should be small due to burst tokens.

    def test_no_bucket_pure_interval(self) -> None:
        gov = _gov(base_interval=0.5, bucket_capacity=0)
        gov.consume(VENUE, EP, now=100.0)
        wait = gov.acquire_wait_time(VENUE, EP, now=100.2)
        assert wait == pytest.approx(0.3, abs=0.01)


# ---------------------------------------------------------------------------
# Consume
# ---------------------------------------------------------------------------


class TestConsume:
    def test_increments_request_count(self) -> None:
        gov = _gov()
        gov.consume(VENUE, EP, now=100.0)
        gov.consume(VENUE, EP, now=101.0)
        state = gov.get_state(VENUE, EP)
        assert state is not None
        assert state.total_requests == 2

    def test_deducts_token(self) -> None:
        gov = _gov(bucket_capacity=5)
        gov.acquire_wait_time(VENUE, EP, now=100.0)  # Creates state, refills.
        state = gov.get_state(VENUE, EP)
        assert state is not None
        tokens_before = state.tokens
        gov.consume(VENUE, EP, now=100.0)
        assert state.tokens == pytest.approx(tokens_before - 1.0, abs=0.01)

    def test_sets_next_request_time(self) -> None:
        gov = _gov(base_interval=0.5)
        gov.consume(VENUE, EP, now=100.0)
        state = gov.get_state(VENUE, EP)
        assert state is not None
        assert state.next_request_time == pytest.approx(100.5)


# ---------------------------------------------------------------------------
# Record response — success
# ---------------------------------------------------------------------------


class TestRecordSuccess:
    def test_increments_streak(self) -> None:
        gov = _gov()
        gov.consume(VENUE, EP, now=100.0)
        gov.record_response(VENUE, EP, status_code=200)
        state = gov.get_state(VENUE, EP)
        assert state is not None
        assert state.success_streak == 1

    def test_interval_decreases_after_streak(self) -> None:
        gov = _gov(
            base_interval=1.0,
            success_streak_threshold=3,
            aimd_increase_factor=0.90,
        )
        gov.consume(VENUE, EP, now=100.0)
        for _ in range(3):
            gov.record_response(VENUE, EP, status_code=200)
        state = gov.get_state(VENUE, EP)
        assert state is not None
        assert state.interval == pytest.approx(0.90)
        assert state.success_streak == 0  # Reset after reduction.

    def test_interval_floored_at_min(self) -> None:
        gov = _gov(
            base_interval=0.02,
            min_interval=0.01,
            success_streak_threshold=1,
            aimd_increase_factor=0.10,
        )
        gov.consume(VENUE, EP, now=100.0)
        gov.record_response(VENUE, EP, status_code=200)
        state = gov.get_state(VENUE, EP)
        assert state is not None
        assert state.interval >= 0.01


# ---------------------------------------------------------------------------
# Record response — 429
# ---------------------------------------------------------------------------


class TestRecord429:
    def test_interval_increases_on_429(self) -> None:
        gov = _gov(base_interval=0.10, aimd_429_factor=1.75)
        gov.consume(VENUE, EP, now=100.0)
        gov.record_response(VENUE, EP, status_code=429)
        state = gov.get_state(VENUE, EP)
        assert state is not None
        assert state.interval == pytest.approx(0.175)

    def test_429_resets_streak(self) -> None:
        gov = _gov()
        gov.consume(VENUE, EP, now=100.0)
        gov.record_response(VENUE, EP, status_code=200)
        gov.record_response(VENUE, EP, status_code=200)
        gov.record_response(VENUE, EP, status_code=429)
        state = gov.get_state(VENUE, EP)
        assert state is not None
        assert state.success_streak == 0

    def test_429_counter(self) -> None:
        gov = _gov()
        gov.consume(VENUE, EP, now=100.0)
        gov.record_response(VENUE, EP, status_code=429)
        gov.record_response(VENUE, EP, status_code=429)
        state = gov.get_state(VENUE, EP)
        assert state is not None
        assert state.total_429s == 2

    def test_retry_after_honored(self) -> None:
        gov = _gov(base_interval=0.10, aimd_429_factor=1.75)
        gov.consume(VENUE, EP, now=100.0)
        gov.record_response(VENUE, EP, status_code=429, retry_after=5.0)
        state = gov.get_state(VENUE, EP)
        assert state is not None
        assert state.interval >= 5.0
        assert state.last_retry_after == 5.0

    def test_retry_after_capped(self) -> None:
        gov = _gov(retry_after_max=10.0)
        gov.consume(VENUE, EP, now=100.0)
        gov.record_response(VENUE, EP, status_code=429, retry_after=300.0)
        state = gov.get_state(VENUE, EP)
        assert state is not None
        assert state.interval <= 10.0

    def test_429_interval_capped_at_max(self) -> None:
        gov = _gov(base_interval=5.0, max_interval=10.0, aimd_429_factor=3.0)
        gov.consume(VENUE, EP, now=100.0)
        gov.record_response(VENUE, EP, status_code=429)
        state = gov.get_state(VENUE, EP)
        assert state is not None
        assert state.interval <= 10.0


# ---------------------------------------------------------------------------
# Record response — 5xx
# ---------------------------------------------------------------------------


class TestRecord5xx:
    def test_interval_increases_on_5xx(self) -> None:
        gov = _gov(base_interval=1.0, aimd_5xx_factor=1.25)
        gov.consume(VENUE, EP, now=100.0)
        gov.record_response(VENUE, EP, status_code=500)
        state = gov.get_state(VENUE, EP)
        assert state is not None
        assert state.interval == pytest.approx(1.25)

    def test_5xx_counter(self) -> None:
        gov = _gov()
        gov.consume(VENUE, EP, now=100.0)
        gov.record_response(VENUE, EP, status_code=502)
        gov.record_response(VENUE, EP, status_code=503)
        state = gov.get_state(VENUE, EP)
        assert state is not None
        assert state.total_5xxs == 2

    def test_5xx_resets_streak(self) -> None:
        gov = _gov()
        gov.consume(VENUE, EP, now=100.0)
        gov.record_response(VENUE, EP, status_code=200)
        gov.record_response(VENUE, EP, status_code=500)
        state = gov.get_state(VENUE, EP)
        assert state is not None
        assert state.success_streak == 0

    def test_5xx_capped_at_max(self) -> None:
        gov = _gov(base_interval=8.0, max_interval=10.0, aimd_5xx_factor=2.0)
        gov.consume(VENUE, EP, now=100.0)
        gov.record_response(VENUE, EP, status_code=500)
        state = gov.get_state(VENUE, EP)
        assert state is not None
        assert state.interval <= 10.0


# ---------------------------------------------------------------------------
# Record response — 4xx (non-429)
# ---------------------------------------------------------------------------


class TestRecord4xx:
    def test_interval_increases_on_4xx(self) -> None:
        gov = _gov(base_interval=1.0, aimd_error_factor=1.15)
        gov.consume(VENUE, EP, now=100.0)
        gov.record_response(VENUE, EP, status_code=400)
        state = gov.get_state(VENUE, EP)
        assert state is not None
        assert state.interval == pytest.approx(1.15)

    def test_error_counter(self) -> None:
        gov = _gov()
        gov.consume(VENUE, EP, now=100.0)
        gov.record_response(VENUE, EP, status_code=403)
        state = gov.get_state(VENUE, EP)
        assert state is not None
        assert state.total_errors == 1

    def test_4xx_resets_streak(self) -> None:
        gov = _gov()
        gov.consume(VENUE, EP, now=100.0)
        gov.record_response(VENUE, EP, status_code=200)
        gov.record_response(VENUE, EP, status_code=404)
        state = gov.get_state(VENUE, EP)
        assert state is not None
        assert state.success_streak == 0


# ---------------------------------------------------------------------------
# Record error (non-HTTP)
# ---------------------------------------------------------------------------


class TestRecordError:
    def test_interval_increases(self) -> None:
        gov = _gov(base_interval=1.0, aimd_error_factor=1.15)
        gov.consume(VENUE, EP, now=100.0)
        gov.record_error(VENUE, EP)
        state = gov.get_state(VENUE, EP)
        assert state is not None
        assert state.interval == pytest.approx(1.15)

    def test_error_counter(self) -> None:
        gov = _gov()
        gov.consume(VENUE, EP, now=100.0)
        gov.record_error(VENUE, EP)
        gov.record_error(VENUE, EP)
        state = gov.get_state(VENUE, EP)
        assert state is not None
        assert state.total_errors == 2

    def test_resets_streak(self) -> None:
        gov = _gov()
        gov.consume(VENUE, EP, now=100.0)
        gov.record_response(VENUE, EP, status_code=200)
        gov.record_error(VENUE, EP)
        state = gov.get_state(VENUE, EP)
        assert state is not None
        assert state.success_streak == 0

    def test_capped_at_max(self) -> None:
        gov = _gov(base_interval=8.0, max_interval=10.0, aimd_error_factor=2.0)
        gov.consume(VENUE, EP, now=100.0)
        gov.record_error(VENUE, EP)
        state = gov.get_state(VENUE, EP)
        assert state is not None
        assert state.interval <= 10.0


# ---------------------------------------------------------------------------
# Token bucket
# ---------------------------------------------------------------------------


class TestTokenBucket:
    def test_initial_full(self) -> None:
        gov = _gov(bucket_capacity=5)
        gov.acquire_wait_time(VENUE, EP, now=100.0)
        state = gov.get_state(VENUE, EP)
        assert state is not None
        assert state.tokens == pytest.approx(5.0)

    def test_refills_over_time(self) -> None:
        gov = _gov(bucket_capacity=5, base_interval=1.0)
        # Use all tokens.
        for i in range(5):
            gov.acquire_wait_time(VENUE, EP, now=100.0)
            gov.consume(VENUE, EP, now=100.0)
        state = gov.get_state(VENUE, EP)
        assert state is not None
        assert state.tokens == pytest.approx(0.0)

        # Wait 3 seconds → refill rate = 1/interval = 1.0 tokens/sec.
        wait = gov.acquire_wait_time(VENUE, EP, now=103.0)
        assert state.tokens == pytest.approx(3.0)

    def test_capacity_capped(self) -> None:
        gov = _gov(bucket_capacity=3, base_interval=0.5)
        # Wait a very long time — should cap at capacity.
        gov.acquire_wait_time(VENUE, EP, now=100.0)
        gov.acquire_wait_time(VENUE, EP, now=1000.0)
        state = gov.get_state(VENUE, EP)
        assert state is not None
        assert state.tokens <= 3.0

    def test_no_bucket_mode(self) -> None:
        gov = _gov(bucket_capacity=0)
        gov.consume(VENUE, EP, now=100.0)
        state = gov.get_state(VENUE, EP)
        assert state is not None
        # No bucket — tokens not used.
        assert state.tokens == 0.0

    def test_custom_refill_rate(self) -> None:
        gov = _gov(bucket_capacity=5, bucket_refill_rate=2.0, base_interval=1.0)
        # Use all tokens.
        for i in range(5):
            gov.acquire_wait_time(VENUE, EP, now=100.0)
            gov.consume(VENUE, EP, now=100.0)
        state = gov.get_state(VENUE, EP)
        assert state is not None
        assert state.tokens == pytest.approx(0.0)

        # Wait 1 second → refill at 2.0 tokens/sec = 2 tokens.
        gov.acquire_wait_time(VENUE, EP, now=101.0)
        assert state.tokens == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Multi-endpoint independence
# ---------------------------------------------------------------------------


class TestMultiEndpoint:
    def test_independent_state(self) -> None:
        gov = _gov(base_interval=1.0)
        gov.consume("kalshi", "orderbook", now=100.0)
        gov.consume("kalshi", "markets", now=100.0)

        state_ob = gov.get_state("kalshi", "orderbook")
        state_mk = gov.get_state("kalshi", "markets")
        assert state_ob is not None
        assert state_mk is not None
        assert state_ob is not state_mk

    def test_independent_intervals(self) -> None:
        gov = _gov(base_interval=1.0, aimd_429_factor=2.0)
        gov.consume("kalshi", "orderbook", now=100.0)
        gov.consume("kalshi", "markets", now=100.0)

        # 429 on orderbook only.
        gov.record_response("kalshi", "orderbook", status_code=429)
        gov.record_response("kalshi", "markets", status_code=200)

        assert gov.get_interval("kalshi", "orderbook") == pytest.approx(2.0)
        assert gov.get_interval("kalshi", "markets") == pytest.approx(1.0)

    def test_cross_venue(self) -> None:
        gov = _gov()
        gov.consume("kalshi", "orderbook", now=100.0)
        gov.consume("polymarket", "orderbook", now=100.0)

        state_k = gov.get_state("kalshi", "orderbook")
        state_p = gov.get_state("polymarket", "orderbook")
        assert state_k is not None
        assert state_p is not None
        assert state_k is not state_p


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


class TestSnapshot:
    def test_empty_snapshot(self) -> None:
        gov = _gov()
        snap = gov.snapshot()
        assert snap.endpoint_count == 0
        assert snap.total_requests == 0
        assert snap.total_429s == 0
        assert snap.total_5xxs == 0
        assert snap.busiest_endpoint == ""

    def test_populated_snapshot(self) -> None:
        gov = _gov()
        # Make some requests.
        gov.consume("kalshi", "orderbook", now=100.0)
        gov.consume("kalshi", "orderbook", now=101.0)
        gov.consume("kalshi", "markets", now=100.0)
        gov.record_response("kalshi", "orderbook", status_code=429)
        gov.record_response("kalshi", "markets", status_code=500)

        snap = gov.snapshot()
        assert snap.endpoint_count == 2
        assert snap.total_requests == 3
        assert snap.total_429s == 1
        assert snap.total_5xxs == 1
        assert snap.busiest_endpoint == "kalshi:orderbook"

    def test_slowest_fastest(self) -> None:
        gov = _gov(base_interval=1.0, aimd_429_factor=2.0)
        gov.consume("kalshi", "orderbook", now=100.0)
        gov.consume("kalshi", "markets", now=100.0)
        gov.record_response("kalshi", "orderbook", status_code=429)

        snap = gov.snapshot()
        assert snap.slowest_interval == pytest.approx(2.0)
        assert snap.fastest_interval == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clears_all(self) -> None:
        gov = _gov()
        gov.consume(VENUE, EP, now=100.0)
        gov.clear()
        assert gov.get_state(VENUE, EP) is None
        snap = gov.snapshot()
        assert snap.endpoint_count == 0


# ---------------------------------------------------------------------------
# Config property
# ---------------------------------------------------------------------------


class TestConfigProperty:
    def test_config_accessible(self) -> None:
        cfg = RateGovernorConfig(base_interval=0.50)
        gov = RateGovernor(cfg)
        assert gov.config.base_interval == 0.50


# ---------------------------------------------------------------------------
# AIMD recovery cycle
# ---------------------------------------------------------------------------


class TestAIMDCycle:
    def test_backoff_and_recovery(self) -> None:
        """Simulate: 429 → backoff → success streak → recovery."""
        gov = _gov(
            base_interval=0.10,
            aimd_429_factor=2.0,
            aimd_increase_factor=0.90,
            success_streak_threshold=3,
            bucket_capacity=0,
        )
        gov.consume(VENUE, EP, now=100.0)

        # 429 → interval doubles.
        gov.record_response(VENUE, EP, status_code=429)
        assert gov.get_interval(VENUE, EP) == pytest.approx(0.20)

        # 3 successes → interval * 0.90.
        for _ in range(3):
            gov.record_response(VENUE, EP, status_code=200)
        assert gov.get_interval(VENUE, EP) == pytest.approx(0.18)

        # 3 more successes → another reduction.
        for _ in range(3):
            gov.record_response(VENUE, EP, status_code=200)
        assert gov.get_interval(VENUE, EP) == pytest.approx(0.162)

    def test_repeated_429_escalation(self) -> None:
        """Repeated 429s compound the backoff."""
        gov = _gov(base_interval=0.10, aimd_429_factor=2.0, max_interval=10.0)
        gov.consume(VENUE, EP, now=100.0)

        gov.record_response(VENUE, EP, status_code=429)
        assert gov.get_interval(VENUE, EP) == pytest.approx(0.20)

        gov.record_response(VENUE, EP, status_code=429)
        assert gov.get_interval(VENUE, EP) == pytest.approx(0.40)

        gov.record_response(VENUE, EP, status_code=429)
        assert gov.get_interval(VENUE, EP) == pytest.approx(0.80)


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_full_lifecycle(self) -> None:
        """Simulate realistic request lifecycle."""
        gov = _gov(
            base_interval=0.10,
            aimd_429_factor=1.75,
            aimd_5xx_factor=1.25,
            aimd_increase_factor=0.95,
            success_streak_threshold=4,
            bucket_capacity=3,
            max_interval=5.0,
        )

        now = 100.0

        # Phase 1: Initial burst of requests (bucket has 3 tokens).
        for i in range(3):
            wait = gov.acquire_wait_time(VENUE, EP, now=now)
            now += wait
            gov.consume(VENUE, EP, now=now)
            gov.record_response(VENUE, EP, status_code=200)

        # Phase 2: Hit rate limit.
        gov.record_response(VENUE, EP, status_code=429, retry_after=1.0)
        state = gov.get_state(VENUE, EP)
        assert state is not None
        assert state.interval >= 1.0

        # Phase 3: Recover with successes.
        for _ in range(8):
            gov.record_response(VENUE, EP, status_code=200)

        # Interval should have decreased from the 429 peak.
        recovered_interval = gov.get_interval(VENUE, EP)
        assert recovered_interval < 1.0

        # Phase 4: Server error.
        gov.record_response(VENUE, EP, status_code=502)
        assert gov.get_interval(VENUE, EP) > recovered_interval

        # Snapshot reflects totals.
        snap = gov.snapshot()
        assert snap.endpoint_count == 1
        assert snap.total_requests == 3
        assert snap.total_429s == 1
        assert snap.total_5xxs == 1

    def test_multi_venue_isolation(self) -> None:
        """Two venues operate completely independently."""
        gov = _gov(
            base_interval=0.10,
            aimd_429_factor=2.0,
            success_streak_threshold=2,
            aimd_increase_factor=0.90,
        )

        # Kalshi gets 429'd.
        gov.consume("kalshi", "orderbook", now=100.0)
        gov.record_response("kalshi", "orderbook", status_code=429)

        # Polymarket runs fine.
        gov.consume("polymarket", "orderbook", now=100.0)
        gov.record_response("polymarket", "orderbook", status_code=200)
        gov.record_response("polymarket", "orderbook", status_code=200)

        kalshi_interval = gov.get_interval("kalshi", "orderbook")
        poly_interval = gov.get_interval("polymarket", "orderbook")

        # Kalshi should be much slower.
        assert kalshi_interval > poly_interval
        assert kalshi_interval == pytest.approx(0.20)
        # Polymarket reduced after 2 successes.
        assert poly_interval == pytest.approx(0.10 * 0.90)
