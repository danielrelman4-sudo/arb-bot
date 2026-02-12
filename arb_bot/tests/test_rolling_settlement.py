"""Tests for Phase 8C: Paper rolling settlement.

Validates that when rolling settlement is enabled, paper positions are settled
once they exceed the minimum hold time rather than waiting for full lifetime
expiry.  This prevents late-cycle capital starvation where early positions
block all available capital and market slots.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone

from arb_bot.config import (
    AppSettings,
    FillModelSettings,
    KalshiSettings,
    OpportunityLaneSettings,
    PolymarketSettings,
    RiskSettings,
    SizingSettings,
    StrategySettings,
    UniverseRankingSettings,
)
from arb_bot.engine import ArbEngine, PaperSimPosition
from arb_bot.models import (
    ArbitrageOpportunity,
    EngineState,
    ExecutionStyle,
    OpportunityKind,
    OpportunityLeg,
    Side,
    TradeLegPlan,
    TradePlan,
)
from arb_bot.risk import RiskManager

NOW = datetime(2026, 2, 12, 12, 0, 0, tzinfo=timezone.utc)


def _settings(
    *,
    lifetime_seconds: int = 600,
    rolling_enabled: bool = False,
    min_hold_seconds: int = 120,
) -> AppSettings:
    return AppSettings(
        live_mode=False,
        run_once=False,
        poll_interval_seconds=60,
        dry_run=True,
        paper_strict_simulation=True,
        paper_position_lifetime_seconds=lifetime_seconds,
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
        paper_rolling_settlement_enabled=rolling_enabled,
        paper_rolling_settlement_min_hold_seconds=min_hold_seconds,
    )


def _make_opportunity(market_id: str = "M1") -> ArbitrageOpportunity:
    legs = (
        OpportunityLeg(
            venue="kalshi",
            market_id=market_id,
            side=Side.YES,
            buy_price=0.48,
            buy_size=100.0,
            metadata={},
        ),
        OpportunityLeg(
            venue="kalshi",
            market_id=market_id,
            side=Side.NO,
            buy_price=0.49,
            buy_size=100.0,
            metadata={},
        ),
    )
    return ArbitrageOpportunity(
        kind=OpportunityKind.INTRA_VENUE,
        execution_style=ExecutionStyle.TAKER,
        legs=legs,
        gross_edge_per_contract=0.03,
        net_edge_per_contract=0.02,
        fee_per_contract=0.0,
        observed_at=NOW,
        match_key=f"test-{market_id}",
    )


def _make_plan(opportunity: ArbitrageOpportunity, contracts: int = 10) -> TradePlan:
    return TradePlan(
        kind=opportunity.kind,
        execution_style=opportunity.execution_style,
        legs=tuple(
            TradeLegPlan(leg.venue, leg.market_id, leg.side, contracts, leg.buy_price)
            for leg in opportunity.legs
        ),
        contracts=contracts,
        capital_required=contracts * 0.97,
        capital_required_by_venue={"kalshi": contracts * 0.97},
        expected_profit=contracts * 0.02,
        edge_per_contract=0.02,
    )


def _make_position(
    market_id: str = "M1",
    opened_at: datetime = NOW,
    lifetime_seconds: int = 600,
    contracts: int = 10,
    expected_profit: float = 0.2,
) -> PaperSimPosition:
    opp = _make_opportunity(market_id)
    plan = _make_plan(opp, contracts)
    return PaperSimPosition(
        opened_at=opened_at,
        release_at=opened_at + timedelta(seconds=lifetime_seconds),
        opportunity=opp,
        plan=plan,
        filled_contracts=contracts,
        committed_capital_by_venue={"kalshi": contracts * 0.97},
        expected_realized_profit=expected_profit,
    )


def _engine_with_positions(
    settings: AppSettings,
    positions: list[PaperSimPosition],
) -> ArbEngine:
    engine = ArbEngine.__new__(ArbEngine)
    engine._settings = settings
    engine._paper_positions = list(positions)
    engine._state = EngineState(
        cash_by_venue={"kalshi": 100.0},
        locked_capital_by_venue={"kalshi": sum(
            sum(p.committed_capital_by_venue.values()) for p in positions
        )},
        open_markets_by_venue={"kalshi": set()},
    )
    engine._risk = RiskManager(settings.risk)
    engine._kelly_realized_profits = []
    engine._kelly_relative_error_window = deque(maxlen=200)
    return engine


# ===================================================================
# Tests for rolling settlement
# ===================================================================


class TestRollingSettlement:
    """Tests for rolling settlement of paper positions."""

    def test_fixed_mode_only_settles_at_lifetime_expiry(self) -> None:
        """Without rolling, positions are not settled until release_at."""
        settings = _settings(lifetime_seconds=600, rolling_enabled=False)
        # Position opened 5 minutes ago (300s), lifetime 600s → not yet settled.
        position = _make_position(
            opened_at=NOW - timedelta(seconds=300),
            lifetime_seconds=600,
        )
        engine = _engine_with_positions(settings, [position])

        settled = engine._process_paper_settlements(NOW)
        assert len(settled) == 0
        assert len(engine._paper_positions) == 1

    def test_fixed_mode_settles_at_expiry(self) -> None:
        """Fixed mode settles positions past release_at."""
        settings = _settings(lifetime_seconds=600, rolling_enabled=False)
        position = _make_position(
            opened_at=NOW - timedelta(seconds=601),
            lifetime_seconds=600,
        )
        engine = _engine_with_positions(settings, [position])

        settled = engine._process_paper_settlements(NOW)
        assert len(settled) == 1
        assert len(engine._paper_positions) == 0

    def test_rolling_settles_past_min_hold(self) -> None:
        """Rolling mode settles positions past min_hold_seconds even before lifetime."""
        settings = _settings(
            lifetime_seconds=600,
            rolling_enabled=True,
            min_hold_seconds=120,
        )
        # Position opened 150s ago — past 120s min hold but before 600s lifetime.
        position = _make_position(
            opened_at=NOW - timedelta(seconds=150),
            lifetime_seconds=600,
        )
        engine = _engine_with_positions(settings, [position])

        settled = engine._process_paper_settlements(NOW)
        assert len(settled) == 1
        assert len(engine._paper_positions) == 0
        assert settled[0].action == "settled"

    def test_rolling_keeps_positions_below_min_hold(self) -> None:
        """Rolling mode does NOT settle positions younger than min_hold_seconds."""
        settings = _settings(
            lifetime_seconds=600,
            rolling_enabled=True,
            min_hold_seconds=120,
        )
        # Position opened 60s ago — below 120s min hold.
        position = _make_position(
            opened_at=NOW - timedelta(seconds=60),
            lifetime_seconds=600,
        )
        engine = _engine_with_positions(settings, [position])

        settled = engine._process_paper_settlements(NOW)
        assert len(settled) == 0
        assert len(engine._paper_positions) == 1

    def test_rolling_settles_oldest_first(self) -> None:
        """Rolling mode settles older positions while keeping newer ones."""
        settings = _settings(
            lifetime_seconds=600,
            rolling_enabled=True,
            min_hold_seconds=120,
        )
        old_position = _make_position(
            market_id="OLD",
            opened_at=NOW - timedelta(seconds=200),  # Past min hold
            lifetime_seconds=600,
        )
        new_position = _make_position(
            market_id="NEW",
            opened_at=NOW - timedelta(seconds=60),  # Below min hold
            lifetime_seconds=600,
        )
        engine = _engine_with_positions(settings, [old_position, new_position])

        settled = engine._process_paper_settlements(NOW)
        assert len(settled) == 1
        assert settled[0].opportunity.match_key == "test-OLD"
        assert len(engine._paper_positions) == 1
        assert engine._paper_positions[0].opportunity.match_key == "test-NEW"

    def test_rolling_frees_capital(self) -> None:
        """Rolling settlement should return committed capital to the engine state."""
        settings = _settings(
            lifetime_seconds=600,
            rolling_enabled=True,
            min_hold_seconds=120,
        )
        position = _make_position(
            opened_at=NOW - timedelta(seconds=200),
            lifetime_seconds=600,
            contracts=10,
            expected_profit=0.5,
        )
        engine = _engine_with_positions(settings, [position])
        initial_cash = engine._state.cash_for("kalshi")

        settled = engine._process_paper_settlements(NOW)
        assert len(settled) == 1
        # Capital + profit should be returned.
        assert engine._state.cash_for("kalshi") > initial_cash

    def test_rolling_mixed_ages(self) -> None:
        """Multiple positions at different ages: only those past min hold settle."""
        settings = _settings(
            lifetime_seconds=600,
            rolling_enabled=True,
            min_hold_seconds=120,
        )
        positions = [
            _make_position(market_id="P1", opened_at=NOW - timedelta(seconds=300)),  # settle
            _make_position(market_id="P2", opened_at=NOW - timedelta(seconds=150)),  # settle
            _make_position(market_id="P3", opened_at=NOW - timedelta(seconds=119)),  # keep
            _make_position(market_id="P4", opened_at=NOW - timedelta(seconds=30)),   # keep
        ]
        engine = _engine_with_positions(settings, positions)

        settled = engine._process_paper_settlements(NOW)
        assert len(settled) == 2
        assert len(engine._paper_positions) == 2
        settled_keys = {s.opportunity.match_key for s in settled}
        assert "test-P1" in settled_keys
        assert "test-P2" in settled_keys

    def test_rolling_still_settles_at_expiry(self) -> None:
        """Rolling mode should also settle positions that have reached their normal expiry."""
        settings = _settings(
            lifetime_seconds=600,
            rolling_enabled=True,
            min_hold_seconds=120,
        )
        # Position expired (lifetime reached).
        position = _make_position(
            opened_at=NOW - timedelta(seconds=601),
            lifetime_seconds=600,
        )
        engine = _engine_with_positions(settings, [position])

        settled = engine._process_paper_settlements(NOW)
        assert len(settled) == 1


# ===================================================================
# Tests for config parameters
# ===================================================================


class TestRollingSettlementConfig:
    """Tests for rolling settlement configuration."""

    def test_default_disabled(self) -> None:
        settings = _settings()
        assert settings.paper_rolling_settlement_enabled is False

    def test_enabled_with_min_hold(self) -> None:
        settings = _settings(rolling_enabled=True, min_hold_seconds=90)
        assert settings.paper_rolling_settlement_enabled is True
        assert settings.paper_rolling_settlement_min_hold_seconds == 90

    def test_reduced_lifetime_default(self) -> None:
        """Verify the plan's recommendation: 300s instead of 600s/900s."""
        settings = _settings(lifetime_seconds=300)
        assert settings.paper_position_lifetime_seconds == 300
