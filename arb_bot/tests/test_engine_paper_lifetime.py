from __future__ import annotations

from datetime import datetime, timedelta, timezone
from dataclasses import replace

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
from arb_bot.engine import ArbEngine, OpportunityDecision
from arb_bot.models import (
    ArbitrageOpportunity,
    BinaryQuote,
    ExecutionStyle,
    OpportunityKind,
    OpportunityLeg,
    Side,
)


def _settings(
    *,
    static_seconds: int = 900,
    dynamic_enabled: bool = False,
    fraction: float = 0.02,
    min_seconds: int = 60,
    max_seconds: int = 600,
) -> AppSettings:
    return AppSettings(
        live_mode=False,
        run_once=False,
        poll_interval_seconds=60,
        dry_run=True,
        paper_strict_simulation=True,
        paper_position_lifetime_seconds=static_seconds,
        paper_dynamic_lifetime_enabled=dynamic_enabled,
        paper_dynamic_lifetime_resolution_fraction=fraction,
        paper_dynamic_lifetime_min_seconds=min_seconds,
        paper_dynamic_lifetime_max_seconds=max_seconds,
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
    )


def _opportunity(
    *,
    kind: OpportunityKind,
    resolution_ts: float | None,
    market_id: str = "M1",
) -> ArbitrageOpportunity:
    metadata = {"resolution_ts": resolution_ts} if resolution_ts is not None else {}
    legs = (
        OpportunityLeg(
            venue="kalshi",
            market_id=market_id,
            side=Side.YES,
            buy_price=0.48,
            buy_size=100.0,
            metadata=metadata,
        ),
        OpportunityLeg(
            venue="kalshi",
            market_id=market_id,
            side=Side.NO,
            buy_price=0.49,
            buy_size=100.0,
            metadata=metadata,
        ),
    )
    return ArbitrageOpportunity(
        kind=kind,
        execution_style=ExecutionStyle.TAKER,
        legs=legs,
        gross_edge_per_contract=0.03,
        net_edge_per_contract=0.02,
        fee_per_contract=0.0,
        observed_at=datetime.now(timezone.utc),
        match_key="test",
    )


def _engine_for(settings: AppSettings) -> ArbEngine:
    engine = ArbEngine.__new__(ArbEngine)
    engine._settings = settings
    return engine


def _two_leg_opp(
    *,
    kind: OpportunityKind,
    left_market_id: str,
    left_side: Side,
    right_market_id: str,
    right_side: Side,
    net_edge: float = 0.02,
) -> ArbitrageOpportunity:
    now = datetime.now(timezone.utc)
    legs = (
        OpportunityLeg(
            venue="kalshi",
            market_id=left_market_id,
            side=left_side,
            buy_price=0.48,
            buy_size=100.0,
            metadata={},
        ),
        OpportunityLeg(
            venue="polymarket",
            market_id=right_market_id,
            side=right_side,
            buy_price=0.49,
            buy_size=100.0,
            metadata={},
        ),
    )
    return ArbitrageOpportunity(
        kind=kind,
        execution_style=ExecutionStyle.TAKER,
        legs=legs,
        gross_edge_per_contract=net_edge,
        net_edge_per_contract=net_edge,
        fee_per_contract=0.0,
        observed_at=now,
        match_key=f"{left_market_id}:{right_market_id}",
        match_score=1.0,
    )


def test_paper_lifetime_uses_static_when_dynamic_disabled() -> None:
    settings = _settings(static_seconds=777, dynamic_enabled=False)
    engine = _engine_for(settings)
    now = datetime(2026, 2, 10, tzinfo=timezone.utc)
    opp = _opportunity(kind=OpportunityKind.CROSS_VENUE, resolution_ts=now.timestamp() + 3600)

    lifetime = engine._paper_position_lifetime_seconds(opportunity=opp, now=now)
    assert lifetime == 777


def test_paper_lifetime_dynamic_uses_resolution_fraction_with_bounds() -> None:
    settings = _settings(
        static_seconds=900,
        dynamic_enabled=True,
        fraction=0.02,
        min_seconds=60,
        max_seconds=600,
    )
    engine = _engine_for(settings)
    now = datetime(2026, 2, 10, tzinfo=timezone.utc)
    opp = _opportunity(kind=OpportunityKind.CROSS_VENUE, resolution_ts=now.timestamp() + (2 * 86400))

    lifetime = engine._paper_position_lifetime_seconds(opportunity=opp, now=now)
    assert lifetime == 600


def test_paper_lifetime_dynamic_falls_back_to_static_when_resolution_unknown() -> None:
    settings = _settings(static_seconds=555, dynamic_enabled=True)
    engine = _engine_for(settings)
    now = datetime(2026, 2, 10, tzinfo=timezone.utc)
    opp = _opportunity(kind=OpportunityKind.CROSS_VENUE, resolution_ts=None)

    lifetime = engine._paper_position_lifetime_seconds(opportunity=opp, now=now)
    assert lifetime == 555


def test_paper_lifetime_dynamic_applies_kind_multiplier() -> None:
    settings = _settings(
        static_seconds=900,
        dynamic_enabled=True,
        fraction=0.02,
        min_seconds=60,
        max_seconds=600,
    )
    engine = _engine_for(settings)
    now = datetime(2026, 2, 10, tzinfo=timezone.utc)
    resolution_ts = (now + timedelta(hours=3)).timestamp()

    cross = _opportunity(kind=OpportunityKind.CROSS_VENUE, resolution_ts=resolution_ts)
    event_tree = _opportunity(kind=OpportunityKind.STRUCTURAL_EVENT_TREE, resolution_ts=resolution_ts)

    cross_lifetime = engine._paper_position_lifetime_seconds(opportunity=cross, now=now)
    event_tree_lifetime = engine._paper_position_lifetime_seconds(opportunity=event_tree, now=now)

    assert cross_lifetime == 216
    assert event_tree_lifetime == 292


def test_lane_fair_order_prioritizes_non_intra_first() -> None:
    engine = ArbEngine.__new__(ArbEngine)

    intra = _opportunity(
        kind=OpportunityKind.INTRA_VENUE,
        resolution_ts=None,
        market_id="I1",
    )
    cross = _opportunity(
        kind=OpportunityKind.CROSS_VENUE,
        resolution_ts=None,
        market_id="C1",
    )
    parity = _opportunity(
        kind=OpportunityKind.STRUCTURAL_PARITY,
        resolution_ts=None,
        market_id="P1",
    )

    scored = [
        (0.09, intra.net_edge_per_contract, intra.match_score, intra),
        (0.08, cross.net_edge_per_contract, cross.match_score, cross),
        (0.07, parity.net_edge_per_contract, parity.match_score, parity),
    ]

    ordered = engine._lane_fair_order(scored)
    assert [opp.kind for opp in ordered[:3]] == [
        OpportunityKind.CROSS_VENUE,
        OpportunityKind.STRUCTURAL_PARITY,
        OpportunityKind.INTRA_VENUE,
    ]


def test_deduplicate_opportunities_suppresses_cross_parity_alias_double_count() -> None:
    cross = _two_leg_opp(
        kind=OpportunityKind.CROSS_VENUE,
        left_market_id="K1",
        left_side=Side.YES,
        right_market_id="P1",
        right_side=Side.NO,
        net_edge=0.02,
    )
    parity = replace(
        cross,
        kind=OpportunityKind.STRUCTURAL_PARITY,
        match_key="parity:K1:P1",
        net_edge_per_contract=0.02,
        gross_edge_per_contract=0.02,
    )

    deduped = ArbEngine._deduplicate_opportunities([cross, parity])

    assert len(deduped) == 1
    assert deduped[0].kind is OpportunityKind.CROSS_VENUE


def test_deduplicate_opportunities_keeps_distinct_cross_and_parity_if_legs_differ() -> None:
    cross = _two_leg_opp(
        kind=OpportunityKind.CROSS_VENUE,
        left_market_id="K1",
        left_side=Side.YES,
        right_market_id="P1",
        right_side=Side.NO,
        net_edge=0.02,
    )
    parity = _two_leg_opp(
        kind=OpportunityKind.STRUCTURAL_PARITY,
        left_market_id="K2",
        left_side=Side.YES,
        right_market_id="P2",
        right_side=Side.NO,
        net_edge=0.019,
    )

    deduped = ArbEngine._deduplicate_opportunities([cross, parity])

    assert len(deduped) == 2
    assert {opp.kind for opp in deduped} == {
        OpportunityKind.CROSS_VENUE,
        OpportunityKind.STRUCTURAL_PARITY,
    }


def test_deduplicate_opportunities_suppresses_cross_parity_when_markets_match_but_sides_differ() -> None:
    cross = _two_leg_opp(
        kind=OpportunityKind.CROSS_VENUE,
        left_market_id="K1",
        left_side=Side.YES,
        right_market_id="P1",
        right_side=Side.NO,
        net_edge=0.018,
    )
    parity_other_side = _two_leg_opp(
        kind=OpportunityKind.STRUCTURAL_PARITY,
        left_market_id="K1",
        left_side=Side.NO,
        right_market_id="P1",
        right_side=Side.YES,
        net_edge=0.02,
    )

    deduped = ArbEngine._deduplicate_opportunities([cross, parity_other_side])

    assert len(deduped) == 1
    assert deduped[0].kind is OpportunityKind.STRUCTURAL_PARITY


def test_opportunity_family_counts_merge_cross_and_parity() -> None:
    cross = _two_leg_opp(
        kind=OpportunityKind.CROSS_VENUE,
        left_market_id="K1",
        left_side=Side.YES,
        right_market_id="P1",
        right_side=Side.NO,
    )
    parity = _two_leg_opp(
        kind=OpportunityKind.STRUCTURAL_PARITY,
        left_market_id="K1",
        left_side=Side.NO,
        right_market_id="P1",
        right_side=Side.YES,
    )
    intra = _opportunity(
        kind=OpportunityKind.INTRA_VENUE,
        resolution_ts=None,
        market_id="I1",
    )

    counts = ArbEngine._opportunity_family_counts([cross, parity, intra])

    assert counts["cross_parity"] == 2
    assert counts[OpportunityKind.INTRA_VENUE.value] == 1
    assert counts[OpportunityKind.STRUCTURAL_BUCKET.value] == 0
    assert counts[OpportunityKind.STRUCTURAL_EVENT_TREE.value] == 0


def test_quote_quality_gate_reports_explicit_reject_reason() -> None:
    now = datetime(2026, 2, 11, tzinfo=timezone.utc)
    valid = BinaryQuote(
        venue="kalshi",
        market_id="VALID",
        yes_buy_price=0.45,
        no_buy_price=0.54,
        yes_buy_size=100.0,
        no_buy_size=100.0,
        observed_at=now,
    )
    invalid = BinaryQuote(
        venue="kalshi",
        market_id="BAD",
        yes_buy_price=1.2,
        no_buy_price=0.1,
        yes_buy_size=100.0,
        no_buy_size=100.0,
        observed_at=now,
    )

    filtered, reject_counts = ArbEngine._filter_quotes_for_quality([valid, invalid], now=now)

    assert len(filtered) == 1
    assert filtered[0].market_id == "VALID"
    assert reject_counts.get("price_out_of_range", 0) == 1


def test_lane_kelly_autotune_raises_floor_on_starvation() -> None:
    settings = _settings()
    engine = _engine_for(settings)
    engine._lane_dynamic_kelly_floor = {kind: 0.0 for kind in OpportunityKind}
    now = datetime(2026, 2, 11, tzinfo=timezone.utc)

    cross = _two_leg_opp(
        kind=OpportunityKind.CROSS_VENUE,
        left_market_id="K1",
        left_side=Side.YES,
        right_market_id="P1",
        right_side=Side.NO,
    )
    decisions = [
        OpportunityDecision(
            timestamp=now,
            action="skipped",
            reason="kelly_fraction_zero",
            opportunity=cross,
            plan=None,
            metrics={},
        )
        for _ in range(8)
    ]
    opportunities_by_kind = {kind.value: 0 for kind in OpportunityKind}
    opportunities_by_kind[OpportunityKind.CROSS_VENUE.value] = 12

    engine._update_dynamic_lane_kelly(opportunities_by_kind=opportunities_by_kind, decisions=decisions)

    assert engine._lane_dynamic_kelly_floor[OpportunityKind.CROSS_VENUE] > 0.0


def test_lane_kelly_autotune_decays_floor_after_open() -> None:
    settings = _settings()
    engine = _engine_for(settings)
    engine._lane_dynamic_kelly_floor = {kind: 0.0 for kind in OpportunityKind}
    engine._lane_dynamic_kelly_floor[OpportunityKind.CROSS_VENUE] = 0.05
    now = datetime(2026, 2, 11, tzinfo=timezone.utc)

    cross = _two_leg_opp(
        kind=OpportunityKind.CROSS_VENUE,
        left_market_id="K1",
        left_side=Side.YES,
        right_market_id="P1",
        right_side=Side.NO,
    )
    decisions = [
        OpportunityDecision(
            timestamp=now,
            action="dry_run",
            reason="paper_position_opened",
            opportunity=cross,
            plan=None,
            metrics={},
        )
    ]
    opportunities_by_kind = {kind.value: 0 for kind in OpportunityKind}
    opportunities_by_kind[OpportunityKind.CROSS_VENUE.value] = 12

    engine._update_dynamic_lane_kelly(opportunities_by_kind=opportunities_by_kind, decisions=decisions)

    assert engine._lane_dynamic_kelly_floor[OpportunityKind.CROSS_VENUE] < 0.05
