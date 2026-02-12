from arb_bot.config import RiskSettings
from arb_bot.models import ExecutionStyle, OpportunityKind, Side, EngineState, TradeLegPlan, TradePlan
from arb_bot.risk import RiskManager


def test_risk_blocks_when_exposure_cap_hit() -> None:
    risk = RiskManager(
        RiskSettings(
            max_exposure_per_venue_usd=50,
            max_open_markets_per_venue=10,
            market_cooldown_seconds=60,
        )
    )
    state = EngineState(cash_by_venue={"kalshi": 100}, locked_capital_by_venue={"kalshi": 45})
    plan = TradePlan(
        kind=OpportunityKind.INTRA_VENUE,
        execution_style=ExecutionStyle.TAKER,
        legs=(
            TradeLegPlan(
                venue="kalshi",
                market_id="M1",
                side=Side.YES,
                contracts=10,
                limit_price=0.4,
            ),
            TradeLegPlan(
                venue="kalshi",
                market_id="M1",
                side=Side.NO,
                contracts=10,
                limit_price=0.4,
            ),
        ),
        contracts=10,
        capital_required=8,
        capital_required_by_venue={"kalshi": 8},
        expected_profit=1.0,
        edge_per_contract=0.1,
    )

    allowed, reason = risk.precheck(plan, state)
    assert not allowed
    assert reason == "venue exposure cap reached (kalshi)"


def test_risk_market_side_cooldown_scope_allows_opposite_side() -> None:
    risk = RiskManager(
        RiskSettings(
            max_exposure_per_venue_usd=1_000,
            max_open_markets_per_venue=20,
            market_cooldown_seconds=300,
            market_cooldown_scope="market_side",
        )
    )

    state = EngineState(
        cash_by_venue={"kalshi": 1000},
        last_trade_ts_by_market={("kalshi", "M1:yes"): 10_000_000_000.0},
    )

    opposite_side_plan = TradePlan(
        kind=OpportunityKind.CROSS_VENUE,
        execution_style=ExecutionStyle.TAKER,
        legs=(
            TradeLegPlan(
                venue="kalshi",
                market_id="M1",
                side=Side.NO,
                contracts=1,
                limit_price=0.4,
            ),
        ),
        contracts=1,
        capital_required=0.4,
        capital_required_by_venue={"kalshi": 0.4},
        expected_profit=0.01,
        edge_per_contract=0.01,
    )
    allowed, reason = risk.precheck(opposite_side_plan, state)
    assert allowed, reason

    same_side_plan = TradePlan(
        kind=OpportunityKind.CROSS_VENUE,
        execution_style=ExecutionStyle.TAKER,
        legs=(
            TradeLegPlan(
                venue="kalshi",
                market_id="M1",
                side=Side.YES,
                contracts=1,
                limit_price=0.4,
            ),
        ),
        contracts=1,
        capital_required=0.4,
        capital_required_by_venue={"kalshi": 0.4},
        expected_profit=0.01,
        edge_per_contract=0.01,
    )
    allowed, reason = risk.precheck(same_side_plan, state)
    assert not allowed
    assert reason == "market cooldown active (kalshi/M1:yes)"


def test_risk_opportunity_cooldown_blocks_repeated_cross_parity_pair() -> None:
    risk = RiskManager(
        RiskSettings(
            max_exposure_per_venue_usd=1_000,
            max_open_markets_per_venue=20,
            market_cooldown_seconds=0,
            market_cooldown_scope="market_side",
            opportunity_cooldown_seconds=600,
        )
    )

    state = EngineState(
        cash_by_venue={"kalshi": 1000, "polymarket": 1000},
        last_trade_ts_by_opportunity={("cross_parity_pair", "kalshi/K1|polymarket/P1"): 10_000_000_000.0},
    )

    plan = TradePlan(
        kind=OpportunityKind.STRUCTURAL_PARITY,
        execution_style=ExecutionStyle.TAKER,
        legs=(
            TradeLegPlan(
                venue="kalshi",
                market_id="K1",
                side=Side.NO,
                contracts=1,
                limit_price=0.4,
            ),
            TradeLegPlan(
                venue="polymarket",
                market_id="P1",
                side=Side.YES,
                contracts=1,
                limit_price=0.5,
            ),
        ),
        contracts=1,
        capital_required=0.9,
        capital_required_by_venue={"kalshi": 0.4, "polymarket": 0.5},
        expected_profit=0.02,
        edge_per_contract=0.02,
        metadata={"match_key": "some-rule:left_no_right_yes"},
    )

    allowed, reason = risk.precheck(plan, state)
    assert not allowed
    assert reason == "opportunity cooldown active (cross_parity_pair/kalshi/K1|polymarket/P1)"


def test_risk_intra_open_market_reserve_blocks_intra() -> None:
    risk = RiskManager(
        RiskSettings(
            max_exposure_per_venue_usd=1_000,
            max_open_markets_per_venue=5,
            non_intra_open_market_reserve_per_venue=2,
            market_cooldown_seconds=0,
        )
    )
    state = EngineState(
        cash_by_venue={"polymarket": 1_000},
        open_markets_by_venue={"polymarket": {"M1", "M2", "M3"}},
    )
    plan = TradePlan(
        kind=OpportunityKind.INTRA_VENUE,
        execution_style=ExecutionStyle.TAKER,
        legs=(
            TradeLegPlan(
                venue="polymarket",
                market_id="M4",
                side=Side.YES,
                contracts=1,
                limit_price=0.4,
            ),
            TradeLegPlan(
                venue="polymarket",
                market_id="M4",
                side=Side.NO,
                contracts=1,
                limit_price=0.5,
            ),
        ),
        contracts=1,
        capital_required=0.9,
        capital_required_by_venue={"polymarket": 0.9},
        expected_profit=0.01,
        edge_per_contract=0.01,
    )

    allowed, reason = risk.precheck(plan, state)
    assert not allowed
    assert reason == "open market cap reached (polymarket; intra reserve)"


def test_risk_non_intra_can_use_reserved_slots() -> None:
    risk = RiskManager(
        RiskSettings(
            max_exposure_per_venue_usd=1_000,
            max_open_markets_per_venue=5,
            non_intra_open_market_reserve_per_venue=2,
            market_cooldown_seconds=0,
        )
    )
    state = EngineState(
        cash_by_venue={"polymarket": 1_000, "kalshi": 1_000},
        open_markets_by_venue={"polymarket": {"M1", "M2", "M3"}},
    )
    plan = TradePlan(
        kind=OpportunityKind.CROSS_VENUE,
        execution_style=ExecutionStyle.TAKER,
        legs=(
            TradeLegPlan(
                venue="polymarket",
                market_id="M4",
                side=Side.YES,
                contracts=1,
                limit_price=0.4,
            ),
            TradeLegPlan(
                venue="kalshi",
                market_id="K1",
                side=Side.NO,
                contracts=1,
                limit_price=0.5,
            ),
        ),
        contracts=1,
        capital_required=0.9,
        capital_required_by_venue={"polymarket": 0.4, "kalshi": 0.5},
        expected_profit=0.01,
        edge_per_contract=0.01,
    )

    allowed, reason = risk.precheck(plan, state)
    assert allowed, reason
