from __future__ import annotations

import time
from typing import TYPE_CHECKING

from arb_bot.config import RiskSettings
from arb_bot.models import EngineState, OpportunityKind, TradePlan

if TYPE_CHECKING:
    from arb_bot.framework.kill_switch import KillSwitchManager


class RiskManager:
    def __init__(
        self,
        settings: RiskSettings,
        kill_switch: KillSwitchManager | None = None,
    ) -> None:
        self._settings = settings
        self._kill_switch = kill_switch

    @property
    def kill_switch(self) -> KillSwitchManager | None:
        return self._kill_switch

    def precheck(self, plan: TradePlan, state: EngineState) -> tuple[bool, str]:
        # Phase 1A: Kill switch checks before anything else.
        if self._kill_switch is not None:
            ks_state = self._kill_switch.check(plan.venues)
            if ks_state.halted:
                return False, f"kill switch: {ks_state.reason}"

        now = time.time()

        for leg in plan.legs:
            market_key = self._cooldown_key(leg_venue=leg.venue, leg_market_id=leg.market_id, leg_side=leg.side.value)
            last_trade_ts = state.last_trade_ts_by_market.get(market_key)
            if last_trade_ts is not None and (now - last_trade_ts) < self._settings.market_cooldown_seconds:
                scope = self._normalized_cooldown_scope()
                if scope == "market_side":
                    return False, f"market cooldown active ({leg.venue}/{leg.market_id}:{leg.side.value})"
                return False, f"market cooldown active ({leg.venue}/{leg.market_id})"

        for venue, required in plan.capital_required_by_venue.items():
            venue_cash = state.cash_for(venue)
            if required > venue_cash:
                return False, f"insufficient cash ({venue})"

            venue_locked = state.locked_for(venue)
            if venue_locked + required > self._settings.max_exposure_per_venue_usd:
                return False, f"venue exposure cap reached ({venue})"

        new_markets_by_venue: dict[str, set[str]] = {}
        for leg in plan.legs:
            new_markets_by_venue.setdefault(leg.venue, set()).add(leg.market_id)

        for venue, new_markets in new_markets_by_venue.items():
            existing_open = state.open_markets_by_venue.get(venue, set())
            additional = len([market for market in new_markets if market not in existing_open])
            venue_open_cap = self._settings.max_open_markets_per_venue
            if plan.kind is OpportunityKind.INTRA_VENUE:
                reserve_for_non_intra = max(
                    0,
                    self._settings.non_intra_open_market_reserve_per_venue,
                )
                venue_open_cap = max(0, venue_open_cap - reserve_for_non_intra)
            if state.open_market_count(venue) + additional > venue_open_cap:
                if plan.kind is OpportunityKind.INTRA_VENUE and self._settings.non_intra_open_market_reserve_per_venue > 0:
                    return False, f"open market cap reached ({venue}; intra reserve)"
                return False, f"open market cap reached ({venue})"

        opportunity_cooldown = max(0, int(self._settings.opportunity_cooldown_seconds))
        if opportunity_cooldown > 0:
            opportunity_key = self._opportunity_cooldown_key(plan)
            last_trade_ts = state.last_trade_ts_by_opportunity.get(opportunity_key)
            if last_trade_ts is not None and (now - last_trade_ts) < opportunity_cooldown:
                return False, f"opportunity cooldown active ({opportunity_key[0]}/{opportunity_key[1]})"

        return True, "ok"

    def record_fill(self, plan: TradePlan, state: EngineState, filled_contracts: int) -> None:
        if filled_contracts <= 0 or plan.contracts <= 0:
            return

        fill_ratio = min(1.0, filled_contracts / plan.contracts)

        for venue, planned_capital in plan.capital_required_by_venue.items():
            committed = planned_capital * fill_ratio
            state.cash_by_venue[venue] = max(0.0, state.cash_for(venue) - committed)
            state.locked_capital_by_venue[venue] = state.locked_for(venue) + committed

        for leg in plan.legs:
            state.mark_open_market(leg.venue, leg.market_id)
            market_key = self._cooldown_key(
                leg_venue=leg.venue,
                leg_market_id=leg.market_id,
                leg_side=leg.side.value,
            )
            state.last_trade_ts_by_market[market_key] = time.time()

        opportunity_cooldown = max(0, int(self._settings.opportunity_cooldown_seconds))
        if opportunity_cooldown > 0:
            opportunity_key = self._opportunity_cooldown_key(plan)
            state.last_trade_ts_by_opportunity[opportunity_key] = time.time()

    def _normalized_cooldown_scope(self) -> str:
        scope = (self._settings.market_cooldown_scope or "market").strip().lower()
        if scope not in {"market", "market_side"}:
            return "market"
        return scope

    def _cooldown_key(
        self,
        *,
        leg_venue: str,
        leg_market_id: str,
        leg_side: str,
    ) -> tuple[str, str]:
        scope = self._normalized_cooldown_scope()
        if scope == "market_side":
            return (leg_venue, f"{leg_market_id}:{leg_side}")
        return (leg_venue, leg_market_id)

    @staticmethod
    def _opportunity_cooldown_key(plan: TradePlan) -> tuple[str, str]:
        match_key = str(plan.metadata.get("match_key") or "").strip()

        if plan.kind in {OpportunityKind.CROSS_VENUE, OpportunityKind.STRUCTURAL_PARITY}:
            # Collapse cross/parity aliases into one cooldown namespace so the same
            # equivalence pair cannot be reopened repeatedly via lane permutations.
            markets = sorted({f"{leg.venue}/{leg.market_id}" for leg in plan.legs})
            if markets:
                return ("cross_parity_pair", "|".join(markets))

        if plan.kind is OpportunityKind.STRUCTURAL_BUCKET and match_key:
            return ("structural_bucket", match_key)
        if plan.kind is OpportunityKind.STRUCTURAL_EVENT_TREE and match_key:
            return ("structural_event_tree", match_key)
        if plan.kind is OpportunityKind.INTRA_VENUE and match_key:
            return ("intra_venue", match_key)
        if plan.kind is OpportunityKind.CROSS_VENUE and match_key:
            return ("cross_venue", match_key)
        if plan.kind is OpportunityKind.STRUCTURAL_PARITY and match_key:
            return ("structural_parity", match_key)

        fallback = "|".join(
            sorted(f"{leg.venue}/{leg.market_id}:{leg.side.value}" for leg in plan.legs)
        )
        return (plan.kind.value, fallback or plan.market_key)
