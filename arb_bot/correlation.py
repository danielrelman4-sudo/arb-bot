from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable

from arb_bot.models import ArbitrageOpportunity, OpportunityKind, Side
from arb_bot.structural_rules import StructuralRuleSet, load_structural_rules

MarketKey = tuple[str, str]


@dataclass(frozen=True)
class CorrelationAssessment:
    cluster_key: str
    min_payout_per_contract: float
    adjusted_payout_per_contract: float
    residual_gross_edge_per_contract: float
    residual_net_edge_per_contract: float
    considered_markets: int
    valid_assignments: int
    assumptions: tuple[str, ...]


class _DisjointSet:
    def __init__(self) -> None:
        self._parent: dict[MarketKey, MarketKey] = {}

    def add(self, key: MarketKey) -> None:
        self._parent.setdefault(key, key)

    def find(self, key: MarketKey) -> MarketKey:
        parent = self._parent.get(key)
        if parent is None:
            self._parent[key] = key
            return key
        if parent == key:
            return key
        root = self.find(parent)
        self._parent[key] = root
        return root

    def union(self, left: MarketKey, right: MarketKey) -> None:
        self.add(left)
        self.add(right)
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self._parent[root_right] = root_left


class CorrelationConstraintModel:
    """Scores opportunity edge against explicit event-dependency constraints."""

    def __init__(
        self,
        structural_rules_path: str | None,
        *,
        enabled: bool = True,
        max_bruteforce_markets: int = 14,
        cross_venue_equivalence_min_match_score: float = 0.9,
        assume_structural_buckets_exhaustive: bool = True,
    ) -> None:
        self._enabled = enabled
        self._max_bruteforce_markets = max(2, max_bruteforce_markets)
        self._cross_equivalence_min_match_score = max(0.0, min(1.0, cross_venue_equivalence_min_match_score))
        self._assume_structural_buckets_exhaustive = assume_structural_buckets_exhaustive

        self._rules: StructuralRuleSet = load_structural_rules(structural_rules_path) if enabled else StructuralRuleSet()
        self._bucket_constraints: list[tuple[MarketKey, ...]] = []
        self._event_tree_constraints: list[tuple[MarketKey, tuple[MarketKey, ...]]] = []
        self._parity_constraints: list[tuple[MarketKey, MarketKey, str]] = []

        self._dsu = _DisjointSet()
        self._component_markets: dict[MarketKey, set[MarketKey]] = {}
        self._market_to_component: dict[MarketKey, MarketKey] = {}
        self._min_payout_cache: dict[tuple[tuple[str, str, str], ...], tuple[float, int, int, tuple[str, ...]]] = {}

        if self._enabled:
            self._build_constraint_index()

    def assess(self, opportunity: ArbitrageOpportunity) -> CorrelationAssessment:
        cluster_key = self._cluster_key_for_opportunity(opportunity)
        if not self._enabled:
            gross_edge = opportunity.payout_per_contract - sum(leg.buy_price for leg in opportunity.legs)
            net_edge = gross_edge - opportunity.fee_per_contract
            return CorrelationAssessment(
                cluster_key=cluster_key,
                min_payout_per_contract=opportunity.payout_per_contract,
                adjusted_payout_per_contract=opportunity.payout_per_contract,
                residual_gross_edge_per_contract=gross_edge,
                residual_net_edge_per_contract=net_edge,
                considered_markets=0,
                valid_assignments=0,
                assumptions=("constraint_pricing_disabled",),
            )

        min_payout, valid_assignments, considered_markets, assumptions = self._minimum_payout(opportunity)
        adjusted_payout = min(opportunity.payout_per_contract, min_payout)
        gross_edge = adjusted_payout - sum(leg.buy_price for leg in opportunity.legs)
        net_edge = gross_edge - opportunity.fee_per_contract

        return CorrelationAssessment(
            cluster_key=cluster_key,
            min_payout_per_contract=min_payout,
            adjusted_payout_per_contract=adjusted_payout,
            residual_gross_edge_per_contract=gross_edge,
            residual_net_edge_per_contract=net_edge,
            considered_markets=considered_markets,
            valid_assignments=valid_assignments,
            assumptions=assumptions,
        )

    def _build_constraint_index(self) -> None:
        for bucket in self._rules.buckets:
            markets = tuple(_key_from_ref(ref.venue, ref.market_id) for ref in bucket.legs)
            if len(markets) < 2:
                continue
            self._bucket_constraints.append(markets)
            self._register_component(markets)

        for rule in self._rules.event_trees:
            parent = _key_from_ref(rule.parent.venue, rule.parent.market_id)
            children = tuple(_key_from_ref(child.venue, child.market_id) for child in rule.children)
            if not children:
                continue
            self._event_tree_constraints.append((parent, children))
            self._register_component((parent, *children))

        for parity in self._rules.parity_checks:
            left = _key_from_ref(parity.left.venue, parity.left.market_id)
            right = _key_from_ref(parity.right.venue, parity.right.market_id)
            relationship = "complement" if parity.relationship == "complement" else "equivalent"
            self._parity_constraints.append((left, right, relationship))
            self._register_component((left, right))

        by_component: dict[MarketKey, set[MarketKey]] = {}
        for market in list(self._dsu._parent.keys()):
            root = self._dsu.find(market)
            by_component.setdefault(root, set()).add(market)
            self._market_to_component[market] = root
        self._component_markets = by_component

    def _register_component(self, markets: tuple[MarketKey, ...]) -> None:
        first = markets[0]
        self._dsu.add(first)
        for other in markets[1:]:
            self._dsu.union(first, other)

    def _cluster_key_for_opportunity(self, opportunity: ArbitrageOpportunity) -> str:
        markets = {_key_from_ref(leg.venue, leg.market_id) for leg in opportunity.legs}
        components = {
            self._market_to_component[market]
            for market in markets
            if market in self._market_to_component
        }
        if components:
            normalized = sorted(_market_key_to_string(component) for component in components)
            return f"constraint:{'+'.join(normalized)}"

        match_key = _normalize_cluster_part(opportunity.match_key)
        if match_key:
            return f"match:{match_key}"

        single = sorted(_market_key_to_string(market) for market in markets)
        if single:
            return f"market:{single[0]}"
        return "market:unknown"

    def _minimum_payout(
        self,
        opportunity: ArbitrageOpportunity,
    ) -> tuple[float, int, int, tuple[str, ...]]:
        cache_key = tuple(
            (leg.venue.lower(), leg.market_id.lower(), leg.side.value)
            for leg in opportunity.legs
        )
        cached = self._min_payout_cache.get(cache_key)
        if cached is not None:
            return cached

        assumptions: list[str] = []
        leg_markets = {_key_from_ref(leg.venue, leg.market_id) for leg in opportunity.legs}
        active_markets = set(leg_markets)
        for market in leg_markets:
            component = self._market_to_component.get(market)
            if component is not None:
                active_markets.update(self._component_markets.get(component, set()))

        if not active_markets:
            result = (opportunity.payout_per_contract, 0, 0, ("no_constraint_markets",))
            self._min_payout_cache[cache_key] = result
            return result

        if len(active_markets) > self._max_bruteforce_markets:
            result = (
                opportunity.payout_per_contract,
                0,
                len(active_markets),
                ("constraint_bruteforce_limit_reached",),
            )
            self._min_payout_cache[cache_key] = result
            return result

        constraints: list[Callable[[dict[MarketKey, int]], bool]] = []
        for markets in self._bucket_constraints:
            if all(market in active_markets for market in markets):
                if self._assume_structural_buckets_exhaustive:
                    constraints.append(
                        lambda assignment, mk=markets: sum(assignment[market] for market in mk) == 1
                    )
                else:
                    # Conservative mode: bucket legs are only assumed mutually exclusive,
                    # not necessarily exhaustive.
                    constraints.append(
                        lambda assignment, mk=markets: sum(assignment[market] for market in mk) <= 1
                    )

        for parent, children in self._event_tree_constraints:
            members = (parent, *children)
            if all(market in active_markets for market in members):
                constraints.append(
                    lambda assignment, p=parent, ch=children: sum(assignment[c] for c in ch) == assignment[p]
                )

        for left, right, relationship in self._parity_constraints:
            if left not in active_markets or right not in active_markets:
                continue
            if relationship == "complement":
                constraints.append(
                    lambda assignment, l=left, r=right: assignment[l] + assignment[r] == 1
                )
            else:
                constraints.append(
                    lambda assignment, l=left, r=right: assignment[l] == assignment[r]
                )

        cross_markets = list({(leg.venue.lower(), leg.market_id.lower()) for leg in opportunity.legs})
        if opportunity.kind is OpportunityKind.CROSS_VENUE and len(cross_markets) >= 2:
            if opportunity.match_score >= self._cross_equivalence_min_match_score:
                first = cross_markets[0]
                for other in cross_markets[1:]:
                    constraints.append(
                        lambda assignment, l=first, r=other: assignment[l] == assignment[r]
                    )
                assumptions.append("cross_equivalence_assumed")
            else:
                assumptions.append("cross_equivalence_not_assumed")

        ordered_markets = tuple(sorted(active_markets))
        valid_assignments = 0
        min_payout = math.inf
        market_count = len(ordered_markets)
        total_states = 1 << market_count

        for state in range(total_states):
            assignment = {
                market: (state >> index) & 1
                for index, market in enumerate(ordered_markets)
            }
            if any(not constraint(assignment) for constraint in constraints):
                continue

            valid_assignments += 1
            payout = 0.0
            for leg in opportunity.legs:
                market_key = _key_from_ref(leg.venue, leg.market_id)
                outcome = assignment[market_key]
                payout += float(outcome if leg.side is Side.YES else 1 - outcome)

            if payout < min_payout:
                min_payout = payout

        if valid_assignments == 0:
            result = (0.0, 0, market_count, tuple(assumptions + ["no_valid_constraint_assignments"]))
            self._min_payout_cache[cache_key] = result
            return result

        if not assumptions:
            assumptions.append("constraint_priced")

        result = (min_payout, valid_assignments, market_count, tuple(assumptions))
        self._min_payout_cache[cache_key] = result
        return result


def _key_from_ref(venue: str, market_id: str) -> MarketKey:
    return venue.strip().lower(), market_id.strip().lower()


def _market_key_to_string(market: MarketKey) -> str:
    return f"{market[0]}:{market[1]}"


def _normalize_cluster_part(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    return normalized.strip("_")
