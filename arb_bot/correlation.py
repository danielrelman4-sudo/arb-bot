from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Callable

from arb_bot.models import ArbitrageOpportunity, OpportunityKind, Side
from arb_bot.structural_rules import StructuralRuleSet, load_structural_rules

# Optional PuLP import for ILP-based constraint solving.
try:
    import pulp  # type: ignore[import-untyped]
    _HAS_PULP = True
except ImportError:
    pulp = None  # type: ignore[assignment]
    _HAS_PULP = False

_LOG = logging.getLogger(__name__)

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
        max_ilp_markets: int = 200,
        cross_venue_equivalence_min_match_score: float = 0.9,
        assume_structural_buckets_exhaustive: bool = True,
    ) -> None:
        self._enabled = enabled
        self._max_bruteforce_markets = max(2, max_bruteforce_markets)
        self._max_ilp_markets = max(self._max_bruteforce_markets, max_ilp_markets)
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

        market_count = len(active_markets)

        # Collect structured constraints for both ILP and brute-force paths.
        bucket_cs: list[tuple[MarketKey, ...]] = []
        for markets in self._bucket_constraints:
            if all(market in active_markets for market in markets):
                bucket_cs.append(markets)

        tree_cs: list[tuple[MarketKey, tuple[MarketKey, ...]]] = []
        for parent, children in self._event_tree_constraints:
            members = (parent, *children)
            if all(market in active_markets for market in members):
                tree_cs.append((parent, children))

        parity_cs: list[tuple[MarketKey, MarketKey, str]] = []
        for left, right, relationship in self._parity_constraints:
            if left in active_markets and right in active_markets:
                parity_cs.append((left, right, relationship))

        cross_equiv_pairs: list[tuple[MarketKey, MarketKey]] = []
        cross_markets = list({(leg.venue.lower(), leg.market_id.lower()) for leg in opportunity.legs})
        if opportunity.kind is OpportunityKind.CROSS_VENUE and len(cross_markets) >= 2:
            if opportunity.match_score >= self._cross_equivalence_min_match_score:
                first = cross_markets[0]
                for other in cross_markets[1:]:
                    cross_equiv_pairs.append((first, other))
                assumptions.append("cross_equivalence_assumed")
            else:
                assumptions.append("cross_equivalence_not_assumed")

        # Choose solver: ILP for large problems, brute-force for small.
        if market_count > self._max_bruteforce_markets:
            if not _HAS_PULP or market_count > self._max_ilp_markets:
                result = (
                    opportunity.payout_per_contract,
                    0,
                    market_count,
                    ("constraint_limit_reached",),
                )
                self._min_payout_cache[cache_key] = result
                return result

            min_payout, valid_assignments = _solve_ilp(
                active_markets=active_markets,
                opportunity=opportunity,
                bucket_constraints=bucket_cs,
                tree_constraints=tree_cs,
                parity_constraints=parity_cs,
                cross_equiv_pairs=cross_equiv_pairs,
                assume_exhaustive=self._assume_structural_buckets_exhaustive,
            )
            assumptions.append("ilp_solver")
        else:
            min_payout, valid_assignments = _solve_bruteforce(
                active_markets=active_markets,
                opportunity=opportunity,
                bucket_constraints=bucket_cs,
                tree_constraints=tree_cs,
                parity_constraints=parity_cs,
                cross_equiv_pairs=cross_equiv_pairs,
                assume_exhaustive=self._assume_structural_buckets_exhaustive,
            )

        if valid_assignments == 0:
            result = (0.0, 0, market_count, tuple(assumptions + ["no_valid_constraint_assignments"]))
            self._min_payout_cache[cache_key] = result
            return result

        if not any(a.startswith("cross_") for a in assumptions):
            assumptions.append("constraint_priced")

        result = (min_payout, valid_assignments, market_count, tuple(assumptions))
        self._min_payout_cache[cache_key] = result
        return result


# ---------------------------------------------------------------------------
# Brute-force solver (original, O(2^N), N ≤ max_bruteforce_markets)
# ---------------------------------------------------------------------------


def _solve_bruteforce(
    *,
    active_markets: set[MarketKey],
    opportunity: ArbitrageOpportunity,
    bucket_constraints: list[tuple[MarketKey, ...]],
    tree_constraints: list[tuple[MarketKey, tuple[MarketKey, ...]]],
    parity_constraints: list[tuple[MarketKey, MarketKey, str]],
    cross_equiv_pairs: list[tuple[MarketKey, MarketKey]],
    assume_exhaustive: bool,
) -> tuple[float, int]:
    """Enumerate all 2^N assignments and return (min_payout, valid_count)."""
    constraints: list[Callable[[dict[MarketKey, int]], bool]] = []

    for markets in bucket_constraints:
        if assume_exhaustive:
            constraints.append(
                lambda a, mk=markets: sum(a[m] for m in mk) == 1
            )
        else:
            constraints.append(
                lambda a, mk=markets: sum(a[m] for m in mk) <= 1
            )

    for parent, children in tree_constraints:
        constraints.append(
            lambda a, p=parent, ch=children: sum(a[c] for c in ch) == a[p]
        )

    for left, right, relationship in parity_constraints:
        if relationship == "complement":
            constraints.append(lambda a, l=left, r=right: a[l] + a[r] == 1)
        else:
            constraints.append(lambda a, l=left, r=right: a[l] == a[r])

    for left, right in cross_equiv_pairs:
        constraints.append(lambda a, l=left, r=right: a[l] == a[r])

    ordered_markets = tuple(sorted(active_markets))
    valid_assignments = 0
    min_payout = math.inf
    total_states = 1 << len(ordered_markets)

    for state in range(total_states):
        assignment = {
            market: (state >> idx) & 1
            for idx, market in enumerate(ordered_markets)
        }
        if any(not c(assignment) for c in constraints):
            continue

        valid_assignments += 1
        payout = 0.0
        for leg in opportunity.legs:
            mkey = _key_from_ref(leg.venue, leg.market_id)
            outcome = assignment[mkey]
            payout += float(outcome if leg.side is Side.YES else 1 - outcome)

        if payout < min_payout:
            min_payout = payout

    return (min_payout if valid_assignments > 0 else 0.0), valid_assignments


# ---------------------------------------------------------------------------
# ILP solver (PuLP + CBC, handles N >> 14)
# ---------------------------------------------------------------------------


def _solve_ilp(
    *,
    active_markets: set[MarketKey],
    opportunity: ArbitrageOpportunity,
    bucket_constraints: list[tuple[MarketKey, ...]],
    tree_constraints: list[tuple[MarketKey, tuple[MarketKey, ...]]],
    parity_constraints: list[tuple[MarketKey, MarketKey, str]],
    cross_equiv_pairs: list[tuple[MarketKey, MarketKey]],
    assume_exhaustive: bool,
) -> tuple[float, int]:
    """Use PuLP ILP to find the minimum payout assignment.

    Returns (min_payout, 1) on success, (0.0, 0) if infeasible.
    The valid_assignments count is not exact (ILP returns one optimal
    solution, not the full enumeration), so we return 1 as a sentinel
    meaning "at least one valid assignment exists".
    """
    if pulp is None:
        return 0.0, 0

    # Build the ILP problem.
    prob = pulp.LpProblem("min_payout", pulp.LpMinimize)

    # Binary variables: one per market (0 or 1 outcome).
    ordered = sorted(active_markets)
    market_idx = {m: i for i, m in enumerate(ordered)}
    x = {m: pulp.LpVariable(f"x_{i}", cat="Binary") for m, i in market_idx.items()}

    # Objective: minimize payout over the opportunity legs.
    payout_expr = []
    for leg in opportunity.legs:
        mkey = _key_from_ref(leg.venue, leg.market_id)
        if mkey not in x:
            continue
        if leg.side is Side.YES:
            payout_expr.append(x[mkey])
        else:
            payout_expr.append(1 - x[mkey])
    prob += pulp.lpSum(payout_expr), "total_payout"

    # Constraints: bucket exclusivity.
    for ci, markets in enumerate(bucket_constraints):
        vs = [x[m] for m in markets if m in x]
        if not vs:
            continue
        if assume_exhaustive:
            prob += pulp.lpSum(vs) == 1, f"bucket_eq_{ci}"
        else:
            prob += pulp.lpSum(vs) <= 1, f"bucket_le_{ci}"

    # Constraints: event tree (sum of children == parent).
    for ci, (parent, children) in enumerate(tree_constraints):
        if parent not in x:
            continue
        child_vars = [x[c] for c in children if c in x]
        if not child_vars:
            continue
        prob += pulp.lpSum(child_vars) == x[parent], f"tree_{ci}"

    # Constraints: parity (complement or equivalent).
    for ci, (left, right, relationship) in enumerate(parity_constraints):
        if left not in x or right not in x:
            continue
        if relationship == "complement":
            prob += x[left] + x[right] == 1, f"parity_comp_{ci}"
        else:
            prob += x[left] == x[right], f"parity_equiv_{ci}"

    # Constraints: cross-venue equivalence.
    for ci, (left, right) in enumerate(cross_equiv_pairs):
        if left not in x or right not in x:
            continue
        prob += x[left] == x[right], f"cross_eq_{ci}"

    # Solve silently.
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if prob.status != pulp.constants.LpStatusOptimal:
        return 0.0, 0

    min_payout = pulp.value(prob.objective)
    if min_payout is None:
        return 0.0, 0

    return float(min_payout), 1


def _key_from_ref(venue: str, market_id: str) -> MarketKey:
    return venue.strip().lower(), market_id.strip().lower()


def _market_key_to_string(market: MarketKey) -> str:
    return f"{market[0]}:{market[1]}"


def _normalize_cluster_part(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    return normalized.strip("_")
