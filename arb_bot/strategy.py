from __future__ import annotations

import logging
import re
from datetime import datetime
from itertools import combinations
from typing import Any

from arb_bot.bucket_quality import BucketQualityModel, bucket_leg_count_map
from arb_bot.cross_mapping import CrossVenueMapping, VenueRef, load_cross_venue_mappings
from arb_bot.models import (
    ArbitrageOpportunity,
    BinaryQuote,
    ExecutionStyle,
    OpportunityKind,
    OpportunityLeg,
    Side,
)
from arb_bot.structural_rules import (
    EventTreeRule,
    ExclusiveBucketRule,
    MarketLegRef,
    ParityRule,
    StructuralRuleSet,
    load_structural_rules,
)

LOGGER = logging.getLogger(__name__)

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "will",
    "with",
}


def _normalize_text(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return " ".join(cleaned.split())


def _tokenize(text: str) -> set[str]:
    normalized = _normalize_text(text)
    tokens = {token for token in normalized.split() if token and token not in _STOPWORDS}
    return {token for token in tokens if len(token) > 1 or token.isdigit()}


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


class ArbitrageFinder:
    """Finds pair and structural binary arbitrage opportunities."""

    def __init__(
        self,
        min_net_edge_per_contract: float,
        enable_cross_venue: bool = True,
        cross_venue_min_match_score: float = 0.62,
        cross_venue_mapping_path: str | None = None,
        cross_venue_mapping_required: bool = False,
        enable_fuzzy_cross_venue_fallback: bool = True,
        enable_maker_estimates: bool = True,
        enable_structural_arb: bool = True,
        structural_rules_path: str | None = None,
        enable_bucket_quality_model: bool = True,
        bucket_quality_history_glob: str | None = "arb_bot/output/paper_*.csv",
        bucket_quality_history_max_files: int = 50,
        bucket_quality_min_observations: int = 8,
        bucket_quality_max_active_buckets: int = 0,
        bucket_quality_explore_fraction: float = 0.15,
        bucket_quality_prior_mean_realized_profit: float = 0.002,
        bucket_quality_prior_strength: float = 12.0,
        bucket_quality_min_score: float = -0.02,
        bucket_quality_leg_count_penalty: float = 0.00025,
        bucket_quality_live_update_interval: int = 25,
    ) -> None:
        self._min_net_edge_per_contract = min_net_edge_per_contract
        self._enable_cross_venue = enable_cross_venue
        self._cross_venue_min_match_score = cross_venue_min_match_score
        self._cross_venue_mapping_required = cross_venue_mapping_required
        self._enable_fuzzy_cross_venue_fallback = enable_fuzzy_cross_venue_fallback
        self._enable_maker_estimates = enable_maker_estimates
        self._enable_structural_arb = enable_structural_arb

        self._mappings: list[CrossVenueMapping] = load_cross_venue_mappings(cross_venue_mapping_path)
        if cross_venue_mapping_path and self._mappings:
            LOGGER.info("loaded %d cross-venue mappings from %s", len(self._mappings), cross_venue_mapping_path)

        self._structural_rules: StructuralRuleSet = load_structural_rules(structural_rules_path)
        if structural_rules_path and not self._structural_rules.is_empty:
            LOGGER.info("loaded structural rules from %s", structural_rules_path)

        self._bucket_quality_model: BucketQualityModel | None = None
        if self._enable_structural_arb and self._structural_rules.buckets:
            bucket_counts = bucket_leg_count_map(
                (bucket.group_id, len(bucket.legs))
                for bucket in self._structural_rules.buckets
            )
            self._bucket_quality_model = BucketQualityModel(
                bucket_leg_counts=bucket_counts,
                enabled=enable_bucket_quality_model,
                history_glob=bucket_quality_history_glob,
                history_max_files=bucket_quality_history_max_files,
                min_observations=bucket_quality_min_observations,
                max_active_buckets=bucket_quality_max_active_buckets,
                explore_fraction=bucket_quality_explore_fraction,
                prior_mean_realized_profit=bucket_quality_prior_mean_realized_profit,
                prior_strength=bucket_quality_prior_strength,
                min_score=bucket_quality_min_score,
                leg_count_penalty=bucket_quality_leg_count_penalty,
                live_update_interval=bucket_quality_live_update_interval,
            )
            LOGGER.info(self._bucket_quality_model.summary())

    def find(
        self,
        quotes: list[BinaryQuote],
        min_net_edge_override: float | None = None,
    ) -> list[ArbitrageOpportunity]:
        return self.find_by_kind(quotes, min_net_edge_override=min_net_edge_override, kinds=None)

    def find_by_kind(
        self,
        quotes: list[BinaryQuote],
        min_net_edge_override: float | None = None,
        kinds: set[OpportunityKind] | None = None,
    ) -> list[ArbitrageOpportunity]:
        threshold = self._min_net_edge_per_contract if min_net_edge_override is None else min_net_edge_override
        selected_kinds = set(kinds) if kinds is not None else set(OpportunityKind)

        opportunities: list[ArbitrageOpportunity] = []
        if OpportunityKind.INTRA_VENUE in selected_kinds:
            opportunities.extend(self._find_intra_venue(quotes, threshold))

        if OpportunityKind.CROSS_VENUE in selected_kinds and self._enable_cross_venue:
            opportunities.extend(self._find_cross_venue(quotes, threshold))

        structural_kinds = {
            OpportunityKind.STRUCTURAL_BUCKET,
            OpportunityKind.STRUCTURAL_EVENT_TREE,
            OpportunityKind.STRUCTURAL_PARITY,
        }
        if selected_kinds & structural_kinds and self._enable_structural_arb and not self._structural_rules.is_empty:
            structural = self._find_structural(quotes, threshold)
            opportunities.extend(opp for opp in structural if opp.kind in selected_kinds)

        opportunities.sort(
            key=lambda opp: (
                opp.net_edge_per_contract,
                opp.match_score,
                opp.gross_edge_per_contract,
            ),
            reverse=True,
        )
        return opportunities

    def find_near(self, quotes: list[BinaryQuote], max_total_cost: float) -> list[ArbitrageOpportunity]:
        threshold = 1.0 - max_total_cost
        near = self.find(quotes, min_net_edge_override=threshold)
        # Near-arb scoring is meaningful for $1 payout structures.
        return [opp for opp in near if abs(opp.payout_per_contract - 1.0) < 1e-9]

    def coverage_snapshot(self, quotes: list[BinaryQuote]) -> dict[str, int]:
        by_venue: dict[str, list[BinaryQuote]] = {}
        for quote in quotes:
            by_venue.setdefault(quote.venue, []).append(quote)

        mapping_k_refs_total: set[str] = set()
        mapping_p_refs_total: set[str] = set()
        mapping_k_refs_seen: set[str] = set()
        mapping_p_refs_seen: set[str] = set()
        mapping_pairs_covered = 0
        for mapping in self._mappings:
            kalshi_identity = self._mapping_ref_identity(mapping.kalshi)
            polymarket_identity = self._mapping_ref_identity(mapping.polymarket)
            mapping_k_refs_total.add(kalshi_identity)
            mapping_p_refs_total.add(polymarket_identity)

            kalshi_quote = self._lookup_mapped_quote("kalshi", mapping.kalshi, by_venue.get("kalshi", []))
            polymarket_quote = self._lookup_mapped_quote("polymarket", mapping.polymarket, by_venue.get("polymarket", []))
            if kalshi_quote is not None:
                mapping_k_refs_seen.add(kalshi_identity)
            if polymarket_quote is not None:
                mapping_p_refs_seen.add(polymarket_identity)
            if kalshi_quote is not None and polymarket_quote is not None:
                mapping_pairs_covered += 1

        quote_keys = {(quote.venue.lower(), quote.market_id.lower()) for quote in quotes}
        parity_rules_total = len(self._structural_rules.parity_checks)
        parity_rules_covered = 0
        parity_markets_total: set[tuple[str, str]] = set()
        parity_markets_seen: set[tuple[str, str]] = set()
        for rule in self._structural_rules.parity_checks:
            parity_markets_total.add(rule.left.key)
            parity_markets_total.add(rule.right.key)
            if rule.left.key in quote_keys:
                parity_markets_seen.add(rule.left.key)
            if rule.right.key in quote_keys:
                parity_markets_seen.add(rule.right.key)
            if rule.left.key in quote_keys and rule.right.key in quote_keys:
                parity_rules_covered += 1

        return {
            "cross_mapping_pairs_total": len(self._mappings),
            "cross_mapping_pairs_covered": mapping_pairs_covered,
            "cross_mapping_kalshi_refs_total": len(mapping_k_refs_total),
            "cross_mapping_kalshi_refs_seen": len(mapping_k_refs_seen),
            "cross_mapping_polymarket_refs_total": len(mapping_p_refs_total),
            "cross_mapping_polymarket_refs_seen": len(mapping_p_refs_seen),
            "structural_parity_rules_total": parity_rules_total,
            "structural_parity_rules_covered": parity_rules_covered,
            "structural_parity_markets_total": len(parity_markets_total),
            "structural_parity_markets_seen": len(parity_markets_seen),
        }

    def _enabled_styles(self) -> tuple[ExecutionStyle, ...]:
        if self._enable_maker_estimates:
            return (ExecutionStyle.TAKER, ExecutionStyle.MAKER_ESTIMATE)
        return (ExecutionStyle.TAKER,)

    def _find_intra_venue(
        self,
        quotes: list[BinaryQuote],
        min_threshold: float,
    ) -> list[ArbitrageOpportunity]:
        opportunities: list[ArbitrageOpportunity] = []

        for quote in quotes:
            for style in self._enabled_styles():
                yes_leg = self._leg_from_quote(quote, Side.YES, style)
                no_leg = self._leg_from_quote(quote, Side.NO, style)
                if yes_leg is None or no_leg is None:
                    continue

                opportunity = self._build_opportunity(
                    kind=OpportunityKind.INTRA_VENUE,
                    execution_style=style,
                    left=yes_leg,
                    right=no_leg,
                    fee_per_contract=quote.fee_per_contract,
                    observed_at=quote.observed_at,
                    match_key=f"{quote.venue}:{quote.market_id}",
                    match_score=1.0,
                    metadata={"market_text": quote.market_text},
                    min_threshold=min_threshold,
                )
                if opportunity is not None:
                    opportunities.append(opportunity)

        return opportunities

    def _find_cross_venue(
        self,
        quotes: list[BinaryQuote],
        min_threshold: float,
    ) -> list[ArbitrageOpportunity]:
        opportunities: list[ArbitrageOpportunity] = []

        if self._mappings:
            opportunities.extend(self._find_cross_venue_from_mappings(quotes, min_threshold))
            if self._cross_venue_mapping_required:
                return opportunities

        if self._enable_fuzzy_cross_venue_fallback:
            opportunities.extend(self._find_cross_venue_fuzzy(quotes, min_threshold))

        return opportunities

    def _find_cross_venue_from_mappings(
        self,
        quotes: list[BinaryQuote],
        min_threshold: float,
    ) -> list[ArbitrageOpportunity]:
        by_venue: dict[str, list[BinaryQuote]] = {}
        for quote in quotes:
            by_venue.setdefault(quote.venue, []).append(quote)

        opportunities: list[ArbitrageOpportunity] = []
        for mapping in self._mappings:
            kalshi_quote = self._lookup_mapped_quote("kalshi", mapping.kalshi, by_venue.get("kalshi", []))
            polymarket_quote = self._lookup_mapped_quote("polymarket", mapping.polymarket, by_venue.get("polymarket", []))

            if kalshi_quote is None or polymarket_quote is None:
                continue

            opportunities.extend(
                self._cross_pair_opportunities(
                    left_quote=kalshi_quote,
                    right_quote=polymarket_quote,
                    match_key=mapping.group_id,
                    match_score=1.0,
                    min_threshold=min_threshold,
                )
            )

        return opportunities

    def _find_cross_venue_fuzzy(
        self,
        quotes: list[BinaryQuote],
        min_threshold: float,
    ) -> list[ArbitrageOpportunity]:
        descriptors = []
        for quote in quotes:
            text = quote.market_text
            tokens = _tokenize(text)
            if not tokens:
                continue
            descriptors.append({"quote": quote, "text": text, "tokens": tokens})

        opportunities: list[ArbitrageOpportunity] = []

        for left_desc, right_desc in combinations(descriptors, 2):
            left_quote: BinaryQuote = left_desc["quote"]
            right_quote: BinaryQuote = right_desc["quote"]

            if left_quote.venue == right_quote.venue:
                continue

            match_score = _jaccard_similarity(left_desc["tokens"], right_desc["tokens"])
            if match_score < self._cross_venue_min_match_score:
                continue

            match_key = _normalize_text(left_desc["text"]) or _normalize_text(right_desc["text"])
            if not match_key:
                match_key = f"{left_quote.market_id}:{right_quote.market_id}"

            opportunities.extend(
                self._cross_pair_opportunities(
                    left_quote=left_quote,
                    right_quote=right_quote,
                    match_key=match_key,
                    match_score=match_score,
                    min_threshold=min_threshold,
                )
            )

        return opportunities

    def _cross_pair_opportunities(
        self,
        left_quote: BinaryQuote,
        right_quote: BinaryQuote,
        match_key: str,
        match_score: float,
        min_threshold: float,
    ) -> list[ArbitrageOpportunity]:
        opportunities: list[ArbitrageOpportunity] = []

        observed_at = max(left_quote.observed_at, right_quote.observed_at)
        pair_fee = left_quote.fee_per_contract + right_quote.fee_per_contract

        for style in self._enabled_styles():
            yes_left = self._leg_from_quote(left_quote, Side.YES, style)
            no_left = self._leg_from_quote(left_quote, Side.NO, style)
            yes_right = self._leg_from_quote(right_quote, Side.YES, style)
            no_right = self._leg_from_quote(right_quote, Side.NO, style)

            if yes_left is None or no_left is None or yes_right is None or no_right is None:
                continue

            first = self._build_opportunity(
                kind=OpportunityKind.CROSS_VENUE,
                execution_style=style,
                left=yes_left,
                right=no_right,
                fee_per_contract=pair_fee,
                observed_at=observed_at,
                match_key=match_key,
                match_score=match_score,
                metadata={
                    "left_text": left_quote.market_text,
                    "right_text": right_quote.market_text,
                },
                min_threshold=min_threshold,
            )
            if first is not None:
                opportunities.append(first)

            second = self._build_opportunity(
                kind=OpportunityKind.CROSS_VENUE,
                execution_style=style,
                left=no_left,
                right=yes_right,
                fee_per_contract=pair_fee,
                observed_at=observed_at,
                match_key=match_key,
                match_score=match_score,
                metadata={
                    "left_text": left_quote.market_text,
                    "right_text": right_quote.market_text,
                },
                min_threshold=min_threshold,
            )
            if second is not None:
                opportunities.append(second)

        return opportunities

    def _find_structural(
        self,
        quotes: list[BinaryQuote],
        min_threshold: float,
    ) -> list[ArbitrageOpportunity]:
        lookup: dict[tuple[str, str], BinaryQuote] = {
            (quote.venue.lower(), quote.market_id.lower()): quote for quote in quotes
        }
        opportunities: list[ArbitrageOpportunity] = []

        for style in self._enabled_styles():
            opportunities.extend(self._find_structural_buckets(lookup, style, min_threshold))
            opportunities.extend(self._find_structural_event_trees(lookup, style, min_threshold))
            opportunities.extend(self._find_structural_parity(lookup, style, min_threshold))

        return opportunities

    def _find_structural_buckets(
        self,
        lookup: dict[tuple[str, str], BinaryQuote],
        style: ExecutionStyle,
        min_threshold: float,
    ) -> list[ArbitrageOpportunity]:
        opportunities: list[ArbitrageOpportunity] = []
        for bucket in self._structural_rules.buckets:
            if self._bucket_quality_model is not None and not self._bucket_quality_model.should_enable_bucket(bucket.group_id):
                continue
            resolved = self._resolve_refs(bucket.legs, lookup, style)
            if resolved is None:
                continue

            legs, source_quotes = resolved
            quality_score = (
                self._bucket_quality_model.score_for(bucket.group_id)
                if self._bucket_quality_model is not None
                else None
            )
            metadata: dict[str, Any] = {
                "structural_class": "mutually_exclusive_bucket",
                "bucket_group_id": bucket.group_id,
            }
            if quality_score is not None:
                metadata["bucket_quality_score"] = quality_score
            opportunity = self._build_structural_opportunity(
                kind=OpportunityKind.STRUCTURAL_BUCKET,
                execution_style=style,
                legs=legs,
                source_quotes=source_quotes,
                payout_per_contract=bucket.payout_per_contract,
                match_key=bucket.group_id,
                match_score=1.0,
                metadata=metadata,
                min_threshold=min_threshold,
            )
            if opportunity is not None:
                opportunities.append(opportunity)
        return opportunities

    def _find_structural_event_trees(
        self,
        lookup: dict[tuple[str, str], BinaryQuote],
        style: ExecutionStyle,
        min_threshold: float,
    ) -> list[ArbitrageOpportunity]:
        opportunities: list[ArbitrageOpportunity] = []
        for rule in self._structural_rules.event_trees:
            opportunities.extend(self._event_tree_opportunities_for_rule(rule, lookup, style, min_threshold))
        return opportunities

    def _event_tree_opportunities_for_rule(
        self,
        rule: EventTreeRule,
        lookup: dict[tuple[str, str], BinaryQuote],
        style: ExecutionStyle,
        min_threshold: float,
    ) -> list[ArbitrageOpportunity]:
        opportunities: list[ArbitrageOpportunity] = []

        basket_one_refs = (self._flip_ref(rule.parent), *rule.children)
        basket_one = self._resolve_refs(basket_one_refs, lookup, style)
        if basket_one is not None:
            legs, source_quotes = basket_one
            opp = self._build_structural_opportunity(
                kind=OpportunityKind.STRUCTURAL_EVENT_TREE,
                execution_style=style,
                legs=legs,
                source_quotes=source_quotes,
                payout_per_contract=1.0,
                match_key=f"{rule.group_id}:parent_no_children_yes",
                match_score=1.0,
                metadata={"structural_class": "event_tree_parent_no_children_yes"},
                min_threshold=min_threshold,
            )
            if opp is not None:
                opportunities.append(opp)

        basket_two_refs = (rule.parent, *[self._flip_ref(child) for child in rule.children])
        basket_two = self._resolve_refs(basket_two_refs, lookup, style)
        if basket_two is not None:
            legs, source_quotes = basket_two
            opp = self._build_structural_opportunity(
                kind=OpportunityKind.STRUCTURAL_EVENT_TREE,
                execution_style=style,
                legs=legs,
                source_quotes=source_quotes,
                payout_per_contract=float(len(rule.children)),
                match_key=f"{rule.group_id}:parent_yes_children_no",
                match_score=1.0,
                metadata={"structural_class": "event_tree_parent_yes_children_no"},
                min_threshold=min_threshold,
            )
            if opp is not None:
                opportunities.append(opp)

        return opportunities

    def _find_structural_parity(
        self,
        lookup: dict[tuple[str, str], BinaryQuote],
        style: ExecutionStyle,
        min_threshold: float,
    ) -> list[ArbitrageOpportunity]:
        opportunities: list[ArbitrageOpportunity] = []
        for rule in self._structural_rules.parity_checks:
            opportunities.extend(self._parity_opportunities_for_rule(rule, lookup, style, min_threshold))
        return opportunities

    def _parity_opportunities_for_rule(
        self,
        rule: ParityRule,
        lookup: dict[tuple[str, str], BinaryQuote],
        style: ExecutionStyle,
        min_threshold: float,
    ) -> list[ArbitrageOpportunity]:
        opportunities: list[ArbitrageOpportunity] = []

        if rule.relationship == "complement":
            pairings = (
                (rule.left, rule.right, "left_yes_right_yes"),
                (self._flip_ref(rule.left), self._flip_ref(rule.right), "left_no_right_no"),
            )
        else:
            pairings = (
                (rule.left, self._flip_ref(rule.right), "left_yes_right_no"),
                (self._flip_ref(rule.left), rule.right, "left_no_right_yes"),
            )

        for left_ref, right_ref, suffix in pairings:
            resolved = self._resolve_refs((left_ref, right_ref), lookup, style)
            if resolved is None:
                continue
            legs, source_quotes = resolved
            opp = self._build_structural_opportunity(
                kind=OpportunityKind.STRUCTURAL_PARITY,
                execution_style=style,
                legs=legs,
                source_quotes=source_quotes,
                payout_per_contract=1.0,
                match_key=f"{rule.group_id}:{suffix}",
                match_score=1.0,
                metadata={"structural_class": f"parity_{rule.relationship}"},
                min_threshold=min_threshold,
            )
            if opp is not None:
                opportunities.append(opp)

        return opportunities

    def _build_structural_opportunity(
        self,
        kind: OpportunityKind,
        execution_style: ExecutionStyle,
        legs: tuple[OpportunityLeg, ...],
        source_quotes: tuple[BinaryQuote, ...],
        payout_per_contract: float,
        match_key: str,
        match_score: float,
        metadata: dict[str, str],
        min_threshold: float,
    ) -> ArbitrageOpportunity | None:
        observed_at = max(quote.observed_at for quote in source_quotes)
        fee_per_contract = sum(quote.fee_per_contract for quote in source_quotes)
        return self._build_opportunity_from_legs(
            kind=kind,
            execution_style=execution_style,
            legs=legs,
            fee_per_contract=fee_per_contract,
            payout_per_contract=payout_per_contract,
            observed_at=observed_at,
            match_key=match_key,
            match_score=match_score,
            metadata=metadata,
            min_threshold=min_threshold,
        )

    def _resolve_refs(
        self,
        refs: tuple[MarketLegRef, ...] | list[MarketLegRef],
        lookup: dict[tuple[str, str], BinaryQuote],
        style: ExecutionStyle,
    ) -> tuple[tuple[OpportunityLeg, ...], tuple[BinaryQuote, ...]] | None:
        legs: list[OpportunityLeg] = []
        source_quotes: list[BinaryQuote] = []
        for ref in refs:
            quote = lookup.get(ref.key)
            if quote is None:
                return None
            leg = self._leg_from_quote(quote, ref.side, style)
            if leg is None:
                return None
            legs.append(leg)
            source_quotes.append(quote)
        return tuple(legs), tuple(source_quotes)

    def _leg_from_quote(
        self,
        quote: BinaryQuote,
        side: Side,
        execution_style: ExecutionStyle,
    ) -> OpportunityLeg | None:
        if execution_style is ExecutionStyle.TAKER:
            if side is Side.YES:
                price, size = quote.yes_buy_price, quote.yes_buy_size
            else:
                price, size = quote.no_buy_price, quote.no_buy_size
        else:
            if side is Side.YES:
                price, size = quote.yes_maker_buy_price, quote.yes_maker_buy_size
            else:
                price, size = quote.no_maker_buy_price, quote.no_maker_buy_size

        if price is None:
            return None
        if price < 0 or price > 1:
            return None
        if size <= 0:
            return None

        return OpportunityLeg(
            venue=quote.venue,
            market_id=quote.market_id,
            side=side,
            buy_price=price,
            buy_size=size,
            metadata=quote.metadata,
        )

    def _build_opportunity(
        self,
        kind: OpportunityKind,
        execution_style: ExecutionStyle,
        left: OpportunityLeg,
        right: OpportunityLeg,
        fee_per_contract: float,
        observed_at: datetime,
        match_key: str,
        match_score: float,
        metadata: dict[str, str],
        min_threshold: float,
    ) -> ArbitrageOpportunity | None:
        return self._build_opportunity_from_legs(
            kind=kind,
            execution_style=execution_style,
            legs=(left, right),
            fee_per_contract=fee_per_contract,
            payout_per_contract=1.0,
            observed_at=observed_at,
            match_key=match_key,
            match_score=match_score,
            metadata=metadata,
            min_threshold=min_threshold,
        )

    def _build_opportunity_from_legs(
        self,
        kind: OpportunityKind,
        execution_style: ExecutionStyle,
        legs: tuple[OpportunityLeg, ...],
        fee_per_contract: float,
        payout_per_contract: float,
        observed_at: datetime,
        match_key: str,
        match_score: float,
        metadata: dict[str, str],
        min_threshold: float,
    ) -> ArbitrageOpportunity | None:
        if len(legs) < 2:
            return None

        gross_edge = payout_per_contract - sum(leg.buy_price for leg in legs)
        net_edge = gross_edge - fee_per_contract
        if net_edge < min_threshold:
            return None

        return ArbitrageOpportunity(
            kind=kind,
            execution_style=execution_style,
            legs=legs,
            gross_edge_per_contract=gross_edge,
            net_edge_per_contract=net_edge,
            fee_per_contract=fee_per_contract,
            observed_at=observed_at,
            match_key=match_key,
            match_score=match_score,
            payout_per_contract=payout_per_contract,
            metadata=metadata,
        )

    def observe_bucket_decisions(self, decisions: list[Any]) -> None:
        if self._bucket_quality_model is None or not decisions:
            return

        for decision in decisions:
            opportunity = getattr(decision, "opportunity", None)
            if opportunity is None or opportunity.kind is not OpportunityKind.STRUCTURAL_BUCKET:
                continue
            group_id = str(getattr(opportunity, "match_key", "") or "").strip()
            if not group_id:
                continue
            metrics = getattr(decision, "metrics", {}) or {}
            self._bucket_quality_model.observe_decision(
                group_id=group_id,
                action=str(getattr(decision, "action", "") or "").strip().lower(),
                detected_edge_per_contract=float(metrics.get("detected_edge_per_contract") or 0.0),
                fill_probability=float(metrics.get("fill_probability") or 0.0),
                expected_realized_profit=float(metrics.get("expected_realized_profit") or 0.0),
                realized_profit=(
                    float(metrics.get("realized_profit"))
                    if metrics.get("realized_profit") is not None
                    else None
                ),
                expected_slippage_per_contract=float(metrics.get("expected_slippage_per_contract") or 0.0),
                execution_latency_ms=(
                    float(metrics.get("execution_latency_ms"))
                    if metrics.get("execution_latency_ms") is not None
                    else None
                ),
            )

    def _lookup_mapped_quote(
        self,
        venue: str,
        mapping_ref: VenueRef,
        venue_quotes: list[BinaryQuote],
    ) -> BinaryQuote | None:
        target = mapping_ref.value.strip().lower()
        if not target:
            return None

        for quote in venue_quotes:
            if quote.venue != venue:
                continue
            if self._quote_matches_ref(quote, mapping_ref, target):
                return quote

        return None

    @staticmethod
    def _quote_matches_ref(quote: BinaryQuote, mapping_ref: VenueRef, target: str) -> bool:
        key = mapping_ref.key.strip().lower()

        if key in {"kalshi_market_id", "kalshi_ticker", "polymarket_market_id", "polymarket_condition_id"}:
            return quote.market_id.strip().lower() == target

        if key == "kalshi_event_ticker":
            value = str(quote.metadata.get("event_ticker") or "").strip().lower()
            return value == target

        if key == "polymarket_slug":
            value = str(quote.metadata.get("slug") or "").strip().lower()
            return value == target

        return False

    @staticmethod
    def _mapping_ref_identity(mapping_ref: VenueRef) -> str:
        return f"{mapping_ref.key.strip().lower()}:{mapping_ref.value.strip().lower()}"

    @staticmethod
    def _flip_ref(ref: MarketLegRef) -> MarketLegRef:
        flipped_side = Side.NO if ref.side is Side.YES else Side.YES
        return MarketLegRef(venue=ref.venue, market_id=ref.market_id, side=flipped_side)


# ---------------------------------------------------------------------------
# Rust dispatch note for strategy module.
#
# The Rust find_opportunities() function exists in arb_engine_rs but uses
# JSON serialization for the complex input types (quotes, rules, mappings).
# For small workloads, JSON overhead negates Rust's compute advantage.
# The Rust strategy port shows benefit only at scale (200+ quotes with
# fuzzy matching).
#
# Future optimization: use PyO3 native types instead of JSON for the
# strategy function to eliminate serialization overhead.
#
# For now, binary_math functions (used internally by strategy) benefit
# from Rust dispatch via ARB_USE_RUST_BINARY_MATH=1.
# ---------------------------------------------------------------------------

_RUST_ACTIVE = False  # Strategy dispatch not yet wired (see note above).
