"""Tests for Phase 9B: ForecastEx cross-venue market mapping.

Validates that the cross-venue mapping system supports 3-venue mappings
(Kalshi + Polymarket + ForecastEx), correctly generates venue pair
combinations, and that the strategy layer finds cross-venue opportunities
across all venue pairs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from arb_bot.cross_mapping import (
    CrossVenueMapping,
    VenuePair,
    VenueRef,
    all_venue_refs,
    load_cross_venue_mappings,
    venue_pairs,
)
from arb_bot.cross_mapping_generator import (
    _build_candidates,
    _build_forecastex_text,
    _forecastex_candidate,
    generate_cross_venue_mapping_rows,
)
from arb_bot.models import (
    ArbitrageOpportunity,
    BinaryQuote,
    ExecutionStyle,
    OpportunityKind,
)
from arb_bot.strategy import ArbitrageFinder


# ===================================================================
# CrossVenueMapping with optional forecastex
# ===================================================================


class TestCrossVenueMappingDataclass:
    """Tests for the extended CrossVenueMapping dataclass."""

    def test_two_venue_mapping_backward_compatible(self) -> None:
        mapping = CrossVenueMapping(
            group_id="test",
            kalshi=VenueRef(key="kalshi_market_id", value="K1"),
            polymarket=VenueRef(key="polymarket_market_id", value="P1"),
        )
        assert mapping.forecastex is None
        assert mapping.kalshi.value == "K1"
        assert mapping.polymarket.value == "P1"

    def test_three_venue_mapping(self) -> None:
        mapping = CrossVenueMapping(
            group_id="test",
            kalshi=VenueRef(key="kalshi_market_id", value="K1"),
            polymarket=VenueRef(key="polymarket_market_id", value="P1"),
            forecastex=VenueRef(key="forecastex_market_id", value="F1"),
        )
        assert mapping.forecastex is not None
        assert mapping.forecastex.value == "F1"

    def test_mapping_is_frozen(self) -> None:
        mapping = CrossVenueMapping(
            group_id="test",
            kalshi=VenueRef(key="kalshi_market_id", value="K1"),
            polymarket=VenueRef(key="polymarket_market_id", value="P1"),
        )
        with pytest.raises(AttributeError):
            mapping.group_id = "changed"  # type: ignore[misc]


# ===================================================================
# venue_pairs() generation
# ===================================================================


class TestVenuePairs:
    """Tests for generating venue pair combinations from a mapping."""

    def test_two_venue_produces_one_pair(self) -> None:
        mapping = CrossVenueMapping(
            group_id="g1",
            kalshi=VenueRef(key="kalshi_market_id", value="K1"),
            polymarket=VenueRef(key="polymarket_market_id", value="P1"),
        )
        pairs = venue_pairs(mapping)
        assert len(pairs) == 1
        assert pairs[0].left_venue == "kalshi"
        assert pairs[0].right_venue == "polymarket"
        assert pairs[0].group_id == "g1"

    def test_three_venue_produces_three_pairs(self) -> None:
        mapping = CrossVenueMapping(
            group_id="g1",
            kalshi=VenueRef(key="kalshi_market_id", value="K1"),
            polymarket=VenueRef(key="polymarket_market_id", value="P1"),
            forecastex=VenueRef(key="forecastex_market_id", value="F1"),
        )
        pairs = venue_pairs(mapping)
        assert len(pairs) == 3
        venue_combos = {(p.left_venue, p.right_venue) for p in pairs}
        assert ("kalshi", "polymarket") in venue_combos
        assert ("kalshi", "forecastex") in venue_combos
        assert ("polymarket", "forecastex") in venue_combos

    def test_empty_kalshi_ref_excluded(self) -> None:
        """A mapping where kalshi ref is empty (e.g. forecastex+polymarket only)."""
        mapping = CrossVenueMapping(
            group_id="g1",
            kalshi=VenueRef(key="kalshi_market_id", value=""),
            polymarket=VenueRef(key="polymarket_market_id", value="P1"),
            forecastex=VenueRef(key="forecastex_market_id", value="F1"),
        )
        pairs = venue_pairs(mapping)
        assert len(pairs) == 1
        assert pairs[0].left_venue == "polymarket"
        assert pairs[0].right_venue == "forecastex"

    def test_empty_polymarket_ref_excluded(self) -> None:
        """A mapping where polymarket ref is empty (kalshi+forecastex only)."""
        mapping = CrossVenueMapping(
            group_id="g1",
            kalshi=VenueRef(key="kalshi_market_id", value="K1"),
            polymarket=VenueRef(key="polymarket_market_id", value=""),
            forecastex=VenueRef(key="forecastex_market_id", value="F1"),
        )
        pairs = venue_pairs(mapping)
        assert len(pairs) == 1
        assert pairs[0].left_venue == "kalshi"
        assert pairs[0].right_venue == "forecastex"

    def test_all_empty_produces_no_pairs(self) -> None:
        mapping = CrossVenueMapping(
            group_id="g1",
            kalshi=VenueRef(key="kalshi_market_id", value=""),
            polymarket=VenueRef(key="polymarket_market_id", value=""),
        )
        pairs = venue_pairs(mapping)
        assert len(pairs) == 0

    def test_pair_refs_correct(self) -> None:
        mapping = CrossVenueMapping(
            group_id="g1",
            kalshi=VenueRef(key="kalshi_market_id", value="KX-CPI"),
            polymarket=VenueRef(key="polymarket_market_id", value="0xabc"),
            forecastex=VenueRef(key="forecastex_market_id", value="FX-CPI"),
        )
        pairs = venue_pairs(mapping)
        kp = [p for p in pairs if p.left_venue == "kalshi" and p.right_venue == "polymarket"]
        kf = [p for p in pairs if p.left_venue == "kalshi" and p.right_venue == "forecastex"]
        pf = [p for p in pairs if p.left_venue == "polymarket" and p.right_venue == "forecastex"]
        assert len(kp) == 1 and kp[0].left_ref.value == "KX-CPI" and kp[0].right_ref.value == "0xabc"
        assert len(kf) == 1 and kf[0].left_ref.value == "KX-CPI" and kf[0].right_ref.value == "FX-CPI"
        assert len(pf) == 1 and pf[0].left_ref.value == "0xabc" and pf[0].right_ref.value == "FX-CPI"


# ===================================================================
# all_venue_refs() iterator
# ===================================================================


class TestAllVenueRefs:
    """Tests for the all_venue_refs iterator."""

    def test_two_venue_yields_two(self) -> None:
        mapping = CrossVenueMapping(
            group_id="g1",
            kalshi=VenueRef(key="kalshi_market_id", value="K1"),
            polymarket=VenueRef(key="polymarket_market_id", value="P1"),
        )
        refs = list(all_venue_refs(mapping))
        assert len(refs) == 2
        venues = [v for v, _ in refs]
        assert "kalshi" in venues
        assert "polymarket" in venues

    def test_three_venue_yields_three(self) -> None:
        mapping = CrossVenueMapping(
            group_id="g1",
            kalshi=VenueRef(key="kalshi_market_id", value="K1"),
            polymarket=VenueRef(key="polymarket_market_id", value="P1"),
            forecastex=VenueRef(key="forecastex_market_id", value="F1"),
        )
        refs = list(all_venue_refs(mapping))
        assert len(refs) == 3
        venues = [v for v, _ in refs]
        assert "forecastex" in venues

    def test_empty_ref_excluded(self) -> None:
        mapping = CrossVenueMapping(
            group_id="g1",
            kalshi=VenueRef(key="kalshi_market_id", value=""),
            polymarket=VenueRef(key="polymarket_market_id", value="P1"),
            forecastex=VenueRef(key="forecastex_market_id", value="F1"),
        )
        refs = list(all_venue_refs(mapping))
        assert len(refs) == 2
        venues = [v for v, _ in refs]
        assert "kalshi" not in venues


# ===================================================================
# CSV loading with forecastex column
# ===================================================================


class TestCSVLoadingThreeVenue:
    """Tests for loading 3-venue CSV files."""

    def test_load_two_venue_csv_backward_compatible(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "map.csv"
        csv_path.write_text(
            "group_id,kalshi_market_id,polymarket_market_id\n"
            "g1,K1,P1\n",
            encoding="utf-8",
        )
        mappings = load_cross_venue_mappings(str(csv_path))
        assert len(mappings) == 1
        assert mappings[0].forecastex is None
        assert mappings[0].kalshi.value == "K1"
        assert mappings[0].polymarket.value == "P1"

    def test_load_three_venue_csv(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "map.csv"
        csv_path.write_text(
            "group_id,kalshi_market_id,polymarket_market_id,forecastex_market_id\n"
            "g1,K1,P1,F1\n",
            encoding="utf-8",
        )
        mappings = load_cross_venue_mappings(str(csv_path))
        assert len(mappings) == 1
        assert mappings[0].forecastex is not None
        assert mappings[0].forecastex.value == "F1"

    def test_load_csv_with_missing_forecastex(self, tmp_path: Path) -> None:
        """forecastex column present but value empty."""
        csv_path = tmp_path / "map.csv"
        csv_path.write_text(
            "group_id,kalshi_market_id,polymarket_market_id,forecastex_market_id\n"
            "g1,K1,P1,\n",
            encoding="utf-8",
        )
        mappings = load_cross_venue_mappings(str(csv_path))
        assert len(mappings) == 1
        # forecastex should be None since value is empty.
        assert mappings[0].forecastex is None

    def test_load_csv_forecastex_plus_kalshi_no_polymarket(self, tmp_path: Path) -> None:
        """Row with kalshi + forecastex but no polymarket."""
        csv_path = tmp_path / "map.csv"
        csv_path.write_text(
            "group_id,kalshi_market_id,polymarket_market_id,forecastex_market_id\n"
            "g1,K1,,F1\n",
            encoding="utf-8",
        )
        mappings = load_cross_venue_mappings(str(csv_path))
        assert len(mappings) == 1
        assert mappings[0].kalshi.value == "K1"
        assert mappings[0].polymarket.value == ""  # Synthetic empty ref
        assert mappings[0].forecastex is not None
        assert mappings[0].forecastex.value == "F1"
        # Should produce 1 pair: kalshi ↔ forecastex
        pairs = venue_pairs(mappings[0])
        assert len(pairs) == 1
        assert pairs[0].left_venue == "kalshi"
        assert pairs[0].right_venue == "forecastex"

    def test_load_csv_only_one_venue_skipped(self, tmp_path: Path) -> None:
        """Row with only one venue should be skipped."""
        csv_path = tmp_path / "map.csv"
        csv_path.write_text(
            "group_id,kalshi_market_id,polymarket_market_id,forecastex_market_id\n"
            "g1,K1,,\n",
            encoding="utf-8",
        )
        mappings = load_cross_venue_mappings(str(csv_path))
        assert len(mappings) == 0

    def test_load_csv_forecastex_symbol_key(self, tmp_path: Path) -> None:
        """Use forecastex_symbol column key."""
        csv_path = tmp_path / "map.csv"
        csv_path.write_text(
            "group_id,kalshi_market_id,polymarket_market_id,forecastex_symbol\n"
            "g1,K1,P1,FEDFUNDS-MAR26\n",
            encoding="utf-8",
        )
        mappings = load_cross_venue_mappings(str(csv_path))
        assert len(mappings) == 1
        assert mappings[0].forecastex is not None
        assert mappings[0].forecastex.key == "forecastex_symbol"
        assert mappings[0].forecastex.value == "FEDFUNDS-MAR26"

    def test_load_csv_multiple_rows_mixed(self, tmp_path: Path) -> None:
        """Mix of 2-venue and 3-venue rows."""
        csv_path = tmp_path / "map.csv"
        csv_path.write_text(
            "group_id,kalshi_market_id,polymarket_market_id,forecastex_market_id\n"
            "g1,K1,P1,F1\n"
            "g2,K2,P2,\n"
            "g3,K3,,F3\n",
            encoding="utf-8",
        )
        mappings = load_cross_venue_mappings(str(csv_path))
        assert len(mappings) == 3

        # g1 is 3-venue
        assert mappings[0].forecastex is not None
        assert len(venue_pairs(mappings[0])) == 3

        # g2 is 2-venue (K+P)
        assert mappings[1].forecastex is None
        assert len(venue_pairs(mappings[1])) == 1

        # g3 is 2-venue (K+F)
        assert mappings[2].forecastex is not None
        assert len(venue_pairs(mappings[2])) == 1

    def test_nonexistent_path_returns_empty(self) -> None:
        mappings = load_cross_venue_mappings("/nonexistent/path.csv")
        assert mappings == []

    def test_none_path_returns_empty(self) -> None:
        mappings = load_cross_venue_mappings(None)
        assert mappings == []


# ===================================================================
# Strategy cross-venue detection with 3 venues
# ===================================================================


def _make_quote(
    venue: str,
    market_id: str,
    yes_buy: float = 0.45,
    no_buy: float = 0.60,
    **metadata: Any,
) -> BinaryQuote:
    return BinaryQuote(
        venue=venue,
        market_id=market_id,
        yes_buy_price=yes_buy,
        no_buy_price=no_buy,
        yes_buy_size=100,
        no_buy_size=100,
        fee_per_contract=0.0,
        metadata=metadata,
    )


class TestStrategyThreeVenueDetection:
    """Tests for cross-venue opportunity detection with 3 venues."""

    def test_kalshi_polymarket_pair_from_mapping(self, tmp_path: Path) -> None:
        """Standard 2-venue K↔P detection still works."""
        csv_path = tmp_path / "map.csv"
        csv_path.write_text(
            "group_id,kalshi_market_id,polymarket_market_id\n"
            "fed_rate,KXFED-MAR26-YES,0xfed123\n",
            encoding="utf-8",
        )
        finder = ArbitrageFinder(
            min_net_edge_per_contract=0.01,
            enable_cross_venue=True,
            cross_venue_mapping_path=str(csv_path),
            cross_venue_mapping_required=True,
            enable_fuzzy_cross_venue_fallback=False,
            enable_maker_estimates=False,
        )
        quotes = [
            _make_quote("kalshi", "KXFED-MAR26-YES", 0.45, 0.60),
            _make_quote("polymarket", "0xfed123", 0.52, 0.50),
        ]
        opps = finder.find(quotes)
        cross = [o for o in opps if o.kind is OpportunityKind.CROSS_VENUE]
        assert len(cross) > 0

    def test_kalshi_forecastex_pair_from_mapping(self, tmp_path: Path) -> None:
        """K↔F pair detected when both venues have quotes."""
        csv_path = tmp_path / "map.csv"
        csv_path.write_text(
            "group_id,kalshi_market_id,polymarket_market_id,forecastex_market_id\n"
            "fed_rate,KXFED-MAR26,,FX-FEDFUNDS-MAR26\n",
            encoding="utf-8",
        )
        finder = ArbitrageFinder(
            min_net_edge_per_contract=0.01,
            enable_cross_venue=True,
            cross_venue_mapping_path=str(csv_path),
            cross_venue_mapping_required=True,
            enable_fuzzy_cross_venue_fallback=False,
            enable_maker_estimates=False,
        )
        quotes = [
            _make_quote("kalshi", "KXFED-MAR26", 0.45, 0.60),
            _make_quote("forecastex", "FX-FEDFUNDS-MAR26", 0.52, 0.50),
        ]
        opps = finder.find(quotes)
        cross = [o for o in opps if o.kind is OpportunityKind.CROSS_VENUE]
        assert len(cross) > 0
        # Match key should contain venue pair info
        assert any("kalshi:forecastex" in o.match_key for o in cross)

    def test_three_venue_produces_multiple_cross_opportunities(self, tmp_path: Path) -> None:
        """3-venue mapping with all 3 quotes should produce opportunities for all pairs."""
        csv_path = tmp_path / "map.csv"
        csv_path.write_text(
            "group_id,kalshi_market_id,polymarket_market_id,forecastex_market_id\n"
            "fed_rate,KXFED,0xfed,FX-FED\n",
            encoding="utf-8",
        )
        finder = ArbitrageFinder(
            min_net_edge_per_contract=0.01,
            enable_cross_venue=True,
            cross_venue_mapping_path=str(csv_path),
            cross_venue_mapping_required=True,
            enable_fuzzy_cross_venue_fallback=False,
            enable_maker_estimates=False,
        )
        quotes = [
            _make_quote("kalshi", "KXFED", 0.45, 0.60),
            _make_quote("polymarket", "0xfed", 0.52, 0.50),
            _make_quote("forecastex", "FX-FED", 0.48, 0.55),
        ]
        opps = finder.find(quotes)
        cross = [o for o in opps if o.kind is OpportunityKind.CROSS_VENUE]
        match_keys = {o.match_key for o in cross}
        # Should see opportunities from at least some of the 3 pairs.
        pair_venues_found = set()
        for key in match_keys:
            if "kalshi:polymarket" in key:
                pair_venues_found.add("kp")
            if "kalshi:forecastex" in key:
                pair_venues_found.add("kf")
            if "polymarket:forecastex" in key:
                pair_venues_found.add("pf")
        # At least K↔P and one other pair should produce opportunities.
        assert len(pair_venues_found) >= 2

    def test_partial_coverage_only_available_pairs(self, tmp_path: Path) -> None:
        """When only 2 of 3 venues have quotes, only those pairs produce opportunities."""
        csv_path = tmp_path / "map.csv"
        csv_path.write_text(
            "group_id,kalshi_market_id,polymarket_market_id,forecastex_market_id\n"
            "fed_rate,KXFED,0xfed,FX-FED\n",
            encoding="utf-8",
        )
        finder = ArbitrageFinder(
            min_net_edge_per_contract=0.01,
            enable_cross_venue=True,
            cross_venue_mapping_path=str(csv_path),
            cross_venue_mapping_required=True,
            enable_fuzzy_cross_venue_fallback=False,
            enable_maker_estimates=False,
        )
        # Only kalshi and forecastex have quotes (polymarket missing).
        quotes = [
            _make_quote("kalshi", "KXFED", 0.45, 0.60),
            _make_quote("forecastex", "FX-FED", 0.52, 0.50),
        ]
        opps = finder.find(quotes)
        cross = [o for o in opps if o.kind is OpportunityKind.CROSS_VENUE]
        for opp in cross:
            # Should not involve polymarket.
            assert "polymarket" not in opp.match_key or "kalshi:forecastex" in opp.match_key

    def test_fuzzy_matching_works_across_forecastex(self, tmp_path: Path) -> None:
        """Fuzzy matching should detect cross-venue opportunities involving forecastex."""
        finder = ArbitrageFinder(
            min_net_edge_per_contract=0.01,
            enable_cross_venue=True,
            cross_venue_min_match_score=0.4,
            enable_fuzzy_cross_venue_fallback=True,
            enable_maker_estimates=False,
        )
        quotes = [
            _make_quote(
                "kalshi",
                "KXFED-MAR26",
                0.45,
                0.60,
                title="Will the Federal Reserve raise rates in March 2026",
            ),
            _make_quote(
                "forecastex",
                "FX-FEDFUNDS",
                0.52,
                0.50,
                title="Federal Reserve interest rate March 2026 raise",
            ),
        ]
        opps = finder.find(quotes)
        cross = [o for o in opps if o.kind is OpportunityKind.CROSS_VENUE]
        # Fuzzy matching should find the match based on shared tokens.
        assert len(cross) > 0


# ===================================================================
# Coverage snapshot with 3 venues
# ===================================================================


class TestCoverageSnapshotThreeVenue:
    """Tests for coverage metrics with ForecastEx included."""

    def test_forecastex_refs_tracked(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "map.csv"
        csv_path.write_text(
            "group_id,kalshi_market_id,polymarket_market_id,forecastex_market_id\n"
            "g1,K1,P1,F1\n"
            "g2,K2,P2,\n",
            encoding="utf-8",
        )
        finder = ArbitrageFinder(
            min_net_edge_per_contract=0.01,
            enable_cross_venue=True,
            cross_venue_mapping_path=str(csv_path),
            cross_venue_mapping_required=True,
            enable_fuzzy_cross_venue_fallback=False,
            enable_maker_estimates=False,
        )
        quotes = [
            _make_quote("kalshi", "K1"),
            _make_quote("polymarket", "P1"),
            _make_quote("forecastex", "F1"),
            _make_quote("kalshi", "K2"),
            _make_quote("polymarket", "P2"),
        ]
        snapshot = finder.coverage_snapshot(quotes)
        assert snapshot["cross_mapping_forecastex_refs_total"] == 1
        assert snapshot["cross_mapping_forecastex_refs_seen"] == 1
        assert snapshot["cross_mapping_kalshi_refs_total"] == 2
        assert snapshot["cross_mapping_kalshi_refs_seen"] == 2
        assert snapshot["cross_mapping_polymarket_refs_total"] == 2
        assert snapshot["cross_mapping_polymarket_refs_seen"] == 2
        assert snapshot["cross_mapping_pairs_covered"] == 2  # Both groups have >=2 venues resolved

    def test_missing_forecastex_quote_not_counted(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "map.csv"
        csv_path.write_text(
            "group_id,kalshi_market_id,polymarket_market_id,forecastex_market_id\n"
            "g1,K1,P1,F1\n",
            encoding="utf-8",
        )
        finder = ArbitrageFinder(
            min_net_edge_per_contract=0.01,
            enable_cross_venue=True,
            cross_venue_mapping_path=str(csv_path),
            cross_venue_mapping_required=True,
            enable_fuzzy_cross_venue_fallback=False,
            enable_maker_estimates=False,
        )
        # ForecastEx quote missing
        quotes = [
            _make_quote("kalshi", "K1"),
            _make_quote("polymarket", "P1"),
        ]
        snapshot = finder.coverage_snapshot(quotes)
        assert snapshot["cross_mapping_forecastex_refs_total"] == 1
        assert snapshot["cross_mapping_forecastex_refs_seen"] == 0
        # Pair still covered (K+P both resolved).
        assert snapshot["cross_mapping_pairs_covered"] == 1

    def test_no_forecastex_mappings_shows_zero(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "map.csv"
        csv_path.write_text(
            "group_id,kalshi_market_id,polymarket_market_id\n"
            "g1,K1,P1\n",
            encoding="utf-8",
        )
        finder = ArbitrageFinder(
            min_net_edge_per_contract=0.01,
            enable_cross_venue=True,
            cross_venue_mapping_path=str(csv_path),
            cross_venue_mapping_required=True,
            enable_fuzzy_cross_venue_fallback=False,
            enable_maker_estimates=False,
        )
        quotes = [_make_quote("kalshi", "K1"), _make_quote("polymarket", "P1")]
        snapshot = finder.coverage_snapshot(quotes)
        assert snapshot["cross_mapping_forecastex_refs_total"] == 0
        assert snapshot["cross_mapping_forecastex_refs_seen"] == 0


# ===================================================================
# ForecastEx ref matching in strategy
# ===================================================================


class TestForecastExRefMatching:
    """Tests for forecastex ref key matching in the strategy layer."""

    def test_forecastex_market_id_matches(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "map.csv"
        csv_path.write_text(
            "group_id,kalshi_market_id,polymarket_market_id,forecastex_market_id\n"
            "g1,K1,,FX-CPI-MAR26\n",
            encoding="utf-8",
        )
        finder = ArbitrageFinder(
            min_net_edge_per_contract=0.01,
            enable_cross_venue=True,
            cross_venue_mapping_path=str(csv_path),
            cross_venue_mapping_required=True,
            enable_fuzzy_cross_venue_fallback=False,
            enable_maker_estimates=False,
        )
        quotes = [
            _make_quote("kalshi", "K1", 0.45, 0.60),
            _make_quote("forecastex", "FX-CPI-MAR26", 0.52, 0.50),
        ]
        opps = finder.find(quotes)
        cross = [o for o in opps if o.kind is OpportunityKind.CROSS_VENUE]
        assert len(cross) > 0

    def test_forecastex_symbol_matches_via_metadata(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "map.csv"
        csv_path.write_text(
            "group_id,kalshi_market_id,polymarket_market_id,forecastex_symbol\n"
            "g1,K1,,FEDFUNDS\n",
            encoding="utf-8",
        )
        finder = ArbitrageFinder(
            min_net_edge_per_contract=0.01,
            enable_cross_venue=True,
            cross_venue_mapping_path=str(csv_path),
            cross_venue_mapping_required=True,
            enable_fuzzy_cross_venue_fallback=False,
            enable_maker_estimates=False,
        )
        quotes = [
            _make_quote("kalshi", "K1", 0.45, 0.60),
            _make_quote("forecastex", "FX-FEDFUNDS-123", 0.52, 0.50, symbol="FEDFUNDS"),
        ]
        opps = finder.find(quotes)
        cross = [o for o in opps if o.kind is OpportunityKind.CROSS_VENUE]
        assert len(cross) > 0

    def test_forecastex_symbol_fallback_to_market_id(self, tmp_path: Path) -> None:
        """forecastex_symbol ref should also match against market_id as fallback."""
        csv_path = tmp_path / "map.csv"
        csv_path.write_text(
            "group_id,kalshi_market_id,polymarket_market_id,forecastex_symbol\n"
            "g1,K1,,FEDFUNDS\n",
            encoding="utf-8",
        )
        finder = ArbitrageFinder(
            min_net_edge_per_contract=0.01,
            enable_cross_venue=True,
            cross_venue_mapping_path=str(csv_path),
            cross_venue_mapping_required=True,
            enable_fuzzy_cross_venue_fallback=False,
            enable_maker_estimates=False,
        )
        quotes = [
            _make_quote("kalshi", "K1", 0.45, 0.60),
            # market_id matches the symbol (no metadata).
            _make_quote("forecastex", "FEDFUNDS", 0.52, 0.50),
        ]
        opps = finder.find(quotes)
        cross = [o for o in opps if o.kind is OpportunityKind.CROSS_VENUE]
        assert len(cross) > 0


# ===================================================================
# ForecastEx candidate builder (generator)
# ===================================================================


class TestForecastExCandidateBuilder:
    """Tests for ForecastEx candidate market extraction."""

    def test_build_forecastex_text_with_description_and_symbol(self) -> None:
        text = _build_forecastex_text({"description": "Fed Funds Rate March", "symbol": "FEDFUNDS-MAR26"})
        assert "Fed Funds Rate March" in text
        assert "FEDFUNDS-MAR26" in text

    def test_build_forecastex_text_description_only(self) -> None:
        text = _build_forecastex_text({"description": "CPI Above 3.0%"})
        assert text == "CPI Above 3.0%"

    def test_build_forecastex_text_symbol_only(self) -> None:
        text = _build_forecastex_text({"symbol": "FEDFUNDS-MAR26"})
        assert text == "FEDFUNDS-MAR26"

    def test_build_forecastex_text_empty(self) -> None:
        text = _build_forecastex_text({})
        assert text == ""

    def test_forecastex_candidate_basic(self) -> None:
        market = {
            "contract_id": "12345",
            "description": "Fed Funds Rate above 4.5% in March 2026",
            "symbol": "FEDFUNDS-MAR26",
            "volume": 500,
        }
        candidate = _forecastex_candidate(market)
        assert candidate is not None
        assert candidate.venue == "forecastex"
        assert candidate.market_id == "12345"
        assert candidate.volume == 500.0

    def test_forecastex_candidate_conid_key(self) -> None:
        market = {
            "conId": "67890",
            "longName": "CPI Year-over-Year above three percent",
            "symbol": "CPI-MAR26",
        }
        candidate = _forecastex_candidate(market)
        assert candidate is not None
        assert candidate.market_id == "67890"

    def test_forecastex_candidate_no_id_returns_none(self) -> None:
        market = {"description": "Some contract"}
        candidate = _forecastex_candidate(market)
        assert candidate is None

    def test_forecastex_candidate_too_few_tokens_returns_none(self) -> None:
        market = {"contract_id": "123", "description": "A"}
        candidate = _forecastex_candidate(market)
        assert candidate is None


# ===================================================================
# _build_candidates with 3 venues
# ===================================================================


class TestBuildCandidatesThreeVenue:
    """Tests for building candidates from mixed 3-venue market data."""

    def test_forecastex_venue_detected(self) -> None:
        markets: list[dict[str, Any]] = [
            {"venue": "forecastex", "contract_id": "F1", "description": "Fed Funds Rate March above 4.5 percent"},
            {"venue": "kalshi", "ticker": "KXFED-MAR26", "title": "Fed rate hike March 2026 yes"},
            {"venue": "polymarket", "conditionId": "0xfed", "question": "Will Fed raise rates March 2026"},
        ]
        kalshi, poly, fx = _build_candidates(markets)
        assert len(kalshi) == 1
        assert len(poly) == 1
        assert len(fx) == 1

    def test_ibkr_exchange_forecastx_detected(self) -> None:
        """Markets with exchange=FORECASTX should be detected as forecastex."""
        markets: list[dict[str, Any]] = [
            {"exchange": "FORECASTX", "conId": "99", "longName": "CPI above three percent event contract"},
        ]
        kalshi, poly, fx = _build_candidates(markets)
        assert len(fx) == 1
        assert fx[0].market_id == "99"

    def test_conid_key_triggers_forecastex(self) -> None:
        markets: list[dict[str, Any]] = [
            {"conId": "42", "description": "GDP growth above 2 percent Q1 2026"},
        ]
        kalshi, poly, fx = _build_candidates(markets)
        assert len(fx) == 1

    def test_mixed_venues_separated(self) -> None:
        markets: list[dict[str, Any]] = [
            {"venue": "kalshi", "ticker": "K1", "title": "Some long question about economics"},
            {"venue": "kalshi", "ticker": "K2", "title": "Another question about interest rates"},
            {"venue": "polymarket", "conditionId": "P1", "question": "Will rates go up this quarter"},
            {"venue": "forecastex", "contract_id": "F1", "description": "Interest rate hike event contract"},
            {"venue": "forecastex", "contract_id": "F2", "description": "GDP growth quarterly report prediction"},
        ]
        kalshi, poly, fx = _build_candidates(markets)
        assert len(kalshi) == 2
        assert len(poly) == 1
        assert len(fx) == 2


# ===================================================================
# generate_cross_venue_mapping_rows with forecastex
# ===================================================================


class TestGenerateMappingRowsThreeVenue:
    """Tests for mapping generation with ForecastEx candidates."""

    def test_kalshi_forecastex_mapping_generated(self) -> None:
        """When only kalshi and forecastex candidates are present, mappings should still be generated."""
        markets: list[dict[str, Any]] = [
            {"venue": "kalshi", "ticker": "K1", "title": "Federal Reserve rate decision March 2026 target"},
            {"venue": "forecastex", "contract_id": "F1", "description": "Federal Reserve rate decision March 2026 target"},
        ]
        rows, diagnostics = generate_cross_venue_mapping_rows(
            markets, min_match_score=0.5, min_shared_tokens=2
        )
        assert len(rows) >= 1
        # Should have forecastex_market_id in the row.
        assert any(row.get("forecastex_market_id") for row in rows)

    def test_three_venue_mapping_includes_all(self) -> None:
        """With all three venues, mapping rows should include forecastex_market_id."""
        markets: list[dict[str, Any]] = [
            {"venue": "kalshi", "ticker": "K1", "title": "CPI inflation above 3 percent in March 2026 forecast"},
            {"venue": "polymarket", "conditionId": "P1", "question": "CPI inflation above 3 percent in March 2026 forecast"},
            {"venue": "forecastex", "contract_id": "F1", "description": "CPI inflation above 3 percent in March 2026 forecast"},
        ]
        rows, diagnostics = generate_cross_venue_mapping_rows(
            markets, min_match_score=0.3, min_shared_tokens=2
        )
        assert len(rows) >= 1
        row = rows[0]
        assert row["kalshi_market_id"] == "K1"
        assert row["polymarket_market_id"] == "P1"
        assert row.get("forecastex_market_id") == "F1"

    def test_no_forecastex_candidates_backward_compatible(self) -> None:
        """Without forecastex candidates, output matches original 2-venue format."""
        markets: list[dict[str, Any]] = [
            {"venue": "kalshi", "ticker": "K1", "title": "Bitcoin above 100k end of year 2026"},
            {"venue": "polymarket", "conditionId": "P1", "question": "Bitcoin above 100k end of year 2026"},
        ]
        rows, diagnostics = generate_cross_venue_mapping_rows(
            markets, min_match_score=0.3, min_shared_tokens=2
        )
        assert len(rows) >= 1
        assert "forecastex_market_id" not in rows[0]

    def test_single_venue_returns_empty(self) -> None:
        """With only one venue, no mappings possible."""
        markets: list[dict[str, Any]] = [
            {"venue": "kalshi", "ticker": "K1", "title": "Some unique market question"},
        ]
        rows, diagnostics = generate_cross_venue_mapping_rows(markets)
        assert len(rows) == 0
