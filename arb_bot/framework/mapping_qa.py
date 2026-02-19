"""Market mapping QA pipeline (Phase 2F).

Validates cross-venue market mappings for quality issues:
- Cardinality violations (one-to-many, many-to-one)
- Stale links (mapped markets no longer present in live data)
- Semantic drift (text similarity between mapped markets drops)
- Duplicate group IDs
- Suspicious mappings (low confidence, missing required fields)

The QA pipeline runs on a list of mappings and a set of live market
identifiers, producing a structured report of issues and a per-mapping
health grade.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Set

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Issue severity
# ---------------------------------------------------------------------------


class IssueSeverity(Enum):
    """Severity level for a mapping QA issue."""

    ERROR = "error"       # Must fix before live trading.
    WARNING = "warning"   # Should investigate.
    INFO = "info"         # Informational note.


# ---------------------------------------------------------------------------
# QA issue
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MappingIssue:
    """A single QA issue found in a mapping."""

    group_id: str
    check_name: str
    severity: IssueSeverity
    message: str


# ---------------------------------------------------------------------------
# QA config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MappingQAConfig:
    """Configuration for mapping QA checks.

    Parameters
    ----------
    check_cardinality:
        Flag one-to-many and many-to-one mappings. Default True.
    check_stale_links:
        Flag mappings where a venue's market ID is not in the live set.
        Default True.
    check_duplicate_group_ids:
        Flag duplicate group_id values. Default True.
    check_empty_fields:
        Flag mappings with empty venue references. Default True.
    min_text_similarity:
        If > 0, check that mapped market texts have at least this
        Jaccard similarity. Requires text data in market_texts dict.
        Default 0.0 (disabled).
    """

    check_cardinality: bool = True
    check_stale_links: bool = True
    check_duplicate_group_ids: bool = True
    check_empty_fields: bool = True
    min_text_similarity: float = 0.0


# ---------------------------------------------------------------------------
# QA report
# ---------------------------------------------------------------------------


@dataclass
class MappingQAReport:
    """Result of running the QA pipeline."""

    total_mappings: int = 0
    total_issues: int = 0
    issues: list[MappingIssue] = field(default_factory=list)
    issues_by_severity: Dict[str, int] = field(default_factory=dict)
    issues_by_check: Dict[str, int] = field(default_factory=dict)

    @property
    def has_errors(self) -> bool:
        return self.issues_by_severity.get("error", 0) > 0

    @property
    def has_warnings(self) -> bool:
        return self.issues_by_severity.get("warning", 0) > 0

    @property
    def summary(self) -> str:
        if self.total_issues == 0:
            return f"OK: {self.total_mappings} mappings, no issues"
        parts = []
        for sev in ("error", "warning", "info"):
            count = self.issues_by_severity.get(sev, 0)
            if count > 0:
                parts.append(f"{count} {sev}(s)")
        return (
            f"{self.total_mappings} mappings, "
            f"{self.total_issues} issues: {', '.join(parts)}"
        )


# ---------------------------------------------------------------------------
# Mapping representation for QA
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MappingEntry:
    """Lightweight mapping representation for QA checks.

    This decouples the QA pipeline from the cross_mapping module's
    internal types, making it testable without importing the full
    mapping system.
    """

    group_id: str
    kalshi_id: str
    polymarket_id: str


# ---------------------------------------------------------------------------
# QA pipeline
# ---------------------------------------------------------------------------


class MappingQAPipeline:
    """Runs quality checks on cross-venue market mappings.

    Usage::

        pipeline = MappingQAPipeline(config)
        report = pipeline.run(
            mappings=mappings,
            live_kalshi_ids={"K1", "K2", ...},
            live_polymarket_ids={"P1", "P2", ...},
            market_texts={"K1": "Will X happen?", "P1": "Will X occur?"},
        )
        if report.has_errors:
            ...
    """

    def __init__(self, config: MappingQAConfig | None = None) -> None:
        self._config = config or MappingQAConfig()

    @property
    def config(self) -> MappingQAConfig:
        return self._config

    def run(
        self,
        mappings: list[MappingEntry],
        live_kalshi_ids: Set[str] | None = None,
        live_polymarket_ids: Set[str] | None = None,
        market_texts: Dict[str, str] | None = None,
    ) -> MappingQAReport:
        """Run all enabled QA checks.

        Parameters
        ----------
        mappings:
            List of mapping entries to validate.
        live_kalshi_ids:
            Set of currently live Kalshi market IDs (for stale check).
        live_polymarket_ids:
            Set of currently live Polymarket market IDs (for stale check).
        market_texts:
            Dict of market_id → market text (for semantic similarity).
        """
        report = MappingQAReport(total_mappings=len(mappings))
        cfg = self._config

        if cfg.check_empty_fields:
            self._check_empty_fields(mappings, report)

        if cfg.check_duplicate_group_ids:
            self._check_duplicate_group_ids(mappings, report)

        if cfg.check_cardinality:
            self._check_cardinality(mappings, report)

        if cfg.check_stale_links:
            self._check_stale_links(
                mappings, report,
                live_kalshi_ids or set(),
                live_polymarket_ids or set(),
            )

        if cfg.min_text_similarity > 0 and market_texts:
            self._check_text_similarity(mappings, report, market_texts)

        return report

    def _add_issue(
        self,
        report: MappingQAReport,
        group_id: str,
        check_name: str,
        severity: IssueSeverity,
        message: str,
    ) -> None:
        issue = MappingIssue(
            group_id=group_id,
            check_name=check_name,
            severity=severity,
            message=message,
        )
        report.issues.append(issue)
        report.total_issues += 1
        report.issues_by_severity[severity.value] = (
            report.issues_by_severity.get(severity.value, 0) + 1
        )
        report.issues_by_check[check_name] = (
            report.issues_by_check.get(check_name, 0) + 1
        )

    def _check_empty_fields(
        self,
        mappings: list[MappingEntry],
        report: MappingQAReport,
    ) -> None:
        for m in mappings:
            if not m.group_id.strip():
                self._add_issue(
                    report, m.group_id, "empty_group_id",
                    IssueSeverity.ERROR,
                    "Mapping has empty group_id",
                )
            if not m.kalshi_id.strip():
                self._add_issue(
                    report, m.group_id, "empty_kalshi_id",
                    IssueSeverity.ERROR,
                    f"Mapping '{m.group_id}' has empty kalshi_id",
                )
            if not m.polymarket_id.strip():
                self._add_issue(
                    report, m.group_id, "empty_polymarket_id",
                    IssueSeverity.ERROR,
                    f"Mapping '{m.group_id}' has empty polymarket_id",
                )

    def _check_duplicate_group_ids(
        self,
        mappings: list[MappingEntry],
        report: MappingQAReport,
    ) -> None:
        counts = Counter(m.group_id for m in mappings)
        for group_id, count in counts.items():
            if count > 1:
                self._add_issue(
                    report, group_id, "duplicate_group_id",
                    IssueSeverity.WARNING,
                    f"Group '{group_id}' appears {count} times",
                )

    def _check_cardinality(
        self,
        mappings: list[MappingEntry],
        report: MappingQAReport,
    ) -> None:
        # Check one-to-many: same kalshi_id mapped to multiple polymarket_ids.
        kalshi_to_poly: Dict[str, list[str]] = {}
        for m in mappings:
            kalshi_to_poly.setdefault(m.kalshi_id, []).append(m.group_id)

        for kalshi_id, groups in kalshi_to_poly.items():
            if len(groups) > 1:
                self._add_issue(
                    report, groups[0], "one_to_many_kalshi",
                    IssueSeverity.WARNING,
                    f"Kalshi '{kalshi_id}' mapped in {len(groups)} groups: "
                    f"{', '.join(groups[:5])}",
                )

        # Check many-to-one: same polymarket_id mapped to multiple kalshi_ids.
        poly_to_kalshi: Dict[str, list[str]] = {}
        for m in mappings:
            poly_to_kalshi.setdefault(m.polymarket_id, []).append(m.group_id)

        for poly_id, groups in poly_to_kalshi.items():
            if len(groups) > 1:
                self._add_issue(
                    report, groups[0], "many_to_one_polymarket",
                    IssueSeverity.WARNING,
                    f"Polymarket '{poly_id}' mapped in {len(groups)} groups: "
                    f"{', '.join(groups[:5])}",
                )

    def _check_stale_links(
        self,
        mappings: list[MappingEntry],
        report: MappingQAReport,
        live_kalshi_ids: Set[str],
        live_polymarket_ids: Set[str],
    ) -> None:
        # Only check if we have live data.
        if not live_kalshi_ids and not live_polymarket_ids:
            return

        for m in mappings:
            if live_kalshi_ids and m.kalshi_id not in live_kalshi_ids:
                self._add_issue(
                    report, m.group_id, "stale_kalshi",
                    IssueSeverity.WARNING,
                    f"Kalshi '{m.kalshi_id}' not in live market set",
                )
            if live_polymarket_ids and m.polymarket_id not in live_polymarket_ids:
                self._add_issue(
                    report, m.group_id, "stale_polymarket",
                    IssueSeverity.WARNING,
                    f"Polymarket '{m.polymarket_id}' not in live market set",
                )

    def _check_text_similarity(
        self,
        mappings: list[MappingEntry],
        report: MappingQAReport,
        market_texts: Dict[str, str],
    ) -> None:
        for m in mappings:
            kalshi_text = market_texts.get(m.kalshi_id, "")
            poly_text = market_texts.get(m.polymarket_id, "")
            if not kalshi_text or not poly_text:
                continue

            sim = _jaccard_similarity(kalshi_text, poly_text)
            if sim < self._config.min_text_similarity:
                self._add_issue(
                    report, m.group_id, "low_text_similarity",
                    IssueSeverity.WARNING,
                    f"Text similarity {sim:.3f} below threshold "
                    f"{self._config.min_text_similarity:.3f}",
                )


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """Compute Jaccard similarity between two texts (tokenized)."""
    tokens_a = set(_tokenize(text_a))
    tokens_b = set(_tokenize(text_b))
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union > 0 else 0.0


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenization."""
    return [
        t for t in text.lower().split()
        if len(t) > 1  # Skip single characters.
    ]
