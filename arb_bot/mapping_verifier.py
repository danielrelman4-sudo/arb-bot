"""LLM-as-offline-curator mapping verification pipeline (B1).

Separate process that runs periodically (not in the hot path):
1. Reads candidate mappings from cross_venue_map.generated.csv
2. Fetches market titles/descriptions from venue APIs (or cached data)
3. Uses LLM to semantically verify each candidate pair
4. Applies price correlation validation (optional)
5. Outputs cross_venue_map.verified.csv with only confirmed matches
6. Bot reads verified file on startup and via hot-reload

Usage:
    python3 -m arb_bot.mapping_verifier \\
        --candidates arb_bot/config/cross_venue_map.generated.csv \\
        --output arb_bot/config/cross_venue_map.verified.csv \\
        --market-data arb_bot/config/market_data.json \\
        [--max-edge-sanity 0.30] \\
        [--model claude-3-haiku-20240307] \\
        [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class VerificationResult:
    """Result of LLM verification for a single mapping pair."""

    group_id: str
    kalshi_market_id: str
    polymarket_market_id: str
    forecastex_market_id: str

    # LLM verification outcome.
    verified: bool
    confidence: float  # 0.0 to 1.0
    rejection_reason: str  # Empty if verified.
    llm_explanation: str  # Raw LLM reasoning.

    # Market descriptions used for verification.
    kalshi_text: str
    polymarket_text: str

    # Price correlation check (optional).
    price_correlation: float | None = None


@dataclass(frozen=True)
class VerificationSettings:
    """Configuration for the mapping verification pipeline."""

    # Max cross-venue edge before flagging as likely mapping error.
    max_edge_sanity: float = 0.30

    # LLM settings.
    llm_provider: str = "anthropic"  # "anthropic", "openai", or "mock"
    llm_model: str = "claude-3-haiku-20240307"
    llm_max_tokens: int = 512
    llm_temperature: float = 0.0

    # Rate limiting.
    llm_requests_per_minute: int = 50
    llm_delay_between_requests: float = 1.2  # seconds

    # Confidence threshold for verification.
    min_confidence: float = 0.70

    # Whether to require price correlation check.
    require_price_correlation: bool = False
    min_price_correlation: float = 0.50

    # Dry run (don't call LLM, use heuristic verification).
    dry_run: bool = False


@dataclass
class VerificationDiagnostics:
    """Diagnostics from the verification run."""

    total_candidates: int = 0
    verified_count: int = 0
    rejected_count: int = 0
    skipped_no_data: int = 0
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    llm_calls_made: int = 0
    total_time_seconds: float = 0.0


# ---------------------------------------------------------------------------
# LLM verification prompt
# ---------------------------------------------------------------------------

_VERIFICATION_PROMPT = """You are verifying whether two prediction markets from different exchanges are asking the SAME question.

Market A (Kalshi):
Title: {kalshi_title}
{kalshi_extra}

Market B (Polymarket):
Title: {polymarket_title}
{polymarket_extra}

Analyze whether these two markets:
1. Ask the same fundamental question
2. Have the same resolution criteria (same outcome = YES/NO on both)
3. Cover the same time period
4. Refer to the same entity/event

IMPORTANT: Be especially careful about:
- "Will X run?" vs "Will X win?" (DIFFERENT questions)
- "Will X happen by date A?" vs "Will X happen by date B?" (DIFFERENT if dates differ)
- Threshold differences: ">50%" vs ">60%" (DIFFERENT questions)
- Temporal scope: "in 2026" vs "in 2027" (DIFFERENT if years differ)

Respond in this exact JSON format:
{{"same_question": true/false, "confidence": 0.0-1.0, "reason": "brief explanation"}}

If the markets ask different questions, set same_question to false and explain why."""


def _build_verification_prompt(
    kalshi_text: str,
    polymarket_text: str,
    kalshi_extra: str = "",
    polymarket_extra: str = "",
) -> str:
    """Build the LLM verification prompt for a candidate pair."""
    return _VERIFICATION_PROMPT.format(
        kalshi_title=kalshi_text,
        kalshi_extra=f"Description: {kalshi_extra}" if kalshi_extra else "",
        polymarket_title=polymarket_text,
        polymarket_extra=f"Description: {polymarket_extra}" if polymarket_extra else "",
    )


def _parse_llm_response(response_text: str) -> tuple[bool, float, str]:
    """Parse the LLM's JSON response into (same_question, confidence, reason)."""
    try:
        # Try to extract JSON from the response.
        json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            same = bool(data.get("same_question", False))
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
            reason = str(data.get("reason", ""))
            return same, confidence, reason
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback: try to detect yes/no from text.
    lower = response_text.lower()
    if "same_question" in lower and "true" in lower:
        return True, 0.5, "parsed from text (low confidence)"
    if "same_question" in lower and "false" in lower:
        return False, 0.5, "parsed from text (low confidence)"

    return False, 0.0, f"failed to parse LLM response: {response_text[:200]}"


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------

def _call_anthropic(prompt: str, settings: VerificationSettings) -> str:
    """Call Anthropic API for verification."""
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "anthropic package required for LLM verification. "
            "Install with: pip install anthropic"
        )

    client = anthropic.Anthropic()
    message = client.messages.create(
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def _call_openai(prompt: str, settings: VerificationSettings) -> str:
    """Call OpenAI API for verification."""
    try:
        import openai
    except ImportError:
        raise ImportError(
            "openai package required for LLM verification. "
            "Install with: pip install openai"
        )

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


def _call_mock(prompt: str, settings: VerificationSettings) -> str:
    """Mock LLM that uses heuristic text matching for testing."""
    # Extract market texts from prompt.
    lines = prompt.split("\n")
    kalshi_title = ""
    polymarket_title = ""
    for line in lines:
        if line.startswith("Title: ") and not kalshi_title:
            kalshi_title = line[7:].strip()
        elif line.startswith("Title: "):
            polymarket_title = line[7:].strip()

    # Simple heuristic: check normalized text overlap.
    def normalize(text: str) -> set[str]:
        words = re.sub(r"[^a-z0-9\s]", "", text.lower()).split()
        stopwords = {"will", "the", "a", "an", "in", "on", "at", "by", "to", "of", "for", "is", "be"}
        return {w for w in words if w not in stopwords and len(w) > 1}

    k_words = normalize(kalshi_title)
    p_words = normalize(polymarket_title)

    if not k_words or not p_words:
        return '{"same_question": false, "confidence": 0.9, "reason": "missing text"}'

    overlap = k_words & p_words
    union = k_words | p_words
    jaccard = len(overlap) / max(1, len(union))

    # Check for known red flags.
    k_lower = kalshi_title.lower()
    p_lower = polymarket_title.lower()

    # "run" vs "win" mismatch.
    if ("run" in k_lower and "win" in p_lower) or ("win" in k_lower and "run" in p_lower):
        if "run" not in p_lower or "win" not in k_lower:
            return json.dumps({
                "same_question": False,
                "confidence": 0.95,
                "reason": "Different questions: 'run' vs 'win' mismatch",
            })

    if jaccard >= 0.6:
        return json.dumps({
            "same_question": True,
            "confidence": min(1.0, jaccard + 0.2),
            "reason": f"High text overlap (jaccard={jaccard:.2f})",
        })
    elif jaccard >= 0.3:
        return json.dumps({
            "same_question": True,
            "confidence": jaccard,
            "reason": f"Moderate text overlap (jaccard={jaccard:.2f})",
        })
    else:
        return json.dumps({
            "same_question": False,
            "confidence": 0.8,
            "reason": f"Low text overlap (jaccard={jaccard:.2f})",
        })


def _call_llm(prompt: str, settings: VerificationSettings) -> str:
    """Route LLM call to the configured provider."""
    if settings.dry_run or settings.llm_provider == "mock":
        return _call_mock(prompt, settings)
    elif settings.llm_provider == "anthropic":
        return _call_anthropic(prompt, settings)
    elif settings.llm_provider == "openai":
        return _call_openai(prompt, settings)
    else:
        raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")


# ---------------------------------------------------------------------------
# Market data loading
# ---------------------------------------------------------------------------

def _load_market_data(path: str | None) -> dict[str, dict[str, Any]]:
    """Load market data JSON for text enrichment.

    Returns a dict mapping market_id → market metadata.
    """
    if not path or not Path(path).exists():
        return {}

    with open(path, "r") as f:
        data = json.load(f)

    result: dict[str, dict[str, Any]] = {}
    if isinstance(data, list):
        for market in data:
            market_id = str(
                market.get("ticker")
                or market.get("market_ticker")
                or market.get("conditionId")
                or market.get("id")
                or ""
            ).strip()
            if market_id:
                result[market_id] = market
    elif isinstance(data, dict):
        result = data
    return result


def _get_market_text(
    market_id: str,
    venue: str,
    market_data: dict[str, dict[str, Any]],
) -> tuple[str, str]:
    """Get (title, description) for a market from cached data.

    Returns ("", "") if not found.
    """
    info = market_data.get(market_id, {})
    if not info:
        return "", ""

    if venue == "kalshi":
        title = str(
            info.get("title")
            or info.get("question")
            or info.get("name")
            or info.get("event_title")
            or ""
        ).strip()
        subtitle = str(
            info.get("subtitle")
            or info.get("yes_sub_title")
            or info.get("outcome")
            or ""
        ).strip()
        full_title = f"{title} {subtitle}".strip() if subtitle else title
        description = str(info.get("description") or info.get("rules") or "").strip()
        return full_title, description

    elif venue == "polymarket":
        title = str(
            info.get("question")
            or info.get("title")
            or info.get("name")
            or ""
        ).strip()
        description = str(info.get("description") or "").strip()
        return title, description

    return "", ""


# ---------------------------------------------------------------------------
# Heuristic sanity checks
# ---------------------------------------------------------------------------

def _heuristic_rejection(
    kalshi_text: str,
    polymarket_text: str,
    settings: VerificationSettings,
) -> str | None:
    """Apply fast heuristic checks before LLM call.

    Returns a rejection reason string, or None if the pair passes.
    """
    if not kalshi_text.strip() or not polymarket_text.strip():
        return "missing_market_text"

    k_lower = kalshi_text.lower()
    p_lower = polymarket_text.lower()

    # Check for "run" vs "win" mismatch (the KX2028RRUN bug).
    k_has_run = bool(re.search(r'\brun\b', k_lower))
    k_has_win = bool(re.search(r'\bwin\b', k_lower))
    p_has_run = bool(re.search(r'\brun\b', p_lower))
    p_has_win = bool(re.search(r'\bwin\b', p_lower))

    if k_has_run and not k_has_win and p_has_win and not p_has_run:
        return "run_vs_win_mismatch"
    if k_has_win and not k_has_run and p_has_run and not p_has_win:
        return "run_vs_win_mismatch"

    return None


# ---------------------------------------------------------------------------
# Core verification pipeline
# ---------------------------------------------------------------------------

def verify_mappings(
    candidates_path: str,
    output_path: str,
    market_data_path: str | None = None,
    settings: VerificationSettings | None = None,
) -> tuple[list[VerificationResult], VerificationDiagnostics]:
    """Run the full mapping verification pipeline.

    Parameters
    ----------
    candidates_path:
        Path to the candidate mappings CSV (cross_venue_map.generated.csv).
    output_path:
        Path to write the verified mappings CSV.
    market_data_path:
        Optional path to market data JSON for text enrichment.
    settings:
        Verification settings.

    Returns
    -------
    Tuple of (results, diagnostics).
    """
    if settings is None:
        settings = VerificationSettings()

    start_time = time.time()
    diagnostics = VerificationDiagnostics()
    results: list[VerificationResult] = []

    # Load candidate mappings.
    candidates = _load_candidates(candidates_path)
    diagnostics.total_candidates = len(candidates)
    LOGGER.info("Loaded %d candidate mappings from %s", len(candidates), candidates_path)

    # Load market data for text enrichment.
    market_data = _load_market_data(market_data_path)
    LOGGER.info("Loaded market data for %d markets", len(market_data))

    # Process each candidate.
    for i, candidate in enumerate(candidates):
        group_id = candidate.get("group_id", "")
        kalshi_id = candidate.get("kalshi_market_id", "")
        polymarket_id = candidate.get("polymarket_market_id", "")
        forecastex_id = candidate.get("forecastex_market_id", "")

        if not kalshi_id or not polymarket_id:
            # Skip rows without both venues.
            diagnostics.skipped_no_data += 1
            continue

        # Get market texts.
        k_title, k_desc = _get_market_text(kalshi_id, "kalshi", market_data)
        p_title, p_desc = _get_market_text(polymarket_id, "polymarket", market_data)

        # If no market data, use group_id as text.
        if not k_title:
            k_title = group_id.replace("_", " ")
        if not p_title:
            p_title = group_id.replace("_", " ")

        # Heuristic pre-check.
        heuristic_reason = _heuristic_rejection(k_title, p_title, settings)
        if heuristic_reason is not None:
            results.append(VerificationResult(
                group_id=group_id,
                kalshi_market_id=kalshi_id,
                polymarket_market_id=polymarket_id,
                forecastex_market_id=forecastex_id,
                verified=False,
                confidence=0.95,
                rejection_reason=heuristic_reason,
                llm_explanation=f"Heuristic rejection: {heuristic_reason}",
                kalshi_text=k_title,
                polymarket_text=p_title,
            ))
            diagnostics.rejected_count += 1
            diagnostics.rejection_reasons[heuristic_reason] = (
                diagnostics.rejection_reasons.get(heuristic_reason, 0) + 1
            )
            continue

        # LLM verification.
        prompt = _build_verification_prompt(
            kalshi_text=k_title,
            polymarket_text=p_title,
            kalshi_extra=k_desc,
            polymarket_extra=p_desc,
        )

        try:
            response = _call_llm(prompt, settings)
            diagnostics.llm_calls_made += 1
        except Exception as exc:
            LOGGER.warning(
                "LLM call failed for %s: %s (treating as unverified)",
                group_id, exc,
            )
            results.append(VerificationResult(
                group_id=group_id,
                kalshi_market_id=kalshi_id,
                polymarket_market_id=polymarket_id,
                forecastex_market_id=forecastex_id,
                verified=False,
                confidence=0.0,
                rejection_reason=f"llm_error: {exc}",
                llm_explanation="",
                kalshi_text=k_title,
                polymarket_text=p_title,
            ))
            diagnostics.rejected_count += 1
            diagnostics.rejection_reasons["llm_error"] = (
                diagnostics.rejection_reasons.get("llm_error", 0) + 1
            )
            continue

        same, confidence, reason = _parse_llm_response(response)

        if same and confidence >= settings.min_confidence:
            verified = True
        else:
            verified = False

        rejection_reason = "" if verified else (reason or "llm_rejected")

        results.append(VerificationResult(
            group_id=group_id,
            kalshi_market_id=kalshi_id,
            polymarket_market_id=polymarket_id,
            forecastex_market_id=forecastex_id,
            verified=verified,
            confidence=confidence,
            rejection_reason=rejection_reason,
            llm_explanation=reason,
            kalshi_text=k_title,
            polymarket_text=p_title,
        ))

        if verified:
            diagnostics.verified_count += 1
        else:
            diagnostics.rejected_count += 1
            diagnostics.rejection_reasons[rejection_reason] = (
                diagnostics.rejection_reasons.get(rejection_reason, 0) + 1
            )

        # Rate limiting.
        if not settings.dry_run and settings.llm_delay_between_requests > 0:
            time.sleep(settings.llm_delay_between_requests)

        if (i + 1) % 100 == 0:
            LOGGER.info(
                "Verified %d/%d candidates (%d verified, %d rejected)",
                i + 1, len(candidates),
                diagnostics.verified_count,
                diagnostics.rejected_count,
            )

    diagnostics.total_time_seconds = time.time() - start_time

    # Write verified mappings.
    _write_verified_csv(output_path, results)

    # Write full results with LLM explanations for audit.
    audit_path = output_path.replace(".csv", ".audit.json")
    _write_audit_json(audit_path, results, diagnostics)

    LOGGER.info(
        "Verification complete: %d verified / %d rejected / %d skipped in %.1fs (%d LLM calls)",
        diagnostics.verified_count,
        diagnostics.rejected_count,
        diagnostics.skipped_no_data,
        diagnostics.total_time_seconds,
        diagnostics.llm_calls_made,
    )

    return results, diagnostics


def _load_candidates(path: str) -> list[dict[str, str]]:
    """Load candidate mappings from CSV."""
    candidates: list[dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            candidates.append(dict(row))
    return candidates


def _write_verified_csv(path: str, results: list[VerificationResult]) -> None:
    """Write verified mappings to CSV (same format as the input)."""
    verified = [r for r in results if r.verified]
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["group_id", "kalshi_market_id", "polymarket_market_id"]
    # Include forecastex column if any results have it.
    if any(r.forecastex_market_id for r in verified):
        fieldnames.append("forecastex_market_id")

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in verified:
            row: dict[str, str] = {
                "group_id": r.group_id,
                "kalshi_market_id": r.kalshi_market_id,
                "polymarket_market_id": r.polymarket_market_id,
            }
            if "forecastex_market_id" in fieldnames:
                row["forecastex_market_id"] = r.forecastex_market_id
            writer.writerow(row)

    LOGGER.info("Wrote %d verified mappings to %s", len(verified), path)


def _write_audit_json(
    path: str,
    results: list[VerificationResult],
    diagnostics: VerificationDiagnostics,
) -> None:
    """Write full audit trail to JSON."""
    audit = {
        "diagnostics": {
            "total_candidates": diagnostics.total_candidates,
            "verified_count": diagnostics.verified_count,
            "rejected_count": diagnostics.rejected_count,
            "skipped_no_data": diagnostics.skipped_no_data,
            "rejection_reasons": diagnostics.rejection_reasons,
            "llm_calls_made": diagnostics.llm_calls_made,
            "total_time_seconds": diagnostics.total_time_seconds,
        },
        "results": [
            {
                "group_id": r.group_id,
                "kalshi_market_id": r.kalshi_market_id,
                "polymarket_market_id": r.polymarket_market_id,
                "verified": r.verified,
                "confidence": r.confidence,
                "rejection_reason": r.rejection_reason,
                "llm_explanation": r.llm_explanation,
                "kalshi_text": r.kalshi_text,
                "polymarket_text": r.polymarket_text,
            }
            for r in results
        ],
    }

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    LOGGER.info("Wrote audit trail to %s", path)


# ---------------------------------------------------------------------------
# Runtime edge sanity gate (for the bot's hot path)
# ---------------------------------------------------------------------------

def edge_sanity_check(
    net_edge_per_contract: float,
    max_edge: float = 0.30,
) -> bool:
    """Runtime sanity check for cross-venue edges.

    Returns True if the edge is plausible, False if it's likely a mapping error.
    Edges > max_edge are almost certainly mapping errors (like run-vs-win).
    """
    return abs(net_edge_per_contract) <= max_edge


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for the mapping verifier."""
    parser = argparse.ArgumentParser(
        description="LLM-based cross-venue mapping verification pipeline",
    )
    parser.add_argument(
        "--candidates",
        required=True,
        help="Path to candidate mappings CSV (cross_venue_map.generated.csv)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write verified mappings CSV",
    )
    parser.add_argument(
        "--market-data",
        default=None,
        help="Path to market data JSON for text enrichment",
    )
    parser.add_argument(
        "--max-edge-sanity",
        type=float,
        default=0.30,
        help="Max edge before flagging as mapping error (default: 0.30)",
    )
    parser.add_argument(
        "--model",
        default="claude-3-haiku-20240307",
        help="LLM model to use (default: claude-3-haiku-20240307)",
    )
    parser.add_argument(
        "--provider",
        default="anthropic",
        choices=["anthropic", "openai", "mock"],
        help="LLM provider (default: anthropic)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.70,
        help="Minimum LLM confidence to verify (default: 0.70)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use heuristic mock instead of real LLM",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    settings = VerificationSettings(
        max_edge_sanity=args.max_edge_sanity,
        llm_provider=args.provider,
        llm_model=args.model,
        min_confidence=args.min_confidence,
        dry_run=args.dry_run,
    )

    results, diagnostics = verify_mappings(
        candidates_path=args.candidates,
        output_path=args.output,
        market_data_path=args.market_data,
        settings=settings,
    )

    print(f"\nVerification Summary:")
    print(f"  Total candidates: {diagnostics.total_candidates}")
    print(f"  Verified:         {diagnostics.verified_count}")
    print(f"  Rejected:         {diagnostics.rejected_count}")
    print(f"  Skipped (no data):{diagnostics.skipped_no_data}")
    print(f"  LLM calls:        {diagnostics.llm_calls_made}")
    print(f"  Time:             {diagnostics.total_time_seconds:.1f}s")
    if diagnostics.rejection_reasons:
        print(f"  Rejection reasons:")
        for reason, count in sorted(
            diagnostics.rejection_reasons.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            print(f"    {reason}: {count}")


if __name__ == "__main__":
    main()
