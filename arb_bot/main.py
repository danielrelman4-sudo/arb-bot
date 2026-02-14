from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import replace
from datetime import datetime

from arb_bot.config import load_settings
from arb_bot.engine import ArbEngine
from arb_bot.logging_setup import configure_logging
from arb_bot.paper import run_paper_session

LOGGER = logging.getLogger(__name__)

# Default output path for auto-refreshed cross-venue mappings.
_DEFAULT_MAPPING_OUTPUT = "arb_bot/config/cross_venue_map.live.csv"


async def _refresh_cross_venue_mappings(
    output_path: str,
    *,
    kalshi_api_base_url: str = "https://api.elections.kalshi.com/trade-api/v2",
    polymarket_gamma_base_url: str = "https://gamma-api.polymarket.com",
    forecastex_catalog_path: str = "",
    forecastex_symbols_csv_path: str = "",
) -> str:
    """Regenerate cross-venue mappings from live Kalshi + Polymarket data.

    Returns the output path so it can be wired into settings.
    """
    from arb_bot.cross_mapping_generator import generate_cross_venue_mapping_rows
    from arb_bot.structural_rule_generator import (
        fetch_kalshi_markets_all_events,
        fetch_polymarket_markets_all_events,
    )

    LOGGER.info("refreshing cross-venue mappings from live market data ...")

    kalshi_markets, polymarket_markets = await asyncio.gather(
        fetch_kalshi_markets_all_events(api_base_url=kalshi_api_base_url),
        fetch_polymarket_markets_all_events(gamma_base_url=polymarket_gamma_base_url),
    )
    LOGGER.info(
        "fetched markets: kalshi=%d polymarket=%d",
        len(kalshi_markets),
        len(polymarket_markets),
    )

    forecastex_markets: list[dict] = []
    if forecastex_catalog_path:
        from arb_bot.cross_mapping_generator import load_forecastex_markets_from_catalog
        forecastex_markets = load_forecastex_markets_from_catalog(
            catalog_path=forecastex_catalog_path,
            symbols_csv_path=forecastex_symbols_csv_path or None,
        )
        LOGGER.info("loaded forecastex catalog markets: %d", len(forecastex_markets))

    if not kalshi_markets:
        raise RuntimeError("Kalshi market fetch returned 0 markets — cannot generate mappings")
    if not polymarket_markets:
        LOGGER.warning("Polymarket market fetch returned 0 markets — cross-venue matching limited to Kalshi↔ForecastEx")

    import csv
    from pathlib import Path

    rows, diagnostics = generate_cross_venue_mapping_rows(
        [*kalshi_markets, *polymarket_markets, *forecastex_markets],
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["group_id", "kalshi_market_id", "polymarket_market_id"]
    if any("forecastex_market_id" in row for row in rows):
        fieldnames.append("forecastex_market_id")
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    LOGGER.info(
        "cross-venue mappings refreshed: %d mappings written to %s "
        "(kalshi_candidates=%d polymarket_candidates=%d unmatched_k=%d unmatched_p=%d)",
        diagnostics.mappings_emitted,
        output_path,
        diagnostics.kalshi_candidates,
        diagnostics.polymarket_candidates,
        diagnostics.unmatched_kalshi,
        diagnostics.unmatched_polymarket,
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prediction-market YES/NO arbitrage bot scaffold",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run in live mode and place orders when risk checks pass",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single polling cycle",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use stream-first detection (with polling refresh fallback)",
    )
    parser.add_argument(
        "--discovery",
        action="store_true",
        help="Enable discovery thresholds for near-arb exploration",
    )
    parser.add_argument(
        "--paper-minutes",
        type=float,
        default=None,
        help="Run a dry-run paper session for N minutes and export a CSV",
    )
    parser.add_argument(
        "--paper-output",
        type=str,
        default=None,
        help="CSV output path for paper session decisions",
    )
    parser.add_argument(
        "--paper-max-cycles",
        type=int,
        default=None,
        help="Optional cap on paper-session cycles",
    )
    parser.add_argument(
        "--refresh-mappings",
        action="store_true",
        help="Regenerate cross-venue mappings from live market data before running. "
        "Required for paper/live runs to avoid stale mappings.",
    )
    parser.add_argument(
        "--mapping-output",
        type=str,
        default=_DEFAULT_MAPPING_OUTPUT,
        help="Output path for refreshed cross-venue mapping CSV "
        f"(default: {_DEFAULT_MAPPING_OUTPUT})",
    )
    return parser.parse_args()


async def _run() -> None:
    args = parse_args()
    settings = load_settings()

    if args.live:
        settings = replace(settings, live_mode=True, dry_run=False)
    if args.once:
        settings = replace(settings, run_once=True)
    if args.stream:
        settings = replace(settings, stream_mode=True)
    if args.discovery:
        strategy = replace(settings.strategy, discovery_mode=True)
        settings = replace(settings, strategy=strategy)

    configure_logging(settings.log_level)

    # Refresh cross-venue mappings from live data if requested.
    if args.refresh_mappings:
        mapping_path = await _refresh_cross_venue_mappings(
            args.mapping_output,
            kalshi_api_base_url=settings.kalshi.api_base_url,
            polymarket_gamma_base_url=settings.polymarket.gamma_base_url,
            forecastex_catalog_path=settings.forecastex.conid_catalog_path,
            forecastex_symbols_csv_path="arb_bot/config/forecastex_symbols.csv" if settings.forecastex.enabled else "",
        )
        strategy = replace(settings.strategy, cross_venue_mapping_path=mapping_path)
        settings = replace(settings, strategy=strategy)

    if args.paper_minutes is not None:
        output_path = args.paper_output
        if not output_path:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"arb_bot/output/paper_session_{ts}.csv"

        paper_settings = replace(settings, live_mode=False, dry_run=True, run_once=False)
        LOGGER.info(
            "paper mode duration=%.2fmin poll_interval=%ss output=%s discovery=%s stream=%s mappings=%s",
            args.paper_minutes,
            paper_settings.poll_interval_seconds,
            output_path,
            paper_settings.strategy.discovery_mode,
            paper_settings.stream_mode,
            paper_settings.strategy.cross_venue_mapping_path or "(none)",
        )

        summary = await run_paper_session(
            settings=paper_settings,
            duration_minutes=args.paper_minutes,
            output_csv=output_path,
            max_cycles=args.paper_max_cycles,
        )

        LOGGER.info(
            "paper summary cycles=%d quotes=%d opportunities=%d near=%d dry_trades=%d settled=%d skipped=%d simulated_pnl=%.2f csv=%s",
            summary.cycles,
            summary.quotes_seen,
            summary.opportunities_seen,
            summary.near_opportunities_seen,
            summary.dry_run_trades,
            summary.settled_trades,
            summary.skipped,
            summary.simulated_pnl_usd,
            summary.output_csv,
        )
        return

    engine = ArbEngine(settings)

    LOGGER.info(
        "bot mode=%s poll_interval=%ss stream=%s discovery=%s",
        "live" if settings.live_mode else "dry-run",
        settings.poll_interval_seconds,
        settings.stream_mode,
        settings.strategy.discovery_mode,
    )
    await engine.run_forever()


if __name__ == "__main__":
    asyncio.run(_run())
