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

    if args.paper_minutes is not None:
        output_path = args.paper_output
        if not output_path:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"arb_bot/output/paper_session_{ts}.csv"

        paper_settings = replace(settings, live_mode=False, dry_run=True, run_once=False)
        LOGGER.info(
            "paper mode duration=%.2fmin poll_interval=%ss output=%s discovery=%s stream=%s",
            args.paper_minutes,
            paper_settings.poll_interval_seconds,
            output_path,
            paper_settings.strategy.discovery_mode,
            paper_settings.stream_mode,
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
