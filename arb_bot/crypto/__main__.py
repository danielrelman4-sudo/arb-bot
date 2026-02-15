"""CLI entry point for the crypto prediction trading module.

Usage::

    python3 -m arb_bot.crypto --paper --duration-minutes 60
    python3 -m arb_bot.crypto --paper --symbols KXBTC KXETH --mc-paths 2000
    python3 -m arb_bot.crypto --paper --min-edge 0.03 --duration-minutes 30
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from arb_bot.crypto.config import CryptoSettings, load_crypto_settings
from arb_bot.crypto.engine import CryptoEngine


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m arb_bot.crypto",
        description="Crypto prediction trading on Kalshi short-term binary markets",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--paper", action="store_true", default=True,
        help="Run in paper trading mode (default)",
    )
    mode.add_argument(
        "--live", action="store_true",
        help="Run in live trading mode (requires Kalshi credentials)",
    )

    parser.add_argument(
        "--symbols", nargs="+", default=None,
        help="Kalshi crypto series to trade (e.g. KXBTC KXETH)",
    )
    parser.add_argument(
        "--mc-paths", type=int, default=None,
        help="Number of Monte Carlo paths (default: 1000)",
    )
    parser.add_argument(
        "--min-edge", type=float, default=None,
        help="Minimum edge fraction to trade (default: 0.05 = 5%%)",
    )
    parser.add_argument(
        "--bankroll", type=float, default=None,
        help="Starting bankroll in USD (default: 500)",
    )
    parser.add_argument(
        "--duration-minutes", type=float, default=0,
        help="How long to run in minutes (0 = indefinitely)",
    )
    parser.add_argument(
        "--scan-interval", type=float, default=None,
        help="Seconds between scan cycles (default: 5.0)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to write trades CSV on exit",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )

    return parser


def _apply_overrides(settings: CryptoSettings, args: argparse.Namespace) -> CryptoSettings:
    """Apply CLI argument overrides to settings."""
    from dataclasses import replace

    overrides = {}

    if args.live:
        overrides["paper_mode"] = False

    if args.symbols is not None:
        overrides["symbols"] = [s.upper() for s in args.symbols]

    if args.mc_paths is not None:
        overrides["mc_num_paths"] = args.mc_paths

    if args.min_edge is not None:
        overrides["min_edge_pct"] = args.min_edge
        overrides["min_edge_cents"] = args.min_edge

    if args.bankroll is not None:
        overrides["bankroll"] = args.bankroll

    if args.scan_interval is not None:
        overrides["scan_interval_seconds"] = args.scan_interval

    if overrides:
        settings = replace(settings, **overrides)

    return settings


async def _async_main(settings: CryptoSettings, args: argparse.Namespace) -> None:
    """Async entry point."""
    kalshi_adapter = None

    if not settings.paper_mode:
        # Try to build Kalshi adapter for live trading
        try:
            from arb_bot.config import load_settings
            from arb_bot.exchanges.kalshi import KalshiAdapter

            app_settings = load_settings()
            kalshi_adapter = KalshiAdapter(app_settings.kalshi)
            await kalshi_adapter.connect()
        except Exception as exc:
            logging.getLogger(__name__).error(
                "Failed to connect Kalshi adapter: %s", exc
            )
            print(f"Error: Could not connect to Kalshi: {exc}", file=sys.stderr)
            print("Run with --paper for paper trading mode.", file=sys.stderr)
            sys.exit(1)

    engine = CryptoEngine(settings, kalshi_adapter=kalshi_adapter)

    try:
        await engine.run(duration_minutes=args.duration_minutes)
    except KeyboardInterrupt:
        engine.stop()
    finally:
        # Export trades if requested
        if args.output:
            csv_data = engine.export_trades_csv()
            with open(args.output, "w") as f:
                f.write(csv_data)
            print(f"Trades exported to {args.output}")

        # Print summary
        trades = engine.trades
        print(f"\n{'='*60}")
        print(f"Crypto Engine Session Summary")
        print(f"{'='*60}")
        print(f"Cycles:    {engine.cycle_count}")
        print(f"Trades:    {len(trades)}")
        if trades:
            wins = sum(1 for t in trades if t.pnl > 0)
            print(f"Wins:      {wins} ({wins/len(trades)*100:.0f}%)")
            print(f"Net PnL:   ${engine.session_pnl:.2f}")
        print(f"Bankroll:  ${engine.bankroll:.2f}")
        print(f"{'='*60}")


def main() -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load settings from env vars, then apply CLI overrides
    settings = load_crypto_settings()
    settings = _apply_overrides(settings, args)

    # Run
    asyncio.run(_async_main(settings, args))


if __name__ == "__main__":
    main()
