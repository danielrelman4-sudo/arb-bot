"""Paper test script for crypto prediction engine.

Fetches live BTC/SOL prices from Binance REST API and crypto market
quotes from Kalshi's public API, runs MC model to find edge.

Usage::

    python3 -m arb_bot.crypto.paper_test --duration-minutes 5
    python3 -m arb_bot.crypto.paper_test --symbols BTC SOL --mc-paths 2000
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import re
import time
from datetime import datetime, timedelta, timezone
from dataclasses import replace

import httpx
import numpy as np

from arb_bot.crypto.config import CryptoSettings
from arb_bot.crypto.edge_detector import CryptoEdge, EdgeDetector, compute_implied_probability
from arb_bot.crypto.engine import CryptoEngine
from arb_bot.crypto.market_scanner import (
    CryptoMarket,
    CryptoMarketMeta,
    CryptoMarketQuote,
)
from arb_bot.crypto.price_feed import PriceTick
from arb_bot.crypto.price_model import PriceModel

LOGGER = logging.getLogger(__name__)

KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"
BINANCE_KLINES = "https://api.binance.us/api/v3/klines"
BINANCE_PRICE = "https://api.binance.us/api/v3/ticker/price"
BINANCE_AGG_TRADES = "https://api.binance.us/api/v3/aggTrades"

# Map underlying to Kalshi series tickers
_SERIES_MAP = {
    "BTC": ["KXBTC15M", "KXBTCD"],
    "ETH": ["KXETH15M", "KXETHD"],
    "SOL": ["KXSOL15M", "KXSOLD"],
}

_BINANCE_MAP = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
}


# ── Binance REST price feed ────────────────────────────────────────

async def fetch_binance_prices(
    symbols: list[str],
    client: httpx.AsyncClient,
) -> dict[str, float]:
    """Fetch current prices from Binance US REST API."""
    prices: dict[str, float] = {}
    for sym in symbols:
        try:
            resp = await client.get(
                BINANCE_PRICE,
                params={"symbol": sym},
                timeout=5.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                prices[sym.lower()] = float(data["price"])
        except Exception as exc:
            LOGGER.debug("Binance price fetch %s failed: %s", sym, exc)
    return prices


async def fetch_binance_klines(
    symbol: str,
    minutes: int,
    client: httpx.AsyncClient,
) -> list[tuple[float, float]]:
    """Fetch 1-minute klines (timestamp, close) from Binance US."""
    try:
        resp = await client.get(
            BINANCE_KLINES,
            params={
                "symbol": symbol,
                "interval": "1m",
                "limit": min(minutes, 1000),
            },
            timeout=10.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            return [(float(k[0]) / 1000.0, float(k[4])) for k in data]
    except Exception as exc:
        LOGGER.warning("Binance klines %s failed: %s", symbol, exc)
    return []


async def fetch_binance_agg_trades(
    symbol: str,
    limit: int,
    client: httpx.AsyncClient,
) -> list[dict]:
    """Fetch recent aggregate trades from Binance US for OFI bootstrap."""
    try:
        resp = await client.get(
            BINANCE_AGG_TRADES,
            params={
                "symbol": symbol,
                "limit": min(limit, 1000),
            },
            timeout=10.0,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception as exc:
        LOGGER.warning("Binance aggTrades %s failed: %s", symbol, exc)
    return []


# ── Kalshi market parsing ──────────────────────────────────────────

def _infer_underlying(ticker: str) -> str:
    """Infer underlying from ticker prefix."""
    t = ticker.upper()
    if "BTC" in t:
        return "BTC"
    elif "ETH" in t:
        return "ETH"
    elif "SOL" in t:
        return "SOL"
    return "UNKNOWN"


def _infer_interval(series: str) -> str:
    """Infer interval from series ticker."""
    s = series.upper()
    if "15M" in s:
        return "15min"
    elif "1H" in s:
        return "1hour"
    elif s.endswith("D"):
        return "daily"
    return "unknown"


def _infer_direction_and_strike(
    ticker: str, subtitle: str, series: str,
) -> tuple[str, float | None]:
    """Infer direction and strike from ticker/subtitle."""
    t = ticker.upper()
    # Daily above/below: has -T or -B followed by number
    m = re.search(r"-T([\d.]+)$", t)
    if m:
        return "above", float(m.group(1))
    m = re.search(r"-B([\d.]+)$", t)
    if m:
        return "below", float(m.group(1))

    # 15M up/down: series ends in 15M, suffix is -00 or similar
    if "15M" in series.upper() or "1H" in series.upper():
        # These are up/down markets — check subtitle or default to up
        sub_lower = subtitle.lower()
        if "down" in sub_lower or "decrease" in sub_lower:
            return "down", None
        return "up", None

    return "up", None


def kalshi_raw_to_quote(
    raw: dict,
    now: datetime,
) -> CryptoMarketQuote | None:
    """Convert a raw Kalshi market dict to CryptoMarketQuote.

    Instead of parsing the ticker format (which varies), we use
    the close_time, series_ticker, subtitle, and price fields directly.
    """
    ticker = raw.get("ticker", "")
    series = raw.get("series_ticker", "")
    subtitle = raw.get("subtitle", "")
    status = raw.get("status", "")

    if status not in ("active", "open"):
        return None

    # Parse close time
    close_str = raw.get("close_time", "")
    if not close_str:
        return None
    try:
        close_str = close_str.replace("Z", "+00:00")
        close_time = datetime.fromisoformat(close_str)
    except (ValueError, TypeError):
        return None

    tte_minutes = (close_time - now).total_seconds() / 60.0
    if tte_minutes <= 0:
        return None

    # Infer metadata
    underlying = _infer_underlying(ticker)
    interval = _infer_interval(series)
    direction, strike = _infer_direction_and_strike(ticker, subtitle, series)

    # Compute interval_start_time for up/down contracts
    interval_start_time = None
    if direction in ("up", "down"):
        # Use open_time from API if available, otherwise derive from close_time
        open_str = raw.get("open_time", "")
        if open_str:
            try:
                open_str_clean = open_str.replace("Z", "+00:00")
                interval_start_time = datetime.fromisoformat(open_str_clean)
            except (ValueError, TypeError):
                pass
        if interval_start_time is None:
            if interval == "15min":
                interval_start_time = close_time - timedelta(minutes=15)
            elif interval == "1hour":
                interval_start_time = close_time - timedelta(hours=1)

    meta = CryptoMarketMeta(
        underlying=underlying,
        interval=interval,
        expiry=close_time,
        strike=strike,
        direction=direction,
        series_ticker=series,
        interval_index=None,
        interval_start_time=interval_start_time,
    )
    market = CryptoMarket(ticker=ticker, meta=meta)

    # Extract prices (Kalshi returns in cents)
    yes_bid = (raw.get("yes_bid") or 0) / 100.0
    yes_ask = (raw.get("yes_ask") or 0) / 100.0
    no_bid = (raw.get("no_bid") or 0) / 100.0
    no_ask = (raw.get("no_ask") or 0) / 100.0

    yes_buy = yes_ask if yes_ask > 0 else 0.50
    no_buy = no_ask if no_ask > 0 else 0.50

    # Skip markets with no real book (yes_ask=1¢ or no_ask=1¢)
    if yes_buy <= 0.02 or no_buy <= 0.02:
        return None

    # ── Book quality filters ──
    # A tradeable market needs BOTH sides to have real bids.
    # If yes_bid=0, there are no YES buyers — the YES ask is just the
    # platform minimum (50¢). Similarly if no_bid=0.
    # Even if one side has bids, we can't reliably compute implied
    # probability from a one-sided book, and we certainly can't trade
    # against phantom liquidity.
    has_yes_book = yes_bid > 0.01  # Real YES bids exist
    has_no_book = no_bid > 0.01   # Real NO bids exist

    if not has_yes_book or not has_no_book:
        # One-sided or empty book — skip entirely
        return None

    # Also skip if the spread is impossibly wide (> 40¢ on either side)
    yes_spread = yes_ask - yes_bid if yes_bid > 0 else 1.0
    no_spread = no_ask - no_bid if no_bid > 0 else 1.0
    if yes_spread > 0.40 or no_spread > 0.40:
        return None

    # Use bid/ask midpoints for more accurate implied probability
    yes_mid = (yes_bid + yes_ask) / 2.0
    no_mid = (no_bid + no_ask) / 2.0
    implied = compute_implied_probability(yes_mid, no_mid)

    return CryptoMarketQuote(
        market=market,
        yes_buy_price=yes_buy,
        no_buy_price=no_buy,
        yes_buy_size=raw.get("volume", 0),
        no_buy_size=raw.get("volume", 0),
        yes_bid_price=yes_bid if yes_bid > 0 else None,
        no_bid_price=no_bid if no_bid > 0 else None,
        time_to_expiry_minutes=tte_minutes,
        implied_probability=implied,
    )


# ── Kalshi market fetching ─────────────────────────────────────────

async def fetch_kalshi_crypto_markets(
    underlyings: list[str],
    client: httpx.AsyncClient,
) -> list[dict]:
    """Fetch crypto markets from Kalshi public API."""
    all_markets: list[dict] = []

    for underlying in underlyings:
        series_list = _SERIES_MAP.get(underlying, [])
        for series in series_list:
            try:
                resp = await client.get(
                    f"{KALSHI_API}/markets",
                    params={
                        "series_ticker": series,
                        "status": "open",
                        "limit": 200,
                    },
                    timeout=10.0,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    markets = data.get("markets", [])
                    # Inject series_ticker since API might not return it
                    for m in markets:
                        m.setdefault("series_ticker", series)
                    all_markets.extend(markets)
                    LOGGER.info("Kalshi: %d markets for %s", len(markets), series)
                else:
                    LOGGER.warning("Kalshi %s: HTTP %d", series, resp.status_code)
            except Exception as exc:
                LOGGER.warning("Kalshi %s: %s", series, exc)

    return all_markets


# ── Main test loop ─────────────────────────────────────────────────

async def run_paper_test(
    underlyings: list[str],
    duration_minutes: float,
    mc_paths: int,
    min_edge: float,
    scan_interval: float,
    max_tte: int,
) -> None:
    """Run the crypto engine paper test."""

    binance_symbols = [_BINANCE_MAP[u] for u in underlyings if u in _BINANCE_MAP]

    settings = CryptoSettings(
        enabled=True,
        paper_mode=True,
        symbols=[f"KX{u}" for u in underlyings],
        price_feed_symbols=[s.lower() for s in binance_symbols],
        mc_num_paths=mc_paths,
        min_edge_pct=min_edge,
        min_edge_pct_daily=0.06,
        min_edge_cents=min_edge,
        max_model_uncertainty=0.20,
        model_uncertainty_multiplier=3.0,
        bankroll=500.0,
        max_position_per_market=50.0,
        max_concurrent_positions=10,
        max_positions_per_underlying=3,
        kelly_fraction_cap=0.10,
        scan_interval_seconds=scan_interval,
        paper_slippage_cents=0.5,
        confidence_level=0.95,
        mc_vol_window_minutes=30,
        min_minutes_to_expiry=1,
        max_minutes_to_expiry=max_tte,
        min_book_depth_contracts=1,
        allowed_directions=["above", "below", "up", "down"],
        # OFI microstructure drift
        ofi_enabled=True,
        ofi_window_seconds=600,
        ofi_alpha=0.0,  # starts neutral, calibrated at runtime
        ofi_recalibrate_interval_hours=4.0,
        # Activity-scaled volatility
        activity_scaling_enabled=True,
        activity_scaling_short_window_seconds=300,
        activity_scaling_long_window_seconds=3600,
        # Jump diffusion
        use_jump_diffusion=True,
        mc_jump_intensity=3.0,
        mc_jump_mean=0.0,
        mc_jump_vol=0.02,
    )

    engine = CryptoEngine(settings)

    print(f"\n{'='*70}")
    print(f"  Crypto Prediction Engine — Paper Test v2")
    print(f"  (OFI drift + activity scaling + jump diffusion + 15m markets)")
    print(f"{'='*70}")
    print(f"  Underlyings:    {', '.join(underlyings)}")
    print(f"  Binance feeds:  {', '.join(binance_symbols)}")
    print(f"  MC paths:       {mc_paths}")
    print(f"  Min edge:       {min_edge*100:.1f}% (daily: 6.0%)")
    print(f"  Duration:       {duration_minutes} min")
    print(f"  Max TTE:        {max_tte} min")
    print(f"  Scan interval:  {scan_interval}s")
    print(f"  Directions:     above, below, up, down (ref-price fixed)")
    print(f"  OFI drift:      enabled (window=600s, recal=4h)")
    print(f"  Activity scale: enabled (short=300s, long=3600s)")
    print(f"  Jump diffusion: enabled (λ=3/day, μ=0, σ_j=0.02)")
    print(f"  Unc multiplier: 3.0x")
    print(f"  Max per-UL:     3 positions")
    print(f"{'='*70}\n")

    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=5.0)) as client:
        # 1. Bootstrap with Binance klines
        print("Loading historical prices from Binance US REST API...")
        for sym in binance_symbols:
            klines = await fetch_binance_klines(sym, 60, client)
            if klines:
                for ts, close in klines:
                    engine.price_feed.inject_tick(
                        PriceTick(symbol=sym.lower(), price=close, timestamp=ts, volume=0)
                    )
                print(f"   {sym}: {len(klines)} 1m candles, latest=${klines[-1][1]:,.2f}")
            else:
                print(f"   {sym}: no data (API may be unavailable)")

        # 2. Fetch current prices
        print("\nFetching current prices...")
        prices = await fetch_binance_prices(binance_symbols, client)
        for sym, price in prices.items():
            engine.price_feed.inject_tick(
                PriceTick(symbol=sym, price=price, timestamp=time.time(), volume=0)
            )
            print(f"   {sym.upper()}: ${price:,.2f}")

        if not prices:
            print("   WARNING: No Binance prices available. Using historical data only.")

        # 2b. Bootstrap OFI data from aggTrades
        print("\nBootstrapping OFI data from Binance aggTrades...")
        for sym in binance_symbols:
            trades = await fetch_binance_agg_trades(sym, 1000, client)
            if trades:
                buy_count = 0
                sell_count = 0
                for trade in trades:
                    try:
                        ts = float(trade["T"]) / 1000.0
                        price = float(trade["p"])
                        qty = float(trade["q"])
                        is_buyer_maker = trade.get("m")
                    except (KeyError, ValueError, TypeError):
                        continue
                    tick = PriceTick(
                        symbol=sym.lower(), price=price, timestamp=ts,
                        volume=qty, is_buyer_maker=is_buyer_maker,
                    )
                    engine.price_feed.inject_tick(tick)
                    if isinstance(is_buyer_maker, bool):
                        if is_buyer_maker:
                            sell_count += 1
                        else:
                            buy_count += 1
                ofi = engine.price_feed.get_ofi(sym.lower(), window_seconds=600)
                print(
                    f"   {sym}: {len(trades)} trades loaded, "
                    f"{buy_count} buys / {sell_count} sells, "
                    f"OFI={ofi:+.3f}"
                )

        # 3. Show vol estimates
        print("\nVolatility estimates:")
        for sym in [s.lower() for s in binance_symbols]:
            returns = engine.price_feed.get_returns(sym, interval_seconds=60)
            if len(returns) >= 2:
                model = PriceModel(num_paths=100)
                vol = model.estimate_volatility(returns, interval_seconds=60)
                print(f"   {sym.upper()}: {vol*100:.1f}% annualized ({len(returns)} returns)")
            else:
                print(f"   {sym.upper()}: insufficient data for vol estimate")

        # 4. Main loop
        start_time = time.monotonic()
        cycle = 0

        while True:
            elapsed = (time.monotonic() - start_time) / 60.0
            if elapsed >= duration_minutes:
                break

            cycle += 1
            now = datetime.now(timezone.utc)
            print(f"\n{'─'*50}")
            print(f"Cycle {cycle} ({elapsed:.1f}m / {duration_minutes}m)")
            print(f"{'─'*50}")

            # Refresh prices + trade flow for OFI
            new_prices = await fetch_binance_prices(binance_symbols, client)
            for sym, price in new_prices.items():
                engine.price_feed.inject_tick(
                    PriceTick(symbol=sym, price=price, timestamp=time.time(), volume=0)
                )

            # Refresh aggTrades for live OFI updates
            for sym in binance_symbols:
                trades = await fetch_binance_agg_trades(sym, 100, client)
                for trade in trades:
                    try:
                        ts = float(trade["T"]) / 1000.0
                        price_t = float(trade["p"])
                        qty = float(trade["q"])
                        is_bm = trade.get("m")
                    except (KeyError, ValueError, TypeError):
                        continue
                    engine.price_feed.inject_tick(PriceTick(
                        symbol=sym.lower(), price=price_t, timestamp=ts,
                        volume=qty, is_buyer_maker=is_bm,
                    ))

            for sym in [s.lower() for s in binance_symbols]:
                p = engine.price_feed.get_current_price(sym)
                ofi = engine.price_feed.get_ofi(sym, window_seconds=300)
                if p:
                    print(f"   {sym.upper()}: ${p:,.2f}  OFI={ofi:+.3f}")

            # Fetch Kalshi markets
            raw_markets = await fetch_kalshi_crypto_markets(underlyings, client)

            # Parse into quotes
            quotes: list[CryptoMarketQuote] = []
            skipped = 0
            for raw in raw_markets:
                q = kalshi_raw_to_quote(raw, now)
                if q is not None and q.time_to_expiry_minutes <= max_tte:
                    quotes.append(q)
                else:
                    skipped += 1

            print(f"   Markets: {len(quotes)} tradeable, {skipped} filtered")

            if quotes:
                # Group by interval type
                by_interval: dict[str, int] = {}
                for q in quotes:
                    by_interval[q.market.meta.interval] = by_interval.get(q.market.meta.interval, 0) + 1
                print(f"   Breakdown: {by_interval}")

                # Show market examples (closest to expiry)
                for q in sorted(quotes, key=lambda x: x.time_to_expiry_minutes)[:8]:
                    strike_str = f"${q.market.meta.strike:,.0f}" if q.market.meta.strike else "N/A"
                    bid_str = f"Ybid={q.yes_bid_price:.2f}" if q.yes_bid_price else "Ybid=   -"
                    nbid_str = f"Nbid={q.no_bid_price:.2f}" if q.no_bid_price else "Nbid=   -"
                    print(
                        f"     {q.market.ticker[:35]:35s} "
                        f"{q.market.meta.direction:6s} "
                        f"strike={strike_str:>10s} "
                        f"YES={q.yes_buy_price:.2f} NO={q.no_buy_price:.2f} "
                        f"{bid_str} {nbid_str} "
                        f"impl={q.implied_probability:.0%} "
                        f"TTE={q.time_to_expiry_minutes:.0f}m"
                    )

                # Run MC model
                edges = await engine.run_cycle_with_quotes(quotes)

                if edges:
                    print(f"\n   EDGES DETECTED: {len(edges)}")
                    for e in edges[:8]:
                        print(
                            f"     {e.market.ticker[:35]:35s} → {e.side.upper():3s} "
                            f"edge={e.edge_cents*100:+.1f}% "
                            f"model={e.model_prob.probability:.0%} "
                            f"market={e.market_implied_prob:.0%} "
                            f"unc={e.model_uncertainty:.3f}"
                        )
                else:
                    print("   No edges above threshold")
            else:
                print("   No markets to evaluate")

            # Position/PnL summary
            if engine.positions:
                print(f"\n   Open positions: {len(engine.positions)}")
                for ticker, pos in engine.positions.items():
                    print(f"     {ticker}: {pos.side.upper()} {pos.contracts}@{pos.entry_price:.2f}")

            if engine.trades:
                wins = sum(1 for t in engine.trades if t.pnl > 0)
                print(f"   Trades: {len(engine.trades)} (wins={wins}) PnL=${engine.session_pnl:.2f}")

            print(f"   Bankroll: ${engine.bankroll:.2f}")

            await asyncio.sleep(scan_interval)

    # Force-settle any remaining open positions using model probability
    # so we can evaluate PnL even on a short test run
    open_positions = list(engine.positions.keys())
    if open_positions:
        print(f"\n  Settling {len(open_positions)} open positions using model probability...")
        for ticker in open_positions:
            pos = engine.positions.get(ticker)
            if pos is None:
                continue
            # Settle using model prob — simulate outcome
            settled_yes = np.random.random() < pos.model_prob
            record = engine.settle_position_with_outcome(ticker, settled_yes)
            if record:
                outcome_str = "YES" if settled_yes else "NO"
                print(
                    f"    {ticker}: settled {outcome_str} → "
                    f"PnL=${record.pnl:+.2f} ({record.side.upper()} "
                    f"{record.contracts}@{record.entry_price:.2f})"
                )

    # Final summary
    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS")
    print(f"{'='*70}")
    print(f"  Cycles:         {cycle}")
    print(f"  Trades:         {len(engine.trades)}")
    if engine.trades:
        wins = sum(1 for t in engine.trades if t.pnl > 0)
        losses = sum(1 for t in engine.trades if t.pnl < 0)
        avg_edge = sum(t.edge_at_entry for t in engine.trades) / len(engine.trades)
        total_pnl = sum(t.pnl for t in engine.trades)
        avg_contracts = sum(t.contracts for t in engine.trades) / len(engine.trades)
        print(f"  Wins:           {wins} ({wins/len(engine.trades)*100:.0f}%)")
        print(f"  Losses:         {losses}")
        print(f"  Avg edge:       {avg_edge*100:.2f}%")
        print(f"  Avg contracts:  {avg_contracts:.0f}")
        print(f"  Total PnL:      ${total_pnl:+.2f}")
        print()
        print(f"  Per-trade breakdown:")
        for t in engine.trades:
            print(
                f"    {t.ticker[:40]:40s} {t.side.upper():3s} "
                f"{t.contracts:3d}@{t.entry_price:.2f} "
                f"PnL=${t.pnl:+.2f} "
                f"edge={t.edge_at_entry*100:.1f}% "
                f"model={t.model_prob_at_entry:.0%} market={t.market_prob_at_entry:.0%}"
            )
    print(f"\n  Net PnL:        ${engine.session_pnl:+.2f}")
    print(f"  Final bankroll: ${engine.bankroll:.2f}")
    print(f"{'='*70}")

    # Export
    csv_data = engine.export_trades_csv()
    if engine.trades:
        fname = f"crypto_paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path = f"/Users/danielelman/Documents/New project/arb_bot/.claude/worktrees/focused-mclean/arb_bot/output/{fname}"
        try:
            with open(path, "w") as f:
                f.write(csv_data)
            print(f"\n  Trades exported to: {path}")
        except Exception as exc:
            print(f"\n  CSV export failed: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Crypto engine paper test")
    parser.add_argument("--symbols", nargs="+", default=["BTC", "SOL"],
                        help="Underlyings to trade (default: BTC SOL)")
    parser.add_argument("--duration-minutes", type=float, default=5,
                        help="How long to run (default: 5)")
    parser.add_argument("--mc-paths", type=int, default=1000,
                        help="Monte Carlo paths (default: 1000)")
    parser.add_argument("--min-edge", type=float, default=0.03,
                        help="Min edge fraction (default: 0.03 = 3%%)")
    parser.add_argument("--scan-interval", type=float, default=15.0,
                        help="Seconds between scans (default: 15)")
    parser.add_argument("--max-tte", type=int, default=600,
                        help="Max time-to-expiry in minutes (default: 600)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    asyncio.run(run_paper_test(
        underlyings=[u.upper() for u in args.symbols],
        duration_minutes=args.duration_minutes,
        mc_paths=args.mc_paths,
        min_edge=args.min_edge,
        scan_interval=args.scan_interval,
        max_tte=args.max_tte,
    ))


if __name__ == "__main__":
    main()
