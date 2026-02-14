#!/usr/bin/env python3
"""Discover all ForecastEx prediction market symbols and contracts.

Uses two complementary approaches:
  1. Client Portal API (localhost:5000) — GET /trsrv/event/category-tree
     Returns the full hierarchy of prediction market categories and symbols.
     Requires the Client Portal Gateway running and authenticated.

  2. TWS/IB Gateway API (localhost:4002) — reqContractDetails per symbol
     Fetches individual contract details for each discovered symbol.
     Requires IB Gateway or TWS running.

Setup:
  Client Portal Gateway:
    1. Download: https://download2.interactivebrokers.com/portal/clientportal.gw.zip
    2. Unzip and run: bin/run.sh root/conf.yaml
    3. Navigate to https://localhost:5000 and authenticate
    4. Keep it running while this script executes

  IB Gateway:
    1. Start IB Gateway (paper: port 4002, live: port 4001)
    2. Enable API connections in Configure → API → Settings

Usage:
    python3 arb_bot/scripts/discover_forecastex_markets.py
    python3 arb_bot/scripts/discover_forecastex_markets.py --cp-port 5000 --tws-port 4002
    python3 arb_bot/scripts/discover_forecastex_markets.py --cp-only
    python3 arb_bot/scripts/discover_forecastex_markets.py --tws-only --symbols FF,GCE,USIP
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import ssl
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import ib_async as ibasync
except ImportError:
    try:
        import ib_insync as ibasync  # type: ignore[no-redef]
    except ImportError:
        ibasync = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ForecastMarket:
    """A single ForecastEx market (group of related contracts)."""
    name: str
    symbol: str
    exchange: str
    conid: int
    category: str = ""
    subcategory: str = ""
    region: str = ""


@dataclass
class ForecastContract:
    """A single tradeable ForecastEx contract."""
    symbol: str
    trading_class: str
    exchange: str
    sec_type: str
    right: str  # C=Yes, P=No
    strike: float
    expiry: str
    conid: int = 0
    description: str = ""


# ---------------------------------------------------------------------------
# Client Portal API discovery
# ---------------------------------------------------------------------------

def _cp_get(host: str, port: int, path: str) -> Any:
    """Make an authenticated GET to the Client Portal Gateway."""
    # Client Portal Gateway uses self-signed certs — skip verification
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.load_default_certs()
    ctx.verify_mode = ssl.CERT_NONE

    url = f"https://{host}:{port}/v1/api{path}"
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        raise RuntimeError(f"HTTP {e.code} from {url}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Cannot reach Client Portal Gateway at {host}:{port}. "
            f"Is it running? Error: {e.reason}"
        ) from e


def discover_via_client_portal(
    host: str = "localhost",
    port: int = 5000,
) -> List[ForecastMarket]:
    """Fetch the full ForecastEx category tree from Client Portal API."""
    print(f"\n{'='*60}")
    print("CLIENT PORTAL API DISCOVERY")
    print(f"{'='*60}")
    print(f"Connecting to Client Portal Gateway at {host}:{port} ...")

    # Check authentication status
    try:
        status = _cp_get(host, port, "/iserver/auth/status")
        print(f"  Auth status: authenticated={status.get('authenticated')}")
        if not status.get("authenticated"):
            print("  WARNING: Not authenticated. Navigate to")
            print(f"  https://{host}:{port} in your browser and log in.")
            return []
    except RuntimeError as e:
        print(f"  ERROR: {e}")
        return []

    # Tickle to keep session alive
    try:
        _cp_get(host, port, "/tickle")
    except Exception:
        pass

    # Fetch category tree
    print("  Fetching category tree ...")
    try:
        tree = _cp_get(host, port, "/trsrv/event/category-tree")
    except RuntimeError as e:
        print(f"  ERROR fetching category tree: {e}")
        return []

    # Parse the tree into markets
    markets: List[ForecastMarket] = []
    categories: Dict[str, dict] = {}

    # First pass: index all categories
    for key, val in tree.items():
        categories[key] = val

    # Second pass: resolve parent chain and extract markets
    def resolve_path(cat_id: str) -> tuple:
        """Return (top_category, subcategory, region) for a leaf."""
        chain = []
        cid = cat_id
        seen = set()
        while cid and cid not in seen:
            seen.add(cid)
            cat = categories.get(cid, {})
            chain.append(cat.get("label", ""))
            cid = cat.get("parentId", "")
        chain.reverse()
        # chain is [top, sub, region, ...]
        top = chain[0] if len(chain) > 0 else ""
        sub = chain[1] if len(chain) > 1 else ""
        region = chain[2] if len(chain) > 2 else ""
        return top, sub, region

    for key, val in tree.items():
        for mkt in val.get("markets", []):
            top, sub, region = resolve_path(key)
            markets.append(ForecastMarket(
                name=mkt.get("name", ""),
                symbol=mkt.get("symbol", ""),
                exchange=mkt.get("exchange", "FORECASTX"),
                conid=mkt.get("conid", 0),
                category=top,
                subcategory=sub,
                region=region,
            ))

    # Print summary
    print(f"\n  Found {len(markets)} ForecastEx markets:")
    by_cat: Dict[str, list] = {}
    for m in markets:
        cat_label = f"{m.category} > {m.subcategory}" if m.subcategory else m.category
        by_cat.setdefault(cat_label, []).append(m)

    for cat, mlist in sorted(by_cat.items()):
        print(f"\n  [{cat}]")
        for m in sorted(mlist, key=lambda x: x.symbol):
            exch = f" ({m.exchange})" if m.exchange != "FORECASTX" else ""
            print(f"    {m.symbol:12s} {m.name}{exch}  [conid={m.conid}]")

    return markets


# ---------------------------------------------------------------------------
# TWS API discovery
# ---------------------------------------------------------------------------

async def discover_via_tws(
    host: str = "127.0.0.1",
    port: int = 4002,
    client_id: int = 51,
    symbols: Optional[List[str]] = None,
    rate_limit_delay: float = 0.5,
) -> List[ForecastContract]:
    """Fetch contract details for given symbols via TWS API."""
    if ibasync is None:
        print("\n  ERROR: ib_insync/ib_async not installed. Cannot use TWS API.")
        return []

    print(f"\n{'='*60}")
    print("TWS API CONTRACT DISCOVERY")
    print(f"{'='*60}")

    if not symbols:
        print("  No symbols provided. Use --symbols or run with --cp-only first.")
        return []

    ib = ibasync.IB()
    print(f"  Connecting to {host}:{port} (clientId={client_id}) ...")

    try:
        await ib.connectAsync(host, port, clientId=client_id, timeout=10)
    except Exception as exc:
        print(f"  ERROR: Could not connect to TWS/Gateway at {host}:{port}")
        print(f"    {type(exc).__name__}: {exc}")
        return []

    print(f"  Connected! Server version: {ib.client.serverVersion()}")

    all_contracts: List[ForecastContract] = []

    for sym in symbols:
        print(f"\n  Querying {sym} ...")
        try:
            partial = ibasync.Contract(
                symbol=sym,
                secType="OPT",
                exchange="FORECASTX",
                currency="USD",
            )
            details = await ib.reqContractDetailsAsync(partial)
            print(f"    Found {len(details)} contracts for {sym}")

            for d in details:
                c = d.contract
                fc = ForecastContract(
                    symbol=c.symbol,
                    trading_class=c.tradingClass,
                    exchange=c.exchange,
                    sec_type=c.secType,
                    right=c.right,
                    strike=c.strike,
                    expiry=c.lastTradeDateOrContractMonth,
                    conid=c.conId,
                    description=getattr(d, "longName", ""),
                )
                all_contracts.append(fc)

            # Print sample
            seen_expiries = set()
            for d in details:
                c = d.contract
                exp = c.lastTradeDateOrContractMonth
                if exp not in seen_expiries:
                    seen_expiries.add(exp)
                    right_label = "YES" if c.right == "C" else "NO"
                    print(f"      {c.tradingClass} {exp} strike={c.strike} {right_label} conid={c.conId}")
                    if len(seen_expiries) >= 5:
                        remaining = len(set(
                            d2.contract.lastTradeDateOrContractMonth for d2 in details
                        )) - 5
                        if remaining > 0:
                            print(f"      ... and {remaining} more expiries")
                        break

        except Exception as exc:
            print(f"    ERROR querying {sym}: {exc}")

        # Rate limit: IB has 10 req/s limit with penalty box
        await asyncio.sleep(rate_limit_delay)

    ib.disconnect()
    print(f"\n  Total: {len(all_contracts)} contracts across {len(symbols)} symbols")
    return all_contracts


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_markets_csv(markets: List[ForecastMarket], path: Path) -> None:
    """Save discovered markets to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "name", "exchange", "conid", "category", "subcategory", "region"])
        for m in sorted(markets, key=lambda x: (x.category, x.symbol)):
            w.writerow([m.symbol, m.name, m.exchange, m.conid, m.category, m.subcategory, m.region])
    print(f"\n  Saved {len(markets)} markets to {path}")


def save_contracts_csv(contracts: List[ForecastContract], path: Path) -> None:
    """Save discovered contracts to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "trading_class", "exchange", "right", "strike", "expiry", "conid", "description"])
        for c in sorted(contracts, key=lambda x: (x.symbol, x.expiry, x.strike, x.right)):
            w.writerow([c.symbol, c.trading_class, c.exchange, c.right, c.strike, c.expiry, c.conid, c.description])
    print(f"  Saved {len(contracts)} contracts to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Discover all ForecastEx prediction market symbols and contracts"
    )
    parser.add_argument("--cp-host", default="localhost", help="Client Portal Gateway host")
    parser.add_argument("--cp-port", type=int, default=5000, help="Client Portal Gateway port")
    parser.add_argument("--tws-host", default="127.0.0.1", help="TWS/IB Gateway host")
    parser.add_argument("--tws-port", type=int, default=4002, help="TWS/IB Gateway port (4002=paper)")
    parser.add_argument("--client-id", type=int, default=51, help="TWS client ID")
    parser.add_argument("--cp-only", action="store_true", help="Only use Client Portal API")
    parser.add_argument("--tws-only", action="store_true", help="Only use TWS API")
    parser.add_argument(
        "--symbols", type=str, default="",
        help="Comma-separated symbols for TWS discovery (e.g. FF,GCE,USIP)"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="arb_bot/config",
        help="Directory to save output CSVs"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    markets: List[ForecastMarket] = []
    contracts: List[ForecastContract] = []

    # Phase 1: Client Portal API discovery
    if not args.tws_only:
        markets = discover_via_client_portal(args.cp_host, args.cp_port)
        if markets:
            save_markets_csv(markets, output_dir / "forecastex_markets.csv")

    # Phase 2: TWS API contract discovery
    if not args.cp_only:
        # Use symbols from CP discovery, or from --symbols flag
        symbols: List[str] = []
        if args.symbols:
            symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        elif markets:
            symbols = list(set(m.symbol for m in markets))
            print(f"\n  Using {len(symbols)} symbols from Client Portal discovery")

        if symbols:
            contracts = asyncio.run(discover_via_tws(
                host=args.tws_host,
                port=args.tws_port,
                client_id=args.client_id,
                symbols=symbols,
            ))
            if contracts:
                save_contracts_csv(contracts, output_dir / "forecastex_contracts.csv")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    if markets:
        print(f"  Markets discovered: {len(markets)}")
        fx_count = sum(1 for m in markets if m.exchange == "FORECASTX")
        cme_count = sum(1 for m in markets if m.exchange != "FORECASTX")
        print(f"    ForecastEx: {fx_count}")
        if cme_count:
            print(f"    CME/other:  {cme_count}")

        # Print unique symbols for easy copy-paste
        symbols_list = sorted(set(m.symbol for m in markets))
        print(f"\n  All symbols ({len(symbols_list)}):")
        print(f"    {','.join(symbols_list)}")
    else:
        print("  No markets discovered via Client Portal API.")

    if contracts:
        print(f"\n  Contracts discovered: {len(contracts)}")
        by_sym = {}
        for c in contracts:
            by_sym.setdefault(c.symbol, []).append(c)
        for sym, clist in sorted(by_sym.items()):
            expiries = sorted(set(c.expiry for c in clist))
            print(f"    {sym}: {len(clist)} contracts, {len(expiries)} expiries")
    elif not args.cp_only:
        print("  No contracts discovered via TWS API.")

    if not markets and not contracts:
        print("\n  Nothing discovered. Check:")
        print("  1. Is Client Portal Gateway running at localhost:5000?")
        print("     Download: https://download2.interactivebrokers.com/portal/clientportal.gw.zip")
        print("  2. Is IB Gateway running on port 4002?")
        print("  3. Are Event Contract permissions enabled?")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
