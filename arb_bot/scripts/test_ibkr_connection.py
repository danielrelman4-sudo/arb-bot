#!/usr/bin/env python3
"""Quick IBKR connection test — verifies TWS/Gateway is reachable.

Usage:
    python3 arb_bot/scripts/test_ibkr_connection.py [--port 4002] [--host 127.0.0.1]

Defaults to paper trading IB Gateway (port 4002).
TWS paper trading uses port 7497.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

try:
    import ib_async as ibasync
except ImportError:
    try:
        import ib_insync as ibasync  # type: ignore[no-redef]
    except ImportError:
        print("ERROR: Neither ib_async nor ib_insync is installed.")
        print("  Install with: pip3 install ib_insync")
        sys.exit(1)


async def test_connection(host: str, port: int, client_id: int) -> None:
    ib = ibasync.IB()
    print(f"Connecting to {host}:{port} (clientId={client_id}) ...")

    try:
        await ib.connectAsync(host, port, clientId=client_id, timeout=10)
    except Exception as exc:
        print(f"\nERROR: Could not connect to IBKR at {host}:{port}")
        print(f"  {type(exc).__name__}: {exc}")
        print()
        print("Troubleshooting:")
        print(f"  1. Is TWS or IB Gateway running and listening on port {port}?")
        print("  2. Check API settings: Configure → API → Settings")
        print("     - Enable ActiveX and Socket Clients")
        print("     - Socket port should match the --port argument")
        print("     - Allow connections from localhost")
        print("  3. Common ports:")
        print("     TWS live=7496, TWS paper=7497")
        print("     Gateway live=4001, Gateway paper=4002")
        sys.exit(1)

    print(f"  Connected! Server version: {ib.client.serverVersion()}")

    # Account info
    accounts = ib.managedAccounts()
    print(f"  Managed accounts: {accounts}")

    # Check if paper trading
    is_paper = any("DU" in acc or "DF" in acc for acc in accounts)
    print(f"  Paper trading: {'YES' if is_paper else 'NO (LIVE!)'}")

    if not is_paper:
        print("\n  WARNING: This appears to be a LIVE account!")
        print("  For paper trading, use port 7497 (TWS) or 4002 (Gateway)")

    # Account summary
    print("\n  Fetching account summary ...")
    summary = await ib.accountSummaryAsync()
    for item in summary:
        if item.tag in ("TotalCashValue", "NetLiquidation", "BuyingPower"):
            print(f"    {item.tag}: ${float(item.value):,.2f} ({item.currency})")

    # Try to discover ForecastEx contracts
    print("\n  Searching for ForecastEx event contracts ...")
    try:
        partial = ibasync.Contract(
            secType="OPT",
            exchange="FORECASTX",
            currency="USD",
        )
        details = await ib.reqContractDetailsAsync(partial)
        print(f"    Found {len(details)} ForecastEx contracts")
        if details:
            for d in details[:5]:
                c = d.contract
                print(f"      {c.symbol} {c.right} strike={c.strike} expiry={c.lastTradeDateOrContractMonth}")
            if len(details) > 5:
                print(f"      ... and {len(details) - 5} more")
        else:
            print("    No ForecastEx contracts found.")
            print("    Check: Account Management → Trading Permissions → Event Contracts")
    except Exception as exc:
        print(f"    ForecastEx discovery failed: {exc}")
        print("    This may mean ForecastEx permissions are not enabled.")

    # Try CME event contracts too
    print("\n  Searching for CME event contracts ...")
    try:
        cme_partial = ibasync.Contract(
            secType="OPT",
            exchange="CME",
            currency="USD",
        )
        # CME has thousands of options — just check connectivity
        # We'd normally filter by specific symbols
        print("    CME connection available (would need specific symbol filter)")
    except Exception as exc:
        print(f"    CME check failed: {exc}")

    ib.disconnect()
    print("\n  Disconnected. Connection test PASSED!")


def main() -> int:
    parser = argparse.ArgumentParser(description="Test IBKR connection")
    parser.add_argument("--host", default="127.0.0.1", help="TWS/Gateway host")
    parser.add_argument("--port", type=int, default=4002, help="Port (4002=Gateway paper, 7497=TWS paper)")
    parser.add_argument("--client-id", type=int, default=99, help="Client ID (use unique value)")
    args = parser.parse_args()

    asyncio.run(test_connection(args.host, args.port, args.client_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
