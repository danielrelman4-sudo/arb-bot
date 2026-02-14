#!/usr/bin/env python3
"""Build ForecastEx conId catalog using qualifyContractsAsync (penalty-box-safe).

Reads contract parameters from the prices CSV and qualifies them in batches.
This approach works even when reqContractDetailsAsync is throttled.

Usage:
    python3 arb_bot/scripts/build_catalog_fast.py --tws-port 4001
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import ib_insync as ibasync
except ImportError:
    try:
        import ib_async as ibasync  # type: ignore[no-redef]
    except ImportError:
        print("ERROR: ib_insync or ib_async required")
        sys.exit(1)


def parse_prices_csv(path: str) -> List[Dict]:
    """Parse prices CSV into contract parameter dicts."""
    contracts: Dict[Tuple, Dict] = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ec = row["event_contract"]
            subtype = row["subtype"]
            full_exp = row.get("expiration_date", "")

            parts = ec.rsplit("_", 2)
            if len(parts) != 3:
                continue

            symbol = parts[0]
            strike_str = parts[2]
            right = "C" if subtype == "YES" else "P"

            try:
                dt = datetime.fromisoformat(full_exp.replace("Z", ""))
                expiry = dt.strftime("%Y%m%d")
            except (ValueError, AttributeError):
                continue

            try:
                strike = float(strike_str)
            except ValueError:
                continue

            key = (symbol, right, strike, expiry)
            if key not in contracts:
                contracts[key] = {
                    "symbol": symbol,
                    "right": right,
                    "strike": strike,
                    "expiry": expiry,
                }

    return list(contracts.values())


async def build_catalog(
    host: str = "127.0.0.1",
    port: int = 4001,
    client_id: int = 61,
    prices_csv: str = "arb_bot/config/forecastex_prices_20260212.csv",
    output_path: str = "arb_bot/config/forecastex_conid_catalog.json",
    batch_size: int = 40,
    pace_seconds: float = 0.5,
) -> int:
    """Build conId catalog by qualifying contracts from prices CSV."""
    print("=" * 60)
    print("FAST conId CATALOG BUILDER")
    print("=" * 60)

    # Parse contracts
    print("Parsing prices CSV ...")
    contract_params = parse_prices_csv(prices_csv)
    symbols = set(c["symbol"] for c in contract_params)
    print("  %d contracts across %d symbols" % (len(contract_params), len(symbols)))

    # Connect
    ib = ibasync.IB()
    print("\nConnecting to %s:%d ..." % (host, port))
    try:
        await ib.connectAsync(host, port, clientId=client_id, timeout=30)
    except Exception as exc:
        print("ERROR: %s" % exc)
        return 1
    print("Connected. Account: %s" % ib.managedAccounts())
    await asyncio.sleep(1)

    # Build IB Contract objects
    ib_contracts = []
    for cp in contract_params:
        c = ibasync.Contract(
            symbol=cp["symbol"],
            secType="OPT",
            exchange="FORECASTX",
            currency="USD",
            right=cp["right"],
            strike=cp["strike"],
            lastTradeDateOrContractMonth=cp["expiry"],
        )
        ib_contracts.append(c)

    # Qualify in batches
    catalog = []
    total_batches = (len(ib_contracts) + batch_size - 1) // batch_size
    success_count = 0
    fail_count = 0
    timeout_count = 0
    t0 = time.time()

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(ib_contracts))
        batch = ib_contracts[start:end]

        try:
            qualified = await asyncio.wait_for(
                ib.qualifyContractsAsync(*batch), timeout=30
            )
            batch_ok = 0
            for c in qualified:
                if c.conId > 0:
                    batch_ok += 1
                    catalog.append({
                        "conid": c.conId,
                        "symbol": c.symbol,
                        "right": c.right,
                        "strike": c.strike,
                        "expiry": c.lastTradeDateOrContractMonth,
                        "trading_class": c.tradingClass,
                        "multiplier": c.multiplier or "1",
                        "long_name": "",
                    })
                else:
                    fail_count += 1
            success_count += batch_ok

            elapsed = time.time() - t0
            rate = success_count / elapsed if elapsed > 0 else 0
            sys.stdout.write(
                "\r  Batch %d/%d | qualified: %d | failed: %d | rate: %.0f/s    "
                % (batch_idx + 1, total_batches, success_count, fail_count, rate)
            )
            sys.stdout.flush()

        except asyncio.TimeoutError:
            timeout_count += 1
            sys.stdout.write(
                "\r  Batch %d/%d | TIMEOUT (count: %d)    "
                % (batch_idx + 1, total_batches, timeout_count)
            )
            sys.stdout.flush()
            if timeout_count >= 5:
                print("\n\n  Too many timeouts — stopping.")
                break
            await asyncio.sleep(3)  # extra backoff
        except Exception as exc:
            sys.stdout.write(
                "\r  Batch %d/%d | ERROR: %s    "
                % (batch_idx + 1, total_batches, exc)
            )
            sys.stdout.flush()

        await asyncio.sleep(pace_seconds)

    ib.disconnect()
    elapsed = time.time() - t0

    # Dedupe by conId (shouldn't be needed, but safety)
    seen = set()
    deduped = []
    for entry in catalog:
        if entry["conid"] not in seen:
            seen.add(entry["conid"])
            deduped.append(entry)

    # Save
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(deduped, f, indent=2)

    print("\n\n" + "=" * 60)
    print("CATALOG COMPLETE")
    print("=" * 60)
    print("  Contracts qualified: %d" % success_count)
    print("  Contracts failed:    %d" % fail_count)
    print("  Timeouts:            %d" % timeout_count)
    print("  Unique conIds saved: %d" % len(deduped))
    print("  Unique symbols:      %d" % len(set(e["symbol"] for e in deduped)))
    print("  Time elapsed:        %.1fs" % elapsed)
    print("  Output:              %s" % out)

    return 0


def main():
    parser = argparse.ArgumentParser(description="Build ForecastEx conId catalog (fast)")
    parser.add_argument("--tws-host", default="127.0.0.1")
    parser.add_argument("--tws-port", type=int, default=4001)
    parser.add_argument("--client-id", type=int, default=61)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--pace", type=float, default=0.5)
    parser.add_argument(
        "--prices-csv",
        default="arb_bot/config/forecastex_prices_20260212.csv",
    )
    parser.add_argument(
        "--output",
        default="arb_bot/config/forecastex_conid_catalog.json",
    )
    args = parser.parse_args()

    return asyncio.run(build_catalog(
        host=args.tws_host,
        port=args.tws_port,
        client_id=args.client_id,
        prices_csv=args.prices_csv,
        output_path=args.output,
        batch_size=args.batch_size,
        pace_seconds=args.pace,
    ))


if __name__ == "__main__":
    raise SystemExit(main())
