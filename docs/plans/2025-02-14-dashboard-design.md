# Arb Bot Dashboard — Design Document

**Date**: 2025-02-14
**Status**: Approved

## Overview

A local web dashboard for monitoring and tuning the arb bot in real time. Shows live PnL, open positions, trade feed, historical analytics, and allows hot-reload parameter tuning — all in a separate process that cannot crash the bot.

## Architecture

**Sidecar pattern**: The dashboard runs as a separate process. Communication with the bot happens over a TCP control socket. Historical data is read directly from the bot's existing SQLite databases.

```
Bot Process                          Dashboard Process
┌──────────────┐   TCP :9120         ┌──────────────────┐
│ ArbEngine    │◄── state push ────► │ FastAPI Server   │
│              │◄── config cmds ───► │                  │
│ ControlSocket│                     │ bot_bridge.py    │
└──────┬───────┘                     │ db_reader.py     │
       │                             │     ▲            │
       │ write                       │     │ WebSocket  │
       ▼                             │     ▼            │
┌──────────────┐    read-only        │ React SPA        │
│ SQLite DBs   │◄────────────────────│ (static files)   │
│ orders.db    │                     └──────────────────┘
│ analytics.db │
└──────────────┘
```

Three data channels:

1. **TCP control socket** (new, ~150 lines in bot) — Bot pushes JSON state snapshot every cycle. Dashboard sends config update commands. Newline-delimited JSON protocol.
2. **SQLite databases** (existing) — Dashboard reads `arb_orders.db` and `analytics.db` directly. Historical trades, positions, daily summaries. Read-only.
3. **WebSocket** (dashboard to browser) — FastAPI bridges control socket data to the React frontend.

## Pages

### 1. Dashboard (home) — Live Trading Terminal

Primary view, split into three zones:

- **Header bar**: Mode badge (LIVE/PAPER), venue balances, locked capital, cycle count, uptime, daily PnL
- **Top half**: PnL line chart (cumulative daily, per-trade dots) alongside open positions table (market, side, contracts, entry, expected profit, time remaining)
- **Bottom half**: Scrolling trade feed (newest on top) showing fills, skips with reasons, and opportunity detection counts

### 2. Analytics — Performance

Historical view from SQLite:

- **Equity curve**: Cumulative PnL over time with high water mark, max drawdown, Sharpe ratio, win rate, avg trade size
- **Daily PnL**: Bar chart by day
- **PnL by lane**: Pie/bar showing bucket vs parity vs cross attribution (toggleable: dollars, trades, percentage)
- **PnL by bucket group**: Sortable table with group_id, trade count, PnL, win rate, avg profit. Filterable by lane.

### 3. Config — Parameter Tuning

Collapsible category sections, each parameter shows current value with an input field and Apply button. Changes hot-reload immediately via control socket.

Categories:
- **Safety Rails**: daily loss cap, consecutive failures, canary mode, kill switch
- **Sizing**: bankroll, max per trade, Kelly settings
- **Lanes**: enable/disable per lane, per-lane thresholds (edge, fill prob, min profit)
- **Bucket Quality**: explore fraction, min score, max legs, failure limit
- **Cooldowns**: market cooldown, opportunity cooldown
- **Exchange**: scan settings (read-only for credentials)

### 4. System — Health

Operational monitoring:
- Stream status per venue (connected/degraded/poll-only)
- Quote coverage (markets scanned vs total)
- Cycle timing (avg duration, timeouts)
- Metrics snapshot (from MetricsRegistry)
- Recent errors/warnings
- Kill switch button (big red emergency stop)

## Control Socket Protocol

TCP on `localhost:9120` (configurable via `ARB_CONTROL_SOCKET_PORT`). Newline-delimited JSON.

### Bot pushes state every cycle:

```json
{
  "type": "state",
  "ts": 1739494823.4,
  "cycle": 347,
  "mode": "live",
  "uptime_seconds": 8040,
  "cash_by_venue": {"kalshi": 94.20},
  "locked_capital_by_venue": {"kalshi": 5.80},
  "open_positions": [...],
  "last_cycle": {
    "duration_ms": 3200,
    "quotes_count": 8749,
    "opportunities_found": 1,
    "near_opportunities": 2,
    "decisions": [...]
  },
  "stream_status": {"kalshi": "connected", "polymarket": "poll_only"},
  "daily_pnl": 3.42,
  "daily_trades": 12,
  "daily_loss_cap_remaining": 6.58,
  "consecutive_failures": 0,
  "config_snapshot": {"ARB_DAILY_LOSS_CAP_USD": 10.0, ...}
}
```

### Dashboard sends config update:

```json
{"type": "update_config", "key": "ARB_DAILY_LOSS_CAP_USD", "value": 15.0}
```

### Bot responds:

```json
{"type": "update_config_ack", "key": "ARB_DAILY_LOSS_CAP_USD", "old": 10.0, "new": 15.0, "ok": true}
```

### Kill switch:

```json
{"type": "kill_switch", "activate": true}
```

### Not hot-reloadable (returns error):

Exchange credentials, structural rules path, database paths, socket port. These require a bot restart.

## Hot-Reload Mechanism

1. Engine starts TCP server on port 9120 as a background asyncio task
2. Per-cycle: engine serializes state, writes to connected clients
3. Config update received: validates key exists, validates type, calls `engine.update_setting(key, value)` using `dataclasses.replace()` on frozen settings, returns ack
4. Kill switch: writes `.kill_switch` file, existing KillSwitchManager handles it

## Tech Stack

### Bot side (minimal)
- `arb_bot/control_socket.py` — ~150 lines, stdlib only (asyncio, json). Zero new dependencies.
- `arb_bot/engine.py` — small modification: start control socket, add `update_setting()` method

### Dashboard server
- FastAPI + Uvicorn (async API server)
- aiosqlite (async SQLite reads)
- WebSocket (FastAPI built-in, bridges to browser)

### Frontend
- React 18 + TypeScript
- Tailwind CSS
- Recharts (charts)
- Vite (build tool)
- React Router (4-page nav)

## Project Structure

```
arb_bot/
  control_socket.py          # NEW: TCP server (~150 lines)
  engine.py                  # MODIFIED: start socket, update_setting()

dashboard/
  server/
    main.py                  # FastAPI app + Uvicorn
    bot_bridge.py            # TCP client to bot control socket
    db_reader.py             # Read-only SQLite queries
    routes/
      state.py               # GET /api/state, WebSocket /ws
      analytics.py           # GET /api/analytics/*
      config.py              # GET/POST /api/config
      system.py              # GET /api/system, POST /api/kill-switch
    requirements.txt         # fastapi, uvicorn, aiosqlite

  frontend/
    src/
      pages/
        Dashboard.tsx        # PnL chart, positions, trade feed
        Analytics.tsx        # Equity curve, attribution, bucket table
        Config.tsx           # Parameter editor
        System.tsx           # Health, kill switch
      components/
        PnlChart.tsx
        EquityCurve.tsx
        PositionsTable.tsx
        TradeFeed.tsx
        ConfigSection.tsx
        StatusBadge.tsx
        KillSwitch.tsx
      hooks/
        useWebSocket.ts      # Real-time state
        useApi.ts            # REST queries
      lib/
        types.ts             # TS types matching bot JSON
    package.json
    tailwind.config.js
    vite.config.ts
```

## Running

```bash
# Terminal 1: Bot (already running)
python3 -m arb_bot.main --live --stream

# Terminal 2: Dashboard
cd dashboard/server && uvicorn main:app --port 8080

# Browser
open http://localhost:8080
```

## Data Sources Summary

| Source | Type | Used For |
|--------|------|----------|
| Control socket state push | Real-time JSON | Dashboard page: PnL, positions, trade feed, cycle stats |
| analytics.db decisions table | SQLite | Analytics page: daily summaries, lane attribution |
| analytics.db fills table | SQLite | Analytics page: equity curve, per-bucket breakdown |
| arb_orders.db order_intents | SQLite | Analytics page: trade history details |
| arb_orders.db positions | SQLite | Dashboard page: open positions (fallback if socket disconnected) |
| MetricsRegistry snapshot | Via control socket | System page: counters, gauges, alerts |
| Config snapshot | Via control socket | Config page: current values for all params |
