# Arb Bot — System Architecture

## High-Level Overview

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                          PREDICTION MARKET EXCHANGES                        ║
║                                                                              ║
║   ┌─────────────┐    ┌──────────────┐    ┌────────────────────┐            ║
║   │   Kalshi     │    │  Polymarket   │    │  ForecastEx / IBKR │            ║
║   │  (REST+WS)   │    │  (REST only)  │    │  (TWS API / REST)  │            ║
║   └──────┬───────┘    └──────┬────────┘    └────────┬──────────┘            ║
╚══════════╪═══════════════════╪══════════════════════╪════════════════════════╝
           │                   │                      │
           ▼                   ▼                      ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                         EXCHANGE ADAPTER LAYER                              ║
║                                                                              ║
║   ┌─────────────────┐  ┌────────────────┐  ┌──────────────────────┐        ║
║   │  KalshiAdapter   │  │ PolymarketAdpt │  │  ForecastExAdapter   │        ║
║   │  2779 LOC        │  │ 1368 LOC       │  │  1131 LOC            │        ║
║   │                   │  │                │  │                      │        ║
║   │ • RSA auth        │  │ • Gamma API    │  │ • conId catalog      │        ║
║   │ • WebSocket sub   │  │ • CLOB orders  │  │ • Options model      │        ║
║   │ • Priority tickers│  │                │  │ • Buy-only (no sell) │        ║
║   │ • Rate throttle   │  │                │  │ • 3-tier discovery   │        ║
║   └────────┬──────────┘  └───────┬────────┘  └──────────┬───────────┘        ║
║            │ ExchangeAdapter ABC │ (base.py)            │                   ║
║            │  fetch_quotes()     │ place_order()        │                   ║
║            │  cancel_order()     │ stream_quotes()      │                   ║
╚════════════╪═════════════════════╪══════════════════════╪════════════════════╝
             │                     │                      │
             └─────────────────────┼──────────────────────┘
                                   ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                        DATA QUALITY PIPELINE                                ║
║                                                                              ║
║   ┌──────────────┐  ┌────────────────┐  ┌─────────────┐  ┌────────────┐   ║
║   │ Stream Health │  │ Reconciliation │  │   Quote     │  │   Dedupe   │   ║
║   │              │  │                │  │  Firewall   │  │            │   ║
║   │ • Bounded    │  │ • Stream vs    │  │             │  │ • Stable   │   ║
║   │   queue      │  │   REST delta   │  │ • Anomaly   │  │   opp IDs  │   ║
║   │ • Backpress. │  │ • Checksums    │  │   detection │  │ • Idempot. │   ║
║   │ • Gap detect │  │ • Force resync │  │ • Staleness │  │   keys     │   ║
║   └──────────────┘  └────────────────┘  └─────────────┘  └────────────┘   ║
╚══════════════════════════════════════════════════════════════════════════════╝
                                   │
                          BinaryQuote objects
                                   │
                                   ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                  ARBITRAGE DETECTION — 5 LANES (strategy.py)                ║
║                                                                              ║
║  ┌───────────────────────────────────────────────────────────────────────┐  ║
║  │                    ArbitrageFinder.find_opportunities()                │  ║
║  └───────────────────────────────────────────────────────────────────────┘  ║
║                                                                              ║
║  ┌─────────────┐ ┌─────────────┐ ┌──────────────┐ ┌────────┐ ┌────────┐  ║
║  │  Lane 1     │ │  Lane 2     │ │  Lane 3      │ │Lane 4  │ │Lane 5  │  ║
║  │  INTRA      │ │  CROSS      │ │  STRUCTURAL  │ │STRUCT  │ │STRUCT  │  ║
║  │  VENUE      │ │  VENUE      │ │  BUCKET      │ │PARITY  │ │EVENT   │  ║
║  │             │ │             │ │              │ │        │ │TREE    │  ║
║  │ YES+NO pair │ │ K↔P, K↔FX  │ │ Mutex groups │ │ A+B=1  │ │Parent/ │  ║
║  │ same market │ │ P↔FX pairs  │ │ (elections,  │ │ comple-│ │child   │  ║
║  │ same venue  │ │ from CSV    │ │  Fed rates,  │ │ ment   │ │hierar- │  ║
║  │             │ │ mappings    │ │  weather)    │ │ pairs  │ │chies   │  ║
║  │ Disabled in │ │             │ │              │ │        │ │        │  ║
║  │ v11 (too    │ │ 743 K↔FX   │ │ Thompson     │ │        │ │        │  ║
║  │ low edge)   │ │ mappings    │ │ sampling     │ │        │ │        │  ║
║  └─────────────┘ └─────────────┘ └──────────────┘ └────────┘ └────────┘  ║
║                                                                              ║
║  Supporting modules:                                                        ║
║  • cross_mapping.py — CSV-based 3-venue mapping loader                     ║
║  • structural_rules.py — bucket/parity/tree rule definitions               ║
║  • bucket_quality.py — Thompson sampling bandit for bucket selection        ║
║  • event_tree.py — parent/child hierarchy logic                            ║
║  • dependency_discovery.py — auto-detect market relationships              ║
╚══════════════════════════════════════════════════════════════════════════════╝
                                   │
                     ArbitrageOpportunity objects
                                   │
                                   ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                     SIZING PIPELINE (12-step, execution-aware)              ║
║                                                                              ║
║   ┌──────────────────────────────────────────────────────────────────┐      ║
║   │                 PositionSizer.build_trade_plan()                  │      ║
║   └──────────────────────────────────────────────────────────────────┘      ║
║                                                                              ║
║   Step 1:  execution_model.py ─── Queue decay, market impact, partials     ║
║   Step 2:  fill_model.py ──────── Fill probability (same-venue correl.)    ║
║   Step 3:  fee_model.py ──────── Per-venue fee schedule (maker/taker)      ║
║   Step 4:  kelly_sizing.py ───── Tail-risk Kelly (Baker-McHale shrinkage)  ║
║   Step 5:  robust_inputs.py ──── IQR fencing, MAD z-scores, winsorize     ║
║   Step 6:  liquidity_impact.py ─ Power-law slippage curve                  ║
║   Step 7:  capital_lock.py ───── Time-to-resolution penalty                ║
║   Step 8:  monte_carlo_sim.py ── PnL distribution, legging risk            ║
║   Step 9:  lane_allocator.py ─── Sharpe-weighted bankroll per lane         ║
║   Step 10: loss_cap.py ──────── Confidence-scaled loss budget              ║
║   Step 11: exposure_manager.py ─ Venue/category/market concentration       ║
║   Step 12: drawdown_manager.py ─ Tiered de-risking (safe/caution/halt)     ║
║                                                                              ║
║   ┌────────────────────── RUST HOT PATH ──────────────────────────┐        ║
║   │  arb_engine_rs (PyO3):                                        │        ║
║   │  • binary_math.rs (401 LOC) — quote decomposition             │        ║
║   │  • strategy.rs (1260 LOC) — all 5 detection lanes             │        ║
║   │  • eval_pipeline.rs (1013 LOC) — sizing + Kelly + fees        │        ║
║   │  Dispatch: env vars (ARB_USE_RUST_ALL=1), p99 < 2ms SLO      │        ║
║   └───────────────────────────────────────────────────────────────┘        ║
╚══════════════════════════════════════════════════════════════════════════════╝
                                   │
                          TradePlan objects
                                   │
                                   ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                        RISK GATES (Pre-Execution)                           ║
║                                                                              ║
║   ┌──────────────┐  ┌────────────────┐  ┌──────────────┐  ┌────────────┐  ║
║   │ Kill Switch   │  │ Circuit Breaker│  │   Cooldowns  │  │  Exposure  │  ║
║   │              │  │                │  │              │  │  Limits    │  ║
║   │ • Global halt │  │ • Per-endpoint │  │ • Market     │  │            │  ║
║   │ • Daily loss  │  │   state machine│  │   cooldown   │  │ • Per-venue│  ║
║   │   cap ($10)   │  │ • CLOSED→OPEN  │  │ • Opportunity│  │   USD cap  │  ║
║   │ • Consec fail │  │   →HALF_OPEN   │  │   cooldown   │  │ • Open mkt │  ║
║   │ • Canary mode │  │ • Jittered     │  │ • Side scope │  │   count    │  ║
║   │              │  │   recovery     │  │              │  │ • Cluster  │  ║
║   └──────────────┘  └────────────────┘  └──────────────┘  └────────────┘  ║
╚══════════════════════════════════════════════════════════════════════════════╝
                                   │
                              Pass/Reject
                                   │
                                   ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                       EXECUTION ENGINE                                      ║
║                                                                              ║
║   engine.py._execute_live_plan()                                            ║
║                                                                              ║
║   For each leg (SEQUENTIAL — not parallel):                                 ║
║                                                                              ║
║   ┌─────────┐    ┌───────────┐    ┌──────────┐    ┌──────────────────┐    ║
║   │ Persist │───▶│  Place    │───▶│  Poll    │───▶│  Record fill /  │    ║
║   │ intent  │    │  order    │    │  status  │    │  cancel+unwind  │    ║
║   │ (SQLite)│    │  on venue │    │  (10s)   │    │  if timeout     │    ║
║   └─────────┘    └───────────┘    └──────────┘    └──────────────────┘    ║
║                                                                              ║
║   Between legs: re-quote next leg → abort if book moved too far            ║
║                                                                              ║
║   ┌─────────────────────────────────────────────────────────────────────┐  ║
║   │                    CRASH RECOVERY                                    │  ║
║   │  order_store.py (SQLite) ← crash_recovery.py (startup reconcile)   │  ║
║   │  Unhedged positions → SAFE MODE (no new trades until ack)           │  ║
║   └─────────────────────────────────────────────────────────────────────┘  ║
╚══════════════════════════════════════════════════════════════════════════════╝
                                   │
                                   ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                     POST-EXECUTION & MONITORING                             ║
║                                                                              ║
║   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ ║
║   │   Balance    │  │  Analytics   │  │   Drift      │  │  Auto-Tuner  │ ║
║   │   Refresh    │  │   Store      │  │  Monitor     │  │              │ ║
║   │              │  │              │  │              │  │ • KPI-driven │ ║
║   │ • 60s sync   │  │ • SQLite/    │  │ • Expected   │  │   threshold  │ ║
║   │ • Reconcile  │  │   DuckDB     │  │   vs actual  │  │   adjustment │ ║
║   │   vs exchange│  │ • Daily PnL  │  │ • Per-lane   │  │ • Safety     │ ║
║   │              │  │ • Fill quality│  │   tracking   │  │   bounds     │ ║
║   └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘ ║
║                                                                              ║
║   ┌────────────────────────────────────────────────────────────────────┐   ║
║   │                     DASHBOARD (React + Python)                      │   ║
║   │  Real-time: positions, trades, lane metrics, circuit breaker state │   ║
║   │  Historical: PnL charts, fill quality, coverage analytics          │   ║
║   │  Controls: kill switch, config editor, system health               │   ║
║   └────────────────────────────────────────────────────────────────────┘   ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## Data Flow Summary

```
Exchange APIs
    │
    ▼
Exchange Adapters (Kalshi, Polymarket, ForecastEx)
    │
    ▼
Data Quality (firewall, reconciliation, dedup)
    │
    ▼
BinaryQuote objects (normalized across all venues)
    │
    ▼
ArbitrageFinder — 5 detection lanes
    │
    ▼
ArbitrageOpportunity objects (with legs, edge, kind)
    │
    ▼
12-step Sizing Pipeline (Kelly + risk + Monte Carlo)
    │
    ▼
TradePlan objects (contracts, capital, expected profit)
    │
    ▼
Risk Gates (kill switch, circuit breakers, exposure)
    │
    ▼
Sequential Leg Execution (SQLite-backed, crash-safe)
    │
    ▼
Post-Execution (balance sync, analytics, auto-tuning)
```

---

## Module Inventory (70+ modules, ~30,000 LOC Python + ~2,700 LOC Rust)

### Core Engine
| Module | LOC | Purpose |
|--------|-----|---------|
| `engine.py` | 3046 | Main loop, cycle evaluation, execution dispatch |
| `models.py` | 246 | BinaryQuote, TradePlan, EngineState, enums |
| `config.py` | 1277 | 150+ env vars → AppSettings dataclass |
| `main.py` | 231 | CLI entry point |
| `paper.py` | 914 | Paper simulation engine |

### Exchange Adapters
| Module | LOC | Purpose |
|--------|-----|---------|
| `exchanges/base.py` | 52 | ExchangeAdapter ABC |
| `exchanges/kalshi.py` | 2779 | Kalshi REST + WebSocket |
| `exchanges/polymarket.py` | 1368 | Polymarket CLOB/Gamma |
| `exchanges/forecastex.py` | 1131 | ForecastEx via IBKR |

### Detection (5 Lanes)
| Module | LOC | Purpose |
|--------|-----|---------|
| `strategy.py` | 906 | ArbitrageFinder (all 5 lanes) |
| `cross_mapping.py` | 196 | Cross-venue mapping loader |
| `cross_mapping_generator.py` | 934 | Auto-generate mappings (Jaccard/embeddings) |
| `structural_rules.py` | 236 | Bucket/parity/tree rule definitions |
| `structural_rule_generator.py` | 1860 | Auto-generate structural rules |
| `bucket_quality.py` | 504 | Thompson sampling bucket selection |
| `event_tree.py` | 458 | Parent/child hierarchy logic |
| `dependency_discovery.py` | 333 | Auto-detect market relationships |
| `binary_math.py` | 311 | Quote math (YES/NO decomposition) |

### Sizing & Models
| Module | LOC | Purpose |
|--------|-----|---------|
| `sizing.py` | 203 | PositionSizer (build_trade_plan) |
| `kelly_sizing.py` | 366 | Tail-risk Kelly (Baker-McHale) |
| `execution_model.py` | 462 | Queue decay, market impact |
| `fill_model.py` | 310 | Fill probability (correlated legs) |
| `fee_model.py` | 502 | Per-venue fee schedules |
| `monte_carlo_sim.py` | 308 | PnL distribution simulation |
| `liquidity_impact.py` | 342 | Power-law slippage curves |
| `capital_lock.py` | 193 | Time-to-resolution penalty |
| `robust_inputs.py` | 285 | Outlier trimming (IQR/MAD) |
| `lane_allocator.py` | 271 | Sharpe-weighted bankroll |
| `loss_cap.py` | 205 | Confidence-scaled loss budgets |
| `exposure_manager.py` | 285 | Concentration limits |
| `drawdown_manager.py` | 221 | Tiered de-risking |

### Risk & Safety
| Module | LOC | Purpose |
|--------|-----|---------|
| `risk.py` | 149 | RiskManager precheck/record_fill |
| `kill_switch.py` | 286 | Global halt, daily loss cap |
| `circuit_breaker.py` | 267 | CLOSED/OPEN/HALF_OPEN FSM |
| `endpoint_breaker.py` | 271 | Per-API-endpoint breakers |
| `mode_degradation.py` | 186 | Stream → poll failover |
| `startup_gate.py` | 221 | Startup health checks |
| `lane_watchdog.py` | 232 | Lane readiness enforcement |
| `drift_monitor.py` | 221 | Expected vs realized tracking |

### Order Management
| Module | LOC | Purpose |
|--------|-----|---------|
| `order_store.py` | 616 | SQLite persistent store |
| `order_poller.py` | 111 | Post-submission status polling |
| `crash_recovery.py` | 151 | Startup reconciliation |
| `balance_refresh.py` | 207 | Periodic cash sync |
| `execution_pipeline.py` | 411 | Stage timing (prebuild→submit→poll) |
| `execution_policy.py` | 346 | Maker-first strategy |
| `latency_slo.py` | 305 | p50/p95/p99 tracking |

### Reliability & Observability
| Module | LOC | Purpose |
|--------|-----|---------|
| `rate_governor.py` | 327 | AIMD + token bucket throttling |
| `poll_scheduler.py` | 373 | Hot/warm/cold tier scheduling |
| `poll_budget.py` | 269 | Per-venue poll budget allocation |
| `coverage_watchdog.py` | 298 | Cross-pair coverage monitoring |
| `keepalive_probes.py` | 282 | Dead market detection |
| `warm_cache.py` | 309 | TTL cache with LRU eviction |
| `analytics_store.py` | 349 | SQLite analytics |
| `metrics.py` | 257 | Prometheus export + alerts |
| `replay_harness.py` | 504 | Deterministic A/B replay |
| `auto_tuner.py` | 474 | KPI-driven threshold adjustment |

### Rust Hot-Path
| Module | LOC | Purpose |
|--------|-----|---------|
| `rust/arb_engine_rs/src/binary_math.rs` | 401 | Quote decomposition |
| `rust/arb_engine_rs/src/strategy.rs` | 1260 | 5-lane detection |
| `rust/arb_engine_rs/src/eval_pipeline.rs` | 1013 | Sizing + Kelly + fees |

---

## Key Design Decisions

1. **Sequential leg execution** — Cross-venue legs execute one at a time (not parallel) to allow abort-on-stale between legs. More conservative but prevents catastrophic legging loss.

2. **SQLite crash recovery** — Every order intent is persisted before placement. On crash, startup reconciles against exchange positions and enters safe mode if unhedged.

3. **Rust hot-path with Python fallback** — Critical compute (detection + sizing) runs in Rust via PyO3 for <2ms p99 latency. Each module has an env var toggle for instant rollback to Python.

4. **Thompson sampling for buckets** — Instead of deterministic bucket scoring, uses Bayesian bandit (Beta posteriors) to balance exploitation of proven buckets with exploration of untested ones.

5. **12-step sizing pipeline** — Each step can independently reject or downsize a trade. This creates a conservative "chain of no" where any concern reduces position size.

6. **Time-regime switching** — Near-term events (≤14d) get edge boosts; far-term events (≥60d) get 25% edge penalties. Prevents capital lockup on ForecastEx's long-dated contracts.

7. **3-venue architecture** — Kalshi (primary, most liquid), ForecastEx/IBKR (economics/commodities), Polymarket (when available). 743 cross-venue mappings enable Kalshi↔ForecastEx arbitrage.
