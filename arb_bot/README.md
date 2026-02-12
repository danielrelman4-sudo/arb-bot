# Prediction-Market Arb Bot (Scaffold)

This bot scans binary markets for arbitrage and supports both same-venue and cross-venue opportunities.

Core behavior:
- Poll Kalshi and/or Polymarket every `N` seconds.
- Build top-of-book quotes with both taker and maker-estimate pricing paths.
- Convert opposite-side bids into executable zero-spread buy prices (`YES = 1 - NO_bid`, `NO = 1 - YES_bid`) and use the best executable path.
- Rank quotes by liquidity/volume/spread/staleness and scan a hotset first (with optional full-universe fallback).
- Detect:
  - Intra-venue arb: `YES + NO < 1`
  - Cross-venue arb: `YES@VenueA + NO@VenueB < 1` (and inverse)
  - Structural arb:
    - Mutually exclusive buckets
    - Event-tree baskets
    - Explicit parity checks
- Apply sizing + risk checks.
- Score fill-adjusted EV and gate opportunities by fill probability / expected realized profit.
- Execute (live) or simulate (dry-run).

## Opportunity Kinds (Detailed)

The bot evaluates opportunities in separate lanes, then merges them into a single allocator ranked by expected realized profit.

Shared pricing rule:
- For any candidate basket, edge is computed from:
  - `gross_edge_per_contract = payout_per_contract - sum(leg_buy_prices)`
  - `net_edge_per_contract = gross_edge_per_contract - fees`
- A candidate is considered only if it passes lane thresholding + fill/EV/risk/sizing gates.

Execution styles:
- `taker`: uses immediate executable prices.
- `maker_estimate`: uses maker-style estimated entry prices and fill model assumptions.

### `intra_venue`
- Definition: same market, same venue, buy `YES` and `NO` together.
- Detection condition: `YES_buy + NO_buy < 1` after fees/thresholds.
- Matching logic: no text matching needed; uses one quote object with both sides.
- Typical strength: frequent but usually low edge per contract.
- Named example: Polymarket market for **"Will Gavin Newsom be the Democratic presidential nominee in 2028?"**
  - If `YES_buy=0.43` and `NO_buy=0.54`, total cost is `0.97`.
  - Since one side must resolve true, payout is `1.00`, so gross edge is `0.03` per contract before fees.

### `cross_venue`
- Definition: semantically equivalent market pair across venues.
- Candidate legs tested:
  - `YES(left) + NO(right)`
  - `NO(left) + YES(right)`
- Pairing sources:
  1. Deterministic mapping (`ARB_CROSS_VENUE_MAPPING_PATH`) when available.
  2. Optional fuzzy text matching fallback (Jaccard token similarity) if enabled.
- Safety note: for production-like runs, set `ARB_CROSS_VENUE_MAPPING_REQUIRED=true` to disable fuzzy-only pairings.
- Named example: Kalshi ticker **`KXPRESNOMD-28-GN`** mapped to the Polymarket Gavin Newsom 2028 nominee market.
  - If `YES(Kalshi)=0.44` and `NO(Polymarket)=0.53`, then `YES+NO=0.97`.
  - Cross-venue basket edge is `1.00 - 0.97 = 0.03` per contract before fees/slippage.

### `structural_bucket`
- Source: `mutually_exclusive_buckets` in structural rules JSON.
- Rule meaning: legs represent outcomes that cannot occur simultaneously.
- Detection: buy the specified leg basket; payout defaults to `1.0` unless overridden by `payout_per_contract`.
- Candidate passes when basket net edge is positive after fees/thresholds.
- Named example: a **2028 Democratic nominee** bucket with mutually exclusive candidates
  (`Gavin Newsom`, `Kamala Harris`, `J.B. Pritzker`, `Josh Shapiro`).
  - Buy `YES` on each candidate leg in the bucket.
  - If prices sum to `0.96`, payout remains `1.00` (exactly one candidate can win), so edge is `0.04`.

### `structural_event_tree`
- Source: `event_trees` in structural rules JSON.
- Rule meaning: parent market and child markets encode hierarchical event logic.
- Two basket constructions are tested per rule:
  - `parent_no_children_yes` with payout `1.0`
  - `parent_yes_children_no` with payout `len(children)`
- Event-tree opportunities appear only when explicit parent/child links exist in rule data.
- Named example: parent market **"Democrat wins State X Senate race"** with child buckets
  **"wins by 0-2"**, **"wins by 3-5"**, **"wins by 6+"**.
  - Basket 1: `NO(parent)+YES(children...)` targets missed mass in child partition (payout `1.0`).
  - Basket 2: `YES(parent)+NO(children...)` targets overpricing in child partition (payout `len(children)`).
  - The bot computes both baskets and keeps only those with positive net edge after fees.

### `structural_parity`
- Source: `cross_market_parity_checks` in structural rules JSON.
- Rule meaning: two markets are tied by parity relationship.
- Supported relationships:
  - `equivalent`:
    - `left_yes + right_no`
    - `left_no + right_yes`
  - `complement`:
    - `left_yes + right_yes`
    - `left_no + right_no`
- Use this lane for explicit cross-market parity constraints generated from mapping or curated rules.
- Named example (`equivalent`): Kalshi and Polymarket versions of
  **"Will Gavin Newsom be the Democratic nominee in 2028?"**
  - Basket A: `YES(Kalshi)+NO(Polymarket)`; if `0.46 + 0.50 = 0.96`, edge is `0.04`.
  - Basket B: `NO(Kalshi)+YES(Polymarket)`; if `0.55 + 0.43 = 0.98`, edge is `0.02`.
  - Either basket can be tradable depending on which side is mispriced at that moment.

## What was added for sensitivity

1. Better market universe selection:
- Paged scanning.
- Liquidity floor.
- Excluded low-signal Kalshi prefixes.
- Optional include-list by event ticker.

2. Explicit cross-venue mapping:
- CSV mapping file support for deterministic market pairing.
- Optional fuzzy fallback.
- Strict mode support: mapped-only cross-venue matching (`ARB_CROSS_VENUE_MAPPING_REQUIRED=true`).

3. Discovery mode:
- Separate low-threshold settings for exploratory runs.
- Near-arb detection based on `ARB_NEAR_ARB_TOTAL_COST_THRESHOLD`.

4. Maker-estimate pricing:
- Opportunities evaluated for taker and maker-estimate execution styles.

5. Binary microstructure math:
- Quote decomposition tracked in metadata:
  - `p = (YES + (1 - NO)) / 2`
  - `edge = (YES + NO - 1) / 2`
- Midpoint-consistency and spread-asymmetry diagnostics logged per quote.

6. Stream-first detection:
- Optional WebSocket stream mode:
  - Polymarket (public stream).
  - Kalshi ticker stream (requires Kalshi API key + private key for websocket auth).
  - Polling refresh fallback remains enabled.

7. Fill-adjusted reporting:
- Decision rows include:
  - detected edge/profit
  - fill probability + partial-fill probability
  - expected realized edge/profit
  - realized edge/profit (when known)
  - execution latency and partial-fill flag

8. Constraint-aware residual edge:
- Opportunity payout is re-priced against explicit structural constraints (buckets/event trees/parity).
- Only residual edge that survives correlated-event constraints is eligible for sizing.

9. Portfolio-aware sizing:
- Modified execution-aware Kelly sizing uses fill probability.
- Cluster-level budgets and open-position caps prevent over-concentration in correlated markets.
- EV is re-estimated after final size selection to avoid overstating fill-adjusted profitability.

## Important caveats

- Two-leg execution can leave unhedged risk if one leg fills and the other does not.
- Maker-estimate opportunities model potential fills; they are not guaranteed fills.
- Cross-venue matching can be wrong without explicit mapping.
- This scaffold does not track settlement lifecycle, realized PnL, or taxes.

## Known pitfalls

### 1) Fuzzy cross-venue mismatch risk
- If `ARB_ENABLE_FUZZY_CROSS_VENUE_FALLBACK=true` and mapping is not required, text-similar but economically different markets can be paired.
- For production-like validation, prefer:
  - `ARB_CROSS_VENUE_MAPPING_PATH=<strict csv>`
  - `ARB_CROSS_VENUE_MAPPING_REQUIRED=true`

### 2) Scaling is not automatic with larger bankroll
- Increasing bankroll does not guarantee proportional PnL growth.
- Common bottlenecks are:
  - per-venue exposure caps (`ARB_MAX_EXPOSURE_PER_VENUE_USD`)
  - per-trade caps (`ARB_MAX_DOLLARS_PER_TRADE`, `ARB_MAX_CONTRACTS_PER_TRADE`)
  - liquidity/depth caps (`ARB_MAX_LIQUIDITY_FRACTION_PER_TRADE`, venue quote depth assumptions)
  - cluster/open-position constraints
- Expect sublinear scaling as size increases due to slippage, queue effects, and partial fills.

### 3) Paper-mode optimism vs live trading
- Paper strict mode still settles positions from modeled expected realized profit after hold time; it is not true venue settlement.
- Fill probabilities are model-derived and can overstate live fill quality during competition or latency spikes.
- Treat short paper runs as directional signal only; validate with longer runs and conservative haircuts.

## Layout

- `arb_bot/main.py`: CLI entrypoint.
- `arb_bot/engine.py`: polling/stream loop, orchestration, and cycle reports.
- `arb_bot/strategy.py`: intra + cross-venue detection, mapping support.
- `arb_bot/structural_rules.py`: structural rule parser.
- `arb_bot/sizing.py`: multi-leg sizing.
- `arb_bot/risk.py`: per-venue exposure/open-market/cooldown checks.
- `arb_bot/fill_model.py`: fill-probability and fill-adjusted EV estimation.
- `arb_bot/universe_ranking.py`: quote ranking and hotset ordering.
- `arb_bot/paper.py`: paper session runner + CSV export.
- `arb_bot/cross_mapping.py`: mapping CSV loader.
- `arb_bot/exchanges/kalshi.py`: Kalshi quote + execution adapter.
- `arb_bot/exchanges/polymarket.py`: Polymarket quote + execution adapter + stream support.
- `arb_bot/config/cross_venue_map.example.csv`: mapping template.
- `arb_bot/config/structural_rules.example.json`: structural-arb rule template.

## Setup

1. Install dependencies:

```bash
pip install -r arb_bot/requirements.txt
```

2. Create env file:

```bash
cp arb_bot/.env.example .env
```

3. Fill credentials and tune limits.

## Run

Single dry-run cycle:

```bash
python -m arb_bot.main --once
```

Continuous dry-run:

```bash
python -m arb_bot.main
```

Stream-first dry-run:

```bash
python -m arb_bot.main --stream
```

Note: `--paper-minutes ... --stream` now runs stream-backed paper sessions (with polling refresh fallback on stream timeouts).
Cycle logs now include per-kind detection counts: `intra_venue`, `cross_venue`, `structural_bucket`, `structural_event_tree`, `structural_parity`.

Streaming credentials:
- Polymarket market-data stream: no auth fields required for quote ingestion in this bot (`POLYMARKET_ENABLE_STREAM=true`).
- Kalshi stream: requires API auth headers, so set `KALSHI_KEY_ID` and either `KALSHI_PRIVATE_KEY_PATH` or `KALSHI_PRIVATE_KEY_PEM`.

## Phase-0 Promotion Workflow

Before tuning or promoting run length, freeze and verify a baseline, then evaluate run gates.

1) Create baseline lock (snapshot of `.env` fingerprint + mapping/rules/profile file digests):

```bash
python -m arb_bot.baseline_lock create \
  --env-file .env \
  --output arb_bot/config/phase0_baseline.lock.json
```

2) Verify baseline lock before each test run:

```bash
python -m arb_bot.baseline_lock verify \
  --env-file .env \
  --lock arb_bot/config/phase0_baseline.lock.json
```

3) Run paper session (example):

```bash
TS=$(date +%Y%m%d_%H%M%S)
OUT="arb_bot/output/paper_phase0_${TS}.csv"
LOG="arb_bot/output/paper_phase0_${TS}.log"
LOG_LEVEL=INFO python -m arb_bot.main --stream --paper-minutes 60 --paper-output "$OUT" 2>&1 | tee "$LOG"
```

4) Evaluate Phase-0 pass/fail gates:

```bash
python -m arb_bot.phase0_gate --csv "$OUT" --log "$LOG"
```

Optional JSON report:

```bash
python -m arb_bot.phase0_gate --csv "$OUT" --log "$LOG" --json-output arb_bot/output/phase0_gate_${TS}.json
```

## Lane architecture

Detection now runs in parallel opportunity lanes (`intra_venue`, `cross_venue`, `structural_bucket`,
`structural_event_tree`, `structural_parity`) and then merges into one global allocator that prioritizes
expected realized profit.

Per-lane tuning env vars:
- `ARB_LANE_<LANE>_ENABLED`
- `ARB_LANE_<LANE>_MIN_NET_EDGE_PER_CONTRACT`
- `ARB_LANE_<LANE>_MIN_EXPECTED_PROFIT_USD`
- `ARB_LANE_<LANE>_MIN_FILL_PROBABILITY`
- `ARB_LANE_<LANE>_MIN_REALIZED_PROFIT_USD`

Where `<LANE>` is one of: `INTRA`, `CROSS`, `STRUCTURAL_BUCKET`, `STRUCTURAL_EVENT_TREE`, `STRUCTURAL_PARITY`.

Discovery mode:

```bash
python -m arb_bot.main --discovery
```

Live mode:

```bash
python -m arb_bot.main --live
```

Paper session with CSV output:

```bash
python -m arb_bot.main --paper-minutes 30 --paper-output arb_bot/output/paper_30m.csv
```

Run with a profile (example `v9`, strict mapping + parity-enabled structural rules):

```bash
set -a && source arb_bot/config/profiles/v9_structural_parity_expanded.env && set +a
ARB_DEFAULT_BANKROLL_USD=1000 ARB_BANKROLL_BY_VENUE=kalshi=500,polymarket=500 \
  python -m arb_bot.main --stream --paper-minutes 10 --paper-output arb_bot/output/paper_lane_profile_v9_10m.csv
```

## Mapping file format

`ARB_CROSS_VENUE_MAPPING_PATH` points to a CSV with one of these key styles:
- `group_id,kalshi_market_id,polymarket_market_id`
- `group_id,kalshi_event_ticker,polymarket_slug`

If `ARB_CROSS_VENUE_MAPPING_REQUIRED=true`, only mapped cross-venue pairs are considered.

Generate a mapping automatically from full open-event books:

```bash
python -m arb_bot.cross_mapping_generator \
  --kalshi-all-events \
  --kalshi-max-pages 20 \
  --kalshi-page-size 200 \
  --polymarket-all-events \
  --polymarket-max-pages 20 \
  --polymarket-page-size 100 \
  --min-match-score 0.62 \
  --output arb_bot/config/cross_venue_map.generated.csv \
  --diagnostics-output arb_bot/output/cross_mapping_diagnostics.json
```

## Structural rule format

`ARB_STRUCTURAL_RULES_PATH` points to a JSON file with:
- `mutually_exclusive_buckets`
- `event_trees`
- `cross_market_parity_checks`

See `arb_bot/config/structural_rules.example.json` for a complete example.

Generate structural rules from specific event selectors:

```bash
python -m arb_bot.structural_rule_generator \
  --kalshi-event-tickers KXPRESNOMD-28,SENATENE-26 \
  --polymarket-event-slugs democratic-nominee-2028,nebraska-senate-2026 \
  --cross-mapping-path arb_bot/config/cross_venue_map.strict.csv \
  --existing-rules arb_bot/config/structural_rules.generated.json \
  --diagnostics-output arb_bot/output/structural_generation_diagnostics.json \
  --output arb_bot/config/structural_rules.generated.json
```

Generate a full ruleset automatically by crawling open events on both venues:

```bash
python -m arb_bot.structural_rule_generator \
  --kalshi-all-events \
  --kalshi-max-pages 20 \
  --kalshi-page-size 200 \
  --polymarket-all-events \
  --polymarket-max-pages 20 \
  --polymarket-page-size 100 \
  --cross-mapping-path arb_bot/config/cross_venue_map.strict.csv \
  --existing-rules arb_bot/config/structural_rules.generated.json \
  --diagnostics-output arb_bot/output/structural_generation_diagnostics.json \
  --output arb_bot/config/structural_rules.generated.json
```

Generate structural rules by crawling all active/open Polymarket events and parsing nested
`events[].markets` directly:

```bash
python -m arb_bot.structural_rule_generator \
  --kalshi-event-tickers KXPRESNOMD-28,SENATENE-26 \
  --polymarket-all-events \
  --polymarket-max-pages 20 \
  --polymarket-page-size 100 \
  --cross-mapping-path arb_bot/config/cross_venue_map.strict.csv \
  --existing-rules arb_bot/config/structural_rules.generated.json \
  --diagnostics-output arb_bot/output/structural_generation_diagnostics.json \
  --output arb_bot/config/structural_rules.generated.json
```

The generator always emits `mutually_exclusive_buckets`. It emits `event_trees` only when the
source market metadata includes explicit parent/child links (for example `parent_market_id` /
`child_market_ids` style fields). `cross_market_parity_checks` can be generated from
`--cross-mapping-path`, and when `--existing-rules` is provided all rule families are merged by
`group_id` (generated entries override duplicates).

## Paper CSV columns

Each row records a decision for one opportunity:
- `action`: `dry_run`, `settled`, `skipped`, `filled`, `execution_failed`
- `reason`: skip/failure reason
- `kind`: `intra_venue`, `cross_venue`, `structural_bucket`, `structural_event_tree`, or `structural_parity`
- `execution_style`: `taker` or `maker_estimate`
- detected vs expected-realized vs realized edge/profit
- fill probability, partial-fill probability, and slippage estimate
- `contracts`, `capital_required`, `expected_profit`, `simulated_pnl`
- correlation/constraint sizing telemetry:
  - `correlation_cluster`
  - `constraint_min_payout`, `constraint_adjusted_payout`
  - `constraint_valid_assignments`, `constraint_considered_markets`, `constraint_assumptions`
  - `cluster_budget_usd`, `cluster_used_usd`, `cluster_remaining_usd`, `cluster_exposure_ratio`
  - `kelly_fraction_raw`, `kelly_fraction_effective`
- both leg venue/market/side/price/size

With strict paper simulation, `simulated_pnl` is recognized at `settled` rows (entry rows are zero).

Strict paper settlement timing can be configured in two ways:
- static hold: `ARB_PAPER_POSITION_LIFETIME_SECONDS`
- dynamic hold by time-to-resolution: set `ARB_PAPER_DYNAMIC_LIFETIME_ENABLED=true` and tune
  `ARB_PAPER_DYNAMIC_LIFETIME_RESOLUTION_FRACTION`,
  `ARB_PAPER_DYNAMIC_LIFETIME_MIN_SECONDS`,
  `ARB_PAPER_DYNAMIC_LIFETIME_MAX_SECONDS`
