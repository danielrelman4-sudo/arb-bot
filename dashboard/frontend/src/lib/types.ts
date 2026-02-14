export interface BotState {
  type: "state";
  ts: number;
  cycle: number;
  mode: "live" | "paper" | "disconnected";
  uptime_seconds: number;
  cash_by_venue: Record<string, number>;
  locked_capital_by_venue: Record<string, number>;
  open_positions: Position[];
  last_cycle: CycleInfo;
  stream_status: Record<string, string>;
  daily_pnl: number;
  daily_trades: number;
  daily_loss_cap_remaining: number;
  consecutive_failures: number;
  config_snapshot: Record<string, number | boolean | string>;
}

export interface CycleInfo {
  duration_ms: number;
  quotes_count: number;
  opportunities_found: number;
  near_opportunities: number;
  decisions: Decision[];
}

export interface Decision {
  action: string;
  kind: string;
  match_key: string;
  reason: string;
  edge: number;
  fill_prob: number;
}

export interface Position {
  market_id: string;
  side: string;
  contracts: number;
  entry: number;
  expected_profit: number;
  opened_at: number;
}

export interface DailyPnl {
  date: string;
  pnl: number;
  trades: number;
  wins: number;
  avg_slippage: number;
  avg_fill_rate: number;
}

export interface EquityPoint {
  ts: number;
  pnl: number;
  trade_pnl: number;
  high_water: number;
  drawdown: number;
  kind: string;
}

export interface LanePnl {
  kind: string;
  pnl: number;
  trades: number;
  wins: number;
}

export interface BucketPnl {
  group_id: string;
  pnl: number;
  trades: number;
  wins: number;
  avg_pnl: number;
}
