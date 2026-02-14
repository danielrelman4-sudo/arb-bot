import { useMemo } from "react";
import type { BotState, Decision } from "../lib/types";
import StatusBadge from "../components/StatusBadge";
import PnlChart from "../components/PnlChart";
import PositionsTable from "../components/PositionsTable";
import TradeFeed from "../components/TradeFeed";

interface DashboardProps {
  state: BotState;
  tradeFeed: Decision[];
}

function formatUptime(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function formatUsd(value: number): string {
  const sign = value < 0 ? "-" : "";
  return `${sign}$${Math.abs(value).toFixed(2)}`;
}

export default function Dashboard({ state, tradeFeed }: DashboardProps) {
  const totalCash = useMemo(
    () => Object.values(state.cash_by_venue).reduce((sum, v) => sum + v, 0),
    [state.cash_by_venue],
  );

  const totalLocked = useMemo(
    () =>
      Object.values(state.locked_capital_by_venue).reduce(
        (sum, v) => sum + v,
        0,
      ),
    [state.locked_capital_by_venue],
  );

  return (
    <div className="space-y-6">
      {/* ---- Header Bar ---- */}
      <div className="flex items-center gap-4 flex-wrap bg-gray-900 rounded-lg border border-gray-800 px-5 py-3">
        <StatusBadge mode={state.mode} />

        {/* Venue balances */}
        <div className="flex items-center gap-3 ml-2">
          {Object.entries(state.cash_by_venue).map(([venue, cash]) => (
            <div key={venue} className="text-xs">
              <span className="text-gray-500 uppercase">{venue}</span>{" "}
              <span className="text-gray-200 font-mono">{formatUsd(cash)}</span>
            </div>
          ))}
          {Object.keys(state.cash_by_venue).length === 0 && (
            <span className="text-xs text-gray-500">No venue data</span>
          )}
        </div>

        <div className="h-5 w-px bg-gray-700" />

        {/* Stat pills */}
        <StatPill label="Total Cash" value={formatUsd(totalCash)} />
        <StatPill label="Locked" value={formatUsd(totalLocked)} />
        <StatPill label="Cycle" value={String(state.cycle)} />
        <StatPill label="Uptime" value={formatUptime(state.uptime_seconds)} />
        <StatPill
          label="Daily PnL"
          value={formatUsd(state.daily_pnl)}
          color={state.daily_pnl >= 0 ? "text-emerald-400" : "text-red-400"}
        />
      </div>

      {/* ---- Top Section: PnL Chart + Positions Table ---- */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div>
          <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-2">
            Cumulative PnL
          </h3>
          <PnlChart tradeFeed={tradeFeed} />
        </div>
        <div>
          <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-2">
            Open Positions
            {state.open_positions.length > 0 && (
              <span className="ml-2 text-gray-500 normal-case">
                ({state.open_positions.length})
              </span>
            )}
          </h3>
          <PositionsTable positions={state.open_positions} />
        </div>
      </div>

      {/* ---- Bottom Section: Trade Feed ---- */}
      <div>
        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-2">
          Trade Feed
          {tradeFeed.length > 0 && (
            <span className="ml-2 text-gray-500 normal-case">
              ({tradeFeed.length} decisions)
            </span>
          )}
        </h3>
        <TradeFeed tradeFeed={tradeFeed} />
      </div>
    </div>
  );
}

/* ---- Helper Components ---- */

function StatPill({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color?: string;
}) {
  return (
    <div className="text-xs">
      <span className="text-gray-500">{label}</span>{" "}
      <span className={`font-mono ${color ?? "text-gray-200"}`}>{value}</span>
    </div>
  );
}
