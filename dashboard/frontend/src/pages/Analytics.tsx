import { useState, useMemo } from "react";
import {
  BarChart,
  Bar,
  Cell,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { useApi } from "../hooks/useApi";
import EquityCurve from "../components/EquityCurve";
import type { EquityPoint, DailyPnl, LanePnl, BucketPnl } from "../lib/types";

/* ------------------------------------------------------------------ */
/*  Sort helpers                                                       */
/* ------------------------------------------------------------------ */

type SortColumn = "group_id" | "trades" | "pnl" | "win_rate" | "avg_pnl";
type SortDir = "asc" | "desc";

function compareBucket(a: BucketPnl, b: BucketPnl, col: SortColumn, dir: SortDir): number {
  let va: number | string;
  let vb: number | string;

  switch (col) {
    case "group_id":
      va = a.group_id;
      vb = b.group_id;
      break;
    case "trades":
      va = a.trades;
      vb = b.trades;
      break;
    case "pnl":
      va = a.pnl;
      vb = b.pnl;
      break;
    case "win_rate":
      va = a.trades > 0 ? a.wins / a.trades : 0;
      vb = b.trades > 0 ? b.wins / b.trades : 0;
      break;
    case "avg_pnl":
      va = a.avg_pnl;
      vb = b.avg_pnl;
      break;
  }

  const cmp = va < vb ? -1 : va > vb ? 1 : 0;
  return dir === "asc" ? cmp : -cmp;
}

/* ------------------------------------------------------------------ */
/*  Summary stat helpers                                               */
/* ------------------------------------------------------------------ */

function computeStats(equity: EquityPoint[], daily: DailyPnl[]) {
  const totalPnl = equity.length > 0 ? equity[equity.length - 1].pnl : 0;
  const maxDrawdown = equity.reduce(
    (worst, pt) => Math.min(worst, -Math.abs(pt.drawdown)),
    0,
  );

  const totalTrades = daily.reduce((s, d) => s + d.trades, 0);
  const totalWins = daily.reduce((s, d) => s + d.wins, 0);
  const winRate = totalTrades > 0 ? totalWins / totalTrades : 0;

  // Approximated Sharpe: mean(daily_pnl) / std(daily_pnl)
  let sharpe = 0;
  if (daily.length >= 2) {
    const pnls = daily.map((d) => d.pnl);
    const mean = pnls.reduce((s, v) => s + v, 0) / pnls.length;
    const variance =
      pnls.reduce((s, v) => s + (v - mean) ** 2, 0) / (pnls.length - 1);
    const std = Math.sqrt(variance);
    sharpe = std > 0 ? mean / std : 0;
  }

  return { totalPnl, maxDrawdown, winRate, sharpe, totalTrades };
}

/* ------------------------------------------------------------------ */
/*  Main component                                                     */
/* ------------------------------------------------------------------ */

export default function Analytics() {
  const { data: equity, loading: eqLoading } = useApi<EquityPoint[]>(
    "/api/analytics/equity-curve",
    [],
  );
  const { data: daily, loading: dayLoading } = useApi<DailyPnl[]>(
    "/api/analytics/daily-pnl",
    [],
  );
  const { data: lanes, loading: laneLoading } = useApi<LanePnl[]>(
    "/api/analytics/by-lane",
    [],
  );
  const { data: buckets, loading: bucketLoading } = useApi<BucketPnl[]>(
    "/api/analytics/by-bucket",
    [],
  );

  const [sortCol, setSortCol] = useState<SortColumn>("pnl");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  const stats = useMemo(() => computeStats(equity, daily), [equity, daily]);

  const sortedBuckets = useMemo(() => {
    const copy = [...buckets];
    copy.sort((a, b) => compareBucket(a, b, sortCol, sortDir));
    return copy;
  }, [buckets, sortCol, sortDir]);

  const handleSort = (col: SortColumn) => {
    if (col === sortCol) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortCol(col);
      setSortDir("desc");
    }
  };

  const loading = eqLoading || dayLoading || laneLoading || bucketLoading;

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-gray-400 text-sm animate-pulse">Loading analytics...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold text-gray-100">Analytics</h2>

      {/* ---- Summary Stats ---- */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
        <StatCard
          label="Total PnL"
          value={formatUsd(stats.totalPnl)}
          color={stats.totalPnl >= 0 ? "text-emerald-400" : "text-red-400"}
        />
        <StatCard
          label="Max Drawdown"
          value={formatUsd(stats.maxDrawdown)}
          color="text-red-400"
        />
        <StatCard
          label="Win Rate"
          value={`${(stats.winRate * 100).toFixed(1)}%`}
          color={stats.winRate >= 0.5 ? "text-emerald-400" : "text-amber-400"}
        />
        <StatCard
          label="Sharpe (approx)"
          value={stats.sharpe.toFixed(2)}
          color={stats.sharpe >= 1 ? "text-emerald-400" : "text-gray-300"}
        />
        <StatCard
          label="Total Trades"
          value={String(stats.totalTrades)}
          color="text-gray-200"
        />
      </div>

      {/* ---- Equity Curve (full width) ---- */}
      <div>
        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-2">
          Equity Curve
        </h3>
        <EquityCurve data={equity} />
      </div>

      {/* ---- Middle: Daily PnL + Lane Attribution ---- */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Daily PnL Bar Chart */}
        <div>
          <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-2">
            Daily PnL
          </h3>
          <DailyPnlChart data={daily} />
        </div>

        {/* Lane Attribution */}
        <div>
          <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-2">
            Lane Attribution
          </h3>
          <LaneChart data={lanes} />
        </div>
      </div>

      {/* ---- Bottom: Bucket Table ---- */}
      <div>
        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-2">
          Bucket Performance
        </h3>
        <BucketTable
          buckets={sortedBuckets}
          sortCol={sortCol}
          sortDir={sortDir}
          onSort={handleSort}
        />
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Sub-components                                                     */
/* ------------------------------------------------------------------ */

function StatCard({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color: string;
}) {
  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 px-4 py-3">
      <p className="text-xs text-gray-500 uppercase tracking-wider">{label}</p>
      <p className={`text-lg font-mono font-semibold mt-1 ${color}`}>{value}</p>
    </div>
  );
}

/* ---- Daily PnL bar chart ---- */

function DailyPnlChart({ data }: { data: DailyPnl[] }) {
  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 bg-gray-900 rounded-lg border border-gray-800">
        <p className="text-gray-500 text-sm">No data</p>
      </div>
    );
  }

  return (
    <div className="h-64 bg-gray-900 rounded-lg border border-gray-800 p-4">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data}>
          <XAxis
            dataKey="date"
            stroke="#6b7280"
            tick={{ fill: "#9ca3af", fontSize: 10 }}
            tickLine={{ stroke: "#4b5563" }}
            interval="preserveStartEnd"
          />
          <YAxis
            stroke="#6b7280"
            tick={{ fill: "#9ca3af", fontSize: 11 }}
            tickLine={{ stroke: "#4b5563" }}
            tickFormatter={(v: number) => `$${v.toFixed(0)}`}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#1f2937",
              border: "1px solid #374151",
              borderRadius: "0.5rem",
              color: "#f3f4f6",
              fontSize: 12,
            }}
            formatter={(value: number | undefined) => [`$${(value ?? 0).toFixed(2)}`, "PnL"]}
          />
          <ReferenceLine y={0} stroke="#4b5563" strokeDasharray="3 3" />
          <Bar dataKey="pnl" radius={[3, 3, 0, 0]}>
            {data.map((entry, idx) => (
              <Cell
                key={`daily-${idx}`}
                fill={entry.pnl >= 0 ? "#10b981" : "#ef4444"}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

/* ---- Lane attribution horizontal bar chart ---- */

function LaneChart({ data }: { data: LanePnl[] }) {
  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 bg-gray-900 rounded-lg border border-gray-800">
        <p className="text-gray-500 text-sm">No data</p>
      </div>
    );
  }

  return (
    <div className="h-64 bg-gray-900 rounded-lg border border-gray-800 p-4">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical">
          <XAxis
            type="number"
            stroke="#6b7280"
            tick={{ fill: "#9ca3af", fontSize: 11 }}
            tickLine={{ stroke: "#4b5563" }}
            tickFormatter={(v: number) => `$${v.toFixed(0)}`}
          />
          <YAxis
            type="category"
            dataKey="kind"
            stroke="#6b7280"
            tick={{ fill: "#9ca3af", fontSize: 11 }}
            tickLine={{ stroke: "#4b5563" }}
            width={100}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#1f2937",
              border: "1px solid #374151",
              borderRadius: "0.5rem",
              color: "#f3f4f6",
              fontSize: 12,
            }}
            formatter={(value: number | undefined) => [`$${(value ?? 0).toFixed(2)}`, "PnL"]}
          />
          <ReferenceLine x={0} stroke="#4b5563" strokeDasharray="3 3" />
          <Bar dataKey="pnl" radius={[0, 3, 3, 0]}>
            {data.map((entry, idx) => (
              <Cell
                key={`lane-${idx}`}
                fill={entry.pnl >= 0 ? "#10b981" : "#ef4444"}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

/* ---- Bucket performance sortable table ---- */

const SORT_ARROW_UP = "\u25B2";
const SORT_ARROW_DOWN = "\u25BC";

function SortIndicator({ col, sortCol, sortDir }: { col: SortColumn; sortCol: SortColumn; sortDir: SortDir }) {
  if (col !== sortCol) return <span className="ml-1 text-gray-600">{SORT_ARROW_UP}</span>;
  return (
    <span className="ml-1 text-gray-300">
      {sortDir === "asc" ? SORT_ARROW_UP : SORT_ARROW_DOWN}
    </span>
  );
}

function BucketTable({
  buckets,
  sortCol,
  sortDir,
  onSort,
}: {
  buckets: BucketPnl[];
  sortCol: SortColumn;
  sortDir: SortDir;
  onSort: (col: SortColumn) => void;
}) {
  if (buckets.length === 0) {
    return (
      <div className="flex items-center justify-center h-32 bg-gray-900 rounded-lg border border-gray-800">
        <p className="text-gray-500 text-sm">No bucket data</p>
      </div>
    );
  }

  const headerClass =
    "px-4 py-3 font-medium cursor-pointer select-none hover:text-gray-200 transition-colors";

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 overflow-hidden">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-gray-800 text-gray-400 text-xs uppercase tracking-wider">
            <th
              className={`text-left ${headerClass}`}
              onClick={() => onSort("group_id")}
            >
              Bucket Group ID
              <SortIndicator col="group_id" sortCol={sortCol} sortDir={sortDir} />
            </th>
            <th
              className={`text-right ${headerClass}`}
              onClick={() => onSort("trades")}
            >
              Trades
              <SortIndicator col="trades" sortCol={sortCol} sortDir={sortDir} />
            </th>
            <th
              className={`text-right ${headerClass}`}
              onClick={() => onSort("pnl")}
            >
              PnL
              <SortIndicator col="pnl" sortCol={sortCol} sortDir={sortDir} />
            </th>
            <th
              className={`text-right ${headerClass}`}
              onClick={() => onSort("win_rate")}
            >
              Win Rate
              <SortIndicator col="win_rate" sortCol={sortCol} sortDir={sortDir} />
            </th>
            <th
              className={`text-right ${headerClass}`}
              onClick={() => onSort("avg_pnl")}
            >
              Avg PnL
              <SortIndicator col="avg_pnl" sortCol={sortCol} sortDir={sortDir} />
            </th>
          </tr>
        </thead>
        <tbody>
          {buckets.map((b) => {
            const wr = b.trades > 0 ? b.wins / b.trades : 0;
            return (
              <tr
                key={b.group_id}
                className="border-b border-gray-800/50 hover:bg-gray-800/40 transition-colors"
              >
                <td className="px-4 py-2.5 text-gray-200 font-mono text-xs truncate max-w-[220px]">
                  {b.group_id}
                </td>
                <td className="px-4 py-2.5 text-right text-gray-300 font-mono">
                  {b.trades}
                </td>
                <td
                  className={`px-4 py-2.5 text-right font-mono ${
                    b.pnl >= 0 ? "text-emerald-400" : "text-red-400"
                  }`}
                >
                  {formatUsd(b.pnl)}
                </td>
                <td className="px-4 py-2.5 text-right text-gray-300 font-mono">
                  {(wr * 100).toFixed(1)}%
                </td>
                <td
                  className={`px-4 py-2.5 text-right font-mono ${
                    b.avg_pnl >= 0 ? "text-emerald-400" : "text-red-400"
                  }`}
                >
                  {formatUsd(b.avg_pnl)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/* ---- Formatting ---- */

function formatUsd(value: number): string {
  const sign = value < 0 ? "-" : "";
  return `${sign}$${Math.abs(value).toFixed(2)}`;
}
