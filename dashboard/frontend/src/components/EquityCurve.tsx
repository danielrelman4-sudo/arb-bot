import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import type { EquityPoint } from "../lib/types";

interface EquityCurveProps {
  data: EquityPoint[];
}

interface ChartDatum {
  ts: number;
  label: string;
  pnl: number;
  high_water: number;
  drawdown: number;
}

function formatTs(ts: number): string {
  const d = new Date(ts * 1000);
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

function formatTsFull(ts: number): string {
  const d = new Date(ts * 1000);
  return d.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function EquityCurve({ data }: EquityCurveProps) {
  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-80 bg-gray-900 rounded-lg border border-gray-800">
        <p className="text-gray-500 text-sm">No data</p>
      </div>
    );
  }

  const chartData: ChartDatum[] = data.map((pt) => ({
    ts: pt.ts,
    label: formatTs(pt.ts),
    pnl: parseFloat(pt.pnl.toFixed(2)),
    high_water: parseFloat(pt.high_water.toFixed(2)),
    drawdown: parseFloat((-Math.abs(pt.drawdown)).toFixed(2)),
  }));

  return (
    <div className="h-80 bg-gray-900 rounded-lg border border-gray-800 p-4">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={chartData}>
          <defs>
            <linearGradient id="pnlGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="ddGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#ef4444" stopOpacity={0.25} />
              <stop offset="95%" stopColor="#ef4444" stopOpacity={0.05} />
            </linearGradient>
          </defs>

          <XAxis
            dataKey="label"
            stroke="#6b7280"
            tick={{ fill: "#9ca3af", fontSize: 11 }}
            tickLine={{ stroke: "#4b5563" }}
            interval="preserveStartEnd"
          />
          <YAxis
            yAxisId="pnl"
            stroke="#6b7280"
            tick={{ fill: "#9ca3af", fontSize: 11 }}
            tickLine={{ stroke: "#4b5563" }}
            tickFormatter={(v: number) => `$${v.toFixed(0)}`}
          />
          <YAxis
            yAxisId="dd"
            orientation="right"
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
            labelFormatter={(_label, payload) => {
              if (payload && payload.length > 0) {
                const point = payload[0].payload as ChartDatum;
                return formatTsFull(point.ts);
              }
              return String(_label);
            }}
            formatter={(value: number | undefined, name: string | undefined) => {
              const v = value ?? 0;
              const label =
                name === "pnl"
                  ? "PnL"
                  : name === "high_water"
                    ? "High Water"
                    : "Drawdown";
              return [`$${v.toFixed(2)}`, label];
            }}
          />

          <ReferenceLine yAxisId="pnl" y={0} stroke="#4b5563" strokeDasharray="3 3" />

          {/* Drawdown area (red, semi-transparent) */}
          <Area
            yAxisId="dd"
            type="monotone"
            dataKey="drawdown"
            stroke="#ef4444"
            strokeWidth={1}
            fill="url(#ddGradient)"
            fillOpacity={1}
          />

          {/* High water mark (dashed gray line) */}
          <Area
            yAxisId="pnl"
            type="monotone"
            dataKey="high_water"
            stroke="#6b7280"
            strokeWidth={1.5}
            strokeDasharray="4 4"
            fill="none"
          />

          {/* Cumulative PnL line (emerald green) */}
          <Area
            yAxisId="pnl"
            type="monotone"
            dataKey="pnl"
            stroke="#10b981"
            strokeWidth={2}
            fill="url(#pnlGradient)"
            fillOpacity={1}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
