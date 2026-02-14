import { useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import type { Decision } from "../lib/types";

interface PnlChartProps {
  tradeFeed: Decision[];
}

interface ChartPoint {
  index: number;
  pnl: number;
}

export default function PnlChart({ tradeFeed }: PnlChartProps) {
  const data = useMemo<ChartPoint[]>(() => {
    // Reverse so oldest is first (tradeFeed is newest-first)
    const chronological = [...tradeFeed].reverse();
    let cumulative = 0;
    return chronological.map((d, i) => {
      // Use edge as proxy for PnL contribution per trade
      cumulative += d.edge;
      return { index: i + 1, pnl: parseFloat(cumulative.toFixed(4)) };
    });
  }, [tradeFeed]);

  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 bg-gray-900 rounded-lg border border-gray-800">
        <p className="text-gray-500 text-sm">No trades yet</p>
      </div>
    );
  }

  return (
    <div className="h-64 bg-gray-900 rounded-lg border border-gray-800 p-4">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <XAxis
            dataKey="index"
            stroke="#6b7280"
            tick={{ fill: "#9ca3af", fontSize: 11 }}
            tickLine={{ stroke: "#4b5563" }}
          />
          <YAxis
            stroke="#6b7280"
            tick={{ fill: "#9ca3af", fontSize: 11 }}
            tickLine={{ stroke: "#4b5563" }}
            tickFormatter={(v: number) => `$${v.toFixed(2)}`}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#1f2937",
              border: "1px solid #374151",
              borderRadius: "0.5rem",
              color: "#f3f4f6",
              fontSize: 12,
            }}
            formatter={(value: number | undefined) => [
              value != null ? `$${value.toFixed(4)}` : "$0.00",
              "PnL",
            ]}
            labelFormatter={(label) => `Trade #${String(label)}`}
          />
          <Line
            type="monotone"
            dataKey="pnl"
            stroke="#10b981"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, fill: "#10b981" }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
