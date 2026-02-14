import type { Position } from "../lib/types";

interface PositionsTableProps {
  positions: Position[];
}

function formatAge(openedAt: number): string {
  const now = Date.now() / 1000; // convert to seconds
  const elapsed = Math.max(0, now - openedAt);

  if (elapsed < 60) return `${Math.floor(elapsed)}s`;
  if (elapsed < 3600) return `${Math.floor(elapsed / 60)}m`;
  if (elapsed < 86400) return `${Math.floor(elapsed / 3600)}h ${Math.floor((elapsed % 3600) / 60)}m`;
  return `${Math.floor(elapsed / 86400)}d ${Math.floor((elapsed % 86400) / 3600)}h`;
}

export default function PositionsTable({ positions }: PositionsTableProps) {
  if (positions.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 bg-gray-900 rounded-lg border border-gray-800">
        <p className="text-gray-500 text-sm">No open positions</p>
      </div>
    );
  }

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 overflow-hidden">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-gray-800 text-gray-400 text-xs uppercase tracking-wider">
            <th className="text-left px-4 py-3 font-medium">Market</th>
            <th className="text-left px-4 py-3 font-medium">Side</th>
            <th className="text-right px-4 py-3 font-medium">Contracts</th>
            <th className="text-right px-4 py-3 font-medium">Entry</th>
            <th className="text-right px-4 py-3 font-medium">Exp. Profit</th>
            <th className="text-right px-4 py-3 font-medium">Age</th>
          </tr>
        </thead>
        <tbody>
          {positions.map((pos, i) => (
            <tr
              key={`${pos.market_id}-${i}`}
              className="border-b border-gray-800/50 hover:bg-gray-800/40 transition-colors"
            >
              <td className="px-4 py-2.5 text-gray-200 font-mono text-xs truncate max-w-[180px]">
                {pos.market_id}
              </td>
              <td className="px-4 py-2.5">
                <span
                  className={`text-xs font-semibold ${
                    pos.side.toLowerCase() === "buy" || pos.side.toLowerCase() === "yes"
                      ? "text-emerald-400"
                      : "text-red-400"
                  }`}
                >
                  {pos.side.toUpperCase()}
                </span>
              </td>
              <td className="px-4 py-2.5 text-right text-gray-300 font-mono">
                {pos.contracts}
              </td>
              <td className="px-4 py-2.5 text-right text-gray-300 font-mono">
                ${pos.entry.toFixed(2)}
              </td>
              <td
                className={`px-4 py-2.5 text-right font-mono ${
                  pos.expected_profit >= 0 ? "text-emerald-400" : "text-red-400"
                }`}
              >
                ${pos.expected_profit.toFixed(2)}
              </td>
              <td className="px-4 py-2.5 text-right text-gray-400 text-xs">
                {formatAge(pos.opened_at)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
