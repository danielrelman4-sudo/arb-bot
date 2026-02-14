import type { Decision } from "../lib/types";

interface TradeFeedProps {
  tradeFeed: Decision[];
}

const ACTION_STYLES: Record<string, string> = {
  filled: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30",
  skipped: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
  detected: "bg-blue-500/20 text-blue-400 border-blue-500/30",
};

function getActionStyle(action: string): string {
  const lower = action.toLowerCase();
  return ACTION_STYLES[lower] ?? "bg-gray-500/20 text-gray-400 border-gray-500/30";
}

function truncate(s: string, maxLen: number): string {
  if (s.length <= maxLen) return s;
  return s.slice(0, maxLen - 1) + "\u2026";
}

export default function TradeFeed({ tradeFeed }: TradeFeedProps) {
  if (tradeFeed.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 bg-gray-900 rounded-lg border border-gray-800">
        <p className="text-gray-500 text-sm">No trade decisions yet</p>
      </div>
    );
  }

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 max-h-96 overflow-y-auto">
      <div className="divide-y divide-gray-800/50">
        {tradeFeed.map((d, i) => (
          <div
            key={`${d.match_key}-${i}`}
            className="px-4 py-3 hover:bg-gray-800/40 transition-colors"
          >
            <div className="flex items-center gap-3 flex-wrap">
              {/* Action badge */}
              <span
                className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold border ${getActionStyle(d.action)}`}
              >
                {d.action.toUpperCase()}
              </span>

              {/* Kind */}
              <span className="text-xs text-gray-300 font-medium">
                {d.kind}
              </span>

              {/* Match key (truncated) */}
              <span className="text-xs text-gray-500 font-mono" title={d.match_key}>
                {truncate(d.match_key, 24)}
              </span>

              {/* Spacer */}
              <span className="flex-1" />

              {/* Edge */}
              <span
                className={`text-xs font-mono ${
                  d.edge >= 0 ? "text-emerald-400" : "text-red-400"
                }`}
              >
                edge: {(d.edge * 100).toFixed(1)}%
              </span>

              {/* Fill prob */}
              <span className="text-xs font-mono text-gray-400">
                fill: {(d.fill_prob * 100).toFixed(0)}%
              </span>
            </div>

            {/* Reason */}
            {d.reason && (
              <p className="text-xs text-gray-500 mt-1 truncate">
                {d.reason}
              </p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
