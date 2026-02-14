import type { BotState } from "../lib/types";
import StatusBadge from "../components/StatusBadge";
import KillSwitch from "../components/KillSwitch";

function formatUptime(seconds: number): string {
  const d = Math.floor(seconds / 86400);
  const h = Math.floor((seconds % 86400) / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  const parts: string[] = [];
  if (d > 0) parts.push(`${d}d`);
  parts.push(`${h}h`);
  parts.push(`${m}m`);
  parts.push(`${s}s`);
  return parts.join(" ");
}

function formatUsd(value: number): string {
  const sign = value < 0 ? "-" : "";
  return `${sign}$${Math.abs(value).toFixed(2)}`;
}

function streamColor(status: string): string {
  const lower = status.toLowerCase();
  if (lower === "connected" || lower === "ok" || lower === "live") {
    return "bg-emerald-500/20 text-emerald-400 border-emerald-500/30";
  }
  if (lower === "degraded" || lower === "slow") {
    return "bg-yellow-500/20 text-yellow-400 border-yellow-500/30";
  }
  return "bg-red-500/20 text-red-400 border-red-500/30";
}

function failureColor(failures: number, limit: number): string {
  if (failures === 0) return "text-emerald-400";
  if (failures >= limit) return "text-red-400";
  return "text-amber-400";
}

export default function System({ state }: { state: BotState }) {
  // Heuristic for "near limit" on consecutive failures
  const failureLimit = 10;

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold text-gray-100">System Health</h2>

      {/* ---- Connection Status ---- */}
      <SectionCard title="Connection Status">
        <div className="flex items-center gap-4">
          <StatusBadge mode={state.mode} />
          <span className="text-sm text-gray-400">
            {state.mode === "disconnected"
              ? "Bot is offline or unreachable"
              : state.mode === "paper"
                ? "Paper trading mode active"
                : "Live trading mode active"}
          </span>
        </div>
      </SectionCard>

      {/* ---- Stream Status ---- */}
      <SectionCard title="Stream Status">
        {Object.keys(state.stream_status).length === 0 ? (
          <p className="text-sm text-gray-500">No stream data available</p>
        ) : (
          <div className="flex flex-wrap gap-3">
            {Object.entries(state.stream_status).map(([venue, status]) => (
              <span
                key={venue}
                className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold border ${streamColor(status)}`}
              >
                <span className="uppercase">{venue}</span>
                <span className="opacity-75">{status}</span>
              </span>
            ))}
          </div>
        )}
      </SectionCard>

      {/* ---- Cycle Info ---- */}
      <SectionCard title="Last Cycle Info">
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <MetricCard
            label="Duration"
            value={`${state.last_cycle.duration_ms.toFixed(0)} ms`}
          />
          <MetricCard
            label="Quotes"
            value={String(state.last_cycle.quotes_count)}
          />
          <MetricCard
            label="Opportunities"
            value={String(state.last_cycle.opportunities_found)}
          />
          <MetricCard
            label="Near Opportunities"
            value={String(state.last_cycle.near_opportunities)}
          />
        </div>
      </SectionCard>

      {/* ---- Uptime ---- */}
      <SectionCard title="Uptime">
        <p className="text-2xl font-mono text-gray-200">
          {formatUptime(state.uptime_seconds)}
        </p>
      </SectionCard>

      {/* ---- Safety Status ---- */}
      <SectionCard title="Safety Status">
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div className="bg-gray-800/50 rounded-lg p-3">
            <span className="block text-xs text-gray-500 mb-1">Daily PnL</span>
            <span
              className={`text-lg font-mono font-bold ${
                state.daily_pnl >= 0 ? "text-emerald-400" : "text-red-400"
              }`}
            >
              {formatUsd(state.daily_pnl)}
            </span>
          </div>
          <div className="bg-gray-800/50 rounded-lg p-3">
            <span className="block text-xs text-gray-500 mb-1">
              Loss Cap Remaining
            </span>
            <span
              className={`text-lg font-mono font-bold ${
                state.daily_loss_cap_remaining > 0
                  ? "text-emerald-400"
                  : "text-red-400"
              }`}
            >
              {formatUsd(state.daily_loss_cap_remaining)}
            </span>
          </div>
          <div className="bg-gray-800/50 rounded-lg p-3">
            <span className="block text-xs text-gray-500 mb-1">
              Consecutive Failures
            </span>
            <span
              className={`text-lg font-mono font-bold ${failureColor(state.consecutive_failures, failureLimit)}`}
            >
              {state.consecutive_failures}
            </span>
          </div>
        </div>
      </SectionCard>

      {/* ---- Kill Switch ---- */}
      <KillSwitch />
    </div>
  );
}

/* ---- Helper Components ---- */

function SectionCard({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 p-5">
      <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
        {title}
      </h3>
      {children}
    </div>
  );
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-gray-800/50 rounded-lg p-3">
      <span className="block text-xs text-gray-500 mb-1">{label}</span>
      <span className="text-lg font-mono font-bold text-gray-200">{value}</span>
    </div>
  );
}
