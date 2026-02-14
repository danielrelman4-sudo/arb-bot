import type { BotState } from "../lib/types";
import ConfigSection from "../components/ConfigSection";

interface ParamDef {
  key: string;
  label: string;
}

interface SectionDef {
  title: string;
  params: ParamDef[];
}

const SECTIONS: SectionDef[] = [
  {
    title: "Safety Rails",
    params: [
      { key: "ARB_DAILY_LOSS_CAP_USD", label: "Daily Loss Cap (USD)" },
      { key: "ARB_MAX_CONSECUTIVE_FAILURES", label: "Max Consecutive Failures" },
    ],
  },
  {
    title: "Sizing",
    params: [
      { key: "ARB_MAX_DOLLARS_PER_TRADE", label: "Max Dollars Per Trade" },
      { key: "ARB_MAX_CONTRACTS_PER_TRADE", label: "Max Contracts Per Trade" },
      {
        key: "ARB_MAX_BANKROLL_FRACTION_PER_TRADE",
        label: "Max Bankroll Fraction Per Trade",
      },
    ],
  },
  {
    title: "Lanes",
    params: [
      {
        key: "ARB_LANE_STRUCTURAL_BUCKET_ENABLED",
        label: "Structural Bucket Lane",
      },
      { key: "ARB_LANE_CROSS_ENABLED", label: "Cross Lane" },
      {
        key: "ARB_LANE_STRUCTURAL_PARITY_ENABLED",
        label: "Structural Parity Lane",
      },
    ],
  },
  {
    title: "Bucket Quality",
    params: [
      { key: "ARB_MAX_BUCKET_LEGS", label: "Max Bucket Legs" },
      {
        key: "ARB_MAX_BUCKET_CONSECUTIVE_FAILURES",
        label: "Max Bucket Consecutive Failures",
      },
    ],
  },
  {
    title: "Cooldowns",
    params: [
      { key: "ARB_MARKET_COOLDOWN_SECONDS", label: "Market Cooldown (seconds)" },
    ],
  },
];

export default function Config({ state }: { state: BotState }) {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold text-gray-100 mb-1">Configuration</h2>
        <p className="text-sm text-gray-400">
          Changes are applied immediately via hot-reload. Some settings
          (credentials, rules path) require a bot restart.
        </p>
      </div>

      {state.mode === "disconnected" && (
        <div className="px-4 py-3 rounded-lg bg-yellow-500/10 border border-yellow-500/30 text-yellow-400 text-sm">
          Bot is not connected. Config values shown may be stale.
        </div>
      )}

      {SECTIONS.map((section) => (
        <ConfigSection
          key={section.title}
          title={section.title}
          params={section.params}
          snapshot={state.config_snapshot}
        />
      ))}
    </div>
  );
}
