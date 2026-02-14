interface StatusBadgeProps {
  mode: "live" | "paper" | "disconnected";
}

const MODE_STYLES: Record<StatusBadgeProps["mode"], string> = {
  live: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30",
  paper: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
  disconnected: "bg-red-500/20 text-red-400 border-red-500/30",
};

const MODE_LABELS: Record<StatusBadgeProps["mode"], string> = {
  live: "LIVE",
  paper: "PAPER",
  disconnected: "DISCONNECTED",
};

export default function StatusBadge({ mode }: StatusBadgeProps) {
  return (
    <span
      className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold border ${MODE_STYLES[mode]}`}
    >
      <span
        className={`w-1.5 h-1.5 rounded-full mr-1.5 ${
          mode === "live"
            ? "bg-emerald-400"
            : mode === "paper"
              ? "bg-yellow-400"
              : "bg-red-400"
        }`}
      />
      {MODE_LABELS[mode]}
    </span>
  );
}
