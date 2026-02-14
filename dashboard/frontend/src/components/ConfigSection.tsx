import { useState, useCallback } from "react";
import { postConfig } from "../hooks/useApi";

interface ParamDef {
  key: string;
  label: string;
}

interface ConfigSectionProps {
  title: string;
  params: ParamDef[];
  snapshot: Record<string, number | boolean | string>;
}

type FeedbackState = Record<string, { msg: string; ok: boolean; timer: number }>;

export default function ConfigSection({ title, params, snapshot }: ConfigSectionProps) {
  const [open, setOpen] = useState(true);
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [feedback, setFeedback] = useState<FeedbackState>({});
  const [submitting, setSubmitting] = useState<Record<string, boolean>>({});

  const showFeedback = useCallback((key: string, msg: string, ok: boolean) => {
    setFeedback((prev) => {
      // Clear any existing timer for this key
      if (prev[key]?.timer) window.clearTimeout(prev[key].timer);
      const timer = window.setTimeout(() => {
        setFeedback((p) => {
          const next = { ...p };
          delete next[key];
          return next;
        });
      }, 3000);
      return { ...prev, [key]: { msg, ok, timer } };
    });
  }, []);

  const handleApply = useCallback(
    async (key: string, currentValue: number | boolean | string) => {
      const raw = drafts[key];
      let value: number | boolean | string;

      if (typeof currentValue === "boolean") {
        // Toggle switches handle their own value
        value = raw === "true";
      } else if (typeof currentValue === "number") {
        value = Number(raw ?? currentValue);
        if (Number.isNaN(value)) {
          showFeedback(key, "Invalid number", false);
          return;
        }
      } else {
        value = raw ?? String(currentValue);
      }

      setSubmitting((prev) => ({ ...prev, [key]: true }));
      try {
        await postConfig(key, value);
        showFeedback(key, "Applied", true);
        // Clear draft so input shows server value on next render
        setDrafts((prev) => {
          const next = { ...prev };
          delete next[key];
          return next;
        });
      } catch {
        showFeedback(key, "Failed to apply", false);
      } finally {
        setSubmitting((prev) => ({ ...prev, [key]: false }));
      }
    },
    [drafts, showFeedback],
  );

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800">
      {/* Section header */}
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex items-center justify-between w-full px-5 py-3 text-left hover:bg-gray-800/50 transition-colors rounded-lg"
      >
        <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wider">
          {title}
        </h3>
        <svg
          className={`w-4 h-4 text-gray-500 transition-transform ${open ? "rotate-180" : ""}`}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={2}
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {/* Param rows */}
      {open && (
        <div className="border-t border-gray-800 divide-y divide-gray-800/60">
          {params.map(({ key, label }) => {
            const currentValue = snapshot[key];
            const isBool = typeof currentValue === "boolean";
            const isNumber = typeof currentValue === "number";
            const fb = feedback[key];
            const busy = submitting[key] ?? false;

            return (
              <div
                key={key}
                className="flex items-center gap-4 px-5 py-3 flex-wrap"
              >
                {/* Label */}
                <div className="min-w-[220px] flex-shrink-0">
                  <span className="text-sm text-gray-300">{label}</span>
                  <span className="block text-xs text-gray-500 font-mono">{key}</span>
                </div>

                {/* Current value */}
                <div className="text-xs text-gray-500 min-w-[80px]">
                  Current:{" "}
                  <span className="text-gray-400 font-mono">
                    {String(currentValue ?? "N/A")}
                  </span>
                </div>

                {/* Input */}
                <div className="flex items-center gap-2 flex-1 min-w-[200px]">
                  {isBool ? (
                    <ToggleSwitch
                      checked={
                        drafts[key] !== undefined
                          ? drafts[key] === "true"
                          : (currentValue as boolean)
                      }
                      onChange={(v) =>
                        setDrafts((prev) => ({ ...prev, [key]: String(v) }))
                      }
                    />
                  ) : (
                    <input
                      type={isNumber ? "number" : "text"}
                      step={isNumber ? "any" : undefined}
                      className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-gray-200 font-mono focus:outline-none focus:border-blue-500 flex-1"
                      value={
                        drafts[key] !== undefined
                          ? drafts[key]
                          : String(currentValue ?? "")
                      }
                      onChange={(e) =>
                        setDrafts((prev) => ({ ...prev, [key]: e.target.value }))
                      }
                    />
                  )}

                  <button
                    type="button"
                    disabled={busy}
                    onClick={() => handleApply(key, currentValue)}
                    className="px-3 py-1.5 text-xs font-medium rounded bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed text-white transition-colors"
                  >
                    {busy ? "..." : "Apply"}
                  </button>
                </div>

                {/* Feedback */}
                {fb && (
                  <span
                    className={`text-xs font-medium ${fb.ok ? "text-emerald-400" : "text-red-400"}`}
                  >
                    {fb.msg}
                  </span>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

/* ---- Toggle Switch ---- */

function ToggleSwitch({
  checked,
  onChange,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
        checked ? "bg-emerald-600" : "bg-gray-700"
      }`}
    >
      <span
        className={`inline-block h-4 w-4 rounded-full bg-white transition-transform ${
          checked ? "translate-x-6" : "translate-x-1"
        }`}
      />
    </button>
  );
}
