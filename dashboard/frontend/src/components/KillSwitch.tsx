import { useState, useCallback } from "react";
import { postKillSwitch } from "../hooks/useApi";

export default function KillSwitch() {
  const [killed, setKilled] = useState(false);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleKill = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      await postKillSwitch(true);
      setKilled(true);
      setDialogOpen(false);
    } catch {
      setError("Failed to activate kill switch");
    } finally {
      setBusy(false);
    }
  }, []);

  const handleResume = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      await postKillSwitch(false);
      setKilled(false);
    } catch {
      setError("Failed to resume trading");
    } finally {
      setBusy(false);
    }
  }, []);

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 p-5">
      <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">
        Kill Switch
      </h3>

      {error && (
        <div className="mb-4 px-3 py-2 rounded bg-red-500/10 border border-red-500/30 text-red-400 text-sm">
          {error}
        </div>
      )}

      {killed ? (
        <div className="space-y-3">
          <div className="flex items-center gap-2 text-red-400 text-sm font-medium">
            <span className="inline-block w-2 h-2 rounded-full bg-red-500 animate-pulse" />
            Trading is halted
          </div>
          <button
            type="button"
            disabled={busy}
            onClick={handleResume}
            className="w-full py-3 rounded-lg font-bold text-white bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-lg"
          >
            {busy ? "Resuming..." : "Resume Trading"}
          </button>
        </div>
      ) : (
        <button
          type="button"
          onClick={() => setDialogOpen(true)}
          className="w-full py-3 rounded-lg font-bold text-white bg-red-600 hover:bg-red-500 transition-colors text-lg"
        >
          EMERGENCY STOP
        </button>
      )}

      {/* Confirmation Dialog */}
      {dialogOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          {/* Backdrop */}
          <div
            className="absolute inset-0 bg-black/70"
            onClick={() => !busy && setDialogOpen(false)}
            onKeyDown={(e) => {
              if (e.key === "Escape" && !busy) setDialogOpen(false);
            }}
            role="button"
            tabIndex={0}
            aria-label="Close dialog"
          />
          {/* Modal */}
          <div className="relative bg-gray-900 border border-gray-700 rounded-xl p-6 max-w-md w-full mx-4 shadow-2xl">
            <h4 className="text-lg font-bold text-gray-100 mb-2">
              Confirm Emergency Stop
            </h4>
            <p className="text-gray-400 text-sm mb-6">
              Are you sure? This will immediately halt all trading.
            </p>
            <div className="flex gap-3 justify-end">
              <button
                type="button"
                disabled={busy}
                onClick={() => setDialogOpen(false)}
                className="px-4 py-2 rounded-lg text-sm font-medium text-gray-300 bg-gray-800 hover:bg-gray-700 disabled:opacity-50 transition-colors"
              >
                Cancel
              </button>
              <button
                type="button"
                disabled={busy}
                onClick={handleKill}
                className="px-4 py-2 rounded-lg text-sm font-bold text-white bg-red-600 hover:bg-red-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {busy ? "Stopping..." : "Confirm Kill"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
