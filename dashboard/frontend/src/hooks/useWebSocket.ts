import { useEffect, useRef, useState, useCallback } from "react";
import type { BotState, Decision } from "../lib/types";

const EMPTY_STATE: BotState = {
  type: "state",
  ts: 0,
  cycle: 0,
  mode: "disconnected",
  uptime_seconds: 0,
  cash_by_venue: {},
  locked_capital_by_venue: {},
  open_positions: [],
  last_cycle: { duration_ms: 0, quotes_count: 0, opportunities_found: 0, near_opportunities: 0, decisions: [] },
  stream_status: {},
  daily_pnl: 0,
  daily_trades: 0,
  daily_loss_cap_remaining: 0,
  consecutive_failures: 0,
  config_snapshot: {},
};

export function useWebSocket() {
  const [state, setState] = useState<BotState>(EMPTY_STATE);
  const [connected, setConnected] = useState(false);
  const [tradeFeed, setTradeFeed] = useState<Decision[]>([]);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${proto}//${window.location.host}/ws`;

    function connect() {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => setConnected(true);
      ws.onclose = () => {
        setConnected(false);
        setTimeout(connect, 3000);
      };
      ws.onmessage = (evt) => {
        try {
          const data = JSON.parse(evt.data) as BotState;
          setState(data);
          if (data.last_cycle?.decisions?.length) {
            setTradeFeed((prev) => [...data.last_cycle.decisions, ...prev].slice(0, 200));
          }
        } catch {}
      };
    }

    connect();
    return () => { wsRef.current?.close(); };
  }, []);

  const sendPing = useCallback(() => {
    wsRef.current?.send("ping");
  }, []);

  return { state, connected, tradeFeed, sendPing };
}
