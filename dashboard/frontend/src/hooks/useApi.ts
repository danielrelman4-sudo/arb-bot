import { useState, useEffect, useCallback } from "react";

export function useApi<T>(url: string, defaultValue: T) {
  const [data, setData] = useState<T>(defaultValue);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const res = await fetch(url);
      const json = await res.json();
      setData(json);
    } catch {
    } finally {
      setLoading(false);
    }
  }, [url]);

  useEffect(() => { refresh(); }, [refresh]);

  return { data, loading, refresh };
}

export async function postConfig(key: string, value: number | boolean | string) {
  const res = await fetch(`/api/config/${key}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ value }),
  });
  return res.json();
}

export async function postKillSwitch(activate: boolean) {
  const res = await fetch("/api/kill-switch", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ activate }),
  });
  return res.json();
}
