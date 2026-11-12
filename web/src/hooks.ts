import { useCallback, useEffect, useRef, useState } from "react";
import { subscribeSse } from "./api";

export interface AsyncState<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
  reload: () => void;
}

/** Run an async loader on mount (and when `deps` change); optional polling interval in ms. */
export function useAsync<T>(loader: () => Promise<T>, deps: unknown[], intervalMs = 0): AsyncState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tick, setTick] = useState(0);
  const loaderRef = useRef(loader);
  useEffect(() => {
    loaderRef.current = loader;
  });

  // `loading` is true from the moment the (deps, tick) key changes until that fetch settles.
  const key = `${JSON.stringify(deps)}|${tick}`;
  const [settledKey, setSettledKey] = useState<string | null>(null);
  const loading = settledKey !== key;

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const run = async () => {
      try {
        const result = await loaderRef.current();
        if (cancelled) return;
        setData(result);
        setError(null);
      } catch (e) {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        if (!cancelled) {
          setSettledKey(key);
          if (intervalMs > 0) timer = setTimeout(run, intervalMs);
        }
      }
    };
    void run();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [key, intervalMs]);

  const reload = useCallback(() => setTick((t) => t + 1), []);
  return { data, error, loading, reload };
}

export interface LiveSnapshot<T> {
  data: T | null;
  /** Last polling error; an SSE failure is not one, it only switches to polling. */
  error: string | null;
  /** `sse` while the stream is open, `poll` after it failed, `done` once nothing more will change. */
  mode: "sse" | "poll" | "done";
}

/**
 * Follow a document the server republishes whenever it changes: SSE first (`url` sends one `event`
 * per new snapshot, then `end`), falling back to polling `load` every `intervalMs` until `terminal`
 * if the EventSource fails. State is per mount: key the component on whatever `url` identifies.
 */
export function useLiveSnapshot<T>(url: string, event: string, load: () => Promise<T>, terminal: boolean, intervalMs = 2000): LiveSnapshot<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<LiveSnapshot<T>["mode"]>("sse");
  const loadRef = useRef(load);
  const terminalRef = useRef(terminal);
  useEffect(() => {
    loadRef.current = load;
    terminalRef.current = terminal;
  });

  useEffect(() => {
    if (mode !== "sse") return;
    const dispose = subscribeSse(
      url,
      [event],
      (name, payload) => {
        if (name === "end") setMode("done");
        else setData(JSON.parse(payload) as T);
      },
      () => {
        // Endpoint unavailable (older server / proxy without SSE) or connection lost: poll instead.
        dispose();
        setMode("poll");
      },
    );
    return dispose;
  }, [url, event, mode]);

  useEffect(() => {
    if (mode !== "poll") return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const tick = async () => {
      // Only a fetch that started after the end is known to hold the final document.
      const ended = terminalRef.current;
      try {
        const result = await loadRef.current();
        if (cancelled) return;
        setData(result);
        setError(null);
      } catch (e) {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : String(e));
      }
      if (ended) setMode("done");
      else timer = setTimeout(tick, intervalMs);
    };
    void tick();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [url, mode, intervalMs]);

  return { data, error, mode };
}

/** Debounce a value by `ms`. */
export function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return v;
}
