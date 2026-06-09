import { useCallback, useEffect, useRef, useState } from "react"
import {
  clearTraces as apiClearTraces,
  fetchEntrypoints,
  fetchTraces,
  parseTracesPayload,
  resolveApiBase,
  traceStreamUrl,
} from "../api"
import type { DashboardConfig, TraceSummary } from "../types"
import { mergeTraces } from "../utils/traces"

export type UseTracesResult = {
  entrypoints: string[]
  traces: TraceSummary[]
  freshUntil: Record<string, number>
  updatedAt: number | null
  error: string | null
  refreshing: boolean
  clearing: boolean
  refresh: (options?: { silent?: boolean }) => Promise<void>
  clear: () => Promise<void>
}

interface State {
  entrypoints: string[]
  traces: TraceSummary[]
  freshUntil: Record<string, number>
  updatedAt: number | null
  error: string | null
  refreshing: boolean
  clearing: boolean
}

const INITIAL_STATE: State = {
  entrypoints: [],
  traces: [],
  freshUntil: {},
  updatedAt: null,
  error: null,
  refreshing: false,
  clearing: false,
}

function sameStrings(a: string[], b: string[]): boolean {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i += 1) {
    if (a[i] !== b[i]) return false
  }
  return true
}

/**
 * Compute the next `freshUntil` map after receiving an incoming batch of traces.
 *
 * - Expired entries (expiresAt <= now) are evicted.
 * - Every trace in `incomingIds` that is new (not already in `seen`) gets a
 *   fresh expiry timestamp of `now + freshMs`.
 */
function computeFreshUntil(
  prev: Record<string, number>,
  incomingIds: string[],
  seen: Set<string>,
  now: number,
  freshMs: number,
): Record<string, number> {
  const freshUntil: Record<string, number> = {}

  for (const [id, expiresAt] of Object.entries(prev)) {
    if (expiresAt > now) freshUntil[id] = expiresAt
  }

  const newIds = incomingIds.filter((id) => !seen.has(id))
  if (newIds.length > 0) {
    const expiresAt = now + freshMs
    for (const id of newIds) freshUntil[id] = expiresAt
  }

  return freshUntil
}

export function useTraces(config: DashboardConfig): UseTracesResult {
  const [state, setState] = useState<State>(INITIAL_STATE)
  // Bumped by clear() to force the SSE connection to reconnect (the server
  // resets its cursor on clear, so an open stream must restart from a fresh
  // snapshot to keep receiving new traces).
  const [streamEpoch, setStreamEpoch] = useState(0)

  /**
   * Refs carry imperative state between refresh cycles without triggering
   * re-renders. `epochRef` cancels stale in-flight results (e.g. after
   * `clear()`). `tracesRef` and `freshUntilRef` shadow the React state so
   * the `refresh` callback stays stable across renders.
   */
  const afterSeqRef = useRef<number | null>(null)
  const seenRef = useRef<Set<string>>(new Set())
  const epochRef = useRef(0)
  const inFlightRef = useRef(false)
  const mountedRef = useRef(true)
  const baseRef = useRef(config.apiBase || resolveApiBase())
  const tracesRef = useRef<TraceSummary[]>([])
  const freshUntilRef = useRef<Record<string, number>>({})

  tracesRef.current = state.traces
  freshUntilRef.current = state.freshUntil

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
    }
  }, [])

  /**
   * Apply an incoming batch of traces to visible state: track freshness for
   * genuinely new traces, then merge/dedup/cap. Shared by the polling path and
   * the SSE stream so both produce identical state transitions.
   */
  const applyIncoming = useCallback(
    (incoming: TraceSummary[]) => {
      const now = Date.now()
      const incomingIds = incoming.map((t) => t.trace_id)
      const freshUntil = computeFreshUntil(freshUntilRef.current, incomingIds, seenRef.current, now, config.freshMs)
      for (const id of incomingIds) seenRef.current.add(id)

      const currentTraces = tracesRef.current
      const capped = currentTraces.length > config.listCap ? currentTraces.slice(0, config.listCap) : currentTraces
      const nextTraces = incoming.length === 0 ? capped : mergeTraces(capped, incoming, config.listCap)

      if (incoming.length === 0 && capped !== currentTraces) {
        seenRef.current = new Set(capped.map((t) => t.trace_id))
      }
      if (nextTraces !== capped) {
        seenRef.current = new Set(nextTraces.map((t) => t.trace_id))
      }

      setState((prev) => ({ ...prev, traces: nextTraces, freshUntil, updatedAt: now, error: null }))
    },
    [config.freshMs, config.listCap],
  )

  const refresh = useCallback(
    async (options?: { silent?: boolean }) => {
      if (inFlightRef.current) return
      inFlightRef.current = true

      const silent = options?.silent ?? false
      epochRef.current += 1
      const epoch = epochRef.current

      if (!silent) setState((prev) => ({ ...prev, refreshing: true, error: null }))

      try {
        const base = baseRef.current

        // 1. Fetch entrypoints
        const nextEntrypoints = await fetchEntrypoints(base)
        if (!isCurrent()) return
        setState((prev) =>
          sameStrings(prev.entrypoints, nextEntrypoints) ? prev : { ...prev, entrypoints: nextEntrypoints },
        )

        /**
         * 2. Fetch traces via cursor-based incremental polling.
         *
         * The backend emits a monotonic cursor (`X-Hayhooks-Trace-Cursor`).
         * We pass `afterSeq` so the server only returns traces created since
         * our last known cursor. If the cursor ever *regresses* (process
         * restart), we fall back to a full re-sync.
         */
        const previousAfterSeq = afterSeqRef.current
        let result = await fetchTraces(base, config.fetchLimit, undefined, previousAfterSeq ?? undefined)
        if (!isCurrent()) return

        let { traces: incoming, nextAfterSeq, hasMore } = result

        if (nextAfterSeq !== null && previousAfterSeq !== null && nextAfterSeq < previousAfterSeq) {
          afterSeqRef.current = null
          seenRef.current = new Set()
          result = await fetchTraces(base, config.fetchLimit)
          if (!isCurrent()) return
          incoming = result.traces
          nextAfterSeq = result.nextAfterSeq
          hasMore = result.hasMore
        }

        if (nextAfterSeq !== null) afterSeqRef.current = nextAfterSeq

        while (hasMore && nextAfterSeq !== null) {
          result = await fetchTraces(base, config.fetchLimit, undefined, nextAfterSeq)
          if (!isCurrent()) return
          incoming = incoming.concat(result.traces)
          nextAfterSeq = result.nextAfterSeq
          hasMore = result.hasMore
          if (nextAfterSeq !== null) afterSeqRef.current = nextAfterSeq
        }

        // 3. Track freshness + merge into the visible list (shared with SSE).
        applyIncoming(incoming)

        if (!isCurrent()) return
      } catch (e) {
        if (isCurrent()) {
          setState((prev) => ({
            ...prev,
            error: e instanceof Error ? e.message : "Unknown error",
          }))
        }
      } finally {
        inFlightRef.current = false
        if (!silent) setState((prev) => ({ ...prev, refreshing: false }))
      }

      function isCurrent(): boolean {
        return mountedRef.current && epoch === epochRef.current
      }
    },
    [applyIncoming, config.fetchLimit],
  )

  const clear = useCallback(async () => {
    setState((prev) => ({ ...prev, clearing: true }))
    epochRef.current += 1

    try {
      await apiClearTraces(baseRef.current)
      if (!mountedRef.current) return
      seenRef.current = new Set()
      afterSeqRef.current = null
      setState((prev) => ({
        ...prev,
        traces: [],
        freshUntil: {},
        updatedAt: Date.now(),
        error: null,
        clearing: false,
      }))
      // Force the SSE stream to reconnect from a fresh (empty) snapshot, since
      // the server resets its trace cursor on clear.
      setStreamEpoch((epoch) => epoch + 1)
    } catch (e) {
      if (mountedRef.current) {
        setState((prev) => ({
          ...prev,
          error: e instanceof Error ? e.message : "Clear failed",
          clearing: false,
        }))
      }
    }
  }, [])

  /**
   * Live updates.
   *
   * Primary path is an SSE stream (`EventSource`): the server pushes a
   * `snapshot` on connect, then `trace` deltas as spans start/finish. If the
   * stream can't connect or repeatedly errors, we fall back to interval polling
   * until it recovers — and when SSE is disabled by the server (or unavailable
   * in the runtime) we use polling directly, preserving the original behavior.
   */
  useEffect(() => {
    const base = baseRef.current

    if (!config.streamEnabled || typeof EventSource === "undefined") {
      const timeout = window.setTimeout(() => void refresh({ silent: true }), 0)
      const interval = window.setInterval(() => void refresh({ silent: true }), config.pollMs)
      return () => {
        window.clearTimeout(timeout)
        window.clearInterval(interval)
      }
    }

    let source: EventSource | null = null
    let pollInterval: number | null = null
    let failures = 0
    let gotData = false
    let disposed = false
    const FAILURE_THRESHOLD = 3
    const CONNECT_TIMEOUT_MS = 4000

    const stopPolling = () => {
      if (pollInterval !== null) {
        window.clearInterval(pollInterval)
        pollInterval = null
      }
    }
    const startPollingFallback = () => {
      if (pollInterval !== null) return
      void refresh({ silent: true })
      pollInterval = window.setInterval(() => void refresh({ silent: true }), config.pollMs)
    }

    const loadEntrypoints = () => {
      void fetchEntrypoints(base)
        .then((next) => {
          if (disposed) return
          setState((prev) => (sameStrings(prev.entrypoints, next) ? prev : { ...prev, entrypoints: next }))
        })
        .catch(() => {
          /* entrypoints are best-effort; ignore transient failures */
        })
    }

    const handlePayload = (raw: string) => {
      try {
        const result = parseTracesPayload(JSON.parse(raw))
        if (result.nextAfterSeq !== null) afterSeqRef.current = result.nextAfterSeq
        // The stream is genuinely delivering data: only now is it safe to treat
        // it as healthy. Resetting on `open` instead would let a connection that
        // flaps (open → error → open …) without ever delivering a payload bounce
        // below FAILURE_THRESHOLD forever and never fall back to polling.
        gotData = true
        failures = 0
        stopPolling()
        applyIncoming(result.traces)
      } catch {
        /* ignore malformed SSE frame */
      }
    }

    source = new EventSource(traceStreamUrl(base, afterSeqRef.current))
    // Safety net: if the stream never delivers data (e.g. `open` never fires, or
    // it opens then stalls), fall back to polling so the UI is never left blank
    // or stale with no error surfaced.
    const watchdog = window.setTimeout(() => {
      if (!disposed && !gotData) startPollingFallback()
    }, CONNECT_TIMEOUT_MS)

    source.addEventListener("open", loadEntrypoints)
    source.addEventListener("snapshot", (event) => handlePayload((event as MessageEvent<string>).data))
    source.addEventListener("trace", (event) => handlePayload((event as MessageEvent<string>).data))
    source.addEventListener("error", () => {
      // EventSource auto-reconnects; after repeated failures, fall back to polling.
      failures += 1
      if (failures >= FAILURE_THRESHOLD) startPollingFallback()
    })

    return () => {
      disposed = true
      window.clearTimeout(watchdog)
      stopPolling()
      source?.close()
    }
  }, [config.streamEnabled, config.pollMs, streamEpoch, refresh, applyIncoming])

  return {
    entrypoints: state.entrypoints,
    traces: state.traces,
    freshUntil: state.freshUntil,
    updatedAt: state.updatedAt,
    error: state.error,
    refreshing: state.refreshing,
    clearing: state.clearing,
    refresh,
    clear,
  }
}
