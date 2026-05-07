import { useCallback, useEffect, useRef, useState } from "react"
import { clearTraces as apiClearTraces, fetchEntrypoints, fetchTraces, resolveApiBase } from "../api"
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

        // 3. Track freshness
        const now = Date.now()
        const incomingIds = incoming.map((t) => t.trace_id)
        const freshUntil = computeFreshUntil(freshUntilRef.current, incomingIds, seenRef.current, now, config.freshMs)
        for (const id of incomingIds) seenRef.current.add(id)

        // 4. Merge incoming traces into the visible list
        const currentTraces = tracesRef.current
        const capped = currentTraces.length > config.listCap
          ? currentTraces.slice(0, config.listCap)
          : currentTraces

        const nextTraces = incoming.length === 0
          ? capped
          : mergeTraces(capped, incoming, config.listCap)

        if (incoming.length === 0 && capped !== currentTraces) {
          seenRef.current = new Set(capped.map((t) => t.trace_id))
        }
        if (nextTraces !== capped) {
          seenRef.current = new Set(nextTraces.map((t) => t.trace_id))
        }

        setState((prev) => ({
          ...prev,
          traces: nextTraces,
          freshUntil,
          updatedAt: now,
          error: null,
        }))

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
    [config.fetchLimit, config.freshMs, config.listCap],
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
   * Initial fetch fires immediately (via setTimeout 0), then repeats on the
   * configured interval. Both use "silent" mode so the UI doesn't flash the
   * loading indicator on background polls.
   */
  useEffect(() => {
    const timeout = window.setTimeout(() => void refresh({ silent: true }), 0)
    const interval = window.setInterval(() => void refresh({ silent: true }), config.pollMs)
    return () => {
      window.clearTimeout(timeout)
      window.clearInterval(interval)
    }
  }, [config.pollMs, refresh])

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
