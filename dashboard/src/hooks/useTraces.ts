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

function sameStrings(a: string[], b: string[]): boolean {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i += 1) {
    if (a[i] !== b[i]) return false
  }
  return true
}

export function useTraces(config: DashboardConfig): UseTracesResult {
  const [entrypoints, setEntrypoints] = useState<string[]>([])
  const [traces, setTraces] = useState<TraceSummary[]>([])
  const [freshUntil, setFreshUntil] = useState<Record<string, number>>({})
  const [updatedAt, setUpdatedAt] = useState<number | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [refreshing, setRefreshing] = useState(false)
  const [clearing, setClearing] = useState(false)

  const sinceRef = useRef<number | null>(null)
  const seenRef = useRef(new Set<string>())
  const baseRef = useRef(resolveApiBase())
  const refreshInFlightRef = useRef(false)
  const mountedRef = useRef(true)

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
    }
  }, [])

  const refresh = useCallback(async (options?: { silent?: boolean }) => {
    const silent = options?.silent ?? false
    if (refreshInFlightRef.current) return
    refreshInFlightRef.current = true
    if (!silent) setRefreshing(true)
    try {
      const base = baseRef.current

      // Step 1: Fetch latest entrypoints
      const nextEntrypoints = await fetchEntrypoints(base)
      if (!mountedRef.current) return
      setEntrypoints((prev) => (sameStrings(prev, nextEntrypoints) ? prev : nextEntrypoints))

      // Step 2: Fetch traces (incremental via since cursor)
      const rawTraces = await fetchTraces(base, config.fetchLimit, sinceRef.current ?? undefined)
      if (!mountedRef.current) return

      // Step 3: Filter to truly new traces when doing incremental fetch
      const cutoff = sinceRef.current
      const incoming = cutoff === null ? rawTraces : rawTraces.filter((t) => t.start_time_ms >= cutoff)

      // Step 4: Track which traces are "fresh" (newly seen)
      const now = Date.now()
      const incomingIds = incoming.map((t) => t.trace_id)
      const freshIds = incomingIds.filter((id) => !seenRef.current.has(id))
      for (const id of incomingIds) seenRef.current.add(id)

      setFreshUntil((prev) => {
        let changed = false
        const next: Record<string, number> = {}
        for (const [k, v] of Object.entries(prev)) {
          if (v > now) {
            next[k] = v
          } else {
            changed = true
          }
        }
        if (freshIds.length === 0) return changed ? next : prev
        const freshUntilMs = now + config.freshMs
        for (const id of freshIds) {
          if (next[id] !== freshUntilMs) changed = true
          next[id] = freshUntilMs
        }
        return changed ? next : prev
      })

      // Step 5: Merge incoming into existing, capped to listCap
      setTraces((prev) => {
        const base = prev.length > config.listCap ? prev.slice(0, config.listCap) : prev
        if (incoming.length === 0) {
          if (base !== prev) {
            seenRef.current = new Set(base.map((trace) => trace.trace_id))
          }
          return base
        }
        const next = mergeTraces(base, incoming, config.listCap)
        if (next.length === base.length && next.every((trace, index) => trace === base[index])) {
          return base
        }
        seenRef.current = new Set(next.map((t) => t.trace_id))
        return next
      })

      // Step 6: Advance the "since" cursor
      const newest = incoming.reduce((m, t) => Math.max(m, t.start_time_ms), sinceRef.current ?? 0)
      if (newest > 0) sinceRef.current = newest

      if (!mountedRef.current) return
      setUpdatedAt(Date.now())
      setError(null)
    } catch (e) {
      if (mountedRef.current) {
        setError(e instanceof Error ? e.message : "Unknown error")
      }
    } finally {
      refreshInFlightRef.current = false
      if (!silent && mountedRef.current) setRefreshing(false)
    }
  }, [config.fetchLimit, config.freshMs, config.listCap])

  const clear = useCallback(async () => {
    setClearing(true)
    setError(null)

    try {
      await apiClearTraces(baseRef.current)
      if (!mountedRef.current) return
      const now = Date.now()
      setTraces([])
      setFreshUntil({})
      seenRef.current = new Set()
      sinceRef.current = now
      setUpdatedAt(now)
    } catch (e) {
      if (mountedRef.current) {
        setError(e instanceof Error ? e.message : "Clear failed")
      }
    } finally {
      if (mountedRef.current) {
        setClearing(false)
      }
    }
  }, [])

  // Initial fetch + polling
  useEffect(() => {
    const timeout = window.setTimeout(() => void refresh({ silent: true }), 0)
    const interval = window.setInterval(() => void refresh({ silent: true }), config.pollMs)
    return () => {
      window.clearTimeout(timeout)
      window.clearInterval(interval)
    }
  }, [config.pollMs, refresh])

  return { entrypoints, traces, freshUntil, updatedAt, error, refreshing, clearing, refresh, clear }
}
