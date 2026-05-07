import { useCallback, useEffect, useMemo, useState } from "react"

import type { TraceSummary } from "../types"
import { useTraceFreshness } from "./useTracesContext"

export function useTraceFreshnessTimer(traces: TraceSummary[]) {
  const { freshUntil } = useTraceFreshness()
  const [nowMs, setNowMs] = useState(() => Date.now())

  const nextExpiryMs = useMemo(() => {
    let next: number | null = null
    for (const trace of traces) {
      const expiresAt = freshUntil[trace.trace_id]
      if (expiresAt === undefined || expiresAt <= nowMs) continue
      if (next === null || expiresAt < next) next = expiresAt
    }
    return next
  }, [traces, freshUntil, nowMs])

  useEffect(() => {
    if (nextExpiryMs === null) return
    const delay = Math.max(nextExpiryMs - Date.now(), 0) + 1
    const id = window.setTimeout(() => setNowMs(Date.now()), delay)
    return () => window.clearTimeout(id)
  }, [nextExpiryMs])

  return useCallback(
    (traceId: string) => (freshUntil[traceId] ?? 0) > nowMs,
    [freshUntil, nowMs],
  )
}
