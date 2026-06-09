import { memo, useMemo } from "react"

import { TooltipProvider } from "@/components/ui/tooltip"
import type { TraceSummary } from "../types"
import { useTraceFreshnessTimer } from "../hooks/useTraceFreshnessTimer"
import { TraceCard } from "./TraceCard"

type TraceCardsProps = {
  traces: TraceSummary[]
  slowComponentMinDurationMs: number
}

export const TraceCards = memo(function TraceCards({ traces, slowComponentMinDurationMs }: TraceCardsProps) {
  const isFresh = useTraceFreshnessTimer(traces)

  // The most recent trace by start time (independent of the list's sort mode);
  // it stays auto-expanded so the latest run is always in view.
  const latestTraceId = useMemo(() => {
    let latest: TraceSummary | null = null
    for (const trace of traces) {
      if (latest === null || trace.start_time_ms > latest.start_time_ms) latest = trace
    }
    return latest?.trace_id ?? null
  }, [traces])

  return (
    <TooltipProvider delay={200}>
      <div className="space-y-2 pr-3">
        {traces.map((trace) => (
          <TraceCard
            key={trace.trace_id}
            trace={trace}
            isFresh={isFresh(trace.trace_id)}
            isLatest={trace.trace_id === latestTraceId}
            slowComponentMinDurationMs={slowComponentMinDurationMs}
          />
        ))}
      </div>
    </TooltipProvider>
  )
})
