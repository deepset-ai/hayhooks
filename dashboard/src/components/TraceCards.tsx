import { memo } from "react"

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

  return (
    <TooltipProvider delay={200}>
      <div className="space-y-2 pr-3">
        {traces.map((trace) => (
          <TraceCard
            key={trace.trace_id}
            trace={trace}
            isFresh={isFresh(trace.trace_id)}
            slowComponentMinDurationMs={slowComponentMinDurationMs}
          />
        ))}
      </div>
    </TooltipProvider>
  )
})
