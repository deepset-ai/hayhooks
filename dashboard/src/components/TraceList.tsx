import { memo, useMemo, useState } from "react"
import { Activity, ArrowDownWideNarrow, GitBranch } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { TooltipProvider } from "@/components/ui/tooltip"
import { useClock } from "../hooks/useClock"
import { useTraceStatus } from "../hooks/useTracesContext"
import type { SortMode, TraceSummary } from "../types"
import { TraceCard } from "./TraceCard"

type TraceListProps = {
  traces: TraceSummary[]
  totalTraces: number
  filter: string | null
  freshUntil: Record<string, number>
  slowComponentMinDurationMs?: number
  onClearFilter: () => void
}

type TraceCardsProps = {
  traces: TraceSummary[]
  freshUntil: Record<string, number>
  slowComponentMinDurationMs: number
}

const TraceCards = memo(function TraceCards({ traces, freshUntil, slowComponentMinDurationMs }: TraceCardsProps) {
  const nowMs = useClock()

  return (
    <TooltipProvider delay={200}>
      <div className="space-y-2 pr-3">
        {traces.map((trace) => (
          <TraceCard
            key={trace.trace_id}
            trace={trace}
            isFresh={(freshUntil[trace.trace_id] ?? 0) > nowMs}
            slowComponentMinDurationMs={slowComponentMinDurationMs}
          />
        ))}
      </div>
    </TooltipProvider>
  )
})

export const TraceList = memo(function TraceList({
  traces,
  totalTraces,
  filter,
  freshUntil,
  slowComponentMinDurationMs = 1000,
  onClearFilter,
}: TraceListProps) {
  const [sortMode, setSortMode] = useState<SortMode>("newest")

  const visibleTraces = useMemo(
    () => (sortMode === "slowest" ? [...traces].sort((a, b) => b.duration_ms - a.duration_ms) : traces),
    [traces, sortMode],
  )

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-medium">Live Traces</h2>
          {filter && (
            <Badge variant="secondary" className="gap-1 rounded px-1.5 text-[11px]">
              {filter}
              <button onClick={onClearFilter} className="ml-0.5 rounded-sm opacity-60 hover:opacity-100" aria-label="Clear filter">
                ×
              </button>
            </Badge>
          )}
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => setSortMode((m) => (m === "newest" ? "slowest" : "newest"))}
            className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
          >
            <ArrowDownWideNarrow className="size-3" />
            {sortMode === "newest" ? "Newest" : "Slowest"}
          </button>
          {visibleTraces.length > 0 && (
            <span className="text-xs tabular-nums text-muted-foreground">
              {filter
                ? `${visibleTraces.length} of ${totalTraces}`
                : `${totalTraces} trace${totalTraces !== 1 ? "s" : ""}`}
            </span>
          )}
        </div>
      </div>
      <ScrollArea className="h-[calc(100vh-240px)] min-h-[400px]">
        {visibleTraces.length === 0 ? (
          <TraceListEmptyState
            filter={filter}
            hasTraces={totalTraces > 0}
            onClearFilter={onClearFilter}
          />
        ) : (
          <TraceCards
            traces={visibleTraces}
            freshUntil={freshUntil}
            slowComponentMinDurationMs={slowComponentMinDurationMs}
          />
        )}
      </ScrollArea>
    </div>
  )
})

const TraceListEmptyState = memo(function TraceListEmptyState({
  filter,
  hasTraces,
  onClearFilter,
}: {
  filter: string | null
  hasTraces: boolean
  onClearFilter: () => void
}) {
  const { updatedAt, error } = useTraceStatus()

  if (filter) {
    return (
      <div className="flex flex-col items-center justify-center gap-3 py-20 text-center text-muted-foreground">
        <GitBranch className="size-8 opacity-30" />
        <div>
          <p className="text-sm font-medium">No traces for <span className="font-mono text-foreground/70">{filter}</span></p>
          <p className="mt-1 text-xs">New traces will appear automatically when this pipeline handles requests.</p>
        </div>
        <button onClick={onClearFilter} className="mt-1 text-xs text-primary hover:underline">
          Show all pipelines
        </button>
      </div>
    )
  }

  if (error !== null) {
    return (
      <div className="flex flex-col items-center justify-center gap-3 py-20 text-center text-muted-foreground">
        <Activity className="size-8 text-destructive/70 opacity-70" />
        <div>
          <p className="text-sm font-medium text-foreground">Unable to load traces</p>
          <p className="mt-1 max-w-md text-xs text-muted-foreground">{error}</p>
        </div>
      </div>
    )
  }

  if (!hasTraces && updatedAt !== null) {
    return (
      <div className="flex flex-col items-center justify-center gap-3 py-20 text-center text-muted-foreground">
        <Activity className="size-8 opacity-30" />
        <div>
          <p className="text-sm font-medium">No traces yet</p>
          <p className="mt-1 text-xs">Send a request to any deployed pipeline and traces will appear here in real time.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex flex-col items-center justify-center gap-3 py-20 text-center text-muted-foreground">
      <div className="size-8 animate-pulse rounded-full bg-muted" />
      <p className="text-sm">Connecting…</p>
    </div>
  )
})
