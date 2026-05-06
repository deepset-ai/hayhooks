import { memo, useEffect, useMemo, useState } from "react"
import { Activity, ArrowDownWideNarrow, GitBranch, Loader2 } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { TooltipProvider } from "@/components/ui/tooltip"
import { useTraceData, useTraceFreshness, useTraceStatus } from "../hooks/useTracesContext"
import type { SortMode, TraceSummary } from "../types"
import { TraceCard } from "./TraceCard"

type TraceListProps = {
  traces: TraceSummary[]
  totalTraces: number
  filter: string | null
  slowComponentMinDurationMs?: number
  onClearFilter: () => void
}

type TraceCardsProps = {
  traces: TraceSummary[]
  slowComponentMinDurationMs: number
}

const TraceCards = memo(function TraceCards({ traces, slowComponentMinDurationMs }: TraceCardsProps) {
  const { freshUntil } = useTraceFreshness()
  const [nowMs, setNowMs] = useState(() => Date.now())
  const nextFreshExpiryMs = useMemo(() => {
    let next: number | null = null
    for (const trace of traces) {
      const expiresAt = freshUntil[trace.trace_id]
      if (expiresAt === undefined || expiresAt <= nowMs) continue
      if (next === null || expiresAt < next) next = expiresAt
    }
    return next
  }, [traces, freshUntil, nowMs])

  useEffect(() => {
    if (nextFreshExpiryMs === null) return
    const delay = Math.max(nextFreshExpiryMs - Date.now(), 0) + 1
    const timeoutId = window.setTimeout(() => setNowMs(Date.now()), delay)
    return () => window.clearTimeout(timeoutId)
  }, [nextFreshExpiryMs])

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
  slowComponentMinDurationMs = 1000,
  onClearFilter,
}: TraceListProps) {
  const [sortMode, setSortMode] = useState<SortMode>("newest")
  const { listCap } = useTraceData()

  const visibleTraces = useMemo(
    () => (sortMode === "slowest" ? [...traces].sort((a, b) => b.duration_ms - a.duration_ms) : traces),
    [traces, sortMode],
  )

  const isCapped = totalTraces >= listCap

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
          <Button
            variant="ghost"
            size="xs"
            onClick={() => setSortMode((m) => (m === "newest" ? "slowest" : "newest"))}
            aria-label={`Sort: ${sortMode === "newest" ? "newest first" : "slowest first"}. Click to toggle.`}
            title="Toggle sort order"
          >
            <ArrowDownWideNarrow className="size-3" />
            {sortMode === "newest" ? "Newest" : "Slowest"}
          </Button>
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
            slowComponentMinDurationMs={slowComponentMinDurationMs}
          />
        )}
      </ScrollArea>
      {isCapped && visibleTraces.length > 0 && (
        <p className="text-[11px] text-muted-foreground/80 text-center pt-1">
          Showing latest {listCap} traces. Older traces are dropped from the live buffer.
        </p>
      )}
    </div>
  )
})

const SkeletonTraceCard = memo(function SkeletonTraceCard() {
  return (
    <div className="rounded-lg border border-l-2 border-l-transparent bg-card px-4 py-3">
      <div className="flex items-center gap-3">
        <div className="size-4 shrink-0 rounded bg-muted" />
        <div className="flex-1 space-y-2">
          <div className="flex items-center gap-2">
            <div className="h-3.5 w-32 rounded bg-muted" />
            <div className="h-3 w-12 rounded bg-muted/70" />
            <div className="h-3 w-10 rounded bg-muted/70" />
          </div>
          <div className="h-2.5 w-1/2 rounded bg-muted/50" />
        </div>
      </div>
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
    <div className="space-y-3 pr-3" aria-busy="true" aria-live="polite">
      <div className="flex items-center justify-center gap-2 py-2 text-xs text-muted-foreground">
        <Loader2 className="size-3.5 animate-spin" />
        <span>Connecting to trace buffer…</span>
      </div>
      <SkeletonTraceCard />
      <SkeletonTraceCard />
      <SkeletonTraceCard />
    </div>
  )
})
