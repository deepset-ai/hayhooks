import { memo, useMemo, useState } from "react"
import { ArrowDownWideNarrow } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { useTraceData } from "../hooks/useTracesContext"
import type { SortMode, TraceSummary } from "../types"
import { TraceCards } from "./TraceCards"
import { TraceListEmptyState } from "./TraceListEmptyState"

type TraceListProps = {
  traces: TraceSummary[]
  totalTraces: number
  filter: string | null
  slowComponentMinDurationMs?: number
  onClearFilter: () => void
}

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
