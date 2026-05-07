import { memo } from "react"
import { Activity, GitBranch, Loader2 } from "lucide-react"

import { useTraceStatus } from "../hooks/useTracesContext"
import { SkeletonTraceCard } from "./SkeletonTraceCard"

type TraceListEmptyStateProps = {
  filter: string | null
  hasTraces: boolean
  onClearFilter: () => void
}

export const TraceListEmptyState = memo(function TraceListEmptyState({
  filter,
  hasTraces,
  onClearFilter,
}: TraceListEmptyStateProps) {
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
