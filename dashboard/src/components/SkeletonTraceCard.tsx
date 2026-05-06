import { memo } from "react"

export const SkeletonTraceCard = memo(function SkeletonTraceCard() {
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
