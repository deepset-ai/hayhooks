import { memo, useMemo } from "react"
import { GitBranch } from "lucide-react"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { cn } from "@/lib/utils"
import type { TraceSummary } from "../types"

type EntrypointsSidebarProps = {
  entrypoints: string[]
  traces: TraceSummary[]
  filter: string | null
  onFilterChange: (filter: string | null) => void
}

export const EntrypointsSidebar = memo(function EntrypointsSidebar({
  entrypoints,
  traces,
  filter,
  onFilterChange,
}: EntrypointsSidebarProps) {
  const traceCounts = useMemo(() => {
    const counts = new Map<string, number>()
    for (const trace of traces) {
      const entrypoint = trace.entrypoint ?? ""
      counts.set(entrypoint, (counts.get(entrypoint) ?? 0) + 1)
    }
    return counts
  }, [traces])

  return (
    <Card className="h-fit lg:sticky lg:top-20">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-sm font-medium">
          <GitBranch className="size-4 text-muted-foreground" />
          Entrypoints
        </CardTitle>
      </CardHeader>
      <CardContent>
        {entrypoints.length === 0 ? (
          <p className="text-xs text-muted-foreground">No deployed pipelines.</p>
        ) : (
          <div className="space-y-0.5">
            <button
              onClick={() => onFilterChange(null)}
              className={cn(
                "flex w-full items-center justify-between rounded-md px-2 py-1.5 text-xs transition-colors",
                filter === null
                  ? "bg-primary/10 font-medium text-primary"
                  : "text-muted-foreground hover:bg-muted hover:text-foreground",
              )}
            >
              <span>All pipelines</span>
              <span className="tabular-nums">{traces.length}</span>
            </button>
            {entrypoints.map((ep) => {
              const isActive = filter === ep
              const count = traceCounts.get(ep) ?? 0
              return (
                <button
                  key={ep}
                  onClick={() => onFilterChange(isActive ? null : ep)}
                  className={cn(
                    "flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-xs transition-colors",
                    isActive ? "bg-primary/10 font-medium text-primary" : "hover:bg-muted",
                  )}
                >
                  <span className={cn("size-1.5 shrink-0 rounded-full", isActive ? "bg-primary" : "bg-haystack-green")} />
                  <span className="min-w-0 truncate font-mono">{ep}</span>
                  <span className={cn("ml-auto shrink-0 tabular-nums", isActive ? "text-primary" : "text-muted-foreground")}>
                    {count}
                  </span>
                </button>
              )
            })}
          </div>
        )}
      </CardContent>
    </Card>
  )
})
