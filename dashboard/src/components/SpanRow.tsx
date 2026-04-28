import { Layers } from "lucide-react"
import { cn } from "@/lib/utils"
import type { TraceSpanNode } from "../types"
import { fmtDur } from "../utils/formatting"
import { spanTagValue } from "../utils/traces"

type SpanRowProps = {
  span: TraceSpanNode
  depth: number
  traceStart: number
  traceDuration: number
  traceEntrypoint: string | null
  isLast?: boolean
}

export function SpanRow({ span, depth, traceStart, traceDuration, traceEntrypoint, isLast = false }: SpanRowProps) {
  const offsetPct = traceDuration > 0 ? ((span.start_time_ms - traceStart) / traceDuration) * 100 : 0
  const widthPct = traceDuration > 0 ? Math.max((span.duration_ms / traceDuration) * 100, 1) : 100
  const pipelineName = spanTagValue(span, "hayhooks.pipeline.name")
  const showPipeline = pipelineName && pipelineName !== traceEntrypoint

  return (
    <>
      <div
        className={cn(
          "flex items-start gap-1.5 px-3 py-1.5 text-xs",
          !isLast && "border-b border-border/40",
        )}
        style={{ paddingLeft: `${depth * 16 + 12}px` }}
      >
        <Layers className="mt-0.5 size-3 shrink-0 text-muted-foreground/60" />
        <span className="min-w-0 break-words font-mono text-foreground/85">
          {span.name}
          {showPipeline && (
            <span className="ml-1.5 inline-flex items-center rounded bg-primary/10 px-1.5 py-0.5 font-sans text-[10px] font-medium text-primary">
              {pipelineName}
            </span>
          )}
        </span>
        <span className="ml-auto shrink-0 whitespace-nowrap tabular-nums text-muted-foreground">
          {fmtDur(span.duration_ms)}
        </span>
      </div>
      <div
        className={cn(
          "relative flex items-center px-3 py-1.5",
          !isLast && "border-b border-border/40",
        )}
      >
        <div className="h-full w-full">
          <div
            className="waterfall-bar"
            style={{ marginLeft: `${offsetPct}%`, width: `${widthPct}%` }}
          />
        </div>
      </div>
      {span.children.map((c, i) => (
        <SpanRow
          key={c.span_id}
          span={c}
          depth={depth + 1}
          traceStart={traceStart}
          traceDuration={traceDuration}
          traceEntrypoint={traceEntrypoint}
          isLast={isLast && i === span.children.length - 1}
        />
      ))}
    </>
  )
}
