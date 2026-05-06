import { memo } from "react"
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
  slowSpanId: string | null
  selectedSpanId: string
  onSelectSpan: (spanId: string) => void
}

export const SpanRow = memo(function SpanRow({
  span,
  depth,
  traceStart,
  traceDuration,
  traceEntrypoint,
  slowSpanId,
  selectedSpanId,
  onSelectSpan,
}: SpanRowProps) {
  const offsetPct = traceDuration > 0 ? ((span.start_time_ms - traceStart) / traceDuration) * 100 : 0
  const widthPct = traceDuration > 0 ? Math.max((span.duration_ms / traceDuration) * 100, 1) : 100
  const pipelineName = spanTagValue(span, "hayhooks.pipeline.name")
  const showPipeline = pipelineName !== undefined && pipelineName !== traceEntrypoint
  const isSlowComponent = slowSpanId === span.span_id
  const isSelected = span.span_id === selectedSpanId

  return (
    <div className="space-y-1.5">
      <button
        type="button"
        onClick={() => onSelectSpan(span.span_id)}
        aria-pressed={isSelected}
        data-slow-component={isSlowComponent}
        className={cn(
          "flex w-full cursor-pointer items-center gap-2 rounded-md border px-2.5 py-2 text-left text-xs transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
          isSelected
            ? "border-primary/40 bg-primary/10 shadow-sm"
            : isSlowComponent
              ? "border-warning-border bg-warning-soft hover:border-primary/30 hover:bg-warning/15"
              : "border-border/40 hover:border-primary/30 hover:bg-muted",
        )}
      >
        <div className="min-w-0 grow" style={{ paddingLeft: depth * 16 }}>
          <div className="flex items-center gap-1.5">
            <Layers className="size-3 shrink-0 text-muted-foreground/60" />
            <span className="min-w-0 break-words font-mono text-foreground/85">
              {span.name}
            </span>
            {showPipeline && (
              <span className="inline-flex items-center rounded bg-primary/10 px-1.5 py-0.5 font-sans text-[10px] font-medium text-primary">
                {pipelineName}
              </span>
            )}
          </div>
        </div>
        <span className="w-14 shrink-0 text-right whitespace-nowrap tabular-nums text-muted-foreground">
          {fmtDur(span.duration_ms)}
        </span>
        <div className={cn("waterfall-track", isSelected && "waterfall-track-selected")}>
          <div
            className={cn("waterfall-bar", isSelected && "waterfall-bar-selected")}
            style={{ marginLeft: `${offsetPct}%`, width: `${widthPct}%` }}
          />
        </div>
      </button>
      {span.children.map((c) => (
        <SpanRow
          key={c.span_id}
          span={c}
          depth={depth + 1}
          traceStart={traceStart}
          traceDuration={traceDuration}
          traceEntrypoint={traceEntrypoint}
          slowSpanId={slowSpanId}
          selectedSpanId={selectedSpanId}
          onSelectSpan={onSelectSpan}
        />
      ))}
    </div>
  )
})
