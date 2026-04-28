import { memo, useMemo, useState } from "react"
import { Activity, ChevronRight, Clock, Layers, Tag, Timer } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Separator } from "@/components/ui/separator"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"
import { KIND_STYLE, SUMMARY_TAG_KEYS } from "../constants"
import type { TraceSummary } from "../types"
import { fmtDur, fmtTime, truncate } from "../utils/formatting"
import { isDestructiveTag, sortTags, tagLabel } from "../utils/tags"
import { collectAllSpans, isFailed, isOngoing, traceKind } from "../utils/traces"
import { SpanRow } from "./SpanRow"

type TraceCardProps = {
  trace: TraceSummary
  isFresh: boolean
}

export const TraceCard = memo(function TraceCard({ trace, isFresh }: TraceCardProps) {
  const [open, setOpen] = useState(false)
  const tags = useMemo(() => sortTags(trace.tags ?? []), [trace.tags])
  const summaryTags = useMemo(() => tags.filter((t) => SUMMARY_TAG_KEYS.has(t.key)), [tags])
  const allSpans = useMemo(() => collectAllSpans(trace.root_span), [trace.root_span])
  const failed = useMemo(() => isFailed(trace), [trace])
  const ongoing = useMemo(() => isOngoing(trace), [trace])
  const kind = useMemo(() => traceKind(trace), [trace])
  const kindStyle = KIND_STYLE[kind]

  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <div
        className={cn(
          "rounded-lg border border-l-2 bg-card text-card-foreground transition-shadow",
          failed ? "border-l-destructive" : kindStyle.border || "border-l-transparent",
          ongoing && "trace-card-ongoing",
          isFresh && !ongoing && `trace-card-fresh trace-card-fresh-${kind}`,
          open && "shadow-sm",
        )}
      >
        <CollapsibleTrigger
          className="flex w-full items-start gap-3 px-4 py-3 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          <ChevronRight
            className={cn(
              "mt-0.5 size-4 shrink-0 text-muted-foreground transition-transform duration-200",
              open && "rotate-90",
            )}
          />
          <div className="min-w-0 flex-1 space-y-1.5">
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
              <span className="font-medium text-sm">
                {trace.entrypoint ?? "unknown"}
              </span>
              <span className={cn(
                "inline-flex items-center rounded border px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide",
                kindStyle.badge,
              )}>
                {kindStyle.label}
              </span>
              <span className="flex items-center gap-1 text-xs tabular-nums text-muted-foreground">
                <Timer className="size-3" />
                {fmtDur(trace.duration_ms)}
              </span>
              <span className="flex items-center gap-1 text-xs tabular-nums text-muted-foreground">
                <Layers className="size-3" />
                {trace.span_count} span{trace.span_count !== 1 && "s"}
              </span>
              <span className="flex items-center gap-1 text-xs text-muted-foreground">
                <Clock className="size-3" />
                {fmtTime(trace.start_time_ms)}
              </span>
              {ongoing && (
                <Badge variant="secondary" className="h-5 gap-1 rounded px-1.5 text-[10px] font-semibold text-haystack-blue">
                  <span className="live-dot live-dot-ongoing" />
                  ONGOING
                </Badge>
              )}
              {isFresh && !ongoing && (
                <Badge variant="secondary" className="h-5 gap-1 rounded px-1.5 text-[10px] font-semibold text-haystack-green">
                  <span className="live-dot" />
                  NEW
                </Badge>
              )}
            </div>
            {summaryTags.length > 0 && (
              <div className="flex flex-wrap items-center gap-1.5">
                {summaryTags.map((t) => {
                  const destructive = isDestructiveTag(t)
                  return (
                    <span
                      key={t.key}
                      className={cn(
                        "inline-flex items-center gap-1 rounded border px-1.5 py-0.5 text-[11px]",
                        destructive
                          ? "border-destructive/30 bg-destructive/10 text-destructive"
                          : "bg-muted/60 text-muted-foreground",
                      )}
                    >
                      <span className="font-medium">{tagLabel(t.key)}</span>
                      <span className={destructive ? "text-destructive/80" : "text-foreground/80"}>
                        {truncate(t.value, 28)}
                      </span>
                    </span>
                  )
                })}
              </div>
            )}
          </div>
        </CollapsibleTrigger>

        <CollapsibleContent>
          <Separator />
          <div className="space-y-3 px-4 py-3">
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <span className="font-medium">Trace</span>
              <code className="rounded bg-muted px-1.5 py-0.5 font-mono text-[11px]">
                {trace.trace_id}
              </code>
            </div>

            {tags.length > 0 && (
              <div className="space-y-1.5">
                <div className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
                  <Tag className="size-3" />
                  Tags
                  <span className="text-[10px] tabular-nums">({tags.length})</span>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {tags.map((tag) => (
                    <Tooltip key={tag.key}>
                      <TooltipTrigger className="cursor-default">
                        <span className="inline-flex items-center gap-1 rounded-md bg-muted/70 px-2 py-1 text-[11px] leading-tight">
                          <span className="font-medium text-muted-foreground">{tagLabel(tag.key)}</span>
                          <span className="text-foreground/85 font-mono">{truncate(tag.value, 40)}</span>
                        </span>
                      </TooltipTrigger>
                      <TooltipContent side="top" className="max-w-sm break-all font-mono text-xs">
                        <p className="font-medium">{tag.key}</p>
                        <p className="mt-0.5 text-muted-foreground">{tag.value}</p>
                      </TooltipContent>
                    </Tooltip>
                  ))}
                </div>
              </div>
            )}

            <div className="space-y-1.5">
              <div className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
                <Activity className="size-3" />
                Spans
                <span className="text-[10px] tabular-nums">({allSpans.length})</span>
              </div>
              <div className="overflow-auto">
                <div className="grid grid-cols-[minmax(240px,3fr)_minmax(120px,2fr)] text-xs">
                  <SpanRow
                    span={trace.root_span}
                    depth={0}
                    traceStart={trace.start_time_ms}
                    traceDuration={trace.duration_ms}
                    traceEntrypoint={trace.entrypoint}
                    isLast
                  />
                </div>
              </div>
            </div>
          </div>
        </CollapsibleContent>
      </div>
    </Collapsible>
  )
})
