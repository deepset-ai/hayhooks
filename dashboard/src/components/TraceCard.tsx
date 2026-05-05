import { memo, useEffect, useMemo, useRef, useState } from "react"
import { Activity, AlertTriangle, Check, ChevronDown, ChevronRight, Clock, Copy, Layers, Tag, Timer } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Separator } from "@/components/ui/separator"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"
import { KIND_STYLE, SUMMARY_TAG_KEYS, DEFAULT_DASHBOARD_CONFIG } from "../constants"
import type { TraceSummary } from "../types"
import { fmtDur, fmtTime, truncate } from "../utils/formatting"
import { isDestructiveTag, sortTags, tagLabel, ERROR_TYPE_TAG_KEY, ERROR_MESSAGE_TAG_KEY, ERROR_STACK_TAG_KEY } from "../utils/tags"
import { collectAllSpans, isFailed, isOngoing, slowestComponentRun, spanTagValue, traceKind } from "../utils/traces"
import { SpanRow } from "./SpanRow"

type TraceCardProps = {
  trace: TraceSummary
  isFresh: boolean
  slowComponentMinDurationMs?: number
}

export const TraceCard = memo(function TraceCard({
  trace,
  isFresh,
  slowComponentMinDurationMs = DEFAULT_DASHBOARD_CONFIG.slowComponentMinDurationMs,
}: TraceCardProps) {
  const [open, setOpen] = useState(false)
  const [selectedSpanId, setSelectedSpanId] = useState(trace.root_span.span_id)
  const [stackExpanded, setStackExpanded] = useState(false)
  const [copied, setCopied] = useState(false)
  const copyTimerRef = useRef<ReturnType<typeof setTimeout>>(null)

  useEffect(() => {
    return () => {
      if (copyTimerRef.current !== null) clearTimeout(copyTimerRef.current)
    }
  }, [])
  const summaryTags = useMemo(
    () => sortTags((trace.tags ?? []).filter((tag) => SUMMARY_TAG_KEYS.has(tag.key))),
    [trace.tags],
  )
  const allSpans = useMemo(() => collectAllSpans(trace.root_span), [trace.root_span])
  const selectedSpan = useMemo(
    () => allSpans.find((span) => span.span_id === selectedSpanId) ?? trace.root_span,
    [allSpans, selectedSpanId, trace.root_span],
  )
  const selectedSpanTags = useMemo(() => sortTags(selectedSpan.tags ?? []), [selectedSpan.tags])
  const slowestRun = useMemo(() => slowestComponentRun(allSpans), [allSpans])
  const highlightedSlowestComponent = useMemo(
    () => (slowestRun !== null && slowestRun.durationMs > slowComponentMinDurationMs ? slowestRun : null),
    [slowestRun, slowComponentMinDurationMs],
  )
  const failed = isFailed(trace)
  const ongoing = isOngoing(trace)
  const kind = traceKind(trace)
  const kindStyle = KIND_STYLE[kind]
  const freshHighlightTone = failed ? "failed" : kind

  const errorType = useMemo(
    () => (failed ? spanTagValue(trace.root_span, ERROR_TYPE_TAG_KEY) : undefined),
    [failed, trace.root_span],
  )
  const errorMessage = useMemo(() => {
    if (!failed) return undefined
    const tag = (trace.tags ?? []).find((t) => t.key === ERROR_MESSAGE_TAG_KEY)
      ?? (trace.root_span.tags ?? []).find((t) => t.key === ERROR_MESSAGE_TAG_KEY)
    return tag?.value
  }, [failed, trace.tags, trace.root_span])
  const errorStack = useMemo(() => {
    if (!failed) return undefined
    const tag = (trace.tags ?? []).find((t) => t.key === ERROR_STACK_TAG_KEY)
      ?? (trace.root_span.tags ?? []).find((t) => t.key === ERROR_STACK_TAG_KEY)
    return tag?.value
  }, [failed, trace.tags, trace.root_span])

  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <div
        className={cn(
          "rounded-lg border border-l-2 bg-card text-card-foreground transition-shadow",
          failed ? "border-l-destructive" : kindStyle.border || "border-l-transparent",
          ongoing && "trace-card-ongoing",
          isFresh && !ongoing && `trace-card-fresh trace-card-fresh-${freshHighlightTone}`,
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

          {failed && (errorType || errorMessage) && (
            <div className="mx-4 mt-3 space-y-2 rounded-md border border-destructive/25 bg-destructive/5 px-3 py-2">
              <div className="flex items-start gap-2">
                <AlertTriangle className="mt-0.5 size-3.5 shrink-0 text-destructive" />
                <div className="min-w-0 flex-1 text-xs">
                  {errorType && (
                    <span className="font-semibold text-destructive">{errorType}</span>
                  )}
                  {errorType && errorMessage && <span className="text-destructive/70">: </span>}
                  {errorMessage && (
                    <span className="text-destructive/80">{errorMessage}</span>
                  )}
                </div>
                {errorStack && (
                  <button
                    type="button"
                    className="shrink-0 rounded p-1 text-destructive/40 hover:text-destructive/80 hover:bg-destructive/10 transition-colors"
                    onClick={async () => {
                      try {
                        await navigator.clipboard.writeText(errorStack)
                        setCopied(true)
                        if (copyTimerRef.current !== null) clearTimeout(copyTimerRef.current)
                        copyTimerRef.current = setTimeout(() => setCopied(false), 1500)
                      } catch {
                        // clipboard write denied — ignore silently
                      }
                    }}
                    aria-label="Copy stack trace"
                  >
                    {copied
                      ? <Check className="size-3" />
                      : <Copy className="size-3" />}
                  </button>
                )}
              </div>
              {errorStack && (
                <button
                  type="button"
                  onClick={() => setStackExpanded((prev) => !prev)}
                  className="flex items-center gap-1 text-[11px] text-destructive/60 hover:text-destructive/90 transition-colors"
                >
                  {stackExpanded ? (
                    <ChevronDown className="size-3" />
                  ) : (
                    <ChevronRight className="size-3" />
                  )}
                  {stackExpanded ? "Hide stack trace" : "Show stack trace"}
                </button>
              )}
              {errorStack && stackExpanded && (
                <pre className="overflow-auto rounded bg-destructive/10 px-2.5 py-2 font-mono text-[10px] leading-relaxed text-destructive/80 whitespace-pre-wrap max-h-64">
                  {errorStack}
                </pre>
              )}
            </div>
          )}

          <div className="space-y-3 px-4 py-3">
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <span className="font-medium">Trace</span>
              <code className="rounded bg-muted px-1.5 py-0.5 font-mono text-[11px]">
                {trace.trace_id}
              </code>
            </div>
            <div className="space-y-1.5">
              <div className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
                <Tag className="size-3" />
                Span tags
                <span className="text-[10px] text-muted-foreground/80">
                  ({selectedSpan.name})
                </span>
                <span className="text-[10px] tabular-nums">({selectedSpanTags.length})</span>
              </div>
              {selectedSpanTags.length > 0 ? (
                <div className="flex flex-wrap gap-1.5">
                  {selectedSpanTags.map((tag) => (
                    <Tooltip key={tag.key}>
                      <TooltipTrigger className="cursor-default">
                        <span className="inline-flex max-w-full items-start gap-1 rounded-md bg-muted/70 px-2 py-1 text-[11px] leading-tight">
                          <span className="shrink-0 font-medium text-muted-foreground">{tagLabel(tag.key)}</span>
                          <span className="break-all text-foreground/85 font-mono">{truncate(tag.value, 40)}</span>
                        </span>
                      </TooltipTrigger>
                      <TooltipContent side="top" className="!block max-w-xl p-3 font-mono text-xs leading-relaxed sm:max-w-2xl">
                        <div className="space-y-2">
                          <p className="break-all text-background/95">{tag.key}</p>
                          <p className="break-all text-background/75">{tag.value}</p>
                        </div>
                      </TooltipContent>
                    </Tooltip>
                  ))}
                </div>
              ) : (
                <p className="text-xs text-muted-foreground">No tags for this span.</p>
              )}
            </div>

            <div className="space-y-1.5">
              <div className="flex flex-wrap items-center gap-1.5 text-xs font-medium text-muted-foreground">
                <Activity className="size-3" />
                <span>Spans</span>
                <span className="text-[10px] tabular-nums">({allSpans.length})</span>
                {highlightedSlowestComponent !== null && (
                  <span className="inline-flex items-center gap-1 rounded border border-amber-400/40 bg-amber-500/10 px-2 py-0.5 text-[11px]">
                    <Timer className="size-3 text-amber-700/90 dark:text-amber-200/90" />
                    <span className="text-muted-foreground">Slowest component</span>
                    <span className="font-medium text-amber-800 dark:text-amber-300">
                      {truncate(highlightedSlowestComponent.componentName, 24)}
                    </span>
                    <span className="tabular-nums text-amber-700/90 dark:text-amber-200/90">
                      {fmtDur(highlightedSlowestComponent.durationMs)}
                    </span>
                  </span>
                )}
              </div>
              <div className="overflow-auto">
                <div className="space-y-1.5 pr-1">
                  <SpanRow
                    span={trace.root_span}
                    depth={0}
                    traceStart={trace.start_time_ms}
                    traceDuration={trace.duration_ms}
                    traceEntrypoint={trace.entrypoint}
                    slowSpanId={highlightedSlowestComponent?.spanId ?? null}
                    selectedSpanId={selectedSpanId}
                    onSelectSpan={setSelectedSpanId}
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
