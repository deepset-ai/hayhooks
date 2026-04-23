import { useEffect, useRef, useState } from "react"
import {
  Activity,
  ChevronRight,
  Clock,
  GitBranch,
  Layers,
  RefreshCw,
  Tag,
  Timer,
  Trash2,
} from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type EntrypointsResponse = { entrypoints: string[] }

type TraceSpanNode = {
  span_id: string
  name: string
  start_time_ms: number
  duration_ms: number
  tags?: TraceTag[]
  children: TraceSpanNode[]
}

type TraceTag = { key: string; value: string }

type TraceSummary = {
  trace_id: string
  start_time_ms: number
  duration_ms: number
  entrypoint: string | null
  tags?: TraceTag[]
  span_count: number
  root_span: TraceSpanNode
}

type TracesResponse = { traces: TraceSummary[] }
type ClearTracesResponse = { ok: boolean; message: string }

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const POLL_MS = 2500
const LIST_CAP = 100
const FETCH_LIMIT = 50
const FRESH_MS = 6000

const TAG_PRIORITY = [
  "hayhooks.pipeline.name",
  "hayhooks.transport",
  "hayhooks.openai.operation",
  "hayhooks.openai.stream_requested",
  "hayhooks.openai.execution_mode",
  "hayhooks.response.stream_type",
  "hayhooks.response.streaming",
  "hayhooks.success",
  "hayhooks.error.type",
  "hayhooks.http.status_code",
  "service.name",
  "serviceName",
]

const TAG_LABELS: Record<string, string> = {
  "hayhooks.pipeline.name": "pipeline",
  "hayhooks.transport": "transport",
  "hayhooks.openai.operation": "openai op",
  "hayhooks.openai.stream_requested": "stream",
  "hayhooks.openai.execution_mode": "exec mode",
  "hayhooks.response.stream_type": "stream type",
  "hayhooks.response.streaming": "streaming",
  "hayhooks.success": "success",
  "hayhooks.error.type": "error",
  "hayhooks.http.status_code": "http",
  "hayhooks.deploy.strategy": "deploy",
  "hayhooks.deploy.save_files": "save files",
  "hayhooks.deploy.file_count": "files",
  "hayhooks.deploy.overwrite": "overwrite",
  "hayhooks.route": "route",
  "hayhooks.payload.keys": "payload keys",
  "hayhooks.payload.has_files": "has files",
  "service.name": "service",
  serviceName: "service",
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function apiBase(): string {
  const env = import.meta.env.VITE_HAYHOOKS_DASHBOARD_API_BASE?.trim()
  if (env) return env.replace(/\/$/, "")
  if (
    window.location.hostname === "localhost" &&
    (window.location.port === "5173" || window.location.port === "4173")
  )
    return "http://localhost:1416/dashboard/api"
  const p = window.location.pathname.replace(/\/index\.html$/, "").replace(/\/$/, "")
  return `${p}/api`
}

function fmtDur(ms: number): string {
  if (ms < 1) return "<1 ms"
  if (ms < 1000) return `${Math.round(ms)} ms`
  return `${(ms / 1000).toFixed(2)}s`
}

function fmtTime(ms: number): string {
  return new Date(ms).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })
}

function safeTags(raw: unknown): TraceTag[] {
  if (!Array.isArray(raw)) return []
  const out: TraceTag[] = []
  for (const t of raw) {
    if (!t || typeof t !== "object") continue
    const k = "key" in t ? t.key : null
    const v = "value" in t ? t.value : null
    if (typeof k === "string" && k && typeof v === "string" && v) out.push({ key: k, value: v })
  }
  return out
}

function sortTags(tags: TraceTag[]): TraceTag[] {
  const dedup = new Map<string, TraceTag>()
  for (const t of tags) if (!dedup.has(t.key)) dedup.set(t.key, t)
  return [...dedup.values()].sort((a, b) => {
    const ai = TAG_PRIORITY.indexOf(a.key)
    const bi = TAG_PRIORITY.indexOf(b.key)
    const an = ai === -1 ? 9999 : ai
    const bn = bi === -1 ? 9999 : bi
    return an !== bn ? an - bn : a.key.localeCompare(b.key)
  })
}

function merge(existing: TraceSummary[], incoming: TraceSummary[]): TraceSummary[] {
  const m = new Map(existing.map((t) => [t.trace_id, { ...t, tags: safeTags(t.tags) }]))
  for (const t of incoming) m.set(t.trace_id, { ...t, tags: safeTags(t.tags) })
  return [...m.values()].sort((a, b) => b.start_time_ms - a.start_time_ms).slice(0, LIST_CAP)
}

function tagLabel(key: string): string {
  return TAG_LABELS[key] ?? key.replace(/^hayhooks\./, "")
}

function truncate(s: string, n: number): string {
  return s.length <= n ? s : `${s.slice(0, n - 1)}…`
}

function collectAllSpans(node: TraceSpanNode): TraceSpanNode[] {
  return [node, ...node.children.flatMap(collectAllSpans)]
}

function spanTagValue(span: TraceSpanNode, key: string): string | undefined {
  return safeTags(span.tags).find((t) => t.key === key)?.value
}

// ---------------------------------------------------------------------------
// Span waterfall row
// ---------------------------------------------------------------------------

function SpanRow({
  span,
  depth,
  traceStart,
  traceDuration,
  traceEntrypoint,
  isLast = false,
}: {
  span: TraceSpanNode
  depth: number
  traceStart: number
  traceDuration: number
  traceEntrypoint: string | null
  isLast?: boolean
}) {
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

// ---------------------------------------------------------------------------
// Single trace card
// ---------------------------------------------------------------------------

function TraceCard({
  trace,
  isFresh,
}: {
  trace: TraceSummary
  isFresh: boolean
}) {
  const [open, setOpen] = useState(false)
  const tags = sortTags(safeTags(trace.tags))
  const summaryKeys = new Set([
    "hayhooks.transport",
    "hayhooks.success",
    "hayhooks.error.type",
  ])
  const summaryTags = tags.filter((t) => summaryKeys.has(t.key))
  const allSpans = collectAllSpans(trace.root_span)

  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <div
        className={cn(
          "rounded-lg border bg-card text-card-foreground transition-shadow",
          isFresh && "trace-card-fresh",
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
                {isFresh && (
                  <Badge variant="secondary" className="h-5 gap-1 rounded px-1.5 text-[10px] font-semibold text-haystack-green">
                    <span className="live-dot" />
                    NEW
                  </Badge>
                )}
              </div>
              {summaryTags.length > 0 && (
                <div className="flex flex-wrap items-center gap-1.5">
                  {summaryTags.map((t) => (
                    <span
                      key={t.key}
                      className={cn(
                        "inline-flex items-center gap-1 rounded border px-1.5 py-0.5 text-[11px]",
                        t.key === "hayhooks.error.type"
                          ? "border-destructive/30 bg-destructive/10 text-destructive"
                          : t.key === "hayhooks.success" && t.value === "false"
                            ? "border-destructive/30 bg-destructive/10 text-destructive"
                            : "bg-muted/60 text-muted-foreground",
                      )}
                    >
                      <span className="font-medium">{tagLabel(t.key)}</span>
                      <span className={cn(
                        t.key === "hayhooks.error.type" || (t.key === "hayhooks.success" && t.value === "false")
                          ? "text-destructive/80"
                          : "text-foreground/80",
                      )}>{truncate(t.value, 28)}</span>
                    </span>
                  ))}
                </div>
              )}
            </div>
        </CollapsibleTrigger>

        <CollapsibleContent>
          <Separator />
          <div className="space-y-3 px-4 py-3">
            {/* Trace ID */}
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <span className="font-medium">Trace</span>
              <code className="rounded bg-muted px-1.5 py-0.5 font-mono text-[11px]">
                {trace.trace_id}
              </code>
            </div>

            {/* Tags as flat chips */}
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

            {/* Waterfall */}
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
                  />
                </div>
              </div>
            </div>
          </div>
        </CollapsibleContent>
      </div>
    </Collapsible>
  )
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

function App() {
  const [entrypoints, setEntrypoints] = useState<string[]>([])
  const [traces, setTraces] = useState<TraceSummary[]>([])
  const [filter, setFilter] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [refreshing, setRefreshing] = useState(false)
  const [clearing, setClearing] = useState(false)
  const [updatedAt, setUpdatedAt] = useState<number | null>(null)
  const [freshUntil, setFreshUntil] = useState<Record<string, number>>({})
  const sinceRef = useRef<number | null>(null)
  const seenRef = useRef(new Set<string>())
  const base = useRef(apiBase())

  const refresh = async () => {
    setRefreshing(true)
    try {
      const ep = await fetch(`${base.current}/entrypoints`)
      if (!ep.ok) throw new Error(`Entrypoints ${ep.status}`)
      setEntrypoints(((await ep.json()) as EntrypointsResponse).entrypoints)

      const params = new URLSearchParams({ limit: String(FETCH_LIMIT) })
      if (sinceRef.current !== null) params.set("since_ms", String(sinceRef.current))
      const tr = await fetch(`${base.current}/traces?${params}`)
      if (!tr.ok) throw new Error(`Traces ${tr.status}`)
      const data = ((await tr.json()) as TracesResponse).traces
      const cutoff = sinceRef.current
      const incoming = cutoff === null ? data : data.filter((t) => t.start_time_ms >= cutoff)
      const now = Date.now()
      const ids = incoming.map((t) => t.trace_id)
      const fresh = ids.filter((id) => !seenRef.current.has(id))
      for (const id of ids) seenRef.current.add(id)
      setFreshUntil((cur) => {
        const next: Record<string, number> = {}
        for (const [k, v] of Object.entries(cur)) if (v > now) next[k] = v
        for (const id of fresh) next[id] = now + FRESH_MS
        return next
      })
      setTraces((cur) => merge(cur, incoming))
      const newest = incoming.reduce((m, t) => Math.max(m, t.start_time_ms), sinceRef.current ?? 0)
      if (newest > 0) sinceRef.current = newest
      setUpdatedAt(Date.now())
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error")
    } finally {
      setRefreshing(false)
    }
  }

  const clear = async () => {
    const now = Date.now()
    setTraces([])
    setFreshUntil({})
    seenRef.current = new Set()
    sinceRef.current = now
    setUpdatedAt(now)
    setError(null)
    setClearing(true)
    try {
      const r = await fetch(`${base.current}/traces/clear`, { method: "POST" })
      if (!r.ok) throw new Error(`Clear failed ${r.status}`)
      const body = (await r.json()) as ClearTracesResponse
      if (!body.ok) throw new Error("Clear failed")
    } catch (e) {
      setError(e instanceof Error ? e.message : "Clear failed")
    } finally {
      setClearing(false)
    }
  }

  useEffect(() => {
    const t0 = setTimeout(() => void refresh(), 0)
    const iv = setInterval(() => void refresh(), POLL_MS)
    return () => {
      clearTimeout(t0)
      clearInterval(iv)
    }
  }, [])

  const now = Date.now()
  const filteredTraces = filter ? traces.filter((t) => t.entrypoint === filter) : traces
  const traceCounts = new Map<string, number>()
  for (const t of traces) {
    const ep = t.entrypoint ?? ""
    traceCounts.set(ep, (traceCounts.get(ep) ?? 0) + 1)
  }

  return (
    <TooltipProvider delay={200}>
      <div className="flex min-h-screen flex-col bg-background">
        {/* ── Header ───────────────────────────────────────────────── */}
        <header className="sticky top-0 z-30 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/80">
          <div className="mx-auto flex h-14 max-w-7xl items-center justify-between gap-4 px-6">
            <div className="flex items-center gap-3">
              <div className="flex size-8 items-center justify-center rounded-lg bg-primary text-primary-foreground">
                <Activity className="size-4" />
              </div>
              <div>
                <h1 className="text-sm font-semibold leading-none">Hayhooks</h1>
                <p className="mt-0.5 text-xs text-muted-foreground">Trace Dashboard</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              {updatedAt !== null && (
                <span className="hidden items-center gap-1.5 text-xs text-muted-foreground sm:flex">
                  <span className="live-dot" />
                  Updated {fmtTime(updatedAt)}
                </span>
              )}
              {error !== null && (
                <Badge variant="destructive" className="text-xs">{error}</Badge>
              )}
              <Separator orientation="vertical" className="mx-1 h-6" />
              <Button
                variant="ghost"
                size="sm"
                onClick={() => void clear()}
                disabled={refreshing || clearing}
                className="gap-1.5 text-xs"
              >
                <Trash2 className="size-3.5" />
                Clear
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => void refresh()}
                disabled={refreshing || clearing}
                className="gap-1.5 text-xs"
              >
                <RefreshCw className={cn("size-3.5", refreshing && "animate-spin")} />
                Refresh
              </Button>
            </div>
          </div>
        </header>

        {/* ── Main ─────────────────────────────────────────────────── */}
        <main className="mx-auto w-full max-w-7xl flex-1 space-y-6 px-6 py-6">
          {/* Stat strip */}
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <StatCard label="Entrypoints" value={entrypoints.length} icon={<GitBranch className="size-4" />} />
            <StatCard label="Traces" value={traces.length} icon={<Layers className="size-4" />} />
            <StatCard
              label="Avg duration"
              value={
                traces.length > 0
                  ? fmtDur(traces.reduce((s, t) => s + t.duration_ms, 0) / traces.length)
                  : "—"
              }
              icon={<Timer className="size-4" />}
            />
            <StatCard
              label="Last trace"
              value={traces.length > 0 ? fmtTime(traces[0].start_time_ms) : "—"}
              icon={<Clock className="size-4" />}
            />
          </div>

          <div className="grid gap-6 lg:grid-cols-[260px_1fr]">
            {/* Entrypoints sidebar */}
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
                      onClick={() => setFilter(null)}
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
                          onClick={() => setFilter(isActive ? null : ep)}
                          className={cn(
                            "flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-xs transition-colors",
                            isActive
                              ? "bg-primary/10 font-medium text-primary"
                              : "hover:bg-muted",
                          )}
                        >
                          <span className={cn(
                            "size-1.5 shrink-0 rounded-full",
                            isActive ? "bg-primary" : "bg-haystack-green",
                          )} />
                          <span className="min-w-0 truncate font-mono">{ep}</span>
                          <span className={cn(
                            "ml-auto shrink-0 tabular-nums",
                            isActive ? "text-primary" : "text-muted-foreground",
                          )}>
                            {count}
                          </span>
                        </button>
                      )
                    })}
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Trace list */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <h2 className="text-sm font-medium">Live Traces</h2>
                  {filter && (
                    <Badge variant="secondary" className="gap-1 rounded px-1.5 text-[11px]">
                      {filter}
                      <button
                        onClick={() => setFilter(null)}
                        className="ml-0.5 rounded-sm opacity-60 hover:opacity-100"
                        aria-label="Clear filter"
                      >
                        ×
                      </button>
                    </Badge>
                  )}
                </div>
                {filteredTraces.length > 0 && (
                  <span className="text-xs tabular-nums text-muted-foreground">
                    {filter
                      ? `${filteredTraces.length} of ${traces.length} trace${traces.length !== 1 ? "s" : ""}`
                      : `Showing ${traces.length} trace${traces.length !== 1 ? "s" : ""}`}
                  </span>
                )}
              </div>
              <ScrollArea className="h-[calc(100vh-240px)] min-h-[400px]">
                {filteredTraces.length === 0 ? (
                  <div className="flex flex-col items-center justify-center gap-2 py-20 text-center text-muted-foreground">
                    <Activity className="size-8 opacity-40" />
                    {filter ? (
                      <>
                        <p className="text-sm">No traces for "{filter}"</p>
                        <button
                          onClick={() => setFilter(null)}
                          className="text-xs text-primary hover:underline"
                        >
                          Show all traces
                        </button>
                      </>
                    ) : (
                      <>
                        <p className="text-sm">Waiting for traces…</p>
                        <p className="text-xs">Traces will appear here as Hayhooks processes requests.</p>
                      </>
                    )}
                  </div>
                ) : (
                  <div className="space-y-2 pr-3">
                    {filteredTraces.map((t) => (
                      <TraceCard
                        key={t.trace_id}
                        trace={t}
                        isFresh={(freshUntil[t.trace_id] ?? 0) > now}
                      />
                    ))}
                  </div>
                )}
              </ScrollArea>
            </div>
          </div>
        </main>
      </div>
    </TooltipProvider>
  )
}

// ---------------------------------------------------------------------------
// Stat card
// ---------------------------------------------------------------------------

function StatCard({
  label,
  value,
  icon,
}: {
  label: string
  value: string | number
  icon: React.ReactNode
}) {
  return (
    <Card className="shadow-none">
      <CardContent className="flex items-center gap-3 py-3">
        <div className="flex size-9 items-center justify-center rounded-md bg-muted text-muted-foreground">
          {icon}
        </div>
        <div>
          <p className="text-xs text-muted-foreground">{label}</p>
          <p className="text-lg font-semibold tabular-nums leading-tight">{value}</p>
        </div>
      </CardContent>
    </Card>
  )
}

export default App
