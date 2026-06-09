import type { TraceTag, TraceSummary } from "./types"
import type { DashboardConfig } from "./types"
import { normalizeDashboardConfig } from "./utils/config"
import { isRecord } from "./utils/tags"

export type FetchTracesResult = {
  traces: TraceSummary[]
  nextAfterSeq: number | null
  hasMore: boolean
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value)
}

function isStringOrNull(value: unknown): value is string | null {
  return value === null || typeof value === "string"
}

function isTraceTag(value: unknown): value is TraceTag {
  return (
    isRecord(value) &&
    typeof value.key === "string" &&
    typeof value.value === "string"
  )
}

function isOptionalTraceTags(value: unknown): value is TraceTag[] | undefined {
  return value === undefined || (Array.isArray(value) && value.every(isTraceTag))
}

function isTraceSpanNode(value: unknown): value is TraceSummary["root_span"] {
  if (!isRecord(value)) return false

  if (
    typeof value.span_id !== "string" ||
    typeof value.name !== "string" ||
    !isFiniteNumber(value.start_time_ms) ||
    !isFiniteNumber(value.duration_ms) ||
    (value.running !== undefined && typeof value.running !== "boolean") ||
    !Array.isArray(value.children) ||
    !isOptionalTraceTags(value.tags)
  ) {
    return false
  }

  return value.children.every(isTraceSpanNode)
}

function isTraceSummary(value: unknown): value is TraceSummary {
  if (!isRecord(value)) return false

  const hasCoreFields =
    typeof value.trace_id === "string" &&
    isFiniteNumber(value.start_time_ms) &&
    isFiniteNumber(value.duration_ms) &&
    isStringOrNull(value.entrypoint) &&
    isFiniteNumber(value.span_count) &&
    isTraceSpanNode(value.root_span)

  return hasCoreFields && isOptionalTraceTags(value.tags)
}

export function resolveApiBase(): string {
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

export async function fetchDashboardConfig(base: string): Promise<DashboardConfig> {
  const response = await fetch(`${base}/config`)
  if (!response.ok) throw new Error(`Config ${response.status}`)
  return normalizeDashboardConfig(await response.json())
}

export async function fetchEntrypoints(base: string): Promise<string[]> {
  const response = await fetch(`${base}/entrypoints`)
  if (!response.ok) throw new Error(`Entrypoints ${response.status}`)

  const payload = await response.json()
  if (
    !isRecord(payload) ||
    !Array.isArray(payload.entrypoints) ||
    payload.entrypoints.some((entrypoint) => typeof entrypoint !== "string")
  ) {
    throw new Error("Entrypoints payload invalid")
  }

  return payload.entrypoints
}

function parseTraceCursor(headerValue: string | null): number | null {
  if (headerValue === null) return null
  const parsed = Number.parseInt(headerValue, 10)
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : null
}

/**
 * Validate and normalize a traces payload (the `{ traces, next_after_seq, has_more }`
 * shape returned by both `GET /traces` and the SSE stream's `snapshot`/`trace`
 * events). When the body omits `next_after_seq`, falls back to the optional
 * response-header cursor.
 */
export function parseTracesPayload(payload: unknown, headerCursor?: string | null): FetchTracesResult {
  if (!isRecord(payload) || !Array.isArray(payload.traces) || payload.traces.some((trace) => !isTraceSummary(trace))) {
    throw new Error("Traces payload invalid")
  }

  const nextAfterSeqFromBody = typeof payload.next_after_seq === "number" && Number.isFinite(payload.next_after_seq)
    ? payload.next_after_seq as number
    : null
  const hasMore = typeof payload.has_more === "boolean" ? payload.has_more : false
  const nextAfterSeq = nextAfterSeqFromBody ?? (headerCursor !== undefined ? parseTraceCursor(headerCursor) : null)

  return {
    traces: payload.traces,
    nextAfterSeq,
    hasMore,
  }
}

export async function fetchTraces(
  base: string,
  limit: number,
  sinceMs?: number,
  afterSeq?: number,
): Promise<FetchTracesResult> {
  const params = new URLSearchParams({ limit: String(limit) })
  if (sinceMs != null) params.set("since_ms", String(sinceMs))
  if (afterSeq != null) params.set("after_seq", String(afterSeq))

  const response = await fetch(`${base}/traces?${params}`)
  if (!response.ok) throw new Error(`Traces ${response.status}`)

  const payload = await response.json()
  return parseTracesPayload(payload, response.headers.get("X-Hayhooks-Trace-Cursor"))
}

export function traceStreamUrl(base: string, afterSeq?: number | null): string {
  const params = new URLSearchParams()
  if (afterSeq != null) params.set("after_seq", String(afterSeq))
  const query = params.toString()
  return query ? `${base}/traces/stream?${query}` : `${base}/traces/stream`
}

export async function clearTraces(base: string): Promise<void> {
  const response = await fetch(`${base}/traces/clear`, { method: "POST" })
  if (!response.ok) throw new Error(`Clear failed ${response.status}`)

  const bodyText = await response.text()
  if (bodyText.trim() === "") return

  let body: unknown
  try {
    body = JSON.parse(bodyText)
  } catch {
    throw new Error("Clear failed")
  }
  if (!isRecord(body) || typeof body.ok !== "boolean" || !body.ok) {
    throw new Error("Clear failed")
  }
}
