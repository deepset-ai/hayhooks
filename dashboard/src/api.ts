import { DEFAULT_DASHBOARD_CONFIG } from "./constants"
import type {
  DashboardConfig,
  DashboardConfigResponse,
  TraceTag,
  TraceSummary,
} from "./types"

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null
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

export function normalizeDashboardConfig(raw: unknown): DashboardConfig {
  if (!raw || typeof raw !== "object") return DEFAULT_DASHBOARD_CONFIG

  const config = raw as Partial<DashboardConfigResponse>
  const listCap =
    typeof config.list_cap === "number" && Number.isFinite(config.list_cap) && config.list_cap > 0
      ? Math.round(config.list_cap)
      : DEFAULT_DASHBOARD_CONFIG.listCap
  const fetchLimitCandidate =
    typeof config.fetch_limit === "number" && Number.isFinite(config.fetch_limit) && config.fetch_limit > 0
      ? Math.round(config.fetch_limit)
      : DEFAULT_DASHBOARD_CONFIG.fetchLimit
  const pollMs =
    typeof config.poll_ms === "number" && Number.isFinite(config.poll_ms) && config.poll_ms >= 250
      ? Math.round(config.poll_ms)
      : DEFAULT_DASHBOARD_CONFIG.pollMs
  const freshMs =
    typeof config.fresh_ms === "number" && Number.isFinite(config.fresh_ms) && config.fresh_ms >= 0
      ? Math.round(config.fresh_ms)
      : DEFAULT_DASHBOARD_CONFIG.freshMs

  return {
    pollMs,
    listCap,
    fetchLimit: Math.min(fetchLimitCandidate, listCap),
    freshMs,
  }
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

export async function fetchTraces(base: string, limit: number, sinceMs?: number): Promise<TraceSummary[]> {
  const params = new URLSearchParams({ limit: String(limit) })
  if (sinceMs != null) params.set("since_ms", String(sinceMs))

  const response = await fetch(`${base}/traces?${params}`)
  if (!response.ok) throw new Error(`Traces ${response.status}`)

  const payload = await response.json()
  if (!isRecord(payload) || !Array.isArray(payload.traces) || payload.traces.some((trace) => !isTraceSummary(trace))) {
    throw new Error("Traces payload invalid")
  }

  return payload.traces
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
