export type TraceTag = { key: string; value: string }

export type TraceSpanNode = {
  span_id: string
  name: string
  start_time_ms: number
  duration_ms: number
  tags?: TraceTag[]
  children: TraceSpanNode[]
}

export type TraceSummary = {
  trace_id: string
  start_time_ms: number
  duration_ms: number
  entrypoint: string | null
  tags?: TraceTag[]
  span_count: number
  root_span: TraceSpanNode
}

export type DashboardConfigResponse = { poll_ms: number; list_cap: number; fetch_limit: number; fresh_ms: number }

export type SortMode = "newest" | "slowest"
export type TraceKind = "deploy" | "undeploy" | "run" | "openai" | "mcp" | "other"

export type DashboardConfig = {
  pollMs: number
  listCap: number
  fetchLimit: number
  freshMs: number
}
