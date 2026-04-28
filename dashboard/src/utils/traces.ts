import type { TraceKind, TraceSpanNode, TraceSummary } from "../types"
import { isDestructiveTag, safeTags } from "./tags"

const SUCCESS_TAG_KEY = "hayhooks.success"
const TRACE_KIND_RULES: Array<{ includes: string; kind: TraceKind }> = [
  { includes: ".undeploy", kind: "undeploy" },
  { includes: ".deploy", kind: "deploy" },
  { includes: ".openai.", kind: "openai" },
  { includes: ".mcp.", kind: "mcp" },
  { includes: ".run", kind: "run" },
]

function normalizeTrace(trace: TraceSummary): TraceSummary {
  return { ...trace, tags: safeTags(trace.tags) }
}

function compareByStartTimeDesc(left: TraceSummary, right: TraceSummary): number {
  return right.start_time_ms - left.start_time_ms
}

export function mergeTraces(existing: TraceSummary[], incoming: TraceSummary[], listCap: number): TraceSummary[] {
  const tracesById = new Map(existing.map((trace) => [trace.trace_id, trace]))
  for (const trace of incoming) {
    tracesById.set(trace.trace_id, normalizeTrace(trace))
  }
  return [...tracesById.values()].sort(compareByStartTimeDesc).slice(0, listCap)
}

export function filterTracesByEntrypoint(traces: TraceSummary[], filter: string | null): TraceSummary[] {
  if (filter === null) return traces
  return traces.filter((trace) => trace.entrypoint === filter)
}

export function collectAllSpans(node: TraceSpanNode): TraceSpanNode[] {
  return [node, ...node.children.flatMap(collectAllSpans)]
}

export function isFailed(trace: TraceSummary): boolean {
  return (trace.tags ?? []).some(isDestructiveTag)
}

export function isOngoing(trace: TraceSummary): boolean {
  if (isFailed(trace)) return false
  const successTag = (trace.tags ?? []).find((tag) => tag.key === SUCCESS_TAG_KEY)
  if (successTag?.value === "true" || successTag?.value === "false") return false
  return trace.root_span.duration_ms === 0
}

export function traceKind(trace: TraceSummary): TraceKind {
  const name = trace.root_span?.name ?? ""
  for (const rule of TRACE_KIND_RULES) {
    if (name.includes(rule.includes)) return rule.kind
  }
  return "other"
}

export function spanTagValue(span: TraceSpanNode, key: string): string | undefined {
  return (span.tags ?? []).find((t) => t.key === key)?.value
}
