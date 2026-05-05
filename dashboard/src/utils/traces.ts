import type { TraceKind, TraceSpanNode, TraceSummary } from "../types"
import { isDestructiveTag, safeTags, SUCCESS_TAG_KEY } from "./tags"

const HAYSTACK_COMPONENT_NAME_TAG_KEY = "haystack.component.name"
const HAYSTACK_COMPONENT_RUN_SPAN_NAME = "haystack.component.run"
const TRACE_KIND_RULES: Array<{ includes: string; kind: TraceKind }> = [
  { includes: ".undeploy", kind: "undeploy" },
  { includes: ".deploy", kind: "deploy" },
  { includes: ".openai.", kind: "openai" },
  { includes: ".mcp.", kind: "mcp" },
  { includes: ".run", kind: "run" },
]

export type SlowComponentRun = {
  spanId: string
  componentName: string
  durationMs: number
}

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

export function slowestComponentRun(spans: TraceSpanNode[]): SlowComponentRun | null {
  let currentSlowest: SlowComponentRun | null = null
  for (const span of spans) {
    if (span.name !== HAYSTACK_COMPONENT_RUN_SPAN_NAME) {
      continue
    }
    const componentTag = spanTagValue(span, HAYSTACK_COMPONENT_NAME_TAG_KEY)
    const componentName = componentTag ?? span.span_id
    const candidate: SlowComponentRun = {
      spanId: span.span_id,
      componentName,
      durationMs: span.duration_ms,
    }
    if (
      currentSlowest === null
      || candidate.durationMs > currentSlowest.durationMs
      || (candidate.durationMs === currentSlowest.durationMs
        && candidate.componentName.localeCompare(currentSlowest.componentName) < 0)
    ) {
      currentSlowest = candidate
    }
  }
  return currentSlowest
}
