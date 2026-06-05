import {
  collectAllSpans,
  filterTracesByEntrypoint,
  isFailed,
  isOngoing,
  mergeTraces,
  slowestComponentRun,
  spanTagValue,
  traceKind,
} from "./traces"
import type { TraceSpanNode, TraceSummary } from "../types"

function makeSpan(overrides: Partial<TraceSpanNode> = {}): TraceSpanNode {
  return {
    span_id: "span-1",
    name: "test-span",
    start_time_ms: 1000,
    duration_ms: 100,
    children: [],
    ...overrides,
  }
}

function makeTrace(overrides: Partial<TraceSummary> = {}): TraceSummary {
  return {
    trace_id: "trace-1",
    start_time_ms: 1000,
    duration_ms: 200,
    entrypoint: "my_pipeline",
    span_count: 1,
    root_span: makeSpan(),
    tags: [],
    ...overrides,
  }
}

describe("mergeTraces", () => {
  it("merges incoming traces into existing, sorted newest-first", () => {
    const existing = [makeTrace({ trace_id: "a", start_time_ms: 100 })]
    const incoming = [makeTrace({ trace_id: "b", start_time_ms: 200 })]
    const result = mergeTraces(existing, incoming, 100)
    expect(result.map((t) => t.trace_id)).toEqual(["b", "a"])
  })

  it("updates existing traces with newer data", () => {
    const existing = [makeTrace({ trace_id: "a", duration_ms: 50 })]
    const incoming = [makeTrace({ trace_id: "a", duration_ms: 150 })]
    const result = mergeTraces(existing, incoming, 100)
    expect(result).toHaveLength(1)
    expect(result[0].duration_ms).toBe(150)
  })

  it("caps output to listCap", () => {
    const traces = Array.from({ length: 5 }, (_, i) =>
      makeTrace({ trace_id: `t-${i}`, start_time_ms: i * 100 }),
    )
    const result = mergeTraces([], traces, 3)
    expect(result).toHaveLength(3)
    expect(result[0].start_time_ms).toBe(400)
  })

  it("normalizes tags via safeTags", () => {
    const incoming = [makeTrace({ tags: [{ key: "k", value: "v" }, null as never] })]
    const result = mergeTraces([], incoming, 100)
    expect(result[0].tags).toEqual([{ key: "k", value: "v" }])
  })
})

describe("collectAllSpans", () => {
  it("returns single span for leaf node", () => {
    const span = makeSpan()
    expect(collectAllSpans(span)).toEqual([span])
  })

  it("flattens nested children", () => {
    const child1 = makeSpan({ span_id: "c1" })
    const child2 = makeSpan({ span_id: "c2" })
    const grandchild = makeSpan({ span_id: "gc1" })
    child1.children = [grandchild]
    const root = makeSpan({ span_id: "root", children: [child1, child2] })

    const all = collectAllSpans(root)
    expect(all.map((s) => s.span_id)).toEqual(["root", "c1", "gc1", "c2"])
  })
})

describe("isFailed", () => {
  it("returns true when trace has error tag", () => {
    const trace = makeTrace({ tags: [{ key: "hayhooks.error.type", value: "RuntimeError" }] })
    expect(isFailed(trace)).toBe(true)
  })

  it("returns true when success is false", () => {
    const trace = makeTrace({ tags: [{ key: "hayhooks.success", value: "false" }] })
    expect(isFailed(trace)).toBe(true)
  })

  it("returns false for successful traces", () => {
    const trace = makeTrace({ tags: [{ key: "hayhooks.success", value: "true" }] })
    expect(isFailed(trace)).toBe(false)
  })

  it("returns false when tags are undefined", () => {
    const trace = makeTrace({ tags: undefined })
    expect(isFailed(trace)).toBe(false)
  })
})

describe("isOngoing", () => {
  it("returns true when trace has no success tag and root span is still open", () => {
    const trace = makeTrace({ duration_ms: 0, root_span: makeSpan({ duration_ms: 0 }), tags: [] })
    expect(isOngoing(trace)).toBe(true)
  })

  it("returns false when success tag is present", () => {
    const trace = makeTrace({ duration_ms: 0, root_span: makeSpan({ duration_ms: 0 }), tags: [{ key: "hayhooks.success", value: "true" }] })
    expect(isOngoing(trace)).toBe(false)
  })

  it("returns false when root span is complete even without success tag", () => {
    const trace = makeTrace({ root_span: makeSpan({ duration_ms: 42 }), tags: [{ key: "service.name", value: "hayhooks" }] })
    expect(isOngoing(trace)).toBe(false)
  })

  it("uses the root running flag over a leaked child success tag", () => {
    const trace = makeTrace({
      root_span: makeSpan({ duration_ms: 0, running: true }),
      tags: [{ key: "hayhooks.success", value: "true" }],
    })
    expect(isOngoing(trace)).toBe(true)
  })

  it("returns false when the root running flag is false", () => {
    const trace = makeTrace({ root_span: makeSpan({ duration_ms: 0, running: false }), tags: [] })
    expect(isOngoing(trace)).toBe(false)
  })
})

describe("traceKind", () => {
  it.each([
    ["hayhooks.undeploy", "undeploy"],
    ["hayhooks.deploy", "deploy"],
    ["hayhooks.openai.chat", "openai"],
    ["hayhooks.mcp.tool", "mcp"],
    ["hayhooks.run", "run"],
    ["something.else", "other"],
  ] as const)("classifies '%s' as '%s'", (spanName, expected) => {
    const trace = makeTrace({ root_span: makeSpan({ name: spanName }) })
    expect(traceKind(trace)).toBe(expected)
  })
})

describe("spanTagValue", () => {
  it("returns the value for an existing tag", () => {
    const span = makeSpan({ tags: [{ key: "hayhooks.pipeline.name", value: "my_pipe" }] })
    expect(spanTagValue(span, "hayhooks.pipeline.name")).toBe("my_pipe")
  })

  it("returns undefined for a missing tag", () => {
    const span = makeSpan({ tags: [] })
    expect(spanTagValue(span, "nonexistent")).toBeUndefined()
  })

  it("handles undefined tags", () => {
    const span = makeSpan({ tags: undefined })
    expect(spanTagValue(span, "any")).toBeUndefined()
  })
})

describe("filterTracesByEntrypoint", () => {
  it("returns all traces when no filter is set", () => {
    const traces = [makeTrace({ trace_id: "a" }), makeTrace({ trace_id: "b", entrypoint: "other" })]
    expect(filterTracesByEntrypoint(traces, null)).toEqual(traces)
  })

  it("returns only traces matching the selected entrypoint", () => {
    const traces = [
      makeTrace({ trace_id: "a", entrypoint: "alpha" }),
      makeTrace({ trace_id: "b", entrypoint: "beta" }),
      makeTrace({ trace_id: "c", entrypoint: "alpha" }),
    ]
    expect(filterTracesByEntrypoint(traces, "alpha").map((t) => t.trace_id)).toEqual(["a", "c"])
  })
})

describe("slowestComponentRun", () => {
  it("returns the slowest haystack.component.run span", () => {
    const root = makeSpan({ span_id: "root", duration_ms: 10 })
    const retrieverFast = makeSpan({
      span_id: "retriever-fast",
      name: "haystack.component.run",
      duration_ms: 90,
      tags: [{ key: "haystack.component.name", value: "retriever" }],
    })
    const retrieverSlow = makeSpan({
      span_id: "retriever-slow",
      name: "haystack.component.run",
      duration_ms: 120,
      tags: [{ key: "haystack.component.name", value: "retriever" }],
    })
    const llm = makeSpan({
      span_id: "llm",
      name: "haystack.component.run",
      duration_ms: 100,
      tags: [{ key: "haystack.component.name", value: "llm" }],
    })
    const ranker = makeSpan({
      span_id: "ranker",
      name: "haystack.component.run",
      duration_ms: 80,
      tags: [{ key: "haystack.component.name", value: "ranker" }],
    })
    const slowerNonComponentRun = makeSpan({
      span_id: "embedder-step",
      name: "haystack.retriever.embed",
      duration_ms: 5000,
      tags: [{ key: "haystack.component.name", value: "embedder" }],
    })
    root.children = [retrieverFast, retrieverSlow, llm, ranker, slowerNonComponentRun]

    const result = slowestComponentRun(collectAllSpans(root))

    expect(result).toEqual({
      spanId: "retriever-slow",
      componentName: "retriever",
      durationMs: 120,
    })
  })

  it("returns null when no haystack.component.run span is present", () => {
    const root = makeSpan({
      span_id: "root",
      name: "hayhooks.pipeline.run",
      duration_ms: 5,
      children: [
        makeSpan({ span_id: "not-component-run", name: "custom.component.step", duration_ms: 55, tags: [] }),
      ],
    })

    const result = slowestComponentRun(collectAllSpans(root))

    expect(result).toBeNull()
  })
})
