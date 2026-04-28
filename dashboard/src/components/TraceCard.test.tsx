import { render, screen } from "@testing-library/react"

import type { TraceSpanNode, TraceSummary } from "../types"
import { TraceCard } from "./TraceCard"

function makeSpan(overrides: Partial<TraceSpanNode> = {}): TraceSpanNode {
  return {
    span_id: "span-1",
    name: "hayhooks.pipeline.run",
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

describe("TraceCard", () => {
  it("shows an ongoing indicator and ongoing class for in-progress traces", () => {
    const trace = makeTrace({
      duration_ms: 0,
      root_span: makeSpan({ duration_ms: 0 }),
      tags: [{ key: "hayhooks.transport", value: "http" }],
    })
    const { container } = render(<TraceCard trace={trace} isFresh={false} />)

    expect(screen.getByText("ONGOING")).toBeInTheDocument()
    expect(container.querySelector(".trace-card-ongoing")).toBeTruthy()
  })

  it("does not show an ongoing indicator for completed traces", () => {
    const trace = makeTrace({
      tags: [{ key: "hayhooks.success", value: "true" }],
    })
    const { container } = render(<TraceCard trace={trace} isFresh={false} />)

    expect(screen.queryByText("ONGOING")).not.toBeInTheDocument()
    expect(container.querySelector(".trace-card-ongoing")).toBeFalsy()
  })
})
