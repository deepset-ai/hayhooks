import { fireEvent, render, screen } from "@testing-library/react"

import type { TraceSpanNode, TraceSummary } from "../types"
import { truncate } from "../utils/formatting"
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

  it("shows related tags for the selected span", () => {
    const childSpan = makeSpan({
      span_id: "span-child",
      name: "haystack.component.run",
      duration_ms: 50,
      tags: [{ key: "haystack.component.name", value: "prompt_builder" }],
    })
    const trace = makeTrace({
      span_count: 2,
      root_span: makeSpan({
        span_id: "span-root",
        duration_ms: 200,
        tags: [{ key: "hayhooks.route", value: "/root-route" }],
        children: [childSpan],
      }),
    })

    render(<TraceCard trace={trace} isFresh={false} />)

    fireEvent.click(screen.getByRole("button", { name: /my_pipeline/i }))
    expect(screen.getByText("/root-route")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: /haystack.component.run/i }))
    expect(screen.getByText("prompt_builder")).toBeInTheDocument()
    expect(screen.queryByText("/root-route")).not.toBeInTheDocument()
  })

  it("truncates selected span tag values in chips", () => {
    const longValue = "component-input-type-value-".repeat(4)
    const trace = makeTrace({
      root_span: makeSpan({
        span_id: "span-root",
        duration_ms: 200,
        tags: [{ key: "hayhooks.route", value: longValue }],
      }),
    })

    render(<TraceCard trace={trace} isFresh={false} />)

    fireEvent.click(screen.getByRole("button", { name: /my_pipeline/i }))
    const label = screen.getByText("route")
    const chip = label.parentElement
    expect(chip).toBeTruthy()
    const valueElement = chip?.querySelector(".font-mono")
    expect(valueElement).toBeTruthy()
    expect(valueElement).toHaveTextContent(truncate(longValue, 40))
    expect(valueElement).not.toHaveTextContent(longValue)
  })

  it("shows span details without a secondary collapsible panel", () => {
    const trace = makeTrace({
      tags: [{ key: "hayhooks.transport", value: "rest" }],
      root_span: makeSpan({
        tags: [{ key: "hayhooks.route", value: "/demo" }],
      }),
    })

    render(<TraceCard trace={trace} isFresh={false} />)

    fireEvent.click(screen.getByRole("button", { name: /my_pipeline/i }))
    expect(screen.queryByRole("button", { name: /Span view/i })).not.toBeInTheDocument()
    expect(screen.queryByText("Trace view")).not.toBeInTheDocument()
    expect(screen.queryByText("Trace tags")).not.toBeInTheDocument()
    expect(screen.getByText("Span tags")).toBeInTheDocument()
    expect(screen.getByText("Spans")).toBeInTheDocument()
  })
})
