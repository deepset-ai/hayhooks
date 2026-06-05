import { render, screen } from "@testing-library/react"
import { SpanRow } from "./SpanRow"
import type { TraceSpanNode } from "../types"

function makeSpan(overrides: Partial<TraceSpanNode> = {}): TraceSpanNode {
  return {
    span_id: "s1",
    name: "haystack.component.run",
    start_time_ms: 1000,
    duration_ms: 0,
    children: [],
    ...overrides,
  }
}

const baseProps = {
  depth: 0,
  traceStart: 1000,
  traceDuration: 0,
  traceEntrypoint: "my_pipeline",
  slowSpanId: null,
  selectedSpanId: "",
  onSelectSpan: () => {},
}

describe("SpanRow", () => {
  it("renders a running span with a live indicator and an indeterminate bar", () => {
    const { container } = render(<SpanRow span={makeSpan({ running: true })} {...baseProps} />)

    expect(screen.getByText("live")).toBeInTheDocument()
    expect(container.querySelector(".waterfall-bar-running")).not.toBeNull()
    expect(container.querySelector(".span-row-running")).not.toBeNull()
    // It must not look like a finished 0ms span.
    expect(screen.queryByText("0ms")).toBeNull()
  })

  it("renders a finished span with a duration and a static sized bar", () => {
    const { container } = render(
      <SpanRow span={makeSpan({ running: false, duration_ms: 120 })} {...baseProps} traceDuration={200} />,
    )

    expect(screen.queryByText("live")).toBeNull()
    expect(container.querySelector(".waterfall-bar-running")).toBeNull()
    expect(container.querySelector(".waterfall-bar")).not.toBeNull()
  })

  it("treats a missing running flag as finished (backward compatible)", () => {
    const { container } = render(
      <SpanRow span={makeSpan({ duration_ms: 50 })} {...baseProps} traceDuration={200} />,
    )

    expect(screen.queryByText("live")).toBeNull()
    expect(container.querySelector(".waterfall-bar-running")).toBeNull()
  })
})
