import { fireEvent, render, screen, within } from "@testing-library/react"

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

  it("uses failed fresh highlight when trace is fresh and failed", () => {
    const trace = makeTrace({
      tags: [{ key: "hayhooks.success", value: "false" }],
      root_span: makeSpan({
        name: "hayhooks.pipeline.run",
      }),
    })
    const { container } = render(<TraceCard trace={trace} isFresh />)

    const freshCard = container.querySelector(".trace-card-fresh")
    expect(freshCard).toBeTruthy()
    expect(freshCard).toHaveClass("trace-card-fresh-failed")
    expect(freshCard).not.toHaveClass("trace-card-fresh-run")
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
    expect(screen.getAllByText("prompt_builder").length).toBeGreaterThan(0)
    expect(screen.queryByText("/root-route")).not.toBeInTheDocument()
  })

  it("shows component names directly on component run span rows", () => {
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
        children: [childSpan],
      }),
    })

    render(<TraceCard trace={trace} isFresh={false} />)
    fireEvent.click(screen.getByRole("button", { name: /my_pipeline/i }))

    const componentRunRow = screen.getByRole("button", { name: /haystack.component.run/i })
    const componentLabel = within(componentRunRow).getByText("prompt_builder")
    expect(componentLabel).toBeInTheDocument()
    expect(componentLabel).toHaveClass("border-primary/30", "bg-primary/10", "text-primary")
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

  it("highlights only the slowest component when above threshold", () => {
    const retriever = makeSpan({
      span_id: "span-retriever",
      name: "haystack.component.run",
      duration_ms: 1200,
      tags: [{ key: "haystack.component.name", value: "retriever" }],
    })
    const llm = makeSpan({
      span_id: "span-llm",
      name: "haystack.component.run",
      duration_ms: 900,
      tags: [{ key: "haystack.component.name", value: "llm" }],
    })
    const ranker = makeSpan({
      span_id: "span-ranker",
      name: "haystack.component.run",
      duration_ms: 700,
      tags: [{ key: "haystack.component.name", value: "ranker" }],
    })
    const nonComponentRun = makeSpan({
      span_id: "span-non-component",
      name: "haystack.retriever.embed",
      duration_ms: 9000,
      tags: [{ key: "haystack.component.name", value: "embedder" }],
    })
    const writer = makeSpan({
      span_id: "span-writer",
      name: "haystack.component.run",
      duration_ms: 40,
      tags: [{ key: "haystack.component.name", value: "writer" }],
    })
    const trace = makeTrace({
      span_count: 5,
      root_span: makeSpan({
        span_id: "span-root",
        duration_ms: 200,
        children: [retriever, llm, ranker, nonComponentRun, writer],
      }),
    })

    render(<TraceCard trace={trace} isFresh={false} slowComponentMinDurationMs={1000} />)
    fireEvent.click(screen.getByRole("button", { name: /my_pipeline/i }))

    expect(screen.getByText("Slowest component")).toBeInTheDocument()
    const spansHeader = screen.getByText("Spans").closest("div")
    expect(spansHeader).toBeTruthy()
    expect(within(spansHeader as HTMLElement).getByText("Slowest component")).toBeInTheDocument()
    expect(within(spansHeader as HTMLElement).getByText("retriever")).toBeInTheDocument()
    expect(within(spansHeader as HTMLElement).queryByText("llm")).not.toBeInTheDocument()
    expect(within(spansHeader as HTMLElement).queryByText("ranker")).not.toBeInTheDocument()
    expect(within(spansHeader as HTMLElement).queryByText("writer")).not.toBeInTheDocument()

    const componentRunButtons = screen.getAllByRole("button", { name: /haystack.component.run/i })
    expect(componentRunButtons).toHaveLength(4)
    expect(componentRunButtons[0]).toHaveAttribute("data-slow-component", "true")
    expect(componentRunButtons[1]).toHaveAttribute("data-slow-component", "false")
    expect(componentRunButtons[2]).toHaveAttribute("data-slow-component", "false")
    expect(componentRunButtons[3]).toHaveAttribute("data-slow-component", "false")
    expect(screen.getByRole("button", { name: /haystack.retriever.embed/i })).toHaveAttribute("data-slow-component", "false")
  })

  it("does not highlight components when slowest duration is below threshold", () => {
    const retriever = makeSpan({
      span_id: "span-retriever",
      name: "haystack.component.run",
      duration_ms: 800,
      tags: [{ key: "haystack.component.name", value: "retriever" }],
    })
    const llm = makeSpan({
      span_id: "span-llm",
      name: "haystack.component.run",
      duration_ms: 700,
      tags: [{ key: "haystack.component.name", value: "llm" }],
    })
    const nonComponentRun = makeSpan({
      span_id: "span-non-component",
      name: "haystack.generator.prepare_prompt",
      duration_ms: 5000,
      tags: [{ key: "haystack.component.name", value: "prompt_builder" }],
    })
    const trace = makeTrace({
      span_count: 4,
      root_span: makeSpan({
        span_id: "span-root",
        duration_ms: 1600,
        children: [retriever, llm, nonComponentRun],
      }),
    })

    render(<TraceCard trace={trace} isFresh={false} slowComponentMinDurationMs={1000} />)
    fireEvent.click(screen.getByRole("button", { name: /my_pipeline/i }))

    expect(screen.queryByText("Slowest component")).not.toBeInTheDocument()
    const componentRunButtons = screen.getAllByRole("button", { name: /haystack.component.run/i })
    expect(componentRunButtons).toHaveLength(2)
    expect(componentRunButtons[0]).toHaveAttribute("data-slow-component", "false")
    expect(componentRunButtons[1]).toHaveAttribute("data-slow-component", "false")
    expect(screen.getByRole("button", { name: /haystack.generator.prepare_prompt/i })).toHaveAttribute("data-slow-component", "false")
  })
})
