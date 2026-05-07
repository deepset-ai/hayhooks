import { render, screen } from "@testing-library/react"

import * as tracesContext from "../hooks/useTracesContext"
import { TraceList } from "./TraceList"

vi.mock("../hooks/useTracesContext", () => ({
  useTraceStatus: vi.fn(),
  useTraceFreshness: vi.fn(),
  useTraceData: vi.fn(),
}))

const mockedUseTraceStatus = vi.mocked(tracesContext.useTraceStatus)
const mockedUseTraceFreshness = vi.mocked(tracesContext.useTraceFreshness)
const mockedUseTraceData = vi.mocked(tracesContext.useTraceData)

beforeEach(() => {
  mockedUseTraceFreshness.mockReturnValue({ freshUntil: {} })
  mockedUseTraceStatus.mockReturnValue({
    updatedAt: null,
    error: null,
    refreshing: false,
    clearing: false,
  })
  mockedUseTraceData.mockReturnValue({
    entrypoints: [],
    traces: [],
    slowComponentMinDurationMs: 1000,
    listCap: 100,
  })
})

describe("TraceList", () => {
  it("shows an explicit error state when initial load fails", () => {
    mockedUseTraceStatus.mockReturnValue({
      updatedAt: null,
      error: "Network error",
      refreshing: false,
      clearing: false,
    })

    render(
      <TraceList
        traces={[]}
        totalTraces={0}
        filter={null}
        onClearFilter={() => {}}
      />,
    )

    expect(screen.getByText("Unable to load traces")).toBeInTheDocument()
    expect(screen.getByText("Network error")).toBeInTheDocument()
  })

  it("shows connecting state while first refresh is in flight", () => {
    render(
      <TraceList
        traces={[]}
        totalTraces={0}
        filter={null}
        onClearFilter={() => {}}
      />,
    )

    expect(screen.getByText(/Connecting to trace buffer/)).toBeInTheDocument()
  })

  it("shows error state even when a previous update timestamp exists", () => {
    mockedUseTraceStatus.mockReturnValue({
      updatedAt: Date.now(),
      error: "Timed out",
      refreshing: false,
      clearing: false,
    })

    render(
      <TraceList
        traces={[]}
        totalTraces={0}
        filter={null}
        onClearFilter={() => {}}
      />,
    )

    expect(screen.getByText("Unable to load traces")).toBeInTheDocument()
    expect(screen.getByText("Timed out")).toBeInTheDocument()
  })
})
