import { render, screen } from "@testing-library/react"

import * as tracesContext from "../hooks/useTracesContext"
import { TraceList } from "./TraceList"

vi.mock("../hooks/useTracesContext", () => ({
  useTraceStatus: vi.fn(),
}))

const mockedUseTraceStatus = vi.mocked(tracesContext.useTraceStatus)

beforeEach(() => {
  mockedUseTraceStatus.mockReturnValue({
    updatedAt: null,
    error: null,
    refreshing: false,
    clearing: false,
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
        freshUntil={{}}
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
        freshUntil={{}}
        onClearFilter={() => {}}
      />,
    )

    expect(screen.getByText("Connecting…")).toBeInTheDocument()
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
        freshUntil={{}}
        onClearFilter={() => {}}
      />,
    )

    expect(screen.getByText("Unable to load traces")).toBeInTheDocument()
    expect(screen.getByText("Timed out")).toBeInTheDocument()
  })
})
