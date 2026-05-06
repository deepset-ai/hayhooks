import { renderHook, act, waitFor } from "@testing-library/react"
import { useTraces } from "./useTraces"
import type { DashboardConfig, TraceSummary, TraceSpanNode } from "../types"
import * as api from "../api"
import type { FetchTracesResult } from "../api"

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

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((res) => { resolve = res })
  return { promise, resolve }
}

function makeFetchResult(overrides: Partial<FetchTracesResult> = {}): FetchTracesResult {
  return { traces: [], nextAfterSeq: null, hasMore: false, ...overrides }
}

const TEST_CONFIG: DashboardConfig = {
  pollMs: 600_000,
  listCap: 100,
  fetchLimit: 50,
  freshMs: 5000,
  slowComponentMinDurationMs: 1000,
  apiBase: "",
}

vi.mock("../api", async (importOriginal) => {
  const actual = await importOriginal<typeof api>()
  return {
    ...actual,
    resolveApiBase: () => "http://test",
    fetchEntrypoints: vi.fn().mockResolvedValue([]),
    fetchTraces: vi.fn().mockResolvedValue(makeFetchResult()),
    clearTraces: vi.fn().mockResolvedValue(undefined),
  }
})

const mockedApi = vi.mocked(api)

beforeEach(() => {
  mockedApi.fetchEntrypoints.mockReset()
  mockedApi.fetchTraces.mockReset()
  mockedApi.clearTraces.mockReset()
  mockedApi.fetchEntrypoints.mockResolvedValue([])
  mockedApi.fetchTraces.mockResolvedValue(makeFetchResult())
  mockedApi.clearTraces.mockResolvedValue(undefined)
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe("useTraces", () => {
  it("fetches entrypoints and traces on mount", async () => {
    mockedApi.fetchEntrypoints.mockResolvedValue(["pipe_a", "pipe_b"])
    mockedApi.fetchTraces.mockResolvedValue(makeFetchResult({ traces: [makeTrace()], nextAfterSeq: 1 }))

    const { result } = renderHook(() => useTraces(TEST_CONFIG))

    await waitFor(() => {
      expect(result.current.entrypoints).toEqual(["pipe_a", "pipe_b"])
      expect(result.current.traces).toHaveLength(1)
      expect(result.current.updatedAt).not.toBeNull()
      expect(result.current.error).toBeNull()
    })
  })

  it("sets error on fetch failure", async () => {
    mockedApi.fetchEntrypoints.mockRejectedValue(new Error("Network error"))

    const { result } = renderHook(() => useTraces(TEST_CONFIG))

    await waitFor(() => {
      expect(result.current.error).toBe("Network error")
    })
  })

  it("clears traces and resets state", async () => {
    mockedApi.fetchEntrypoints.mockResolvedValue(["pipe_a"])
    mockedApi.fetchTraces.mockResolvedValue(makeFetchResult({ traces: [makeTrace()], nextAfterSeq: 1 }))

    const { result } = renderHook(() => useTraces(TEST_CONFIG))

    await waitFor(() => expect(result.current.traces).toHaveLength(1))

    await act(async () => { await result.current.clear() })

    expect(result.current.traces).toHaveLength(0)
    expect(result.current.error).toBeNull()
    expect(mockedApi.clearTraces).toHaveBeenCalledWith("http://test")
  })

  it("sets error when clear fails", async () => {
    mockedApi.fetchEntrypoints.mockResolvedValue(["pipe_a"])
    mockedApi.fetchTraces.mockResolvedValue(makeFetchResult({ traces: [makeTrace()], nextAfterSeq: 1 }))
    mockedApi.clearTraces.mockRejectedValue(new Error("Clear failed"))

    const { result } = renderHook(() => useTraces(TEST_CONFIG))

    await waitFor(() => expect(result.current.traces).toHaveLength(1))

    await act(async () => { await result.current.clear() })

    expect(result.current.error).toBe("Clear failed")
    expect(result.current.traces).toHaveLength(1)
  })

  it("marks new traces as fresh", async () => {
    const trace = makeTrace({ trace_id: "new-trace", start_time_ms: Date.now() })
    mockedApi.fetchEntrypoints.mockResolvedValue([])
    mockedApi.fetchTraces.mockResolvedValue(makeFetchResult({ traces: [trace], nextAfterSeq: 1 }))

    const { result } = renderHook(() => useTraces(TEST_CONFIG))

    await waitFor(() => {
      expect(result.current.freshUntil["new-trace"]).toBeDefined()
      expect(result.current.freshUntil["new-trace"]).toBeGreaterThan(Date.now())
    })
  })

  it("triggers manual refresh", async () => {
    const { result } = renderHook(() => useTraces(TEST_CONFIG))

    await waitFor(() => expect(mockedApi.fetchEntrypoints).toHaveBeenCalled())
    const callsBefore = mockedApi.fetchEntrypoints.mock.calls.length

    await act(async () => { await result.current.refresh() })

    expect(mockedApi.fetchEntrypoints.mock.calls.length).toBeGreaterThan(callsBefore)
  })

  it("ignores stale refresh results that complete after clear", async () => {
    const staleRefresh = deferred<FetchTracesResult>()
    mockedApi.fetchEntrypoints.mockResolvedValue(["pipe_a"])
    mockedApi.fetchTraces
      .mockReturnValueOnce(staleRefresh.promise)
      .mockResolvedValue(makeFetchResult())
    mockedApi.fetchTraces.mockClear()

    const { result } = renderHook(() => useTraces(TEST_CONFIG))

    await waitFor(() => expect(mockedApi.fetchTraces).toHaveBeenCalledTimes(1))
    await act(async () => { await result.current.clear() })

    await act(async () => {
      staleRefresh.resolve(makeFetchResult({ traces: [makeTrace({ trace_id: "stale" })], nextAfterSeq: 1 }))
      await Promise.resolve()
    })

    await waitFor(() => expect(result.current.refreshing).toBe(false))
    expect(result.current.traces).toHaveLength(0)
  })

  it("applies new listCap even when incremental refresh is empty", async () => {
    mockedApi.fetchEntrypoints.mockResolvedValue(["pipe_a"])
    mockedApi.fetchTraces
      .mockResolvedValueOnce(makeFetchResult({
        traces: [
          makeTrace({ trace_id: "newer", start_time_ms: 2000 }),
          makeTrace({ trace_id: "older", start_time_ms: 1000 }),
        ],
        nextAfterSeq: 2,
      }))
      .mockResolvedValue(makeFetchResult())

    const initialConfig: DashboardConfig = { ...TEST_CONFIG, listCap: 2 }
    const reducedConfig: DashboardConfig = { ...initialConfig, listCap: 1 }

    const { result, rerender } = renderHook(
      ({ config }) => useTraces(config),
      { initialProps: { config: initialConfig } },
    )

    await waitFor(() => expect(result.current.traces).toHaveLength(2))

    rerender({ config: reducedConfig })
    await act(async () => { await result.current.refresh() })

    expect(result.current.traces).toHaveLength(1)
    expect(result.current.traces[0].trace_id).toBe("newer")
  })

  it("does not skip traces that share the same start time across incremental polls", async () => {
    const traceA = makeTrace({ trace_id: "trace-a", start_time_ms: 1000 })
    const traceB = makeTrace({ trace_id: "trace-b", start_time_ms: 1000 })
    mockedApi.fetchEntrypoints.mockResolvedValue([])
    mockedApi.fetchTraces
      .mockResolvedValueOnce(makeFetchResult({ traces: [traceA], nextAfterSeq: 11 }))
      .mockResolvedValueOnce(makeFetchResult({ traces: [traceB], nextAfterSeq: 12 }))

    const { result } = renderHook(() => useTraces(TEST_CONFIG))

    await waitFor(() => expect(result.current.traces.map((t) => t.trace_id)).toEqual(["trace-a"]))
    await act(async () => { await result.current.refresh() })

    expect(new Set(result.current.traces.map((t) => t.trace_id))).toEqual(new Set(["trace-a", "trace-b"]))
    expect(mockedApi.fetchTraces).toHaveBeenNthCalledWith(1, "http://test", 50, undefined, undefined)
    expect(mockedApi.fetchTraces).toHaveBeenNthCalledWith(2, "http://test", 50, undefined, 11)
  })

  it("continues paginated catch-up from the last returned cursor", async () => {
    const traceA = makeTrace({ trace_id: "trace-a", start_time_ms: 1000 })
    const traceB = makeTrace({ trace_id: "trace-b", start_time_ms: 1001 })
    mockedApi.fetchEntrypoints.mockResolvedValue([])
    mockedApi.fetchTraces
      .mockResolvedValueOnce(makeFetchResult({ traces: [traceA], nextAfterSeq: 11, hasMore: true }))
      .mockResolvedValueOnce(makeFetchResult({ traces: [traceB], nextAfterSeq: 12, hasMore: false }))

    const { result } = renderHook(() => useTraces(TEST_CONFIG))

    await waitFor(() => {
      expect(new Set(result.current.traces.map((t) => t.trace_id))).toEqual(new Set(["trace-a", "trace-b"]))
    })
    expect(mockedApi.fetchTraces).toHaveBeenNthCalledWith(1, "http://test", 50, undefined, undefined)
    expect(mockedApi.fetchTraces).toHaveBeenNthCalledWith(2, "http://test", 50, undefined, 11)
  })

  it("recovers automatically when backend cursor regresses", async () => {
    const traceA = makeTrace({ trace_id: "trace-a", start_time_ms: 1000 })
    const traceB = makeTrace({ trace_id: "trace-b", start_time_ms: 1200 })
    mockedApi.fetchEntrypoints.mockResolvedValue([])
    mockedApi.fetchTraces
      .mockResolvedValueOnce(makeFetchResult({ traces: [traceA], nextAfterSeq: 10 }))
      .mockResolvedValueOnce(makeFetchResult({ nextAfterSeq: 2 }))
      .mockResolvedValueOnce(makeFetchResult({ traces: [traceB], nextAfterSeq: 3 }))

    const { result } = renderHook(() => useTraces(TEST_CONFIG))

    await waitFor(() => expect(result.current.traces.map((t) => t.trace_id)).toEqual(["trace-a"]))
    await act(async () => { await result.current.refresh() })

    expect(new Set(result.current.traces.map((t) => t.trace_id))).toEqual(new Set(["trace-a", "trace-b"]))
    expect(mockedApi.fetchTraces).toHaveBeenNthCalledWith(1, "http://test", 50, undefined, undefined)
    expect(mockedApi.fetchTraces).toHaveBeenNthCalledWith(2, "http://test", 50, undefined, 10)
    expect(mockedApi.fetchTraces).toHaveBeenNthCalledWith(3, "http://test", 50)
  })
})
