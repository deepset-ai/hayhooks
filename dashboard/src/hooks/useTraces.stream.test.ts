import { renderHook, act, waitFor } from "@testing-library/react"
import { useTraces } from "./useTraces"
import type { DashboardConfig, TraceSummary } from "../types"
import * as api from "../api"
import type { FetchTracesResult } from "../api"

function makeTrace(overrides: Partial<TraceSummary> = {}): TraceSummary {
  return {
    trace_id: "trace-1",
    start_time_ms: 1000,
    duration_ms: 200,
    entrypoint: "my_pipeline",
    span_count: 1,
    root_span: { span_id: "s1", name: "root", start_time_ms: 1000, duration_ms: 200, children: [] },
    tags: [],
    ...overrides,
  }
}

function makeFetchResult(overrides: Partial<FetchTracesResult> = {}): FetchTracesResult {
  return { traces: [], nextAfterSeq: null, hasMore: false, ...overrides }
}

type Listener = (event: { data?: string }) => void

class MockEventSource {
  static instances: MockEventSource[] = []
  url: string
  closed = false
  private listeners: Record<string, Listener[]> = {}

  constructor(url: string) {
    this.url = url
    MockEventSource.instances.push(this)
  }

  addEventListener(type: string, cb: Listener) {
    ;(this.listeners[type] ||= []).push(cb)
  }

  removeEventListener() {
    /* not needed for these tests */
  }

  close() {
    this.closed = true
  }

  emit(type: string, data?: string) {
    for (const cb of this.listeners[type] ?? []) cb({ data })
  }
}

const STREAM_CONFIG: DashboardConfig = {
  pollMs: 600_000,
  listCap: 100,
  fetchLimit: 50,
  freshMs: 5000,
  slowComponentMinDurationMs: 1000,
  apiBase: "",
  streamEnabled: true,
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
  MockEventSource.instances = []
  ;(globalThis as unknown as { EventSource: unknown }).EventSource = MockEventSource
  mockedApi.fetchEntrypoints.mockReset().mockResolvedValue([])
  mockedApi.fetchTraces.mockReset().mockResolvedValue(makeFetchResult())
  mockedApi.clearTraces.mockReset().mockResolvedValue(undefined)
})

afterEach(() => {
  vi.restoreAllMocks()
  delete (globalThis as unknown as { EventSource?: unknown }).EventSource
})

describe("useTraces (SSE)", () => {
  it("opens an EventSource to the stream endpoint and applies pushed traces", async () => {
    const { result } = renderHook(() => useTraces(STREAM_CONFIG))

    await waitFor(() => expect(MockEventSource.instances).toHaveLength(1))
    const source = MockEventSource.instances[0]
    expect(source.url).toContain("/traces/stream")

    act(() => {
      source.emit("open")
      source.emit(
        "snapshot",
        JSON.stringify({ traces: [makeTrace()], next_after_seq: 1, has_more: false }),
      )
    })

    await waitFor(() => {
      expect(result.current.traces).toHaveLength(1)
      expect(result.current.traces[0].trace_id).toBe("trace-1")
    })

    // No polling while the stream is healthy.
    expect(mockedApi.fetchTraces).not.toHaveBeenCalled()
  })

  it("merges subsequent trace delta events", async () => {
    const { result } = renderHook(() => useTraces(STREAM_CONFIG))
    await waitFor(() => expect(MockEventSource.instances).toHaveLength(1))
    const source = MockEventSource.instances[0]

    act(() => {
      source.emit("open")
      source.emit("snapshot", JSON.stringify({ traces: [makeTrace()], next_after_seq: 1, has_more: false }))
    })
    await waitFor(() => expect(result.current.traces).toHaveLength(1))

    act(() => {
      source.emit(
        "trace",
        JSON.stringify({ traces: [makeTrace({ trace_id: "trace-2", start_time_ms: 2000 })], next_after_seq: 2, has_more: false }),
      )
    })

    await waitFor(() => expect(result.current.traces).toHaveLength(2))
  })

  it("falls back to polling after repeated stream errors", async () => {
    mockedApi.fetchTraces.mockResolvedValue(makeFetchResult({ traces: [makeTrace()], nextAfterSeq: 5 }))
    renderHook(() => useTraces(STREAM_CONFIG))
    await waitFor(() => expect(MockEventSource.instances).toHaveLength(1))
    const source = MockEventSource.instances[0]

    act(() => {
      source.emit("error")
      source.emit("error")
      source.emit("error")
    })

    // Third failure crosses the threshold and triggers a polling fetch.
    await waitFor(() => expect(mockedApi.fetchTraces).toHaveBeenCalled())
  })

  it("falls back when the connection flaps (open then error) without ever delivering data", async () => {
    mockedApi.fetchTraces.mockResolvedValue(makeFetchResult({ traces: [makeTrace()], nextAfterSeq: 5 }))
    renderHook(() => useTraces(STREAM_CONFIG))
    await waitFor(() => expect(MockEventSource.instances).toHaveLength(1))
    const source = MockEventSource.instances[0]

    // open must NOT reset the failure counter — otherwise a flapping connection
    // that never sends a payload would bounce below the threshold forever.
    act(() => {
      source.emit("open")
      source.emit("error")
      source.emit("open")
      source.emit("error")
      source.emit("open")
      source.emit("error")
    })

    await waitFor(() => expect(mockedApi.fetchTraces).toHaveBeenCalled())
  })
})
