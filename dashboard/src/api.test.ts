import { clearTraces, fetchEntrypoints, fetchTraces } from "./api"
import { normalizeDashboardConfig } from "./utils/config"
import { DEFAULT_DASHBOARD_CONFIG } from "./constants"

describe("normalizeDashboardConfig", () => {
  it("returns defaults for null/undefined input", () => {
    expect(normalizeDashboardConfig(null)).toEqual(DEFAULT_DASHBOARD_CONFIG)
    expect(normalizeDashboardConfig(undefined)).toEqual(DEFAULT_DASHBOARD_CONFIG)
  })

  it("returns defaults for non-object input", () => {
    expect(normalizeDashboardConfig("string")).toEqual(DEFAULT_DASHBOARD_CONFIG)
    expect(normalizeDashboardConfig(42)).toEqual(DEFAULT_DASHBOARD_CONFIG)
  })

  it("parses valid config", () => {
    const raw = {
      poll_ms: 5000,
      list_cap: 200,
      fetch_limit: 100,
      fresh_ms: 3000,
      slow_component_min_duration_ms: 1750,
    }
    expect(normalizeDashboardConfig(raw)).toEqual({
      pollMs: 5000,
      listCap: 200,
      fetchLimit: 100,
      freshMs: 3000,
      slowComponentMinDurationMs: 1750,
      apiBase: "",
      streamEnabled: true,
    })
  })

  it("clamps fetchLimit to listCap", () => {
    const raw = { poll_ms: 1000, list_cap: 50, fetch_limit: 200, fresh_ms: 3000 }
    const result = normalizeDashboardConfig(raw)
    expect(result.fetchLimit).toBe(50)
  })

  it("uses default for poll_ms below minimum (250)", () => {
    const raw = { poll_ms: 100 }
    const result = normalizeDashboardConfig(raw)
    expect(result.pollMs).toBe(DEFAULT_DASHBOARD_CONFIG.pollMs)
  })

  it("accepts poll_ms at minimum boundary", () => {
    const raw = { poll_ms: 250 }
    const result = normalizeDashboardConfig(raw)
    expect(result.pollMs).toBe(250)
  })

  it("uses default for negative list_cap", () => {
    const raw = { list_cap: -10 }
    const result = normalizeDashboardConfig(raw)
    expect(result.listCap).toBe(DEFAULT_DASHBOARD_CONFIG.listCap)
  })

  it("rounds fractional values", () => {
    const raw = {
      poll_ms: 1000.7,
      list_cap: 50.3,
      fetch_limit: 25.9,
      fresh_ms: 4000.1,
      slow_component_min_duration_ms: 1234.9,
    }
    const result = normalizeDashboardConfig(raw)
    expect(result.pollMs).toBe(1001)
    expect(result.listCap).toBe(50)
    expect(result.fetchLimit).toBe(26)
    expect(result.freshMs).toBe(4000)
    expect(result.slowComponentMinDurationMs).toBe(1235)
  })

  it("accepts fresh_ms of 0", () => {
    const raw = { fresh_ms: 0 }
    const result = normalizeDashboardConfig(raw)
    expect(result.freshMs).toBe(0)
  })

  it("uses default for non-positive slow_component_min_duration_ms", () => {
    const raw = { slow_component_min_duration_ms: 0 }
    const result = normalizeDashboardConfig(raw)
    expect(result.slowComponentMinDurationMs).toBe(DEFAULT_DASHBOARD_CONFIG.slowComponentMinDurationMs)
  })
})

describe("fetchEntrypoints", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("throws when payload does not contain a string array", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ entrypoints: [123] }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    )

    await expect(fetchEntrypoints("http://localhost")).rejects.toThrow("Entrypoints payload invalid")
  })
})

describe("clearTraces", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("accepts successful responses with no JSON body", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(null, { status: 200 }))
    await expect(clearTraces("http://localhost")).resolves.toBeUndefined()
  })

  it("fails when response body is invalid JSON", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response("{ invalid json", {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    )

    await expect(clearTraces("http://localhost")).rejects.toThrow("Clear failed")
  })
})

describe("fetchTraces", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("returns traces with the next cursor from response headers", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ traces: [] }), {
        status: 200,
        headers: {
          "Content-Type": "application/json",
          "X-Hayhooks-Trace-Cursor": "42",
        },
      }),
    )

    const result = await fetchTraces("http://localhost", 10, undefined, 7)

    expect(result).toEqual({ traces: [], nextAfterSeq: 42, hasMore: false })
    expect(globalThis.fetch).toHaveBeenCalledWith("http://localhost/traces?limit=10&after_seq=7")
  })

  it("prefers pagination state from the traces response body", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ traces: [], next_after_seq: 7, has_more: true }), {
        status: 200,
        headers: {
          "Content-Type": "application/json",
          "X-Hayhooks-Trace-Cursor": "42",
        },
      }),
    )

    const result = await fetchTraces("http://localhost", 10)

    expect(result).toEqual({ traces: [], nextAfterSeq: 7, hasMore: true })
  })

  it("throws when traces payload items are malformed", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ traces: [{ trace_id: 123 }] }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    )

    await expect(fetchTraces("http://localhost", 10)).rejects.toThrow("Traces payload invalid")
  })
})
