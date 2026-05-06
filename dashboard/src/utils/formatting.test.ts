import { fmtDur, fmtRelativeTime, fmtTime, truncate } from "./formatting"

describe("fmtDur", () => {
  it("returns fallback for invalid values", () => {
    expect(fmtDur(Number.NaN)).toBe("—")
    expect(fmtDur(Number.POSITIVE_INFINITY)).toBe("—")
  })

  it("returns '<1 ms' for sub-millisecond durations", () => {
    expect(fmtDur(0)).toBe("<1 ms")
    expect(fmtDur(0.5)).toBe("<1 ms")
  })

  it("rounds milliseconds", () => {
    expect(fmtDur(1)).toBe("1 ms")
    expect(fmtDur(42.7)).toBe("43 ms")
    expect(fmtDur(999)).toBe("999 ms")
  })

  it("formats seconds with two decimal places", () => {
    expect(fmtDur(1000)).toBe("1.00s")
    expect(fmtDur(1234)).toBe("1.23s")
    expect(fmtDur(60500)).toBe("60.50s")
  })
})

describe("fmtTime", () => {
  it("returns fallback for invalid timestamps", () => {
    expect(fmtTime(Number.NaN)).toBe("—")
    expect(fmtTime(Number.POSITIVE_INFINITY)).toBe("—")
  })

  it("formats a timestamp as HH:MM:SS", () => {
    const result = fmtTime(0)
    expect(result).toMatch(/\d{2}:\d{2}:\d{2}/)
  })
})

describe("fmtRelativeTime", () => {
  it("returns fallback for invalid timestamps", () => {
    expect(fmtRelativeTime(Number.NaN, 0)).toBe("—")
    expect(fmtRelativeTime(Number.POSITIVE_INFINITY, 0)).toBe("—")
  })

  it("returns 'just now' for < 5 seconds", () => {
    const now = 1_000_000
    expect(fmtRelativeTime(now, now)).toBe("just now")
    expect(fmtRelativeTime(now - 4_000, now)).toBe("just now")
  })

  it("formats seconds, minutes, hours, days", () => {
    const now = 10_000_000_000
    expect(fmtRelativeTime(now - 12_000, now)).toBe("12s ago")
    expect(fmtRelativeTime(now - 90_000, now)).toBe("2m ago")
    expect(fmtRelativeTime(now - 7_200_000, now)).toBe("2h ago")
    expect(fmtRelativeTime(now - 2 * 86_400_000, now)).toBe("2d ago")
  })

  it("falls back to absolute time after a week", () => {
    const now = 10_000_000_000
    const result = fmtRelativeTime(now - 14 * 86_400_000, now)
    expect(result).toMatch(/\d{2}:\d{2}:\d{2}/)
  })
})

describe("truncate", () => {
  it("returns the string unchanged when within limit", () => {
    expect(truncate("abc", 5)).toBe("abc")
    expect(truncate("abc", 3)).toBe("abc")
  })

  it("truncates and appends ellipsis when string exceeds limit", () => {
    expect(truncate("abcdef", 4)).toBe("abc…")
    expect(truncate("hello world", 6)).toBe("hello…")
  })
})
