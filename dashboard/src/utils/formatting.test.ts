import { fmtDur, fmtTime, truncate } from "./formatting"

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
