import { renderHook, act } from "@testing-library/react"
import { useClock } from "./useClock"

beforeEach(() => vi.useFakeTimers())
afterEach(() => vi.useRealTimers())

describe("useClock", () => {
  it("returns current timestamp initially", () => {
    const now = Date.now()
    const { result } = renderHook(() => useClock(1000))
    expect(result.current).toBeGreaterThanOrEqual(now)
  })

  it("ticks at the specified interval", () => {
    const { result } = renderHook(() => useClock(500))
    const initial = result.current

    act(() => vi.advanceTimersByTime(500))
    expect(result.current).toBeGreaterThan(initial)
  })

  it("stops ticking on unmount", () => {
    const { result, unmount } = renderHook(() => useClock(500))
    unmount()

    const afterUnmount = result.current
    act(() => vi.advanceTimersByTime(2000))
    expect(result.current).toBe(afterUnmount)
  })
})
