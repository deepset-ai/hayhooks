import { renderHook, act } from "@testing-library/react"
import { useDarkMode } from "./useDarkMode"

beforeEach(() => {
  document.documentElement.classList.remove("dark")
})

describe("useDarkMode", () => {
  it("initializes from document class", () => {
    document.documentElement.classList.add("dark")
    const { result } = renderHook(() => useDarkMode())
    expect(result.current.dark).toBe(true)
  })

  it("defaults to light when dark class is absent", () => {
    const { result } = renderHook(() => useDarkMode())
    expect(result.current.dark).toBe(false)
  })

  it("toggles dark mode and updates DOM", () => {
    const { result } = renderHook(() => useDarkMode())

    act(() => result.current.toggle())
    expect(result.current.dark).toBe(true)
    expect(document.documentElement.classList.contains("dark")).toBe(true)

    act(() => result.current.toggle())
    expect(result.current.dark).toBe(false)
    expect(document.documentElement.classList.contains("dark")).toBe(false)
  })
})
