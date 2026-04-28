import { isDestructiveTag, safeTags, sortTags, tagLabel } from "./tags"
import type { TraceTag } from "../types"

describe("safeTags", () => {
  it("returns empty array for non-array input", () => {
    expect(safeTags(null)).toEqual([])
    expect(safeTags(undefined)).toEqual([])
    expect(safeTags("string")).toEqual([])
    expect(safeTags(42)).toEqual([])
  })

  it("filters out malformed entries", () => {
    const raw = [
      { key: "a", value: "1" },
      null,
      { key: "", value: "2" },
      { key: "b" },
      { key: "c", value: "3" },
    ]
    expect(safeTags(raw)).toEqual([
      { key: "a", value: "1" },
      { key: "c", value: "3" },
    ])
  })

  it("returns empty array for empty input", () => {
    expect(safeTags([])).toEqual([])
  })
})

describe("sortTags", () => {
  it("sorts by TAG_PRIORITY order, known keys first", () => {
    const tags: TraceTag[] = [
      { key: "custom.key", value: "x" },
      { key: "hayhooks.success", value: "true" },
      { key: "hayhooks.transport", value: "http" },
    ]
    const sorted = sortTags(tags)
    expect(sorted.map((t) => t.key)).toEqual([
      "hayhooks.transport",
      "hayhooks.success",
      "custom.key",
    ])
  })

  it("deduplicates by key, keeping first occurrence", () => {
    const tags: TraceTag[] = [
      { key: "hayhooks.transport", value: "first" },
      { key: "hayhooks.transport", value: "second" },
    ]
    const sorted = sortTags(tags)
    expect(sorted).toHaveLength(1)
    expect(sorted[0].value).toBe("first")
  })

  it("sorts unknown keys alphabetically", () => {
    const tags: TraceTag[] = [
      { key: "z.custom", value: "1" },
      { key: "a.custom", value: "2" },
    ]
    const sorted = sortTags(tags)
    expect(sorted.map((t) => t.key)).toEqual(["a.custom", "z.custom"])
  })
})

describe("tagLabel", () => {
  it("returns known label for mapped keys", () => {
    expect(tagLabel("hayhooks.transport")).toBe("transport")
    expect(tagLabel("hayhooks.success")).toBe("success")
    expect(tagLabel("service.name")).toBe("service")
  })

  it("strips hayhooks prefix for unknown keys", () => {
    expect(tagLabel("hayhooks.custom.field")).toBe("custom.field")
  })

  it("returns key as-is if no hayhooks prefix", () => {
    expect(tagLabel("some.other.key")).toBe("some.other.key")
  })
})

describe("isDestructiveTag", () => {
  it("returns true for error type tags", () => {
    expect(isDestructiveTag({ key: "hayhooks.error.type", value: "RuntimeError" })).toBe(true)
  })

  it("returns true for success=false", () => {
    expect(isDestructiveTag({ key: "hayhooks.success", value: "false" })).toBe(true)
  })

  it("returns false for success=true", () => {
    expect(isDestructiveTag({ key: "hayhooks.success", value: "true" })).toBe(false)
  })

  it("returns false for unrelated tags", () => {
    expect(isDestructiveTag({ key: "hayhooks.transport", value: "http" })).toBe(false)
  })
})
