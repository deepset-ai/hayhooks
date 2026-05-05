import { TAG_LABELS, TAG_PRIORITY } from "../constants"
import type { TraceTag } from "../types"

const UNKNOWN_TAG_PRIORITY = Number.MAX_SAFE_INTEGER
export const SUCCESS_TAG_KEY = "hayhooks.success"
export const ERROR_TYPE_TAG_KEY = "hayhooks.error.type"
export const ERROR_MESSAGE_TAG_KEY = "hayhooks.error.message"
export const ERROR_STACK_TAG_KEY = "hayhooks.error.stack"
const TAG_PRIORITY_INDEX = new Map(TAG_PRIORITY.map((key, index) => [key, index]))

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.length > 0
}

function toTraceTag(candidate: unknown): TraceTag | null {
  if (!isRecord(candidate)) return null
  if (!isNonEmptyString(candidate.key) || !isNonEmptyString(candidate.value)) return null
  return { key: candidate.key, value: candidate.value }
}

function tagPriority(tagKey: string): number {
  return TAG_PRIORITY_INDEX.get(tagKey) ?? UNKNOWN_TAG_PRIORITY
}

export function safeTags(raw: unknown): TraceTag[] {
  if (!Array.isArray(raw)) return []
  const tags: TraceTag[] = []
  for (const candidate of raw) {
    const tag = toTraceTag(candidate)
    if (tag !== null) tags.push(tag)
  }
  return tags
}

export function sortTags(tags: TraceTag[]): TraceTag[] {
  const deduplicated = new Map<string, TraceTag>()
  for (const tag of tags) {
    if (!deduplicated.has(tag.key)) deduplicated.set(tag.key, tag)
  }

  return [...deduplicated.values()].sort((left, right) => {
    const priorityDiff = tagPriority(left.key) - tagPriority(right.key)
    return priorityDiff !== 0 ? priorityDiff : left.key.localeCompare(right.key)
  })
}

export function tagLabel(key: string): string {
  return TAG_LABELS[key] ?? key.replace(/^hayhooks\./, "")
}

export function isDestructiveTag(tag: TraceTag): boolean {
  if (tag.key === ERROR_TYPE_TAG_KEY) return true
  return tag.key === SUCCESS_TAG_KEY && tag.value === "false"
}
