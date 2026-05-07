export function fmtDur(ms: number): string {
  if (!Number.isFinite(ms)) return "—"
  if (ms < 1) return "<1 ms"
  if (ms < 1000) return `${Math.round(ms)} ms`
  return `${(ms / 1000).toFixed(2)}s`
}

export function fmtTime(ms: number): string {
  if (!Number.isFinite(ms)) return "—"
  return new Date(ms).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })
}

/**
 * Coarse relative time ("just now", "12s ago", "3m ago", "2h ago", "1d ago").
 * For older timestamps falls back to the absolute clock time.
 */
export function fmtRelativeTime(ms: number, nowMs: number = Date.now()): string {
  if (!Number.isFinite(ms)) return "—"
  const diffSec = Math.max(0, Math.round((nowMs - ms) / 1000))
  if (diffSec < 5) return "just now"
  if (diffSec < 60) return `${diffSec}s ago`
  const diffMin = Math.round(diffSec / 60)
  if (diffMin < 60) return `${diffMin}m ago`
  const diffHr = Math.round(diffMin / 60)
  if (diffHr < 24) return `${diffHr}h ago`
  const diffDay = Math.round(diffHr / 24)
  if (diffDay < 7) return `${diffDay}d ago`
  return fmtTime(ms)
}

export function truncate(s: string, n: number): string {
  return s.length <= n ? s : `${s.slice(0, n - 1)}…`
}
