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

export function truncate(s: string, n: number): string {
  return s.length <= n ? s : `${s.slice(0, n - 1)}…`
}
