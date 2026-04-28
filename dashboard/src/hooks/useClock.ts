import { useEffect, useState } from "react"

/**
 * Ticks every `intervalMs` so components can compare timestamps against "now"
 * (e.g. freshness indicators).
 */
export function useClock(intervalMs = 1000) {
  const [now, setNow] = useState(() => Date.now())

  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), intervalMs)
    return () => window.clearInterval(id)
  }, [intervalMs])

  return now
}
