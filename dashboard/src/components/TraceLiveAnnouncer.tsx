import { memo, useEffect, useRef, useState } from "react"

import { useTraceData } from "../hooks/useTracesContext"

/**
 * Visually-hidden polite live region that announces new traces to assistive
 * tech. Without this, screen reader users have no way of knowing when the
 * trace feed updates.
 */
export const TraceLiveAnnouncer = memo(function TraceLiveAnnouncer() {
  const { traces } = useTraceData()
  const seenRef = useRef<Set<string> | null>(null)
  const [message, setMessage] = useState("")

  useEffect(() => {
    if (seenRef.current === null) {
      seenRef.current = new Set(traces.map((t) => t.trace_id))
      return
    }
    const newOnes = traces.filter((t) => !seenRef.current!.has(t.trace_id))
    if (newOnes.length === 0) return

    for (const t of newOnes) seenRef.current.add(t.trace_id)

    if (newOnes.length === 1) {
      const ep = newOnes[0].entrypoint ?? "unknown"
      setMessage(`New trace from ${ep}.`)
    } else {
      setMessage(`${newOnes.length} new traces received.`)
    }
  }, [traces])

  return (
    <div role="status" aria-live="polite" aria-atomic="true" className="sr-only">
      {message}
    </div>
  )
})
