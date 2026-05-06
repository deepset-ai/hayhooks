import { useCallback, useEffect, useMemo, useState } from "react"

import { useDarkMode } from "./hooks/useDarkMode"
import { useTraceData } from "./hooks/useTracesContext"
import { fmtDur, fmtRelativeTime, fmtTime } from "./utils/formatting"
import { filterTracesByEntrypoint, isFailed } from "./utils/traces"
import { EntrypointsSidebar } from "./components/EntrypointsSidebar"
import { Header } from "./components/Header"
import { Stat, StatStrip } from "./components/Stats"
import { TraceList } from "./components/TraceList"
import { TraceLiveAnnouncer } from "./components/TraceLiveAnnouncer"

const SPARKLINE_LIMIT = 30

export default function App() {
  const { dark, toggle: toggleDark } = useDarkMode()
  const { entrypoints, traces, slowComponentMinDurationMs } = useTraceData()

  const [nowMs, setNowMs] = useState(() => Date.now())
  useEffect(() => {
    const id = window.setInterval(() => setNowMs(Date.now()), 10_000)
    return () => window.clearInterval(id)
  }, [])

  const [filter, setFilter] = useState<string | null>(null)
  const filteredTraces = useMemo(
    () => filterTracesByEntrypoint(traces, filter),
    [traces, filter],
  )
  const isFiltered = filter !== null

  const avgDurationLabel = useMemo(() => {
    if (filteredTraces.length === 0) return "—"
    const total = filteredTraces.reduce((sum, t) => sum + t.duration_ms, 0)
    return fmtDur(total / filteredTraces.length)
  }, [filteredTraces])

  /**
   * Sparkline expects oldest → newest. The trace list is newest-first, so we
   * take the most recent SPARKLINE_LIMIT and reverse.
   */
  const durationHistory = useMemo(
    () =>
      filteredTraces
        .slice(0, SPARKLINE_LIMIT)
        .map((t) => t.duration_ms)
        .reverse(),
    [filteredTraces],
  )

  const failureCount = useMemo(
    () => filteredTraces.reduce((n, t) => (isFailed(t) ? n + 1 : n), 0),
    [filteredTraces],
  )

  const lastTrace = filteredTraces[0]
  const lastTraceFailed = lastTrace !== undefined && isFailed(lastTrace)

  const handleClearFilter = useCallback(() => {
    setFilter(null)
  }, [])

  return (
    <div className="flex min-h-screen flex-col bg-background">
      <TraceLiveAnnouncer />
      <Header
        dark={dark}
        onToggleDark={toggleDark}
      />

      <main className="mx-auto w-full max-w-7xl flex-1 space-y-6 px-6 py-6">
        <StatStrip>
          <Stat
            label="Traces"
            value={filteredTraces.length}
            hint={isFiltered ? `of ${traces.length}` : undefined}
          />
          <Stat
            label="Failures"
            value={failureCount}
            tone={failureCount > 0 ? "destructive" : "default"}
          />
          <Stat
            label="Avg duration"
            value={avgDurationLabel}
            sparkline={durationHistory}
          />
          <Stat
            label="Last trace"
            value={lastTrace !== undefined ? fmtRelativeTime(lastTrace.start_time_ms, nowMs) : "—"}
            title={lastTrace !== undefined ? fmtTime(lastTrace.start_time_ms) : undefined}
            tone={lastTraceFailed ? "destructive" : "default"}
            hint={lastTrace !== undefined ? (lastTraceFailed ? "failed" : undefined) : undefined}
          />
        </StatStrip>

        <div className="grid gap-6 lg:grid-cols-[260px_1fr]">
          <EntrypointsSidebar
            entrypoints={entrypoints}
            traces={traces}
            filter={filter}
            onFilterChange={setFilter}
          />
          <TraceList
            traces={filteredTraces}
            totalTraces={traces.length}
            filter={filter}
            slowComponentMinDurationMs={slowComponentMinDurationMs}
            onClearFilter={handleClearFilter}
          />
        </div>
      </main>
    </div>
  )
}

