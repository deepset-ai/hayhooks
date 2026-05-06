import { memo, useCallback, useEffect, useMemo, useState } from "react"
import { Clock, GitBranch, Layers, Timer } from "lucide-react"

import { useDarkMode } from "./hooks/useDarkMode"
import { useTraceData } from "./hooks/useTracesContext"
import { fmtDur, fmtRelativeTime, fmtTime } from "./utils/formatting"
import { filterTracesByEntrypoint } from "./utils/traces"
import { EntrypointsSidebar } from "./components/EntrypointsSidebar"
import { Header } from "./components/Header"
import { StatCard } from "./components/StatCard"
import { TraceList } from "./components/TraceList"
import { TraceLiveAnnouncer } from "./components/TraceLiveAnnouncer"

const ENTRYPOINTS_ICON = <GitBranch className="size-4" />
const TRACES_ICON = <Layers className="size-4" />
const AVG_DURATION_ICON = <Timer className="size-4" />
const LAST_TRACE_ICON = <Clock className="size-4" />

const App = memo(function App() {
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

  const avgDuration = useMemo(() => {
    if (filteredTraces.length === 0) return "—"
    const totalDuration = filteredTraces.reduce((sum, trace) => sum + trace.duration_ms, 0)
    return fmtDur(totalDuration / filteredTraces.length)
  }, [filteredTraces])
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
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <StatCard label="Entrypoints" value={entrypoints.length} icon={ENTRYPOINTS_ICON} />
          <StatCard label="Traces" value={filteredTraces.length} icon={TRACES_ICON} />
          <StatCard label="Avg duration" value={avgDuration} icon={AVG_DURATION_ICON} />
          <StatCard
            label="Last trace"
            value={filteredTraces.length > 0 ? fmtRelativeTime(filteredTraces[0].start_time_ms, nowMs) : "—"}
            title={filteredTraces.length > 0 ? fmtTime(filteredTraces[0].start_time_ms) : undefined}
            icon={LAST_TRACE_ICON}
          />
        </div>

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
})

export default App
