import { useMemo } from "react"
import type { ReactNode } from "react"

import { useDashboardConfig } from "./useDashboardConfig"
import { useTraces } from "./useTraces"
import {
  TraceActionsContext,
  TraceDataContext,
  TraceStatusContext,
} from "./useTracesContext"

export function TracesProvider({ children }: { children: ReactNode }) {
  const config = useDashboardConfig()
  const traces = useTraces(config)

  const dataValue = useMemo(
    () => ({
      entrypoints: traces.entrypoints,
      traces: traces.traces,
      freshUntil: traces.freshUntil,
    }),
    [traces.entrypoints, traces.traces, traces.freshUntil],
  )

  const statusValue = useMemo(
    () => ({
      updatedAt: traces.updatedAt,
      error: traces.error,
      refreshing: traces.refreshing,
      clearing: traces.clearing,
    }),
    [traces.updatedAt, traces.error, traces.refreshing, traces.clearing],
  )

  const actionsValue = useMemo(
    () => ({
      refresh: traces.refresh,
      clear: traces.clear,
    }),
    [traces.refresh, traces.clear],
  )

  return (
    <TraceActionsContext.Provider value={actionsValue}>
      <TraceStatusContext.Provider value={statusValue}>
        <TraceDataContext.Provider value={dataValue}>
          {children}
        </TraceDataContext.Provider>
      </TraceStatusContext.Provider>
    </TraceActionsContext.Provider>
  )
}
