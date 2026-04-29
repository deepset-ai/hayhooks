import { createContext, useContext } from "react"
import type { Context } from "react"

import type { UseTracesResult } from "./useTraces"

type TraceData = Pick<UseTracesResult, "entrypoints" | "traces"> & {
  slowComponentMinDurationMs: number
}
type TraceFreshness = Pick<UseTracesResult, "freshUntil">
type TraceStatus = Pick<UseTracesResult, "updatedAt" | "error" | "refreshing" | "clearing">
type TraceActions = Pick<UseTracesResult, "refresh" | "clear">

export const TraceDataContext = createContext<TraceData | null>(null)
export const TraceFreshnessContext = createContext<TraceFreshness | null>(null)
export const TraceStatusContext = createContext<TraceStatus | null>(null)
export const TraceActionsContext = createContext<TraceActions | null>(null)

function useRequiredContext<T>(context: Context<T | null>, hookName: string): T {
  const value = useContext(context)
  if (value === null) {
    throw new Error(`${hookName} must be used within TracesProvider`)
  }
  return value
}

export function useTraceData(): TraceData {
  return useRequiredContext(TraceDataContext, "useTraceData")
}

export function useTraceFreshness(): TraceFreshness {
  return useRequiredContext(TraceFreshnessContext, "useTraceFreshness")
}

export function useTraceStatus(): TraceStatus {
  return useRequiredContext(TraceStatusContext, "useTraceStatus")
}

export function useTraceActions(): TraceActions {
  return useRequiredContext(TraceActionsContext, "useTraceActions")
}
