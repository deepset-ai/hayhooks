import { useEffect, useState } from "react"
import { fetchDashboardConfig, resolveApiBase } from "../api"
import { DEFAULT_DASHBOARD_CONFIG } from "../constants"
import type { DashboardConfig } from "../types"

export function useDashboardConfig() {
  const [config, setConfig] = useState<DashboardConfig>(DEFAULT_DASHBOARD_CONFIG)

  useEffect(() => {
    let cancelled = false
    const base = resolveApiBase()

    const loadConfig = async () => {
      try {
        const nextConfig = await fetchDashboardConfig(base)
        if (!cancelled) setConfig(nextConfig)
      } catch (error: unknown) {
        if (import.meta.env.DEV) {
          console.warn("Failed to load dashboard config, using defaults.", error)
        }
      }
    }

    void loadConfig()

    return () => { cancelled = true }
  }, [])

  return config
}
