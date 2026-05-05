import { DEFAULT_DASHBOARD_CONFIG } from "../constants"
import type { DashboardConfig, DashboardConfigResponse } from "../types"

/**
 * Parse a single config integer from raw API response.
 * Returns the default when the value is missing, not a finite number,
 * or below the inclusive minimum.
 */
function parseConfigInt(
  rawConfig: Partial<DashboardConfigResponse>,
  key: keyof DashboardConfigResponse,
  defaultValue: number,
  minInclusive: number,
): number {
  const value = rawConfig[key]
  if (typeof value !== "number" || !Number.isFinite(value)) return defaultValue
  const rounded = Math.round(value)
  return rounded < minInclusive ? defaultValue : rounded
}

export function normalizeDashboardConfig(raw: unknown): DashboardConfig {
  if (typeof raw !== "object" || raw === null) return DEFAULT_DASHBOARD_CONFIG

  const config = raw as Partial<DashboardConfigResponse>

  const listCap =         parseConfigInt(config, "list_cap",         DEFAULT_DASHBOARD_CONFIG.listCap, 1)
  const fetchLimit =      parseConfigInt(config, "fetch_limit",      DEFAULT_DASHBOARD_CONFIG.fetchLimit, 1)
  const pollMs =          parseConfigInt(config, "poll_ms",          DEFAULT_DASHBOARD_CONFIG.pollMs, 250)
  const freshMs =         parseConfigInt(config, "fresh_ms",         DEFAULT_DASHBOARD_CONFIG.freshMs, 0)
  const slowComponentMs = parseConfigInt(config, "slow_component_min_duration_ms", DEFAULT_DASHBOARD_CONFIG.slowComponentMinDurationMs, 1)

  return {
    pollMs,
    listCap,
    fetchLimit: Math.min(fetchLimit, listCap),
    freshMs,
    slowComponentMinDurationMs: slowComponentMs,
  }
}
