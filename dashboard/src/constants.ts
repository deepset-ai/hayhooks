import type { DashboardConfig, TraceKind } from "./types"

export const DEFAULT_DASHBOARD_CONFIG: DashboardConfig = {
  pollMs: 2500,
  listCap: 100,
  fetchLimit: 50,
  freshMs: 6000,
  slowComponentMinDurationMs: 1000,
}

export const TAG_PRIORITY = [
  "hayhooks.pipeline.name",
  "hayhooks.transport",
  "hayhooks.openai.operation",
  "hayhooks.openai.stream_requested",
  "hayhooks.openai.execution_mode",
  "hayhooks.response.stream_type",
  "hayhooks.response.streaming",
  "hayhooks.success",
  "hayhooks.error.type",
  "hayhooks.http.status_code",
  "service.name",
  "serviceName",
]

export const TAG_LABELS: Record<string, string> = {
  "hayhooks.pipeline.name": "pipeline",
  "hayhooks.transport": "transport",
  "hayhooks.openai.operation": "openai op",
  "hayhooks.openai.stream_requested": "stream",
  "hayhooks.openai.execution_mode": "exec mode",
  "hayhooks.response.stream_type": "stream type",
  "hayhooks.response.streaming": "streaming",
  "hayhooks.success": "success",
  "hayhooks.error.type": "error",
  "hayhooks.http.status_code": "http",
  "hayhooks.deploy.strategy": "deploy",
  "hayhooks.deploy.save_files": "save files",
  "hayhooks.deploy.file_count": "files",
  "hayhooks.deploy.overwrite": "overwrite",
  "hayhooks.route": "route",
  "hayhooks.payload.values": "payload values",
  "hayhooks.payload.has_files": "has files",
  "service.name": "service",
  serviceName: "service",
}

export const SUMMARY_TAG_KEYS = new Set([
  "hayhooks.transport",
  "hayhooks.success",
  "hayhooks.error.type",
])

export const KIND_STYLE: Record<TraceKind, { label: string; badge: string; border: string }> = {
  deploy: {
    label: "deploy",
    badge: "bg-kind-deploy-soft text-kind-deploy border-kind-deploy-border",
    border: "border-l-kind-deploy",
  },
  undeploy: {
    label: "undeploy",
    badge: "bg-muted text-muted-foreground border-border",
    border: "border-l-muted-foreground",
  },
  run: {
    label: "run",
    badge: "bg-primary/10 text-primary border-primary/20",
    border: "border-l-primary",
  },
  openai: {
    label: "openai",
    badge: "bg-kind-openai-soft text-kind-openai border-kind-openai-border",
    border: "border-l-kind-openai",
  },
  mcp: {
    label: "mcp",
    badge: "bg-kind-mcp-soft text-kind-mcp border-kind-mcp-border",
    border: "border-l-kind-mcp",
  },
  other: {
    label: "trace",
    badge: "bg-muted text-muted-foreground border-border",
    border: "",
  },
}
