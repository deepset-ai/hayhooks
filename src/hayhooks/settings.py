from enum import Enum
from pathlib import Path
from typing import Literal

from dotenv import find_dotenv, load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from hayhooks.server.logger import log

load_dotenv(dotenv_path=find_dotenv(usecwd=True))

# NOTE: We intentionally do not set HAYSTACK_DESERIALIZATION_ALLOWLIST. Haystack v3 gates pipeline
# deserialization behind a module allowlist, and since Hayhooks deserializes operator-supplied YAML we
# keep Haystack's secure default. Operators configure the allowlist (or opt out with "*") via the env var.


APP_TITLE = "Hayhooks"
APP_DESCRIPTION = "Hayhooks makes it easy to deploy and serve Haystack pipelines as REST APIs or MCP Tools"


class DeployConcurrencyPolicy(str, Enum):
    """Controls how runtime deploy/undeploy operations are synchronized."""

    SERIALIZED = "serialized"
    PARALLEL = "parallel"


class StartupDeployStrategy(str, Enum):
    """Controls how pipelines are deployed at startup."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


class AppSettings(BaseSettings):
    # Root path for the FastAPI app
    root_path: str = ""

    # Path to the directory containing the pipelines
    # Default to project root / pipelines
    pipelines_dir: str = str(Path.cwd() / "pipelines")

    # Additional Python path to be added to the Python path
    additional_python_path: str = ""

    # Hayhooks Host
    host: str = "localhost"

    # Hayhooks Port
    port: int = 1416

    # Max seconds to wait for in-flight connections to drain on shutdown (Ctrl-C)
    # before they are cancelled. Bounds graceful shutdown so long-lived
    # connections (e.g. the dashboard SSE stream) don't block server exit
    # indefinitely. Passed to uvicorn as ``timeout_graceful_shutdown``.
    graceful_shutdown_timeout: int = Field(default=5, ge=0, le=300)

    # Whether to use HTTPS when running CLI commands
    # Example: `hayhooks status`
    # NOTE: This is NOT used to specify the protocol for the uvicorn server
    use_https: bool = False

    # Host for the MCP app
    mcp_host: str = "localhost"

    # Port for the MCP app
    mcp_port: int = 1417

    # Host for the A2A app
    a2a_host: str = "localhost"

    # Port for the A2A app
    a2a_port: int = 1418

    # Base URL advertised in A2A agent cards (e.g. when behind a reverse proxy).
    # When empty, defaults to http://{a2a_host}:{a2a_port}
    a2a_external_url: str = ""

    # Accept A2A spec 0.3 requests on the same endpoints (many clients and
    # tools, e.g. the a2a-inspector, still speak 0.3 during the 1.0 transition)
    a2a_v0_3_compat: bool = True

    # Disable SSL verification when making requests from the CLI
    disable_ssl: bool = False

    # Files to ignore when reading pipeline files from a directory
    files_to_ignore_patterns: list[str] = ["*.pyc", "*.pyo", "*.pyd", "__pycache__", "*.so", "*.egg", "*.egg-info"]

    # Show tracebacks on errors during pipeline execution and deployment
    show_tracebacks: bool = False

    # Default streaming components configuration
    # Can be:
    # - Empty string (default): enable stream only for the LAST capable component
    # - "all": enable stream for ALL capable components
    # - Comma-separated list: "llm_1,llm_2" to enable stream for specific components
    streaming_components: str = ""

    # Deploy concurrency policy for runtime (API/MCP) deploy/undeploy operations.
    # "serialized" (default): one deploy/undeploy at a time (safe, predictable).
    # "parallel": allow concurrent deploy/undeploy (higher throughput, higher risk).
    deploy_concurrency: DeployConcurrencyPolicy = DeployConcurrencyPolicy.SERIALIZED

    # Strategy for deploying pipelines at startup from HAYHOOKS_PIPELINES_DIR.
    # "sequential": deploy one pipeline at a time (original behavior).
    # "parallel" (default): prepare pipelines in parallel, then commit serially.
    startup_deploy_strategy: StartupDeployStrategy = StartupDeployStrategy.PARALLEL

    # Max worker threads for parallel startup deployment (only used when
    # startup_deploy_strategy is "parallel"). Defaults to 4.
    startup_deploy_workers: int = Field(default=4, gt=0, le=32)

    # CORS Settings
    cors_allow_origins: list[str] = ["*"]
    cors_allow_methods: list[str] = ["*"]
    cors_allow_headers: list[str] = ["*"]
    cors_allow_credentials: bool = False
    cors_allow_origin_regex: str | None = None
    cors_expose_headers: list[str] = ["X-Hayhooks-Trace-Cursor"]
    cors_max_age: int = 600

    # Stdlib loggers to intercept and route through loguru.
    # Only these loggers are patched; everything else (httpx, haystack, etc.)
    # keeps its default behaviour.
    intercepted_loggers: list[str] = ["uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"]
    # Access-log path prefixes to suppress from uvicorn.access output.
    # Defaults hide noisy dashboard polling endpoints.
    access_log_excluded_path_prefixes: list[str] = [
        "/dashboard/api/config",
        "/dashboard/api/entrypoints",
        "/dashboard/api/traces",
    ]

    # Exclude low-level ASGI send/receive spans from framework instrumentation
    # to keep streaming traces readable by default.
    tracing_excluded_spans: list[Literal["send", "receive"]] = ["send", "receive"]

    # Dashboard trace API settings (in-process live buffer)
    dashboard_trace_default_limit: int = Field(default=25, gt=0, le=200)
    dashboard_trace_max_limit: int = Field(default=100, gt=0, le=500)
    dashboard_trace_buffer_capacity: int = Field(default=200, gt=0, le=10_000)
    dashboard_trace_include_haystack_spans: bool = True
    dashboard_ui_poll_ms: int = Field(default=2500, ge=250, le=60_000)
    dashboard_ui_list_cap: int = Field(default=100, gt=0, le=500)
    dashboard_ui_fetch_limit: int = Field(default=50, gt=0, le=500)
    dashboard_ui_fresh_ms: int = Field(default=6000, ge=0, le=60_000)
    dashboard_ui_slow_component_min_duration_ms: int = Field(default=1000, gt=0)
    # Real-time trace streaming (SSE). When enabled, the dashboard receives trace
    # updates over a Server-Sent Events stream instead of polling; polling stays
    # as an automatic client-side fallback. Heartbeat keeps the connection alive
    # through proxies during idle periods.
    dashboard_stream_enabled: bool = True
    dashboard_stream_heartbeat_ms: int = Field(default=15_000, ge=1_000, le=120_000)
    dashboard_enabled: bool = False
    dashboard_path: str = "/dashboard"
    dashboard_dist_dir: str = str(Path.cwd() / "dashboard" / "dist")
    dashboard_trace_include_payload_values: bool = False

    # Chainlit Settings
    # Enable embedded Chainlit UI frontend
    chainlit_enabled: bool = False
    # URL path where Chainlit UI will be mounted
    chainlit_path: str = "/chat"
    # Custom Chainlit app file (optional, uses default if not set)
    chainlit_app: str = ""
    # Default pipeline/model to auto-select in the Chainlit UI (empty = auto-select if only one)
    chainlit_default_model: str = ""
    # Timeout (seconds) for chat completion requests from the Chainlit UI
    chainlit_request_timeout: float = 120.0
    # Directory containing custom .jsx element files for the Chainlit UI
    chainlit_custom_elements_dir: str = ""

    _WILDCARD_HOSTS = {"0.0.0.0", "::"}  # noqa: S104

    @property
    def chainlit_base_url(self) -> str:
        """Base URL for the Chainlit UI to reach the Hayhooks backend."""
        protocol = "https" if self.use_https else "http"
        host = "127.0.0.1" if self.host in self._WILDCARD_HOSTS else self.host
        base = f"{protocol}://{host}:{self.port}"
        if self.root_path:
            base = f"{base}/{self.root_path.strip('/')}"
        return base

    # Prefix for the environment variables to avoid conflicts
    # with other similar environment variables
    model_config = SettingsConfigDict(env_prefix="hayhooks_")


settings = AppSettings()


def check_cors_settings():
    """
    Check if the CORS settings are set to the default values.
    """
    if (
        settings.cors_allow_origins == ["*"]
        and settings.cors_allow_methods == ["*"]
        and settings.cors_allow_headers == ["*"]
    ):
        log.warning("Using default CORS settings - All origins, methods, and headers are allowed.")
