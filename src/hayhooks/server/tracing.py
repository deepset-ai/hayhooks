"""
OpenTelemetry / Haystack tracing integration for Hayhooks.

This module centralizes:
  - Span-name constants used throughout the server.
  - Opt-in FastAPI / Starlette OpenTelemetry instrumentation (via Haystack's
    ``LazyImport`` so the ``tracing`` extra stays optional).
  - A small :func:`trace_operation` context manager that every Hayhooks
    domain span flows through, with consistent success / error / timing tags
    and Loguru log correlation.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator, Generator, Iterator, Mapping
from contextlib import contextmanager, nullcontext
from contextvars import copy_context
from time import monotonic
from typing import Any

from fastapi import HTTPException
from haystack.lazy_imports import LazyImport
from haystack.tracing import OpenTelemetryTracer, auto_enable_tracing, enable_tracing, is_tracing_enabled, tracer

from hayhooks.server.logger import log, normalize_trace_correlation_data
from hayhooks.settings import settings

_TRACING_EXTRA_INSTALL_CMD = 'pip install "hayhooks[tracing]"'
_LAZY_IMPORT_HINT = f"Run '{_TRACING_EXTRA_INSTALL_CMD}' to install tracing support."
_INSTALL_HINT = f"Install with '{_TRACING_EXTRA_INSTALL_CMD}'."

SPAN_PIPELINE_DEPLOY = "hayhooks.pipeline.deploy"
SPAN_PIPELINE_DEPLOY_PREPARE = "hayhooks.pipeline.deploy.prepare"
SPAN_PIPELINE_DEPLOY_COMMIT = "hayhooks.pipeline.deploy.commit"
SPAN_PIPELINE_UNDEPLOY = "hayhooks.pipeline.undeploy"
SPAN_PIPELINE_RUN = "hayhooks.pipeline.run"
SPAN_PIPELINE_STARTUP_DEPLOY = "hayhooks.pipeline.startup.deploy"
SPAN_OPENAI_RUN = "hayhooks.openai.run"
SPAN_OPENAI_FILE_UPLOAD = "hayhooks.openai.file_upload"
SPAN_MCP_LIST_TOOLS = "hayhooks.mcp.list_tools"
SPAN_MCP_CALL_TOOL = "hayhooks.mcp.call_tool"
SPAN_MCP_RUN_PIPELINE_TOOL = "hayhooks.mcp.run_pipeline_tool"

_TAG_SUCCESS = "hayhooks.success"
_TAG_ERROR_TYPE = "hayhooks.error.type"
_TAG_ELAPSED_MS = "hayhooks.elapsed_ms"
_TAG_HTTP_STATUS = "hayhooks.http.status_code"
_TAG_RESPONSE_STREAMING = "hayhooks.response.streaming"
_TAG_RESPONSE_STREAM_TYPE = "hayhooks.response.stream_type"

_FASTAPI_STATE_FLAG = "_hayhooks_fastapi_instrumented"
_STARLETTE_STATE_FLAG = "_hayhooks_starlette_instrumented"
_DEFAULT_SERVICE_NAME = "hayhooks"
_OTLP_HTTP_PROTOBUF = "http/protobuf"
_OTLP_GRPC = "grpc"
_OTLP_ENDPOINT_ENV_KEYS = ("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "OTEL_EXPORTER_OTLP_ENDPOINT")
_OTLP_PROTOCOL_ENV_KEYS = ("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "OTEL_EXPORTER_OTLP_PROTOCOL")

with LazyImport(_LAZY_IMPORT_HINT) as fastapi_tracing_import:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor  # ty: ignore[unresolved-import]

with LazyImport(_LAZY_IMPORT_HINT) as starlette_tracing_import:
    from opentelemetry.instrumentation.starlette import StarletteInstrumentor  # ty: ignore[unresolved-import]

with LazyImport(_LAZY_IMPORT_HINT) as otel_sdk_import:
    from opentelemetry import trace as otel_trace  # ty: ignore[unresolved-import]
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource  # ty: ignore[unresolved-import]
    from opentelemetry.sdk.trace import TracerProvider  # ty: ignore[unresolved-import]
    from opentelemetry.sdk.trace.export import BatchSpanProcessor  # ty: ignore[unresolved-import]

with LazyImport(_LAZY_IMPORT_HINT) as otlp_http_exporter_import:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # ty: ignore[unresolved-import]
        OTLPSpanExporter as OTLPHTTPSpanExporter,
    )

with LazyImport(_LAZY_IMPORT_HINT) as otlp_grpc_exporter_import:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # ty: ignore[unresolved-import]
        OTLPSpanExporter as OTLPGRPCSpanExporter,
    )


def _first_set_env(*keys: str) -> str | None:
    for key in keys:
        if value := os.getenv(key):
            return value
    return None


def _normalize_otlp_protocol(raw_protocol: str | None) -> str | None:
    if raw_protocol is None:
        return _OTLP_HTTP_PROTOBUF

    normalized = raw_protocol.strip().lower()
    if normalized in {"http", _OTLP_HTTP_PROTOBUF}:
        return _OTLP_HTTP_PROTOBUF
    if normalized == _OTLP_GRPC:
        return _OTLP_GRPC
    return None


def _load_fastapi_instrumentor() -> type[Any] | None:
    """Return FastAPI OTel instrumentor when tracing extras are installed."""
    try:
        fastapi_tracing_import.check()
    except ImportError:
        return None
    return FastAPIInstrumentor


def _load_starlette_instrumentor() -> type[Any] | None:
    """Return Starlette OTel instrumentor when tracing extras are installed."""
    try:
        starlette_tracing_import.check()
    except ImportError:
        return None
    return StarletteInstrumentor


def _build_otlp_span_exporter(protocol: str) -> Any | None:
    if protocol == _OTLP_HTTP_PROTOBUF:
        try:
            otlp_http_exporter_import.check()
        except ImportError:
            log.warning(
                "OTLP HTTP exporter unavailable. {} "
                "Reinstall tracing extras to ensure OTLP exporter dependencies are installed.",
                _INSTALL_HINT,
            )
            return None
        # Let the exporter resolve endpoint/path from standard OTEL env vars.
        # This avoids accidentally overriding path handling (e.g. /v1/traces).
        return OTLPHTTPSpanExporter()

    if protocol == _OTLP_GRPC:
        try:
            otlp_grpc_exporter_import.check()
        except ImportError:
            log.warning(
                "OTLP gRPC exporter unavailable. {} Install with: pip install opentelemetry-exporter-otlp-proto-grpc",
                _INSTALL_HINT,
            )
            return None
        return OTLPGRPCSpanExporter()

    log.warning(
        "OTLP protocol '{}' is not supported by Hayhooks built-in tracing bootstrap. "
        "Supported protocols are '{}' and '{}'.",
        protocol,
        _OTLP_HTTP_PROTOBUF,
        _OTLP_GRPC,
    )
    return None


def _configure_otel_tracer_from_env() -> bool:
    """
    Bootstrap OpenTelemetry tracing when OTLP endpoint vars are present.

    This gives Hayhooks a sensible out-of-the-box path for local/self-hosted
    tracing setups without requiring users to write explicit SDK bootstrap code.
    """
    endpoint = _first_set_env(*_OTLP_ENDPOINT_ENV_KEYS)
    if endpoint is None:
        return False

    raw_protocol = _first_set_env(*_OTLP_PROTOCOL_ENV_KEYS)
    protocol = _normalize_otlp_protocol(raw_protocol)
    if protocol is None:
        log.warning(
            "OTLP protocol '{}' is not supported by Hayhooks built-in tracing bootstrap. "
            "Supported values are: {}, {}, {}.",
            raw_protocol,
            "http",
            _OTLP_HTTP_PROTOBUF,
            _OTLP_GRPC,
        )
        return False

    try:
        otel_sdk_import.check()
    except ImportError:
        log.warning(
            "OpenTelemetry OTLP runtime unavailable. {} "
            "If you upgraded Hayhooks, reinstall extras to pull new tracing dependencies.",
            _INSTALL_HINT,
        )
        return False

    exporter = _build_otlp_span_exporter(protocol)
    if exporter is None:
        return False

    try:
        service_name = os.getenv("OTEL_SERVICE_NAME", _DEFAULT_SERVICE_NAME)
        provider = TracerProvider(resource=Resource.create({SERVICE_NAME: service_name}))
        provider.add_span_processor(BatchSpanProcessor(exporter))
        otel_trace.set_tracer_provider(provider)
        enable_tracing(OpenTelemetryTracer(otel_trace.get_tracer(service_name)))
    except Exception as exc:  # pragma: no cover - defensive guard for SDK/runtime failures
        log.warning("Failed to bootstrap OpenTelemetry tracing from environment: {}", exc)
        return False

    log.info("Hayhooks tracing enabled with OTLP {} exporter endpoint '{}'", protocol, endpoint)
    return True


def configure_tracing() -> bool:
    """
    Ensure Haystack tracing is enabled when possible.

    Order:
      1. No-op if tracing is already enabled.
      2. Try Haystack's ``auto_enable_tracing()`` for externally configured providers.
      3. Fallback to Hayhooks OTLP bootstrap when OTLP env vars are present.
    """
    if is_tracing_enabled():
        return True

    auto_enable_tracing()
    if is_tracing_enabled():
        log.debug("Hayhooks tracing enabled by Haystack auto-configuration")
        return True

    return _configure_otel_tracer_from_env()


def _instrument_app(
    app: Any,
    *,
    state_attr: str,
    instrumentor: type[Any] | None,
    framework_name: str,
) -> bool:
    """Attach *instrumentor* to *app* exactly once. Returns ``True`` on success."""
    if getattr(app.state, state_attr, False):
        return True

    if instrumentor is None:
        log.debug("OpenTelemetry {} instrumentation unavailable. {}", framework_name, _INSTALL_HINT)
        return False

    try:
        instrument_kwargs: dict[str, Any] = {}
        if settings.tracing_excluded_spans:
            instrument_kwargs["exclude_spans"] = settings.tracing_excluded_spans
        try:
            instrumentor.instrument_app(app, **instrument_kwargs)
        except TypeError as exc:
            if instrument_kwargs and "exclude_spans" in str(exc):
                log.debug(
                    "{} instrumentation does not support exclude_spans; retrying without excluded span filtering",
                    framework_name,
                )
                instrumentor.instrument_app(app)
            else:
                raise
    except Exception as exc:  # pragma: no cover - defensive guard for third-party errors
        log.warning("Failed to instrument {} app with OpenTelemetry: {}", framework_name, exc)
        return False

    setattr(app.state, state_attr, True)
    log.debug("OpenTelemetry {} instrumentation enabled", framework_name)
    return True


def instrument_fastapi_app(app: Any) -> bool:
    """
    Enable OpenTelemetry FastAPI instrumentation for *app* if available.

    Returns ``True`` when instrumentation is enabled or already active.
    """
    return _instrument_app(
        app,
        state_attr=_FASTAPI_STATE_FLAG,
        instrumentor=_load_fastapi_instrumentor(),
        framework_name="FastAPI",
    )


def instrument_starlette_app(app: Any) -> bool:
    """
    Enable OpenTelemetry Starlette instrumentation for *app* if available.

    Returns ``True`` when instrumentation is enabled or already active.
    """
    return _instrument_app(
        app,
        state_attr=_STARLETTE_STATE_FLAG,
        instrumentor=_load_starlette_instrumentor(),
        framework_name="Starlette",
    )


def build_trace_tags(tags: Mapping[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
    """
    Merge tag mappings and drop keys whose value is ``None``.

    ``None`` values are skipped so optional attributes don't pollute spans with
    empty fields. Keyword arguments override values from ``tags``.
    """
    merged: dict[str, Any] = {**(tags or {}), **kwargs}
    return {key: value for key, value in merged.items() if value is not None}


def build_streaming_trace_tags(tags: Mapping[str, Any] | None = None, *, stream_type: str) -> dict[str, Any]:
    return build_trace_tags(tags, **{_TAG_RESPONSE_STREAMING: True, _TAG_RESPONSE_STREAM_TYPE: stream_type})


def get_trace_log_context() -> dict[str, str]:
    """
    Return normalized trace correlation values for Loguru ``contextualize``.

    Returns an empty dict when tracing is disabled or correlation data is unavailable.
    """
    current_span = tracer.current_span()
    if current_span is None:
        return {}

    try:
        correlation_data = current_span.get_correlation_data_for_logs()
    except Exception:  # pragma: no cover - defensive guard for third-party tracer errors
        return {}

    return normalize_trace_correlation_data(correlation_data) if correlation_data else {}


def _mark_success(span: Any) -> None:
    span.set_tag(_TAG_SUCCESS, value=True)


def _mark_failure(span: Any, exc: BaseException) -> None:
    span.set_tag(_TAG_SUCCESS, value=False)
    span.set_tag(_TAG_ERROR_TYPE, type(exc).__name__)


def _mark_http_exception(span: Any, exc: HTTPException) -> None:
    """
    Record an HTTPException outcome.

    Domain-operation spans treat every HTTPException as a failure while still
    recording the HTTP status code for debugging and transport correlation.
    """
    span.set_tag(_TAG_HTTP_STATUS, exc.status_code)
    _mark_failure(span, exc)


class _OperationTrace:
    def __init__(self, operation_name: str, *, tags: Mapping[str, Any] | None = None):
        self._span_cm = tracer.trace(operation_name, tags=dict(tags or {}))
        self._span: Any | None = None
        self._log_context_cm: Any | None = None
        self._started = 0.0
        self._finished = False

    @property
    def span(self) -> Any:
        return self.start()

    def start(self) -> Any:
        if self._span is not None:
            return self._span

        self._span = self._span_cm.__enter__()
        self._started = monotonic()
        trace_context = get_trace_log_context()
        self._log_context_cm = log.contextualize(**trace_context) if trace_context else nullcontext()
        self._log_context_cm.__enter__()
        return self._span

    def finish(self, exc: BaseException | None = None) -> None:
        span = self.start()
        if self._finished:
            return

        try:
            if exc is None:
                _mark_success(span)
            elif isinstance(exc, HTTPException):
                _mark_http_exception(span, exc)
            else:
                _mark_failure(span, exc)
        finally:
            span.set_tag(_TAG_ELAPSED_MS, (monotonic() - self._started) * 1000)
            exc_type = type(exc) if exc is not None else None
            exc_tb = exc.__traceback__ if exc is not None else None
            if self._log_context_cm is not None:
                self._log_context_cm.__exit__(exc_type, exc, exc_tb)
            self._span_cm.__exit__(exc_type, exc, exc_tb)
            self._finished = True


def start_trace_operation(
    operation_name: str,
    *,
    tags: Mapping[str, Any] | None = None,
) -> _OperationTrace:
    operation = _OperationTrace(operation_name, tags=tags)
    operation.start()
    return operation


def trace_sync_stream(
    stream: Generator[Any, None, None],
    operation_name: str,
    *,
    tags: Mapping[str, Any] | None = None,
) -> Generator[Any, None, None]:
    operation = _OperationTrace(operation_name, tags=tags)
    stream_context = copy_context()

    def traced_stream() -> Generator[Any, None, None]:
        stream_context.run(operation.start)
        try:
            while True:
                try:
                    item = stream_context.run(next, stream)
                except StopIteration:
                    stream_context.run(operation.finish)
                    return
                except BaseException as exc:
                    stream_context.run(operation.finish, exc)
                    raise
                else:
                    yield item
        except BaseException as exc:
            stream_context.run(operation.finish, exc)
            raise
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                stream_context.run(close)

    return traced_stream()


def trace_async_stream(
    stream: AsyncGenerator[Any, None],
    operation_name: str,
    *,
    tags: Mapping[str, Any] | None = None,
) -> AsyncGenerator[Any, None]:
    async def traced_stream() -> AsyncGenerator[Any, None]:
        try:
            with trace_operation(operation_name, tags=tags):
                async for item in stream:
                    yield item
        finally:
            aclose = getattr(stream, "aclose", None)
            if callable(aclose):
                await aclose()

    return traced_stream()


@contextmanager
def trace_operation(
    operation_name: str,
    *,
    tags: Mapping[str, Any] | None = None,
) -> Iterator[Any]:
    """
    Wrap a block in a Haystack span and attach standard Hayhooks attributes.

    On exit the span is tagged with:
      - ``hayhooks.success`` (``bool``)
      - ``hayhooks.error.type`` (only on failure)
      - ``hayhooks.http.status_code`` (only for :class:`HTTPException`)
      - ``hayhooks.elapsed_ms`` (wall-clock duration)

    The wrapped block also inherits ``trace_id`` / ``span_id`` in its Loguru
    log context so logs emitted inside it can be correlated with the span.
    """
    operation = start_trace_operation(operation_name, tags=tags)
    try:
        yield operation.span
    except BaseException as exc:
        operation.finish(exc)
        raise
    else:
        operation.finish()
