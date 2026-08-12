"""Runtime-owned durable deployment services for wrappers and A2A projections."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import uuid
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Protocol, cast, get_type_hints

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from hayhooks.durable.adapters import HaystackDurableAdapter, _run_fenced_thread, execution_kind
from hayhooks.durable.backend import ExecutionIdempotencyConflictError
from hayhooks.durable.context import DurableContext
from hayhooks.durable.manager import DurableExecutionManager
from hayhooks.durable.mode import DurableAuthoringMode, _durable_method_implementations, durable_authoring_mode
from hayhooks.durable.models import ExecutionKind, ExecutionRecord, JsonValue, json_safe, validate_json
from hayhooks.durable.store import ExecutionStore, InMemoryExecutionStoreProvider, RedisExecutionStoreProvider
from hayhooks.server.exceptions import PipelineWrapperError
from hayhooks.server.logger import log
from hayhooks.server.pipelines.registry import registry
from hayhooks.server.tracing import SPAN_DURABLE_ATTEMPT, SPAN_DURABLE_SUBMIT, build_trace_tags, trace_operation
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper
from hayhooks.settings import AppSettings, settings


class ExecutionStoreProvider(Protocol):
    """Create application-owned stores for durable deployments."""

    def create_execution_store(self, deployment_name: str) -> ExecutionStore: ...

    async def close(self) -> None: ...


def _runtime_settings(provider: ExecutionStoreProvider | None, app_settings: AppSettings | None) -> AppSettings | None:
    provider_settings = getattr(provider, "app_settings", None)
    if isinstance(provider_settings, AppSettings):
        if app_settings is not None and app_settings != provider_settings:
            msg = "Durable runtime and execution-store provider settings must match"
            raise ValueError(msg)
        app_settings = provider_settings
    return app_settings.model_copy(deep=True) if app_settings is not None else None


class DurableDeployment:
    """One deployment's records, manager, validated callable, and adapter."""

    def __init__(
        self,
        name: str,
        wrapper: BasePipelineWrapper,
        provider: ExecutionStoreProvider,
        *,
        app_settings: AppSettings | None = None,
    ) -> None:
        self.name = name
        self.wrapper = wrapper
        self.app_settings = _runtime_settings(provider, app_settings) or settings.model_copy(deep=True)
        pipeline = wrapper.pipeline
        try:
            kind = execution_kind(pipeline)
        except TypeError as error:
            raise PipelineWrapperError(str(error)) from error
        self.authoring_mode = durable_authoring_mode(wrapper)
        self.builtin_agent = kind is ExecutionKind.AGENT and self.authoring_mode is DurableAuthoringMode.MANAGED_AGENT
        if self.builtin_agent:
            self.method = None
            self.is_async = True
            self.request_type = _DurableAgentRequest
            self.result_type = Any
        else:
            self.method, self.is_async, self.request_type, self.result_type = _durable_method_contract(wrapper)
        self.result_adapter = TypeAdapter(self.result_type) if self.result_type is not Any else None
        self.resume_type = getattr(wrapper, "durable_resume_model", None)
        if self.resume_type is not None and (
            not inspect.isclass(self.resume_type) or not issubclass(self.resume_type, BaseModel)
        ):
            msg = "durable_resume_model must be a Pydantic model class or None"
            raise PipelineWrapperError(msg)
        revision = getattr(wrapper, "durable_revision", None)
        if not isinstance(revision, str) or not revision.strip():
            msg = "Durable wrappers must declare a non-empty durable_revision"
            raise PipelineWrapperError(msg)
        self.kind = kind
        self.revision = revision.strip()
        self.adapter = HaystackDurableAdapter(pipeline, kind)
        self.store = provider.create_execution_store(name)
        self.store.set_definition_revision(self.revision)
        self.manager = DurableExecutionManager(
            name,
            self.store,
            self._run,
            self.adapter,
            concurrency=self.app_settings.durable_execution_concurrency,
            poll_interval=self.app_settings.durable_poll_interval,
            shutdown_grace_period=self.app_settings.durable_shutdown_grace_period,
            max_attempts=self.app_settings.durable_max_attempts,
            retry_base_delay=self.app_settings.durable_retry_base_delay,
            retry_max_delay=self.app_settings.durable_retry_max_delay,
        )

    async def start(self) -> None:
        """Prepare and activate this deployment's execution manager."""
        await self.manager.start()

    async def prepare(self) -> None:
        """Initialize the execution store without allowing the candidate to claim work."""
        await self.manager.prepare()

    def activate(self) -> None:
        """Start prepared workers at the deployment publication boundary."""
        self.manager.activate()

    def deactivate(self) -> None:
        """Reject new submissions while an active deployment is being replaced."""
        self.manager.deactivate()

    async def quiesce(self) -> None:
        """Close submission admission before this deployment is stopped."""
        await self.manager.quiesce()

    async def close(self) -> None:
        """Stop this deployment's workers and submission admission."""
        await self.manager.close()

    async def submit(
        self,
        payload: Mapping[str, Any],
        *,
        execution_id: str | None = None,
        owner_id: str | None = None,
    ) -> tuple[bool, ExecutionRecord]:
        """Validate and idempotently submit one durable execution."""
        if not self.manager.accepting:
            msg = f"Durable deployment '{self.name}' is not accepting submissions"
            raise RuntimeError(msg)
        request = self.request_type.model_validate(dict(payload))
        if execution_id is not None and owner_id is not None:
            execution_id = execution_id_for(owner_id, execution_id)
        execution_id = execution_id or uuid.uuid4().hex
        validated_input = cast(
            dict[str, JsonValue],
            validate_json(
                request.model_dump(mode="json"), limit=self.app_settings.durable_max_record_bytes, label="request"
            ),
        )
        fingerprint_input = cast(dict[str, JsonValue], _canonical_json(request.model_dump(mode="python")))
        operation_fingerprint = _operation_fingerprint(
            self.name,
            self.revision,
            fingerprint_input,
            owner_id=owner_id,
        )
        record = ExecutionRecord(
            execution_id=execution_id,
            execution_kind=self.kind,
            deployment_name=self.name,
            definition_revision=self.revision,
            validated_input=validated_input,
            operation_fingerprint=operation_fingerprint,
            owner_id=owner_id,
            max_progress_events=self.app_settings.durable_max_progress_events,
            max_record_bytes=self.app_settings.durable_max_record_bytes,
        )
        with trace_operation(
            SPAN_DURABLE_SUBMIT,
            tags=build_trace_tags(
                {
                    "hayhooks.pipeline.name": self.name,
                    "hayhooks.durable.execution_id": execution_id,
                    "hayhooks.durable.definition_revision": self.revision,
                }
            ),
        ) as span:
            try:
                created, persisted = await self.manager.submit_with_record(record)
            except ExecutionIdempotencyConflictError as error:
                msg = "Idempotency-Key was already used for a different durable operation"
                raise IdempotencyConflictError(msg) from error
            persisted = self._validated_record(
                execution_id,
                persisted,
                owner_id=owner_id,
                enforce_owner=owner_id is not None,
            )
            if not created and persisted.operation_fingerprint != operation_fingerprint:
                msg = "Idempotency-Key was already used for a different durable operation"
                raise IdempotencyConflictError(msg)
            span.set_tag("hayhooks.durable.idempotent_replay", not created)
            log.bind(
                deployment=self.name,
                execution_id=persisted.execution_id,
                revision=self.revision,
                kind=self.kind.value,
                created=created,
            ).debug("Accepted durable execution submission")
            return created, persisted

    async def get(
        self,
        execution_id: str,
        *,
        owner_id: str | None = None,
        enforce_owner: bool = False,
        allow_revision_mismatch: bool = False,
    ) -> ExecutionRecord:
        """Return one execution after owner and definition-revision validation."""
        record = await self.store.get(execution_id)
        return self._validated_record(
            execution_id,
            record,
            owner_id=owner_id,
            enforce_owner=enforce_owner,
            allow_revision_mismatch=allow_revision_mismatch,
        )

    async def request_cancel(
        self,
        execution_id: str,
        *,
        owner_id: str | None = None,
        enforce_owner: bool = False,
        reason: str | None = None,
    ) -> bool:
        """Request cooperative cancellation after validating record ownership."""
        await self.get(
            execution_id,
            owner_id=owner_id,
            enforce_owner=enforce_owner,
            allow_revision_mismatch=True,
        )
        accepted = await self.store.request_cancel(execution_id, reason)
        log.bind(deployment=self.name, execution_id=execution_id, accepted=accepted).debug(
            "Processed durable execution cancellation request"
        )
        return accepted

    async def resume(
        self,
        execution_id: str,
        update: JsonValue | None = None,
        *,
        owner_id: str | None = None,
        enforce_owner: bool = False,
    ) -> bool:
        """Validate and enqueue a resume update for a waiting execution."""
        await self.get(
            execution_id,
            owner_id=owner_id,
            enforce_owner=enforce_owner,
            allow_revision_mismatch=True,
        )
        if self.resume_type is not None and update is None:
            msg = f"Execution '{execution_id}' requires a resume request body"
            raise ValueError(msg)
        if self.resume_type is not None:
            update = cast(JsonValue, self.resume_type.model_validate(update).model_dump(mode="json"))
        resumed = await self.store.resume(execution_id, update)
        log.bind(deployment=self.name, execution_id=execution_id, resumed=resumed).debug(
            "Processed durable execution resume request"
        )
        return resumed

    def _validated_record(
        self,
        execution_id: str,
        record: ExecutionRecord | None,
        *,
        owner_id: str | None = None,
        enforce_owner: bool = False,
        allow_revision_mismatch: bool = False,
    ) -> ExecutionRecord:
        if record is None or record.deployment_name != self.name:
            raise KeyError(execution_id)
        if enforce_owner and record.owner_id != owner_id:
            raise KeyError(execution_id)
        if record.definition_revision != self.revision and not record.terminal and not allow_revision_mismatch:
            msg = (
                f"Durable execution '{execution_id}' was created for a different definition revision and cannot resume."
            )
            raise DefinitionRevisionConflictError(msg)
        return record

    async def _run(self, context: DurableContext) -> JsonValue:
        if context.record.definition_revision != self.revision:
            msg = (
                f"Durable execution '{context.execution_id}' was created for a different "
                "definition revision and cannot resume."
            )
            raise DefinitionRevisionConflictError(msg)
        with trace_operation(
            SPAN_DURABLE_ATTEMPT,
            tags=build_trace_tags(
                {
                    "hayhooks.pipeline.name": self.name,
                    "hayhooks.durable.execution_id": context.execution_id,
                    "hayhooks.durable.attempt": context.attempt,
                    "hayhooks.durable.kind": self.kind.value,
                    "hayhooks.durable.queue_latency_ms": max(
                        0,
                        int((context.record.updated_at - context.record.created_at).total_seconds() * 1_000),
                    ),
                }
            ),
        ):
            if self.builtin_agent:
                request = _DurableAgentRequest.model_validate(context.record.validated_input)
                from haystack.dataclasses import ChatMessage

                messages = [ChatMessage.from_dict(message) for message in request.messages]
                resume_input = context.take_resume_input() if context.record.checkpoint is None else None
                if isinstance(resume_input, dict):
                    resumed_messages = resume_input.get("messages")
                    if isinstance(resumed_messages, list):
                        messages.extend(
                            ChatMessage.from_dict(cast(dict[str, Any], message))
                            for message in resumed_messages
                            if isinstance(message, dict)
                        )
                return json_safe(await context.run_agent_async(messages=messages))
            method = cast(Callable[[DurableContext, BaseModel], Any], self.method)
            request = self.request_type.model_validate(context.record.validated_input)
            if self.is_async:
                result = await cast(Awaitable[Any], method(context, request))
            else:
                result = await _run_fenced_thread(method, context, request)
            if self.result_adapter is not None:
                try:
                    result = self.result_adapter.validate_python(result)
                except ValidationError as error:
                    msg = "Durable method result does not match its declared return annotation"
                    raise ValueError(msg) from error
            serializer = getattr(result, "model_dump", None)
            return json_safe(serializer(mode="json") if callable(serializer) else result)


class _DurableAgentRequest(BaseModel):
    """Private A2A input mapping; REST wrappers always provide their own model."""

    messages: list[dict[str, Any]] = Field(min_length=1)
    a2a_context_id: str | None = Field(default=None, min_length=1)


class DurableRuntime:
    """Application-owned provider lifecycle and deployed durable services."""

    def __init__(
        self,
        provider: ExecutionStoreProvider | None = None,
        *,
        app_settings: AppSettings | None = None,
    ) -> None:
        self._store_provider = provider
        self._app_settings = _runtime_settings(provider, app_settings)
        self._deployments: dict[str, DurableDeployment] = {}
        self._started = False
        self._provider_close_task: asyncio.Task[None] | None = None
        self._registry: Any | None = None

    def has_capability(self, wrapper: BasePipelineWrapper) -> bool:
        return durable_authoring_mode(wrapper) is not DurableAuthoringMode.NONE

    @property
    def started(self) -> bool:
        return self._started

    @property
    def provider(self) -> ExecutionStoreProvider | None:
        """Return the runtime-owned provider selected at construction or first use."""
        return self._store_provider

    @property
    def app_settings(self) -> AppSettings:
        """Return configured or provider settings, falling back to Hayhooks' global settings."""
        if self._app_settings is not None:
            return self._app_settings
        provider_settings = getattr(self.provider, "app_settings", None)
        return provider_settings if isinstance(provider_settings, AppSettings) else settings

    def create_deployment(self, name: str, wrapper: BasePipelineWrapper) -> DurableDeployment | None:
        """Build an uncached candidate so route closures cannot capture an old deployment."""
        if not self.has_capability(wrapper):
            return None
        return DurableDeployment(name, wrapper, self._provider(), app_settings=self.app_settings)

    def current_deployment(self, name: str) -> DurableDeployment | None:
        """Return the currently published durable deployment, if any."""
        return self._deployments.get(name)

    def install_deployment(self, name: str, deployment: DurableDeployment | None) -> None:
        """Publish a prepared deployment, or clear a removed durable capability."""
        if deployment is None:
            self._deployments.pop(name, None)
        else:
            self._deployments[name] = deployment

    def deployment(self, name: str, wrapper: BasePipelineWrapper | None = None) -> DurableDeployment:
        """Return the published deployment or create an inactive one for the wrapper."""
        existing = self._deployments.get(name)
        if existing is not None and (wrapper is None or existing.wrapper is wrapper):
            return existing
        wrapper = wrapper or (self._registry.get(name) if self._registry is not None else None)
        if wrapper is None or not self.has_capability(wrapper):
            msg = f"Pipeline '{name}' does not expose durable execution"
            raise KeyError(msg)
        if existing is not None and existing.manager.started:
            msg = (
                f"Pipeline '{name}' has an active durable deployment; "
                "use the async deployment transaction to replace it"
            )
            raise RuntimeError(msg)
        deployment = DurableDeployment(name, wrapper, self._provider(), app_settings=self.app_settings)
        self._deployments[name] = deployment
        return deployment

    async def start(self) -> None:
        """Start runtime-owned deployments after optional registry discovery."""
        if self._started:
            return
        started: list[DurableDeployment] = []
        self._started = True
        try:
            if self._registry is not None:
                for name in self._registry.get_names():
                    wrapper = self._registry.get(name)
                    if wrapper is not None and self.has_capability(wrapper):
                        self.deployment(name, wrapper)
            for deployment in list(self._deployments.values()):
                if deployment.manager.started:
                    continue
                await deployment.start()
                started.append(deployment)
            if started:
                log.bind(deployments=len(started), store=type(self.provider).__name__).info("Durable runtime ready")
        except BaseException:
            self._started = False
            for deployment in reversed(started):
                await deployment.close()
            raise

    async def close(self) -> None:
        """Stop deployments and close their shared provider after draining work."""
        self._started = False
        deployments = list(self._deployments.values())
        log.bind(deployments=len(deployments)).debug("Closing durable runtime")
        for deployment in reversed(deployments):
            await deployment.close()
        self._deployments.clear()
        if self.provider is not None:
            provider = self.provider
            self._store_provider = None
            draining = [deployment.manager for deployment in deployments if deployment.manager.draining]
            if draining:

                async def close_after_drain() -> None:
                    await asyncio.gather(*(manager.wait_drained() for manager in draining))
                    await provider.close()

                self._provider_close_task = asyncio.create_task(
                    close_after_drain(),
                    name="durable-provider-close",
                )
            else:
                await provider.close()

    async def health(self) -> dict[str, JsonValue]:
        """Return aggregate health for all published durable deployments."""
        deployments = dict(
            zip(
                self._deployments,
                await asyncio.gather(
                    *(deployment.manager.health_snapshot() for deployment in self._deployments.values())
                ),
                strict=True,
            )
        )
        return {
            "healthy": all(bool(health["healthy"]) for health in deployments.values()),
            "deployments": cast(JsonValue, deployments),
        }

    def _provider(self) -> ExecutionStoreProvider:
        provider = self._store_provider
        if provider is None:
            if self.app_settings.durable_store == "memory":
                log.warning("Durable execution uses volatile in-memory storage; queued work is lost on process exit")
                provider = InMemoryExecutionStoreProvider(app_settings=self.app_settings)
            else:
                provider = RedisExecutionStoreProvider(app_settings=self.app_settings)
            self._store_provider = provider
        return provider


class IdempotencyConflictError(RuntimeError):
    """An idempotency key was reused with a different operation fingerprint."""


class DefinitionRevisionConflictError(RuntimeError):
    """A nonterminal record belongs to an incompatible deployment revision."""

    code = "definition_revision_conflict"


def execution_id_for(owner_id: str, external_task_id: str) -> str:
    """Return the internal fixed-size durable ID for one A2A task."""
    return hashlib.sha256(owner_id.encode("utf-8") + b"\0" + external_task_id.encode("utf-8")).hexdigest()


def _durable_method_contract(wrapper: BasePipelineWrapper) -> tuple[Any, bool, type[BaseModel], Any]:
    sync, asynchronous = _durable_method_implementations(wrapper)
    if sync == asynchronous:
        msg = "Implement exactly one of run_durable and run_durable_async"
        raise PipelineWrapperError(msg)
    method = wrapper.run_durable_async if asynchronous else wrapper.run_durable
    parameters = list(inspect.signature(method).parameters.values())
    expected_parameters = 2
    if len(parameters) != expected_parameters:
        msg = "Durable methods must accept exactly (context: DurableContext, request: PydanticModel)"
        raise PipelineWrapperError(msg)
    try:
        annotations = get_type_hints(method)
    except (NameError, TypeError) as error:
        msg = f"Invalid durable method annotation: {error}"
        raise PipelineWrapperError(msg) from error
    context_annotation = annotations.get(parameters[0].name)
    request_type = annotations.get(parameters[1].name)
    if context_annotation is not DurableContext:
        msg = "The first durable method parameter must be annotated DurableContext"
        raise PipelineWrapperError(msg)
    if not inspect.isclass(request_type) or not issubclass(request_type, BaseModel):
        msg = "The durable request parameter must be an annotated Pydantic model"
        raise PipelineWrapperError(msg)
    return method, asynchronous, cast(type[BaseModel], request_type), annotations.get("return", Any)


def _operation_fingerprint(
    deployment_name: str,
    definition_revision: str,
    validated_input: Mapping[str, JsonValue],
    *,
    owner_id: str | None,
) -> str:
    payload = json.dumps(
        {
            "deployment_name": deployment_name,
            "definition_revision": definition_revision,
            "validated_input": validated_input,
            "owner_id": owner_id,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> JsonValue:
    """Serialize Pydantic input deterministically without changing list semantics."""
    if isinstance(value, Mapping):
        return {str(key): _canonical_json(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_canonical_json(item) for item in value]
    if isinstance(value, set | frozenset):
        return sorted(
            (_canonical_json(item) for item in value),
            key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        )
    return cast(JsonValue, TypeAdapter(Any).dump_python(value, mode="json"))


durable_runtime = DurableRuntime()
durable_runtime._registry = registry


__all__ = [
    "DefinitionRevisionConflictError",
    "DurableDeployment",
    "DurableRuntime",
    "ExecutionStoreProvider",
    "IdempotencyConflictError",
    "durable_runtime",
]
