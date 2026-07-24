"""Runtime-owned durable deployment services for wrappers and A2A projections."""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import inspect
import json
import uuid
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, cast, get_type_hints

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from hayhooks.durable import DurableOptions
from hayhooks.durable.adapters import HaystackDurableAdapter, definition_revision, execution_kind
from hayhooks.durable.context import DurableContext, DurableExecutionStoreProvider
from hayhooks.durable.manager import DurableExecutionManager
from hayhooks.durable.memory import InMemoryExecutionStoreProvider
from hayhooks.durable.mode import DurableAuthoringMode, durable_authoring_mode
from hayhooks.durable.models import ExecutionKind, ExecutionRecord, JsonValue, json_safe, validate_json
from hayhooks.durable.redis import RedisExecutionStoreProvider
from hayhooks.server.exceptions import PipelineWrapperError
from hayhooks.server.logger import log
from hayhooks.server.pipelines.registry import registry
from hayhooks.server.tracing import SPAN_DURABLE_ATTEMPT, SPAN_DURABLE_SUBMIT, build_trace_tags, trace_operation
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper
from hayhooks.settings import settings


class DurableDeployment:
    """One deployment's records, manager, validated callable, and adapter."""

    def __init__(self, name: str, wrapper: BasePipelineWrapper, provider: DurableExecutionStoreProvider) -> None:
        self.name = name
        self.wrapper = wrapper
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
        options = getattr(wrapper, "durable_options", None)
        if options is not None and not isinstance(options, DurableOptions):
            msg = "durable_options must be DurableOptions or None"
            raise PipelineWrapperError(msg)
        self.kind = kind
        self.revision = definition_revision(pipeline, options.revision if options else None, wrapper=wrapper)
        self.adapter = HaystackDurableAdapter(pipeline, kind)
        self.store = provider.create_execution_store(name)
        self.manager = DurableExecutionManager(
            name,
            self.store,
            self._run,
            self.adapter,
            concurrency=settings.durable_execution_concurrency,
            shutdown_grace_period=settings.durable_shutdown_grace_period,
            max_attempts=settings.durable_max_attempts,
            retry_base_delay=settings.durable_retry_base_delay,
            retry_max_delay=settings.durable_retry_max_delay,
        )

    async def start(self) -> None:
        if self.manager.started:
            return
        await self.prepare()
        await self.retire_incompatible()
        self.activate()

    async def prepare(self) -> None:
        """Initialize the execution store without allowing the candidate to claim work."""
        await self.manager.prepare()

    def activate(self) -> None:
        """Start prepared workers at the deployment publication boundary."""
        self.manager.activate()

    def deactivate(self) -> None:
        """Reject new submissions while an active deployment is being replaced."""
        self.manager.deactivate()

    async def close(self) -> None:
        await self.manager.close()

    async def retire_incompatible(self) -> int:
        """Terminalize unclaimed records created for replaced definitions."""
        return await self.store.retire_incompatible(self.revision)

    async def submit(
        self,
        payload: Mapping[str, Any],
        *,
        execution_id: str | None = None,
        owner_id: str | None = None,
    ) -> tuple[bool, ExecutionRecord]:
        if not self.manager.accepting:
            msg = f"Durable deployment '{self.name}' is not accepting submissions"
            raise RuntimeError(msg)
        request = self.request_type.model_validate(dict(payload))
        if execution_id is not None and owner_id is not None:
            owner_namespace = hashlib.sha256(owner_id.encode("utf-8")).hexdigest()
            execution_id = f"{owner_namespace}-{execution_id}"
        execution_id = execution_id or uuid.uuid4().hex
        validated_input = cast(
            dict[str, JsonValue],
            validate_json(request.model_dump(mode="json"), limit=settings.durable_max_record_bytes, label="request"),
        )
        operation_fingerprint = _operation_fingerprint(
            self.name,
            self.revision,
            validated_input,
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
            max_progress_events=settings.durable_max_progress_events,
            max_record_bytes=settings.durable_max_record_bytes,
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
            created = await self.manager.submit(record)
            persisted = await self.get(execution_id, owner_id=owner_id, enforce_owner=owner_id is not None)
            if not created and persisted.operation_fingerprint != operation_fingerprint:
                msg = "Idempotency-Key was already used for a different durable operation"
                raise IdempotencyConflictError(msg)
            span.set_tag("hayhooks.durable.idempotent_replay", not created)
            return created, persisted

    async def get(
        self,
        execution_id: str,
        *,
        owner_id: str | None = None,
        enforce_owner: bool = False,
        allow_revision_mismatch: bool = False,
    ) -> ExecutionRecord:
        record = await self.store.get(execution_id)
        return self._validated_record(
            execution_id,
            record,
            owner_id=owner_id,
            enforce_owner=enforce_owner,
            allow_revision_mismatch=allow_revision_mismatch,
        )

    async def get_changed(
        self,
        known_sequences: Mapping[str, int],
    ) -> dict[str, ExecutionRecord | None]:
        """Load only records whose store-maintained sequence changed."""
        changed = getattr(self.store, "get_changed", None)
        if callable(changed):
            records = await changed(known_sequences)
        else:
            records = dict(
                zip(
                    known_sequences,
                    await asyncio.gather(*(self.store.get(execution_id) for execution_id in known_sequences)),
                    strict=True,
                )
            )
        validated: dict[str, ExecutionRecord | None] = {}
        for execution_id, record in records.items():
            if record is None:
                validated[execution_id] = None
                continue
            try:
                validated[execution_id] = self._validated_record(execution_id, record)
            except (KeyError, DefinitionRevisionConflictError):
                # Batch consumers isolate missing/incompatible records rather than
                # failing unrelated projections in the same reconciliation pass.
                validated[execution_id] = None
        return validated

    async def request_cancel(
        self,
        execution_id: str,
        *,
        owner_id: str | None = None,
        enforce_owner: bool = False,
        reason: str | None = None,
    ) -> bool:
        await self.get(
            execution_id,
            owner_id=owner_id,
            enforce_owner=enforce_owner,
            allow_revision_mismatch=True,
        )
        return await self.store.request_cancel(execution_id, reason)

    async def resume(
        self,
        execution_id: str,
        update: JsonValue | None = None,
        *,
        owner_id: str | None = None,
        enforce_owner: bool = False,
    ) -> bool:
        await self.get(execution_id, owner_id=owner_id, enforce_owner=enforce_owner)
        if self.resume_type is not None and update is not None:
            update = cast(JsonValue, self.resume_type.model_validate(update).model_dump(mode="json"))
        return await self.store.resume(execution_id, update)

    async def submit_agent_messages(
        self,
        messages: list[Any],
        *,
        execution_id: str | None = None,
        owner_id: str | None = None,
    ) -> tuple[bool, ExecutionRecord]:
        if not self.builtin_agent:
            msg = "This deployment does not use the built-in durable Agent adapter"
            raise RuntimeError(msg)
        return await self.submit(
            {"messages": [json_safe(message) for message in messages]},
            execution_id=execution_id,
            owner_id=owner_id,
        )

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
                call_context = contextvars.copy_context()
                thread_task = asyncio.create_task(
                    asyncio.to_thread(call_context.run, method, context, request),
                    name=f"durable-thread:{self.name}:{context.execution_id}",
                )
                try:
                    result = await asyncio.shield(thread_task)
                except asyncio.CancelledError:
                    # Cancelling the await does not stop the underlying thread. Keep
                    # this claim fenced until the synchronous method actually exits.
                    result = await thread_task
            if self.result_adapter is not None:
                try:
                    result = self.result_adapter.validate_python(result)
                except ValidationError as error:
                    msg = "Durable method result does not match its declared return annotation"
                    raise ValueError(msg) from error
            return json_safe(_model_dump(result))


class _DurableAgentRequest(BaseModel):
    """Private A2A input mapping; REST wrappers always provide their own model."""

    messages: list[dict[str, Any]] = Field(min_length=1)


class DurableRuntime:
    """Application-owned provider lifecycle and deployed durable services."""

    def __init__(self, provider: DurableExecutionStoreProvider | None = None) -> None:
        self.provider = provider
        self._deployments: dict[str, DurableDeployment] = {}
        self._started = False
        self._provider_close_task: asyncio.Task[None] | None = None
        self._background_tasks: set[asyncio.Task[None]] = set()

    def has_capability(self, wrapper: BasePipelineWrapper) -> bool:
        return durable_authoring_mode(wrapper) is not DurableAuthoringMode.NONE

    @property
    def started(self) -> bool:
        return self._started

    def create_deployment(self, name: str, wrapper: BasePipelineWrapper) -> DurableDeployment | None:
        """Build an uncached candidate so route closures cannot capture an old deployment."""
        if not self.has_capability(wrapper):
            return None
        return DurableDeployment(name, wrapper, self._provider())

    def current_deployment(self, name: str) -> DurableDeployment | None:
        return self._deployments.get(name)

    def install_deployment(self, name: str, deployment: DurableDeployment | None) -> None:
        """Publish a prepared deployment, or clear a removed durable capability."""
        if deployment is None:
            self._deployments.pop(name, None)
        else:
            self._deployments[name] = deployment

    def deployment(self, name: str, wrapper: BasePipelineWrapper | None = None) -> DurableDeployment:
        existing = self._deployments.get(name)
        if existing is not None and (wrapper is None or existing.wrapper is wrapper):
            return existing
        wrapper = wrapper or registry.get(name)
        if wrapper is None or not self.has_capability(wrapper):
            msg = f"Pipeline '{name}' does not expose durable execution"
            raise KeyError(msg)
        if existing is not None and existing.manager.started:
            msg = (
                f"Pipeline '{name}' has an active durable deployment; "
                "use the async deployment transaction to replace it"
            )
            raise RuntimeError(msg)
        deployment = DurableDeployment(name, wrapper, self._provider())
        self._deployments[name] = deployment
        return deployment

    async def start_wrapper(self, name: str, wrapper: BasePipelineWrapper) -> None:
        if not self.has_capability(wrapper):
            return
        deployment = self.deployment(name, wrapper)
        if self._started:
            await deployment.start()

    async def close_wrapper(self, name: str) -> None:
        deployment = self._deployments.pop(name, None)
        if deployment is not None:
            await deployment.close()

    async def start(self) -> None:
        if self._started:
            return
        started: list[DurableDeployment] = []
        self._started = True
        try:
            for name in registry.get_names():
                wrapper = registry.get(name)
                if wrapper is not None and self.has_capability(wrapper):
                    deployment = self.deployment(name, wrapper)
                    await deployment.start()
                    started.append(deployment)
        except BaseException:
            self._started = False
            for deployment in reversed(started):
                await deployment.close()
            raise

    async def close(self) -> None:
        self._started = False
        deployments = list(self._deployments.values())
        for deployment in reversed(deployments):
            await deployment.close()
        self._deployments.clear()
        if self.provider is not None:
            provider = self.provider
            self.provider = None
            draining = [deployment.manager for deployment in deployments if deployment.manager.draining]
            background = list(self._background_tasks)
            if draining or background:

                async def close_after_drain() -> None:
                    await asyncio.gather(*(manager.wait_drained() for manager in draining))
                    if background:
                        await asyncio.gather(*background, return_exceptions=True)
                    await provider.close()

                self._provider_close_task = asyncio.create_task(
                    close_after_drain(),
                    name="durable-provider-close",
                )
            else:
                await provider.close()

    def track_background_task(self, awaitable: Awaitable[None], *, name: str) -> None:
        """Keep replacement cleanup alive and surface failures without losing the task."""

        async def resolve() -> None:
            await awaitable

        task = asyncio.create_task(resolve(), name=name)
        self._background_tasks.add(task)

        def completed(done: asyncio.Task[None]) -> None:
            self._background_tasks.discard(done)
            if done.cancelled():
                return
            error = done.exception()
            if error is not None:
                log.opt(exception=error).error("Durable background task '{}' failed", done.get_name())

        task.add_done_callback(completed)

    async def health(self) -> dict[str, JsonValue]:
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

    def shared_redis_client(self) -> Any | None:
        """Return the application-owned Redis client when the built-in provider is active."""
        provider = self._provider()
        return provider.redis if isinstance(provider, RedisExecutionStoreProvider) else None

    def _provider(self) -> DurableExecutionStoreProvider:
        if self.provider is None:
            if settings.durable_store == "memory":
                log.warning("Durable execution uses volatile in-memory storage; queued work is lost on process exit")
                self.provider = InMemoryExecutionStoreProvider(max_progress_events=settings.durable_max_progress_events)
            else:
                self.provider = RedisExecutionStoreProvider()
        return self.provider


class IdempotencyConflictError(RuntimeError):
    """An idempotency key was reused with a different operation fingerprint."""


class DefinitionRevisionConflictError(RuntimeError):
    """A nonterminal record belongs to an incompatible deployment revision."""

    code = "definition_revision_conflict"


def _durable_method_contract(wrapper: BasePipelineWrapper) -> tuple[Any, bool, type[BaseModel], Any]:
    sync = bool(getattr(wrapper, "_is_run_durable_implemented", False))
    asynchronous = bool(getattr(wrapper, "_is_run_durable_async_implemented", False))
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


def _model_dump(value: Any) -> Any:
    serializer = getattr(value, "model_dump", None)
    if callable(serializer):
        return serializer(mode="json")
    return value


durable_runtime = DurableRuntime()


__all__ = [
    "DefinitionRevisionConflictError",
    "DurableDeployment",
    "DurableRuntime",
    "IdempotencyConflictError",
    "durable_runtime",
]
