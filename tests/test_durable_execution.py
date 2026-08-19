import asyncio
import importlib.metadata
import json
import threading
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from haystack import Pipeline, component
from haystack.components.agents import Agent
from haystack.components.agents.state import State
from haystack.core.errors import PipelineRuntimeError
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.dataclasses.breakpoints import PipelineSnapshot
from haystack.tools import Tool
from pydantic import BaseModel

from hayhooks import BasePipelineWrapper, DurableContext
from hayhooks.durable.adapters import (
    HaystackDurableAdapter,
    _agent_exits_after_tools,
    _checkpoint_agent_state,
    _checkpoint_data,
    _restore_agent_state,
)
from hayhooks.durable.context import execution_context_scope
from hayhooks.durable.models import ExecutionCheckpoint, ExecutionKind, ExecutionRecord, ExecutionStatus
from hayhooks.durable.runtime import DurableDeployment, DurableRuntime, _canonical_json, _operation_fingerprint
from hayhooks.durable.settings import DurableSettings
from hayhooks.durable.store import InMemoryExecutionStoreProvider
from hayhooks.server.a2a.app import create_a2a_app
from hayhooks.server.app import create_app
from hayhooks.server.logger import log
from hayhooks.server.pipelines.registry import registry
from hayhooks.server.utils.deploy_utils import add_pipeline_api_route
from hayhooks.server.utils.mcp_utils import create_mcp_server, create_starlette_app
from hayhooks.server.utils.module_loader import (
    _set_method_implementation_flags,
    create_pipeline_wrapper_instance,
    load_pipeline_module,
    unload_pipeline_modules,
)
from hayhooks.settings import settings
from tests.durable_helpers import wait_for_record, wait_for_status, wait_until_async

pytestmark = pytest.mark.skipif(
    not importlib.metadata.version("haystack-ai").startswith("3."), reason="durable execution requires Haystack 3"
)

_DURABLE_EXECUTION_EXAMPLE = Path("examples/durable_execution/pipelines/durable_job")
_DURABLE_A2A_EXAMPLE = Path("examples/a2a_long_running/pipelines/long_running_agent")


class Request(BaseModel):
    value: int


class Result(BaseModel):
    value: int


class ResumeInput(BaseModel):
    approved: bool


def test_set_valued_input_has_a_stable_idempotency_fingerprint() -> None:
    class SetRequest(BaseModel):
        tags: set[str]

    canonical = _canonical_json(SetRequest(tags={"zeta", "alpha"}).model_dump(mode="python"))
    assert canonical == {"tags": ["alpha", "zeta"]}
    assert _operation_fingerprint("job", "revision", canonical, owner_id=None) == _operation_fingerprint(
        "job", "revision", {"tags": ["alpha", "zeta"]}, owner_id=None
    )
    assert _operation_fingerprint("job", "revision", canonical, owner_id=None) != _operation_fingerprint(
        "job", "revision", {"tags": ["zeta", "alpha"]}, owner_id=None
    )


async def test_portable_runtime_starts_its_own_deployments() -> None:
    wrapper = Wrapper()
    wrapper.setup()
    runtime = DurableRuntime(InMemoryExecutionStoreProvider())
    deployment = runtime.deployment("portable", wrapper)

    try:
        await runtime.start()
        assert runtime.started
        assert deployment.manager.started
        assert deployment.manager.accepting
    finally:
        await runtime.close()


async def test_rest_app_runtimes_are_isolated(monkeypatch) -> None:
    monkeypatch.setattr(settings, "pipelines_dir", "")
    provider_a = InMemoryExecutionStoreProvider()
    provider_b = InMemoryExecutionStoreProvider()
    close_a = AsyncMock(wraps=provider_a.close)
    close_b = AsyncMock(wraps=provider_b.close)
    monkeypatch.setattr(provider_a, "close", close_a)
    monkeypatch.setattr(provider_b, "close", close_b)
    runtime_a = DurableRuntime(provider_a)
    runtime_b = DurableRuntime(provider_b)
    wrapper = Wrapper()
    wrapper.setup()
    runtime_a.deployment("only-a", wrapper)
    app_a = create_app(durable_runtime=runtime_a)
    app_b = create_app(durable_runtime=runtime_b)

    assert app_a.state.durable_runtime is runtime_a
    assert app_b.state.durable_runtime is runtime_b
    assert runtime_b.current_deployment("only-a") is None

    lifespan_a = app_a.router.lifespan_context(app_a)
    lifespan_b = app_b.router.lifespan_context(app_b)
    await lifespan_a.__aenter__()
    await lifespan_b.__aenter__()
    try:
        await lifespan_a.__aexit__(None, None, None)
        assert not runtime_a.started
        assert runtime_b.started
        close_a.assert_awaited_once()
        close_b.assert_not_awaited()
    finally:
        await lifespan_b.__aexit__(None, None, None)
    close_b.assert_awaited_once()


def test_app_factories_retain_the_supplied_runtime() -> None:
    rest_runtime = DurableRuntime(InMemoryExecutionStoreProvider())
    a2a_runtime = DurableRuntime(InMemoryExecutionStoreProvider())
    mcp_runtime = DurableRuntime(InMemoryExecutionStoreProvider())

    rest = create_app(durable_runtime=rest_runtime)
    a2a = create_a2a_app(durable_runtime=a2a_runtime)
    mcp_server = create_mcp_server(durable_runtime=mcp_runtime)
    mcp = create_starlette_app(mcp_server, durable_runtime=mcp_runtime)

    assert rest.state.durable_runtime is rest_runtime
    assert a2a.state.durable_runtime is a2a_runtime
    assert mcp.state.durable_runtime is mcp_runtime
    assert create_app().state.durable_runtime is not create_app().state.durable_runtime


def _create_rest_app(runtime: DurableRuntime):
    return create_app(durable_runtime=runtime)


def _create_a2a_test_app(runtime: DurableRuntime):
    return create_a2a_app(durable_runtime=runtime)


def _create_mcp_app(runtime: DurableRuntime):
    return create_starlette_app(
        create_mcp_server(durable_runtime=runtime),
        durable_runtime=runtime,
    )


@pytest.mark.parametrize(
    "app_factory",
    [_create_rest_app, _create_a2a_test_app, _create_mcp_app],
    ids=["rest", "a2a", "mcp"],
)
async def test_app_lifespans_close_durable_runtime_when_start_fails(monkeypatch, app_factory) -> None:
    closed = 0

    async def fail_start() -> None:
        msg = "durable startup failed"
        raise RuntimeError(msg)

    async def close() -> None:
        nonlocal closed
        closed += 1

    monkeypatch.setattr(settings, "pipelines_dir", "")
    runtime = DurableRuntime(InMemoryExecutionStoreProvider())
    monkeypatch.setattr(runtime, "start", fail_start)
    monkeypatch.setattr(runtime, "close", close)
    app = app_factory(runtime)

    with pytest.raises(Exception) as exc_info:
        async with app.router.lifespan_context(app):
            pass

    assert "durable startup failed" in repr(exc_info.value)
    assert closed == 1


async def test_mcp_lifespan_closes_durable_runtime_when_session_manager_start_fails(monkeypatch) -> None:
    session_manager = MagicMock()
    session_manager.run.return_value.__aenter__.side_effect = RuntimeError("MCP session manager startup failed")
    provider = InMemoryExecutionStoreProvider()
    close = AsyncMock(wraps=provider.close)
    monkeypatch.setattr(provider, "close", close)
    monkeypatch.setattr(
        "hayhooks.server.utils.mcp_utils.StreamableHTTPSessionManager", lambda **_kwargs: session_manager
    )
    runtime = DurableRuntime(provider)
    app = _create_mcp_app(runtime)

    with pytest.raises(RuntimeError, match="MCP session manager startup failed"):
        async with app.router.lifespan_context(app):
            pass

    close.assert_awaited_once()


def _checkpoint_test_tool(value: str) -> str:
    return value


class _Claim:
    def __init__(self, record: ExecutionRecord) -> None:
        self.record = record
        self.checkpoints = 0

    async def checkpoint(self) -> None:
        self.checkpoints += 1

    async def cancellation_requested(self) -> bool:
        return False


def _agent_record(execution_id: str, deployment_name: str = "agent", **changes) -> ExecutionRecord:
    fields = {
        "execution_id": execution_id,
        "execution_kind": ExecutionKind.AGENT,
        "deployment_name": deployment_name,
        "definition_revision": "revision",
        "validated_input": {"messages": []},
    }
    return ExecutionRecord(**{**fields, **changes})


class Wrapper(BasePipelineWrapper):
    durable_revision = "test-wrapper"

    def setup(self) -> None:
        self.pipeline = Pipeline()

    async def run_durable_async(self, context: DurableContext, request: Request) -> Result:
        await context.report_progress("working")
        return Result(value=request.value + 1)


class InvalidResultWrapper(Wrapper):
    async def run_durable_async(self, context: DurableContext, request: Request) -> Result:
        return {"value": "not-an-integer"}  # type: ignore[return-value]


class SyncWrapper(BasePipelineWrapper):
    durable_revision = "sync-wrapper"

    def setup(self) -> None:
        self.pipeline = Pipeline()

    def run_durable(self, context: DurableContext, request: Request) -> Result:
        context.report_progress_sync("working in a worker thread")
        context.check_cancelled_sync()
        return Result(value=request.value + 2)


class BlockingAdmissionWrapper(BasePipelineWrapper):
    durable_revision = "blocking-admission"

    def setup(self) -> None:
        self.pipeline = Pipeline()
        self.started = threading.Event()
        self.release = threading.Event()

    async def run_durable_async(self, context: DurableContext, request: Request) -> Result:
        self.started.set()
        await asyncio.to_thread(self.release.wait)
        return Result(value=request.value + 1)


async def test_durable_lifecycle_logs_identifiers_without_payload(monkeypatch) -> None:
    monkeypatch.setattr(settings, "durable_poll_interval", 0.01)
    deployment = _deployment("logged-job")
    records = []
    sink = log.add(lambda message: records.append(message.record), level="DEBUG")
    try:
        await deployment.start()
        await deployment.submit({"value": 8675309}, execution_id="logged-execution")
        expected = {
            "Accepted durable execution submission",
            "Claimed durable execution",
            "Finished durable execution attempt",
        }
        await wait_until_async(
            lambda: expected <= {record["message"] for record in records},
            "durable lifecycle logs were not emitted",
        )
    finally:
        await deployment.close()
        log.remove(sink)

    lifecycle = [record for record in records if record["message"] in expected]
    assert all(record["extra"]["deployment"] == "logged-job" for record in lifecycle)
    assert all(record["extra"]["execution_id"] == "logged-execution" for record in lifecycle)
    assert "8675309" not in str(lifecycle)


async def test_quiesce_waits_for_an_admitted_submission_and_rejects_later_ones(monkeypatch) -> None:
    monkeypatch.setattr(settings, "durable_poll_interval", 0.01)
    monkeypatch.setattr(settings, "durable_shutdown_grace_period", 0.001)
    deployment = _deployment("admission-gate")
    await deployment.start()
    entered = asyncio.Event()
    release = asyncio.Event()
    submit_with_record = deployment.store.submit_with_record

    async def paused_submit(record):
        entered.set()
        await release.wait()
        return await submit_with_record(record)

    monkeypatch.setattr(deployment.store, "submit_with_record", paused_submit)
    submission = asyncio.create_task(deployment.submit({"value": 1}))
    await entered.wait()
    quiescing = asyncio.create_task(deployment.quiesce())
    await asyncio.sleep(0)
    assert not quiescing.done()

    release.set()
    await submission
    await quiescing
    assert (await deployment.store.operational_counts())["nonterminal"] == 1
    with pytest.raises(RuntimeError, match="not accepting submissions"):
        await deployment.submit({"value": 2})
    await deployment.close()


async def test_deployment_claims_reject_incompatible_work_without_a_revision_scan() -> None:
    provider = InMemoryExecutionStoreProvider()
    store = provider.create_execution_store("revision-safe")
    await store.initialize()

    for execution_id, revision in (("waiting-old", "old"), ("waiting-current", "current")):
        assert await store.submit(
            ExecutionRecord(
                execution_id=execution_id,
                execution_kind=ExecutionKind.PIPELINE,
                deployment_name="revision-safe",
                definition_revision=revision,
                validated_input={"value": 1},
            )
        )
        store.set_definition_revision(revision)
        waiting = await store.claim_next("worker")
        assert waiting is not None
        async with waiting:
            waiting.record.status = ExecutionStatus.WAITING
            waiting.record.wait = {"kind": "input"}
            await waiting.suspend()
    assert await store.submit(
        ExecutionRecord(
            execution_id="queued-old",
            execution_kind=ExecutionKind.PIPELINE,
            deployment_name="revision-safe",
            definition_revision="old",
            validated_input={"value": 1},
        )
    )

    wrapper = Wrapper()
    wrapper.durable_revision = "current"
    async with _started(_deployment("revision-safe", wrapper, provider)):
        queued_old = await wait_for_record(
            store, "queued-old", message="incompatible queued work was not rejected by its first claim"
        )
        waiting_old = await store.get("waiting-old")
        waiting_current = await store.get("waiting-current")

    assert queued_old.status is ExecutionStatus.FAILED
    assert queued_old.error is not None
    assert waiting_old is not None
    assert waiting_old.status is ExecutionStatus.WAITING
    assert waiting_current is not None
    assert waiting_current.status is ExecutionStatus.WAITING


@component
class _CheckpointIncrement:
    def __init__(self, *, fail_once: bool = False) -> None:
        self.fail_once = fail_once
        self.calls = 0

    @component.output_types(value=int)
    def run(self, value: int) -> dict[str, int]:
        self.calls += 1
        if self.fail_once and self.calls == 1:
            msg = "interrupted component"
            raise RuntimeError(msg)
        return {"value": value + 1}


class CheckpointPipelineWrapper(BasePipelineWrapper):
    durable_revision = "checkpoint-pipeline"

    def setup(self) -> None:
        self.first = _CheckpointIncrement()
        self.second = _CheckpointIncrement(fail_once=True)
        self.pipeline = Pipeline()
        self.pipeline.add_component("first", self.first)
        self.pipeline.add_component("second", self.second)
        self.pipeline.connect("first.value", "second.value")

    async def run_durable_async(self, context: DurableContext, request: Request) -> Result:
        try:
            result = await context.run_pipeline_async(
                {"first": {"value": request.value}},
                checkpoint_at=["first", "second"],
            )
        except PipelineRuntimeError:
            await context.retry("retry interrupted pipeline", delay=0)
        return Result(value=result["second"]["value"])


class WaitingWrapper(Wrapper):
    durable_resume_model = ResumeInput

    async def run_durable_async(self, context: DurableContext, request: Request) -> Result:
        if context.resume_input is None:
            await context.suspend(
                {
                    "kind": "approval",
                    "message": "Approve this job",
                    "expected_input_schema": ResumeInput.model_json_schema(),
                    "private_tool_arguments": {"secret": True},
                }
            )
        resume = ResumeInput.model_validate(context.take_resume_input())
        return Result(value=request.value if resume.approved else -1)


@component
class FakeChatGenerator:
    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage], tools=None):
        return {"replies": [ChatMessage.from_assistant("done")]}


class AgentRequest(BaseModel):
    message: str


class AgentWrapper(BasePipelineWrapper):
    durable_revision = "test-agent"

    def setup(self) -> None:
        self.pipeline = Agent(chat_generator=FakeChatGenerator(), tools=[])

    async def run_durable_async(self, context: DurableContext, request: AgentRequest) -> dict:
        return await context.run_agent_async(messages=[ChatMessage.from_user(request.message)])


class BuiltinAgentWrapper(BasePipelineWrapper):
    durable = True
    durable_revision = "builtin-agent"

    def setup(self) -> None:
        self.pipeline = Agent(chat_generator=FakeChatGenerator(), tools=[])


def _deployment(name: str, wrapper: BasePipelineWrapper | None = None, provider=None, **options) -> DurableDeployment:
    """Prepare a wrapper the way the module loader does, then deploy it."""
    wrapper = wrapper or Wrapper()
    wrapper.setup()
    _set_method_implementation_flags(wrapper)
    return DurableDeployment(name, wrapper, provider or InMemoryExecutionStoreProvider(), **options)


@contextmanager
def _example_module(module_name: str, source: Path):
    """Load one bundled example wrapper module and always unload it."""
    module = load_pipeline_module(module_name, source)
    try:
        yield module
    finally:
        unload_pipeline_modules(module_name)


@asynccontextmanager
async def _started(deployment: DurableDeployment):
    """Run one deployment for the body of a test and always close it."""
    await deployment.start()
    try:
        yield deployment
    finally:
        await deployment.close()


@pytest.fixture(autouse=True)
def clean_registry():
    registry.clear()
    yield
    registry.clear()


def _durable_app(monkeypatch, name, wrapper):
    monkeypatch.setattr(settings, "durable_store", "memory")
    monkeypatch.setattr(settings, "durable_poll_interval", 0.05)
    wrapper.setup()
    _set_method_implementation_flags(wrapper)
    registry.add(name, wrapper)
    app = create_app()
    add_pipeline_api_route(app, name, wrapper)
    return app


def test_durable_rest_submission_is_direct_typed_and_idempotent(monkeypatch) -> None:
    app = _durable_app(monkeypatch, "job", Wrapper())

    with TestClient(app) as client:
        submitted = client.post("/job/run-durable", json={"value": 4}, headers={"Idempotency-Key": "same"})
        duplicate = client.post("/job/run-durable", json={"value": 4}, headers={"Idempotency-Key": "same"})
        assert submitted.status_code == 202
        assert duplicate.headers["Idempotent-Replay"] == "true"
        body = submitted.json()
        assert set(body) == {
            "execution_id",
            "status",
            "attempt",
            "sequence",
            "progress",
            "result",
            "error",
            "waiting",
            "cancellation_requested_at",
            "created_at",
            "updated_at",
            "links",
        }
        inspected = wait_for_status(client, body["links"]["self"], "completed")

    assert inspected["result"] == {"value": 5}
    assert submitted.headers["Location"] == body["links"]["self"]


def test_durable_rest_admission_is_retryable_and_preserves_owner_isolation(monkeypatch) -> None:
    monkeypatch.setattr(settings, "durable_trusted_owner_header", "X-Authenticated-Owner")
    monkeypatch.setattr(settings, "durable_max_nonterminal_executions", 1)
    wrapper = BlockingAdmissionWrapper()
    app = _durable_app(monkeypatch, "admission", wrapper)

    try:
        with TestClient(app) as client:
            alice_headers = {"Idempotency-Key": "alice-first", "X-Authenticated-Owner": "alice"}
            first = client.post("/admission/run-durable", json={"value": 1}, headers=alice_headers)
            assert first.status_code == 202
            assert wrapper.started.wait(timeout=1)

            replay = client.post("/admission/run-durable", json={"value": 1}, headers=alice_headers)
            assert replay.status_code == 202
            assert replay.headers["Idempotent-Replay"] == "true"

            globally_limited = client.post(
                "/admission/run-durable",
                json={"value": 2},
                headers={"Idempotency-Key": "alice-second", "X-Authenticated-Owner": "alice"},
            )
            assert globally_limited.status_code == 503
            assert globally_limited.headers["Retry-After"] == "1"
            assert "deployment_nonterminal" in globally_limited.json()["detail"]

            bob = client.post(
                "/admission/run-durable",
                json={"value": 3},
                headers={"Idempotency-Key": "bob-first", "X-Authenticated-Owner": "bob"},
            )
            assert bob.status_code == 503
            wrapper.release.set()
    finally:
        wrapper.release.set()


def test_durable_rest_can_inspect_and_cancel_an_execution_from_an_old_revision(monkeypatch) -> None:
    wrapper = BlockingAdmissionWrapper()
    app = _durable_app(monkeypatch, "rolling", wrapper)

    try:
        with TestClient(app) as client:
            submitted = client.post("/rolling/run-durable", json={"value": 1})
            assert wrapper.started.wait(timeout=5)
            deployment = app.state.durable_runtime.current_deployment("rolling")
            assert deployment is not None
            deployment.revision = "replacement"

            links = submitted.json()["links"]
            assert client.get(links["self"]).status_code == 200
            canceled = client.post(links["cancel"])
            assert canceled.status_code == 202
            assert canceled.json()["cancellation_requested_at"] is not None
            wrapper.release.set()
            wait_for_status(client, links["self"], "canceled")
            assert client.post(links["cancel"]).status_code == 200
    finally:
        wrapper.release.set()


def test_durable_result_annotation_is_validated_before_completion(monkeypatch) -> None:
    app = _durable_app(monkeypatch, "invalid-result", InvalidResultWrapper())

    with TestClient(app) as client:
        submitted = client.post("/invalid-result/run-durable", json={"value": 4})
        url = submitted.json()["links"]["self"]
        inspected = wait_for_status(client, url, "failed")

    assert inspected["result"] is None
    assert inspected["error"] == {
        "type": "ValueError",
        "message": "Durable method result does not match its declared return annotation",
        "retryable": False,
        "code": None,
    }


def test_durable_rest_rejects_mismatched_idempotency_payload(monkeypatch) -> None:
    app = _durable_app(monkeypatch, "job", Wrapper())

    with TestClient(app) as client:
        first = client.post("/job/run-durable", json={"value": 4}, headers={"Idempotency-Key": "same"})
        conflict = client.post("/job/run-durable", json={"value": 5}, headers={"Idempotency-Key": "same"})

    assert first.status_code == 202
    assert conflict.status_code == 409


def test_durable_rest_uses_the_execution_id_key_grammar_at_every_boundary(monkeypatch) -> None:
    app = _durable_app(monkeypatch, "job", Wrapper())

    with TestClient(app) as client:
        accepted = client.post("/job/run-durable", json={"value": 4}, headers={"Idempotency-Key": "a" * 128})
        rejected = [
            client.post("/job/run-durable", json={"value": 4}, headers={"Idempotency-Key": key})
            for key in ("part/child", "part.child", "part~child", "a" * 129)
        ]
        invalid_paths = [
            client.get("/job/executions/part.child"),
            client.post("/job/executions/part.child/cancel"),
            client.post("/job/executions/part.child/resume"),
        ]

    assert accepted.status_code == 202
    assert all(response.status_code == 422 for response in rejected + invalid_paths)


def test_durable_rest_bounds_owner_scoped_idempotency_keys(monkeypatch) -> None:
    monkeypatch.setattr(settings, "durable_trusted_owner_header", "X-Authenticated-Owner")
    app = _durable_app(monkeypatch, "job", Wrapper())

    with TestClient(app) as client:
        accepted = client.post(
            "/job/run-durable",
            json={"value": 4},
            headers={"Idempotency-Key": "a" * 63, "X-Authenticated-Owner": "owner"},
        )
        rejected = client.post(
            "/job/run-durable",
            json={"value": 4},
            headers={"Idempotency-Key": "a" * 64, "X-Authenticated-Owner": "owner"},
        )

    assert accepted.status_code == 202
    assert rejected.status_code == 422
    assert "63" in rejected.json()["detail"]


def test_durable_rest_maps_oversized_validated_request_to_422(monkeypatch) -> None:
    monkeypatch.setattr(settings, "durable_max_record_bytes", 1_024)
    app = _durable_app(monkeypatch, "job", Wrapper())

    with TestClient(app) as client:
        response = client.post("/job/run-durable", json={"value": int("1" * 1_100)})

    assert response.status_code == 422
    assert "durable execution limit" in response.json()["detail"]


def test_durable_waiting_resume_is_typed_private_and_revision_safe(monkeypatch) -> None:
    app = _durable_app(monkeypatch, "approval", WaitingWrapper())

    with TestClient(app) as client:
        submitted = client.post("/approval/run-durable", json={"value": 7})
        url = submitted.json()["links"]["self"]
        waiting = wait_for_status(client, url, "waiting")
        assert waiting["waiting"] == {
            "kind": "approval",
            "message": "Approve this job",
            "expected_input_schema": ResumeInput.model_json_schema(),
        }
        deployment = app.state.durable_runtime.current_deployment("approval")
        assert deployment is not None
        missing = client.post(f"{url}/resume")
        assert missing.status_code == 422
        invalid = client.post(f"{url}/resume", json={"approved": "not-a-bool"})
        assert invalid.status_code == 422
        revision = deployment.revision
        deployment.revision = "replacement"
        conflict = client.post(f"{url}/resume", json={"approved": True})
        assert conflict.status_code == 409
        assert client.get(url).json()["status"] == "waiting"
        deployment.revision = revision
        resumed = client.post(f"{url}/resume", json={"approved": True})
        assert resumed.status_code == 202
        completed = wait_for_status(client, url, "completed")

    assert completed["result"] == {"value": 7}
    openapi = app.openapi()
    resume_schema = openapi["paths"]["/approval/executions/{execution_id}/resume"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]
    assert "ResumeInput" in str(resume_schema)


def test_durable_rest_enforces_configured_trusted_owner_header(monkeypatch) -> None:
    monkeypatch.setattr(settings, "durable_trusted_owner_header", "X-Authenticated-Owner")
    app = _durable_app(monkeypatch, "owned", Wrapper())

    with TestClient(app) as client:
        assert client.post("/owned/run-durable", json={"value": 1}).status_code == 401
        submitted = client.post(
            "/owned/run-durable",
            json={"value": 1},
            headers={"X-Authenticated-Owner": "alice"},
        )
        assert submitted.status_code == 202
        url = submitted.json()["links"]["self"]
        assert client.get(url, headers={"X-Authenticated-Owner": "bob"}).status_code == 404
        assert client.get(url, headers={"X-Authenticated-Owner": "alice"}).status_code == 200
        oversized = client.post(
            "/owned/run-durable",
            json={"value": 1},
            headers={"X-Authenticated-Owner": "x" * 513},
        )
        assert oversized.status_code == 400
        assert "exceeds 512 characters" in oversized.json()["detail"]


def test_durable_deployment_requires_an_explicit_revision() -> None:
    class MissingRevisionWrapper(BasePipelineWrapper):
        def setup(self) -> None:
            self.pipeline = Pipeline()

        async def run_durable_async(self, context: DurableContext, request: Request) -> Result:
            return Result(value=request.value)

    with pytest.raises(Exception, match="non-empty durable_revision"):
        _deployment("missing-revision", MissingRevisionWrapper())


def test_sync_durable_wrapper_uses_context_sync_controls(monkeypatch) -> None:
    app = _durable_app(monkeypatch, "sync-job", SyncWrapper())

    with TestClient(app) as client:
        submitted = client.post("/sync-job/run-durable", json={"value": 4})
        url = submitted.json()["links"]["self"]
        inspected = wait_for_status(client, url, "completed")

    assert inspected["result"] == {"value": 6}
    assert inspected["progress"][0]["message"] == "working in a worker thread"


async def test_sync_work_retains_claim_after_shutdown_grace_until_thread_exits(monkeypatch) -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingWrapper(BasePipelineWrapper):
        durable_revision = "blocking-wrapper"

        def setup(self) -> None:
            self.pipeline = Pipeline()

        def run_durable(self, context: DurableContext, request: Request) -> Result:
            started.set()
            assert release.wait(timeout=5)
            return Result(value=request.value)

    durable_settings = DurableSettings(durable_store="memory", durable_shutdown_grace_period=0.001)
    provider = InMemoryExecutionStoreProvider(durable_settings=durable_settings)
    deployment = _deployment("blocking", BlockingWrapper(), provider)
    await deployment.start()
    _, submitted = await deployment.submit({"value": 9})
    assert await asyncio.to_thread(started.wait, 1)

    await deployment.close()
    assert deployment.manager.draining
    assert await deployment.store.claim_next("replacement") is None

    release.set()
    await deployment.manager.wait_drained()
    completed = await deployment.store.get(submitted.execution_id)
    assert completed is not None
    assert completed.status.value == "completed"


async def test_async_pipeline_thread_fallback_retains_cancellation_fence(monkeypatch) -> None:
    started = threading.Event()
    release = threading.Event()
    adapter = HaystackDurableAdapter(Pipeline(), ExecutionKind.PIPELINE)

    def blocking_run(_context, _data, *, checkpoint_at):
        assert checkpoint_at == ["component"]
        started.set()
        assert release.wait(timeout=5)
        return {"done": True}

    monkeypatch.setattr(adapter, "run_pipeline", blocking_run)
    task = asyncio.create_task(
        adapter.run_pipeline_async(object(), {}, checkpoint_at=["component"]),  # type: ignore[arg-type]
    )
    assert await asyncio.to_thread(started.wait, 1)

    task.cancel()
    await asyncio.sleep(0.01)
    assert not task.done()
    release.set()
    assert await task == {"done": True}


async def test_pipeline_snapshot_round_trip_skips_completed_components_after_retry(monkeypatch) -> None:
    monkeypatch.setattr(settings, "durable_retry_base_delay", 0)
    monkeypatch.setattr(settings, "durable_retry_max_delay", 0)
    wrapper = CheckpointPipelineWrapper()
    async with _started(_deployment("checkpoint-pipeline", wrapper)) as deployment:
        _, submitted = await deployment.submit({"value": 3})
        completed = await wait_for_record(
            deployment.store, submitted.execution_id, message="checkpointed Pipeline did not finish its retry"
        )

    assert completed.status.value == "completed"
    assert completed.result == {"value": 5}
    assert completed.attempt == 2
    assert completed.checkpoint is not None
    PipelineSnapshot.from_dict(completed.checkpoint.data["snapshot"])
    assert wrapper.first.calls == 1
    assert wrapper.second.calls == 2
    checkpoint_events = [event for event in completed.progress if event.kind == "checkpoint"]
    assert len(checkpoint_events) >= 2


async def test_agent_state_checkpoint_restores_custom_state_and_typed_resume_messages() -> None:
    schema = {
        "counter": {"type": int},
        "tools": {"type": list},
        "hook_context": {"type": dict},
    }
    record = _agent_record("agent-state")
    checkpoint_state = State(
        schema=schema,
        data={
            "counter": 7,
            "messages": [ChatMessage.from_user("before restart")],
            "tools": ["old tool"],
            "hook_context": {"request": "old"},
        },
    )
    checkpoint_context = DurableContext(_Claim(record), adapter=object())
    checkpoint_data = _checkpoint_data(checkpoint_state, checkpoint_context)
    checkpoint_payload = checkpoint_data["data"]
    assert "tools" not in checkpoint_payload["serialized_data"]
    assert "hook_context" not in checkpoint_payload["serialized_data"]
    assert "tools" not in checkpoint_payload["serialization_schema"]["properties"]
    assert "hook_context" not in checkpoint_payload["serialization_schema"]["properties"]
    store = InMemoryExecutionStoreProvider().create_execution_store("agent")
    await store.initialize()
    assert await store.submit(record)
    store.set_definition_revision("revision")
    waiting_claim = await store.claim_next("before-resume")
    assert waiting_claim is not None
    async with waiting_claim:
        waiting_claim.record.checkpoint = ExecutionCheckpoint(ExecutionKind.AGENT, checkpoint_data)
        await waiting_claim.checkpoint()
        waiting_claim.record.status = ExecutionStatus.WAITING
        await waiting_claim.suspend()
    assert await store.resume(
        record.execution_id,
        {"messages": [ChatMessage.from_user("after restart").to_dict()]},
    )
    resumed_claim = await store.claim_next("after-resume")
    assert resumed_claim is not None
    context = DurableContext(resumed_claim, adapter=object())

    restored_state = State(
        schema=schema,
        data={"counter": 0, "tools": ["live tool"], "hook_context": {"request": "live"}},
    )
    _restore_agent_state(context, restored_state)

    assert restored_state.data["counter"] == 7
    assert restored_state.data["tools"] == ["live tool"]
    assert restored_state.data["hook_context"] == {"request": "live"}
    assert [message.text for message in restored_state.data["messages"]] == ["before restart", "after restart"]
    assert context.resume_input is None


async def test_agent_checkpoint_excludes_custom_tool_deserialization() -> None:
    tool = Tool(
        name="custom_tool",
        description="A tool defined by the deployed wrapper",
        parameters={"type": "object", "properties": {"value": {"type": "string"}}},
        function=_checkpoint_test_tool,
    )
    record = _agent_record("agent-tool-checkpoint")
    checkpoint_context = DurableContext(_Claim(record), adapter=object())
    checkpoint_state = State(
        schema={"tools": {"type": list}, "hook_context": {"type": dict}},
        data={"tools": [tool], "hook_context": {"request": "old"}},
    )
    checkpoint_data = _checkpoint_data(checkpoint_state, checkpoint_context)
    checkpoint_payload = checkpoint_data["data"]
    assert "tools" not in checkpoint_payload["serialized_data"]
    assert "tools" not in checkpoint_payload["serialization_schema"]["properties"]
    record.checkpoint = ExecutionCheckpoint(ExecutionKind.AGENT, checkpoint_state.to_dict())

    restored_state = State(
        schema={"tools": {"type": list}, "hook_context": {"type": dict}},
        data={"tools": [tool], "hook_context": {"request": "live"}},
    )
    _restore_agent_state(DurableContext(_Claim(record), adapter=object()), restored_state)

    assert restored_state.data["tools"] == [tool]
    assert restored_state.data["hook_context"] == {"request": "live"}


async def test_agent_checkpoint_and_progress_share_one_store_write() -> None:
    record = _agent_record("coalesced-agent-checkpoint")
    claim = _Claim(record)
    context = DurableContext(claim, adapter=object())
    state = State(schema={"counter": {"type": int}}, data={"counter": 1})

    await _checkpoint_agent_state(context, state)

    assert claim.checkpoints == 1
    assert record.checkpoint is not None
    assert [event.kind for event in record.progress] == ["checkpoint"]


def test_agent_tool_exit_uses_only_the_current_tool_result_batch() -> None:
    failed_then_retried = State(
        schema={},
        data={
            "messages": [
                ChatMessage.from_tool("failed", origin=ToolCall(tool_name="finish", arguments={}), error=True),
                ChatMessage.from_assistant("retrying"),
                ChatMessage.from_tool("done", origin=ToolCall(tool_name="finish", arguments={})),
            ]
        },
    )
    current_non_exit_tool = State(
        schema={},
        data={
            "messages": [
                ChatMessage.from_tool("done", origin=ToolCall(tool_name="finish", arguments={})),
                ChatMessage.from_assistant("continuing"),
                ChatMessage.from_tool("working", origin=ToolCall(tool_name="search", arguments={})),
            ]
        },
    )

    assert _agent_exits_after_tools(failed_then_retried, ["finish"])
    assert not _agent_exits_after_tools(current_non_exit_tool, ["finish"])


async def test_agent_after_run_saves_a_final_checkpoint() -> None:
    record = _agent_record("agent-after-run")
    adapter = HaystackDurableAdapter(Agent(chat_generator=FakeChatGenerator(), tools=[]), ExecutionKind.AGENT)
    claim = _Claim(record)
    context = DurableContext(claim, adapter=adapter)

    with execution_context_scope(context):
        result = await adapter.run_agent_async(context, messages=[ChatMessage.from_user("question")])

    assert result["last_message"].text == "done"
    assert record.checkpoint is not None
    assert record.checkpoint.data["_hayhooks_agent_checkpoint_phase"] == "after_run"
    assert claim.checkpoints == 1


async def test_durable_agent_hooks_ignore_ordinary_sync_and_async_runs() -> None:
    agent = Agent(chat_generator=FakeChatGenerator(), tools=[])
    HaystackDurableAdapter(agent, ExecutionKind.AGENT)
    hook_counts = {name: len(hooks) for name, hooks in agent.hooks.items()}
    HaystackDurableAdapter(agent, ExecutionKind.AGENT)

    sync_result = agent.run(messages=[ChatMessage.from_user("sync")])
    async_result = await agent.run_async(messages=[ChatMessage.from_user("async")])

    assert {name: len(hooks) for name, hooks in agent.hooks.items()} == hook_counts
    assert sync_result["last_message"].text == "done"
    assert async_result["last_message"].text == "done"


async def test_agent_after_run_checkpoint_recovers_without_another_llm_call() -> None:
    record = _agent_record("agent-final-state")
    adapter = HaystackDurableAdapter(Agent(chat_generator=FakeChatGenerator(), tools=[]), ExecutionKind.AGENT)
    claim = _Claim(record)
    context = DurableContext(claim, adapter=adapter)
    final_state = State(
        schema={"counter": {"type": int}},
        data={
            "counter": 2,
            "messages": [ChatMessage.from_user("question"), ChatMessage.from_assistant("already complete")],
        },
    )

    record.checkpoint = ExecutionCheckpoint(ExecutionKind.AGENT, _checkpoint_data(final_state, context, final=True))
    result = await adapter.run_agent_async(context, messages=[ChatMessage.from_user("must not run")])

    assert result["counter"] == 2
    assert result["last_message"].text == "already complete"
    assert claim.checkpoints == 0


async def test_agent_on_exit_continuation_checkpoints_application_state() -> None:
    from haystack.hooks import hook

    @component
    class FailAfterExit:
        @component.output_types(replies=list[ChatMessage])
        def run(self, messages: list[ChatMessage], tools=None) -> dict:
            del messages, tools
            if hasattr(self, "called"):
                msg = "interrupted after continuation"
                raise RuntimeError(msg)
            self.called = True
            return {"replies": [ChatMessage.from_assistant("first exit")]}

    @hook
    def continue_once(state: State) -> None:
        state.set("marker", "saved")
        state.set("continue_run", True)

    record = _agent_record("agent-on-exit")
    adapter = HaystackDurableAdapter(
        Agent(
            chat_generator=FailAfterExit(),
            tools=[],
            state_schema={"marker": {"type": str}},
            hooks={"on_exit": [continue_once]},
        ),
        ExecutionKind.AGENT,
    )
    claim = _Claim(record)
    context = DurableContext(claim, adapter=adapter)

    with execution_context_scope(context), pytest.raises(RuntimeError, match="interrupted after continuation"):
        await adapter.run_agent_async(context, messages=[ChatMessage.from_user("question")])

    assert record.checkpoint is not None
    data = record.checkpoint.data["data"]["serialized_data"]
    assert data["marker"] == "saved"
    assert data["continue_run"] is True
    assert claim.checkpoints == 1


async def test_builtin_agent_leaves_resume_input_for_checkpoint_restoration() -> None:
    class RestoringAdapter:
        restored_messages: list[str]

        async def run_agent_async(self, context, *, messages, **_kwargs):
            state = State(
                schema={"messages": {"type": list[ChatMessage]}},
                data={"messages": messages},
            )
            _restore_agent_state(context, state)
            self.restored_messages = [message.text for message in state.data["messages"]]
            return {"messages": [message.to_dict() for message in state.data["messages"]]}

    wrapper = BuiltinAgentWrapper()
    deployment = _deployment("builtin-agent", wrapper)
    checkpoint_state = State(
        schema={"messages": {"type": list[ChatMessage]}},
        data={"messages": [ChatMessage.from_user("before restart")]},
    )
    checkpoint_context = DurableContext(_Claim(_agent_record("checkpoint", "builtin-agent")), adapter=object())
    record = _agent_record(
        "agent-resume",
        "builtin-agent",
        definition_revision=deployment.revision,
        validated_input={"messages": [ChatMessage.from_user("initial").to_dict()]},
        checkpoint=ExecutionCheckpoint(ExecutionKind.AGENT, _checkpoint_data(checkpoint_state, checkpoint_context)),
        application_state={"__hayhooks_resume_input": {"messages": [ChatMessage.from_user("after restart").to_dict()]}},
    )
    adapter = RestoringAdapter()
    context = DurableContext(_Claim(record), adapter=adapter)

    await deployment._run(context)

    assert adapter.restored_messages == ["before restart", "after restart"]
    assert context.resume_input is None


def test_durable_agent_uses_native_run_and_public_hooks(monkeypatch) -> None:
    app = _durable_app(monkeypatch, "agent", AgentWrapper())

    with TestClient(app) as client:
        submitted = client.post("/agent/run-durable", json={"message": "hello"})
        url = submitted.json()["links"]["self"]
        inspected = wait_for_status(client, url, "completed")

    assert inspected["result"]["last_message"]["content"][0]["text"] == "done"
    assert inspected["progress"][0]["kind"] == "checkpoint"


@pytest.mark.parametrize(
    ("module_name", "source", "kind"),
    [
        ("durable_execution_example", _DURABLE_EXECUTION_EXAMPLE, ExecutionKind.PIPELINE),
        ("durable_a2a_example", _DURABLE_A2A_EXAMPLE, ExecutionKind.AGENT),
    ],
)
def test_durable_examples_load(monkeypatch, module_name, source, kind) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with _example_module(module_name, source) as module:
        deployment = DurableDeployment(
            module_name, create_pipeline_wrapper_instance(module), InMemoryExecutionStoreProvider()
        )
        assert deployment.kind is kind
        assert deployment.revision


async def test_durable_execution_example_completes_retry_approval_and_real_pipeline(monkeypatch) -> None:
    monkeypatch.setattr(settings, "durable_retry_base_delay", 0)
    monkeypatch.setattr(settings, "durable_retry_max_delay", 0)
    module_name = "durable_execution_end_to_end_example"
    with _example_module(module_name, _DURABLE_EXECUTION_EXAMPLE) as module:
        wrapper = create_pipeline_wrapper_instance(module)
        deployment = DurableDeployment("durable-execution-example", wrapper, InMemoryExecutionStoreProvider())
        async with _started(deployment):
            _, submitted = await deployment.submit(
                {
                    "documents": [{"document_id": "guide", "content": "durable document preparation"}],
                    "fail_first_attempt": True,
                    "require_approval": True,
                    "demo_delay_seconds": 0.01,
                }
            )
            await wait_for_record(
                deployment,
                submitted.execution_id,
                lambda record: record.status is ExecutionStatus.WAITING,
                message="durable execution example did not reach approval",
            )

            assert await deployment.resume(submitted.execution_id, {"approved": True})
            completed = await wait_for_record(
                deployment, submitted.execution_id, message="durable execution example did not complete"
            )

    assert completed.status is ExecutionStatus.COMPLETED
    assert completed.attempt == 3
    assert completed.result["document_count"] == 1
    assert completed.result["chunk_count"] == 1
    assert completed.checkpoint is not None
    assert {event.kind for event in completed.progress} >= {
        "accepted",
        "retry_demo",
        "waiting",
        "checkpoint",
        "demo_delay",
        "completed",
    }


async def test_durable_a2a_example_tool_replays_its_external_effect_idempotently(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("HAYHOOKS_EXAMPLE_INDEX_DB", str(tmp_path / "indexing-effects.sqlite3"))
    monkeypatch.setenv("HAYHOOKS_EXAMPLE_TOOL_DELAY_SECONDS", "0")
    module_name = "durable_a2a_tool_example"
    with _example_module(module_name, _DURABLE_A2A_EXAMPLE) as module:
        record = _agent_record("a2a-tool-replay", "long-running-agent")
        claim = _Claim(record)
        context = DurableContext(claim, adapter=object())

        with execution_context_scope(context):
            first = await asyncio.to_thread(
                module.prepare_document_for_indexing.invoke,
                document_id="guide",
                content="one two three",
            )
            replay = await asyncio.to_thread(
                module.prepare_document_for_indexing.invoke,
                document_id="guide",
                content="one two three",
            )

        assert json.loads(first)["side_effect_applied"] is True
        assert json.loads(replay)["side_effect_applied"] is False
        assert claim.checkpoints == 2
        assert [event.kind for event in record.progress] == ["side_effect_committed", "side_effect_committed"]
