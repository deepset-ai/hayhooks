import asyncio
import importlib
import importlib.metadata
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from haystack import Pipeline, component
from haystack.components.agents import Agent
from haystack.components.agents.state import State
from haystack.core.errors import PipelineRuntimeError
from haystack.dataclasses import ChatMessage
from haystack.dataclasses.breakpoints import PipelineSnapshot
from pydantic import BaseModel

from hayhooks import BasePipelineWrapper, DurableContext, DurableOptions
from hayhooks.durable_adapters import (
    HaystackDurableAdapter,
    _checkpoint_data,
    _restore_agent_state,
    definition_revision,
)
from hayhooks.durable_runtime import DurableDeployment
from hayhooks.execution import (
    ExecutionCheckpoint,
    ExecutionKind,
    ExecutionRecord,
    ExecutionStatus,
    InMemoryExecutionStore,
    InMemoryExecutionStoreProvider,
)
from hayhooks.server.app import create_app
from hayhooks.server.pipelines.registry import registry
from hayhooks.server.utils.deploy_utils import add_pipeline_api_route
from hayhooks.server.utils.module_loader import (
    _set_method_implementation_flags,
    create_pipeline_wrapper_instance,
    load_pipeline_module,
    unload_pipeline_modules,
)
from hayhooks.settings import settings

pytestmark = pytest.mark.skipif(
    not importlib.metadata.version("haystack-ai").startswith("3."), reason="durable execution requires Haystack 3"
)

_DURABLE_CHAT_EXAMPLE = Path("examples/durable_chat_with_website/pipelines/chat_with_website")


class Request(BaseModel):
    value: int


class Result(BaseModel):
    value: int


class ResumeInput(BaseModel):
    approved: bool


class Wrapper(BasePipelineWrapper):
    def setup(self) -> None:
        self.pipeline = Pipeline()

    async def run_durable_async(self, context: DurableContext, request: Request) -> Result:
        await context.report_progress("working")
        return Result(value=request.value + 1)


class InvalidResultWrapper(Wrapper):
    async def run_durable_async(self, context: DurableContext, request: Request) -> Result:
        return {"value": "not-an-integer"}  # type: ignore[return-value]


class SyncWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        self.pipeline = Pipeline()

    def run_durable(self, context: DurableContext, request: Request) -> Result:
        context.report_progress_sync("working in a worker thread")
        context.check_cancelled_sync()
        return Result(value=request.value + 2)


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
    durable_options = DurableOptions(revision="checkpoint-pipeline-v1")

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
    durable_options = DurableOptions(revision="test-agent-v1")

    def setup(self) -> None:
        self.pipeline = Agent(chat_generator=FakeChatGenerator(), tools=[])

    async def run_durable_async(self, context: DurableContext, request: AgentRequest) -> dict:
        return await context.run_agent_async(messages=[ChatMessage.from_user(request.message)])


class BuiltinAgentWrapper(BasePipelineWrapper):
    durable = True
    durable_options = DurableOptions(revision="builtin-agent-v1")

    def setup(self) -> None:
        self.pipeline = Agent(chat_generator=FakeChatGenerator(), tools=[])


@pytest.fixture(autouse=True)
def clean_registry():
    registry.clear()
    yield
    registry.clear()


def test_durable_rest_submission_is_direct_typed_and_idempotent(monkeypatch) -> None:
    monkeypatch.setattr(settings, "durable_store", "memory")
    wrapper = Wrapper()
    wrapper.setup()
    _set_method_implementation_flags(wrapper)
    registry.add("job", wrapper)
    app = create_app()
    add_pipeline_api_route(app, "job", wrapper)

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
        for _ in range(100):
            inspected = client.get(body["links"]["self"])
            if inspected.json()["status"] == "completed":
                break
            time.sleep(0.01)
        else:
            pytest.fail("durable execution did not complete")

    assert inspected.json()["result"] == {"value": 5}
    assert submitted.headers["Location"] == body["links"]["self"]


def test_durable_result_annotation_is_validated_before_completion(monkeypatch) -> None:
    monkeypatch.setattr(settings, "durable_store", "memory")
    wrapper = InvalidResultWrapper()
    wrapper.setup()
    _set_method_implementation_flags(wrapper)
    registry.add("invalid-result", wrapper)
    app = create_app()
    add_pipeline_api_route(app, "invalid-result", wrapper)

    with TestClient(app) as client:
        submitted = client.post("/invalid-result/run-durable", json={"value": 4})
        url = submitted.json()["links"]["self"]
        for _ in range(100):
            inspected = client.get(url)
            if inspected.json()["status"] == "failed":
                break
            time.sleep(0.01)
        else:
            pytest.fail("invalid durable result did not become a terminal failure")

    assert inspected.json()["result"] is None
    assert inspected.json()["error"] == {
        "type": "ValueError",
        "message": "Durable method result does not match its declared return annotation",
        "retryable": False,
        "code": None,
    }


def test_durable_rest_rejects_mismatched_idempotency_payload(monkeypatch) -> None:
    monkeypatch.setattr(settings, "durable_store", "memory")
    wrapper = Wrapper()
    wrapper.setup()
    _set_method_implementation_flags(wrapper)
    registry.add("job", wrapper)
    app = create_app()
    add_pipeline_api_route(app, "job", wrapper)

    with TestClient(app) as client:
        first = client.post("/job/run-durable", json={"value": 4}, headers={"Idempotency-Key": "same"})
        conflict = client.post("/job/run-durable", json={"value": 5}, headers={"Idempotency-Key": "same"})

    assert first.status_code == 202
    assert conflict.status_code == 409


def test_durable_rest_rejects_non_path_safe_idempotency_keys(monkeypatch) -> None:
    monkeypatch.setattr(settings, "durable_store", "memory")
    wrapper = Wrapper()
    wrapper.setup()
    _set_method_implementation_flags(wrapper)
    registry.add("job", wrapper)
    app = create_app()
    add_pipeline_api_route(app, "job", wrapper)

    with TestClient(app) as client:
        response = client.post("/job/run-durable", json={"value": 4}, headers={"Idempotency-Key": "part/child"})

    assert response.status_code == 422
    assert "URL-safe" in response.json()["detail"]


def test_durable_rest_maps_oversized_validated_request_to_422(monkeypatch) -> None:
    monkeypatch.setattr(settings, "durable_store", "memory")
    monkeypatch.setattr(settings, "durable_max_record_bytes", 5)
    wrapper = Wrapper()
    wrapper.setup()
    _set_method_implementation_flags(wrapper)
    registry.add("job", wrapper)
    app = create_app()
    add_pipeline_api_route(app, "job", wrapper)

    with TestClient(app) as client:
        response = client.post("/job/run-durable", json={"value": 123})

    assert response.status_code == 422
    assert "durable execution limit" in response.json()["detail"]


def test_durable_waiting_resume_is_typed_and_private_state_is_hidden(monkeypatch) -> None:
    monkeypatch.setattr(settings, "durable_store", "memory")
    wrapper = WaitingWrapper()
    wrapper.setup()
    _set_method_implementation_flags(wrapper)
    registry.add("approval", wrapper)
    app = create_app()
    add_pipeline_api_route(app, "approval", wrapper)

    with TestClient(app) as client:
        submitted = client.post("/approval/run-durable", json={"value": 7})
        url = submitted.json()["links"]["self"]
        for _ in range(100):
            waiting = client.get(url)
            if waiting.json()["status"] == "waiting":
                break
            time.sleep(0.01)
        else:
            pytest.fail("execution did not wait")
        assert waiting.json()["waiting"] == {
            "kind": "approval",
            "message": "Approve this job",
            "expected_input_schema": ResumeInput.model_json_schema(),
        }
        invalid = client.post(f"{url}/resume", json={"approved": "not-a-bool"})
        assert invalid.status_code == 422
        resumed = client.post(f"{url}/resume", json={"approved": True})
        assert resumed.status_code == 202
        for _ in range(100):
            completed = client.get(url)
            if completed.json()["status"] == "completed":
                break
            time.sleep(0.01)
        else:
            pytest.fail("resumed execution did not complete")

    assert completed.json()["result"] == {"value": 7}
    openapi = app.openapi()
    resume_schema = openapi["paths"]["/approval/executions/{execution_id}/resume"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]
    assert "ResumeInput" in str(resume_schema)


def test_durable_rest_enforces_configured_trusted_owner_header(monkeypatch) -> None:
    monkeypatch.setattr(settings, "durable_store", "memory")
    monkeypatch.setattr(settings, "durable_trusted_owner_header", "X-Authenticated-Owner")
    wrapper = Wrapper()
    wrapper.setup()
    _set_method_implementation_flags(wrapper)
    registry.add("owned", wrapper)
    app = create_app()
    add_pipeline_api_route(app, "owned", wrapper)

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


def test_automatic_revision_includes_wrapper_implementation() -> None:
    class FirstWrapper(BasePipelineWrapper):
        def setup(self) -> None:
            self.pipeline = Pipeline()

        def run_api(self, value: int) -> int:
            return value + 1

    class SecondWrapper(BasePipelineWrapper):
        def setup(self) -> None:
            self.pipeline = Pipeline()

        def run_api(self, value: int) -> int:
            return value + 2

    first = FirstWrapper()
    first.setup()
    second = SecondWrapper()
    second.setup()
    FirstWrapper.__name__ = SecondWrapper.__name__ = "PipelineWrapper"
    FirstWrapper.__qualname__ = SecondWrapper.__qualname__ = "PipelineWrapper"

    assert definition_revision(first.pipeline, wrapper=first) != definition_revision(second.pipeline, wrapper=second)


def test_automatic_revision_includes_custom_component_implementation(monkeypatch, tmp_path) -> None:
    module_path = tmp_path / "revision_component.py"
    source = """
from haystack import component

@component
class RevisionComponent:
    @component.output_types(value=int)
    def run(self, value: int):
        return {"value": value + INCREMENT}
"""
    monkeypatch.syspath_prepend(str(tmp_path))
    module_path.write_text(f"INCREMENT = 1\n{source}")
    first_module = importlib.import_module("revision_component")
    first_pipeline = Pipeline()
    first_pipeline.add_component("component", first_module.RevisionComponent())
    first_revision = definition_revision(first_pipeline)

    module_path.write_text(f"INCREMENT = 200\n{source}")
    importlib.invalidate_caches()
    sys.modules.pop("revision_component", None)
    second_module = importlib.import_module("revision_component")
    second_pipeline = Pipeline()
    second_pipeline.add_component("component", second_module.RevisionComponent())

    assert definition_revision(second_pipeline) != first_revision


def test_sync_durable_wrapper_uses_context_sync_controls(monkeypatch) -> None:
    monkeypatch.setattr(settings, "durable_store", "memory")
    wrapper = SyncWrapper()
    wrapper.setup()
    _set_method_implementation_flags(wrapper)
    registry.add("sync-job", wrapper)
    app = create_app()
    add_pipeline_api_route(app, "sync-job", wrapper)

    with TestClient(app) as client:
        submitted = client.post("/sync-job/run-durable", json={"value": 4})
        url = submitted.json()["links"]["self"]
        for _ in range(100):
            inspected = client.get(url)
            if inspected.json()["status"] == "completed":
                break
            time.sleep(0.01)
        else:
            pytest.fail("sync durable execution did not complete")

    assert inspected.json()["result"] == {"value": 6}
    assert inspected.json()["progress"][0]["message"] == "working in a worker thread"


@pytest.mark.asyncio
async def test_sync_work_retains_claim_after_shutdown_grace_until_thread_exits(monkeypatch) -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingWrapper(BasePipelineWrapper):
        def setup(self) -> None:
            self.pipeline = Pipeline()

        def run_durable(self, context: DurableContext, request: Request) -> Result:
            started.set()
            assert release.wait(timeout=5)
            return Result(value=request.value)

    monkeypatch.setattr(settings, "durable_shutdown_grace_period", 0.001)
    provider = InMemoryExecutionStoreProvider()
    wrapper = BlockingWrapper()
    wrapper.setup()
    _set_method_implementation_flags(wrapper)
    deployment = DurableDeployment("blocking", wrapper, provider)
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_pipeline_snapshot_round_trip_skips_completed_components_after_retry(monkeypatch) -> None:
    monkeypatch.setattr(settings, "durable_retry_base_delay", 0)
    monkeypatch.setattr(settings, "durable_retry_max_delay", 0)
    provider = InMemoryExecutionStoreProvider()
    wrapper = CheckpointPipelineWrapper()
    wrapper.setup()
    _set_method_implementation_flags(wrapper)
    deployment = DurableDeployment("checkpoint-pipeline", wrapper, provider)
    await deployment.start()
    try:
        _, submitted = await deployment.submit({"value": 3})
        for _ in range(200):
            completed = await deployment.store.get(submitted.execution_id)
            if completed is not None and completed.terminal:
                break
            await asyncio.sleep(0.005)
        else:
            pytest.fail("checkpointed Pipeline did not finish its retry")
    finally:
        await deployment.close()

    assert completed is not None
    assert completed.status.value == "completed"
    assert completed.result == {"value": 5}
    assert completed.attempt == 2
    assert completed.checkpoint is not None
    PipelineSnapshot.from_dict(completed.checkpoint.data["snapshot"])
    assert wrapper.first.calls == 1
    assert wrapper.second.calls == 2
    checkpoint_events = [event for event in completed.progress if event.kind == "checkpoint"]
    assert len(checkpoint_events) >= 2


@pytest.mark.asyncio
async def test_agent_state_checkpoint_restores_custom_state_and_typed_resume_messages() -> None:
    class Claim:
        def __init__(self, record: ExecutionRecord) -> None:
            self.record = record

    schema = {
        "counter": {"type": int},
        "tools": {"type": list},
        "hook_context": {"type": dict},
    }
    record = ExecutionRecord(
        execution_id="agent-state",
        execution_kind=ExecutionKind.AGENT,
        deployment_name="agent",
        definition_revision="revision",
        validated_input={"messages": []},
    )
    checkpoint_state = State(
        schema=schema,
        data={
            "counter": 7,
            "messages": [ChatMessage.from_user("before restart")],
            "tools": ["old tool"],
            "hook_context": {"request": "old"},
        },
    )
    checkpoint_context = DurableContext(Claim(record), adapter=object())
    checkpoint_data = _checkpoint_data(checkpoint_state, checkpoint_context)
    assert "tools" not in checkpoint_data["data"]
    assert "hook_context" not in checkpoint_data["data"]
    record.checkpoint = ExecutionCheckpoint(ExecutionKind.AGENT, checkpoint_data)

    store = InMemoryExecutionStore()
    await store.initialize()
    assert await store.submit(record)
    waiting_claim = await store.claim_next("before-resume")
    assert waiting_claim is not None
    async with waiting_claim:
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


@pytest.mark.asyncio
async def test_builtin_agent_leaves_resume_input_for_checkpoint_restoration() -> None:
    class Claim:
        def __init__(self, record: ExecutionRecord) -> None:
            self.record = record

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
    wrapper.setup()
    _set_method_implementation_flags(wrapper)
    deployment = DurableDeployment("builtin-agent", wrapper, InMemoryExecutionStoreProvider())
    checkpoint_state = State(
        schema={"messages": {"type": list[ChatMessage]}},
        data={"messages": [ChatMessage.from_user("before restart")]},
    )
    record = ExecutionRecord(
        execution_id="agent-resume",
        execution_kind=ExecutionKind.AGENT,
        deployment_name="builtin-agent",
        definition_revision=deployment.revision,
        validated_input={"messages": [ChatMessage.from_user("initial").to_dict()]},
        checkpoint=ExecutionCheckpoint(
            ExecutionKind.AGENT,
            _checkpoint_data(
                checkpoint_state,
                DurableContext(
                    Claim(
                        ExecutionRecord(
                            execution_id="checkpoint",
                            execution_kind=ExecutionKind.AGENT,
                            deployment_name="builtin-agent",
                            definition_revision=deployment.revision,
                            validated_input={"messages": []},
                        )
                    ),
                    adapter=object(),
                ),
            ),
        ),
        application_state={
            "__hayhooks_resume_input": {
                "messages": [ChatMessage.from_user("after restart").to_dict()],
            }
        },
    )
    adapter = RestoringAdapter()
    context = DurableContext(Claim(record), adapter=adapter)

    await deployment._run(context)

    assert adapter.restored_messages == ["before restart", "after restart"]
    assert context.resume_input is None


def test_durable_agent_uses_native_run_and_public_hooks(monkeypatch) -> None:
    monkeypatch.setattr(settings, "durable_store", "memory")
    wrapper = AgentWrapper()
    wrapper.setup()
    _set_method_implementation_flags(wrapper)
    registry.add("agent", wrapper)
    app = create_app()
    add_pipeline_api_route(app, "agent", wrapper)

    with TestClient(app) as client:
        submitted = client.post("/agent/run-durable", json={"message": "hello"})
        url = submitted.json()["links"]["self"]
        for _ in range(100):
            inspected = client.get(url)
            if inspected.json()["status"] == "completed":
                break
            time.sleep(0.01)
        else:
            pytest.fail("durable Agent did not complete")

    assert inspected.json()["result"]["last_message"]["content"][0]["text"] == "done"
    assert inspected.json()["progress"][0]["kind"] == "checkpoint"


@pytest.mark.asyncio
async def test_durable_chat_with_website_example_loads_and_maps_pipeline_input(monkeypatch) -> None:
    class ExampleContext:
        def __init__(self) -> None:
            self.progress: list[tuple[str, str]] = []
            self.pipeline_input = None
            self.checkpoint_at = None

        async def report_progress(self, message, *, kind="progress", metadata=None):
            self.progress.append((kind, message))

        async def run_pipeline_async(self, data, *, checkpoint_at):
            self.pipeline_input = data
            self.checkpoint_at = checkpoint_at
            return {"llm": {"replies": [ChatMessage.from_assistant(text="Generators yield values lazily.")]}}

        async def retry(self, message, *, delay=None):
            pytest.fail(f"unexpected retry: {message}, delay={delay}")

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    module_name = "durable_chat_with_website_example"
    module = load_pipeline_module(module_name, _DURABLE_CHAT_EXAMPLE)
    try:
        wrapper = create_pipeline_wrapper_instance(module)
        assert list(wrapper.pipeline.graph.nodes) == ["converter", "fetcher", "llm", "prompt"]
        assert wrapper._is_run_api_implemented
        assert wrapper._is_run_durable_async_implemented

        request = module.WebsiteQuestionRequest(
            urls=["https://docs.python.org/3/howto/functional.html"],
            question="What is a generator?",
        )
        context = ExampleContext()
        result = await wrapper.run_durable_async(context, request)

        assert context.pipeline_input == {
            "fetcher": {"urls": ["https://docs.python.org/3/howto/functional.html"]},
            "prompt": {"query": "What is a generator?"},
        }
        assert context.checkpoint_at == ["converter", "prompt", "llm"]
        assert context.progress == [
            ("accepted", "Website question accepted"),
            ("completed", "Website answer completed"),
        ]
        assert result.model_dump() == {
            "answer": "Generators yield values lazily.",
            "sources": ["https://docs.python.org/3/howto/functional.html"],
        }
    finally:
        unload_pipeline_modules(module_name)
