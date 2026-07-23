import importlib.util
from collections.abc import AsyncGenerator
from types import SimpleNamespace

import pytest
from haystack.dataclasses import StreamingChunk

from hayhooks.events import PipelineEvent
from hayhooks.server.pipelines import registry
from hayhooks.server.tracing import SPAN_A2A_RUN_AGENT
from hayhooks.server.utils.a2a_utils import (
    RESPONSE_ARTIFACT_NAME,
    _build_openai_messages,
    _execute_agent_task,
    _stream_item_to_text,
    create_agent_card,
    create_agent_executor,
    get_a2a_base_url,
    is_a2a_exposable,
)
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper
from hayhooks.server.utils.module_loader import _set_method_implementation_flags, create_pipeline_wrapper_instance

A2A_AVAILABLE = importlib.util.find_spec("a2a") is not None

# NOTE: Skip all tests in this file if a2a-sdk is not available
pytestmark = [
    pytest.mark.skipif(not A2A_AVAILABLE, reason="'a2a-sdk' package not installed"),
    pytest.mark.a2a,
]


@pytest.fixture(autouse=True)
def cleanup_test_pipelines():
    yield
    registry.clear()


class RecordingQueue:
    """Minimal EventQueue stand-in recording enqueued events."""

    def __init__(self):
        self.events = []

    async def enqueue_event(self, event):
        self.events.append(event)


class AsyncChatWrapper(BasePipelineWrapper):
    def setup(self):
        self.pipeline = object()

    async def run_chat_completion_async(self, model: str, messages: list[dict], body: dict) -> AsyncGenerator:
        async def generator():
            yield "Hello, "
            yield StreamingChunk(content="world")
            yield PipelineEvent(type="test_event", data={})  # must be skipped
            yield f" (question: {messages[-1]['content']})"

        return generator()


class SyncChatWrapper(BasePipelineWrapper):
    def setup(self):
        self.pipeline = object()

    def run_chat_completion(self, model: str, messages: list[dict], body: dict) -> str:
        return "sync response"


class FailingChatWrapper(BasePipelineWrapper):
    def setup(self):
        self.pipeline = object()

    async def run_chat_completion_async(self, model: str, messages: list[dict], body: dict) -> str:
        msg = "boom"
        raise RuntimeError(msg)


class ApiOnlyWrapper(BasePipelineWrapper):
    def setup(self):
        self.pipeline = object()

    def run_api(self, question: str) -> str:
        return question


def register_wrapper(name: str, wrapper_cls: type[BasePipelineWrapper], metadata: dict | None = None) -> None:
    wrapper = wrapper_cls()
    wrapper.setup()
    _set_method_implementation_flags(wrapper)
    base_metadata = {"description": f"{name} description", "skip_a2a": wrapper.skip_a2a, "a2a_card": wrapper.a2a_card}
    registry.add(name, wrapper, metadata={**base_metadata, **(metadata or {})})


def make_context(text: str = "hi", current_task=None):
    from a2a.helpers import new_text_message
    from a2a.types import Role

    message = new_text_message(text, role=Role.ROLE_USER) if text is not None else None
    return SimpleNamespace(message=message, current_task=current_task)


def get_status_states(events) -> list:
    from a2a.types import TaskStatusUpdateEvent

    return [event.status.state for event in events if isinstance(event, TaskStatusUpdateEvent)]


def get_artifact_events(events) -> list:
    from a2a.types import TaskArtifactUpdateEvent

    return [event for event in events if isinstance(event, TaskArtifactUpdateEvent)]


# --- Exposure rules ---


def test_is_a2a_exposable_unknown_pipeline():
    assert not is_a2a_exposable("non_existent_pipeline")


def test_is_a2a_exposable_chat_async():
    register_wrapper("chat_agent", AsyncChatWrapper)
    assert is_a2a_exposable("chat_agent")


def test_is_a2a_exposable_chat_sync():
    register_wrapper("sync_agent", SyncChatWrapper)
    assert is_a2a_exposable("sync_agent")


def test_is_a2a_exposable_api_only():
    register_wrapper("api_only", ApiOnlyWrapper)
    assert not is_a2a_exposable("api_only")


def test_is_a2a_exposable_skip_a2a():
    register_wrapper("chat_agent", AsyncChatWrapper, metadata={"skip_a2a": True})
    assert not is_a2a_exposable("chat_agent")


def test_is_a2a_exposable_native_only():
    from a2a.server.agent_execution import AgentExecutor

    from hayhooks.a2a import A2APipelineWrapper

    class NativeExecutor(AgentExecutor):
        async def execute(self, context, event_queue):
            pass

        async def cancel(self, context, event_queue):
            pass

    class NativeWrapper(A2APipelineWrapper):
        def setup(self):
            self.pipeline = object()

        def create_a2a_agent_executor(self):
            return NativeExecutor()

    register_wrapper("native_agent", NativeWrapper)
    assert is_a2a_exposable("native_agent")


def test_native_only_wrapper_passes_normal_deployment_validation():
    from a2a.server.agent_execution import AgentExecutor

    from hayhooks.a2a import A2APipelineWrapper

    class NativeExecutor(AgentExecutor):
        async def execute(self, context, event_queue):
            pass

        async def cancel(self, context, event_queue):
            pass

    class NativeOnlyWrapper(A2APipelineWrapper):
        def setup(self):
            self.pipeline = object()

        def create_a2a_agent_executor(self):
            return NativeExecutor()

    wrapper = create_pipeline_wrapper_instance(SimpleNamespace(PipelineWrapper=NativeOnlyWrapper))

    assert isinstance(wrapper, NativeOnlyWrapper)
    assert not wrapper._is_run_chat_completion_implemented
    assert not wrapper._is_run_chat_completion_async_implemented


# --- Base URL ---


def test_get_a2a_base_url_default(test_settings):
    test_settings.a2a_external_url = ""
    assert get_a2a_base_url() == f"http://{test_settings.a2a_host}:{test_settings.a2a_port}"


def test_get_a2a_base_url_external(test_settings):
    test_settings.a2a_external_url = "https://agents.example.com/"
    try:
        assert get_a2a_base_url() == "https://agents.example.com"
    finally:
        test_settings.a2a_external_url = ""


# --- Agent card ---


def test_create_agent_card_defaults():
    register_wrapper("chat_agent", AsyncChatWrapper)
    card = create_agent_card("chat_agent", "http://test:1418")

    assert card.name == "chat_agent"
    assert card.description == "chat_agent description"
    assert card.version == "1.0.0"
    assert card.capabilities.streaming is True
    assert list(card.default_input_modes) == ["text/plain"]
    assert len(card.supported_interfaces) == 1
    assert card.supported_interfaces[0].url == "http://test:1418/chat_agent/"
    assert card.supported_interfaces[0].protocol_binding == "JSONRPC"
    assert len(card.skills) == 1
    assert card.skills[0].id == "chat_agent"
    assert list(card.skills[0].tags) == ["haystack", "hayhooks"]


def test_create_agent_card_empty_description_fallback():
    register_wrapper("chat_agent", AsyncChatWrapper, metadata={"description": ""})
    card = create_agent_card("chat_agent", "http://test:1418")
    assert card.description == "Haystack pipeline 'chat_agent' deployed with Hayhooks"


def test_create_agent_card_overrides():
    overrides = {
        "name": "Weather Agent",
        "description": "Provides weather forecasts",
        "version": "2.1.0",
        "skills": [
            {
                "id": "get_weather",
                "name": "Get weather",
                "description": "Current weather for a location",
                "tags": ["weather"],
                "examples": ["What's the weather in Berlin?"],
            }
        ],
    }
    register_wrapper("chat_agent", AsyncChatWrapper, metadata={"a2a_card": overrides})
    card = create_agent_card("chat_agent", "http://test:1418")

    assert card.name == "Weather Agent"
    assert card.description == "Provides weather forecasts"
    assert card.version == "2.1.0"
    # URL is always derived from the pipeline name, not overridable
    assert card.supported_interfaces[0].url == "http://test:1418/chat_agent/"
    assert len(card.skills) == 1
    assert card.skills[0].id == "get_weather"
    assert list(card.skills[0].tags) == ["weather"]
    assert list(card.skills[0].examples) == ["What's the weather in Berlin?"]


# --- Stream item mapping ---


def test_stream_item_to_text():
    assert _stream_item_to_text(StreamingChunk(content="hello")) == "hello"
    assert _stream_item_to_text(StreamingChunk(content="")) is None
    assert _stream_item_to_text("plain") == "plain"
    assert _stream_item_to_text("") is None
    assert _stream_item_to_text(b"bytes") == "bytes"
    assert _stream_item_to_text(PipelineEvent(type="test_event", data={})) is None
    assert _stream_item_to_text({"type": "event"}) is None
    assert _stream_item_to_text(None) is None


# --- OpenAI message mapping ---


def test_build_openai_messages_from_message_only():
    context = make_context("what is the weather?")
    assert _build_openai_messages(context) == [{"role": "user", "content": "what is the weather?"}]


def test_build_openai_messages_with_task_history():
    from a2a.helpers import new_task_from_user_message, new_text_message
    from a2a.types import Role

    first_message = new_text_message("first question", role=Role.ROLE_USER)
    task = new_task_from_user_message(first_message)
    task.history.append(new_text_message("first answer", role=Role.ROLE_AGENT))

    context = make_context("second question", current_task=task)
    assert _build_openai_messages(context) == [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
    ]


def test_build_openai_messages_deduplicates_current_message():
    from a2a.helpers import new_task_from_user_message, new_text_message
    from a2a.types import Role

    message = new_text_message("hello", role=Role.ROLE_USER)
    task = new_task_from_user_message(message)  # history already contains the message

    context = SimpleNamespace(message=message, current_task=task)
    assert _build_openai_messages(context) == [{"role": "user", "content": "hello"}]


def test_build_openai_messages_keeps_new_message_matching_history_text():
    from a2a.helpers import new_task_from_user_message, new_text_message
    from a2a.types import Role

    # The agent's last reply is "yes" and the user replies with the identical text:
    # dedup is by message id, so the new user turn must NOT be dropped
    task = new_task_from_user_message(new_text_message("continue?", role=Role.ROLE_USER))
    task.history.append(new_text_message("yes", role=Role.ROLE_AGENT))

    context = make_context("yes", current_task=task)
    assert _build_openai_messages(context) == [
        {"role": "user", "content": "continue?"},
        {"role": "assistant", "content": "yes"},
        {"role": "user", "content": "yes"},
    ]


# --- Task execution ---


@pytest.mark.asyncio
async def test_execute_agent_task_string_result():
    from a2a.types import Task, TaskState

    register_wrapper("sync_agent", SyncChatWrapper)
    queue = RecordingQueue()

    await _execute_agent_task("sync_agent", make_context(), queue)

    assert isinstance(queue.events[0], Task)
    assert get_status_states(queue.events) == [TaskState.TASK_STATE_WORKING, TaskState.TASK_STATE_COMPLETED]

    artifact_events = get_artifact_events(queue.events)
    assert len(artifact_events) == 1
    assert artifact_events[0].artifact.name == RESPONSE_ARTIFACT_NAME
    assert artifact_events[0].artifact.parts[0].text == "sync response"
    assert artifact_events[0].last_chunk is True


@pytest.mark.asyncio
async def test_execute_agent_task_streaming_result():
    from a2a.types import TaskState

    register_wrapper("chat_agent", AsyncChatWrapper)
    queue = RecordingQueue()

    await _execute_agent_task("chat_agent", make_context("hi"), queue)

    assert get_status_states(queue.events)[-1] == TaskState.TASK_STATE_COMPLETED

    artifact_events = get_artifact_events(queue.events)
    # PipelineEvent items are skipped, text chunks are streamed incrementally
    assert len(artifact_events) == 4
    texts = [event.artifact.parts[0].text for event in artifact_events]
    assert texts == ["Hello, ", "world", " (question: hi)", ""]
    # All chunks belong to the same artifact; an empty marker finalizes iterator output
    assert len({event.artifact.artifact_id for event in artifact_events}) == 1
    assert [event.last_chunk for event in artifact_events] == [False, False, False, True]
    assert artifact_events[0].append is False
    assert artifact_events[1].append is True


@pytest.mark.asyncio
async def test_execute_agent_task_error_sets_failed_state():
    from a2a.types import TaskState

    register_wrapper("failing_agent", FailingChatWrapper)
    queue = RecordingQueue()

    await _execute_agent_task("failing_agent", make_context(), queue)

    states = get_status_states(queue.events)
    assert states[-1] == TaskState.TASK_STATE_FAILED
    failed_events = [e for e in queue.events if getattr(e, "status", None) and e.status.state == states[-1]]
    assert "boom" in failed_events[-1].status.message.parts[0].text


@pytest.mark.asyncio
async def test_execute_agent_task_none_result_fails():
    from a2a.types import TaskState

    class NoneResultWrapper(BasePipelineWrapper):
        def setup(self):
            self.pipeline = object()

        def run_chat_completion(self, model: str, messages: list[dict], body: dict):
            return None  # contract violation: must return str or generator

    register_wrapper("none_agent", NoneResultWrapper)
    queue = RecordingQueue()

    await _execute_agent_task("none_agent", make_context(), queue)

    assert get_status_states(queue.events)[-1] == TaskState.TASK_STATE_FAILED


@pytest.mark.asyncio
async def test_execute_agent_task_unknown_pipeline_fails():
    from a2a.types import TaskState

    queue = RecordingQueue()
    await _execute_agent_task("non_existent", make_context(), queue)
    assert get_status_states(queue.events)[-1] == TaskState.TASK_STATE_FAILED


@pytest.mark.asyncio
async def test_execute_agent_task_emits_trace_span(recording_tracer):
    register_wrapper("sync_agent", SyncChatWrapper)
    await _execute_agent_task("sync_agent", make_context(), RecordingQueue())

    spans = [span for span in recording_tracer.spans if span.operation_name == SPAN_A2A_RUN_AGENT]
    assert spans
    assert spans[-1].tags["hayhooks.pipeline.name"] == "sync_agent"
    assert spans[-1].tags["hayhooks.transport"] == "a2a"


def test_create_agent_executor():
    executor = create_agent_executor("some_pipeline")
    assert executor.pipeline_name == "some_pipeline"
    assert hasattr(executor, "execute")
    assert hasattr(executor, "cancel")


def test_create_agent_executor_selects_native_executor():
    from a2a.server.agent_execution import AgentExecutor

    from hayhooks.a2a import A2APipelineWrapper
    from hayhooks.server.a2a.executor import ChatCompletionAgentExecutor

    class NativeExecutor(AgentExecutor):
        async def execute(self, context, event_queue):
            pass

        async def cancel(self, context, event_queue):
            pass

    class HybridWrapper(A2APipelineWrapper):
        def __init__(self):
            super().__init__()
            self.factory_calls = 0

        def setup(self):
            self.pipeline = object()

        def run_chat_completion(self, model: str, messages: list[dict], body: dict) -> str:
            return "chat"

        def create_a2a_agent_executor(self):
            self.factory_calls += 1
            return NativeExecutor()

    wrapper = HybridWrapper()
    wrapper.setup()
    _set_method_implementation_flags(wrapper)

    executor = create_agent_executor(wrapper, "hybrid_agent")

    assert isinstance(executor, NativeExecutor)
    assert not isinstance(executor, ChatCompletionAgentExecutor)
    assert wrapper.factory_calls == 1


def test_create_agent_executor_rejects_invalid_native_executor():
    from hayhooks.a2a import A2APipelineWrapper

    class BadNativeWrapper(A2APipelineWrapper):
        def setup(self):
            self.pipeline = object()

        def create_a2a_agent_executor(self):
            return object()

    wrapper = BadNativeWrapper()
    wrapper.setup()
    _set_method_implementation_flags(wrapper)

    with pytest.raises(TypeError, match="bad_agent"):
        create_agent_executor(wrapper, "bad_agent")


def test_create_a2a_app_isolates_invalid_native_executor():
    from hayhooks.a2a import A2APipelineWrapper
    from hayhooks.server.utils.a2a_utils import create_a2a_app

    class BadNativeWrapper(A2APipelineWrapper):
        def setup(self):
            self.pipeline = object()

        def create_a2a_agent_executor(self):
            return object()

    register_wrapper("bad_agent", BadNativeWrapper)
    register_wrapper("chat_agent", AsyncChatWrapper)

    app = create_a2a_app(base_url="http://test:1418")
    route_paths = [getattr(route, "path", "") for route in app.routes]

    assert "/chat_agent" in route_paths
    assert "/bad_agent" not in route_paths


def test_create_a2a_app_manages_native_executor_lifecycle():
    from a2a.server.agent_execution import AgentExecutor
    from starlette.testclient import TestClient

    from hayhooks.a2a import A2APipelineWrapper
    from hayhooks.server.a2a import create_a2a_app

    events = []

    class ManagedExecutor(AgentExecutor):
        async def start(self):
            events.append("start")

        async def close(self):
            events.append("close")

        async def execute(self, context, event_queue):
            pass

        async def cancel(self, context, event_queue):
            pass

    class ManagedWrapper(A2APipelineWrapper):
        def setup(self):
            self.pipeline = object()

        def create_a2a_agent_executor(self):
            return ManagedExecutor()

    register_wrapper("managed_agent", ManagedWrapper)
    app = create_a2a_app(base_url="http://test:1418")

    with TestClient(app):
        assert events == ["start"]

    assert events == ["start", "close"]


def test_runtime_passes_agent_name_to_task_store_provider():
    from a2a.server.tasks import InMemoryTaskStore

    from hayhooks.a2a import TaskStoreProvider
    from hayhooks.server.a2a import A2ARuntime

    class RecordingTaskStoreProvider(TaskStoreProvider):
        def __init__(self):
            self.agent_names = []

        def create_task_store(self, agent_name):
            self.agent_names.append(agent_name)
            return InMemoryTaskStore()

    provider = RecordingTaskStoreProvider()
    runtime = A2ARuntime(task_store_provider=provider)

    first_store = runtime.create_task_store("first_agent")
    second_store = runtime.create_task_store("second_agent")

    assert isinstance(first_store, InMemoryTaskStore)
    assert isinstance(second_store, InMemoryTaskStore)
    assert first_store is not second_store
    assert provider.agent_names == ["first_agent", "second_agent"]


def test_runtime_rejects_invalid_task_store_from_provider():
    from hayhooks.a2a import TaskStoreProvider
    from hayhooks.server.a2a import A2ARuntime

    class InvalidTaskStoreProvider(TaskStoreProvider):
        def create_task_store(self, _agent_name):
            return object()

    runtime = A2ARuntime(task_store_provider=InvalidTaskStoreProvider())

    with pytest.raises(TypeError, match=r"InvalidTaskStoreProvider.*invalid_agent"):
        runtime.create_task_store("invalid_agent")


def test_load_task_store_provider(monkeypatch):
    from a2a.server.tasks import InMemoryTaskStore

    from hayhooks.a2a import TaskStoreProvider
    from hayhooks.server.a2a import load_task_store_provider

    class CustomTaskStoreProvider(TaskStoreProvider):
        def create_task_store(self, _agent_name):
            return InMemoryTaskStore()

    module = SimpleNamespace(CustomTaskStoreProvider=CustomTaskStoreProvider)
    monkeypatch.setattr("hayhooks.server.a2a.runtime.importlib.import_module", lambda _name: module)

    provider = load_task_store_provider("my_project.a2a:CustomTaskStoreProvider")

    assert isinstance(provider, CustomTaskStoreProvider)


@pytest.mark.parametrize("import_path", ["", "module", ":Provider", "module:"])
def test_load_task_store_provider_rejects_invalid_import_path(import_path):
    from hayhooks.server.a2a import load_task_store_provider

    with pytest.raises(ValueError, match="module:ClassName"):
        load_task_store_provider(import_path)


@pytest.mark.asyncio
async def test_runtime_closes_task_store_provider():
    from a2a.server.tasks import InMemoryTaskStore

    from hayhooks.a2a import TaskStoreProvider
    from hayhooks.server.a2a import A2ARuntime

    class CloseableTaskStoreProvider(TaskStoreProvider):
        def __init__(self):
            self.closed = False

        def create_task_store(self, _agent_name):
            return InMemoryTaskStore()

        async def close(self):
            self.closed = True

    provider = CloseableTaskStoreProvider()

    await A2ARuntime(task_store_provider=provider).close()

    assert provider.closed


@pytest.mark.asyncio
async def test_runtime_starts_and_closes_executor_lifecycles_before_provider():
    from a2a.server.tasks import InMemoryTaskStore

    from hayhooks.a2a import TaskStoreProvider
    from hayhooks.server.a2a import A2ARuntime

    events = []

    class RecordingProvider(TaskStoreProvider):
        def create_task_store(self, _agent_name):
            return InMemoryTaskStore()

        async def close(self):
            events.append("provider.close")

    class LifecycleExecutor:
        async def start(self):
            events.append("executor.start")

        async def close(self):
            events.append("executor.close")

    runtime = A2ARuntime(task_store_provider=RecordingProvider())
    runtime.register_agent_executor(LifecycleExecutor())

    await runtime.start()
    await runtime.close()

    assert events == ["executor.start", "executor.close", "provider.close"]


def test_create_a2a_app_loads_and_closes_configured_task_store_provider(monkeypatch):
    from a2a.server.tasks import InMemoryTaskStore
    from starlette.testclient import TestClient

    from hayhooks.a2a import TaskStoreProvider
    from hayhooks.server.a2a import create_a2a_app
    from hayhooks.settings import settings

    class ConfiguredTaskStoreProvider(TaskStoreProvider):
        def __init__(self):
            self.agent_names = []
            self.closed = False

        def create_task_store(self, agent_name):
            self.agent_names.append(agent_name)
            return InMemoryTaskStore()

        async def close(self):
            self.closed = True

    provider = ConfiguredTaskStoreProvider()
    provider_configurations = []
    monkeypatch.setattr(settings, "a2a_task_store", "memory")
    monkeypatch.setattr(settings, "a2a_task_store_provider", "my_project.a2a:ConfiguredTaskStoreProvider")
    monkeypatch.setattr(settings, "a2a_redis_url", "redis://localhost:6379/0")
    monkeypatch.setattr(settings, "a2a_redis_key_prefix", "hayhooks:a2a")
    monkeypatch.setattr(
        "hayhooks.server.a2a.app.create_task_store_provider",
        lambda **kwargs: provider_configurations.append(kwargs) or provider,
    )
    register_wrapper("configured_agent", AsyncChatWrapper)

    app = create_a2a_app(base_url="http://test:1418")
    with TestClient(app):
        assert not provider.closed

    assert provider_configurations == [
        {
            "backend": "memory",
            "custom_provider": "my_project.a2a:ConfiguredTaskStoreProvider",
            "redis_url": "redis://localhost:6379/0",
            "redis_key_prefix": "hayhooks:a2a",
        }
    ]
    assert provider.agent_names == ["configured_agent"]
    assert provider.closed


def test_a2a_app_does_not_share_durable_redis_client_with_a_different_endpoint(monkeypatch):
    from hayhooks.server.a2a.app import _create_app_task_store_provider
    from hayhooks.settings import settings

    configurations = []
    monkeypatch.setattr(settings, "a2a_task_store", "redis")
    monkeypatch.setattr(settings, "a2a_task_store_provider", "")
    monkeypatch.setattr(settings, "a2a_redis_url", "redis://a2a.example:6379/0")
    monkeypatch.setattr(settings, "a2a_redis_key_prefix", "hayhooks:a2a")
    monkeypatch.setattr(settings, "durable_store", "redis")
    monkeypatch.setattr(settings, "durable_redis_url", "redis://durable.example:6379/0")
    shared_client = object()
    monkeypatch.setattr(
        "hayhooks.server.a2a.app.durable_runtime.shared_redis_client",
        lambda: shared_client,
    )
    monkeypatch.setattr(
        "hayhooks.server.a2a.app.create_task_store_provider",
        lambda **kwargs: configurations.append(kwargs) or object(),
    )

    _create_app_task_store_provider(durable_agents_deployed=True)

    assert configurations == [
        {
            "backend": "redis",
            "custom_provider": "",
            "redis_url": "redis://a2a.example:6379/0",
            "redis_key_prefix": "hayhooks:a2a",
        }
    ]
