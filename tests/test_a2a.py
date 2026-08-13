import importlib.util
from collections.abc import AsyncGenerator
from types import SimpleNamespace

import pytest
from haystack.dataclasses import StreamingChunk

from hayhooks.events import PipelineEvent
from hayhooks.server.a2a.cards import create_agent_card, get_a2a_base_url, is_a2a_exposable
from hayhooks.server.a2a.executor import RESPONSE_ARTIFACT_NAME, _stream_item_to_text, create_agent_executor
from hayhooks.server.a2a.messages import build_openai_messages
from hayhooks.server.logger import log
from hayhooks.server.pipelines import registry
from hayhooks.server.tracing import SPAN_A2A_RUN_AGENT
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper
from hayhooks.server.utils.module_loader import _set_method_implementation_flags

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


async def execute_agent_task(pipeline_name, context, event_queue):
    wrapper = registry.get(pipeline_name)
    assert wrapper is not None
    await create_agent_executor(wrapper, pipeline_name).execute(context, event_queue)


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


@pytest.mark.parametrize(
    ("name", "wrapper", "metadata", "expected"),
    [
        ("non_existent_pipeline", None, None, False),
        ("chat_agent", AsyncChatWrapper, None, True),
        ("sync_agent", SyncChatWrapper, None, True),
        ("api_only", ApiOnlyWrapper, None, False),
        ("skipped_agent", AsyncChatWrapper, {"skip_a2a": True}, False),
    ],
)
def test_is_a2a_exposable(name, wrapper, metadata, expected):
    if wrapper is not None:
        register_wrapper(name, wrapper, metadata=metadata)
    assert is_a2a_exposable(name) is expected


# --- Base URL ---


@pytest.mark.parametrize(
    ("external_url", "expected"),
    [("", None), ("https://agents.example.com/", "https://agents.example.com")],
)
def test_get_a2a_base_url(test_settings, external_url, expected):
    test_settings.a2a_external_url = external_url
    expected = expected or f"http://{test_settings.a2a_host}:{test_settings.a2a_port}"
    assert get_a2a_base_url() == expected


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
    assert build_openai_messages(context) == [{"role": "user", "content": "what is the weather?"}]


def test_build_openai_messages_with_task_history():
    from a2a.helpers import new_task_from_user_message, new_text_message
    from a2a.types import Role

    first_message = new_text_message("first question", role=Role.ROLE_USER)
    task = new_task_from_user_message(first_message)
    task.history.append(new_text_message("first answer", role=Role.ROLE_AGENT))

    context = make_context("second question", current_task=task)
    assert build_openai_messages(context) == [
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
    assert build_openai_messages(context) == [{"role": "user", "content": "hello"}]


def test_build_openai_messages_keeps_new_message_matching_history_text():
    from a2a.helpers import new_task_from_user_message, new_text_message
    from a2a.types import Role

    # The agent's last reply is "yes" and the user replies with the identical text:
    # dedup is by message id, so the new user turn must NOT be dropped
    task = new_task_from_user_message(new_text_message("continue?", role=Role.ROLE_USER))
    task.history.append(new_text_message("yes", role=Role.ROLE_AGENT))

    context = make_context("yes", current_task=task)
    assert build_openai_messages(context) == [
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

    await execute_agent_task("sync_agent", make_context(), queue)

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

    await execute_agent_task("chat_agent", make_context("hi"), queue)

    assert get_status_states(queue.events)[-1] == TaskState.TASK_STATE_COMPLETED

    artifact_events = get_artifact_events(queue.events)
    # PipelineEvent items are skipped, text chunks are streamed incrementally
    assert len(artifact_events) == 3
    texts = [event.artifact.parts[0].text for event in artifact_events]
    assert texts == ["Hello, ", "world", " (question: hi)"]
    # All chunks belong to the same artifact; terminal task status closes iterator output.
    assert len({event.artifact.artifact_id for event in artifact_events}) == 1
    assert [event.last_chunk for event in artifact_events] == [False, False, False]
    assert artifact_events[0].append is False
    assert artifact_events[1].append is True


@pytest.mark.asyncio
async def test_execute_agent_task_error_sets_failed_state():
    from a2a.types import TaskState

    register_wrapper("failing_agent", FailingChatWrapper)
    queue = RecordingQueue()

    await execute_agent_task("failing_agent", make_context(), queue)

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

    await execute_agent_task("none_agent", make_context(), queue)

    assert get_status_states(queue.events)[-1] == TaskState.TASK_STATE_FAILED


@pytest.mark.asyncio
async def test_execute_agent_task_emits_trace_and_safe_lifecycle_logs(recording_tracer):
    register_wrapper("sync_agent", SyncChatWrapper)
    records = []
    sink = log.add(lambda message: records.append(message.record), level="DEBUG")
    try:
        await execute_agent_task("sync_agent", make_context("private message"), RecordingQueue())
    finally:
        log.remove(sink)

    spans = [span for span in recording_tracer.spans if span.operation_name == SPAN_A2A_RUN_AGENT]
    assert spans
    assert spans[-1].tags["hayhooks.pipeline.name"] == "sync_agent"
    assert spans[-1].tags["hayhooks.transport"] == "a2a"
    lifecycle = [record for record in records if record["message"] in {"Started A2A task", "Completed A2A task"}]
    assert [record["message"] for record in lifecycle] == ["Started A2A task", "Completed A2A task"]
    assert all(record["extra"]["pipeline_name"] == "sync_agent" for record in lifecycle)
    assert all(record["extra"]["task_id"] for record in lifecycle)
    assert "private message" not in str(lifecycle)


def test_runtime_passes_agent_name_to_task_store_provider():
    from a2a.server.tasks import InMemoryTaskStore

    from hayhooks.a2a import TaskStoreProvider
    from hayhooks.server.a2a.runtime import A2ARuntime

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
    from hayhooks.server.a2a.runtime import A2ARuntime

    class InvalidTaskStoreProvider(TaskStoreProvider):
        def create_task_store(self, _agent_name):
            return object()

    runtime = A2ARuntime(task_store_provider=InvalidTaskStoreProvider())

    with pytest.raises(TypeError, match=r"InvalidTaskStoreProvider.*invalid_agent"):
        runtime.create_task_store("invalid_agent")


async def test_runtime_closes_task_store_provider():
    from a2a.server.tasks import InMemoryTaskStore

    from hayhooks.a2a import TaskStoreProvider
    from hayhooks.server.a2a.runtime import A2ARuntime

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


@pytest.mark.parametrize(
    ("health", "expected_error"),
    [
        ({"healthy": False, "provider": "TestProvider", "error": "ConnectionError"}, "ConnectionError"),
        (True, "InvalidHealthPayload"),
    ],
)
def test_a2a_status_returns_503_for_unhealthy_task_store(health, expected_error):
    from a2a.server.tasks import InMemoryTaskStore
    from starlette.testclient import TestClient

    from hayhooks.a2a import TaskStoreProvider
    from hayhooks.server.a2a.app import create_a2a_app
    from hayhooks.server.a2a.runtime import A2ARuntime

    class TestProvider(TaskStoreProvider):
        def create_task_store(self, _agent_name):
            return InMemoryTaskStore()

        async def health(self):
            return health

    register_wrapper("chat_agent", AsyncChatWrapper)
    app = create_a2a_app(
        base_url="http://test:1418",
        runtime=A2ARuntime(task_store_provider=TestProvider()),
    )

    with TestClient(app) as client:
        response = client.get("/status")

    assert response.status_code == 503
    assert response.json()["status"] == "unavailable"
    assert response.json()["components"]["task_store"]["error"] == expected_error
