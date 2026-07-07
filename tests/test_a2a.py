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
    assert len(artifact_events) == 3
    texts = [event.artifact.parts[0].text for event in artifact_events]
    assert texts == ["Hello, ", "world", " (question: hi)"]
    # All chunks belong to the same artifact; only the last one is marked last_chunk
    assert len({event.artifact.artifact_id for event in artifact_events}) == 1
    assert [event.last_chunk for event in artifact_events] == [False, False, True]
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
