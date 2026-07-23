from dataclasses import dataclass
from typing import Any

import haystack
import pytest
from haystack import Document, Pipeline, component
from haystack.components.converters.txt import TextFileToDocument
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.joiners import BranchJoiner
from haystack.dataclasses import (
    ByteStream,
    ChatMessage,
    ChatRole,
    ExtractedAnswer,
    FileContent,
    GeneratedAnswer,
    ImageContent,
    ReasoningContent,
    SparseEmbedding,
    TextContent,
    ToolCall,
    ToolCallResult,
)
from haystack.utils import Secret
from pydantic import BaseModel

from hayhooks.server.pipelines.utils import coerce_pipeline_inputs

# The round-trip cases assert how Pydantic serializes Haystack dataclasses, which targets Haystack v3
# serialization. hayhooks also runs its suite against Haystack v2, so skip there.
HAYSTACK_MAJOR_VERSION = int(haystack.__version__.split(".")[0])

pytestmark = pytest.mark.skipif(
    HAYSTACK_MAJOR_VERSION < 3, reason="coerce_pipeline_inputs coercion tests target Haystack v3 serialization"
)


@component
class TypedEcho:
    """Echoes back a single input whose socket type is set at construction time."""

    def __init__(self, type_: Any) -> None:
        component.set_input_types(self, value=type_)
        component.set_output_types(self, value=Any)

    def run(self, **kwargs: Any) -> dict[str, Any]:
        return {"value": kwargs.get("value")}


@component
class MultiInputEcho:
    """A component with one coercible socket alongside plain, non-coercible sockets."""

    @component.output_types(messages=list[ChatMessage])
    def run(self, messages: list[ChatMessage], top_k: int = 3, config: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"messages": messages}


@dataclass
class PlainDataclass:
    name: str
    count: int


class PydanticModel(BaseModel):
    a: int
    b: str


class Conversation(BaseModel):
    messages: list[ChatMessage]
    documents: list[Document]


class SerializationEnvelope(BaseModel):
    value: Any


def serialize_for_client(instance: Any) -> Any:
    """
    Serialize an instance the way a FastAPI app does: instances arrive inside a loosely-typed Pydantic field, so
    Pydantic (not `to_dict`) produces the JSON payload.
    """
    return SerializationEnvelope(value=instance).model_dump(mode="json")["value"]


TOOL_CALL = ToolCall(tool_name="t", arguments={"a": 1}, id="1")

# Round-tripping these through Pydantic relies on a from_dict fix that is in Haystack main but not yet released.
# Non-strict so the cases pass (xpass) automatically once the fix ships (expected in Haystack v3.1).
NEEDS_HAYSTACK_MAIN = pytest.mark.xfail(
    reason="Requires an unreleased Haystack from_dict fix for ChatMessage/GeneratedAnswer (expected in v3.1)",
    strict=False,
)

COERCIBLE_INSTANCES = [
    # A Document with all non-bytes fields and a ChatMessage with every content-part type, to exercise the
    # serialization of each field / content type.
    pytest.param(
        Document(
            content="Hello",
            meta={"k": "v"},
            score=0.5,
            embedding=[0.1, 0.2],
            sparse_embedding=SparseEmbedding(indices=[0, 2], values=[0.1, 0.2]),
        ),
        id="Document",
    ),
    pytest.param(
        ChatMessage(
            _role=ChatRole.ASSISTANT,
            _content=[
                TextContent(text="hi"),
                TOOL_CALL,
                ToolCallResult(result="r", origin=TOOL_CALL, error=False),
                ImageContent(base64_image="aGVsbG8=", mime_type="image/png"),
                ReasoningContent(reasoning_text="because"),
                FileContent(base64_data="aGVsbG8=", mime_type="application/pdf"),
            ],
            _name="n",
            _meta={"k": "v"},
        ),
        id="ChatMessage",
        marks=NEEDS_HAYSTACK_MAIN,
    ),
    pytest.param(
        GeneratedAnswer(data="answer", query="q", documents=[Document(content="d")]),
        id="GeneratedAnswer",
        marks=NEEDS_HAYSTACK_MAIN,
    ),
    pytest.param(ImageContent(base64_image="aGVsbG8=", mime_type="image/png"), id="ImageContent"),
    pytest.param(
        ByteStream(data=b"hello", mime_type="text/plain", meta={"k": "v"}),
        id="ByteStream",
        marks=pytest.mark.xfail(
            reason="Pydantic serializes the bytes field as a JSON string that ByteStream.from_dict cannot rebuild",
            raises=TypeError,
            strict=True,
        ),
    ),
    pytest.param(PydanticModel(a=1, b="x"), id="pydantic-model"),
    pytest.param(PlainDataclass(name="x", count=2), id="stdlib-dataclass"),
    pytest.param(
        Conversation(messages=[ChatMessage.from_user("Hi")], documents=[Document(content="D")]),
        id="pydantic-model-nesting-haystack",
    ),
]


class TestCoercePipelineInputs:
    @pytest.fixture
    def pipeline(self):
        pipe = Pipeline()
        pipe.add_component("echo", MultiInputEcho())
        return pipe

    @pytest.mark.parametrize("instance", COERCIBLE_INSTANCES)
    def test_single_socket(self, instance):
        pipe = Pipeline()
        pipe.add_component("echo", TypedEcho(type(instance)))
        coerced = coerce_pipeline_inputs(pipe, {"echo": {"value": serialize_for_client(instance)}})
        assert coerced["echo"]["value"] == instance

    @pytest.mark.parametrize("instance", COERCIBLE_INSTANCES)
    def test_list_socket(self, instance):
        pipe = Pipeline()
        pipe.add_component("echo", TypedEcho(list[type(instance)]))  # type: ignore[misc]
        coerced = coerce_pipeline_inputs(pipe, {"echo": {"value": [serialize_for_client(instance)]}})
        assert coerced["echo"]["value"] == [instance]

    @pytest.mark.parametrize("instance", COERCIBLE_INSTANCES)
    def test_optional_socket(self, instance):
        pipe = Pipeline()
        pipe.add_component("echo", TypedEcho(type(instance) | None))
        coerced = coerce_pipeline_inputs(pipe, {"echo": {"value": serialize_for_client(instance)}})
        assert coerced["echo"]["value"] == instance

    def test_nested_format(self, pipeline):
        messages = [ChatMessage.from_user("Hi"), ChatMessage.from_assistant("Hello")]
        data: dict[str, Any] = {
            "echo": {"messages": [serialize_for_client(message) for message in messages], "top_k": 5}
        }
        coerced = coerce_pipeline_inputs(pipeline, data)
        assert coerced == {"echo": {"messages": messages, "top_k": 5}}
        # the input data is not mutated
        assert isinstance(data["echo"]["messages"][0], dict)

    def test_flat_format(self, pipeline):
        messages = [ChatMessage.from_user("Hi")]
        data = {"messages": [serialize_for_client(message) for message in messages], "top_k": 5}
        coerced = coerce_pipeline_inputs(pipeline, data)
        assert coerced == {"messages": messages, "top_k": 5}

    def test_instances_pass_through(self, pipeline):
        messages = [ChatMessage.from_user("Hi")]
        data = {"echo": {"messages": messages}}
        coerced = coerce_pipeline_inputs(pipeline, data)
        assert coerced["echo"]["messages"][0] is messages[0]

    def test_mixed_list(self, pipeline):
        message = ChatMessage.from_user("Hi")
        other = ChatMessage.from_assistant("Hello")
        data = {"echo": {"messages": [message, serialize_for_client(other)]}}
        coerced = coerce_pipeline_inputs(pipeline, data)
        assert coerced["echo"]["messages"] == [message, other]

    def test_non_coercible_values_untouched(self, pipeline):
        data = {"echo": {"messages": [], "top_k": 5, "config": {"a": 1}}}
        coerced = coerce_pipeline_inputs(pipeline, data)
        assert coerced == data

    def test_unknown_inputs_untouched(self, pipeline):
        message = serialize_for_client(ChatMessage.from_user("Hi"))
        data = {"unknown_input": [message], "top_k": 5}
        coerced = coerce_pipeline_inputs(pipeline, data)
        assert coerced == {"unknown_input": [message], "top_k": 5}

    def test_unknown_component_untouched(self, pipeline):
        message = serialize_for_client(ChatMessage.from_user("Hi"))
        data = {"unknown_component": {"messages": [message]}}
        coerced = coerce_pipeline_inputs(pipeline, data)
        assert coerced == {"unknown_component": {"messages": [message]}}

    def test_variadic_socket(self):
        pipe = Pipeline()
        pipe.add_component("joiner", BranchJoiner(list[ChatMessage]))
        messages = [ChatMessage.from_user("Hi")]
        data = {"value": [serialize_for_client(message) for message in messages], "unrelated": 1}
        coerced = coerce_pipeline_inputs(pipe, data)
        assert coerced["value"] == messages

    def test_chat_generator_messages_socket(self):
        # OpenAIChatGenerator's `messages` socket is `list[ChatMessage] | str`: the list of serialized messages
        # is coerced, since ChatMessage is the only coercible member of the union.
        pipe = Pipeline()
        pipe.add_component("generator", OpenAIChatGenerator(api_key=Secret.from_token("test-key")))
        messages = [ChatMessage.from_user("Hi"), ChatMessage.from_system("Sys")]
        data = {"generator": {"messages": [serialize_for_client(message) for message in messages]}}
        coerced = coerce_pipeline_inputs(pipe, data)
        assert coerced == {"generator": {"messages": messages}}

    def test_chat_generator_messages_socket_string_passes_through(self):
        # The `str` arm of `list[ChatMessage] | str` is not coercible, so a bare string is left untouched.
        pipe = Pipeline()
        pipe.add_component("generator", OpenAIChatGenerator(api_key=Secret.from_token("test-key")))
        data = {"generator": {"messages": "just a string"}}
        coerced = coerce_pipeline_inputs(pipe, data)
        assert coerced == {"generator": {"messages": "just a string"}}

    def test_converter_sources_socket(self):
        # A converter's `sources` socket is `list[str | Path | ByteStream]`: ByteStream is the only coercible
        # member, so a serialized ByteStream is coerced while plain string paths pass through. ByteStream is
        # serialized with `to_dict` since it has no lossless Pydantic JSON form.
        pipe = Pipeline()
        pipe.add_component("converter", TextFileToDocument())
        byte_stream = ByteStream(data=b"hello", mime_type="text/plain")
        data = {"converter": {"sources": ["/path/to/file.txt", byte_stream.to_dict()]}}
        coerced = coerce_pipeline_inputs(pipe, data)
        assert coerced["converter"]["sources"] == ["/path/to/file.txt", byte_stream]

    def test_ambiguous_union_socket_raises(self):
        # A socket that involves more than one coercible class cannot be disambiguated from the payload.
        pipe = Pipeline()
        pipe.add_component("echo", TypedEcho(GeneratedAnswer | ExtractedAnswer))
        answer = GeneratedAnswer(data="a", query="q", documents=[])
        data = {"echo": {"value": serialize_for_client(answer)}}
        with pytest.raises(ValueError, match="multiple deserializable members"):
            coerce_pipeline_inputs(pipe, data)

    def test_ambiguous_union_socket_passes_through_instances(self):
        # An already-deserialized value needs no coercion, so an ambiguous socket does not raise for it.
        pipe = Pipeline()
        pipe.add_component("echo", TypedEcho(GeneratedAnswer | ExtractedAnswer))
        answer = GeneratedAnswer(data="a", query="q", documents=[])
        coerced = coerce_pipeline_inputs(pipe, {"echo": {"value": answer}})
        assert coerced["echo"]["value"] is answer
