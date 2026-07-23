import json
from dataclasses import is_dataclass
from types import UnionType
from typing import TYPE_CHECKING, Any, Union, get_args, get_origin
from uuid import uuid4

from fastapi_openai_compat import Message
from haystack.dataclasses import ChatMessage, ToolCall
from pydantic import BaseModel, TypeAdapter

from hayhooks.server.pipelines.streaming import (
    OnPipelineEnd,
    OnReasoning,
    OnToolCallEnd,
    OnToolCallStart,
    ToolCallbackReturn,
    async_streaming_generator,
    find_all_streaming_components,
    is_streaming_component,
    parse_streaming_components_setting,
    streaming_generator,
)

if TYPE_CHECKING:
    from haystack.core.pipeline.base import PipelineBase

__all__ = [
    "OnPipelineEnd",
    "OnReasoning",
    "OnToolCallEnd",
    "OnToolCallStart",
    "ToolCallbackReturn",
    "async_streaming_generator",
    "chat_messages_from_openai_response",
    "coerce_pipeline_inputs",
    "find_all_streaming_components",
    "get_content",
    "get_input_files",
    "get_last_user_input_text",
    "get_last_user_message",
    "is_streaming_component",
    "is_user_message",
    "parse_streaming_components_setting",
    "streaming_generator",
]


def is_user_message(msg: Message | dict) -> bool:
    if isinstance(msg, Message):
        return msg.role == "user"
    return msg.get("role") == "user"


def get_content(msg: Message | dict) -> str:
    if isinstance(msg, Message):
        return msg.content or ""
    return msg.get("content", "")


def get_last_user_message(messages: list[Message | dict]) -> str | None:
    user_messages = (msg for msg in reversed(messages) if is_user_message(msg))

    for message in user_messages:
        return get_content(message)

    return None


def get_input_files(input_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Extract all ``input_file`` content parts from OpenAI Responses API input items.

    Returns a list of dicts, each containing at least ``"file_id"`` (and any
    other keys present on the content part, e.g. ``"filename"``).  Items that
    are not user messages or content parts that are not ``input_file`` are
    silently skipped.
    """
    files: list[dict[str, Any]] = []
    for item in input_items:
        if not isinstance(item, dict) or item.get("role") != "user":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        files.extend(
            part
            for part in content
            if isinstance(part, dict) and part.get("type") == "input_file" and part.get("file_id")
        )
    return files


def get_last_user_input_text(input_items: list[dict[str, Any]]) -> str | None:
    """
    Extract the last user text from OpenAI Responses API input items.

    Input items use a different structure than chat messages:
    ``[{"role": "user", "type": "message", "content": [{"type": "input_text", "text": "..."}]}]``

    String shorthand (``"content": "Hello"``) is also supported.
    """
    for item in reversed(input_items):
        if not isinstance(item, dict) or item.get("role") != "user":
            continue
        if item.get("type") not in ("message", None):
            continue

        content = item.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for content_part in reversed(content):
                if isinstance(content_part, dict) and content_part.get("type") == "input_text":
                    text = content_part.get("text")
                    return text if isinstance(text, str) else None
    return None


def _content_to_text(content: object) -> str:
    """Flatten a Responses API content field into plain text."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if isinstance(text, str) and text:
            parts.append(text)
    return "\n".join(parts)


def _parse_tool_arguments(raw: object) -> dict:
    """Parse tool-call arguments that may arrive as a dict or a JSON string."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _extract_call_id(item: dict[str, Any], *, generate_if_missing: bool) -> str | None:
    """Resolve ``call_id`` or ``id`` from a Responses item, optionally generating one."""
    for key in ("call_id", "id"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    if generate_if_missing:
        return f"call_{uuid4().hex[:24]}"
    return None


def chat_messages_from_openai_response(input_items: list[dict[str, Any]]) -> list[ChatMessage]:
    """
    Convert OpenAI Responses API input items to Haystack ``ChatMessage`` objects.

    Accepts both ``InputItem`` dicts (from ``fastapi-openai-compat``) and plain
    ``dict[str, Any]`` — they are the same underlying type.

    Handles three item types:

    - **message** (user / system / developer / assistant) → ``ChatMessage``
    - **function_call** → ``ChatMessage.from_assistant`` with a ``ToolCall``
    - **function_call_output** → ``ChatMessage.from_tool`` with matching origin

    This is the Responses API counterpart of building a ``messages`` list from
    the Chat Completions format.  It supports the full Codex multi-turn loop
    where the client sends back ``function_call`` / ``function_call_output``
    items alongside regular messages.
    """
    messages: list[ChatMessage] = []
    tool_calls_by_id: dict[str, ToolCall] = {}

    for item in input_items:
        if not isinstance(item, dict):
            continue

        item_type = item.get("type")

        if item_type == "function_call":
            arguments = _parse_tool_arguments(item.get("arguments", "{}"))
            call_id = _extract_call_id(item, generate_if_missing=True)
            tool_name = item.get("name") or "function"

            tool_call = ToolCall(tool_name=tool_name, arguments=arguments, id=call_id)
            tool_calls_by_id[call_id] = tool_call  # ty: ignore[invalid-assignment]
            messages.append(ChatMessage.from_assistant(tool_calls=[tool_call]))
            continue

        if item_type == "function_call_output":
            call_id = _extract_call_id(item, generate_if_missing=False)
            raw_output = item.get("output", "")
            output = raw_output if isinstance(raw_output, str) else json.dumps(raw_output)

            if isinstance(call_id, str) and call_id in tool_calls_by_id:
                origin = tool_calls_by_id[call_id]
            else:
                origin = ToolCall(tool_name="function", arguments={}, id=str(call_id) if call_id else None)

            messages.append(ChatMessage.from_tool(tool_result=output, origin=origin))
            continue

        role = item.get("role")
        text = _content_to_text(item.get("content"))
        if not text:
            continue
        if role == "user":
            messages.append(ChatMessage.from_user(text))
        elif role in ("system", "developer"):
            messages.append(ChatMessage.from_system(text))
        elif role == "assistant":
            messages.append(ChatMessage.from_assistant(text))

    return messages


def _is_union(type_: Any) -> bool:
    """Whether `type_` is a typing.Union or a PEP 604 `X | Y` union."""
    return get_origin(type_) in (Union, UnionType)


def _coercible_classes(type_: Any) -> set[type]:
    """
    Collect the distinct classes involved in `type_` that can deserialize a plain dictionary.

    A class is coercible if it exposes a `from_dict` method (Haystack dataclasses and Components), is a Pydantic
    model, or is a standard-library dataclass. Follows `list[T]` and optionals/unions of those.

    :param type_: The type to inspect.
    :returns: The set of coercible classes `type_` involves (empty if none).
    """
    if _is_union(type_):
        classes: set[type] = set()
        for arm in get_args(type_):
            classes |= _coercible_classes(arm)
        return classes
    if get_origin(type_) is list:
        args = get_args(type_)
        return _coercible_classes(args[0]) if args else set()
    if isinstance(type_, type) and (hasattr(type_, "from_dict") or issubclass(type_, BaseModel) or is_dataclass(type_)):
        return {type_}
    return set()


def _deserialize_one(value: dict[str, Any], target_class: Any) -> Any:
    """
    Deserialize a single dictionary into an instance of `target_class`.

    `from_dict`-capable classes take priority so Haystack objects keep their native deserialization. Pydantic
    models and standard-library dataclasses are deserialized with Pydantic.
    """
    if hasattr(target_class, "from_dict"):
        return target_class.from_dict(value)
    if issubclass(target_class, BaseModel):
        return target_class.model_validate(value)
    return TypeAdapter(target_class).validate_python(value)


def _deserialize_from_dict(value: Any, target_class: Any) -> Any:
    """
    Deserialize `value` into instances of `target_class`.

    Dictionaries are deserialized directly, lists element-wise (leaving non-dictionary items untouched). Any other
    value is returned unchanged.
    """
    if isinstance(value, dict):
        return _deserialize_one(value, target_class)
    if isinstance(value, list):
        return [_deserialize_one(item, target_class) if isinstance(item, dict) else item for item in value]
    return value


def _needs_coercion(value: Any) -> bool:
    """Whether `value` carries dictionaries that could be deserialized (a dict, or a list containing one)."""
    return isinstance(value, dict) or (isinstance(value, list) and any(isinstance(item, dict) for item in value))


def _coerce_input_value(value: Any, socket_types: list[Any]) -> Any:
    if not _needs_coercion(value):
        return value
    for type_ in socket_types:
        classes = _coercible_classes(type_)
        if len(classes) > 1:
            names = ", ".join(sorted(cls.__name__ for cls in classes))
            msg = (
                f"Cannot coerce input for socket type '{type_}': it has multiple deserializable members "
                f"({names}) and the serialized payload carries no type information to choose between them. "
                f"Provide this input as an already-deserialized object."
            )
            raise ValueError(msg)
        if classes:
            return _deserialize_from_dict(value, next(iter(classes)))
    return value


def coerce_pipeline_inputs(pipeline: "PipelineBase", data: dict[str, Any]) -> dict[str, Any]:
    """
    Deserialize serialized Haystack objects in pipeline input data, based on the pipeline's input socket types.

    Web frameworks such as FastAPI hand pipeline inputs over as plain JSON (often typed loosely as
    `dict[str, Any]`), so serialized Haystack objects arrive as plain dictionaries rather than instances. This
    utility recovers them: for every provided value whose input socket type involves a coercible class, plain
    dictionaries are converted into instances of that class. A class is coercible if it exposes a `from_dict`
    method (such as `ChatMessage` or `Document`), is a Pydantic model, or is a standard-library dataclass.
    Socket types of the form `T`, `list[T]`, and optionals/unions of those are supported. Values that are already
    deserialized, or whose socket types involve no coercible class, are returned unchanged. A socket type that
    involves more than one distinct coercible class (such as `GeneratedAnswer | ExtractedAnswer`) is ambiguous,
    since the payload carries no type information to select one; coercing a dictionary against it raises a
    `ValueError`.

    Like `Pipeline.run`, `data` accepts the nested format (`{"component_name": {"input_name": value}}`) and the
    flat format (`{"input_name": value}`). The same format detection rules apply and the returned dictionary keeps
    the format of `data`.

    Usage example:
    ```python
    from hayhooks import coerce_pipeline_inputs

    inputs = coerce_pipeline_inputs(pipeline, serialized_inputs)
    result = pipeline.run(inputs)
    ```

    :param pipeline: The pipeline whose input socket types drive the coercion.
    :param data: The pipeline input data, in nested or flat format.
    :returns: A new dictionary with the same structure as `data` and deserialized values.
    :raises ValueError: If a socket type involves more than one distinct coercible class and the corresponding
        value is a serialized dictionary.
    """
    # mirrors the format detection in Pipeline.run: nested if all values are dictionaries
    if all(isinstance(value, dict) for value in data.values()):
        available_inputs = pipeline.inputs(include_components_with_connected_inputs=True)
        coerced: dict[str, Any] = {}
        for component_name, component_inputs in data.items():
            sockets = available_inputs.get(component_name, {})
            coerced[component_name] = {
                input_name: _coerce_input_value(value, [sockets[input_name]["type"]] if input_name in sockets else [])
                for input_name, value in component_inputs.items()
            }
        return coerced

    # flat format: input names are matched across components like in Pipeline.run
    available_inputs = pipeline.inputs()
    return {
        input_name: _coerce_input_value(
            value, [sockets[input_name]["type"] for sockets in available_inputs.values() if input_name in sockets]
        )
        for input_name, value in data.items()
    }
