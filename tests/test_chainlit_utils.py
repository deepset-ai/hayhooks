import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from hayhooks.server.chainlit_app import utils


@pytest.mark.parametrize(
    "models,default_model,expected",
    [
        ([{"id": "model-a"}, {"id": "model-b"}], "model-b", "model-b"),
        ([{"id": "model-a"}, {"id": "model-b"}], "model-x", None),
        ([{"id": "only-model"}], "", "only-model"),
        ([{"id": "model-a"}, {"id": "model-b"}], "", None),
        ([], "", None),
    ],
    ids=[
        "returns_default_when_available",
        "returns_none_when_default_not_available",
        "returns_single_model",
        "returns_none_for_multiple_models_no_default",
        "returns_none_for_empty_list",
    ],
)
def test_select_model_automatically(models, default_model, expected):
    assert utils.select_model_automatically(models, default_model) == expected


@pytest.mark.parametrize(
    "description,expected_name,expected_type",
    [
        ("Calling 'weather_tool' tool...", "🔧 weather_tool", "tool"),
        ("Calling 'my_custom_tool' tool", "🔧 my_custom_tool", "tool"),
        ("Processing data...", "Processing", "run"),
        ("weather_tool' tool", "Processing", "run"),
        ("Calling 'weather_tool'", "Processing", "run"),
        ("", "Processing", "run"),
    ],
    ids=[
        "extracts_tool_name",
        "extracts_tool_with_underscores",
        "non_tool_description",
        "missing_prefix",
        "missing_suffix",
        "empty_string",
    ],
)
def test_extract_tool_name(description, expected_name, expected_type):
    step_name, step_type = utils.extract_tool_name(description)
    assert step_name == expected_name
    assert step_type == expected_type


def test_format_tool_result_with_dict_arguments():
    arguments = {"location": "Paris", "units": "celsius"}
    result = "The weather is sunny"
    formatted = utils.format_tool_result(arguments, result)

    assert "**Arguments:**" in formatted
    assert "```json" in formatted
    assert '"location": "Paris"' in formatted
    assert "**Result:**" in formatted
    assert "The weather is sunny" in formatted


def test_format_tool_result_with_non_dict_arguments():
    formatted = utils.format_tool_result("simple string", "Result text")
    assert "simple string" in formatted
    assert "Result text" in formatted


def test_build_chat_request():
    history = [{"role": "user", "content": "Hello"}]
    request = utils.build_chat_request("test-model", history)

    assert request == {"model": "test-model", "messages": history, "stream": True}


@pytest.mark.parametrize(
    "line,expected",
    [
        ("event: message", None),
        ("data: [DONE]", None),
        ("data: not json", None),
        ("data: {}", None),
        (f'data: {json.dumps({"choices": []})}', None),
        (f'data: {json.dumps({"choices": [{"delta": {}}]})}', None),
        (f'data: {json.dumps({"choices": [{"delta": {"content": ""}}]})}', None),
        (f'data: {json.dumps({"choices": [{"delta": {"content": "Hello"}}]})}', "Hello"),
    ],
    ids=[
        "non_data_line",
        "done_signal",
        "invalid_json",
        "empty_object",
        "empty_choices",
        "missing_content",
        "empty_content",
        "valid_content",
    ],
)
@pytest.mark.asyncio
async def test_process_sse_chunk(line, expected):
    assert await utils.process_sse_chunk(line) == expected


@pytest.mark.asyncio
async def test_process_sse_chunk_calls_event_handler():
    handler = AsyncMock()
    event_data = {"type": "status", "data": {"description": "Processing"}}
    chunk = {"event": event_data}

    result = await utils.process_sse_chunk(f"data: {json.dumps(chunk)}", event_handler=handler)

    assert result is None
    handler.assert_called_once_with(event_data)


@pytest.mark.parametrize(
    "event",
    [
        {"data": {"description": "Processing"}},  # missing type
        {"type": "status"},  # missing data
    ],
    ids=["missing_type", "missing_data"],
)
@pytest.mark.asyncio
async def test_process_sse_chunk_skips_invalid_events(event):
    handler = AsyncMock()
    chunk = {"event": event}

    await utils.process_sse_chunk(f"data: {json.dumps(chunk)}", event_handler=handler)

    handler.assert_not_called()


@pytest.mark.parametrize(
    "status_code,expected",
    [(200, True), (500, False)],
    ids=["healthy", "unhealthy"],
)
@pytest.mark.asyncio
async def test_check_backend_health_status_codes(status_code, expected):
    mock_response = MagicMock()
    mock_response.status_code = status_code

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
        assert await utils.check_backend_health() == expected


@pytest.mark.parametrize(
    "exception",
    [httpx.ConnectError("Connection refused"), httpx.TimeoutException("Timeout")],
    ids=["connection_error", "timeout"],
)
@pytest.mark.asyncio
async def test_check_backend_health_returns_false_on_error(exception):
    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(side_effect=exception)
        assert await utils.check_backend_health() is False


@pytest.mark.asyncio
async def test_get_available_models_returns_models():
    mock_response = MagicMock()
    mock_response.json.return_value = {"data": [{"id": "model-a"}, {"id": "model-b"}]}
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
        result = await utils.get_available_models()

    assert result == [{"id": "model-a"}, {"id": "model-b"}]


@pytest.mark.parametrize(
    "scenario",
    ["http_error", "no_data_key"],
    ids=["http_error", "no_data_key"],
)
@pytest.mark.asyncio
async def test_get_available_models_returns_empty_list(scenario):
    with patch("httpx.AsyncClient") as mock_client:
        if scenario == "http_error":
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=httpx.HTTPStatusError("Error", request=MagicMock(), response=MagicMock())
            )
        else:
            mock_response = MagicMock()
            mock_response.json.return_value = {}
            mock_response.raise_for_status = MagicMock()
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)

        assert await utils.get_available_models() == []
