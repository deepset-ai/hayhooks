import asyncio
import importlib.util
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from hayhooks.server.pipelines.registry import registry
from hayhooks.server.routers.deploy import router as deploy_router
from hayhooks.server.routers.openai import (
    _CHAT_COMPLETION_DISPATCH,
    _RESPONSE_DISPATCH,
    _run_pipeline_method,
    create_openai_router,
)
from hayhooks.server.utils.a2a_utils import _run_chat_completion
from hayhooks.server.utils.mcp_utils import list_pipelines_as_tools, run_pipeline_as_tool
from hayhooks.settings import settings

METHODS = ["run_api", "run_chat_completion", "run_response"]
METHODS += [method + "_async" for method in METHODS]
BASE_SOURCE = """\
import asyncio
import time
from collections.abc import Generator, AsyncGenerator
from fastapi import UploadFile
from starlette.datastructures import Headers
from hayhooks import BasePipelineWrapper

class PipelineWrapper(BasePipelineWrapper):
    def setup(self):
        pass
"""


@pytest.fixture
def headers_client():
    registry.clear()
    app = FastAPI()
    app.include_router(deploy_router)
    app.include_router(create_openai_router(registry))
    with TestClient(app) as client:
        yield client
    registry.clear()


def deploy(client, source):
    response = client.post(
        "/deploy_files",
        json={
            "name": "headers_test",
            "files": {"pipeline_wrapper.py": source},
            "save_files": False,
        },
    )
    assert response.status_code == 200, response.text


def wrapper_source(method, stream=False, files=False):
    is_async = method.endswith("_async")
    arguments = (
        "query: str"
        if method.startswith("run_api")
        else (
            "model: str, messages: list[dict], body: dict"
            if "chat" in method
            else "model: str, input_items: list[dict], body: dict"
        )
    )
    if files:
        arguments += ", files: list[UploadFile]"
    prefix = "async " if is_async else ""
    delay = "await asyncio.sleep(0.01)" if is_async else "time.sleep(0.01)"
    return_type = ("AsyncGenerator" if is_async else "Generator") if stream else "str"
    header_value = "headers.get('Authorization', 'missing') if headers is not None else 'no-context'"
    result = (
        f"        {prefix}def chunks():\n            {delay}\n"
        f"            yield {header_value}\n        return chunks()\n"
        if stream
        else f"        return {header_value}\n"
    )
    return BASE_SOURCE + (
        f"    {prefix}def {method}(self, {arguments}, *, headers: Headers | None = None) -> {return_type}:\n"
        f"        {delay}\n" + result
    )


def request_for(method, stream=False, prefix="/v1"):
    if method.startswith("run_api"):
        return "/headers_test/run", {"query": "hi", "headers": {"authorization": "body-spoof"}}
    body = {"model": "headers_test", "stream": stream, "headers": {"authorization": "body-spoof"}}
    if "chat" in method:
        return prefix + "/chat/completions", {**body, "messages": [{"role": "user", "content": "hi"}]}
    return prefix + "/responses", {**body, "input": "hi"}


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("postponed", [False, True])
async def test_headers_are_request_local_through_execution_and_streaming(headers_client, method, stream, postponed):
    source = wrapper_source(method, stream)
    deploy(headers_client, "from __future__ import annotations\n" + source if postponed else source)
    if method.startswith("run_api"):
        schema = headers_client.get("/openapi.json").json()["components"]["schemas"]["headers_testRunRequest"]
        assert set(schema["properties"]) == {"query"}
    url, body = request_for(method, stream)
    tokens = ["Bearer alice-test", "Bearer bob-test", None]
    async with AsyncClient(transport=ASGITransport(app=headers_client.app), base_url="http://test") as client:
        responses = await asyncio.gather(
            *[
                client.post(url, json=body, headers={"Authorization": token} if token is not None else {})
                for token in tokens
            ]
        )
    for token, response in zip(tokens, responses, strict=True):
        assert response.status_code == 200, response.text
        assert (token or "missing") in response.text
        assert "body-spoof" not in response.text
        assert all((other or "missing") not in response.text for other in tokens if other != token)


@pytest.mark.parametrize("method", [m for m in METHODS if not m.startswith("run_api")])
def test_openai_aliases_forward_headers(headers_client, method):
    deploy(headers_client, wrapper_source(method))
    url, body = request_for(method, prefix="")
    response = headers_client.post(url, json=body, headers={"Authorization": "alias-token"})
    assert response.status_code == 200, response.text
    assert "alias-token" in response.text


@pytest.mark.parametrize("method", ["run_api", "run_api_async"])
@pytest.mark.parametrize("postponed", [False, True])
def test_multipart_headers_are_injected_outside_the_form(headers_client, method, postponed):
    source = wrapper_source(method, files=True)
    deploy(headers_client, "from __future__ import annotations\n" + source if postponed else source)
    response = headers_client.post(
        "/headers_test/run",
        data={"query": "hi", "headers": "body-spoof"},
        files={"files": ("test.txt", b"test", "text/plain")},
        headers={"Authorization": "upload-token"},
    )
    assert response.status_code == 200, response.text
    assert response.json() == {"result": "upload-token"}


@pytest.mark.parametrize("method", ["run_api", "run_api_async"])
@pytest.mark.skipif(importlib.util.find_spec("mcp") is None, reason="MCP is not installed")
@pytest.mark.mcp
async def test_injected_headers_stay_out_of_mcp_schema_and_arguments(headers_client, method):
    deploy(headers_client, wrapper_source(method))
    tools = await list_pipelines_as_tools(registry)
    assert set(tools[0].inputSchema["properties"]) == {"query"}
    result = await run_pipeline_as_tool(registry, "headers_test", {"query": "hi"})
    assert result[0].text == "no-context"
    with pytest.raises(ValueError, match="cannot be supplied as pipeline arguments"):
        await run_pipeline_as_tool(
            registry, "headers_test", {"query": "hi", "headers": {"authorization": "tool-spoof"}}
        )


@pytest.mark.parametrize("method", [m for m in METHODS if not m.startswith("run_api")])
async def test_non_http_openai_calls_use_the_default(headers_client, method):
    deploy(headers_client, wrapper_source(method))
    dispatch, kwargs = (
        (_CHAT_COMPLETION_DISPATCH, {"messages": []}) if "chat" in method else (_RESPONSE_DISPATCH, {"input_items": []})
    )
    result = await _run_pipeline_method(registry, dispatch, model="headers_test", kwargs=kwargs, body={}, headers=None)
    assert result == "no-context"


@pytest.mark.parametrize("method", ["run_chat_completion", "run_chat_completion_async"])
@pytest.mark.skipif(importlib.util.find_spec("a2a") is None, reason="A2A is not installed")
@pytest.mark.a2a
async def test_a2a_calls_use_the_default(headers_client, method):
    deploy(headers_client, wrapper_source(method))
    assert (
        await _run_chat_completion(registry, "headers_test", SimpleNamespace(message=None, current_task=None))
        == "no-context"
    )


@pytest.mark.parametrize(
    "extra,assertion",
    [
        ("", "True"),
        (", **kwargs", "kwargs == {}"),
        (", *headers", "headers == ()"),
        (", headers: dict[str, str] | None = None", "headers is None"),
        (", headers: 'NotImportedAtRuntime' = None", "headers is None"),
        (", headers: dict[str, Headers] = None", "headers is None"),
    ],
)
@pytest.mark.parametrize("method", [m for m in METHODS if not m.startswith("run_api")])
def test_legacy_openai_signatures_do_not_receive_headers(headers_client, extra, assertion, method):
    args = (
        "model: str, messages: list[dict], body: dict"
        if "chat" in method
        else "model: str, input_items: list[dict], body: dict"
    )
    prefix = "async " if method.endswith("_async") else ""
    deploy(
        headers_client,
        BASE_SOURCE
        + (
            f"    {prefix}def {method}(self, {args}{extra}) -> str:\n"
            f"        assert {assertion}\n        return 'legacy'\n"
        ),
    )
    url, body = request_for(method)
    response = headers_client.post(url, json=body, headers={"Authorization": "must-not-inject"})
    assert response.status_code == 200, response.text
    assert "legacy" in response.text


@pytest.mark.parametrize("method", ["run_api", "run_api_async"])
@pytest.mark.parametrize(
    "transport",
    [
        "http",
        pytest.param(
            "mcp",
            marks=[
                pytest.mark.mcp,
                pytest.mark.skipif(importlib.util.find_spec("mcp") is None, reason="MCP is not installed"),
            ],
        ),
    ],
)
async def test_regular_headers_body_field_keeps_its_schema_and_value(headers_client, method, transport):
    prefix = "async " if method.endswith("_async") else ""
    deploy(
        headers_client,
        BASE_SOURCE
        + (
            f"    {prefix}def {method}(self, headers: dict[str, str]) -> str:\n"
            "        return headers['authorization']\n"
        ),
    )
    if transport == "http":
        schema = headers_client.get("/openapi.json").json()["components"]["schemas"]["headers_testRunRequest"]
        assert schema["required"] == ["headers"]
        response = headers_client.post(
            "/headers_test/run",
            json={"headers": {"authorization": "body-value"}},
            headers={"Authorization": "transport-value"},
        )
        assert response.status_code == 200, response.text
        assert response.json() == {"result": "body-value"}
    else:
        tools = await list_pipelines_as_tools(registry)
        assert tools[0].inputSchema["required"] == ["headers"]
        result = await run_pipeline_as_tool(registry, "headers_test", {"headers": {"authorization": "tool-value"}})
        assert result[0].text == "tool-value"


@pytest.mark.parametrize(
    "declaration",
    [
        "headers: Headers",
        "headers: Headers | None",
        "headers: Headers = None",
        "headers: Headers | str | None = None",
        "headers: Headers | None = None, /",
        "*headers: Headers | None",
        "**headers: Headers | None",
    ],
)
def test_invalid_opt_in_is_rejected_at_deployment(headers_client, declaration):
    source = BASE_SOURCE + f"    def run_api(self, {declaration}) -> str:\n        return 'unused'\n"
    response = headers_client.post(
        "/deploy_files",
        json={
            "name": "invalid_headers",
            "files": {"pipeline_wrapper.py": source},
            "save_files": False,
        },
    )
    assert response.status_code == 422, response.text
    assert "headers: Headers | None = None" in response.json()["detail"]
    assert registry.get("invalid_headers") is None


def test_postponed_header_annotation_is_resolved_without_resolving_unrelated_types(headers_client):
    source = "from __future__ import annotations\n" + wrapper_source("run_chat_completion")
    source = source.replace("messages: list[dict]", "messages: NotImportedAtRuntime")
    deploy(headers_client, source)
    response = headers_client.post(
        "/chat/completions",
        json={"model": "headers_test", "messages": []},
        headers={"Authorization": "resolved-token"},
    )
    assert response.status_code == 200, response.text
    assert "resolved-token" in response.text


def test_injected_headers_are_not_payload_logs_or_trace_tags(headers_client, recording_tracer, caplog, monkeypatch):
    monkeypatch.setattr(settings, "dashboard_trace_include_payload_values", True)
    deploy(headers_client, wrapper_source("run_api"))
    response = headers_client.post(
        "/headers_test/run",
        json={"query": "hi"},
        headers={"Authorization": "secret-transport-token"},
    )
    assert response.status_code == 200
    assert "secret-transport-token" not in caplog.text
    assert "secret-transport-token" not in repr([span.tags for span in recording_tracer.spans])
