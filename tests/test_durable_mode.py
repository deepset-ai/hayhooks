"""Durable mode: a pipeline set fixed at startup, with every mutation surface absent or rejected."""

import asyncio
import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hayhooks.server import app as server_app
from hayhooks.server.app import create_app
from hayhooks.server.exceptions import PipelineModeError
from hayhooks.server.pipelines.loader import REGISTRY_ROOT, load_pipeline_registry
from hayhooks.server.pipelines.registry import registry
from hayhooks.server.utils import deploy_utils
from hayhooks.server.utils.models import PreparedPipeline
from hayhooks.server.utils.yaml_pipeline_wrapper import YAMLPipelineWrapper
from hayhooks.settings import settings
from tests.pipeline_sources import CALC_YAML, CHAT_WRAPPER, DURABLE_WRAPPER, HAYSTACK_V3, ORDINARY_WRAPPER, write_tree

MUTATION_ROUTES = [("post", "/deploy_files"), ("post", "/deploy-yaml"), ("post", "/undeploy/double")]
FILES = {"pipeline_wrapper.py": ORDINARY_WRAPPER}


def _registry_modules() -> set[str]:
    return {name for name in sys.modules if name == REGISTRY_ROOT or name.startswith(f"{REGISTRY_ROOT}.")}


@pytest.fixture
def immutable_app(durable_pipelines_dir: Path) -> FastAPI:
    samples = Path(__file__).parent / "test_files/files"
    write_tree(
        durable_pipelines_dir,
        {
            "double/pipeline_wrapper.py": ORDINARY_WRAPPER,
            "chat/pipeline_wrapper.py": CHAT_WRAPPER,
            "calc.yml": CALC_YAML,
            **{
                f"{name}/pipeline_wrapper.py": (samples / sample / "pipeline_wrapper.py").read_text()
                for name, sample in (
                    ("stream", "run_api_streaming"),
                    ("image", "file_response"),
                    ("upload", "upload_files"),
                )
            },
        },
    )
    registry.clear()
    return create_app()


def test_immutable_server_serves_fixed_pipelines_without_deployment_routes(
    immutable_app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The mode is captured at construction; changing the setting afterwards changes nothing.
    monkeypatch.setattr(settings, "durable_mode", False)

    with TestClient(immutable_app) as client:
        assert client.post("/double/run", json={"value": 21}).json() == {"result": 42}
        assert client.post("/calc/run", json={"value": 1}).status_code == 200
        assert client.post("/stream/run", json={"query": "one two"}).text == "one two "
        assert client.post("/image/run", json={}).headers["content-type"] == "image/png"
        uploaded = client.post("/upload/run", data={"test_param": "x"}, files={"files": ("a.txt", b"a", "text/plain")})
        assert uploaded.json() == {"result": "Received files: a.txt with param x"}
        completion = client.post("/chat/completions", json={"model": "chat", "messages": []}).json()
        assert completion["choices"][0]["message"]["content"] == "hello from chat"
        names = ["calc", "chat", "double", "image", "stream", "upload"]
        assert sorted(model["id"] for model in client.get("/models").json()["data"]) == names
        assert client.get("/dashboard/api/entrypoints").json() == {"entrypoints": names}
        assert client.get("/status/double").status_code == 200

        status = client.get("/status").json()
        assert status["durable_mode"] is True
        assert status["pipelines"] == names
        assert status["durable"] == {"healthy": True, "deployments": {}}

        for method, path in MUTATION_ROUTES:
            assert getattr(client, method)(path, json={}).status_code == 404
        paths = client.get("/openapi.json").json()["paths"]
        assert not {"/deploy_files", "/deploy-yaml", "/undeploy/{pipeline_name}"} & set(paths)

    # Ordinary pipelines need neither Redis nor a durable runtime, and nothing reaches the mutable singleton.
    assert immutable_app.state.durable_runtime is None
    assert immutable_app.state.durable_redis_clients == ()
    assert registry.get_names() == []


def test_default_mode_status_reports_live_deployment(client: TestClient) -> None:
    assert client.get("/status").json()["durable_mode"] is False


def _call_async(function: Callable[..., Any]) -> Callable[..., Any]:
    return lambda **kwargs: asyncio.run(function(**kwargs))


DEPLOY_FILES = {"pipeline_name": "new", "files": FILES}
DEPLOY_YAML = {"pipeline_name": "new", "source_code": CALC_YAML}


# Helpers that take the app are rejected by the app's registry type as well as by the setting.
APP_CALLS = [
    pytest.param(lambda app: deploy_utils.deploy_pipeline_files(**DEPLOY_FILES, app=app), id="deploy-files"),
    pytest.param(lambda app: deploy_utils.deploy_pipeline_yaml(**DEPLOY_YAML, app=app), id="deploy-yaml"),
    pytest.param(lambda app: deploy_utils.undeploy_pipeline("double", app=app), id="undeploy"),
    pytest.param(
        lambda app: _call_async(deploy_utils.deploy_pipeline_files_async)(**DEPLOY_FILES, app=app),
        id="deploy-files-async",
    ),
    pytest.param(
        lambda app: _call_async(deploy_utils.deploy_pipeline_yaml_async)(**DEPLOY_YAML, app=app),
        id="deploy-yaml-async",
    ),
    pytest.param(
        lambda app: _call_async(deploy_utils.undeploy_pipeline_async)(pipeline_name="double", app=app),
        id="undeploy-async",
    ),
    pytest.param(
        lambda app: deploy_utils.commit_prepared_pipeline(
            PreparedPipeline("new", YAMLPipelineWrapper.from_yaml(CALC_YAML)), app=app
        ),
        id="commit",
    ),
    pytest.param(
        lambda app: deploy_utils.add_pipeline_api_route(app, "double", YAMLPipelineWrapper.from_yaml(CALC_YAML)),
        id="add-route",
    ),
    pytest.param(lambda app: server_app.deploy_pipelines(app, settings.pipelines_dir), id="app-startup"),
]
APPLESS_CALLS = [
    pytest.param(lambda _app: deploy_utils.prepare_pipeline_files("new", FILES), id="prepare-files"),
    pytest.param(lambda _app: deploy_utils.prepare_pipeline_yaml("new", CALC_YAML), id="prepare-yaml"),
    pytest.param(lambda _app: deploy_utils.save_pipeline_files("new", FILES, settings.pipelines_dir), id="save"),
    pytest.param(lambda _app: deploy_utils.remove_pipeline_files("double", settings.pipelines_dir), id="remove"),
    pytest.param(lambda _app: server_app.init_pipeline_dir(settings.pipelines_dir), id="init-dir"),
    pytest.param(lambda _app: deploy_utils.deploy_pipelines(), id="standalone-startup"),
]


@pytest.mark.parametrize(
    ("call", "setting_enabled"),
    [
        *(pytest.param(case.values[0], True, id=f"setting-{case.id}") for case in [*APP_CALLS, *APPLESS_CALLS]),
        *(pytest.param(case.values[0], False, id=f"app-registry-{case.id}") for case in APP_CALLS),
    ],
)
def test_legacy_mutation_is_rejected_before_side_effects(
    immutable_app: FastAPI,
    durable_pipelines_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    call: Callable[[FastAPI], object],
    setting_enabled: bool,
) -> None:
    monkeypatch.setattr(settings, "durable_mode", setting_enabled)
    source = {path: path.read_bytes() for path in durable_pipelines_dir.rglob("*") if path.is_file()}
    modules = _registry_modules()
    routes = list(immutable_app.routes)

    with TestClient(immutable_app) as client:
        with pytest.raises(PipelineModeError, match="durable mode"):
            call(immutable_app)
        # An existing endpoint keeps serving its original wrapper.
        assert client.post("/double/run", json={"value": 21}).json() == {"result": 42}

    assert {path: path.read_bytes() for path in durable_pipelines_dir.rglob("*") if path.is_file()} == source
    assert _registry_modules() == modules
    assert immutable_app.routes == routes
    assert registry.get_names() == []


@pytest.mark.parametrize("helper", [server_app.init_pipeline_dir, lambda path: server_app.deploy_pipelines(None, path)])
def test_startup_helpers_do_not_create_a_missing_directory(
    durable_pipelines_dir: Path, tmp_path: Path, helper: Callable[[Path], object]
) -> None:
    missing = tmp_path / "missing"

    with pytest.raises(PipelineModeError):
        helper(missing)

    assert not missing.exists()


@pytest.mark.parametrize(
    ("files", "dashboard_path", "match"),
    [
        pytest.param(
            {"status/pipeline_wrapper.py": ORDINARY_WRAPPER}, "/dashboard", r"\['status'\] conflict", id="status"
        ),
        pytest.param({"draw/pipeline_wrapper.py": ORDINARY_WRAPPER}, "/dashboard", r"\['draw'\] conflict", id="draw"),
        pytest.param(
            {"dashboard/pipeline_wrapper.py": ORDINARY_WRAPPER},
            "/dashboard",
            r"\['dashboard'\] conflict",
            id="dashboard",
        ),
        pytest.param(
            {"ops/pipeline_wrapper.py": ORDINARY_WRAPPER}, "/ops", r"\['ops'\] conflict", id="configured-mount"
        ),
        pytest.param(
            {"jobs/pipeline_wrapper.py": DURABLE_WRAPPER.replace("self.pipeline = Pipeline()", "self.pipeline = None")},
            "/dashboard",
            "durable deployment of pipeline 'jobs'",
            id="durable-adapter",
        ),
    ],
)
def test_host_construction_failures_fail_startup_and_unload_modules(
    durable_pipelines_dir: Path, monkeypatch: pytest.MonkeyPatch, files: dict[str, str], dashboard_path: str, match: str
) -> None:
    for setting, value in (("dashboard_enabled", True), ("dashboard_path", dashboard_path)):
        monkeypatch.setattr(settings, setting, value)
    monkeypatch.setattr(settings, "dashboard_dist_dir", str(durable_pipelines_dir.parent))
    # "v1" shares a first path segment with /v1/models without matching its run path, so it loads.
    write_tree(durable_pipelines_dir, {**files, "v1.yml": CALC_YAML})

    with pytest.raises(PipelineModeError, match=match):
        create_app()

    assert _registry_modules() == set()


@pytest.mark.skipif(not HAYSTACK_V3, reason="durable adapters require Haystack 3.1+")
@pytest.mark.parametrize(
    "dashboard_path",
    [
        "/jobs/run-durable",
        "/jobs/executions",
        *(f"/jobs/executions/{'a' * 32}{suffix}" for suffix in ("", "/cancel", "/resume", "/stream")),
    ],
)
def test_durable_routes_cannot_be_shadowed_by_nested_mounts(
    durable_pipelines_dir: Path, monkeypatch: pytest.MonkeyPatch, dashboard_path: str
) -> None:
    write_tree(durable_pipelines_dir, {"jobs/pipeline_wrapper.py": DURABLE_WRAPPER})
    monkeypatch.setattr(settings, "durable_store", "memory")
    monkeypatch.setattr(settings, "dashboard_enabled", True)
    monkeypatch.setattr(settings, "dashboard_path", dashboard_path)
    monkeypatch.setattr(settings, "dashboard_dist_dir", str(durable_pipelines_dir.parent))

    with pytest.raises(PipelineModeError, match="conflict"):
        create_app()

    assert _registry_modules() == set()


@pytest.mark.skipif(not HAYSTACK_V3, reason="durable adapters require Haystack 3.1+")
def test_durable_routes_can_share_a_prefix_with_an_unrelated_mount(
    durable_pipelines_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_tree(durable_pipelines_dir, {"jobs/pipeline_wrapper.py": DURABLE_WRAPPER})
    monkeypatch.setattr(settings, "durable_store", "memory")
    monkeypatch.setattr(settings, "dashboard_enabled", True)
    monkeypatch.setattr(settings, "dashboard_path", "/jobs/monitor")
    monkeypatch.setattr(settings, "dashboard_dist_dir", str(durable_pipelines_dir.parent))

    with TestClient(create_app()) as client:
        response = client.post("/jobs/run-durable", json={"value": 2})
        assert response.status_code == 202
        assert client.get(response.headers["Location"]).status_code == 200


def test_additional_python_path_is_importable_while_loading(
    durable_pipelines_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared = write_tree(tmp_path / "shared", {"shared_factor.py": "FACTOR = 5\n"})
    monkeypatch.setattr(settings, "additional_python_path", str(shared))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delitem(sys.modules, "shared_factor", raising=False)
    source = "from shared_factor import FACTOR\n" + ORDINARY_WRAPPER.replace("value * 2", "value * FACTOR")
    write_tree(durable_pipelines_dir, {"scaled/pipeline_wrapper.py": source})

    with TestClient(create_app()) as client:
        assert client.post("/scaled/run", json={"value": 2}).json() == {"result": 10}


@pytest.mark.skipif(importlib.util.find_spec("mcp") is None, reason="MCP is not installed")
@pytest.mark.mcp
async def test_immutable_mcp_server_has_no_mutation_tools(
    durable_pipelines_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mcp.shared.memory import create_connected_server_and_client_session

    from hayhooks.server.utils import mcp_utils

    write_tree(durable_pipelines_dir, {"double/pipeline_wrapper.py": ORDINARY_WRAPPER})
    server = mcp_utils.create_mcp_server(registry=load_pipeline_registry(durable_pipelines_dir))
    notified = []
    monkeypatch.setattr(mcp_utils, "notify_client", lambda _server: notified.append(True))

    async with create_connected_server_and_client_session(server) as session:
        tools = {tool.name for tool in (await session.list_tools()).tools}
        assert tools == {"double", "get_all_pipeline_statuses", "get_pipeline_status"}
        assert (await session.call_tool("double", {"value": 2})).content[0].text == "4"  # ty: ignore[unresolved-attribute]
        for tool in ("deploy_pipeline", "undeploy_pipeline"):
            result = await session.call_tool(tool, {"name": "double", "pipeline_name": "double"})
            assert result.isError
            assert "durable mode" in result.content[0].text  # ty: ignore[unresolved-attribute]

    assert notified == []


@pytest.mark.parametrize(
    ("server", "files", "match"),
    [
        pytest.param("mcp", {"deploy_pipeline/pipeline_wrapper.py": ORDINARY_WRAPPER}, "MCP core tools", id="mcp-name"),
        pytest.param("mcp", {"jobs/pipeline_wrapper.py": DURABLE_WRAPPER}, "main HTTP server", id="mcp-durable"),
        pytest.param("a2a", {"status/pipeline_wrapper.py": CHAT_WRAPPER}, "reserved by the A2A", id="a2a-path"),
        pytest.param("a2a", {"jobs/pipeline_wrapper.py": DURABLE_WRAPPER}, "main HTTP server", id="a2a-durable"),
        pytest.param(
            "a2a",
            {
                "chat/pipeline_wrapper.py": CHAT_WRAPPER.replace(
                    "    def setup", "    a2a_card = {'skills': [1]}\n\n    def setup"
                )
            },
            "has no attribute 'get'",
            id="a2a-malformed-card",
        ),
    ],
)
def test_standalone_immutable_host_construction_fails_and_unloads_modules(
    durable_pipelines_dir: Path, server: str, files: dict[str, str], match: str
) -> None:
    pytest.importorskip({"mcp": "mcp", "a2a": "a2a"}[server])
    if "durable" in match or "HTTP" in match:
        pytest.importorskip("haystack.components.agents")
    from hayhooks.server.pipelines.loader import build_immutable_host
    from hayhooks.server.utils.a2a_utils import create_a2a_app
    from hayhooks.server.utils.mcp_utils import create_mcp_server

    build = {"mcp": lambda reg: create_mcp_server(registry=reg), "a2a": lambda reg: create_a2a_app(registry=reg)}
    write_tree(durable_pipelines_dir, files)

    with pytest.raises(Exception, match=match):
        build_immutable_host(durable_pipelines_dir, build[server])

    assert _registry_modules() == set()


@pytest.mark.parametrize("factory", ["mcp", "a2a"])
def test_standalone_factories_require_an_explicit_registry_in_durable_mode(durable_pipelines_dir: Path, factory: str):
    pytest.importorskip(factory)
    from hayhooks.server.utils.a2a_utils import create_a2a_app
    from hayhooks.server.utils.mcp_utils import create_mcp_server

    with pytest.raises(PipelineModeError, match="explicit pipeline registry"):
        {"mcp": create_mcp_server, "a2a": create_a2a_app}[factory]()


@pytest.mark.skipif(importlib.util.find_spec("a2a") is None, reason="A2A is not installed")
@pytest.mark.a2a
def test_immutable_a2a_exposes_only_chat_pipelines(durable_pipelines_dir: Path) -> None:
    from hayhooks.server.utils.a2a_utils import create_a2a_app

    skipped = CHAT_WRAPPER.replace("    def setup", "    skip_a2a = True\n\n    def setup")
    write_tree(
        durable_pipelines_dir,
        {
            "chat/pipeline_wrapper.py": CHAT_WRAPPER,
            "hidden/pipeline_wrapper.py": skipped,
            "double/pipeline_wrapper.py": ORDINARY_WRAPPER,
        },
    )

    with TestClient(create_a2a_app(registry=load_pipeline_registry(durable_pipelines_dir))) as client:
        assert client.get("/status").json()["agents"] == ["chat"]
        assert client.get("/chat/.well-known/agent-card.json").status_code == 200


@pytest.mark.skipif(not HAYSTACK_V3, reason="durable adapters require Haystack 3.1+")
def test_durable_wrapper_runs_detached_while_ordinary_requests_stay_ordinary(
    durable_pipelines_dir: Path, monkeypatch: pytest.MonkeyPatch, recording_tracer, wait_for_execution
) -> None:
    for name, value in (("durable_store", "memory"), ("durable_release_running_on_shutdown", True)):
        monkeypatch.setattr(settings, name, value)
    write_tree(durable_pipelines_dir, {"jobs/pipeline_wrapper.py": DURABLE_WRAPPER})
    app = create_app()
    [deployment] = app.state.durable_runtime._deployments.values()
    assert deployment.config.release_running_on_close is True
    assert app.state.durable_redis_clients == ()

    with TestClient(app) as client:
        assert client.post("/jobs/run", json={"value": 2}).json() == {"result": 3}
        submitted = client.post("/jobs/run-durable", json={"value": 2})
        assert submitted.status_code == 202
        completed = wait_for_execution(client, submitted.headers["Location"], "completed")
        assert completed["result"] == {"value": 20}
        assert client.get("/status").json()["durable"]["deployments"]["jobs"]["healthy"] is True

    [attempt] = [span for span in recording_tracer.spans if span.operation_name == "hayhooks.durable.attempt"]
    assert {
        "hayhooks.durable.execution_id": submitted.json()["execution_id"],
        "hayhooks.transport": "durable",
        "hayhooks.pipeline.name": "jobs",
        "hayhooks.durable.attempt": 1,
        "hayhooks.durable.kind": "pipeline",
        "hayhooks.durable.definition_revision": "v1",
        "hayhooks.success": True,
    }.items() <= attempt.tags.items()
