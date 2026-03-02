from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hayhooks.server.app import create_app
from hayhooks.server.pipelines.registry import registry
from hayhooks.server.utils.deploy_utils import (
    commit_prepared_pipeline,
    deploy_pipeline_files_async,
    deploy_pipeline_yaml,
    deploy_pipeline_yaml_async,
    prepare_pipeline_yaml,
    rebuild_openapi,
    undeploy_pipeline_async,
)
from hayhooks.server.utils.yaml_utils import (
    get_inputs_outputs_from_yaml,
    get_streaming_components_from_yaml,
    parse_yaml_pipeline,
)
from hayhooks.settings import DeployConcurrencyPolicy, StartupDeployStrategy, settings

YAML_DIR = Path(__file__).parent / "test_files/yaml"
SAMPLE_YAML = (YAML_DIR / "sample_calc_pipeline.yml").read_text()
SAMPLE_FILES_WRAPPER = Path("tests/test_files/files/no_chat/pipeline_wrapper.py").read_text()
MIN_EXPECTED_DEPLOYED = 6  # 7 YAML + 1 dir; broken dir always fails, trafilatura is optional


@pytest.fixture(autouse=True)
def _cleanup():
    registry.clear()
    yield
    registry.clear()


def test_default_settings():
    assert settings.deploy_concurrency == DeployConcurrencyPolicy.SERIALIZED
    assert settings.startup_deploy_strategy == StartupDeployStrategy.PARALLEL
    assert settings.startup_deploy_workers == 4


def test_get_inputs_outputs_reuses_parsed(monkeypatch):
    parsed = parse_yaml_pipeline(SAMPLE_YAML)
    calls = []
    monkeypatch.setattr("hayhooks.server.utils.yaml_utils.yaml.safe_load", lambda *_a, **_kw: calls.append(1))
    get_inputs_outputs_from_yaml(SAMPLE_YAML, _parsed=parsed)
    assert calls == []


def test_get_streaming_components_reuses_parsed(monkeypatch):
    parsed = parse_yaml_pipeline(SAMPLE_YAML)
    calls = []
    monkeypatch.setattr("hayhooks.server.utils.yaml_utils.yaml.safe_load", lambda *_a, **_kw: calls.append(1))
    get_streaming_components_from_yaml(SAMPLE_YAML, _parsed=parsed)
    assert calls == []


@pytest.mark.asyncio
async def test_deploy_pipeline_yaml_async(monkeypatch):
    monkeypatch.setattr(settings, "deploy_concurrency", DeployConcurrencyPolicy.SERIALIZED)

    result = await deploy_pipeline_yaml_async(
        pipeline_name="calc_async_test",
        source_code=SAMPLE_YAML,
        options={"save_file": False},
    )

    assert result == {"name": "calc_async_test"}
    assert registry.get("calc_async_test") is not None


@pytest.mark.asyncio
async def test_deploy_pipeline_files_async(monkeypatch):
    monkeypatch.setattr(settings, "deploy_concurrency", DeployConcurrencyPolicy.SERIALIZED)

    files = {"pipeline_wrapper.py": SAMPLE_FILES_WRAPPER}

    result = await deploy_pipeline_files_async(
        pipeline_name="files_async_test",
        files=files,
        save_files=False,
    )

    assert result == {"name": "files_async_test"}
    assert registry.get("files_async_test") is not None


@pytest.mark.asyncio
async def test_undeploy_pipeline_async(monkeypatch):
    monkeypatch.setattr(settings, "deploy_concurrency", DeployConcurrencyPolicy.SERIALIZED)

    deploy_pipeline_yaml(
        pipeline_name="undeploy_async_test",
        source_code=SAMPLE_YAML,
        options={"save_file": False},
    )
    assert registry.get("undeploy_async_test") is not None

    await undeploy_pipeline_async(pipeline_name="undeploy_async_test")
    assert registry.get("undeploy_async_test") is None


@pytest.mark.asyncio
async def test_serialized_policy_wraps_with_lock(monkeypatch):
    monkeypatch.setattr(settings, "deploy_concurrency", DeployConcurrencyPolicy.SERIALIZED)

    lock_calls = []
    original_with_lock = __import__(
        "hayhooks.server.utils.deploy_utils", fromlist=["_with_deploy_lock"]
    )._with_deploy_lock

    def tracking_with_lock(func):
        lock_calls.append(func.__name__)
        return original_with_lock(func)

    monkeypatch.setattr(
        "hayhooks.server.utils.deploy_utils._with_deploy_lock",
        tracking_with_lock,
    )

    await deploy_pipeline_yaml_async(
        pipeline_name="lock_test",
        source_code=SAMPLE_YAML,
        options={"save_file": False},
    )

    assert lock_calls == ["deploy_pipeline_yaml"]


def test_defer_openapi_rebuild_skips_setup():
    mock_app = MagicMock(spec=FastAPI)
    mock_app.routes = []

    deploy_pipeline_yaml(
        pipeline_name="defer_test",
        source_code=SAMPLE_YAML,
        app=mock_app,
        options={"save_file": False},
        _defer_openapi_rebuild=True,
    )

    mock_app.setup.assert_not_called()


def test_no_defer_calls_setup():
    mock_app = MagicMock(spec=FastAPI)
    mock_app.routes = []

    deploy_pipeline_yaml(
        pipeline_name="no_defer_test",
        source_code=SAMPLE_YAML,
        app=mock_app,
        options={"save_file": False},
    )

    mock_app.setup.assert_called()


def test_rebuild_openapi():
    mock_app = MagicMock(spec=FastAPI)
    mock_app.openapi_schema = {"existing": "schema"}

    rebuild_openapi(mock_app)

    assert mock_app.openapi_schema is None
    mock_app.setup.assert_called_once()


def test_prepare_then_commit_pipeline():
    prepared = prepare_pipeline_yaml("prep_test", source_code=SAMPLE_YAML, options={"save_file": False})
    assert prepared.name == "prep_test"
    assert prepared.wrapper is not None
    assert prepared.extra_metadata is not None

    result = commit_prepared_pipeline(prepared)
    assert result == {"name": "prep_test"}
    assert registry.get("prep_test") is not None


def test_deploy_pipelines_sequential(test_settings, monkeypatch):
    monkeypatch.setattr(test_settings, "pipelines_dir", str(YAML_DIR))
    monkeypatch.setattr(test_settings, "startup_deploy_strategy", StartupDeployStrategy.SEQUENTIAL)

    app = create_app()
    with TestClient(app) as client:
        response = client.get("/status")
        assert response.status_code == 200
        assert len(response.json()["pipelines"]) >= MIN_EXPECTED_DEPLOYED


def test_deploy_pipelines_parallel(test_settings, monkeypatch):
    monkeypatch.setattr(test_settings, "pipelines_dir", str(YAML_DIR))
    monkeypatch.setattr(test_settings, "startup_deploy_strategy", StartupDeployStrategy.PARALLEL)
    monkeypatch.setattr(test_settings, "startup_deploy_workers", 2)

    app = create_app()
    with TestClient(app) as client:
        response = client.get("/status")
        assert response.status_code == 200
        assert len(response.json()["pipelines"]) >= MIN_EXPECTED_DEPLOYED


def test_parallel_and_sequential_deploy_same_result(test_settings, monkeypatch):
    monkeypatch.setattr(test_settings, "pipelines_dir", str(YAML_DIR))

    monkeypatch.setattr(test_settings, "startup_deploy_strategy", StartupDeployStrategy.SEQUENTIAL)
    seq_app = create_app()
    with TestClient(seq_app) as client:
        seq_pipelines = set(client.get("/status").json()["pipelines"])

    registry.clear()

    monkeypatch.setattr(test_settings, "startup_deploy_strategy", StartupDeployStrategy.PARALLEL)
    monkeypatch.setattr(test_settings, "startup_deploy_workers", 4)
    par_app = create_app()
    with TestClient(par_app) as client:
        par_pipelines = set(client.get("/status").json()["pipelines"])

    assert seq_pipelines == par_pipelines
    assert len(seq_pipelines) >= 3


def test_parallel_deploy_rebuilds_openapi_exactly_once(test_settings, monkeypatch):
    monkeypatch.setattr(test_settings, "pipelines_dir", str(YAML_DIR))
    monkeypatch.setattr(test_settings, "startup_deploy_strategy", StartupDeployStrategy.PARALLEL)

    calls = []
    original_rebuild = rebuild_openapi

    def tracking_rebuild(app):
        calls.append(1)
        return original_rebuild(app)

    monkeypatch.setattr("hayhooks.server.app.rebuild_openapi", tracking_rebuild)

    app = create_app()
    with TestClient(app) as client:
        pipelines = client.get("/status").json()["pipelines"]
        assert len(pipelines) >= 3

    assert len(calls) == 1
