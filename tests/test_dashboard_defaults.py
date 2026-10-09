"""Default local capture and authenticated dashboard discovery."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from hayhooks.dashboard_auth import local_dashboard_token
from hayhooks.server.app import create_app
from hayhooks.server.utils.live_trace_buffer import clear_live_traces
from hayhooks.settings import AppSettings, settings
from hayhooks.tui.client import DashboardClient
from tests.pipeline_sources import DURABLE_WRAPPER, HAYSTACK_V3, ORDINARY_WRAPPER, write_tree


@pytest.fixture(autouse=True)
def private_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv("HAYHOOKS_DASHBOARD_TOKEN", raising=False)
    monkeypatch.delenv("HAYHOOKS_DASHBOARD_API_KEY", raising=False)


def test_defaults_enable_both_monitoring_sources(monkeypatch):
    monkeypatch.delenv("HAYHOOKS_MONITORING_ENABLED", raising=False)
    monkeypatch.delenv("HAYHOOKS_TRACING_ENABLED", raising=False)
    defaults = AppSettings(_env_file=None)
    assert defaults.monitoring_enabled and defaults.tracing_enabled


def test_local_credential_is_atomic_private_and_never_sent_remotely(tmp_path):
    assert local_dashboard_token() is None
    with ThreadPoolExecutor(max_workers=4) as pool:
        tokens = list(pool.map(lambda _: local_dashboard_token(create=True), range(8)))
    assert len(set(tokens)) == 1
    assert local_dashboard_token() == tokens[0]
    assert (tmp_path / "hayhooks/dashboard-token").stat().st_mode & 0o777 == 0o600
    for host in ("localhost", "127.0.0.1", "[::1]"):
        with closing(DashboardClient(f"http://{host}:1416")) as client:
            assert client.session.headers["Authorization"] == f"Bearer {tokens[0]}"
    for host in ("example.com", "localhost.example.com", "127.0.0.1.example.com"):
        with closing(DashboardClient(f"https://{host}")) as client:
            assert "Authorization" not in client.session.headers


@pytest.mark.parametrize("problem", ["public-file", "public-directory", "symlink", "invalid"])
def test_local_credentials_fail_closed(tmp_path, problem):
    local_dashboard_token(create=True)
    path = tmp_path / "hayhooks/dashboard-token"
    if problem == "public-file":
        path.chmod(0o644)
    elif problem == "public-directory":
        path.parent.chmod(0o755)
    elif problem == "symlink":
        other = path.with_name("other")
        path.rename(other)
        path.symlink_to(other)
    else:
        path.write_text("invalid")
    with pytest.raises((OSError, ValueError)):
        local_dashboard_token()


def test_explicit_credentials_take_precedence_and_remote_hosts_require_them(monkeypatch):
    monkeypatch.setattr(settings, "host", "0.0.0.0")
    monkeypatch.setattr(settings, "tracing_enabled", True)
    monkeypatch.setattr(settings, "tracing_token", None)
    with TestClient(create_app()) as server:
        assert server.get("/traces").status_code == 403
    assert local_dashboard_token() is None
    monkeypatch.setattr(settings, "tracing_token", SecretStr("explicit"))
    monkeypatch.setenv("HAYHOOKS_DASHBOARD_TOKEN", "explicit")
    with closing(DashboardClient("https://example.com")) as client, TestClient(create_app()) as server:
        assert server.get("/traces", headers=client.session.headers).status_code == 200
        assert server.get("/traces").status_code == 401
    assert local_dashboard_token() is None


@pytest.mark.skipif(not HAYSTACK_V3, reason="durable adapters require Haystack 3.1+")
def test_default_host_captures_sync_async_streaming_and_durable_runs(
    durable_pipelines_dir, monkeypatch, wait_for_execution
):
    monkeypatch.setattr(settings, "durable_store", "memory")
    monkeypatch.setattr(settings, "host", "localhost")
    monkeypatch.setattr(settings, "monitoring_enabled", True)
    monkeypatch.setattr(settings, "tracing_enabled", True)
    monkeypatch.setattr(settings, "tracing_token", None)
    write_tree(durable_pipelines_dir, {
        "sync/pipeline_wrapper.py": ORDINARY_WRAPPER,
        "async/pipeline_wrapper.py": ORDINARY_WRAPPER.replace("def run_api(", "async def run_api_async("),
        "stream/pipeline_wrapper.py": ORDINARY_WRAPPER.replace(
            "from hayhooks", "from collections.abc import Generator\nfrom hayhooks"
        ).replace("-> int:", "-> Generator[str, None, None]:").replace("return value * 2", "yield str(value)"),
        "jobs/pipeline_wrapper.py": DURABLE_WRAPPER,
    })
    app = create_app()
    with TestClient(app) as server, closing(DashboardClient("http://localhost:1416")) as dashboard:
        headers = dashboard.session.headers
        assert server.get("/traces").status_code == 401
        assert server.get("/jobs/executions").status_code == 401
        clear_live_traces()
        for name in ("sync", "async", "stream"):
            assert server.post(f"/{name}/run", json={"value": 2}).status_code == 200
        submitted = server.post("/jobs/run-durable", json={"value": 3}).json()
        run_id = submitted["execution_id"]
        wait_for_execution(server, f"/jobs/executions/{run_id}", "completed")
        traces = server.get("/traces", headers=headers).json()["traces"]
        assert {trace["entrypoint"] for trace in traces} >= {"sync", "async", "stream", "jobs"}
        assert all(not trace["root_span"]["running"] for trace in traces)
        executions = server.get("/jobs/executions", headers=headers).json()["executions"]
        assert [row["execution_id"] for row in executions] == [run_id]
        assert executions[0]["status"] == "completed"
        assert server.get("/status").json()["capabilities"]["execution_enumeration"] is True


def test_host_operator_dependency_overrides_apply_to_both_sources(monkeypatch):
    from fastapi import HTTPException
    from hayhooks.durable import DurableDeployment, MemoryExecutionStore, create_durable_router
    from hayhooks.durable.store import StoreConfig
    from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper
    from hayhooks.server.routers.traces import require_tracing_access
    from tests.test_durable_monitoring import RequestModel, no_run

    class Wrapper(BasePipelineWrapper):
        def setup(self):
            pass

    monkeypatch.setattr(settings, "host", "localhost")
    monkeypatch.setattr(settings, "tracing_token", None)
    monkeypatch.setattr(settings, "tracing_enabled", True)
    app = create_app()
    deployment = DurableDeployment("jobs", "v1", MemoryExecutionStore("jobs", config=StoreConfig(monitoring_enabled=True)), RequestModel, no_run)
    app.include_router(create_durable_router(deployment, owner_id_dependency=None, monitoring_enabled=True,
                       monitoring_dependency=Wrapper().durable_monitoring_access), prefix="/jobs")
    app.dependency_overrides[require_tracing_access] = lambda: None
    with TestClient(app) as client:
        assert client.get("/traces").status_code == 200
        assert client.get("/jobs/executions").status_code == 200

        def deny():
            raise HTTPException(403, "Operator required")

        app.dependency_overrides[require_tracing_access] = deny
        assert client.get("/traces").status_code == 403
        assert client.get("/jobs/executions").status_code == 403


def test_automatic_credentials_bypass_proxies_and_preserve_tls_ca(monkeypatch):
    local_dashboard_token(create=True)
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example.test:8888")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example.test:8888")
    monkeypatch.setenv("ALL_PROXY", "http://proxy.example.test:8888")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", "/private-ca.pem")
    with closing(DashboardClient("http://localhost:1416")) as client:
        transport = client.session.merge_environment_settings(client.url, {}, False, client.session.verify, None)
        assert not transport["proxies"]
        assert transport["verify"] == "/private-ca.pem"
    monkeypatch.setenv("HAYHOOKS_DASHBOARD_TOKEN", "explicit")
    with closing(DashboardClient("https://example.com")) as client:
        assert client.session.trust_env


def test_unavailable_automatic_credentials_do_not_prevent_host_startup(tmp_path, monkeypatch):
    directory = tmp_path / "hayhooks"
    directory.mkdir(mode=0o755)
    monkeypatch.setattr(settings, "host", "localhost")
    monkeypatch.setattr(settings, "tracing_enabled", True)
    monkeypatch.setattr(settings, "tracing_token", None)
    app = create_app()
    assert app.state.dashboard_token is None
    with TestClient(app) as client:
        assert client.get("/status").status_code == 200
        assert client.get("/traces").status_code == 403


def test_verified_operator_middleware_is_not_preempted_by_automatic_credentials(monkeypatch):
    monkeypatch.setattr(settings, "host", "localhost")
    monkeypatch.setattr(settings, "tracing_enabled", True)
    monkeypatch.setattr(settings, "tracing_token", None)
    app = create_app()
    assert app.state.dashboard_token

    @app.middleware("http")
    async def verified_operator(request, call_next):
        request.state.tracing_operator = True
        return await call_next(request)

    with TestClient(app) as client:
        assert client.get("/traces").status_code == 200
