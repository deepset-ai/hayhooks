from fastapi import HTTPException
from fastapi.testclient import TestClient

from hayhooks.server.app import create_app
from hayhooks.server.routers.traces import require_tracing_access
from hayhooks.settings import settings


def test_trace_api_is_opt_in_and_browser_dashboard_is_gone(monkeypatch):
    monkeypatch.setattr(settings, "tracing_enabled", False)
    client = TestClient(create_app())
    assert client.get("/traces").status_code == 404
    assert client.get("/dashboard").status_code == 404
    assert client.get("/dashboard/api/traces").status_code == 404


def test_all_trace_routes_require_operator_access(monkeypatch):
    monkeypatch.setattr(settings, "tracing_enabled", True)
    app = create_app()
    client = TestClient(app)
    for path in ("/traces", "/traces/config", "/traces/entrypoints", "/traces/stream"):
        assert client.get(path).status_code == 403
    assert client.post("/traces/clear").status_code == 403
    app.dependency_overrides[require_tracing_access] = lambda: None
    assert client.get("/traces").status_code == 200
    assert client.get("/traces/config").json()["scope"].startswith("Responding process")

    def deny():
        raise HTTPException(401, "Unauthorized")

    app.dependency_overrides[require_tracing_access] = deny
    assert client.get("/traces").status_code == 401


def test_trace_operator_token_checks_all_routes_and_masks_settings(monkeypatch):
    from pydantic import SecretStr

    monkeypatch.setattr(settings, "tracing_enabled", True)
    monkeypatch.setattr(settings, "tracing_token", SecretStr("private-test-token"))
    client = TestClient(create_app())
    for path in ("/traces", "/traces/config", "/traces/entrypoints", "/traces/stream"):
        assert client.get(path).status_code == 401
        assert client.get(path, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/traces", headers={"Authorization": "Bearer private-test-token"}).status_code == 200
    assert "private-test-token" not in settings.model_dump_json()
