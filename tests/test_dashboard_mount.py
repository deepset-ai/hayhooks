from fastapi.testclient import TestClient

from hayhooks.server.app import create_app
from hayhooks.settings import settings


def test_dashboard_static_ui_mounts_when_enabled(monkeypatch, tmp_path):
    dist_dir = tmp_path / "dashboard-dist"
    dist_dir.mkdir(parents=True, exist_ok=True)
    (dist_dir / "index.html").write_text("<html><body>dashboard-ui</body></html>", encoding="utf-8")

    monkeypatch.setattr(settings, "dashboard_enabled", True, raising=False)
    monkeypatch.setattr(settings, "dashboard_path", "/dashboard", raising=False)
    monkeypatch.setattr(settings, "dashboard_dist_dir", str(dist_dir), raising=False)
    monkeypatch.setattr(settings, "chainlit_enabled", False, raising=False)
    monkeypatch.setattr(settings, "root_path", "", raising=False)

    app = create_app()
    client = TestClient(app)
    response = client.get("/dashboard")

    assert response.status_code == 200
    assert "dashboard-ui" in response.text


def test_dashboard_static_ui_not_mounted_when_disabled(monkeypatch, tmp_path):
    dist_dir = tmp_path / "dashboard-dist"
    dist_dir.mkdir(parents=True, exist_ok=True)
    (dist_dir / "index.html").write_text("<html><body>dashboard-ui</body></html>", encoding="utf-8")

    monkeypatch.setattr(settings, "dashboard_enabled", False, raising=False)
    monkeypatch.setattr(settings, "dashboard_path", "/dashboard", raising=False)
    monkeypatch.setattr(settings, "dashboard_dist_dir", str(dist_dir), raising=False)
    monkeypatch.setattr(settings, "chainlit_enabled", False, raising=False)
    monkeypatch.setattr(settings, "root_path", "", raising=False)

    app = create_app()
    client = TestClient(app)
    response = client.get("/dashboard")

    assert response.status_code == 404


def test_dashboard_static_ui_mounts_at_custom_path(monkeypatch, tmp_path):
    dist_dir = tmp_path / "dashboard-dist"
    dist_dir.mkdir(parents=True, exist_ok=True)
    (dist_dir / "index.html").write_text("<html><body>custom-dashboard-ui</body></html>", encoding="utf-8")

    monkeypatch.setattr(settings, "dashboard_enabled", True, raising=False)
    monkeypatch.setattr(settings, "dashboard_path", "/observability", raising=False)
    monkeypatch.setattr(settings, "dashboard_dist_dir", str(dist_dir), raising=False)
    monkeypatch.setattr(settings, "chainlit_enabled", False, raising=False)
    monkeypatch.setattr(settings, "root_path", "", raising=False)

    app = create_app()
    client = TestClient(app)

    response_default = client.get("/dashboard")
    assert response_default.status_code == 404

    response_custom = client.get("/observability")
    assert response_custom.status_code == 200
    assert "custom-dashboard-ui" in response_custom.text


def test_dashboard_api_routes_under_custom_path(monkeypatch, tmp_path):
    dist_dir = tmp_path / "dashboard-dist"
    dist_dir.mkdir(parents=True, exist_ok=True)
    (dist_dir / "index.html").write_text("<html><body>dashboard</body></html>", encoding="utf-8")

    monkeypatch.setattr(settings, "dashboard_enabled", True, raising=False)
    monkeypatch.setattr(settings, "dashboard_path", "/obs", raising=False)
    monkeypatch.setattr(settings, "dashboard_dist_dir", str(dist_dir), raising=False)
    monkeypatch.setattr(settings, "chainlit_enabled", False, raising=False)
    monkeypatch.setattr(settings, "root_path", "", raising=False)
    monkeypatch.setattr(settings, "dashboard_trace_default_limit", 10, raising=False)
    monkeypatch.setattr(settings, "dashboard_trace_max_limit", 25, raising=False)

    app = create_app()
    client = TestClient(app)

    response_old = client.get("/dashboard/api/config")
    assert response_old.status_code == 404

    response_new = client.get("/obs/api/config")
    assert response_new.status_code == 200
    body = response_new.json()
    assert body["api_base"] == "/obs/api"


def test_dashboard_static_ui_not_mounted_when_dist_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "dashboard_enabled", True, raising=False)
    monkeypatch.setattr(settings, "dashboard_path", "/dashboard", raising=False)
    monkeypatch.setattr(settings, "dashboard_dist_dir", str(tmp_path / "missing-dashboard-dist"), raising=False)
    monkeypatch.setattr(settings, "chainlit_enabled", False, raising=False)
    monkeypatch.setattr(settings, "root_path", "", raising=False)

    app = create_app()
    client = TestClient(app)
    response = client.get("/dashboard")

    assert response.status_code == 404
