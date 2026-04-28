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
