import pytest
from fastapi import FastAPI

from hayhooks.server.app import run_app
from hayhooks.settings import settings


@pytest.fixture(autouse=True)
def default_settings():
    settings.host = "127.0.0.1"
    settings.port = 8080


def test_run_app_defaults_from_settings(monkeypatch):
    """run_app(app) with no host/port should read from settings."""
    import uvicorn

    calls: list[tuple] = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(uvicorn, "run", fake_run)

    app = FastAPI()
    run_app(app)

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] is app
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 8080


def test_run_app_explicit_overrides(monkeypatch):
    """Explicit host/port should override settings."""
    import uvicorn

    calls: list[tuple] = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(uvicorn, "run", fake_run)

    app = FastAPI()
    run_app(app, host="0.0.0.0", port=9999)

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] is app
    assert kwargs["host"] == "0.0.0.0"
    assert kwargs["port"] == 9999


def test_run_app_passes_app_object(monkeypatch):
    """run_app must pass the FastAPI instance directly, not a string import path."""
    import uvicorn

    calls: list[tuple] = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(uvicorn, "run", fake_run)

    app = FastAPI()
    run_app(app)

    args, _kwargs = calls[0]
    assert isinstance(args[0], FastAPI)
    assert not isinstance(args[0], str)
