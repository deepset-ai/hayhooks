"""
Integration tests for the API key authentication example.

These tests verify the auth layer only — they don't require OpenAI keys
or any external service.  Run with:

    hatch run test:pytest examples/programmatic/api_key_auth/test_api_key_auth.py -v
"""

import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest  # type: ignore[import-not-found]
from fastapi import status
from fastapi.testclient import TestClient

from hayhooks.server.pipelines.registry import registry

TEST_API_KEY_1 = "test-secret-key-1"
TEST_API_KEY_2 = "test-secret-key-2"
API_KEYS_ENV_VAR = "HAYHOOKS_API_KEYS"
API_KEY_ENV_VAR = "HAYHOOKS_API_KEY"
APP_PATH = Path(__file__).with_name("app.py")
AUTH_PATH = Path(__file__).with_name("auth.py")
PIPELINE_WRAPPER_PATH = APP_PATH.parent / "pipelines" / "weather_agent" / "pipeline_wrapper.py"


@pytest.fixture(autouse=True)
def _set_api_key(monkeypatch):
    monkeypatch.setenv(API_KEYS_ENV_VAR, f"{TEST_API_KEY_1},{TEST_API_KEY_2}")
    monkeypatch.delenv(API_KEY_ENV_VAR, raising=False)


@pytest.fixture(autouse=True)
def _clear_registry():
    registry.clear()
    yield
    registry.clear()


@pytest.fixture()
def app_module():
    return runpy.run_path(str(APP_PATH))


@pytest.fixture()
def auth_app(app_module):
    return app_module["create_authenticated_app"]()


@pytest.fixture()
def auth_module():
    return runpy.run_path(str(AUTH_PATH))


@pytest.fixture()
def wrapper_module():
    return runpy.run_path(str(PIPELINE_WRAPPER_PATH))


@pytest.fixture()
def client(auth_app):
    with TestClient(auth_app) as c:
        yield c


def _auth_header(key: str = TEST_API_KEY_1) -> dict:
    return {"X-API-Key": key}


# -- Missing key --------------------------------------------------------


class TestMissingKey:
    def test_status_returns_403(self, client):
        resp = client.get("/status")
        assert resp.status_code == status.HTTP_403_FORBIDDEN
        assert resp.json()["detail"] == "Not authenticated"

    def test_deploy_yaml_returns_403(self, client):
        resp = client.post("/deploy-yaml", json={"name": "p", "source_code": ""})
        assert resp.status_code == status.HTTP_403_FORBIDDEN


class TestWrongKey:
    def test_status_returns_403(self, client):
        resp = client.get("/status", headers=_auth_header("wrong-key"))
        assert resp.status_code == status.HTTP_403_FORBIDDEN
        assert resp.json()["detail"] == "Invalid API key"

    def test_deploy_yaml_returns_403(self, client):
        resp = client.post(
            "/deploy-yaml",
            json={"name": "p", "source_code": ""},
            headers=_auth_header("wrong-key"),
        )
        assert resp.status_code == status.HTTP_403_FORBIDDEN
        assert resp.json()["detail"] == "Invalid API key"


class TestValidKey:
    @pytest.mark.parametrize("key", [TEST_API_KEY_1, TEST_API_KEY_2])
    def test_status_returns_200_for_any_valid_key(self, client, key):
        resp = client.get("/status", headers=_auth_header(key))
        assert resp.status_code == status.HTTP_200_OK
        assert "pipelines" in resp.json()

    def test_docs_returns_200_without_key(self, client):
        resp = client.get("/docs")
        assert resp.status_code == status.HTTP_200_OK

    def test_openapi_schema_returns_200_without_key(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == status.HTTP_200_OK


class TestSwaggerAuthorize:
    def test_schema_includes_api_key_security_scheme(self, client):
        resp = client.get("/openapi.json")
        schema = resp.json()

        security_schemes = schema.get("components", {}).get("securitySchemes", {})
        assert "APIKeyHeader" in security_schemes
        assert security_schemes["APIKeyHeader"]["type"] == "apiKey"
        assert security_schemes["APIKeyHeader"]["in"] == "header"
        assert security_schemes["APIKeyHeader"]["name"] == "X-API-Key"

    def test_schema_has_global_security_requirement(self, client):
        resp = client.get("/openapi.json")
        schema = resp.json()
        assert {"APIKeyHeader": []} in schema.get("security", [])

    def test_schema_merge_preserves_existing_security(self, auth_module):
        add_openapi_security = auth_module["_add_openapi_security"]

        class FakeApp:
            def __init__(self):
                self.openapi_schema = None

            def openapi(self):
                return {
                    "openapi": "3.1.0",
                    "info": {"title": "fake", "version": "1.0.0"},
                    "paths": {},
                    "components": {
                        "securitySchemes": {
                            "ExistingBearer": {"type": "http", "scheme": "bearer"},
                        }
                    },
                    "security": [{"ExistingBearer": []}],
                }

        app = FakeApp()
        add_openapi_security(app)
        schema = app.openapi()

        schemes = schema["components"]["securitySchemes"]
        assert "ExistingBearer" in schemes
        assert "APIKeyHeader" in schemes
        assert {"ExistingBearer": []} in schema["security"]
        assert {"APIKeyHeader": []} in schema["security"]


class TestPublicPathMatching:
    def test_allows_docs_paths_with_and_without_trailing_slash(self, auth_module):
        is_public = auth_module["_is_public_path"]
        assert is_public("/docs")
        assert is_public("/docs/")
        assert is_public("/redoc")
        assert is_public("/redoc/")
        assert is_public("/openapi.json")

    def test_allows_docs_paths_under_root_path_prefix(self, auth_module):
        is_public = auth_module["_is_public_path"]
        assert is_public("/api/docs", "/api")
        assert is_public("/api/docs/", "/api")
        assert is_public("/api/redoc", "/api")
        assert is_public("/api/openapi.json", "/api")

    def test_does_not_allow_non_public_paths(self, auth_module):
        is_public = auth_module["_is_public_path"]
        assert not is_public("/status")
        assert not is_public("/api/status", "/api")
        assert not is_public("/docs-private")


class TestApiKeyLoading:
    def test_loads_comma_separated_keys(self, auth_module, monkeypatch):
        load_api_keys = auth_module["_load_api_keys"]
        monkeypatch.setenv(API_KEYS_ENV_VAR, " alpha , beta,alpha ,  ")
        monkeypatch.delenv(API_KEY_ENV_VAR, raising=False)
        assert load_api_keys() == {"alpha", "beta"}

    def test_loads_single_key_fallback(self, auth_module, monkeypatch):
        load_api_keys = auth_module["_load_api_keys"]
        monkeypatch.delenv(API_KEYS_ENV_VAR, raising=False)
        monkeypatch.setenv(API_KEY_ENV_VAR, "single")
        assert load_api_keys() == {"single"}


class TestWeatherAgentRunApi:
    @pytest.mark.asyncio
    async def test_run_api_uses_messages_kwarg_and_last_message_text(self, wrapper_module):
        PipelineWrapper = wrapper_module["PipelineWrapper"]

        class FakeAgent:
            async def run_async(self, *, messages):
                self.messages = messages
                return {"last_message": SimpleNamespace(text="ok")}

        wrapper = PipelineWrapper()
        wrapper.agent = FakeAgent()

        reply = await wrapper.run_api_async("weather in Berlin?")
        assert reply == "ok"
        assert len(wrapper.agent.messages) == 1
