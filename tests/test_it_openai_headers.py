"""Request headers reach pipeline wrappers that declare a `headers` parameter, and only those."""

import shutil
from pathlib import Path

import pytest
from fastapi_openai_compat import ChatRequest

from hayhooks.server.pipelines import registry
from hayhooks.server.routers.openai import _build_call_kwargs, _method_accepts_kwarg
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper
from hayhooks.settings import settings

# Headers can only reach a wrapper if the installed fastapi-openai-compat forwards them to the
# run_completion callback. Skip the end-to-end checks on versions that predate that.
try:
    from fastapi_openai_compat._shared import callable_accepts_kwarg as _compat_opt_in

    _COMPAT_FORWARDS_HEADERS = callable(_compat_opt_in)
except ImportError:  # pragma: no cover - depends on the installed version
    _COMPAT_FORWARDS_HEADERS = False

requires_header_forwarding = pytest.mark.skipif(
    not _COMPAT_FORWARDS_HEADERS,
    reason="installed fastapi-openai-compat does not forward request headers to run_completion",
)

TEST_FILES_DIR_WITH_HEADERS = Path(__file__).parent / "test_files/files/chat_with_headers"
PIPELINE_FILES_WITH_HEADERS = {
    "pipeline_wrapper.py": (TEST_FILES_DIR_WITH_HEADERS / "pipeline_wrapper.py").read_text(),
}

TEST_FILES_DIR_WITHOUT_HEADERS = Path(__file__).parent / "test_files/files/chat_without_headers"
PIPELINE_FILES_WITHOUT_HEADERS = {
    "pipeline_wrapper.py": (TEST_FILES_DIR_WITHOUT_HEADERS / "pipeline_wrapper.py").read_text(),
}


@pytest.fixture(autouse=True)
def clear_registry():
    registry.clear()
    if Path(settings.pipelines_dir).exists():
        shutil.rmtree(settings.pipelines_dir)
    yield


def _chat(client, model: str, headers: dict[str, str] | None = None):
    request = ChatRequest(stream=False, model=model, messages=[{"role": "user", "content": "who am I?"}])
    return client.post("/chat/completions", json=request.model_dump(), headers=headers)


@requires_header_forwarding
def test_headers_forwarded_when_wrapper_declares_them(client, deploy_files):
    assert deploy_files(client, "with_headers", PIPELINE_FILES_WITH_HEADERS).status_code == 200

    response = _chat(client, "with_headers", {"Authorization": "Bearer alice-token"})

    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"]
    assert content == "authorization=Bearer alice-token"


def test_wrapper_without_headers_parameter_is_unaffected(client, deploy_files):
    """The pre-existing (model, messages, body) signature must keep working untouched."""
    assert deploy_files(client, "without_headers", PIPELINE_FILES_WITHOUT_HEADERS).status_code == 200

    response = _chat(client, "without_headers", {"Authorization": "Bearer bob-token"})

    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"]
    assert content == "no headers parameter declared"


@requires_header_forwarding
def test_wrapper_declaring_headers_without_request_headers(client, deploy_files):
    """A request always carries some headers, so the wrapper still gets a dict, just without ours."""
    assert deploy_files(client, "with_headers", PIPELINE_FILES_WITH_HEADERS).status_code == 200

    response = _chat(client, "with_headers")

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "authorization=missing"


# --- the opt-in helpers ----------------------------------------------------------------------


class _WithHeaders(BasePipelineWrapper):
    def setup(self) -> None: ...

    def run_chat_completion(self, model: str, messages: list[dict], body: dict, headers: dict[str, str]) -> str:
        return "ok"


class _WithKwargs(BasePipelineWrapper):
    def setup(self) -> None: ...

    def run_chat_completion(self, model: str, messages: list[dict], body: dict, **kwargs) -> str:
        return "ok"


class _WithoutHeaders(BasePipelineWrapper):
    def setup(self) -> None: ...

    def run_chat_completion(self, model: str, messages: list[dict], body: dict) -> str:
        return "ok"


def test_method_accepts_kwarg_detects_explicit_parameter():
    assert _method_accepts_kwarg(_WithHeaders().run_chat_completion, "headers") is True


def test_method_accepts_kwarg_detects_var_keyword():
    assert _method_accepts_kwarg(_WithKwargs().run_chat_completion, "headers") is True


def test_method_accepts_kwarg_rejects_missing_parameter():
    assert _method_accepts_kwarg(_WithoutHeaders().run_chat_completion, "headers") is False


def test_method_accepts_kwarg_on_unintrospectable_callable():
    """Callables without a signature must not raise; they simply do not opt in."""
    assert _method_accepts_kwarg(print, "headers") is False


def test_build_call_kwargs_includes_headers_for_opted_in_wrapper():
    call_kwargs = _build_call_kwargs(
        _WithHeaders(), "run_chat_completion", {"messages": []}, {"stream": False}, {"authorization": "Bearer x"}
    )

    assert call_kwargs == {"messages": [], "body": {"stream": False}, "headers": {"authorization": "Bearer x"}}


def test_build_call_kwargs_omits_headers_for_other_wrappers():
    call_kwargs = _build_call_kwargs(
        _WithoutHeaders(), "run_chat_completion", {"messages": []}, {"stream": False}, {"authorization": "Bearer x"}
    )

    assert call_kwargs == {"messages": [], "body": {"stream": False}}


def test_build_call_kwargs_omits_headers_when_none_available():
    """Non-HTTP entry points pass headers=None, which must not reach the wrapper."""
    call_kwargs = _build_call_kwargs(_WithHeaders(), "run_chat_completion", {"messages": []}, {}, None)

    assert call_kwargs == {"messages": [], "body": {}}
