"""Runnable coverage for the durable-only examples."""

import asyncio
import json
from importlib.metadata import version
from pathlib import Path

import pytest
from haystack.dataclasses import ByteStream

from examples.durable_streams import _http_url
from hayhooks.durable.engine import ExecutionStatus, PayloadKind
from hayhooks.durable.haystack import HaystackDurableAdapter
from hayhooks.durable.runtime import DurableDeployment, RuntimeConfig
from hayhooks.durable.store import CHUNK_CURSOR_START, MemoryExecutionStore, StoreConfig, StoredExecution
from hayhooks.server.utils.module_loader import (
    create_pipeline_wrapper_instance,
    load_pipeline_module,
    unload_pipeline_modules,
)

_HAYSTACK_VERSION = tuple(int(part) for part in version("haystack-ai").split(".", maxsplit=2)[:2])
pytestmark = pytest.mark.skipif(_HAYSTACK_VERSION < (3, 1), reason="durable examples require Haystack 3.1+")


def test_stream_example_accepts_only_http_urls() -> None:
    assert _http_url("https://example.com/api/", "jobs") == "https://example.com/api/jobs"
    with pytest.raises(ValueError, match="unsupported URL scheme"):
        _http_url("https://example.com", "file:///etc/passwd")


async def wait_for_execution(deployment: DurableDeployment, run_id: str, expected: ExecutionStatus) -> StoredExecution:
    for _ in range(400):
        stored = await deployment.get(run_id)
        if stored.control.status is expected:
            return stored
        await asyncio.sleep(0.005)
    raise AssertionError(f"execution did not reach {expected.value}")


def test_standalone_fastapi_example_exposes_typed_durable_routes() -> None:
    from examples.durable_fastapi import app as durable_fastapi

    openapi = durable_fastapi.app.openapi()
    paths = openapi["paths"]
    prefix = "/jobs/document-analysis"
    assert "DocumentRequest" in str(paths[f"{prefix}/run-durable"]["post"]["requestBody"])
    assert "Approval" in str(paths[f"{prefix}/executions/{{execution_id}}/resume"]["post"]["requestBody"])
    assert openapi["components"]["securitySchemes"]["HTTPBearer"]["scheme"] == "bearer"


async def test_standalone_fastapi_example_waits_resumes_and_runs_pipeline() -> None:
    from examples.durable_fastapi import app as durable_fastapi

    name = "standalone_fastapi_example"
    adapter = HaystackDurableAdapter(durable_fastapi.build_pipeline())
    deployment = DurableDeployment(
        name,
        durable_fastapi.DEPLOYMENT_REVISION,
        MemoryExecutionStore(name, config=StoreConfig(lease_commit_safety_ms=10)),
        durable_fastapi.DocumentRequest,
        durable_fastapi.run_document_analysis,
        result_model=durable_fastapi.AnalysisResult,
        resume_model=durable_fastapi.Approval,
        adapter=adapter,
        kind=adapter.kind,
        config=RuntimeConfig(poll_interval_seconds=0.005, lease_duration_ms=500),
    )

    try:
        await deployment.start()
        submitted = await deployment.submit(
            {
                "document_id": "document-42",
                "text": "One durable pipeline, one durable result.",
                "require_approval": True,
            }
        )
        run_id = submitted.control.run_id
        waiting = await wait_for_execution(deployment, run_id, ExecutionStatus.WAITING)
        assert json.loads(waiting.payloads[PayloadKind.WAIT])["kind"] == "approval"
        await deployment.resume(run_id, {"approved": True})
        completed = await wait_for_execution(deployment, run_id, ExecutionStatus.COMPLETED)
        assert json.loads(completed.payloads[PayloadKind.RESULT]) == {
            "document_id": "document-42",
            "word_count": 6,
            "unique_terms": 4,
        }
    finally:
        await deployment.close()


async def test_durable_execution_example_retries_resumes_and_skips_checkpointed_work() -> None:
    name = "durable_execution_example"
    module = load_pipeline_module(name, Path("examples/durable_execution/pipelines/durable_execution"))
    wrapper = create_pipeline_wrapper_instance(module)
    adapter = HaystackDurableAdapter(wrapper.pipeline)
    deployment = DurableDeployment(
        name,
        wrapper.durable_revision,
        MemoryExecutionStore(name, config=StoreConfig(lease_commit_safety_ms=10)),
        module.ExecutionRequest,
        wrapper.run_durable_async,
        result_model=module.ExecutionResult,
        resume_model=module.Approval,
        adapter=adapter,
        kind=adapter.kind,
        config=RuntimeConfig(
            worker_concurrency=1,
            poll_interval_seconds=0.005,
            lease_duration_ms=500,
            max_run_attempts=4,
        ),
    )

    try:
        await deployment.start()
        submitted = await deployment.submit(
            {"value": 20, "require_approval": True, "fail_once": True, "retry_delay_seconds": 0}
        )
        run_id = submitted.control.run_id
        await wait_for_execution(deployment, run_id, ExecutionStatus.WAITING)
        await deployment.resume(run_id, {"approved": True})
        completed = await wait_for_execution(deployment, run_id, ExecutionStatus.COMPLETED)
        assert completed.control.run_attempt == 3
        assert json.loads(completed.payloads[PayloadKind.RESULT]) == {"result": 41}
        assert wrapper.pipeline.get_component("prepare").calls == 1
    finally:
        await deployment.close()
        unload_pipeline_modules(name)


async def test_durable_website_example_keeps_concurrent_streams_isolated(monkeypatch) -> None:
    name = "durable_website_example"
    module = load_pipeline_module(
        name,
        Path("examples/durable_chat_with_website/pipelines/durable_chat_with_website"),
    )
    wrapper = create_pipeline_wrapper_instance(module)

    def fetch(urls: list[str]) -> dict[str, list[ByteStream]]:
        return {
            "streams": [
                ByteStream.from_string(
                    f"<html><body>{url} contains {url.split('//')[1].split('.')[0]} facts.</body></html>"
                )
                for url in urls
            ]
        }

    monkeypatch.setattr(wrapper.pipeline.get_component("fetch"), "run", fetch)
    adapter = HaystackDurableAdapter(wrapper.pipeline)
    deployment = DurableDeployment(
        name,
        wrapper.durable_revision,
        MemoryExecutionStore(name, config=StoreConfig(lease_commit_safety_ms=10)),
        module.WebsiteRequest,
        wrapper.run_durable_async,
        result_model=module.WebsiteAnswer,
        adapter=adapter,
        kind=adapter.kind,
        config=RuntimeConfig(worker_concurrency=2, poll_interval_seconds=0.005, lease_duration_ms=500),
    )

    try:
        await deployment.start()
        alpha, beta = await asyncio.gather(
            deployment.submit({"urls": ["https://alpha.example"], "question": "What alpha facts?"}),
            deployment.submit({"urls": ["https://beta.example"], "question": "What beta facts?"}),
        )
        completed = await asyncio.gather(
            wait_for_execution(deployment, alpha.control.run_id, ExecutionStatus.COMPLETED),
            wait_for_execution(deployment, beta.control.run_id, ExecutionStatus.COMPLETED),
        )
        streams = []
        for stored in completed:
            chunks = await deployment.store.read_chunks(stored.control.run_id, CHUNK_CURSOR_START)
            streams.append("".join(json.loads(chunk.data)["text"] for chunk in chunks))
        assert "alpha" in streams[0] and "beta" not in streams[0]
        assert "beta" in streams[1] and "alpha" not in streams[1]
    finally:
        await deployment.close()
        unload_pipeline_modules(name)
