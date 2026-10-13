import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from hayhooks.server.app import create_app
from hayhooks.server.routers.traces import require_tracing_access
from hayhooks.server.utils.live_trace_buffer import clear_live_traces, record_live_span_finish, record_live_span_start
from hayhooks.settings import settings
from tests.pipeline_sources import DURABLE_WRAPPER, HAYSTACK_V3, write_tree


def setup_function():
    clear_live_traces()


async def test_snapshot_through_app_lifespan_reads_worker_thread_spans(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "pipelines_dir", str(tmp_path))
    monkeypatch.setattr(settings, "tracing_enabled", True)
    app = create_app()
    app.dependency_overrides[require_tracing_access] = lambda: None
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
    ):
        assert (await client.get("/traces")).json()["traces"] == []

        def record():
            record_live_span_start(
                trace_id="t1",
                span_id="s1",
                parent_span_id=None,
                operation_name="hayhooks.pipeline.run",
                start_time_ms=1000,
            )
            record_live_span_finish(trace_id="t1", span_id="s1", duration_ms=5)

        await asyncio.to_thread(record)
        snapshot = (await client.get("/traces")).json()
        assert snapshot["traces"][0]["trace_id"] == "t1"
        assert snapshot["traces"][0]["root_span"]["running"] is False


@pytest.mark.skipif(not HAYSTACK_V3, reason="durable adapters require Haystack 3.1+")
async def test_durable_worker_attempt_reaches_trace_snapshot(durable_pipelines_dir, monkeypatch):
    monkeypatch.setattr(settings, "durable_store", "memory")
    monkeypatch.setattr(settings, "tracing_enabled", True)
    write_tree(durable_pipelines_dir, {"jobs/pipeline_wrapper.py": DURABLE_WRAPPER})
    app = create_app()
    app.dependency_overrides[require_tracing_access] = lambda: None
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
    ):
        submitted = await client.post("/jobs/run-durable", json={"value": 1, "wait": True})
        assert submitted.status_code == 202
        execution_id = submitted.json()["execution_id"]
        for _ in range(100):
            traces = (await client.get("/traces")).json()["traces"]
            attempt = next(
                (trace["root_span"] for trace in traces if trace["root_span"]["name"] == "hayhooks.durable.attempt"),
                None,
            )
            if attempt is not None and not attempt["running"]:
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("durable attempt did not finish in the trace snapshot")

    assert {
        "hayhooks.pipeline.name": "jobs",
        "hayhooks.durable.execution_id": execution_id,
        "hayhooks.durable.attempt": "1",
        "hayhooks.durable.kind": "pipeline",
        "hayhooks.success": "true",
        "hayhooks.checkpoint": "true",
    }.items() <= {tag["key"]: tag["value"] for tag in attempt["tags"]}.items()
