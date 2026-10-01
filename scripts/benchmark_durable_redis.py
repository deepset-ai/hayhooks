"""
Measure a synthetic streaming run against an isolated, otherwise idle Redis server.

Run with PYTHONPATH pointing at the checkout's src directory to compare revisions.
Uses real runtime and SSE code, without an LLM or HTTP network transport. Counts
RESP payload bytes (not TCP/IP headers), client command batches, and server calls.
"""
# ruff: noqa: PLR2004, T201

import argparse
import asyncio
import inspect
import json
import time
import uuid
from collections import Counter

from fastapi import FastAPI, Request
from fastapi.routing import APIRoute
from haystack.dataclasses import StreamingChunk
from loguru import logger
from pydantic import BaseModel
from redis.asyncio import Redis
from redis.asyncio.connection import Connection, ConnectionPool

from hayhooks.durable.engine import initial_control
from hayhooks.durable.fastapi import create_durable_router
from hayhooks.durable.models import ExecutionKind
from hayhooks.durable.redis import RedisExecutionStore
from hayhooks.durable.runtime import DurableDeployment
from hayhooks.durable.store import StoreConfig


def command_names(data):
    """Decode command names from redis-py's packed RESP2 arrays."""
    offset = 0
    while offset < len(data):
        end = data.index(b"\r\n", offset)
        count = int(data[offset + 1 : end])
        offset = end + 2
        for index in range(count):
            end = data.index(b"\r\n", offset)
            length = int(data[offset + 1 : end])
            offset = end + 2
            if index == 0:
                yield data[offset : offset + length].decode().upper()
            offset += length + 2


class RunInput(BaseModel):
    text: str


async def benchmark(port, duration, submissions):  # noqa: C901, PLR0915
    metrics = Counter()
    commands = Counter()
    measuring = False

    class TrackedConnection(Connection):
        async def send_packed_command(self, command, check_health=True):
            if measuring:
                data = command if isinstance(command, bytes) else b"".join(command)
                commands.update(command_names(data))
                metrics["client_batches"] += 1
            return await super().send_packed_command(command, check_health=check_health)

    async def proxy(reader, writer):
        upstream_reader, upstream_writer = await asyncio.open_connection("127.0.0.1", port)

        async def forward(source, target, label):
            try:
                while data := await source.read(65_536):
                    if measuring:
                        metrics[label] += len(data)
                    target.write(data)
                    await target.drain()
            finally:
                target.close()
                await target.wait_closed()

        await asyncio.gather(
            forward(reader, upstream_writer, "request_bytes"),
            forward(upstream_reader, writer, "response_bytes"),
        )

    server = await asyncio.start_server(proxy, "127.0.0.1", 0)
    proxy_port = server.sockets[0].getsockname()[1]
    admin = Redis(host="127.0.0.1", port=port)
    clients = [
        Redis(connection_pool=ConnectionPool(connection_class=TrackedConnection, port=proxy_port, protocol=2))
        for _ in range(2)
    ]
    for client in clients:
        await client.ping()
    prefix = f"benchmark-{uuid.uuid4().hex}"
    options = (
        {"viewer_client": clients[1]} if "viewer_client" in inspect.signature(RedisExecutionStore).parameters else {}
    )
    store = RedisExecutionStore(clients[0], "bench", key_prefix=prefix, **options)

    async def runner(context, _request):
        started = time.monotonic()
        context.state["checkpoint"] = "c" * 30_000
        for turn in range(6):
            await context.check_cancelled()
            for token in range(250):
                deadline = started + (turn * 250 + token + 1) * duration / 1_500
                await asyncio.sleep(max(0, deadline - time.monotonic()))
                await context.stream_chunk(StreamingChunk(content="token ", index=token))
            if turn < 5:
                await context.check_cancelled()
                await context.report_progress("Tool call finished", metadata={"turn": turn})
            await context.checkpoint()
        return {"text": "r" * 5_000}

    deployment = DurableDeployment("bench", "v1", store, RunInput, runner, kind=ExecutionKind.AGENT)
    app = FastAPI()
    router = create_durable_router(deployment, owner_id_dependency=None)
    app.include_router(router)
    request = Request(
        {
            "type": "http",
            "scheme": "http",
            "server": ("benchmark", 80),
            "headers": [],
            "path": "/",
            "root_path": "",
            "app": app,
            "router": app.router,
        }
    )
    stream = next(
        route.endpoint for route in router.routes if isinstance(route, APIRoute) and route.name.endswith(".stream")
    )
    try:
        await store.initialize()
        submission_burst = None
        if submissions:
            burst = RedisExecutionStore(
                clients[0],
                "bench-burst",
                key_prefix=prefix,
                config=StoreConfig(max_nonterminal_executions=0),
            )
            pool = clients[0].connection_pool
            burst_connections = [await pool.get_connection() for _ in range(submissions)]
            for connection in burst_connections:
                await pool.release(connection)
            measuring = True
            batches = metrics["client_batches"]
            results = await asyncio.gather(
                *(
                    burst.submit(
                        initial_control(
                            run_id=f"burst-{index}",
                            idempotency_digest=f"burst-{index}",
                            idempotency_binding_digest="b",
                            deployment="bench-burst",
                            definition_revision="v1",
                            owner_id=None,
                            kind="pipeline",
                            now_ms=0,
                        ),
                        b"{}",
                    )
                    for index in range(submissions)
                ),
                return_exceptions=True,
            )
            errors = Counter(type(result).__name__ for result in results if isinstance(result, BaseException))
            submission_burst = {
                "submissions": submissions,
                "created": sum(not isinstance(result, BaseException) and result.created for result in results),
                "errors": dict(errors),
                "exchanges_per_submission": (metrics["client_batches"] - batches) / submissions,
            }
            metrics.clear()
            commands.clear()
        before = await admin.info("commandstats")
        measuring = True
        started = time.monotonic()
        await deployment.start()
        submission = await deployment.submit({"text": "i" * 20_000})
        # Match the public snapshot read performed by the submit route.
        await deployment.get(submission.control.run_id)
        response = await stream(submission.control.run_id, request, owner_id=None, last_event_id=None)
        events = Counter()

        async def consume():
            async for frame in response.body_iterator:
                for line in frame.splitlines():
                    if line.startswith("event: "):
                        events[line.removeprefix("event: ")] += 1

        await asyncio.wait_for(consume(), duration + 30)
        await deployment.close()
        elapsed = time.monotonic() - started
        measuring = False
        after = await admin.info("commandstats")
        assert events == {"chunk": 1_500, "completed": 1}, events
        server_commands = {
            key.removeprefix("cmdstat_"): value["calls"] - before.get(key, {}).get("calls", 0)
            for key, value in after.items()
            if key != "cmdstat_info"
        }
        version = await admin.info("server")
        return {
            "redis_version": version["redis_version"],
            "valkey_version": version.get("valkey_version"),
            "duration_seconds": duration,
            "submission_burst": submission_burst,
            "elapsed_seconds": round(elapsed, 3),
            "events": dict(events),
            "client_commands": commands.total(),
            **metrics,
            "resp_bytes": metrics["request_bytes"] + metrics["response_bytes"],
            "server_commands": sum(server_commands.values()),
            "client_command_mix": dict(commands),
            "server_command_mix": server_commands,
        }
    finally:
        await deployment.close()
        # Only remove this run's namespace; never flush the supplied server.
        keys = [key async for key in admin.scan_iter(match=f"{prefix}:*")]
        if keys:
            await admin.delete(*keys)
        for client in clients:
            await client.aclose()
            await client.connection_pool.aclose()
        await admin.aclose()
        server.close()
        await server.wait_closed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=16_479)
    parser.add_argument("--duration", type=float, default=40)
    parser.add_argument("--submissions", type=int, default=100)
    args = parser.parse_args()
    if args.submissions < 0:
        parser.error("--submissions cannot be negative")
    logger.disable("hayhooks")
    print(json.dumps(asyncio.run(benchmark(args.port, args.duration, args.submissions)), indent=2))
