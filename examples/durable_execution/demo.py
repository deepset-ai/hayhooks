"""Submit and follow the durable Pipeline recovery demonstration."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from collections.abc import Callable
from typing import Any
from urllib.parse import urljoin

import httpx
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel

_PAUSE_SECONDS = 5
_TERMINAL_STATUSES = {"completed", "failed", "canceled"}


def request(console: Console, client: httpx.Client, method: str, url: str, **kwargs: Any) -> httpx.Response | None:
    console.print(Panel.fit(f"[bold cyan]{method}[/] {url}", title="Request", border_style="cyan"))
    if payload := kwargs.get("json"):
        console.print(JSON.from_data(payload))
    try:
        response = client.request(method, url, **kwargs)
    except httpx.HTTPError as error:
        console.print(Panel(str(error), title="Connection failed", border_style="red"))
        return None
    try:
        body = JSON.from_data(response.json())
    except json.JSONDecodeError:
        body = response.text
    console.print(Panel(body, title=f"{response.status_code} {response.url}", border_style="green"))
    return response


def pause(console: Console) -> None:
    console.print(f"[dim]Waiting {_PAUSE_SECONDS} seconds before the next request...[/]")
    time.sleep(_PAUSE_SECONDS)


def poll_until(
    console: Console,
    client: httpx.Client,
    execution_url: str,
    matches: Callable[[dict[str, Any]], bool],
) -> dict[str, Any] | None:
    while True:
        response = request(console, client, "GET", execution_url)
        if response is not None:
            body = response.json()
            if matches(body):
                return body
            if body["status"] in _TERMINAL_STATUSES:
                return None
        pause(console)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:1416")
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")
    execution_id = f"prepare-hayhooks-guide-{uuid.uuid4().hex[:12]}"
    console = Console()

    with httpx.Client(timeout=2) as client:
        while (
            submitted := request(
                console,
                client,
                "POST",
                f"{base_url}/durable_job/run-durable",
                headers={"Idempotency-Key": execution_id},
                json={
                    "documents": [
                        {
                            "document_id": "hayhooks-guide",
                            "content": "Hayhooks durable Pipelines survive restarts.",
                        }
                    ],
                    "fail_first_attempt": True,
                    "require_approval": True,
                    "demo_delay_seconds": 30,
                },
            )
        ) is None:
            pause(console)
        if submitted.is_error:
            return 1
        links = submitted.json()["links"]
        execution_url = urljoin(f"{base_url}/", links["self"])
        resume_url = urljoin(f"{base_url}/", links["resume"])

        pause(console)
        if poll_until(console, client, execution_url, lambda body: body["status"] == "waiting") is None:
            return 1

        pause(console)
        resumed = request(console, client, "POST", resume_url, json={"approved": True})
        if resumed is None or resumed.is_error:
            return 1

        pause(console)
        checkpoint = poll_until(
            console,
            client,
            execution_url,
            lambda body: any(event["kind"] == "demo_delay" for event in body["progress"]),
        )
        if checkpoint is None:
            return 1
        console.print(
            Panel(
                "Kill Hayhooks now, then restart it. This client will keep polling.",
                title="The clean checkpoint is persisted",
                border_style="yellow",
            )
        )

        pause(console)
        completed = poll_until(console, client, execution_url, lambda body: body["status"] in _TERMINAL_STATUSES)
        return 0 if completed is not None and completed["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
