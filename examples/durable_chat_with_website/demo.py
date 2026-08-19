"""Submit a durable chat-with-website question and follow its token stream."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from collections.abc import Iterator
from typing import Any
from urllib.parse import urljoin

import httpx
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel

_PAUSE_SECONDS = 2
# Detaching mid-answer is the point of the demo: the execution keeps running and
# the chunk log keeps every token until this client reattaches.
_DETACH_AFTER_CHUNKS = 8


def sse_events(client: httpx.Client, url: str, cursor: str | None) -> Iterator[dict[str, str]]:
    """Yield one dict of SSE fields per event, resuming from *cursor* when given."""
    headers = {"Last-Event-ID": cursor} if cursor else {}
    with client.stream("GET", url, headers=headers, timeout=None) as response:
        response.raise_for_status()
        fields: dict[str, str] = {}
        for line in response.iter_lines():
            if line.startswith(":"):
                continue  # a heartbeat comment, sent while the execution is quiet
            if line:
                name, _, value = line.partition(": ")
                fields[name] = value
            elif fields:
                yield fields
                fields = {}


class StreamDroppedError(Exception):
    """The server ended the stream with an error event instead of a terminal one."""


def follow(
    console: Console, client: httpx.Client, url: str, cursor: str | None, *, detach_after: int | None
) -> tuple[str | None, dict[str, Any] | None]:
    """Print chunks from one connection; return the cursor and the terminal body."""
    attempt: int | None = None
    for printed, event in enumerate(sse_events(client, url, cursor), start=1):
        cursor = event.get("id", cursor)
        if event["event"] == "error":
            raise StreamDroppedError(json.loads(event["data"]).get("detail", "execution stream interrupted"))
        if event["event"] != "chunk":
            console.print()
            return cursor, json.loads(event["data"])
        chunk = json.loads(event["data"])
        if attempt is None:
            attempt = chunk["attempt"]
        elif chunk["attempt"] != attempt:
            # A retried attempt replays from its checkpoint, so tokens printed
            # before the crash arrive again. Printing cannot retract them; a
            # client with a rewritable buffer would reset it here instead.
            attempt = chunk["attempt"]
            console.print()
            console.print(f"[dim]Attempt {attempt}: the execution resumed from its checkpoint, so tokens repeat.[/dim]")
        console.print(chunk["payload"]["content"] or "", end="")
        if printed == detach_after:
            return cursor, None
    return cursor, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:1416")
    parser.add_argument("--question", default="What does Haystack do, and how does Redis fit in?")
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")
    console = Console()

    with httpx.Client(timeout=10) as client:
        submitted = client.post(
            f"{base_url}/chat_with_website/run-durable",
            headers={"Idempotency-Key": f"ask-{uuid.uuid4().hex[:12]}"},
            json={"question": args.question},
        )
        console.print(Panel(JSON.from_data(submitted.json()), title=f"{submitted.status_code} submitted"))
        if submitted.is_error:
            return 1
        stream_url = urljoin(f"{base_url}/", submitted.json()["links"]["stream"])

        console.print(Panel.fit(f"[bold cyan]GET[/] {stream_url}", title="Streaming", border_style="cyan"))
        cursor, terminal, detached = None, None, False
        while terminal is None:
            try:
                cursor, terminal = follow(
                    console, client, stream_url, cursor, detach_after=None if detached else _DETACH_AFTER_CHUNKS
                )
            except (httpx.HTTPError, StreamDroppedError) as error:
                console.print()
                console.print(Panel(str(error), title="Stream interrupted", border_style="red"))
            if terminal is None:
                detached = True
                console.print()
                console.print(
                    Panel(
                        f"Reattaching from Last-Event-ID {cursor}. Nothing between here and there is lost.",
                        title="Detached",
                        border_style="yellow",
                    )
                )
                time.sleep(_PAUSE_SECONDS)

        console.print(Panel(JSON.from_data(terminal), title=f"Terminal event: {terminal['status']}"))
        return 0 if terminal["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
