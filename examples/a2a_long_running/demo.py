"""Run the durable A2A example's submit, approval, and completion flow."""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

import httpx
from a2a.client import A2AClientError, Client, ClientConfig, create_client
from a2a.helpers import new_text_message
from a2a.types import GetTaskRequest, Role, SendMessageConfiguration, SendMessageRequest, Task, TaskState
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

TERMINAL_STATES = {
    TaskState.TASK_STATE_COMPLETED,
    TaskState.TASK_STATE_FAILED,
    TaskState.TASK_STATE_CANCELED,
    TaskState.TASK_STATE_REJECTED,
}


class A2ADemoError(RuntimeError):
    """A readable failure raised by the demo client."""


async def send_message(client: Client, text: str, *, task_id: str = "") -> Task:
    request = SendMessageRequest(
        message=new_text_message(text, task_id=task_id or None, role=Role.ROLE_USER),
        configuration=SendMessageConfiguration(return_immediately=True),
    )
    async for response in client.send_message(request):
        if response.HasField("task"):
            return response.task
    msg = "A2A response did not contain a task"
    raise A2ADemoError(msg)


async def wait_for_state(  # noqa: PLR0913 - polling controls are explicit CLI inputs
    client: Client,
    task_id: str,
    expected: set[int],
    *,
    deadline: float,
    poll_interval: float,
    console: Console,
) -> Task:
    previous_state = None
    while time.monotonic() < deadline:
        task = await client.get_task(GetTaskRequest(id=task_id))
        state = task.status.state
        if state != previous_state:
            console.print(f"  [cyan]A2A state[/cyan] → [bold]{TaskState.Name(state)}[/bold]")
            previous_state = state
        if state in expected:
            return task
        if state in TERMINAL_STATES:
            names = ", ".join(TaskState.Name(item) for item in sorted(expected))
            msg = f"Task became {TaskState.Name(state)} before reaching {names}"
            raise A2ADemoError(msg)
        await asyncio.sleep(poll_interval)
    names = ", ".join(TaskState.Name(item) for item in sorted(expected))
    msg = f"Timed out waiting for {names}"
    raise A2ADemoError(msg)


def print_summary(task: Task, console: Console) -> None:
    table = Table(title="Durable A2A result", show_header=False)
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("Task", task.id)
    table.add_row("State", TaskState.Name(task.status.state))
    table.add_row("History messages", str(len(task.history)))
    table.add_row("Artifacts", ", ".join(artifact.name or "unnamed" for artifact in task.artifacts))
    console.print(table)

    result = next((artifact for artifact in task.artifacts if artifact.name == "durable-result"), None)
    if result:
        text = "\n".join(part.text for part in result.parts if part.WhichOneof("content") == "text")
        if text:
            console.print(Panel(text, title="Agent result", border_style="green"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:1418/long_running_agent/", help="A2A agent base URL")
    parser.add_argument("--document-id", default="hayhooks-guide")
    parser.add_argument("--content", default="Hayhooks durable A2A work survives restarts.")
    parser.add_argument("--timeout", type=float, default=120, help="Overall timeout in seconds")
    parser.add_argument("--poll-interval", type=float, default=0.5)
    return parser.parse_args()


async def run(args: argparse.Namespace, console: Console) -> None:
    if args.timeout <= 0 or args.poll_interval <= 0:
        msg = "timeout and poll interval must be positive"
        raise A2ADemoError(msg)

    deadline = time.monotonic() + args.timeout
    async with httpx.AsyncClient(timeout=15) as http:
        client = await create_client(
            args.url,
            ClientConfig(streaming=False, polling=True, httpx_client=http),
        )
        console.print(Panel.fit("Submit → input required → approve → complete", title="Durable A2A demo"))
        task = await send_message(
            client,
            f"Prepare this document for indexing. document_id: {args.document_id}. content: {args.content}",
        )
        console.print(f"[green]✓[/green] Submitted task [bold]{task.id}[/bold]")
        await wait_for_state(
            client,
            task.id,
            {TaskState.TASK_STATE_INPUT_REQUIRED},
            deadline=deadline,
            poll_interval=args.poll_interval,
            console=console,
        )
        console.print("[green]✓[/green] Approval requested")
        await send_message(client, "Approved; proceed.", task_id=task.id)
        console.print("[green]✓[/green] Approval sent")
        print_summary(
            await wait_for_state(
                client,
                task.id,
                {TaskState.TASK_STATE_COMPLETED},
                deadline=deadline,
                poll_interval=args.poll_interval,
                console=console,
            ),
            console,
        )


def main() -> int:
    console = Console()
    try:
        asyncio.run(run(parse_args(), console))
    except (A2ADemoError, A2AClientError, httpx.HTTPError, ValueError) as error:
        console.print(f"[bold red]Demo failed:[/bold red] {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
