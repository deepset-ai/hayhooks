"""
One-command A2A demo.

    python run_demo.py                    run the full scripted demo, then wait for Ctrl-C
    python run_demo.py "your question"    same, with a custom question
    python run_demo.py --serve            start everything and keep it running,
                                          so you can chat, curl, or attach the
                                          a2a-inspector (Ctrl-C to stop)
    python run_demo.py --step-wait 0      skip the visible waits between steps

Requires: pip install -r requirements.txt, and OPENAI_API_KEY set.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text

DEMO_DIR = Path(__file__).parent
A2A_PORT = int(os.getenv("HAYHOOKS_A2A_PORT", "1418"))
DEFAULT_STEP_WAIT_SECONDS = int(os.getenv("A2A_DEMO_STEP_WAIT", "5"))
HTTP_OK = 200
console = Console()


def serve_banner() -> Panel:
    return Panel(
        Group(
            Text("Everything is up. Try these from another terminal.", style="bold green"),
            "",
            Text("Agent discovery", style="bold cyan"),
            Text(f"  curl -s http://localhost:{A2A_PORT}/weather_agent/.well-known/agent-card.json | jq"),
            Text(f"  curl -s http://localhost:{A2A_PORT}/trip_planner_agent/.well-known/agent-card.json | jq"),
            "",
            Text("Talk to the agents", style="bold cyan"),
            Text("  python chat.py                    trip planner, delegates weather over A2A"),
            Text("  python chat.py weather_agent      weather agent directly"),
            Text('  python client.py "I\'m in Lisbon today, what should I do?"'),
            "",
            Text("Official a2a-inspector", style="bold cyan"),
            Text(f"  Connect to http://localhost:{A2A_PORT}/weather_agent"),
            Text("  Project: https://github.com/a2aproject/a2a-inspector"),
            "",
            Text("Press Ctrl-C to stop all servers.", style="dim"),
        ),
        title="Hayhooks A2A demo",
        border_style="cyan",
    )


def hayhooks_command() -> list[str]:
    return ["hayhooks"]


def wait_for(url: str, name: str, *, require_ok: bool = False, timeout: int = 30) -> None:
    """
    Wait until `url` answers.

    MCP streamable-http endpoints answer 406 to a plain GET, so by default any
    HTTP response counts as "up"; pass require_ok=True to require a 200.
    """
    with console.status(f"[bold]Waiting for {name}[/bold] at {url}", spinner="dots"):
        for _ in range(timeout):
            try:
                response = httpx.get(url, timeout=2)
                if not require_ok or response.status_code == HTTP_OK:
                    console.print(f"[green]OK[/green] {name} is up at [bold]{url}[/bold]")
                    return
            except httpx.HTTPError:
                pass
            time.sleep(1)
    msg = f"{name} did not come up at {url}"
    raise RuntimeError(msg)


def start(command: list[str]) -> subprocess.Popen:
    return subprocess.Popen(command, cwd=DEMO_DIR)  # noqa: S603


def pause_between_steps(seconds: int, next_step: str) -> None:
    if seconds <= 0:
        return
    console.print()
    for remaining in range(seconds, 0, -1):
        console.print(f"[dim]{next_step} in {remaining}...[/dim]")
        time.sleep(1)


def wait_for_shutdown() -> None:
    console.print()
    console.print(
        Panel(
            "The scripted run is complete, and all servers are still running.\n"
            "Press Ctrl-C when you want to stop the demo.",
            title="Demo running",
            border_style="green",
        )
    )
    while True:
        time.sleep(3600)


def print_flow_summary(question: str) -> None:
    console.print()
    console.print(
        Panel(
            Group(
                Text("What just happened", style="bold green"),
                "",
                Text("1. The demo started two private MCP tool servers on ports 8001 and 8002."),
                Text(f"2. Hayhooks started one A2A server on port {A2A_PORT} and exposed two agents:"),
                Text("   - weather_agent: answers weather questions with the get_weather MCP tool."),
                Text(
                    "   - trip_planner_agent: plans the day, delegates weather to weather_agent, "
                    "then calls suggest_activities."
                ),
                Text("3. The client discovered both agents by reading their A2A agent cards."),
                Text(f"4. The client asked trip_planner_agent: {question!r}"),
                Text("5. trip_planner_agent called ask_weather_agent, which discovered weather_agent over A2A."),
                Text("6. weather_agent called get_weather on its MCP server, which fetched Open-Meteo data."),
                Text("7. trip_planner_agent called suggest_activities on its own MCP server."),
                Text("8. trip_planner_agent streamed the final plan back to the client."),
                "",
                Text("The servers are still running so you can inspect, curl, or chat with the agents.", style="dim"),
            ),
            title="A2A + MCP flow",
            border_style="green",
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the A2A multi-agent demo.")
    parser.add_argument("--serve", action="store_true", help="keep the servers running instead of exiting")
    parser.add_argument(
        "--step-wait",
        type=int,
        default=DEFAULT_STEP_WAIT_SECONDS,
        help="seconds to wait between visible demo steps (default: %(default)s)",
    )
    parser.add_argument("question", nargs="*", help="question for the trip planner (default: plan a day in Berlin)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    step_wait = max(args.step_wait, 0)

    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set")

    # Route SIGINT/SIGTERM through KeyboardInterrupt so the finally block below
    # always shuts the child servers down (a plain SIGTERM would otherwise kill
    # this process without cleanup, leaving the servers orphaned)
    def _interrupt(signum, frame):  # noqa: ARG001
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _interrupt)
    signal.signal(signal.SIGTERM, _interrupt)

    processes: list[subprocess.Popen] = []
    try:
        console.print(
            Panel.fit(
                "[bold]A2A multi-agent demo[/bold]\n"
                "MCP tool servers -> Hayhooks A2A server -> streamed trip planner response",
                border_style="cyan",
            )
        )

        console.rule("[bold cyan]1. MCP tool servers")
        console.print("Starting weather and activities MCP servers ...")
        processes.append(start([sys.executable, "mcp_servers/weather_server.py"]))
        processes.append(start([sys.executable, "mcp_servers/activities_server.py"]))
        # The agents connect to their MCP servers eagerly at deploy time, so
        # the servers must be reachable before the A2A server starts
        wait_for("http://localhost:8001/mcp", "weather MCP server")
        wait_for("http://localhost:8002/mcp", "activities MCP server")

        pause_between_steps(step_wait, "Starting Hayhooks A2A server")
        console.rule("[bold cyan]2. Hayhooks A2A server")
        console.print(f"Starting Hayhooks A2A server on port [bold]{A2A_PORT}[/bold] ...")
        processes.append(
            start([*hayhooks_command(), "a2a", "run", "--pipelines-dir", "pipelines", "--port", str(A2A_PORT)])
        )
        wait_for(f"http://localhost:{A2A_PORT}/status", "A2A server", require_ok=True)

        if args.serve:
            console.print()
            console.print(serve_banner())
            while True:
                time.sleep(3600)
        else:
            question = " ".join(args.question) or "I'm visiting Berlin today - check the weather and plan my day."
            pause_between_steps(step_wait, "Starting scripted A2A run")
            console.rule("[bold cyan]3. Scripted A2A run")
            console.print(Panel(Text(question), title="Question", border_style="green"))
            subprocess.run(  # noqa: S603
                [sys.executable, "client.py", "--step-wait", str(step_wait), *args.question],
                cwd=DEMO_DIR,
                check=False,
            )
            print_flow_summary(question)
            wait_for_shutdown()
    except KeyboardInterrupt:
        pass
    finally:
        console.rule("[bold cyan]Shutdown")
        console.print("Stopping child servers ...")
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        console.print("[green]Done[/green]")


if __name__ == "__main__":
    main()
