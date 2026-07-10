"""
Demo client for the A2A multi-agent example.

Discovers both A2A agents through their cards, then asks the trip planner to
plan a day. The trip planner delegates the weather check to the weather agent
over A2A, and picks activities via its own MCP tools.

Run with:
    python client.py [question]
    python client.py --step-wait 0 [question]
"""

import argparse
import asyncio
import os
import sys
import termios
import tty

import httpx
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

A2A_SERVER_URL = os.getenv("A2A_SERVER_URL", "http://localhost:1418")
AGENTS = ["weather_agent", "trip_planner_agent"]
TOOLS_BY_AGENT = {
    "weather_agent": [
        ("get_weather", "MCP -> weather_server", "Fetches live weather from Open-Meteo."),
    ],
    "trip_planner_agent": [
        ("ask_weather_agent", "A2A -> weather_agent", "Delegates weather questions to the weather agent."),
        ("suggest_activities", "MCP -> activities_server", "Picks activities for the city and weather."),
    ],
}

DEFAULT_QUESTION = "I'm visiting Berlin today - check the weather and plan my day."
DEFAULT_STEP_WAIT_SECONDS = int(os.getenv("A2A_DEMO_STEP_WAIT", "5"))
console = Console()


def skill_lines(card) -> list[Text]:
    lines = [Text("Advertised A2A skills", style="bold cyan")]
    for skill in card.skills:
        tags = f" [{', '.join(skill.tags)}]" if skill.tags else ""
        lines.append(Text(f"  - {skill.name}{tags}: {skill.description or 'No description'}"))
    return lines


def tool_lines(agent: str) -> list[Text]:
    lines = [Text("Demo tools / integrations", style="bold cyan")]
    for name, route, purpose in TOOLS_BY_AGENT.get(agent, []):
        lines.append(Text(f"  - {name}: {route}. {purpose}"))
    return lines


async def pause_between_steps(seconds: int, next_step: str) -> None:
    if seconds <= 0:
        return
    console.print()
    for remaining in range(seconds, 0, -1):
        console.print(f"[dim]{next_step} in {remaining}...[/dim]")
        await asyncio.sleep(1)


def wait_for_keypress() -> None:
    if not sys.stdin.isatty():
        console.print("[dim]No interactive input available; starting the run.[/dim]")
        return

    console.print("\n[bold green]Press any key to start the scripted A2A run...[/bold green]", end="")
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    console.print()


def agent_panel(agent: str, card) -> Panel:
    content = [
        Text(f"{card.name} v{card.version}", style="bold green"),
        Text(card.description or ""),
        Text(f"card: {A2A_SERVER_URL}/{agent}/.well-known/agent-card.json", style="dim"),
        "",
        *skill_lines(card),
        "",
        *tool_lines(agent),
    ]
    return Panel(
        Group(*content),
        title="Discovered A2A agent",
        border_style="green",
    )


def flow_preview() -> Panel:
    return Panel(
        Group(
            Text("Expected flow", style="bold cyan"),
            Text("1. Client sends one A2A message to trip_planner_agent."),
            Text("2. trip_planner_agent discovers and calls weather_agent over A2A."),
            Text("3. weather_agent calls get_weather through its MCP server."),
            Text("4. trip_planner_agent calls suggest_activities through its MCP server."),
            Text("5. trip_planner_agent streams the final plan back here."),
        ),
        title="Before the run",
        border_style="cyan",
    )


def answer_panel(answer: str) -> Panel:
    body = Markdown(answer) if answer.strip() else Text("Waiting for streamed response...", style="dim")
    return Panel(body, title="trip_planner_agent streamed answer", border_style="green")


async def show_agent_cards(httpx_client: httpx.AsyncClient) -> dict:
    from a2a.client import A2ACardResolver

    cards = {}
    console.rule("[bold cyan]A2A discovery")
    console.print(f"Reading agent cards from [bold]{A2A_SERVER_URL}[/bold]")
    for agent in AGENTS:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=f"{A2A_SERVER_URL}/{agent}")
        card = await resolver.get_agent_card()
        cards[agent] = card
        console.print(agent_panel(agent, card))
    return cards


async def ask_trip_planner(httpx_client: httpx.AsyncClient, card, question: str, step_wait: int) -> None:
    from a2a.client import ClientConfig, create_client
    from a2a.helpers import get_stream_response_text, new_text_message
    from a2a.types import Role, SendMessageRequest

    console.rule("[bold cyan]A2A request")
    console.print(Panel(Text(question), title="Question for trip_planner_agent", border_style="cyan"))
    console.print(flow_preview())
    wait_for_keypress()
    await pause_between_steps(step_wait, "Sending A2A message")
    client = await create_client(agent=card, client_config=ClientConfig(streaming=True, httpx_client=httpx_client))
    try:
        request = SendMessageRequest(message=new_text_message(question, role=Role.ROLE_USER))
        console.rule("[bold cyan]Streaming answer")
        answer = ""
        console.print("[dim]Waiting for streamed response...[/dim]")
        live: Live | None = None
        try:
            async for response in client.send_message(request):
                if response.HasField("artifact_update"):
                    chunk = get_stream_response_text(response)
                    if not chunk:
                        continue
                    answer += chunk
                    if live is None:
                        live = Live(answer_panel(answer), console=console, auto_refresh=False, transient=True)
                        live.start()
                    else:
                        live.update(answer_panel(answer), refresh=True)
            if live is not None:
                live.update(answer_panel(answer), refresh=True)
        finally:
            if live is not None:
                live.stop()
        console.print(answer_panel(answer))
    finally:
        await client.close()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run the scripted A2A demo client.")
    parser.add_argument(
        "--step-wait",
        type=int,
        default=DEFAULT_STEP_WAIT_SECONDS,
        help="seconds to wait between visible demo steps (default: %(default)s)",
    )
    parser.add_argument("question", nargs="*", help="question for the trip planner")
    args = parser.parse_args()

    question = " ".join(args.question) or DEFAULT_QUESTION
    async with httpx.AsyncClient(timeout=180) as httpx_client:
        cards = await show_agent_cards(httpx_client)
        await ask_trip_planner(httpx_client, cards["trip_planner_agent"], question, max(args.step_wait, 0))


if __name__ == "__main__":
    asyncio.run(main())
