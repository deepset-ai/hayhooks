from typing import Annotated

import typer

from hayhooks.cli.utils import get_server_url


def chat(
    ctx: typer.Context,
    pipeline: Annotated[str, typer.Argument(help="Name of a deployed chat-capable pipeline")],
) -> None:
    """Chat with a deployed Agent or pipeline through its existing chat endpoint."""
    try:
        from hayhooks.chat.app import ChatApp
        from hayhooks.chat.client import ChatClient
    except ModuleNotFoundError as error:
        if error.name not in {"textual", "httpx"}:
            raise
        typer.echo("Install terminal chat with: pip install 'hayhooks[chat]'", err=True)
        raise typer.Exit(1) from error
    server_url = get_server_url(ctx.obj["host"], ctx.obj["port"], https=ctx.obj["use_https"])
    ChatApp(ChatClient(server_url, pipeline, verify_ssl=not ctx.obj["disable_ssl"])).run()
