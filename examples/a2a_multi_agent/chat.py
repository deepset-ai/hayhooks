"""
Interactive chat with one of the demo's A2A agents.

Run with:
    python chat.py                     # chat with the trip planner
    python chat.py weather_agent       # chat with the weather agent directly
"""

import asyncio
import os
import sys

import httpx
from a2a.client import A2ACardResolver, ClientConfig, create_client
from a2a.helpers import get_stream_response_text, new_text_message
from a2a.types import Role, SendMessageRequest

A2A_SERVER_URL = os.getenv("A2A_SERVER_URL", "http://localhost:1418")


async def main() -> None:
    agent = sys.argv[1] if len(sys.argv) > 1 else "trip_planner_agent"

    async with httpx.AsyncClient(timeout=180) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=f"{A2A_SERVER_URL}/{agent}")
        card = await resolver.get_agent_card()
        print(f"Chatting with {card.name} - {card.description}")
        print("(empty line, 'exit' or Ctrl-D to quit; each message is a new A2A task)\n")

        client = await create_client(agent=card, client_config=ClientConfig(streaming=True, httpx_client=httpx_client))
        try:
            while True:
                try:
                    question = input("you> ").strip()
                except EOFError:
                    break
                if not question or question in {"exit", "quit"}:
                    break

                request = SendMessageRequest(message=new_text_message(question, role=Role.ROLE_USER))
                print(f"{card.name}> ", end="", flush=True)
                async for response in client.send_message(request):
                    if response.HasField("artifact_update"):
                        print(get_stream_response_text(response), end="", flush=True)
                print("\n")
        finally:
            await client.close()


if __name__ == "__main__":
    asyncio.run(main())
