"""
Default Chainlit app for Hayhooks.

This app provides a chat interface that communicates with Hayhooks'
OpenAI-compatible endpoints, allowing users to interact with deployed
Haystack pipelines through a conversational UI.
"""

import os

import chainlit as cl  # type: ignore
import httpx

# Configuration - these can be overridden via environment variables
HAYHOOKS_BASE_URL = os.getenv("HAYHOOKS_BASE_URL", "http://localhost:1416")
DEFAULT_MODEL = os.getenv("HAYHOOKS_DEFAULT_MODEL", "")


async def get_available_models() -> list[dict]:
    """Fetch available models (deployed pipelines) from Hayhooks."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{HAYHOOKS_BASE_URL}/v1/models", timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                return data.get("data", [])
    except Exception as e:
        cl.logger.warning(f"Failed to fetch models: {e}")
    return []


@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat session."""
    # Fetch available models
    models = await get_available_models()

    if not models:
        await cl.Message(
            content="⚠️ No pipelines are currently deployed. "
            "Deploy a pipeline with `run_chat_completion` method to start chatting."
        ).send()
        return

    # If there's a default model set and it exists, use it
    model_ids = [m["id"] for m in models]
    if DEFAULT_MODEL and DEFAULT_MODEL in model_ids:
        cl.user_session.set("model", DEFAULT_MODEL)
        await cl.Message(content=f"🚀 Connected to **{DEFAULT_MODEL}**. How can I help you?").send()
    elif len(models) == 1:
        # Only one model available, use it automatically
        cl.user_session.set("model", models[0]["id"])
        await cl.Message(content=f"🚀 Connected to **{models[0]['id']}**. How can I help you?").send()
    else:
        # Multiple models - let user choose
        actions = [
            cl.Action(name="select_model", payload={"model": m["id"]}, label=m["id"]) for m in models[:10]  # Limit to 10
        ]
        await cl.Message(content="**Select a pipeline to chat with:**", actions=actions).send()


@cl.action_callback("select_model")
async def on_model_select(action: cl.Action):
    """Handle model selection."""
    model = action.payload["model"]
    cl.user_session.set("model", model)
    await cl.Message(content=f"🚀 Connected to **{model}**. How can I help you?").send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages."""
    model = cl.user_session.get("model")

    if not model:
        # No model selected - prompt user to select one
        models = await get_available_models()
        if not models:
            await cl.Message(
                content="⚠️ No pipelines are deployed. Please deploy a pipeline first."
            ).send()
            return

        actions = [
            cl.Action(name="select_model", payload={"model": m["id"]}, label=m["id"]) for m in models[:10]
        ]
        await cl.Message(content="**Please select a pipeline first:**", actions=actions).send()
        return

    # Get message history for context
    history = cl.user_session.get("history", [])
    history.append({"role": "user", "content": message.content})

    # Create response message for streaming
    response_msg = cl.Message(content="")
    await response_msg.send()

    try:
        # Call Hayhooks OpenAI-compatible endpoint with streaming
        async with httpx.AsyncClient() as client, client.stream(
            "POST",
            f"{HAYHOOKS_BASE_URL}/v1/chat/completions",
            json={
                "model": model,
                "messages": history,
                "stream": True,
            },
            timeout=120.0,
        ) as response:
            if response.status_code != 200:
                error_text = await response.aread()
                await response_msg.stream_token(f"❌ Error: {error_text.decode()}")
                await response_msg.update()
                return

            full_response = ""
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        import json

                        chunk = json.loads(data)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            full_response += content
                            await response_msg.stream_token(content)
                    except json.JSONDecodeError:
                        continue

            # Update history with assistant response
            if full_response:
                history.append({"role": "assistant", "content": full_response})
                cl.user_session.set("history", history)

    except httpx.TimeoutException:
        await response_msg.stream_token("❌ Request timed out. Please try again.")
    except Exception as e:
        await response_msg.stream_token(f"❌ Error: {e!s}")

    await response_msg.update()


@cl.on_settings_update
async def on_settings_update(settings: dict):
    """Handle settings updates."""
    if "model" in settings:
        cl.user_session.set("model", settings["model"])
        await cl.Message(content=f"🔄 Switched to **{settings['model']}**").send()
