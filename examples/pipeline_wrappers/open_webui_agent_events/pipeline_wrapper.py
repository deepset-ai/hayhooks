from collections.abc import AsyncGenerator
from typing import Union

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage, StreamingChunk

from hayhooks import BasePipelineWrapper, async_streaming_generator
from hayhooks.open_webui import OpenWebUIEvent, create_details_tag, create_status_event


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        self.agent = Agent(
            chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
            system_prompt="You're a helpful agent",
        )

    async def run_chat_completion_async(
        self,
        model: str,  # noqa: ARG002
        messages: list[dict],
        body: dict,  # noqa: ARG002
    ) -> AsyncGenerator[Union[StreamingChunk, OpenWebUIEvent, str], None]:
        chat_messages = [ChatMessage.from_openai_dict_format(message) for message in messages]

        async def main_async_generator():
            # Send a status event to open-webui to indicate that the pipeline is running
            # This will trigger the "Running the pipeline!" status in the open-webui UI
            yield create_status_event(
                description="Running the pipeline!",
                done=False,
            )

            # Now we consume the agent's response chunks and yield them
            async for chunk in async_streaming_generator(
                pipeline=self.agent,
                pipeline_run_args={
                    "messages": chat_messages,
                },
            ):
                yield chunk

            # Now we send a status event to open-webui to indicate that the pipeline has completed
            # This will trigger the "Pipeline completed!" status in the open-webui UI
            yield create_status_event(
                description="Pipeline completed!",
                done=True,
            )

            # We can event send a <details> tag to open-webui to provide more details about the pipeline
            # This will be displayed as a collapsible section in the open-webui UI with the summary
            # "Pipeline completed!" and the content "Pipeline successfully completed!"
            yield "\n" + create_details_tag(
                tool_name="Pipeline",
                summary="Pipeline completed!",
                content="Pipeline successfully completed!",
            )

        return main_async_generator()
