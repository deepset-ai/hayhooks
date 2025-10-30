from collections.abc import Generator
from typing import Any, Union

from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage, StreamingChunk
from haystack.utils import Secret

from hayhooks import BasePipelineWrapper, get_last_user_message, streaming_generator


class PipelineWrapper(BasePipelineWrapper):
    """
    A pipeline with two sequential LLM components that can both stream.

    The first LLM (low reasoning) provides a concise answer, and the second LLM
    (medium reasoning) refines and expands it with more detail.

    This example demonstrates the streaming_components parameter which controls which
    components should stream their responses.
    """

    def setup(self) -> None:
        """Initialize the pipeline with two streaming LLM components."""
        self.pipeline = Pipeline()

        # First stage: Initial answer
        self.pipeline.add_component(
            "prompt_builder_1",
            ChatPromptBuilder(
                template=[
                    ChatMessage.from_system(
                        "You are a helpful assistant. \nAnswer the user's question in a short and concise manner."
                    ),
                    ChatMessage.from_user("{{query}}"),
                ],
                required_variables="*",
            ),
        )
        self.pipeline.add_component(
            "llm_1",
            OpenAIChatGenerator(
                api_key=Secret.from_env_var("OPENAI_API_KEY"),
                model="gpt-5-nano",
                generation_kwargs={
                    "reasoning_effort": "low",
                },
            ),
        )

        # Second stage: Refinement
        # The prompt builder can directly access ChatMessage attributes via Jinja2
        self.pipeline.add_component(
            "prompt_builder_2",
            ChatPromptBuilder(
                template=[
                    ChatMessage.from_system("You are a helpful assistant that refines and improves responses."),
                    ChatMessage.from_user(
                        "Here is the previous response:\n\n{{previous_response[0].text}}\n\n"
                        "Please refine and improve this response. "
                        "Make it a bit more detailed, clear, and professional. "
                        "Please state that you're refining the response in the beginning of your answer."
                    ),
                ],
                required_variables="*",
            ),
        )
        self.pipeline.add_component(
            "llm_2",
            OpenAIChatGenerator(
                api_key=Secret.from_env_var("OPENAI_API_KEY"),
                model="gpt-5-nano",
                generation_kwargs={
                    "reasoning_effort": "medium",
                },
                streaming_callback=None,
            ),
        )

        # Connect the components
        self.pipeline.connect("prompt_builder_1.prompt", "llm_1.messages")
        self.pipeline.connect("llm_1.replies", "prompt_builder_2.previous_response")
        self.pipeline.connect("prompt_builder_2.prompt", "llm_2.messages")

    def run_api(self, query: str) -> dict[str, Any]:
        """Run the pipeline in non-streaming mode."""
        result = self.pipeline.run(
            {
                "prompt_builder_1": {"query": query},
            }
        )
        return {"reply": result["llm_2"]["replies"][0].text if result["llm_2"]["replies"] else ""}

    def run_chat_completion(self, model: str, messages: list[dict], body: dict) -> Union[str, Generator]:  # noqa: ARG002
        """
        Run the pipeline in streaming mode.

        This demonstrates the streaming_components parameter which controls which components stream.
        By default (streaming_components=None), only the last streaming-capable component (llm_2) streams.
        To enable streaming for both LLMs, use: streaming_components=["llm_1", "llm_2"]

        We inject a visual separator between LLM 1 and LLM 2 outputs.
        """
        question = get_last_user_message(messages)

        def custom_streaming():
            """
            Enhanced streaming that injects a visual separator between LLM outputs.

            Uses StreamingChunk.component_info.name to reliably detect which component
            is streaming, avoiding fragile chunk counting or heuristics.

            NOTE: This is simply a workaround to inject a visual separator between LLM outputs.
            """
            llm2_started = False

            # Enable streaming for both LLM components
            # To stream only the last component (default), omit streaming_components or set to None
            for chunk in streaming_generator(
                pipeline=self.pipeline,
                pipeline_run_args={
                    "prompt_builder_1": {"query": question},
                },
                streaming_components=["llm_1", "llm_2"],  # Or use streaming_components="all"
            ):
                # Use component_info to detect which LLM is streaming
                if hasattr(chunk, "component_info") and chunk.component_info:
                    component_name = chunk.component_info.name

                    # When we see llm_2 for the first time, inject a visual separator
                    if component_name == "llm_2" and not llm2_started:
                        llm2_started = True
                        yield StreamingChunk(content="\n\n**[LLM 2 - Refining the response]**\n\n")

                yield chunk

        return custom_streaming()
