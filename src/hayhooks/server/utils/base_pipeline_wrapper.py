from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Generator
from typing import Union


class BasePipelineWrapper(ABC):
    # Class attribute to skip MCP listing of the pipeline
    # If True, the pipeline will not be listed as an MCP tool
    # Even if it has a description and a request model
    skip_mcp: bool = False

    def __init__(self):
        self.pipeline = None
        self._is_run_api_implemented = False
        self._is_run_chat_completion_implemented = False
        self._is_run_api_async_implemented = False
        self._is_run_chat_completion_async_implemented = False

    @abstractmethod
    def setup(self) -> None:
        """
        Initialize and configure the pipeline.

        This method will be called before any pipeline operations.
        It should initialize `self.pipeline` with the appropriate pipeline.

        Pipelines can be loaded from YAML or provided directly as code.
        """
        pass

    # Execute the pipeline in API mode.
    #
    # This method provides a generic interface for running the pipeline
    # with implementation-specific parameters.
    #
    # This method will be used as the handler for the `/run` API endpoint.
    #
    # Pydantic models will be automatically generated based on this method's
    # signature and return type for request validation and response serialization.
    #
    # NOTE: we don't provide a default docstring for this method as it will be
    #       used in MCP as a tool description, so by default it will be empty.
    #       If you want to provide a custom description, you can override the
    #       docstring in the subclass.
    def run_api(self):
        msg = "run_api not implemented"
        raise NotImplementedError(msg)

    # Asynchronous version of run_api.
    async def run_api_async(self):
        msg = "run_api_async not implemented"
        raise NotImplementedError(msg)

    def run_chat_completion(self, model: str, messages: list[dict], body: dict) -> Union[str, Generator]:
        """
        This method is called when a user sends an OpenAI-compatible chat completion request.

        This method handles conversational interactions with the pipeline,
        maintaining context and processing chat-specific parameters.

        This method will be used as the handler for the `/chat` API endpoint.

        Args:
            model: The `name` of the deployed Haystack pipeline to run
            messages: The history of messages as OpenAI-compatible list of dicts
            body: Additional parameters and configuration options
        """
        msg = "run_chat_completion not implemented"
        raise NotImplementedError(msg)

    async def run_chat_completion_async(
        self, model: str, messages: list[dict], body: dict
    ) -> Union[str, AsyncGenerator]:
        """
        Asynchronous version of run_chat_completion.
        """
        msg = "run_chat_completion_async not implemented"
        raise NotImplementedError(msg)
