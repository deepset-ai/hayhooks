from abc import ABC, abstractmethod
from typing import Generator, List, Union


class BasePipelineWrapper(ABC):
    def __init__(self):
        self.pipeline = None
        self._is_run_api_implemented = False
        self._is_run_chat_completion_implemented = False

    @abstractmethod
    def setup(self) -> None:
        """
        Initialize and configure the pipeline.

        This method will be called before any pipeline operations.
        It should initialize `self.pipeline` with the appropriate pipeline.

        Pipelines can be loaded from YAML or provided directly as code.
        """
        pass

    def run_api(self):
        """
        Execute the pipeline in API mode.

        This method provides a generic interface for running the pipeline
        with implementation-specific parameters.

        This method will be used as the handler for the `/run` API endpoint.

        Pydantic models will be automatically generated based on this method's
        signature and return type for request validation and response serialization.
        """
        raise NotImplementedError("run_api not implemented")

    def run_chat_completion(self, model: str, messages: List[dict], body: dict) -> Union[str, Generator]:
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
        raise NotImplementedError("run_chat_completion not implemented")
