from abc import ABC, abstractmethod
from typing import List


class BasePipelineWrapper(ABC):
    def __init__(self):
        self.pipeline = None

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

        An API endpoint will call this method and will use dynamically created
        pydantic models for request and response validation.
        """
        raise NotImplementedError("run_api not implemented")

    def run_chat(self, model_id: str, messages: List[dict], body: dict):
        """
        This method is called when a user sends an OpenAI-compatible chat completion request.

        This method handles conversational interactions with the pipeline,
        maintaining context and processing chat-specific parameters.

        Args:
            model_id: The model (Haystack pipeline) to run
            messages: List of previous conversation messages for context
            body: Additional parameters and configuration options
        """
        raise NotImplementedError("run_chat not implemented")
