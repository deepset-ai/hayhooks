from abc import ABC, abstractmethod
from typing import List


class BasePipelineWrapper(ABC):
    def __init__(self):
        self.pipeline = None

    @abstractmethod
    def setup(self) -> None:
        """
        Setup the pipeline.

        This method should be called before using the pipeline.
        """
        pass

    @abstractmethod
    def run_api(self, urls: List[str], question: str) -> dict:
        """
        Run the pipeline in API mode.

        Args:
            urls: List of URLs to fetch content from
            question: Question to be answered

        Returns:
            dict: Pipeline execution results
        """
        pass

    @abstractmethod
    def run_chat(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> dict:
        """
        Run the pipeline in chat mode.

        Args:
            user_message: Message from the user
            model_id: ID of the model to use
            messages: List of previous messages
            body: Additional request body parameters

        Returns:
            dict: Pipeline execution results
        """
        pass
