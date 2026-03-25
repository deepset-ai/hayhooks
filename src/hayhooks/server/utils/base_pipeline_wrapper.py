from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Generator


class BasePipelineWrapper(ABC):
    # Class attribute to skip MCP listing of the pipeline
    # If True, the pipeline will not be listed as an MCP tool
    # Even if it has a description and a request model
    skip_mcp: bool = False

    def __init__(self):
        self.pipeline = None
        self._is_run_api_implemented = False
        self._is_run_api_async_implemented = False
        self._is_run_chat_completion_implemented = False
        self._is_run_chat_completion_async_implemented = False
        self._is_run_response_implemented = False
        self._is_run_response_async_implemented = False
        self._is_run_file_upload_implemented = False

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

    def run_chat_completion(self, model: str, messages: list[dict], body: dict) -> str | Generator:
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

    async def run_chat_completion_async(self, model: str, messages: list[dict], body: dict) -> str | AsyncGenerator:
        """
        Asynchronous version of run_chat_completion.
        """
        msg = "run_chat_completion_async not implemented"
        raise NotImplementedError(msg)

    def run_response(self, model: str, input_items: list[dict], body: dict) -> str | Generator:
        """
        Handle an OpenAI-compatible Responses API request.

        This method processes requests sent to the `/v1/responses` endpoint,
        using the Responses API input format instead of chat messages.

        Args:
            model: The `name` of the deployed Haystack pipeline to run
            input_items: Normalized input items in OpenAI Responses API format
            body: Additional parameters and configuration options (e.g. temperature, tools, instructions)
        """
        msg = "run_response not implemented"
        raise NotImplementedError(msg)

    async def run_response_async(self, model: str, input_items: list[dict], body: dict) -> str | AsyncGenerator:
        """
        Asynchronous version of run_response.
        """
        msg = "run_response_async not implemented"
        raise NotImplementedError(msg)

    def run_file_upload(self, filename: str | None, content_type: str | None, content: bytes, purpose: str) -> dict:
        """
        Handle a file uploaded via the ``/v1/files`` endpoint.

        Override this method to store or process uploaded files.  The returned
        dict must conform to the OpenAI ``FileObject`` schema (fields: ``id``,
        ``object``, ``bytes``, ``created_at``, ``filename``, ``purpose``).

        Args:
            filename: Original uploaded filename (may be ``None``).
            content_type: MIME content type (may be ``None``).
            content: Raw file bytes.
            purpose: Upload purpose string from the request.
        """
        msg = "run_file_upload not implemented"
        raise NotImplementedError(msg)
