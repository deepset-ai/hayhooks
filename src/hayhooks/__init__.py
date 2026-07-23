"""Public Hayhooks authoring API."""

from hayhooks.a2a import A2APipelineWrapper
from hayhooks.callbacks import default_on_pipeline_end, default_on_tool_call_end, default_on_tool_call_start
from hayhooks.durable import (
    DurableOptions,
    ExecutionProgress,
    ExecutionResult,
    current_durable_context,
    current_execution_id,
)
from hayhooks.events import PipelineEvent
from hayhooks.execution import DurableContext, ExecutionStatus
from hayhooks.server.app import create_app, run_app
from hayhooks.server.logger import log
from hayhooks.server.pipelines.sse import SSEStream
from hayhooks.server.pipelines.streaming import async_streaming_generator, streaming_generator
from hayhooks.server.pipelines.utils import (
    chat_messages_from_openai_response,
    get_input_files,
    get_last_user_input_text,
    get_last_user_message,
    is_user_message,
)
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper
from hayhooks.server.utils.haystack_compat import AsyncPipeline, Pipeline
from hayhooks.server.utils.yaml_pipeline_wrapper import YAMLPipelineWrapper

__all__ = [
    "A2APipelineWrapper",
    "AsyncPipeline",
    "BasePipelineWrapper",
    "DurableContext",
    "DurableOptions",
    "ExecutionProgress",
    "ExecutionResult",
    "ExecutionStatus",
    "Pipeline",
    "PipelineEvent",
    "SSEStream",
    "YAMLPipelineWrapper",
    "async_streaming_generator",
    "chat_messages_from_openai_response",
    "create_app",
    "current_durable_context",
    "current_execution_id",
    "default_on_pipeline_end",
    "default_on_tool_call_end",
    "default_on_tool_call_start",
    "get_input_files",
    "get_last_user_input_text",
    "get_last_user_message",
    "is_user_message",
    "log",
    "run_app",
    "streaming_generator",
]
