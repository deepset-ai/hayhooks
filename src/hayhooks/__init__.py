"""Public Hayhooks API, loaded on first use."""

from importlib import import_module
from typing import Any

__all__ = [
    "AsyncPipeline",
    "BasePipelineWrapper",
    "Pipeline",
    "PipelineEvent",
    "SSEStream",
    "YAMLPipelineWrapper",
    "async_streaming_generator",
    "chat_messages_from_openai_response",
    "coerce_pipeline_inputs",
    "create_app",
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

_EXPORT_MODULES = {
    "AsyncPipeline": "hayhooks.server.utils.haystack_compat",
    "BasePipelineWrapper": "hayhooks.server.utils.base_pipeline_wrapper",
    "Pipeline": "hayhooks.server.utils.haystack_compat",
    "PipelineEvent": "hayhooks.events",
    "SSEStream": "hayhooks.server.pipelines.sse",
    "YAMLPipelineWrapper": "hayhooks.server.utils.yaml_pipeline_wrapper",
    "async_streaming_generator": "hayhooks.server.pipelines.utils",
    "chat_messages_from_openai_response": "hayhooks.server.pipelines.utils",
    "coerce_pipeline_inputs": "hayhooks.server.pipelines.utils",
    "create_app": "hayhooks.server.app",
    "default_on_pipeline_end": "hayhooks.callbacks",
    "default_on_tool_call_end": "hayhooks.callbacks",
    "default_on_tool_call_start": "hayhooks.callbacks",
    "get_input_files": "hayhooks.server.pipelines.utils",
    "get_last_user_input_text": "hayhooks.server.pipelines.utils",
    "get_last_user_message": "hayhooks.server.pipelines.utils",
    "is_user_message": "hayhooks.server.pipelines.utils",
    "log": "hayhooks.server.logger",
    "run_app": "hayhooks.server.app",
    "streaming_generator": "hayhooks.server.pipelines.utils",
}


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORT_MODULES[name]
    except KeyError:
        raise AttributeError(name) from None
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
