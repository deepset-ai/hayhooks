"""
Compatibility imports for the A2A server implementation.

The implementation lives in :mod:`hayhooks.server.a2a`. This module is kept so
older imports continue to work.
"""

from hayhooks.server.a2a import (
    A2ARuntime,
    ChatCompletionAgentExecutor,
    DurableAgentExecutor,
    InMemoryTaskStoreProvider,
    RedisTaskStore,
    RedisTaskStoreProvider,
    TaskStoreProvider,
    a2a_import,
    create_a2a_app,
    create_agent_card,
    create_agent_executor,
    create_task_store_provider,
    get_a2a_base_url,
    is_a2a_exposable,
    load_task_store_provider,
)
from hayhooks.server.a2a.app import _create_agent_mount
from hayhooks.server.a2a.executor import (
    RESPONSE_ARTIFACT_NAME,
    _build_openai_messages,
    _execute_agent_task,
    _iter_text_chunks,
    _run_chat_completion,
    _stream_item_to_text,
    _stream_result_as_artifact,
)

__all__ = [
    "RESPONSE_ARTIFACT_NAME",
    "A2ARuntime",
    "ChatCompletionAgentExecutor",
    "DurableAgentExecutor",
    "InMemoryTaskStoreProvider",
    "RedisTaskStore",
    "RedisTaskStoreProvider",
    "TaskStoreProvider",
    "_build_openai_messages",
    "_create_agent_mount",
    "_execute_agent_task",
    "_iter_text_chunks",
    "_run_chat_completion",
    "_stream_item_to_text",
    "_stream_result_as_artifact",
    "a2a_import",
    "create_a2a_app",
    "create_agent_card",
    "create_agent_executor",
    "create_task_store_provider",
    "get_a2a_base_url",
    "is_a2a_exposable",
    "load_task_store_provider",
]
