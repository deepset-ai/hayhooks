from haystack.lazy_imports import LazyImport

INSTALL_A2A_MESSAGE = "Run 'pip install \"hayhooks[a2a]\"' to install A2A support."

with LazyImport(INSTALL_A2A_MESSAGE) as a2a_import:
    from a2a.helpers import get_message_text, new_task_from_user_message, new_text_part
    from a2a.server.agent_execution import (
        AgentExecutor,
        RequestContext,
        RequestContextBuilder,
        SimpleRequestContextBuilder,
    )
    from a2a.server.events import EventQueue
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
    from a2a.server.tasks import InMemoryTaskStore, TaskStore, TaskUpdater
    from a2a.server.tasks.task_manager import append_artifact_to_task
    from a2a.types import (
        AgentCapabilities,
        AgentCard,
        AgentInterface,
        AgentSkill,
        ListTasksRequest,
        ListTasksResponse,
        Role,
        Task,
        TaskArtifactUpdateEvent,
        TaskState,
        TaskStatusUpdateEvent,
    )
    from a2a.utils.constants import DEFAULT_LIST_TASKS_PAGE_SIZE
    from a2a.utils.errors import InvalidParamsError
    from a2a.utils.task import decode_page_token, encode_page_token

a2a_import.check()

__all__ = [
    "DEFAULT_LIST_TASKS_PAGE_SIZE",
    "AgentCapabilities",
    "AgentCard",
    "AgentExecutor",
    "AgentInterface",
    "AgentSkill",
    "DefaultRequestHandler",
    "EventQueue",
    "InMemoryTaskStore",
    "InvalidParamsError",
    "ListTasksRequest",
    "ListTasksResponse",
    "RequestContext",
    "RequestContextBuilder",
    "Role",
    "SimpleRequestContextBuilder",
    "Task",
    "TaskArtifactUpdateEvent",
    "TaskState",
    "TaskStatusUpdateEvent",
    "TaskStore",
    "TaskUpdater",
    "append_artifact_to_task",
    "create_agent_card_routes",
    "create_jsonrpc_routes",
    "decode_page_token",
    "encode_page_token",
    "get_message_text",
    "new_task_from_user_message",
    "new_text_part",
]
