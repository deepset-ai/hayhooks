from typing import TYPE_CHECKING

from haystack.lazy_imports import LazyImport

INSTALL_A2A_MESSAGE = "Run 'pip install \"hayhooks[a2a]\"' to install A2A support."

if TYPE_CHECKING:
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
    from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill, Role
    from a2a.utils.errors import InvalidParamsError

    a2a_import: LazyImport
else:
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
        from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill, Role
        from a2a.utils.errors import InvalidParamsError

    if "AgentExecutor" not in globals():
        AgentExecutor = object
        RequestContext = object
        RequestContextBuilder = object
        SimpleRequestContextBuilder = object
        EventQueue = object
        DefaultRequestHandler = None
        InMemoryTaskStore = None
        TaskStore = object
        TaskUpdater = object
        AgentCapabilities = None
        AgentCard = object
        AgentInterface = None
        AgentSkill = None
        Role = None
        InvalidParamsError = RuntimeError

        def get_message_text(*_args, **_kwargs):
            a2a_import.check()

        def new_task_from_user_message(*_args, **_kwargs):
            a2a_import.check()

        def new_text_part(*_args, **_kwargs):
            a2a_import.check()

        def create_agent_card_routes(*_args, **_kwargs):
            a2a_import.check()

        def create_jsonrpc_routes(*_args, **_kwargs):
            a2a_import.check()
