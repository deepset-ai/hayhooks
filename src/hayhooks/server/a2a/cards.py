from typing import Any

from hayhooks.durable.mode import DurableAuthoringMode, durable_authoring_mode
from hayhooks.server.a2a.imports import AgentCapabilities, AgentCard, AgentInterface, AgentSkill, a2a_import
from hayhooks.server.logger import log
from hayhooks.server.pipelines.registry import registry
from hayhooks.settings import settings


def get_a2a_base_url() -> str:
    """Base URL advertised in agent cards, without trailing slash."""
    base_url = settings.a2a_external_url or f"http://{settings.a2a_host}:{settings.a2a_port}"
    return base_url.rstrip("/")


def is_a2a_exposable(pipeline_name: str) -> bool:
    """
    Whether a deployed pipeline can be exposed as an A2A agent.

    A pipeline is exposable when it provides a native A2A or durable Agent hook,
    implements ``run_chat_completion`` / ``run_chat_completion_async``, and does
    not set ``skip_a2a = True``.
    """
    pipeline_wrapper = registry.get(pipeline_name)
    if pipeline_wrapper is None:
        return False

    metadata = registry.get_metadata(name=pipeline_name) or {}
    if metadata.get("skip_a2a"):
        log.debug("Skipping pipeline '{}': skip_a2a is set", pipeline_name)
        return False

    exposable = (
        hasattr(pipeline_wrapper, "create_a2a_agent_executor")
        or durable_authoring_mode(pipeline_wrapper) is DurableAuthoringMode.MANAGED_AGENT
        or pipeline_wrapper._is_run_chat_completion_implemented
        or pipeline_wrapper._is_run_chat_completion_async_implemented
    )
    if not exposable:
        log.debug("Skipping pipeline '{}': no A2A or chat completion method implemented", pipeline_name)
    return exposable


def create_agent_card(pipeline_name: str, base_url: str, *, push_notifications: bool = False) -> "AgentCard":
    """
    Build an A2A agent card for a deployed pipeline.

    Card fields are derived from the pipeline's registry metadata and can be
    overridden via the wrapper's ``a2a_card`` class attribute.
    """
    a2a_import.check()

    metadata = registry.get_metadata(name=pipeline_name) or {}
    overrides = metadata.get("a2a_card") or {}

    name = overrides.get("name") or pipeline_name
    description = (
        overrides.get("description")
        or metadata.get("description")
        or f"Haystack pipeline '{pipeline_name}' deployed with Hayhooks"
    )
    version = overrides.get("version") or "1.0.0"
    agent_url = f"{base_url.rstrip('/')}/{pipeline_name}/"

    skills_spec: list[dict[str, Any]] = overrides.get("skills") or [
        {"id": pipeline_name, "name": name, "description": description, "tags": ["haystack", "hayhooks"]}
    ]
    skills = [
        AgentSkill(
            id=skill.get("id", pipeline_name),
            name=skill.get("name", name),
            description=skill.get("description", description),
            tags=list(skill.get("tags", [])),
            examples=list(skill.get("examples", [])),
        )
        for skill in skills_spec
    ]

    log.debug(
        "Built A2A agent card for pipeline '{}' with name='{}', url='{}', skills={}",
        pipeline_name,
        name,
        agent_url,
        [skill.id for skill in skills],
    )

    return AgentCard(
        name=name,
        description=description,
        version=version,
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(streaming=True, push_notifications=push_notifications),
        supported_interfaces=[AgentInterface(protocol_binding="JSONRPC", url=agent_url)],
        skills=skills,
    )
