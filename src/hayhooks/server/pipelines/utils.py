import asyncio
import inspect
import threading
from collections.abc import AsyncGenerator, Generator
from queue import Queue
from typing import Any, Callable, Literal, Optional, Union

from haystack import AsyncPipeline, Pipeline
from haystack.components.agents import Agent
from haystack.core.component import Component
from haystack.dataclasses import StreamingChunk

from hayhooks.open_webui import OpenWebUIEvent
from hayhooks.server.logger import log
from hayhooks.server.routers.openai import Message
from hayhooks.settings import settings

ToolCallbackReturn = Union[OpenWebUIEvent, str, None, list[Union[OpenWebUIEvent, str]]]
OnToolCallStart = Optional[Callable[[str, Optional[str], Optional[str]], ToolCallbackReturn]]
OnToolCallEnd = Optional[Callable[[str, dict[str, Any], str, bool], ToolCallbackReturn]]
OnPipelineEnd = Optional[Callable[[Any], Optional[str]]]


def is_user_message(msg: Union[Message, dict]) -> bool:
    if isinstance(msg, Message):
        return msg.role == "user"
    return msg.get("role") == "user"


def get_content(msg: Union[Message, dict]) -> str:
    if isinstance(msg, Message):
        return msg.content
    return msg.get("content", "")


def get_last_user_message(messages: list[Union[Message, dict]]) -> Union[str, None]:
    user_messages = (msg for msg in reversed(messages) if is_user_message(msg))

    for message in user_messages:
        return get_content(message)

    return None


def find_all_streaming_components(pipeline: Union[Pipeline, AsyncPipeline]) -> list[tuple[Component, str]]:
    """
    Finds all components in the pipeline that support streaming_callback.

    Returns:
        A list of tuples containing (component, component_name) for all streaming components
    """
    streaming_components = []

    for name, component in pipeline.walk():
        # Check if the component's run() method accepts streaming_callback parameter
        if hasattr(component, "run"):
            sig = inspect.signature(component.run)
            if "streaming_callback" in sig.parameters:
                log.trace("streaming_callback run parameter found in '{}'", name)
                streaming_components.append((component, name))

    if not streaming_components:
        msg = "No streaming-capable components found in the pipeline"
        raise ValueError(msg)

    return streaming_components


def _parse_streaming_components_setting(setting_value: str) -> Union[list[str], Literal["all"], None]:
    """
    Parse the HAYHOOKS_STREAMING_COMPONENTS environment variable.

    Args:
        setting_value: The raw setting value from environment variable

    Returns:
        - None if empty string (use default behavior)
        - "all" if the value is "all"
        - list[str] if it's a comma-separated list of component names
    """
    if not setting_value or setting_value.strip() == "":
        return None

    setting_value = setting_value.strip()

    # Check for "all" keyword
    if setting_value.lower() == "all":
        return "all"

    # Parse as comma-separated list
    components = [c.strip() for c in setting_value.split(",") if c.strip()]
    if components:
        return components

    return None


def _setup_streaming_callback_for_pipeline(
    pipeline: Union[Pipeline, AsyncPipeline],
    pipeline_run_args: dict[str, Any],
    streaming_callback: Any,
    streaming_components: Optional[Union[list[str], Literal["all"]]] = None,
) -> dict[str, Any]:
    """
    Sets up streaming callbacks for streaming-capable components in the pipeline.

    By default, only the last streaming-capable component will stream. You can customize this
    behavior using the streaming_components parameter or HAYHOOKS_STREAMING_COMPONENTS env var.

    Args:
        pipeline: The pipeline to configure
        pipeline_run_args: Arguments for pipeline execution
        streaming_callback: The callback function to set
        streaming_components: Optional config for which components should stream.
                             Can be:
                             - None: use HAYHOOKS_STREAMING_COMPONENTS or default (last only)
                             - "all": stream all capable components
                             - list[str]: ["llm_1", "llm_2"] to enable specific components

    Returns:
        Updated pipeline run arguments
    """
    all_streaming_components = find_all_streaming_components(pipeline)

    # If streaming_components not provided, check environment variable
    if streaming_components is None:
        streaming_components = _parse_streaming_components_setting(settings.streaming_components)

    # Determine which components should stream
    components_to_stream = []

    # Stream all capable components
    if streaming_components == "all":
        components_to_stream = all_streaming_components
        log.trace("Streaming enabled for all components via 'all' keyword '{}'", components_to_stream)

    # Default behavior: stream only the last capable component
    elif streaming_components is None:
        if all_streaming_components:
            components_to_stream = [all_streaming_components[-1]]
            log.trace("Streaming enabled for last component only '{}'", components_to_stream)

    # Use explicit list of component names
    elif isinstance(streaming_components, list):
        enabled_component_names = set(streaming_components)
        for component, component_name in all_streaming_components:
            if component_name in enabled_component_names:
                components_to_stream.append((component, component_name))
        log.trace("Streaming enabled for components '{}'", [name for _, name in components_to_stream])

    for _, component_name in components_to_stream:
        # Ensure component args exist and make a copy to avoid mutating original
        if component_name not in pipeline_run_args:
            pipeline_run_args[component_name] = {}
        else:
            # Create a copy of the existing component args to avoid modifying the original
            pipeline_run_args[component_name] = pipeline_run_args[component_name].copy()

        pipeline_run_args[component_name]["streaming_callback"] = streaming_callback
        log.trace("Streaming callback set for component '{}'", component_name)

    return pipeline_run_args


def _setup_streaming_callback_for_agent(pipeline_run_args: dict[str, Any], streaming_callback: Any) -> dict[str, Any]:
    """
    Sets up streaming callback for agent execution.

    Args:
        pipeline_run_args: Arguments for agent execution
        streaming_callback: The callback function to set

    Returns:
        Updated pipeline run arguments
    """
    pipeline_run_args["streaming_callback"] = streaming_callback
    return pipeline_run_args


def _setup_streaming_callback(
    pipeline: Union[Pipeline, AsyncPipeline, Agent],
    pipeline_run_args: dict[str, Any],
    streaming_callback: Any,
    streaming_components: Optional[Union[list[str], Literal["all"]]] = None,
) -> dict[str, Any]:
    """
    Configures streaming callback for the given pipeline or agent.

    Args:
        pipeline: The pipeline or agent to configure
        pipeline_run_args: Execution arguments
        streaming_callback: The callback function
        streaming_components: Optional config - list[str], "all", or None (pipelines only)

    Returns:
        Updated pipeline run arguments
    """
    pipeline_run_args = pipeline_run_args.copy()

    if isinstance(pipeline, (Pipeline, AsyncPipeline)):
        return _setup_streaming_callback_for_pipeline(
            pipeline, pipeline_run_args, streaming_callback, streaming_components
        )

    if isinstance(pipeline, Agent):
        return _setup_streaming_callback_for_agent(pipeline_run_args, streaming_callback)

    msg = f"Unsupported pipeline type: {type(pipeline)}"
    raise ValueError(msg)


def _yield_callback_results(result: ToolCallbackReturn) -> Generator[Union[OpenWebUIEvent, str], None, None]:
    """
    Yields callback results, handling both single values and lists.

    Args:
        result: The callback result to yield (can be None, single value, or list)

    Yields:
        OpenWebUIEvent or str: The callback results
    """
    if result:
        if isinstance(result, list):
            yield from result
        else:
            yield result


def _process_tool_call_start(
    chunk: StreamingChunk, on_tool_call_start: OnToolCallStart
) -> Generator[Union[OpenWebUIEvent, str], None, None]:
    """
    Process tool call start events from a streaming chunk.

    Args:
        chunk: The streaming chunk that may contain tool calls
        on_tool_call_start: Callback function for tool call start

    Yields:
        OpenWebUIEvent or str: Results from the callback
    """
    if on_tool_call_start and hasattr(chunk, "tool_calls") and chunk.tool_calls:
        for tool_call in chunk.tool_calls:
            if tool_call.tool_name:
                result = on_tool_call_start(tool_call.tool_name, tool_call.arguments, tool_call.id)
                yield from _yield_callback_results(result)


def _process_tool_call_end(
    chunk: StreamingChunk, on_tool_call_end: OnToolCallEnd
) -> Generator[Union[OpenWebUIEvent, str], None, None]:
    """
    Process tool call end events from a streaming chunk.

    Args:
        chunk: The streaming chunk that may contain tool call results
        on_tool_call_end: Callback function for tool call end

    Yields:
        OpenWebUIEvent or str: Results from the callback
    """
    if on_tool_call_end and hasattr(chunk, "tool_call_result") and chunk.tool_call_result:
        result = on_tool_call_end(
            chunk.tool_call_result.origin.tool_name,
            chunk.tool_call_result.origin.arguments,
            chunk.tool_call_result.result,
            bool(chunk.tool_call_result.error),
        )
        yield from _yield_callback_results(result)


def _process_pipeline_end(result: dict[str, Any], on_pipeline_end: OnPipelineEnd) -> Optional[StreamingChunk]:
    """
    Process pipeline end callback and return a StreamingChunk if there's content.

    Args:
        result: The pipeline execution result
        on_pipeline_end: Optional callback function for pipeline end

    Returns:
        StreamingChunk with content from callback, or None
    """
    if on_pipeline_end:
        try:
            on_pipeline_end_result = on_pipeline_end(result)
            if on_pipeline_end_result:
                return StreamingChunk(content=on_pipeline_end_result)
        except Exception as e:
            log.error("Error in on_pipeline_end callback: {}", e, exc_info=True)
    return None


def _execute_pipeline_sync(
    pipeline: Union[Pipeline, AsyncPipeline, Agent],
    pipeline_run_args: dict[str, Any],
    include_outputs_from: Optional[set[str]] = None,
) -> dict[str, Any]:
    """
    Executes pipeline synchronously based on its type.

    Args:
        pipeline: The pipeline or agent to execute
        pipeline_run_args: Execution arguments
        include_outputs_from: Optional set of component names to include outputs from (Pipeline/AsyncPipeline only)
    """
    if isinstance(pipeline, Agent):
        return pipeline.run(**pipeline_run_args)

    kwargs: dict[str, Any] = {"data": pipeline_run_args}
    if include_outputs_from is not None:
        kwargs["include_outputs_from"] = include_outputs_from

    return pipeline.run(**kwargs)


def streaming_generator(  # noqa: PLR0913
    pipeline: Union[Pipeline, AsyncPipeline, Agent],
    *,
    pipeline_run_args: Optional[dict[str, Any]] = None,
    on_tool_call_start: OnToolCallStart = None,
    on_tool_call_end: OnToolCallEnd = None,
    on_pipeline_end: OnPipelineEnd = None,
    streaming_components: Optional[Union[list[str], Literal["all"]]] = None,
    include_outputs_from: Optional[set[str]] = None,
) -> Generator[Union[StreamingChunk, OpenWebUIEvent, str], None, None]:
    """
    Creates a generator that yields streaming chunks from a pipeline or agent execution.

    By default, only the last streaming-capable component in pipelines will stream.
    You can control which components stream using streaming_components or HAYHOOKS_STREAMING_COMPONENTS.

    Args:
        pipeline: The Pipeline, AsyncPipeline, or Agent to execute
        pipeline_run_args: Arguments for execution
        on_tool_call_start: Callback for tool call start
        on_tool_call_end: Callback for tool call end
        on_pipeline_end: Callback for pipeline end
        streaming_components: Optional config for which components should stream.
                             Can be:
                             - None: use HAYHOOKS_STREAMING_COMPONENTS or default (last only)
                             - "all": stream all capable components
                             - list[str]: ["llm_1", "llm_2"] to enable specific components
        include_outputs_from: Optional set of component names to include outputs from (Pipeline/AsyncPipeline only)

    Yields:
        StreamingChunk: Individual chunks from the streaming execution
        OpenWebUIEvent: Event for tool call
        str: Tool name or stream content

    NOTE: This generator works with sync/async pipelines and agents. Pipeline components
          which support streaming must have a _sync_ `streaming_callback`. By default,
          only the last streaming-capable component will stream.
    """
    if pipeline_run_args is None:
        pipeline_run_args = {}
    queue: Queue[Union[StreamingChunk, None, Exception]] = Queue()

    def streaming_callback(chunk: StreamingChunk) -> None:
        queue.put(chunk)

    # Configure streaming callback
    try:
        configured_args = _setup_streaming_callback(
            pipeline, pipeline_run_args, streaming_callback, streaming_components
        )
        log.trace("Streaming pipeline run args '{}'", configured_args)
    except Exception as e:
        log.error("Error in streaming callback setup: {}", e, exc_info=True)
        raise

    def run_pipeline() -> None:
        try:
            result = _execute_pipeline_sync(pipeline, configured_args, include_outputs_from)
            # Process pipeline end callback
            final_chunk = _process_pipeline_end(result, on_pipeline_end)
            if final_chunk:
                queue.put(final_chunk)
            # Signal completion
            queue.put(None)
        except Exception as e:
            log.error("Error in pipeline execution thread for streaming_generator: {}", e, exc_info=True)
            queue.put(e)  # Signal error

    thread = threading.Thread(target=run_pipeline)
    thread.start()

    try:
        while True:
            item = queue.get()
            if isinstance(item, Exception):
                raise item
            if item is None:
                break

            # Handle tool calls
            yield from _process_tool_call_start(item, on_tool_call_start)
            yield from _process_tool_call_end(item, on_tool_call_end)
            yield item
    finally:
        thread.join()


def _create_hybrid_streaming_callback(
    queue: asyncio.Queue[StreamingChunk], component: Component, component_name: str, loop: asyncio.AbstractEventLoop
) -> Callable:
    """
    Creates a streaming callback (sync or async) based on component capabilities.

    For components without run_async (sync-only), wraps the sync callback to work with asyncio.
    For components with run_async (async-capable), returns an async callback.

    Args:
        queue: The asyncio queue to put chunks into
        component: The component to check for async support
        component_name: Name of the component (for logging)
        loop: The event loop to use for thread-safe operations

    Returns:
        A streaming callback function (sync or async) appropriate for the component
    """
    has_async_support = hasattr(component, "run_async")

    if has_async_support:

        async def async_callback(chunk: StreamingChunk) -> None:
            await queue.put(chunk)

        log.trace("Using async streaming callback for component '{}'", component_name)
        return async_callback
    else:

        def sync_callback(chunk: StreamingChunk) -> None:
            # Bridge sync callback to async queue using thread-safe operation
            future = asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)

            def handle_error(fut: Any) -> None:
                try:
                    if not fut.cancelled():
                        fut.result()
                except Exception as e:
                    log.error(
                        "Error in hybrid streaming callback for component '{}': {}",
                        component_name,
                        e,
                        exc_info=True,
                    )

            future.add_done_callback(handle_error)

        log.trace("Using sync streaming callback (hybrid mode) for component '{}'", component_name)
        return sync_callback


def _validate_async_streaming_support(
    pipeline: Union[Pipeline, AsyncPipeline],
    allow_sync_streaming_callbacks: bool = False,
) -> tuple[bool, list[tuple[Component, str]]]:
    """
    Validates that all streaming components in the pipeline support async streaming callbacks.

    Args:
        pipeline: The pipeline to validate
        allow_sync_streaming_callbacks: Controls validation behavior:
            - False (default): Strict mode - all components must support async, raises error otherwise
            - True: Automatically detect and enable hybrid mode only if needed

    Returns:
        A tuple of (use_hybrid_mode, streaming_components):
            - use_hybrid_mode: True if hybrid mode should be used, False otherwise
            - streaming_components: List of (component, component_name) tuples

    Raises:
        ValueError: If any streaming component doesn't support async streaming and
                   allow_sync_streaming_callbacks is False
    """
    # Get all streaming components once to avoid multiple pipeline walks
    streaming_components = find_all_streaming_components(pipeline)

    if allow_sync_streaming_callbacks:
        needs_hybrid = any(not hasattr(component, "run_async") for component, _ in streaming_components)
        return (needs_hybrid, streaming_components)

    for streaming_component, streaming_component_name in streaming_components:
        if not hasattr(streaming_component, "run_async"):
            component_type = type(streaming_component).__name__
            msg = (
                f"Component '{streaming_component_name}' of type '{component_type}' seems to not support async "
                "streaming callbacks. Use the sync 'streaming_generator' function instead, switch to a component "
                "that supports async streaming callbacks (e.g., OpenAIChatGenerator instead of OpenAIGenerator), "
                "or set allow_sync_streaming_callbacks=True to enable hybrid mode."
            )
            raise ValueError(msg)

    return (False, streaming_components)


def _setup_hybrid_streaming_callbacks_for_pipeline(
    pipeline_run_args: dict[str, Any],
    queue: asyncio.Queue[StreamingChunk],
    loop: asyncio.AbstractEventLoop,
    all_streaming_components: list[tuple[Component, str]],
    streaming_components: Optional[Union[list[str], Literal["all"]]] = None,
) -> dict[str, Any]:
    """
    Sets up hybrid streaming callbacks (sync or async) for pipeline components.

    This function creates appropriate callbacks based on each component's capabilities:
    - For async-capable components: uses async callbacks
    - For sync-only components: wraps sync callbacks for async context

    Args:
        pipeline_run_args: Arguments for pipeline execution
        queue: Asyncio queue to put chunks into
        loop: The event loop to use for thread-safe operations
        all_streaming_components: Pre-computed list of all streaming components
        streaming_components: Optional config for which components should stream

    Returns:
        Updated pipeline run arguments
    """
    if streaming_components is None:
        streaming_components = _parse_streaming_components_setting(settings.streaming_components)

    components_to_stream = []

    if streaming_components == "all":
        components_to_stream = all_streaming_components
        log.trace(
            "Hybrid streaming enabled for all components via 'all' keyword '{}'",
            components_to_stream,
        )
    elif streaming_components is None:
        if all_streaming_components:
            components_to_stream = [all_streaming_components[-1]]
            log.trace(
                "Hybrid streaming enabled for last component only '{}'",
                components_to_stream,
            )
    elif isinstance(streaming_components, list):
        enabled_component_names = set(streaming_components)
        for component, component_name in all_streaming_components:
            if component_name in enabled_component_names:
                components_to_stream.append((component, component_name))
        log.trace(
            "Hybrid streaming enabled for components '{}'",
            [name for _, name in components_to_stream],
        )

    for component, component_name in components_to_stream:
        streaming_callback = _create_hybrid_streaming_callback(queue, component, component_name, loop)

        # Copy component args to avoid mutating the original
        if component_name not in pipeline_run_args:
            pipeline_run_args[component_name] = {}
        else:
            pipeline_run_args[component_name] = pipeline_run_args[component_name].copy()

        pipeline_run_args[component_name]["streaming_callback"] = streaming_callback
        log.trace("Hybrid streaming callback set for component '{}'", component_name)

    return pipeline_run_args


async def _execute_pipeline_async(
    pipeline: Union[Pipeline, AsyncPipeline, Agent],
    pipeline_run_args: dict[str, Any],
    include_outputs_from: Optional[set[str]] = None,
) -> asyncio.Task:
    """
    Creates and returns an async task for pipeline execution.

    Args:
        pipeline: The pipeline or agent to execute
        pipeline_run_args: Execution arguments
        include_outputs_from: Optional set of component names to include outputs from (Pipeline/AsyncPipeline only)

    Returns:
        Async task for pipeline execution
    """
    if isinstance(pipeline, Agent):
        return asyncio.create_task(pipeline.run_async(**pipeline_run_args))

    kwargs: dict[str, Any] = {"data": pipeline_run_args}
    if include_outputs_from is not None:
        kwargs["include_outputs_from"] = include_outputs_from

    if isinstance(pipeline, AsyncPipeline):
        return asyncio.create_task(pipeline.run_async(**kwargs))
    else:  # Regular Pipeline
        return asyncio.create_task(asyncio.to_thread(pipeline.run, **kwargs))


async def _stream_chunks_from_queue(
    queue: asyncio.Queue[StreamingChunk], pipeline_task: asyncio.Task
) -> AsyncGenerator[StreamingChunk, None]:
    """
    Streams chunks from the queue while the pipeline is running.

    Args:
        queue: Queue containing streaming chunks
        pipeline_task: The async task running the pipeline

    Yields:
        StreamingChunk: Individual chunks from the pipeline
    """
    while not pipeline_task.done() or not queue.empty():
        # Check for pipeline completion with exception
        if pipeline_task.done():
            exception = pipeline_task.exception()
            if exception is not None:
                raise exception

        try:
            chunk = await asyncio.wait_for(queue.get(), timeout=0.1)
            yield chunk
        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            log.warning("Async streaming generator was cancelled")
            break
        except Exception as e:
            log.error("Unexpected error in async streaming generator: {}", e, exc_info=True)
            raise e


async def _cleanup_pipeline_task(pipeline_task: asyncio.Task) -> None:
    """
    Cleans up the pipeline task if it's still running.

    Args:
        pipeline_task: The task to clean up
    """
    if not pipeline_task.done():
        pipeline_task.cancel()
        try:
            await asyncio.wait_for(pipeline_task, timeout=1.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        except Exception as e:
            log.warning("Error during pipeline task cleanup: {}", e, exc_info=True)
            raise e


async def async_streaming_generator(  # noqa: PLR0913
    pipeline: Union[Pipeline, AsyncPipeline, Agent],
    *,
    pipeline_run_args: Optional[dict[str, Any]] = None,
    on_tool_call_start: OnToolCallStart = None,
    on_tool_call_end: OnToolCallEnd = None,
    on_pipeline_end: OnPipelineEnd = None,
    streaming_components: Optional[Union[list[str], Literal["all"]]] = None,
    include_outputs_from: Optional[set[str]] = None,
    allow_sync_streaming_callbacks: bool = False,
) -> AsyncGenerator[Union[StreamingChunk, OpenWebUIEvent, str], None]:
    """
    Creates an async generator that yields streaming chunks from a pipeline or agent execution.

    By default, only the last streaming-capable component in pipelines will stream.
    You can control which components stream using streaming_components or HAYHOOKS_STREAMING_COMPONENTS.

    Args:
        pipeline: The Pipeline, AsyncPipeline, or Agent to execute
        pipeline_run_args: Arguments for execution
        on_tool_call_start: Callback for tool call start
        on_tool_call_end: Callback for tool call end
        on_pipeline_end: Callback for pipeline end
        streaming_components: Optional config for which components should stream.
                             Can be:
                             - None: use HAYHOOKS_STREAMING_COMPONENTS or default (last only)
                             - "all": stream all capable components
                             - list[str]: ["llm_1", "llm_2"] to enable specific components
        include_outputs_from: Optional set of component names to include outputs from (Pipeline/AsyncPipeline only)
        allow_sync_streaming_callbacks: Controls hybrid streaming mode:
                                       - False (default): Strict mode - all components must support async callbacks
                                       - True: Automatically detect and enable hybrid mode only if needed
                                       When True, the system automatically detects components with sync-only
                                       streaming callbacks (e.g., OpenAIGenerator) and enables hybrid mode to
                                       bridge them to work in async pipelines. If all components support async,
                                       no bridging is applied (pure async mode).

    Yields:
        StreamingChunk: Individual chunks from the streaming execution
        OpenWebUIEvent: Event for tool call
        str: Tool name or stream content

    NOTE: This generator works with sync/async pipelines and agents. For pipelines, the streaming components
          should support an _async_ `streaming_callback`. However, if allow_sync_streaming_callbacks=True,
          components with only sync callbacks (e.g., OpenAIGenerator) will also work by automatically
          enabling hybrid mode when needed. Agents have built-in async streaming support. By default, only
          the last streaming-capable component will stream.
    """
    if pipeline_run_args is None:
        pipeline_run_args = {}

    use_hybrid_mode = False
    all_streaming_components: list[tuple[Component, str]] = []

    if isinstance(pipeline, (AsyncPipeline, Pipeline)):
        use_hybrid_mode, all_streaming_components = _validate_async_streaming_support(
            pipeline, allow_sync_streaming_callbacks
        )

    queue: asyncio.Queue[StreamingChunk] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    if use_hybrid_mode:
        configured_args = _setup_hybrid_streaming_callbacks_for_pipeline(
            pipeline_run_args, queue, loop, all_streaming_components, streaming_components
        )
    else:

        async def streaming_callback(chunk: StreamingChunk) -> None:
            await queue.put(chunk)

        configured_args = _setup_streaming_callback(
            pipeline, pipeline_run_args, streaming_callback, streaming_components
        )

    pipeline_task = await _execute_pipeline_async(pipeline, configured_args, include_outputs_from)

    try:
        async for chunk in _stream_chunks_from_queue(queue, pipeline_task):
            for result in _process_tool_call_start(chunk, on_tool_call_start):
                yield result
            for result in _process_tool_call_end(chunk, on_tool_call_end):
                yield result
            yield chunk

        await pipeline_task
        final_chunk = _process_pipeline_end(pipeline_task.result(), on_pipeline_end)
        if final_chunk:
            yield final_chunk

    except Exception as e:
        log.error("Unexpected error in async streaming generator: {}", e, exc_info=True)
        raise e
    finally:
        await _cleanup_pipeline_task(pipeline_task)
