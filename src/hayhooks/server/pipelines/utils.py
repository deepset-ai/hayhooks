import asyncio
import threading
from collections.abc import AsyncGenerator, Generator
from queue import Queue
from typing import Any, Callable, Optional, Union

from haystack import AsyncPipeline, Pipeline
from haystack.components.agents import Agent
from haystack.core.component import Component
from haystack.dataclasses import StreamingChunk

from hayhooks.open_webui import OpenWebUIEvent
from hayhooks.server.logger import log
from hayhooks.server.routers.openai import Message

ToolCallbackReturn = Union[OpenWebUIEvent, str, None, list[Union[OpenWebUIEvent, str]]]
OnToolCallStart = Optional[Callable[[str, Optional[str], Optional[str]], ToolCallbackReturn]]
OnToolCallEnd = Optional[Callable[[str, dict[str, Any], str, bool], ToolCallbackReturn]]


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


def find_streaming_component(pipeline: Union[Pipeline, AsyncPipeline]) -> tuple[Component, str]:
    """
    Finds the component in the pipeline that supports streaming_callback

    Returns:
        The first component that supports streaming
    """
    streaming_component = None
    streaming_component_name = ""

    for name, component in pipeline.walk():
        if hasattr(component, "streaming_callback"):
            log.trace(f"Streaming component found in '{name}' with type {type(component)}")
            streaming_component = component
            streaming_component_name = name
    if not streaming_component:
        msg = "No streaming-capable component found in the pipeline"
        raise ValueError(msg)

    return streaming_component, streaming_component_name


def _setup_streaming_callback_for_pipeline(
    pipeline: Union[Pipeline, AsyncPipeline], pipeline_run_args: dict[str, Any], streaming_callback: Any
) -> dict[str, Any]:
    """
    Sets up streaming callback for pipeline components.

    Args:
        pipeline: The pipeline to configure
        pipeline_run_args: Arguments for pipeline execution
        streaming_callback: The callback function to set

    Returns:
        Updated pipeline run arguments
    """
    _, streaming_component_name = find_streaming_component(pipeline)

    # Ensure component args exist in pipeline run args
    if streaming_component_name not in pipeline_run_args:
        pipeline_run_args[streaming_component_name] = {}

    # Set the streaming callback on the component
    streaming_component = pipeline.get_component(streaming_component_name)
    assert hasattr(streaming_component, "streaming_callback")
    streaming_component.streaming_callback = streaming_callback

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
    pipeline: Union[Pipeline, AsyncPipeline, Agent], pipeline_run_args: dict[str, Any], streaming_callback: Any
) -> dict[str, Any]:
    """
    Configures streaming callback for the given pipeline or agent.

    Args:
        pipeline: The pipeline or agent to configure
        pipeline_run_args: Execution arguments
        streaming_callback: The callback function

    Returns:
        Updated pipeline run arguments
    """
    pipeline_run_args = pipeline_run_args.copy()

    if isinstance(pipeline, (Pipeline, AsyncPipeline)):
        return _setup_streaming_callback_for_pipeline(pipeline, pipeline_run_args, streaming_callback)
    elif isinstance(pipeline, Agent):
        return _setup_streaming_callback_for_agent(pipeline_run_args, streaming_callback)
    else:
        msg = f"Unsupported pipeline type: {type(pipeline)}"
        raise ValueError(msg)


def _execute_pipeline_sync(pipeline: Union[Pipeline, AsyncPipeline, Agent], pipeline_run_args: dict[str, Any]) -> None:
    """
    Executes pipeline synchronously based on its type.

    Args:
        pipeline: The pipeline or agent to execute
        pipeline_run_args: Execution arguments
    """
    if isinstance(pipeline, Agent):
        pipeline.run(**pipeline_run_args)
    else:
        pipeline.run(data=pipeline_run_args)


def streaming_generator(  # noqa: C901, PLR0912
    pipeline: Union[Pipeline, AsyncPipeline, Agent],
    *,
    pipeline_run_args: Optional[dict[str, Any]] = None,
    on_tool_call_start: OnToolCallStart = None,
    on_tool_call_end: OnToolCallEnd = None,
) -> Generator[Union[StreamingChunk, OpenWebUIEvent, str], None, None]:
    """
    Creates a generator that yields streaming chunks from a pipeline or agent execution.

    Automatically finds the streaming-capable component in pipelines or uses the agent's streaming callback.

    Args:
        pipeline: The Pipeline, AsyncPipeline, or Agent to execute
        pipeline_run_args: Arguments for execution
        on_tool_call_start: Callback for tool call start
        on_tool_call_end: Callback for tool call end

    Yields:
        StreamingChunk: Individual chunks from the streaming execution
        OpenWebUIEvent: Event for tool call
        str: Tool name or stream content

    NOTE: This generator works with sync/async pipelines and agents, but pipeline components
          which support streaming must have a _sync_ `streaming_callback`.
    """
    if pipeline_run_args is None:
        pipeline_run_args = {}
    queue: Queue[Union[StreamingChunk, None, Exception]] = Queue()

    def streaming_callback(chunk: StreamingChunk) -> None:
        queue.put(chunk)

    # Configure streaming callback
    configured_args = _setup_streaming_callback(pipeline, pipeline_run_args, streaming_callback)
    log.trace(f"Streaming pipeline run args: {configured_args}")

    def run_pipeline() -> None:
        try:
            _execute_pipeline_sync(pipeline, configured_args)
            queue.put(None)  # Signal completion
        except Exception as e:
            log.error(f"Error in pipeline execution thread for streaming_generator: {e}", exc_info=True)
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
            if on_tool_call_start and hasattr(item, "tool_calls") and item.tool_calls:
                for tool_call in item.tool_calls:
                    if tool_call.tool_name:
                        res = on_tool_call_start(tool_call.tool_name, tool_call.arguments, tool_call.id)
                        if res:
                            if isinstance(res, list):
                                for r in res:
                                    yield r
                            else:
                                yield res

            if on_tool_call_end and hasattr(item, "tool_call_result") and item.tool_call_result:
                res = on_tool_call_end(
                    item.tool_call_result.origin.tool_name,
                    item.tool_call_result.origin.arguments,
                    item.tool_call_result.result,
                    bool(item.tool_call_result.error),
                )
                if res:
                    if isinstance(res, list):
                        for r in res:
                            yield r
                    else:
                        yield res
            yield item
    finally:
        thread.join()


def _validate_async_streaming_support(pipeline: Union[Pipeline, AsyncPipeline]) -> None:
    """
    Validates that the pipeline supports async streaming callbacks.

    Args:
        pipeline: The pipeline to validate

    Raises:
        ValueError: If the pipeline doesn't support async streaming
    """
    streaming_component, streaming_component_name = find_streaming_component(pipeline)

    # Check if the streaming component supports async streaming callbacks
    # We check for run_async method as an indicator of async support
    if not hasattr(streaming_component, "run_async"):
        component_type = type(streaming_component).__name__
        msg = (
            f"Component '{streaming_component_name}' of type '{component_type}' seems to not support async streaming "
            "callbacks. Use the sync 'streaming_generator' function instead, or switch to a component that supports "
            "async streaming callbacks (e.g., OpenAIChatGenerator instead of OpenAIGenerator)."
        )
        raise ValueError(msg)


async def _execute_pipeline_async(
    pipeline: Union[Pipeline, AsyncPipeline, Agent], pipeline_run_args: dict[str, Any]
) -> asyncio.Task:
    """
    Creates and returns an async task for pipeline execution.

    Args:
        pipeline: The pipeline or agent to execute
        pipeline_run_args: Execution arguments

    Returns:
        Async task for pipeline execution
    """
    if isinstance(pipeline, AsyncPipeline):
        return asyncio.create_task(pipeline.run_async(data=pipeline_run_args))
    elif isinstance(pipeline, Agent):
        return asyncio.create_task(pipeline.run_async(**pipeline_run_args))
    else:  # Regular Pipeline
        return asyncio.create_task(asyncio.to_thread(pipeline.run, data=pipeline_run_args))


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
            log.error(f"Unexpected error in async streaming generator: {e}")
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
            log.warning(f"Error during pipeline task cleanup: {e}")
            raise e


async def async_streaming_generator(  # noqa: C901, PLR0912
    pipeline: Union[Pipeline, AsyncPipeline, Agent],
    *,
    pipeline_run_args: Optional[dict[str, Any]] = None,
    on_tool_call_start: OnToolCallStart = None,
    on_tool_call_end: OnToolCallEnd = None,
) -> AsyncGenerator[Union[StreamingChunk, OpenWebUIEvent, str], None]:
    """
    Creates an async generator that yields streaming chunks from a pipeline or agent execution.

    Automatically finds the streaming-capable component in pipelines or uses the agent's streaming callback.

    Args:
        pipeline: The Pipeline, AsyncPipeline, or Agent to execute
        pipeline_run_args: Arguments for execution
        on_tool_call_start: Callback for tool call start
        on_tool_call_end: Callback for tool call end

    Yields:
        StreamingChunk: Individual chunks from the streaming execution
        OpenWebUIEvent: Event for tool call
        str: Tool name or stream content

    NOTE: This generator works with sync/async pipelines and agents. For pipelines, the streaming component
          must support an _async_ `streaming_callback`. Agents have built-in async streaming support.
    """
    # Validate async streaming support for pipelines (not needed for agents)
    if pipeline_run_args is None:
        pipeline_run_args = {}
    if isinstance(pipeline, (AsyncPipeline, Pipeline)):
        _validate_async_streaming_support(pipeline)

    # Create async queue and streaming callback
    queue: asyncio.Queue[StreamingChunk] = asyncio.Queue()

    async def streaming_callback(chunk: StreamingChunk) -> None:
        await queue.put(chunk)

    # Configure streaming callback
    configured_args = _setup_streaming_callback(pipeline, pipeline_run_args, streaming_callback)

    # Start pipeline execution
    pipeline_task = await _execute_pipeline_async(pipeline, configured_args)

    try:
        async for chunk in _stream_chunks_from_queue(queue, pipeline_task):
            # Handle tool calls
            if on_tool_call_start and hasattr(chunk, "tool_calls") and chunk.tool_calls:
                for tool_call in chunk.tool_calls:
                    if tool_call.tool_name:
                        res = on_tool_call_start(tool_call.tool_name, tool_call.arguments, tool_call.id)
                        if res:
                            if isinstance(res, list):
                                for r in res:
                                    yield r
                            else:
                                yield res

            if on_tool_call_end and hasattr(chunk, "tool_call_result") and chunk.tool_call_result:
                res = on_tool_call_end(
                    chunk.tool_call_result.origin.tool_name,
                    chunk.tool_call_result.origin.arguments,
                    chunk.tool_call_result.result,
                    bool(chunk.tool_call_result.error),
                )
                if res:
                    if isinstance(res, list):
                        for r in res:
                            yield r
                    else:
                        yield res
            yield chunk

        await pipeline_task

    except Exception as e:
        log.error(f"Unexpected error in async streaming generator: {e}")
        raise e
    finally:
        await _cleanup_pipeline_task(pipeline_task)
