import asyncio
import threading
from queue import Queue
from typing import AsyncGenerator, Generator, List, Union, Dict, Tuple
from haystack import AsyncPipeline, Pipeline
from haystack.core.component import Component
from hayhooks.server.logger import log
from hayhooks.server.routers.openai import Message


def is_user_message(msg: Union[Message, Dict]) -> bool:
    if isinstance(msg, Message):
        return msg.role == "user"
    return msg.get("role") == "user"


def get_content(msg: Union[Message, Dict]) -> str:
    if isinstance(msg, Message):
        return msg.content
    return msg.get("content", "")


def get_last_user_message(messages: List[Union[Message, Dict]]) -> Union[str, None]:
    user_messages = (msg for msg in reversed(messages) if is_user_message(msg))

    for message in user_messages:
        return get_content(message)

    return None


def find_streaming_component(pipeline: Pipeline) -> Tuple[Component, str]:
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
        raise ValueError("No streaming-capable component found in the pipeline")

    return streaming_component, streaming_component_name


def streaming_generator(pipeline: Pipeline, pipeline_run_args: Dict) -> Generator:
    """
    Creates a generator that yields streaming chunks from a pipeline execution.
    Automatically finds the streaming-capable component in the pipeline.
    """
    queue: Queue[Union[str, None, Exception]] = Queue()

    def streaming_callback(chunk):
        queue.put(chunk.content)

    _, streaming_component_name = find_streaming_component(pipeline)
    pipeline_run_args = pipeline_run_args.copy()

    if streaming_component_name not in pipeline_run_args:
        pipeline_run_args[streaming_component_name] = {}

    streaming_component = pipeline.get_component(streaming_component_name)
    streaming_component.streaming_callback = streaming_callback
    log.trace(f"Streaming pipeline run args: {pipeline_run_args}")

    def run_pipeline():
        try:
            pipeline.run(data=pipeline_run_args)
            queue.put(None)
        except Exception as e:
            log.error(f"Error in pipeline execution thread for streaming_generator: {e}", exc_info=True)
            queue.put(e)

    thread = threading.Thread(target=run_pipeline)
    thread.start()

    try:
        while True:
            item = queue.get()
            if isinstance(item, Exception):
                raise item
            if item is None:
                break
            yield item
    finally:
        thread.join()


async def async_streaming_generator(
    pipeline: Union[Pipeline, AsyncPipeline], pipeline_run_args: Dict
) -> AsyncGenerator:
    """
    Creates an async generator that yields streaming chunks from a pipeline execution.
    Automatically finds the streaming-capable component in the pipeline.
    """

    queue: asyncio.Queue = asyncio.Queue()

    async def streaming_callback(chunk):
        await queue.put(chunk.content)

    _, streaming_component_name = find_streaming_component(pipeline)
    pipeline_run_args = pipeline_run_args.copy()

    if streaming_component_name not in pipeline_run_args:
        pipeline_run_args[streaming_component_name] = {}

    streaming_component = pipeline.get_component(streaming_component_name)
    streaming_component.streaming_callback = streaming_callback

    pipeline_task = asyncio.create_task(pipeline.run_async(data=pipeline_run_args))

    try:
        while not pipeline_task.done() or not queue.empty():
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
    except Exception as e:
        log.error(f"Unexpected error in async streaming generator: {e}")
        raise e
    finally:
        if not pipeline_task.done():
            pipeline_task.cancel()
            try:
                await asyncio.wait_for(pipeline_task, timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
            except Exception as e:
                log.warning(f"Error during pipeline task cleanup: {e}")
                raise e
