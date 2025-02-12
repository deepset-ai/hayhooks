import threading
from queue import Queue
from typing import Generator, List, Union, Dict, Tuple
from haystack import Pipeline
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
    queue: Queue[str] = Queue()

    def streaming_callback(chunk):
        queue.put(chunk.content)

    _, streaming_component_name = find_streaming_component(pipeline)
    pipeline_run_args = pipeline_run_args.copy()

    if streaming_component_name not in pipeline_run_args:
        pipeline_run_args[streaming_component_name] = {}

    pipeline_run_args[streaming_component_name]["streaming_callback"] = streaming_callback
    log.trace(f"Streaming pipeline run args: {pipeline_run_args}")

    def run_pipeline():
        try:
            pipeline.run(pipeline_run_args)
        finally:
            queue.put(None)

    thread = threading.Thread(target=run_pipeline)
    thread.start()

    while True:
        chunk = queue.get()
        if chunk is None:
            break
        yield chunk

    thread.join()
