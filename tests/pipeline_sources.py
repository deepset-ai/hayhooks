"""Pipeline sources and a directory writer shared by the durable-mode tests."""

from importlib.metadata import version
from pathlib import Path

try:
    _HAYSTACK_VERSION = tuple(int(part) for part in version("haystack-ai").split(".", maxsplit=2)[:2])
except ValueError:
    _HAYSTACK_VERSION = (0, 0)
HAYSTACK_V3 = (3, 1) <= _HAYSTACK_VERSION < (4, 0)

CALC_YAML = (Path(__file__).parent / "test_files/yaml/sample_calc_pipeline.yml").read_text()

ORDINARY_WRAPPER = '''
from hayhooks import BasePipelineWrapper


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        self.pipeline = None

    def run_api(self, value: int) -> int:
        """Double a value."""
        return value * 2
'''

CHAT_WRAPPER = """
from hayhooks import BasePipelineWrapper


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        self.pipeline = None

    def run_chat_completion(self, model: str, messages: list[dict], body: dict) -> str:
        return "hello from " + model
"""

# Ordinary requests stay ordinary; /run-durable runs the durable method. Requires Haystack 3.1.
DURABLE_WRAPPER = '''
from haystack import Pipeline
from pydantic import BaseModel

from hayhooks import BasePipelineWrapper, DurableContext


class Request(BaseModel):
    value: int
    wait: bool = False


class Result(BaseModel):
    value: int


class PipelineWrapper(BasePipelineWrapper):
    durable_revision = "v1"

    def setup(self) -> None:
        self.pipeline = Pipeline()

    def run_api(self, value: int) -> int:
        """Increment a value."""
        return value + 1

    def run_durable(self, context: DurableContext, request: Request) -> Result:
        if request.wait and context.resume_input is None:
            context.suspend_sync({"kind": "approval"})
        return Result(value=request.value * 10)
'''


def write_tree(root: Path, files: dict[str, str]) -> Path:
    """Write ``{relative path: content}`` under *root* and return it."""
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return root
