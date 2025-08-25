from pathlib import Path

import pytest
from haystack import Pipeline

from hayhooks.server.pipelines import registry
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper


@pytest.mark.skip(reason="Requests to https://mermaid.ink seem to be failing")
def test_draw_pipeline(client, deploy_pipeline, draw_pipeline):
    pipeline_file = Path(__file__).parent / "test_files/yaml" / "working_pipelines/test_pipeline_01.yml"
    pipeline_data = {"name": pipeline_file.stem, "source_code": pipeline_file.read_text()}

    deploy_pipeline(client, pipeline_data["name"], pipeline_data["source_code"])

    draw_response = draw_pipeline(client, pipeline_data["name"])
    assert draw_response.status_code == 200

    assert draw_response.headers["Content-Type"] == "image/png"
    assert len(draw_response.content) > 0


def test_draw_non_existent_pipeline(client, draw_pipeline):
    draw_response = draw_pipeline(client, "non_existent_pipeline")
    assert draw_response.status_code == 404


@pytest.mark.skip(reason="Requests to https://mermaid.ink seem to be failing")
def test_draw_pipeline_wrapper(client, deploy_pipeline, draw_pipeline):
    class TestPipelineWrapper(BasePipelineWrapper):
        def setup(self) -> None:
            pipeline_file = Path(__file__).parent / "test_files/yaml" / "working_pipelines/test_pipeline_01.yml"
            self.pipeline = Pipeline.loads(pipeline_file.read_text())

        def run_api(self, urls: list[str], question: str) -> dict:
            return {}

        def run_chat_completion(self, model: str, messages: list[dict], body: dict) -> dict:
            return {}

    wrapper = TestPipelineWrapper()
    wrapper.setup()
    registry.add("test_wrapper", wrapper)

    draw_response = draw_pipeline(client, "test_wrapper")
    assert draw_response.status_code == 200
    assert draw_response.headers["Content-Type"] == "image/png"
    assert len(draw_response.content) > 0
