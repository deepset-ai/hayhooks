from hayhooks import AsyncPipeline, Pipeline
from hayhooks.server.utils.haystack_compat import Pipeline as InternalPipeline


def test_pipeline_is_public_compat_alias() -> None:
    assert Pipeline is InternalPipeline
    assert AsyncPipeline is Pipeline
