from haystack import Pipeline

from hayhooks import BasePipelineWrapper, get_last_user_input_text, log


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        self.pipeline = Pipeline()

    def run_response(self, model: str, input_items: list[dict], body: dict) -> str:
        log.trace(
            "Running pipeline with model: {}, input_items: {}, body: {}",
            model,
            input_items,
            body,
        )

        question = get_last_user_input_text(input_items)
        log.trace("Question: {}", question)

        # NOTE: This is used in tests, please don't change it
        return "This is a mock response from the pipeline"
