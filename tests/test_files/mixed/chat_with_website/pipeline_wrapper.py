from pathlib import Path

from haystack import Pipeline

from hayhooks import BasePipelineWrapper, get_last_user_message, log

URLS = ["https://haystack.deepset.ai", "https://www.redis.io"]


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        pipeline_yaml = (Path(__file__).parent / "chat_with_website.yml").read_text()
        self.pipeline = Pipeline.loads(pipeline_yaml)

    def run_api(self, urls: list[str], question: str) -> str:
        log.trace("Running pipeline with urls: {} and question: {}", urls, question)
        result = self.pipeline.run({"fetcher": {"urls": urls}, "prompt": {"query": question}})
        return result["llm"]["replies"][0]

    def run_chat_completion(self, model: str, messages: list[dict], body: dict) -> str:
        log.trace(
            "Running pipeline with model: {}, messages: {}, body: {}",
            model,
            messages,
            body,
        )

        question = get_last_user_message(messages)
        result = self.pipeline.run({"fetcher": {"urls": URLS}, "prompt": {"query": question}})

        return result["llm"]["replies"][0]
