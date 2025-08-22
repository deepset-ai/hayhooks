from haystack import Pipeline

from hayhooks.server.logger import log
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        from haystack.components.builders import ChatPromptBuilder
        from haystack.components.embedders import SentenceTransformersTextEmbedder
        from haystack.components.generators.chat import HuggingFaceAPIChatGenerator
        from haystack.dataclasses import ChatMessage
        from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchEmbeddingRetriever
        from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore

        document_store = ElasticsearchDocumentStore(hosts="http://localhost:9200")

        template = [
            ChatMessage.from_user(
                """
        Answer the questions based on the given context.

        Context:
        {% for document in documents %}
            {{ document.content }}
        {% endfor %}

        Question: {{ question }}
        Answer:
        """
            )
        ]
        pipe = Pipeline()
        pipe.add_component(
            "embedder",
            SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"),
        )
        pipe.add_component("retriever", ElasticsearchEmbeddingRetriever(document_store=document_store))
        pipe.add_component("chat_prompt_builder", ChatPromptBuilder(template=template, required_variables="*"))
        pipe.add_component(
            "llm",
            HuggingFaceAPIChatGenerator(
                api_type="serverless_inference_api",
                api_params={"model": "Qwen/Qwen2.5-7B-Instruct", "provider": "together"},
            ),
        )

        pipe.connect("embedder.embedding", "retriever.query_embedding")
        pipe.connect("retriever", "chat_prompt_builder.documents")
        pipe.connect("chat_prompt_builder.prompt", "llm.messages")

        self.pipeline = pipe

    def run_api(self, question: str) -> str:
        log.trace(f"Running pipeline with question: {question}")
        result = self.pipeline.run(
            {
                "embedder": {"text": question},
                "chat_prompt_builder": {"question": question},
            }
        )

        return result["llm"]["replies"][0].text
