import tempfile
from pathlib import Path
from typing import List, Optional
from fastapi import UploadFile
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        from haystack.components.writers import DocumentWriter
        from haystack.components.converters import (
            MarkdownToDocument,
            PyPDFToDocument,
            TextFileToDocument,
        )
        from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
        from haystack.components.routers import FileTypeRouter
        from haystack.components.joiners import DocumentJoiner
        from haystack.components.embedders import SentenceTransformersDocumentEmbedder
        from haystack import Pipeline
        from haystack.document_stores.types import DuplicatePolicy
        from haystack_integrations.document_stores.elasticsearch import (
            ElasticsearchDocumentStore,
        )

        document_store = ElasticsearchDocumentStore(hosts="http://localhost:9200")
        file_type_router = FileTypeRouter(mime_types=["text/plain", "application/pdf", "text/markdown"])
        text_file_converter = TextFileToDocument()
        markdown_converter = MarkdownToDocument()
        pdf_converter = PyPDFToDocument()
        document_joiner = DocumentJoiner()

        document_cleaner = DocumentCleaner()
        document_splitter = DocumentSplitter(split_by="word", split_length=150, split_overlap=50)

        document_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
        document_writer = DocumentWriter(document_store, policy=DuplicatePolicy.OVERWRITE)

        pipe = Pipeline()
        pipe.add_component(instance=file_type_router, name="file_type_router")
        pipe.add_component(instance=text_file_converter, name="text_file_converter")
        pipe.add_component(instance=markdown_converter, name="markdown_converter")
        pipe.add_component(instance=pdf_converter, name="pypdf_converter")
        pipe.add_component(instance=document_joiner, name="document_joiner")
        pipe.add_component(instance=document_cleaner, name="document_cleaner")
        pipe.add_component(instance=document_splitter, name="document_splitter")
        pipe.add_component(instance=document_embedder, name="document_embedder")
        pipe.add_component(instance=document_writer, name="document_writer")

        pipe.connect("file_type_router.text/plain", "text_file_converter.sources")
        pipe.connect("file_type_router.application/pdf", "pypdf_converter.sources")
        pipe.connect("file_type_router.text/markdown", "markdown_converter.sources")
        pipe.connect("text_file_converter", "document_joiner")
        pipe.connect("pypdf_converter", "document_joiner")
        pipe.connect("markdown_converter", "document_joiner")
        pipe.connect("document_joiner", "document_cleaner")
        pipe.connect("document_cleaner", "document_splitter")
        pipe.connect("document_splitter", "document_embedder")
        pipe.connect("document_embedder", "document_writer")

        self.pipeline = pipe

    def run_api(self, files: Optional[List[UploadFile]] = None) -> dict:
        try:
            if files:
                with tempfile.TemporaryDirectory() as temp_dir:
                    for file in files:
                        file_path = Path(temp_dir) / file.filename
                        file_path.write_bytes(file.file.read())
                        self.pipeline.run({"file_type_router": {"sources": [file_path]}})

                return {"message": f"Files indexed successfully: {[file.filename for file in files]}"}
            else:
                return {"message": "No files provided"}
        except Exception as e:
            return {"message": f"Error indexing files: {e}"}
