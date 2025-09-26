# RAG System Example

This example demonstrates how to build a complete Retrieval-Augmented Generation (RAG) system using Hayhooks, including document ingestion, vector storage, and question answering.

## Overview

The RAG system pipeline provides:

- Document upload and processing
- Vector embedding and storage
- Similarity-based document retrieval
- Context-aware question answering
- Multiple document format support

## Pipeline Implementation

### Complete RAG Pipeline

```python
# pipeline_wrapper.py
from typing import List, Dict, Any, Optional
from fastapi import UploadFile
from haystack import Pipeline
from haystack.components.converters import (
    PyPDFToDocument,
    TextFileToDocument,
    DocxToDocument,
    MarkdownToDocument
)
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.writers import InMemoryDocumentWriter
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.dataclasses import Document
from hayhooks import BasePipelineWrapper
import tempfile
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class RAGPipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        """Setup the RAG system pipeline"""
        # Initialize document store
        self.document_store = self._create_document_store()

        # Initialize components
        self.pdf_converter = PyPDFToDocument()
        self.text_converter = TextFileToDocument()
        self.docx_converter = DocxToDocument()
        self.markdown_converter = MarkdownToDocument()
        self.cleaner = DocumentCleaner(
            remove_empty_lines=True,
            remove_extra_whitespaces=True,
            remove_repeated_substrings=True
        )
        self.splitter = DocumentSplitter(
            split_by="word",
            split_length=300,
            split_overlap=50
        )
        self.embedder = SentenceTransformersDocumentEmbedder(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.writer = InMemoryDocumentWriter(document_store=self.document_store)
        self.text_embedder = SentenceTransformersTextEmbedder(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.retriever = InMemoryEmbeddingRetriever(
            document_store=self.document_store,
            top_k=5
        )
        self.prompt_builder = PromptBuilder(
            template="""You are a helpful assistant that answers questions based on the provided context.

Context:
{% for doc in documents %}
- {{ doc.content }}
{% endfor %}

Question: {{ query }}

Please provide a comprehensive answer based on the context above. If the context doesn't contain enough information to answer the question, please say so.

Answer:"""
        )
        self.llm = OpenAIGenerator(
            model="gpt-3.5-turbo"
        )

        # Build indexing pipeline
        self.indexing_pipeline = Pipeline()
        self.indexing_pipeline.add_component("pdf_converter", self.pdf_converter)
        self.indexing_pipeline.add_component("text_converter", self.text_converter)
        self.indexing_pipeline.add_component("docx_converter", self.docx_converter)
        self.indexing_pipeline.add_component("markdown_converter", self.markdown_converter)
        self.indexing_pipeline.add_component("cleaner", self.cleaner)
        self.indexing_pipeline.add_component("splitter", self.splitter)
        self.indexing_pipeline.add_component("embedder", self.embedder)
        self.indexing_pipeline.add_component("writer", self.writer)

        # Connect indexing pipeline
        self.indexing_pipeline.connect("pdf_converter", "cleaner")
        self.indexing_pipeline.connect("text_converter", "cleaner")
        self.indexing_pipeline.connect("docx_converter", "cleaner")
        self.indexing_pipeline.connect("markdown_converter", "cleaner")
        self.indexing_pipeline.connect("cleaner", "splitter")
        self.indexing_pipeline.connect("splitter", "embedder")
        self.indexing_pipeline.connect("embedder", "writer")

        # Build query pipeline
        self.query_pipeline = Pipeline()
        self.query_pipeline.add_component("text_embedder", self.text_embedder)
        self.query_pipeline.add_component("retriever", self.retriever)
        self.query_pipeline.add_component("prompt_builder", self.prompt_builder)
        self.query_pipeline.add_component("llm", self.llm)

        # Connect query pipeline
        self.query_pipeline.connect("text_embedder", "retriever")
        self.query_pipeline.connect("retriever", "prompt_builder.documents")
        self.query_pipeline.connect("prompt_builder", "llm")

    def _create_document_store(self):
        """Create in-memory document store"""
        from haystack.document_stores import InMemoryDocumentStore
        return InMemoryDocumentStore()

    def process_file(self, file_path: str, file_type: str) -> List[Document]:
        """Process a single file based on its type"""
        try:
            if file_type == "pdf":
                result = self.indexing_pipeline.run({"pdf_converter": {"sources": [file_path]}})
            elif file_type == "txt":
                result = self.indexing_pipeline.run({"text_converter": {"sources": [file_path]}})
            elif file_type == "docx":
                result = self.indexing_pipeline.run({"docx_converter": {"sources": [file_path]}})
            elif file_type == "md":
                result = self.indexing_pipeline.run({"markdown_converter": {"sources": [file_path]}})
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

            return result.get("writer", {}).get("documents", [])

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise

    def run_api(self, query: str = "", files: Optional[List[UploadFile]] = None) -> str:
        """Run the RAG system"""
        # Process uploaded files
        if files:
            for file in files:
                try:
                    # Save file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp_file:
                        content = file.file.read()
                        tmp_file.write(content)
                        tmp_file_path = tmp_file.name

                    # Determine file type
                    file_ext = Path(file.filename).suffix.lower()
                    file_type = file_ext[1:] if file_ext else "txt"

                    # Process the file
                    documents = self.process_file(tmp_file_path, file_type)
                    logger.info(f"Processed {file.filename}, created {len(documents)} document chunks")

                    # Clean up
                    os.unlink(tmp_file_path)

                except Exception as e:
                    logger.error(f"Error processing file {file.filename}: {str(e)}")
                    continue

        # Run the query
        if not query:
            return "Please provide a question to answer."

        try:
            result = self.query_pipeline.run({
                "text_embedder": {"text": query},
                "prompt_builder": {"query": query}
            })

            return result["llm"]["replies"][0]

        except Exception as e:
            logger.error(f"Error running query: {str(e)}")
            return f"Error processing your question: {str(e)}"

    def get_document_count(self) -> int:
        """Get the number of documents in the store"""
        return len(self.document_store.filter_documents())

    def clear_documents(self):
        """Clear all documents from the store"""
        self.document_store.delete_documents()
        logger.info("Cleared all documents from the store")
```

### Enhanced RAG with Metadata

```python
# enhanced_rag_pipeline.py
from typing import List, Dict, Any, Optional
from fastapi import UploadFile
from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument, TextFileToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.writers import InMemoryDocumentWriter
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIChatGenerator
from haystack.components.routers import ConditionalRouter
from hayhooks import BasePipelineWrapper, get_last_user_message
import tempfile
import os
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

class EnhancedRAGPipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        """Setup enhanced RAG system with metadata and filtering"""
        # Initialize document store
        self.document_store = self._create_document_store()

        # Initialize components
        self.pdf_converter = PyPDFToDocument()
        self.text_converter = TextFileToDocument()
        self.cleaner = DocumentCleaner()
        self.splitter = DocumentSplitter(
            split_by="word",
            split_length=300,
            split_overlap=50
        )
        self.embedder = SentenceTransformersDocumentEmbedder(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.writer = InMemoryDocumentWriter(document_store=self.document_store)
        self.text_embedder = SentenceTransformersTextEmbedder(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.retriever = InMemoryEmbeddingRetriever(
            document_store=self.document_store,
            top_k=5
        )
        self.prompt_builder = PromptBuilder(
            template="""You are a helpful assistant that answers questions based on the provided context.

Context Information:
{% for doc in documents %}
Document: {{ doc.meta.file_name }}
Type: {{ doc.meta.file_type }}
Uploaded: {{ doc.meta.upload_date }}

Content:
{{ doc.content }}
---
{% endfor %}

Question: {{ query }}
{% if metadata %}
Additional Context: {{ metadata }}
{% endif %}

Please provide a comprehensive answer based on the context above. If the context doesn't contain enough information to answer the question, please say so.

Answer:"""
        )
        self.llm = OpenAIChatGenerator(
            model="gpt-4",
            streaming_callback=lambda x: None
        )

        # Build indexing pipeline
        self.indexing_pipeline = Pipeline()
        self.indexing_pipeline.add_component("pdf_converter", self.pdf_converter)
        self.indexing_pipeline.add_component("text_converter", self.text_converter)
        self.indexing_pipeline.add_component("cleaner", self.cleaner)
        self.indexing_pipeline.add_component("splitter", self.splitter)
        self.indexing_pipeline.add_component("embedder", self.embedder)
        self.indexing_pipeline.add_component("writer", self.writer)

        # Connect indexing pipeline
        self.indexing_pipeline.connect("pdf_converter", "cleaner")
        self.indexing_pipeline.connect("text_converter", "cleaner")
        self.indexing_pipeline.connect("cleaner", "splitter")
        self.indexing_pipeline.connect("splitter", "embedder")
        self.indexing_pipeline.connect("embedder", "writer")

        # Build query pipeline
        self.query_pipeline = Pipeline()
        self.query_pipeline.add_component("text_embedder", self.text_embedder)
        self.query_pipeline.add_component("retriever", self.retriever)
        self.query_pipeline.add_component("prompt_builder", self.prompt_builder)
        self.query_pipeline.add_component("llm", self.llm)

        # Connect query pipeline
        self.query_pipeline.connect("text_embedder", "retriever")
        self.query_pipeline.connect("retriever", "prompt_builder.documents")
        self.query_pipeline.connect("prompt_builder", "llm")

    def _create_document_store(self):
        """Create in-memory document store"""
        from haystack.document_stores import InMemoryDocumentStore
        return InMemoryDocumentStore()

    def process_file(self, file_path: str, file_name: str, file_type: str) -> List[Document]:
        """Process a single file with metadata"""
        try:
            if file_type == "pdf":
                result = self.indexing_pipeline.run({"pdf_converter": {"sources": [file_path]}})
            elif file_type == "txt":
                result = self.indexing_pipeline.run({"text_converter": {"sources": [file_path]}})
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

            # Add metadata to documents
            documents = result.get("writer", {}).get("documents", [])
            for doc in documents:
                doc.meta.update({
                    "file_name": file_name,
                    "file_type": file_type,
                    "upload_date": str(Path(file_path).stat().st_ctime),
                    "file_size": Path(file_path).stat().st_size
                })

            return documents

        except Exception as e:
            logger.error(f"Error processing file {file_name}: {str(e)}")
            raise

    def run_api(self, query: str = "", files: Optional[List[UploadFile]] = None, metadata: Optional[str] = None) -> str:
        """Run the enhanced RAG system"""
        # Process uploaded files
        processed_files = []
        if files:
            for file in files:
                try:
                    # Save file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp_file:
                        content = file.file.read()
                        tmp_file.write(content)
                        tmp_file_path = tmp_file.name

                    # Determine file type
                    file_ext = Path(file.filename).suffix.lower()
                    file_type = file_ext[1:] if file_ext else "txt"

                    # Process the file
                    documents = self.process_file(tmp_file_path, file.filename, file_type)
                    processed_files.append({
                        "filename": file.filename,
                        "chunks": len(documents),
                        "status": "success"
                    })

                    logger.info(f"Processed {file.filename}, created {len(documents)} document chunks")

                    # Clean up
                    os.unlink(tmp_file_path)

                except Exception as e:
                    processed_files.append({
                        "filename": file.filename,
                        "status": "error",
                        "error": str(e)
                    })
                    logger.error(f"Error processing file {file.filename}: {str(e)}")
                    continue

        # Run the query
        if not query:
            return json.dumps({
                "status": "ready",
                "processed_files": processed_files,
                "document_count": self.get_document_count(),
                "message": "Please provide a question to answer."
            })

        try:
            result = self.query_pipeline.run({
                "text_embedder": {"text": query},
                "prompt_builder": {"query": query, "metadata": metadata or ""}
            })

            response = {
                "answer": result["llm"]["replies"][0].content,
                "processed_files": processed_files,
                "retrieved_documents": len(result["retriever"]["documents"]),
                "document_count": self.get_document_count()
            }

            return json.dumps(response, indent=2)

        except Exception as e:
            logger.error(f"Error running query: {str(e)}")
            return json.dumps({
                "error": str(e),
                "processed_files": processed_files
            })

    def run_chat_completion(self, model: str, messages: List[dict], body: dict) -> str:
        """Run chat completion with RAG support"""
        question = get_last_user_message(messages)

        # Extract files from body if present
        files = body.get("files", [])
        metadata = body.get("metadata", "")

        # Process files
        if files:
            processed_files = []
            for file_info in files:
                try:
                    # Extract file content from base64 or path
                    file_content = file_info.get("content", "")
                    file_name = file_info.get("filename", "unknown")
                    file_type = file_info.get("type", "txt")

                    # Save and process file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file_name}") as tmp_file:
                        if file_content:
                            # Handle base64 content if needed
                            import base64
                            try:
                                decoded_content = base64.b64decode(file_content)
                                tmp_file.write(decoded_content)
                            except:
                                tmp_file.write(file_content.encode())
                        tmp_file_path = tmp_file.name

                    documents = self.process_file(tmp_file_path, file_name, file_type)
                    processed_files.append({
                        "filename": file_name,
                        "chunks": len(documents),
                        "status": "success"
                    })

                    os.unlink(tmp_file_path)

                except Exception as e:
                    processed_files.append({
                        "filename": file_name,
                        "status": "error",
                        "error": str(e)
                    })

        # Run the query
        result = self.query_pipeline.run({
            "text_embedder": {"text": question},
            "prompt_builder": {"query": question, "metadata": metadata}
        })

        return result["llm"]["replies"][0].content

    def get_document_count(self) -> int:
        """Get the number of documents in the store"""
        return len(self.document_store.filter_documents())

    def clear_documents(self):
        """Clear all documents from the store"""
        self.document_store.delete_documents()
        logger.info("Cleared all documents from the store")

    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about the document store"""
        documents = self.document_store.filter_documents()
        stats = {
            "total_documents": len(documents),
            "file_types": {},
            "total_size": 0
        }

        for doc in documents:
            file_type = doc.meta.get("file_type", "unknown")
            file_size = doc.meta.get("file_size", 0)

            stats["file_types"][file_type] = stats["file_types"].get(file_type, 0) + 1
            stats["total_size"] += file_size

        return stats
```

## Deployment

### Using the CLI

```bash
# Deploy the basic RAG pipeline
hayhooks pipeline deploy-files -n rag-system ./rag-system --description "Basic RAG system for document Q&A"

# Deploy the enhanced RAG pipeline
hayhooks pipeline deploy-files -n rag-system-enhanced ./rag-system-enhanced --description "Enhanced RAG system with metadata and filtering"
```

### Using YAML Configuration

```yaml
# rag-system.yml
name: rag-system
description: "Retrieval-Augmented Generation system"
inputs:
  - name: query
    type: str
    description: "Question to answer"
  - name: files
    type: List[UploadFile]
    description: "Documents to process"
  - name: metadata
    type: str
    description: "Additional metadata context"
outputs:
  - name: answer
    type: str
    description: "Generated answer with context"
```

```bash
# Deploy from YAML
hayhooks pipeline deploy-yaml -n rag-system rag-system.yml
```

## Usage Examples

### API Usage

```bash
# Upload documents and ask questions
curl -X POST http://localhost:1416/rag-system/run \
  -F 'files=@document1.pdf' \
  -F 'files=@document2.txt' \
  -F 'query="What are the main points discussed in these documents?"'

# Get system status
curl -X POST http://localhost:1416/rag-system/run \
  -H 'Content-Type: application/json' \
  -d '{"query": ""}'

# Clear documents
curl -X POST http://localhost:1416/rag-system/clear
```

### CLI Usage

```bash
# Upload documents and ask question
hayhooks pipeline run rag-system --file document.pdf --file notes.txt --param 'query="Summarize the key information"'

# Check document count
hayhooks pipeline run rag-system --param 'query=""'
```

### OpenWebUI Integration

```python
# Use with OpenWebUI for file upload support
curl -X POST http://localhost:1416/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "rag-system-enhanced",
    "messages": [
      {
        "role": "user",
        "content": "What is the main topic of the uploaded document?"
      }
    ],
    "files": [
      {
        "filename": "document.pdf",
        "content": "base64-encoded-content",
        "type": "pdf"
      }
    ]
  }'
```

## Testing

### Test Script

```python
# test_rag_system.py
import requests
import json
import base64

def test_rag_system():
    """Test the RAG system"""
    # Test document upload
    with open("test_document.pdf", "rb") as f:
        files = {"files": f}
        data = {"query": "What is this document about?"}

        response = requests.post(
            "http://localhost:1416/rag-system/run",
            files=files,
            data=data
        )

        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")

def test_enhanced_rag():
    """Test the enhanced RAG system"""
    payload = {
        "query": "Summarize the key points",
        "metadata": "Focus on technical details"
    }

    response = requests.post(
        "http://localhost:1416/rag-system-enhanced/run",
        json=payload
    )

    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")

def test_openwebui_integration():
    """Test OpenWebUI integration"""
    # Read and encode a test file
    with open("test.txt", "rb") as f:
        file_content = base64.b64encode(f.read()).decode()

    payload = {
        "model": "rag-system-enhanced",
        "messages": [
            {
                "role": "user",
                "content": "What does this document say?"
            }
        ],
        "files": [
            {
                "filename": "test.txt",
                "content": file_content,
                "type": "txt"
            }
        ]
    }

    response = requests.post(
        "http://localhost:1416/v1/chat/completions",
        json=payload
    )

    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

if __name__ == "__main__":
    test_rag_system()
    test_enhanced_rag()
    test_openwebui_integration()
```

## Extensions and Customizations

### Adding Custom Document Processing

```python
# custom_document_processor.py
from haystack.components.converters import FileTypeConverter

class CustomDocumentConverter(FileTypeConverter):
    """Custom document converter for additional formats"""

    def __init__(self):
        super().__init__(supported_formats=["csv", "json", "xml"])

    def _convert(self, file_path: str, meta: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Convert custom document formats"""
        file_ext = Path(file_path).suffix.lower()

        if file_ext == ".csv":
            return self._convert_csv(file_path, meta)
        elif file_ext == ".json":
            return self._convert_json(file_path, meta)
        elif file_ext == ".xml":
            return self._convert_xml(file_path, meta)
        else:
            raise ValueError(f"Unsupported format: {file_ext}")

    def _convert_csv(self, file_path: str, meta: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Convert CSV to document"""
        import pandas as pd

        df = pd.read_csv(file_path)
        content = df.to_string()

        return [Document(content=content, meta=meta or {})]

    def _convert_json(self, file_path: str, meta: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Convert JSON to document"""
        with open(file_path, 'r') as f:
            data = json.load(f)

        content = json.dumps(data, indent=2)
        return [Document(content=content, meta=meta or {})]

    def _convert_xml(self, file_path: str, meta: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Convert XML to document"""
        import xml.etree.ElementTree as ET

        tree = ET.parse(file_path)
        root = tree.getroot()

        content = ET.tostring(root, encoding='unicode')
        return [Document(content=content, meta=meta or {})]
```

### Adding Persistent Storage

```python
# persistent_rag_pipeline.py
from haystack.document_stores import FAISSDocumentStore
import os

class PersistentRAGPipelineWrapper(BasePipelineWrapper):
    def __init__(self):
        self.storage_path = "./rag_storage"
        os.makedirs(self.storage_path, exist_ok=True)

    def _create_document_store(self):
        """Create persistent document store"""
        return FAISSDocumentStore(
            embedding_dim=384,  # Match embedding model dimension
            index_path=f"{self.storage_path}/faiss.index",
            config_path=f"{self.storage_path}/faiss_config.json"
        )

    def save_index(self):
        """Save the document store index"""
        self.document_store.save(f"{self.storage_path}/faiss.index")

    def load_index(self):
        """Load the document store index"""
        if os.path.exists(f"{self.storage_path}/faiss.index"):
            self.document_store.load(f"{self.storage_path}/faiss.index")
```

## Best Practices

### 1. Document Processing

- Choose appropriate chunk sizes for your use case
- Use overlapping chunks for better context
- Clean and preprocess documents properly
- Handle different file formats gracefully

### 2. Embedding Models

- Choose embedding models appropriate for your domain
- Consider model size and performance trade-offs
- Use consistent embedding models for indexing and retrieval

### 3. Retrieval Strategy

- Adjust the number of retrieved documents (top_k)
- Consider using hybrid retrieval (dense + sparse)
- Implement relevance scoring and filtering

### 4. Performance Optimization

- Use persistent storage for large document collections
- Implement caching for frequent queries
- Monitor memory usage and scale accordingly
- Consider batch processing for document ingestion

### 5. Error Handling

- Provide clear error messages for users
- Log errors for debugging and monitoring
- Implement retry logic for transient failures
- Validate input documents before processing

## Next Steps

- [Chat with Website Example](chat-with-website.md) - Website analysis
- [Async Operations](async-operations.md) - Asynchronous patterns
- [OpenWebUI Events](openwebui-events.md) - OpenWebUI integration
