# Code Sharing

This section covers strategies for sharing code between pipelines, creating reusable components, and managing dependencies effectively.

## Overview

Code sharing in Hayhooks allows you to:

- Create reusable components across multiple pipelines
- Share utility functions and classes
- Manage common dependencies efficiently
- Reduce code duplication and improve maintainability

## Directory Structure for Shared Code

### Recommended Structure

```
hayhooks-project/
├── pipelines/
│   ├── chat_pipeline/
│   │   ├── pipeline_wrapper.py
│   │   └── config.yaml
│   ├── rag_pipeline/
│   │   ├── pipeline_wrapper.py
│   │   └── config.yaml
│   └── translation_pipeline/
│       ├── pipeline_wrapper.py
│       └── config.yaml
├── shared_components/
│   ├── __init__.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── text_processing.py
│   │   └── document_processing.py
│   ├── connectors/
│   │   ├── __init__.py
│   │   ├── database.py
│   │   └── api_clients.py
│   └── custom_components/
│       ├── __init__.py
│       ├── custom_llm.py
│       └── custom_retriever.py
├── requirements/
│   ├── base.txt
│   ├── dev.txt
│   └── prod.txt
└── config/
    ├── settings.py
    └── logging.py
```

### Configuration

```bash
# Set additional Python path for shared code
export HAYHOOKS_ADDITIONAL_PYTHON_PATH=./shared_components
```

## Shared Components

### Utility Functions

```python
# shared_components/utils/text_processing.py
import re
from typing import List, Dict, Any

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters if needed
    text = re.sub(r'[^\w\s\.\,\!\?]', '', text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks"""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract keywords from text"""
    # Simple keyword extraction
    words = text.lower().split()
    word_freq = {}

    for word in words:
        if len(word) > 3:  # Skip short words
            word_freq[word] = word_freq.get(word, 0) + 1

    # Return top keywords
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_words[:max_keywords]]
```

### Document Processing

```python
# shared_components/utils/document_processing.py
from pathlib import Path
from typing import List, Dict, Any, Optional
import tempfile
import os

class DocumentProcessor:
    """Base class for document processing"""

    def __init__(self, max_size: int = 10 * 1024 * 1024):
        self.max_size = max_size

    def process_file(self, file_path: str) -> Dict[str, Any]:
        """Process a single file"""
        raise NotImplementedError

    def validate_file(self, file_path: str) -> bool:
        """Validate file before processing"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_size = os.path.getsize(file_path)
        if file_size > self.max_size:
            raise ValueError(f"File too large: {file_size} bytes")

        return True

class PDFProcessor(DocumentProcessor):
    """PDF document processor"""

    def process_file(self, file_path: str) -> Dict[str, Any]:
        self.validate_file(file_path)

        try:
            import PyPDF2

            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""

                for page in reader.pages:
                    text += page.extract_text() + "\n"

                return {
                    "content": text,
                    "metadata": {
                        "filename": Path(file_path).name,
                        "pages": len(reader.pages),
                        "file_size": os.path.getsize(file_path)
                    }
                }

        except ImportError:
            raise ImportError("PyPDF2 is required for PDF processing")
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")

class TextProcessor(DocumentProcessor):
    """Text file processor"""

    def process_file(self, file_path: str) -> Dict[str, Any]:
        self.validate_file(file_path)

        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            return {
                "content": content,
                "metadata": {
                    "filename": Path(file_path).name,
                    "file_size": os.path.getsize(file_path),
                    "encoding": "utf-8"
                }
            }

        except Exception as e:
            raise Exception(f"Error processing text file: {str(e)}")
```

### Custom Connectors

```python
# shared_components/connectors/database.py
from sqlalchemy import create_engine, Column, String, JSON, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from typing import List, Dict, Any, Optional

Base = declarative_base()

class DocumentStore(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True)
    content = Column(String)
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class DatabaseConnector:
    """Database connector for document storage"""

    def __init__(self, database_url: str):
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        Base.metadata.create_all(bind=self.engine)

    def save_document(self, doc_id: str, content: str, metadata: Dict[str, Any]) -> bool:
        """Save document to database"""
        try:
            session = self.SessionLocal()
            doc = DocumentStore(id=doc_id, content=content, metadata=metadata)
            session.add(doc)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            raise Exception(f"Error saving document: {str(e)}")
        finally:
            session.close()

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve document from database"""
        try:
            session = self.SessionLocal()
            doc = session.query(DocumentStore).filter(DocumentStore.id == doc_id).first()

            if doc:
                return {
                    "id": doc.id,
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "created_at": doc.created_at,
                    "updated_at": doc.updated_at
                }
            return None
        except Exception as e:
            raise Exception(f"Error retrieving document: {str(e)}")
        finally:
            session.close()

    def search_documents(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search documents (simple text search)"""
        try:
            session = self.SessionLocal()
            docs = session.query(DocumentStore).filter(
                DocumentStore.content.contains(query)
            ).limit(limit).all()

            return [
                {
                    "id": doc.id,
                    "content": doc.content[:200] + "...",
                    "metadata": doc.metadata,
                    "created_at": doc.created_at
                }
                for doc in docs
            ]
        except Exception as e:
            raise Exception(f"Error searching documents: {str(e)}")
        finally:
            session.close()
```

### API Clients

```python
# shared_components/connectors/api_clients.py
import requests
from typing import Dict, Any, Optional
import json

class OpenAIClient:
    """OpenAI API client"""

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def chat_completion(self, messages: List[Dict[str, str]], model: str = "gpt-3.5-turbo") -> Dict[str, Any]:
        """Create chat completion"""
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.7
        }

        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()

        return response.json()

    def embedding(self, text: str, model: str = "text-embedding-ada-002") -> List[float]:
        """Create text embedding"""
        url = f"{self.base_url}/embeddings"

        payload = {
            "model": model,
            "input": text
        }

        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()

        return response.json()["data"][0]["embedding"]

class SerpAPIClient:
    """SerpAPI client for web search"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://serpapi.com/search"

    def search(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        """Perform web search"""
        params = {
            "q": query,
            "api_key": self.api_key,
            "num": num_results
        }

        response = requests.get(self.base_url, params=params)
        response.raise_for_status()

        results = response.json()

        return [
            {
                "title": result.get("title", ""),
                "link": result.get("link", ""),
                "snippet": result.get("snippet", "")
            }
            for result in results.get("organic_results", [])
        ]
```

### Custom Haystack Components

```python
# shared_components/custom_components/custom_llm.py
from haystack.components.generators import OpenAIGenerator
from typing import List, Dict, Any, Optional
import requests

class CustomOpenAIGenerator(OpenAIGenerator):
    """Custom OpenAI generator with additional features"""

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", **kwargs):
        super().__init__(model=model, api_key=api_key, **kwargs)
        self.api_key = api_key
        self.model = model

    def generate_with_context(self, prompt: str, context: str) -> str:
        """Generate text with additional context"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {prompt}"}
        ]

        return self._call_openai_api(messages)

    def _call_openai_api(self, messages: List[Dict[str, str]]) -> str:
        """Internal method to call OpenAI API"""
        url = "https://api.openai.com/v1/chat/completions"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()

        return response.json()["choices"][0]["message"]["content"]
```

## Pipeline Templates

### Base Pipeline Template

```python
# shared_components/templates/base_pipeline.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from hayhooks import BasePipelineWrapper

class BaseTemplatePipeline(BasePipelineWrapper):
    """Base template for creating consistent pipelines"""

    def __init__(self):
        self.pipeline = None
        self.components = {}
        self.config = {}

    @abstractmethod
    def setup_components(self):
        """Setup pipeline components"""
        pass

    @abstractmethod
    def build_pipeline(self):
        """Build the pipeline connections"""
        pass

    def setup(self):
        """Setup the pipeline"""
        self.setup_components()
        self.build_pipeline()

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data"""
        return True

    def preprocess_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess input data"""
        return input_data

    def postprocess_output(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        """Postprocess output data"""
        return output_data

    def get_component(self, name: str):
        """Get component by name"""
        return self.components.get(name)
```

### Chat Pipeline Template

```python
# shared_components/templates/chat_pipeline.py
from .base_pipeline import BaseTemplatePipeline
from haystack.components import PromptBuilder, OpenAIChatGenerator
from hayhooks import get_last_user_message

class ChatPipelineTemplate(BaseTemplatePipeline):
    """Template for chat-based pipelines"""

    def __init__(self, system_prompt: str = "You are a helpful assistant."):
        super().__init__()
        self.system_prompt = system_prompt

    def setup_components(self):
        """Setup chat pipeline components"""
        self.prompt_builder = PromptBuilder(
            template=f"System: {self.system_prompt}\n\nUser: {{query}}\n\nAssistant:"
        )
        self.llm = OpenAIChatGenerator()

        self.components = {
            "prompt_builder": self.prompt_builder,
            "llm": self.llm
        }

    def build_pipeline(self):
        """Build chat pipeline connections"""
        from haystack import Pipeline

        self.pipeline = Pipeline()
        self.pipeline.add_component("prompt_builder", self.prompt_builder)
        self.pipeline.add_component("llm", self.llm)
        self.pipeline.connect("prompt_builder", "llm")

    def run_chat_completion(self, model: str, messages: List[dict], body: dict):
        """Run chat completion"""
        question = get_last_user_message(messages)

        result = self.pipeline.run({
            "prompt_builder": {"query": question}
        })

        return result["llm"]["replies"][0].content
```

### RAG Pipeline Template

```python
# shared_components/templates/rag_pipeline.py
from .base_pipeline import BaseTemplatePipeline
from haystack.components import (
    DocumentSplitter,
    SentenceTransformersDocumentEmbedder,
    InMemoryDocumentStore,
    InMemoryEmbeddingRetriever,
    PromptBuilder,
    OpenAIGenerator
)

class RAGPipelineTemplate(BaseTemplatePipeline):
    """Template for RAG pipelines"""

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        super().__init__()
        self.embedding_model = embedding_model

    def setup_components(self):
        """Setup RAG pipeline components"""
        self.document_store = InMemoryDocumentStore()
        self.splitter = DocumentSplitter()
        self.embedder = SentenceTransformersDocumentEmbedder(model=self.embedding_model)
        self.retriever = InMemoryEmbeddingRetriever(document_store=self.document_store)
        self.prompt_builder = PromptBuilder(
            template="Answer this question: {{query}}\n\nContext: {{documents}}"
        )
        self.generator = OpenAIGenerator()

        self.components = {
            "document_store": self.document_store,
            "splitter": self.splitter,
            "embedder": self.embedder,
            "retriever": self.retriever,
            "prompt_builder": self.prompt_builder,
            "generator": self.generator
        }

    def build_pipeline(self):
        """Build RAG pipeline connections"""
        from haystack import Pipeline

        self.pipeline = Pipeline()
        self.pipeline.add_component("splitter", self.splitter)
        self.pipeline.add_component("embedder", self.embedder)
        self.pipeline.add_component("retriever", self.retriever)
        self.pipeline.add_component("prompt_builder", self.prompt_builder)
        self.pipeline.add_component("generator", self.generator)

        self.pipeline.connect("splitter", "embedder")
        self.pipeline.connect("embedder", "retriever")
        self.pipeline.connect("retriever", "prompt_builder.documents")
        self.pipeline.connect("prompt_builder", "generator")

    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the RAG system"""
        self.pipeline.run({
            "splitter": {"documents": documents}
        })

    def run_query(self, query: str) -> str:
        """Run query against the RAG system"""
        result = self.pipeline.run({
            "retriever": {"query": query},
            "prompt_builder": {"query": query}
        })

        return result["generator"]["replies"][0]
```

## Using Shared Components

### Example: Chat Pipeline

```python
# pipelines/chat_pipeline/pipeline_wrapper.py
from shared_components.templates.chat_pipeline import ChatPipelineTemplate
from shared_components.utils.text_processing import clean_text
from hayhooks import get_last_user_message

class PipelineWrapper(ChatPipelineTemplate):
    def __init__(self):
        super().__init__(system_prompt="You are a helpful AI assistant.")

    def run_chat_completion(self, model: str, messages: List[dict], body: dict):
        """Enhanced chat completion with text cleaning"""
        question = get_last_user_message(messages)

        # Clean the input
        cleaned_question = clean_text(question)

        result = self.pipeline.run({
            "prompt_builder": {"query": cleaned_question}
        })

        return result["llm"]["replies"][0].content
```

### Example: RAG Pipeline

```python
# pipelines/rag_pipeline/pipeline_wrapper.py
from shared_components.templates.rag_pipeline import RAGPipelineTemplate
from shared_components.utils.document_processing import PDFProcessor, TextProcessor
from shared_components.connectors.database import DatabaseConnector

class PipelineWrapper(RAGPipelineTemplate):
    def __init__(self):
        super().__init__()
        self.pdf_processor = PDFProcessor()
        self.text_processor = TextProcessor()
        self.db_connector = DatabaseConnector("sqlite:///rag_documents.db")

    def setup(self):
        """Setup with additional components"""
        super().setup()

        # Load existing documents from database
        self.load_existing_documents()

    def load_existing_documents(self):
        """Load documents from database"""
        documents = self.db_connector.get_all_documents()

        if documents:
            haydocs = [
                {"content": doc["content"], "meta": doc["metadata"]}
                for doc in documents
            ]
            self.add_documents(haydocs)

    def run_api(self, query: str = "", files: Optional[List[UploadFile]] = None) -> str:
        """Run API with file processing support"""
        if files:
            # Process uploaded files
            for file in files:
                try:
                    # Save and process file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp_file:
                        content = file.file.read()
                        tmp_file.write(content)
                        tmp_file_path = tmp_file.name

                    # Process based on file type
                    if file.filename.endswith('.pdf'):
                        doc_data = self.pdf_processor.process_file(tmp_file_path)
                    else:
                        doc_data = self.text_processor.process_file(tmp_file_path)

                    # Add to database and pipeline
                    self.db_connector.save_document(
                        doc_id=file.filename,
                        content=doc_data["content"],
                        metadata=doc_data["metadata"]
                    )

                    # Add to RAG system
                    self.add_documents([{
                        "content": doc_data["content"],
                        "meta": doc_data["metadata"]
                    }])

                    # Clean up
                    os.unlink(tmp_file_path)

                except Exception as e:
                    return f"Error processing file {file.filename}: {str(e)}"

        # Run the query
        return self.run_query(query)
```

## Dependency Management

### Requirements Files

```txt
# requirements/base.txt
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
haystack-ai>=2.0.0
pydantic>=2.5.0
sqlalchemy>=2.0.0
python-multipart>=0.0.6
```

```txt
# requirements/dev.txt
-r base.txt
pytest>=7.4.0
pytest-asyncio>=0.21.0
black>=23.0.0
ruff>=0.1.0
mypy>=1.7.0
```

```txt
# requirements/prod.txt
-r base.txt
gunicorn>=21.2.0
prometheus-client>=0.19.0
redis>=5.0.0
```

### Setup Script

```python
# setup_shared_components.py
import subprocess
import sys
from pathlib import Path

def install_requirements():
    """Install requirements for shared components"""
    base_dir = Path(__file__).parent

    # Install base requirements
    base_requirements = base_dir / "requirements" / "base.txt"
    if base_requirements.exists():
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(base_requirements)
        ], check=True)

    # Install development requirements if needed
    dev_requirements = base_dir / "requirements" / "dev.txt"
    if dev_requirements.exists():
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(dev_requirements)
        ], check=True)

def setup_python_path():
    """Setup Python path for shared components"""
    import sys
    from pathlib import Path

    shared_path = Path(__file__).parent / "shared_components"
    if shared_path.exists() and str(shared_path) not in sys.path:
        sys.path.insert(0, str(shared_path))

if __name__ == "__main__":
    install_requirements()
    setup_python_path()
    print("Shared components setup completed!")
```

## Best Practices

### 1. Code Organization

- Keep shared components in a dedicated directory
- Use clear naming conventions for components
- Implement proper error handling and logging
- Write comprehensive tests for shared components

### 2. Component Design

- Design components to be reusable and configurable
- Use dependency injection for better testability
- Implement proper interfaces and abstract classes
- Document component APIs and usage examples

### 3. Performance Considerations

- Optimize frequently used functions
- Implement caching where appropriate
- Use lazy loading for heavy components
- Monitor resource usage

### 4. Version Control

- Keep shared components in version control
- Use semantic versioning for components
- Implement proper change management
- Document breaking changes

## Next Steps

- [Advanced Configuration](advanced-configuration.md) - Configuration options
- [Custom Routes](custom-routes.md) - Custom endpoint development
- [Examples](../examples/overview.md) - Working examples
