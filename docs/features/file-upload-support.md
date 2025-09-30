# File Upload Support

Hayhooks provides built-in support for handling file uploads in your pipelines, making it easy to create applications that process documents, images, and other files.

## Overview

File upload support enables you to:

- Accept file uploads through REST APIs
- Process multiple files in a single request
- Combine file uploads with other parameters
- Support various file formats (PDF, DOCX, images, etc.)
- Integrate with document processing pipelines

## Basic Implementation

### Adding File Upload Support

Add a `files` parameter to your `run_api` method:

```python
from fastapi import UploadFile
from typing import Optional, List

class PipelineWrapper(BasePipelineWrapper):
    def run_api(self, files: Optional[List[UploadFile]] = None, query: str = "") -> str:
        if files and len(files) > 0:
            filenames = [f.filename for f in files if f.filename is not None]
            file_contents = [f.file.read() for f in files]

            return f"Received {len(files)} files: {', '.join(filenames)}"

        return "No files received"
```

### Processing Pattern

```python
from fastapi import UploadFile
from typing import Optional, List
from pathlib import Path
import tempfile
import os

class PipelineWrapper(BasePipelineWrapper):
    def run_api(self, files: Optional[List[UploadFile]] = None, query: str = "") -> str:
        if not files:
            return "No files provided"

        file_info = []
        for file in files:
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(file.file.read())
                tmp_path = tmp.name

            try:
                # Here you can process the file based on your needs
                # Use Haystack converters, custom logic, etc.

                # ...

                # Add file info to the list to return it to the user
                file_info.append({
                    "name": file.filename,
                    "size": file.size,
                    "type": Path(file.filename).suffix
                })
            finally:
                # Always clean up temporary file to avoid memory leaks
                os.unlink(tmp_path)

        return f"Received {len(files)} files: {file_info}"
```

For a complete RAG example with file uploads and Haystack converters, see [RAG System Example](../examples/rag-system.md).

## API Usage

### Multipart Form Data Requests

File uploads use `multipart/form-data` format:

```bash
# Upload single file
curl -X POST \
  http://localhost:1416/my_pipeline/run \
  -F 'files=@document.pdf' \
  -F 'query="Summarize this document"'

# Upload multiple files
curl -X POST \
  http://localhost:1416/my_pipeline/run \
  -F 'files=@document1.pdf' \
  -F 'files=@document2.txt' \
  -F 'query="Compare these documents"'

# Upload with additional parameters
curl -X POST \
  http://localhost:1416/my_pipeline/run \
  -F 'files=@document.pdf' \
  -F 'query="Analyze this document"' \
  -F 'temperature=0.7'
```

### Python Client Example

```python
import requests

# Upload files
files = [
    ('files', open('document.pdf', 'rb')),
    ('files', open('notes.txt', 'rb'))
]

data = {
    'query': 'Analyze these documents',
    'temperature': '0.7'
}

response = requests.post(
    'http://localhost:1416/my_pipeline/run',
    files=files,
    data=data
)

print(response.json())
```

## CLI Usage

Hayhooks CLI supports file uploads:

```bash
# Upload single file
hayhooks pipeline run my_pipeline --file document.pdf --param 'query="Summarize this"'

# Upload directory
hayhooks pipeline run my_pipeline --dir ./documents --param 'query="Analyze all documents"'

# Upload multiple files
hayhooks pipeline run my_pipeline --file doc1.pdf --file doc2.txt --param 'query="Compare documents"'

# Upload with parameters
hayhooks pipeline run my_pipeline --file document.pdf --param 'query="Analyze"' --param 'temperature=0.7'
```

## File Type Support

### Supported File Types

| File Type | Extension | Processing Library | Dependencies |
|-----------|-----------|-------------------|--------------|
| PDF | .pdf | PyPDFToDocument (Haystack) | `pip install pypdf` |
| Text | .txt | Built-in | None |
| Word | .docx, .doc | python-docx | `pip install python-docx` |
| Markdown | .md | Built-in | None |
| Images | .jpg, .png | PIL/Pillow | `pip install Pillow` |
| CSV | .csv | pandas | `pip install pandas` |

### Adding Custom File Types

```python
def _process_custom_file(self, file_path: str) -> str:
    """Process custom file type"""
    file_ext = Path(file_path).suffix.lower()

    if file_ext == '.json':
        return self._process_json(file_path)
    elif file_ext == '.xml':
        return self._process_xml(file_path)
    else:
        return f"Unsupported file type: {file_ext}"

def _process_json(self, file_path: str) -> str:
    """Process JSON file"""
    import json
    with open(file_path, 'r') as f:
        data = json.load(f)
    return str(data)[:5000]
```

## Security Considerations

### File Validation

```python
from fastapi import UploadFile, HTTPException
from pathlib import Path

class PipelineWrapper(BasePipelineWrapper):
    def _validate_file(self, file: UploadFile) -> bool:
        """Validate uploaded file"""
        # Check file size (e.g., max 10MB)
        max_size = 10 * 1024 * 1024  # 10MB
        if file.size and file.size > max_size:
            raise HTTPException(status_code=413, detail="File too large")

        # Check file extension
        allowed_extensions = {'.pdf', '.txt', '.docx', '.md'}
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(status_code=400, detail="File type not supported")

        # Check filename for security
        if '..' in file.filename or '/' in file.filename:
            raise HTTPException(status_code=400, detail="Invalid filename")

        return True

    def run_api(self, files: Optional[List[UploadFile]] = None, query: str = "") -> str:
        if files:
            for file in files:
                self._validate_file(file)

        # Continue with processing...
```

## Error Handling

Handle errors gracefully when processing uploaded files:

```python
from fastapi import HTTPException
from hayhooks import log

class PipelineWrapper(BasePipelineWrapper):
    def run_api(self, files: Optional[List[UploadFile]] = None, query: str = "") -> str:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")

        try:
            # Validate files
            for file in files:
                self._validate_file(file)

            # Process files (your custom logic here)
            filenames = [f.filename for f in files]
            return f"Successfully received {len(files)} files: {', '.join(filenames)}"

        except Exception as e:
            log.error(f"Error processing files: {e}")
            raise HTTPException(status_code=500, detail=str(e))
```

## Complete Example

### RAG System with File Upload

```python
class RAGPipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        # Initialize RAG pipeline components
        from haystack.components.preprocessors import DocumentSplitter
        from haystack.components.embedders import SentenceTransformersDocumentEmbedder
        from haystack.document_stores.in_memory import InMemoryDocumentStore
        from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
        from haystack.components.builders import PromptBuilder
        from haystack.components.generators import OpenAIGenerator

        self.document_store = InMemoryDocumentStore()
        self.splitter = DocumentSplitter()
        self.embedder = SentenceTransformersDocumentEmbedder()
        self.retriever = InMemoryEmbeddingRetriever(document_store=self.document_store)
        self.prompt_builder = PromptBuilder(
            template="Answer this question: {{query}}\n\nContext: {{documents}}"
        )
        self.generator = OpenAIGenerator()

        # Build pipeline
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

    def run_api(self, files: Optional[List[UploadFile]] = None, query: str = "") -> str:
        if files:
            # Index uploaded documents
            documents = []
            for file in files:
                doc_content = self._process_file(file)
                documents.append({"content": doc_content, "meta": {"source": file.filename}})

            # Add to document store
            self.pipeline.run({"splitter": {"documents": documents}})

        # Query the indexed documents
        result = self.pipeline.run({
            "retriever": {"query": query},
            "prompt_builder": {"query": query}
        })

        return result["generator"]["replies"][0]
```

## Best Practices

### 1. File Validation

- Validate file types (extensions)
- Enforce file size limits
- Sanitize file names
- Check for required file formats

### 2. File Management

- Use temporary files for processing
- Clean up files after processing
- Handle multiple files efficiently
- Store files securely if needed

### 3. Error Handling

- Validate files before processing
- Provide clear error messages
- Log errors for debugging
- Handle partial failures gracefully

## Next Steps

- [PipelineWrapper](../concepts/pipeline-wrapper.md) - Learn about wrapper implementation
- [Examples](../examples/overview.md) - See working examples
- [CLI Commands](cli-commands.md) - CLI usage for file uploads
