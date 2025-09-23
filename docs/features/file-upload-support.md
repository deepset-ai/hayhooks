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

### Complete File Processing Example

```python
from fastapi import UploadFile
from typing import Optional, List
from pathlib import Path
import tempfile
import os

class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        # Initialize your pipeline
        from haystack.components import PromptBuilder, OpenAIGenerator

        self.prompt_builder = PromptBuilder(
            template="Analyze this content: {{documents}}\n\nQuestion: {{query}}"
        )
        self.llm = OpenAIGenerator()

        # Build pipeline
        self.pipeline = Pipeline()
        self.pipeline.add_component("prompt_builder", self.prompt_builder)
        self.pipeline.add_component("llm", self.llm)
        self.pipeline.connect("prompt_builder", "llm")

    def run_api(self, files: Optional[List[UploadFile]] = None, query: str = "") -> str:
        if not files:
            return "No files provided. Please upload files to analyze."

        documents = []

        # Process each uploaded file
        for file in files:
            try:
                # Save file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp_file:
                    content = file.file.read()
                    tmp_file.write(content)
                    tmp_file_path = tmp_file.name

                # Process file based on type
                file_ext = Path(file.filename).suffix.lower()

                if file_ext == '.pdf':
                    doc_text = self._process_pdf(tmp_file_path)
                elif file_ext == '.txt':
                    doc_text = self._process_text(tmp_file_path)
                elif file_ext in ['.docx', '.doc']:
                    doc_text = self._process_docx(tmp_file_path)
                else:
                    doc_text = f"Unsupported file type: {file_ext}"

                documents.append({
                    "filename": file.filename,
                    "content": doc_text
                })

                # Clean up temporary file
                os.unlink(tmp_file_path)

            except Exception as e:
                documents.append({
                    "filename": file.filename,
                    "content": f"Error processing file: {str(e)}"
                })

        # Run analysis pipeline
        result = self.pipeline.run({
            "prompt_builder": {
                "documents": "\n\n".join([f"{doc['filename']}: {doc['content']}" for doc in documents]),
                "query": query
            }
        })

        return result["llm"]["replies"][0]

    def _process_pdf(self, file_path: str) -> str:
        """Process PDF file"""
        try:
            import PyPDF2

            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text[:5000]  # Limit text length

        except ImportError:
            return "PDF processing requires PyPDF2: pip install PyPDF2"
        except Exception as e:
            return f"Error processing PDF: {str(e)}"

    def _process_text(self, file_path: str) -> str:
        """Process text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()[:5000]  # Limit text length
        except Exception as e:
            return f"Error processing text file: {str(e)}"

    def _process_docx(self, file_path: str) -> str:
        """Process DOCX file"""
        try:
            import docx

            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text[:5000]  # Limit text length

        except ImportError:
            return "DOCX processing requires python-docx: pip install python-docx"
        except Exception as e:
            return f"Error processing DOCX: {str(e)}"
```

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
| PDF | .pdf | PyPDF2 | `pip install PyPDF2` |
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

### Virus Scanning

```python
import clamd

def _scan_file(self, file_path: str) -> bool:
    """Scan file for viruses"""
    try:
        cd = clamd.ClamdUnixSocket()
        scan_result = cd.scan_file(file_path)
        return scan_result is None  # None means no virus found
    except Exception:
        # If scanning fails, you might want to block the file or log it
        return False
```

## Error Handling

### Comprehensive Error Handling

```python
from fastapi import HTTPException
from hayhooks import log

class PipelineWrapper(BasePipelineWrapper):
    def run_api(self, files: Optional[List[UploadFile]] = None, query: str = "") -> str:
        try:
            if not files:
                return "No files provided"

            processed_files = []
            for file in files:
                try:
                    # Validate file
                    self._validate_file(file)

                    # Process file
                    content = self._process_file(file)
                    processed_files.append({
                        "filename": file.filename,
                        "status": "success",
                        "content": content[:1000]  # Preview
                    })

                except Exception as e:
                    log.error(f"Error processing file {file.filename}: {e}")
                    processed_files.append({
                        "filename": file.filename,
                        "status": "error",
                        "error": str(e)
                    })

            # Generate summary
            successful_files = [f for f in processed_files if f["status"] == "success"]
            if not successful_files:
                raise HTTPException(status_code=400, detail="No files could be processed")

            return f"Processed {len(successful_files)} files successfully"

        except Exception as e:
            log.error(f"Pipeline execution failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
```

## Performance Optimization

### File Processing Optimization

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class PipelineWrapper(BasePipelineWrapper):
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)

    def run_api(self, files: Optional[List[UploadFile]] = None, query: str = "") -> str:
        if not files:
            return "No files provided"

        # Process files in parallel
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            tasks = [loop.run_in_executor(self.executor, self._process_file_async, file) for file in files]
            results = loop.run_until_complete(asyncio.gather(*tasks))

            return f"Processed {len(files)} files: {results}"
        finally:
            loop.close()

    async def _process_file_async(self, file: UploadFile) -> dict:
        """Process file asynchronously"""
        try:
            content = await asyncio.to_thread(file.file.read)
            # Process content...
            return {"filename": file.filename, "status": "success"}
        except Exception as e:
            return {"filename": file.filename, "status": "error", "error": str(e)}
```

## Examples

### RAG System with File Upload

```python
class RAGPipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        # Initialize RAG pipeline components
        from haystack.components import (
            DocumentSplitter,
            SentenceTransformersDocumentEmbedder,
            InMemoryDocumentStore,
            InMemoryEmbeddingRetriever,
            PromptBuilder,
            OpenAIGenerator
        )

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

### 1. File Management
- Use temporary files for processing
- Clean up files after processing
- Implement file size limits
- Validate file types and names

### 2. Security
- Scan uploaded files for malware
- Validate file contents
- Implement rate limiting
- Use secure file storage

### 3. Performance
- Process files in parallel when possible
- Use streaming for large files
- Implement caching where appropriate
- Monitor resource usage

### 4. Error Handling
- Provide clear error messages
- Log errors for debugging
- Implement graceful degradation
- Validate inputs before processing

## Next Steps

- [PipelineWrapper](../concepts/pipeline-wrapper.md) - Learn about wrapper implementation
- [Examples](../examples/overview.md) - See working examples
- [CLI Commands](cli-commands.md) - CLI usage for file uploads