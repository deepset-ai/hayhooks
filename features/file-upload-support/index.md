# File Upload Support

Hayhooks provides built-in support for handling file uploads in your pipelines, making it easy to create applications that process documents, images, and other files.

## Overview

File upload support enables you to:

- Accept file uploads through REST APIs
- Process multiple files in a single request
- Combine file uploads with other parameters
- Build document processing and RAG pipelines

## Basic Implementation

To accept file uploads in your pipeline, add a `files` parameter to your `run_api` method:

```
from fastapi import UploadFile

class PipelineWrapper(BasePipelineWrapper):
    def run_api(self, files: list[UploadFile] | None = None, query: str = "") -> str:
        if not files:
            return "No files provided"

        # Process files here...
        return f"Processed {len(files)} files"
```

For a complete implementation, see the [RAG System Example](https://deepset-ai.github.io/hayhooks/examples/rag-system/index.md).

## API Usage

### Multipart Form Data Requests

File uploads use `multipart/form-data` format:

```
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
  -F 'query="Analyze this document"'
```

### Python Client Example

```
import requests

# Upload files
files = [
    ('files', open('document.pdf', 'rb')),
    ('files', open('notes.txt', 'rb'))
]

data = {
    'query': 'Analyze these documents'
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

```
# Upload single file
hayhooks pipeline run my_pipeline --file document.pdf --param 'query="Summarize this"'

# Upload directory
hayhooks pipeline run my_pipeline --dir ./documents --param 'query="Analyze all documents"'

# Upload multiple files
hayhooks pipeline run my_pipeline --file doc1.pdf --file doc2.txt --param 'query="Compare documents"'

# Upload with parameters
hayhooks pipeline run my_pipeline --file document.pdf --param 'query="Analyze"'
```

## Combining Files with Other Parameters

You can handle both files and parameters in the same request by adding them as arguments to the `run_api` method:

```
from fastapi import UploadFile

class PipelineWrapper(BasePipelineWrapper):
    def run_api(
        self,
        files: list[UploadFile] | None = None,
        query: str = "",
        additional_param: str = "default"
    ) -> str:
        if files and len(files) > 0:
            filenames = [f.filename for f in files if f.filename is not None]
            return f"Received files: {', '.join(filenames)} with query: {query}"

        return "No files received"
```

## Complete Example: RAG System with File Upload

For a complete, production-ready example of a RAG system with file uploads, including document indexing and querying with Elasticsearch, see:

- [RAG System Example](https://deepset-ai.github.io/hayhooks/examples/rag-system/index.md) - Full RAG implementation guide
- [examples/rag_indexing_query](https://github.com/deepset-ai/hayhooks/tree/main/examples/rag_indexing_query) - Complete working code with:
- Document indexing pipeline with file upload support
- Query pipeline for retrieving and generating answers
- Elasticsearch integration
- Support for PDF, Markdown, and text files

## Next Steps

- [PipelineWrapper](https://deepset-ai.github.io/hayhooks/concepts/pipeline-wrapper/index.md) - Learn about wrapper implementation
- [Examples](https://deepset-ai.github.io/hayhooks/examples/overview/index.md) - See working examples
- [CLI Commands](https://deepset-ai.github.io/hayhooks/features/cli-commands/index.md) - CLI usage for file uploads
