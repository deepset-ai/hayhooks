# File Response Support

Hayhooks supports returning binary files (images, PDFs, audio, etc.) directly from `run_api` endpoints. When `run_api` returns a FastAPI [`Response`](https://fastapi.tiangolo.com/advanced/response-directly/) object, Hayhooks passes it straight to the client — bypassing JSON serialization entirely. See also the FastAPI docs on [custom responses](https://fastapi.tiangolo.com/advanced/custom-response/).

## Overview

File response support enables you to:

- Return images, PDFs, audio files, or any binary content from pipelines
- Use FastAPI's `FileResponse`, `StreamingResponse`, or plain `Response`
- Serve generated content (e.g. AI-generated images) directly to clients
- Set custom headers like `Content-Disposition` for download behavior

## Basic Implementation

To return a file from your pipeline, return a FastAPI `Response` (or any subclass) from `run_api`:

```
from fastapi.responses import FileResponse

class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        pass

    def run_api(self, prompt: str) -> FileResponse:
        # Generate or retrieve a file...
        image_path = generate_image(prompt)

        return FileResponse(
            path=image_path,
            media_type="image/png",
            filename="result.png",
        )
```

## Response Types

You can use any FastAPI/Starlette response class:

### FileResponse

Best for serving files from disk:

```
from fastapi.responses import FileResponse

def run_api(self, document_id: str) -> FileResponse:
    path = self.get_document_path(document_id)
    return FileResponse(
        path=path,
        media_type="application/pdf",
        filename="document.pdf",
    )
```

### Response

Best for returning in-memory bytes:

```
from fastapi.responses import Response

def run_api(self, prompt: str) -> Response:
    image_bytes = self.generate_image(prompt)
    return Response(
        content=image_bytes,
        media_type="image/png",
        headers={"Content-Disposition": 'inline; filename="image.png"'},
    )
```

### StreamingResponse

Best for large files or on-the-fly generation:

```
import io
from fastapi.responses import StreamingResponse

def run_api(self, query: str) -> StreamingResponse:
    audio_buffer = self.generate_audio(query)
    return StreamingResponse(
        content=io.BytesIO(audio_buffer),
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="audio.wav"'},
    )
```

## How It Works

When Hayhooks deploys a pipeline whose `run_api` return type is a `Response` subclass (or a generator), three things happen at deploy time:

1. **`response_model=None`**: `create_response_model_from_callable` detects the `Response` return type and returns `None` instead of a Pydantic model. This tells FastAPI to skip response validation and not generate a JSON schema for this endpoint.

1. **`response_class`**: `get_response_class_from_callable` returns the concrete response class (e.g. `FileResponse`, `StreamingResponse`) so that OpenAPI docs show the correct Content-Type for the endpoint instead of `application/json`.

1. **At runtime**: The endpoint handler checks if the result is a `Response` instance and returns it directly, skipping JSON wrapping:

   ```
   # From deploy_utils.py
   if isinstance(result, Response):
       return result
   ```

This is the same mechanism used for streaming generators — both bypass Pydantic serialization. Generators additionally get `StreamingResponse` as their `response_class`.

## API Usage

### curl

```
# Generate and download an image
curl -X POST "http://localhost:1416/image_pipeline/run" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A sunset over mountains"}' \
  --output result.png
```

### Python

```
import requests

response = requests.post(
    "http://localhost:1416/image_pipeline/run",
    json={"prompt": "A sunset over mountains"},
)

# Save the binary content
with open("result.png", "wb") as f:
    f.write(response.content)
```

### Browser

File responses with `Content-Disposition: inline` will display directly in the browser. Use `Content-Disposition: attachment` to trigger a download.

## Complete Example: Image Generation

This example uses the Hugging Face Inference API to generate images from text prompts:

```
import tempfile

from fastapi.responses import FileResponse

from hayhooks import BasePipelineWrapper, log

DEFAULT_MODEL = "black-forest-labs/FLUX.1-schnell"


class PipelineWrapper(BasePipelineWrapper):
    """Generate images from text prompts using Hugging Face Inference API."""

    def setup(self) -> None:
        from huggingface_hub import InferenceClient

        self.client = InferenceClient()

    def run_api(
        self,
        prompt: str,
        width: int = 512,
        height: int = 512,
        model: str = DEFAULT_MODEL,
    ) -> FileResponse:
        """Generate an image from a text prompt and return it as a PNG file."""
        log.info("Generating image for prompt: '{}'", prompt)

        image = self.client.text_to_image(
            prompt=prompt, model=model, width=width, height=height,
        )

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp, format="PNG")

        return FileResponse(
            path=tmp.name,
            media_type="image/png",
            filename="generated_image.png",
        )
```

Deploy and test:

```
# Deploy
hayhooks run --pipelines-dir examples/pipeline_wrappers/image_generation

# Generate an image
curl -X POST "http://localhost:1416/image_generation/run" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A cat sitting on a rainbow"}' \
  --output generated_image.png
```

For the full example code, see [examples/pipeline_wrappers/image_generation](https://github.com/deepset-ai/hayhooks/tree/main/examples/pipeline_wrappers/image_generation).

## Next Steps

- [PipelineWrapper](https://deepset-ai.github.io/hayhooks/concepts/pipeline-wrapper/index.md) - Learn about wrapper implementation
- [File Upload Support](https://deepset-ai.github.io/hayhooks/features/file-upload-support/index.md) - Accept file uploads in pipelines
- [Examples](https://deepset-ai.github.io/hayhooks/examples/overview/index.md) - See working examples
