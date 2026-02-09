# Image Generation Example

This example demonstrates how to return binary files (images) from a `run_api` endpoint using hayhooks.

It uses the [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) model via the Hugging Face Inference API to generate images from text prompts.

## Key Concept

When `run_api` returns a FastAPI `Response` object (such as `FileResponse`, `StreamingResponse`, or plain `Response`), hayhooks passes it directly to the client — bypassing the usual Pydantic JSON response wrapper. This makes it straightforward to return images, PDFs, audio files, or any other binary content.

## Prerequisites

- A Hugging Face API token set as `HF_TOKEN` environment variable (get one at https://huggingface.co/settings/tokens)
- Install the required dependency:

```bash
pip install huggingface_hub
```

## Deploy and Run

Start hayhooks and deploy this pipeline:

```bash
hayhooks run --pipelines-dir examples/pipeline_wrappers/image_generation
```

Then generate an image:

```bash
curl -X POST "http://localhost:1416/image_generation/run" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A cat sitting on a rainbow", "width": 512, "height": 512}' \
  --output generated_image.png
```

The image will be saved to `generated_image.png`.

You can also open the URL directly in a browser or use the Swagger UI at `http://localhost:1416/docs` to test the endpoint interactively.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str` | *required* | Text description of the image to generate |
| `width` | `int` | `512` | Width of the generated image in pixels |
| `height` | `int` | `512` | Height of the generated image in pixels |
| `model` | `str` | `black-forest-labs/FLUX.1-schnell` | Hugging Face model ID for text-to-image |

## How It Works

The `pipeline_wrapper.py`:

1. Uses `huggingface_hub.InferenceClient` to call a text-to-image model
2. Saves the returned PIL Image to a temporary file
3. Returns a FastAPI `FileResponse` pointing to that file

Because `FileResponse` is a subclass of `Response`, hayhooks detects this at deploy time and replaces
the return type with `Any` for the Pydantic response model (similar to how `Generator`/`AsyncGenerator`
are handled). At runtime, the `Response` object is detected and returned directly to the client:

```python
# From hayhooks internals (deploy_utils.py)
if isinstance(result, Response):
    return result
```
