import tempfile

from fastapi.responses import FileResponse

from hayhooks import BasePipelineWrapper, log

# Requires: pip install huggingface_hub

DEFAULT_MODEL = "black-forest-labs/FLUX.1-schnell"


class PipelineWrapper(BasePipelineWrapper):
    """Generate images from text prompts using Hugging Face Inference API."""

    def setup(self) -> None:
        from huggingface_hub import InferenceClient

        self.client = InferenceClient(provider="auto")

    def run_api(
        self,
        prompt: str,
        width: int = 512,
        height: int = 512,
        model: str = DEFAULT_MODEL,
    ) -> FileResponse:
        """
        Generate an image from a text prompt and return it as a PNG file.

        Args:
            prompt: Text description of the image to generate.
            width: Width of the generated image in pixels.
            height: Height of the generated image in pixels.
            model: Hugging Face model ID for text-to-image generation.
        """
        log.info("Generating image with model '{}' for prompt: '{}' ({}x{})", model, prompt, width, height)

        image = self.client.text_to_image(
            prompt=prompt,
            model=model,
            width=width,
            height=height,
        )

        # Save the image to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp, format="PNG")

        log.info("Generated image saved to: {}", tmp.name)

        return FileResponse(
            path=tmp.name,
            media_type="image/png",
            filename="generated_image.png",
        )
