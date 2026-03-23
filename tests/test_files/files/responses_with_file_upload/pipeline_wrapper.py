import time
from uuid import uuid4

from hayhooks import BasePipelineWrapper


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        self.pipeline = None

    def run_file_upload(
        self, filename: str | None, content_type: str | None, content: bytes, purpose: str
    ) -> dict:
        # NOTE: This is used in tests, please don't change it
        return {
            "id": f"custom-{uuid4().hex[:12]}",
            "object": "file",
            "bytes": len(content),
            "created_at": int(time.time()),
            "filename": filename or "",
            "purpose": purpose,
        }

    def run_response(self, model: str, input_items: list[dict], body: dict) -> str:
        return "Response with file upload support"
