import random
import struct
import tempfile
import zlib

from fastapi.responses import FileResponse

from hayhooks import BasePipelineWrapper


def _generate_random_png(width: int, height: int) -> bytes:
    """Generate a random RGB image as PNG bytes using only the standard library."""
    raw_data = bytearray()
    for _ in range(height):
        raw_data.append(0)  # filter byte
        for _ in range(width):
            raw_data.extend([random.randint(0, 255) for _ in range(3)])  # noqa: S311

    def _make_chunk(chunk_type: bytes, data: bytes) -> bytes:
        chunk = chunk_type + data
        return struct.pack(">I", len(data)) + chunk + struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)

    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _make_chunk(b"IHDR", ihdr_data)
        + _make_chunk(b"IDAT", zlib.compress(bytes(raw_data)))
        + _make_chunk(b"IEND", b"")
    )


class PipelineWrapper(BasePipelineWrapper):
    """Generate a random PNG image to test FileResponse support."""

    def setup(self) -> None:
        pass

    def run_api(self, width: int = 64, height: int = 64) -> FileResponse:
        """Generate a random PNG image, write it to a temp file, and return it as a FileResponse."""
        image_bytes = _generate_random_png(width, height)

        # Write to a temporary file (delete=False so FileResponse can read it)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(image_bytes)

        return FileResponse(
            path=tmp.name,
            media_type="image/png",
            filename="random_image.png",
        )
