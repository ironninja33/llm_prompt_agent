"""Server-side image compression for LLM quality scoring."""

import io
import logging

from PIL import Image

logger = logging.getLogger(__name__)


def compress_for_llm(file_path: str, max_dimension: int = 512,
                     quality: int = 80) -> bytes | None:
    """Compress an image to JPEG bytes for LLM scoring.

    Resizes so the largest dimension is at most ``max_dimension`` pixels,
    converts to RGB, and encodes as JPEG.

    Returns JPEG bytes or None if the file is unreadable.
    """
    try:
        img = Image.open(file_path)
        img.thumbnail((max_dimension, max_dimension), Image.LANCZOS)
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        img.close()
        return buf.getvalue()
    except Exception:
        logger.warning("Failed to compress image for LLM: %s", file_path)
        return None
