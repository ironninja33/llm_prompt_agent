"""Shared helpers for API route modules."""

import json


# ── Placeholder SVG for missing/deleted images ───────────────────────────

MISSING_IMAGE_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 256 256">'
    '<rect width="256" height="256" fill="#3a3a4a"/>'
    '<line x1="80" y1="80" x2="176" y2="176" stroke="#888" stroke-width="6" stroke-linecap="round"/>'
    '<line x1="176" y1="80" x2="80" y2="176" stroke="#888" stroke-width="6" stroke-linecap="round"/>'
    '<text x="128" y="210" text-anchor="middle" fill="#888" font-family="sans-serif" font-size="14">'
    'Image not found</text>'
    '</svg>'
)


def _sse_event(event_type: str, data: dict) -> str:
    """Format a Server-Sent Event string."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def _cleanup_missing_image(job_id: str, image_id: int, file_path: str | None = None):
    """Clean up database records for a missing image file.

    Deletes the image record from generated_images and removes the
    corresponding embedding from the vector store.

    Args:
        job_id: The generation job ID.
        image_id: The image record ID.
        file_path: Normalized file path used as the vector store doc ID.
    """
    import logging
    import os
    from src.models import generation as gen_model, vector_store

    logger = logging.getLogger(__name__)

    # Delete the image record
    deleted = gen_model.delete_image(image_id)
    if deleted:
        logger.warning(
            "Deleted missing image record id=%d for job %s", image_id, job_id
        )

    # Remove this image's embedding from the vector store
    if file_path:
        doc_id = os.path.normpath(file_path)
        removed = vector_store.delete_document(doc_id, "output")
        if removed:
            logger.warning(
                "Deleted vector store document %s for job %s", doc_id, job_id,
            )
