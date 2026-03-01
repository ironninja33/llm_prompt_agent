"""Browser controller — business logic for the image browser."""

import os
import logging

from src.models import browser as browser_model
from src.models import generation as gen_model

logger = logging.getLogger(__name__)


def get_root_listing(sort: str = "date") -> list[dict]:
    """Virtual root: all output directories with preview thumbnails."""
    roots = browser_model.get_root_directories()
    for root in roots:
        root["previews"] = browser_model.get_directory_previews(root["path"])
        root["image_count"] = browser_model._count_images_in_dir(root["path"])

    if sort == "date":
        for root in roots:
            root["_latest"] = browser_model.get_latest_image_timestamp(root["path"])
        roots.sort(key=lambda r: str(r["_latest"]) if r["_latest"] is not None else "", reverse=True)
        for root in roots:
            root.pop("_latest", None)

    return roots


def get_directory_contents(virtual_path: str, offset: int = 0, limit: int = 50,
                           sort: str = "date") -> dict:
    """Resolve virtual path to absolute path and return paginated results.

    Two-phase approach:
    1. Fast-register all files (os.listdir + os.stat, no PIL) — ~300ms for 10k files
    2. Query DB for paginated results
    3. Parse metadata for only the page's pending images — ~1-2s max for 50 images
    4. Re-fetch page if any were pending (now with full metadata)
    """
    abs_path = _resolve_virtual_path(virtual_path)
    if abs_path is None:
        return {"error": "Invalid path", "directories": [], "images": [],
                "total_image_count": 0, "has_more": False}

    # Phase 1: Fast-register (current dir + immediate subdirs for previews)
    try:
        new_count = browser_model.fast_register_images(abs_path)
        if os.path.isdir(abs_path):
            for entry in os.listdir(abs_path):
                sub = os.path.join(abs_path, entry)
                if os.path.isdir(sub):
                    new_count += browser_model.fast_register_images(sub)
        if new_count > 0:
            logger.info("Fast-registered %d new images in %s", new_count, abs_path)
    except Exception as e:
        logger.warning("Error fast-registering images in %s: %s", abs_path, e)

    # Phase 2: Query DB for paginated results
    result = browser_model.get_directory_contents(abs_path, offset, limit, sort=sort)

    # Phase 3: Parse metadata for any pending images on this page
    pending_ids = [img["id"] for img in result["images"] if img.get("metadata_status") == "pending"]
    if pending_ids:
        try:
            parsed_count = browser_model.parse_pending_for_page(pending_ids)
            if parsed_count > 0:
                logger.info("Parsed metadata for %d images", parsed_count)
                # Phase 4: Re-fetch page with full metadata
                result = browser_model.get_directory_contents(abs_path, offset, limit, sort=sort)
        except Exception as e:
            logger.warning("Error parsing pending metadata: %s", e)

    # Add previews for subdirectories
    for d in result["directories"]:
        d["previews"] = browser_model.get_directory_previews(d["path"])

    return result


def get_breadcrumb(virtual_path: str) -> list[dict]:
    """Build breadcrumb segments for navigation."""
    if not virtual_path:
        return [{"name": "Root", "path": ""}]

    segments = [{"name": "Root", "path": ""}]
    parts = virtual_path.strip("/").split("/")
    current = ""
    for part in parts:
        current = f"{current}/{part}" if current else part
        segments.append({"name": part, "path": current})
    return segments


def search_keyword(keywords_str: str, offset: int = 0, limit: int = 50) -> dict:
    """Split by comma, trim, query generation_settings.positive_prompt with LIKE."""
    keywords = [kw.strip() for kw in keywords_str.split(",") if kw.strip()]
    if not keywords:
        return {"images": [], "total_count": 0, "has_more": False}
    return gen_model.search_by_keywords(keywords, offset, limit)


def search_embedding(query: str, offset: int = 0, limit: int = 50) -> dict:
    """Embed query, search vector store, map results back to generated_images."""
    try:
        from src.services import embedding_service
        from src.models import vector_store

        embedding = embedding_service.embed(query)
        if not embedding:
            return {"images": [], "total_count": 0, "has_more": False}

        results = vector_store.search_similar(
            embedding, source_type="output", k=limit
        )

        if not results or not results.get("ids"):
            return {"images": [], "total_count": 0, "has_more": False}

        # Map doc_ids back to generated_images
        # Doc IDs from ingestion are file paths; from generation are "gen_{job_id}"
        doc_ids = results["ids"][0] if results["ids"] else []
        job_ids = []
        file_paths = []
        for doc_id in doc_ids:
            if doc_id.startswith("gen_"):
                job_ids.append(doc_id[4:])  # Strip "gen_" prefix
            else:
                file_paths.append(doc_id)

        images = []

        # Get images by job IDs
        if job_ids:
            images.extend(gen_model.get_images_by_job_ids(job_ids))

        # Get images by file paths
        if file_paths:
            from src.models.database import get_db
            from sqlalchemy import text
            with get_db() as conn:
                for fp in file_paths:
                    result = conn.execute(
                        text("""SELECT gi.id as image_id, gi.job_id, gi.filename, gi.subfolder,
                                  gi.width, gi.height, gi.created_at, gi.file_size, gi.file_path,
                                  gj.status, gj.source,
                                  gs.positive_prompt, gs.negative_prompt, gs.base_model, gs.loras,
                                  gs.output_folder, gs.seed, gs.num_images, gs.sampler, gs.cfg_scale,
                                  gs.scheduler, gs.steps
                           FROM generated_images gi
                           JOIN generation_jobs gj ON gi.job_id = gj.id
                           LEFT JOIN generation_settings gs ON gi.job_id = gs.job_id
                           WHERE gi.file_path = :fp"""),
                        {"fp": fp},
                    )
                    rows = result.fetchall()
                    if rows:
                        images.extend(gen_model._rows_to_image_list(rows))

        return {
            "images": images[:limit],
            "total_count": len(images),
            "has_more": False,  # Embedding search doesn't paginate naturally
        }

    except Exception as e:
        logger.error("Embedding search error: %s", e, exc_info=True)
        return {"images": [], "total_count": 0, "has_more": False}


def poll_new_files(virtual_path: str, since: float) -> dict:
    """Check for new files/directories since timestamp.

    Uses fast_register_images (no parsing) so polls stay lightweight.
    Also scans immediate subdirectories so newly created output folders
    (e.g. from a generation with a new subfolder) are detected.
    Returns has_new_files flag so the frontend can trigger a full reload.
    """
    abs_path = _resolve_virtual_path(virtual_path) if virtual_path else None

    if abs_path is None:
        # Root level: check all output directories + their immediate subdirs
        roots = browser_model.get_root_directories()
        new_count = 0
        for root in roots:
            new_count += _register_dir_and_subdirs(root["path"])
        return {"new_count": new_count, "has_new_files": new_count > 0}

    # Current directory + immediate subdirectories
    new_count = _register_dir_and_subdirs(abs_path)
    return {"new_count": new_count, "has_new_files": new_count > 0}


def _register_dir_and_subdirs(abs_path: str) -> int:
    """Fast-register images in a directory and its immediate subdirectories."""
    new_count = browser_model.fast_register_images(abs_path)
    try:
        for entry in os.listdir(abs_path):
            sub = os.path.join(abs_path, entry)
            if os.path.isdir(sub):
                new_count += browser_model.fast_register_images(sub)
    except OSError:
        pass
    return new_count


def _resolve_virtual_path(virtual_path: str) -> str | None:
    """Map a virtual path like 'output_dir/subdir' to an absolute filesystem path.

    The first segment matches a data_directories entry by basename.
    """
    if not virtual_path:
        return None

    parts = virtual_path.strip("/").split("/", 1)
    root_name = parts[0]
    rest = parts[1] if len(parts) > 1 else ""

    roots = browser_model.get_root_directories()
    for root in roots:
        if os.path.basename(root["path"]) == root_name:
            abs_path = os.path.join(root["path"], rest) if rest else root["path"]
            abs_path = os.path.normpath(abs_path)
            # Security: ensure resolved path is still under the root
            if abs_path.startswith(os.path.normpath(root["path"])):
                return abs_path
            return None

    return None
