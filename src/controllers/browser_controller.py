"""Browser controller — business logic for the image browser."""

import os
import logging

from src.config import IGNORE_DIR_PREFIX
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
    abs_path = resolve_virtual_path(virtual_path)
    if abs_path is None:
        return {"error": "Invalid path", "directories": [], "images": [],
                "total_image_count": 0, "has_more": False}

    # Phase 1: Fast-register (current dir + immediate subdirs for previews)
    try:
        new_count = browser_model.fast_register_images(abs_path)
        if os.path.isdir(abs_path):
            for entry in os.listdir(abs_path):
                sub = os.path.join(abs_path, entry)
                if os.path.isdir(sub) and not entry.startswith(IGNORE_DIR_PREFIX):
                    new_count += browser_model.fast_register_images(sub)
        if new_count > 0:
            logger.info("Fast-registered %d new images in %s", new_count, abs_path)
    except Exception as e:
        logger.warning("Error fast-registering images in %s: %s", abs_path, e)

    # Phase 2: Query DB for paginated results
    result = browser_model.get_directory_contents(abs_path, offset, limit, sort=sort)

    # Phase 3: Parse metadata for any pending images on this page
    pending_ids = [img["id"] for img in result["images"] if img.get("metadata_status") == "pending"]
    parsed_images = []
    if pending_ids:
        try:
            parsed_count, parsed_images = browser_model.parse_pending_for_page(pending_ids)
            if parsed_count > 0:
                logger.info("Parsed metadata for %d images", parsed_count)
                # Phase 4: Re-fetch page with full metadata
                result = browser_model.get_directory_contents(abs_path, offset, limit, sort=sort)
        except Exception as e:
            logger.warning("Error parsing pending metadata: %s", e)

    # Phase 3b: Trigger embedding for newly parsed images
    if parsed_images:
        from src.controllers.generation_controller import embed_image
        for pi in parsed_images:
            if pi.get("prompt") and pi.get("file_path"):
                embed_image(pi["file_path"], pi["prompt"])

    # Derive output_folder from actual file location so it's always correct
    # even if the stored DB value is stale from a previous move/reorg.
    for img in result["images"]:
        if img.get("file_path"):
            img["settings"]["output_folder"] = browser_model._output_folder_from_path(img["file_path"])

    # Recursive image count (current dir + all subdirs)
    result["recursive_image_count"] = browser_model._count_images_in_dir(abs_path)

    # Add previews for subdirectories
    for d in result["directories"]:
        d["previews"] = browser_model.get_directory_previews(d["path"])

    return result


def get_breadcrumb(virtual_path: str) -> list[dict]:
    """Build breadcrumb segments for navigation."""
    from src.models.browser import parse_concept_name

    if not virtual_path:
        return [{"name": "Root", "path": ""}]

    segments = [{"name": "Root", "path": ""}]
    parts = virtual_path.strip("/").split("/")
    current = ""
    for idx, part in enumerate(parts):
        current = f"{current}/{part}" if current else part
        seg = {"name": part, "path": current}
        # Concept-level segments (depth 2: root_dir/concept)
        if idx == 1:
            parsed = parse_concept_name(part)
            seg["category"] = parsed["category"]
            seg["display_name"] = parsed["display_name"]
        segments.append(seg)
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

        if not results:
            return {"images": [], "total_count": 0, "has_more": False}

        # Map doc_ids back to generated_images
        # search_similar returns list[dict] with "id" key per entry.
        # Doc IDs from ingestion are file paths; from generation are "gen_{job_id}"
        doc_ids = [r["id"] for r in results]
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

    Read-only: uses os.stat mtime to detect directory changes without
    writing to the DB. Actual registration happens on full loadBrowserContents.
    Also reports agent_busy so the frontend can back off polling.
    """
    from src.agent import runner

    abs_path = resolve_virtual_path(virtual_path) if virtual_path else None
    agent_busy = runner.has_active_runs()

    if abs_path is None:
        # Root level: check all output directories for mtime changes
        roots = browser_model.get_root_directories()
        has_changes = any(_dir_changed_since(root["path"], since) for root in roots)
        return {"has_new_files": has_changes, "agent_busy": agent_busy}

    has_changes = _dir_changed_since(abs_path, since)
    return {"has_new_files": has_changes, "agent_busy": agent_busy}


def _dir_changed_since(abs_path: str, since: float) -> bool:
    """Check if a directory or any immediate subdirectory was modified since timestamp."""
    try:
        if os.stat(abs_path).st_mtime > since:
            return True
        for entry in os.listdir(abs_path):
            sub = os.path.join(abs_path, entry)
            if os.path.isdir(sub) and os.stat(sub).st_mtime > since:
                return True
    except OSError:
        pass
    return False


def recluster_folder(virtual_path: str, k: int) -> dict:
    """Save per-folder k and trigger single-folder recluster."""
    from src.models import settings
    from src.services import clustering_service

    if clustering_service.is_running():
        return {"error": "Clustering already running"}

    # Extract concept name (same logic as suggest_subfolders)
    parts = virtual_path.strip("/").split("/")
    concept_name = parts[1] if len(parts) >= 2 else parts[0]

    # Save per-folder k override (browser operates on output folders)
    settings.update_setting(f"cluster_k_intra:{concept_name}:output", str(k))

    # Start single-folder recluster for output source type
    clustering_service.start_clustering_single(concept_name, k, source_type="output")
    return {"ok": True}


def suggest_subfolders(virtual_path: str) -> dict:
    """Get proposed subfolder split based on intra-folder clusters."""
    import re

    from src.models.database import get_db, row_to_dict
    from sqlalchemy import text

    abs_path = resolve_virtual_path(virtual_path)
    if abs_path is None:
        return {"error": "Invalid path", "subfolders": []}

    # Get the concept name — matches ingestion logic where the concept is the
    # first subdirectory within a registered root, not the root itself.
    parts = virtual_path.strip("/").split("/")
    concept_name = parts[1] if len(parts) >= 2 else parts[0]

    # Get intra-folder clusters for this concept (output source type — browser operates on output dirs)
    with get_db() as conn:
        result = conn.execute(
            text("""SELECT c.id, c.label, c.prompt_count
               FROM clusters c
               WHERE c.cluster_type = 'intra_folder' AND c.folder_path = :folder_path
               AND c.source_type = 'output'
               ORDER BY c.id"""),
            {"folder_path": concept_name},
        )
        clusters = [dict(row._mapping) for row in result.fetchall()]

    if not clusters:
        return {"subfolders": [], "message": "No clusters found. Run clustering first."}

    subfolders = []
    with get_db() as conn:
        for cluster in clusters:
            # Get image IDs assigned to this cluster.
            # Match on exact file_path first, fall back to basename match
            # (handles files moved by a previous reorg where doc_id is stale).
            result = conn.execute(
                text("""SELECT ca.doc_id, gi.id as image_id, gi.job_id
                   FROM cluster_assignments ca
                   JOIN generated_images gi ON (
                       ca.doc_id = gi.file_path
                       OR ca.doc_id LIKE '%/' || gi.filename
                   )
                   WHERE ca.cluster_id = :cluster_id"""),
                {"cluster_id": cluster["id"]},
            )
            members = [dict(row._mapping) for row in result.fetchall()]
            image_ids = [m["image_id"] for m in members]

            if not image_ids:
                continue

            # Normalize label to subfolder name
            label = cluster["label"] or f"group_{cluster['id']}"
            subfolder_name = _normalize_label(label)

            # Get 4 sample image IDs for preview thumbnails
            sample_ids = image_ids[:4]
            sample_previews = []
            for sid in sample_ids:
                row = conn.execute(
                    text("SELECT id, job_id FROM generated_images WHERE id = :id"),
                    {"id": sid},
                ).fetchone()
                if row:
                    r = row._mapping
                    sample_previews.append({"image_id": r["id"], "job_id": r["job_id"]})

            subfolders.append({
                "name": subfolder_name,
                "label": cluster["label"],
                "image_count": len(image_ids),
                "image_ids": image_ids,
                "sample_previews": sample_previews,
            })

    # Find images with no cluster assignment -> 'misc' group
    with get_db() as conn:
        assigned_ids = set()
        for sf in subfolders:
            assigned_ids.update(sf["image_ids"])

        result = conn.execute(
            text("""SELECT gi.id as image_id, gi.job_id
               FROM generated_images gi
               LEFT JOIN generation_settings gs ON gs.job_id = gi.job_id
               WHERE gi.file_path LIKE :path_prefix
               AND gi.file_path IS NOT NULL"""),
            {"path_prefix": abs_path + "%"},
        )
        all_in_dir = [dict(row._mapping) for row in result.fetchall()]
        misc_images = [m for m in all_in_dir if m["image_id"] not in assigned_ids]

        if misc_images:
            misc_ids = [m["image_id"] for m in misc_images]
            subfolders.append({
                "name": "misc",
                "label": "Unclustered",
                "image_count": len(misc_ids),
                "image_ids": misc_ids,
                "sample_previews": [{"image_id": m["image_id"], "job_id": m["job_id"]}
                                    for m in misc_images[:4]],
            })

    return {"subfolders": subfolders}


def execute_reorg(virtual_path: str, subfolders: list[dict]) -> dict:
    """Move files into subfolders and update DB/ChromaDB."""
    from src.models.database import get_db
    from src.models import vector_store
    from sqlalchemy import text

    abs_path = resolve_virtual_path(virtual_path)
    if abs_path is None:
        return {"error": "Invalid path", "moved": 0, "errors": []}

    moved = 0
    errors = []

    for subfolder in subfolders:
        subfolder_name = subfolder["name"]
        image_ids = subfolder.get("image_ids", [])
        if not image_ids or not subfolder_name:
            continue

        new_dir = os.path.join(abs_path, subfolder_name)
        os.makedirs(new_dir, exist_ok=True)

        for image_id in image_ids:
            try:
                with get_db() as conn:
                    row = conn.execute(
                        text("SELECT id, file_path, job_id FROM generated_images WHERE id = :id"),
                        {"id": image_id},
                    ).fetchone()

                    if not row:
                        continue
                    r = row._mapping
                    old_path = r["file_path"]
                    if not old_path or not os.path.isfile(old_path):
                        continue

                    filename = os.path.basename(old_path)
                    new_path = os.path.join(new_dir, filename)

                    # Move file
                    os.rename(old_path, new_path)

                    # Update DB
                    conn.execute(
                        text("UPDATE generated_images SET file_path = :new_path WHERE id = :id"),
                        {"new_path": new_path, "id": image_id},
                    )

                    # Update output_folder in generation_settings
                    new_output_folder = browser_model._output_folder_from_path(new_path)
                    conn.execute(
                        text("UPDATE generation_settings SET output_folder = :folder WHERE job_id = :jid"),
                        {"folder": new_output_folder, "jid": r["job_id"]},
                    )

                    # Update cluster_assignments doc_id so future lookups
                    # don't rely on the filename fallback
                    conn.execute(
                        text("UPDATE cluster_assignments SET doc_id = :new_path WHERE doc_id = :old_path"),
                        {"new_path": new_path, "old_path": old_path},
                    )

                moved += 1

            except Exception as e:
                errors.append({"image_id": image_id, "error": str(e)})
                logger.error("Reorg move failed for image %s: %s", image_id, e)

    return {"moved": moved, "errors": errors}


def _normalize_label(label: str) -> str:
    """Normalize a cluster label to a valid subfolder name."""
    import re
    # Remove punctuation, lowercase, replace spaces with underscores
    name = re.sub(r'[^\w\s-]', '', label.lower())
    name = re.sub(r'[\s-]+', '_', name).strip('_')
    # Truncate to reasonable length
    if len(name) > 40:
        name = name[:40].rstrip('_')
    return name or "group"


def resolve_virtual_path(virtual_path: str) -> str | None:
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
