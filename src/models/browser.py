"""Browser model — directory listing and image discovery for the image browser."""

import json
import os
import time
import uuid
import logging
from datetime import datetime, timezone

from sqlalchemy import text

from src.config import IGNORE_DIR_PREFIX
from src.models.database import get_db, row_to_dict
from src.models import settings
from src.services.image_parser import parse_file

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def parse_concept_name(raw_name: str) -> dict:
    """Parse 'category__display_name' folder naming convention.

    Returns {"raw": raw_name, "category": category_or_None, "display_name": name_part}.
    Folders without a ``__`` delimiter return category=None and display_name=raw_name.
    """
    if "__" in raw_name:
        category, _, display_name = raw_name.partition("__")
        return {"raw": raw_name, "category": category, "display_name": display_name}
    return {"raw": raw_name, "category": None, "display_name": raw_name}


def _output_folder_from_path(filepath: str) -> str:
    """Compute output_folder relative to the configured output root directory.

    For a file at ``<output_root>/character/subdir/image.png`` returns
    ``"character/subdir"`` instead of just ``"subdir"``.
    """
    dirpath = os.path.normpath(os.path.dirname(filepath))
    for root in get_root_directories():
        root_path = os.path.normpath(root["path"])
        if dirpath.startswith(root_path):
            rel = os.path.relpath(dirpath, root_path)
            return rel if rel != "." else ""
    # Fallback: immediate parent only (no matching root found)
    return os.path.basename(dirpath)


def get_root_directories() -> list[dict]:
    """Get all active output data_directories as virtual root entries."""
    dirs = settings.get_data_directories(active_only=True)
    roots = []
    for d in dirs:
        if d["dir_type"] != "output":
            continue
        if not os.path.isdir(d["path"]):
            continue
        roots.append({
            "name": os.path.basename(d["path"]),
            "path": d["path"],
            "dir_type": d["dir_type"],
            "id": d["id"],
        })
    return roots


def get_directory_contents(abs_dir_path: str, offset: int = 0, limit: int = 50,
                           sort: str = "date") -> dict:
    """Get paginated listing of subdirectories + images for a directory.

    Returns { directories: [...], images: [...], total_image_count, has_more }
    Directories come from filesystem scan; images from generated_images table.

    Sort modes:
        "date" (default): directories by most recent image DESC, images by created_at DESC
        "name": directories alphabetical, images by filename ASC
    """
    from src.models import generation as gen_model

    directories = []
    if os.path.isdir(abs_dir_path):
        try:
            entries = sorted(os.listdir(abs_dir_path))
        except OSError:
            entries = []

        for entry in entries:
            full_path = os.path.join(abs_dir_path, entry)
            if os.path.isdir(full_path) and not entry.startswith(IGNORE_DIR_PREFIX):
                image_count = _count_images_in_dir(full_path)
                parsed = parse_concept_name(entry)
                directories.append({
                    "name": entry,
                    "path": full_path,
                    "image_count": image_count,
                    "category": parsed["category"],
                    "display_name": parsed["display_name"],
                })

    # Sort directories
    if sort == "date":
        for d in directories:
            d["_latest"] = get_latest_image_timestamp(d["path"])
        directories.sort(key=lambda d: str(d["_latest"]) if d["_latest"] is not None else "", reverse=True)
        for d in directories:
            d.pop("_latest", None)
    # sort="name" keeps the existing alphabetical order from sorted(os.listdir)

    # Get images from DB for this directory
    result = gen_model.get_images_for_directory(abs_dir_path, offset, limit, sort=sort)

    return {
        "directories": directories,
        "images": result["images"],
        "total_image_count": result["total_count"],
        "has_more": result["has_more"],
    }


def fast_register_images(dir_path: str) -> int:
    """Fast-register images in a directory without parsing metadata.

    Only uses os.listdir + os.stat — no PIL, no metadata parsing.
    New files get metadata_status='pending'. Returns count of newly registered images.
    """
    if not os.path.isdir(dir_path):
        return 0

    try:
        entries = os.listdir(dir_path)
    except OSError:
        return 0

    # Filter to image files and build full paths
    image_files = []
    for fname in entries:
        ext = os.path.splitext(fname)[1].lower()
        if ext in IMAGE_EXTS:
            image_files.append((fname, os.path.normpath(os.path.join(dir_path, fname))))

    disk_paths = set(fp for _, fp in image_files)

    # Remove DB records for files that were deleted from disk.
    norm_dir = os.path.normpath(dir_path) + os.sep
    with get_db() as conn:
        result = conn.execute(
            text("""SELECT gi.id, gi.job_id, gi.file_path
               FROM generated_images gi
               WHERE gi.file_path LIKE :like AND gi.file_path NOT LIKE :not_like"""),
            {"like": norm_dir + "%", "not_like": norm_dir + "%/%"},
        )
        for row in result.fetchall():
            r = row._mapping
            if r["file_path"] not in disk_paths:
                conn.execute(
                    text("DELETE FROM generated_images WHERE id = :id"),
                    {"id": r["id"]},
                )
                # Clean up orphan job if no images remain
                remaining = conn.execute(
                    text("SELECT COUNT(*) as cnt FROM generated_images WHERE job_id = :job_id"),
                    {"job_id": r["job_id"]},
                ).fetchone()._mapping["cnt"]
                if remaining == 0:
                    conn.execute(
                        text("DELETE FROM generation_jobs WHERE id = :id"),
                        {"id": r["job_id"]},
                    )
                logger.debug("Removed stale DB record for deleted file: %s", r["file_path"])

    # If the directory is now empty (no images, no subdirs), remove it from disk.
    if not image_files:
        _maybe_remove_empty_dir(dir_path)

    if not image_files:
        return 0

    # Batch existence check (chunked at 500)
    all_paths = [fp for _, fp in image_files]
    existing_paths = set()
    with get_db() as conn:
        for i in range(0, len(all_paths), 500):
            chunk = all_paths[i:i + 500]
            placeholders = ",".join([f":p{j}" for j in range(len(chunk))])
            params = {f"p{j}": v for j, v in enumerate(chunk)}
            result = conn.execute(
                text(f"SELECT file_path FROM generated_images WHERE file_path IN ({placeholders})"),
                params,
            )
            existing_paths.update(row._mapping["file_path"] for row in result.fetchall())

    # Register new files
    new_files = [(fname, fp) for fname, fp in image_files if fp not in existing_paths]
    if not new_files:
        return 0

    # Check for existing records from generation (file_path IS NULL, same filename).
    new_filenames = [fname for fname, _ in new_files]
    existing_by_name = {}
    with get_db() as conn:
        for i in range(0, len(new_filenames), 500):
            chunk = new_filenames[i:i + 500]
            placeholders = ",".join([f":p{j}" for j in range(len(chunk))])
            params = {f"p{j}": v for j, v in enumerate(chunk)}
            result = conn.execute(
                text(f"""SELECT gi.id, gi.filename, gi.job_id
                    FROM generated_images gi
                    WHERE gi.filename IN ({placeholders}) AND gi.file_path IS NULL
                    ORDER BY gi.id DESC"""),
                params,
            )
            for row in result.fetchall():
                r = row._mapping
                # Keep the most recent record per filename (highest id)
                if r["filename"] not in existing_by_name:
                    existing_by_name[r["filename"]] = dict(r)

    count = 0
    now = time.time()
    with get_db() as conn:
        for fname, filepath in new_files:
            try:
                stat = os.stat(filepath)
                file_size = stat.st_size
                file_mtime_epoch = stat.st_mtime
                file_mtime = datetime.fromtimestamp(file_mtime_epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            except OSError:
                continue

            # Skip very recent files — likely from an in-progress generation.
            # The completion callback will create the proper record with the
            # correct job_id and settings.
            if now - file_mtime_epoch < 2.0 and fname not in existing_by_name:
                continue

            if fname in existing_by_name:
                # Update existing generation record with file_path
                existing = existing_by_name[fname]
                conn.execute(
                    text("""UPDATE generated_images
                       SET file_path = :file_path, file_size = :file_size, metadata_status = 'complete'
                       WHERE id = :id"""),
                    {"file_path": filepath, "file_size": file_size, "id": existing["id"]},
                )
                count += 1
            else:
                # Truly new file — create job + image record
                job_id = str(uuid.uuid4())
                conn.execute(
                    text("""INSERT OR IGNORE INTO generation_jobs
                       (id, chat_id, message_id, prompt_id, status, source, created_at, completed_at)
                       VALUES (:id, NULL, NULL, NULL, 'completed', 'scan', :created_at, :completed_at)"""),
                    {"id": job_id, "created_at": file_mtime, "completed_at": file_mtime},
                )
                result = conn.execute(
                    text("""INSERT OR IGNORE INTO generated_images
                       (job_id, filename, subfolder, file_size, file_path, metadata_status)
                       VALUES (:job_id, :filename, '', :file_size, :file_path, 'pending')"""),
                    {"job_id": job_id, "filename": fname, "file_size": file_size, "file_path": filepath},
                )
                if result.rowcount > 0:
                    count += 1
                else:
                    # Image already existed (race condition), clean up orphan job
                    conn.execute(
                        text("DELETE FROM generation_jobs WHERE id = :id"),
                        {"id": job_id},
                    )

    return count


def parse_pending_for_page(image_ids: list[int]) -> int:
    """Parse metadata for images that have metadata_status='pending'.

    For each pending image: parse the file, INSERT generation_settings,
    UPDATE metadata_status to 'complete'. If parse fails, insert minimal
    empty settings and still mark complete (no infinite retry).

    Returns count of images that were parsed.
    """
    if not image_ids:
        return 0

    placeholders = ",".join([f":p{i}" for i in range(len(image_ids))])
    params = {f"p{i}": v for i, v in enumerate(image_ids)}

    with get_db() as conn:
        result = conn.execute(
            text(f"""SELECT gi.id, gi.job_id, gi.file_path
                FROM generated_images gi
                WHERE gi.id IN ({placeholders}) AND gi.metadata_status = 'pending'"""),
            params,
        )
        pending = result.fetchall()

    if not pending:
        return 0

    count = 0
    for row in pending:
        r = row._mapping
        image_id = r["id"]
        job_id = r["job_id"]
        filepath = r["file_path"]

        parsed = parse_file(filepath) if filepath else None

        with get_db() as conn:
            if parsed:
                loras_json = json.dumps(parsed.loras) if parsed.loras else None
                conn.execute(
                    text("""INSERT OR IGNORE INTO generation_settings
                       (job_id, positive_prompt, negative_prompt, base_model, loras,
                        output_folder, seed, num_images, sampler, cfg_scale, scheduler, steps)
                       VALUES (:job_id, :positive_prompt, :negative_prompt, :base_model, :loras,
                        :output_folder, :seed, 1, :sampler, :cfg_scale, :scheduler, :steps)"""),
                    {
                        "job_id": job_id,
                        "positive_prompt": parsed.prompt,
                        "negative_prompt": parsed.negative_prompt,
                        "base_model": parsed.base_model,
                        "loras": loras_json,
                        "output_folder": _output_folder_from_path(filepath),
                        "seed": parsed.seed if parsed.seed is not None else -1,
                        "sampler": parsed.sampler,
                        "cfg_scale": parsed.cfg_scale,
                        "scheduler": parsed.scheduler,
                        "steps": parsed.steps,
                    },
                )
                # Update width/height if available
                if parsed.raw_workflow is None:
                    try:
                        from PIL import Image
                        img = Image.open(filepath)
                        w, h = img.size
                        img.close()
                        conn.execute(
                            text("UPDATE generated_images SET width = :w, height = :h WHERE id = :id"),
                            {"w": w, "h": h, "id": image_id},
                        )
                    except Exception:
                        pass
            else:
                # Parse failed — insert minimal empty settings so we don't retry
                conn.execute(
                    text("""INSERT OR IGNORE INTO generation_settings
                       (job_id, positive_prompt, negative_prompt, base_model, loras,
                        output_folder, seed, num_images)
                       VALUES (:job_id, '', NULL, NULL, NULL, :output_folder, -1, 1)"""),
                    {"job_id": job_id,
                     "output_folder": _output_folder_from_path(filepath) if filepath else ""},
                )

            conn.execute(
                text("UPDATE generated_images SET metadata_status = 'complete' WHERE id = :id"),
                {"id": image_id},
            )
        count += 1

    return count


def get_directory_previews(dir_path: str, count: int = 4) -> list[dict]:
    """Get up to `count` image job_id/image_id pairs for directory thumbnail previews."""
    if not dir_path.endswith(os.sep):
        dir_path = dir_path + os.sep
    like_pattern = dir_path + "%"

    with get_db() as conn:
        result = conn.execute(
            text("""SELECT gi.id as image_id, gi.job_id
               FROM generated_images gi
               JOIN generation_jobs gj ON gi.job_id = gj.id
               WHERE gi.file_path LIKE :like
               ORDER BY gj.created_at DESC
               LIMIT :limit"""),
            {"like": like_pattern, "limit": count},
        )
        return [{"image_id": row._mapping["image_id"], "job_id": row._mapping["job_id"]}
                for row in result.fetchall()]


def get_latest_image_timestamp(dir_path: str) -> str | None:
    """Get the most recent image creation date under dir_path (recursive).

    Uses generation_jobs.created_at which stores the file modification time
    for scanned images (not the DB insertion time).
    """
    if not dir_path.endswith(os.sep):
        dir_path = dir_path + os.sep

    with get_db() as conn:
        result = conn.execute(
            text("""SELECT MAX(gj.created_at) as latest
               FROM generated_images gi
               JOIN generation_jobs gj ON gi.job_id = gj.id
               WHERE gi.file_path LIKE :like"""),
            {"like": dir_path + "%"},
        )
        row = result.fetchone()
        return row._mapping["latest"] if row else None


def _maybe_remove_empty_dir(dir_path: str):
    """Remove empty subdirectories, walking up to (but not including) root output dirs.

    Only deletes if the directory contains no files and no subdirectories.
    Root output directories (configured in data_directories) are never removed.
    Walks up the tree so nested empty directories are cleaned up.
    """
    roots = get_root_directories()
    root_paths = {os.path.normpath(r["path"]) for r in roots}

    current = os.path.normpath(dir_path)
    while current not in root_paths:
        try:
            if not os.listdir(current):
                os.rmdir(current)
                logger.info("Removed empty directory: %s", current)
                current = os.path.dirname(current)
            else:
                break  # Not empty, stop walking up
        except OSError:
            break  # Permission error or not a directory — stop


def _count_images_in_dir(dir_path: str) -> int:
    """Count images in the DB whose file_path starts with the given directory."""
    if not dir_path.endswith(os.sep):
        dir_path = dir_path + os.sep
    like_pattern = dir_path + "%"

    with get_db() as conn:
        result = conn.execute(
            text("SELECT COUNT(*) as cnt FROM generated_images WHERE file_path LIKE :like"),
            {"like": like_pattern},
        )
        return result.fetchone()._mapping["cnt"]
