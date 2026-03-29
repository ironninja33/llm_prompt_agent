"""Data sampler — query DB for images, copy to working directory with prompt sidecars."""

import logging
import os
import random
import shutil
from dataclasses import dataclass

from sqlalchemy import text

from src.models.database import get_db

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = "data/images"


@dataclass
class SampledImage:
    """An image sampled from the database and copied to the working directory."""

    file_path: str  # path to the copy in output_dir
    original_path: str
    filename: str
    prompt: str | None
    file_size: int
    width: int
    height: int


def sample_images(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    query: str | None = None,
    folder: str | None = None,
    k: int = 10,
    seed: int | None = None,
) -> tuple[list[SampledImage], int]:
    """Query DB for images, copy to output_dir with .txt prompt sidecar files.

    Args:
        output_dir: Directory to copy images into (cleared first).
        query: Comma-separated keywords to LIKE-match against positive_prompt.
        folder: Virtual path (e.g. "output/character__alice") or absolute path
                to filter images. Includes subfolders.
        k: Maximum number of images to sample.
        seed: Random seed for reproducible sampling. Generated if None.

    Returns:
        Tuple of (list of SampledImage, seed used).
    """
    # Clear output dir to avoid stale data from previous runs
    if os.path.isdir(output_dir):
        for entry in os.listdir(output_dir):
            full = os.path.join(output_dir, entry)
            if os.path.isfile(full):
                os.remove(full)
    os.makedirs(output_dir, exist_ok=True)

    # Build query
    conditions = ["gi.file_path IS NOT NULL"]
    params: dict = {}

    # Keyword filter
    if query:
        keywords = [kw.strip() for kw in query.split(",") if kw.strip()]
        for i, kw in enumerate(keywords):
            conditions.append(f"gs.positive_prompt LIKE :kw{i}")
            params[f"kw{i}"] = f"%{kw}%"

    # Folder filter
    if folder:
        abs_folder = _resolve_folder(folder)
        if abs_folder:
            if not abs_folder.endswith(os.sep):
                abs_folder += os.sep
            conditions.append("gi.file_path LIKE :folder_prefix")
            params["folder_prefix"] = abs_folder + "%"
            logger.info("Folder filter: %s", abs_folder)
        else:
            logger.warning("Could not resolve folder path: %s", folder)

    where_clause = " AND ".join(conditions)

    # Generate seed if not provided
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    logger.info("Random seed: %d", seed)

    # Fetch all matching rows (random sampling done in Python for seedability)
    with get_db() as conn:
        result = conn.execute(
            text(f"""SELECT gi.file_path, gi.filename, gi.width, gi.height, gi.file_size,
                       gs.positive_prompt
                FROM generated_images gi
                JOIN generation_settings gs ON gi.job_id = gs.job_id
                WHERE {where_clause}"""),
            params,
        )
        rows = result.fetchall()

    if not rows:
        logger.warning("No images matched the query")
        return [], seed

    # Randomly sample k images using the seed
    rng = random.Random(seed)
    total_candidates = len(rows)
    if len(rows) > k:
        rows = rng.sample(rows, k)
    else:
        rng.shuffle(rows)

    logger.info("Sampled %d images from %d candidates", len(rows), total_candidates)

    # Copy images and create sidecar .txt files
    sampled = []
    used_names: set[str] = set()

    for row in rows:
        r = row._mapping
        original_path = r["file_path"]
        if not original_path or not os.path.isfile(original_path):
            logger.warning("Image file missing from disk: %s", original_path)
            continue

        # Handle filename collisions
        filename = r["filename"] or os.path.basename(original_path)
        filename = _deduplicate_filename(filename, used_names)
        used_names.add(filename)

        dest_path = os.path.join(output_dir, filename)
        shutil.copy2(original_path, dest_path)

        # Write prompt sidecar
        prompt = r["positive_prompt"]
        stem = os.path.splitext(filename)[0]
        txt_path = os.path.join(output_dir, f"{stem}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(prompt or "")

        sampled.append(SampledImage(
            file_path=dest_path,
            original_path=original_path,
            filename=filename,
            prompt=prompt,
            file_size=r["file_size"] or 0,
            width=r["width"] or 0,
            height=r["height"] or 0,
        ))

    logger.info("Copied %d images to %s", len(sampled), output_dir)
    return sampled, seed


def _resolve_folder(folder: str) -> str | None:
    """Resolve a virtual path or absolute path to an absolute directory path."""
    # If it's already an absolute path that exists, use it directly
    if os.path.isabs(folder) and os.path.isdir(folder):
        return os.path.normpath(folder)

    # Try resolving as a virtual path via browser_controller
    try:
        from src.controllers.browser_controller import resolve_virtual_path

        resolved = resolve_virtual_path(folder)
        if resolved and os.path.isdir(resolved):
            return os.path.normpath(resolved)
    except Exception as e:
        logger.debug("Virtual path resolution failed: %s", e)

    return None


def _deduplicate_filename(filename: str, used: set[str]) -> str:
    """Append numeric suffix if filename already used."""
    if filename not in used:
        return filename

    stem, ext = os.path.splitext(filename)
    counter = 2
    while True:
        candidate = f"{stem}_{counter}{ext}"
        if candidate not in used:
            return candidate
        counter += 1
