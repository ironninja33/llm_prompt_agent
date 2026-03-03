"""Cleanup service — heuristic scoring, near-duplicate detection, bulk parse."""

import logging
import os
import re
import threading
from datetime import datetime, timezone

import numpy as np
from sqlalchemy import text

from src.models.database import get_db
from src.models import scoring as scoring_model
from src.models import vector_store

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bulk metadata parse
# ---------------------------------------------------------------------------

_parse_lock = threading.Lock()
_parse_progress = {"running": False, "parsed": 0, "total": 0}


def get_parse_progress() -> dict:
    """Return current bulk parse progress.

    When a parse is not running, queries the DB for the pending count so the
    frontend can decide whether to trigger one.
    """
    if _parse_progress["running"]:
        return dict(_parse_progress)

    # Not running — check DB for pending images
    try:
        with get_db() as conn:
            result = conn.execute(
                text("SELECT COUNT(*) as cnt FROM generated_images WHERE metadata_status = 'pending'")
            )
            pending = result.fetchone()._mapping["cnt"]
        return {"running": False, "parsed": 0, "total": pending}
    except Exception:
        return dict(_parse_progress)


def start_bulk_parse():
    """Start a background thread to parse metadata for all pending images.

    Reuses ``browser.parse_pending_for_page()`` in batches.
    """
    if not _parse_lock.acquire(blocking=False):
        logger.info("Bulk parse already running, skipping")
        return

    def _run():
        try:
            from src.models.browser import parse_pending_for_page

            # Get all pending image IDs
            with get_db() as conn:
                result = conn.execute(
                    text("""SELECT id FROM generated_images
                       WHERE metadata_status = 'pending'
                       ORDER BY id""")
                )
                pending_ids = [row._mapping["id"] for row in result.fetchall()]

            _parse_progress["total"] = len(pending_ids)
            _parse_progress["parsed"] = 0
            _parse_progress["running"] = True

            if not pending_ids:
                logger.info("No pending images to parse")
                return

            logger.info("Bulk metadata parse starting: %d images", len(pending_ids))
            batch_size = 50
            for i in range(0, len(pending_ids), batch_size):
                batch = pending_ids[i:i + batch_size]
                parsed = parse_pending_for_page(batch)
                _parse_progress["parsed"] += parsed
                logger.debug("Bulk parse progress: %d / %d",
                             _parse_progress["parsed"], _parse_progress["total"])

            logger.info("Bulk metadata parse complete: %d images parsed",
                        _parse_progress["parsed"])
        except Exception:
            logger.exception("Bulk metadata parse failed")
        finally:
            _parse_progress["running"] = False
            _parse_lock.release()

    thread = threading.Thread(target=_run, daemon=True, name="bulk-metadata-parse")
    thread.start()


# ---------------------------------------------------------------------------
# Heuristic keep-score computation
# ---------------------------------------------------------------------------

def compute_keep_scores(folder_filter: str | None = None,
                        dupe_groups: list[dict] | None = None) -> list[dict]:
    """Compute heuristic keep scores for all output images.

    Returns list of dicts with keys:
        image_id, job_id, file_path, filename, file_size, output_folder,
        keep_score, wave, created_at, positive_prompt, seed

    Score formula (each signal 0.0-1.0, weighted):
      - age_score:          newer = higher
      - prompt_score:       longer prompts = higher
      - file_size_score:    very small files = lower
      - seed_dupe_score:    unique seed+prompt = higher
      - cluster_density:    smaller clusters = higher
      - folder_score:       finetune/compare folders = lower
      - dupe_member_score:  non-best dupe members = lower
      - llm_score:          from image_quality_scores if available

    Wave assignment:
      wave 1: keep_score < 0.55
      wave 2: 0.55 <= keep_score < 0.70
      wave 3: 0.70 <= keep_score < 0.82
      (>= 0.82: not shown in cleanup UI)
    """
    # Fetch all output images with metadata
    folder_clause = ""
    params = {}
    if folder_filter:
        folder_clause = "AND gs.output_folder = :folder"
        params["folder"] = folder_filter

    with get_db() as conn:
        result = conn.execute(
            text(f"""SELECT gi.id as image_id, gi.job_id, gi.file_path, gi.filename,
                    gi.file_size, gi.created_at as img_created,
                    gj.created_at, gj.source,
                    gs.positive_prompt, gs.seed, gs.output_folder
                FROM generated_images gi
                JOIN generation_jobs gj ON gi.job_id = gj.id
                LEFT JOIN generation_settings gs ON gs.job_id = gi.job_id
                WHERE gi.file_path IS NOT NULL
                {folder_clause}
                ORDER BY gi.id"""),
            params,
        )
        rows = [dict(row._mapping) for row in result.fetchall()]

    if not rows:
        return []

    # Get keep flags and LLM scores
    keep_flags = scoring_model.get_keep_flags()
    llm_scores = scoring_model.get_all_quality_scores()

    # Get cluster assignment counts for density signal
    cluster_sizes = _get_cluster_sizes()

    # Detect seed+prompt duplicates
    seed_prompt_counts = _count_seed_prompt_dupes()

    # Get near-duplicate group membership for scoring signal
    dupe_best_picks: set[int] = set()
    dupe_non_best: set[int] = set()
    if dupe_groups is None:
        try:
            dupe_groups = detect_near_duplicates(folder_filter=folder_filter)
        except Exception:
            logger.warning("Failed to compute dupe groups for scoring signal", exc_info=True)
            dupe_groups = []
    for g in dupe_groups:
        best = g.get("best_pick_id")
        for mid in g["image_ids"]:
            if mid == best:
                dupe_best_picks.add(mid)
            else:
                dupe_non_best.add(mid)

    # Compute age range for normalization
    now = datetime.now(timezone.utc)
    max_age_days = 365.0

    results = []
    for row in rows:
        image_id = row["image_id"]

        # Skip kept images
        if image_id in keep_flags:
            continue

        # --- Individual signal scores (0.0-1.0, higher = more keepable) ---

        # Age: newer images score higher
        age_score = 1.0
        created_at = row.get("created_at")
        if created_at:
            try:
                if isinstance(created_at, str):
                    dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                else:
                    dt = datetime.fromtimestamp(float(created_at), tz=timezone.utc)
                days_old = (now - dt).total_seconds() / 86400.0
                age_score = max(0.0, 1.0 - (days_old / max_age_days))
            except (ValueError, TypeError, OSError):
                age_score = 0.5

        # Prompt quality: longer prompts tend to be more intentional
        prompt = row.get("positive_prompt") or ""
        prompt_len = len(prompt.strip())
        if prompt_len == 0:
            prompt_score = 0.1
        elif prompt_len < 20:
            prompt_score = 0.3
        elif prompt_len < 100:
            prompt_score = 0.6
        else:
            prompt_score = min(1.0, 0.6 + (prompt_len - 100) / 500.0)

        # File size: very small files are likely corrupt/failed
        file_size = row.get("file_size") or 0
        if file_size < 10_000:  # < 10 KB
            file_size_score = 0.0
        elif file_size < 50_000:  # < 50 KB
            file_size_score = 0.3
        else:
            file_size_score = 1.0

        # Seed duplication: same seed + same prompt = identical output
        seed = row.get("seed")
        seed_key = (seed, prompt[:100]) if seed and seed != -1 and prompt else None
        if seed_key and seed_prompt_counts.get(seed_key, 0) > 1:
            seed_dupe_score = 0.0
        else:
            seed_dupe_score = 1.0

        # Cluster density: images in oversized clusters are more expendable
        cluster_density = _get_image_cluster_density(image_id, cluster_sizes)

        # Folder score: finetune/compare folders are lower value
        output_folder = row.get("output_folder") or ""
        folder_name = output_folder.split("/")[0] if output_folder else ""
        if re.match(r"finetune", folder_name, re.IGNORECASE):
            folder_score = 0.3
        elif re.match(r".*compare", folder_name, re.IGNORECASE):
            folder_score = 0.3
        else:
            folder_score = 1.0

        # Near-duplicate membership signal
        if image_id in dupe_non_best:
            dupe_member_score = 0.2   # expendable — not the best pick
        elif image_id in dupe_best_picks:
            dupe_member_score = 0.8   # the one to keep
        else:
            dupe_member_score = 0.5   # neutral — not in any dupe group

        # LLM quality score (if available)
        llm_data = llm_scores.get(image_id)
        has_llm = llm_data is not None

        if has_llm:
            llm_score = llm_data["overall"]
            # Weights when LLM score is available
            keep_score = (
                0.08 * age_score +
                0.06 * prompt_score +
                0.04 * file_size_score +
                0.08 * seed_dupe_score +
                0.06 * cluster_density +
                0.08 * folder_score +
                0.15 * dupe_member_score +
                0.45 * llm_score
            )
        else:
            # Weights when no LLM score — redistribute LLM weight
            keep_score = (
                0.15 * age_score +
                0.10 * prompt_score +
                0.05 * file_size_score +
                0.10 * seed_dupe_score +
                0.15 * cluster_density +
                0.25 * folder_score +
                0.20 * dupe_member_score
            )

        # Wave assignment
        if keep_score < 0.55:
            wave = 1
        elif keep_score < 0.70:
            wave = 2
        elif keep_score < 0.82:
            wave = 3
        else:
            continue  # Not shown in cleanup UI

        results.append({
            "image_id": image_id,
            "job_id": row["job_id"],
            "file_path": row["file_path"],
            "filename": row["filename"],
            "file_size": file_size,
            "output_folder": output_folder,
            "keep_score": round(keep_score, 4),
            "wave": wave,
            "created_at": row.get("created_at"),
            "positive_prompt": prompt[:200] if prompt else None,
            "seed": seed,
            "llm_overall": llm_data["overall"] if has_llm else None,
        })

    results.sort(key=lambda r: r["keep_score"])
    return results


def _get_cluster_sizes() -> dict[int, int]:
    """Get cluster_id → assignment_count mapping."""
    with get_db() as conn:
        result = conn.execute(
            text("""SELECT cluster_id, COUNT(*) as cnt
               FROM cluster_assignments
               GROUP BY cluster_id""")
        )
        return {row._mapping["cluster_id"]: row._mapping["cnt"]
                for row in result.fetchall()}


def _get_image_cluster_density(image_id: int, cluster_sizes: dict[int, int]) -> float:
    """Get cluster density score for an image (0.0 = very dense cluster, 1.0 = small).

    Uses the image's file_path as the doc_id lookup in cluster_assignments.
    """
    with get_db() as conn:
        # Look up by file_path (scanned) or gen_{job_id} (generated)
        result = conn.execute(
            text("""SELECT ca.cluster_id
               FROM cluster_assignments ca
               JOIN generated_images gi ON (ca.doc_id = gi.file_path OR ca.doc_id = 'gen_' || gi.job_id)
               WHERE gi.id = :image_id
               LIMIT 1"""),
            {"image_id": image_id},
        )
        row = result.fetchone()

    if not row:
        return 0.5  # No cluster assignment — neutral

    cluster_id = row._mapping["cluster_id"]
    size = cluster_sizes.get(cluster_id, 1)
    max_size = max(cluster_sizes.values()) if cluster_sizes else 1
    # Normalize: larger cluster → lower score
    return max(0.0, 1.0 - (size / max_size))


def _count_seed_prompt_dupes() -> dict[tuple, int]:
    """Count images sharing the same (seed, prompt_prefix) combination."""
    with get_db() as conn:
        result = conn.execute(
            text("""SELECT seed, SUBSTR(positive_prompt, 1, 100) as prompt_prefix, COUNT(*) as cnt
               FROM generation_settings
               WHERE seed IS NOT NULL AND seed != -1 AND positive_prompt != ''
               GROUP BY seed, prompt_prefix
               HAVING cnt > 1""")
        )
        return {(row._mapping["seed"], row._mapping["prompt_prefix"]): row._mapping["cnt"]
                for row in result.fetchall()}


# ---------------------------------------------------------------------------
# Near-duplicate detection
# ---------------------------------------------------------------------------

def _refine_dupe_groups(
    groups_map: dict[int, list[int]],
    embeddings_normed: np.ndarray,
    threshold: float,
) -> dict[int, list[int]]:
    """Break transitive chains in dupe groups.

    For groups with >3 members, verify that each member has sufficient
    average similarity to the rest of the group.  Members whose average
    falls below ``threshold - 0.02`` are removed.  Groups that shrink
    below 2 members are dropped entirely.
    """
    refined = {}
    for root, indices in groups_map.items():
        if len(indices) <= 3:
            refined[root] = indices
            continue

        idx_arr = np.array(indices)
        group_emb = embeddings_normed[idx_arr]
        pairwise = group_emb @ group_emb.T  # (G, G) similarities
        # Average similarity of each member to all other members
        n = len(indices)
        avg_sims = (pairwise.sum(axis=1) - 1.0) / max(n - 1, 1)  # exclude self

        relaxed = threshold - 0.02
        core = [idx for idx, avg in zip(indices, avg_sims) if avg >= relaxed]
        if len(core) >= 2:
            refined[root] = core

    return refined


def detect_near_duplicates(folder_filter: str | None = None,
                           threshold: float = 0.98) -> list[dict]:
    """Find near-duplicate image groups via ChromaDB cosine similarity.

    Algorithm:
      1. Get all output doc embeddings from ChromaDB
      2. For each doc, find neighbors with cosine similarity > threshold
      3. Build adjacency graph and find connected components (union-find)
      4. Map doc_ids back to image_ids
      5. For each group, compute heuristic "best pick"

    Returns list of groups:
      [{group_id, image_ids: [...], best_pick_id, folders: [...]}]
    """
    # Get all output embeddings
    collection = vector_store._generated_collection
    if collection is None or collection.count() == 0:
        return []

    count = collection.count()
    all_ids = []
    all_embeddings = []

    page_size = 5000
    offset = 0
    while offset < count:
        result = collection.get(
            limit=page_size,
            offset=offset,
            include=["embeddings"],
        )
        if not result["ids"]:
            break
        all_ids.extend(result["ids"])
        all_embeddings.extend(result["embeddings"])
        offset += len(result["ids"])

    if len(all_ids) < 2:
        return []

    # Convert to numpy for fast cosine similarity
    embeddings = np.array(all_embeddings, dtype=np.float32)
    # Normalize for cosine similarity via dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings_normed = embeddings / norms

    # Union-Find
    parent = list(range(len(all_ids)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Batch cosine similarity: process in chunks to avoid memory explosion
    chunk_size = 500
    # ChromaDB uses L2 distance by default; cosine similarity = 1 - distance/2
    # for normalized vectors. But since we have the raw embeddings, compute directly.
    for i in range(0, len(all_ids), chunk_size):
        chunk = embeddings_normed[i:i + chunk_size]
        # Compute similarities: chunk × all embeddings
        sims = chunk @ embeddings_normed.T  # shape: (chunk_size, N)
        for ci in range(len(chunk)):
            global_i = i + ci
            for j in range(global_i + 1, len(all_ids)):
                if sims[ci, j] > threshold:
                    union(global_i, j)

    # Collect connected components
    groups_map: dict[int, list[int]] = {}
    for idx in range(len(all_ids)):
        root = find(idx)
        if root not in groups_map:
            groups_map[root] = []
        groups_map[root].append(idx)

    # Filter to groups with 2+ members
    groups_map = {k: v for k, v in groups_map.items() if len(v) >= 2}

    if not groups_map:
        return []

    # Refine groups: break transitive chains where members don't all
    # meet minimum average similarity to the rest of the group
    groups_map = _refine_dupe_groups(groups_map, embeddings_normed, threshold)

    # Map doc_ids to image_ids
    doc_to_image = _map_doc_ids_to_image_ids(all_ids)

    # Build output groups
    llm_scores = scoring_model.get_all_quality_scores()
    result_groups = []
    group_id = 0

    for indices in groups_map.values():
        members = []
        folders = set()
        for idx in indices:
            doc_id = all_ids[idx]
            img_info = doc_to_image.get(doc_id)
            if img_info:
                members.append({
                    "image_id": img_info["image_id"],
                    "job_id": img_info.get("job_id"),
                    "file_path": img_info.get("file_path"),
                    "file_size": img_info.get("file_size"),
                })
                if img_info.get("output_folder"):
                    folders.add(img_info["output_folder"])

        if len(members) < 2:
            continue

        image_ids = [m["image_id"] for m in members]

        # Filter by folder if requested
        if folder_filter:
            filtered = [
                m for m in members
                if any(doc_to_image.get(all_ids[idx], {}).get("image_id") == m["image_id"]
                       and doc_to_image.get(all_ids[idx], {}).get("output_folder", "").startswith(folder_filter)
                       for idx in indices)
            ]
            if not filtered:
                continue

        # Determine best pick: prefer highest LLM score, else newest, else largest file
        best_pick = _pick_best(image_ids, llm_scores)

        result_groups.append({
            "group_id": group_id,
            "image_ids": image_ids,
            "members": members,
            "best_pick_id": best_pick,
            "folders": sorted(folders),
        })
        group_id += 1

    result_groups.sort(key=lambda g: len(g["image_ids"]), reverse=True)
    return result_groups


def _map_doc_ids_to_image_ids(doc_ids: list[str]) -> dict[str, dict]:
    """Map ChromaDB doc_ids to image information.

    Doc IDs can be:
      - 'gen_{job_id}' for chat-generated images
      - file_path for scanned images
    """
    mapping = {}

    # Split into gen_ prefixed and file paths
    gen_job_ids = []
    file_paths = []
    for doc_id in doc_ids:
        if doc_id.startswith("gen_"):
            gen_job_ids.append(doc_id[4:])  # strip 'gen_' prefix
        else:
            file_paths.append(doc_id)

    with get_db() as conn:
        # Batch lookup for gen_ doc_ids
        for i in range(0, len(gen_job_ids), 500):
            chunk = gen_job_ids[i:i + 500]
            placeholders = ",".join([f":p{j}" for j in range(len(chunk))])
            params = {f"p{j}": v for j, v in enumerate(chunk)}
            result = conn.execute(
                text(f"""SELECT gi.id as image_id, gi.job_id, gi.file_path,
                        gi.file_size, gs.output_folder, gj.created_at
                    FROM generated_images gi
                    JOIN generation_jobs gj ON gi.job_id = gj.id
                    LEFT JOIN generation_settings gs ON gs.job_id = gi.job_id
                    WHERE gi.job_id IN ({placeholders})"""),
                params,
            )
            for row in result.fetchall():
                r = row._mapping
                mapping[f"gen_{r['job_id']}"] = {
                    "image_id": r["image_id"],
                    "job_id": r["job_id"],
                    "file_path": r["file_path"],
                    "file_size": r["file_size"],
                    "output_folder": r["output_folder"],
                    "created_at": r["created_at"],
                }

        # Batch lookup for file_path doc_ids
        for i in range(0, len(file_paths), 500):
            chunk = file_paths[i:i + 500]
            placeholders = ",".join([f":p{j}" for j in range(len(chunk))])
            params = {f"p{j}": v for j, v in enumerate(chunk)}
            result = conn.execute(
                text(f"""SELECT gi.id as image_id, gi.job_id, gi.file_path,
                        gi.file_size, gs.output_folder, gj.created_at
                    FROM generated_images gi
                    JOIN generation_jobs gj ON gi.job_id = gj.id
                    LEFT JOIN generation_settings gs ON gs.job_id = gi.job_id
                    WHERE gi.file_path IN ({placeholders})"""),
                params,
            )
            for row in result.fetchall():
                r = row._mapping
                mapping[r["file_path"]] = {
                    "image_id": r["image_id"],
                    "job_id": r["job_id"],
                    "file_path": r["file_path"],
                    "file_size": r["file_size"],
                    "output_folder": r["output_folder"],
                    "created_at": r["created_at"],
                }

    return mapping


def _pick_best(image_ids: list[int], llm_scores: dict[int, dict]) -> int:
    """Pick the best image from a near-duplicate group.

    Priority: highest LLM overall score > newest > largest file size.
    """
    best_id = image_ids[0]
    best_llm = -1.0
    best_created = ""
    best_size = 0

    with get_db() as conn:
        placeholders = ",".join([f":p{i}" for i in range(len(image_ids))])
        params = {f"p{i}": v for i, v in enumerate(image_ids)}
        result = conn.execute(
            text(f"""SELECT gi.id, gi.file_size, gj.created_at
                FROM generated_images gi
                JOIN generation_jobs gj ON gi.job_id = gj.id
                WHERE gi.id IN ({placeholders})"""),
            params,
        )
        for row in result.fetchall():
            r = row._mapping
            iid = r["id"]
            llm = llm_scores.get(iid, {}).get("overall", -1.0)
            created = str(r["created_at"] or "")
            size = r["file_size"] or 0

            if (llm > best_llm or
                (llm == best_llm and created > best_created) or
                (llm == best_llm and created == best_created and size > best_size)):
                best_id = iid
                best_llm = llm
                best_created = created
                best_size = size

    return best_id


# ---------------------------------------------------------------------------
# Folder summary
# ---------------------------------------------------------------------------

def get_folder_summary() -> list[dict]:
    """Get all output folders with image count, disk size, and cleanup score.

    Returns list of dicts:
      [{folder, image_count, disk_size_bytes, cleanup_pct}]
    sorted by cleanup_pct descending (most deletable first).
    """
    with get_db() as conn:
        result = conn.execute(
            text("""SELECT gs.output_folder,
                    COUNT(*) as image_count,
                    COALESCE(SUM(gi.file_size), 0) as disk_size_bytes
                FROM generated_images gi
                JOIN generation_jobs gj ON gi.job_id = gj.id
                LEFT JOIN generation_settings gs ON gs.job_id = gi.job_id
                WHERE gi.file_path IS NOT NULL
                GROUP BY gs.output_folder
                ORDER BY gs.output_folder""")
        )
        folders = [dict(row._mapping) for row in result.fetchall()]

    # Compute cleanup percentage: what fraction of images would be in waves 1-3
    # This is an approximation using folder-level heuristics (fast, no per-image scoring)
    for folder in folders:
        name = (folder.get("output_folder") or "").split("/")[0]
        if re.match(r"finetune", name, re.IGNORECASE) or re.match(r".*compare", name, re.IGNORECASE):
            folder["cleanup_pct"] = 75  # finetune folders: high cleanup potential
        else:
            folder["cleanup_pct"] = 20  # character folders: low cleanup potential

    folders.sort(key=lambda f: f["cleanup_pct"], reverse=True)
    return folders


# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------

def delete_images(image_ids: list[int]) -> dict:
    """Delete images from disk, DB, and ChromaDB.

    Returns {deleted_count, freed_bytes, errors: [...]}.
    """
    if not image_ids:
        return {"deleted_count": 0, "freed_bytes": 0, "errors": []}

    deleted = 0
    freed = 0
    errors = []

    # Get file paths and job IDs
    with get_db() as conn:
        placeholders = ",".join([f":p{i}" for i in range(len(image_ids))])
        params = {f"p{i}": v for i, v in enumerate(image_ids)}
        result = conn.execute(
            text(f"""SELECT gi.id, gi.job_id, gi.file_path, gi.file_size
                FROM generated_images gi
                WHERE gi.id IN ({placeholders})"""),
            params,
        )
        images = [dict(row._mapping) for row in result.fetchall()]

    for img in images:
        file_path = img.get("file_path")
        try:
            # Delete from disk
            if file_path and os.path.isfile(file_path):
                freed += img.get("file_size") or 0
                os.remove(file_path)

            # Delete from ChromaDB
            if file_path:
                vector_store.delete_document(file_path, "output")
            # Also try gen_ prefix doc ID
            vector_store.delete_document(f"gen_{img['job_id']}", "output")

            # Delete from DB
            with get_db() as conn:
                conn.execute(
                    text("DELETE FROM generated_images WHERE id = :id"),
                    {"id": img["id"]},
                )
                # Clean up quality scores
                conn.execute(
                    text("DELETE FROM image_quality_scores WHERE image_id = :id"),
                    {"id": img["id"]},
                )
                # Clean up keep flags
                conn.execute(
                    text("DELETE FROM image_keep_flags WHERE image_id = :id"),
                    {"id": img["id"]},
                )
                # Clean up orphan jobs (no images remain)
                remaining = conn.execute(
                    text("SELECT COUNT(*) as cnt FROM generated_images WHERE job_id = :jid"),
                    {"jid": img["job_id"]},
                ).fetchone()._mapping["cnt"]
                if remaining == 0:
                    conn.execute(
                        text("DELETE FROM generation_settings WHERE job_id = :jid"),
                        {"jid": img["job_id"]},
                    )
                    conn.execute(
                        text("DELETE FROM generation_jobs WHERE id = :jid"),
                        {"jid": img["job_id"]},
                    )

            deleted += 1

        except Exception as e:
            errors.append({"image_id": img["id"], "error": str(e)})
            logger.error("Failed to delete image %s: %s", img["id"], e)

    # Clean up empty directories
    if images:
        _cleanup_empty_dirs(images)

    logger.info("Deleted %d images, freed %d bytes, %d errors",
                deleted, freed, len(errors))
    return {"deleted_count": deleted, "freed_bytes": freed, "errors": errors}


def _cleanup_empty_dirs(deleted_images: list[dict]):
    """Remove directories left empty after deletion."""
    from src.models.browser import get_root_directories

    roots = {os.path.normpath(r["path"]) for r in get_root_directories()}
    dirs_to_check = set()
    for img in deleted_images:
        if img.get("file_path"):
            dirs_to_check.add(os.path.dirname(img["file_path"]))

    for dir_path in dirs_to_check:
        current = os.path.normpath(dir_path)
        while current not in roots:
            try:
                if os.path.isdir(current) and not os.listdir(current):
                    os.rmdir(current)
                    logger.info("Removed empty directory: %s", current)
                    current = os.path.dirname(current)
                else:
                    break
            except OSError:
                break
