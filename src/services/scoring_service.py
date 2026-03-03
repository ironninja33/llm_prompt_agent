"""LLM scoring service — compress, upload, batch-score images via Gemini."""

import io
import json
import logging
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.services import image_compression
from src.services.llm_service import get_client
from src.models import scoring as scoring_model
from src.models.database import get_db
from sqlalchemy import text

logger = logging.getLogger(__name__)

# Progress tracking
_scoring_lock = threading.Lock()
_scoring_progress = {
    "phase": "idle",        # idle | uploading | submitted | processing | completed | failed
    "uploaded": 0,
    "total": 0,
    "batch_db_id": None,
}

SCORING_RUBRIC = """Score this image on 6 dimensions using a 0.0-1.0 scale.

Anchor definitions:
  0.0 = completely wrong / broken
  0.25 = major issues
  0.5 = acceptable
  0.75 = good
  1.0 = excellent / flawless

Dimensions:
  character: Face/body accuracy, proportions, anatomy correctness
  composition: Framing, visual balance, lighting, color harmony
  artifacts: Generation quality (1.0 = no artifacts, 0.0 = severe distortion)
  theme: Coherence with apparent prompt intent, style consistency
  detail: Texture quality, sharpness, background quality
  expression: Facial expression naturalness, pose quality

Return JSON only, no other text:
{"overall": <float>, "character": <float>, "composition": <float>, "artifacts": <float>, "theme": <float>, "detail": <float>, "expression": <float>, "notes": "<brief observation>"}

The overall score should be a weighted composite:
  overall = 0.25*character + 0.20*composition + 0.25*artifacts + 0.10*theme + 0.10*detail + 0.10*expression
"""


def get_scoring_progress() -> dict:
    """Return current scoring progress."""
    return dict(_scoring_progress)


def submit_scoring_batch(image_ids: list[int],
                         model: str = "gemini-2.5-flash-preview",
                         progress_callback=None) -> int | None:
    """Compress, upload, and submit a Gemini batch for quality scoring.

    Returns the batch DB id, or None if submission fails.
    Runs in the calling thread (caller should spawn a background thread).
    """
    client = get_client()
    if not client:
        logger.error("Cannot submit scoring batch: Gemini client not initialized")
        return None

    if not _scoring_lock.acquire(blocking=False):
        logger.info("Scoring batch already in progress")
        return None

    try:
        # Get file paths for all images
        with get_db() as conn:
            placeholders = ",".join([f":p{i}" for i in range(len(image_ids))])
            params = {f"p{i}": v for i, v in enumerate(image_ids)}
            result = conn.execute(
                text(f"""SELECT id, file_path FROM generated_images
                   WHERE id IN ({placeholders}) AND file_path IS NOT NULL"""),
                params,
            )
            image_files = [(row._mapping["id"], row._mapping["file_path"])
                           for row in result.fetchall()]

        if not image_files:
            logger.warning("No valid image files for scoring batch")
            return None

        # Create batch record
        import uuid
        batch_id = f"score_{uuid.uuid4().hex[:8]}"
        db_id = scoring_model.create_scoring_batch(batch_id, len(image_files))
        scoring_model.add_batch_items(db_id, [img_id for img_id, _ in image_files])

        _scoring_progress["phase"] = "uploading"
        _scoring_progress["uploaded"] = 0
        _scoring_progress["total"] = len(image_files)
        _scoring_progress["batch_db_id"] = db_id

        # Concurrent compress + upload
        uri_map = {}  # image_id -> gemini file name

        def _compress_and_upload(img_id, file_path):
            jpeg_bytes = image_compression.compress_for_llm(file_path)
            if not jpeg_bytes:
                return (img_id, None)

            # Upload to Gemini Files API
            try:
                from google.genai import types
                buf = io.BytesIO(jpeg_bytes)
                buf.name = f"img_{img_id}.jpg"
                uploaded = client.files.upload(
                    file=buf,
                    config=types.UploadFileConfig(
                        display_name=f"score_img_{img_id}",
                        mime_type="image/jpeg",
                    ),
                )
                return (img_id, uploaded.name)
            except Exception as e:
                logger.error("Failed to upload image %s: %s", img_id, e)
                return (img_id, None)

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {
                pool.submit(_compress_and_upload, img_id, fp): img_id
                for img_id, fp in image_files
            }
            for future in as_completed(futures):
                img_id, uri = future.result()
                if uri:
                    uri_map[img_id] = uri
                _scoring_progress["uploaded"] += 1
                if progress_callback:
                    progress_callback(_scoring_progress["uploaded"], len(image_files))

        if not uri_map:
            logger.error("No images uploaded successfully for scoring batch")
            scoring_model.update_batch_status(db_id, "failed")
            _scoring_progress["phase"] = "failed"
            return db_id

        # Build JSONL batch request
        requests = []
        for img_id, file_name in uri_map.items():
            requests.append({
                "key": str(img_id),
                "request": {
                    "contents": [{
                        "parts": [
                            {"file_data": {"file_uri": file_name, "mime_type": "image/jpeg"}},
                            {"text": SCORING_RUBRIC},
                        ]
                    }],
                    "generation_config": {
                        "temperature": 0.2,
                        "response_mime_type": "application/json",
                    },
                },
            })

        # Write JSONL to temp file and upload
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for req in requests:
                f.write(json.dumps(req) + "\n")
            jsonl_path = f.name

        try:
            from google.genai import types
            uploaded_jsonl = client.files.upload(
                file=jsonl_path,
                config=types.UploadFileConfig(
                    display_name=f"batch_{batch_id}",
                    mime_type="jsonl",
                ),
            )
        finally:
            os.unlink(jsonl_path)

        # Submit batch job
        _scoring_progress["phase"] = "submitted"
        batch_job = client.batches.create(
            model=model,
            src=uploaded_jsonl.name,
            config={"display_name": f"score_{batch_id}"},
        )

        # Store the Gemini batch job name
        scoring_model.update_batch_status(db_id, "submitted")

        # Store job name and uri_map for later retrieval
        _store_batch_metadata(db_id, batch_job.name, uri_map)

        logger.info("Scoring batch submitted: db_id=%d, gemini_job=%s, images=%d",
                     db_id, batch_job.name, len(uri_map))

        _scoring_progress["phase"] = "processing"
        return db_id

    except Exception as e:
        logger.exception("Failed to submit scoring batch: %s", e)
        _scoring_progress["phase"] = "failed"
        if 'db_id' in dir():
            scoring_model.update_batch_status(db_id, "failed")
        return None
    finally:
        _scoring_lock.release()


def poll_batch_status(batch_db_id: int) -> dict:
    """Check Gemini API for batch completion and process results.

    Returns {status, scored_count, total_images}.
    """
    client = get_client()
    if not client:
        return {"status": "error", "message": "Gemini client not initialized"}

    batch = scoring_model.get_batch_by_id(batch_db_id)
    if not batch:
        return {"status": "error", "message": "Batch not found"}

    # Already terminal
    if batch["status"] in ("completed", "failed"):
        return {
            "status": batch["status"],
            "scored_count": batch["scored_count"],
            "total_images": batch["total_images"],
        }

    # Get stored Gemini job name
    meta = _get_batch_metadata(batch_db_id)
    if not meta or not meta.get("gemini_job_name"):
        return {"status": "error", "message": "Batch metadata missing"}

    try:
        batch_job = client.batches.get(name=meta["gemini_job_name"])
        state = batch_job.state.name if batch_job.state else "UNKNOWN"

        if state == "JOB_STATE_SUCCEEDED":
            # Retrieve results
            scored = _process_batch_results(batch_db_id, batch_job, meta)
            scoring_model.update_batch_status(batch_db_id, "completed", scored_count=scored)
            _scoring_progress["phase"] = "completed"

            # Background cleanup of uploaded files
            uri_map = meta.get("uri_map", {})
            if uri_map:
                thread = threading.Thread(
                    target=_delete_remote_files, args=(uri_map,),
                    daemon=True, name="cleanup-remote-files"
                )
                thread.start()

            return {"status": "completed", "scored_count": scored,
                    "total_images": batch["total_images"]}

        elif state in ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"):
            scoring_model.update_batch_status(batch_db_id, "failed")
            _scoring_progress["phase"] = "failed"
            return {"status": "failed", "scored_count": 0,
                    "total_images": batch["total_images"]}

        else:
            # Still processing
            scoring_model.update_batch_status(batch_db_id, "processing")
            _scoring_progress["phase"] = "processing"
            return {"status": "processing", "scored_count": 0,
                    "total_images": batch["total_images"]}

    except Exception as e:
        logger.error("Failed to poll batch status: %s", e)
        return {"status": "error", "message": str(e)}


def _process_batch_results(batch_db_id: int, batch_job, meta: dict) -> int:
    """Parse batch results and upsert quality scores. Returns count scored."""
    client = get_client()
    scored = 0

    try:
        result_file_name = batch_job.dest.file_name
        if not result_file_name:
            logger.error("Batch job has no result file")
            return 0

        file_content = client.files.download(file=result_file_name)
        content_str = file_content.decode('utf-8') if isinstance(file_content, bytes) else str(file_content)

        for line in content_str.strip().split('\n'):
            if not line.strip():
                continue
            try:
                result = json.loads(line)
                key = result.get("key", "")
                image_id = int(key) if key.isdigit() else None
                if image_id is None:
                    continue

                response = result.get("response", {})
                # Extract text from candidates
                candidates = response.get("candidates", [])
                if not candidates:
                    continue
                parts = candidates[0].get("content", {}).get("parts", [])
                text_content = ""
                for part in parts:
                    if "text" in part:
                        text_content += part["text"]

                if not text_content:
                    continue

                scores = json.loads(text_content)
                scoring_model.upsert_quality_score(
                    image_id=image_id,
                    overall=_clamp(scores.get("overall", 0.5)),
                    character=_clamp(scores.get("character", 0.5)),
                    composition=_clamp(scores.get("composition", 0.5)),
                    artifacts=_clamp(scores.get("artifacts", 0.5)),
                    theme=_clamp(scores.get("theme", 0.5)),
                    detail=_clamp(scores.get("detail", 0.5)),
                    expression=_clamp(scores.get("expression", 0.5)),
                    notes=scores.get("notes"),
                    model_used=meta.get("model", "gemini"),
                )
                scored += 1
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.warning("Failed to parse batch result line: %s", e)
                continue

    except Exception as e:
        logger.exception("Failed to process batch results: %s", e)

    return scored


def _delete_remote_files(uri_map: dict):
    """Background cleanup: delete uploaded files from Gemini Files API."""
    client = get_client()
    if not client:
        return

    for img_id, file_name in uri_map.items():
        try:
            client.files.delete(name=file_name)
        except Exception:
            logger.debug("Failed to delete remote file: %s", file_name)


def _clamp(value, min_val=0.0, max_val=1.0) -> float:
    """Clamp a score to [min_val, max_val]."""
    try:
        return max(min_val, min(max_val, float(value)))
    except (TypeError, ValueError):
        return 0.5


# ---------------------------------------------------------------------------
# Batch metadata storage (simple key-value in a JSON file)
# ---------------------------------------------------------------------------

_BATCH_META_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'batch_meta')


def _store_batch_metadata(batch_db_id: int, gemini_job_name: str, uri_map: dict):
    """Persist batch metadata for later retrieval."""
    os.makedirs(_BATCH_META_DIR, exist_ok=True)
    path = os.path.join(_BATCH_META_DIR, f"batch_{batch_db_id}.json")
    with open(path, 'w') as f:
        json.dump({
            "gemini_job_name": gemini_job_name,
            "uri_map": {str(k): v for k, v in uri_map.items()},
        }, f)


def _get_batch_metadata(batch_db_id: int) -> dict | None:
    """Retrieve persisted batch metadata."""
    path = os.path.join(_BATCH_META_DIR, f"batch_{batch_db_id}.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
