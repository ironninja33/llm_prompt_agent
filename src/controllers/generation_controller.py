"""Generation controller — business logic for ComfyUI image generation."""

import logging
import os
import random
import threading
import time
from src.models import generation as gen_model
from src.models.settings import get_setting
from src.services import comfyui_service

logger = logging.getLogger(__name__)


def submit_generation(chat_id: str | None, message_id: int | None, settings: dict,
                      source: str = "chat", session_id: str | None = None,
                      parent_job_id: str | None = None):
    """Submit an image generation job.

    1. Create job record in database
    2. Load and prepare the workflow via workflow_controller
    3. Submit to ComfyUI
    4. Start polling for progress
    5. Return the job dict

    settings dict should contain:
    - positive_prompt (required)
    - negative_prompt (optional, defaults to setting)
    - base_model (optional, defaults to setting)
    - loras (optional, list of {"name": str, "strength": float})
    - output_folder (optional)
    - seed (optional, default -1)
    - num_images (optional, default 1)
    """
    from src.controllers import workflow_controller
    from src.models import metrics

    # Get defaults from settings for missing values
    if not settings.get("negative_prompt"):
        settings["negative_prompt"] = get_setting("comfyui_default_negative") or ""
    if not settings.get("base_model"):
        settings["base_model"] = get_setting("comfyui_default_model") or ""

    # Apply default sampler settings for fields not present in the submission.
    # The overlay UI doesn't expose these yet, so they always come from defaults.
    if settings.get("sampler") is None:
        settings["sampler"] = get_setting("comfyui_default_sampler") or None
    if settings.get("cfg_scale") is None:
        default_cfg = get_setting("comfyui_default_cfg")
        settings["cfg_scale"] = float(default_cfg) if default_cfg else None
    if settings.get("scheduler") is None:
        settings["scheduler"] = get_setting("comfyui_default_scheduler") or None
    if settings.get("steps") is None:
        default_steps = get_setting("comfyui_default_steps")
        settings["steps"] = int(default_steps) if default_steps else None

    # Resolve seed: if -1, generate a random seed now so the actual value
    # is stored in the database alongside the generated images.
    seed_value = settings.get("seed", -1)
    if seed_value is None or seed_value == -1:
        seed_value = random.randint(0, 1125899906842624)
        settings["seed"] = seed_value
        logger.info("Resolved seed -1 → %d", seed_value)

    # Load and prepare workflow with user settings applied
    prepared_api, prepared_ui = workflow_controller.prepare_for_generation(settings)

    # Determine source: if no chat_id, it's a browser generation
    if chat_id is None:
        source = "browser"

    # Session tracking
    resolved_session = metrics.resolve_session(session_id)

    # Lineage computation
    lineage_depth = 0
    if parent_job_id:
        parent = gen_model.get_job(parent_job_id)
        if parent:
            lineage_depth = (parent.get("lineage_depth") or 0) + 1

    # Create job in database
    job = gen_model.create_job(
        chat_id, message_id, settings, source=source,
        session_id=resolved_session,
        parent_job_id=parent_job_id,
        lineage_depth=lineage_depth,
    )
    job_id = job["id"]

    try:
        num_images = settings.get("num_images", 1) or 1

        # Submit to ComfyUI
        result = comfyui_service.submit_prompt(
            prepared_api,
            client_id=comfyui_service.COMFYUI_CLIENT_ID,
            ui_workflow=prepared_ui,
        )

        if not result.success:
            gen_model.update_job_status(job_id, "failed")
            job["status"] = "failed"
            job["error"] = result.error
            return job

        # Update job with prompt_id
        gen_model.update_job_status(job_id, "queued", prompt_id=result.prompt_id)
        job["status"] = "queued"
        job["prompt_id"] = result.prompt_id

        # Start polling in background
        comfyui_service.poll_job(job_id, result.prompt_id, total_images=num_images)

        return job

    except Exception as e:
        logger.error(f"Failed to submit generation job {job_id}: {e}", exc_info=True)
        gen_model.update_job_status(job_id, "failed")
        job["status"] = "failed"
        job["error"] = str(e)
        return job


def submit_grid_generation(chat_id: str | None, message_id: int | None,
                           grid_settings: dict, source: str = "chat",
                           session_id: str | None = None,
                           parent_job_id: str | None = None) -> dict:
    """Submit multiple generation jobs for a parameter grid.

    Computes the Cartesian product of all parameter combinations,
    generates N random seeds (or reuses provided ones), and calls
    submit_generation() for each seed × combination.

    Returns dict with 'jobs', 'total_combinations', 'total_images',
    and 'seeds_used'.
    """
    combinations = _generate_parameter_combinations(grid_settings)

    # Use provided seeds or generate new ones
    provided_seeds = grid_settings.get("seeds")
    if provided_seeds and isinstance(provided_seeds, list):
        seeds = [int(s) for s in provided_seeds]
    else:
        num_seeds = grid_settings.get("num_seeds", 1)
        seeds = [random.randint(0, 1125899906842624) for _ in range(num_seeds)]

    jobs = []
    for seed in seeds:
        for combo in combinations:
            settings = {
                "positive_prompt": grid_settings.get("positive_prompt", ""),
                "negative_prompt": grid_settings.get("negative_prompt", ""),
                "output_folder": grid_settings.get("output_folder", ""),
                "base_model": combo["base_model"],
                "loras": combo["loras"],
                "sampler": combo["sampler"],
                "scheduler": combo["scheduler"],
                "cfg_scale": combo["cfg_scale"],
                "steps": combo["steps"],
                "seed": seed,
                "num_images": 1,
            }
            job = submit_generation(
                chat_id, message_id, settings, source,
                session_id, parent_job_id,
            )
            jobs.append(job)

    return {
        "jobs": jobs,
        "total_combinations": len(combinations),
        "total_images": len(seeds) * len(combinations),
        "seeds_used": seeds,
    }


def _generate_parameter_combinations(grid_settings: dict) -> list[dict]:
    """Generate Cartesian product of all parameter values.

    Each multi-select (models, loras, samplers, schedulers) contributes
    ONE value per combination.  LoRAs are individual — each combination
    uses exactly one LoRA from the selection.
    """
    base_models = grid_settings.get("base_models", [])
    samplers = grid_settings.get("samplers", [])
    schedulers = grid_settings.get("schedulers", [])

    lora_selections = grid_settings.get("loras", [])
    lora_combos = [[lora] for lora in lora_selections] if lora_selections else [[]]

    cfg_values = _expand_float_range(
        grid_settings.get("cfg_range", {}).get("start"),
        grid_settings.get("cfg_range", {}).get("stop"),
        grid_settings.get("cfg_range", {}).get("step"),
    )
    steps_values = _expand_int_range(
        grid_settings.get("steps_range", {}).get("start"),
        grid_settings.get("steps_range", {}).get("stop"),
        grid_settings.get("steps_range", {}).get("step"),
    )

    combinations = []
    for model in base_models:
        for loras in lora_combos:
            for sampler in samplers:
                for scheduler in schedulers:
                    for cfg in cfg_values:
                        for steps in steps_values:
                            combinations.append({
                                "base_model": model,
                                "loras": loras,
                                "sampler": sampler,
                                "scheduler": scheduler,
                                "cfg_scale": cfg,
                                "steps": steps,
                            })
    return combinations


def _expand_float_range(start, stop, step):
    """Expand [start, stop] inclusive with increment. Step=0/None → [start]."""
    if start is None:
        return [None]
    if stop is None or step is None or step == 0:
        return [start]
    values = []
    current = float(start)
    stop_f = float(stop)
    step_f = float(step)
    while current <= stop_f + 1e-9:
        values.append(round(current, 4))
        current += step_f
    if values and abs(values[-1] - stop_f) > 1e-9 and stop_f > values[-1]:
        values.append(stop_f)
    return values or [start]


def _expand_int_range(start, stop, step):
    """Expand integer range [start, stop] with increment. Step=0/None → [start]."""
    if start is None:
        return [None]
    if stop is None or step is None or step == 0:
        return [int(start)]
    values = list(range(int(start), int(stop) + 1, int(step)))
    if values and values[-1] != int(stop):
        values.append(int(stop))
    return values or [int(start)]


def get_generation_status(job_id: str) -> dict | None:
    """Get current status of a generation job."""
    return gen_model.get_job(job_id)


def get_chat_generations(chat_id: str) -> list[dict]:
    """Get all generation data for a chat (for rebuilding UI on reload)."""
    return gen_model.get_generation_data_for_chat(chat_id)


def _handle_generation_status(progress):
    """Handle non-completion status updates (failures).

    Registered as a general status listener.
    """
    if progress.phase == "failed":
        gen_model.update_job_status(progress.job_id, "failed")


def handle_generation_complete(progress):
    """Handle generation completion — store images and trigger embedding.

    Registered as a completion callback so DB writes finish before
    SSE events reach the browser.  Uses register_image() which is shared
    with the mtime flow — on duplicate file_path (mtime already inserted),
    adopts the scan record and bails without re-embedding.
    """
    gen_model.update_job_status(progress.job_id, "completed")

    if not progress.output_images:
        return

    # Fast path: all images already under this job_id
    existing = gen_model.get_job_images(progress.job_id)
    if existing:
        logger.debug("Images already recorded for job %s, skipping", progress.job_id)
        return

    any_new = False
    settings = gen_model.get_job_settings(progress.job_id)
    prompt = settings.get("positive_prompt", "") if settings else ""
    # Use OUR output_folder setting (what we told ComfyUI to use) instead of
    # ComfyUI's reported subfolder, which can be wrong for custom save nodes.
    output_folder = settings.get("output_folder", "") if settings else ""

    for img in progress.output_images:
        filename = img.get("filename", "")
        file_path = comfyui_service.construct_output_path(filename, output_folder)
        if not file_path:
            logger.warning(
                "Could not construct file_path for %s/%s in job %s "
                "(no output directories configured?)",
                output_folder, filename, progress.job_id,
            )
            continue

        # Validate the file actually exists on disk — don't create phantom records
        if not os.path.isfile(file_path):
            logger.warning(
                "Output file not found at constructed path: %s (job %s)",
                file_path, progress.job_id,
            )
            continue

        result = register_image(
            file_path=file_path,
            filename=filename,
            subfolder=output_folder,
            job_id=progress.job_id,
        )

        if result == "inserted":
            any_new = True
        # "adopted" / "exists" → mtime already handled, skip embedding

    if any_new and prompt:
        # Embed only images we actually inserted
        images = gen_model.get_job_images(progress.job_id)
        for img in images:
            fp = img.get("file_path")
            if fp:
                embed_image(fp, prompt)


def _embed_generated_prompt(job_id: str):
    """Embed the generated prompt using file-path doc IDs.

    Uses the same normalized file path that ingestion uses as the ChromaDB
    doc ID, so ingestion will see these as "already indexed" and skip them.
    Runs in a background thread to avoid blocking the completion callback.
    """

    def _do_embed():
        try:
            settings = gen_model.get_job_settings(job_id)
            if not settings:
                logger.warning(f"No settings found for job {job_id}, skipping embedding")
                return

            prompt = settings.get("positive_prompt", "")
            if not prompt:
                logger.debug(f"No positive prompt for job {job_id}, skipping embedding")
                return

            from src.services import embedding_service
            from src.models import vector_store
            from src.services import clustering_service
            from src.models.settings import get_data_directories

            # Get file paths from the generated_images records
            images = gen_model.get_job_images(job_id)
            file_paths = [
                os.path.normpath(img["file_path"])
                for img in images
                if img.get("file_path")
            ]
            if not file_paths:
                logger.warning(f"No file paths for job {job_id}, skipping embedding")
                return

            # Skip any already embedded
            file_paths = [
                fp for fp in file_paths
                if not vector_store.document_exists(fp, "output")
            ]
            if not file_paths:
                logger.debug(f"All images for job {job_id} already embedded, skipping")
                return

            # Embed once — same prompt for all images in this job
            embedding = embedding_service.embed(prompt)
            if not embedding:
                logger.warning(f"Empty embedding returned for job {job_id}")
                return

            # Build metadata matching ingestion_service format
            output_dirs = [
                d for d in get_data_directories(active_only=True)
                if d.get("dir_type") == "output"
            ]

            doc_ids = []
            concepts = []
            for fp in file_paths:
                base_dir, concept, source_file = _resolve_ingestion_metadata(
                    fp, output_dirs, settings,
                )
                metadata = {
                    "concept": concept,
                    "base_dir": base_dir,
                    "source_file": source_file,
                    "dir_type": "output",
                    "base_model": settings.get("base_model", "") or "",
                    "loras": settings.get("loras", []) or [],
                }
                vector_store.add_document(
                    doc_id=fp,
                    text=prompt,
                    embedding=embedding,
                    source_type="output",
                    metadata=metadata,
                )
                doc_ids.append(fp)
                concepts.append(concept)

            logger.info(f"Embedded {len(doc_ids)} image(s) for job {job_id}")

            # Assign to nearest existing clusters
            clustering_service.assign_new_docs_to_clusters(
                doc_ids=doc_ids,
                embeddings=[embedding] * len(doc_ids),
                source_types=["output"] * len(doc_ids),
                concepts=concepts,
            )

        except Exception as e:
            logger.error(
                f"Failed to embed generated prompt for job {job_id}: {e}",
                exc_info=True,
            )

    thread = threading.Thread(target=_do_embed, daemon=True)
    thread.start()


def _resolve_ingestion_metadata(
    file_path: str,
    output_dirs: list[dict],
    settings: dict,
) -> tuple[str, str, str]:
    """Derive base_dir/concept/source_file the same way ingestion does."""
    for d in output_dirs:
        dir_path = d["path"]
        if file_path.startswith(os.path.normpath(dir_path) + os.sep):
            rel_path = os.path.relpath(file_path, dir_path)
            parts = rel_path.split(os.sep)
            concept = parts[0] if len(parts) > 1 else os.path.basename(dir_path)
            base_dir = os.path.basename(dir_path)
            source_file = os.path.basename(file_path)
            return base_dir, concept, source_file
    # Fallback if file not under any known output dir
    output_folder = settings.get("output_folder", "") or ""
    concept = output_folder.split("/")[0] if output_folder else "generated"
    return "generated", concept, os.path.basename(file_path)



# ---------------------------------------------------------------------------
# Shared image registration (used by both mtime and comfy done flows)
# ---------------------------------------------------------------------------

def register_image(file_path, filename, subfolder="", job_id=None, file_size=None):
    """Register a generated image. Single entry point for both flows.

    Try INSERT.  On duplicate-key (IntegrityError from unique index on
    file_path), the handler checks which flow is calling:
      - mtime  (job_id=None):  comfy already inserted → bail.
      - comfy  (job_id=real):  mtime created scan record → UPDATE job_id → bail.

    Returns:
        "inserted" — new record. Caller should do embedding.
        "adopted"  — comfy claimed mtime's scan record. Caller should bail.
        "exists"   — mtime found comfy's record. Caller should bail.
    """
    actual_job_id = job_id if job_id else _create_scan_job()

    # Scan records (mtime) need file-based metadata parsing; comfy done
    # records already have generation_settings from job submission.
    metadata_status = "pending" if job_id is None else "complete"

    try:
        gen_model.add_generated_image_strict(
            actual_job_id, filename, subfolder,
            file_path=file_path, file_size=file_size,
            metadata_status=metadata_status,
        )
        logger.info("Registered image %s under job %s", file_path, actual_job_id)
        return "inserted"
    except Exception as exc:
        # Check for IntegrityError (sqlite3 or SQLAlchemy wrapper)
        if "UNIQUE constraint failed" in str(exc) or "IntegrityError" in type(exc).__name__:
            if job_id is None:
                # mtime flow — comfy already inserted this file. bail.
                gen_model.delete_job(actual_job_id)
                logger.info("Image already registered (comfy won): %s", file_path)
                return "exists"
            else:
                # comfy flow — mtime created a scan record. claim it.
                gen_model.update_image_job(file_path, job_id, subfolder)
                _cleanup_orphan_scan_jobs()
                logger.info("Adopted scan record for %s → job %s", file_path, job_id)
                return "adopted"
        raise  # Re-raise unexpected errors


def _create_scan_job():
    """Create a minimal scan generation_jobs record. Returns the job ID."""
    import uuid
    from src.models.database import get_db
    from sqlalchemy import text

    job_id = str(uuid.uuid4())
    with get_db() as conn:
        conn.execute(
            text("""INSERT INTO generation_jobs
                   (id, chat_id, message_id, prompt_id, status, source, created_at, completed_at)
                   VALUES (:id, NULL, NULL, NULL, 'completed', 'scan',
                           CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"""),
            {"id": job_id},
        )
    return job_id


def _cleanup_orphan_scan_jobs():
    """Delete scan jobs that have no remaining images."""
    from src.models.database import get_db
    from sqlalchemy import text

    with get_db() as conn:
        conn.execute(
            text("""DELETE FROM generation_jobs
                   WHERE source = 'scan'
                   AND id NOT IN (SELECT DISTINCT job_id FROM generated_images)"""),
        )


def embed_image(file_path, prompt):
    """Embed an image into ChromaDB. Takes file_path and prompt only.

    Derives metadata (concept, base_dir) from file_path internally.
    Checks document_exists() first — idempotent safety net.
    Runs in a background thread to avoid blocking the caller.
    """
    def _do_embed():
        try:
            from src.services import embedding_service
            from src.models import vector_store
            from src.services import clustering_service
            from src.models.settings import get_data_directories

            norm_path = os.path.normpath(file_path)

            if vector_store.document_exists(norm_path, "output"):
                logger.debug("Already embedded, skipping: %s", norm_path)
                return

            if not prompt:
                logger.debug("No prompt for embedding, skipping: %s", norm_path)
                return

            embedding = embedding_service.embed(prompt)
            if not embedding:
                logger.warning("Empty embedding returned for %s", norm_path)
                return

            # Derive metadata from file path
            output_dirs = [
                d for d in get_data_directories(active_only=True)
                if d.get("dir_type") == "output"
            ]
            base_dir = "generated"
            concept = "generated"
            source_file = os.path.basename(norm_path)

            for d in output_dirs:
                dir_path = d["path"]
                if norm_path.startswith(os.path.normpath(dir_path) + os.sep):
                    rel_path = os.path.relpath(norm_path, dir_path)
                    parts = rel_path.split(os.sep)
                    concept = parts[0] if len(parts) > 1 else os.path.basename(dir_path)
                    base_dir = os.path.basename(dir_path)
                    break

            metadata = {
                "concept": concept,
                "base_dir": base_dir,
                "source_file": source_file,
                "dir_type": "output",
            }
            vector_store.add_document(
                doc_id=norm_path,
                text=prompt,
                embedding=embedding,
                source_type="output",
                metadata=metadata,
            )
            logger.info("Embedded image: %s", norm_path)

            # Assign to nearest existing cluster
            clustering_service.assign_new_docs_to_clusters(
                doc_ids=[norm_path],
                embeddings=[embedding],
                source_types=["output"],
                concepts=[concept],
            )

        except Exception as e:
            logger.error("Failed to embed image %s: %s", file_path, e, exc_info=True)

    thread = threading.Thread(target=_do_embed, daemon=True)
    thread.start()


def trigger_embedding(job_id):
    """Public wrapper — trigger embedding for all images in a job.

    Reads prompt from generation_settings, then calls embed_image()
    for each image. Used by browser_controller after parse_pending_for_page.
    """
    settings = gen_model.get_job_settings(job_id)
    if not settings:
        return
    prompt = settings.get("positive_prompt", "")
    if not prompt:
        return

    images = gen_model.get_job_images(job_id)
    for img in images:
        fp = img.get("file_path")
        if fp:
            embed_image(fp, prompt)


def initialize():
    """Register the generation completion listener.

    Called during app startup.
    """
    comfyui_service.add_completion_callback(handle_generation_complete)
    comfyui_service.add_status_listener(_handle_generation_status)
    comfyui_service.start_poller()

    # Resume polling for any active jobs from a previous session
    active_jobs = gen_model.get_active_jobs()
    for job in active_jobs:
        if job.get("prompt_id"):
            logger.info(
                f"Resuming poll for job {job['id']} "
                f"(prompt_id={job['prompt_id']})"
            )
            comfyui_service.poll_job(
                job["id"],
                job["prompt_id"],
                total_images=job.get("num_images", 1) or 1,
            )
