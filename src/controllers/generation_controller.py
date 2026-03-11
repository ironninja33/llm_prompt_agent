"""Generation controller — business logic for ComfyUI image generation."""

import logging
import random
import threading
from src.models import generation as gen_model
from src.models.settings import get_setting
from src.services import comfyui_service

logger = logging.getLogger(__name__)


def submit_generation(chat_id: str | None, message_id: int | None, settings: dict,
                      source: str = "chat"):
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

    # Create job in database
    job = gen_model.create_job(chat_id, message_id, settings, source=source)
    job_id = job["id"]

    try:
        num_images = settings.get("num_images", 1) or 1

        # Submit to ComfyUI
        result = comfyui_service.submit_prompt(prepared_api, ui_workflow=prepared_ui)

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


def get_generation_status(job_id: str) -> dict | None:
    """Get current status of a generation job."""
    return gen_model.get_job(job_id)


def get_chat_generations(chat_id: str) -> list[dict]:
    """Get all generation data for a chat (for rebuilding UI on reload)."""
    return gen_model.get_generation_data_for_chat(chat_id)


def handle_generation_complete(progress):
    """Callback for when a generation job completes.

    Called from the polling system to update the database with output images.
    This is registered as a listener on comfyui_service.
    """
    if progress.phase in ("completed", "failed"):
        gen_model.update_job_status(
            progress.job_id,
            progress.phase,
        )

        if progress.phase == "completed" and progress.output_images:
            # Idempotency: skip if images already recorded for this job
            existing = gen_model.get_job_images(progress.job_id)
            if existing:
                logger.debug("Images already recorded for job %s, skipping", progress.job_id)
                return

            for img in progress.output_images:
                gen_model.add_generated_image(
                    job_id=progress.job_id,
                    filename=img.get("filename", ""),
                    subfolder=img.get("subfolder", ""),
                )

            # Trigger embedding and clustering for the generated prompt
            _embed_generated_prompt(progress.job_id)


def _embed_generated_prompt(job_id: str):
    """Embed the generated prompt and assign to nearest clusters.

    Runs in a background thread to avoid blocking the completion callback,
    since embedding involves an API call to the LLM service.
    """

    def _do_embed():
        try:
            # Get job settings for prompt and metadata
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

            # Create a document ID based on job_id
            doc_id = f"gen_{job_id}"

            # Check if already embedded
            if vector_store.document_exists(doc_id, "output"):
                logger.debug(f"Document {doc_id} already embedded, skipping")
                return

            # Get embedding for the prompt text
            embedding = embedding_service.embed(prompt)
            if not embedding:
                logger.warning(f"Empty embedding returned for job {job_id}")
                return

            # Build metadata matching the ingestion_service pattern
            output_folder = settings.get("output_folder", "") or ""
            concept = output_folder.split("/")[0] if output_folder else "generated"
            metadata = {
                "concept": concept,
                "base_dir": "generated",
                "source_file": f"generated_{job_id}",
                "dir_type": "output",
                "base_model": settings.get("base_model", "") or "",
                "loras": settings.get("loras", []) or [],
            }

            # Add to generated_prompts ChromaDB collection
            vector_store.add_document(
                doc_id=doc_id,
                text=prompt,
                embedding=embedding,
                source_type="output",
                metadata=metadata,
            )

            logger.info(f"Embedded generated prompt for job {job_id}")

            # Assign to nearest existing clusters (cross-folder and intra-folder)
            clustering_service.assign_new_docs_to_clusters(
                doc_ids=[doc_id],
                embeddings=[embedding],
                source_types=["output"],
                concepts=[concept],
            )

            logger.info(f"Assigned generated prompt {doc_id} to nearest clusters")

            # Auto-slot into subfolder if enabled
            _auto_slot_if_enabled(job_id, embedding, concept)

        except Exception as e:
            logger.error(
                f"Failed to embed generated prompt for job {job_id}: {e}",
                exc_info=True,
            )

    thread = threading.Thread(target=_do_embed, daemon=True)
    thread.start()


def _auto_slot_if_enabled(job_id: str, embedding: list[float], concept: str):
    """Silently move generated image to best-matching subfolder.

    1. Check setting: auto_organize_output. If disabled, return.
    2. Check if output_folder already has a subfolder (user specified). If so, return.
    3. Find intra-folder cluster centroids for this concept.
    4. Compute cosine similarity to each centroid.
    5. If best > 0.7, move file to that cluster's subfolder.
    """
    import json
    import os
    import re

    import numpy as np

    from src.models.settings import get_setting
    from src.models.database import get_db
    from sqlalchemy import text

    try:
        auto_organize = get_setting("auto_organize_output")
        if auto_organize != "true":
            return

        settings = gen_model.get_job_settings(job_id)
        if not settings:
            return

        output_folder = settings.get("output_folder") or ""
        # If user already specified a subfolder (contains /), skip
        if "/" in output_folder:
            return

        # Get intra-folder cluster centroids for this concept (output clusters)
        with get_db() as conn:
            result = conn.execute(
                text("""SELECT id, label, centroid
                   FROM clusters
                   WHERE cluster_type = 'intra_folder'
                   AND folder_path = :concept
                   AND source_type = 'output'
                   AND centroid IS NOT NULL"""),
                {"concept": concept},
            )
            clusters = []
            for row in result.fetchall():
                r = row._mapping
                centroid = json.loads(r["centroid"]) if isinstance(r["centroid"], str) else r["centroid"]
                if centroid:
                    clusters.append({
                        "id": r["id"],
                        "label": r["label"],
                        "centroid": np.array(centroid, dtype=np.float32),
                    })

        if not clusters:
            return

        # Compute cosine similarity
        emb = np.array(embedding, dtype=np.float32)
        emb_norm = emb / (np.linalg.norm(emb) or 1.0)

        best_sim = -1.0
        best_cluster = None
        for c in clusters:
            c_norm = c["centroid"] / (np.linalg.norm(c["centroid"]) or 1.0)
            sim = float(np.dot(emb_norm, c_norm))
            if sim > best_sim:
                best_sim = sim
                best_cluster = c

        if best_sim < 0.7 or best_cluster is None:
            return

        # Normalize cluster label to subfolder name
        label = best_cluster["label"] or f"group_{best_cluster['id']}"
        subfolder = re.sub(r'[^\w\s-]', '', label.lower())
        subfolder = re.sub(r'[\s-]+', '_', subfolder).strip('_')
        if len(subfolder) > 40:
            subfolder = subfolder[:40].rstrip('_')
        subfolder = subfolder or "group"

        # Get the generated image file path
        images = gen_model.get_images_by_job_ids([job_id])
        if not images:
            return

        from src.models.browser import get_root_directories
        roots = get_root_directories()
        if not roots:
            return

        for img in images:
            file_path = img.get("file_path")
            # file_path is often NULL for freshly-generated images; resolve
            # from filename + subfolder against the output data directories.
            if not file_path:
                file_path = comfyui_service.resolve_image_path(
                    img.get("filename", ""), img.get("subfolder", ""),
                )
            if not file_path or not os.path.isfile(file_path):
                continue

            # Compute new path
            current_dir = os.path.dirname(file_path)
            new_dir = os.path.join(current_dir, subfolder)
            os.makedirs(new_dir, exist_ok=True)

            filename = os.path.basename(file_path)
            new_path = os.path.join(new_dir, filename)

            # Move file
            os.rename(file_path, new_path)

            # Update DB
            with get_db() as conn:
                conn.execute(
                    text("UPDATE generated_images SET file_path = :new_path WHERE id = :id"),
                    {"new_path": new_path, "id": img["id"]},
                )
                new_output_folder = f"{output_folder}/{subfolder}" if output_folder else subfolder
                conn.execute(
                    text("UPDATE generation_settings SET output_folder = :folder WHERE job_id = :jid"),
                    {"folder": new_output_folder, "jid": job_id},
                )

            logger.info("Auto-slotted image %s to subfolder %s (similarity=%.3f)",
                         img["id"], subfolder, best_sim)

    except Exception as e:
        logger.warning("Auto-slot failed for job %s: %s", job_id, e, exc_info=True)


def initialize():
    """Register the generation completion listener.

    Called during app startup.
    """
    comfyui_service.add_status_listener(handle_generation_complete)

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
