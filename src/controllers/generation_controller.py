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
            concept = output_folder if output_folder else "generated"
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

        except Exception as e:
            logger.error(
                f"Failed to embed generated prompt for job {job_id}: {e}",
                exc_info=True,
            )

    thread = threading.Thread(target=_do_embed, daemon=True)
    thread.start()


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
