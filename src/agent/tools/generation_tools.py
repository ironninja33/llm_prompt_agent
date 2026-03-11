"""Generation tool implementations (ComfyUI integration)."""


def _generate_image(args: dict, context: dict) -> dict:
    """Submit a single prompt for image generation."""
    from src.services import comfyui_service
    from src.controllers import generation_controller

    prompt = args.get("prompt", "").strip()
    if not prompt:
        return {"error": "prompt is required and cannot be empty"}

    # Check ComfyUI connectivity first
    try:
        health = comfyui_service.check_health()
        if not health:
            return {"error": "ComfyUI is not reachable. Check that ComfyUI is running."}
    except Exception:
        return {"error": "ComfyUI is not reachable. Check that ComfyUI is running."}

    # Build settings dict
    settings = {"positive_prompt": prompt}

    # Map optional args to settings, only if provided (non-None)
    if args.get("negative_prompt") is not None:
        settings["negative_prompt"] = args["negative_prompt"]
    if args.get("base_model") is not None:
        settings["base_model"] = args["base_model"]
    if args.get("output_folder") is not None:
        settings["output_folder"] = args["output_folder"]
    if args.get("seed") is not None:
        settings["seed"] = args["seed"]
    if args.get("num_images") is not None:
        settings["num_images"] = args["num_images"]
    if args.get("sampler") is not None:
        settings["sampler"] = args["sampler"]
    if args.get("cfg_scale") is not None:
        settings["cfg_scale"] = args["cfg_scale"]
    if args.get("scheduler") is not None:
        settings["scheduler"] = args["scheduler"]
    if args.get("steps") is not None:
        settings["steps"] = args["steps"]

    # Handle loras: validate against available loras
    if args.get("loras"):
        available = comfyui_service.get_available_models("loras")
        for name in args["loras"]:
            if name not in available:
                return {
                    "error": f"LoRA '{name}' not found",
                    "valid_options": available,
                }
        settings["loras"] = list(args["loras"])

    chat_id = context.get("chat_id")
    job = generation_controller.submit_generation(
        chat_id=chat_id, message_id=None, settings=settings, source="chat",
    )

    if job.get("status") == "failed":
        return {"error": job.get("error", "Generation failed"), "job_id": job["id"]}

    return {
        "status": "submitted",
        "job_id": job["id"],
        "prompt": prompt,
        "settings_used": {
            k: v for k, v in job.get("settings", {}).items()
            if v is not None and k != "extra_settings"
        },
    }


def _get_available_loras(args: dict) -> dict:
    """List available LoRA models."""
    from src.services import comfyui_service

    try:
        loras = comfyui_service.get_available_models("loras")
    except Exception as e:
        return {"error": f"Failed to fetch loras from ComfyUI: {e}"}

    if not loras:
        return {"error": "No loras found. ComfyUI may not be running or has no loras installed."}

    return {
        "count": len(loras),
        "loras": loras,
    }


def _get_output_directories(args: dict) -> dict:
    """List available output directories."""
    from src.services import comfyui_service

    try:
        folders = comfyui_service.get_output_subfolders()
    except Exception as e:
        return {"error": f"Failed to list output directories: {e}"}

    return {
        "count": len(folders),
        "directories": folders,
    }


def _get_last_generation_settings(args: dict, context: dict) -> dict:
    """Get settings from the most recent completed generation job."""
    from src.models import generation as gen_model

    output_folder = args.get("output_folder")
    chat_id = context.get("chat_id") if args.get("current_chat") else None
    settings = gen_model.get_latest_job_settings(
        output_folder=output_folder, chat_id=chat_id,
    )

    if not settings:
        msg = "No completed generation jobs found"
        if output_folder:
            msg += f" in output folder '{output_folder}'"
        if chat_id:
            msg += " in the current chat"
        return {"error": msg}

    return {
        "settings": {
            k: v for k, v in settings.items()
            if v is not None and k != "extra_settings" and k != "workflow_name"
        },
    }


def _get_last_generated_prompts(args: dict, context: dict) -> dict:
    """Get prompts from the most recent generation jobs."""
    from src.models import generation as gen_model

    count = args.get("count", 1) or 1
    current_chat = args.get("current_chat")
    if current_chat is None:
        current_chat = True
    chat_id = context.get("chat_id") if current_chat else None
    jobs = gen_model.get_recent_job_prompts(count=count, chat_id=chat_id)

    if not jobs:
        msg = "No completed generation jobs found"
        if chat_id:
            msg += " in the current chat"
        return {"error": msg}

    return {
        "count": len(jobs),
        "jobs": jobs,
    }
