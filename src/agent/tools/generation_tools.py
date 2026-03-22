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


def _get_generation_outcomes(args: dict, context: dict) -> dict:
    """Check what happened with previously suggested prompts."""
    from difflib import SequenceMatcher
    from src.models import generation as gen_model
    from src.models import metrics

    chat_id = context.get("chat_id")
    message_id = args.get("message_id")

    # Default to most recent assistant message with generations
    if not message_id:
        if not chat_id:
            return {"error": "No chat context available"}
        message_id = gen_model.get_latest_agent_message_id(chat_id)
    if not message_id:
        return {"error": "No generation jobs found in this chat"}

    # Get all jobs for this message, partition into roots and children
    all_jobs = gen_model.get_jobs_by_message_id(message_id)
    if not all_jobs:
        return {"error": "No generation jobs found for that message"}

    root_jobs = [j for j in all_jobs if not j.get("parent_job_id")]
    non_roots = [j for j in all_jobs if j.get("parent_job_id")]

    # Build children map: parent_job_id → list of child jobs
    root_ids = {j["job_id"] for j in root_jobs}
    children_map: dict[str, list] = {rid: [] for rid in root_ids}
    for child in non_roots:
        parent = child["parent_job_id"]
        if parent in children_map:
            children_map[parent].append(child)
        elif len(root_ids) == 1:
            # Single root — assign orphaned children to it
            children_map[next(iter(root_ids))].append(child)

    # Collect all job IDs for deletion lookup
    all_job_ids = [j["job_id"] for j in all_jobs]

    # Get deletion records
    deletions_map = metrics.get_deletions_for_jobs(all_job_ids)

    # Build outcomes
    outcomes = []
    for i, root in enumerate(root_jobs):
        root_deletions = deletions_map.get(root["job_id"], [])
        root_images = gen_model.get_job_images(root["job_id"])
        root_kept = len(root_images)
        root_deleted = len(root_deletions)
        root_reasons = _count_reasons(root_deletions)

        # Process children (regenerations)
        children = children_map.get(root["job_id"], [])
        total_kept = root_kept
        total_deleted = root_deleted
        total_reasons = dict(root_reasons)

        variations = []
        for child in children:
            child_deletions = deletions_map.get(child["job_id"], [])
            child_images = gen_model.get_job_images(child["job_id"])
            child_kept = len(child_images)
            child_deleted = len(child_deletions)
            child_reasons = _count_reasons(child_deletions)

            total_kept += child_kept
            total_deleted += child_deleted
            for reason, count in child_reasons.items():
                total_reasons[reason] = total_reasons.get(reason, 0) + count

            # Check if prompt was modified
            child_prompt = child.get("positive_prompt", "")
            root_prompt = root.get("positive_prompt", "")
            if child_prompt and child_prompt != root_prompt:
                diff = _compact_diff(root_prompt, child_prompt)
                variations.append({
                    "diff": diff,
                    "total_images": child_kept + child_deleted,
                    "kept": child_kept,
                    "deleted": child_deleted,
                    "deletion_reasons": child_reasons,
                })

        # Deduplicate variations: group by diff text, keep only the
        # most recent unique diff (the user's final version is what matters).
        if variations:
            seen_diffs: dict[str, dict] = {}
            for v in variations:
                seen_diffs[v["diff"]] = v  # last one wins
            unique = list(seen_diffs.values())
            # Keep only the last unique variation (the user's final edit)
            variations = unique[-1:]

        outcome = {
            "position": i + 1,
            "total_images": total_kept + total_deleted,
            "kept": total_kept,
            "deleted": total_deleted,
        }
        if total_reasons:
            outcome["deletion_reasons"] = total_reasons
        if variations:
            outcome["variations"] = variations

        outcomes.append(outcome)

    return {"message_id": message_id, "outcomes": outcomes}


def _count_reasons(deletions: list[dict]) -> dict:
    """Count deletion reasons."""
    counts: dict[str, int] = {}
    for d in deletions:
        reason = d["reason"]
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _compact_diff(original: str, modified: str) -> str:
    """Produce a compact inline diff showing what changed between two prompts."""
    from difflib import SequenceMatcher

    orig_words = original.split()
    mod_words = modified.split()
    matcher = SequenceMatcher(None, orig_words, mod_words)

    parts = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            equal_words = orig_words[i1:i2]
            if len(equal_words) > 6:
                parts.append(" ".join(equal_words[:3]) + " ... " + " ".join(equal_words[-3:]))
            else:
                parts.append(" ".join(equal_words))
        elif tag == "replace":
            parts.append("~~" + " ".join(orig_words[i1:i2]) + "~~ → ++" + " ".join(mod_words[j1:j2]) + "++")
        elif tag == "delete":
            parts.append("~~" + " ".join(orig_words[i1:i2]) + "~~")
        elif tag == "insert":
            parts.append("++" + " ".join(mod_words[j1:j2]) + "++")

    return " ".join(parts)
