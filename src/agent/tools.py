"""Tool definitions and execution for the agent loop.

Uses native Gemini tool calling via function declarations.
"""

import json
import logging

from google.genai import types

from src.services import embedding_service
from src.models import vector_store

logger = logging.getLogger(__name__)


# ── Tool declarations for Gemini ─────────────────────────────────────────

TOOL_DECLARATIONS = [
    types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="search_similar_prompts",
            description=(
                "Search the prompt database for prompts semantically similar to a query. "
                "Use this when the user describes what they want or provides a seed phrase. "
                "Expand concept-level requests into rich semantic queries."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "query": types.Schema(type="STRING", description="Search query text to find similar prompts"),
                    "k": types.Schema(type="INTEGER", description="Number of results to return (default 10)"),
                    "source_type": types.Schema(
                        type="STRING",
                        description="Filter by source: 'training', 'output', or null for both",
                        nullable=True,
                    ),
                    "concept": types.Schema(
                        type="STRING",
                        description="Filter by concept/subdirectory name",
                        nullable=True,
                    ),
                },
                required=["query"],
            ),
        ),
        types.FunctionDeclaration(
            name="search_diverse_prompts",
            description=(
                "Search for prompts that are different/distant from the query. "
                "Use this to find contrasting examples for variety."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "query": types.Schema(type="STRING", description="Query text to find diverse prompts away from"),
                    "k": types.Schema(type="INTEGER", description="Number of results (default 10)"),
                    "source_type": types.Schema(
                        type="STRING",
                        description="Filter by source: 'training', 'output', or null for both",
                        nullable=True,
                    ),
                },
                required=["query"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_random_prompts",
            description="Get random prompts from the database for inspiration.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "k": types.Schema(type="INTEGER", description="Number of random prompts (default 10)"),
                    "source_type": types.Schema(
                        type="STRING",
                        description="Filter by source: 'training', 'output', or null for both",
                        nullable=True,
                    ),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="get_opposite_prompts",
            description="Find prompts most dissimilar to the query. Use when the user wants the opposite.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "query": types.Schema(type="STRING", description="Query to find opposites of"),
                    "k": types.Schema(type="INTEGER", description="Number of results (default 10)"),
                    "source_type": types.Schema(
                        type="STRING",
                        description="Filter by source: 'training', 'output', or null for both",
                        nullable=True,
                    ),
                },
                required=["query"],
            ),
        ),
        types.FunctionDeclaration(
            name="list_concepts",
            description="List available concept names and their counts in the database.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "source_type": types.Schema(
                        type="STRING",
                        description="Filter by source: 'training', 'output', or null for both",
                        nullable=True,
                    ),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="get_dataset_overview",
            description=(
                "Get a high-level overview of the entire dataset. Returns folder names, "
                "source types, prompt counts, per-folder summary terms, and cross-folder "
                "themes. This is pre-loaded in your context — only call if data may have "
                "changed mid-conversation."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={},
            ),
        ),
        types.FunctionDeclaration(
            name="get_folder_themes",
            description=(
                "Get the intra-folder cluster themes for a specific concept folder. "
                "Returns theme labels and prompt counts. Call this to explore the "
                "thematic variety within a folder before searching."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "folder_name": types.Schema(
                        type="STRING",
                        description="The concept folder name (e.g. 'salma', 'clothes_gown')",
                    ),
                },
                required=["folder_name"],
            ),
        ),
        types.FunctionDeclaration(
            name="query_themed_prompts",
            description=(
                "Search for prompts using a comprehensive themed query. "
                "Returns directly similar prompts, prompts from matching "
                "intra-folder themes, prompts from matching cross-folder themes, "
                "and optionally random/opposite prompts. This is your primary "
                "search tool — use it for initial exploration. The k-values for "
                "each category come from the app settings."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "query": types.Schema(
                        type="STRING",
                        description="Search query text — expand concept-level requests into rich semantic queries",
                    ),
                    "include_random": types.Schema(
                        type="BOOLEAN",
                        description="Include random prompts for variety (default: false)",
                        nullable=True,
                    ),
                    "include_opposite": types.Schema(
                        type="BOOLEAN",
                        description="Include opposite/contrasting prompts (default: false)",
                        nullable=True,
                    ),
                    "source_type": types.Schema(
                        type="STRING",
                        description="Filter by source: 'training', 'output', or null for both",
                        nullable=True,
                    ),
                },
                required=["query"],
            ),
        ),
        # ── Generation tools ──────────────────────────────────────────
        types.FunctionDeclaration(
            name="generate_image",
            description=(
                "Submit a single prompt for image generation via ComfyUI. "
                "Call this multiple times in one turn to generate different prompts "
                "with different settings. Only call when the user explicitly asks "
                "you to auto-generate. Seed should be -1 (random) unless the user "
                "explicitly specifies a seed or says 'use the same seed'. "
                "Unspecified optional settings use the user's configured defaults."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "prompt": types.Schema(type="STRING", description="The positive prompt text to generate"),
                    "negative_prompt": types.Schema(
                        type="STRING",
                        description="Negative prompt (defaults to user's configured default)",
                        nullable=True,
                    ),
                    "base_model": types.Schema(
                        type="STRING",
                        description="Diffusion model filename (defaults to user's configured default)",
                        nullable=True,
                    ),
                    "loras": types.Schema(
                        type="ARRAY",
                        description="LoRA filenames to apply (strength defaults to 1.0)",
                        items=types.Schema(type="STRING", description="LoRA filename"),
                        nullable=True,
                    ),
                    "output_folder": types.Schema(
                        type="STRING",
                        description="Output subdirectory name",
                        nullable=True,
                    ),
                    "seed": types.Schema(
                        type="INTEGER",
                        description="Seed (-1 for random, which is the default)",
                        nullable=True,
                    ),
                    "num_images": types.Schema(
                        type="INTEGER",
                        description="Number of images to generate (default 1)",
                        nullable=True,
                    ),
                    "sampler": types.Schema(
                        type="STRING",
                        description="Sampler name",
                        nullable=True,
                    ),
                    "cfg_scale": types.Schema(
                        type="NUMBER",
                        description="CFG scale value",
                        nullable=True,
                    ),
                    "scheduler": types.Schema(
                        type="STRING",
                        description="Scheduler name",
                        nullable=True,
                    ),
                    "steps": types.Schema(
                        type="INTEGER",
                        description="Number of sampling steps",
                        nullable=True,
                    ),
                },
                required=["prompt"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_available_loras",
            description=(
                "List available LoRA model filenames from ComfyUI. "
                "Call this when the user mentions a LoRA by description and you need "
                "the exact filename to pass to generate_image."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={},
            ),
        ),
        types.FunctionDeclaration(
            name="get_output_directories",
            description=(
                "List available output subdirectories. Call this when the user references "
                "an output folder or when you need to choose where to save generated images."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={},
            ),
        ),
        types.FunctionDeclaration(
            name="get_last_generation_settings",
            description=(
                "Get the full settings from the most recent completed generation job. "
                "Use when the user says 're-use settings', 'same as last time', or "
                "'pull settings from [folder]'. Set current_chat=true to only look at "
                "generations from this conversation."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "output_folder": types.Schema(
                        type="STRING",
                        description="If specified, get settings from the latest job in this output folder",
                        nullable=True,
                    ),
                    "current_chat": types.Schema(
                        type="BOOLEAN",
                        description="If true, only search generations from the current chat (default: false)",
                        nullable=True,
                    ),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="get_last_generated_prompts",
            description=(
                "Get the actual prompts submitted in the most recent generation jobs. "
                "These may differ from your generated_prompts if the user edited them "
                "before submitting. Use for refinement when the user wants to tweak "
                "a previously generated prompt. Set current_chat=true to only look at "
                "generations from this conversation."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "count": types.Schema(
                        type="INTEGER",
                        description="Number of recent jobs to retrieve prompts from (default 1)",
                        nullable=True,
                    ),
                    "current_chat": types.Schema(
                        type="BOOLEAN",
                        description="If true, only search generations from the current chat (default: false)",
                        nullable=True,
                    ),
                },
            ),
        ),
        # ── State management ─────────────────────────────────────────
        types.FunctionDeclaration(
            name="update_state",
            description=(
                "Update the agent's working state to track progress through the workflow. "
                "Call this to record prompt requirements as you learn them (as a JSON string of key-value pairs), "
                "record dataset knowledge, manage tasks, change the workflow phase, or save generated prompts. "
                "You MUST call this whenever you learn new requirements or change phase."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "phase": types.Schema(
                        type="STRING",
                        description=(
                            "Current workflow phase: 'gathering_info', 'searching', "
                            "'generating', 'refining', or 'complete'"
                        ),
                        nullable=True,
                    ),
                    "task_completed": types.Schema(
                        type="STRING",
                        description="Name of a task to mark as completed (moves from in_progress/pending to completed)",
                        nullable=True,
                    ),
                    "task_started": types.Schema(
                        type="STRING",
                        description="Name of a task to start working on (moves from pending to in_progress)",
                        nullable=True,
                    ),
                    "task_added": types.Schema(
                        type="STRING",
                        description="Name of a new task to add to pending",
                        nullable=True,
                    ),
                    "prompt_requirements": types.Schema(
                        type="STRING",
                        description=(
                            "JSON string of prompt requirements to merge into state. "
                            "Agent decides the keys (e.g. subject, style, mood, lighting, etc.)"
                        ),
                        nullable=True,
                    ),
                    "dataset_knowledge": types.Schema(
                        type="STRING",
                        description=(
                            "JSON string of dataset knowledge to merge into state. "
                            "Agent records what it learns about the dataset (e.g. relevant concepts, themes, folder notes)"
                        ),
                        nullable=True,
                    ),
                    "generated_prompt": types.Schema(
                        type="STRING",
                        description="A prompt you just generated, to save in the state",
                        nullable=True,
                    ),
                    "refinement_note": types.Schema(
                        type="STRING",
                        description="A note about a refinement requested by the user",
                        nullable=True,
                    ),
                },
            ),
        ),
    ]),
]


# ── Tool execution ───────────────────────────────────────────────────────

def execute_tool(name: str, args: dict, context: dict | None = None) -> dict:
    """Execute a tool call and return the result as a dict.

    Args:
        name: Tool name.
        args: Tool arguments from the LLM.
        context: Optional context dict with keys like 'chat_id' injected
                 by the agent loop for tools that need session awareness.

    Note: 'update_state' is handled here as a passthrough — it returns
    the updates dict. The caller (loop.py) is responsible for applying
    these updates to the actual agent state.
    """
    try:
        if name == "search_similar_prompts":
            return _search_similar(args)
        elif name == "search_diverse_prompts":
            return _search_diverse(args)
        elif name == "get_random_prompts":
            return _get_random(args)
        elif name == "get_opposite_prompts":
            return _get_opposite(args)
        elif name == "list_concepts":
            return _list_concepts(args)
        elif name == "get_dataset_overview":
            return _get_dataset_overview(args)
        elif name == "get_folder_themes":
            return _get_folder_themes(args)
        elif name == "query_themed_prompts":
            return _query_themed_prompts(args)
        elif name == "generate_image":
            return _generate_image(args, context or {})
        elif name == "get_available_loras":
            return _get_available_loras(args)
        elif name == "get_output_directories":
            return _get_output_directories(args)
        elif name == "get_last_generation_settings":
            return _get_last_generation_settings(args, context or {})
        elif name == "get_last_generated_prompts":
            return _get_last_generated_prompts(args, context or {})
        elif name == "update_state":
            return _validate_and_passthrough_state_update(args)
        else:
            return {"error": f"Unknown tool: {name}"}
    except Exception as e:
        logger.error(f"Tool execution error ({name}): {e}", exc_info=True)
        return {"error": str(e)}


_UPDATE_STATE_FIELDS = {
    "phase", "task_completed", "task_started", "task_added",
    "prompt_requirements", "dataset_knowledge",
    "generated_prompt", "refinement_note",
}


def _validate_and_passthrough_state_update(args: dict) -> dict:
    """Validate update_state args and return as a passthrough for loop.py to apply.

    Checks that at least one recognized field has a non-None value.
    Returns an error message to the agent if no valid fields are found.
    """
    valid_updates = {k: v for k, v in args.items() if k in _UPDATE_STATE_FIELDS and v is not None}

    if not valid_updates:
        return {
            "error": (
                "update_state called with no recognized fields. "
                f"You must provide at least one of: {', '.join(sorted(_UPDATE_STATE_FIELDS))}. "
                f"Received: {json.dumps(args)}"
            )
        }

    return {"status": "ok", "updates_applied": valid_updates}


def _search_similar(args: dict) -> dict:
    query = args.get("query", "")
    k = args.get("k", 10)
    source_type = args.get("source_type")
    concept = args.get("concept")

    query_embedding = embedding_service.embed(query)
    results = vector_store.search_similar(
        query_embedding, k=k, source_type=source_type, concept=concept
    )

    return {
        "query": query,
        "count": len(results),
        "prompts": [
            {
                "text": r["document"],
                "concept": r["metadata"].get("concept", ""),
                "source": r["metadata"].get("dir_type", ""),
                "distance": round(r["distance"], 4),
            }
            for r in results
        ],
    }


def _search_diverse(args: dict) -> dict:
    query = args.get("query", "")
    k = args.get("k", 10)
    source_type = args.get("source_type")

    query_embedding = embedding_service.embed(query)
    results = vector_store.search_diverse(query_embedding, k=k, source_type=source_type)

    return {
        "query": query,
        "count": len(results),
        "prompts": [
            {
                "text": r["document"],
                "concept": r["metadata"].get("concept", ""),
                "source": r["metadata"].get("dir_type", ""),
            }
            for r in results
        ],
    }


def _get_random(args: dict) -> dict:
    k = args.get("k", 10)
    source_type = args.get("source_type")

    results = vector_store.get_random(k=k, source_type=source_type)

    return {
        "count": len(results),
        "prompts": [
            {
                "text": r["document"],
                "concept": r["metadata"].get("concept", ""),
                "source": r["metadata"].get("dir_type", ""),
            }
            for r in results
        ],
    }


def _get_opposite(args: dict) -> dict:
    query = args.get("query", "")
    k = args.get("k", 10)
    source_type = args.get("source_type")

    query_embedding = embedding_service.embed(query)
    # Use a large offset to get the most distant prompts
    results = vector_store.search_diverse(query_embedding, k=k, offset=100, source_type=source_type)

    return {
        "query": query,
        "count": len(results),
        "prompts": [
            {
                "text": r["document"],
                "concept": r["metadata"].get("concept", ""),
                "source": r["metadata"].get("dir_type", ""),
            }
            for r in results
        ],
    }


def _list_concepts(args: dict) -> dict:
    source_type = args.get("source_type")
    concepts = vector_store.list_concepts(source_type=source_type)

    return {
        "count": len(concepts),
        "concepts": concepts,
    }


def _get_dataset_overview(args: dict) -> dict:
    """Get the lightweight dataset overview (no intra-folder cluster details)."""
    from src.services import clustering_service
    return clustering_service.get_dataset_overview()


def _get_folder_themes(args: dict) -> dict:
    """Get intra-folder cluster themes for a specific folder."""
    from src.services import clustering_service
    folder_name = args.get("folder_name", "")
    if not folder_name:
        return {"error": "folder_name is required"}
    return clustering_service.get_folder_themes(folder_name)


def _query_themed_prompts(args: dict) -> dict:
    """Execute a themed prompt query across multiple sources."""
    from src.services import clustering_service

    query = args.get("query", "")
    include_random = args.get("include_random", False)
    include_opposite = args.get("include_opposite", False)
    source_type = args.get("source_type")

    # Generate embedding for the query
    query_embedding = embedding_service.embed(query)

    # Read k-values from settings
    from src.models.settings import get_setting
    k_random = int(get_setting("query_k_random") or "3") if include_random else 0
    k_opposite = int(get_setting("query_k_random") or "3") if include_opposite else 0

    result = clustering_service.get_themed_prompts(
        query_embedding=query_embedding,
        k_random=k_random,
        k_opposite=k_opposite,
        source_type=source_type,
    )

    return result


# ── Generation tool implementations ──────────────────────────────────────

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
    chat_id = context.get("chat_id") if args.get("current_chat") else None
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
