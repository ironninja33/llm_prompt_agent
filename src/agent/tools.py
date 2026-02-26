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
            name="get_dataset_map",
            description=(
                "Get a map of the dataset showing folder structure, "
                "intra-folder themes, cross-cutting themes, and prompt counts. "
                "Use this first to understand what data is available before searching."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={},
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

def execute_tool(name: str, args: dict) -> dict:
    """Execute a tool call and return the result as a dict.

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
        elif name == "get_dataset_map":
            return _get_dataset_map(args)
        elif name == "query_themed_prompts":
            return _query_themed_prompts(args)
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


def _get_dataset_map(args: dict) -> dict:
    """Get the dataset map showing themes, folders, and stats."""
    from src.services import clustering_service
    return clustering_service.get_dataset_map()


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
