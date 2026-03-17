"""Tool execution dispatcher — routes tool calls to implementations."""

import logging

from src.agent.tools.search_tools import (
    _search_similar,
    _search_diverse,
    _get_random,
    _get_opposite,
    _list_concepts,
    _get_dataset_overview,
    _get_folder_themes,
    _query_dataset_map,
)
from src.agent.tools.query_diverse import _query_diverse_prompts
from src.agent.tools.generation_tools import (
    _generate_image,
    _get_available_loras,
    _get_output_directories,
    _get_last_generation_settings,
    _get_last_generated_prompts,
)
from src.agent.tools.state_tools import _validate_and_passthrough_state_update

logger = logging.getLogger(__name__)


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
        elif name == "query_diverse_prompts":
            return _query_diverse_prompts(args)
        elif name == "query_dataset_map":
            return _query_dataset_map(args)
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
