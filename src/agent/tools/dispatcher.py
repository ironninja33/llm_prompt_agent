"""Tool execution dispatcher — routes tool calls to implementations."""

import logging

from src.agent.tools.search_tools import (
    _search_similar,
    _get_folder_themes,
    _query_dataset_map,
)
from src.agent.tools.query_diverse import _query_diverse_prompts
from src.agent.tools.generation_tools import (
    _generate_image,
    _get_available_loras,
    _get_output_directories,
    _get_last_generation_settings,
)
from src.agent.tools.metrics_tools import (
    _get_deletion_insights,
    _get_successful_patterns,
)

logger = logging.getLogger(__name__)


def execute_tool(name: str, args: dict, context: dict | None = None) -> dict:
    """Execute a tool call and return the result as a dict.

    Args:
        name: Tool name.
        args: Tool arguments from the LLM.
        context: Optional context dict with keys like 'chat_id' injected
                 by the agent loop for tools that need session awareness.
    """
    try:
        if name == "search_similar_prompts":
            return _search_similar(args)
        elif name == "get_folder_themes":
            return _get_folder_themes(args)
        elif name == "query_diverse_prompts":
            return _query_diverse_prompts(args)
        elif name == "query_dataset_map":
            return _query_dataset_map(args)
        elif name == "get_deletion_insights":
            return _get_deletion_insights(args)
        elif name == "get_successful_patterns":
            return _get_successful_patterns(args)
        elif name == "generate_image":
            return _generate_image(args, context or {})
        elif name == "get_available_loras":
            return _get_available_loras(args)
        elif name == "get_output_directories":
            return _get_output_directories(args)
        elif name == "get_last_generation_settings":
            return _get_last_generation_settings(args, context or {})
        else:
            return {"error": f"Unknown tool: {name}"}
    except Exception as e:
        logger.error(f"Tool execution error ({name}): {e}", exc_info=True)
        return {"error": str(e)}
