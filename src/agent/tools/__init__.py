"""Tool definitions and execution for the agent loop.

Uses native Gemini tool calling via function declarations.
"""

from src.agent.tools.declarations import TOOL_DECLARATIONS
from src.agent.tools.dispatcher import execute_tool
from src.agent.tools.summarizers import summarize_tool_result

__all__ = ["TOOL_DECLARATIONS", "execute_tool", "summarize_tool_result"]
