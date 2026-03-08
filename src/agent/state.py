"""Agent state management."""

import json
import logging

logger = logging.getLogger(__name__)

# Maximum number of generated prompts to retain in state.
MAX_RECENT_PROMPTS = 5


def create_initial_state() -> dict:
    """Create a fresh agent state for a new chat."""
    return {
        "phase": "gathering_info",
        "prompt_requirements": {},
        "generated_prompts": [],
        "context": "",
    }


def state_to_context(state: dict) -> str:
    """Convert agent state to a context string for inclusion in the system prompt.

    This is read-only context for the model. State modifications happen
    exclusively through the update_state tool call.
    """
    return (
        "\n\n## Current Agent State\n\n"
        "Below is your current working state for this conversation. "
        "Use the `update_state` tool to modify it as you progress through the workflow.\n\n"
        f"```json\n{json.dumps(state, indent=2)}\n```"
    )


def apply_state_update(state: dict, updates: dict) -> dict:
    """Apply updates from the update_state tool call to the agent state.

    Args:
        state: The current agent state dict (modified in-place).
        updates: Dict of field names to new values from the tool call.

    Returns:
        The updated state dict.
    """
    if "phase" in updates and updates["phase"]:
        state["phase"] = updates["phase"]

    if "prompt_requirements" in updates and updates["prompt_requirements"]:
        try:
            parsed = json.loads(updates["prompt_requirements"])
            state.setdefault("prompt_requirements", {}).update(parsed)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to parse prompt_requirements JSON: {e}")

    if "generated_prompts" in updates and updates["generated_prompts"]:
        try:
            parsed = json.loads(updates["generated_prompts"])
            if isinstance(parsed, list):
                prompts = state.setdefault("generated_prompts", [])
                prompts.extend(parsed)
                # Keep only the most recent prompts
                if len(prompts) > MAX_RECENT_PROMPTS:
                    state["generated_prompts"] = prompts[-MAX_RECENT_PROMPTS:]
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to parse generated_prompts JSON: {e}")

    if "context" in updates and updates["context"] is not None:
        state["context"] = updates["context"]

    return state
