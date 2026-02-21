"""Agent state management."""

import json
import logging

logger = logging.getLogger(__name__)


def create_initial_state() -> dict:
    """Create a fresh agent state for a new chat."""
    return {
        "phase": "gathering_info",
        "tasks": {
            "completed": [],
            "in_progress": [],
            "pending": ["understand_request", "explore_dataset", "generate_prompts"],
        },
        "prompt_requirements": {},
        "dataset_knowledge": {},
        "generated_prompts": [],
        "refinement_notes": [],
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
    tasks = state.setdefault("tasks", {"completed": [], "in_progress": [], "pending": []})

    if "phase" in updates and updates["phase"]:
        state["phase"] = updates["phase"]

    if "task_completed" in updates and updates["task_completed"]:
        task = updates["task_completed"]
        # Remove from in_progress or pending
        if task in tasks["in_progress"]:
            tasks["in_progress"].remove(task)
        elif task in tasks["pending"]:
            tasks["pending"].remove(task)
        # Add to completed if not already there
        if task not in tasks["completed"]:
            tasks["completed"].append(task)

    if "task_started" in updates and updates["task_started"]:
        task = updates["task_started"]
        # Remove from pending if present
        if task in tasks["pending"]:
            tasks["pending"].remove(task)
        # Add to in_progress if not already there
        if task not in tasks["in_progress"]:
            tasks["in_progress"].append(task)

    if "task_added" in updates and updates["task_added"]:
        task = updates["task_added"]
        if task not in tasks["pending"] and task not in tasks["in_progress"] and task not in tasks["completed"]:
            tasks["pending"].append(task)

    if "prompt_requirements" in updates and updates["prompt_requirements"]:
        try:
            parsed = json.loads(updates["prompt_requirements"])
            state.setdefault("prompt_requirements", {}).update(parsed)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to parse prompt_requirements JSON: {e}")

    if "dataset_knowledge" in updates and updates["dataset_knowledge"]:
        try:
            parsed = json.loads(updates["dataset_knowledge"])
            state.setdefault("dataset_knowledge", {}).update(parsed)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to parse dataset_knowledge JSON: {e}")

    if "generated_prompt" in updates and updates["generated_prompt"]:
        state.setdefault("generated_prompts", []).append(updates["generated_prompt"])

    if "refinement_note" in updates and updates["refinement_note"]:
        state.setdefault("refinement_notes", []).append(updates["refinement_note"])

    return state
