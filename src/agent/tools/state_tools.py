"""State management tool: validate and passthrough update_state calls."""

import json
import logging

logger = logging.getLogger(__name__)

_UPDATE_STATE_FIELDS = {
    "phase", "prompt_requirements", "generated_prompts", "context",
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
