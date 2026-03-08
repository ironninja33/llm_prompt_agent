"""Tool call persistence — save and retrieve tool calls for messages."""

import json
import logging
from sqlalchemy import text
from src.models.database import get_db, row_to_dict

logger = logging.getLogger(__name__)

# Maximum length for response_summary stored in DB
MAX_SUMMARY_LENGTH = 500


def save_tool_calls(message_id: int, calls: list[dict]):
    """Persist tool calls for an assistant message.

    Each call should have keys: tool, args, result, and optionally iteration.
    The result is truncated to ~500 chars for storage.
    Calls are stored with an explicit sequence number preserving execution order.
    """
    if not calls:
        return

    with get_db() as conn:
        for sequence, call in enumerate(calls):
            tool_name = call.get("tool", "unknown")
            parameters = json.dumps(call.get("args", {}))
            iteration = call.get("iteration", 1)

            # Truncate result to a summary
            result = call.get("result", {})
            result_str = json.dumps(result)
            if len(result_str) > MAX_SUMMARY_LENGTH:
                response_summary = result_str[:MAX_SUMMARY_LENGTH] + "..."
            else:
                response_summary = result_str

            conn.execute(
                text("""INSERT INTO tool_calls
                    (message_id, tool_name, parameters, response_summary, sequence, iteration)
                    VALUES (:message_id, :tool_name, :parameters, :response_summary,
                            :sequence, :iteration)"""),
                {"message_id": message_id, "tool_name": tool_name,
                 "parameters": parameters, "response_summary": response_summary,
                 "sequence": sequence, "iteration": iteration},
            )


def get_tool_calls(message_id: int) -> list[dict]:
    """Get saved tool calls for a message, ordered by execution sequence."""
    with get_db() as conn:
        result = conn.execute(
            text("""SELECT tool_name, parameters, response_summary, sequence, iteration
               FROM tool_calls WHERE message_id = :message_id ORDER BY sequence ASC"""),
            {"message_id": message_id},
        )
        calls = []
        for row in result.fetchall():
            r = row_to_dict(row)
            params = {}
            if r["parameters"]:
                try:
                    params = json.loads(r["parameters"])
                except (json.JSONDecodeError, TypeError):
                    pass

            result_val = {}
            if r["response_summary"]:
                try:
                    result_val = json.loads(r["response_summary"])
                except (json.JSONDecodeError, TypeError):
                    result_val = {"summary": r["response_summary"]}

            calls.append({
                "tool": r["tool_name"],
                "args": params,
                "result": result_val,
                "sequence": r.get("sequence", 0),
                "iteration": r.get("iteration", 1),
            })
        return calls
