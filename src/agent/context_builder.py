"""Centralized context injection for the agent loop.

Builds the message array sent to the LLM, including:
- State prefix (conversation context summary)
- Dataset overview (early turns only)
- Generation outcomes from agent's last turn
- Browser lineage feedback (browser gens traced back to this chat)
- Chat history with sliding window trimming
"""

import json
import logging

from src.services import clustering_service
from src.models import settings, chat as chat_model
from src.models import generation as gen_model
from src.models import metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_messages(
    chat_id: str,
    agent_state: dict,
) -> list[dict]:
    """Build the full message array for the LLM.

    Order:
        [state_prefix] + [overview_prefix] + [outcomes] + [browser_lineage] + [history]

    outcomes and browser_lineage are inserted before the final user message.
    """
    state_prefix = _build_state_prefix(agent_state)
    overview_prefix = _build_overview_prefix(agent_state)
    history = _build_message_history(chat_id, agent_state)

    messages = state_prefix + overview_prefix + history

    # Collect injection pairs to insert before the final user message
    injections = []

    outcomes = _build_outcomes_prefix(chat_id)
    if outcomes:
        injections.extend(outcomes)

    lineage = _build_browser_lineage_prefix(chat_id)
    if lineage:
        injections.extend(lineage)

    if injections and len(messages) > 0:
        messages = messages[:-1] + injections + messages[-1:]
        logger.info("Injected %d context messages (outcomes + lineage)", len(injections))

    return messages


def state_has_content(agent_state: dict) -> bool:
    """Check if agent state has meaningful content worth preserving over history."""
    return bool(
        agent_state.get("direction")
        or agent_state.get("initial_request")
        or agent_state.get("recent_prompts")
    )


# ---------------------------------------------------------------------------
# State prefix
# ---------------------------------------------------------------------------

def state_to_context(state: dict) -> str:
    """Convert agent state to a natural-language context block."""
    parts = ["## Conversation Context"]

    if state.get("initial_request"):
        parts.append(f"**Original request:** {state['initial_request']}")

    if state.get("direction"):
        parts.append(f"**Direction:** {state['direction']}")

    prompts = state.get("recent_prompts", [])
    if prompts:
        parts.append(f"**Your last {len(prompts)} prompt(s):**")
        for i, p in enumerate(prompts, 1):
            parts.append(f"{i}. {p}")

    search_ctx = state.get("search_context", {})
    concepts = search_ctx.get("active_concepts", [])
    if concepts:
        parts.append(f"**Explored concepts:** {', '.join(concepts[:8])}")

    queries = search_ctx.get("recent_queries", [])
    if queries:
        parts.append(f"**Recent searches:** {'; '.join(q[:60] for q in queries[-3:])}")

    return "\n\n".join(parts)


def _build_state_prefix(agent_state: dict) -> list[dict]:
    """Build the state prefix message pair."""
    return [
        {"role": "user", "content": state_to_context(agent_state)},
        {"role": "model", "content": "Acknowledged."},
    ]


# ---------------------------------------------------------------------------
# Dataset overview
# ---------------------------------------------------------------------------

def _build_overview_prefix(agent_state: dict) -> list[dict]:
    """Inject dataset overview for early turns (before state accumulates)."""
    if state_has_content(agent_state):
        return []

    try:
        overview = clustering_service.get_dataset_overview()
        overview_json = json.dumps(overview, separators=(",", ":"))
        logger.info("Injected dataset overview into messages (%d chars)",
                     len(overview_json))
        return [
            {"role": "user", "content": (
                "Dataset overview (pre-loaded context \u2014 do not call "
                "get_dataset_overview, this data is already available):\n"
                + overview_json
            )},
            {"role": "model", "content": (
                "I have the dataset overview with folder structure, "
                "cross-folder themes, and statistics. I'll call "
                "get_folder_themes when I need intra-folder details "
                "for a specific folder. Ready to help."
            )},
        ]
    except Exception as e:
        logger.warning("Failed to load dataset overview: %s", e)
        return []


# ---------------------------------------------------------------------------
# Generation outcomes (agent's own gens)
# ---------------------------------------------------------------------------

def _build_outcomes_prefix(chat_id: str) -> list[dict]:
    """Build injection if the last assistant turn had generation jobs."""
    try:
        from src.agent.tools.generation_tools import _get_generation_outcomes

        db_messages = chat_model.get_messages(chat_id)
        last_assistant_id = None
        for msg in reversed(db_messages):
            if msg["role"] == "assistant":
                last_assistant_id = msg["id"]
                break

        if not last_assistant_id:
            return []

        result = _get_generation_outcomes(
            {"message_id": last_assistant_id},
            {"chat_id": chat_id},
        )

        if "error" in result or not result.get("outcomes"):
            return []

        lines = ["Generation outcomes from your last suggestions:"]
        for o in result["outcomes"]:
            parts = []
            parts.append(f"{o['total_images']} images")
            parts.append(f"{o['kept']} kept")
            if o["deleted"] > 0:
                reasons = ", ".join(
                    f"{v} {k}" for k, v in o.get("deletion_reasons", {}).items()
                )
                parts.append(
                    f"{o['deleted']} deleted ({reasons})" if reasons
                    else f"{o['deleted']} deleted"
                )
            line = f"- Prompt {o['position']}: {', '.join(parts)}"
            lines.append(line)

            for v in o.get("variations", []):
                var_parts = [f"{v['total_images']} images, {v['kept']} kept"]
                if v["deleted"] > 0:
                    reasons = ", ".join(
                        f"{ct} {r}" for r, ct in v.get("deletion_reasons", {}).items()
                    )
                    var_parts.append(
                        f"{v['deleted']} deleted ({reasons})" if reasons
                        else f"{v['deleted']} deleted"
                    )
                lines.append(f"  \u2514 User modified prompt: {v['diff']}")
                lines.append(f"    {', '.join(var_parts)}")

        text = "\n".join(lines)
        return [
            {"role": "user", "content": text},
            {"role": "model", "content": (
                "Understood. I'll use these outcomes and any "
                "user-modified prompts as my starting point."
            )},
        ]

    except Exception as e:
        logger.warning("Failed to build outcomes prefix: %s", e)
        return []


# ---------------------------------------------------------------------------
# Browser lineage feedback
# ---------------------------------------------------------------------------

def _build_browser_lineage_prefix(chat_id: str) -> list[dict]:
    """Build injection showing browser activity traced to this chat.

    Finds browser-sourced jobs whose parent_job_id links to a job from
    this chat. Shows kept/deleted/modified info.
    """
    try:
        descendants = gen_model.get_browser_descendants_for_chat(chat_id)
        if not descendants:
            return []

        job_ids = [d["id"] for d in descendants]
        deletions_map = metrics.get_deletions_for_jobs(job_ids)

        # Group by parent (the chat-originated job)
        by_parent: dict[str, list] = {}
        for d in descendants:
            by_parent.setdefault(d["parent_job_id"], []).append(d)

        lines = ["Browser activity on your prompts from this chat:"]
        for parent_id, children in by_parent.items():
            parent_settings = gen_model.get_job_settings(parent_id)
            parent_prompt = (parent_settings or {}).get("positive_prompt", "")

            kept = 0
            deleted = 0
            reasons: dict[str, int] = {}
            diffs = []

            for child in children:
                child_dels = deletions_map.get(child["id"], [])
                if child_dels:
                    deleted += len(child_dels)
                    for d in child_dels:
                        reasons[d["reason"]] = reasons.get(d["reason"], 0) + 1
                else:
                    # No deletion record + completed = kept
                    if child.get("status") == "completed":
                        kept += 1

                # Check for prompt modification
                child_prompt = child.get("positive_prompt", "")
                if child_prompt and child_prompt != parent_prompt and len(diffs) < 2:
                    from src.agent.tools.generation_tools import _compact_diff
                    diff = _compact_diff(parent_prompt, child_prompt)
                    if diff:
                        diffs.append(diff[:120])

            total = len(children)
            prompt_preview = parent_prompt[:80] + ("..." if len(parent_prompt) > 80 else "")
            line = f'- "{prompt_preview}" \u2192 {total} browser regen{"s" if total != 1 else ""}'
            if kept:
                line += f", {kept} kept"
            if deleted:
                reason_str = ", ".join(f"{c} {r}" for r, c in reasons.items())
                line += f", {deleted} deleted ({reason_str})"
            lines.append(line)

            for diff in diffs:
                lines.append(f"  Variation: {diff}")

        if len(lines) <= 1:
            return []  # No meaningful content

        return [
            {"role": "user", "content": "\n".join(lines)},
            {"role": "model", "content": "Noted. I'll factor in the browser regeneration feedback."},
        ]

    except Exception as e:
        logger.warning("Failed to build browser lineage prefix: %s", e)
        return []


# ---------------------------------------------------------------------------
# Chat history with sliding window
# ---------------------------------------------------------------------------

def _build_message_history(
    chat_id: str,
    agent_state: dict | None = None,
) -> list[dict]:
    """Build messages from chat history with sliding window trimming."""
    db_messages = chat_model.get_messages(chat_id)

    messages = []
    for msg in db_messages:
        role = msg["role"]
        content = msg["content"]

        # Skip error, empty, and partial messages
        if role == "assistant" and isinstance(msg.get("metadata"), dict):
            meta = msg["metadata"]
            if meta.get("is_error") or meta.get("is_partial"):
                continue
        if role == "assistant" and not content.strip():
            continue

        if role == "user":
            messages.append({"role": "user", "content": content})
        elif role == "assistant":
            messages.append({"role": "model", "content": content})

    # Sliding window when state has accumulated context
    if agent_state and state_has_content(agent_state):
        max_messages = int(settings.get_setting("context_history_pairs") or 3) * 2
        if len(messages) > max_messages:
            trimmed = len(messages) - max_messages
            messages = messages[-max_messages:]
            logger.info("Trimmed %d old history messages (keeping %d)",
                        trimmed, max_messages)

    return messages
