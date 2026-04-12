"""Agent state management — implicit extraction from conversation turns.

State is built automatically from the agent's tool calls and text response,
not from explicit update_state tool calls. The state captures:
- initial_request: First user message (set once, never changes)
- recent_prompts: Last 5 prompts from agent's ```prompt blocks
- direction: Brief summary of user's evolving preferences (LLM-generated)
- search_context: Recent queries and explored concepts
"""

import json
import logging
import re

logger = logging.getLogger(__name__)

MAX_RECENT_PROMPTS = 5


def create_initial_state() -> dict:
    """Create a fresh agent state for a new chat."""
    return {
        "initial_request": "",
        "recent_prompts": [],
        "direction": "",
        "search_context": {
            "recent_queries": [],
            "active_concepts": [],
        },
    }


def extract_initial_request(chat_id: str) -> str:
    """Get the first user message from a chat for the initial_request field."""
    from src.models import chat as chat_model

    messages = chat_model.get_messages(chat_id)
    for msg in messages:
        if msg["role"] == "user":
            return msg["content"][:500]
    return ""


def extract_prompt_blocks(response_text: str) -> list[str]:
    """Parse ```prompt fenced code blocks from agent response text."""
    pattern = r'```prompt\s*\n(.*?)```'
    matches = re.findall(pattern, response_text, re.DOTALL)
    return [m.strip() for m in matches[-MAX_RECENT_PROMPTS:]]


def extract_search_context(tool_call_log: list[dict]) -> dict:
    """Extract search queries and explored concepts from tool call log."""
    queries = []
    concepts = set()

    for call in tool_call_log:
        tool = call.get("tool", "")
        args = call.get("args", {})
        result = call.get("result", {})

        # Capture queries from search tools
        if tool in ("query_diverse_prompts", "search_similar_prompts", "query_dataset_map"):
            q = args.get("query", "")
            if q:
                queries.append(q)

        # Capture concepts from folder theme exploration
        if tool == "get_folder_themes":
            folder = args.get("folder_name", "")
            if folder:
                concepts.add(folder)

        # Capture concepts from search results
        for p in result.get("prompts", [])[:5]:
            concept = p.get("concept", "")
            if concept:
                concepts.add(concept)

        # Capture concepts from dataset map results
        for f in result.get("folders", [])[:5]:
            name = f.get("name", "")
            if name:
                concepts.add(name)

    return {
        "recent_queries": queries[-3:],
        "active_concepts": sorted(concepts)[:10],
    }


def update_direction(
    previous_direction: str,
    user_message: str,
    outcomes_text: str | None,
    tool_summary: str | None,
) -> str:
    """Update the direction field using the configured summary model.

    Makes a single cheap LLM call to produce a 1-2 sentence summary of what
    the user wants and doesn't want, incorporating deletion feedback and
    search activity.
    """
    from src.services import llm_service
    from src.models.settings import get_setting
    from src.config import DEFAULT_MODEL_SUMMARY

    model = get_setting("model_summary") or DEFAULT_MODEL_SUMMARY

    prompt = (
        "Update this conversation direction summary in 1-2 sentences.\n\n"
        f"Previous direction: {previous_direction or '(none yet)'}\n"
        f"User's latest message: {user_message}\n"
    )
    if outcomes_text:
        prompt += f"Generation feedback: {outcomes_text}\n"
    if tool_summary:
        prompt += f"Agent searched for: {tool_summary}\n"
    prompt += (
        "\nRules:\n"
        "- Capture what the user wants and doesn't want\n"
        "- Note quality/direction feedback from image deletions\n"
        "- Note specific preferences (lighting, pose, style, etc.)\n"
        "- Keep under 200 characters\n"
        "- Output ONLY the summary, nothing else"
    )

    try:
        messages = [{"role": "user", "content": prompt}]
        response = llm_service.generate(model=model, messages=messages)
        text = response.text if response and response.text else ""
        return text.strip()[:250] if text else previous_direction
    except Exception as e:
        logger.warning("Failed to update direction via LLM: %s", e)
        return previous_direction


def build_tool_summary(tool_call_log: list[dict]) -> str | None:
    """Build a brief summary of tool activity for the direction updater."""
    search_queries = []
    gen_count = 0
    for call in tool_call_log:
        tool = call.get("tool", "")
        if tool in ("query_diverse_prompts", "search_similar_prompts"):
            q = call.get("args", {}).get("query", "")
            if q:
                search_queries.append(q[:60])
        elif tool == "generate_image":
            gen_count += 1

    parts = []
    if search_queries:
        parts.append(f"Searched: {'; '.join(search_queries[:3])}")
    if gen_count:
        parts.append(f"Generated {gen_count} image(s)")
    return "; ".join(parts) if parts else None


def build_outcomes_text(chat_id: str) -> str | None:
    """Build a brief outcomes summary for the direction updater."""
    try:
        from src.agent.tools.generation_tools import _get_generation_outcomes
        from src.models import chat as chat_model

        db_messages = chat_model.get_messages(chat_id)
        last_assistant_id = None
        for msg in reversed(db_messages):
            if msg["role"] == "assistant":
                last_assistant_id = msg["id"]
                break

        if not last_assistant_id:
            return None

        result = _get_generation_outcomes(
            {"message_id": last_assistant_id},
            {"chat_id": chat_id},
        )
        if "error" in result or not result.get("outcomes"):
            return None

        parts = []
        for o in result["outcomes"]:
            p = f"{o['kept']} kept, {o['deleted']} deleted"
            if o.get("deletion_reasons"):
                reasons = ", ".join(f"{v} {k}" for k, v in o["deletion_reasons"].items())
                p += f" ({reasons})"
            parts.append(p)
        return "; ".join(parts)
    except Exception:
        return None


def apply_implicit_state_update(
    state: dict,
    chat_id: str,
    user_message: str,
    full_response: str,
    tool_call_log: list[dict],
) -> dict:
    """Update state implicitly from the completed agent turn.

    Called after the agent loop finishes, before saving the response.
    """
    # Set initial_request on first turn
    if not state.get("initial_request"):
        state["initial_request"] = extract_initial_request(chat_id)

    # Extract prompts from agent's response
    prompts = extract_prompt_blocks(full_response)
    if prompts:
        state["recent_prompts"] = prompts

    # Extract search context from tool calls
    if tool_call_log:
        new_search = extract_search_context(tool_call_log)
        existing = state.get("search_context", {})

        # Merge queries (keep last 3)
        all_queries = existing.get("recent_queries", []) + new_search["recent_queries"]
        merged_concepts = set(existing.get("active_concepts", [])) | set(new_search["active_concepts"])

        state["search_context"] = {
            "recent_queries": all_queries[-3:],
            "active_concepts": sorted(merged_concepts)[:10],
        }

    # Update direction via summary model
    outcomes_text = build_outcomes_text(chat_id)
    tool_summary = build_tool_summary(tool_call_log)
    # Only call LLM if there's something new to incorporate
    if user_message or outcomes_text or tool_summary:
        state["direction"] = update_direction(
            state.get("direction", ""),
            user_message,
            outcomes_text,
            tool_summary,
        )

    return state
