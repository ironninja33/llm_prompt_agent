"""Background agent execution manager.

Decouples the agent loop from the SSE connection so that:
- The agent always runs to completion (saves response + summarizes)
- SSE clients can disconnect and reconnect without losing data
- Multiple tabs can read the same stream via the event queue
"""

import logging
import queue
import threading
from dataclasses import dataclass, field

from src.agent.loop import run_agent_turn

logger = logging.getLogger(__name__)


@dataclass
class AgentRun:
    chat_id: str
    events: queue.Queue = field(default_factory=queue.Queue)
    thread: threading.Thread = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    status: str = "running"  # running | done | error


# Active runs keyed by chat_id
_active_runs: dict[str, AgentRun] = {}
_lock = threading.Lock()


def start_run(
    chat_id: str,
    content: str,
    attachments: list | None = None,
    attachment_urls: list[str] | None = None,
) -> AgentRun:
    """Spawn a daemon thread that runs run_agent_turn() to completion.

    Events are put on a Queue for any SSE consumer to read.
    If no consumer is reading, the agent still finishes and saves.
    """
    run = AgentRun(chat_id=chat_id)

    def _run():
        try:
            for event in run_agent_turn(
                chat_id, content,
                attachments=attachments,
                attachment_urls=attachment_urls,
                cancel_event=run.cancel_event,
            ):
                run.events.put(event)
        except Exception as e:
            logger.error("Agent runner error for chat %s: %s", chat_id, e, exc_info=True)
            run.events.put({"type": "error", "message": f"Internal error: {str(e)}"})
            run.status = "error"
        finally:
            # Sentinel signals completion
            run.events.put(None)
            if run.status == "running":
                run.status = "done"
            with _lock:
                _active_runs.pop(chat_id, None)

    thread = threading.Thread(target=_run, daemon=True)
    run.thread = thread

    with _lock:
        # Cancel any existing run for this chat
        existing = _active_runs.get(chat_id)
        if existing:
            existing.cancel_event.set()
        _active_runs[chat_id] = run

    thread.start()
    return run


def get_run(chat_id: str) -> AgentRun | None:
    """Get the active run for a chat, if any."""
    with _lock:
        return _active_runs.get(chat_id)


def has_active_runs() -> bool:
    """Check if any agent runs are currently active."""
    with _lock:
        return len(_active_runs) > 0


def cancel_run(chat_id: str):
    """Signal cancellation for a chat's active run."""
    with _lock:
        run = _active_runs.get(chat_id)
        if run:
            run.cancel_event.set()
