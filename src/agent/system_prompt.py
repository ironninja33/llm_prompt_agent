"""System prompt loader — reads the default prompt from a markdown file."""

import os

_PROMPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_PROMPT_PATH = os.path.join(_PROMPT_DIR, "default_system_prompt.md")


def load_default_system_prompt() -> str:
    """Read the default system prompt from the markdown file."""
    with open(_DEFAULT_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


# Constant available at import time for database seeding
DEFAULT_SYSTEM_PROMPT = load_default_system_prompt()
