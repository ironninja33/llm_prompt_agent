"""Settings controller — business logic for settings management."""

import logging
from src.models import settings, vector_store
from src.services import llm_service
from src.agent.system_prompt import load_default_system_prompt

logger = logging.getLogger(__name__)


def get_all_settings() -> dict:
    """Get all settings. Masks the API key for security."""
    all_settings = settings.get_all_settings()
    # Mask API key
    if "gemini_api_key" in all_settings and all_settings["gemini_api_key"]:
        key = all_settings["gemini_api_key"]
        all_settings["gemini_api_key_masked"] = key[:8] + "..." + key[-4:] if len(key) > 12 else "***"
        all_settings["gemini_api_key_set"] = True
    else:
        all_settings["gemini_api_key_masked"] = ""
        all_settings["gemini_api_key_set"] = False
    # Don't send the raw key
    all_settings.pop("gemini_api_key", None)
    return all_settings


def update_settings(new_settings: dict):
    """Update settings. Re-initializes LLM client if API key changes."""
    # Handle API key separately
    api_key = new_settings.pop("gemini_api_key", None)
    if api_key is not None and api_key != "":
        settings.update_setting("gemini_api_key", api_key)
        llm_service.initialize(api_key)

    # Update other settings
    if new_settings:
        settings.update_settings(new_settings)


def reset_system_prompt() -> str:
    """Reset the system prompt to the default from the markdown file."""
    default_prompt = load_default_system_prompt()
    settings.update_setting("system_prompt", default_prompt)
    return default_prompt


def get_available_models() -> list[dict]:
    """List available Gemini models."""
    return llm_service.list_models()


def get_data_directories() -> list[dict]:
    """Get all data directories."""
    return settings.get_data_directories(active_only=False)


def add_data_directory(path: str, dir_type: str) -> dict:
    """Add a new data directory."""
    return settings.add_data_directory(path, dir_type)


def update_data_directory(dir_id: int, **kwargs) -> bool:
    """Update a data directory."""
    return settings.update_data_directory(dir_id, **kwargs)


def delete_data_directory(dir_id: int) -> bool:
    """Delete a data directory and purge its entries from the vector store."""
    # Look up the directory before deleting so we can clean up the vector store
    directory = settings.get_data_directory(dir_id)
    if directory is None:
        return False

    # Remove associated documents from ChromaDB
    dir_path = directory["path"]
    dir_type = directory["dir_type"]
    deleted_count = vector_store.delete_documents_by_directory(dir_path, dir_type)
    logger.info(
        f"Purged {deleted_count} vector store entries for directory "
        f"{dir_path!r} (type={dir_type})"
    )

    # Remove the directory record from SQLite
    return settings.delete_data_directory(dir_id)
