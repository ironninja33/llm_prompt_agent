"""Build LLM prompt strings from loaded data using templates.

Templates are loaded from settings DB first, falling back to bundled defaults
in ``src/agent/prompts/``.
"""

from __future__ import annotations

import os
from pathlib import Path

from src.models import settings

from .data_loader import CrossFolderInput, FolderInput

# Bundled defaults directory
PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "agent" / "prompts"
CONCEPT_OVERLAYS_DIR = PROMPTS_DIR / "summarizer_concept_overlays"


def _load_template_from_settings_or_file(setting_key: str, default_filename: str) -> str:
    """Load a template from settings DB, falling back to bundled default file."""
    value = settings.get_setting(setting_key)
    if value and value.strip():
        return value.strip()
    default_path = PROMPTS_DIR / default_filename
    if default_path.is_file():
        return default_path.read_text(encoding="utf-8").strip()
    return ""


def load_system_prompt() -> str:
    """Load summarizer system prompt from settings or default file."""
    return _load_template_from_settings_or_file(
        "summarizer_system_prompt", "summarizer_system_prompt.md"
    )


def load_cross_folder_template() -> str:
    """Load cross-folder prompt template."""
    return _load_template_from_settings_or_file(
        "summarizer_cross_folder_template", "summarizer_cross_folder.txt"
    )


def load_folder_template() -> str:
    """Load folder prompt template."""
    return _load_template_from_settings_or_file(
        "summarizer_folder_template", "summarizer_folder.txt"
    )


def load_intra_folder_template() -> str:
    """Load intra-folder prompt template."""
    return _load_template_from_settings_or_file(
        "summarizer_intra_folder_template", "summarizer_intra_folder.txt"
    )


def load_default_templates() -> dict[str, str]:
    """Load all bundled default templates (for the 'Reset to Defaults' button)."""
    defaults = {}
    for key, filename in [
        ("summarizer_system_prompt", "summarizer_system_prompt.md"),
        ("summarizer_cross_folder_template", "summarizer_cross_folder.txt"),
        ("summarizer_folder_template", "summarizer_folder.txt"),
        ("summarizer_intra_folder_template", "summarizer_intra_folder.txt"),
    ]:
        path = PROMPTS_DIR / filename
        if path.is_file():
            defaults[key] = path.read_text(encoding="utf-8").strip()
        else:
            defaults[key] = ""
    return defaults


def build_cross_folder_prompt(item: CrossFolderInput) -> str:
    """Render a cross-folder cluster prompt from the template."""
    template = load_cross_folder_template()

    contributing = ", ".join(
        f"{f['label']} ({f['folder_path']})"
        for f in item.contributing_folders
    )
    sample_lines = "\n".join(f"'{p}'" for p in item.sample_prompts)

    return template.format_map({
        "contributing_folders": contributing or "N/A",
        "sample_count": len(item.sample_prompts),
        "sample_prompts": sample_lines,
    })


def _load_concept_overlay(folder_path: str) -> str:
    """Load a concept-specific prompt overlay if one exists."""
    if "__" in folder_path:
        concept = folder_path.split("__", 1)[0]
        concept_path = CONCEPT_OVERLAYS_DIR / f"prompt__{concept}.txt"
        if concept_path.is_file():
            return concept_path.read_text(encoding="utf-8").strip()
    return ""


def build_folder_prompt(item: FolderInput) -> str:
    """Render a folder summary prompt from the template."""
    template = load_folder_template()

    name = Path(item.folder_path).name if "/" in item.folder_path or "\\" in item.folder_path else item.folder_path
    folder_label = name.split("__", 1)[1] if "__" in name else name
    sample_lines = "\n".join(f"'{p}'" for p in item.sample_prompts)

    values: dict[str, str] = {
        "folder_label": folder_label,
        "concept": item.concept,
        "sample_prompts": sample_lines,
    }

    has_concept_prompt_placeholder = "{concept_prompt}" in template
    concept_prompt = ""
    if has_concept_prompt_placeholder:
        concept_prompt = _load_concept_overlay(item.folder_path)
        values["concept_prompt"] = concept_prompt

    result = template.format_map(values)

    if has_concept_prompt_placeholder and not concept_prompt:
        result = "\n".join(line for line in result.splitlines() if line.strip())
    return result


def build_intra_cluster_prompt(cluster_info, folder_input: FolderInput) -> str:
    """Build a prompt for an intra-cluster summary using the intra-folder template."""
    template = load_intra_folder_template()

    name = Path(folder_input.folder_path).name if "/" in folder_input.folder_path or "\\" in folder_input.folder_path else folder_input.folder_path
    folder_label = name.split("__", 1)[1] if "__" in name else name

    sample_lines = "\n".join(f"'{p}'" for p in cluster_info.sample_prompts)
    diff_lines = "\n".join(f"'{p}'" for p in cluster_info.difference_prompts)

    return template.format_map({
        "folder_label": folder_label,
        "concept": folder_input.concept,
        "sample_prompts": sample_lines,
        "difference_prompts": diff_lines,
    })
