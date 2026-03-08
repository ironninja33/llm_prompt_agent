"""Build LLM prompt strings from loaded data using user-editable templates."""

from __future__ import annotations

import os
from pathlib import Path

from ..data_loader import CrossFolderInput, FolderInput

PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_template(path: str | None, default_name: str) -> str:
    """Load a template file, falling back to the bundled default."""
    if path and os.path.isfile(path):
        return Path(path).read_text(encoding="utf-8").strip()
    default_path = PROMPTS_DIR / default_name
    return default_path.read_text(encoding="utf-8").strip()


def build_cross_folder_prompt(
    item: CrossFolderInput,
    template_path: str | None = None,
) -> str:
    """Render a cross-folder cluster prompt from the template."""
    template = _load_template(template_path, "cross_folder.txt")

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


def _load_concept_prompt(folder_path: str) -> str:
    """Load a concept-specific prompt if one exists for the folder's prefix."""
    if "__" in folder_path:
        concept = folder_path.split("__", 1)[0]
        concept_path = PROMPTS_DIR / f"prompt__{concept}.txt"
        if concept_path.is_file():
            return concept_path.read_text(encoding="utf-8").strip()
    return ""


def build_folder_prompt(
    item: FolderInput,
    template_path: str | None = None,
) -> str:
    """Render a folder summary prompt from the template."""
    template = _load_template(template_path, "folder.txt")

    name = Path(item.folder_path).name if "/" in item.folder_path or "\\" in item.folder_path else item.folder_path
    folder_label = name.split("__", 1)[1] if "__" in name else name
    sample_lines = "\n".join(f"'{p}'" for p in item.sample_prompts)
    # Only load and inject concept_prompt if the template uses it
    values: dict[str, str] = {
        "folder_label": folder_label,
        "concept": item.concept,
        "sample_prompts": sample_lines,
    }

    has_concept_prompt_placeholder = "{concept_prompt}" in template
    if has_concept_prompt_placeholder:
        concept_prompt = _load_concept_prompt(item.folder_path)
        values["concept_prompt"] = concept_prompt

    result = template.format_map(values)

    # Remove blank line left behind when no concept prompt was injected
    if has_concept_prompt_placeholder and not concept_prompt:
        result = "\n".join(line for line in result.splitlines() if line.strip())
    return result
