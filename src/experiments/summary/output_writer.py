"""Format and write summary output text files."""

from __future__ import annotations

import os
from pathlib import Path

from .data_loader import CrossFolderInput, FolderInput

SEPARATOR = "=" * 80


def _format_folder_label(folder_path: str) -> str:
    """Convert 'category__label' folder path to 'label (category)' display format."""
    if "__" in folder_path:
        category, label = folder_path.split("__", 1)
        return f"{label} ({category})"
    return folder_path


def _format_contributing(folders: list[dict]) -> str:
    """Format contributing folders, collapsing duplicates after normalization."""
    seen: dict[str, None] = {}
    for f in folders:
        label = _format_folder_label(f["folder_path"])
        seen[label] = None
    return ", ".join(seen) if seen else "N/A"


def write_cross_folder_summaries(
    inputs: list[CrossFolderInput],
    summaries: list[str],
    output_dir: str,
    summary_label: str = "LLM Summary",
) -> str:
    """Write cross-folder summaries to a text file.

    Returns:
        Path to the written file.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "cross_folder_summaries.txt")

    lines = []
    for item, summary in zip(inputs, summaries):
        contributing = _format_contributing(item.contributing_folders)
        lines.append(SEPARATOR)
        lines.append(f"Cross-Folder Cluster #{item.cluster_id}")
        lines.append(f"TF-IDF Label: {item.tfidf_label}")
        lines.append(f"{summary_label}:  {summary}")
        lines.append(f"Contributing Folders: {contributing}")
        lines.append(SEPARATOR)
        lines.append("")

    Path(path).write_text("\n".join(lines), encoding="utf-8")
    return path


def write_cross_folder_summaries_raw(
    inputs: list[CrossFolderInput],
    summaries_raw: list[str],
    output_dir: str,
    summary_label: str = "LLM Summary (raw)",
) -> str:
    """Write cross-folder summaries with raw model output (including thinking tokens).

    Returns:
        Path to the written file.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "cross_folder_summaries_raw.txt")

    lines = []
    for item, summary in zip(inputs, summaries_raw):
        contributing = _format_contributing(item.contributing_folders)
        lines.append(SEPARATOR)
        lines.append(f"Cross-Folder Cluster #{item.cluster_id}")
        lines.append(f"TF-IDF Label: {item.tfidf_label}")
        lines.append(f"{summary_label}:  {summary}")
        lines.append(f"Contributing Folders: {contributing}")
        lines.append(SEPARATOR)
        lines.append("")

    Path(path).write_text("\n".join(lines), encoding="utf-8")
    return path


def write_folder_summaries(
    inputs: list[FolderInput],
    summaries: list[str],
    output_dir: str,
    summary_label: str = "LLM Summary",
    cluster_summaries: list[list[tuple[str, str]]] | None = None,
) -> str:
    """Write folder summaries to a text file.

    Returns:
        Path to the written file.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "folder_summaries.txt")

    lines = []
    for i, (item, summary) in enumerate(zip(inputs, summaries)):
        lines.append(SEPARATOR)
        lines.append(f"Folder: {item.folder_path}")
        lines.append(f"Data Root: {item.data_root or 'unknown'}")
        lines.append(f"Concept: {item.concept}")
        lines.append(f"TF-IDF Summary: {item.tfidf_summary}")
        lines.append(f"{summary_label}:    {summary}")
        if cluster_summaries and i < len(cluster_summaries) and cluster_summaries[i]:
            lines.append("  Clusters:")
            for label, cl_summary in cluster_summaries[i]:
                lines.append(f"    \u2022 [{label}]: {cl_summary}")
        lines.append(SEPARATOR)
        lines.append("")

    Path(path).write_text("\n".join(lines), encoding="utf-8")
    return path


def write_folder_summaries_raw(
    inputs: list[FolderInput],
    summaries_raw: list[str],
    output_dir: str,
    summary_label: str = "LLM Summary (raw)",
    cluster_summaries: list[list[tuple[str, str]]] | None = None,
) -> str:
    """Write folder summaries with raw model output (including thinking tokens).

    Returns:
        Path to the written file.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "folder_summaries_raw.txt")

    lines = []
    for i, (item, summary) in enumerate(zip(inputs, summaries_raw)):
        lines.append(SEPARATOR)
        lines.append(f"Folder: {item.folder_path}")
        lines.append(f"Data Root: {item.data_root or 'unknown'}")
        lines.append(f"Concept: {item.concept}")
        lines.append(f"TF-IDF Summary: {item.tfidf_summary}")
        lines.append(f"{summary_label}:    {summary}")
        if cluster_summaries and i < len(cluster_summaries) and cluster_summaries[i]:
            lines.append("  Clusters:")
            for label, cl_summary in cluster_summaries[i]:
                lines.append(f"    \u2022 [{label}]: {cl_summary}")
        lines.append(SEPARATOR)
        lines.append("")

    Path(path).write_text("\n".join(lines), encoding="utf-8")
    return path
