"""Main summarizer orchestrator — loads data, builds prompts, runs LLM, writes results."""

from __future__ import annotations

import logging
from typing import Callable

from sqlalchemy import text

from src.models import settings
from src.models.database import get_db

from . import engine
from .data_loader import (
    CrossFolderInput,
    FolderInput,
    load_cross_folder_inputs,
    load_folder_inputs,
)
from .prompts import (
    build_cross_folder_prompt,
    build_folder_prompt,
    build_intra_cluster_prompt,
    load_system_prompt,
)

logger = logging.getLogger(__name__)


def _post_process(raw: str, max_tags: int, max_words: int) -> str:
    """Clean up a raw LLM summary: limit tags and words per tag."""
    tags = [t.strip() for t in raw.split(",")][:max_tags]
    return ", ".join(" ".join(t.split()[:max_words]) for t in tags if t)


def run_summarization(progress_callback: Callable[[str], None] | None = None):
    """Run LLM summarization on all clusters and folders. Called after clustering."""
    max_tags = int(settings.get_setting("summarizer_max_tags") or 5)
    max_words = int(settings.get_setting("summarizer_max_words") or 12)
    model_name = settings.get_setting("summarizer_model") or "Qwen/Qwen3-4B-FP8"

    if progress_callback:
        progress_callback("Loading data for summarization...")

    # 1. Load data
    cross_inputs = load_cross_folder_inputs()
    folder_inputs = load_folder_inputs()

    if not cross_inputs and not folder_inputs:
        logger.warning("No data to summarize.")
        return

    # 2. Build all prompts
    system_prompt = load_system_prompt()
    all_prompts: list[str] = []
    prompt_map: list[tuple[str, int | str]] = []  # (type, id)

    for ci in cross_inputs:
        all_prompts.append(build_cross_folder_prompt(ci))
        prompt_map.append(("cross", ci.cluster_id))

    for fi in folder_inputs:
        all_prompts.append(build_folder_prompt(fi))
        prompt_map.append(("folder", fi.folder_path))
        for ic in fi.intra_clusters:
            all_prompts.append(build_intra_cluster_prompt(ic, fi))
            prompt_map.append(("intra", ic.cluster_id))

    if progress_callback:
        progress_callback(f"Built {len(all_prompts)} prompts. Loading model...")

    logger.info(f"Summarization: {len(all_prompts)} prompts "
                f"({sum(1 for t, _ in prompt_map if t == 'cross')} cross, "
                f"{sum(1 for t, _ in prompt_map if t == 'folder')} folder, "
                f"{sum(1 for t, _ in prompt_map if t == 'intra')} intra)")

    # 3. Load model, batch generate, unload
    llm = engine.load_model(model_name)
    try:
        if progress_callback:
            progress_callback(f"Running inference on {len(all_prompts)} prompts...")
        clean_results, _ = engine.batch_generate(llm, all_prompts, system_prompt, no_think=True)
    finally:
        engine.unload_model(llm)

    if progress_callback:
        progress_callback("Writing summaries to database...")

    # 4. Post-process and write to DB
    _write_results(clean_results, prompt_map, max_tags, max_words)

    logger.info("Summarization complete.")


def run_folder_summarization(folder_path: str, source_type: str | None = None):
    """Summarize a single folder after recluster."""
    max_tags = int(settings.get_setting("summarizer_max_tags") or 5)
    max_words = int(settings.get_setting("summarizer_max_words") or 12)
    model_name = settings.get_setting("summarizer_model") or "Qwen/Qwen3-4B-FP8"

    folder_inputs = load_folder_inputs(
        folder_path=folder_path,
        source_type=source_type,
    )

    if not folder_inputs:
        logger.warning(f"No folder data found for '{folder_path}'.")
        return

    system_prompt = load_system_prompt()
    all_prompts: list[str] = []
    prompt_map: list[tuple[str, int | str]] = []

    for fi in folder_inputs:
        all_prompts.append(build_folder_prompt(fi))
        prompt_map.append(("folder", fi.folder_path))
        for ic in fi.intra_clusters:
            all_prompts.append(build_intra_cluster_prompt(ic, fi))
            prompt_map.append(("intra", ic.cluster_id))

    if not all_prompts:
        return

    logger.info(f"Folder summarization for '{folder_path}': {len(all_prompts)} prompts")

    llm = engine.load_model(model_name)
    try:
        clean_results, _ = engine.batch_generate(llm, all_prompts, system_prompt, no_think=True)
    finally:
        engine.unload_model(llm)

    _write_results(clean_results, prompt_map, max_tags, max_words)

    logger.info(f"Folder summarization complete for '{folder_path}'.")


def _write_results(
    clean_results: list[str],
    prompt_map: list[tuple[str, int | str]],
    max_tags: int,
    max_words: int,
):
    """Post-process summaries and write to database."""
    with get_db() as conn:
        for i, summary in enumerate(clean_results):
            cleaned = _post_process(summary, max_tags, max_words)
            type_, id_ = prompt_map[i]

            if type_ == "cross":
                conn.execute(
                    text("UPDATE clusters SET label = :label WHERE id = :id"),
                    {"label": cleaned, "id": id_},
                )
            elif type_ == "folder":
                conn.execute(
                    text("INSERT OR REPLACE INTO folder_summaries (folder_path, summary, updated_at) "
                         "VALUES (:fp, :s, CURRENT_TIMESTAMP)"),
                    {"fp": id_, "s": cleaned},
                )
            elif type_ == "intra":
                conn.execute(
                    text("UPDATE clusters SET label = :label WHERE id = :id"),
                    {"label": cleaned, "id": id_},
                )
