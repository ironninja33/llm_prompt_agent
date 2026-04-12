"""Fetch deleted and surviving prompts, and folder-level deletion rates."""

import logging
from dataclasses import dataclass

from sqlalchemy import text

from src.models.database import get_db

logger = logging.getLogger(__name__)


@dataclass
class DeletedPrompt:
    """A prompt that was deleted for a tracked reason."""
    job_id: str
    positive_prompt: str
    output_folder: str | None
    reason: str


def fetch_deleted_prompts(base_model: str | None, reasons: list[str]) -> list[DeletedPrompt]:
    """Query deletion_log filtered by reasons, optionally filtered by base_model.

    When base_model is provided, uses a LEFT JOIN with generation_settings to filter
    where possible. Deleted jobs whose generation_settings were CASCADE-deleted are
    included (they cannot be excluded by model since the info is gone).

    Deduplicates by prompt text (keeps first occurrence).
    """
    reason_placeholders = ", ".join(f":r{i}" for i in range(len(reasons)))
    params: dict = {f"r{i}": r for i, r in enumerate(reasons)}

    if base_model:
        params["base_model"] = base_model
        query = f"""
            SELECT dl.job_id, dl.positive_prompt, dl.output_folder, dl.reason
            FROM deletion_log dl
            LEFT JOIN generation_settings gs ON gs.job_id = dl.job_id
            WHERE dl.reason IN ({reason_placeholders})
              AND (gs.base_model = :base_model OR gs.job_id IS NULL)
              AND dl.positive_prompt IS NOT NULL
              AND dl.positive_prompt != ''
            ORDER BY dl.deleted_at DESC
        """
    else:
        query = f"""
            SELECT dl.job_id, dl.positive_prompt, dl.output_folder, dl.reason
            FROM deletion_log dl
            WHERE dl.reason IN ({reason_placeholders})
              AND dl.positive_prompt IS NOT NULL
              AND dl.positive_prompt != ''
            ORDER BY dl.deleted_at DESC
        """

    with get_db() as conn:
        rows = conn.execute(text(query), params).fetchall()

    seen_prompts: set[str] = set()
    results: list[DeletedPrompt] = []
    for row in rows:
        m = row._mapping
        prompt_text = m["positive_prompt"]
        if prompt_text not in seen_prompts:
            seen_prompts.add(prompt_text)
            results.append(DeletedPrompt(
                job_id=m["job_id"],
                positive_prompt=prompt_text,
                output_folder=m["output_folder"],
                reason=m["reason"],
            ))
    return results


def fetch_surviving_prompts(base_model: str | None) -> list[str]:
    """Fetch distinct positive_prompt values from completed, non-deleted jobs.

    These are the "good" prompts — jobs that completed and whose images were
    NOT deleted for any reason.
    """
    conditions = [
        "gj.status = 'completed'",
        "gs.positive_prompt IS NOT NULL",
        "gs.positive_prompt != ''",
        "gs.job_id NOT IN (SELECT job_id FROM deletion_log)",
    ]
    params: dict = {}

    if base_model:
        conditions.append("gs.base_model = :base_model")
        params["base_model"] = base_model

    where = " AND ".join(conditions)
    query = f"""
        SELECT DISTINCT gs.positive_prompt
        FROM generation_settings gs
        JOIN generation_jobs gj ON gs.job_id = gj.id
        WHERE {where}
    """

    with get_db() as conn:
        rows = conn.execute(text(query), params).fetchall()

    return [row._mapping["positive_prompt"] for row in rows]


def fetch_folder_deletion_rates(
    base_model: str | None, reasons: list[str]
) -> list[dict]:
    """Compute per-folder deletion rates.

    Returns list of dicts sorted by deletion_rate DESC:
        {folder, total_deleted, total_surviving, total_generated, deletion_rate}

    Deleted jobs have their generation_settings CASCADE-deleted, so we count
    deletions from deletion_log and surviving jobs from generation_settings
    separately, then combine by folder.
    """
    reason_placeholders = ", ".join(f":r{i}" for i in range(len(reasons)))
    params: dict = {f"r{i}": r for i, r in enumerate(reasons)}

    # Count deletions per folder from deletion_log
    del_query = f"""
        SELECT output_folder AS folder, COUNT(DISTINCT job_id) AS del_count
        FROM deletion_log
        WHERE reason IN ({reason_placeholders})
          AND output_folder IS NOT NULL
        GROUP BY output_folder
    """

    # Count surviving jobs per folder from generation_settings
    surv_conditions = [
        "gj.status = 'completed'",
        "gs.output_folder IS NOT NULL",
    ]
    surv_params: dict = {}
    if base_model:
        surv_conditions.append("gs.base_model = :base_model")
        surv_params["base_model"] = base_model

    surv_where = " AND ".join(surv_conditions)
    surv_query = f"""
        SELECT gs.output_folder AS folder, COUNT(DISTINCT gs.job_id) AS surv_count
        FROM generation_settings gs
        JOIN generation_jobs gj ON gs.job_id = gj.id
        WHERE {surv_where}
        GROUP BY gs.output_folder
    """

    with get_db() as conn:
        del_rows = conn.execute(text(del_query), params).fetchall()
        surv_rows = conn.execute(text(surv_query), surv_params).fetchall()

    del_map = {row._mapping["folder"]: row._mapping["del_count"] for row in del_rows}
    surv_map = {row._mapping["folder"]: row._mapping["surv_count"] for row in surv_rows}

    all_folders = set(del_map.keys()) | set(surv_map.keys())
    results = []
    for folder in all_folders:
        deleted = del_map.get(folder, 0)
        surviving = surv_map.get(folder, 0)
        if deleted == 0:
            continue
        total = deleted + surviving
        results.append({
            "folder": folder,
            "total_deleted": deleted,
            "total_surviving": surviving,
            "total_generated": total,
            "deletion_rate": round(deleted / total, 4) if total > 0 else 0,
        })

    results.sort(key=lambda r: r["deletion_rate"], reverse=True)
    return results
