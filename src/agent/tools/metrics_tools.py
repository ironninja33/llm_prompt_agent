"""Metrics-based tools — deletion insights and successful generation patterns."""

import logging

from sqlalchemy import text

from src.models.database import get_db

logger = logging.getLogger(__name__)


def _get_deletion_insights(args: dict) -> dict:
    """Get patterns from deleted prompts to help avoid past failures."""
    output_folder = args.get("output_folder")
    concept = args.get("concept")
    k = args.get("k", 5)

    # Query deletion_log for quality/wrong_direction deletions
    conditions = ["reason IN ('quality', 'wrong_direction')"]
    params: dict = {"k": k}

    if output_folder:
        conditions.append("output_folder = :folder")
        params["folder"] = output_folder

    where = " AND ".join(conditions)

    with get_db() as conn:
        rows = conn.execute(
            text(f"""SELECT positive_prompt, reason, COUNT(*) as del_count,
                          AVG(lineage_depth) as avg_depth
                   FROM deletion_log
                   WHERE {where} AND positive_prompt IS NOT NULL
                   GROUP BY positive_prompt, reason
                   ORDER BY del_count DESC
                   LIMIT :k"""),
            params,
        ).fetchall()

    deleted_prompts = [
        {
            "prompt": row._mapping["positive_prompt"][:200],
            "reason": row._mapping["reason"],
            "times_deleted": row._mapping["del_count"],
            "avg_lineage_depth": round(row._mapping["avg_depth"] or 0, 1),
        }
        for row in rows
    ]

    # Optionally search graveyard embeddings by concept similarity
    graveyard_matches = []
    if concept:
        try:
            from src.services import embedding_service
            from src.models import vector_store

            query_emb = embedding_service.embed(concept)
            graveyard_matches = vector_store.search_graveyard(query_emb, k=k)
        except Exception as e:
            logger.warning("Graveyard search failed: %s", e)

    # Summary counts
    quality_count = sum(1 for r in rows if r._mapping["reason"] == "quality")
    wrong_dir_count = sum(1 for r in rows if r._mapping["reason"] == "wrong_direction")

    return {
        "count": len(deleted_prompts),
        "deleted_prompts": deleted_prompts,
        "graveyard_matches": [
            {
                "text": r["document"][:200],
                "reason": r["metadata"].get("deletion_reason", ""),
            }
            for r in graveyard_matches
        ] if graveyard_matches else [],
        "summary": {
            "total_quality": quality_count,
            "total_wrong_direction": wrong_dir_count,
        },
    }


def _get_successful_patterns(args: dict) -> dict:
    """Get prompts that led to productive regeneration chains."""
    output_folder = args.get("output_folder")
    min_depth = args.get("min_depth", 3)
    k = args.get("k", 5)

    with get_db() as conn:
        # Find deep-chain leaf jobs and trace back to roots
        conditions = ["gj.lineage_depth >= :min_depth", "gj.status = 'completed'"]
        params: dict = {"min_depth": min_depth, "k": k * 3}

        if output_folder:
            conditions.append("gs.output_folder = :folder")
            params["folder"] = output_folder

        where = " AND ".join(conditions)

        leaves = conn.execute(
            text(f"""SELECT gj.id, gj.parent_job_id, gj.lineage_depth,
                          gs.positive_prompt, gs.output_folder
                   FROM generation_jobs gj
                   JOIN generation_settings gs ON gs.job_id = gj.id
                   WHERE {where}
                   ORDER BY gj.lineage_depth DESC
                   LIMIT :k"""),
            params,
        ).fetchall()

        # For each leaf, walk parent chain to find root prompt
        patterns = []
        seen_roots: set[str] = set()

        for leaf in leaves:
            lm = leaf._mapping
            current_id = lm["parent_job_id"]
            root_prompt = lm["positive_prompt"]
            depth = 0

            while current_id and depth < 15:
                parent = conn.execute(
                    text("""SELECT gj.parent_job_id, gs.positive_prompt
                           FROM generation_jobs gj
                           JOIN generation_settings gs ON gs.job_id = gj.id
                           WHERE gj.id = :id"""),
                    {"id": current_id},
                ).fetchone()
                if not parent:
                    break
                root_prompt = parent._mapping["positive_prompt"]
                current_id = parent._mapping["parent_job_id"]
                depth += 1

            # Deduplicate by root prompt
            root_key = (root_prompt or "")[:100]
            if root_key in seen_roots:
                continue
            seen_roots.add(root_key)

            patterns.append({
                "root_prompt": (root_prompt or "")[:300],
                "leaf_prompt": (lm["positive_prompt"] or "")[:300],
                "max_depth": lm["lineage_depth"],
                "output_folder": lm["output_folder"] or "",
            })

            if len(patterns) >= k:
                break

    return {"count": len(patterns), "patterns": patterns}
