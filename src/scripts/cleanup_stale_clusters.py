"""One-time cleanup: remove training clusters whose assignments are all output-only.

These stale clusters occur when a concept only has data in the output collection
but ghost training cluster entries persist in SQLite.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sqlalchemy import text
from src.models.database import get_db


def cleanup_stale_training_clusters(dry_run: bool = False):
    with get_db() as conn:
        # Find training clusters where all assignments have source_type='output'
        stale = conn.execute(text(
            "SELECT c.id, c.folder_path FROM clusters c "
            "WHERE c.cluster_type = 'intra_folder' AND c.source_type = 'training' "
            "  AND NOT EXISTS ("
            "    SELECT 1 FROM cluster_assignments ca "
            "    WHERE ca.cluster_id = c.id AND ca.source_type = 'training'"
            "  ) "
            "  AND EXISTS ("
            "    SELECT 1 FROM cluster_assignments ca "
            "    WHERE ca.cluster_id = c.id AND ca.source_type = 'output'"
            "  )"
        )).fetchall()

        if not stale:
            print("No stale training clusters found.")
            return

        stale_ids = [row._mapping["id"] for row in stale]
        folders = sorted(set(row._mapping["folder_path"] for row in stale))

        print(f"Found {len(stale_ids)} stale training cluster(s) across {len(folders)} folder(s):")
        for f in folders:
            print(f"  - {f}")

        if dry_run:
            print("\nDry run — no changes made.")
            return

        # Delete assignments first, then clusters
        conn.execute(
            text("DELETE FROM cluster_assignments WHERE cluster_id IN "
                 f"({','.join(':id' + str(i) for i in range(len(stale_ids)))})"),
            {f"id{i}": sid for i, sid in enumerate(stale_ids)},
        )
        conn.execute(
            text("DELETE FROM clusters WHERE id IN "
                 f"({','.join(':id' + str(i) for i in range(len(stale_ids)))})"),
            {f"id{i}": sid for i, sid in enumerate(stale_ids)},
        )

        print(f"\nDeleted {len(stale_ids)} stale cluster(s) and their assignments.")

    # Verify no mismatches remain
    with get_db() as conn:
        mismatches = conn.execute(text(
            "SELECT COUNT(*) as cnt FROM clusters c "
            "JOIN cluster_assignments ca ON ca.cluster_id = c.id "
            "WHERE c.source_type != ca.source_type"
        )).scalar()
        print(f"Verification: {mismatches} source_type mismatch(es) remaining.")


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    cleanup_stale_training_clusters(dry_run=dry)
