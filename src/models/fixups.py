"""One-time data fixup functions run during database initialization or startup."""

import os
import logging

from sqlalchemy import text

from src.models.database import get_db

logger = logging.getLogger(__name__)


def fix_truncated_output_folders(conn):
    """Fix output_folder values that were truncated to just the leaf directory.

    Scanned images in nested directories like ``character/subdir`` had
    output_folder set to ``"subdir"`` instead of ``"character/subdir"``.
    Recompute from file_path relative to the configured output root.
    """
    # Get output root directories
    result = conn.execute(
        text("SELECT path FROM data_directories WHERE dir_type = 'output' AND active = 1")
    )
    output_roots = [os.path.normpath(row[0]) for row in result.fetchall()]
    if not output_roots:
        return

    # Find all images with file_path set and a generation_settings record
    result = conn.execute(
        text("""SELECT gs.job_id, gs.output_folder, gi.file_path
           FROM generation_settings gs
           JOIN generated_images gi ON gs.job_id = gi.job_id
           WHERE gi.file_path IS NOT NULL""")
    )
    rows = result.fetchall()

    fixed = 0
    for row in rows:
        job_id = row[0]
        stored_folder = row[1] or ""
        filepath = row[2]

        dirpath = os.path.normpath(os.path.dirname(filepath))
        correct_folder = None
        for root in output_roots:
            if dirpath.startswith(root):
                rel = os.path.relpath(dirpath, root)
                correct_folder = rel if rel != "." else ""
                break

        if correct_folder is not None and correct_folder != stored_folder:
            conn.execute(
                text("UPDATE generation_settings SET output_folder = :folder WHERE job_id = :job_id"),
                {"folder": correct_folder, "job_id": job_id},
            )
            fixed += 1

    if fixed:
        logger.info(f"Fixed {fixed} truncated output_folder values")


def fix_subfolder_concepts():
    """Fix ChromaDB documents where concept is a subfolder instead of a base concept.

    Two sources of bad concepts:
    1. Slash paths (e.g. ``"character/scenes"``) -- generation_controller used
       the full output_folder as the concept.
    2. Bare subfolder names (e.g. ``"scenes"``) -- files in user-created
       subfolders at the output root level got the subfolder as the concept.

    For slash paths, the fix is ``concept.split("/")[0]``.
    For bare subfolders, we determine the correct parent from:
    - File-path IDs: derive parent from the path relative to the output root.
    - ``gen_*`` IDs: look up ``output_folder`` in ``generation_settings``.

    Also cleans up orphan cluster/assignment rows in SQLite.
    """
    try:
        from src.models import vector_store
        if vector_store._generated_collection is None:
            return

        col = vector_store._generated_collection
        count = col.count()
        if count == 0:
            return

        # Build set of valid base-level concepts from registered output dirs
        valid_base_concepts = set()
        output_roots = []
        with get_db() as conn:
            result = conn.execute(
                text("SELECT path FROM data_directories WHERE dir_type = 'output' AND active = 1")
            )
            output_roots = [os.path.normpath(row[0]) for row in result.fetchall()]

        for root in output_roots:
            if os.path.isdir(root):
                for name in os.listdir(root):
                    if os.path.isdir(os.path.join(root, name)):
                        valid_base_concepts.add(name)

        if not valid_base_concepts:
            return

        # Build a gen_job_id -> output_folder lookup for gen_* docs
        gen_output_folders = {}
        with get_db() as conn:
            result = conn.execute(text("SELECT job_id, output_folder FROM generation_settings"))
            for row in result.fetchall():
                gen_output_folders[row[0]] = row[1] or ""

        # Scan all generated docs for bad concepts
        all_data = col.get(limit=count, include=["metadatas"])
        fix_ids = []
        fix_metadatas = []
        delete_ids = []
        fixed_concepts = set()

        for doc_id, meta in zip(all_data["ids"], all_data["metadatas"]):
            concept = meta.get("concept", "")

            # Case 1: slash in concept -- always fixable
            if "/" in concept:
                correct = concept.split("/")[0]
                fixed_meta = dict(meta)
                fixed_meta["concept"] = correct
                fix_ids.append(doc_id)
                fix_metadatas.append(fixed_meta)
                fixed_concepts.add(concept)
                continue

            # Case 2: concept not in valid base set -- it's a subfolder name
            if concept and concept not in valid_base_concepts:
                correct = None

                # For file-path IDs, derive from path
                if not doc_id.startswith("gen_"):
                    norm_path = os.path.normpath(doc_id)
                    for root in output_roots:
                        if norm_path.startswith(root):
                            rel = os.path.relpath(norm_path, root)
                            parts = rel.split(os.sep)
                            if len(parts) > 2:
                                # Nested under a base concept subfolder
                                correct = parts[0]
                            break

                # For gen_* IDs, look up output_folder
                else:
                    job_id = doc_id[4:]
                    output_folder = gen_output_folders.get(job_id, "")
                    if output_folder:
                        correct = output_folder.split("/")[0]

                if correct and correct in valid_base_concepts and correct != concept:
                    fixed_meta = dict(meta)
                    fixed_meta["concept"] = correct
                    fix_ids.append(doc_id)
                    fix_metadatas.append(fixed_meta)
                    fixed_concepts.add(concept)
                elif not correct or correct not in valid_base_concepts:
                    # Orphan document -- folder gone and no parent derivable
                    delete_ids.append(doc_id)
                    fixed_concepts.add(concept)

        if fix_ids:
            chunk_size = 500
            for i in range(0, len(fix_ids), chunk_size):
                col.update(
                    ids=fix_ids[i:i + chunk_size],
                    metadatas=fix_metadatas[i:i + chunk_size],
                )
            logger.info(f"Fixed {len(fix_ids)} ChromaDB documents with subfolder concepts")

        if delete_ids:
            chunk_size = 500
            for i in range(0, len(delete_ids), chunk_size):
                col.delete(ids=delete_ids[i:i + chunk_size])
            logger.info(f"Deleted {len(delete_ids)} orphan ChromaDB documents with no valid parent concept")

        # Clean up orphan cluster rows in SQLite for the fixed concepts
        if fixed_concepts:
            with get_db() as conn:
                for bad_concept in fixed_concepts:
                    conn.execute(
                        text("DELETE FROM cluster_assignments WHERE cluster_id IN "
                             "(SELECT id FROM clusters WHERE folder_path = :fp)"),
                        {"fp": bad_concept},
                    )
                    conn.execute(
                        text("DELETE FROM clusters WHERE folder_path = :fp"),
                        {"fp": bad_concept},
                    )
            logger.info(f"Removed orphan cluster rows for {len(fixed_concepts)} subfolder concepts")

    except Exception as e:
        logger.warning(f"Could not fix subfolder concepts: {e}")
