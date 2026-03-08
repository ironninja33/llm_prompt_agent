"""Folder/concept rename logic — filesystem, SQLite, ChromaDB."""

import logging
import os

from sqlalchemy import text

from src.models import vector_store
from src.models.database import get_db

logger = logging.getLogger(__name__)


def rename_concept(old_concept: str, new_concept: str, parent_dirs: list[str]) -> dict:
    """Rename a concept folder on disk and update all DB/ChromaDB references."""
    # 1. Rename physical directories
    renamed_dirs = []
    for parent in parent_dirs:
        old_dir = os.path.join(parent, old_concept)
        new_dir = os.path.join(parent, new_concept)
        if not os.path.isdir(old_dir):
            continue
        try:
            os.rename(old_dir, new_dir)
            renamed_dirs.append((parent, old_dir, new_dir))
        except OSError as e:
            for _, rd_old, rd_new in renamed_dirs:
                try:
                    os.rename(rd_new, rd_old)
                except OSError:
                    pass
            return {"ok": False, "error": f"Failed to rename {old_dir}: {e}"}

    if not renamed_dirs:
        return {"ok": False, "error": f"No directories found for concept '{old_concept}'"}

    # 2. Update SQLite
    try:
        with get_db() as conn:
            _rename_sqlite(conn, old_concept, new_concept, renamed_dirs)
    except Exception as e:
        logger.error(f"SQLite update failed during rename: {e}")
        for _, rd_old, rd_new in renamed_dirs:
            try:
                os.rename(rd_new, rd_old)
            except OSError:
                pass
        return {"ok": False, "error": f"Database update failed: {e}"}

    # 3. Update ChromaDB
    try:
        _rename_chromadb(old_concept, new_concept, renamed_dirs)
    except Exception as e:
        logger.warning(f"ChromaDB update failed during rename (filesystem+SQL already committed): {e}")

    from src.services.cache_service import cache_manager
    cache_manager.invalidate()

    logger.info(f"Renamed concept '{old_concept}' -> '{new_concept}' across {len(renamed_dirs)} dir(s)")
    return {"ok": True}


def _rename_sqlite(conn, old_concept: str, new_concept: str, renamed_dirs: list[tuple]):
    """Update all SQLite tables for a concept rename."""
    conn.execute(
        text("UPDATE clusters SET folder_path = :new WHERE folder_path = :old"),
        {"new": new_concept, "old": old_concept},
    )
    conn.execute(
        text("UPDATE clustering_runs SET folder_path = :new WHERE folder_path = :old"),
        {"new": new_concept, "old": old_concept},
    )

    row = conn.execute(
        text("SELECT summary FROM folder_summaries WHERE folder_path = :old"),
        {"old": old_concept},
    ).fetchone()
    if row:
        conn.execute(
            text("INSERT OR REPLACE INTO folder_summaries (folder_path, summary, updated_at) "
                 "VALUES (:new, :summary, CURRENT_TIMESTAMP)"),
            {"new": new_concept, "summary": row._mapping["summary"]},
        )
        conn.execute(
            text("DELETE FROM folder_summaries WHERE folder_path = :old"),
            {"old": old_concept},
        )

    for parent, old_dir, new_dir in renamed_dirs:
        old_prefix = old_dir + os.sep
        new_prefix = new_dir + os.sep

        conn.execute(
            text("UPDATE generated_images SET file_path = :new_prefix || SUBSTR(file_path, :old_len + 1) "
                 "WHERE file_path LIKE :old_like"),
            {"new_prefix": new_prefix, "old_len": len(old_prefix), "old_like": old_prefix + "%"},
        )

        conn.execute(
            text("UPDATE generated_images SET subfolder = :new_sub || SUBSTR(subfolder, :old_len + 1) "
                 "WHERE subfolder LIKE :old_like"),
            {"new_sub": new_concept, "old_len": len(old_concept), "old_like": old_concept + "%"},
        )

        conn.execute(
            text("UPDATE generation_settings SET output_folder = :new_sub || SUBSTR(output_folder, :old_len + 1) "
                 "WHERE output_folder LIKE :old_like"),
            {"new_sub": new_concept, "old_len": len(old_concept), "old_like": old_concept + "%"},
        )

        conn.execute(
            text("UPDATE cluster_assignments SET doc_id = :new_prefix || SUBSTR(doc_id, :old_len + 1) "
                 "WHERE doc_id LIKE :old_like"),
            {"new_prefix": new_prefix, "old_len": len(old_prefix), "old_like": old_prefix + "%"},
        )

        conn.execute(
            text("UPDATE thumbnail_cache SET file_path = :new_prefix || SUBSTR(file_path, :old_len + 1) "
                 "WHERE file_path LIKE :old_like"),
            {"new_prefix": new_prefix, "old_len": len(old_prefix), "old_like": old_prefix + "%"},
        )

    # settings: cluster_k_intra:<old> -> cluster_k_intra:<new>
    old_key = f"cluster_k_intra:{old_concept}"
    new_key = f"cluster_k_intra:{new_concept}"
    row = conn.execute(
        text("SELECT value FROM settings WHERE key = :old_key"),
        {"old_key": old_key},
    ).fetchone()
    if row:
        conn.execute(
            text("INSERT OR REPLACE INTO settings (key, value, updated_at) "
                 "VALUES (:new_key, :value, CURRENT_TIMESTAMP)"),
            {"new_key": new_key, "value": row._mapping["value"]},
        )
        conn.execute(
            text("DELETE FROM settings WHERE key = :old_key"),
            {"old_key": old_key},
        )


def _rename_chromadb(old_concept: str, new_concept: str, renamed_dirs: list[tuple]):
    """Update ChromaDB: concept metadata + re-key file-path doc IDs."""
    for source_type in ("training", "output"):
        collection = vector_store._get_collection(source_type)
        count = collection.count()
        if count == 0:
            continue

        all_data = collection.get(
            limit=count,
            include=["metadatas", "embeddings", "documents"],
            where={"concept": old_concept},
        )
        if not all_data["ids"]:
            continue

        rekey_indices = []
        metadata_only_indices = []

        for i, doc_id in enumerate(all_data["ids"]):
            needs_rekey = any(
                doc_id.startswith(old_dir + os.sep) or doc_id.startswith(old_dir + "/")
                for _, old_dir, _ in renamed_dirs
            )
            if needs_rekey:
                rekey_indices.append(i)
            else:
                metadata_only_indices.append(i)

        if metadata_only_indices:
            update_ids = [all_data["ids"][i] for i in metadata_only_indices]
            update_metas = []
            for i in metadata_only_indices:
                meta = dict(all_data["metadatas"][i])
                meta["concept"] = new_concept
                update_metas.append(meta)
            for cs in range(0, len(update_ids), 500):
                ce = cs + 500
                collection.update(ids=update_ids[cs:ce], metadatas=update_metas[cs:ce])

        if rekey_indices:
            old_ids = []
            new_ids = []
            new_docs = []
            new_embeds = []
            new_metas = []

            for i in rekey_indices:
                old_id = all_data["ids"][i]
                new_id = old_id
                for _, old_dir, new_dir in renamed_dirs:
                    for sep in (os.sep, "/"):
                        prefix = old_dir + sep
                        if old_id.startswith(prefix):
                            new_id = new_dir + sep + old_id[len(prefix):]
                            break
                    if new_id != old_id:
                        break

                meta = dict(all_data["metadatas"][i])
                meta["concept"] = new_concept
                old_ids.append(old_id)
                new_ids.append(new_id)
                new_docs.append(all_data["documents"][i])
                new_embeds.append(all_data["embeddings"][i])
                new_metas.append(meta)

            for cs in range(0, len(old_ids), 500):
                ce = cs + 500
                collection.delete(ids=old_ids[cs:ce])
            for cs in range(0, len(new_ids), 500):
                ce = cs + 500
                collection.add(
                    ids=new_ids[cs:ce],
                    documents=new_docs[cs:ce],
                    embeddings=new_embeds[cs:ce],
                    metadatas=new_metas[cs:ce],
                )
