#!/usr/bin/env python3
"""One-time migration script to rename concept folders to category__name format.

Usage:
    python migrate_folder_names.py generate               # Create TSV mapping file
    python migrate_folder_names.py execute [--dry-run]     # Apply renames from TSV
    python migrate_folder_names.py verify                  # Check consistency after migration
"""

import argparse
import os
import sys
import logging

# Add project root to path so we can import src modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import text

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TSV_FILENAME = "folder_rename_mapping.tsv"


def _get_data_dirs():
    """Get active data directories grouped by dir_type."""
    from src.models.database import get_db
    with get_db() as conn:
        result = conn.execute(
            text("SELECT id, path, dir_type FROM data_directories WHERE active = 1")
        )
        return [dict(r._mapping) for r in result.fetchall()]


def _discover_concepts(data_dirs):
    """List first-level subdirectories under each data directory.

    Returns list of (dir_type_label/concept_name, abs_parent_path, concept_name).
    """
    entries = []
    for dd in data_dirs:
        parent = dd["path"].rstrip("/")
        label = os.path.basename(parent)
        if not os.path.isdir(parent):
            continue
        for name in sorted(os.listdir(parent)):
            full = os.path.join(parent, name)
            if os.path.isdir(full):
                entries.append((f"{label}/{name}", parent, name))
    return entries


# ── generate ────────────────────────────────────────────────────────────

def cmd_generate(args):
    from src.models.database import initialize_database
    initialize_database()

    data_dirs = _get_data_dirs()
    if not data_dirs:
        logger.error("No active data directories found.")
        return 1

    entries = _discover_concepts(data_dirs)
    if not entries:
        logger.error("No concept folders found under data directories.")
        return 1

    with open(TSV_FILENAME, "w") as f:
        f.write("# Edit this file: fill in the second column with the new name (category__descriptive_name).\n")
        f.write("# Leave the second column empty or identical to skip renaming that folder.\n")
        f.write("# Lines starting with # are comments.\n")
        f.write("#\n")
        f.write("# current_path\tnew_name\n")
        for display_path, _parent, concept_name in entries:
            f.write(f"{display_path}\t{concept_name}\n")

    logger.info(f"Wrote {len(entries)} entries to {TSV_FILENAME}")
    logger.info(f"Edit the file, then run: python {sys.argv[0]} execute --dry-run")
    return 0


# ── execute ─────────────────────────────────────────────────────────────

def _parse_tsv():
    """Parse the TSV mapping file.

    Returns list of (display_path, new_name) tuples for entries that need renaming.
    """
    if not os.path.isfile(TSV_FILENAME):
        logger.error(f"{TSV_FILENAME} not found. Run 'generate' first.")
        return None

    mappings = []
    with open(TSV_FILENAME) as f:
        for line_no, line in enumerate(f, 1):
            line = line.rstrip("\n\r")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t", 1)
            display_path = parts[0].strip()
            new_name = parts[1].strip() if len(parts) > 1 else ""
            if not display_path:
                continue
            if not new_name or new_name == display_path.split("/", 1)[-1]:
                continue  # Skip unchanged
            mappings.append((display_path, new_name))
    return mappings


def _resolve_display_path(display_path, data_dirs):
    """Resolve 'label/concept' to (abs_parent_path, old_concept_name)."""
    label, _, concept = display_path.partition("/")
    for dd in data_dirs:
        if os.path.basename(dd["path"].rstrip("/")) == label:
            return dd["path"], concept
    return None, None


def _validate_mappings(mappings, data_dirs):
    """Pre-flight validation. Returns (valid_renames, errors).

    valid_renames: list of (old_concept, new_concept, list_of_abs_parent_dirs)
    """
    errors = []

    # Group by old concept name -> new name, collecting parent dirs
    concept_map = {}  # old_concept -> {new_name, parents: set}
    for display_path, new_name in mappings:
        parent, old_concept = _resolve_display_path(display_path, data_dirs)
        if parent is None:
            errors.append(f"Cannot resolve data directory for: {display_path}")
            continue

        old_dir = os.path.join(parent, old_concept)
        if not os.path.isdir(old_dir):
            errors.append(f"Source directory does not exist: {old_dir}")
            continue

        if old_concept in concept_map:
            if concept_map[old_concept]["new_name"] != new_name:
                errors.append(
                    f"Inconsistent rename for '{old_concept}': "
                    f"'{concept_map[old_concept]['new_name']}' vs '{new_name}'"
                )
                continue
            concept_map[old_concept]["parents"].add(parent)
        else:
            concept_map[old_concept] = {"new_name": new_name, "parents": {parent}}

    # Check for duplicate targets within the same parent directory
    # (different old concepts in different dirs mapping to the same new name is fine)
    targets_by_parent = {}  # parent -> {new_name: old_concept}
    for old_concept, info in concept_map.items():
        for parent in info["parents"]:
            if parent not in targets_by_parent:
                targets_by_parent[parent] = {}
            new_name = info["new_name"]
            if new_name in targets_by_parent[parent]:
                errors.append(
                    f"Duplicate target name '{new_name}' in {parent} for concepts "
                    f"'{targets_by_parent[parent][new_name]}' and '{old_concept}'"
                )
            targets_by_parent[parent][new_name] = old_concept

    # Check destination conflicts on disk
    for old_concept, info in concept_map.items():
        for parent in info["parents"]:
            dest = os.path.join(parent, info["new_name"])
            if os.path.exists(dest) and os.path.basename(dest) != old_concept:
                errors.append(f"Destination already exists: {dest}")

    valid_renames = []
    for old_concept, info in concept_map.items():
        valid_renames.append((old_concept, info["new_name"], sorted(info["parents"])))

    return valid_renames, errors


def _execute_rename(old_concept, new_concept, parent_dirs, dry_run):
    """Execute a single concept rename across all parent dirs + DB + ChromaDB."""
    from src.models.database import get_db

    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Renaming: {old_concept} -> {new_concept}")

    # 1. Rename physical directories
    renamed_dirs = []
    for parent in parent_dirs:
        old_dir = os.path.join(parent, old_concept)
        new_dir = os.path.join(parent, new_concept)
        if not os.path.isdir(old_dir):
            logger.warning(f"  Skipping missing directory: {old_dir}")
            continue
        logger.info(f"  Rename dir: {old_dir} -> {new_dir}")
        if not dry_run:
            try:
                os.rename(old_dir, new_dir)
                renamed_dirs.append((parent, old_dir, new_dir))
            except OSError as e:
                logger.error(f"  FAILED to rename {old_dir}: {e}")
                return False
        else:
            renamed_dirs.append((parent, old_dir, new_dir))

    if not renamed_dirs:
        logger.warning(f"  No directories renamed for {old_concept}")
        return False

    # 2. Update SQLite
    logger.info(f"  Updating SQLite records...")
    if not dry_run:
        with get_db() as conn:
            _update_sqlite(conn, old_concept, new_concept, renamed_dirs)
    else:
        _preview_sqlite(old_concept, new_concept, renamed_dirs)

    # 3. Update ChromaDB
    logger.info(f"  Updating ChromaDB...")
    if not dry_run:
        _update_chromadb(old_concept, new_concept, renamed_dirs)
    else:
        _preview_chromadb(old_concept, new_concept)

    return True


def _update_sqlite(conn, old_concept, new_concept, renamed_dirs):
    """Update all SQLite tables for a concept rename."""

    # clusters.folder_path
    result = conn.execute(
        text("UPDATE clusters SET folder_path = :new WHERE folder_path = :old"),
        {"new": new_concept, "old": old_concept},
    )
    if result.rowcount:
        logger.info(f"    clusters.folder_path: {result.rowcount} rows")

    # clustering_runs.folder_path
    result = conn.execute(
        text("UPDATE clustering_runs SET folder_path = :new WHERE folder_path = :old"),
        {"new": new_concept, "old": old_concept},
    )
    if result.rowcount:
        logger.info(f"    clustering_runs.folder_path: {result.rowcount} rows")

    # folder_summaries (PK = folder_path, so INSERT new + DELETE old)
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
        logger.info(f"    folder_summaries: migrated")

    # Path-based updates for each renamed directory
    for parent, old_dir, new_dir in renamed_dirs:
        old_prefix = old_dir + os.sep
        new_prefix = new_dir + os.sep

        # generated_images.file_path
        result = conn.execute(
            text("UPDATE generated_images SET file_path = :new_prefix || SUBSTR(file_path, :old_len + 1) "
                 "WHERE file_path LIKE :old_like"),
            {"new_prefix": new_prefix, "old_len": len(old_prefix),
             "old_like": old_prefix + "%"},
        )
        if result.rowcount:
            logger.info(f"    generated_images.file_path: {result.rowcount} rows (under {old_dir})")

        # Also update file_path entries that match the dir exactly (files directly in the concept dir)
        # Handle files like /path/to/concept/image.png where file_path starts with old_dir/
        # (already handled by the LIKE above)

        # generated_images.subfolder — replace old concept prefix
        old_sub = old_concept
        new_sub = new_concept
        result = conn.execute(
            text("UPDATE generated_images SET subfolder = :new_sub || SUBSTR(subfolder, :old_len + 1) "
                 "WHERE subfolder LIKE :old_like"),
            {"new_sub": new_sub, "old_len": len(old_sub), "old_like": old_sub + "%"},
        )
        if result.rowcount:
            logger.info(f"    generated_images.subfolder: {result.rowcount} rows")

        # generation_settings.output_folder
        result = conn.execute(
            text("UPDATE generation_settings SET output_folder = :new_sub || SUBSTR(output_folder, :old_len + 1) "
                 "WHERE output_folder LIKE :old_like"),
            {"new_sub": new_sub, "old_len": len(old_sub), "old_like": old_sub + "%"},
        )
        if result.rowcount:
            logger.info(f"    generation_settings.output_folder: {result.rowcount} rows")

        # cluster_assignments.doc_id (file paths)
        result = conn.execute(
            text("UPDATE cluster_assignments SET doc_id = :new_prefix || SUBSTR(doc_id, :old_len + 1) "
                 "WHERE doc_id LIKE :old_like"),
            {"new_prefix": new_prefix, "old_len": len(old_prefix),
             "old_like": old_prefix + "%"},
        )
        if result.rowcount:
            logger.info(f"    cluster_assignments.doc_id: {result.rowcount} rows (under {old_dir})")

        # thumbnail_cache.file_path
        result = conn.execute(
            text("UPDATE thumbnail_cache SET file_path = :new_prefix || SUBSTR(file_path, :old_len + 1) "
                 "WHERE file_path LIKE :old_like"),
            {"new_prefix": new_prefix, "old_len": len(old_prefix),
             "old_like": old_prefix + "%"},
        )
        if result.rowcount:
            logger.info(f"    thumbnail_cache.file_path: {result.rowcount} rows (under {old_dir})")

    # settings: rename cluster_k_intra:<old> -> cluster_k_intra:<new>
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
        logger.info(f"    settings: renamed {old_key} -> {new_key}")


def _preview_sqlite(old_concept, new_concept, renamed_dirs):
    """Preview SQLite changes without executing."""
    from src.models.database import get_db

    with get_db() as conn:
        # Count affected rows
        r = conn.execute(
            text("SELECT COUNT(*) as cnt FROM clusters WHERE folder_path = :old"),
            {"old": old_concept},
        ).fetchone()._mapping["cnt"]
        if r:
            logger.info(f"    clusters.folder_path: {r} rows would update")

        r = conn.execute(
            text("SELECT COUNT(*) as cnt FROM clustering_runs WHERE folder_path = :old"),
            {"old": old_concept},
        ).fetchone()._mapping["cnt"]
        if r:
            logger.info(f"    clustering_runs.folder_path: {r} rows would update")

        r = conn.execute(
            text("SELECT COUNT(*) as cnt FROM folder_summaries WHERE folder_path = :old"),
            {"old": old_concept},
        ).fetchone()._mapping["cnt"]
        if r:
            logger.info(f"    folder_summaries: would migrate {r} row(s)")

        for _parent, old_dir, _new_dir in renamed_dirs:
            old_prefix = old_dir + os.sep
            r = conn.execute(
                text("SELECT COUNT(*) as cnt FROM generated_images WHERE file_path LIKE :like"),
                {"like": old_prefix + "%"},
            ).fetchone()._mapping["cnt"]
            if r:
                logger.info(f"    generated_images.file_path: {r} rows (under {old_dir})")

            r = conn.execute(
                text("SELECT COUNT(*) as cnt FROM cluster_assignments WHERE doc_id LIKE :like"),
                {"like": old_prefix + "%"},
            ).fetchone()._mapping["cnt"]
            if r:
                logger.info(f"    cluster_assignments.doc_id: {r} rows (under {old_dir})")

            r = conn.execute(
                text("SELECT COUNT(*) as cnt FROM thumbnail_cache WHERE file_path LIKE :like"),
                {"like": old_prefix + "%"},
            ).fetchone()._mapping["cnt"]
            if r:
                logger.info(f"    thumbnail_cache.file_path: {r} rows (under {old_dir})")

        r = conn.execute(
            text("SELECT COUNT(*) as cnt FROM generation_settings WHERE output_folder LIKE :like"),
            {"like": old_concept + "%"},
        ).fetchone()._mapping["cnt"]
        if r:
            logger.info(f"    generation_settings.output_folder: {r} rows")

        old_key = f"cluster_k_intra:{old_concept}"
        r = conn.execute(
            text("SELECT COUNT(*) as cnt FROM settings WHERE key = :key"),
            {"key": old_key},
        ).fetchone()._mapping["cnt"]
        if r:
            logger.info(f"    settings: would rename {old_key}")


def _update_chromadb(old_concept, new_concept, renamed_dirs):
    """Update ChromaDB: concept metadata + re-key file-path doc IDs."""
    from src.models import vector_store
    vector_store.initialize()

    for source_type in ("training", "output"):
        collection = vector_store._get_collection(source_type)
        count = collection.count()
        if count == 0:
            continue

        # Get all docs with old concept metadata
        all_data = collection.get(
            limit=count,
            include=["metadatas", "embeddings", "documents"],
            where={"concept": old_concept},
        )

        if not all_data["ids"]:
            continue

        # Separate into: path-based IDs that need re-keying vs others
        rekey_indices = []
        metadata_only_indices = []

        for i, doc_id in enumerate(all_data["ids"]):
            needs_rekey = False
            for _parent, old_dir, _new_dir in renamed_dirs:
                if doc_id.startswith(old_dir + os.sep) or doc_id.startswith(old_dir + "/"):
                    needs_rekey = True
                    break
            if needs_rekey:
                rekey_indices.append(i)
            else:
                metadata_only_indices.append(i)

        # Update metadata-only docs (gen_* IDs etc.)
        if metadata_only_indices:
            update_ids = []
            update_metas = []
            for i in metadata_only_indices:
                update_ids.append(all_data["ids"][i])
                meta = dict(all_data["metadatas"][i])
                meta["concept"] = new_concept
                update_metas.append(meta)
            for chunk_start in range(0, len(update_ids), 500):
                chunk_end = chunk_start + 500
                collection.update(
                    ids=update_ids[chunk_start:chunk_end],
                    metadatas=update_metas[chunk_start:chunk_end],
                )
            logger.info(f"    {source_type}: updated concept metadata for {len(update_ids)} docs")

        # Re-key docs with file-path IDs (delete old + add with new ID)
        if rekey_indices:
            old_ids = []
            new_ids = []
            new_docs = []
            new_embeds = []
            new_metas = []

            for i in rekey_indices:
                old_id = all_data["ids"][i]
                # Compute new ID by replacing old dir prefix with new
                new_id = old_id
                for _parent, old_dir, new_dir in renamed_dirs:
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

            # Delete old IDs
            for chunk_start in range(0, len(old_ids), 500):
                chunk_end = chunk_start + 500
                collection.delete(ids=old_ids[chunk_start:chunk_end])

            # Add with new IDs
            for chunk_start in range(0, len(new_ids), 500):
                chunk_end = chunk_start + 500
                collection.add(
                    ids=new_ids[chunk_start:chunk_end],
                    documents=new_docs[chunk_start:chunk_end],
                    embeddings=new_embeds[chunk_start:chunk_end],
                    metadatas=new_metas[chunk_start:chunk_end],
                )
            logger.info(f"    {source_type}: re-keyed {len(new_ids)} file-path doc IDs")


def _preview_chromadb(old_concept, new_concept):
    """Preview ChromaDB changes."""
    from src.models import vector_store
    vector_store.initialize()

    for source_type in ("training", "output"):
        collection = vector_store._get_collection(source_type)
        count = collection.count()
        if count == 0:
            continue

        try:
            results = collection.get(
                limit=count,
                include=["metadatas"],
                where={"concept": old_concept},
            )
            n = len(results["ids"]) if results["ids"] else 0
            if n:
                path_ids = sum(1 for doc_id in results["ids"] if not doc_id.startswith("gen_"))
                logger.info(f"    {source_type}: {n} docs with concept='{old_concept}' "
                            f"({path_ids} path-based, {n - path_ids} gen-based)")
        except Exception:
            pass


def cmd_execute(args):
    from src.models.database import initialize_database
    initialize_database()

    mappings = _parse_tsv()
    if mappings is None:
        return 1
    if not mappings:
        logger.info("No renames to apply (all entries empty or unchanged).")
        return 0

    data_dirs = _get_data_dirs()
    valid_renames, errors = _validate_mappings(mappings, data_dirs)

    if errors:
        logger.error("Validation errors:")
        for e in errors:
            logger.error(f"  - {e}")
        if not args.dry_run:
            logger.error("Fix errors before executing. Aborting.")
            return 1
        logger.warning("Continuing dry run despite errors...")

    if not valid_renames:
        logger.info("No valid renames to apply.")
        return 0

    if not args.dry_run:
        logger.warning("=== EXECUTING RENAMES ===")
        logger.warning("Back up app.db and chroma_db/ before proceeding!")
    else:
        logger.info("=== DRY RUN ===")

    success = 0
    failed = 0
    for old_concept, new_concept, parent_dirs in valid_renames:
        ok = _execute_rename(old_concept, new_concept, parent_dirs, args.dry_run)
        if ok:
            success += 1
        else:
            failed += 1

    logger.info(f"\nDone. {success} concept(s) {'would be ' if args.dry_run else ''}renamed, {failed} failed.")
    if not args.dry_run and success > 0:
        logger.info(f"Run 'python {sys.argv[0]} verify' to check consistency.")
    return 0 if failed == 0 else 1


# ── verify ──────────────────────────────────────────────────────────────

def cmd_verify(args):
    from src.models.database import initialize_database, get_db
    from src.models import vector_store

    initialize_database()
    vector_store.initialize()

    data_dirs = _get_data_dirs()
    output_dirs = [dd["path"] for dd in data_dirs if dd["dir_type"] == "output"]
    training_dirs = [dd["path"] for dd in data_dirs if dd["dir_type"] == "training"]
    all_dirs = output_dirs + training_dirs

    errors = []

    # 1. Check generated_images.file_path values point to existing files
    logger.info("Checking generated_images.file_path...")
    with get_db() as conn:
        result = conn.execute(
            text("SELECT id, file_path FROM generated_images WHERE file_path IS NOT NULL")
        )
        missing = 0
        total = 0
        for row in result.fetchall():
            total += 1
            fp = row._mapping["file_path"]
            if not os.path.isfile(fp):
                missing += 1
                if missing <= 10:
                    errors.append(f"generated_images.file_path missing: {fp}")
        if missing > 10:
            errors.append(f"  ... and {missing - 10} more missing file_path entries")
        logger.info(f"  {total} records, {missing} missing files")

    # 2. Check clusters.folder_path match directories on disk
    logger.info("Checking clusters.folder_path...")
    with get_db() as conn:
        result = conn.execute(
            text("SELECT DISTINCT folder_path FROM clusters WHERE folder_path IS NOT NULL")
        )
        for row in result.fetchall():
            folder_path = row._mapping["folder_path"]
            found = False
            for parent in all_dirs:
                if os.path.isdir(os.path.join(parent, folder_path)):
                    found = True
                    break
            if not found:
                errors.append(f"clusters.folder_path has no matching dir: {folder_path}")

    # 3. Check ChromaDB concept metadata matches disk folders
    logger.info("Checking ChromaDB concepts...")
    disk_concepts = set()
    for parent in all_dirs:
        if os.path.isdir(parent):
            for name in os.listdir(parent):
                if os.path.isdir(os.path.join(parent, name)):
                    disk_concepts.add(name)

    for source_type in ("training", "output"):
        collection = vector_store._get_collection(source_type)
        count = collection.count()
        if count == 0:
            continue
        all_data = collection.get(limit=count, include=["metadatas"])
        chroma_concepts = set()
        for meta in all_data["metadatas"]:
            c = meta.get("concept", "")
            if c:
                chroma_concepts.add(c)
        stale = chroma_concepts - disk_concepts
        if stale:
            errors.append(f"ChromaDB {source_type} has concepts not on disk: {stale}")

    # 4. Check no stale cluster_k_intra:* settings keys
    logger.info("Checking settings keys...")
    with get_db() as conn:
        result = conn.execute(
            text("SELECT key FROM settings WHERE key LIKE 'cluster_k_intra:%'")
        )
        for row in result.fetchall():
            key = row._mapping["key"]
            concept = key.split(":", 1)[1]
            if concept not in disk_concepts:
                errors.append(f"Stale settings key: {key} (concept '{concept}' not on disk)")

    if errors:
        logger.error(f"\n{len(errors)} issue(s) found:")
        for e in errors:
            logger.error(f"  - {e}")
        return 1
    else:
        logger.info("\nAll checks passed!")
        return 0


# ── main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Rename concept folders to category__name format."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("generate", help="Create TSV mapping file from current folders")

    exec_parser = subparsers.add_parser("execute", help="Apply renames from TSV")
    exec_parser.add_argument("--dry-run", action="store_true",
                             help="Preview changes without executing")

    subparsers.add_parser("verify", help="Check consistency after migration")

    args = parser.parse_args()

    handlers = {
        "generate": cmd_generate,
        "execute": cmd_execute,
        "verify": cmd_verify,
    }
    sys.exit(handlers[args.command](args))


if __name__ == "__main__":
    main()
