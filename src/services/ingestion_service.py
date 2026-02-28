"""Data ingestion pipeline — scans directories, parses files, generates embeddings."""

import json
import os
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.services.image_parser import parse_file, ParsedImageData
from src.services import embedding_service
from src.models import vector_store, settings
from src.models.database import get_db
from src.config import EMBEDDING_BATCH_SIZE

logger = logging.getLogger(__name__)

# Global ingestion state
_ingestion_lock = threading.Lock()
_ingestion_running = False
_status_listeners: list = []  # list of callback functions


@dataclass
class IngestionProgress:
    """Current state of an ingestion run."""
    phase: str = "idle"
    message: str = ""
    directories_scanned: int = 0
    total_files: int = 0
    new_files: int = 0
    already_indexed: int = 0
    current: int = 0
    current_dir: str = ""
    errors: int = 0
    complete: bool = False


def add_status_listener(callback):
    """Register a callback for ingestion status updates."""
    _status_listeners.append(callback)


def remove_status_listener(callback):
    """Remove a status listener."""
    if callback in _status_listeners:
        _status_listeners.remove(callback)


def _emit_status(progress: IngestionProgress):
    """Notify all listeners of a status update."""
    for listener in _status_listeners[:]:
        try:
            listener(progress)
        except Exception as e:
            logger.error(f"Error in status listener: {e}")


def is_running() -> bool:
    """Check if ingestion is currently running."""
    return _ingestion_running


def start_ingestion(output_only: bool = False):
    """Start ingestion in a background thread.

    Args:
        output_only: If True, only re-scan output directories.
    """
    global _ingestion_running
    with _ingestion_lock:
        if _ingestion_running:
            logger.warning("Ingestion already running, skipping")
            return
        _ingestion_running = True

    thread = threading.Thread(
        target=_run_ingestion,
        args=(output_only,),
        daemon=True,
    )
    thread.start()


def _run_ingestion(output_only: bool = False):
    """Main ingestion logic — runs in a background thread."""
    global _ingestion_running
    progress = IngestionProgress()

    try:
        # Phase 1: Discovery
        progress.phase = "discovery"
        progress.message = "Loading data directories..."
        _emit_status(progress)

        dirs = settings.get_data_directories(active_only=True)
        if output_only:
            dirs = [d for d in dirs if d["dir_type"] == "output"]

        if not dirs:
            progress.message = "No data directories configured."
            progress.phase = "complete"
            progress.complete = True
            _emit_status(progress)
            return

        progress.directories_scanned = len(dirs)
        progress.message = f"Scanning {len(dirs)} {'output ' if output_only else ''}director{'y' if len(dirs) == 1 else 'ies'}..."
        _emit_status(progress)

        # Collect files to process
        files_to_process = []
        for d in dirs:
            dir_path = d["path"]
            dir_type = d["dir_type"]

            if not os.path.isdir(dir_path):
                logger.warning(f"Directory not found: {dir_path}")
                continue

            dir_files = _scan_directory(dir_path, dir_type)
            progress.total_files += len(dir_files)

            for filepath, parsed_dir_type in dir_files:
                files_to_process.append((filepath, parsed_dir_type, d))

        progress.message = f"Found {len(dirs)} directories with {progress.total_files} total files"
        _emit_status(progress)

        # Check which files are already indexed.
        # Normalise stored IDs so they match the normalised scan paths,
        # even if earlier runs stored non-normalised paths.
        training_existing = {
            os.path.normpath(p) for p in vector_store.get_existing_ids("training")
        }
        output_existing = {
            os.path.normpath(p) for p in vector_store.get_existing_ids("output")
        }

        new_files = []
        for filepath, dir_type, dir_info in files_to_process:
            existing = training_existing if dir_type == "training" else output_existing
            if filepath not in existing:
                new_files.append((filepath, dir_type, dir_info))
            else:
                progress.already_indexed += 1

        progress.new_files = len(new_files)
        progress.message = (
            f"{progress.new_files} new files to process, "
            f"{progress.already_indexed} already indexed"
        )
        _emit_status(progress)

        if not new_files:
            progress.phase = "complete"
            progress.complete = True
            progress.message = "No new files to index"
            _emit_status(progress)
            return

        # Phase 2: Parse and embed
        progress.phase = "embedding"
        batch_ids = []
        batch_texts = []
        batch_metadatas = []
        batch_source_types = []

        for i, (filepath, dir_type, dir_info) in enumerate(new_files):
            progress.current = i + 1
            rel_path = os.path.relpath(filepath, dir_info["path"])
            dir_name = os.path.basename(dir_info["path"])

            # Determine concept from subdirectory
            parts = rel_path.split(os.sep)
            concept = parts[0] if len(parts) > 1 else dir_name

            progress.current_dir = f"{dir_name}/{concept}" if concept != dir_name else dir_name
            progress.message = f"Processing {progress.current}/{progress.new_files}: {progress.current_dir}"
            _emit_status(progress)

            # Parse the file
            parsed = parse_file(filepath)
            if parsed is None:
                progress.errors += 1
                logger.warning(f"Failed to parse: {filepath}")
                continue

            # For output files, create SQLite records so the browser can query them
            if dir_type == "output":
                try:
                    _ensure_sqlite_records(filepath, parsed, d)
                except Exception as e:
                    logger.warning(f"Failed to create SQLite records for {filepath}: {e}")

            metadata = {
                "concept": concept,
                "base_dir": dir_name,
                "source_file": parsed.source_file,
                "dir_type": dir_type,
            }

            # Add model/LoRA metadata for output files
            if dir_type == "output":
                metadata["base_model"] = parsed.base_model or ""
                metadata["loras"] = parsed.loras

            batch_ids.append(filepath)
            batch_texts.append(parsed.prompt)
            batch_metadatas.append(metadata)
            batch_source_types.append(dir_type)

            # Process in batches — rate limiting is handled by llm_service
            if len(batch_ids) >= EMBEDDING_BATCH_SIZE:
                _process_batch(batch_ids, batch_texts, batch_metadatas, batch_source_types)
                batch_ids, batch_texts, batch_metadatas, batch_source_types = [], [], [], []

        # Process remaining batch
        if batch_ids:
            _process_batch(batch_ids, batch_texts, batch_metadatas, batch_source_types)

        # Phase 3: Clustering
        from src.services import clustering_service
        from src.models.database import get_db

        progress.phase = "clustering"
        progress.message = "Checking clustering status..."
        _emit_status(progress)

        # Check if any clusters exist
        with get_db() as conn:
            cursor = conn.execute("SELECT COUNT(*) as cnt FROM clusters")
            cluster_count = cursor.fetchone()["cnt"]

        if cluster_count == 0 and progress.new_files > 0:
            # No clusters exist — run full clustering
            progress.message = "Generating initial clusters..."
            _emit_status(progress)

            try:
                clustering_service.generate_cross_folder_clusters()
                progress.message = "Cross-folder clusters generated. Processing folders..."
                _emit_status(progress)

                clustering_service.generate_intra_folder_clusters()
                progress.message = "All clusters generated"
                _emit_status(progress)
            except Exception as e:
                logger.error(f"Clustering error: {e}", exc_info=True)
                progress.message = f"Clustering warning: {str(e)}"
                _emit_status(progress)

        elif cluster_count > 0 and new_files:
            # Clusters exist — assign new docs to existing clusters
            progress.message = "Assigning new documents to existing clusters..."
            _emit_status(progress)

            try:
                # Gather new doc info for assignment
                new_doc_ids = []
                new_embeddings = []
                new_source_types = []
                new_concepts = []

                for filepath, dir_type, dir_info in new_files:
                    rel_path = os.path.relpath(filepath, dir_info["path"])
                    parts = rel_path.split(os.sep)
                    concept = parts[0] if len(parts) > 1 else os.path.basename(dir_info["path"])
                    new_doc_ids.append(filepath)
                    new_source_types.append(dir_type)
                    new_concepts.append(concept)

                # Fetch embeddings from ChromaDB for the new docs
                from src.models import vector_store as _vs
                for stype in set(new_source_types):
                    collection = _vs._get_collection(stype)
                    stype_ids = [did for did, st in zip(new_doc_ids, new_source_types) if st == stype]
                    if not stype_ids:
                        continue
                    # Fetch in chunks to avoid too-large queries
                    chunk_size = 500
                    for ci in range(0, len(stype_ids), chunk_size):
                        chunk_ids = stype_ids[ci:ci + chunk_size]
                        try:
                            result = collection.get(ids=chunk_ids, include=["embeddings"])
                            if result["embeddings"]:
                                for doc_id, embedding in zip(result["ids"], result["embeddings"]):
                                    idx = new_doc_ids.index(doc_id)
                                    new_embeddings.append((idx, embedding))
                        except Exception as e:
                            logger.warning(f"Could not fetch embeddings for chunk: {e}")

                # Sort by original index and extract just the embeddings
                if new_embeddings:
                    new_embeddings.sort(key=lambda x: x[0])
                    ordered_embeddings = [emb for _, emb in new_embeddings]
                    ordered_ids = [new_doc_ids[idx] for idx, _ in new_embeddings]
                    ordered_types = [new_source_types[idx] for idx, _ in new_embeddings]
                    ordered_concepts = [new_concepts[idx] for idx, _ in new_embeddings]

                    clustering_service.assign_new_docs_to_clusters(
                        ordered_ids, ordered_embeddings, ordered_types, ordered_concepts
                    )
                    progress.message = f"Assigned {len(ordered_ids)} new documents to clusters"
                    _emit_status(progress)

                # Check for new folders needing intra-clustering
                # Get folders that have docs but no intra-folder clusters
                with get_db() as conn:
                    existing_folders = set()
                    cursor = conn.execute(
                        "SELECT DISTINCT folder_path FROM clusters WHERE cluster_type = 'intra_folder'"
                    )
                    for row in cursor:
                        if row["folder_path"]:
                            existing_folders.add(row["folder_path"])

                new_folder_concepts = set(new_concepts) - existing_folders
                if new_folder_concepts:
                    progress.message = f"Checking {len(new_folder_concepts)} new folders for clustering..."
                    _emit_status(progress)
                    for concept in new_folder_concepts:
                        try:
                            clustering_service.generate_intra_folder_clusters(folder_path=concept)
                        except Exception as e:
                            logger.warning(f"Intra-clustering for {concept} skipped: {e}")

            except Exception as e:
                logger.error(f"Post-ingestion clustering error: {e}", exc_info=True)
                progress.message = f"Clustering assignment warning: {str(e)}"
                _emit_status(progress)

        # Phase 4: Complete
        progress.phase = "complete"
        progress.complete = True
        progress.message = (
            f"Indexing complete: {progress.new_files - progress.errors} new, "
            f"{progress.already_indexed} existing, {progress.errors} errors"
        )
        _emit_status(progress)

    except Exception as e:
        logger.error(f"Ingestion error: {e}", exc_info=True)
        progress.phase = "error"
        progress.message = f"Ingestion failed: {str(e)}"
        _emit_status(progress)

    finally:
        _ingestion_running = False


def _scan_directory(dir_path: str, dir_type: str) -> list[tuple[str, str]]:
    """Scan a directory and return list of (filepath, dir_type) for parseable files.

    All file paths are normalised with os.path.normpath so they match
    the IDs already stored in the vector store, preventing duplicates
    and ensuring partially-indexed folders are completed on re-index.
    """
    IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
    files = []

    for root, _dirs, filenames in os.walk(dir_path):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            filepath = os.path.normpath(os.path.join(root, fname))

            if dir_type == "training" and ext == ".txt":
                files.append((filepath, "training"))
            elif dir_type == "output" and ext in IMAGE_EXTS:
                files.append((filepath, "output"))

    return files


def _process_batch(
    ids: list[str],
    texts: list[str],
    metadatas: list[dict],
    source_types: list[str],
):
    """Generate embeddings and store a batch of documents."""
    try:
        embeddings = embedding_service.embed_batch(texts)

        # Group by source type for ChromaDB
        groups: dict[str, tuple[list, list, list, list]] = {}
        for doc_id, text, embedding, metadata, stype in zip(
            ids, texts, embeddings, metadatas, source_types
        ):
            if stype not in groups:
                groups[stype] = ([], [], [], [])
            g = groups[stype]
            g[0].append(doc_id)
            g[1].append(text)
            g[2].append(embedding)
            g[3].append(metadata)

        for stype, (gids, gtexts, gembs, gmetas) in groups.items():
            vector_store.add_documents_batch(gids, gtexts, gembs, stype, gmetas)

    except Exception as e:
        logger.error(f"Error processing batch: {e}", exc_info=True)


def _ensure_sqlite_records(filepath: str, parsed: ParsedImageData, dir_info: dict):
    """Create generation_jobs + generation_settings + generated_images records
    for a scanned output file, so the browser can query it.

    Skips if a generated_images record with the same file_path already exists.
    """
    norm_path = os.path.normpath(filepath)

    with get_db() as conn:
        # Check for existing record
        cursor = conn.execute(
            "SELECT 1 FROM generated_images WHERE file_path = ?", (norm_path,)
        )
        if cursor.fetchone():
            return  # Already registered

        # Get file metadata
        try:
            stat = os.stat(filepath)
            file_size = stat.st_size
            file_mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        except OSError:
            file_size = None
            file_mtime = None

        job_id = str(uuid.uuid4())

        # Create generation_jobs record (chat_id = NULL, source = 'scan')
        conn.execute(
            """INSERT INTO generation_jobs (id, chat_id, message_id, prompt_id, status, source, created_at, completed_at)
               VALUES (?, NULL, NULL, NULL, 'completed', 'scan', ?, ?)""",
            (job_id, file_mtime, file_mtime),
        )

        # Build loras JSON
        loras_json = json.dumps(parsed.loras) if parsed.loras else None

        # Create generation_settings record
        conn.execute(
            """INSERT INTO generation_settings
               (job_id, positive_prompt, negative_prompt, base_model, loras,
                output_folder, seed, num_images, sampler, cfg_scale, scheduler, steps)
               VALUES (?, ?, ?, ?, ?, ?, -1, 1, ?, ?, ?, ?)""",
            (
                job_id,
                parsed.prompt,
                parsed.negative_prompt,
                parsed.base_model,
                loras_json,
                os.path.basename(os.path.dirname(filepath)),
                parsed.sampler,
                parsed.cfg_scale,
                parsed.scheduler,
                parsed.steps,
            ),
        )

        # Create generated_images record
        filename = os.path.basename(filepath)
        subfolder = os.path.relpath(os.path.dirname(filepath), dir_info["path"])
        if subfolder == ".":
            subfolder = ""

        conn.execute(
            """INSERT INTO generated_images
               (job_id, filename, subfolder, file_size, file_path)
               VALUES (?, ?, ?, ?, ?)""",
            (job_id, filename, subfolder, file_size, norm_path),
        )
