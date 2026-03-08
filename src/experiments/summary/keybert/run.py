"""CLI entry point for the KeyBERT keyword extraction experiment."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
DEFAULT_OUTPUT_DIR = str(SCRIPT_DIR.parent / "output" / "keybert")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract keywords from cluster/folder prompts using KeyBERT.",
    )
    parser.add_argument(
        "--model", type=str, default="all-MiniLM-L6-v2",
        help="Sentence-transformer model name (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--top-n", type=int, default=5,
        help="Number of keywords to extract per cluster/folder (default: 5)",
    )
    parser.add_argument(
        "--ngram-min", type=int, default=1,
        help="Minimum n-gram size for keyphrases (default: 1)",
    )
    parser.add_argument(
        "--ngram-max", type=int, default=3,
        help="Maximum n-gram size for keyphrases (default: 3)",
    )
    parser.add_argument(
        "--diversity", type=float, default=0.5,
        help="MMR diversity factor 0-1 (default: 0.5)",
    )
    parser.add_argument(
        "--no-mmr", action="store_true", default=False,
        help="Disable Maximal Marginal Relevance diversification",
    )
    parser.add_argument(
        "--max-cross-clusters", type=int, default=None,
        help="Randomly sample N cross-folder clusters (default: all)",
    )
    parser.add_argument(
        "--max-folders", type=int, default=None,
        help="Randomly sample N folders (default: all)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--distance-threshold", type=float, default=1.5,
        help="Max distance for contributing intra-folder clusters (default: 1.5)",
    )
    parser.add_argument(
        "--top-k-folders", type=int, default=5,
        help="Contributing intra-folder clusters per cross-folder centroid (default: 5)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Initialize DB and vector store
    logger.info("Initializing database and vector store...")
    from src.models import vector_store
    from src.models.database import initialize_database
    initialize_database()
    vector_store.initialize()

    # Load structural data (sample_prompts=0 since we fetch all prompts separately)
    from ..data_loader import (
        get_all_prompts_for_cluster,
        get_all_prompts_for_folder,
        load_cross_folder_inputs,
        load_folder_inputs,
    )

    logger.info("Loading cross-folder cluster data...")
    cross_inputs = load_cross_folder_inputs(
        max_clusters=args.max_cross_clusters,
        sample_prompts=0,
        distance_threshold=args.distance_threshold,
        top_k_folders=args.top_k_folders,
    )
    logger.info(f"Loaded {len(cross_inputs)} cross-folder clusters.")

    logger.info("Loading folder data...")
    folder_inputs = load_folder_inputs(
        max_folders=args.max_folders,
        sample_prompts=0,
    )
    logger.info(f"Loaded {len(folder_inputs)} folders.")

    if not cross_inputs and not folder_inputs:
        logger.warning("No data to process. Exiting.")
        return

    # Fetch ALL prompts for each cluster/folder
    logger.info("Fetching all prompts for cross-folder clusters...")
    cross_prompt_lists = []
    for item in cross_inputs:
        prompts = get_all_prompts_for_cluster(item.cluster_id)
        cross_prompt_lists.append(prompts)
        logger.info(f"  Cluster #{item.cluster_id}: {len(prompts)} prompts")

    logger.info("Fetching all prompts for folders...")
    folder_prompt_lists = []
    for item in folder_inputs:
        prompts = get_all_prompts_for_folder(item.folder_path)
        folder_prompt_lists.append(prompts)
        logger.info(f"  {item.folder_path}: {len(prompts)} prompts")

    # Load KeyBERT model
    from . import keybert_engine

    kw_model = keybert_engine.load_model(model_name=args.model)

    ngram_range = (args.ngram_min, args.ngram_max)
    use_mmr = not args.no_mmr

    # Extract keywords
    all_prompt_lists = cross_prompt_lists + folder_prompt_lists
    logger.info(f"Extracting keywords for {len(all_prompt_lists)} items...")
    all_keywords = keybert_engine.batch_extract(
        kw_model, all_prompt_lists,
        top_n=args.top_n,
        ngram_range=ngram_range,
        diversity=args.diversity,
        use_mmr=use_mmr,
    )

    cross_keywords = all_keywords[:len(cross_inputs)]
    folder_keywords = all_keywords[len(cross_inputs):]

    # Write output
    from ..output_writer import write_cross_folder_summaries, write_folder_summaries

    if cross_inputs:
        path = write_cross_folder_summaries(
            cross_inputs, cross_keywords, args.output_dir,
            summary_label="KeyBERT Summary",
        )
        logger.info(f"Wrote cross-folder summaries to {path}")

    if folder_inputs:
        path = write_folder_summaries(
            folder_inputs, folder_keywords, args.output_dir,
            summary_label="KeyBERT Summary",
        )
        logger.info(f"Wrote folder summaries to {path}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
