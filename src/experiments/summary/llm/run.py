"""CLI entry point for the LLM cluster summarization experiment."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
DEFAULT_OUTPUT_DIR = str(SCRIPT_DIR.parent / "output" / "llm")
DEFAULT_SYSTEM_PROMPT = str(SCRIPT_DIR / "prompts" / "system_prompt.md")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LLM summaries for cross-folder clusters and folder summaries.",
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
        "--sample-prompts", type=int, default=5,
        help="Number of sample prompts per cluster/folder (default: 5)",
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-4B-FP8",
        help="HuggingFace model ID (default: Qwen/Qwen3-4B-FP8)",
    )
    parser.add_argument(
        "--quantization", type=str, default=None, choices=["awq", "gptq"],
        help="Quantization method (default: None)",
    )
    parser.add_argument(
        "--dtype", type=str, default="auto",
        choices=["auto", "half", "float16", "bfloat16", "float", "float32"],
        help="Model weight precision (default: auto)",
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
    parser.add_argument(
        "--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT,
        help="Path to system prompt file",
    )
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.85,
        help="Fraction of GPU memory for vLLM (default: 0.85)",
    )
    parser.add_argument(
        "--cross-folder-template", type=str, default=None,
        help="Path to cross-folder prompt template (default: bundled template)",
    )
    parser.add_argument(
        "--folder-template", type=str, default=None,
        help="Path to folder prompt template (default: bundled template)",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=4096,
        help="Maximum model context length in tokens (default: 4096)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=1024,
        help="Maximum output tokens per response (default: 1024)",
    )
    parser.add_argument(
        "--no-think", action="store_true", default=False,
        help="Disable model thinking/reasoning (appends /no_think to prompts)",
    )
    parser.add_argument(
        "--no-intra-clusters", action="store_true", default=False,
        help="Disable per-cluster summaries within folders",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Load system prompt
    system_prompt_path = Path(args.system_prompt)
    if not system_prompt_path.is_file():
        logger.error(f"System prompt not found: {system_prompt_path}")
        sys.exit(1)
    system_prompt = system_prompt_path.read_text(encoding="utf-8").strip()

    # Initialize DB and vector store
    logger.info("Initializing database and vector store...")
    from src.models import vector_store
    from src.models.database import initialize_database
    initialize_database()
    vector_store.initialize()

    # Load data
    from ..data_loader import FolderInput, load_cross_folder_inputs, load_folder_inputs

    logger.info("Loading cross-folder cluster data...")
    cross_inputs = load_cross_folder_inputs(
        max_clusters=args.max_cross_clusters,
        sample_prompts=args.sample_prompts,
        distance_threshold=args.distance_threshold,
        top_k_folders=args.top_k_folders,
    )
    logger.info(f"Loaded {len(cross_inputs)} cross-folder clusters.")

    include_intra = not args.no_intra_clusters
    logger.info("Loading folder data...")
    folder_inputs = load_folder_inputs(
        max_folders=args.max_folders,
        sample_prompts=args.sample_prompts,
        include_intra_clusters=include_intra,
    )
    logger.info(f"Loaded {len(folder_inputs)} folders.")

    if not cross_inputs and not folder_inputs:
        logger.warning("No data to summarize. Exiting.")
        return

    # Build prompts
    from .prompt_builder import build_cross_folder_prompt, build_folder_prompt

    cross_prompts = [
        build_cross_folder_prompt(item, args.cross_folder_template)
        for item in cross_inputs
    ]
    folder_prompts = [
        build_folder_prompt(item, args.folder_template)
        for item in folder_inputs
    ]

    # Build intra-cluster prompts
    intra_prompts = []
    intra_map: list[tuple[int, int]] = []  # (folder_index, cluster_index)
    if include_intra:
        for fi, folder_input in enumerate(folder_inputs):
            for ci, cluster in enumerate(folder_input.intra_clusters):
                tmp_input = FolderInput(
                    folder_path=folder_input.folder_path,
                    tfidf_summary=folder_input.tfidf_summary,
                    concept=folder_input.concept,
                    data_root=folder_input.data_root,
                    sample_prompts=cluster.sample_prompts,
                )
                intra_prompts.append(build_folder_prompt(tmp_input, args.folder_template))
                intra_map.append((fi, ci))

    all_prompts = cross_prompts + folder_prompts + intra_prompts
    logger.info(f"Built {len(all_prompts)} prompts total "
                f"({len(cross_prompts)} cross, {len(folder_prompts)} folder, "
                f"{len(intra_prompts)} intra-cluster).")

    # Load model and run inference
    from . import llm_engine

    llm = llm_engine.load_model(
        model_id=args.model,
        quantization=args.quantization,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    try:
        all_clean, all_raw = llm_engine.batch_generate(
            llm, all_prompts, system_prompt,
            max_tokens=args.max_tokens, no_think=args.no_think,
        )
    finally:
        llm_engine.unload_model(llm)

    # Split results
    n_cross = len(cross_inputs)
    n_folder = len(folder_inputs)
    cross_clean = all_clean[:n_cross]
    cross_raw = all_raw[:n_cross]
    folder_clean = all_clean[n_cross:n_cross + n_folder]
    folder_raw = all_raw[n_cross:n_cross + n_folder]
    intra_clean = all_clean[n_cross + n_folder:]
    intra_raw = all_raw[n_cross + n_folder:]

    # Reconstruct per-folder cluster summaries
    cluster_summaries: list[list[tuple[str, str]]] | None = None
    cluster_summaries_raw: list[list[tuple[str, str]]] | None = None
    if include_intra and intra_map:
        cluster_summaries = [[] for _ in folder_inputs]
        cluster_summaries_raw = [[] for _ in folder_inputs]
        for idx, (fi, ci) in enumerate(intra_map):
            label = folder_inputs[fi].intra_clusters[ci].label
            cluster_summaries[fi].append((label, intra_clean[idx]))
            cluster_summaries_raw[fi].append((label, intra_raw[idx]))

    # Write output
    from ..output_writer import (
        write_cross_folder_summaries,
        write_cross_folder_summaries_raw,
        write_folder_summaries,
        write_folder_summaries_raw,
    )

    if cross_inputs:
        path = write_cross_folder_summaries(cross_inputs, cross_clean, args.output_dir)
        logger.info(f"Wrote cross-folder summaries to {path}")
        path = write_cross_folder_summaries_raw(cross_inputs, cross_raw, args.output_dir)
        logger.info(f"Wrote cross-folder raw summaries to {path}")

    if folder_inputs:
        path = write_folder_summaries(
            folder_inputs, folder_clean, args.output_dir,
            cluster_summaries=cluster_summaries,
        )
        logger.info(f"Wrote folder summaries to {path}")
        path = write_folder_summaries_raw(
            folder_inputs, folder_raw, args.output_dir,
            cluster_summaries=cluster_summaries_raw,
        )
        logger.info(f"Wrote folder raw summaries to {path}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
