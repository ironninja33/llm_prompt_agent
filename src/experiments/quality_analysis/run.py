"""CLI entry point for the bad quality concept analysis experiment."""

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
DEFAULT_OUTPUT_DIR = str(SCRIPT_DIR / "output")
VALID_REASONS = {"quality", "wrong_direction", "duplicate", "space"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Extract concepts from deleted prompts and compare frequencies "
                    "against surviving prompts to find what causes bad generations.",
    )

    # Data selection
    parser.add_argument(
        "--base-model", type=str, default=None,
        help="Base model to filter by (default: most recent model used in a generation)",
    )
    parser.add_argument(
        "--reasons", type=str, default="quality",
        help="Comma-separated deletion reasons to include (default: quality). "
             f"Valid: {', '.join(sorted(VALID_REASONS))}",
    )

    # vLLM model flags
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
        "--gpu-memory-utilization", type=float, default=0.85,
        help="Fraction of GPU memory for vLLM (default: 0.85)",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=8192,
        help="Maximum model context length in tokens (default: 8192)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=1024,
        help="Maximum output tokens per response (default: 1024)",
    )
    parser.add_argument(
        "--no-think", action="store_true", default=False,
        help="Disable model thinking/reasoning (appends /no_think to prompts)",
    )

    # Analysis flags
    parser.add_argument(
        "--min-bad-count", type=int, default=2,
        help="Minimum occurrences in bad set to report a tag (default: 2)",
    )
    parser.add_argument(
        "--top-n", type=int, default=50,
        help="Number of top over-represented tags to show (default: 50)",
    )

    # Output
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for results (default: {DEFAULT_OUTPUT_DIR})",
    )

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Parse and validate reasons
    reasons = [r.strip() for r in args.reasons.split(",") if r.strip()]
    invalid = set(reasons) - VALID_REASONS
    if invalid:
        logger.error("Invalid reasons: %s. Valid: %s", invalid, VALID_REASONS)
        sys.exit(1)
    if not reasons:
        logger.error("No reasons specified.")
        sys.exit(1)

    # Initialize database only (no vector store or Gemini needed)
    from src.models.database import initialize_database
    initialize_database()

    # Resolve base model
    if args.base_model:
        base_model = args.base_model
    else:
        from src.models.generation import get_latest_job_settings
        settings = get_latest_job_settings()
        if settings is None or not settings.get("base_model"):
            logger.error("No completed jobs found; cannot determine base model. Use --base-model.")
            sys.exit(1)
        base_model = settings["base_model"]

    logger.info("Base model: %s", base_model)
    logger.info("Reasons: %s", ", ".join(reasons))
    logger.info("LLM model: %s", args.model)

    # Fetch data
    from .data import fetch_deleted_prompts, fetch_surviving_prompts, fetch_folder_deletion_rates

    deleted = fetch_deleted_prompts(base_model, reasons)
    if not deleted:
        logger.warning("No deletions found for model '%s' with reasons %s. Exiting.", base_model, reasons)
        sys.exit(0)

    bad_texts = [d.positive_prompt for d in deleted]
    logger.info("Found %d unique deleted prompts", len(bad_texts))

    surviving_texts = fetch_surviving_prompts(base_model)
    logger.info("Found %d unique surviving prompts", len(surviving_texts))

    folder_rates = fetch_folder_deletion_rates(base_model, reasons)
    logger.info("Found %d folders with deletions", len(folder_rates))

    # Extract concepts from both sets in one model load
    from .concept_extraction import (
        SYSTEM_PROMPT, build_extraction_prompt, parse_extraction_response,
    )
    from src.experiments.common import vllm_engine

    all_prompts = [build_extraction_prompt(p) for p in bad_texts + surviving_texts]
    n_bad = len(bad_texts)

    logger.info("Extracting concepts from %d prompts (%d bad + %d surviving)...",
                len(all_prompts), n_bad, len(surviving_texts))

    llm = vllm_engine.load_model(
        model_id=args.model,
        quantization=args.quantization,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    try:
        clean_results, _ = vllm_engine.batch_generate(
            llm, all_prompts, SYSTEM_PROMPT,
            max_tokens=args.max_tokens, no_think=args.no_think,
        )
    finally:
        vllm_engine.unload_model(llm)

    # Parse responses
    bad_concepts = [
        parse_extraction_response(r, t)
        for r, t in zip(clean_results[:n_bad], bad_texts)
    ]
    surviving_concepts = [
        parse_extraction_response(r, t)
        for r, t in zip(clean_results[n_bad:], surviving_texts)
    ]

    bad_parsed = sum(1 for c in bad_concepts if c.tags)
    surv_parsed = sum(1 for c in surviving_concepts if c.tags)
    logger.info("Parsed: %d/%d bad, %d/%d surviving", bad_parsed, n_bad, surv_parsed, len(surviving_texts))

    # Frequency analysis
    from .frequency_analysis import compute_tag_frequencies

    result = compute_tag_frequencies(
        bad_concepts, surviving_concepts,
        base_model=base_model,
        reasons=reasons,
        folder_rates=folder_rates,
        min_bad_count=args.min_bad_count,
    )

    # Write report
    from .output_writer import write_report

    output_path = write_report(result, args.output_dir, top_n=args.top_n)
    logger.info("Report written to %s", output_path)


if __name__ == "__main__":
    main()
