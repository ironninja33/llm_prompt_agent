"""CLI entry point for the image quality assessment experiment."""

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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate image quality using CLIP-IQA+ and/or ImageReward.",
    )
    parser.add_argument(
        "--algorithm", type=str, default="both",
        choices=["clip-iqa+", "imagereward", "both"],
        help="Which algorithm to run (default: both)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for results (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run models on (default: cuda)",
    )

    # Data sampling parameters
    parser.add_argument(
        "--query", type=str, default=None,
        help="Comma-separated keywords to match against prompts (default: random selection)",
    )
    parser.add_argument(
        "--folder", type=str, default=None,
        help="Virtual path or absolute path to filter images (includes subfolders)",
    )
    parser.add_argument(
        "--k", type=int, default=10,
        help="Maximum number of images to sample (default: 10)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducible image sampling (default: auto-generated)",
    )

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Check device availability
    if args.device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                args.device = "cpu"
        except ImportError:
            logger.warning("PyTorch not found, falling back to CPU")
            args.device = "cpu"

    # Initialize database (needed for data sampling)
    from src.models.database import initialize_database
    initialize_database()

    # Sample images from database
    from .data_sampler import sample_images

    logger.info("Sampling up to %d images...", args.k)
    if args.query:
        logger.info("Query: %s", args.query)
    if args.folder:
        logger.info("Folder: %s", args.folder)

    sampled, seed = sample_images(
        query=args.query,
        folder=args.folder,
        k=args.k,
        seed=args.seed,
    )
    logger.info("Seed: %d", seed)

    if not sampled:
        logger.warning("No images to score. Exiting.")
        sys.exit(0)

    # Build assessor list
    from .assessors import ASSESSORS

    if args.algorithm == "both":
        algo_names = list(ASSESSORS.keys())
    else:
        algo_names = [args.algorithm]

    assessors = [ASSESSORS[name]() for name in algo_names]
    logger.info("Algorithms: %s", ", ".join(algo_names))
    logger.info("Device: %s", args.device)

    # Run pipeline
    from .pipeline import run_pipeline

    report = run_pipeline(
        sampled_images=sampled,
        assessors=assessors,
        device=args.device,
    )

    if not report.scores:
        logger.warning("No images were scored.")
        return

    # Write results
    from .output_writer import write_results

    output_path = write_results(
        report, args.output_dir,
        query=args.query, folder=args.folder, k=args.k, seed=seed,
    )
    logger.info("Results written to %s", output_path)


if __name__ == "__main__":
    main()
