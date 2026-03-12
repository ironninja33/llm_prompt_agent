"""Run query strategy experiments.

Usage:
    python -m src.experiments.tool_study.run_experiment \
        --strategy baseline \
        --queries queries.txt \
        --k 10

    python -m src.experiments.tool_study.run_experiment \
        --strategy source_balanced \
        --query "salma hayek elegant gown dramatic lighting" \
        --k 10 \
        --training-ratio 0.5
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from src.models.database import initialize_database
from src.models import vector_store
from src.services import embedding_service, llm_service
from src.models.settings import get_setting

from .strategies import STRATEGIES
from .strategies.baseline import SearchResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
DEFAULT_QUERIES_FILE = str(SCRIPT_DIR / "queries.txt")
DEFAULT_OUTPUT_DIR = str(SCRIPT_DIR / "output")

MIN_BATCH_SIZE = 20


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run query strategy experiments.")
    parser.add_argument(
        "--strategy", type=str, required=True,
        choices=list(STRATEGIES.keys()),
        help=f"Strategy to test: {', '.join(STRATEGIES.keys())}",
    )
    parser.add_argument(
        "--queries", type=str, default=None,
        help=f"Path to queries file (default: {DEFAULT_QUERIES_FILE})",
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="Single query to run (alternative to --queries file)",
    )
    parser.add_argument("--k", type=int, default=10, help="Results per query (default: 10)")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--training-ratio", type=float, default=0.5,
                        help="Training ratio for source_balanced strategy (default: 0.5)")
    parser.add_argument("--k-per-concept", type=int, default=5,
                        help="Results per concept for decomposed strategy (default: 5)")
    parser.add_argument("--similarity-threshold", type=float, default=0.85,
                        help="Similarity threshold for dedup_aware strategy (default: 0.85)")
    parser.add_argument("--top-clusters", type=int, default=6,
                        help="Top intra-folder concepts for cluster_diverse strategy (default: 6)")
    parser.add_argument("--min-slots-per-concept", type=int, default=1,
                        help="Minimum result slots per concept for cluster_diverse strategy (default: 1)")
    parser.add_argument("--discovery-limit", type=int, default=2,
                        help="Max concepts to add from cross-folder discovery (default: 2)")
    parser.add_argument("--no-save", action="store_true", help="Don't save output files")
    return parser.parse_args(argv)


def load_queries(path: str) -> list[str]:
    """Load queries from a text file, skipping comments and blanks."""
    queries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                queries.append(line)
    return queries


def format_result_table(
    query: str,
    strategy_name: str,
    params: dict,
    results: list[SearchResult],
) -> str:
    """Format results into the specified output format."""
    lines = []
    lines.append(f"QUERY: {query}")
    lines.append(f"STRATEGY: {strategy_name}")
    param_str = ", ".join(f"{k}={v}" for k, v in params.items())
    lines.append(f"PARAMS: {param_str}")
    lines.append("---")
    lines.append(f"{'RANK':>4} | {'DISTANCE':>8} | {'SOURCE':8} | {'CONCEPT':30} | TEXT (truncated)")

    for r in results:
        text_trunc = r.text[:80].replace("\n", " ")
        lines.append(
            f"{r.rank:4d} | {r.distance:8.4f} | {r.source:8} | {r.concept[:30]:30} | {text_trunc}"
        )

    lines.append("---")

    # Compute metrics
    training_count = sum(1 for r in results if r.source == "training")
    output_count = sum(1 for r in results if r.source != "training")
    training_distances = [r.distance for r in results if r.source == "training"]
    output_distances = [r.distance for r in results if r.source != "training"]
    unique_concepts = len({r.concept for r in results})

    lines.append("METRICS:")
    lines.append(f"  training_count: {training_count}")
    lines.append(f"  output_count: {output_count}")
    avg_t = sum(training_distances) / len(training_distances) if training_distances else 0
    avg_o = sum(output_distances) / len(output_distances) if output_distances else 0
    lines.append(f"  avg_distance_training: {avg_t:.4f}")
    lines.append(f"  avg_distance_output: {avg_o:.4f}")
    lines.append(f"  unique_concepts: {unique_concepts}")
    lines.append("")

    return "\n".join(lines)


def format_summary(
    strategy_name: str,
    all_results: list[tuple[str, list[SearchResult]]],
) -> str:
    """Format an aggregate summary across all queries."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"AGGREGATE SUMMARY — {strategy_name}")
    lines.append("=" * 70)

    total_training = 0
    total_output = 0
    total_concepts = set()

    for query, results in all_results:
        for r in results:
            if r.source == "training":
                total_training += 1
            else:
                total_output += 1
            total_concepts.add(r.concept)

    total = total_training + total_output
    lines.append(f"  Queries: {len(all_results)}")
    lines.append(f"  Total results: {total}")
    lines.append(f"  Training: {total_training} ({total_training / max(total, 1) * 100:.1f}%)")
    lines.append(f"  Output: {total_output} ({total_output / max(total, 1) * 100:.1f}%)")
    lines.append(f"  Unique concepts: {len(total_concepts)}")

    # Per-query breakdown
    lines.append(f"\n  Per-query training %:")
    for query, results in all_results:
        t = sum(1 for r in results if r.source == "training")
        pct = t / max(len(results), 1) * 100
        marker = " ***" if pct < 30 else ""
        lines.append(f"    {pct:5.1f}% ({t}/{len(results)}) {query[:60]}{marker}")

    return "\n".join(lines)


def main(argv: list[str] | None = None):
    args = parse_args(argv)

    logger.info("Initializing database and vector store...")
    initialize_database()
    vector_store.initialize()

    api_key = get_setting("gemini_api_key")
    if not api_key:
        print("ERROR: Gemini API key required. Set it in the app settings.")
        sys.exit(1)
    llm_service.initialize(api_key)

    # Load queries
    if args.query:
        queries = [args.query]
    elif args.queries:
        queries = load_queries(args.queries)
    else:
        queries = load_queries(DEFAULT_QUERIES_FILE)

    if not queries:
        print("No queries to run.")
        return

    # Build strategy
    strategy_kwargs = {
        "training_ratio": args.training_ratio,
        "k_per_concept": args.k_per_concept,
        "similarity_threshold": args.similarity_threshold,
        "top_clusters": args.top_clusters,
        "min_slots": args.min_slots_per_concept,
        "discovery_limit": args.discovery_limit,
    }
    strategy_cls = STRATEGIES[args.strategy]
    strategy = strategy_cls(**strategy_kwargs)

    params = {"k": args.k}
    params.update({k: v for k, v in strategy_kwargs.items()
                   if k in strategy_cls.__init__.__code__.co_varnames})

    logger.info(f"Strategy: {args.strategy}, Queries: {len(queries)}, k={args.k}")

    # Batch embed all queries using Gemini batch API (min batch size = 20)
    logger.info(f"Embedding {len(queries)} queries in batches of {MIN_BATCH_SIZE}...")
    all_embeddings = []
    for i in range(0, len(queries), MIN_BATCH_SIZE):
        batch = queries[i:i + MIN_BATCH_SIZE]
        embeddings = embedding_service.embed_batch(batch)
        all_embeddings.extend(embeddings)
        logger.info(f"  Embedded batch {i // MIN_BATCH_SIZE + 1} ({len(batch)} queries)")

    # Run experiments
    all_results: list[tuple[str, list[SearchResult]]] = []
    all_output_text = []

    for i, (query, embedding) in enumerate(zip(queries, all_embeddings)):
        logger.info(f"  Running query {i + 1}/{len(queries)}: {query[:50]}...")
        results = strategy.search(
            query_embedding=embedding,
            k=args.k,
            query_text=query,
        )
        all_results.append((query, results))

        table = format_result_table(query, args.strategy, params, results)
        all_output_text.append(table)
        print(table)

    # Summary
    summary = format_summary(args.strategy, all_results)
    all_output_text.append(summary)
    print(summary)

    # Save output
    if not args.no_save:
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{args.strategy}_{timestamp}.txt"
        output_path = os.path.join(args.output_dir, filename)
        with open(output_path, "w") as f:
            f.write("\n\n".join(all_output_text))
        logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
