"""Analyze output-data bias in search results.

Needs Gemini API (for embeddings). Re-executes real agent queries against
ChromaDB and measures training vs output result ratios.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict

from sqlalchemy import text

from src.models.database import get_db, initialize_database, row_to_dict
from src.models import vector_store
from src.services import embedding_service, llm_service
from src.models.settings import get_setting

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SEARCH_TOOLS = {
    "search_similar_prompts",
    "query_themed_prompts",
}

MIN_BATCH_SIZE = 20


def extract_queries_from_db(conn, limit: int | None = None) -> list[str]:
    """Extract unique query strings from search tool calls."""
    result = conn.execute(text("""
        SELECT DISTINCT tc.parameters
        FROM tool_calls tc
        WHERE tc.tool_name IN ('search_similar_prompts', 'query_themed_prompts')
    """))

    queries = []
    seen = set()
    for row in result.fetchall():
        params_str = row._mapping["parameters"]
        try:
            params = json.loads(params_str) if isinstance(params_str, str) else params_str
        except (json.JSONDecodeError, TypeError):
            continue
        query = params.get("query", "").strip()
        if query and query.lower() not in seen:
            seen.add(query.lower())
            queries.append(query)

    if limit:
        queries = queries[:limit]
    return queries


def analyze_query_bias(
    query: str,
    query_embedding: list[float],
    k: int = 10,
) -> dict:
    """Run a query against ChromaDB and measure training/output ratio."""
    # Search without source_type filter (how the agent usually queries)
    results = vector_store.search_similar(query_embedding, k=k, source_type=None)

    training_results = []
    output_results = []
    for r in results:
        source = r["metadata"].get("dir_type", "")
        if source == "training":
            training_results.append(r)
        else:
            output_results.append(r)

    training_distances = [r["distance"] for r in training_results]
    output_distances = [r["distance"] for r in output_results]

    return {
        "query": query,
        "total_results": len(results),
        "training_count": len(training_results),
        "output_count": len(output_results),
        "output_pct": round(len(output_results) / max(len(results), 1) * 100, 1),
        "avg_distance_training": round(sum(training_distances) / max(len(training_distances), 1), 4) if training_distances else None,
        "avg_distance_output": round(sum(output_distances) / max(len(output_distances), 1), 4) if output_distances else None,
        "results": [
            {
                "rank": i + 1,
                "distance": round(r["distance"], 4),
                "source": r["metadata"].get("dir_type", ""),
                "concept": r["metadata"].get("concept", ""),
                "text": r["document"][:100],
            }
            for i, r in enumerate(results)
        ],
    }


def embed_queries_batch(queries: list[str]) -> list[list[float]]:
    """Embed all queries using Gemini batch API with minimum batch size of 20."""
    all_embeddings = []
    for i in range(0, len(queries), MIN_BATCH_SIZE):
        batch = queries[i:i + MIN_BATCH_SIZE]
        # Pad to minimum batch size if needed (last batch)
        if len(batch) < MIN_BATCH_SIZE and i + MIN_BATCH_SIZE < len(queries) + MIN_BATCH_SIZE:
            embeddings = embedding_service.embed_batch(batch)
        else:
            embeddings = embedding_service.embed_batch(batch)
        all_embeddings.extend(embeddings[:len(batch)])
        logger.info(f"  Embedded batch {i // MIN_BATCH_SIZE + 1} ({len(batch)} queries)")
    return all_embeddings


def print_report(analyses: list[dict]):
    """Print bias analysis report."""
    print("\n" + "=" * 70)
    print("OUTPUT BIAS ANALYSIS")
    print("=" * 70)

    high_bias = [a for a in analyses if a["output_pct"] > 70]
    moderate_bias = [a for a in analyses if 50 < a["output_pct"] <= 70]
    balanced = [a for a in analyses if a["output_pct"] <= 50]

    print(f"\n--- Overview ({len(analyses)} queries analyzed) ---")
    print(f"  High output bias (>70%):   {len(high_bias)}")
    print(f"  Moderate bias (50-70%):    {len(moderate_bias)}")
    print(f"  Balanced (<=50%):          {len(balanced)}")
    avg_output_pct = sum(a["output_pct"] for a in analyses) / max(len(analyses), 1)
    print(f"  Average output %:          {avg_output_pct:.1f}%")

    print(f"\n--- High Bias Queries (>70% output) ---")
    for a in sorted(high_bias, key=lambda x: -x["output_pct"]):
        print(f"\n  Query: {a['query'][:70]}")
        print(f"    Output: {a['output_count']}/{a['total_results']} ({a['output_pct']}%)")
        print(f"    Avg distance — training: {a['avg_distance_training']}  output: {a['avg_distance_output']}")

    print(f"\n--- Per-Query Details ---")
    for a in analyses:
        marker = " *** HIGH BIAS ***" if a["output_pct"] > 70 else ""
        print(f"\n  Query: {a['query'][:70]}{marker}")
        print(f"    Training: {a['training_count']}  Output: {a['output_count']}  ({a['output_pct']}% output)")
        for r in a["results"]:
            src_tag = "T" if r["source"] == "training" else "O"
            print(f"      {r['rank']:2d}. [{src_tag}] d={r['distance']:.4f} [{r['concept'][:25]}] {r['text'][:60]}")


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Analyze output-data bias in search results.")
    parser.add_argument("--limit", type=int, default=None, help="Limit queries to analyze")
    parser.add_argument("--k", type=int, default=10, help="Results per query (default: 10)")
    parser.add_argument("--queries-file", type=str, default=None,
                        help="Read queries from file instead of database")
    args = parser.parse_args(argv)

    logger.info("Initializing database and vector store...")
    initialize_database()
    vector_store.initialize()

    api_key = get_setting("gemini_api_key")
    if not api_key:
        print("ERROR: Gemini API key required. Set it in the app settings.")
        return
    llm_service.initialize(api_key)

    if args.queries_file:
        with open(args.queries_file) as f:
            queries = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    else:
        with get_db() as conn:
            queries = extract_queries_from_db(conn, limit=args.limit)

    if not queries:
        print("No queries found.")
        return

    logger.info(f"Analyzing {len(queries)} queries...")

    # Batch embed all queries
    logger.info("Embedding queries in batches...")
    embeddings = embed_queries_batch(queries)

    analyses = []
    for i, (query, embedding) in enumerate(zip(queries, embeddings)):
        logger.info(f"  Analyzing query {i + 1}/{len(queries)}: {query[:50]}...")
        analysis = analyze_query_bias(query, embedding, k=args.k)
        analyses.append(analysis)

    print_report(analyses)


if __name__ == "__main__":
    main()
