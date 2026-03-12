"""Analyze agent tool call patterns from the database.

No API key needed — pure SQLite analysis.

Outputs:
- Tool frequency distribution
- Query strings extracted from search tool parameters
- Per-chat tool call sequences (detecting refinement patterns)
- source_type filter usage stats
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict

from sqlalchemy import text

from src.models.database import get_db, initialize_database, row_to_dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SEARCH_TOOLS = {
    "search_similar_prompts",
    "search_diverse_prompts",
    "get_opposite_prompts",
    "query_themed_prompts",
    "query_dataset_map",
    "get_folder_themes",
}


def get_all_tool_calls(conn) -> list[dict]:
    """Fetch all tool calls joined with message and chat info."""
    result = conn.execute(text("""
        SELECT
            tc.id,
            tc.tool_name,
            tc.parameters,
            tc.response_summary,
            tc.sequence,
            tc.iteration,
            tc.created_at,
            m.chat_id,
            m.role,
            m.content as message_content
        FROM tool_calls tc
        JOIN messages m ON tc.message_id = m.id
        ORDER BY m.chat_id, tc.created_at, tc.sequence
    """))
    return [row_to_dict(r) for r in result.fetchall()]


def analyze_tool_frequency(calls: list[dict]) -> dict[str, int]:
    """Count how often each tool is called."""
    counter = Counter(c["tool_name"] for c in calls)
    return dict(counter.most_common())


def extract_query_strings(calls: list[dict]) -> list[dict]:
    """Extract query strings from search tool parameters."""
    queries = []
    for c in calls:
        if c["tool_name"] not in SEARCH_TOOLS:
            continue
        try:
            params = json.loads(c["parameters"]) if isinstance(c["parameters"], str) else c["parameters"]
        except (json.JSONDecodeError, TypeError):
            continue

        query = params.get("query", "")
        if not query:
            continue

        queries.append({
            "tool": c["tool_name"],
            "query": query,
            "source_type": params.get("source_type"),
            "concept": params.get("concept"),
            "k": params.get("k"),
            "chat_id": c["chat_id"],
        })
    return queries


def analyze_source_type_usage(queries: list[dict]) -> dict:
    """Analyze how often source_type filters are used."""
    total = len(queries)
    if total == 0:
        return {"total": 0}

    by_filter = Counter(q.get("source_type") or "none" for q in queries)
    return {
        "total": total,
        "by_source_type": dict(by_filter.most_common()),
        "pct_no_filter": round(by_filter.get("none", 0) / total * 100, 1),
    }


def analyze_per_chat_sequences(calls: list[dict]) -> list[dict]:
    """Group tool calls by chat and detect refinement patterns.

    A refinement pattern is: search tool → (assistant generates) → search tool again.
    """
    by_chat: dict[str, list[dict]] = defaultdict(list)
    for c in calls:
        by_chat[c["chat_id"]].append(c)

    chat_summaries = []
    for chat_id, chat_calls in by_chat.items():
        tool_sequence = [c["tool_name"] for c in chat_calls]
        search_calls = [c for c in chat_calls if c["tool_name"] in SEARCH_TOOLS]

        # Detect refinement: consecutive search calls in the same chat
        refinement_pairs = []
        for i in range(1, len(search_calls)):
            prev = search_calls[i - 1]
            curr = search_calls[i]
            try:
                prev_params = json.loads(prev["parameters"]) if isinstance(prev["parameters"], str) else prev["parameters"]
                curr_params = json.loads(curr["parameters"]) if isinstance(curr["parameters"], str) else curr["parameters"]
            except (json.JSONDecodeError, TypeError):
                continue

            prev_query = prev_params.get("query", "")
            curr_query = curr_params.get("query", "")
            if prev_query and curr_query:
                refinement_pairs.append({
                    "first_tool": prev["tool_name"],
                    "first_query": prev_query,
                    "second_tool": curr["tool_name"],
                    "second_query": curr_query,
                })

        chat_summaries.append({
            "chat_id": chat_id,
            "total_calls": len(chat_calls),
            "search_calls": len(search_calls),
            "tool_sequence": tool_sequence,
            "refinement_pairs": refinement_pairs,
        })

    chat_summaries.sort(key=lambda x: x["total_calls"], reverse=True)
    return chat_summaries


def print_report(
    freq: dict,
    queries: list[dict],
    source_usage: dict,
    chat_sequences: list[dict],
):
    """Print a formatted analysis report."""
    print("\n" + "=" * 70)
    print("TOOL CALL PATTERN ANALYSIS")
    print("=" * 70)

    print("\n--- Tool Frequency Distribution ---")
    for tool, count in freq.items():
        print(f"  {tool:40s} {count:>5d}")

    print(f"\n--- Extracted Queries ({len(queries)} total) ---")
    for i, q in enumerate(queries[:50], 1):
        src = q["source_type"] or "any"
        concept = f" [concept={q['concept']}]" if q.get("concept") else ""
        print(f"  {i:3d}. [{q['tool']:30s}] (src={src}{concept}) {q['query'][:80]}")
    if len(queries) > 50:
        print(f"  ... and {len(queries) - 50} more")

    print(f"\n--- source_type Filter Usage ---")
    print(f"  Total search calls: {source_usage['total']}")
    for k, v in source_usage.get("by_source_type", {}).items():
        print(f"    {k}: {v}")
    print(f"  No filter: {source_usage.get('pct_no_filter', 0)}%")

    print(f"\n--- Per-Chat Sequences ({len(chat_sequences)} chats) ---")
    total_refinements = 0
    for cs in chat_sequences:
        n_ref = len(cs["refinement_pairs"])
        total_refinements += n_ref
        if n_ref > 0:
            print(f"\n  Chat {cs['chat_id'][:8]}... ({cs['total_calls']} calls, {cs['search_calls']} searches, {n_ref} refinements)")
            for rp in cs["refinement_pairs"]:
                print(f"    {rp['first_tool']:30s} → {rp['second_tool']}")
                print(f"      Q1: {rp['first_query'][:70]}")
                print(f"      Q2: {rp['second_query'][:70]}")

    print(f"\n  Total refinement pairs across all chats: {total_refinements}")

    # Export unique queries for queries.txt
    print(f"\n--- Unique Queries (for queries.txt) ---")
    seen = set()
    unique_queries = []
    for q in queries:
        normalized = q["query"].strip().lower()
        if normalized not in seen:
            seen.add(normalized)
            unique_queries.append(q["query"].strip())
    for i, uq in enumerate(unique_queries, 1):
        print(f"  {i:3d}. {uq}")
    print(f"\n  Total unique queries: {len(unique_queries)}")


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Analyze agent tool call patterns.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tool calls to analyze")
    args = parser.parse_args(argv)

    logger.info("Initializing database...")
    initialize_database()

    with get_db() as conn:
        logger.info("Fetching tool calls...")
        calls = get_all_tool_calls(conn)

    if not calls:
        print("No tool calls found in database.")
        return

    logger.info(f"Found {len(calls)} tool calls.")
    if args.limit:
        calls = calls[:args.limit]

    freq = analyze_tool_frequency(calls)
    queries = extract_query_strings(calls)
    source_usage = analyze_source_type_usage(queries)
    chat_sequences = analyze_per_chat_sequences(calls)

    print_report(freq, queries, source_usage, chat_sequences)


if __name__ == "__main__":
    main()
