"""Analyze refinement query redundancy.

Needs Gemini API (for embeddings). For each chat, reconstructs the pattern:
agent generates prompts → agent searches again, and computes similarity
between refinement queries and recently generated prompts.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict

import numpy as np
from sqlalchemy import text

from src.models.database import get_db, initialize_database, row_to_dict
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
    "search_diverse_prompts",
}

MIN_BATCH_SIZE = 20


def get_chat_timeline(conn, chat_id: str) -> list[dict]:
    """Get interleaved messages and tool calls for a chat, in chronological order."""
    messages = conn.execute(text("""
        SELECT id, role, content, created_at
        FROM messages
        WHERE chat_id = :chat_id
        ORDER BY created_at, id
    """), {"chat_id": chat_id}).fetchall()

    tool_calls = conn.execute(text("""
        SELECT tc.id as tc_id, tc.tool_name, tc.parameters, tc.response_summary,
               tc.sequence, m.id as message_id, m.created_at
        FROM tool_calls tc
        JOIN messages m ON tc.message_id = m.id
        WHERE m.chat_id = :chat_id
        ORDER BY m.created_at, tc.sequence
    """), {"chat_id": chat_id}).fetchall()

    return (
        [row_to_dict(m) for m in messages],
        [row_to_dict(tc) for tc in tool_calls],
    )


def extract_refinement_sessions(conn) -> list[dict]:
    """Find chats where the agent searched, generated, then searched again."""
    chats = conn.execute(text("""
        SELECT DISTINCT m.chat_id
        FROM tool_calls tc
        JOIN messages m ON tc.message_id = m.id
        WHERE tc.tool_name IN ('search_similar_prompts', 'query_themed_prompts', 'search_diverse_prompts')
        GROUP BY m.chat_id
        HAVING COUNT(*) >= 2
    """)).fetchall()

    sessions = []
    for row in chats:
        chat_id = row._mapping["chat_id"]
        messages, tool_calls = get_chat_timeline(conn, chat_id)

        # Find search queries
        search_queries = []
        for tc in tool_calls:
            if tc["tool_name"] not in SEARCH_TOOLS:
                continue
            try:
                params = json.loads(tc["parameters"]) if isinstance(tc["parameters"], str) else tc["parameters"]
            except (json.JSONDecodeError, TypeError):
                continue
            query = params.get("query", "").strip()
            if query:
                search_queries.append({
                    "query": query,
                    "tool": tc["tool_name"],
                    "message_id": tc["message_id"],
                })

        # Find assistant-generated prompts between search calls
        # Look for assistant messages that contain prompt-like content
        assistant_outputs = []
        for msg in messages:
            if msg["role"] == "assistant" and msg["content"]:
                content = msg["content"]
                # Heuristic: if content is long and contains prompt-like text
                if len(content) > 100:
                    assistant_outputs.append({
                        "message_id": msg["id"],
                        "content": content[:500],
                    })

        if len(search_queries) >= 2 and assistant_outputs:
            sessions.append({
                "chat_id": chat_id,
                "search_queries": search_queries,
                "assistant_outputs": assistant_outputs,
            })

    return sessions


def compute_similarities(sessions: list[dict]) -> list[dict]:
    """Compute cosine similarity between refinement queries and preceding outputs."""
    # Collect all texts that need embedding
    all_texts = []
    text_map = []  # (session_idx, type, sub_idx)

    for si, session in enumerate(sessions):
        for qi, sq in enumerate(session["search_queries"]):
            all_texts.append(sq["query"])
            text_map.append((si, "query", qi))
        for ai, ao in enumerate(session["assistant_outputs"]):
            # Truncate long outputs for embedding
            all_texts.append(ao["content"][:500])
            text_map.append((si, "output", ai))

    if not all_texts:
        return []

    # Batch embed all texts
    logger.info(f"Embedding {len(all_texts)} texts in batches of {MIN_BATCH_SIZE}...")
    all_embeddings = []
    for i in range(0, len(all_texts), MIN_BATCH_SIZE):
        batch = all_texts[i:i + MIN_BATCH_SIZE]
        embeddings = embedding_service.embed_batch(batch)
        all_embeddings.extend(embeddings)
        logger.info(f"  Embedded batch {i // MIN_BATCH_SIZE + 1}")

    # Map embeddings back to sessions
    for idx, (si, text_type, sub_idx) in enumerate(text_map):
        emb = all_embeddings[idx]
        if text_type == "query":
            sessions[si]["search_queries"][sub_idx]["embedding"] = emb
        else:
            sessions[si]["assistant_outputs"][sub_idx]["embedding"] = emb

    # Compute similarities within each session
    results = []
    for session in sessions:
        queries_with_emb = [q for q in session["search_queries"] if "embedding" in q]
        outputs_with_emb = [o for o in session["assistant_outputs"] if "embedding" in o]

        if len(queries_with_emb) < 2 or not outputs_with_emb:
            continue

        # For refinement queries (all after the first), compare to preceding outputs
        for qi in range(1, len(queries_with_emb)):
            refinement_q = queries_with_emb[qi]
            ref_emb = np.array(refinement_q["embedding"])

            # Compare to all assistant outputs
            max_sim = 0.0
            most_similar_output = ""
            for out in outputs_with_emb:
                out_emb = np.array(out["embedding"])
                # Cosine similarity
                dot = np.dot(ref_emb, out_emb)
                norm = np.linalg.norm(ref_emb) * np.linalg.norm(out_emb)
                sim = float(dot / norm) if norm > 0 else 0.0
                if sim > max_sim:
                    max_sim = sim
                    most_similar_output = out["content"][:200]

            # Also compare to the preceding search query
            prev_q = queries_with_emb[qi - 1]
            prev_emb = np.array(prev_q["embedding"])
            dot = np.dot(ref_emb, prev_emb)
            norm = np.linalg.norm(ref_emb) * np.linalg.norm(prev_emb)
            query_sim = float(dot / norm) if norm > 0 else 0.0

            results.append({
                "chat_id": session["chat_id"],
                "previous_query": prev_q["query"],
                "refinement_query": refinement_q["query"],
                "query_to_query_similarity": round(query_sim, 4),
                "query_to_output_similarity": round(max_sim, 4),
                "most_similar_output": most_similar_output,
                "is_redundant": max_sim > 0.85,
            })

    return results


def print_report(results: list[dict]):
    """Print refinement redundancy report."""
    print("\n" + "=" * 70)
    print("REFINEMENT REDUNDANCY ANALYSIS")
    print("=" * 70)

    if not results:
        print("\nNo refinement patterns found.")
        return

    redundant = [r for r in results if r["is_redundant"]]
    print(f"\n--- Overview ({len(results)} refinement pairs) ---")
    print(f"  Redundant (>0.85 similarity to output): {len(redundant)}")
    print(f"  Redundancy rate: {len(redundant) / len(results) * 100:.1f}%")

    avg_q2q = sum(r["query_to_query_similarity"] for r in results) / len(results)
    avg_q2o = sum(r["query_to_output_similarity"] for r in results) / len(results)
    print(f"  Avg query-to-query similarity: {avg_q2q:.4f}")
    print(f"  Avg query-to-output similarity: {avg_q2o:.4f}")

    print(f"\n--- Redundant Refinements ---")
    for r in sorted(redundant, key=lambda x: -x["query_to_output_similarity"]):
        print(f"\n  Chat: {r['chat_id'][:8]}...")
        print(f"    Prev query:   {r['previous_query'][:70]}")
        print(f"    Refine query: {r['refinement_query'][:70]}")
        print(f"    Q→Q sim: {r['query_to_query_similarity']:.4f}  Q→Output sim: {r['query_to_output_similarity']:.4f}")
        print(f"    Similar output: {r['most_similar_output'][:100]}...")

    print(f"\n--- All Refinement Pairs ---")
    for r in results:
        tag = " *** REDUNDANT ***" if r["is_redundant"] else ""
        print(f"  Q→Q: {r['query_to_query_similarity']:.4f}  Q→O: {r['query_to_output_similarity']:.4f}{tag}")
        print(f"    {r['previous_query'][:50]} → {r['refinement_query'][:50]}")


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Analyze refinement query redundancy.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of sessions to analyze")
    args = parser.parse_args(argv)

    logger.info("Initializing database...")
    initialize_database()

    api_key = get_setting("gemini_api_key")
    if not api_key:
        print("ERROR: Gemini API key required. Set it in the app settings.")
        return
    llm_service.initialize(api_key)

    with get_db() as conn:
        logger.info("Extracting refinement sessions...")
        sessions = extract_refinement_sessions(conn)

    if not sessions:
        print("No refinement sessions found.")
        return

    if args.limit:
        sessions = sessions[:args.limit]

    logger.info(f"Found {len(sessions)} sessions with refinement patterns.")
    results = compute_similarities(sessions)
    print_report(results)


if __name__ == "__main__":
    main()
