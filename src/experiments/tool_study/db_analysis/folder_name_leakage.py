"""Detect folder name leakage in agent tool calls and responses.

Needs ChromaDB initialized (no API key). Scans tool call parameters and
assistant message content for folder names and underscore-joined display names.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter, defaultdict

from sqlalchemy import text

from src.models.database import get_db, initialize_database, row_to_dict
from src.models import vector_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_concept_names() -> list[dict]:
    """Load all concept names from ChromaDB."""
    concepts = vector_store.list_concepts()
    enriched = []
    for c in concepts:
        name = c["concept"]
        parts = name.split("__", 1)
        category = parts[0] if len(parts) > 1 else ""
        display = parts[1] if len(parts) > 1 else name
        enriched.append({
            "full_name": name,
            "category": category,
            "display_name": display,
            "underscore_display": display,  # already underscored
            "source_type": c["source_type"],
            "count": c["count"],
        })
    return enriched


def build_patterns(concepts: list[dict]) -> list[dict]:
    """Build regex patterns for detecting folder name leakage."""
    patterns = []
    seen = set()
    for c in concepts:
        full = c["full_name"]
        display = c["display_name"]

        # Full name pattern: category__display_name
        if full not in seen and "__" in full:
            patterns.append({
                "pattern": re.compile(re.escape(full), re.IGNORECASE),
                "name": full,
                "type": "full_name",
            })
            seen.add(full)

        # Underscore display name (e.g., "against_wall")
        if display not in seen and "_" in display:
            patterns.append({
                "pattern": re.compile(r"\b" + re.escape(display) + r"\b", re.IGNORECASE),
                "name": display,
                "type": "underscore_display",
            })
            seen.add(display)

    return patterns


def scan_tool_parameters(conn, patterns: list[dict]) -> list[dict]:
    """Scan tool call parameters for folder name patterns."""
    result = conn.execute(text("""
        SELECT tc.id, tc.tool_name, tc.parameters, m.chat_id
        FROM tool_calls tc
        JOIN messages m ON tc.message_id = m.id
    """))
    rows = result.fetchall()

    matches = []
    for row in rows:
        r = row._mapping
        params_str = r["parameters"] or ""
        if isinstance(params_str, str):
            try:
                params = json.loads(params_str)
                # Scan all string values in params
                text_to_scan = " ".join(
                    str(v) for v in params.values() if isinstance(v, str)
                )
            except (json.JSONDecodeError, TypeError):
                text_to_scan = params_str
        else:
            text_to_scan = str(params_str)

        for p in patterns:
            if p["pattern"].search(text_to_scan):
                matches.append({
                    "location": "tool_parameters",
                    "tool_call_id": r["id"],
                    "tool_name": r["tool_name"],
                    "chat_id": r["chat_id"],
                    "matched_name": p["name"],
                    "match_type": p["type"],
                    "context": text_to_scan[:200],
                })
    return matches


def scan_assistant_messages(conn, patterns: list[dict]) -> list[dict]:
    """Scan assistant message content for folder name patterns."""
    result = conn.execute(text("""
        SELECT id, chat_id, content
        FROM messages
        WHERE role = 'assistant' AND content IS NOT NULL
    """))
    rows = result.fetchall()

    matches = []
    for row in rows:
        r = row._mapping
        content = r["content"] or ""
        for p in patterns:
            found = p["pattern"].findall(content)
            if found:
                # Find context around first match
                m = p["pattern"].search(content)
                start = max(0, m.start() - 40)
                end = min(len(content), m.end() + 40)
                context = content[start:end]

                matches.append({
                    "location": "assistant_message",
                    "message_id": r["id"],
                    "chat_id": r["chat_id"],
                    "matched_name": p["name"],
                    "match_type": p["type"],
                    "occurrences": len(found),
                    "context": context,
                })
    return matches


def print_report(
    concepts: list[dict],
    param_matches: list[dict],
    msg_matches: list[dict],
):
    """Print leakage analysis report."""
    print("\n" + "=" * 70)
    print("FOLDER NAME LEAKAGE ANALYSIS")
    print("=" * 70)

    print(f"\n--- Concept Names Loaded: {len(concepts)} ---")
    with_underscore = [c for c in concepts if "_" in c["display_name"]]
    print(f"  With underscores in display name: {len(with_underscore)}")

    # Parameter leakage
    print(f"\n--- Tool Parameter Leakage ({len(param_matches)} matches) ---")
    by_name = Counter(m["matched_name"] for m in param_matches)
    by_type = Counter(m["match_type"] for m in param_matches)
    print(f"  By match type: {dict(by_type)}")
    print(f"\n  Top leaked names in parameters:")
    for name, count in by_name.most_common(20):
        print(f"    {name:40s} {count:>4d}")

    by_tool = Counter(m["tool_name"] for m in param_matches)
    print(f"\n  By tool:")
    for tool, count in by_tool.most_common():
        print(f"    {tool:40s} {count:>4d}")

    # Message leakage
    print(f"\n--- Assistant Message Leakage ({len(msg_matches)} matches) ---")
    by_name_msg = Counter(m["matched_name"] for m in msg_matches)
    by_type_msg = Counter(m["match_type"] for m in msg_matches)
    print(f"  By match type: {dict(by_type_msg)}")
    print(f"\n  Top leaked names in assistant messages:")
    for name, count in by_name_msg.most_common(20):
        print(f"    {name:40s} {count:>4d}")

    # Show some example contexts
    print(f"\n--- Example Leakage Contexts ---")
    shown = set()
    for m in msg_matches[:20]:
        key = m["matched_name"]
        if key in shown:
            continue
        shown.add(key)
        print(f"\n  Name: {m['matched_name']} ({m['match_type']})")
        print(f"  Context: ...{m['context']}...")

    # Summary
    all_leaked = set(m["matched_name"] for m in param_matches) | set(m["matched_name"] for m in msg_matches)
    print(f"\n--- Summary ---")
    print(f"  Unique names leaked in parameters: {len(set(m['matched_name'] for m in param_matches))}")
    print(f"  Unique names leaked in messages: {len(set(m['matched_name'] for m in msg_matches))}")
    print(f"  Total unique leaked names: {len(all_leaked)}")


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Detect folder name leakage in agent behavior.")
    parser.parse_args(argv)

    logger.info("Initializing database and vector store...")
    initialize_database()
    vector_store.initialize()

    logger.info("Loading concept names from ChromaDB...")
    concepts = load_concept_names()
    logger.info(f"Loaded {len(concepts)} concept entries.")

    patterns = build_patterns(concepts)
    logger.info(f"Built {len(patterns)} search patterns.")

    with get_db() as conn:
        logger.info("Scanning tool parameters...")
        param_matches = scan_tool_parameters(conn, patterns)
        logger.info(f"Found {len(param_matches)} parameter matches.")

        logger.info("Scanning assistant messages...")
        msg_matches = scan_assistant_messages(conn, patterns)
        logger.info(f"Found {len(msg_matches)} message matches.")

    print_report(concepts, param_matches, msg_matches)


if __name__ == "__main__":
    main()
