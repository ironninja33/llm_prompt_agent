"""Tool result summarizers for context truncation.

Each summarizer takes (args, result) and returns a compact dict.
Used by context truncation to replace verbose tool results with summaries.
"""


def _unique_values(items: list[dict], key: str, limit: int = 3) -> list[str]:
    """Extract unique values for *key* from a list of dicts, up to *limit*."""
    seen = []
    for item in items:
        val = item.get(key, "")
        if val and val not in seen:
            seen.append(val)
            if len(seen) >= limit:
                break
    return seen


def _summarize_search_similar(args: dict, result: dict) -> dict:
    summary = {"query": args.get("query", ""), "count": result.get("count", 0)}
    if args.get("source_type"):
        summary["source_type"] = args["source_type"]
    if args.get("concept"):
        summary["concept"] = args["concept"]
    summary["top_concepts"] = _unique_values(result.get("prompts", []), "concept")
    if result.get("warning"):
        summary["warning"] = result["warning"]
    return summary


def _summarize_search_diverse(args: dict, result: dict) -> dict:
    return {
        "query": args.get("query", ""),
        "count": result.get("count", 0),
        "top_concepts": _unique_values(result.get("prompts", []), "concept"),
    }


def _summarize_get_random(args: dict, result: dict) -> dict:
    return {
        "count": result.get("count", 0),
        "top_concepts": _unique_values(result.get("prompts", []), "concept"),
    }


def _summarize_get_opposite(args: dict, result: dict) -> dict:
    return {
        "query": args.get("query", ""),
        "count": result.get("count", 0),
        "top_concepts": _unique_values(result.get("prompts", []), "concept"),
    }


def _summarize_list_concepts(args: dict, result: dict) -> dict:
    return {"count": result.get("count", 0)}


def _summarize_dataset_overview(args: dict, result: dict) -> dict:
    return {
        "total_prompts": result.get("total_prompts", 0),
        "folder_count": len(result.get("folders", [])),
        "cross_theme_count": len(result.get("cross_folder_themes", [])),
    }


def _summarize_folder_themes(args: dict, result: dict) -> dict:
    themes = result.get("themes", [])
    return {
        "folder": args.get("folder_name", ""),
        "source_type": args.get("source_type", ""),
        "theme_count": len(themes),
        "top_themes": [t.get("label", "") for t in themes[:5]],
    }


def _summarize_query_themed(args: dict, result: dict) -> dict:
    summary = {"query": args.get("query", "")}
    for section in ("similar", "intra_folder_themes", "cross_folder_themes", "random", "opposite"):
        data = result.get(section)
        if data and isinstance(data, dict):
            summary[section] = data.get("count", len(data.get("prompts", [])))
        elif data and isinstance(data, list):
            summary[section] = len(data)
    return summary


def _summarize_generate_image(args: dict, result: dict) -> dict:
    summary = {"status": result.get("status", result.get("error", "unknown"))}
    if result.get("job_id"):
        summary["job_id"] = result["job_id"]
    prompt = args.get("prompt", "")
    summary["prompt_preview"] = prompt[:80] + ("..." if len(prompt) > 80 else "")
    return summary


def _summarize_get_loras(args: dict, result: dict) -> dict:
    return {"count": result.get("count", 0)}


def _summarize_get_output_dirs(args: dict, result: dict) -> dict:
    return {"count": result.get("count", 0)}


def _summarize_last_gen_settings(args: dict, result: dict) -> dict:
    s = result.get("settings", {})
    return {"found": bool(s), "model": s.get("base_model", "")}


def _summarize_last_gen_prompts(args: dict, result: dict) -> dict:
    return {"count": result.get("count", 0)}


def _summarize_update_state(args: dict, result: dict) -> dict:
    return {"status": result.get("status", "ok")}


def _summarize_query_dataset_map(args: dict, result: dict) -> dict:
    return {"query": args.get("query", ""), "count": result.get("count", 0)}


TOOL_SUMMARIES = {
    "search_similar_prompts": _summarize_search_similar,
    "search_diverse_prompts": _summarize_search_diverse,
    "get_random_prompts": _summarize_get_random,
    "get_opposite_prompts": _summarize_get_opposite,
    "list_concepts": _summarize_list_concepts,
    "get_dataset_overview": _summarize_dataset_overview,
    "get_folder_themes": _summarize_folder_themes,
    "query_themed_prompts": _summarize_query_themed,
    "generate_image": _summarize_generate_image,
    "get_available_loras": _summarize_get_loras,
    "get_output_directories": _summarize_get_output_dirs,
    "get_last_generation_settings": _summarize_last_gen_settings,
    "get_last_generated_prompts": _summarize_last_gen_prompts,
    "update_state": _summarize_update_state,
    "query_dataset_map": _summarize_query_dataset_map,
}


def summarize_tool_result(tool_name: str, args: dict, result: dict) -> dict:
    """Return a compact summary of a tool result for context truncation."""
    summarizer = TOOL_SUMMARIES.get(tool_name)
    if summarizer:
        return summarizer(args, result)
    return {"_summary": True, "tool": tool_name, "status": "completed"}
