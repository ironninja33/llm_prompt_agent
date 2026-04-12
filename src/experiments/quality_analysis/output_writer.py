"""Format and write the concept frequency analysis report."""

import math
import os
from datetime import datetime
from pathlib import Path

from .frequency_analysis import FrequencyAnalysisResult

SEPARATOR = "=" * 80
SUB_SEPARATOR = "-" * 80


def write_report(
    result: FrequencyAnalysisResult, output_dir: str, top_n: int = 50
) -> str:
    """Write the frequency analysis report to a timestamped text file.

    Returns the absolute path to the written report file.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"quality_analysis_{timestamp}.txt"
    path = os.path.join(output_dir, filename)

    lines: list[str] = []

    # Header
    lines.append(SEPARATOR)
    lines.append(f"Bad Quality Concept Analysis Report — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Base Model: {result.base_model}")
    lines.append(f"Reasons: {', '.join(result.reasons)}")
    lines.append(f"Total bad prompts analyzed: {result.total_bad}")
    lines.append(f"Total surviving prompts analyzed: {result.total_surviving}")
    lines.append(SEPARATOR)

    # Split tags into over-represented (also in surviving) and exclusive (only in bad)
    overrep_tags = [t for t in result.tags if not math.isinf(t.overrep_ratio)]
    exclusive_tags = [t for t in result.tags if math.isinf(t.overrep_ratio)]

    # Section 1: Over-represented concepts
    lines.append("")
    lines.append("SECTION 1: OVER-REPRESENTED CONCEPTS IN BAD GENERATIONS")
    lines.append(SEPARATOR)
    lines.append("Concepts that appear disproportionately in deleted (bad) prompts")
    lines.append("compared to surviving prompts, sorted by over-representation ratio.")
    lines.append("")

    if overrep_tags:
        header = f" {'#':>3} | {'Category':<16} | {'Tag':<24} | {'Bad':>6} | {'Surv':>6} | {'Ratio':>7} | {'p-value':>7}"
        lines.append(header)
        lines.append(f" {'---':>3}-+-{'-' * 16}-+-{'-' * 24}-+-{'-' * 6}-+-{'-' * 6}-+-{'-' * 7}-+-{'-' * 7}")

        for i, t in enumerate(overrep_tags[:top_n], 1):
            p_str = "<0.001" if t.fisher_p_value < 0.001 else f"{t.fisher_p_value:.3f}"
            lines.append(
                f" {i:3d} | {t.category:<16} | {t.tag:<24} | "
                f"{t.bad_rate * 100:5.1f}% | {t.surviving_rate * 100:5.1f}% | "
                f"{t.overrep_ratio:6.2f}x | {p_str:>7}"
            )
    else:
        lines.append("  (no over-represented tags found)")

    # Section 2: Concepts exclusive to bad generations
    lines.append("")
    lines.append("SECTION 2: CONCEPTS ONLY IN BAD GENERATIONS")
    lines.append(SEPARATOR)
    lines.append("Tags that appear in deleted prompts but NEVER in surviving prompts.")
    lines.append("")

    if exclusive_tags:
        header = f" {'#':>3} | {'Category':<16} | {'Tag':<24} | {'Bad Count':>9} | {'Bad Rate':>8}"
        lines.append(header)
        lines.append(f" {'---':>3}-+-{'-' * 16}-+-{'-' * 24}-+-{'-' * 9}-+-{'-' * 8}")

        for i, t in enumerate(exclusive_tags, 1):
            lines.append(
                f" {i:3d} | {t.category:<16} | {t.tag:<24} | "
                f"{t.bad_count:9d} | {t.bad_rate * 100:6.1f}%"
            )
    else:
        lines.append("  (no exclusive tags found)")

    # Section 3: Folder deletion rates
    lines.append("")
    lines.append("SECTION 3: FOLDER DELETION RATES")
    lines.append(SEPARATOR)
    lines.append("Per-folder deletion rates (folders with at least one deletion).")
    lines.append("")

    if result.folder_rates:
        header = f" {'#':>3} | {'Folder':<45} | {'Generated':>9} | {'Deleted':>7} | {'Rate':>6}"
        lines.append(header)
        lines.append(f" {'---':>3}-+-{'-' * 45}-+-{'-' * 9}-+-{'-' * 7}-+-{'-' * 6}")

        for i, f in enumerate(result.folder_rates, 1):
            lines.append(
                f" {i:3d} | {f['folder']:<45} | "
                f"{f['total_generated']:9d} | {f['total_deleted']:7d} | "
                f"{f['deletion_rate'] * 100:5.1f}%"
            )
    else:
        lines.append("  (no folder deletion data)")

    lines.append("")
    lines.append(SEPARATOR)

    Path(path).write_text("\n".join(lines), encoding="utf-8")
    return path
