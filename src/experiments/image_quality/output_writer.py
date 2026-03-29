"""Output writer — format and write scoring results to text file."""

import os
from datetime import datetime
from pathlib import Path

from .pipeline import PipelineReport

SEPARATOR = "=" * 80
SUB_SEPARATOR = "-" * 40


def write_results(
    report: PipelineReport,
    output_dir: str,
    query: str | None = None,
    folder: str | None = None,
    k: int = 10,
    seed: int | None = None,
) -> str:
    """Write scoring results to a timestamped text file.

    Returns path to the written file.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"quality_scores_{timestamp}.txt"
    path = os.path.join(output_dir, filename)

    lines = []

    # Header
    lines.append(SEPARATOR)
    lines.append(f"Image Quality Assessment — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if query:
        lines.append(f"Query: {query}")
    if folder:
        lines.append(f"Folder: {folder}")
    lines.append(f"Sample size (k): {k}")
    if seed is not None:
        lines.append(f"Seed: {seed}")
    lines.append(f"Images: {report.total_images} total, {report.failed_images} failed")
    if report.missing_prompts > 0:
        lines.append(f"Missing prompts: {report.missing_prompts} (affects ImageReward accuracy)")
    lines.append(SEPARATOR)

    # Performance summary
    lines.append("")
    lines.append("PERFORMANCE SUMMARY")
    lines.append(SUB_SEPARATOR)
    lines.append(f"Overall wall-clock time: {report.overall_elapsed_s:.2f}s")
    lines.append("")
    for algo, perf in report.performance.items():
        if "error" in perf:
            lines.append(f"  {algo}: FAILED — {perf['error']}")
            continue
        lines.append(f"  {algo}:")
        lines.append(f"    Model load:       {perf['model_load_time_s']:.2f}s")
        lines.append(
            f"    Scoring:          {perf['total_score_time_s']:.2f}s total, "
            f"{perf['avg_time_per_image_s']:.4f}s/image avg"
        )
        lines.append(f"    Images scored:    {perf['images_scored']}")
        if perf.get("gpu_memory_mb"):
            lines.append(f"    GPU memory:       ~{perf['gpu_memory_mb']:.0f} MB")
        lines.append("")

    # Per-image results
    lines.append("")
    lines.append("PER-IMAGE RESULTS")
    for img_score in report.scores:
        lines.append(SEPARATOR)
        lines.append(f"Filename: {img_score.filename}")
        lines.append(f"Path: {img_score.file_path}")
        size_mb = img_score.file_size / (1024 * 1024)
        lines.append(
            f"Size: {size_mb:.2f} MB | Resolution: {img_score.width}x{img_score.height}"
        )
        if img_score.prompt:
            prompt_display = img_score.prompt[:200]
            if len(img_score.prompt) > 200:
                prompt_display += "..."
            lines.append(f'Prompt: "{prompt_display}"')
        else:
            lines.append("Prompt: [not found]")

        if img_score.load_error:
            lines.append(f"ERROR: {img_score.load_error}")
        else:
            lines.append(SUB_SEPARATOR)
            for algo_name, result in img_score.results.items():
                lines.append(f"  {algo_name.upper()}:")
                for key, val in result.scores.items():
                    lines.append(f"    {key}: {val:.4f}")
                lines.append(f"    normalized (0-1): {result.normalized_score:.4f}")
                lines.append(f"    time: {result.elapsed_s:.4f}s")

    lines.append(SEPARATOR)

    # Ranking summaries
    for algo_name in report.performance:
        if "error" in report.performance[algo_name]:
            continue

        scored_images = [
            (s.filename, s.results[algo_name])
            for s in report.scores
            if algo_name in s.results
        ]
        scored_images.sort(key=lambda x: x[1].normalized_score, reverse=True)

        lines.append("")
        lines.append(f"RANKING BY {algo_name.upper()} (best to worst)")
        lines.append(SUB_SEPARATOR)
        lines.append(f"{'RANK':>4} | {'RAW':>10} | {'NORMALIZED':>10} | {'TIME':>8} | FILENAME")
        for i, (fname, result) in enumerate(scored_images, 1):
            lines.append(
                f"{i:4d} | {result.raw_score:10.4f} | "
                f"{result.normalized_score:10.4f} | "
                f"{result.elapsed_s:7.4f}s | {fname}"
            )

    lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    return path
