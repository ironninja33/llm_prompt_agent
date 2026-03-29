"""Scoring pipeline — orchestrate image loading, assessment, and result collection."""

import logging
import os
import time
from dataclasses import dataclass, field

from PIL import Image

from .assessors.base import AssessmentResult, QualityAssessor
from .data_sampler import SampledImage

logger = logging.getLogger(__name__)


@dataclass
class ImageScore:
    """All assessment results for a single image."""

    file_path: str
    filename: str
    file_size: int
    width: int
    height: int
    prompt: str | None
    results: dict[str, AssessmentResult] = field(default_factory=dict)
    load_error: str | None = None


@dataclass
class PipelineReport:
    """Full pipeline run report."""

    scores: list[ImageScore]
    performance: dict  # {algo: {total_time, avg_time, images_scored, model_load_time, gpu_memory_mb}}
    overall_elapsed_s: float
    total_images: int
    failed_images: int
    missing_prompts: int


def run_pipeline(
    sampled_images: list[SampledImage],
    assessors: list[QualityAssessor],
    device: str = "cuda",
) -> PipelineReport:
    """Run the scoring pipeline over sampled images.

    Models are loaded and unloaded sequentially (never two on GPU at once).
    Images are opened one at a time and closed immediately after scoring.
    """
    total_images = len(sampled_images)
    if not sampled_images:
        return PipelineReport(
            scores=[], performance={}, overall_elapsed_s=0.0,
            total_images=0, failed_images=0, missing_prompts=0,
        )

    missing_prompts = sum(1 for s in sampled_images if not s.prompt)

    # Pre-create ImageScore containers
    image_scores: dict[str, ImageScore] = {}
    for s in sampled_images:
        image_scores[s.file_path] = ImageScore(
            file_path=s.file_path,
            filename=s.filename,
            file_size=s.file_size,
            width=s.width,
            height=s.height,
            prompt=s.prompt,
        )

    performance: dict[str, dict] = {}
    failed_count = 0
    overall_start = time.time()

    for assessor in assessors:
        algo_name = assessor.name
        logger.info("Loading model: %s", algo_name)

        t_load_start = time.time()
        try:
            assessor.load_model(device=device)
        except Exception as e:
            logger.error("Failed to load %s: %s", algo_name, e)
            performance[algo_name] = {"error": str(e)}
            continue
        t_load = time.time() - t_load_start
        gpu_mem = assessor.get_gpu_memory_mb()
        logger.info(
            "Model loaded in %.1fs%s",
            t_load,
            f" ({gpu_mem:.0f} MB GPU)" if gpu_mem else "",
        )

        scored = 0
        t_score_start = time.time()

        for sample in sampled_images:
            img_score = image_scores[sample.file_path]

            # Load image
            try:
                img = Image.open(sample.file_path)
                img.load()
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")
            except Exception as e:
                logger.warning("Failed to load %s: %s", sample.filename, e)
                if not img_score.load_error:
                    img_score.load_error = str(e)
                    failed_count += 1
                continue

            # Score
            try:
                prompt = sample.prompt if assessor.needs_prompt else None
                t_img_start = time.time()
                result = assessor.score(img, prompt=prompt)
                result.elapsed_s = time.time() - t_img_start
                img_score.results[algo_name] = result
                scored += 1
            except Exception as e:
                logger.warning(
                    "Scoring failed for %s with %s: %s",
                    sample.filename, algo_name, e,
                )
            finally:
                img.close()

        t_score_total = time.time() - t_score_start

        assessor.unload_model()

        performance[algo_name] = {
            "model_load_time_s": round(t_load, 2),
            "total_score_time_s": round(t_score_total, 2),
            "avg_time_per_image_s": round(t_score_total / max(scored, 1), 4),
            "images_scored": scored,
            "gpu_memory_mb": gpu_mem,
        }
        logger.info(
            "%s: scored %d images in %.1fs (%.3fs/image)",
            algo_name, scored, t_score_total,
            t_score_total / max(scored, 1),
        )

    overall_elapsed = time.time() - overall_start

    return PipelineReport(
        scores=sorted(image_scores.values(), key=lambda s: s.file_path),
        performance=performance,
        overall_elapsed_s=round(overall_elapsed, 2),
        total_images=total_images,
        failed_images=failed_count,
        missing_prompts=missing_prompts,
    )
