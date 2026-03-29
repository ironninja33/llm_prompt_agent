"""ImageReward assessor — human preference scoring for text-to-image generation."""

import logging

from .base import AssessmentResult, QualityAssessor

logger = logging.getLogger(__name__)


class ImageRewardAssessor(QualityAssessor):
    """ImageReward human preference assessment.

    Outputs a float typically in [-2, 2] where higher = better human preference.
    Requires both the image and the text prompt that generated it.
    """

    def __init__(self):
        self._model = None
        self._device = "cuda"

    @property
    def name(self) -> str:
        return "imagereward"

    @property
    def needs_prompt(self) -> bool:
        return True

    def load_model(self, device: str = "cuda") -> None:
        import ImageReward as IR

        self._device = device
        self._model = IR.load("ImageReward-v1.0", device=device)
        logger.info("ImageReward model loaded on %s", device)

    def score(self, image, prompt=None) -> AssessmentResult:
        if prompt is None:
            prompt = ""

        raw = self._model.score(prompt, image)

        # Normalize from typical [-2, 2] range to [0, 1]
        normalized = max(0.0, min(1.0, (raw + 2.0) / 4.0))

        return AssessmentResult(
            algorithm=self.name,
            scores={"preference": raw},
            raw_score=raw,
            normalized_score=normalized,
            prompt_used=prompt[:200] if prompt else None,
        )

    def unload_model(self) -> None:
        import torch

        del self._model
        self._model = None
        torch.cuda.empty_cache()

    def get_gpu_memory_mb(self) -> float | None:
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024 * 1024)
        except Exception:
            pass
        return None
