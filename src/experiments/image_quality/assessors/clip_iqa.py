"""CLIP-IQA+ assessor — no-reference image quality assessment via pyiqa."""

import logging

from .base import AssessmentResult, QualityAssessor

logger = logging.getLogger(__name__)


class ClipIqaAssessor(QualityAssessor):
    """CLIP-IQA+ image quality assessment.

    Outputs a single float in [0, 1] where higher = better technical quality.
    Evaluates sharpness, noise, distortion, and overall perceptual quality.
    Does not require a text prompt.
    """

    def __init__(self):
        self._model = None
        self._device = "cuda"

    @property
    def name(self) -> str:
        return "clip-iqa+"

    @property
    def needs_prompt(self) -> bool:
        return False

    def load_model(self, device: str = "cuda") -> None:
        import pyiqa

        self._device = device
        self._model = pyiqa.create_metric("clipiqa+", device=device)
        logger.info("CLIP-IQA+ model loaded on %s", device)

    def score(self, image, prompt=None) -> AssessmentResult:
        import torchvision.transforms.functional as TF

        tensor = TF.to_tensor(image).unsqueeze(0).to(self._device)
        raw = self._model(tensor).item()

        return AssessmentResult(
            algorithm=self.name,
            scores={"quality": raw},
            raw_score=raw,
            normalized_score=raw,  # already 0-1
            prompt_used=None,
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
