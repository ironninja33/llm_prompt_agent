"""Abstract base class for image quality assessors."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image


@dataclass
class AssessmentResult:
    """Score result from a single assessor for a single image."""

    algorithm: str
    scores: dict[str, float]
    raw_score: float
    normalized_score: float
    prompt_used: str | None = None
    elapsed_s: float = 0.0


class QualityAssessor(ABC):
    """Abstract base for image quality assessment algorithms."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Algorithm identifier string."""
        ...

    @property
    @abstractmethod
    def needs_prompt(self) -> bool:
        """Whether this assessor requires the generation prompt."""
        ...

    @abstractmethod
    def load_model(self, device: str = "cuda") -> None:
        """Load model weights onto the specified device."""
        ...

    @abstractmethod
    def score(self, image: "Image.Image", prompt: str | None = None) -> AssessmentResult:
        """Score a single image."""
        ...

    def score_batch(
        self,
        images: list["Image.Image"],
        prompts: list[str | None] | None = None,
    ) -> list[AssessmentResult]:
        """Score a batch of images. Default: sequential fallback."""
        if prompts is None:
            prompts = [None] * len(images)
        return [self.score(img, p) for img, p in zip(images, prompts)]

    @abstractmethod
    def unload_model(self) -> None:
        """Release GPU memory."""
        ...

    def get_gpu_memory_mb(self) -> float | None:
        """Return approximate GPU memory usage in MB after model load."""
        return None
