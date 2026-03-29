"""Quality assessor registry."""

from .clip_iqa import ClipIqaAssessor
from .image_reward import ImageRewardAssessor

ASSESSORS = {
    "clip-iqa+": ClipIqaAssessor,
    "imagereward": ImageRewardAssessor,
}
