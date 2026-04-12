"""Compare concept tag frequencies between bad and surviving prompt sets."""

import logging
from collections import Counter
from dataclasses import dataclass, field

from scipy.stats import fisher_exact

from .concept_extraction import ExtractedConcepts

logger = logging.getLogger(__name__)


@dataclass
class TagFrequency:
    """Frequency stats for a single tag value."""
    category: str
    tag: str
    bad_count: int
    bad_rate: float           # bad_count / total_bad
    surviving_count: int
    surviving_rate: float     # surviving_count / total_surviving
    overrep_ratio: float      # bad_rate / surviving_rate (inf if surviving_rate == 0)
    fisher_p_value: float     # one-sided Fisher's exact test


@dataclass
class FrequencyAnalysisResult:
    """Full result of tag frequency comparison."""
    base_model: str
    reasons: list[str]
    total_bad: int
    total_surviving: int
    tags: list[TagFrequency] = field(default_factory=list)
    folder_rates: list[dict] = field(default_factory=list)


def compute_tag_frequencies(
    bad_concepts: list[ExtractedConcepts],
    surviving_concepts: list[ExtractedConcepts],
    base_model: str,
    reasons: list[str],
    folder_rates: list[dict],
    min_bad_count: int = 2,
) -> FrequencyAnalysisResult:
    """Compare tag frequencies between bad and surviving corpora.

    For each unique (category, tag) pair:
    1. Count occurrences in bad set and surviving set
    2. Compute rates (count / total prompts in that set)
    3. Compute over-representation ratio (bad_rate / surviving_rate)
    4. Run Fisher's exact test for statistical significance
    5. Filter to tags with bad_count >= min_bad_count
    6. Sort by overrep_ratio DESC
    """
    total_bad = len(bad_concepts)
    total_surviving = len(surviving_concepts)

    bad_counter = _count_tags(bad_concepts)
    surviving_counter = _count_tags(surviving_concepts)

    results: list[TagFrequency] = []
    for (category, tag), bad_count in bad_counter.items():
        if bad_count < min_bad_count:
            continue

        surviving_count = surviving_counter.get((category, tag), 0)
        bad_rate = bad_count / total_bad if total_bad > 0 else 0
        surviving_rate = surviving_count / total_surviving if total_surviving > 0 else 0

        if surviving_rate > 0:
            overrep_ratio = bad_rate / surviving_rate
        else:
            overrep_ratio = float("inf")

        # Fisher's exact test: is this tag significantly more common in bad set?
        table = [
            [bad_count, total_bad - bad_count],
            [surviving_count, total_surviving - surviving_count],
        ]
        _, p_value = fisher_exact(table, alternative="greater")

        results.append(TagFrequency(
            category=category,
            tag=tag,
            bad_count=bad_count,
            bad_rate=bad_rate,
            surviving_count=surviving_count,
            surviving_rate=surviving_rate,
            overrep_ratio=overrep_ratio,
            fisher_p_value=p_value,
        ))

    results.sort(key=lambda t: t.overrep_ratio, reverse=True)

    return FrequencyAnalysisResult(
        base_model=base_model,
        reasons=reasons,
        total_bad=total_bad,
        total_surviving=total_surviving,
        tags=results,
        folder_rates=folder_rates,
    )


def _count_tags(concepts: list[ExtractedConcepts]) -> Counter:
    """Flatten all (category, tag) pairs from a list of ExtractedConcepts."""
    counter: Counter = Counter()
    for concept in concepts:
        for category, tags in concept.tags.items():
            for tag in tags:
                normalized = tag.lower().strip()
                if normalized:
                    counter[(category, normalized)] += 1
    return counter
