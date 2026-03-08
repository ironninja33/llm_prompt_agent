"""KeyBERT keyword extraction using sentence-transformers + cosine similarity."""

from __future__ import annotations

import logging

from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_model(model_name: str = "all-MiniLM-L6-v2"):
    """Load KeyBERT with specified sentence-transformer.

    Returns:
        A ``KeyBERT`` instance.
    """
    from keybert import KeyBERT

    logger.info(f"Loading KeyBERT model: {model_name}")
    kw_model = KeyBERT(model=model_name)
    logger.info("KeyBERT model loaded.")
    return kw_model


MAX_DOC_CHARS = 50_000


def extract_keywords(
    kw_model,
    texts: list[str],
    top_n: int = 5,
    ngram_range: tuple[int, int] = (1, 3),
    diversity: float = 0.5,
    use_mmr: bool = True,
) -> str:
    """Extract keywords at each n-gram size separately, then combine.

    Extracts top_n candidates per n-gram size, dedupes, and returns the top_n
    overall ranked by score. This prevents longer n-grams from crowding out
    shorter ones.

    If the concatenated text exceeds MAX_DOC_CHARS, it is truncated at a word
    boundary to avoid crashing the sentence-transformer tokenizer.

    Returns:
        Comma-separated keyword string.
    """
    doc = " ".join(texts)
    if len(doc) > MAX_DOC_CHARS:
        logger.info(f"Truncating document from {len(doc)} to ~{MAX_DOC_CHARS} chars")
        doc = doc[:MAX_DOC_CHARS].rsplit(" ", 1)[0]

    ngram_min, ngram_max = ngram_range
    all_keywords: dict[str, float] = {}
    for n in range(ngram_min, ngram_max + 1):
        keywords = kw_model.extract_keywords(
            doc,
            keyphrase_ngram_range=(n, n),
            stop_words="english",
            top_n=top_n,
            use_mmr=use_mmr,
            diversity=diversity,
        )
        for kw, score in keywords:
            if kw not in all_keywords or score > all_keywords[kw]:
                all_keywords[kw] = score

    ranked = sorted(all_keywords.items(), key=lambda x: x[1], reverse=True)
    return ", ".join(kw for kw, _ in ranked[:top_n])


def batch_extract(
    kw_model,
    prompt_lists: list[list[str]],
    top_n: int = 5,
    ngram_range: tuple[int, int] = (1, 3),
    diversity: float = 0.5,
    use_mmr: bool = True,
) -> list[str]:
    """Extract keywords for multiple folders/clusters with tqdm progress."""
    results = []
    for texts in tqdm(prompt_lists, desc="Extracting keywords"):
        if texts:
            kw = extract_keywords(
                kw_model, texts,
                top_n=top_n,
                ngram_range=ngram_range,
                diversity=diversity,
                use_mmr=use_mmr,
            )
        else:
            kw = ""
        results.append(kw)
    return results
