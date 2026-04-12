"""LLM-based concept extraction from image generation prompts."""

import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a structured data extraction assistant. You analyze image generation prompts \
and extract concept tags into a fixed taxonomy. Output ONLY valid JSON with no \
additional text.

Categories:
- pose: Body position (e.g., "standing", "bent over", "kneeling", "doggy style", \
"missionary", "reverse cowgirl", "lying on back", "straddling", "on all fours")
- camera_angle: Camera perspective (e.g., "low angle", "high angle", "pov", \
"from behind", "frontal", "close-up", "wide shot", "floor level", "side view")
- action: What is happening (e.g., "penetration", "oral sex", "blowjob", \
"cunnilingus", "masturbation", "posing", "dancing", "squirting", "foreplay")
- lighting: Lighting style (e.g., "neon", "low-key", "dramatic", "natural", \
"studio", "rim lighting", "backlit", "cinematic", "candlelight")
- setting: Location (e.g., "strip club", "bedroom", "alley", "studio", "stage", \
"bathroom", "office", "outdoor", "pool", "hotel room")
- clothing: What is worn (e.g., "stockings", "garter belt", "high heels", \
"lingerie", "nude", "fishnet", "pencil skirt", "bra")
- body_descriptor: Physical emphasis (e.g., "large breasts", "wide hips", \
"hourglass", "muscular", "thick thighs", "petite", "voluptuous", "pawg")
- expression: Facial/emotional state (e.g., "ecstasy", "moaning", "eye contact", \
"helpless", "lustful", "seductive", "intense", "submissive")
- num_subjects: Count (e.g., "solo", "couple", "group")
- explicitness: Level (e.g., "softcore", "explicit", "implied", "non-sexual")
- prop: Notable objects (e.g., "pedestal", "couch", "mirror", "neon sign", \
"pole", "chair", "bed")
- composition: Compositional elements (e.g., "reflection", "foreground element", \
"silhouette", "symmetry", "framing")

Use short, lowercase, normalized tags. Omit categories not present in the prompt.
Output format: {"pose": ["kneeling"], "action": ["oral sex"], ...}"""


@dataclass
class ExtractedConcepts:
    """Structured concept tags extracted from a single prompt."""
    source_prompt: str
    tags: dict[str, list[str]] = field(default_factory=dict)
    raw_response: str = ""


def build_extraction_prompt(prompt_text: str) -> str:
    """Build the user-message for concept extraction from a single prompt."""
    return f"Extract concept tags from this image generation prompt:\n\n{prompt_text}"


def parse_extraction_response(response: str, source_prompt: str) -> ExtractedConcepts:
    """Parse the LLM's JSON response into an ExtractedConcepts object.

    Handles code fences and malformed JSON gracefully.
    """
    cleaned = response.strip()

    # Strip markdown code fences
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first line (```json) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    try:
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            logger.warning("LLM response is not a JSON object: %s", cleaned[:200])
            return ExtractedConcepts(source_prompt=source_prompt, raw_response=response)

        # Normalize: ensure all values are lists of strings
        tags: dict[str, list[str]] = {}
        for category, values in data.items():
            if isinstance(values, list):
                tags[category] = [str(v).lower().strip() for v in values if v]
            elif isinstance(values, str):
                tags[category] = [values.lower().strip()]

        return ExtractedConcepts(
            source_prompt=source_prompt,
            tags=tags,
            raw_response=response,
        )
    except (json.JSONDecodeError, ValueError):
        logger.warning("Failed to parse LLM response as JSON: %s", cleaned[:200])
        return ExtractedConcepts(source_prompt=source_prompt, raw_response=response)


def extract_concepts_batch(
    prompts: list[str],
    model_id: str = "Qwen/Qwen3-4B-FP8",
    quantization: str | None = None,
    dtype: str = "auto",
    gpu_memory_utilization: float = 0.85,
    max_model_len: int = 8192,
    max_tokens: int = 1024,
    no_think: bool = False,
) -> list[ExtractedConcepts]:
    """Load vLLM, extract concepts from all prompts in one batch, unload model.

    Returns list of ExtractedConcepts, one per input prompt.
    """
    from src.experiments.common import vllm_engine

    extraction_prompts = [build_extraction_prompt(p) for p in prompts]

    llm = vllm_engine.load_model(
        model_id=model_id,
        quantization=quantization,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )

    try:
        clean_results, _ = vllm_engine.batch_generate(
            llm, extraction_prompts, SYSTEM_PROMPT,
            max_tokens=max_tokens, no_think=no_think,
        )
    finally:
        vllm_engine.unload_model(llm)

    results = []
    for response, source in zip(clean_results, prompts):
        results.append(parse_extraction_response(response, source))

    parsed_ok = sum(1 for r in results if r.tags)
    logger.info("Parsed %d/%d responses successfully", parsed_ok, len(results))

    return results
