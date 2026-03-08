"""vLLM model loading, batched inference, and cleanup."""

from __future__ import annotations

import gc
import logging
import re

logger = logging.getLogger(__name__)


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks (and unclosed <think>... from truncated output)."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL)
    cleaned = cleaned.strip()
    if cleaned:
        return cleaned
    inner = re.sub(r"</?think>", "", text).strip()
    return inner


def load_model(
    model_id: str,
    quantization: str | None = None,
    dtype: str = "auto",
    gpu_memory_utilization: float = 0.85,
    max_model_len: int = 4096,
):
    """Load a vLLM model. Downloads from HuggingFace if needed.

    Returns:
        A ``vllm.LLM`` instance.
    """
    from vllm import LLM

    kwargs = {
        "model": model_id,
        "trust_remote_code": True,
        "max_model_len": max_model_len,
        "dtype": dtype,
        "gpu_memory_utilization": gpu_memory_utilization,
    }
    if quantization:
        kwargs["quantization"] = quantization

    logger.info(f"Loading model {model_id} (dtype={dtype}, quantization={quantization})...")
    llm = LLM(**kwargs)
    logger.info("Model loaded.")
    return llm


def batch_generate(
    llm,
    prompts: list[str],
    system_prompt: str,
    max_tokens: int = 1024,
    no_think: bool = False,
) -> tuple[list[str], list[str]]:
    """Run batched inference with chat template formatting.

    Returns:
        Tuple of (clean_results, raw_results).
    """
    from vllm import SamplingParams

    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=0.3,
        max_tokens=max_tokens,
        top_p=0.9,
    )

    formatted_prompts = []
    for prompt in prompts:
        user_content = f"{prompt} /no_think" if no_think else prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        formatted_prompts.append(formatted)

    logger.info(f"Running inference on {len(formatted_prompts)} prompts...")
    outputs = llm.generate(formatted_prompts, sampling_params)

    results_clean = []
    results_raw = []
    for output in outputs:
        raw = output.outputs[0].text.strip()
        results_raw.append(raw)
        results_clean.append(_strip_thinking(raw))

    return results_clean, results_raw


def unload_model(llm) -> None:
    """Free GPU VRAM by deleting the model and clearing caches."""
    import torch

    del llm
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("Model unloaded and GPU memory freed.")
