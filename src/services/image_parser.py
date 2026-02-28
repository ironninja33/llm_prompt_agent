"""Image metadata parser for extracting prompts and generation info from images.

Handles:
- PNG files with ComfyUI workflow data embedded in tEXt chunks
- JPG files with EXIF UserComment containing the prompt (UTF-16-BE encoded)
- TXT files (training data captions — only these are read from training dirs)
"""

import json
import os
import re
import logging
from dataclasses import dataclass, field

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class ParsedImageData:
    """Data extracted from an image or text file."""
    prompt: str
    source_file: str
    base_model: str | None = None
    loras: list[str] = field(default_factory=list)
    negative_prompt: str | None = None
    raw_workflow: dict | None = None
    sampler: str | None = None
    cfg_scale: float | None = None
    scheduler: str | None = None
    steps: int | None = None
    seed: int | None = None


def parse_file(filepath: str) -> ParsedImageData | None:
    """Parse a file and extract prompt data.

    Args:
        filepath: Path to the image or text file.

    Returns:
        ParsedImageData if prompt was successfully extracted, None otherwise.
    """
    ext = os.path.splitext(filepath)[1].lower()

    try:
        if ext == ".txt":
            return _parse_text_file(filepath)
        elif ext == ".png":
            return _parse_png_file(filepath)
        elif ext in (".jpg", ".jpeg"):
            return _parse_jpg_file(filepath)
        else:
            logger.warning(f"Unsupported file type: {ext} for {filepath}")
            return None
    except Exception as e:
        logger.error(f"Error parsing {filepath}: {e}", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Text file parsing (training data)
# ---------------------------------------------------------------------------

def _parse_text_file(filepath: str) -> ParsedImageData | None:
    """Parse a training data text file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if not content:
            return None

        return ParsedImageData(
            prompt=content,
            source_file=os.path.basename(filepath),
        )
    except Exception as e:
        logger.error(f"Error reading text file {filepath}: {e}")
        return None


# ---------------------------------------------------------------------------
# PNG file parsing (ComfyUI workflow in tEXt chunks)
# ---------------------------------------------------------------------------

def _parse_png_file(filepath: str) -> ParsedImageData | None:
    """Parse a PNG file with embedded ComfyUI workflow or A1111-style data.

    ComfyUI stores the API-format workflow in the ``prompt`` tEXt chunk.
    Some PNGs may instead have A1111-style text in ``parameters`` or other
    tEXt chunks containing the prompt/negative/generation info.
    """
    try:
        img = Image.open(filepath)
        info = img.info
        img.close()

        # 1. Try ComfyUI workflow (JSON in the "prompt" chunk)
        workflow_data = None
        if "prompt" in info:
            try:
                workflow_data = json.loads(info["prompt"])
            except (json.JSONDecodeError, TypeError):
                pass

        if workflow_data and isinstance(workflow_data, dict):
            return _extract_from_comfyui_workflow(workflow_data, filepath)

        # 2. Try A1111-style text in "parameters" chunk
        if "parameters" in info:
            text = info["parameters"]
            if isinstance(text, str) and len(text) > 10:
                positive, negative, base_model, loras, sinfo = _parse_a1111_metadata(text)
                if positive:
                    return ParsedImageData(
                        prompt=positive,
                        source_file=os.path.basename(filepath),
                        negative_prompt=negative,
                        base_model=base_model,
                        loras=loras,
                        **sinfo,
                    )

        # 3. If "prompt" chunk existed but wasn't valid JSON, treat as A1111 text
        if "prompt" in info and not workflow_data:
            text = info["prompt"]
            if isinstance(text, str) and len(text) > 10:
                positive, negative, base_model, loras, sinfo = _parse_a1111_metadata(text)
                if positive:
                    return ParsedImageData(
                        prompt=positive,
                        source_file=os.path.basename(filepath),
                        negative_prompt=negative,
                        base_model=base_model,
                        loras=loras,
                        **sinfo,
                    )

        # 4. Check any other text chunks that might contain prompt data
        for key in ("Comment", "Description", "UserComment"):
            if key in info:
                text = info[key]
                if isinstance(text, str) and len(text) > 10:
                    positive, negative, base_model, loras, sinfo = _parse_a1111_metadata(text)
                    if positive:
                        return ParsedImageData(
                            prompt=positive,
                            source_file=os.path.basename(filepath),
                            negative_prompt=negative,
                            base_model=base_model,
                            loras=loras,
                            **sinfo,
                        )

        # 5. Fallback: try to extract from filename
        return _extract_from_filename(filepath)

    except Exception as e:
        logger.error(f"Error parsing PNG {filepath}: {e}")
        return None


def _extract_from_comfyui_workflow(
    workflow: dict, filepath: str
) -> ParsedImageData | None:
    """Extract prompt, model, and LoRA info from a ComfyUI API workflow.

    Real-world node types observed:
    - ``PrimitiveStringMultiline`` with title containing "Positive" / "Negative"
    - ``CLIPTextEncode`` (sometimes used directly with a ``text`` input)
    - ``UNETLoader`` with ``unet_name``
    - ``CheckpointLoaderSimple`` with ``ckpt_name``
    - ``Lora Loader Stack (rgthree)`` with ``lora_01``, ``lora_02``, …
    - ``LoraLoader`` with ``lora_name``
    """
    positive_prompt = None
    negative_prompt = None
    base_model = None
    loras: list[str] = []

    # Collect all candidate text values with their "positivity" signal
    text_candidates: list[tuple[str, str, str]] = []  # (text, title, class_type)

    for node_id, node_data in workflow.items():
        class_type = node_data.get("class_type", "")
        inputs = node_data.get("inputs", {})
        meta = node_data.get("_meta", {})
        title = meta.get("title", "")

        # --- Text / prompt nodes ---
        # PrimitiveStringMultiline stores prompts as "value"
        if class_type in ("PrimitiveStringMultiline", "Text Multiline"):
            value = inputs.get("value", "")
            if isinstance(value, str) and value.strip():
                text_candidates.append((value.strip(), title, class_type))

        # CLIPTextEncode stores prompts as "text"
        if class_type in ("CLIPTextEncode", "CLIPTextEncodeSDXL"):
            text = inputs.get("text", "")
            if isinstance(text, str) and text.strip():
                text_candidates.append((text.strip(), title, class_type))

        # --- Model loaders ---
        if class_type in (
            "CheckpointLoaderSimple", "CheckpointLoader",
            "CheckpointLoaderNF4",
        ):
            ckpt_name = inputs.get("ckpt_name", "")
            if ckpt_name:
                base_model = ckpt_name

        if class_type == "UNETLoader":
            unet_name = inputs.get("unet_name", "")
            if unet_name:
                base_model = unet_name

        # --- LoRA loaders ---
        # Single LoRA loader
        if "lora" in class_type.lower() and "loader" in class_type.lower():
            # rgthree stacked loader: lora_01, lora_02, lora_03, lora_04
            for key in sorted(inputs.keys()):
                if re.match(r"^lora_\d+$", key):
                    lora_name = inputs[key]
                    if isinstance(lora_name, str) and lora_name.strip() and lora_name != "None":
                        if lora_name not in loras:
                            loras.append(lora_name)

            # Standard single LoRA loader
            lora_name = inputs.get("lora_name", "")
            if isinstance(lora_name, str) and lora_name.strip() and lora_name != "None":
                if lora_name not in loras:
                    loras.append(lora_name)

    # Classify text candidates as positive or negative
    for text, title, class_type in text_candidates:
        title_lower = title.lower()
        if "positive" in title_lower:
            if positive_prompt is None:
                positive_prompt = text
        elif "negative" in title_lower:
            if negative_prompt is None:
                negative_prompt = text

    # If we didn't find an explicitly titled positive prompt, use the first
    # long text candidate that isn't the negative
    if positive_prompt is None:
        for text, title, class_type in text_candidates:
            if text != negative_prompt and len(text) > 10:
                positive_prompt = text
                break

    if not positive_prompt:
        return _extract_from_filename(filepath)

    # Extract sampler settings from the workflow
    sampler_info = _extract_sampler_settings(workflow)

    return ParsedImageData(
        prompt=positive_prompt,
        source_file=os.path.basename(filepath),
        base_model=base_model,
        loras=loras,
        negative_prompt=negative_prompt,
        raw_workflow=workflow,
        **sampler_info,
    )


# ---------------------------------------------------------------------------
# Sampler settings extraction from ComfyUI workflows
# ---------------------------------------------------------------------------

def _resolve_node_value(workflow: dict, value, as_type=None):
    """Resolve a ComfyUI workflow input value, following node references.

    In ComfyUI API format, inputs that come from other nodes are stored as
    [node_id, output_index]. This follows the reference to get the source
    node's value (checking common keys like 'seed', 'value', 'noise_seed').

    Args:
        workflow: The full workflow dict.
        value: The raw input value (direct value or [node_id, output_index]).
        as_type: Optional type to cast the result to (int, float, str).

    Returns:
        The resolved value, or None if unresolvable.
    """
    # Direct value (not a reference)
    if not isinstance(value, list):
        if value is None:
            return None
        if as_type:
            try:
                return as_type(value)
            except (ValueError, TypeError):
                return None
        return value

    # Node reference: [node_id, output_index]
    if len(value) != 2:
        return None

    ref_node_id = str(value[0])
    ref_node = workflow.get(ref_node_id)
    if not ref_node:
        return None

    ref_inputs = ref_node.get("inputs", {})
    # Try common value keys used by primitive/seed nodes
    for key in ("seed", "value", "noise_seed", "SEED", "Value"):
        candidate = ref_inputs.get(key)
        if candidate is not None and not isinstance(candidate, list):
            if as_type:
                try:
                    return as_type(candidate)
                except (ValueError, TypeError):
                    continue
            return candidate

    return None


def _extract_sampler_settings(workflow: dict) -> dict:
    """Extract sampler, CFG, scheduler, steps, and seed from a ComfyUI API workflow.

    Generically scans for any node whose class_type contains "ksampler" or
    "sampler" (case-insensitive). Covers KSampler, KSamplerAdvanced, and
    custom sampler nodes. Follows node references for seed/steps values.

    Returns:
        Dict with keys: sampler, cfg_scale, scheduler, steps, seed (None if not found).
    """
    result = {"sampler": None, "cfg_scale": None, "scheduler": None, "steps": None, "seed": None}

    for node_id, node_data in workflow.items():
        class_type = (node_data.get("class_type") or "").lower()
        if "sampler" not in class_type:
            continue

        inputs = node_data.get("inputs", {})

        # sampler_name is the standard key for KSampler nodes
        if result["sampler"] is None:
            sampler_name = inputs.get("sampler_name") or inputs.get("sampler")
            if isinstance(sampler_name, str) and sampler_name.strip():
                result["sampler"] = sampler_name.strip()

        if result["cfg_scale"] is None:
            cfg = inputs.get("cfg")
            if cfg is not None:
                resolved = _resolve_node_value(workflow, cfg, as_type=float)
                if resolved is not None:
                    result["cfg_scale"] = resolved

        if result["scheduler"] is None:
            scheduler = inputs.get("scheduler")
            if isinstance(scheduler, str) and scheduler.strip():
                result["scheduler"] = scheduler.strip()

        if result["steps"] is None:
            steps = inputs.get("steps")
            if steps is not None:
                resolved = _resolve_node_value(workflow, steps, as_type=int)
                if resolved is not None:
                    result["steps"] = resolved

        if result["seed"] is None:
            # KSampler uses "seed", KSamplerAdvanced uses "noise_seed"
            for key in ("seed", "noise_seed"):
                raw = inputs.get(key)
                if raw is not None:
                    resolved = _resolve_node_value(workflow, raw, as_type=int)
                    if resolved is not None:
                        result["seed"] = resolved
                        break

        # If we found everything, stop scanning
        if all(v is not None for v in result.values()):
            break

    return result


# ---------------------------------------------------------------------------
# JPG file parsing (EXIF UserComment)
# ---------------------------------------------------------------------------

def _parse_jpg_file(filepath: str) -> ParsedImageData | None:
    """Parse a JPG file for embedded prompt data.

    The JPG output files from the user's pipeline store the prompt in the
    EXIF UserComment tag (0x9286) inside the ExifIFD, encoded as UTF-16-BE
    with a ``UNICODE\\x00`` 8-byte prefix.
    """
    try:
        result = _extract_from_jpg_exif(filepath)
        if result:
            return result

        # Fallback: extract from filename
        return _extract_from_filename(filepath)

    except Exception as e:
        logger.error(f"Error parsing JPG {filepath}: {e}")
        return None


def _extract_from_jpg_exif(filepath: str) -> ParsedImageData | None:
    """Extract prompt, negative prompt, model, and LoRA from EXIF UserComment.

    The A1111/ComfyUI-style format embedded in the UserComment is::

        <positive prompt>
        Negative prompt: <negative prompt>
        Steps: 20, Sampler: ..., Model: <name>, ..., Extra info: <lora>, None, None
    """
    try:
        img = Image.open(filepath)
        exif_data = img.getexif()
        img.close()

        if not exif_data:
            return None

        # The UserComment lives inside the Exif sub-IFD (tag 0x8769)
        exif_ifd = exif_data.get_ifd(0x8769)
        user_comment = exif_ifd.get(0x9286) if exif_ifd else None

        # Also try the flat EXIF (some Pillow versions expose it directly)
        if user_comment is None:
            user_comment = exif_data.get(0x9286)

        if user_comment is None:
            return None

        text = _decode_user_comment(user_comment)
        if not text or len(text) < 10:
            return None

        # Parse the full A1111-style embedded metadata
        positive, negative, base_model, loras, sinfo = _parse_a1111_metadata(text)

        if not positive:
            return None

        return ParsedImageData(
            prompt=positive,
            source_file=os.path.basename(filepath),
            negative_prompt=negative,
            base_model=base_model,
            loras=loras,
            **sinfo,
        )

    except Exception as e:
        logger.debug(f"No usable EXIF in {filepath}: {e}")
        return None


def _decode_user_comment(raw) -> str | None:
    """Decode an EXIF UserComment value to a plain string."""
    if isinstance(raw, str):
        return raw.strip()

    if not isinstance(raw, bytes):
        return str(raw).strip() if raw else None

    # Standard prefixes per EXIF spec
    if raw[:8] == b"UNICODE\x00":
        # The actual data observed is UTF-16-BE
        payload = raw[8:]
        # Try BE first, then LE
        for enc in ("utf-16-be", "utf-16-le"):
            try:
                decoded = payload.decode(enc)
                # Sanity check: should produce mostly ASCII-range chars for English prompts
                ascii_ratio = sum(1 for c in decoded[:50] if ord(c) < 128) / max(len(decoded[:50]), 1)
                if ascii_ratio > 0.5:
                    return decoded.strip()
            except (UnicodeDecodeError, ValueError):
                continue
        return payload.decode("utf-16-be", errors="ignore").strip()

    if raw[:8] == b"ASCII\x00\x00\x00":
        return raw[8:].decode("ascii", errors="ignore").strip()

    if raw[:4] == b"JIS\x00":
        return raw[4:].decode("shift_jis", errors="ignore").strip()

    # No recognized prefix — try UTF-8
    return raw.decode("utf-8", errors="ignore").strip()


def _parse_a1111_metadata(text: str) -> tuple[str, str | None, str | None, list[str], dict]:
    """Parse A1111/ComfyUI-style embedded metadata text.

    Format::

        <positive prompt>
        Negative prompt: <negative prompt>
        Steps: 20, Sampler: ..., CFG scale: 7.0, Scheduler: ..., Model: <name>, ..., Extra info: <lora>, None, None

    Returns:
        (positive, negative, base_model, loras, sampler_info)
        where sampler_info is a dict with keys: sampler, cfg_scale, scheduler, steps
    """
    positive = None
    negative = None
    base_model = None
    loras: list[str] = []
    sampler_info = {"sampler": None, "cfg_scale": None, "scheduler": None, "steps": None, "seed": None}

    # Split on "Negative prompt:" to get positive prompt
    neg_match = re.search(r'\nNegative prompt:\s*', text)
    if neg_match:
        positive = text[:neg_match.start()].strip()
        rest = text[neg_match.end():]
    else:
        # No negative prompt marker; look for the parameters line
        # which starts with "Steps:" or similar
        params_match = re.search(r'\nSteps:\s*', text)
        if params_match:
            positive = text[:params_match.start()].strip()
            rest = text[params_match.start():]
        else:
            positive = text.strip()
            rest = ""

    # Extract negative prompt (between "Negative prompt:" and the params line)
    if neg_match and rest:
        params_match = re.search(r'\nSteps:\s*', rest)
        if params_match:
            negative = rest[:params_match.start()].strip()
            params_line = rest[params_match.start():]
        else:
            negative = rest.strip()
            params_line = ""
    elif not neg_match:
        params_line = rest
    else:
        params_line = rest

    # Parse the parameters line for Model, sampler settings, and Extra info (LoRAs)
    if params_line:
        # Extract Model name
        model_match = re.search(r'\bModel:\s*([^,]+)', params_line)
        if model_match:
            base_model = model_match.group(1).strip()

        # Extract sampler settings
        steps_match = re.search(r'\bSteps:\s*(\d+)', params_line)
        if steps_match:
            sampler_info["steps"] = int(steps_match.group(1))

        sampler_match = re.search(r'\bSampler:\s*([^,]+)', params_line)
        if sampler_match:
            sampler_info["sampler"] = sampler_match.group(1).strip()

        cfg_match = re.search(r'\bCFG scale:\s*([\d.]+)', params_line)
        if cfg_match:
            try:
                sampler_info["cfg_scale"] = float(cfg_match.group(1))
            except ValueError:
                pass

        scheduler_match = re.search(r'\bScheduler:\s*([^,]+)', params_line)
        if scheduler_match:
            sampler_info["scheduler"] = scheduler_match.group(1).strip()

        seed_match = re.search(r'\bSeed:\s*(\d+)', params_line)
        if seed_match:
            sampler_info["seed"] = int(seed_match.group(1))

        # Extract LoRAs from "Extra info:" field
        # Format: "Extra info: lora_name.safetensors, None, None"
        extra_match = re.search(r'\bExtra info:\s*(.+?)(?:,\s*Version:|$)', params_line)
        if extra_match:
            extra_parts = extra_match.group(1).strip()
            for part in extra_parts.split(","):
                part = part.strip()
                if part and part != "None" and (
                    part.endswith(".safetensors") or part.endswith(".pt") or part.endswith(".ckpt")
                ):
                    loras.append(part)

    return positive or "", negative, base_model, loras, sampler_info


def _split_positive_negative(text: str) -> tuple[str, str | None]:
    """Simple split of text into positive and negative prompt sections."""
    pos, neg, _, _, _ = _parse_a1111_metadata(text)
    return pos, neg


# ---------------------------------------------------------------------------
# Filename fallback
# ---------------------------------------------------------------------------

def _extract_from_filename(filepath: str) -> ParsedImageData | None:
    """Try to extract prompt from the filename as a last resort.

    Observed filename patterns:
    - ``chroma-<prompt>_00001_.png``
    - ``finetune_v9_2-6-<seed>-<prompt>-sf.jpg``
    """
    basename = os.path.splitext(os.path.basename(filepath))[0]

    # Remove trailing generation suffixes
    cleaned = re.sub(r'_\d+_$', '', basename)
    cleaned = re.sub(r'-sf\d*$', '', cleaned)
    cleaned = re.sub(r'_sf\d*$', '', cleaned)

    # Pattern: finetune_v9_2-6-<seed>-<prompt>
    prefix_match = re.match(r'^[\w]+-(?:\d+-)*\d+-(.+)$', cleaned)
    if prefix_match:
        cleaned = prefix_match.group(1)
    elif cleaned.startswith("chroma-"):
        cleaned = cleaned[7:]

    if cleaned and len(cleaned) > 10:
        return ParsedImageData(
            prompt=cleaned,
            source_file=os.path.basename(filepath),
        )

    return None
