"""Workflow abstraction layer for ComfyUI workflow JSON manipulation.

Provides an abstract interface for reading/modifying ComfyUI workflow files
so that workflow structure changes don't break the application.  Each concrete
:class:`WorkflowDefinition` subclass knows how to extract user-configurable
settings from — and apply settings to — a specific workflow JSON structure.

Supports **both** workflow formats:

- **API format** (flat dict keyed by node-ID strings with ``inputs`` dicts)
  — this is what ComfyUI's ``/prompt`` endpoint expects.
- **UI format** (``{"nodes": [{...}, ...]}`` with ``widgets_values`` arrays)
  — the JSON exported from ComfyUI's graph editor.

The primary target is the API format since that's what gets submitted.
"""

from __future__ import annotations

import copy
import json
import logging
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def is_api_format(workflow: dict) -> bool:
    """Check if *workflow* is in API format vs UI format.

    API format is a flat dict where every key is a string node-ID and every
    value is a dict with at least ``class_type`` and ``inputs``.

    UI format has a top-level ``"nodes"`` array.
    """
    if "nodes" in workflow:
        return False
    # Quick sanity: all values should be dicts (the node objects)
    return bool(workflow) and all(isinstance(v, dict) for v in workflow.values())


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FieldDef:
    """Definition of an extractable/settable field in a workflow.

    Attributes:
        name:       Programmatic key, e.g. ``"positive_prompt"``.
        field_type: Widget hint for the frontend — one of ``"text"``,
                    ``"model_select"``, ``"lora_multi"``, ``"folder"``,
                    ``"integer"``.
        label:      Human-readable label shown in the UI.
        default:    Default value when no user override is provided.
        required:   Whether the field must be set before submission.
    """

    name: str
    field_type: str
    label: str
    default: Any = None
    required: bool = False


@dataclass
class NodeLookup:
    """Describes how to locate a node inside a workflow.

    The search tries *title first* — if ``title`` is not ``None`` the node
    whose ``title`` field (or ``_meta.title`` in API format) contains the
    given substring (case-insensitive) is returned.  If ``title`` is ``None``
    (or no match is found and ``class_type`` is set), the first node whose
    ``type``/``class_type`` matches exactly is returned instead.
    """

    title: str | None = None
    class_type: str | None = None


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class WorkflowDefinition(ABC):
    """Abstract base class for ComfyUI workflow definitions.

    Each subclass knows how to extract settings from — and apply settings
    to — a specific workflow JSON structure.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this workflow definition."""

    @property
    @abstractmethod
    def filename_pattern(self) -> str:
        """Regex pattern to match workflow filenames."""

    @abstractmethod
    def get_field_definitions(self) -> list[FieldDef]:
        """Return the fields this workflow supports for user configuration."""

    @abstractmethod
    def extract_settings(self, workflow: dict) -> dict:
        """Extract current settings from a workflow JSON.

        Returns:
            dict mapping :pyattr:`FieldDef.name` to current values.
        """

    @abstractmethod
    def apply_settings(self, workflow: dict, settings: dict) -> dict:
        """Apply user settings to a workflow JSON.

        Must **deep-copy** the workflow before mutating so the original
        is never modified.

        Returns:
            The modified workflow dict (a deep copy).
        """

    def matches_workflow(self, filename: str) -> bool:
        """Check if this definition matches a workflow *filename*."""
        return bool(re.search(self.filename_pattern, filename))

    # -- helpers available to all subclasses ----------------------------------

    @staticmethod
    def _find_node(
        workflow: dict,
        lookup: NodeLookup,
    ) -> dict | None:
        """Locate a single node inside a UI-format workflow.

        Searches the ``nodes`` array.  Returns the first matching node dict
        or ``None``.
        """
        nodes: list[dict] = workflow.get("nodes", [])

        # 1. Match by title substring (case-insensitive)
        if lookup.title is not None:
            needle = lookup.title.lower()
            for node in nodes:
                node_title = node.get("title", "")
                if isinstance(node_title, str) and needle in node_title.lower():
                    return node

        # 2. Fallback: match by class_type (exact)
        if lookup.class_type is not None:
            for node in nodes:
                if node.get("type") == lookup.class_type:
                    return node

        return None

    @staticmethod
    def _find_api_node(
        workflow: dict,
        lookup: NodeLookup,
    ) -> tuple[str | None, dict | None]:
        """Locate a node inside an API-format workflow.

        Returns ``(node_id, node_dict)`` or ``(None, None)`` if not found.
        Searches by ``_meta.title`` first, then by ``class_type``.
        """
        # 1. Match by _meta.title substring (case-insensitive)
        if lookup.title is not None:
            needle = lookup.title.lower()
            for node_id, node in workflow.items():
                meta_title = node.get("_meta", {}).get("title", "")
                if isinstance(meta_title, str) and needle in meta_title.lower():
                    return node_id, node

        # 2. Fallback: match by class_type (exact)
        if lookup.class_type is not None:
            for node_id, node in workflow.items():
                if node.get("class_type") == lookup.class_type:
                    return node_id, node

        return None, None


# ---------------------------------------------------------------------------
# Concrete implementation: Chroma Subgraph workflow
# ---------------------------------------------------------------------------

class ChromaSubgraphWorkflow(WorkflowDefinition):
    """Workflow definition for the *chroma_subgraph* family of workflows.

    Supports both API and UI format workflows.

    **API format** node mapping (from ``api_chroma_subgraph_png_2025-02-21.json``):

    +-----------------+----+-------------------------------+--------------------+-----------------+
    | Field           | ID | class_type                    | _meta.title        | inputs key      |
    +=================+====+===============================+====================+=================+
    | positive_prompt |  2 | PrimitiveStringMultiline      | String (Positive)  | value           |
    | negative_prompt |  3 | PrimitiveStringMultiline      | String (Negative)  | value           |
    | base_model      | 94 | UNETLoader                    | Load Diffusion …   | unet_name       |
    | loras           | 93 | Lora Loader Stack (rgthree)   | load_lora          | lora_01 … 04    |
    | output_folder   | 62 | PrimitiveStringMultiline      | String (Output …)  | value           |
    | seed            | 13 | Seed (rgthree)                | Seed (rgthree)     | seed            |
    +-----------------+----+-------------------------------+--------------------+-----------------+

    **UI format** node mapping (from ``chroma_subgraph_png_2025-02-21.json``):
    Values are stored in each node's ``widgets_values`` array.
    """

    # -- node lookups (title preferred, class_type as fallback) ---------------

    _POSITIVE_LOOKUP = NodeLookup(title="Positive", class_type="PrimitiveStringMultiline")
    _NEGATIVE_LOOKUP = NodeLookup(title="Negative", class_type=None)
    _MODEL_LOOKUP = NodeLookup(title="Load Diffusion Model", class_type="UNETLoader")
    _LORA_LOOKUP = NodeLookup(title="load_lora", class_type="Lora Loader Stack (rgthree)")
    _FOLDER_LOOKUP = NodeLookup(title="Output Folder", class_type=None)
    _SEED_LOOKUP = NodeLookup(title="seed", class_type="Seed (rgthree)")

    # Sampler node — try custom node first, then standard KSampler variants
    _SAMPLER_LOOKUPS = [
        NodeLookup(title=None, class_type="ClownsharKSampler_Beta"),
        NodeLookup(title=None, class_type="KSampler"),
        NodeLookup(title=None, class_type="KSamplerAdvanced"),
    ]

    # -- LoRA stack: API-format input key patterns ----------------------------
    _LORA_SLOT_KEYS: list[tuple[str, str]] = [
        ("lora_01", "strength_01"),
        ("lora_02", "strength_02"),
        ("lora_03", "strength_03"),
        ("lora_04", "strength_04"),
    ]

    # -- LoRA stack layout (UI format): (name_idx, strength_idx) pairs --------
    _LORA_SLOTS: list[tuple[int, int]] = [(0, 1), (2, 3), (4, 5), (6, 7)]

    # -- ABC implementation ---------------------------------------------------

    @property
    def name(self) -> str:
        return "chroma_subgraph"

    @property
    def filename_pattern(self) -> str:
        return r"chroma_subgraph"

    def get_field_definitions(self) -> list[FieldDef]:
        return [
            FieldDef(
                name="positive_prompt",
                field_type="text",
                label="Positive Prompt",
                default="",
                required=True,
            ),
            FieldDef(
                name="negative_prompt",
                field_type="text",
                label="Negative Prompt",
                default="",
            ),
            FieldDef(
                name="base_model",
                field_type="model_select",
                label="Base Model",
                default="",
                required=True,
            ),
            FieldDef(
                name="loras",
                field_type="lora_multi",
                label="LoRA(s)",
                default=[],
            ),
            FieldDef(
                name="output_folder",
                field_type="folder",
                label="Output Folder",
                default="",
            ),
            FieldDef(
                name="seed",
                field_type="integer",
                label="Seed",
                default=-1,
            ),
        ]

    # -- sampler node lookup (tries multiple class_types) ----------------------

    def _find_sampler_node_api(self, workflow: dict) -> tuple[str | None, dict | None]:
        """Locate the sampler node in an API-format workflow."""
        for lookup in self._SAMPLER_LOOKUPS:
            node_id, node = self._find_api_node(workflow, lookup)
            if node is not None:
                return node_id, node
        return None, None

    def _find_sampler_node_ui(self, workflow: dict) -> dict | None:
        """Locate the sampler node in a UI-format workflow."""
        for lookup in self._SAMPLER_LOOKUPS:
            node = self._find_node(workflow, lookup)
            if node is not None:
                return node
        return None

    # -- extract / apply: dispatch by format ----------------------------------

    def extract_settings(self, workflow: dict) -> dict:
        """Read current values from the workflow's nodes.

        Automatically detects API vs UI format and delegates.
        """
        if is_api_format(workflow):
            return self._extract_settings_api(workflow)
        return self._extract_settings_ui(workflow)

    def apply_settings(self, workflow: dict, settings: dict) -> dict:
        """Apply user-provided *settings* to a **deep copy** of *workflow*.

        Automatically detects API vs UI format and delegates.
        """
        if is_api_format(workflow):
            return self._apply_settings_api(workflow, settings)
        return self._apply_settings_ui(workflow, settings)

    # ======================================================================
    # API FORMAT
    # ======================================================================

    def _extract_settings_api(self, workflow: dict) -> dict:
        """Extract settings from an API-format workflow."""
        settings: dict[str, Any] = {}

        # -- Positive prompt --------------------------------------------------
        _, node = self._find_api_node(workflow, self._POSITIVE_LOOKUP)
        if node:
            settings["positive_prompt"] = node.get("inputs", {}).get("value", "")
        else:
            logger.warning("ChromaSubgraphWorkflow(API): Positive prompt node not found")
            settings["positive_prompt"] = ""

        # -- Negative prompt --------------------------------------------------
        _, node = self._find_api_node(workflow, self._NEGATIVE_LOOKUP)
        if node:
            settings["negative_prompt"] = node.get("inputs", {}).get("value", "")
        else:
            logger.warning("ChromaSubgraphWorkflow(API): Negative prompt node not found")
            settings["negative_prompt"] = ""

        # -- Base model -------------------------------------------------------
        _, node = self._find_api_node(workflow, self._MODEL_LOOKUP)
        if node:
            settings["base_model"] = node.get("inputs", {}).get("unet_name", "")
        else:
            logger.warning("ChromaSubgraphWorkflow(API): Base model node not found")
            settings["base_model"] = ""

        # -- LoRAs ------------------------------------------------------------
        _, node = self._find_api_node(workflow, self._LORA_LOOKUP)
        if node:
            settings["loras"] = self._extract_loras_api(node)
        else:
            logger.warning("ChromaSubgraphWorkflow(API): LoRA node not found")
            settings["loras"] = []

        # -- Output folder ----------------------------------------------------
        _, node = self._find_api_node(workflow, self._FOLDER_LOOKUP)
        if node:
            settings["output_folder"] = node.get("inputs", {}).get("value", "")
        else:
            logger.warning("ChromaSubgraphWorkflow(API): Output folder node not found")
            settings["output_folder"] = ""

        # -- Seed -------------------------------------------------------------
        _, node = self._find_api_node(workflow, self._SEED_LOOKUP)
        if node:
            settings["seed"] = node.get("inputs", {}).get("seed", -1)
        else:
            logger.warning("ChromaSubgraphWorkflow(API): Seed node not found")
            settings["seed"] = -1

        # -- Sampler settings (sampler, scheduler, cfg, steps) ----------------
        _, sampler_node = self._find_sampler_node_api(workflow)
        if sampler_node:
            inputs = sampler_node.get("inputs", {})
            # Remap node keys → settings keys (sampler_name→sampler, cfg→cfg_scale)
            settings["sampler"] = inputs.get("sampler_name")
            settings["cfg_scale"] = inputs.get("cfg")
            settings["scheduler"] = inputs.get("scheduler")
            settings["steps"] = inputs.get("steps")

        return settings

    def _apply_settings_api(self, workflow: dict, settings: dict) -> dict:
        """Apply user settings to an API-format workflow (deep copy)."""
        wf = copy.deepcopy(workflow)

        # -- Positive prompt --------------------------------------------------
        if "positive_prompt" in settings:
            _, node = self._find_api_node(wf, self._POSITIVE_LOOKUP)
            if node:
                node.setdefault("inputs", {})["value"] = settings["positive_prompt"]
            else:
                logger.warning("ChromaSubgraphWorkflow(API): Positive prompt node not found — skipping")

        # -- Negative prompt --------------------------------------------------
        if "negative_prompt" in settings:
            _, node = self._find_api_node(wf, self._NEGATIVE_LOOKUP)
            if node:
                node.setdefault("inputs", {})["value"] = settings["negative_prompt"]
            else:
                logger.warning("ChromaSubgraphWorkflow(API): Negative prompt node not found — skipping")

        # -- Base model -------------------------------------------------------
        if "base_model" in settings:
            _, node = self._find_api_node(wf, self._MODEL_LOOKUP)
            if node:
                node.setdefault("inputs", {})["unet_name"] = settings["base_model"]
            else:
                logger.warning("ChromaSubgraphWorkflow(API): Base model node not found — skipping")

        # -- LoRAs ------------------------------------------------------------
        if "loras" in settings:
            _, node = self._find_api_node(wf, self._LORA_LOOKUP)
            if node:
                self._apply_loras_api(node, settings["loras"])
            else:
                logger.warning("ChromaSubgraphWorkflow(API): LoRA node not found — skipping")

        # -- Output folder ----------------------------------------------------
        if "output_folder" in settings:
            _, node = self._find_api_node(wf, self._FOLDER_LOOKUP)
            if node:
                node.setdefault("inputs", {})["value"] = settings["output_folder"]
            else:
                logger.warning("ChromaSubgraphWorkflow(API): Output folder node not found — skipping")

        # -- Seed -------------------------------------------------------------
        if "seed" in settings:
            _, node = self._find_api_node(wf, self._SEED_LOOKUP)
            if node:
                seed_value = settings["seed"]
                if seed_value == -1:
                    seed_value = random.randint(0, 1125899906842624)
                node.setdefault("inputs", {})["seed"] = seed_value
            else:
                logger.warning("ChromaSubgraphWorkflow(API): Seed node not found — skipping")

        # -- Sampler settings (sampler, scheduler, cfg, steps) ----------------
        _, sampler_node = self._find_sampler_node_api(wf)
        if sampler_node:
            inputs = sampler_node.setdefault("inputs", {})
            # Remap settings keys → node keys (sampler→sampler_name, cfg_scale→cfg)
            if settings.get("sampler") is not None:
                inputs["sampler_name"] = settings["sampler"]
            if settings.get("cfg_scale") is not None:
                inputs["cfg"] = float(settings["cfg_scale"])
            if settings.get("scheduler") is not None:
                inputs["scheduler"] = settings["scheduler"]
            if settings.get("steps") is not None:
                inputs["steps"] = int(settings["steps"])

        return wf

    # -- API-format LoRA helpers ----------------------------------------------

    def _extract_loras_api(self, node: dict) -> list[dict[str, Any]]:
        """Extract LoRA names and strengths from an API-format Lora Loader Stack node.

        Returns a list of dicts like::

            [{"name": "my_lora.safetensors", "strength": 1.0}, ...]

        Slots whose name is ``"None"`` or empty are skipped.
        """
        inputs = node.get("inputs", {})
        loras: list[dict[str, Any]] = []
        for lora_key, strength_key in self._LORA_SLOT_KEYS:
            lora_name = inputs.get(lora_key)
            if isinstance(lora_name, str) and lora_name.strip() and lora_name != "None":
                strength = inputs.get(strength_key, 1.0)
                loras.append({"name": lora_name, "strength": float(strength)})
        return loras

    def _apply_loras_api(
        self,
        node: dict,
        loras: list[dict[str, Any] | str],
    ) -> None:
        """Write LoRA names and strengths into an API-format Lora Loader Stack node.

        *loras* may be a list of dicts (``{"name": ..., "strength": ...}``)
        or plain strings (just the filename).  Unused slots are set to
        ``"None"`` / ``0.5``.
        """
        inputs = node.setdefault("inputs", {})
        for slot_idx, (lora_key, strength_key) in enumerate(self._LORA_SLOT_KEYS):
            if slot_idx < len(loras):
                lora = _normalize_lora(loras[slot_idx])
                inputs[lora_key] = lora["name"]
                inputs[strength_key] = float(lora["strength"])
            else:
                inputs[lora_key] = "None"
                inputs[strength_key] = 0.5

    # ======================================================================
    # UI FORMAT (backward-compatible)
    # ======================================================================

    def _extract_settings_ui(self, workflow: dict) -> dict:
        """Extract settings from a UI-format workflow (``nodes`` array)."""
        settings: dict[str, Any] = {}

        # -- Positive prompt --------------------------------------------------
        node = self._find_node(workflow, self._POSITIVE_LOOKUP)
        if node:
            settings["positive_prompt"] = _widget_value(node, 0, "")
        else:
            logger.warning("ChromaSubgraphWorkflow(UI): Positive prompt node not found")
            settings["positive_prompt"] = ""

        # -- Negative prompt --------------------------------------------------
        node = self._find_node(workflow, self._NEGATIVE_LOOKUP)
        if node:
            settings["negative_prompt"] = _widget_value(node, 0, "")
        else:
            logger.warning("ChromaSubgraphWorkflow(UI): Negative prompt node not found")
            settings["negative_prompt"] = ""

        # -- Base model -------------------------------------------------------
        node = self._find_node(workflow, self._MODEL_LOOKUP)
        if node:
            settings["base_model"] = _widget_value(node, 0, "")
        else:
            logger.warning("ChromaSubgraphWorkflow(UI): Base model node not found")
            settings["base_model"] = ""

        # -- LoRAs ------------------------------------------------------------
        node = self._find_node(workflow, self._LORA_LOOKUP)
        if node:
            settings["loras"] = self._extract_loras_ui(node)
        else:
            logger.warning("ChromaSubgraphWorkflow(UI): LoRA node not found")
            settings["loras"] = []

        # -- Output folder ----------------------------------------------------
        node = self._find_node(workflow, self._FOLDER_LOOKUP)
        if node:
            settings["output_folder"] = _widget_value(node, 0, "")
        else:
            logger.warning("ChromaSubgraphWorkflow(UI): Output folder node not found")
            settings["output_folder"] = ""

        # -- Seed -------------------------------------------------------------
        node = self._find_node(workflow, self._SEED_LOOKUP)
        if node:
            settings["seed"] = _widget_value(node, 0, -1)
        else:
            logger.warning("ChromaSubgraphWorkflow(UI): Seed node not found")
            settings["seed"] = -1

        # -- Sampler settings (sampler, scheduler, cfg, steps) ----------------
        # UI format: widget_values layout depends on node type.  For
        # ClownsharKSampler_Beta the order observed in the workflow JSON is:
        #   [eta, sampler_name, scheduler, steps, cfg, seed_link, ...]
        # We don't hard-code indices — instead look up the node's inputs spec
        # at runtime only when we have object_info.  For now, skip UI extract
        # since API format is the primary submission target.

        return settings

    def _apply_settings_ui(self, workflow: dict, settings: dict) -> dict:
        """Apply user settings to a UI-format workflow (deep copy)."""
        wf = copy.deepcopy(workflow)

        # -- Positive prompt --------------------------------------------------
        if "positive_prompt" in settings:
            node = self._find_node(wf, self._POSITIVE_LOOKUP)
            if node:
                _set_widget_value(node, 0, settings["positive_prompt"])
            else:
                logger.warning("ChromaSubgraphWorkflow(UI): Positive prompt node not found — skipping")

        # -- Negative prompt --------------------------------------------------
        if "negative_prompt" in settings:
            node = self._find_node(wf, self._NEGATIVE_LOOKUP)
            if node:
                _set_widget_value(node, 0, settings["negative_prompt"])
            else:
                logger.warning("ChromaSubgraphWorkflow(UI): Negative prompt node not found — skipping")

        # -- Base model -------------------------------------------------------
        if "base_model" in settings:
            node = self._find_node(wf, self._MODEL_LOOKUP)
            if node:
                _set_widget_value(node, 0, settings["base_model"])
            else:
                logger.warning("ChromaSubgraphWorkflow(UI): Base model node not found — skipping")

        # -- LoRAs ------------------------------------------------------------
        if "loras" in settings:
            node = self._find_node(wf, self._LORA_LOOKUP)
            if node:
                self._apply_loras_ui(node, settings["loras"])
            else:
                logger.warning("ChromaSubgraphWorkflow(UI): LoRA node not found — skipping")

        # -- Output folder ----------------------------------------------------
        if "output_folder" in settings:
            node = self._find_node(wf, self._FOLDER_LOOKUP)
            if node:
                _set_widget_value(node, 0, settings["output_folder"])
            else:
                logger.warning("ChromaSubgraphWorkflow(UI): Output folder node not found — skipping")

        # -- Seed -------------------------------------------------------------
        if "seed" in settings:
            node = self._find_node(wf, self._SEED_LOOKUP)
            if node:
                seed_value = settings["seed"]
                if seed_value == -1:
                    seed_value = random.randint(0, 1125899906842624)
                _set_widget_value(node, 0, seed_value)
            else:
                logger.warning("ChromaSubgraphWorkflow(UI): Seed node not found — skipping")

        # -- Sampler settings (sampler, scheduler, cfg, steps) ----------------
        # UI format: skip for now — API format is the submission target.
        # UI workflow is only passed as extra_pnginfo for introspection nodes.

        return wf

    # -- UI-format LoRA helpers -----------------------------------------------

    def _extract_loras_ui(self, node: dict) -> list[dict[str, Any]]:
        """Extract LoRA names and strengths from a UI-format ``Lora Loader Stack`` node."""
        widgets = node.get("widgets_values", [])
        loras: list[dict[str, Any]] = []
        for name_idx, strength_idx in self._LORA_SLOTS:
            if name_idx >= len(widgets):
                break
            lora_name = widgets[name_idx]
            if isinstance(lora_name, str) and lora_name.strip() and lora_name != "None":
                strength = widgets[strength_idx] if strength_idx < len(widgets) else 1.0
                loras.append({"name": lora_name, "strength": float(strength)})
        return loras

    def _apply_loras_ui(
        self,
        node: dict,
        loras: list[dict[str, Any] | str],
    ) -> None:
        """Write LoRA names and strengths into a UI-format ``Lora Loader Stack`` node."""
        widgets = node.get("widgets_values", [])
        while len(widgets) < 8:
            widgets.append(None)
        node["widgets_values"] = widgets

        for slot_idx, (name_idx, strength_idx) in enumerate(self._LORA_SLOTS):
            if slot_idx < len(loras):
                lora = _normalize_lora(loras[slot_idx])
                widgets[name_idx] = lora["name"]
                widgets[strength_idx] = float(lora["strength"])
            else:
                widgets[name_idx] = "None"
                widgets[strength_idx] = 0.5


# ---------------------------------------------------------------------------
# Private helper functions
# ---------------------------------------------------------------------------

def _normalize_lora(lora: dict | str) -> dict[str, Any]:
    """Normalize a LoRA item to ``{"name": str, "strength": float}``.

    Accepts either a dict with ``name``/``strength`` keys or a plain string
    (just the filename, strength defaults to 1.0).
    """
    if isinstance(lora, str):
        return {"name": lora, "strength": 1.0}
    return {"name": lora.get("name", "None"), "strength": float(lora.get("strength", 1.0))}


def _widget_value(node: dict, index: int, default: Any = None) -> Any:
    """Safely read ``widgets_values[index]`` from a node."""
    widgets = node.get("widgets_values", [])
    if index < len(widgets):
        return widgets[index]
    return default


def _set_widget_value(node: dict, index: int, value: Any) -> None:
    """Safely write *value* into ``widgets_values[index]``, extending if needed."""
    widgets = node.get("widgets_values")
    if widgets is None:
        widgets = []
        node["widgets_values"] = widgets
    while len(widgets) <= index:
        widgets.append(None)
    widgets[index] = value


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class WorkflowRegistry:
    """Registry of available :class:`WorkflowDefinition` instances.

    Matches workflow files to their appropriate definition.
    """

    def __init__(self) -> None:
        self._definitions: list[WorkflowDefinition] = []

    def register(self, definition: WorkflowDefinition) -> None:
        """Add a definition to the registry."""
        self._definitions.append(definition)
        logger.debug("Registered workflow definition: %s", definition.name)

    def get_definition(self, filename: str) -> WorkflowDefinition | None:
        """Find the :class:`WorkflowDefinition` that matches *filename*."""
        for defn in self._definitions:
            if defn.matches_workflow(filename):
                return defn
        return None

    def list_definitions(self) -> list[dict]:
        """List all registered definitions with their names and field defs."""
        return [
            {
                "name": d.name,
                "filename_pattern": d.filename_pattern,
                "fields": d.get_field_definitions(),
            }
            for d in self._definitions
        ]


# ---------------------------------------------------------------------------
# Module-level singleton & convenience functions
# ---------------------------------------------------------------------------

_registry = WorkflowRegistry()


def get_registry() -> WorkflowRegistry:
    """Return the module-level :class:`WorkflowRegistry` singleton."""
    return _registry


def get_definition_for_workflow(filename: str) -> WorkflowDefinition | None:
    """Shortcut — find the definition matching *filename*."""
    return _registry.get_definition(filename)


def load_workflow(filepath: str) -> dict:
    """Load a workflow JSON file from disk.

    Args:
        filepath: Path (absolute or relative) to the ``.json`` file.

    Returns:
        The parsed workflow dict.

    Raises:
        FileNotFoundError: If *filepath* does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    path = Path(filepath)
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_workflow_from_string(json_string: str) -> dict:
    """Parse a workflow from a JSON string.

    Args:
        json_string: Raw JSON text of the workflow.

    Returns:
        The parsed workflow dict.

    Raises:
        json.JSONDecodeError: If the string is not valid JSON.
    """
    return json.loads(json_string)


def prepare_workflow(filepath: str, settings: dict) -> dict:
    """Load a workflow and apply *settings*.

    Automatically finds the correct :class:`WorkflowDefinition` for the file
    based on its filename, then calls :meth:`apply_settings`.

    Args:
        filepath: Path to the workflow ``.json`` file.
        settings: Dict mapping field names to desired values.

    Returns:
        The modified workflow dict (ready to submit to ComfyUI if API format).

    Raises:
        FileNotFoundError: If *filepath* does not exist.
        ValueError: If no registered definition matches the filename.
    """
    path = Path(filepath)
    workflow = load_workflow(filepath)

    defn = _registry.get_definition(path.name)
    if defn is None:
        raise ValueError(
            f"No workflow definition registered for '{path.name}'. "
            f"Registered patterns: "
            f"{[d.filename_pattern for d in _registry._definitions]}"
        )

    return defn.apply_settings(workflow, settings)


def prepare_workflow_from_json(
    workflow_json: str,
    filename: str,
    settings: dict,
) -> dict:
    """Parse a workflow from a JSON string and apply *settings*.

    Args:
        workflow_json: Raw JSON text of the workflow.
        filename: Original filename, used to match a WorkflowDefinition.
        settings: Dict mapping field names to desired values.

    Returns:
        The modified workflow dict (ready to submit to ComfyUI if API format).

    Raises:
        json.JSONDecodeError: If *workflow_json* is not valid JSON.
        ValueError: If no registered definition matches *filename*.
    """
    workflow = json.loads(workflow_json)

    defn = _registry.get_definition(filename)
    if defn is None:
        raise ValueError(
            f"No workflow definition registered for '{filename}'. "
            f"Registered patterns: "
            f"{[d.filename_pattern for d in _registry._definitions]}"
        )

    return defn.apply_settings(workflow, settings)


# ---------------------------------------------------------------------------
# Auto-register known workflow definitions
# ---------------------------------------------------------------------------

_registry.register(ChromaSubgraphWorkflow())
