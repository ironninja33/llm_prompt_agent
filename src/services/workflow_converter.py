"""Generic ComfyUI UI→API workflow format converter.

Converts a UI-format workflow (exported from ComfyUI's graph editor with a
``"nodes"`` array and ``"links"`` array) to API format (flat dict keyed by
node-ID strings with ``"inputs"``, ``"class_type"``, ``"_meta"``).

Requires the ``/object_info`` response from ComfyUI which describes each
node class's input types and names.

Handles two categories of frontend-only nodes that are not present in
``/object_info`` and must be resolved/expanded during conversion:

- **Reroute nodes** (``Reroute``, ``Reroute (rgthree)``) — simple
  passthrough nodes whose references are traced back to the real upstream
  source.
- **Group/component nodes** (identified by UUID or ``workflow/...`` type
  names, defined in ``extra.groupNodes``) — composite nodes that are
  expanded into their constituent internal API nodes.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Primitive widget types — inputs with these types get their values
# from widgets_values rather than from link connections.
_WIDGET_TYPES = frozenset({"INT", "FLOAT", "STRING", "BOOLEAN"})

# Reroute node class types — frontend-only passthrough nodes that don't
# appear in /object_info and must be resolved to their upstream source.
_REROUTE_TYPES = frozenset({"Reroute", "Reroute (rgthree)"})


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def convert_ui_to_api(ui_workflow: dict, object_info: dict) -> dict:
    """Convert a UI-format workflow to API format.

    Args:
        ui_workflow: Parsed UI-format workflow dict (has ``"nodes"`` and
            ``"links"``).
        object_info: Response from ComfyUI's ``GET /object_info`` endpoint.

    Returns:
        API-format workflow dict — flat dict keyed by node-ID strings,
        each value having ``"inputs"``, ``"class_type"``, ``"_meta"``.

    Raises:
        ValueError: If the workflow is missing required structure.
    """
    if "nodes" not in ui_workflow:
        raise ValueError("UI workflow missing 'nodes' array")
    if "links" not in ui_workflow:
        raise ValueError("UI workflow missing 'links' array")

    link_map = _build_link_map(ui_workflow["links"])
    node_by_id: dict[int, dict] = {
        node["id"]: node for node in ui_workflow["nodes"] if "id" in node
    }

    # -- Phase 1: Build reroute resolution map --------------------------------
    reroute_map = _build_reroute_resolution(
        ui_workflow["nodes"], link_map, node_by_id,
    )

    # -- Phase 2: Expand group/component nodes --------------------------------
    group_api_nodes, group_output_remap = _expand_group_nodes(
        ui_workflow, object_info, link_map, node_by_id,
    )

    # -- Phase 3: Convert regular nodes ---------------------------------------
    api_workflow: dict[str, dict] = {}
    skipped_ids: set[str] = set()

    for node in ui_workflow["nodes"]:
        class_type = node.get("type", "")
        node_id = node.get("id")
        node_id_str = str(node_id)

        # Skip reroute nodes — resolved in post-processing
        if class_type in _REROUTE_TYPES:
            skipped_ids.add(node_id_str)
            continue

        # Skip group nodes — expanded into internal nodes
        if node_id_str in group_output_remap:
            skipped_ids.add(node_id_str)
            continue

        result = _convert_node(node, object_info, link_map)
        if result is not None:
            api_workflow[node_id_str] = result
        else:
            skipped_ids.add(node_id_str)

    # -- Phase 4: Merge expanded group internal nodes -------------------------
    api_workflow.update(group_api_nodes)

    # -- Phase 5: Resolve dangling references ---------------------------------
    _resolve_references(api_workflow, reroute_map, group_output_remap)

    logger.info(
        "Converted UI workflow: %d UI nodes → %d API nodes "
        "(%d reroutes resolved, %d group nodes expanded into %d internal nodes)",
        len(ui_workflow["nodes"]),
        len(api_workflow),
        len(reroute_map),
        len(group_output_remap),
        len(group_api_nodes),
    )
    return api_workflow


# ---------------------------------------------------------------------------
# Link map
# ---------------------------------------------------------------------------

def _build_link_map(links: list) -> dict[int, tuple[int, int]]:
    """Build a mapping from link_id to (source_node_id, source_output_index).

    Each link entry is::

        [link_id, source_node_id, source_output_idx,
         target_node_id, target_input_idx, type_string]

    Args:
        links: The ``"links"`` array from a UI-format workflow.

    Returns:
        Dict mapping ``link_id`` → ``(source_node_id, source_output_index)``.
    """
    link_map: dict[int, tuple[int, int]] = {}
    for link in links:
        if isinstance(link, (list, tuple)) and len(link) >= 4:
            link_id = link[0]
            source_node_id = link[1]
            source_output_idx = link[2]
            link_map[link_id] = (source_node_id, source_output_idx)
    return link_map


# ---------------------------------------------------------------------------
# Reroute resolution
# ---------------------------------------------------------------------------

def _build_reroute_resolution(
    nodes: list[dict],
    link_map: dict[int, tuple[int, int]],
    node_by_id: dict[int, dict],
) -> dict[str, tuple[str, int]]:
    """Build a resolution map that traces through reroute chains.

    Reroute nodes are frontend-only passthrough nodes (one input, one
    output, same type).  They don't exist in ``/object_info`` and can't
    appear in the API workflow.  This function traces each reroute's input
    back through any chain of reroutes to the *real* upstream source node.

    Args:
        nodes: The ``"nodes"`` array from a UI-format workflow.
        link_map: Mapping from link_id to ``(source_node_id, output_index)``.
        node_by_id: Mapping from node ID to node dict.

    Returns:
        Dict mapping ``reroute_node_id_str`` →
        ``(real_source_node_id_str, real_source_output_idx)``.
    """
    reroute_ids: set[int] = set()
    for node in nodes:
        if node.get("type", "") in _REROUTE_TYPES:
            reroute_ids.add(node["id"])

    if not reroute_ids:
        return {}

    # For each reroute, find its direct input source from the link map.
    direct_source: dict[int, tuple[int, int]] = {}
    for node_id in reroute_ids:
        node = node_by_id.get(node_id)
        if not node:
            continue
        for inp in (node.get("inputs") or []):
            if not isinstance(inp, dict):
                continue
            link_id = inp.get("link")
            if link_id is not None and link_id in link_map:
                direct_source[node_id] = link_map[link_id]
                break  # reroute has exactly one input

    # Chase through reroute chains with cycle detection.
    resolution: dict[str, tuple[str, int]] = {}

    def _resolve(node_id: int, visited: frozenset[int] = frozenset()) -> tuple[int, int] | None:
        if node_id in visited:
            logger.warning("Cycle detected in reroute chain at node %s", node_id)
            return None
        if node_id not in direct_source:
            return None
        src_id, src_slot = direct_source[node_id]
        if src_id in reroute_ids:
            return _resolve(src_id, visited | {node_id})
        return (src_id, src_slot)

    for rr_id in reroute_ids:
        result = _resolve(rr_id)
        if result is not None:
            resolution[str(rr_id)] = (str(result[0]), result[1])
            logger.debug(
                "Reroute %d → node %d output %d", rr_id, result[0], result[1],
            )
        else:
            logger.warning("Reroute node %d could not be resolved to a source", rr_id)

    return resolution


# ---------------------------------------------------------------------------
# Group / component node expansion
# ---------------------------------------------------------------------------

def _expand_group_nodes(
    ui_workflow: dict,
    object_info: dict,
    link_map: dict[int, tuple[int, int]],
    node_by_id: dict[int, dict],
) -> tuple[dict[str, dict], dict[str, dict[int, tuple[str, int]]]]:
    """Expand group/component nodes into their constituent internal API nodes.

    ComfyUI group nodes (sometimes called component nodes) are composite
    nodes defined in ``ui_workflow["extra"]["groupNodes"]``.  Their
    ``class_type`` in the UI workflow is typically a UUID or
    ``"workflow/GroupName"`` — not present in ``/object_info``.

    This function:

    1. Identifies group node instances (nodes whose type is not in
       ``object_info`` and not a reroute).
    2. Looks up their definition in ``extra.groupNodes``.
    3. Converts each internal node to API format with a unique composite
       ID (``"{group_instance_id}:{internal_index}"``).
    4. Wires internal links between the expanded nodes.
    5. Maps external input connections to the correct internal nodes.
    6. Builds an *output remapping* so downstream references like
       ``["110", 2]`` can be rewritten to the correct internal node.

    Args:
        ui_workflow: Full UI-format workflow dict.
        object_info: ``/object_info`` response.
        link_map: Link ID to ``(source_node_id, output_index)`` map.
        node_by_id: Node ID to node dict map.

    Returns:
        Tuple of ``(expanded_api_nodes, output_remap)``.

        - ``expanded_api_nodes``: dict of new API nodes keyed by composite
          ID strings (e.g. ``"110:0"``, ``"110:1"``).
        - ``output_remap``: dict mapping ``group_instance_id_str`` →
          ``{output_slot: (internal_node_id_str, internal_output_slot)}``.
    """
    group_defs = ui_workflow.get("extra", {}).get("groupNodes", {})
    expanded_api_nodes: dict[str, dict] = {}
    output_remap: dict[str, dict[int, tuple[str, int]]] = {}

    # DEBUG: Log available group definitions
    if group_defs:
        logger.warning(
            "DEBUG: extra.groupNodes keys: %r", list(group_defs.keys()),
        )
    else:
        logger.warning("DEBUG: No extra.groupNodes found in workflow")
        # Also log what keys are in extra
        extra = ui_workflow.get("extra", {})
        if extra:
            logger.warning("DEBUG: extra keys: %r", list(extra.keys()))

    for node in ui_workflow["nodes"]:
        class_type = node.get("type", "")
        node_id = node.get("id")
        node_id_str = str(node_id)

        # Skip nodes that are already in object_info (regular nodes)
        if class_type in object_info:
            continue
        # Skip reroute nodes (handled separately)
        if class_type in _REROUTE_TYPES:
            continue
        # Skip nodes with no type
        if not class_type:
            continue

        logger.warning(
            "DEBUG: Trying group expansion for node %s type='%s'",
            node_id, class_type,
        )
        # DEBUG: Dump full node properties for identification
        props = node.get("properties", {})
        logger.warning(
            "DEBUG: node %s properties=%r",
            node_id, props,
        )
        # Log inputs and outputs for matching
        logger.warning(
            "DEBUG: node %s inputs=%r",
            node_id, node.get("inputs", []),
        )
        logger.warning(
            "DEBUG: node %s outputs=%r",
            node_id, node.get("outputs", []),
        )
        logger.warning(
            "DEBUG: node %s widgets_values=%r",
            node_id, node.get("widgets_values", []),
        )

        # Try to find the group definition
        group_def = _find_group_definition(class_type, group_defs)
        if group_def is None:
            logger.warning(
                "Node %s has unresolvable class_type '%s' — "
                "not in object_info or extra.groupNodes, skipping",
                node_id, class_type,
            )
            continue

        # Expand this group node
        internal_nodes, slot_remap = _expand_single_group(
            node, group_def, object_info, link_map, node_id_str,
        )

        expanded_api_nodes.update(internal_nodes)
        output_remap[node_id_str] = slot_remap

        logger.debug(
            "Expanded group node %s (%s) into %d internal nodes, "
            "output remap: %s",
            node_id, class_type, len(internal_nodes), slot_remap,
        )

    return expanded_api_nodes, output_remap


def _find_group_definition(
    class_type: str,
    group_defs: dict,
) -> dict | None:
    """Look up a group node definition by class_type.

    Tries several matching strategies:

    1. Exact match on class_type.
    2. Strip ``"workflow/"`` prefix and match.
    3. Match by UUID if any group definition has a matching ``id`` field.

    Args:
        class_type: The node's ``type`` field from the UI workflow.
        group_defs: The ``extra.groupNodes`` dict from the workflow.

    Returns:
        The group definition dict, or ``None`` if not found.
    """
    if not group_defs:
        return None

    # 1. Exact match
    if class_type in group_defs:
        return group_defs[class_type]

    # 2. Strip "workflow/" prefix
    stripped = class_type
    if stripped.startswith("workflow/"):
        stripped = stripped[len("workflow/"):]
        if stripped in group_defs:
            return group_defs[stripped]

    # 3. Check if any group definition has a matching "id" field
    for _name, defn in group_defs.items():
        if isinstance(defn, dict) and defn.get("id") == class_type:
            return defn

    return None


def _expand_single_group(
    group_instance: dict,
    group_def: dict,
    object_info: dict,
    link_map: dict[int, tuple[int, int]],
    group_id_str: str,
) -> tuple[dict[str, dict], dict[int, tuple[str, int]]]:
    """Expand a single group node instance into internal API nodes.

    Args:
        group_instance: The group node dict from the UI workflow.
        group_def: The group definition from ``extra.groupNodes``.
        object_info: ``/object_info`` response.
        link_map: Link map for external connections.
        group_id_str: String ID of the group instance node.

    Returns:
        Tuple of ``(internal_api_nodes, output_slot_remap)``.
    """
    internal_nodes_def = group_def.get("nodes", [])
    internal_links = group_def.get("links", [])
    external_defs = group_def.get("external", [])

    internal_api_nodes: dict[str, dict] = {}
    # Map internal node index → composite API node ID string
    internal_id_map: dict[int, str] = {}

    # -- Step 1: Distribute widgets_values to internal nodes ------------------
    #
    # The group instance's widgets_values is a flat concatenation of all
    # internal nodes' widget values.  We distribute them by iterating
    # internal nodes in order and consuming widget values.
    instance_widgets = group_instance.get("widgets_values", []) or []
    widget_cursor = 0

    for idx, inode_def in enumerate(internal_nodes_def):
        inode_type = inode_def.get("type", "")
        composite_id = f"{group_id_str}:{idx}"
        internal_id_map[idx] = composite_id

        class_info = object_info.get(inode_type)
        if class_info is None:
            # Internal node type not in object_info (could be a nested
            # reroute or unknown node) — skip but still advance cursor.
            logger.debug(
                "Group %s internal node %d has unknown type '%s', skipping",
                group_id_str, idx, inode_type,
            )
            # Try to advance cursor using the definition's widgets_values count
            def_widgets = inode_def.get("widgets_values", []) or []
            widget_cursor += len(def_widgets)
            continue

        # Determine how many widget values this internal node consumes
        widget_names = _get_widget_names(class_info)
        api_inputs: dict[str, Any] = {}

        for w_idx, w_name in enumerate(widget_names):
            abs_idx = widget_cursor + w_idx
            if abs_idx < len(instance_widgets):
                value = instance_widgets[abs_idx]
                # Skip button-type widget values (rgthree UI artefacts)
                if isinstance(value, dict) and value.get("type") == "button":
                    continue
                api_inputs[w_name] = value

        widget_cursor += len(widget_names)

        title = inode_def.get("title") or inode_type
        internal_api_nodes[composite_id] = {
            "inputs": api_inputs,
            "class_type": inode_type,
            "_meta": {"title": f"{group_instance.get('title', 'Group')} / {title}"},
        }

    # -- Step 2: Wire internal links ------------------------------------------
    #
    # Internal links use node indices (into the internal_nodes_def array),
    # not node IDs.  Format:
    #   [link_id, src_node_idx, src_output_slot,
    #    dst_node_idx, dst_input_slot, type_string]
    for ilink in internal_links:
        if not isinstance(ilink, (list, tuple)) or len(ilink) < 6:
            continue
        _link_id, src_idx, src_slot, dst_idx, dst_slot, _type_str = ilink[:6]

        src_composite = internal_id_map.get(src_idx)
        dst_composite = internal_id_map.get(dst_idx)

        if src_composite is None or dst_composite is None:
            continue

        dst_api = internal_api_nodes.get(dst_composite)
        if dst_api is None:
            continue

        # Determine the input name for dst_input_slot
        dst_type = internal_nodes_def[dst_idx].get("type", "")
        dst_class_info = object_info.get(dst_type)
        if dst_class_info is None:
            continue

        all_input_names = _get_all_input_names(dst_class_info)
        # dst_slot is the connection input slot index.  Connection inputs
        # are those that are NOT widget inputs, in order.  We need to find
        # the right input name.
        conn_names = _get_connection_input_names(dst_class_info)
        if dst_slot < len(conn_names):
            inp_name = conn_names[dst_slot]
        elif dst_slot < len(all_input_names):
            inp_name = all_input_names[dst_slot]
        else:
            logger.debug(
                "Group %s internal link: dst slot %d exceeds input count "
                "for node %s (%s)",
                group_id_str, dst_slot, dst_composite, dst_type,
            )
            continue

        dst_api["inputs"][inp_name] = [src_composite, src_slot]

    # -- Step 3: Map external inputs ------------------------------------------
    #
    # The group instance's "inputs" array lists external connections.
    # Each entry has "name", "link", and possibly "slot_index".
    # These connect to internal nodes that have unconnected inputs.
    #
    # The external_defs array (from the group definition) maps external
    # input/output slots to internal node indices and slots.
    instance_inputs = group_instance.get("inputs", []) or []
    for inp in instance_inputs:
        if not isinstance(inp, dict):
            continue
        link_id = inp.get("link")
        if link_id is None or link_id not in link_map:
            continue

        source_node_id, source_output_idx = link_map[link_id]
        inp_name = inp.get("name", "")
        slot_index = inp.get("slot_index", 0)

        # Find which internal node this external input maps to.
        # Strategy: look for an internal node that has an input with
        # matching name that isn't already wired.
        _map_external_input_to_internal(
            internal_api_nodes, internal_nodes_def, internal_id_map,
            object_info, inp_name, source_node_id, source_output_idx,
            external_defs, slot_index, is_input=True,
        )

    # -- Step 4: Build output slot remapping ----------------------------------
    #
    # For each output slot of the group instance, determine which internal
    # node's output provides it.
    output_slot_remap: dict[int, tuple[str, int]] = {}
    instance_outputs = group_instance.get("outputs", []) or []

    # Strategy 1: Use external_defs if available
    # The external array typically has entries for outputs that map
    # group output slots to internal node indices and output slots.
    output_external_entries = _extract_output_externals(
        external_defs, len(instance_inputs), len(instance_outputs),
    )

    if output_external_entries:
        for out_slot, (inode_idx, inode_slot) in enumerate(output_external_entries):
            composite_id = internal_id_map.get(inode_idx)
            if composite_id is not None:
                output_slot_remap[out_slot] = (composite_id, inode_slot)

    # Strategy 2: If no external defs, find internal nodes whose outputs
    # aren't consumed by any internal link (they must be external).
    if not output_slot_remap and instance_outputs:
        output_slot_remap = _infer_output_remap(
            internal_nodes_def, internal_links, internal_id_map,
            object_info, instance_outputs,
        )

    return internal_api_nodes, output_slot_remap


def _get_connection_input_names(class_info: dict) -> list[str]:
    """Extract ordered list of *connection* (non-widget) input names.

    These are inputs whose type is a non-primitive connection type
    (e.g. ``MODEL``, ``CLIP``, ``IMAGE``).

    Args:
        class_info: The object_info entry for a single node class.

    Returns:
        Ordered list of connection input names.
    """
    conn_names: list[str] = []
    input_def = class_info.get("input", {})

    for section in ("required", "optional"):
        section_inputs = input_def.get(section)
        if not section_inputs or not isinstance(section_inputs, dict):
            continue
        for name, spec in section_inputs.items():
            if not isinstance(spec, (list, tuple)) or len(spec) < 1:
                continue
            type_spec = spec[0]
            if not _is_widget_input(type_spec):
                conn_names.append(name)

    return conn_names


def _map_external_input_to_internal(
    internal_api_nodes: dict[str, dict],
    internal_nodes_def: list[dict],
    internal_id_map: dict[int, str],
    object_info: dict,
    inp_name: str,
    source_node_id: int,
    source_output_idx: int,
    external_defs: list,
    slot_index: int,
    is_input: bool,
) -> None:
    """Map an external input connection to the correct internal node.

    Searches internal nodes for one that has an input with matching name
    that hasn't been wired yet (i.e., isn't set to a link reference).
    """
    source_ref = [str(source_node_id), source_output_idx]

    # Try to find an internal node with a matching input name
    for idx, inode_def in enumerate(internal_nodes_def):
        composite_id = internal_id_map.get(idx)
        if composite_id is None or composite_id not in internal_api_nodes:
            continue

        inode_type = inode_def.get("type", "")
        class_info = object_info.get(inode_type)
        if class_info is None:
            continue

        all_names = _get_all_input_names(class_info)
        if inp_name in all_names:
            api_node = internal_api_nodes[composite_id]
            # Only set if not already wired by an internal link
            current = api_node["inputs"].get(inp_name)
            if current is None or not isinstance(current, list):
                api_node["inputs"][inp_name] = source_ref
                return

    # Fallback: just log a warning
    logger.debug(
        "Could not map external input '%s' to any internal node "
        "(source: node %d slot %d)",
        inp_name, source_node_id, source_output_idx,
    )


def _extract_output_externals(
    external_defs: list,
    num_inputs: int,
    num_outputs: int,
) -> list[tuple[int, int]]:
    """Extract output slot mappings from the group's external definitions.

    The ``external`` array may contain both input and output mappings.
    Output entries typically come after input entries.  Each entry is
    ``[internal_node_index, internal_slot_index]`` or
    ``[internal_node_index, internal_slot_index, name]``.

    Returns:
        List of ``(internal_node_index, internal_output_slot)`` for each
        output slot, or empty list if the format is unrecognised.
    """
    if not external_defs or not isinstance(external_defs, list):
        return []

    # The external array often has num_inputs entries for inputs followed
    # by num_outputs entries for outputs.
    total_expected = num_inputs + num_outputs
    if len(external_defs) == total_expected and total_expected > 0:
        output_entries = external_defs[num_inputs:]
        result = []
        for entry in output_entries:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                result.append((entry[0], entry[1]))
            else:
                return []  # format not recognised
        return result

    # If the external array length matches just the outputs
    if len(external_defs) == num_outputs and num_outputs > 0:
        result = []
        for entry in external_defs:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                result.append((entry[0], entry[1]))
            else:
                return []
        return result

    return []


def _infer_output_remap(
    internal_nodes_def: list[dict],
    internal_links: list,
    internal_id_map: dict[int, str],
    object_info: dict,
    instance_outputs: list[dict],
) -> dict[int, tuple[str, int]]:
    """Infer output slot remapping by finding internal outputs not consumed internally.

    An internal node output that has no internal link consuming it is an
    external output of the group.  We match these to the group instance's
    output slots by type.

    Returns:
        Dict mapping output slot index → ``(composite_id, internal_output_slot)``.
    """
    # Collect all (dst_node_idx, dst_input_slot) pairs from internal links
    consumed_outputs: set[tuple[int, int]] = set()
    for ilink in internal_links:
        if isinstance(ilink, (list, tuple)) and len(ilink) >= 4:
            src_idx, src_slot = ilink[1], ilink[2]
            consumed_outputs.add((src_idx, src_slot))

    # Find unconsumed outputs
    unconsumed: list[tuple[int, int, str]] = []  # (node_idx, output_slot, type)
    for idx, inode_def in enumerate(internal_nodes_def):
        inode_type = inode_def.get("type", "")
        class_info = object_info.get(inode_type)
        if class_info is None:
            continue
        outputs = class_info.get("output", [])
        for out_slot, out_type in enumerate(outputs):
            if (idx, out_slot) not in consumed_outputs:
                unconsumed.append((idx, out_slot, out_type))

    # Match unconsumed outputs to instance output slots by type
    remap: dict[int, tuple[str, int]] = {}
    used_unconsumed: set[int] = set()

    for out_slot_idx, out_def in enumerate(instance_outputs):
        out_type = out_def.get("type", "")
        for uc_idx, (node_idx, node_slot, uc_type) in enumerate(unconsumed):
            if uc_idx in used_unconsumed:
                continue
            if uc_type == out_type or out_type == "*":
                composite_id = internal_id_map.get(node_idx)
                if composite_id is not None:
                    remap[out_slot_idx] = (composite_id, node_slot)
                    used_unconsumed.add(uc_idx)
                    break

    return remap


# ---------------------------------------------------------------------------
# Reference resolution (post-processing)
# ---------------------------------------------------------------------------

def _resolve_references(
    api_workflow: dict[str, dict],
    reroute_map: dict[str, tuple[str, int]],
    group_output_remap: dict[str, dict[int, tuple[str, int]]],
) -> None:
    """Rewrite any input references that point to reroute or group nodes.

    Walks every input in the API workflow.  Any value that is a link
    reference ``[node_id_str, output_idx]`` pointing to a reroute or
    group node is replaced with the resolved/remapped target.

    Mutates *api_workflow* in place.
    """
    for node_id, node in api_workflow.items():
        inputs = node.get("inputs", {})
        for inp_name, inp_value in list(inputs.items()):
            if not isinstance(inp_value, list) or len(inp_value) != 2:
                continue

            ref_node_str = str(inp_value[0])
            ref_slot = inp_value[1]

            # Check reroute map
            if ref_node_str in reroute_map:
                resolved_node, resolved_slot = reroute_map[ref_node_str]
                inputs[inp_name] = [resolved_node, resolved_slot]
                logger.debug(
                    "Node %s input '%s': resolved reroute %s → %s:%d",
                    node_id, inp_name, ref_node_str,
                    resolved_node, resolved_slot,
                )
                continue

            # Check group output remap
            if ref_node_str in group_output_remap:
                slot_map = group_output_remap[ref_node_str]
                if ref_slot in slot_map:
                    resolved_node, resolved_slot = slot_map[ref_slot]
                    inputs[inp_name] = [resolved_node, resolved_slot]
                    logger.debug(
                        "Node %s input '%s': resolved group %s slot %d → %s:%d",
                        node_id, inp_name, ref_node_str, ref_slot,
                        resolved_node, resolved_slot,
                    )
                else:
                    logger.warning(
                        "Node %s input '%s': group %s output slot %d has no "
                        "remap — leaving reference as-is",
                        node_id, inp_name, ref_node_str, ref_slot,
                    )
                continue

            # Check if the referenced node is in the API workflow
            if ref_node_str not in api_workflow:
                logger.warning(
                    "Node %s input '%s': references missing node %s "
                    "(not in API workflow, reroute map, or group remap)",
                    node_id, inp_name, ref_node_str,
                )


# ---------------------------------------------------------------------------
# Widget / input type helpers
# ---------------------------------------------------------------------------

def _is_widget_input(type_spec: Any) -> bool:
    """Determine if an input type specification represents a widget (vs a connection).

    Widget inputs get values from ``widgets_values``.
    Connection inputs get values from link references.

    Args:
        type_spec: The first element of the input definition tuple from
            object_info.  May be a list (combo/enum) or a string (type name).

    Returns:
        ``True`` if this is a widget input, ``False`` if it's a connection.
    """
    # If type_spec is a list, it's a combo/enum → widget
    if isinstance(type_spec, list):
        return True
    # If type_spec is a string in _WIDGET_TYPES → widget
    if isinstance(type_spec, str) and type_spec in _WIDGET_TYPES:
        return True
    # Otherwise (e.g., "MODEL", "CLIP", "IMAGE", "LATENT") → connection
    return False


def _get_widget_names(class_info: dict) -> list[str]:
    """Extract the ordered list of widget input names from a class's object_info.

    Iterates ``required`` then ``optional`` inputs, returning only those
    whose type specification indicates a widget input (primitives and
    combo/enums), not connection inputs.

    Args:
        class_info: The object_info entry for a single node class.  Expected
            to have ``class_info["input"]["required"]`` and optionally
            ``class_info["input"]["optional"]``.

    Returns:
        Ordered list of widget input names.
    """
    widget_names: list[str] = []
    input_def = class_info.get("input", {})

    for section in ("required", "optional"):
        section_inputs = input_def.get(section)
        if not section_inputs or not isinstance(section_inputs, dict):
            continue
        for name, spec in section_inputs.items():
            # spec is typically [type_spec, config_dict] or [type_spec]
            if not isinstance(spec, (list, tuple)) or len(spec) < 1:
                continue
            type_spec = spec[0]
            if _is_widget_input(type_spec):
                widget_names.append(name)

    return widget_names


def _get_all_input_names(class_info: dict) -> list[str]:
    """Extract all input names (both widget and connection) from a class's object_info.

    Args:
        class_info: The object_info entry for a single node class.

    Returns:
        Ordered list of all input names (required first, then optional).
    """
    all_names: list[str] = []
    input_def = class_info.get("input", {})

    for section in ("required", "optional"):
        section_inputs = input_def.get(section)
        if not section_inputs or not isinstance(section_inputs, dict):
            continue
        for name in section_inputs:
            all_names.append(name)

    return all_names


# ---------------------------------------------------------------------------
# Single node conversion
# ---------------------------------------------------------------------------

def _convert_node(
    node: dict,
    object_info: dict,
    link_map: dict[int, tuple[int, int]],
) -> dict | None:
    """Convert a single UI-format node to API format.

    Steps:

    1. Look up the node's ``class_type`` in object_info.
    2. Determine widget input names and map ``widgets_values`` positionally.
    3. Resolve connection inputs from the node's ``inputs`` array using
       the link map.
    4. Assemble the API-format node dict.

    Args:
        node: A single node dict from the UI workflow's ``"nodes"`` array.
        object_info: Full ``/object_info`` response.
        link_map: Mapping from link_id to ``(source_node_id, output_index)``.

    Returns:
        API-format node dict with ``"inputs"``, ``"class_type"``,
        ``"_meta"``, or ``None`` if the node's class type is not found
        in object_info.
    """
    class_type = node.get("type", "")
    if not class_type:
        logger.debug("Skipping node %s with no type", node.get("id"))
        return None

    class_info = object_info.get(class_type)
    if class_info is None:
        logger.warning(
            "Node %s has unknown class_type '%s' — not in object_info, skipping",
            node.get("id"),
            class_type,
        )
        return None

    # --- Step 1: Map widgets_values to widget input names ---
    widget_names = _get_widget_names(class_info)
    widgets_values = node.get("widgets_values", []) or []
    api_inputs: dict[str, Any] = {}

    for idx, name in enumerate(widget_names):
        if idx < len(widgets_values):
            value = widgets_values[idx]
            # Skip non-serialisable widget values (e.g. button dicts from
            # rgthree nodes that have {"type": "button", ...}).  These are
            # UI-only artefacts that should not appear in the API workflow.
            if isinstance(value, dict) and value.get("type") == "button":
                continue
            api_inputs[name] = value
        else:
            # More widget names than widgets_values — leave unset
            logger.debug(
                "Node %s (%s): widget '%s' at index %d has no matching "
                "widgets_value (only %d values)",
                node.get("id"),
                class_type,
                name,
                idx,
                len(widgets_values),
            )

    # --- Step 2: Resolve connection inputs from node's inputs array ---
    # The node's "inputs" array lists all connectable inputs.  Each entry
    # has a "name", "type", and "link" (null if not connected, or an int
    # link_id if connected).  Entries with a "widget" key are "converted
    # widgets" — widget inputs exposed as connectable.  If their link is
    # non-null, the connection overrides the widget value.
    node_inputs = node.get("inputs", []) or []
    for inp in node_inputs:
        if not isinstance(inp, dict):
            continue
        link_id = inp.get("link")
        if link_id is None:
            continue
        # This input has an active connection
        inp_name = inp.get("name", "")
        if not inp_name:
            continue
        if link_id not in link_map:
            logger.debug(
                "Node %s (%s): input '%s' references link_id %s not in link map",
                node.get("id"),
                class_type,
                inp_name,
                link_id,
            )
            continue
        source_node_id, source_output_idx = link_map[link_id]
        # API format references: [str(source_node_id), output_index]
        api_inputs[inp_name] = [str(source_node_id), source_output_idx]

    # --- Step 3: Assemble the API node ---
    title = node.get("title") or class_type
    return {
        "inputs": api_inputs,
        "class_type": class_type,
        "_meta": {"title": title},
    }
