# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent workflow_static_v1 reference: traces."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from typing import cast

from tests.graph_sdk.reference._workflow_static_v1.model import (
    SEMANTIC_ORDERS,
    Json,
    Object,
    _canonical,
    _obj,
)
from tests.graph_sdk.reference._workflow_static_v1.validation import (
    _id_label,
    _list,
    _nodes,
    judge,
)


def _events(declaration: Object, *, substitution: bool) -> list[Json]:
    events: list[Json] = [{"op": "new_workflow", "workflow": declaration["workflow"]}, {"op": "declare_interface"}]
    for node in _nodes(declaration):
        events.append({"node": node["id"], "op": "declare_subgraph" if node["kind"] == "subgraph" else "declare_node"})
    for binding in _list(declaration["input_bindings"]):
        events.append({"destination": _obj(binding)["destination"], "op": "bind_input"})
    for binding in _list(declaration["output_bindings"]):
        events.append({"destination": _obj(binding)["destination"], "op": "bind_output"})
    for binding in _list(declaration["outcome_bindings"]):
        events.append({"destination": _obj(binding)["destination"], "op": "bind_output"})
    for edge in _list(declaration["sequence"]):
        events.append({"edge": edge, "op": "add_sequence"})
    for choice in _list(declaration["choices"]):
        events.append({"choice": choice, "op": "add_choice"})
    for requirement in _list(declaration["protection"]):
        events.append({"op": "declare_protection", "requirement": requirement})
    events.append({"op": "admit"})
    if substitution:
        events.append({"op": "substitute", "target": declaration["substitution_target"]})
    return events


def independent(left: Object, right: Object) -> bool:
    """Return the exact conditional symmetric trace-independence relation."""
    left_op, right_op = left.get("op"), right.get("op")
    if left_op in {"declare_node", "declare_subgraph"} and right_op in {"declare_node", "declare_subgraph"}:
        return left.get("node") != right.get("node")
    if left_op in {"bind_input", "bind_output"} and right_op in {"bind_input", "bind_output"}:
        return left.get("destination") != right.get("destination")
    if left_op == right_op == "add_sequence":
        return left.get("edge") != right.get("edge")
    if left_op == right_op == "add_choice":
        left_choice, right_choice = _obj(cast(Json, left["choice"])), _obj(cast(Json, right["choice"]))
        left_members = {
            _canonical(member)
            for branch in map(_obj, _list(left_choice["branches"]))
            for member in _list(branch["members"])
        }
        right_members = {
            _canonical(member)
            for branch in map(_obj, _list(right_choice["branches"]))
            for member in _list(branch["members"])
        }
        left_nodes = left_members | {_canonical(left_choice["selector"])}
        right_nodes = right_members | {_canonical(right_choice["selector"])}
        return not left_nodes & right_nodes
    if left_op == right_op == "declare_protection":
        keys = ("outcome", "meaning", "subject_port")
        left_requirement, right_requirement = (
            _obj(cast(Json, left["requirement"])),
            _obj(cast(Json, right["requirement"])),
        )
        return tuple(left_requirement[key] for key in keys) != tuple(right_requirement[key] for key in keys)
    return False


def _rename_maps(declaration: Object, replacement: Object | None) -> dict[str, dict[str, str]]:
    labels = [_id_label(node["id"]) for node in _nodes(declaration)]
    if len(labels) == 1:
        nodes = {labels[0]: labels[0]}
    elif len(labels) == 2:
        nodes = {labels[0]: labels[1], labels[1]: labels[0]}
    else:
        ordered = sorted(labels)
        nodes = {label: ordered[(index + 1) % len(ordered)] for index, label in enumerate(ordered)}
    text = json.dumps((declaration, replacement), sort_keys=True)
    mappings = {"node": nodes}
    for role, order in SEMANTIC_ORDERS.items():
        present = [item for item in order if f'"{item}"' in text]
        mappings[role] = (
            {item: present[(index + 1) % len(present)] for index, item in enumerate(present)}
            if len(present) > 1
            else {item: item for item in present}
        )
    return mappings


def _rewrite_role(value: str, role: str, mappings: Mapping[str, Mapping[str, str]]) -> str:
    return mappings.get(role, {}).get(value, value)


def _rewrite(value: Json, mappings: Mapping[str, Mapping[str, str]], *, parent_key: str = "") -> Json:
    if isinstance(value, str):
        if parent_key == "label":
            return _rewrite_role(value, "node", mappings)
        if parent_key in {
            "port",
            "subject_port",
            "identity_input",
            "output",
            "inputs",
            "produced_ports",
            "consumed_ports",
        }:
            rewritten = _rewrite_role(value, "input_port", mappings)
            return _rewrite_role(rewritten, "output_port", mappings)
        if parent_key == "meaning":
            rewritten = _rewrite_role(value, "context", mappings)
            return _rewrite_role(rewritten, "evidence", mappings)
        if parent_key == "capability":
            return _rewrite_role(value, "model", mappings)
        return value
    if isinstance(value, list):
        return [_rewrite(item, mappings, parent_key=parent_key) for item in value]
    if isinstance(value, dict):
        rewritten = {key: _rewrite(item, mappings, parent_key=key) for key, item in value.items()}
        name = value.get("name")
        if isinstance(name, str):
            if "artifact_type" in value:
                port = _rewrite_role(name, "input_port", mappings)
                rewritten["name"] = _rewrite_role(port, "output_port", mappings)
            elif value.get("kind") in {"field", "source_view", "evaluation", "absence"}:
                rewritten["name"] = _rewrite_role(name, "coverage", mappings)
            elif value.get("kind") in {"read", "write"}:
                rewritten["name"] = _rewrite_role(name, "state", mappings)
            elif "consumed_ports" in value and "subject_port" in value:
                rewritten["name"] = _rewrite_role(name, "evidence", mappings)
        return rewritten
    return value


def _traces(declaration: Object, replacement: Object | None, expected: Object) -> list[Json]:
    mappings = _rename_maps(declaration, replacement)
    renamed = cast(Object, _rewrite(declaration, mappings))
    renamed_replacement = cast(Object | None, _rewrite(replacement, mappings)) if replacement is not None else None
    traces: list[Json] = [
        {
            "declaration": renamed,
            "events": _events(renamed, substitution=replacement is not None),
            "expected": judge(renamed, renamed_replacement),
            "replacement": renamed_replacement,
            "transformation": "rename",
        }
    ]
    if len(_nodes(declaration)) >= 2:
        reversed_declaration = deepcopy(declaration)
        reversed_declaration["nodes"] = list(reversed(_list(reversed_declaration["nodes"])))
        traces.append(
            {
                "declaration": reversed_declaration,
                "events": _events(reversed_declaration, substitution=replacement is not None),
                "expected": judge(reversed_declaration, replacement),
                "replacement": deepcopy(replacement),
                "transformation": "reverse_declaration_tuple",
            }
        )
    invariant_keys = ("status", "code", "topology")
    for trace_value in traces:
        trace = _obj(trace_value)
        if any(_obj(trace["expected"])[key] != expected[key] for key in invariant_keys):
            raise ValueError("metamorphic invariant failed")
    return traces
