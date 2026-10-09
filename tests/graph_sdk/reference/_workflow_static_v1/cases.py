# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent workflow_static_v1 reference: cases."""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Sequence
from copy import deepcopy
from typing import cast

from tests.graph_sdk.reference._workflow_static_v1.builders import (
    _ceiling,
    _choice,
    _dependency,
    _input,
    _node,
    _one,
    _outcome_binding,
    _output,
    _p,
    _pipe,
    _q,
    _ref,
    _reowner_workflow,
    _workflow,
    _wrap,
    _z,
)
from tests.graph_sdk.reference._workflow_static_v1.model import (
    LIMIT_FIELDS,
    RESOURCE_FIELDS,
    Object,
    _obj,
    _sorted,
)
from tests.graph_sdk.reference._workflow_static_v1.traces import (
    _traces,
)
from tests.graph_sdk.reference._workflow_static_v1.validation import (
    _cycle_and_sinks,
    _list,
    _metrics,
    _nodes,
    _operation_outcomes,
    judge,
)


def _case(
    family: str, coordinates: Sequence[int], mutation: str, declaration: Object, replacement: Object | None = None
) -> Object:
    mode = "substitution" if replacement is not None else "admission"
    expected = judge(declaration, replacement)
    coordinate_text = "/".join(f"{value:03d}" for value in coordinates)
    case_id = f"{family}/{coordinate_text}/{mutation}"
    return {
        "case_id": case_id,
        "declaration": declaration,
        "expected": expected,
        "family": family,
        "mode": mode,
        "replacement": replacement,
        "traces": _traces(declaration, replacement, expected),
    }


def _topology_cases() -> Iterator[Object]:
    for n_ordinal, n in enumerate((1, 2, 3)):
        labels = tuple(f"N{index}" for index in range(n))
        domain = tuple((left, right) for left in labels for right in labels if left != right)
        for subset_ordinal, mask in enumerate(range(1 << len(domain))):
            edges = tuple(edge for index, edge in enumerate(domain) if mask & (1 << index))
            for permutation_ordinal, permutation in enumerate(itertools.permutations(labels)):
                declaration = _workflow(_z(n), tuple(_node(label, _z()) for label in permutation))
                declaration["family_marker"] = "topology"
                declaration["sequence"] = [{"after": _ref(after), "before": _ref(before)} for before, after in edges]
                cyclic, sinks = _cycle_and_sinks(declaration)
                if cyclic:
                    sink = "N0"
                elif sinks == 1:
                    outgoing = {before for before, _ in edges}
                    sink = next(label for label in labels if label not in outgoing)
                else:
                    sink = ""
                if sink:
                    declaration["outcome_bindings"] = [
                        _outcome_binding(sink, "ok", "ok"),
                        _outcome_binding(sink, "fail", "fail"),
                    ]
                yield _case("topology", (n_ordinal, subset_ordinal, permutation_ordinal), "edges", declaration)


def _ports_cases() -> Iterator[Object]:
    mutations = (
        "base",
        "input_type_A1",
        "output_type_A1",
        "remove_input_binding",
        "remove_output_binding",
        "duplicate_input_destination",
        "duplicate_output_destination",
        "remove_output_dependency",
        "duplicate_output_dependency",
        "missing_dependency_output",
        "missing_dependency_input",
        "missing_source_node",
        "missing_source_port",
        "missing_destination_node",
        "missing_destination_port",
        "internal_type_A1",
    )
    for base_ordinal, factory in enumerate((_one_q, _pipe_q)):
        for mutation_ordinal, mutation in enumerate(mutations):
            if mutation == "internal_type_A1" and base_ordinal == 0:
                continue
            declaration = factory()
            if mutation == "input_type_A1":
                _obj(_list(_obj(declaration["interface"])["inputs"])[0])["artifact_type"] = "A1@1"
            elif mutation == "output_type_A1":
                _obj(_list(_obj(declaration["interface"])["outputs"])[0])["artifact_type"] = "A1@1"
            elif mutation == "remove_input_binding":
                _list(declaration["input_bindings"]).pop(0)
            elif mutation == "remove_output_binding":
                _list(declaration["output_bindings"]).pop()
            elif mutation == "duplicate_input_destination":
                _list(declaration["input_bindings"]).append(deepcopy(_list(declaration["input_bindings"])[0]))
            elif mutation == "duplicate_output_destination":
                _list(declaration["output_bindings"]).append(deepcopy(_list(declaration["output_bindings"])[0]))
            elif mutation == "remove_output_dependency":
                _list(_obj(_nodes(declaration)[0]["operation"])["output_dependencies"]).clear()
            elif mutation == "duplicate_output_dependency":
                dependencies = _list(_obj(_nodes(declaration)[0]["operation"])["output_dependencies"])
                dependencies.append(deepcopy(dependencies[0]))
            elif mutation == "missing_dependency_output":
                _obj(_list(_obj(_nodes(declaration)[0]["operation"])["output_dependencies"])[0])["output"] = "o1"
            elif mutation == "missing_dependency_input":
                _obj(_list(_obj(_nodes(declaration)[0]["operation"])["output_dependencies"])[0])["inputs"] = ["i1"]
            elif mutation == "missing_source_node":
                _obj(_obj(_list(declaration["output_bindings"])[0])["source"])["node"] = _ref("N2")
            elif mutation == "missing_source_port":
                _obj(_obj(_list(declaration["output_bindings"])[0])["source"])["port"] = "o1"
            elif mutation == "missing_destination_node":
                _obj(_obj(_list(declaration["input_bindings"])[0])["destination"])["node"] = _ref("N2")
            elif mutation == "missing_destination_port":
                _obj(_obj(_list(declaration["input_bindings"])[0])["destination"])["port"] = "i1"
            elif mutation == "internal_type_A1":
                _obj(_list(_obj(_nodes(declaration)[1]["operation"])["inputs"])[0])["artifact_type"] = "A1@1"
            yield _case("ports", (base_ordinal, mutation_ordinal), mutation, declaration)


def _one_q() -> Object:
    return _one(_q())


def _pipe_q() -> Object:
    return _pipe()


def _choice_cases() -> Iterator[Object]:
    mutations = (
        "CHOICE_Z",
        "unmap_fail",
        "overlap_outcome",
        "overlap_member",
        "unknown_outcome",
        "selector_member",
        "remove_selector_edge",
        "second_choice_membership",
    )
    for ordinal, mutation in enumerate(mutations):
        declaration = _choice()
        branch_values = _list(_obj(_list(declaration["choices"])[0])["branches"])
        branches = [_obj(value) for value in branch_values]
        if mutation == "unmap_fail":
            branch_values.pop()
            declaration["nodes"] = _list(declaration["nodes"])[:2]
            declaration["sequence"] = _list(declaration["sequence"])[:1]
            declaration["outcome_bindings"] = _list(declaration["outcome_bindings"])[:2] + [
                _outcome_binding("N0", "fail", "fail")
            ]
        elif mutation == "overlap_outcome":
            _list(branches[1]["outcomes"]).append("ok")
        elif mutation == "overlap_member":
            _list(branches[1]["members"]).append(_ref("N1"))
        elif mutation == "unknown_outcome":
            branches[1]["outcomes"] = ["other"]
        elif mutation == "selector_member":
            _list(branches[0]["members"]).append(_ref("N0"))
        elif mutation == "remove_selector_edge":
            _list(declaration["sequence"]).pop(0)
        elif mutation == "second_choice_membership":
            _list(declaration["choices"]).append(
                {"branches": [{"members": [_ref("N1")], "outcomes": ["ok"]}], "selector": _ref("N2")}
            )
        yield _case("choice", (ordinal,), mutation, declaration)


def _change_semantic(operation: Object, mutation: str) -> None:
    if mutation == "input_type_A1":
        _obj(_list(operation["inputs"])[0])["artifact_type"] = "A1@1"
    elif mutation == "output_type_A1":
        _obj(_list(operation["outputs"])[0])["artifact_type"] = "A1@1"
    elif mutation == "dependency_empty":
        dependency = _obj(_list(operation["output_dependencies"])[0])
        dependency["inputs"] = []
        dependency["identity_input"] = None
    elif mutation == "remove_fail_outcome":
        operation["outcomes"] = [value for value in _list(operation["outcomes"]) if _obj(value)["name"] != "fail"]
    elif mutation == "context_alt":
        for outcome in map(_obj, _list(operation["outcomes"])):
            if _list(outcome["context"]):
                _obj(_list(outcome["context"])[0])["meaning"] = "context_alt"
    elif mutation == "evidence_meaning_alt":
        evidence = _list(_obj(_list(operation["outcomes"])[0])["evidence"])
        _obj(evidence[0])["meaning"] = "assessment_alt"
    elif mutation == "remove_field_coverage":
        evidence = _obj(_list(_obj(_list(operation["outcomes"])[0])["evidence"])[0])
        evidence["coverage"] = [value for value in _list(evidence["coverage"]) if _obj(value)["name"] != "field"]
    elif mutation == "state_alt":
        for outcome in map(_obj, _list(operation["outcomes"])):
            _obj(_list(outcome["state_effects"])[0])["name"] = "state_alt"
    elif mutation == "model_revision_2":
        for outcome in map(_obj, _list(operation["outcomes"])):
            _obj(_list(outcome["model_requirements"])[0])["revision"] = 2


def _consistent_body_mutation(body: Object, mutation: str) -> None:
    interface = _obj(body["interface"])
    affected_nodes = _nodes(body)
    if mutation in {"input_type_A1", "output_type_A1", "dependency_empty"}:
        _change_semantic(interface, mutation)
        target = affected_nodes[0] if mutation != "output_type_A1" or len(affected_nodes) == 1 else affected_nodes[-1]
        target_operation = _obj(target["operation"])
        _change_semantic(target_operation, mutation)
        if len(affected_nodes) > 1 and mutation in {"input_type_A1", "output_type_A1"}:
            _obj(_list(interface["output_dependencies"])[0])["identity_input"] = None
            _obj(_list(target_operation["output_dependencies"])[0])["identity_input"] = None
    else:
        _change_semantic(interface, mutation)
        target = affected_nodes[-1] if mutation == "remove_fail_outcome" else affected_nodes[0]
        _change_semantic(_obj(target["operation"]), mutation)
        if mutation == "remove_fail_outcome":
            body["outcome_bindings"] = [
                value for value in _list(body["outcome_bindings"]) if _obj(_obj(value)["source"])["outcome"] != "fail"
            ]


def _subgraph_cases() -> Iterator[Object]:
    bodies = (
        (_one(_z()), ()),
        (
            _one(_p()),
            (
                "input_type_A1",
                "output_type_A1",
                "dependency_empty",
                "context_alt",
                "evidence_meaning_alt",
                "remove_field_coverage",
                "state_alt",
                "model_revision_2",
            ),
        ),
        (_pipe(), ("input_type_A1", "output_type_A1", "dependency_empty")),
    )
    resources = tuple(f"widen_{field}_by_1" for field in RESOURCE_FIELDS)
    for body_ordinal, (base_body, extras) in enumerate(bodies):
        mutations = ("equal", "remove_fail_outcome", *resources, *extras)
        for mutation_ordinal, mutation in enumerate(mutations):
            body = deepcopy(base_body)
            declaration = _wrap(base_body)
            if mutation.startswith("widen_"):
                field = mutation.removeprefix("widen_").removesuffix("_by_1")
                for operation in (_obj(body["interface"]), _obj(_nodes(body)[0]["operation"])):
                    for outcome in map(_obj, _list(operation["outcomes"])):
                        ceiling = _obj(outcome["ceiling"])
                        ceiling[field] = cast(int, ceiling[field]) + 1
            elif mutation != "equal":
                _consistent_body_mutation(body, mutation)
            cast(list[Object], declaration["nodes"])[0]["body"] = _reowner_workflow(body, "W1")
            yield _case("subgraph", (body_ordinal, mutation_ordinal), mutation, declaration)


def _replacement_mutation(replacement: Object, mutation: str) -> None:
    interface = _obj(replacement["interface"])
    operation = _obj(_nodes(replacement)[0]["operation"])
    if mutation.startswith(("narrow_", "widen_")):
        direction = -1 if mutation.startswith("narrow_") else 1
        rest = mutation.split("_", 1)[1].removesuffix("_by_1")
        outcome_name, field = rest.split("_", 1)
        for current in (interface, operation):
            outcome = _operation_outcomes(current)[outcome_name]
            ceiling = _obj(outcome["ceiling"])
            ceiling[field] = cast(int, ceiling[field]) + direction
    else:
        _change_semantic(interface, mutation)
        _change_semantic(operation, mutation)
        if mutation == "remove_fail_outcome":
            replacement["outcome_bindings"] = [
                value
                for value in _list(replacement["outcome_bindings"])
                if _obj(_obj(value)["source"])["outcome"] != "fail"
            ]


def _substitution_cases() -> Iterator[Object]:
    base = _one(_p())
    base["substitution_target"] = _ref("N0")
    ceiling_mutations: list[str] = []
    for outcome, ceiling in (("ok", _ceiling(1, 1, 8, 8)), ("fail", _ceiling(1, 1, 8, 0))):
        for field in RESOURCE_FIELDS:
            if cast(int, ceiling[field]) > 0:
                ceiling_mutations.append(f"narrow_{outcome}_{field}_by_1")
            ceiling_mutations.append(f"widen_{outcome}_{field}_by_1")
    mutations = (
        "equal",
        *ceiling_mutations,
        "input_type_A1",
        "output_type_A1",
        "dependency_empty",
        "remove_fail_outcome",
        "context_alt",
        "evidence_meaning_alt",
        "remove_field_coverage",
        "state_alt",
        "model_revision_2",
    )
    for ordinal, mutation in enumerate(mutations):
        replacement = _reowner_workflow(_one(_p()), "W1")
        if mutation != "equal":
            _replacement_mutation(replacement, mutation)
        yield _case("substitution", (ordinal,), mutation, deepcopy(base), replacement)


def _protection_cases() -> Iterator[Object]:
    mutations = ("exact", "meaning_alt", "subject_i0", "add_consumed_i1", "add_field_alt", "add_absence")
    for ordinal, mutation in enumerate(mutations):
        declaration = _one(_p(two_inputs=True))
        requirement: Object = {
            "consumed_ports": ["i0"],
            "coverage": _sorted(({"kind": "field", "name": "field"}, {"kind": "source_view", "name": "source"})),
            "meaning": "assessment",
            "outcome": "ok",
            "subject_port": "o0",
        }
        if mutation == "meaning_alt":
            requirement["meaning"] = "assessment_alt"
        elif mutation == "subject_i0":
            requirement["subject_port"] = "i0"
        elif mutation == "add_consumed_i1":
            _list(requirement["consumed_ports"]).append("i1")
        elif mutation == "add_field_alt":
            _list(requirement["coverage"]).append({"kind": "field", "name": "field_alt"})
            requirement["coverage"] = _sorted(_list(requirement["coverage"]))
        elif mutation == "add_absence":
            _list(requirement["coverage"]).append({"kind": "absence", "name": "absence"})
            requirement["coverage"] = _sorted(_list(requirement["coverage"]))
        declaration["protection"] = [requirement]
        yield _case("protection", (ordinal,), mutation, declaration)


def _lineage_cases() -> Iterator[Object]:
    mutations = ("PIPE_L", "dependency_fanin", "zero_upstream", "zero_downstream", "many_downstream")
    for ordinal, mutation in enumerate(mutations):
        declaration = _pipe(lineage=True)
        if mutation == "dependency_fanin":
            _list(_obj(declaration["interface"])["inputs"]).append({"artifact_type": "A0@1", "name": "i1"})
            _list(_obj(_nodes(declaration)[0]["operation"])["inputs"]).append({"artifact_type": "A0@1", "name": "i1"})
            _list(declaration["input_bindings"]).append(_input("i1", "N0", "i1"))
            for operation in (_obj(declaration["interface"]), _obj(_nodes(declaration)[0]["operation"])):
                _obj(_list(operation["output_dependencies"])[0])["inputs"] = ["i0", "i1"]
        elif mutation == "zero_upstream":
            _obj(_list(_obj(_nodes(declaration)[0]["operation"])["output_dependencies"])[0])["identity_input"] = None
        elif mutation == "zero_downstream":
            _obj(_list(_obj(_nodes(declaration)[1]["operation"])["output_dependencies"])[0])["identity_input"] = None
        elif mutation == "many_downstream":
            interface = _obj(declaration["interface"])
            _list(interface["outputs"]).append({"artifact_type": "A0@1", "name": "o1"})
            _list(interface["output_dependencies"]).append(_dependency("o1", ("i0",), "i0"))
            for outcome in map(_obj, _list(interface["outcomes"])):
                _list(outcome["produced_ports"]).append("o1")
            _list(declaration["output_bindings"]).append(_output("N1", "o0", "o1"))
        yield _case("lineage", (ordinal,), mutation, declaration)


def _exact_limits(declaration: Object) -> None:
    metrics = _metrics(declaration)
    declaration["limits"] = {
        "max_bindings": metrics["bindings"],
        "max_branch_members": metrics["branch_members"],
        "max_choice_states": metrics["choice_states"],
        "max_choices": metrics["choices"],
        "max_nodes": metrics["nodes"],
        "max_sequence_edges": metrics["sequence_edges"],
        "max_subgraph_depth": metrics["subgraph_depth"],
    }


def _limits_cases() -> Iterator[Object]:
    bases = (_one(_p()), _pipe(), _choice())
    metric_for_limit = dict(
        zip(
            LIMIT_FIELDS,
            ("nodes", "bindings", "sequence_edges", "choices", "branch_members", "subgraph_depth", "choice_states"),
            strict=True,
        )
    )
    for base_ordinal, base in enumerate(bases):
        mutations = ["exact_all"]
        metrics = _metrics(base)
        for limit, metric in metric_for_limit.items():
            if metrics[metric] > (1 if limit in {"max_nodes", "max_subgraph_depth", "max_choice_states"} else 0):
                mutations.append(f"one_under_{metric}")
        for mutation_ordinal, mutation in enumerate(mutations):
            declaration = deepcopy(base)
            _exact_limits(declaration)
            if mutation != "exact_all":
                metric = mutation.removeprefix("one_under_")
                limit = next(key for key, value in metric_for_limit.items() if value == metric)
                _obj(declaration["limits"])[limit] = cast(int, _obj(declaration["limits"])[limit]) - 1
            yield _case("limits_ownership", (0, base_ordinal, mutation_ordinal), mutation, declaration)
    for wrapper_ordinal, body in enumerate((_one(_z()), _pipe())):
        base = _wrap(body)
        for mutation_ordinal, mutation in enumerate(("exact_all", "one_under_nodes", "one_under_subgraph_depth")):
            declaration = deepcopy(base)
            _exact_limits(declaration)
            if mutation == "one_under_nodes":
                _obj(declaration["limits"])["max_nodes"] = cast(int, _obj(declaration["limits"])["max_nodes"]) - 1
            elif mutation == "one_under_subgraph_depth":
                _obj(declaration["limits"])["max_subgraph_depth"] = 1
            yield _case("limits_ownership", (1, wrapper_ordinal, mutation_ordinal), mutation, declaration)
    foreign_specs = (
        ("operation_node_id", _one(_p())),
        ("subgraph_node_id", _wrap(_one(_z()))),
        ("output_ref", _one(_p())),
        ("input_ref", _one(_p())),
        ("outcome_ref", _one(_p())),
        ("sequence_before", _pipe()),
        ("sequence_after", _pipe()),
        ("choice_selector", _choice()),
        ("choice_member", _choice()),
        ("substitution_target", _one(_p())),
    )
    for ordinal, (location, declaration) in enumerate(foreign_specs):
        replacement: Object | None = None
        if location in {"operation_node_id", "subgraph_node_id"}:
            foreign_label = "N2" if location == "subgraph_node_id" else "N0"
            _nodes(declaration)[0]["id"] = _ref(foreign_label, "W1")
        elif location == "output_ref":
            _obj(_obj(_list(declaration["output_bindings"])[0])["source"])["node"] = _ref("N0", "W1")
        elif location == "input_ref":
            _obj(_obj(_list(declaration["input_bindings"])[0])["destination"])["node"] = _ref("N0", "W1")
        elif location == "outcome_ref":
            _obj(_obj(_list(declaration["outcome_bindings"])[0])["source"])["node"] = _ref("N0", "W1")
        elif location.startswith("sequence_"):
            _obj(_list(declaration["sequence"])[0])[location.split("_")[1]] = _ref("N0", "W1")
        elif location == "choice_selector":
            _obj(_list(declaration["choices"])[0])["selector"] = _ref("N0", "W1")
        elif location == "choice_member":
            _list(_obj(_list(_obj(_list(declaration["choices"])[0])["branches"])[0])["members"])[0] = _ref("N1", "W1")
        else:
            declaration["substitution_target"] = _ref("N2", "W1")
            replacement = _reowner_workflow(_one(_p()), "W1")
        yield _case("limits_ownership", (2, ordinal), f"foreign_owner_{location}", declaration, replacement)
