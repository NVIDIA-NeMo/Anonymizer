# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent workflow_static_v1 reference: builders."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from copy import deepcopy
from typing import cast

from tests.graph_sdk.reference._workflow_static_v1.model import (
    Json,
    Object,
    _obj,
    _sorted,
)


def _ref(label: str, owner: str = "W0") -> Object:
    return {"label": label, "owner": owner}


def _ceiling(a: int, m: int, i: int, o: int) -> Object:
    return {
        "max_activations": a,
        "max_input_bytes": i,
        "max_model_requests": m,
        "max_output_bytes": o,
    }


def _outcome(
    name: str,
    category: str,
    produced: Iterable[str],
    ceiling: Object,
    *,
    context: Iterable[Object] = (),
    evidence: Iterable[Object] = (),
    state: Iterable[Object] = (),
    models: Iterable[Object] = (),
) -> Object:
    return {
        "category": category,
        "ceiling": deepcopy(ceiling),
        "context": _sorted(deepcopy(list(context))),
        "evidence": _sorted(deepcopy(list(evidence))),
        "model_requirements": _sorted(deepcopy(list(models))),
        "name": name,
        "produced_ports": sorted(produced),
        "state_effects": _sorted(deepcopy(list(state))),
    }


def _dependency(output: str, inputs: Iterable[str], identity: str | None) -> Object:
    return {"identity_input": identity, "inputs": sorted(inputs), "output": output}


def _operation(
    name: str,
    inputs: Sequence[tuple[str, str]],
    outputs: Sequence[tuple[str, str]],
    dependencies: Sequence[Object],
    outcomes: Sequence[Object],
) -> Object:
    return {
        "inputs": [{"artifact_type": artifact, "name": port} for port, artifact in inputs],
        "name": name,
        "outcomes": deepcopy(list(outcomes)),
        "output_dependencies": deepcopy(list(dependencies)),
        "outputs": [{"artifact_type": artifact, "name": port} for port, artifact in outputs],
    }


def _context(port: str, meaning: str) -> Object:
    return {"capture": "whole_artifact", "meaning": meaning, "port": port}


def _evidence(name: str, meaning: str, subject: str, consumed: Iterable[str], coverage: Iterable[str]) -> Object:
    return {
        "consumed_ports": sorted(consumed),
        "coverage": _sorted({"kind": "source_view" if item == "source" else item, "name": item} for item in coverage),
        "meaning": meaning,
        "name": name,
        "subject_port": subject,
    }


def _z(n: int = 1) -> Object:
    ceiling = _ceiling(n, 0, 0, 0)
    return _operation(
        "Z" if n == 1 else f"Z[{n}]",
        (),
        (),
        (),
        (_outcome("ok", "success", (), ceiling), _outcome("fail", "failure", (), ceiling)),
    )


def _p(*, two_inputs: bool = False) -> Object:
    contexts = [_context("i0", "context")]
    inputs = [("i0", "A0@1")]
    if two_inputs:
        inputs.append(("i1", "A0@1"))
        contexts.append(_context("i1", "context_alt"))
    state: list[Object] = [{"kind": "read", "name": "state"}]
    models: list[Object] = [{"capability": "model", "revision": 1}]
    return _operation(
        "P2I" if two_inputs else "P",
        inputs,
        (("o0", "A0@1"),),
        (_dependency("o0", ("i0",), None),),
        (
            _outcome(
                "ok",
                "success",
                ("o0",),
                _ceiling(1, 1, 8, 8),
                context=contexts,
                evidence=(_evidence("assessment", "assessment", "o0", ("i0",), ("field", "source")),),
                state=state,
                models=models,
            ),
            _outcome(
                "fail",
                "failure",
                (),
                _ceiling(1, 1, 8, 0),
                context=contexts,
                state=state,
                models=models,
            ),
        ),
    )


def _q(*, factor: int = 1) -> Object:
    ceiling = _ceiling(factor, 0, 8 * factor, 8 * factor)
    return _operation(
        "Q" if factor == 1 else "Q2",
        (("i0", "A0@1"),),
        (("o0", "A0@1"),),
        (_dependency("o0", ("i0",), "i0"),),
        (_outcome("ok", "success", ("o0",), ceiling), _outcome("fail", "failure", ("o0",), ceiling)),
    )


def _l(index: int) -> Object:
    meaning = "context" if index == 0 else "context_alt"
    assessment = "assessment" if index == 0 else "assessment_alt"
    result = _q()
    result["name"] = f"L{index}"
    for outcome in cast(list[Object], result["outcomes"]):
        outcome["context"] = [_context("i0", meaning)]
        outcome["evidence"] = [_evidence(assessment, assessment, "o0", ("i0",), ("field", "source"))]
    return result


def _l01() -> Object:
    result = _q(factor=2)
    result["name"] = "L01"
    for outcome in cast(list[Object], result["outcomes"]):
        outcome["context"] = _sorted((_context("i0", "context"), _context("i0", "context_alt")))
        outcome["evidence"] = _sorted(
            (
                _evidence("assessment", "assessment", "o0", ("i0",), ("field", "source")),
                _evidence("assessment_alt", "assessment_alt", "o0", ("i0",), ("field", "source")),
            )
        )
    return result


def _limits(
    *,
    nodes: int = 3,
    bindings: int = 16,
    edges: int = 8,
    choices: int = 2,
    members: int = 4,
    depth: int = 2,
    states: int = 4,
) -> Object:
    return {
        "max_bindings": bindings,
        "max_branch_members": members,
        "max_choice_states": states,
        "max_choices": choices,
        "max_nodes": nodes,
        "max_sequence_edges": edges,
        "max_subgraph_depth": depth,
    }


def _node(label: str, operation: Object, *, owner: str = "W0", body: Object | None = None) -> Object:
    value: Object = {
        "id": _ref(label, owner),
        "kind": "subgraph" if body is not None else "operation",
        "operation": deepcopy(operation),
    }
    if body is not None:
        value["body"] = deepcopy(body)
    return value


def _workflow(interface: Object, nodes: Sequence[Object], *, limits: Object | None = None) -> Object:
    return {
        "choices": [],
        "input_bindings": [],
        "interface": deepcopy(interface),
        "limits": deepcopy(limits or _limits()),
        "nodes": deepcopy(list(nodes)),
        "outcome_bindings": [],
        "output_bindings": [],
        "protection": [],
        "sequence": [],
        "workflow": "W0",
    }


def _input(source: str, node: str, port: str, *, owner: str = "W0") -> Object:
    source_value: Object = {"kind": "workflow_input", "port": source}
    if source.startswith("N"):
        source_value = {"kind": "node_output", "node": _ref(source), "port": "o0"}
    return {"destination": {"node": _ref(node, owner), "port": port}, "source": source_value}


def _output(node: str, source_port: str, destination: str, *, owner: str = "W0") -> Object:
    return {
        "destination": {"port": destination},
        "source": {"kind": "node_output", "node": _ref(node, owner), "port": source_port},
    }


def _outcome_binding(node: str, outcome: str, destination: str, *, owner: str = "W0") -> Object:
    return {"destination": {"outcome": destination}, "source": {"node": _ref(node, owner), "outcome": outcome}}


def _one(operation: Object) -> Object:
    declaration = _workflow(operation, (_node("N0", operation),))
    inputs = cast(list[Object], operation["inputs"])
    outputs = cast(list[Object], operation["outputs"])
    declaration["input_bindings"] = [_input(cast(str, port["name"]), "N0", cast(str, port["name"])) for port in inputs]
    declaration["output_bindings"] = [
        _output("N0", cast(str, port["name"]), cast(str, port["name"])) for port in outputs
    ]
    declaration["outcome_bindings"] = [_outcome_binding("N0", "ok", "ok"), _outcome_binding("N0", "fail", "fail")]
    return declaration


def _pipe(*, lineage: bool = False) -> Object:
    first, second = (_l(0), _l(1)) if lineage else (_q(), _q())
    interface = _l01() if lineage else _q(factor=2)
    declaration = _workflow(interface, (_node("N0", first), _node("N1", second)))
    declaration["input_bindings"] = [_input("i0", "N0", "i0"), _input("N0", "N1", "i0")]
    declaration["output_bindings"] = [_output("N1", "o0", "o0")]
    declaration["outcome_bindings"] = [_outcome_binding("N1", "ok", "ok"), _outcome_binding("N1", "fail", "fail")]
    declaration["sequence"] = [{"after": _ref("N1"), "before": _ref("N0")}]
    return declaration


def _choice() -> Object:
    declaration = _workflow(_z(2), tuple(_node(f"N{i}", _z()) for i in range(3)))
    declaration["sequence"] = [
        {"after": _ref("N1"), "before": _ref("N0")},
        {"after": _ref("N2"), "before": _ref("N0")},
    ]
    declaration["choices"] = [
        {
            "branches": [
                {"members": [_ref("N1")], "outcomes": ["ok"]},
                {"members": [_ref("N2")], "outcomes": ["fail"]},
            ],
            "selector": _ref("N0"),
        }
    ]
    declaration["outcome_bindings"] = [
        _outcome_binding("N1", "ok", "ok"),
        _outcome_binding("N1", "fail", "fail"),
        _outcome_binding("N2", "ok", "ok"),
        _outcome_binding("N2", "fail", "fail"),
    ]
    return declaration


def _wrap(body: Object) -> Object:
    interface = deepcopy(_obj(body["interface"]))
    declaration = _one(interface)
    cast(list[Object], declaration["nodes"])[0] = _node("N0", interface, body=_reowner_workflow(body, "W1"))
    return declaration


def _reowner_workflow(declaration: Object, owner: str) -> Object:
    result = deepcopy(declaration)
    old_owner = cast(str, result["workflow"])
    result["workflow"] = owner

    def replace_owner(value: Json) -> None:
        if isinstance(value, list):
            for item in value:
                replace_owner(item)
        elif isinstance(value, dict):
            if value.get("owner") == old_owner and "label" in value:
                value["owner"] = owner
            for item in value.values():
                replace_owner(item)

    replace_owner(result)
    return result
