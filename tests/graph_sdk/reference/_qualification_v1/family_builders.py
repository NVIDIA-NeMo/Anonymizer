# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent qualification_v1 reference: family builders."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import cast

from tests.graph_sdk.reference._qualification_v1.case import (
    _replace_refs,
)
from tests.graph_sdk.reference._qualification_v1.declarations import (
    declaration,
    set_output_dependency,
    set_production_consumed,
)
from tests.graph_sdk.reference._qualification_v1.events import (
    artifact,
    assessment,
    assessment_submission,
    base_events,
    binding_receipt,
    binding_ref,
    input_producer,
    port,
    provenance,
)
from tests.graph_sdk.reference._qualification_v1.map_builders import (
    map_item_events,
)
from tests.graph_sdk.reference._qualification_v1.model import (
    Obj,
    arr,
    obj,
)


def execution_only_shape() -> tuple[Obj, list[Obj]]:
    declaration_value = declaration(purpose="execution_only")
    declaration_value["productions"] = []
    declaration_value["requirements"] = []
    events = [
        value
        for value in base_events(("A",), assess=False)
        if not (value.get("kind") == "artifact" and value.get("role") == "evidence")
        and not (value.get("kind") == "port" and value.get("role") == "evidence")
        and not (value.get("kind") == "provenance" and value.get("key") == "EVID:A")
        and not (value.get("kind") == "revision" and value.get("key") == "EA")
        and not (value.get("kind") == "revision" and value.get("collection") != "artifacts")
    ]
    return declaration_value, events


def wire_result(d: Obj, e: list[Obj], target: str, producer: str, input_name: str) -> None:
    output = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == f"OUT:{target}")
    producer_fact = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == producer)
    e.insert(
        e.index(output),
        port(f"ROOT:{target}", target, input_name, cast(str, producer_fact["artifact"]), "artifact"),
    )
    e.insert(e.index(output), input_producer(f"ROOT:{target}", target, "N", input_name, producer))
    output["parents"] = [producer]
    set_output_dependency(d, "N", "result", (input_name,))


def retain_root_input(e: list[Obj], target: str, port_name: str, ref: str, *, node: str = "N") -> None:
    output = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == f"OUT:{target}")
    key = f"ROOT:{target}:{port_name}"
    e.insert(
        e.index(output),
        provenance(key, ref, target, source="root_input", node=node, port_name=port_name, activation=f"ROOT:{target}"),
    )
    e.insert(e.index(output), input_producer(f"ROOT:{target}", target, node, port_name, key))


def wire_evidence_consumed(d: Obj, e: list[Obj], ports: Sequence[str]) -> None:
    set_production_consumed(d, ports)
    set_output_dependency(d, "N", "evidence", ports)
    evidence = next(value for value in e if value.get("kind") == "provenance" and value.get("key") == "EVID:A")
    producers = {
        cast(str, value["port"]): cast(str, value["producer"])
        for value in e
        if value.get("kind") == "input_producer" and value.get("activation") == "ROOT:A" and value.get("node") == "N"
    }
    evidence["parents"] = [producers[port_name] for port_name in ports]


def multi_promise_declaration() -> Obj:
    d = declaration()
    partial = obj(arr(d["productions"])[0])
    partial["promise"] = "P_PART"
    partial["coverage"] = ["K0"]
    full = deepcopy(partial)
    full.update({"coverage": ["K0", "K1"], "evidence_port": "evidence2", "promise": "P_FULL"})
    arr(d["productions"]).append(full)
    obj(arr(d["requirements"])[0])["coverage"] = ["K0", "K1"]
    set_output_dependency(d, "N", "evidence2", ("context",))
    return d


def multi_promise_events(*, partial: bool, complete: bool) -> list[Obj]:
    base = base_events(("A",))
    e = deepcopy(base)
    partial_assessment = next(x for x in e if x.get("kind") == "assessment")
    partial_assessment.update({"coverage": ["K0"], "promise": "P_PART"})
    partial_assessment["fact"] = "F:A:P_PART"
    partial_submission = next(x for x in e if x.get("kind") == "assessment_submission")
    partial_submission["fact"] = "F:A:P_PART"
    if not partial:
        e.remove(partial_submission)
    out = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "OUT:A")
    idx = e.index(out)
    complete_submission = assessment_submission("A", "P_FULL")
    e[idx:idx] = [
        artifact("E2v0", "A", "evidence"),
        port("ROOT:A", "A", "evidence2", "E2v0", "evidence"),
        provenance(
            "EVID2:A",
            "E2v0",
            "A",
            ("ROOT:A:context",),
            port_name="evidence2",
            activation="ROOT:A",
        ),
        assessment(
            "A",
            coverage=["K0", "K1"],
            evidence_artifact="E2v0",
            evidence_port="evidence2",
            promise="P_FULL",
        ),
        complete_submission,
    ]
    if not complete:
        e.remove(complete_submission)
    seal = next(i for i, x in enumerate(e) if x.get("kind") == "seal_revision")
    e.insert(seal, {"collection": "artifacts", "key": "E2", "kind": "revision", "value": 0})
    return e


def request_history(failure: str | None, purpose: str | None, unknown: bool = False) -> list[Obj]:
    base = base_events(("A",))
    e = deepcopy(base)
    idx = next(i for i, x in enumerate(e) if x.get("kind") == "revision")
    facts: list[Obj] = [
        {
            "associations": ["A"],
            "kind": "request_attempt",
            "policy": "P",
            "predecessor": None,
            "purpose": "initial",
            "request": "R0",
        },
        {
            "condition": "failure" if failure else "result",
            "failure": failure,
            "kind": "request_terminal",
            "request": "R0",
        },
        {
            "kind": "settlement",
            "remote_stopped": not unknown,
            "request": "R0",
            "usage": "unknown" if unknown else 1,
        },
    ]
    if purpose:
        facts += [
            {
                "associations": ["A"],
                "kind": "request_attempt",
                "policy": "P",
                "predecessor": "R0",
                "purpose": purpose,
                "request": "R1",
            },
            {"condition": "result", "failure": None, "kind": "request_terminal", "request": "R1"},
            {"kind": "settlement", "remote_stopped": True, "request": "R1", "usage": 1},
        ]
    e[idx:idx] = facts
    return e


def binding_cleanup_events(
    *, disposition: str | None, association_targets: Sequence[str] | None, duplicate: bool = False
) -> list[Obj]:
    e = base_events(("A", "C"))
    e.insert(0, binding_receipt((), ()))
    idx = next(i for i, x in enumerate(e) if x.get("kind") == "revision")
    facts: list[Obj] = []
    if association_targets is not None:
        association: Obj = {
            "kind": "binding_cleanup_association",
            "owner": "sdk",
            "purpose": "accounting",
            "resource": "BIND:R0",
            "targets": list(association_targets),
        }
        facts.append(association)
        if duplicate:
            facts.append(deepcopy(association))
    if disposition is not None:
        facts.append({"disposition": disposition, "kind": "binding_cleanup", "resource": "BIND:R0"})
    e[idx:idx] = facts
    return e


def bound_events(node: str, declaration_id: str, key: str) -> list[Obj]:
    base = base_events(("A",))
    e = deepcopy(base)
    site = (declaration_id, node, "p", "A")
    e.insert(0, binding_receipt((site,), ((declaration_id, 0, 1),)))
    out = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "OUT:A")
    e.insert(
        e.index(out),
        provenance(
            key,
            "XAv0",
            "A",
            source="bound_input",
            node=node,
            port_name="p",
            binding_artifact=binding_ref(declaration_id),
        ),
    )
    e.insert(e.index(out), port("ROOT:A", "A", "bound_0", "XAv0", "artifact"))
    e.insert(e.index(out), input_producer("ROOT:A", "A", "N", "bound_0", key))
    out["parents"] = [key, "ROOT:A:subject"]
    return e


def initial_collection_events() -> list[Obj]:
    base = base_events(("A",))
    e = deepcopy(base)
    site = ("D1", "N0", "items", "A")
    e.insert(0, binding_receipt((site,), (("D1", 0, 1), ("D1", 1, 1))))
    out = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "OUT:A")
    idx = e.index(out)
    facts = [artifact("B0v0", "A", "artifact"), artifact("B1v0", "A", "artifact")]
    for item_key, ref in enumerate(("B0v0", "B1v0")):
        facts.append(
            provenance(
                f"BOUND:D1:{item_key}",
                ref,
                "A",
                source="bound_input",
                node="N0",
                port_name="items",
                binding_artifact=binding_ref("D1", item_key),
            )
        )
    facts.append(
        provenance(
            "INITIAL",
            "XAv0",
            "A",
            ("BOUND:D1:0", "BOUND:D1:1"),
            source="initial_collection",
            node="N0",
            port_name="items",
            declaration="D1",
        )
    )
    facts.append(input_producer("ROOT:A", "A", "N", "bound_0", "INITIAL"))
    facts.append(port("ROOT:A", "A", "bound_0", "XAv0", "artifact"))
    e[idx:idx] = facts
    out["parents"] = ["INITIAL", "ROOT:A:subject"]
    return e


def map_version_events() -> list[Obj]:
    events = map_item_events(2)
    events = cast(list[Obj], _replace_refs(events, {"MI1v1": "MI0v2"}))
    second_artifact = next(value for value in events if value.get("kind") == "artifact" and value.get("ref") == "MI0v2")
    second_artifact.update({"key": "MI0", "version": 2})
    collection = next(value for value in events if value.get("kind") == "collection_value")
    obj(arr(collection["items"])[1]).update({"key": 0, "version": 2})
    second_item = next(
        value for value in events if value.get("kind") == "provenance" and value.get("key") == "MAPITEM:1"
    )
    second_item.update({"item_key": 0, "item_version": 2})
    return events


def initial_version_seed(*, limit_changes: Mapping[str, int] | None = None) -> tuple[Obj, list[Obj]]:
    d = declaration(limit_changes=limit_changes)
    events = base_events(("A",))
    old = next(value for value in events if value.get("kind") == "artifact" and value.get("ref") == "XAv0")
    events.insert(events.index(old) + 1, artifact("XAv1", "A", "artifact"))
    next(
        value
        for value in events
        if value.get("kind") == "revision" and value.get("collection") == "artifacts" and value.get("key") == "XA"
    )["value"] = 1
    return d, events
