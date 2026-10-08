# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent qualification_v1 reference: lineage cases."""

from __future__ import annotations

from copy import deepcopy

from tests.graph_sdk.reference._qualification_v1.case import (
    case,
    mutate,
)
from tests.graph_sdk.reference._qualification_v1.declarations import (
    declaration,
    set_output_dependency,
)
from tests.graph_sdk.reference._qualification_v1.events import (
    artifact,
    base_events,
    binding_declaration,
    entry,
    input_producer,
    port,
    provenance,
    terminal,
)
from tests.graph_sdk.reference._qualification_v1.family_builders import (
    bound_events,
    initial_collection_events,
    retain_root_input,
    wire_evidence_consumed,
)
from tests.graph_sdk.reference._qualification_v1.map_builders import (
    map_declaration,
    map_item_events,
)
from tests.graph_sdk.reference._qualification_v1.model import (
    Obj,
    arr,
    obj,
)


def lineage_cases() -> list[Obj]:
    c: list[Obj] = []
    base = base_events(("A",))
    for name, d, failed in (
        ("independent", declaration(targets=("A", "B", "C")), "A"),
        ("a_to_b", declaration(targets=("A", "B", "C"), deps=(("A", "B"),)), "A"),
        ("b_to_a", declaration(targets=("A", "B", "C"), deps=(("B", "A"),)), "A"),
        ("atomic_ab", declaration(targets=("A", "B", "C"), atomic=(("A", "B"),)), "A"),
        ("atomic_bc", declaration(targets=("A", "B", "C"), atomic=(("B", "C"),)), "B"),
    ):
        e = base_events(("A", "B", "C"))
        next(x for x in e if x.get("kind") == "assessment" and x.get("target") == failed)["finding"] = "unsatisfied"
        c.append(case("propagation", name, d, e))

    for name, chain in (("direct", ("DEC",)), ("transitive", ("MID", "DEC")), ("unrelated", ())):
        d = declaration()
        e = deepcopy(base)
        out = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "OUT:A")
        parents = []
        if chain:
            e.insert(1, artifact("DAv0", "A", "decision"))
            out_index = e.index(out)
            e.insert(out_index, port("ROOT:A", "A", "decision_source", "DAv0", "decision"))
            e.insert(
                out_index + 1,
                provenance(
                    "DEC",
                    "DAv0",
                    "A",
                    decision=True,
                    source="operation_output",
                    port_name="decision_source",
                    activation="ROOT:A",
                ),
            )
            e.insert(out_index + 2, input_producer("ROOT:A", "A", "N", "decision_source", "DEC"))
            set_output_dependency(d, "N", "decision_source", ())
            parents = ["DEC"]
            if name == "transitive":
                e.insert(out_index + 2, port("ROOT:A", "A", "mid", "Av0", "candidate"))
                e.insert(
                    out_index + 3,
                    provenance(
                        "MID",
                        "Av0",
                        "A",
                        ("DEC",),
                        source="operation_output",
                        port_name="mid",
                        activation="ROOT:A",
                    ),
                )
                e.insert(out_index + 5, input_producer("ROOT:A", "A", "N", "mid", "MID"))
                set_output_dependency(d, "N", "mid", ("decision_source", "subject"), identity_input="subject")
                set_output_dependency(d, "N", "result", ("mid", "subject"), identity_input="subject")
                next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "MID")["parents"] = [
                    "DEC",
                    "ROOT:A:subject",
                ]
                parents = ["MID", "ROOT:A:subject"]
            else:
                set_output_dependency(d, "N", "result", ("decision_source", "subject"), identity_input="subject")
                parents.append("ROOT:A:subject")
            out["parents"] = parents
            seal = next(i for i, x in enumerate(e) if x.get("kind") == "seal_revision")
            e.insert(seal, {"collection": "artifacts", "key": "DA", "kind": "revision", "value": 0})
        c.append(case("decisions", name, d, e))

    e = mutate(base, "assessment", "consumed", {"context": "XAv0", "decision": "DAv0"})

    e.insert(1, artifact("DAv0", "A", "decision"))

    e.insert(
        next(i for i, x in enumerate(e) if x.get("kind") == "assessment"),
        port("ROOT:A", "A", "decision", "DAv0", "decision"),
    )

    retain_root_input(e, "A", "decision", "DAv0")

    seal = next(i for i, x in enumerate(e) if x.get("kind") == "seal_revision")

    e.insert(seal, {"collection": "artifacts", "key": "DA", "kind": "revision", "value": 0})

    d = declaration()

    wire_evidence_consumed(d, e, ("context", "decision"))

    c.append(case("decisions", "consumed", d, e))

    d = declaration()

    obj(arr(d["productions"])[0])["subject_source"] = "input"

    d["root_outputs"] = [
        {"input_port": "subject", "port": "result", "source_node": None, "source_port": None, "target": "A"}
    ]

    e = [x for x in deepcopy(base) if not (x.get("kind") == "provenance" and x.get("key") == "OUT:A")]

    f = next(x for x in e if x.get("kind") == "final")

    f["producer"] = "ROOT:A:subject"

    c.append(case("provenance", "root", d, e))

    wrong_root = deepcopy(e)

    next(x for x in wrong_root if x.get("kind") == "final")["producer"] = "ROOT:A:context"

    c.append(case("provenance", "direct_root_wrong_input", deepcopy(d), wrong_root))

    d = declaration()

    set_output_dependency(d, "N", "alias", ("context",), identity_input="context")

    set_output_dependency(d, "N", "result", ("alias", "subject"), identity_input="subject")

    e = deepcopy(base)

    out = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "OUT:A")

    idx = e.index(out)

    e[idx:idx] = [
        port("ROOT:A", "A", "alias", "XAv0", "artifact"),
        provenance("ALIAS", "XAv0", "A", ("ROOT:A:context",), port_name="alias", activation="ROOT:A"),
        input_producer("ROOT:A", "A", "N", "alias", "ALIAS"),
    ]

    out["parents"] = ["ALIAS", "ROOT:A:subject"]

    c.append(case("provenance", "identity_alias", d, e))

    missing = [x for x in deepcopy(e) if not (x.get("kind") == "input_producer" and x.get("port") == "alias")]

    c.append(case("provenance", "missing_executed_input_producer", deepcopy(d), missing))

    extra = deepcopy(e)

    next(x for x in extra if x.get("kind") == "provenance" and x.get("key") == "ALIAS")["parents"] = [
        "ROOT:A:context",
        "ROOT:A:subject",
    ]

    c.append(case("provenance", "extra_dependency_parent", deepcopy(d), extra))

    swapped = deepcopy(base)

    subject_input = next(x for x in swapped if x.get("kind") == "input_producer" and x.get("port") == "subject")

    context_input = next(x for x in swapped if x.get("kind") == "input_producer" and x.get("port") == "context")

    subject_input["producer"], context_input["producer"] = context_input["producer"], subject_input["producer"]

    c.append(case("provenance", "swapped_dependency_parents", declaration(), swapped))

    commuted_parents = deepcopy(base)

    next(x for x in commuted_parents if x.get("kind") == "provenance" and x.get("key") == "OUT:A")["parents"] = [
        "ROOT:A:context",
        "ROOT:A:subject",
    ]

    c.append(case("provenance", "parent_order_invariant", declaration(), commuted_parents))

    extra_input = deepcopy(base)

    extra_input.insert(
        next(i for i, x in enumerate(extra_input) if x.get("kind") == "provenance" and x.get("key") == "OUT:A"),
        input_producer("ROOT:A", "A", "N", "not_an_executed_input", "ROOT:A:subject"),
    )

    c.append(case("provenance", "extra_input_producer", declaration(), extra_input))

    d = declaration()

    obj(d["node_kinds"]).update({"SG": "container", "BN": "operation"})

    d["subgraphs"] = [
        {
            "body_input": "input",
            "body_node": "BN",
            "body_outcome": "ok",
            "body_port": "result",
            "body_source": "node_output",
            "input_port": "input",
            "node": "SG",
            "outcome": "ok",
            "port": "result",
        }
    ]

    set_output_dependency(d, "BN", "result", ("input",), identity_input="input")

    set_output_dependency(d, "N", "result", ("subgraph_result", "subject"), identity_input="subject")

    e = deepcopy(base)

    root_members = next(x for x in e if x.get("kind") == "membership" and x.get("parent") is None)

    arr(root_members["members"]).append("SUB")

    idx = next(i for i, x in enumerate(e) if x.get("kind") == "provenance" and x.get("key") == "OUT:A")

    e[idx:idx] = [
        entry("SUB", "A", "SG", node_kind="container", occurrence=1),
        terminal("SUB", "A", structural=True),
        port("SUB", "A", "input", "XAv0", "artifact", "SG"),
        input_producer("SUB", "A", "SG", "input", "ROOT:A:context"),
        {
            "closed": True,
            "expansion_outcome": "ok",
            "kind": "membership",
            "members": ["BODY"],
            "parent": "SUB",
            "status": "closed",
            "target": "A",
        },
        entry("BODY", "A", "BN", occurrence=0, parent="SUB"),
        terminal("BODY", "A"),
        port("BODY", "A", "input", "XAv0", "artifact", "BN"),
        input_producer("BODY", "A", "BN", "input", "ROOT:A:context"),
        port("BODY", "A", "result", "XAv0", "artifact", "BN"),
        provenance("BODYOUT", "XAv0", "A", ("ROOT:A:context",), node="BN", activation="BODY"),
        port("SUB", "A", "result", "XAv0", "artifact", "SG"),
        provenance("SUBGRAPH", "XAv0", "A", ("BODYOUT",), node="SG", activation="SUB"),
        port("ROOT:A", "A", "subgraph_result", "XAv0", "artifact"),
        input_producer("ROOT:A", "A", "N", "subgraph_result", "SUBGRAPH"),
    ]

    out = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "OUT:A")

    out["parents"] = ["SUBGRAPH", "ROOT:A:subject"]

    c.append(case("provenance", "subgraph", d, e))

    wrong_projection = deepcopy(e)

    next(x for x in wrong_projection if x.get("kind") == "provenance" and x.get("key") == "SUBGRAPH")["parents"] = [
        "ROOT:A:context"
    ]

    c.append(case("provenance", "subgraph_wrong_body_projection", deepcopy(d), wrong_projection))

    wrong_passthrough = deepcopy(e)

    next(
        x
        for x in wrong_passthrough
        if x.get("kind") == "input_producer" and x.get("activation") == "BODY" and x.get("port") == "input"
    )["producer"] = "ROOT:A:subject"

    c.append(case("provenance", "subgraph_wrong_input_passthrough", deepcopy(d), wrong_passthrough))

    d = declaration()

    obj(d["node_kinds"])["SG"] = "container"

    d["subgraphs"] = [
        {
            "body_input": "input",
            "body_node": None,
            "body_outcome": "ok",
            "body_port": None,
            "body_source": "workflow_input",
            "input_port": "input",
            "node": "SG",
            "outcome": "ok",
            "port": "result",
        }
    ]

    set_output_dependency(d, "N", "result", ("passthrough", "subject"), identity_input="subject")

    e = deepcopy(base)

    arr(next(x for x in e if x.get("kind") == "membership" and x.get("parent") is None)["members"]).append("SUB")

    out = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "OUT:A")

    idx = e.index(out)

    e[idx:idx] = [
        entry("SUB", "A", "SG", node_kind="container", occurrence=1),
        terminal("SUB", "A", structural=True),
        port("SUB", "A", "input", "XAv0", "artifact", "SG"),
        input_producer("SUB", "A", "SG", "input", "ROOT:A:context"),
        {
            "closed": True,
            "expansion_outcome": "ok",
            "kind": "membership",
            "members": [],
            "parent": "SUB",
            "status": "closed",
            "target": "A",
        },
        port("SUB", "A", "result", "XAv0", "artifact", "SG"),
        provenance("PASSTHROUGH", "XAv0", "A", ("ROOT:A:context",), node="SG", activation="SUB"),
        port("ROOT:A", "A", "passthrough", "XAv0", "artifact"),
        input_producer("ROOT:A", "A", "N", "passthrough", "PASSTHROUGH"),
    ]

    out["parents"] = ["PASSTHROUGH", "ROOT:A:subject"]

    c.append(case("provenance", "subgraph_nested_workflow_input_passthrough", d, e))

    missing_capture = [
        x
        for x in deepcopy(e)
        if not (x.get("kind") == "input_producer" and x.get("activation") == "SUB" and x.get("port") == "input")
    ]

    c.append(case("provenance", "subgraph_passthrough_missing_capture", deepcopy(d), missing_capture))

    wrong_capture = deepcopy(e)

    next(
        x
        for x in wrong_capture
        if x.get("kind") == "input_producer" and x.get("activation") == "SUB" and x.get("port") == "input"
    )["producer"] = "ROOT:A:subject"

    c.append(case("provenance", "subgraph_passthrough_wrong_capture", deepcopy(d), wrong_capture))

    d = declaration()

    set_output_dependency(d, "N", "version", ("context",), identity_input="context")

    set_output_dependency(d, "N", "result", ("version", "subject"), identity_input="subject")

    e = deepcopy(base)

    out = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "OUT:A")

    idx = e.index(out)

    e[idx:idx] = [
        port("ROOT:A", "A", "version", "XAv0", "artifact"),
        provenance("VERSION", "XAv0", "A", ("ROOT:A:context",), port_name="version", activation="ROOT:A"),
        input_producer("ROOT:A", "A", "N", "version", "VERSION"),
    ]

    out["parents"] = ["VERSION", "ROOT:A:subject"]

    c.append(case("provenance", "version_edge", d, e))

    c.append(
        case(
            "provenance",
            "bound_n0",
            binding_declaration(("D0", "N0", "p", "A")),
            bound_events("N0", "D0", "BOUND:N0:p"),
        )
    )

    c.append(
        case(
            "provenance",
            "bound_n1",
            binding_declaration(("D2", "N1", "p", "A")),
            bound_events("N1", "D2", "BOUND:N1:p"),
        )
    )

    initial_declaration = binding_declaration(("D1", "N0", "items", "A"), collections=("D1",))

    c.append(case("provenance", "initial_collection", initial_declaration, initial_collection_events()))

    c.append(case("provenance", "map_item", map_declaration(1), map_item_events()))

    c.append(case("provenance", "map_item_two_members", map_declaration(2), map_item_events(2)))

    d = map_declaration(2, operation_expander=True)

    c.append(
        case("provenance", "map_item_two_members_operation_expander", d, map_item_events(2, operation_expander=True))
    )

    return c
