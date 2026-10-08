# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent qualification_v1 reference: map evidence cases."""

from __future__ import annotations

from copy import deepcopy
from typing import cast

from tests.graph_sdk.reference._qualification_v1.case import (
    _replace_refs,
    case,
)
from tests.graph_sdk.reference._qualification_v1.declarations import (
    set_output_dependency,
)
from tests.graph_sdk.reference._qualification_v1.events import (
    artifact,
    assessment,
    assessment_submission,
    entry,
    input_producer,
    port,
    provenance,
    terminal,
)
from tests.graph_sdk.reference._qualification_v1.map_evidence_builders import (
    block_keyed_join,
    direct_item_declaration,
    direct_item_events,
    map_item_endpoint,
)
from tests.graph_sdk.reference._qualification_v1.model import (
    Obj,
    arr,
    obj,
)


def map_evidence_cases() -> list[Obj]:
    c: list[Obj] = []
    for count in range(3):
        c.append(
            case("map_item_evidence", f"direct_{count}", direct_item_declaration(count), direct_item_events(count))
        )

    d = direct_item_declaration(2)

    obj(d["limits"])["max_submissions"] = 2

    c.append(case("map_item_bounds", "submissions_one_over", d, direct_item_events(2)))

    d = direct_item_declaration(2)

    obj(d["limits"])["max_verified_evidence"] = 2

    c.append(case("map_item_bounds", "verified_one_over", d, direct_item_events(2)))

    c.append(
        case(
            "map_item_evidence",
            "typed_consumed_endpoint",
            direct_item_declaration(1, consumed_only=True),
            direct_item_events(1, consumed_only=True),
        )
    )

    e = direct_item_events(1)

    e = [
        event
        for event in e
        if not (event.get("kind") == "assessment_submission" and event.get("fact") == "F:M0:P_ITEM")
    ]

    c.append(case("map_item_evidence", "missing_submission", direct_item_declaration(1), e))

    e = direct_item_events(1)

    next(event for event in e if event.get("kind") == "assessment_submission" and event.get("fact") == "F:M0:P_ITEM")[
        "fact"
    ] = "F:FOREIGN:P_ITEM"

    c.append(case("map_item_evidence", "foreign_submission", direct_item_declaration(1), e))

    e = direct_item_events(1)

    insertion = next(i for i, event in enumerate(e) if event.get("kind") == "revision")

    e.insert(insertion, assessment_submission("M0", "P_ITEM"))

    c.append(case("map_item_evidence", "repeated_submission", direct_item_declaration(1), e))

    e = [
        event
        for event in direct_item_events(1)
        if not (
            event.get("kind") == "assessment"
            and event.get("activation") == "M0"
            or event.get("kind") == "assessment_submission"
            and event.get("fact") == "F:M0:P_ITEM"
        )
    ]

    c.append(case("map_item_evidence", "missing_fact", direct_item_declaration(1), e))

    e = direct_item_events(2)

    next(event for event in e if event.get("kind") == "assessment_submission" and event.get("fact") == "F:M0:P_ITEM")[
        "fact"
    ] = "F:M1:P_ITEM"

    c.append(case("map_item_evidence", "copied_member_submission", direct_item_declaration(2), e))

    for field, value in (
        ("expander", "OTHER"),
        ("member", "OTHER"),
        ("item_input", "other"),
        ("membership_port", "other"),
        ("expansion_outcome", "other"),
    ):
        d = direct_item_declaration(1)
        obj(obj(arr(d["map_item_requirements"])[0])["subject_endpoint"])[field] = value
        c.append(case("map_item_admission", f"invalid_{field}", d, [], "admission"))

    d = direct_item_declaration(1, candidate_port="other")

    c.append(case("map_item_admission", "invalid_candidate_port", d, [], "admission"))

    d = direct_item_declaration(1)

    obj(obj(arr(d["map_item_requirements"])[0])["subject_endpoint"])["path"] = ["MN"]

    c.append(case("map_item_admission", "invalid_path_owner", d, [], "admission"))

    d = direct_item_declaration(1)

    typed_requirement = obj(arr(d["map_item_requirements"])[0])

    obj(typed_requirement["subject_endpoint"])["expander"] = "OTHER"

    obj(arr(typed_requirement["consumed_endpoints"])[0])["expander"] = "OTHER"

    c.append(case("map_item_admission", "unresolved_projection", d, [], "admission"))

    d = direct_item_declaration(1)

    obj(d["node_kinds"])["UNRELATED"] = "container"

    typed_requirement = obj(arr(d["map_item_requirements"])[0])

    obj(typed_requirement["subject_endpoint"])["path"] = ["UNRELATED"]

    obj(arr(typed_requirement["consumed_endpoints"])[0])["path"] = ["UNRELATED"]

    c.append(case("map_item_admission", "valid_container_wrong_route", d, [], "admission"))

    d = direct_item_declaration(1)

    arr(d["map_item_requirements"]).append(deepcopy(arr(d["map_item_requirements"])[0]))

    c.append(case("map_item_admission", "duplicate_typed_requirement", d, [], "admission"))

    e = direct_item_events(1, consumed_only=True)

    next(event for event in e if event.get("kind") == "provenance" and event.get("key") == "MAPITEM:0")["member"] = (
        "OTHER"
    )

    c.append(
        case(
            "map_item_evidence",
            "typed_consumed_wrong_owner",
            direct_item_declaration(1, consumed_only=True),
            e,
        )
    )

    e = direct_item_events(1)

    item_assessment = next(
        event for event in e if event.get("kind") == "assessment" and event.get("activation") == "M0"
    )

    item_assessment["subject_artifact"] = "Av0"

    c.append(case("map_item_evidence", "wrong_subject_artifact", direct_item_declaration(1), e))

    e = direct_item_events(1)

    next(event for event in e if event.get("kind") == "provenance" and event.get("key") == "MAPITEM:0")[
        "item_version"
    ] = 2

    c.append(case("map_item_evidence", "wrong_item_version", direct_item_declaration(1), e))

    e = direct_item_events(1)

    next(event for event in e if event.get("kind") == "provenance" and event.get("key") == "MAPITEM:0")["item_key"] = 1

    c.append(case("map_item_evidence", "wrong_item_key", direct_item_declaration(1), e))

    e = direct_item_events(1)

    next(
        event
        for event in e
        if event.get("kind") == "input_producer" and event.get("activation") == "M0" and event.get("port") == "item"
    )["producer"] = "ROOT:A:subject"

    c.append(case("map_item_evidence", "wrong_item_owner", direct_item_declaration(1), e))

    e = direct_item_events(1)

    e = [
        event
        for event in e
        if not (
            event.get("kind") == "revision" and event.get("collection") == "artifacts" and event.get("key") == "MI0"
        )
    ]

    c.append(case("map_item_evidence", "item_unknown", direct_item_declaration(1), e))

    e = direct_item_events(1)

    e = [
        event
        for event in e
        if not (
            event.get("kind") == "assessment"
            and event.get("activation") == "M0"
            or event.get("kind") == "assessment_submission"
            and event.get("fact") == "F:M0:P_ITEM"
            or event.get("kind") == "artifact"
            and event.get("ref") == "ME0v0"
            or event.get("kind") == "port"
            and event.get("activation") == "M0"
            and event.get("port") == "evidence"
            or event.get("kind") == "provenance"
            and event.get("key") == "EVID:M0"
            or event.get("kind") == "revision"
            and event.get("collection") == "artifacts"
            and event.get("key") == "ME0"
        )
    ]

    member_entry = next(event for event in e if event.get("kind") == "entry" and event.get("activation") == "M0")

    member_entry.update({"state_category": "failure", "state_outcome": None})

    member_terminal = next(event for event in e if event.get("kind") == "terminal" and event.get("activation") == "M0")

    member_terminal.clear()

    member_terminal.update(terminal("M0", "A", "failure", None))

    block_keyed_join(e)

    c.append(case("map_item_evidence", "member_non_success", direct_item_declaration(1), e))

    e = direct_item_events(1)

    e = [
        event
        for event in e
        if not (
            event.get("kind") == "assessment"
            and event.get("activation") == "M0"
            or event.get("kind") == "assessment_submission"
            and event.get("fact") == "F:M0:P_ITEM"
            or event.get("kind") == "artifact"
            and event.get("ref") == "ME0v0"
            or event.get("kind") == "port"
            and event.get("activation") == "M0"
            and event.get("port") == "evidence"
            or event.get("kind") == "provenance"
            and event.get("key") == "EVID:M0"
            or event.get("kind") == "revision"
            and event.get("collection") == "artifacts"
            and event.get("key") == "ME0"
        )
    ]

    member_entry = next(event for event in e if event.get("kind") == "entry" and event.get("activation") == "M0")

    member_entry.update({"closed_unstarted": True, "state_category": "blocked", "state_outcome": None})

    member_terminal = next(event for event in e if event.get("kind") == "terminal" and event.get("activation") == "M0")

    member_terminal.clear()

    member_terminal.update(terminal("M0", "A", "blocked", None))

    block_keyed_join(e)

    c.append(case("map_item_evidence", "member_blocked_unreached", direct_item_declaration(1), e))

    for status in ("failed", "overflow"):
        e = direct_item_events(0)
        membership = next(event for event in e if event.get("kind") == "membership" and event.get("parent") == "MAP")
        membership.update({"expansion_outcome": None, "status": status})
        block_keyed_join(e)
        if status == "failed":
            e.insert(
                e.index(membership),
                {"activation": "M0", "kind": "reservation", "parent": "MAP", "selected": True, "target": "A"},
            )
        c.append(case("map_item_evidence", f"expansion_{status}", direct_item_declaration(0), e))

    e = direct_item_events(0)

    membership = next(event for event in e if event.get("kind") == "membership" and event.get("parent") == "MAP")

    membership.update({"closed": False, "status": "open"})

    e = [event for event in e if not (event.get("kind") == "terminal" and event.get("activation") == "JOIN")]

    c.append(case("map_item_evidence", "expansion_open", direct_item_declaration(0), e))

    d = direct_item_declaration(1)

    set_output_dependency(d, "N", "result", ("subject",), identity_input="subject")

    e = direct_item_events(1)

    e = [
        event
        for event in e
        if not (
            event.get("activation") == "ROOT:A"
            and event.get("port") == "membership"
            and event.get("kind") in {"port", "input_producer"}
        )
    ]

    next(event for event in e if event.get("kind") == "provenance" and event.get("key") == "OUT:A")["parents"] = [
        "ROOT:A:subject"
    ]

    c.append(case("map_item_evidence", "different_final_ancestry", d, e))

    d = direct_item_declaration(1)

    d["root_outputs"] = [
        {"input_port": "subject", "port": "result", "source_node": None, "source_port": None, "target": "A"}
    ]

    e = direct_item_events(1)

    next(event for event in e if event.get("kind") == "final")["producer"] = "ROOT:A:subject"

    next(
        event
        for event in e
        if event.get("kind") == "port"
        and event.get("activation") == "ROOT:A"
        and event.get("node") == "N"
        and event.get("port") == "result"
    )["role"] = "artifact"

    c.append(case("map_item_evidence", "candidate_passthrough_unrelated", d, e))

    nested_endpoint = map_item_endpoint(path=("SG",))

    d = direct_item_declaration(1, endpoint=nested_endpoint)

    obj(d["node_kinds"])["SG"] = "container"

    arr(d["subgraphs"]).append(
        {
            "body_input": "context",
            "body_node": "EXP",
            "body_outcome": "ok",
            "body_port": "members",
            "body_source": "node_output",
            "input_port": "context",
            "node": "SG",
            "outcome": "ok",
            "port": "nested_members",
        }
    )

    e = direct_item_events(1)

    root_membership = next(event for event in e if event.get("kind") == "membership" and event.get("parent") is None)

    root_members = arr(root_membership["members"])

    root_members[root_members.index("MAP")] = "WRAP"

    root_members.remove("JOIN")

    next(event for event in e if event.get("kind") == "entry" and event.get("activation") == "MAP")["parent"] = "WRAP"

    next(event for event in e if event.get("kind") == "entry" and event.get("activation") == "JOIN")["parent"] = "WRAP"

    next(
        event
        for event in e
        if event.get("kind") == "input_producer"
        and event.get("activation") == "ROOT:A"
        and event.get("port") == "membership"
    )["producer"] = "SGOUT:WRAP:A:nested_members"

    next(event for event in e if event.get("kind") == "provenance" and event.get("key") == "OUT:A")["parents"] = [
        "SGOUT:WRAP:A:nested_members",
        "ROOT:A:subject",
    ]

    insertion = next(
        i for i, event in enumerate(e) if event.get("kind") == "provenance" and event.get("key") == "OUT:A"
    )

    e[insertion:insertion] = [
        {"activation": "WRAP", "kind": "reservation", "parent": None, "selected": True, "target": "A"},
        entry("WRAP", "A", "SG", node_kind="container"),
        terminal("WRAP", "A", structural=True),
        port("WRAP", "A", "nested_members", "CAv0", "artifact", "SG"),
        provenance(
            "SGOUT:WRAP:A:nested_members",
            "CAv0",
            "A",
            ("OP:MAP:A:members",),
            node="SG",
            port_name="nested_members",
            activation="WRAP",
        ),
        {"activation": "MAP", "kind": "reservation", "parent": "WRAP", "selected": True, "target": "A"},
        {"activation": "JOIN", "kind": "reservation", "parent": "WRAP", "selected": True, "target": "A"},
        {
            "closed": True,
            "expansion_outcome": None,
            "kind": "membership",
            "members": ["MAP", "JOIN"],
            "parent": "WRAP",
            "status": "closed",
            "target": "A",
        },
    ]

    nested_occurrences = {"WRAP": 0, "MAP": 1, "M0": 2, "M1": 3, "JOIN": 4, "ROOT:A": 5}

    for event in e:
        if event.get("kind") == "entry" and event.get("activation") in nested_occurrences:
            event["occurrence"] = nested_occurrences[cast(str, event["activation"])]

    c.append(case("map_item_evidence", "nested_path", d, e))

    alternate_endpoint = map_item_endpoint(expansion_outcome="alternate", membership_port="alternate_members")

    d = direct_item_declaration(1, endpoint=alternate_endpoint)

    map_input = obj(arr(d["map_inputs"])[0])

    map_input.update({"membership_port": "alternate_members", "outcome": "alternate"})

    expander_dependency = next(
        dependency
        for dependency in map(obj, arr(d["output_dependencies"]))
        if dependency.get("node") == "EXP" and dependency.get("port") == "members"
    )

    expander_dependency.update({"outcome": "alternate", "port": "alternate_members"})

    e = cast(
        list[Obj],
        _replace_refs(
            direct_item_events(1),
            {"OP:MAP:A:members": "OP:MAP:A:alternate_members"},
        ),
    )

    next(event for event in e if event.get("kind") == "entry" and event.get("activation") == "MAP")["state_outcome"] = (
        "alternate"
    )

    next(event for event in e if event.get("kind") == "terminal" and event.get("activation") == "MAP")["outcome"] = (
        "alternate"
    )

    next(event for event in e if event.get("kind") == "membership" and event.get("parent") == "MAP")[
        "expansion_outcome"
    ] = "alternate"

    next(
        event
        for event in e
        if event.get("kind") == "port" and event.get("activation") == "MAP" and event.get("port") == "members"
    )["port"] = "alternate_members"

    next(
        event for event in e if event.get("kind") == "provenance" and event.get("key") == "OP:MAP:A:alternate_members"
    )["port"] = "alternate_members"

    c.append(case("map_item_evidence", "distinct_outcome_port", d, e))

    e = cast(list[Obj], _replace_refs(direct_item_events(2), {"MI1v1": "MI0v2"}))

    second_artifact = next(event for event in e if event.get("kind") == "artifact" and event.get("ref") == "MI0v2")

    second_artifact.update({"key": "MI0", "version": 2})

    second_item = next(event for event in e if event.get("kind") == "provenance" and event.get("key") == "MAPITEM:1")

    second_item.update({"item_key": 0, "item_version": 2})

    collection = next(event for event in e if event.get("kind") == "collection_value")

    obj(arr(collection["items"])[1]).update({"key": 0, "version": 2})

    e = [
        event
        for event in e
        if not (
            event.get("kind") == "revision" and event.get("collection") == "artifacts" and event.get("key") == "MI1"
        )
    ]

    next(
        event
        for event in e
        if event.get("kind") == "revision" and event.get("collection") == "artifacts" and event.get("key") == "MI0"
    )["value"] = 2

    c.append(case("map_item_evidence", "item_stale", direct_item_declaration(2), e))

    for name, field, value in (
        ("wrong_expander", "expander", "OTHER"),
        ("wrong_member", "member", "OTHER"),
        ("wrong_target", "target", "B"),
    ):
        e = direct_item_events(1)
        next(event for event in e if event.get("kind") == "provenance" and event.get("key") == "MAPITEM:0")[field] = (
            value
        )
        c.append(case("map_item_evidence", name, direct_item_declaration(1), e))

    e = direct_item_events(1)

    next(event for event in e if event.get("kind") == "artifact" and event.get("ref") == "MI0v1")["invocation"] = "I1"

    c.append(case("map_item_evidence", "wrong_invocation", direct_item_declaration(1), e))

    d = direct_item_declaration(1)

    obj(d["limits"])["max_productions"] = 3

    obj(d["node_kinds"]).update({"EXP2": "operation", "J2": "operation", "MN2": "operation"})

    arr(d["map_inputs"]).append(
        {"expander": "EXP2", "item_input": "item2", "membership_port": "members2", "outcome": "ok"}
    )

    second_production = deepcopy(
        next(production for production in map(obj, arr(d["productions"])) if production.get("promise") == "P_ITEM")
    )

    second_production.update(
        {
            "consumed_ports": ["item2"],
            "meaning": "item_privacy_2",
            "node": "MN2",
            "promise": "P_ITEM_2",
            "subject_port": "item2",
        }
    )

    arr(d["productions"]).append(second_production)

    arr(d["output_dependencies"]).extend(
        [
            {"identity_input": None, "inputs": ["context"], "node": "EXP2", "outcome": "ok", "port": "members2"},
            {"identity_input": None, "inputs": ["item2"], "node": "MN2", "outcome": "ok", "port": "evidence"},
        ]
    )

    second_endpoint = map_item_endpoint(expander="EXP2", item_input="item2", member="MN2", membership_port="members2")

    arr(d["map_item_requirements"]).append(
        {
            "candidate_port": "result",
            "consumed_endpoints": [deepcopy(second_endpoint)],
            "coverage": ["K0"],
            "meaning": "item_privacy_2",
            "promise": "P_ITEM_2",
            "subject_endpoint": deepcopy(second_endpoint),
            "target": "A",
        }
    )

    arr(d["map_routes"]).append(
        {
            "expander": "EXP2",
            "item_input": "item2",
            "member": "MN2",
            "membership_port": "members2",
            "outcome": "ok",
            "path": [],
        }
    )

    arr(d["keyed_joins"]).append(
        {
            "accepted_categories": ["success"],
            "join": "J2",
            "reduction": "all_by_key",
            "source": "EXP2",
        }
    )

    set_output_dependency(d, "N", "result", ("membership", "membership2", "subject"), identity_input="subject")

    e = direct_item_events(1)

    root_membership = next(event for event in e if event.get("kind") == "membership" and event.get("parent") is None)

    arr(root_membership["members"]).extend(("MAP2", "JOIN2"))

    out = next(event for event in e if event.get("kind") == "provenance" and event.get("key") == "OUT:A")

    arr(out["parents"]).insert(1, "OP:MAP2:A:members2")

    insertion = e.index(out)

    e[insertion:insertion] = [
        artifact("CBv0", "A", "artifact"),
        artifact("MJ0v1", "A", "artifact"),
        artifact("MF0v0", "A", "evidence"),
        {
            "artifact": "CBv0",
            "items": [{"key": 0, "version": 1}],
            "kind": "collection_value",
            "producer": "OP:MAP2:A:members2",
            "target": "A",
        },
        port("ROOT:A", "A", "membership2", "CBv0", "artifact"),
        input_producer("ROOT:A", "A", "N", "membership2", "OP:MAP2:A:members2"),
        port("MAP2", "A", "members2", "CBv0", "artifact", "EXP2"),
        port("MAP2", "A", "context", "XAv0", "artifact", "EXP2"),
        input_producer("MAP2", "A", "EXP2", "context", "ROOT:A:context"),
        provenance(
            "OP:MAP2:A:members2",
            "CBv0",
            "A",
            ("ROOT:A:context",),
            node="EXP2",
            port_name="members2",
            activation="MAP2",
        ),
        port("Z0", "A", "item2", "MJ0v1", "artifact", "MN2"),
        provenance(
            "MAPITEM2:0",
            "MJ0v1",
            "A",
            ("OP:MAP2:A:members2",),
            source="map_item",
            node=None,
            port_name="item2",
            expander="MAP2",
            member="Z0",
            item_key=0,
            item_version=1,
        ),
        input_producer("Z0", "A", "MN2", "item2", "MAPITEM2:0"),
        port("Z0", "A", "evidence", "MF0v0", "evidence", "MN2"),
        provenance(
            "EVID:Z0",
            "MF0v0",
            "A",
            ("MAPITEM2:0",),
            port_name="evidence",
            activation="Z0",
            node="MN2",
        ),
        assessment(
            "A",
            activation="Z0",
            consumed={"item2": "MJ0v1"},
            evidence_artifact="MF0v0",
            environment={"absences": {}, "configurations": {"MN2": "c0"}, "state": {}},
            fact="F:Z0:P_ITEM_2",
            node="MN2",
            promise="P_ITEM_2",
            subject_artifact="MJ0v1",
            subject_port="item2",
        ),
        assessment_submission("Z0", "P_ITEM_2"),
        entry("MAP2", "A", "EXP2", occurrence=5),
        terminal("MAP2", "A"),
        entry("JOIN2", "A", "J2", occurrence=8),
        terminal("JOIN2", "A"),
        {"activation": "Z0", "kind": "reservation", "parent": "MAP2", "selected": True, "target": "A"},
        entry("Z0", "A", "MN2", occurrence=6, parent="MAP2"),
        terminal("Z0", "A"),
        {
            "closed": True,
            "expansion_outcome": "ok",
            "kind": "membership",
            "members": ["Z0"],
            "parent": "MAP2",
            "status": "closed",
            "target": "A",
        },
    ]

    seal = next(i for i, event in enumerate(e) if event.get("kind") == "seal_revision")

    e[seal:seal] = [
        {"collection": "artifacts", "key": "MF0", "kind": "revision", "value": 0},
        {"collection": "artifacts", "key": "MJ0", "kind": "revision", "value": 1},
        {"collection": "configurations", "key": "MN2", "kind": "revision", "value": "c0"},
    ]

    c.append(case("map_item_evidence", "two_independent_maps", d, e))

    missing_second = [
        deepcopy(event)
        for event in e
        if not (event.get("kind") == "assessment_submission" and event.get("fact") == "F:Z0:P_ITEM_2")
    ]

    c.append(case("map_item_evidence", "two_maps_no_crossproduct", deepcopy(d), missing_second))

    cross_owner = deepcopy(e)

    next(event for event in cross_owner if event.get("kind") == "assessment" and event.get("activation") == "Z0")[
        "subject_artifact"
    ] = "MI0v1"

    c.append(case("map_item_evidence", "two_maps_cross_owner", deepcopy(d), cross_owner))

    return c
