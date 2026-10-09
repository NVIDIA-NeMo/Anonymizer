# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent qualification_v1 reference: dynamic builders."""

from __future__ import annotations

from copy import deepcopy
from typing import cast

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
from tests.graph_sdk.reference._qualification_v1.map_builders import (
    map_declaration,
    map_item_events,
)
from tests.graph_sdk.reference._qualification_v1.model import (
    Obj,
    arr,
    obj,
)


def assessed_map_declaration(_count: int) -> Obj:
    """One count-independent typed map declaration for dynamic cases."""
    d = map_declaration(0, operation_expander=True)
    obj(d["node_kinds"]).update({"MN": "operation", "J": "operation"})
    member_production = deepcopy(obj(arr(d["productions"])[0]))
    member_production.update(
        {
            "absence_queries": [],
            "consumed_ports": ["item"],
            "evidence_port": "evidence",
            "meaning": "item_privacy",
            "node": "MN",
            "promise": "P_MEMBER",
            "read_state": [],
            "subject_port": "item",
            "subject_source": "input",
        }
    )
    arr(d["productions"]).append(member_production)
    endpoint: Obj = {
        "expander": "EXP",
        "expansion_outcome": "ok",
        "item_input": "item",
        "member": "MN",
        "membership_port": "members",
        "path": [],
    }
    d["map_item_requirements"] = [
        {
            "candidate_port": "result",
            "consumed_endpoints": [deepcopy(endpoint)],
            "coverage": ["K0"],
            "meaning": "item_privacy",
            "promise": "P_MEMBER",
            "subject_endpoint": deepcopy(endpoint),
            "target": "A",
        }
    ]
    d["map_routes"] = [
        {
            "expander": "EXP",
            "item_input": "item",
            "member": "MN",
            "membership_port": "members",
            "outcome": "ok",
            "path": [],
        }
    ]
    d["keyed_joins"] = [{"accepted_categories": ["success"], "join": "J", "reduction": "all_by_key", "source": "EXP"}]
    set_output_dependency(d, "MN", "evidence", ("item",))
    set_output_dependency(d, "N", "result", ("membership", "subject"), identity_input="subject")
    return d


def assessed_map_events(count: int, *, submit_members: bool = False) -> list[Obj]:
    """Retain an executable N/EXP/MN/J record with actual member owners."""
    e = map_item_events(count, operation_expander=True)
    root_membership = next(event for event in e if event.get("kind") == "membership" and event.get("parent") is None)
    arr(root_membership["members"]).append("JOIN")
    cut = next(i for i, event in enumerate(e) if event.get("kind") == "revision")
    e[cut:cut] = [entry("JOIN", "A", "J"), terminal("JOIN", "A")]
    e = [
        event
        for event in e
        if not (
            event.get("activation") == "ROOT:A"
            and event.get("kind") in {"port", "input_producer"}
            and cast(str, event.get("port")).startswith("mapped_")
        )
    ]
    out = next(event for event in e if event.get("kind") == "provenance" and event.get("key") == "OUT:A")
    out["parents"] = ["OP:MAP:A:members", "ROOT:A:subject"]
    insert_at = next(
        i for i, event in enumerate(e) if event.get("kind") == "provenance" and event.get("key") == "OUT:A"
    )
    member_facts: list[Obj] = [
        port("ROOT:A", "A", "membership", "CAv0", "artifact"),
        input_producer("ROOT:A", "A", "N", "membership", "OP:MAP:A:members"),
    ]
    for item_key in range(count):
        member = f"M{item_key}"
        item_ref = f"MI{item_key}v1"
        evidence_ref = f"ME{item_key}v0"
        entry_fact = next(event for event in e if event.get("kind") == "entry" and event.get("activation") == member)
        entry_fact["node"] = "MN"
        item_port = next(
            event
            for event in e
            if event.get("kind") == "port" and event.get("activation") == member and event.get("port") == "item"
        )
        item_port.update({"node": "MN", "role": "artifact"})
        item_input = next(
            event
            for event in e
            if event.get("kind") == "input_producer"
            and event.get("activation") == member
            and event.get("port") == "item"
        )
        item_input["node"] = "MN"
        member_facts.extend(
            [
                artifact(evidence_ref, "A", "evidence"),
                port(member, "A", "evidence", evidence_ref, "evidence", "MN"),
                provenance(
                    f"EVID:{member}",
                    evidence_ref,
                    "A",
                    (f"MAPITEM:{item_key}",),
                    port_name="evidence",
                    activation=member,
                    node="MN",
                ),
                assessment(
                    "A",
                    activation=member,
                    consumed={"item": item_ref},
                    evidence_artifact=evidence_ref,
                    environment={
                        "absences": {},
                        "configurations": {"MN": "c0"},
                        "state": {},
                    },
                    fact=f"F:{member}:P_MEMBER",
                    node="MN",
                    promise="P_MEMBER",
                    subject_artifact=item_ref,
                    subject_port="item",
                ),
            ]
        )
    e[insert_at:insert_at] = member_facts
    seal = next(i for i, event in enumerate(e) if event.get("kind") == "seal_revision")
    for item_key in range(count):
        e.insert(
            seal,
            {"collection": "artifacts", "key": f"ME{item_key}", "kind": "revision", "value": 0},
        )
        seal += 1
        e.insert(
            seal,
            {"collection": "artifacts", "key": f"MI{item_key}", "kind": "revision", "value": 1},
        )
        seal += 1
    e.insert(seal, {"collection": "configurations", "key": "MN", "kind": "revision", "value": "c0"})
    occurrence_by_activation = {"MAP": 0, "M0": 1, "M1": 2, "ROOT:A": 3, "JOIN": 4}
    for event in e:
        if event.get("kind") == "entry" and event.get("activation") in occurrence_by_activation:
            event["occurrence"] = occurrence_by_activation[cast(str, event["activation"])]
    if submit_members:
        insertion = (
            next(
                i
                for i, event in enumerate(e)
                if event.get("kind") == "assessment_submission" and event.get("fact") == "F:A:P"
            )
            + 1
        )
        e[insertion:insertion] = [assessment_submission(f"M{index}", "P_MEMBER") for index in range(count)]
    return e


def submit_dynamic_occurrences(count: int) -> list[Obj]:
    return assessed_map_events(count, submit_members=True)


def inject_member_assessment(events: list[Obj]) -> list[Obj]:
    injected = deepcopy(events)
    insertion = next(i for i, event in enumerate(injected) if event.get("kind") == "revision")
    injected.insert(
        insertion,
        assessment(
            "A",
            activation="M0",
            consumed={"item": "MI0v1"},
            evidence_artifact="ME0v0",
            environment={"absences": {}, "configurations": {"MN": "c0"}, "state": {}},
            fact="F:M0:P_MEMBER",
            node="MN",
            promise="P_MEMBER",
            subject_artifact="MI0v1",
            subject_port="item",
        ),
    )
    return injected


def remove_member_evidence(events: list[Obj], member: str = "M0") -> list[Obj]:
    evidence_ref = f"ME{member.removeprefix('M')}v0"
    evidence_key = evidence_ref.rsplit("v", 1)[0]
    return [
        event
        for event in events
        if not (
            event.get("kind") == "assessment"
            and event.get("activation") == member
            or event.get("kind") == "assessment_submission"
            and event.get("fact") == f"F:{member}:P_MEMBER"
            or event.get("kind") == "artifact"
            and event.get("ref") == evidence_ref
            or event.get("kind") == "port"
            and event.get("activation") == member
            and event.get("port") == "evidence"
            or event.get("kind") == "provenance"
            and event.get("key") == f"EVID:{member}"
            or event.get("kind") == "revision"
            and event.get("collection") == "artifacts"
            and event.get("key") == evidence_key
        )
    ]


def non_success_member_events(category: str, *, closed_unstarted: bool) -> list[Obj]:
    events = remove_member_evidence(assessed_map_events(1))
    member_entry = next(event for event in events if event.get("kind") == "entry" and event.get("activation") == "M0")
    member_entry.update(
        {
            "closed_unstarted": closed_unstarted,
            "node": "MN",
            "state_category": category,
            "state_outcome": None,
        }
    )
    member_terminal = next(
        event for event in events if event.get("kind") == "terminal" and event.get("activation") == "M0"
    )
    replacement = terminal("M0", "A", category, None)
    member_terminal.clear()
    member_terminal.update(replacement)
    join_entry = next(event for event in events if event.get("kind") == "entry" and event.get("activation") == "JOIN")
    join_entry.update({"closed_unstarted": True, "state_category": "blocked", "state_outcome": None})
    join_terminal = next(
        event for event in events if event.get("kind") == "terminal" and event.get("activation") == "JOIN"
    )
    join_terminal.clear()
    join_terminal.update(terminal("JOIN", "A", "blocked", None))
    if category == "blocked":
        obj_kind = obj(assessed_map_declaration(1)["node_kinds"])
        assert obj_kind["MN"] == "operation"
        root_membership = next(
            event for event in events if event.get("kind") == "membership" and event.get("parent") is None
        )
        arr(root_membership["members"]).append("FAILED")
        insertion = next(i for i, event in enumerate(events) if event.get("kind") == "revision")
        events[insertion:insertion] = [
            entry("FAILED", "A", "FAILED_SOURCE", state_category="failure", state_outcome=None),
            terminal("FAILED", "A", "failure", None),
        ]
        events = [
            event
            for event in events
            if not (
                event.get("activation") == "M0"
                and event.get("port") == "item"
                and event.get("kind") in {"port", "input_producer"}
            )
        ]
    return events
