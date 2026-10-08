# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent qualification_v1 reference: map evidence builders."""

from __future__ import annotations

from collections.abc import Sequence
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


def map_item_endpoint(
    *,
    expander: str = "EXP",
    expansion_outcome: str = "ok",
    item_input: str = "item",
    member: str = "MN",
    membership_port: str = "members",
    path: Sequence[str] = (),
) -> Obj:
    return {
        "expander": expander,
        "expansion_outcome": expansion_outcome,
        "item_input": item_input,
        "member": member,
        "membership_port": membership_port,
        "path": list(path),
    }


def direct_item_declaration(
    count: int,
    *,
    consumed_only: bool = False,
    endpoint: Obj | None = None,
    candidate_port: str = "result",
) -> Obj:
    d = map_declaration(count, operation_expander=True)
    actual_endpoint = deepcopy(endpoint or map_item_endpoint())
    obj(d["node_kinds"]).update({cast(str, actual_endpoint["member"]): "operation", "J": "operation"})
    member_production = deepcopy(obj(arr(d["productions"])[0]))
    member_production.update(
        {
            "absence_queries": [],
            "consumed_ports": [actual_endpoint["item_input"]],
            "evidence_port": "evidence",
            "meaning": "item_privacy",
            "node": actual_endpoint["member"],
            "promise": "P_ITEM",
            "read_state": [],
            "subject_port": "subject" if consumed_only else actual_endpoint["item_input"],
            "subject_source": "input",
        }
    )
    arr(d["productions"]).append(member_production)
    d["map_item_requirements"] = [
        {
            "candidate_port": candidate_port,
            "consumed_endpoints": [deepcopy(actual_endpoint)],
            "coverage": ["K0"],
            "meaning": "item_privacy",
            "promise": "P_ITEM",
            "subject_endpoint": None if consumed_only else deepcopy(actual_endpoint),
            "target": "A",
        }
    ]
    d["map_routes"] = [
        {
            "expander": actual_endpoint["expander"],
            "item_input": actual_endpoint["item_input"],
            "member": actual_endpoint["member"],
            "membership_port": actual_endpoint["membership_port"],
            "outcome": actual_endpoint["expansion_outcome"],
            "path": deepcopy(actual_endpoint["path"]),
        }
    ]
    d["keyed_joins"] = [
        {
            "accepted_categories": ["success"],
            "join": "J",
            "reduction": "all_by_key",
            "source": actual_endpoint["expander"],
        }
    ]
    set_output_dependency(d, "MN", "evidence", (cast(str, actual_endpoint["item_input"]),))
    set_output_dependency(d, "N", "result", ("membership", "subject"), identity_input="subject")
    return d


def direct_item_events(count: int, *, consumed_only: bool = False, submit: bool = True) -> list[Obj]:
    events = map_item_events(count, operation_expander=True)
    root_membership = next(
        event for event in events if event.get("kind") == "membership" and event.get("parent") is None
    )
    arr(root_membership["members"]).append("JOIN")
    cut = next(i for i, event in enumerate(events) if event.get("kind") == "revision")
    events[cut:cut] = [entry("JOIN", "A", "J"), terminal("JOIN", "A")]
    events[:] = [
        event
        for event in events
        if not (
            event.get("activation") == "ROOT:A"
            and (
                (event.get("kind") == "port" and cast(str, event.get("port")).startswith("mapped_"))
                or (event.get("kind") == "input_producer" and cast(str, event.get("port")).startswith("mapped_"))
            )
        )
    ]
    out = next(event for event in events if event.get("kind") == "provenance" and event.get("key") == "OUT:A")
    out["parents"] = ["OP:MAP:A:members", "ROOT:A:subject"]
    out_index = events.index(out)
    events[out_index:out_index] = [
        port("ROOT:A", "A", "membership", "CAv0", "artifact"),
        input_producer("ROOT:A", "A", "N", "membership", "OP:MAP:A:members"),
    ]
    member_facts: list[Obj] = []
    for item_key in range(count):
        member = f"M{item_key}"
        item_ref = f"MI{item_key}v1"
        evidence_ref = f"ME{item_key}v0"
        next(event for event in events if event.get("kind") == "entry" and event.get("activation") == member)[
            "node"
        ] = "MN"
        next(
            event
            for event in events
            if event.get("kind") == "port" and event.get("activation") == member and event.get("port") == "item"
        )["node"] = "MN"
        next(
            event
            for event in events
            if event.get("kind") == "input_producer"
            and event.get("activation") == member
            and event.get("port") == "item"
        )["node"] = "MN"
        facts = [
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
        ]
        subject_ref = item_ref
        subject_port = "item"
        if consumed_only:
            facts.extend(
                [
                    port(member, "A", "subject", "Av0", "candidate", "MN"),
                    input_producer(member, "A", "MN", "subject", "ROOT:A:subject"),
                ]
            )
            subject_ref = "Av0"
            subject_port = "subject"
        facts.append(
            assessment(
                "A",
                activation=member,
                consumed={"item": item_ref},
                evidence_artifact=evidence_ref,
                environment={"absences": {}, "configurations": {"MN": "c0"}, "state": {}},
                fact=f"F:{member}:P_ITEM",
                node="MN",
                promise="P_ITEM",
                subject_artifact=subject_ref,
                subject_port=subject_port,
            )
        )
        if submit:
            facts.append(assessment_submission(member, "P_ITEM"))
        member_facts.extend(facts)
    events[out_index:out_index] = member_facts
    seal = next(i for i, event in enumerate(events) if event.get("kind") == "seal_revision")
    for item_key in range(count):
        events.insert(
            seal,
            {"collection": "artifacts", "key": f"ME{item_key}", "kind": "revision", "value": 0},
        )
        seal += 1
        events.insert(
            seal,
            {"collection": "artifacts", "key": f"MI{item_key}", "kind": "revision", "value": 1},
        )
        seal += 1
    events.insert(seal, {"collection": "configurations", "key": "MN", "kind": "revision", "value": "c0"})
    occurrence_by_activation = {"MAP": 0, "M0": 1, "M1": 2, "ROOT:A": 3, "JOIN": 4}
    for event in events:
        if event.get("kind") == "entry" and event.get("activation") in occurrence_by_activation:
            event["occurrence"] = occurrence_by_activation[cast(str, event["activation"])]
    return events


def block_keyed_join(events: list[Obj], activation: str = "JOIN") -> None:
    join_entry = next(
        event for event in events if event.get("kind") == "entry" and event.get("activation") == activation
    )
    join_entry.update({"closed_unstarted": True, "state_category": "blocked", "state_outcome": None})
    join_terminal = next(
        event for event in events if event.get("kind") == "terminal" and event.get("activation") == activation
    )
    join_terminal.clear()
    join_terminal.update(terminal(activation, "A", "blocked", None))
