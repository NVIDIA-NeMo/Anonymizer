# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent qualification_v1 reference: map builders."""

from __future__ import annotations

from copy import deepcopy

from tests.graph_sdk.reference._qualification_v1.declarations import (
    declaration,
    set_output_dependency,
)
from tests.graph_sdk.reference._qualification_v1.events import (
    artifact,
    base_events,
    entry,
    input_producer,
    port,
    provenance,
    terminal,
)
from tests.graph_sdk.reference._qualification_v1.model import (
    Obj,
    arr,
    obj,
)


def map_events(
    n: int, *, parent_terminal: bool = True, nested: bool = False, operation_expander: bool = False
) -> list[Obj]:
    base = base_events(("A",))
    e = deepcopy(base)
    cut = next(i for i, x in enumerate(e) if x.get("kind") == "revision")
    root_membership = next(x for x in e if x.get("kind") == "membership" and x.get("parent") is None)
    arr(root_membership["members"]).append("MAP")
    expander_kind = "operation" if operation_expander else "container"
    facts: list[Obj] = [entry("MAP", "A", "EXP", node_kind=expander_kind)]
    if parent_terminal:
        facts.append(terminal("MAP", "A", structural=not operation_expander))
    members = []
    for i in range(n):
        m = f"M{i}"
        members.append(m)
        facts += [
            {"activation": m, "kind": "reservation", "parent": "MAP", "selected": True, "target": "A"},
            entry(m, "A", "MEM", parent="MAP"),
            terminal(m, "A"),
        ]
    facts.append(
        {
            "closed": True,
            "expansion_outcome": "ok",
            "kind": "membership",
            "members": members,
            "parent": "MAP",
            "status": "closed",
            "target": "A",
        }
    )
    if nested:
        members.append("INNER")
        facts += [
            {"activation": "INNER", "kind": "reservation", "parent": "MAP", "selected": True, "target": "A"},
            entry("INNER", "A", "EXP", node_kind="container", parent="MAP"),
            terminal("INNER", "A", structural=True),
            {"activation": "IM0", "kind": "reservation", "parent": "INNER", "selected": True, "target": "A"},
            entry("IM0", "A", "MEM", parent="INNER"),
            terminal("IM0", "A"),
            {
                "closed": True,
                "expansion_outcome": "ok",
                "kind": "membership",
                "members": ["IM0"],
                "parent": "INNER",
                "status": "closed",
                "target": "A",
            },
        ]
    e[cut:cut] = facts
    return e


def map_item_events(count: int = 1, *, operation_expander: bool = False) -> list[Obj]:
    e = map_events(count, operation_expander=operation_expander)
    out = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "OUT:A")
    idx = e.index(out)
    producer = "OP:MAP:A:members"
    facts: list[Obj] = [
        artifact("CAv0", "A", "artifact"),
        {
            "artifact": "CAv0",
            "items": [{"key": item_key, "version": 1} for item_key in range(count)],
            "kind": "collection_value",
            "producer": producer,
            "target": "A",
        },
        port("MAP", "A", "members", "CAv0", "artifact", "EXP"),
        port("MAP", "A", "context", "XAv0", "artifact", "EXP"),
        input_producer("MAP", "A", "EXP", "context", "ROOT:A:context"),
        provenance(
            producer,
            "CAv0",
            "A",
            ("ROOT:A:context",),
            source="operation_output",
            node="EXP",
            port_name="members",
            activation="MAP",
        ),
    ]
    item_parents = []
    for item_key in range(count):
        member = f"M{item_key}"
        ref = f"MI{item_key}v1"
        key = f"MAPITEM:{item_key}"
        facts.extend(
            [
                artifact(ref, "A", "artifact"),
                port(member, "A", "item", ref, "artifact", "MEM"),
                provenance(
                    key,
                    ref,
                    "A",
                    (producer,),
                    source="map_item",
                    node=None,
                    port_name="item",
                    expander="MAP",
                    member=member,
                    item_key=item_key,
                    item_version=1,
                ),
                input_producer(member, "A", "MEM", "item", key),
                input_producer("ROOT:A", "A", "N", f"mapped_{item_key}", key),
                port("ROOT:A", "A", f"mapped_{item_key}", ref, "artifact"),
            ]
        )
        item_parents.append(key)
    e[idx:idx] = facts
    out["parents"] = [*item_parents, "ROOT:A:subject"]
    return e


def map_declaration(count: int, *, operation_expander: bool = False) -> Obj:
    d = declaration()
    if operation_expander:
        obj(d["node_kinds"])["EXP"] = "operation"
    set_output_dependency(
        d,
        "N",
        "result",
        (*tuple(f"mapped_{item_key}" for item_key in range(count)), "subject"),
        identity_input="subject",
    )
    return d
