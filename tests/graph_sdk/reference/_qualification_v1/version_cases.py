# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent qualification_v1 reference: version cases."""

from __future__ import annotations

from copy import deepcopy

from tests.graph_sdk.reference._qualification_v1.case import (
    case,
)
from tests.graph_sdk.reference._qualification_v1.declarations import (
    declaration,
    set_output_dependency,
)
from tests.graph_sdk.reference._qualification_v1.events import (
    artifact,
    base_events,
    binding_declaration,
    input_producer,
    port,
    provenance,
)
from tests.graph_sdk.reference._qualification_v1.family_builders import (
    bound_events,
    initial_collection_events,
    map_version_events,
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


def version_cases() -> list[Obj]:
    c: list[Obj] = []
    base = base_events(("A",))
    initial_declaration = binding_declaration(("D1", "N0", "items", "A"), collections=("D1",))
    c.append(case("lineage", "map_two_versions", map_declaration(2), map_version_events()))

    e = map_version_events()

    next(value for value in e if value.get("kind") == "provenance" and value.get("key") == "MAPITEM:1")["expander"] = (
        "OTHER"
    )

    c.append(case("lineage", "expander_crossover", map_declaration(2), e))

    e = map_version_events()

    next(value for value in e if value.get("kind") == "provenance" and value.get("key") == "MAPITEM:1")["target"] = "B"

    c.append(case("lineage", "map_target_crossover", map_declaration(2), e))

    e = map_version_events()

    next(value for value in e if value.get("kind") == "artifact" and value.get("ref") == "MI0v2")["invocation"] = "I1"

    c.append(case("lineage", "map_invocation_crossover", map_declaration(2), e))

    e = bound_events("N0", "D0", "BOUND:N0:p")

    next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "BOUND:N0:p")["binding_artifact"] = "XAv0"

    c.append(
        case(
            "provenance",
            "bound_invocation_ref_substitution",
            binding_declaration(("D0", "N0", "p", "A")),
            e,
        )
    )

    e = bound_events("N0", "D0", "BOUND:N0:p")

    obj(next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "BOUND:N0:p")["binding_artifact"])[
        "version"
    ] = 0

    c.append(case("provenance", "bound_invalid_version", binding_declaration(("D0", "N0", "p", "A")), e))

    e = bound_events("N0", "D0", "BOUND:N0:p")

    obj(next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "BOUND:N0:p")["binding_artifact"])[
        "key"
    ] = 999

    c.append(
        case(
            "provenance",
            "bound_unretained_reference",
            binding_declaration(("D0", "N0", "p", "A")),
            e,
        )
    )

    e = bound_events("N0", "D0", "BOUND:N0:p")

    obj(arr(next(x for x in e if x.get("kind") == "binding_receipt")["artifacts"])[0])["source"] = "OTHER@1"

    c.append(
        case(
            "provenance",
            "bound_foreign_receipt_source",
            binding_declaration(("D0", "N0", "p", "A")),
            e,
        )
    )

    e = initial_collection_events()

    next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "INITIAL")["parents"] = ["BOUND:D1:0"]

    c.append(case("provenance", "initial_missing_bound_parent", initial_declaration, e))

    e = initial_collection_events()

    next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "INITIAL")["declaration"] = "D0"

    c.append(case("provenance", "initial_wrong_declaration", initial_declaration, e))

    e = initial_collection_events()

    obj(next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "BOUND:D1:1")["binding_artifact"])[
        "key"
    ] = 999

    c.append(case("provenance", "initial_unretained_bound_parent", initial_declaration, e))

    for name, field, value in (
        ("map_wrong_member", "member", "MAP"),
        ("map_wrong_port", "port", "not_item_input"),
        ("map_negative_item_key", "item_key", -1),
        ("map_zero_item_version", "item_version", 0),
        ("map_wrong_expander", "expander", "M0"),
    ):
        e = map_item_events()
        next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "MAPITEM:0")[field] = value
        c.append(case("provenance", name, declaration(), e))

    e = map_item_events()

    next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "MAPITEM:0")["parents"] = []

    c.append(case("provenance", "map_missing_collection_parent", declaration(), e))

    e = map_item_events()

    next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "MAPITEM:0")["parents"] = ["ROOT:A:context"]

    c.append(case("provenance", "map_wrong_collection_parent", declaration(), e))

    e = map_item_events()

    next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "MAPITEM:0")["parents"] = [
        "OP:MAP:A:members",
        "ROOT:A:context",
    ]

    c.append(case("provenance", "map_multiple_collection_parents", declaration(), e))

    e = map_item_events()

    obj(arr(next(x for x in e if x.get("kind") == "collection_value")["items"])[0])["key"] = 1

    c.append(case("provenance", "map_collection_identity_mismatch", declaration(), e))

    e = map_item_events()

    collection_value = next(x for x in e if x.get("kind") == "collection_value")

    collection_value["producer"] = "EVID:A"

    collection_value["artifact"] = "EAv0"

    next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "MAPITEM:0")["parents"] = ["EVID:A"]

    evidence_producer = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "EVID:A")

    e.remove(evidence_producer)

    e.insert(e.index(collection_value), evidence_producer)

    c.append(case("provenance", "map_unrelated_collection_producer", declaration(), e))

    for name, field, value in (
        ("map_wrong_membership_port", "membership_port", "other"),
        ("map_wrong_expansion_outcome", "outcome", "other"),
    ):
        d = declaration()
        obj(arr(d["map_inputs"])[0])[field] = value
        c.append(case("provenance", name, d, map_item_events()))

    e = map_item_events()

    producer_fact = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "OP:MAP:A:members")

    producer_fact["activation"] = "ROOT:A"

    c.append(case("provenance", "map_wrong_producer_activation", declaration(), e))

    e = map_item_events(2)

    first_item = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "MAPITEM:0")

    second_item = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "MAPITEM:1")

    first_item["item_key"], second_item["item_key"] = second_item["item_key"], first_item["item_key"]

    c.append(case("provenance", "map_swapped_member_items", declaration(), e))

    e = map_item_events()

    e.insert(1, artifact("MAv0", "B", "artifact"))

    item = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "MAPITEM:0")

    item["artifact"] = "MAv0"

    item["target"] = "B"

    item_port = next(
        x for x in e if x.get("kind") == "port" and x.get("activation") == "M0" and x.get("port") == "item"
    )

    item_port["artifact"] = "MAv0"

    item_port["target"] = "B"

    c.append(case("provenance", "map_wrong_target", declaration(targets=("A", "B")), e))

    e = deepcopy(base)

    out = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "OUT:A")

    idx = e.index(out)

    e[idx:idx] = [
        port("ROOT:A", "A", "decision_alias", "Av0", "decision"),
        provenance(
            "DECISION:ALIAS",
            "Av0",
            "A",
            ("ROOT:A:subject",),
            decision=True,
            source="operation_output",
            port_name="decision_alias",
            activation="ROOT:A",
        ),
        input_producer("ROOT:A", "A", "N", "decision_alias", "DECISION:ALIAS"),
    ]

    out["parents"] = ["DECISION:ALIAS", "ROOT:A:subject"]

    d = declaration()

    set_output_dependency(d, "N", "decision_alias", ("subject",), identity_input="subject")

    set_output_dependency(d, "N", "result", ("decision_alias", "subject"), identity_input="subject")

    c.append(case("roles", "candidate_decision_distinct_occurrence_alias", d, e))

    return c
