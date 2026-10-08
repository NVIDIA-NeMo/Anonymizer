# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent qualification_v1 reference: mutation cases."""

from __future__ import annotations

from copy import deepcopy
from typing import cast

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
    binding_receipt,
    binding_ref,
    input_producer,
    port,
    provenance,
)
from tests.graph_sdk.reference._qualification_v1.family_builders import (
    execution_only_shape,
    request_history,
    retain_root_input,
    wire_evidence_consumed,
)
from tests.graph_sdk.reference._qualification_v1.map_builders import (
    map_events,
)
from tests.graph_sdk.reference._qualification_v1.model import (
    Obj,
    arr,
    obj,
)


def mutation_cases(direct_case: Obj) -> list[Obj]:
    c: list[Obj] = []
    base = base_events(("A",))
    c.append(
        case(
            "record", "open_execution_only", declaration(purpose="execution_only"), map_events(0, parent_terminal=False)
        )
    )

    for name, field, value in (("plan", "plan", "P1"), ("invocation", "invocation", "I1"), ("graph", "graph", "G1")):
        e = base_events(("A",))
        owner: Obj = {
            "factory": "EXEC:I0",
            "graph": "G0",
            "invocation": "I0",
            "kind": "result_owner",
            "plan": "P0",
        }
        owner[field] = value
        e.insert(0, owner)
        c.append(case("record", name, declaration(), e))

    e = [x for x in deepcopy(base) if not (x.get("kind") == "port" and x.get("port") == "context")]

    c.append(case("authentication", "missing_consumed_port_fact", declaration(), e))

    for name, kind, selector, field, value in (
        ("evidence_port_node", "port", "evidence", "node", "OTHER"),
        ("evidence_port_target", "port", "evidence", "target", "B"),
        ("entry_node", "entry", "ROOT:A", "node", "OTHER"),
        ("entry_target", "entry", "ROOT:A", "target", "B"),
    ):
        e = deepcopy(base)
        fact = next(
            x for x in e if x.get("kind") == kind and (x.get("port") == selector or x.get("activation") == selector)
        )
        fact[field] = value
        c.append(case("authentication", name, declaration(), e))

    e = deepcopy(base)

    next(x for x in e if x.get("kind") == "entry" and x.get("activation") == "ROOT:A")["state_outcome"] = "other"

    c.append(case("authentication", "terminal_outcome", declaration(), e))

    e = deepcopy(base)

    next(x for x in e if x.get("kind") == "terminal" and x.get("activation") == "ROOT:A")["target"] = "B"

    c.append(case("authentication", "terminal_target", declaration(), e))

    for name, environment in (
        ("missing_absence_environment", {"absences": {}, "configurations": {"N": "c0"}, "state": {"read": 1}}),
        ("missing_configuration_environment", {"absences": {"Q0": 1}, "configurations": {}, "state": {"read": 1}}),
        ("missing_state_environment", {"absences": {"Q0": 1}, "configurations": {"N": "c0"}, "state": {}}),
    ):
        c.append(case("authentication", name, declaration(), mutate(base, "assessment", "environment", environment)))

    for name, field, value in (
        ("subject_port", "subject_port", "other"),
        ("consumed_port", "consumed_ports", ["other"]),
    ):
        d = declaration()
        obj(arr(d["requirements"])[0])[field] = value
        c.append(case("admission", f"requirement_{name}", d, [], "admission"))

    for name, failure, purpose in (
        ("permanent_retry", "permanent", "retry"),
        ("retryable_failover", "retryable", "failover"),
        ("malformed_retry", "malformed", "retry"),
    ):
        c.append(case("request", name, declaration(), request_history(failure, purpose)))

    e = request_history("retryable", "retry")

    next(x for x in e if x.get("kind") == "request_attempt" and x.get("request") == "R1")["policy"] = "OTHER"

    c.append(case("request", "changed_policy", declaration(), e))

    e = request_history("retryable", "retry")

    next(x for x in e if x.get("kind") == "request_attempt" and x.get("request") == "R1")["associations"] = ["B"]

    c.append(case("request", "changed_association", declaration(), e))

    e = request_history("retryable", "retry")

    idx = next(i for i, x in enumerate(e) if x.get("kind") == "revision")

    e[idx:idx] = [
        {
            "associations": ["A"],
            "kind": "request_attempt",
            "policy": "P",
            "predecessor": "R1",
            "purpose": "retry",
            "request": "R2",
        },
        {"condition": "failure", "failure": "retryable", "kind": "request_terminal", "request": "R2"},
        {"kind": "settlement", "remote_stopped": True, "request": "R2", "usage": 1},
    ]

    c.append(case("request", "latest_failure_authority", declaration(), e))

    e = request_history("lost", None, False)

    c.append(case("request", "lost_known_usage", declaration(), e))

    e = request_history(None, None, True)

    c.append(case("request", "success_unknown_usage", declaration(), e))

    e = request_history(None, None)

    next(x for x in e if x.get("kind") == "request_attempt")["associations"] = ["B"]

    c.append(case("request", "foreign_association", declaration(), e))

    e = deepcopy(base)

    idx = next(i for i, x in enumerate(e) if x.get("kind") == "revision")

    e[idx:idx] = [
        {"kind": "cleanup_association", "owner": "sdk", "purpose": "verification", "resource": "Q0", "targets": ["A"]},
        {"disposition": "left_open", "kind": "cleanup", "resource": "Q0"},
    ]

    c.append(case("cleanup", "sdk_left_open", declaration(), e))

    e = base_events(("A", "C"))

    idx = next(i for i, x in enumerate(e) if x.get("kind") == "revision")

    e[idx:idx] = [
        {"kind": "cleanup_association", "owner": "sdk", "purpose": "verification", "resource": "QC", "targets": ["C"]},
        {"disposition": "close_failed", "kind": "cleanup", "resource": "QC"},
    ]

    c.append(case("cleanup", "unrelated_c", declaration(targets=("A", "C")), e))

    e = deepcopy(base)

    idx = next(i for i, x in enumerate(e) if x.get("kind") == "revision")

    e[idx:idx] = [
        {"kind": "cleanup_association", "owner": "sdk", "purpose": "verification", "resource": "QX", "targets": ["B"]},
        {"disposition": "close_failed", "kind": "cleanup", "resource": "QX"},
    ]

    c.append(case("cleanup", "foreign_target", declaration(), e))

    e = map_events(0)

    next(x for x in e if x.get("kind") == "entry" and x.get("activation") == "MAP")["state_outcome"] = "other"

    next(x for x in e if x.get("kind") == "terminal" and x.get("activation") == "MAP")["outcome"] = "other"

    d = declaration()

    arr(d["output_dependencies"]).append(
        {"identity_input": None, "inputs": [], "node": "EXP", "outcome": "other", "port": "alternate"}
    )

    c.append(case("membership", "wrong_expansion_outcome", d, e))

    for name, closed, nested in (("open", False, False), ("nested_open", False, True)):
        e = map_events(1, nested=nested)
        memberships = [x for x in e if x.get("kind") == "membership"]
        memberships[-1]["closed"] = closed
        memberships[-1]["status"] = "closed" if closed else "open"
        if not closed:
            root_membership = next(x for x in memberships if x.get("parent") is None)
            root_membership["closed"] = False
            root_membership["status"] = "open"
        c.append(case("membership", name, declaration(), e))

    for category in ("blocked", "cancelled", "lost", "inconsistent"):
        e = map_events(0)
        map_entry = next(x for x in e if x.get("kind") == "entry" and x.get("activation") == "MAP")
        map_entry["state_category"] = category
        map_entry["state_outcome"] = None
        map_terminal = next(x for x in e if x.get("kind") == "terminal" and x.get("activation") == "MAP")
        map_terminal["category"] = category
        map_terminal["outcome"] = None
        map_terminal["reasons"] = [
            {
                "blocked": "prerequisite",
                "cancelled": "cancel_requested",
                "inconsistent": "contradictory",
                "lost": "transport_lost",
            }[category]
        ]
        membership = next(x for x in e if x.get("kind") == "membership" and x.get("parent") == "MAP")
        membership["status"] = "failed"
        membership["expansion_outcome"] = None
        c.append(case("membership", f"terminal_{category}", declaration(), e))

    e = map_events(0)

    next(x for x in e if x.get("kind") == "membership" and x.get("parent") == "MAP")["target"] = "B"

    c.append(case("membership", "foreign_target", declaration(), e))

    c.append(case("structural", "valid_container", declaration(), map_events(0)))

    e = deepcopy(base)

    next(x for x in e if x.get("kind") == "terminal")["structural"] = 1

    c.append(case("structural", "non_bool_flag", declaration(), e))

    e = deepcopy(base)

    next(x for x in e if x.get("kind") == "terminal")["reasons"] = ["unexpected"]

    c.append(case("structural", "success_with_reason", declaration(), e))

    e = map_events(0)

    next(x for x in e if x.get("kind") == "terminal" and x.get("activation") == "MAP")["attempt"] = "TASK:MAP"

    c.append(case("structural", "container_attempt_forbidden", declaration(), e))

    e = deepcopy(base)

    root_terminal = next(x for x in e if x.get("kind") == "terminal" and x.get("activation") == "ROOT:A")

    root_terminal["structural"] = True

    root_terminal["attempt"] = None

    c.append(case("structural", "structural_on_operation", declaration(), e))

    e = map_events(0)

    map_terminal = next(x for x in e if x.get("kind") == "terminal" and x.get("activation") == "MAP")

    map_terminal["structural"] = False

    map_terminal["attempt"] = "TASK:MAP"

    c.append(case("structural", "operation_on_container", declaration(), e))

    e = deepcopy(base)

    next(x for x in e if x.get("kind") == "terminal" and x.get("activation") == "ROOT:A")["attempt"] = None

    c.append(case("structural", "operation_success_missing_attempt", declaration(), e))

    for name, category in (("failed", "failure"), ("cancelled", "cancelled"), ("lost", "lost")):
        e = deepcopy(base)
        root_entry = next(x for x in e if x.get("kind") == "entry" and x.get("activation") == "ROOT:A")
        root_entry["state_category"] = category
        root_entry["state_outcome"] = None
        root_terminal = next(x for x in e if x.get("kind") == "terminal" and x.get("activation") == "ROOT:A")
        root_terminal["attempt"] = None
        root_terminal["category"] = category
        root_terminal["outcome"] = None
        root_terminal["reasons"] = [
            {"failure": "execution_failed", "cancelled": "cancel_requested", "lost": "transport_lost"}[category]
        ]
        c.append(case("structural", f"operation_{name}_missing_attempt", declaration(), e))

    e = map_events(0)

    map_terminal = next(x for x in e if x.get("kind") == "terminal" and x.get("activation") == "MAP")

    map_terminal["category"] = "failure"

    map_terminal["outcome"] = None

    map_terminal["reasons"] = ["execution_failed"]

    c.append(case("structural", "category_mismatch", declaration(), e))

    e = map_events(0)

    map_entry = next(x for x in e if x.get("kind") == "entry" and x.get("activation") == "MAP")

    map_entry["state_category"] = "failure"

    map_entry["state_outcome"] = None

    map_terminal = next(x for x in e if x.get("kind") == "terminal" and x.get("activation") == "MAP")

    map_terminal["category"] = "failure"

    map_terminal["outcome"] = None

    map_terminal["reasons"] = []

    c.append(case("structural", "failure_missing_reason", declaration(), e))

    e = map_events(0)

    next(x for x in e if x.get("kind") == "entry" and x.get("activation") == "MAP")["node_kind"] = "operation"

    c.append(case("structural", "node_kind_mismatch", declaration(), e))

    for status, category in (("failed", "failure"), ("overflow", "inconsistent")):
        e = map_events(1)
        map_entry = next(x for x in e if x.get("kind") == "entry" and x.get("activation") == "MAP")
        map_entry["state_category"] = category
        map_entry["state_outcome"] = None
        map_terminal = next(x for x in e if x.get("kind") == "terminal" and x.get("activation") == "MAP")
        map_terminal["category"] = category
        map_terminal["outcome"] = None
        map_terminal["reasons"] = ["execution_failed" if status == "failed" else "contradictory"]
        membership = next(x for x in e if x.get("kind") == "membership" and x.get("parent") == "MAP")
        membership["status"] = status
        membership["expansion_outcome"] = None
        insertion = next(i for i, x in enumerate(e) if x.get("kind") == "revision")
        e.insert(
            insertion,
            {"activation": "M1", "kind": "reservation", "parent": "MAP", "selected": True, "target": "A"},
        )
        c.append(case("structural", f"{status}_actual_members", declaration(), e))

    for name, field, value in (("missing_artifact", "artifact", "MISSINGv0"), ("foreign_target", "target", "B")):
        direct = direct_case
        e = cast(list[Obj], deepcopy(direct["events"]))
        decision_fact = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "DEC")
        decision_fact[field] = value
        c.append(case("decisions", name, cast(Obj, deepcopy(direct["declaration"])), e))

    e = [x for x in deepcopy(base) if not (x.get("kind") == "provenance" and x.get("key") == "OUT:A")]

    c.append(case("provenance", "missing_producer", declaration(), e))

    e = deepcopy(base)

    out = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "OUT:A")

    e.insert(e.index(out) + 1, deepcopy(out))

    c.append(case("provenance", "duplicate_producer", declaration(), e))

    d, e = execution_only_shape()

    c.append(case("release", "execution_only_empty", d, e))

    for name, deps_value in (
        ("a_only", {"context": "XAv0"}),
        ("b_only", {"decision": "YAv0"}),
        ("a_b", {"context": "XAv0", "decision": "YAv0"}),
    ):
        e = base_events(("A",))
        a = next(x for x in e if x.get("kind") == "assessment")
        if "decision" in deps_value:
            e.insert(1, artifact("YAv0", "A", "artifact"))
            e.insert(e.index(a), port("ROOT:A", "A", "decision", "YAv0", "artifact"))
            retain_root_input(e, "A", "decision", "YAv0")
        a["consumed"] = deps_value
        if "decision" in deps_value:
            seal = next(i for i, x in enumerate(e) if x.get("kind") == "seal_revision")
            e.insert(seal, {"collection": "artifacts", "key": "YA", "kind": "revision", "value": 1})
            e.insert(1, artifact("YAv1", "A", "artifact"))
        d = declaration()
        wire_evidence_consumed(d, e, tuple(deps_value))
        obj(arr(d["requirements"])[0])["consumed_ports"] = list(deps_value)
        c.append(case("selective", name, d, e))

    for dependency, consumed in (
        ("a_only", {"context": "XAv0"}),
        ("b_only", {"decision": "YAv0"}),
        ("a_b", {"context": "XAv0", "decision": "YAv0"}),
        ("decision", {"decision": "YAv0"}),
    ):
        for mode in ("current", "stale", "unknown"):
            e = base_events(("A",))
            assessment_fact = next(x for x in e if x.get("kind") == "assessment")
            e.insert(1, artifact("YAv0", "A", "decision"))
            e.insert(e.index(assessment_fact), port("ROOT:A", "A", "decision", "YAv0", "decision"))
            if "decision" in consumed:
                retain_root_input(e, "A", "decision", "YAv0")
            assessment_fact["consumed"] = consumed
            seal = next(i for i, x in enumerate(e) if x.get("kind") == "seal_revision")
            e.insert(seal, {"collection": "artifacts", "key": "YA", "kind": "revision", "value": 0})
            changed_key = "XA" if dependency == "a_only" else "YA"
            revision = next(
                x
                for x in e
                if x.get("kind") == "revision" and x.get("collection") == "artifacts" and x.get("key") == changed_key
            )
            if mode == "stale":
                revision["value"] = 1
                e.insert(1, artifact("XAv1" if changed_key == "XA" else "YAv1", "A", "artifact"))
            elif mode == "unknown":
                e.remove(revision)
            d = declaration()
            wire_evidence_consumed(d, e, tuple(consumed))
            obj(arr(d["requirements"])[0])["consumed_ports"] = list(consumed)
            c.append(case("selective", f"{dependency}_{mode}", d, e))

    e = deepcopy(base)

    e.insert(1, artifact("A1v0", "A", "candidate"))

    result_port = next(x for x in e if x.get("kind") == "port" and x.get("port") == "result")

    result_port["artifact"] = "A1v0"

    f = next(x for x in e if x.get("kind") == "final")

    f["candidate"] = "A1v0"

    p = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "OUT:A")

    p["artifact"] = "A1v0"

    seal = next(i for i, x in enumerate(e) if x.get("kind") == "seal_revision")

    e.insert(seal, {"collection": "artifacts", "key": "A1", "kind": "revision", "value": 0})

    d = declaration()

    set_output_dependency(d, "N", "result", ("subject", "context"))

    c.append(case("selective", "intermediate_a0_final_a1", d, e))

    for collection, key in (("artifacts", "A"), ("absences", "Q0"), ("configurations", "N"), ("state", "read")):
        e = deepcopy(base)
        source = next(
            x for x in e if x.get("kind") == "revision" and x.get("collection") == collection and x.get("key") == key
        )
        duplicate = deepcopy(source)
        if collection == "state":
            duplicate["value"] = cast(int, source["value"]) + 1
        e.insert(e.index(source) + 1, duplicate)
        c.append(case("bounds", f"duplicate_{collection}_key", declaration(), e))

    for name, field, value in (
        ("foreign_target", "target", "B"),
        ("foreign_evidence", "evidence_artifact", "EBv0"),
        ("foreign_subject", "subject_artifact", "Bv0"),
    ):
        c.append(case("assessment", name, declaration(), mutate(base, "assessment", field, value)))

    e = deepcopy(base)

    next(x for x in e if x.get("kind") == "final")["target"] = "B"

    c.append(case("final_output", "wrong_target", declaration(), e))

    e = deepcopy(base)

    next(x for x in e if x.get("kind") == "final")["outcome"] = "other"

    c.append(case("final_output", "wrong_outcome", declaration(), e))

    e = base_events(("A", "B"))

    e.insert(1, artifact("Dv0", "shared", "decision"))

    first_out = next(i for i, x in enumerate(e) if x.get("kind") == "provenance" and x.get("key") == "OUT:A")

    e[first_out:first_out] = [
        port("ROOT:A", "A", "shared_decision", "Dv0", "decision"),
        port("ROOT:B", "B", "shared_decision", "Dv0", "decision"),
        provenance("SHARED:DEC:A", "Dv0", "A", decision=True, port_name="shared_decision", activation="ROOT:A"),
        provenance("SHARED:DEC:B", "Dv0", "B", decision=True, port_name="shared_decision", activation="ROOT:B"),
        input_producer("ROOT:A", "A", "N", "shared_decision", "SHARED:DEC:A"),
        input_producer("ROOT:B", "B", "N", "shared_decision", "SHARED:DEC:B"),
    ]

    for target in ("A", "B"):
        next(x for x in e if x.get("kind") == "provenance" and x.get("key") == f"OUT:{target}")["parents"] = [
            f"SHARED:DEC:{target}",
            f"ROOT:{target}:subject",
        ]

    seal = next(i for i, x in enumerate(e) if x.get("kind") == "seal_revision")

    e.insert(seal, {"collection": "artifacts", "key": "D", "kind": "revision", "value": 0})

    d = declaration(targets=("A", "B"))

    set_output_dependency(d, "N", "shared_decision", ())

    set_output_dependency(d, "N", "result", ("shared_decision", "subject"), identity_input="subject")

    c.append(case("decisions", "shared_deduplicated", d, e))

    e = deepcopy(base)

    sites = (("D0", "N0", "p", "A"), ("D2", "N1", "p", "A"))

    e.insert(0, binding_receipt(sites, (("D0", 0, 1), ("D2", 0, 1))))

    out = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "OUT:A")

    idx = e.index(out)

    e[idx:idx] = [
        provenance(
            "BOUND:0",
            "XAv0",
            "A",
            source="bound_input",
            node="N0",
            port_name="p",
            binding_artifact=binding_ref("D0"),
        ),
        provenance(
            "BOUND:1",
            "XAv0",
            "A",
            source="bound_input",
            node="N1",
            port_name="p",
            binding_artifact=binding_ref("D2"),
        ),
        input_producer("ROOT:A", "A", "N", "bound_0", "BOUND:0"),
        input_producer("ROOT:A", "A", "N", "bound_1", "BOUND:1"),
        port("ROOT:A", "A", "bound_0", "XAv0", "artifact"),
        port("ROOT:A", "A", "bound_1", "XAv0", "artifact"),
    ]

    out["parents"] = ["BOUND:0", "BOUND:1", "ROOT:A:subject"]

    c.append(case("provenance", "multiple_bound_artifacts", binding_declaration(*sites), e))

    for name, kind, selector, field, value in (
        ("producer_node", "port", "result", "node", "OTHER"),
        ("producer_activation", "port", "result", "activation", "OTHER:A"),
        ("final_node", "final", None, "node", "OTHER"),
    ):
        e = deepcopy(base)
        fact = next(x for x in e if x.get("kind") == kind and (selector is None or x.get("port") == selector))
        fact[field] = value
        c.append(case("final_output", name, declaration(), e))

    c.append(
        case(
            "assessment",
            "wrong_kind_coverage",
            declaration(),
            mutate(base, "assessment", "coverage", ["OTHER"]),
        )
    )

    for name, mode in (("evidence_output_removed", "removed"), ("evidence_output_replaced", "replaced")):
        e = deepcopy(base)
        rev = next(x for x in e if x.get("kind") == "revision" and x.get("key") == "EA")
        if mode == "removed":
            e.remove(rev)
        else:
            rev["value"] = 1
            e.insert(1, artifact("EAv1", "A", "artifact"))
        c.append(case("validity", name, declaration(), e))

    e = deepcopy(base)

    next(
        x for x in e if x.get("kind") == "revision" and x.get("collection") == "configurations" and x.get("key") == "N"
    )["value"] = 0

    c.append(case("validity", "configuration_type_changed", declaration(), e))

    return c
