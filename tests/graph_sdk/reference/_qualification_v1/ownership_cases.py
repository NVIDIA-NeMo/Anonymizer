# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent qualification_v1 reference: ownership cases."""

from __future__ import annotations

from copy import deepcopy
from typing import cast

from tests.graph_sdk.reference._qualification_v1.case import (
    case,
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
    binding_declaration,
    binding_receipt,
    binding_ref,
    entry,
    input_producer,
    port,
    provenance,
    terminal,
)
from tests.graph_sdk.reference._qualification_v1.family_builders import (
    initial_version_seed,
    retain_root_input,
)
from tests.graph_sdk.reference._qualification_v1.map_builders import (
    map_events,
)
from tests.graph_sdk.reference._qualification_v1.model import (
    Obj,
    arr,
    obj,
)


def ownership_cases(direct_case: Obj) -> list[Obj]:
    c: list[Obj] = []
    base = base_events(("A",))
    e = base_events(("A", "B"))

    e.insert(1, artifact("Sv0", "shared", "artifact"))

    out_a = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "OUT:A")

    out_b = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "OUT:B")

    first = min(e.index(out_a), e.index(out_b))

    e[first:first] = [
        port("ROOT:A", "A", "shared_context", "Sv0", "artifact"),
        port("ROOT:B", "B", "shared_context", "Sv0", "artifact"),
        provenance(
            "SHARED:A",
            "Sv0",
            "A",
            source="root_input",
            port_name="shared_context",
            activation="ROOT:A",
        ),
        provenance(
            "SHARED:B",
            "Sv0",
            "B",
            source="root_input",
            port_name="shared_context",
            activation="ROOT:B",
        ),
        input_producer("ROOT:A", "A", "N", "shared_context", "SHARED:A"),
        input_producer("ROOT:B", "B", "N", "shared_context", "SHARED:B"),
    ]

    out_a["parents"] = ["SHARED:A", "ROOT:A:subject"]

    out_b["parents"] = ["SHARED:B", "ROOT:B:subject"]

    seal = next(i for i, x in enumerate(e) if x.get("kind") == "seal_revision")

    e.insert(seal, {"collection": "artifacts", "key": "S", "kind": "revision", "value": 0})

    d = declaration(targets=("A", "B"))

    set_output_dependency(d, "N", "result", ("shared_context", "subject"), identity_input="subject")

    c.append(case("provenance", "shared_context_identity", d, e))

    for name, declaration_id, target, d in (
        (
            "mismatched_bound_root",
            "D2",
            "A",
            binding_declaration(("D0", "N0", "p", "A")),
        ),
        (
            "foreign_bound_target",
            "D0",
            "B",
            binding_declaration(("D0", "N0", "p", "A"), targets=("A", "B")),
        ),
    ):
        e = deepcopy(base)
        e.insert(0, binding_receipt((("D0", "N0", "p", "A"),), (("D0", 0, 1),)))
        out = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "OUT:A")
        idx = e.index(out)
        e.insert(
            idx,
            provenance(
                "BOUND:BAD",
                "XAv0",
                target,
                source="bound_input",
                node="N0",
                port_name="p",
                binding_artifact=binding_ref(declaration_id),
            ),
        )
        out["parents"] = ["BOUND:BAD"]
        c.append(case("provenance", name, d, e))

    for name, linked in (("withheld_unrelated", False), ("withheld_linked", True)):
        e = base_events(("A", "B"))
        next(x for x in e if x.get("kind") == "assessment" and x.get("target") == "A")["finding"] = "unsatisfied"
        e.insert(1, artifact("Dv0", "shared", "decision"))
        out_a = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "OUT:A")
        out_b = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "OUT:B")
        insertion = min(e.index(out_a), e.index(out_b))
        e[insertion:insertion] = [
            port("ROOT:A", "A", "withheld_decision", "Dv0", "decision"),
            port("ROOT:B", "B", "withheld_decision", "Dv0", "decision"),
            provenance(
                "WITHHELD:DEC:A",
                "Dv0",
                "A",
                decision=True,
                port_name="withheld_decision",
                activation="ROOT:A",
            ),
            provenance(
                "WITHHELD:DEC:B",
                "Dv0",
                "B",
                decision=True,
                port_name="withheld_decision",
                activation="ROOT:B",
            ),
        ]
        d = declaration(targets=("A", "B"))
        set_output_dependency(d, "N", "withheld_decision", ())
        if linked:
            e.insert(e.index(out_a), input_producer("ROOT:A", "A", "N", "withheld_decision", "WITHHELD:DEC:A"))
            e.insert(e.index(out_b), input_producer("ROOT:B", "B", "N", "withheld_decision", "WITHHELD:DEC:B"))
            out_a["parents"] = ["WITHHELD:DEC:A", "ROOT:A:subject"]
            out_b["parents"] = ["WITHHELD:DEC:B", "ROOT:B:subject"]
            set_output_dependency(d, "N", "result", ("withheld_decision", "subject"), identity_input="subject")
        seal = next(i for i, x in enumerate(e) if x.get("kind") == "seal_revision")
        e.insert(seal, {"collection": "artifacts", "key": "D", "kind": "revision", "value": 0})
        c.append(case("decisions", name, d, e))

    e = [x for x in deepcopy(base) if not (x.get("kind") == "membership" and x.get("parent") is None)]

    c.append(case("membership", "missing_root", declaration(), e))

    e = deepcopy(base)

    idx = next(i for i, x in enumerate(e) if x.get("kind") == "revision")

    e[idx:idx] = [entry("ORPHAN", "A"), terminal("ORPHAN", "A")]

    c.append(case("membership", "orphan_entry", declaration(), e))

    e = deepcopy(base)

    idx = next(i for i, x in enumerate(e) if x.get("kind") == "revision")

    e.insert(idx, terminal("STRAY", "A"))

    c.append(case("membership", "stray_terminal", declaration(), e))

    e = map_events(1)

    next(x for x in e if x.get("kind") == "entry" and x.get("activation") == "M0")["target"] = "B"

    next(x for x in e if x.get("kind") == "terminal" and x.get("activation") == "M0")["target"] = "B"

    c.append(case("membership", "cross_target_member", declaration(targets=("A", "B")), e))

    revision_mutants: tuple[tuple[str, Obj], ...] = (
        ("invented_artifact", {"collection": "artifacts", "key": "INVENTED", "kind": "revision", "value": 0}),
        ("invented_absence", {"collection": "absences", "key": "Q1", "kind": "revision", "value": 1}),
        ("invented_configuration", {"collection": "configurations", "key": "OTHER", "kind": "revision", "value": "c0"}),
        ("invented_state", {"collection": "state", "key": "write", "kind": "revision", "value": 1}),
    )

    for name, event in revision_mutants:
        e = deepcopy(base)
        e.insert(next(i for i, x in enumerate(e) if x.get("kind") == "seal_revision"), event)
        c.append(case("revision", name, declaration(), e))

    e = deepcopy(base)

    next(x for x in e if x.get("kind") == "artifact" and x.get("ref") == "EAv0")["role"] = "anything"

    c.append(case("roles", "global_role_ignored", declaration(), e))

    e = deepcopy(base)

    next(x for x in e if x.get("kind") == "port" and x.get("port") == "evidence")["role"] = "candidate"

    c.append(case("roles", "distinct_evidence_wrong_role", declaration(), e))

    d = declaration()

    obj(arr(d["productions"])[0])["subject_source"] = "input"

    d["root_outputs"] = [
        {"input_port": None, "port": "evidence", "source_node": "N", "source_port": "evidence", "target": "A"}
    ]

    e = deepcopy(base)

    next(x for x in e if x.get("kind") == "port" and x.get("port") == "evidence")["role"] = "candidate"

    f = next(x for x in e if x.get("kind") == "final")

    f.update({"candidate": "EAv0", "port": "evidence", "producer": "EVID:A"})

    c.append(case("roles", "root_bound_distinct_evidence_candidate", d, e))

    e = deepcopy(base)

    next(x for x in e if x.get("kind") == "port" and x.get("port") == "subject")["role"] = "decision"

    c.append(case("roles", "subject_decision_collision", declaration(), e))

    e = deepcopy(base)

    next(x for x in e if x.get("kind") == "port" and x.get("port") == "context")["role"] = "other"

    c.append(case("roles", "consumed_unsupported_role", declaration(), e))

    d = declaration()

    set_production_consumed(d, ("subject", "context"))

    obj(arr(d["requirements"])[0])["consumed_ports"] = ["subject", "context"]

    set_output_dependency(d, "N", "evidence", ("subject", "context"), identity_input="subject")

    e = deepcopy(base)

    next(x for x in e if x.get("kind") == "port" and x.get("port") == "evidence")["artifact"] = "Av0"

    evidence_provenance = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "EVID:A")

    evidence_provenance.update({"artifact": "Av0", "parents": ["ROOT:A:subject", "ROOT:A:context"]})

    e = [x for x in e if not (x.get("kind") == "artifact" and x.get("ref") == "EAv0")]

    e = [
        x
        for x in e
        if not (x.get("kind") == "revision" and x.get("collection") == "artifacts" and x.get("key") == "EA")
    ]

    a = next(x for x in e if x.get("kind") == "assessment")

    a["evidence_artifact"] = "Av0"

    a["consumed"] = {"context": "XAv0", "subject": "Av0"}

    c.append(case("roles", "candidate_evidence_alias", d, e))

    for name, ref in (("missing_parent_artifact", "MISSINGv0"), ("mismatched_source_artifact", "Av0")):
        e = deepcopy(base)
        next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "ROOT:A:context")["artifact"] = ref
        c.append(case("provenance", name, declaration(), e))

    d = declaration()

    p2 = deepcopy(obj(arr(d["productions"])[0]))

    p2.update(
        {"consumed_ports": ["decision"], "evidence_port": "evidence2", "promise": "P2", "subject_port": "subject2"}
    )

    arr(d["productions"]).append(p2)

    set_output_dependency(d, "N", "evidence2", ("decision",))

    e = deepcopy(base)

    insertion = next(i for i, x in enumerate(e) if x.get("kind") == "revision")

    e[insertion:insertion] = [
        artifact("E2v0", "A", "evidence"),
        artifact("Dv0", "A", "decision"),
        port("ROOT:A", "A", "evidence2", "E2v0", "evidence"),
        port("ROOT:A", "A", "subject2", "Av0", "candidate"),
        port("ROOT:A", "A", "decision", "Dv0", "decision"),
        assessment(
            "A",
            promise="P2",
            evidence_port="evidence2",
            subject_port="subject2",
            evidence_artifact="E2v0",
            consumed={"decision": "Dv0"},
        ),
        assessment_submission("A", "P2"),
        provenance(
            "EVID2:A",
            "E2v0",
            "A",
            ("ROOT:A:decision",),
            port_name="evidence2",
            activation="ROOT:A",
        ),
    ]

    retain_root_input(e, "A", "decision", "Dv0")

    seal = next(i for i, x in enumerate(e) if x.get("kind") == "seal_revision")

    e[seal:seal] = [
        {"collection": "artifacts", "key": "E2", "kind": "revision", "value": 0},
        {"collection": "artifacts", "key": "D", "kind": "revision", "value": 0},
    ]

    c.append(case("selection", "unrelated_assessment", d, e))

    d = declaration(targets=("A", "B"))

    e = base_events(("A", "B"))

    idx = next(i for i, x in enumerate(e) if x.get("kind") == "revision")

    e[idx:idx] = [
        {
            "associations": ["A", "B"],
            "kind": "request_attempt",
            "policy": "P",
            "predecessor": None,
            "purpose": "initial",
            "request": "R0",
        },
        {"condition": "failure", "failure": "retryable", "kind": "request_terminal", "request": "R0"},
        {"kind": "settlement", "remote_stopped": True, "request": "R0", "usage": 1},
        {
            "associations": ["B", "A"],
            "kind": "request_attempt",
            "policy": "P",
            "predecessor": "R0",
            "purpose": "retry",
            "request": "R1",
        },
        {"condition": "result", "failure": None, "kind": "request_terminal", "request": "R1"},
        {"kind": "settlement", "remote_stopped": True, "request": "R1", "usage": 1},
    ]

    c.append(case("request", "association_order_invariant", d, e))

    c.append(case("bounds", "exact", declaration(), base))

    for name, exact, over, e in (
        ("ports", {"max_port_facts": 4}, {"max_port_facts": 3}, base),
        ("consumed", {"max_consumed_per_assessment": 1}, {"max_consumed_per_assessment": 0}, base),
        ("edges", {"max_provenance_edges": 3}, {"max_provenance_edges": 2}, base),
        ("submissions", {"max_submissions": 1}, {"max_submissions": 0}, base),
        ("revisions", {"max_revision_entries": 5}, {"max_revision_entries": 4}, base),
        ("absences", {"max_absence_revisions": 1}, {"max_absence_revisions": 0}, base),
    ):
        c.append(case("bounds", f"{name}_exact", declaration(limit_changes=exact), deepcopy(e)))
        c.append(case("bounds", f"{name}_one_over", declaration(limit_changes=over), deepcopy(e)))

    for name, exact, over in (
        ("productions", {"max_productions": 1}, {"max_productions": 0}),
        ("coverage", {"max_coverage_atoms": 2}, {"max_coverage_atoms": 1}),
        ("verified", {"max_verified_evidence": 1}, {"max_verified_evidence": 0}),
    ):
        c.append(case("bounds", f"{name}_exact", declaration(limit_changes=exact), deepcopy(base)))
        boundary = "admission" if name in ("productions", "coverage") else "qualification"
        c.append(
            case(
                "bounds",
                f"{name}_one_over",
                declaration(limit_changes=over),
                [] if boundary == "admission" else deepcopy(base),
                boundary,
            )
        )

    decision_events = cast(list[Obj], deepcopy(direct_case["events"]))

    decision_declaration = cast(Obj, deepcopy(direct_case["declaration"]))

    obj(decision_declaration["limits"])["max_required_decisions"] = 1

    c.append(
        case(
            "bounds",
            "required_decisions_exact",
            decision_declaration,
            decision_events,
        )
    )

    c.append(
        case(
            "bounds",
            "required_decisions_one_over",
            dict(decision_declaration, limits=dict(obj(decision_declaration["limits"]), max_required_decisions=0)),
            deepcopy(decision_events),
        )
    )

    fixed_events = base_events(("A", "B", "C"))

    c.append(
        case(
            "bounds",
            "fixed_point_exact",
            declaration(targets=("A", "B", "C"), limit_changes={"max_fixed_point_steps": 3}),
            fixed_events,
        )
    )

    c.append(
        case(
            "bounds",
            "fixed_point_one_over",
            declaration(targets=("A", "B", "C"), limit_changes={"max_fixed_point_steps": 2}),
            [],
            "admission",
        )
    )

    for name in (
        "initial_two_versions",
        "two_selected_current_versions",
        "invented_version",
        "declaration_crossover",
        "target_crossover",
        "invocation_crossover",
        "rejected_publication_rollback",
        "latest_older_selected",
        "latest_missing_binding_request",
        "latest_foreign_binding_association",
        "latest_missing_binding_settlement",
        "latest_missing_binding_cleanup",
        "latest_foreign_cleanup_target",
    ):
        d, e = initial_version_seed()
        c.append(case("lineage", name, d, e))

    for name, limit_changes in (
        ("artifacts_exact", {"max_artifacts": 4}),
        ("artifacts_one_over", {"max_artifacts": 3}),
        ("bytes_exact", {"max_artifact_bytes": 4}),
        ("bytes_one_over", {"max_artifact_bytes": 3}),
        ("provenance_exact", {"max_provenance_edges": 3}),
        ("provenance_one_over", {"max_provenance_edges": 2}),
    ):
        d, e = initial_version_seed(limit_changes=limit_changes)
        c.append(case("lineage", name, d, e))

    d = binding_declaration(("D0", "N0", "p", "A"))

    e = base_events(("A",))

    receipt = binding_receipt((("D0", "N0", "p", "A"),), (("D0", 0, 1), ("D0", 0, 1)))

    e.insert(0, receipt)

    c.append(case("lineage", "source_duplicate_pair", d, e))

    for name, d in (
        ("duplicate_production", declaration()),
        ("fixed_point", declaration(limit_changes={"max_fixed_point_steps": 0})),
        ("foreign_dependency", declaration(deps=(("A", "B"),))),
        ("production_one_over", declaration(limit_changes={"max_productions": 0})),
        ("overlapping_atomic", declaration(targets=("A", "B", "C"), atomic=(("A", "B"), ("B", "C")))),
        ("unsupported_outcome", declaration()),
        ("missing_promise", declaration()),
    ):
        if name == "duplicate_production":
            arr(d["productions"]).append(deepcopy(arr(d["productions"])[0]))
        elif name == "unsupported_outcome":
            obj(arr(d["requirements"])[0])["outcome"] = "other"
        elif name == "missing_promise":
            obj(arr(d["requirements"])[0])["meaning"] = "other"
        c.append(case("admission", name, d, [], "admission"))

    d = declaration(targets=("A", "B", "C"), atomic=(("A", "B"),))

    d["atomic"] = [["A", "B"]]

    c.append(case("admission", "incomplete_atomic_partition", d, [], "admission"))

    d = binding_declaration(("D0", "N0", "p", "A"))

    obj(arr(d["binding_inputs"])[0])["node"] = "OTHER"

    c.append(case("admission", "binding_foreign_node", d, [], "admission"))

    d = declaration()

    arr(d["initial_collections"]).append({"declaration": "OTHER", "node": "N0", "port": "items", "target": "A"})

    c.append(case("admission", "initial_collection_without_binding", d, [], "admission"))

    d = declaration()

    arr(d["map_inputs"]).append(deepcopy(arr(d["map_inputs"])[0]))

    c.append(case("admission", "duplicate_map_input", d, [], "admission"))

    d = declaration()

    set_output_dependency(d, "N", "evidence", ("subject", "context"))

    c.append(case("admission", "evidence_dependency_consumed_mismatch", d, [], "admission"))

    d = binding_declaration(("D0", "N0", "p", "A"), ("D1", "N0", "items", "A"))

    obj(arr(d["binding_inputs"])[1])["declaration"] = "D0"

    c.append(case("admission", "duplicate_binding_declaration", d, [], "admission"))

    c.append(case("admission", "empty_atomic_group", declaration(atomic=((),)), [], "admission"))

    d = declaration()

    primary = deepcopy(base)

    alternate = deepcopy(primary)

    a = next(i for i, x in enumerate(alternate) if x.get("kind") == "revision" and x.get("key") == "A")

    q = next(i for i, x in enumerate(alternate) if x.get("kind") == "revision" and x.get("key") == "Q0")

    alternate[a], alternate[q] = alternate[q], alternate[a]

    c.append(case("commutation", "independent_revisions", d, primary, alternates=(alternate,)))

    return c
