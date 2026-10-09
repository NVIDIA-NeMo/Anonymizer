# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent qualification_v1 reference: release cases."""

from __future__ import annotations

from copy import deepcopy

from tests.graph_sdk.reference._qualification_v1.case import (
    case,
    mutate,
)
from tests.graph_sdk.reference._qualification_v1.declarations import (
    declaration,
)
from tests.graph_sdk.reference._qualification_v1.events import (
    artifact,
    base_events,
)
from tests.graph_sdk.reference._qualification_v1.family_builders import (
    execution_only_shape,
    multi_promise_declaration,
    multi_promise_events,
)
from tests.graph_sdk.reference._qualification_v1.model import (
    Obj,
)


def release_cases() -> list[Obj]:
    c: list[Obj] = []
    base = base_events(("A",))
    c.append(case("release", "protection_success", declaration(), base))

    execution_declaration, execution_events = execution_only_shape()

    c.append(case("release", "execution_only", execution_declaration, execution_events))

    for name, kind, field, value in (
        ("missing_candidate_artifact", "artifact", "ref", "DROP"),
        ("missing_evidence_artifact", "artifact", "ref", "DROP"),
        ("producer_artifact_mismatch", "provenance", "artifact", "XAv0"),
        ("producer_target_mismatch", "provenance", "target", "B"),
        ("evidence_port_swap", "assessment", "evidence_port", "wrong"),
        ("subject_port_swap", "assessment", "subject_port", "wrong"),
        ("invocation_mismatch", "artifact", "invocation", "I1"),
        ("node_mismatch", "assessment", "node", "OTHER"),
        ("outcome_mismatch", "assessment", "outcome", "other"),
        ("promise_mismatch", "assessment", "promise", "other"),
    ):
        e = deepcopy(base)
        if name == "missing_candidate_artifact":
            e = [x for x in e if not (x.get("kind") == "artifact" and x.get("ref") == "Av0")]
        elif name == "missing_evidence_artifact":
            e = [x for x in e if not (x.get("kind") == "artifact" and x.get("ref") == "EAv0")]
        else:
            target = next(x for x in e if x.get("kind") == kind and (kind != "provenance" or x.get("key") == "OUT:A"))
            target[field] = value
        c.append(case("joins", name, declaration(), e))

    for label, field, key in (
        ("candidate", "artifacts", "A"),
        ("evidence", "artifacts", "EA"),
        ("consumed", "artifacts", "XA"),
        ("absence", "absences", "Q0"),
        ("configuration", "configurations", "N"),
        ("state", "state", "read"),
    ):
        for mode in ("current", "stale", "unknown"):
            e = deepcopy(base)
            rev = [
                x for x in e if x.get("kind") == "revision" and x.get("collection") == field and x.get("key") == key
            ][0]
            if mode == "stale":
                rev["value"] = 2 if field != "configurations" else "c1"
                if field == "artifacts":
                    ref = {"A": "Av2", "EA": "EAv2", "XA": "XAv2"}[key]
                    e.insert(0, artifact(ref, "A", "artifact"))
            elif mode == "unknown":
                e.remove(rev)
            c.append(case("validity", f"{label}_{mode}", declaration(), e))

    e = deepcopy(base)

    next(x for x in e if x.get("kind") == "revision" and x.get("key") == "A")["value"] = 1

    e.insert(0, artifact("Av1", "A", "artifact"))

    c.append(case("validity", "final_candidate_sibling_substitution", declaration(), e))

    e = deepcopy(base)

    e = [value for value in e if value.get("kind") != "assessment_submission"]

    next(value for value in e if value.get("kind") == "revision" and value.get("key") == "A")["value"] = 2

    e.insert(0, artifact("Av2", "A", "artifact"))

    c.append(case("validity", "candidate_stale_no_assessment", declaration(), e))

    e = deepcopy(base)

    e = [x for x in e if not (x.get("kind") == "revision" and x.get("key") == "Q0")]

    next(x for x in e if x.get("kind") == "revision" and x.get("key") == "XA")["value"] = 1

    e.insert(0, artifact("XAv1", "A", "artifact"))

    c.append(case("validity", "stale_precedes_unknown", declaration(), e))

    for name, field, value in (
        ("unsatisfied", "finding", "unsatisfied"),
        ("unknown", "finding", "unknown"),
        ("incomplete_coverage", "coverage", []),
        ("extra_coverage", "coverage", ["K0", "K1", "K2"]),
        ("duplicate_coverage", "coverage", ["K0", "K0"]),
        ("caller_copy", "authenticated_factory", "CALLER"),
        ("consumed_port", "consumed", {"wrong": "XAv0"}),
        ("foreign_consumed", "consumed", {"context": "XBv0"}),
    ):
        c.append(case("assessment", name, declaration(), mutate(base, "assessment", field, value)))

    c.append(
        case(
            "assessment",
            "missing",
            declaration(),
            [x for x in base if x.get("kind") != "assessment_submission"],
        )
    )

    e = deepcopy(base)

    e.insert(
        e.index(next(x for x in e if x.get("kind") == "revision")),
        deepcopy(next(x for x in e if x.get("kind") == "assessment_submission")),
    )

    c.append(case("assessment", "duplicate", declaration(), e))

    c.append(case("assessment", "unsupported_finding", declaration(), mutate(base, "assessment", "finding", "other")))

    c.append(
        case(
            "assessment",
            "absence_query",
            declaration(),
            mutate(
                base,
                "assessment",
                "environment",
                {"absences": {"Q1": 1}, "configurations": {"N": "c0"}, "state": {"read": 1}},
            ),
        )
    )

    for name, partial, complete in (
        ("partial_promise_only", True, False),
        ("complete_promise_only", False, True),
        ("partial_and_complete_promises", True, True),
    ):
        c.append(
            case(
                "assessment",
                name,
                multi_promise_declaration(),
                multi_promise_events(partial=partial, complete=complete),
            )
        )

    c.append(
        case(
            "assessment",
            "caller_replacement",
            declaration(),
            mutate(base, "assessment", "authenticated_factory", "CALLER"),
        )
    )

    return c
