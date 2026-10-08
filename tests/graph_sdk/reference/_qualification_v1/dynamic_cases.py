# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent qualification_v1 reference: dynamic cases."""

from __future__ import annotations

from copy import deepcopy

from tests.graph_sdk.reference._qualification_v1.case import (
    case,
)
from tests.graph_sdk.reference._qualification_v1.declarations import (
    declaration,
)
from tests.graph_sdk.reference._qualification_v1.dynamic_builders import (
    assessed_map_declaration,
    assessed_map_events,
    inject_member_assessment,
    non_success_member_events,
    submit_dynamic_occurrences,
)
from tests.graph_sdk.reference._qualification_v1.events import (
    assessment_submission,
    base_events,
    terminal,
)
from tests.graph_sdk.reference._qualification_v1.model import (
    Obj,
    obj,
)


def dynamic_cases() -> list[Obj]:
    c: list[Obj] = []
    base = base_events(("A",))
    for count in range(3):
        c.append(
            case(
                "assessment",
                f"dynamic_occurrences_{count}",
                assessed_map_declaration(count),
                assessed_map_events(count),
            )
        )

    e = assessed_map_events(1)

    e = [event for event in e if not (event.get("kind") == "assessment" and event.get("activation") == "M0")]

    c.append(case("assessment", "dynamic_occurrence_missing", assessed_map_declaration(1), e))

    e = assessed_map_events(1)

    duplicate = deepcopy(
        next(event for event in e if event.get("kind") == "assessment" and event.get("activation") == "M0")
    )

    duplicate["fact"] = "F:M0:P_MEMBER:DUPLICATE"

    e.insert(next(i for i, event in enumerate(e) if event.get("kind") == "revision"), duplicate)

    c.append(case("assessment", "dynamic_occurrence_duplicate", assessed_map_declaration(1), e))

    for count in (1, 2):
        c.append(
            case(
                "assessment",
                f"dynamic_submissions_{count}",
                assessed_map_declaration(count),
                submit_dynamic_occurrences(count),
            )
        )

    e = submit_dynamic_occurrences(1)

    insertion = next(i for i, event in enumerate(e) if event.get("kind") == "revision")

    e.insert(insertion, assessment_submission("M0", "P_MEMBER"))

    c.append(case("assessment", "dynamic_submission_repeated", assessed_map_declaration(1), e))

    e = submit_dynamic_occurrences(1)

    insertion = next(i for i, event in enumerate(e) if event.get("kind") == "revision")

    e.insert(insertion, {"fact": "F:FOREIGN:P_MEMBER", "kind": "assessment_submission"})

    c.append(case("assessment", "dynamic_submission_foreign", assessed_map_declaration(1), e))

    e = assessed_map_events(0)

    e = [
        event
        for event in e
        if not (
            event.get("kind") == "assessment"
            or event.get("kind") == "assessment_submission"
            or event.get("kind") == "final"
            or event.get("kind") == "artifact"
            and event.get("ref") in {"EAv0", "CAv0"}
            or event.get("kind") == "collection_value"
            or event.get("kind") == "port"
            and (event.get("activation"), event.get("port"))
            in {
                ("ROOT:A", "subject"),
                ("ROOT:A", "context"),
                ("ROOT:A", "evidence"),
                ("ROOT:A", "result"),
                ("ROOT:A", "membership"),
                ("MAP", "members"),
            }
            or event.get("kind") == "input_producer"
            and event.get("activation") == "ROOT:A"
            or event.get("kind") == "provenance"
            and event.get("key") not in {"ROOT:A:subject", "ROOT:A:context"}
            or event.get("kind") == "revision"
            and event.get("collection") == "artifacts"
            and event.get("key") == "EA"
        )
    ]

    next(event for event in e if event.get("kind") == "artifact" and event.get("ref") == "Av0")["role"] = "artifact"

    map_entry = next(event for event in e if event.get("kind") == "entry" and event.get("activation") == "MAP")

    map_entry.update({"state_category": "failure", "state_outcome": None})

    map_terminal = next(event for event in e if event.get("kind") == "terminal" and event.get("activation") == "MAP")

    map_terminal.update({"category": "failure", "outcome": None, "reasons": ["execution_failed"]})

    membership = next(event for event in e if event.get("kind") == "membership" and event.get("parent") == "MAP")

    membership.update({"expansion_outcome": None, "status": "failed"})

    root_entry = next(event for event in e if event.get("kind") == "entry" and event.get("activation") == "ROOT:A")

    root_entry.update({"closed_unstarted": True, "state_category": "blocked", "state_outcome": None})

    root_terminal = next(
        event for event in e if event.get("kind") == "terminal" and event.get("activation") == "ROOT:A"
    )

    root_terminal.clear()

    root_terminal.update(terminal("ROOT:A", "A", "blocked", None))

    join_entry = next(event for event in e if event.get("kind") == "entry" and event.get("activation") == "JOIN")

    join_entry.update({"closed_unstarted": True, "state_category": "blocked", "state_outcome": None})

    join_terminal = next(event for event in e if event.get("kind") == "terminal" and event.get("activation") == "JOIN")

    join_terminal.clear()

    join_terminal.update(terminal("JOIN", "A", "blocked", None))

    insertion = e.index(membership)

    e.insert(
        insertion,
        {"activation": "M0", "kind": "reservation", "parent": "MAP", "selected": True, "target": "A"},
    )

    c.append(case("assessment", "dynamic_unreached_failed_expansion", assessed_map_declaration(0), e))

    c.append(
        case(
            "assessment",
            "dynamic_unreached_failed_expansion_injected",
            assessed_map_declaration(0),
            inject_member_assessment(e),
        )
    )

    for name, category, closed_unstarted in (
        ("blocked_unreached", "blocked", True),
        ("started_failure", "failure", False),
    ):
        e = non_success_member_events(category, closed_unstarted=closed_unstarted)
        d = assessed_map_declaration(1)
        if category == "blocked":
            obj(d["node_kinds"])["FAILED_SOURCE"] = "operation"
        c.append(case("assessment", f"dynamic_{name}", d, e))
        c.append(
            case(
                "assessment",
                f"dynamic_{name}_injected",
                d,
                inject_member_assessment(e),
            )
        )

    root_missing_terminal = [
        deepcopy(event)
        for event in base
        if not (event.get("kind") == "terminal" and event.get("activation") == "ROOT:A")
    ]

    root_unsubmitted = [event for event in root_missing_terminal if event.get("kind") != "assessment_submission"]

    c.append(case("assessment", "root_missing_terminal_unsubmitted", declaration(), root_unsubmitted))

    c.append(case("assessment", "root_missing_terminal_submitted", declaration(), root_missing_terminal))

    dynamic_missing_terminal = [
        event
        for event in assessed_map_events(1)
        if not (event.get("kind") == "terminal" and event.get("activation") == "M0")
    ]

    c.append(
        case(
            "assessment",
            "dynamic_missing_terminal_unsubmitted",
            assessed_map_declaration(1),
            dynamic_missing_terminal,
        )
    )

    dynamic_submitted = deepcopy(dynamic_missing_terminal)

    insertion = next(i for i, event in enumerate(dynamic_submitted) if event.get("kind") == "revision")

    dynamic_submitted.insert(insertion, assessment_submission("M0", "P_MEMBER"))

    c.append(
        case(
            "assessment",
            "dynamic_missing_terminal_submitted",
            assessed_map_declaration(1),
            dynamic_submitted,
        )
    )

    return c
