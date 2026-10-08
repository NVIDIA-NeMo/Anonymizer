# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent qualification_v1 reference: lifecycle cases."""

from __future__ import annotations

from copy import deepcopy

from tests.graph_sdk.reference._qualification_v1.case import (
    case,
)
from tests.graph_sdk.reference._qualification_v1.declarations import (
    declaration,
)
from tests.graph_sdk.reference._qualification_v1.events import (
    base_events,
    entry,
    terminal,
)
from tests.graph_sdk.reference._qualification_v1.family_builders import (
    binding_cleanup_events,
    request_history,
)
from tests.graph_sdk.reference._qualification_v1.map_builders import (
    map_events,
)
from tests.graph_sdk.reference._qualification_v1.model import (
    Obj,
    arr,
)


def lifecycle_cases() -> list[Obj]:
    c: list[Obj] = []
    base = base_events(("A",))
    for n in (0, 1, 2):
        c.append(case("membership", f"closed_{n}", declaration(), map_events(n)))

    c.append(case("membership", "nested_closed", declaration(), map_events(1, nested=True)))

    c.append(case("membership", "missing_expander_terminal", declaration(), map_events(0, parent_terminal=False)))

    e = map_events(1)

    e = [x for x in e if not (x.get("kind") == "terminal" and x.get("activation") == "M0")]

    c.append(case("membership", "missing_member_terminal", declaration(), e))

    e = map_events(1)

    idx = next(i for i, x in enumerate(e) if x.get("kind") == "provenance" and x.get("key") == "OUT:A")

    e.insert(idx, dict(terminal("M0", "A"), category="success"))

    c.append(case("membership", "duplicate_terminal", declaration(), e))

    e = deepcopy(base)

    idx = next(i for i, x in enumerate(e) if x.get("kind") == "revision")

    arr(next(x for x in e if x.get("kind") == "membership" and x.get("parent") is None)["members"]).append("CHOICE")

    e[idx:idx] = [
        {"activation": "CHOICE", "kind": "reservation", "parent": None, "selected": True, "target": "A"},
        entry("CHOICE", "A", closed=True, state_category="blocked", state_outcome=None),
        terminal("CHOICE", "A", "blocked", None),
    ]

    c.append(case("membership", "selected_closed_unstarted", declaration(), e))

    e = deepcopy(base)

    idx = next(i for i, x in enumerate(e) if x.get("kind") == "revision")

    e.insert(idx, {"activation": "OTHER", "kind": "reservation", "parent": None, "selected": False, "target": "A"})

    c.append(case("membership", "unselected_choice", declaration(), e))

    e = map_events(1)

    m = next(x for x in e if x.get("kind") == "membership" and x.get("parent") == "MAP")

    arr(m["members"]).append("M0")

    c.append(case("membership", "duplicate_member", declaration(), e))

    for purpose, failure in (("retry", "retryable"), ("correction", "malformed"), ("failover", "permanent")):
        c.append(case("request", f"{purpose}_recovered", declaration(), request_history(failure, purpose)))

    c.append(case("request", "lost_unknown", declaration(), request_history("lost", None, True)))

    c.append(case("request", "final_failure", declaration(), request_history("permanent", None)))

    for purpose in ("initial_binding", "adaptive_retrieval", "repair"):
        e = deepcopy(base)
        idx = next(i for i, x in enumerate(e) if x.get("kind") == "revision")
        e[idx:idx] = [
            {
                "associations": ["A"],
                "kind": "request_attempt",
                "policy": "P",
                "predecessor": None,
                "purpose": purpose,
                "request": f"R:{purpose}",
            },
            {"condition": "result", "failure": None, "kind": "request_terminal", "request": f"R:{purpose}"},
            {"kind": "settlement", "remote_stopped": True, "request": f"R:{purpose}", "usage": 1},
        ]
        c.append(case("request", f"fresh_{purpose}", declaration(), e))

    for name, purpose, disp, associated in (
        ("verification_failed", "verification", "close_failed", True),
        ("accounting_unknown", "accounting", "close_unknown", True),
        ("transport_failed", "transport_only", "close_failed", True),
        ("caller_left_open", "verification", "left_open", True),
        ("missing_association", "verification", "close_failed", False),
    ):
        e = deepcopy(base)
        idx = next(i for i, x in enumerate(e) if x.get("kind") == "revision")
        facts: list[Obj] = []
        if associated:
            facts.append(
                {
                    "kind": "cleanup_association",
                    "owner": "caller" if name == "caller_left_open" else "sdk",
                    "purpose": purpose,
                    "resource": "Q0",
                    "targets": ["A"],
                }
            )
        facts.append({"disposition": disp, "kind": "cleanup", "resource": "Q0"})
        e[idx:idx] = facts
        c.append(case("cleanup", name, declaration(), e))

    for name, purpose in (
        ("empty_transport_only", "transport_only"),
        ("empty_verification", "verification"),
        ("empty_accounting", "accounting"),
    ):
        e = deepcopy(base)
        idx = next(i for i, x in enumerate(e) if x.get("kind") == "revision")
        e[idx:idx] = [
            {"kind": "cleanup_association", "owner": "sdk", "purpose": purpose, "resource": "EMPTY", "targets": []},
            {"disposition": "close_failed", "kind": "cleanup", "resource": "EMPTY"},
        ]
        c.append(case("cleanup", name, declaration(), e))

    binding_cleanup_declaration = declaration(targets=("A", "C"))

    for name, disposition, association_targets, duplicate in (
        ("binding_closed", "closed", ("A",), False),
        ("binding_failed_local_a", "close_failed", ("A",), False),
        ("binding_missing_association", "close_failed", None, False),
        ("binding_foreign_association", "close_failed", ("Z",), False),
        ("binding_extra_association", None, ("A",), False),
        ("binding_duplicate_association", "closed", ("A",), True),
    ):
        c.append(
            case(
                "cleanup",
                name,
                deepcopy(binding_cleanup_declaration),
                binding_cleanup_events(
                    disposition=disposition, association_targets=association_targets, duplicate=duplicate
                ),
            )
        )

    return c
