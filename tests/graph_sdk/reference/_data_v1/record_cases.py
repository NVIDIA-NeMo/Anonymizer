# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent data_v1 reference: record cases."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from copy import deepcopy

from tests.graph_sdk.reference._data_v1.model import (
    JsonObject,
    JsonValue,
    Ref,
    _as_list,
    _as_object,
)
from tests.graph_sdk.reference._data_v1.validation import (
    _json_ref,
)


def _record(boundary: str, **facts: JsonValue) -> JsonObject:
    return {"kind": "record", "boundary": boundary, "facts": facts}


def _activation(
    occurrence: int | bool = 0,
    *,
    invocation: str = "I",
    parent: JsonObject | None = None,
    iteration: int | bool | None = None,
) -> JsonObject:
    return {
        "invocation": invocation,
        "occurrence": occurrence,
        "parent": deepcopy(parent),
        "iteration": iteration,
    }


def _status(
    target: Ref = ("local", 0),
    *,
    completion: str = "closed",
    qualification: str = "met",
    artifact_available: bool = True,
    protection_available: bool = False,
) -> JsonObject:
    return {
        "target": _json_ref(target),
        "completion": completion,
        "qualification": qualification,
        "artifact_available": artifact_available,
        "protection_available": protection_available,
    }


def _terminal(
    *,
    activation: JsonObject | None = None,
    attempt: str | None = "T",
    attempt_activation: JsonObject | None = None,
    category: str = "success",
    reasons: Sequence[str] = (),
) -> JsonObject:
    activation = _activation() if activation is None else activation
    owned_activation = activation if attempt_activation is None else attempt_activation
    return {
        "activation": deepcopy(activation),
        "attempt": None if attempt is None else {"id": attempt, "activation": deepcopy(owned_activation)},
        "category": category,
        "reasons": list(reasons),
    }


def _artifact(*, invocation: str = "I", key: int | bool = 0, version: int | bool = 1) -> JsonObject:
    return {"invocation": invocation, "key": key, "version": version}


def _canonical() -> JsonObject:
    evidence_artifact = _artifact(key=0)
    dependency_artifact = _artifact(key=1)
    activation = _activation()
    return {
        "plan": "P",
        "invocation": "I",
        "invocation_plan": "P",
        "graph": "G",
        "targets": [_json_ref(("local", 0))],
        "memberships": [
            {
                "invocation": "I",
                "parent": None,
                "members": [activation],
                "closed": True,
            }
        ],
        "terminals": [_terminal(activation=activation)],
        "artifacts": [evidence_artifact, dependency_artifact],
        "evidence": [
            {
                "artifact": evidence_artifact,
                "consumed": [{"kind": "artifact", "ref": dependency_artifact}],
            }
        ],
        "statuses": [_status()],
    }


def _record_cases() -> Iterator[tuple[str, JsonObject, str]]:
    statuses = (
        ("closed-unassessed", "closed", "not_assessed", True, False),
        ("closed-unmet", "closed", "unmet", True, False),
        ("pending-unknown", "pending", "unknown", False, False),
        ("closed-met-withheld", "closed", "met", True, False),
        ("closed-met-available", "closed", "met", True, True),
        ("pending-protected", "pending", "not_assessed", True, True),
        ("unmet-protected", "closed", "unmet", True, True),
    )
    for label, completion, qualification, artifact, protection in statuses:
        yield (
            "record",
            _record(
                "status",
                target=_json_ref(("local", 0)),
                completion=completion,
                qualification=qualification,
                artifact_available=artifact,
                protection_available=protection,
            ),
            label,
        )
    terminal_reasons: dict[str, list[str]] = {
        "success": [],
        "failure": ["execution_failed"],
        "cancelled": ["cancel_requested"],
        "lost": ["transport_lost"],
        "blocked": ["prerequisite"],
        "inconsistent": ["contradictory"],
    }
    for category, reasons in terminal_reasons.items():
        attempt = None if category in ("blocked", "inconsistent") else "T"
        facts = _terminal(attempt=attempt, category=category, reasons=reasons)
        yield "record", {"kind": "record", "boundary": "terminal", "facts": facts}, f"terminal-{category}"
    yield (
        "record",
        {"kind": "record", "boundary": "terminal", "facts": _terminal(reasons=["execution_failed"])},
        "success-reason",
    )
    yield (
        "record",
        {"kind": "record", "boundary": "terminal", "facts": _terminal(category="failure")},
        "failure-no-reason",
    )
    yield (
        "record",
        {
            "kind": "record",
            "boundary": "terminal",
            "facts": _terminal(attempt=None, category="failure", reasons=["execution_failed"]),
        },
        "failure-no-attempt",
    )
    yield (
        "record",
        {
            "kind": "record",
            "boundary": "terminal",
            "facts": _terminal(attempt_activation=_activation(1)),
        },
        "attempt-mismatch",
    )
    yield "record", _record("activation", **_activation()), "activation-valid"
    yield "record", _record("activation", **_activation(-1)), "activation-negative-occurrence"
    yield "record", _record("activation", **_activation(True)), "activation-bool-occurrence"
    yield "record", _record("activation", **_activation(iteration=-1)), "activation-negative-iteration"
    yield "record", _record("activation", **_activation(iteration=True)), "activation-bool-iteration"
    yield (
        "record",
        _record("activation", **_activation(1, parent=_activation(invocation="J"))),
        "activation-foreign-parent",
    )
    yield "record", _record("artifact", invocation="I", key=-1, version=1), "artifact-negative-key"
    yield "record", _record("artifact", invocation="I", key=True, version=1), "artifact-bool-key"
    yield "record", _record("absence", invocation="I", query=-1, scope_revision=1), "absence-negative-query"
    yield "record", _record("absence", invocation="I", query=True, scope_revision=1), "absence-bool-query"
    yield "record", _record("status", **_status(completion="unexpected")), "unknown-completion"
    yield "record", _record("status", **_status(qualification="unexpected")), "unknown-qualification"
    yield "record", _record("status", **{**_status(), "artifact_available": 1}), "nonbool-status"
    yield "record", _record("terminal", **_terminal(category="unexpected")), "unknown-terminal-category"
    yield (
        "record",
        _record("terminal", **_terminal(category="failure", reasons=["unexpected"])),
        "unknown-reason",
    )
    yield (
        "record",
        _record("terminal", **{**_terminal(), "category": 1}),
        "terminal-vocabulary-wrong-type",
    )
    parent = _activation(10)
    yield (
        "record",
        _record("membership", invocation="I", parent=parent, members=[parent], closed=True),
        "membership-member-equals-parent",
    )
    yield (
        "record",
        _record(
            "membership",
            invocation="I",
            parent=parent,
            members=[_activation(11, parent=_activation(12))],
            closed=True,
        ),
        "membership-parent-mismatch",
    )
    yield (
        "record",
        _record("membership", invocation="I", parent=None, members=[_activation()], closed=1),
        "membership-closed-wrong-type",
    )
    yield (
        "record",
        _record(
            "evidence",
            artifact=_artifact(invocation="I", key=0),
            consumed=[{"kind": "artifact", "ref": _artifact(invocation="J", key=1)}],
        ),
        "evidence-foreign-dependency",
    )
    yield "record", _record("artifact", invocation=1, key=0, version=1), "identity-wrong-type"
    yield "record", _record("terminal", **{**_terminal(), "reasons": [1]}), "reason-element-wrong-type"
    yield "record", _record("terminal", **{**_terminal(), "reasons": "not-a-list"}), "reasons-wrong-collection"
    yield "record", _record("status", **{**_status(), "target": "not-a-reference"}), "reference-wrong-type"
    canonical_cases: list[tuple[str, JsonObject]] = []
    facts = _canonical()
    _as_list(facts["terminals"]).append(deepcopy(_as_list(facts["terminals"])[0]))
    canonical_cases.append(("duplicate_terminal", facts))
    facts = _canonical()
    terminal = _as_object(_as_list(facts["terminals"])[0])
    replacement = _activation(9)
    terminal["activation"] = replacement
    _as_object(terminal["attempt"])["activation"] = replacement
    canonical_cases.append(("undeclared_terminal", facts))
    facts = _canonical()
    facts["artifacts"] = [_artifact(invocation="J", key=0), _artifact(invocation="J", key=1)]
    evidence = _as_object(_as_list(facts["evidence"])[0])
    _as_object(evidence["artifact"])["invocation"] = "J"
    _as_object(_as_object(_as_list(evidence["consumed"])[0])["ref"])["invocation"] = "J"
    canonical_cases.append(("foreign_invocation", facts))
    facts = _canonical()
    facts["invocation_plan"] = "Q"
    canonical_cases.append(("foreign_plan", facts))
    facts = _canonical()
    _as_list(facts["memberships"]).append(deepcopy(_as_list(facts["memberships"])[0]))
    canonical_cases.append(("duplicate_membership_parent", facts))
    facts = _canonical()
    parent = _activation(10)
    child = _activation(11, parent=parent)
    _as_list(facts["memberships"]).append(
        {
            "invocation": "I",
            "parent": parent,
            "members": [child],
            "closed": True,
        }
    )
    canonical_cases.append(("missing_parent", facts))
    facts = _canonical()
    _as_list(facts["evidence"]).append(deepcopy(_as_list(facts["evidence"])[0]))
    canonical_cases.append(("duplicate_evidence", facts))
    facts = _canonical()
    facts["artifacts"] = [_as_list(facts["artifacts"])[0]]
    canonical_cases.append(("missing_consumed_artifact", facts))
    facts = _canonical()
    facts["artifacts"] = [_as_list(facts["artifacts"])[1]]
    canonical_cases.append(("missing_evidence_artifact", facts))
    facts = _canonical()
    consumed = _as_object(_as_list(_as_object(_as_list(facts["evidence"])[0])["consumed"])[0])
    consumed["kind"] = "candidate"
    consumed["target"] = _json_ref(("foreign", 0))
    canonical_cases.append(("foreign_candidate_target", facts))
    facts = _canonical()
    consumed = _as_object(_as_list(_as_object(_as_list(facts["evidence"])[0])["consumed"])[0])
    consumed["kind"] = "candidate"
    consumed["target"] = _json_ref(("local", 2))
    canonical_cases.append(("candidate-outside-targets", facts))
    facts = _canonical()
    _as_list(facts["statuses"])[0] = _status(("foreign", 0))
    canonical_cases.append(("foreign-status-target", facts))
    facts = _canonical()
    facts["targets"] = [_json_ref(("foreign", 0))]
    facts["statuses"] = [_status(("foreign", 0))]
    canonical_cases.append(("foreign-selected-target", facts))
    for label, facts in canonical_cases:
        yield "record", {"kind": "record", "boundary": "canonical", "facts": facts}, label
    for version in (1, 0, True):
        yield "record", _record("artifact", invocation="I", key=0, version=version), f"artifact-version-{version!s}"
    for revision in (1, 0, True):
        yield (
            "record",
            _record("absence", invocation="I", query=0, scope_revision=revision),
            f"absence-revision-{revision!s}",
        )
    for available, consumed_version, label in (
        ([1], 1, "consume-v1"),
        ([1], 2, "missing-v2"),
        ([1, 2], 2, "consume-v2"),
    ):
        facts = _canonical()
        dependency = _artifact(key=1, version=consumed_version)
        facts["artifacts"] = [_artifact(key=0), *[_artifact(key=1, version=version) for version in available]]
        _as_object(_as_list(_as_object(_as_list(facts["evidence"])[0])["consumed"])[0])["ref"] = dependency
        yield "record", {"kind": "record", "boundary": "canonical", "facts": facts}, label
    facts = _canonical()
    _as_list(facts["memberships"]).append(deepcopy(_as_list(facts["memberships"])[0]))
    yield (
        "record",
        {"kind": "record", "boundary": "canonical", "facts": facts},
        "member-repeated-across-manifests-and-duplicate-parent",
    )
    facts = _canonical()
    _as_list(facts["statuses"]).append(deepcopy(_as_list(facts["statuses"])[0]))
    yield "record", {"kind": "record", "boundary": "canonical", "facts": facts}, "duplicate-status"
    facts = _canonical()
    facts["statuses"] = []
    yield "record", {"kind": "record", "boundary": "canonical", "facts": facts}, "missing-status"
    facts = _canonical()
    _as_list(facts["statuses"]).append(_status(("local", 2)))
    yield "record", {"kind": "record", "boundary": "canonical", "facts": facts}, "status-outside-targets"


def _record_precedence_cases() -> Iterator[tuple[str, JsonObject, str]]:
    rp3 = {
        "kind": "record",
        "boundary": "terminal",
        "facts": _terminal(attempt_activation=_activation(1), reasons=["execution_failed"]),
    }
    rp4 = _canonical()
    rp4["invocation_plan"] = "Q"
    _as_list(rp4["terminals"]).append(deepcopy(_as_list(rp4["terminals"])[0]))
    rp5 = _canonical()
    _as_list(rp5["evidence"]).append(deepcopy(_as_list(rp5["evidence"])[0]))
    rp5["artifacts"] = [_as_list(rp5["artifacts"])[0]]
    rp6 = _canonical()
    _as_list(rp6["statuses"]).append(_status(("local", 2)))
    dependency = _as_object(_as_list(_as_object(_as_list(rp6["evidence"])[0])["consumed"])[0])
    _as_object(dependency["ref"])["invocation"] = "J"
    _as_object(_as_object(_as_list(rp6["evidence"])[0])["artifact"])["invocation"] = "J"
    rp6["artifacts"] = [_artifact(invocation="J", key=0), _artifact(invocation="J", key=1)]
    rp7 = _canonical()
    rp7["statuses"] = []
    parent = _activation(10)
    _as_list(rp7["memberships"]).append(
        {
            "invocation": "I",
            "parent": parent,
            "members": [_activation(11, parent=parent)],
            "closed": True,
        }
    )
    terminal = _as_object(_as_list(rp7["terminals"])[0])
    replacement = _activation(9)
    terminal["activation"] = replacement
    _as_object(terminal["attempt"])["activation"] = replacement
    cases: tuple[JsonObject, ...] = (
        _record("artifact", invocation="I", key=-1, version=True),
        _record("absence", invocation="I", query=-1, scope_revision=True),
        rp3,
        {"kind": "record", "boundary": "canonical", "facts": rp4},
        {"kind": "record", "boundary": "canonical", "facts": rp5},
        {"kind": "record", "boundary": "canonical", "facts": rp6},
        {"kind": "record", "boundary": "canonical", "facts": rp7},
    )
    for index, declaration in enumerate(cases, 1):
        yield "record-precedence", declaration, f"rp{index}"
