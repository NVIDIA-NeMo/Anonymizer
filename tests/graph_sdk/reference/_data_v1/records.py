# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent data_v1 reference: records."""

from __future__ import annotations

from tests.graph_sdk.reference._data_v1.model import (
    Activation,
    JsonObject,
    JsonValue,
    Ref,
    _as_list,
    _as_object,
    _ref,
    _Reject,
)


def _string(value: JsonValue | object) -> str:
    if not isinstance(value, str):
        raise _Reject("invalid_type")
    return value


def _boolean(value: JsonValue | object) -> bool:
    if not isinstance(value, bool):
        raise _Reject("invalid_type")
    return value


def _activation_key(value: JsonObject) -> Activation:
    invocation = _string(value.get("invocation"))
    occurrence_value = value.get("occurrence")
    iteration_value = value.get("iteration")
    if isinstance(occurrence_value, bool) or not isinstance(occurrence_value, int):
        raise _Reject("invalid_type")
    if iteration_value is not None and (isinstance(iteration_value, bool) or not isinstance(iteration_value, int)):
        raise _Reject("invalid_type")
    parent_value = value.get("parent")
    parent = None if parent_value is None else _activation_key(_as_object(parent_value))
    if occurrence_value < 0 or (iteration_value is not None and iteration_value < 0):
        raise _Reject("invalid_value")
    if parent is not None and parent[0] != invocation:
        raise _Reject("foreign_owner")
    return invocation, occurrence_value, parent, iteration_value


def _artifact_key(value: JsonObject) -> tuple[str, int, int]:
    invocation = _string(value.get("invocation"))
    key_value = value.get("key")
    version_value = value.get("version")
    if isinstance(key_value, bool) or not isinstance(key_value, int):
        raise _Reject("invalid_type")
    if isinstance(version_value, bool) or not isinstance(version_value, int):
        raise _Reject("invalid_type")
    if key_value < 0 or version_value <= 0:
        raise _Reject("invalid_value")
    return invocation, key_value, version_value


def _absence_key(value: JsonObject) -> tuple[str, int, int]:
    invocation = _string(value.get("invocation"))
    query_value = value.get("query")
    revision_value = value.get("scope_revision")
    if isinstance(query_value, bool) or not isinstance(query_value, int):
        raise _Reject("invalid_type")
    if isinstance(revision_value, bool) or not isinstance(revision_value, int):
        raise _Reject("invalid_type")
    if query_value < 0 or revision_value <= 0:
        raise _Reject("invalid_value")
    return invocation, query_value, revision_value


def _status_target(value: JsonObject) -> Ref:
    target = _ref(value.get("target"))
    completion = _string(value.get("completion"))
    qualification = _string(value.get("qualification"))
    _boolean(value.get("artifact_available"))
    protection = _boolean(value.get("protection_available"))
    if completion not in ("pending", "closed") or qualification not in (
        "not_assessed",
        "met",
        "unmet",
        "unknown",
    ):
        raise _Reject("invalid_value")
    if protection and not (completion == "closed" and qualification == "met"):
        raise _Reject("contradictory")
    return target


def _consumed_ref(value: JsonObject) -> tuple[str, tuple[str, int, int], Ref | None]:
    kind = _string(value.get("kind"))
    if kind == "absence":
        return kind, _absence_key(_as_object(value.get("ref"))), None
    if kind not in ("artifact", "candidate", "decision"):
        raise _Reject("invalid_value")
    target = _ref(value.get("target")) if kind == "candidate" else None
    return kind, _artifact_key(_as_object(value.get("ref"))), target


def _evidence(value: JsonObject) -> tuple[tuple[str, int, int], list[tuple[str, int, int]], list[Ref]]:
    artifact = _artifact_key(_as_object(value.get("artifact")))
    consumed_artifacts: list[tuple[str, int, int]] = []
    candidate_targets: list[Ref] = []
    for raw_consumed in _as_list(value.get("consumed")):
        kind, dependency, target = _consumed_ref(_as_object(raw_consumed))
        if dependency[0] != artifact[0]:
            raise _Reject("foreign_owner")
        if kind != "absence":
            consumed_artifacts.append(dependency)
        if target is not None:
            candidate_targets.append(target)
    return artifact, consumed_artifacts, candidate_targets


def _membership(value: JsonObject) -> tuple[str, Activation | None, frozenset[Activation]]:
    invocation = _string(value.get("invocation"))
    parent_value = value.get("parent")
    parent = None if parent_value is None else _activation_key(_as_object(parent_value))
    _boolean(value.get("closed"))
    members = frozenset(_activation_key(_as_object(raw)) for raw in _as_list(value.get("members")))
    if parent is not None and parent[0] != invocation:
        raise _Reject("foreign_owner")
    if any(member[0] != invocation for member in members):
        raise _Reject("foreign_owner")
    if any(member == parent or member[2] != parent for member in members):
        raise _Reject("contradictory")
    return invocation, parent, members


def _terminal_key(value: JsonObject) -> tuple[str, Activation]:
    activation = _activation_key(_as_object(value.get("activation")))
    attempt = value.get("attempt")
    if attempt is not None:
        attempt_value = _as_object(attempt)
        _string(attempt_value.get("id"))
        if _activation_key(_as_object(attempt_value.get("activation"))) != activation:
            raise _Reject("foreign_owner")
    category = _string(value.get("category"))
    reasons = _as_list(value.get("reasons"))
    if not all(isinstance(reason, str) for reason in reasons):
        raise _Reject("invalid_type")
    allowed_reasons = {
        "execution_failed",
        "cancel_requested",
        "transport_lost",
        "missing",
        "duplicate",
        "foreign",
        "stale",
        "contradictory",
        "prerequisite",
    }
    if any(reason not in allowed_reasons for reason in reasons):
        raise _Reject("invalid_value")
    if category not in ("success", "failure", "cancelled", "lost", "blocked", "inconsistent"):
        raise _Reject("invalid_value")
    if (category == "success" and reasons) or (category != "success" and not reasons):
        raise _Reject("contradictory")
    if attempt is None and category not in ("blocked", "inconsistent"):
        raise _Reject("contradictory")
    return activation[0], activation


def _validate_canonical(facts: JsonObject) -> None:
    plan = _string(facts.get("plan"))
    invocation = _string(facts.get("invocation"))
    invocation_plan = _string(facts.get("invocation_plan"))
    _string(facts.get("graph"))
    targets = [_ref(value) for value in _as_list(facts.get("targets"))]
    memberships = [_as_object(value) for value in _as_list(facts.get("memberships"))]
    terminals = [_as_object(value) for value in _as_list(facts.get("terminals"))]
    artifacts = [_as_object(value) for value in _as_list(facts.get("artifacts"))]
    evidence = [_as_object(value) for value in _as_list(facts.get("evidence"))]
    statuses = [_as_object(value) for value in _as_list(facts.get("statuses"))]

    artifact_keys = [_artifact_key(item) for item in artifacts]
    terminal_keys = [_terminal_key(item) for item in terminals]
    status_targets = [_status_target(item) for item in statuses]
    parsed_memberships = [_membership(item) for item in memberships]
    parents = [parent for _, parent, _ in parsed_memberships]
    members = [member for _, _, manifest in parsed_memberships for member in manifest]
    parsed_evidence = [_evidence(item) for item in evidence]
    evidence_artifacts = [artifact for artifact, _, _ in parsed_evidence]
    consumed_artifacts = [item for _, consumed, _ in parsed_evidence for item in consumed]
    candidate_targets = [target for _, _, candidates in parsed_evidence for target in candidates]
    nested_invocations = [key[0] for key in (*artifact_keys, *terminal_keys, *evidence_artifacts, *consumed_artifacts)]
    nested_invocations.extend(member[0] for member in members)
    nested_invocations.extend(owner for owner, _, _ in parsed_memberships)

    target_set = set(targets)
    if any(target[0] == "local" and target not in target_set for target in (*status_targets, *candidate_targets)):
        raise _Reject("invalid_value")
    if invocation_plan != plan or any(value != invocation for value in nested_invocations):
        raise _Reject("foreign_owner")
    if any(target[0] != "local" for target in (*targets, *status_targets, *candidate_targets)):
        raise _Reject("foreign_owner")
    if len(parents) != len(set(parents)) or len(members) != len(set(members)):
        raise _Reject("duplicate")
    if len(terminal_keys) != len({activation for _, activation in terminal_keys}):
        raise _Reject("duplicate")
    if len(evidence_artifacts) != len(set(evidence_artifacts)) or len(status_targets) != len(set(status_targets)):
        raise _Reject("duplicate")
    member_set = set(members)
    if any(parent is not None and parent not in member_set for parent in parents):
        raise _Reject("missing")
    if any(activation not in member_set for _, activation in terminal_keys):
        raise _Reject("missing")
    artifact_set = set(artifact_keys)
    if any(item not in artifact_set for item in (*evidence_artifacts, *consumed_artifacts)):
        raise _Reject("missing")
    if any(target not in set(status_targets) for target in targets):
        raise _Reject("missing")
