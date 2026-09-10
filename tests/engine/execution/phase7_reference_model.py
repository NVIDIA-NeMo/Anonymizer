# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure Phase 7 oracle with no production, pandas, or DataDesigner imports."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from itertools import combinations
from typing import cast

REFERENCE_MODEL_VERSION = "phase7-reference-model/v1"
GENERATOR_VERSION = "phase7-finite-envelope/v1"
MAX_EXOGENOUS_OBSERVATIONS = 16
EVENT_ALPHABET = (
    "dispatch_accepted",
    "dispatch_rejected",
    "candidate_rows",
    "failed_record",
    "backend_exception",
    "cancellation",
    "trusted_stop",
    "loss",
    "transform",
    "verify",
    "finalize",
    "cleanup",
    "immutable_accept",
    "teardown",
    "release",
)
INDEPENDENCE_RULES = (
    "different_scopes",
    "different_datums",
    "different_groups",
    "disjoint_scope_and_datum",
    "disjoint_scope_and_group",
    "disjoint_datum_and_group",
)
OWNER_CORPUS_VERSION = "anonymizer-phase7-owner-contract-corpus/v1"
OWNER_CASE_IDS = (
    "valid_empty_scope_zero_dispatch",
    "valid_single_given_name",
    "valid_given_family_email_relation",
    "valid_phone_source_mask",
    "unknown_contract_version",
    "contract_digest_mismatch",
    "missing_detector_disposition",
    "unknown_role",
    "unknown_relation",
    "unknown_mask",
    "unsupported_detector_label",
    "selector_resolves_zero_slots",
    "selector_resolves_multiple_slots",
    "relation_crosses_scopes",
    "email_relation_wrong_roles",
    "distinct_slots_same_canonical_value",
    "candidate_matches_own_original",
    "candidate_matches_other_slot_original",
    "email_local_part_omits_name",
    "count_limits_exact",
    "count_limits_one_over",
    "byte_limits_exact",
    "byte_limits_one_over",
    "runtime_capability_missing",
    "trusted_task_failure",
    "unattributable_failure",
    "cleanup_attestation_verified",
    "cleanup_attestation_missing",
    "cleanup_attestation_contradictory",
    "redact_policy_role_bearing_scope",
)
ROLE_BY_LABEL = {
    "email": "email_address",
    "fax_number": "fax_number",
    "first_name": "person_given_name",
    "last_name": "person_family_name",
    "phone_number": "voice_phone_number",
    "user_name": "user_name",
}
ROLE_CONTRACT = {
    "email_address": ("email_addr_spec_ascii/v1", "none/v1"),
    "fax_number": ("telephone_ascii/v1", "digit_literal/v1"),
    "person_family_name": ("unicode_person_name/v1", "none/v1"),
    "person_given_name": ("unicode_person_name/v1", "none/v1"),
    "user_name": ("username_ascii/v1", "none/v1"),
    "voice_phone_number": ("telephone_ascii/v1", "digit_literal/v1"),
}
SLOT_LABELS = ("first_name", "last_name", "email", "phone_number")
SLOT_SOURCES = ("Alice", "Adams", "alice@example.com", "555-0100")
SLOT_CANDIDATES = ("Nova", "Vale", "nova.vale@example.test", "555-0199")


class ReferencePolicy(str, Enum):
    CURRENT_EMPTY = "phase6-redact-empty/v1"
    FUTURE_V1 = "phase6-substitute-role-policy/v1"


class ReferenceEventKind(str, Enum):
    DISPATCH_ACCEPTED = "dispatch_accepted"
    DISPATCH_REJECTED = "dispatch_rejected"
    CANDIDATE_ROWS = "candidate_rows"
    FAILED_RECORD = "failed_record"
    BACKEND_EXCEPTION = "backend_exception"
    CANCELLATION = "cancellation"
    TRUSTED_STOP = "trusted_stop"
    LOSS = "loss"
    TRANSFORM = "transform"
    VERIFY = "verify"
    FINALIZE = "finalize"
    CLEANUP = "cleanup"
    IMMUTABLE_ACCEPT = "immutable_accept"
    TEARDOWN = "teardown"
    RELEASE = "release"


@dataclass(frozen=True, slots=True)
class ReferenceDatum:
    id: str
    text: str


@dataclass(frozen=True, slots=True)
class ReferenceScope:
    members: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReferenceMention:
    datum: str
    start: int
    end: int
    source: str
    label: str
    cluster: str


@dataclass(frozen=True, slots=True)
class ReferenceSelector:
    cluster: str
    role: str


@dataclass(frozen=True, slots=True)
class ReferenceRelation:
    version: str
    upstream: tuple[ReferenceSelector, ...]
    downstream: ReferenceSelector


@dataclass(frozen=True, slots=True)
class ReferenceDeclaration:
    datums: tuple[ReferenceDatum, ...]
    scopes: tuple[ReferenceScope, ...]
    mentions: tuple[ReferenceMention, ...] = ()
    relations: tuple[ReferenceRelation, ...] = ()
    dependencies: tuple[tuple[str, str], ...] = ()
    groups: tuple[tuple[str, ...], ...] = ()
    policy: ReferencePolicy = ReferencePolicy.FUTURE_V1
    capability: bool = True
    contract_version: str = "anonymizer-phase7-stable-substitute/v1"
    contract_digest_valid: bool = True
    detector_universe_complete: bool = True
    declared_role: str | None = None
    declared_mask: str | None = None


@dataclass(frozen=True, slots=True)
class ReferenceEvent:
    kind: ReferenceEventKind
    subject_kind: str = "invocation"
    subject: int = 0
    attempt: int = 0
    assignments: tuple[tuple[str, str], ...] = ()
    outcome: str = "accepted"


@dataclass(frozen=True, slots=True)
class ReferenceCase:
    name: str
    declaration: ReferenceDeclaration
    events: tuple[ReferenceEvent, ...] = field(default_factory=tuple)
    owner_case: str | None = None


@dataclass(frozen=True, slots=True)
class ReferenceSlot:
    key: str
    cluster: str
    role: str
    format: str
    mask: str
    mention_indexes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ReferenceManifest:
    members: tuple[str, ...]
    slots: tuple[ReferenceSlot, ...]
    required_pairs: tuple[tuple[str, str], ...]
    relations: tuple[tuple[str, tuple[str, ...], str], ...]


@dataclass(frozen=True, slots=True)
class ReferenceResult:
    admission: str
    manifests: tuple[ReferenceManifest, ...]
    scope_outcomes: tuple[str, ...]
    reason_codes: tuple[str | None, ...]
    bundles: tuple[tuple[tuple[str, str], ...], ...]
    task_outcomes: tuple[tuple[str, str, str], ...]
    outputs: tuple[tuple[str, str], ...]
    released_groups: tuple[int, ...]
    released_datums: tuple[str, ...]
    invocation: str
    cleanup: str
    immutable_result: bool
    dispatch_count: int
    attempt_count: int
    event_count: int
    max_event_count: int


def events_commute(
    left: ReferenceEvent,
    right: ReferenceEvent,
    declaration: ReferenceDeclaration | None = None,
) -> bool:
    """Return whether two adjacent exogenous observations are independent."""
    if left.subject_kind == "invocation" or right.subject_kind == "invocation":
        return False
    if left.subject_kind == right.subject_kind:
        return left.subject != right.subject
    if declaration is None:
        return True
    scope_members = tuple(set(scope.members) for scope in declaration.scopes)
    scoped_datums = {("scope", scope_index): members for scope_index, members in enumerate(scope_members)}
    group_members = {("group", group_index): set(group) for group_index, group in enumerate(declaration.groups)}

    def affected(event: ReferenceEvent) -> set[str]:
        if event.subject_kind == "scope":
            return scoped_datums.get(("scope", event.subject), set())
        if event.subject_kind == "group":
            return group_members.get(("group", event.subject), set())
        if event.subject_kind == "datum" and event.subject < len(declaration.datums):
            return {declaration.datums[event.subject].id}
        return set()

    return affected(left).isdisjoint(affected(right))


def canonical_events(
    events: tuple[ReferenceEvent, ...],
    declaration: ReferenceDeclaration | None = None,
) -> tuple[ReferenceEvent, ...]:
    """Order only adjacent commuting observations by opaque declaration position."""
    rank = {value: index for index, value in enumerate(EVENT_ALPHABET)}
    canonical = list(events)
    changed = True
    while changed:
        changed = False
        for index in range(len(canonical) - 1):
            left = canonical[index]
            right = canonical[index + 1]
            left_key = (left.subject_kind, left.subject, rank[left.kind.value], left.attempt, left.outcome)
            right_key = (right.subject_kind, right.subject, rank[right.kind.value], right.attempt, right.outcome)
            if events_commute(left, right, declaration) and left_key > right_key:
                canonical[index], canonical[index + 1] = right, left
                changed = True
    return tuple(canonical)


def reduce_reference(case: ReferenceCase) -> ReferenceResult:
    """Derive the bounded Phase 7 result without accepting a production verdict."""
    if len(case.events) > MAX_EXOGENOUS_OBSERVATIONS:
        raise AssertionError("trace exceeds the frozen 16-observation bound")
    if any(not isinstance(event, ReferenceEvent) for event in case.events):
        raise AssertionError("trace contains an invalid observation")
    compiled = _compile_reference(case.declaration)
    if isinstance(compiled, str):
        return ReferenceResult(
            compiled,
            (),
            (),
            (),
            (),
            (),
            (),
            (),
            (),
            "not_opened",
            "not_entered",
            False,
            0,
            0,
            len(case.events),
            MAX_EXOGENOUS_OBSERVATIONS,
        )
    manifests, mention_slots = compiled
    groups = case.declaration.groups or tuple((datum.id,) for datum in case.declaration.datums)
    scope_outcomes = [
        "planned"
        if not manifest.slots
        else "blocked"
        if case.declaration.policy is ReferencePolicy.CURRENT_EMPTY
        else "reserved"
        for manifest in manifests
    ]
    reasons: list[str | None] = ["prerequisite_blocked" if outcome == "blocked" else None for outcome in scope_outcomes]
    bundles: list[tuple[tuple[str, str], ...]] = [() for _manifest in manifests]
    dispatched = [False for _manifest in manifests]
    attempts = [-1 for _manifest in manifests]
    transformed: dict[str, str] = {}
    transform_failed: set[str] = set()
    verified_groups: set[int] = set()
    dispatch_count = 0
    attempt_count = 0
    cancellation = False
    cancellation_after_release = False
    global_inconsistent = False
    loss = False
    finalized = False
    cleanup = "unconfirmed"
    immutable_result = False
    release_observed = False
    teardown = "unconfirmed"

    for event in case.events:
        if event.kind is ReferenceEventKind.CANCELLATION:
            if release_observed and immutable_result:
                cancellation_after_release = True
                continue
            cancellation = True
            for index, outcome in enumerate(scope_outcomes):
                if outcome == "reserved":
                    scope_outcomes[index] = "cancelling" if dispatched[index] else "cancelled"
            continue
        if event.kind in {
            ReferenceEventKind.DISPATCH_ACCEPTED,
            ReferenceEventKind.DISPATCH_REJECTED,
            ReferenceEventKind.CANDIDATE_ROWS,
            ReferenceEventKind.FAILED_RECORD,
            ReferenceEventKind.BACKEND_EXCEPTION,
            ReferenceEventKind.TRUSTED_STOP,
            ReferenceEventKind.LOSS,
        }:
            if event.subject_kind != "scope" or not 0 <= event.subject < len(manifests):
                global_inconsistent = True
                continue
            scope_index = event.subject
            state = scope_outcomes[scope_index]
            if event.kind is ReferenceEventKind.DISPATCH_ACCEPTED:
                if state == "reserved" and not dispatched[scope_index]:
                    dispatched[scope_index] = True
                    attempts[scope_index] = event.attempt
                    dispatch_count += 1
                    attempt_count += 1
                else:
                    global_inconsistent = True
            elif event.kind is ReferenceEventKind.DISPATCH_REJECTED:
                if state == "reserved":
                    scope_outcomes[scope_index] = "failed"
                    reasons[scope_index] = "backend_failed"
                elif state != "planned":
                    global_inconsistent = True
            elif event.kind is ReferenceEventKind.CANDIDATE_ROWS:
                if state == "planned":
                    continue
                if state != "reserved" or not dispatched[scope_index] or event.attempt != attempts[scope_index]:
                    if state in {"cancelling", "cancelled", "lost", "failed", "inconsistent", "blocked"}:
                        continue
                    scope_outcomes[scope_index] = "inconsistent"
                    reasons[scope_index] = "evidence_unattributable"
                    global_inconsistent = True
                    continue
                validated = _validate_candidate(
                    case.declaration,
                    manifests[scope_index],
                    event.assignments,
                    mention_slots,
                )
                if isinstance(validated, str):
                    reasons[scope_index] = validated
                    if validated in {"partial_bundle", "duplicate_slot", "foreign_slot"}:
                        scope_outcomes[scope_index] = "inconsistent"
                        global_inconsistent = True
                    else:
                        scope_outcomes[scope_index] = "failed"
                else:
                    scope_outcomes[scope_index] = "planned"
                    bundles[scope_index] = validated
            elif event.kind is ReferenceEventKind.FAILED_RECORD:
                if state == "planned":
                    continue
                if (
                    state == "reserved"
                    and dispatched[scope_index]
                    and event.attempt == attempts[scope_index]
                    and event.outcome == "attributed"
                ):
                    scope_outcomes[scope_index] = "failed"
                    reasons[scope_index] = "backend_failed"
                else:
                    scope_outcomes[scope_index] = "inconsistent"
                    reasons[scope_index] = "evidence_unattributable"
                    global_inconsistent = True
            elif event.kind is ReferenceEventKind.BACKEND_EXCEPTION:
                if state == "reserved" and dispatched[scope_index] and event.attempt == attempts[scope_index]:
                    scope_outcomes[scope_index] = "failed"
                    reasons[scope_index] = "backend_failed"
                elif state != "planned":
                    global_inconsistent = True
            elif event.kind is ReferenceEventKind.TRUSTED_STOP:
                if state in {"reserved", "cancelling"} and dispatched[scope_index]:
                    scope_outcomes[scope_index] = "cancelled"
                elif state != "planned":
                    global_inconsistent = True
            elif event.kind is ReferenceEventKind.LOSS:
                if state in {"reserved", "cancelling"} and dispatched[scope_index]:
                    scope_outcomes[scope_index] = "lost"
                    reasons[scope_index] = "transport_lost"
                    loss = True
                elif state != "planned":
                    global_inconsistent = True
            continue
        if event.kind is ReferenceEventKind.TRANSFORM:
            if event.subject_kind != "datum" or not 0 <= event.subject < len(case.declaration.datums):
                global_inconsistent = True
                continue
            datum = case.declaration.datums[event.subject]
            if event.outcome != "accepted":
                transform_failed.add(datum.id)
                continue
            output = _apply_anchored(case.declaration, manifests, bundles, mention_slots, datum.id)
            if output is None:
                transform_failed.add(datum.id)
            else:
                transformed[datum.id] = output
        elif event.kind is ReferenceEventKind.VERIFY:
            if event.subject_kind != "group" or not 0 <= event.subject < len(groups):
                global_inconsistent = True
            elif event.outcome == "accepted":
                verified_groups.add(event.subject)
        elif event.kind is ReferenceEventKind.FINALIZE:
            finalized = event.outcome == "accepted"
            global_inconsistent = global_inconsistent or not finalized
        elif event.kind is ReferenceEventKind.CLEANUP:
            cleanup = event.outcome
            global_inconsistent = global_inconsistent or cleanup != "verified"
        elif event.kind is ReferenceEventKind.IMMUTABLE_ACCEPT:
            immutable_result = (
                event.outcome == "accepted"
                and all(
                    outcome in {"planned", "blocked", "failed", "cancelled", "lost", "inconsistent"}
                    for outcome in scope_outcomes
                )
                and finalized
                and cleanup == "verified"
                and not global_inconsistent
                and not cancellation
                and not loss
            )
        elif event.kind is ReferenceEventKind.TEARDOWN:
            teardown = event.outcome
            if teardown == "failed" and not immutable_result:
                global_inconsistent = True
        elif event.kind is ReferenceEventKind.RELEASE:
            release_observed = True

    for index, outcome in enumerate(scope_outcomes):
        if outcome in {"reserved", "cancelling"}:
            scope_outcomes[index] = "lost" if dispatched[index] else "blocked"
            reasons[index] = "transport_lost" if dispatched[index] else "prerequisite_blocked"
            loss = loss or dispatched[index]

    eligible = _eligible_datums(
        case.declaration,
        manifests,
        tuple(scope_outcomes),
        transformed,
        transform_failed,
    )
    while True:
        before = set(eligible)
        for group in groups:
            if not set(group).issubset(eligible):
                eligible.difference_update(group)
        for prerequisite, dependent in case.declaration.dependencies:
            if prerequisite not in eligible:
                eligible.discard(dependent)
        if eligible == before:
            break
    legal_groups = tuple(
        index for index, group in enumerate(groups) if set(group).issubset(eligible) and index in verified_groups
    )
    released_groups = legal_groups if immutable_result and release_observed else ()
    released_datums = tuple(datum_id for group_index in released_groups for datum_id in groups[group_index])
    outputs = tuple(
        (datum.id, transformed.get(datum.id, datum.text)) for datum in case.declaration.datums if datum.id in eligible
    )
    task_outcomes = tuple(
        ("scope", str(index), _phase4_scope_outcome(outcome)) for index, outcome in enumerate(scope_outcomes)
    ) + tuple(
        (
            "datum",
            datum.id,
            "succeeded" if datum.id in eligible else "failed" if datum.id in transform_failed else "blocked",
        )
        for datum in case.declaration.datums
    )
    if immutable_result:
        invocation = "completed"
    elif global_inconsistent or cleanup != "verified":
        invocation = "inconsistent"
    elif loss:
        invocation = "lost"
    elif cancellation:
        invocation = "cancelled"
    elif any(outcome == "failed" for outcome in scope_outcomes):
        invocation = "failed"
    else:
        invocation = "completed"
    if cancellation_after_release:
        invocation = "completed"
    return ReferenceResult(
        "admitted",
        manifests,
        tuple(scope_outcomes),
        tuple(reasons),
        tuple(bundles),
        task_outcomes,
        outputs,
        released_groups,
        released_datums,
        invocation,
        cleanup,
        immutable_result,
        dispatch_count,
        attempt_count,
        len(case.events),
        MAX_EXOGENOUS_OBSERVATIONS,
    )


def _compile_reference(
    declaration: ReferenceDeclaration,
) -> tuple[tuple[ReferenceManifest, ...], tuple[str, ...]] | str:
    if declaration.contract_version != "anonymizer-phase7-stable-substitute/v1":
        return "contract_invalid"
    if not declaration.contract_digest_valid:
        return "digest_mismatch"
    if not declaration.detector_universe_complete:
        return "detector_universe_incomplete"
    if declaration.declared_role is not None and declaration.declared_role not in ROLE_CONTRACT:
        return "unsupported_role"
    if declaration.declared_mask is not None and declaration.declared_mask not in {"none/v1", "digit_literal/v1"}:
        return "unsupported_mask"
    datum_ids = tuple(datum.id for datum in declaration.datums)
    if (
        len(datum_ids) > 4
        or len(declaration.scopes) > 2
        or len(declaration.mentions) > 6
        or len({mention.cluster for mention in declaration.mentions}) > 3
        or any(len(datum.text.encode("utf-8")) > 1536 for datum in declaration.datums)
        or any(len(mention.source.encode("utf-8")) > 256 for mention in declaration.mentions)
    ):
        return "limit_exceeded"
    scopes = declaration.scopes
    if any(not scope.members for scope in scopes):
        return "empty_scope"
    normalized_scopes = tuple(tuple(sorted(scope.members)) for scope in scopes)
    if len(set(normalized_scopes)) != len(normalized_scopes):
        return "duplicate_scope"
    if any(len(set(scope.members)) != len(scope.members) for scope in scopes):
        return "duplicate_scope_member"
    members = tuple(member for scope in scopes for member in scope.members)
    if any(member not in set(datum_ids) for member in members):
        return "unknown_scope_datum"
    if set(members) != set(datum_ids):
        return "scope_coverage_gap"
    if len(members) != len(set(members)):
        scope_sets = tuple(set(scope.members) for scope in scopes)
        if any(left < right or right < left for left, right in combinations(scope_sets, 2)):
            return "unsupported_scope_nesting"
        return "scope_overlap"
    groups = declaration.groups or tuple((datum_id,) for datum_id in datum_ids)
    group_members = tuple(member for group in groups for member in group)
    if set(group_members) != set(datum_ids) or len(group_members) != len(set(group_members)):
        return "malformed_graph"
    if any(
        prerequisite not in set(datum_ids) or dependent not in set(datum_ids) or prerequisite == dependent
        for prerequisite, dependent in declaration.dependencies
    ):
        return "malformed_graph"
    datum_by_id = {datum.id: datum for datum in declaration.datums}
    scope_by_datum = {datum_id: scope_index for scope_index, scope in enumerate(scopes) for datum_id in scope.members}
    structural_slots: dict[tuple[object, ...], list[int]] = {}
    for mention_index, mention in enumerate(declaration.mentions):
        datum = datum_by_id.get(mention.datum)
        if datum is None:
            return "phase6_handoff_mismatch"
        if (
            mention.start < 0
            or mention.end <= mention.start
            or mention.end > len(datum.text)
            or datum.text[mention.start : mention.end] != mention.source
        ):
            return "phase6_handoff_mismatch"
        role = ROLE_BY_LABEL.get(mention.label)
        if role is None:
            return "unsupported_label"
        scope_index = scope_by_datum[mention.datum]
        structural_key = (scope_index, mention.cluster, role)
        structural_slots.setdefault(structural_key, []).append(mention_index)
    if len(structural_slots) > 4:
        return "limit_exceeded"
    slots_by_scope: list[list[ReferenceSlot]] = [[] for _scope in scopes]
    mention_slots = ["" for _mention in declaration.mentions]
    for scope_index in range(len(scopes)):
        keys = tuple(key for key in structural_slots if key[0] == scope_index)
        for slot_index, key in enumerate(keys):
            _scope, cluster, role = key
            mention_indexes = tuple(structural_slots[key])
            slot_key = f"slot-{scope_index}-{slot_index}"
            format_name, mask = ROLE_CONTRACT[cast(str, role)]
            slot = ReferenceSlot(
                slot_key,
                cast(str, cluster),
                cast(str, role),
                format_name,
                mask,
                mention_indexes,
            )
            slots_by_scope[scope_index].append(slot)
            for mention_index in mention_indexes:
                mention_slots[mention_index] = slot_key
    manifests: list[ReferenceManifest] = []
    for scope_index, scope in enumerate(scopes):
        slots = tuple(slots_by_scope[scope_index])
        required_pairs = tuple((left.key, right.key) for left, right in combinations(slots, 2))
        if len(required_pairs) > 6:
            return "limit_exceeded"
        manifests.append(ReferenceManifest(tuple(sorted(scope.members)), slots, required_pairs, ()))
    compiled_relations: list[list[tuple[str, tuple[str, ...], str]]] = [[] for _scope in scopes]
    all_slots = tuple(slot for slots in slots_by_scope for slot in slots)
    for relation in declaration.relations:
        if relation.version != "email_from_name/v1":
            return "unsupported_constraint"
        if len(set(relation.upstream)) != len(relation.upstream):
            return "selector_ambiguous"

        def resolve(selector: ReferenceSelector) -> tuple[ReferenceSlot, ...]:
            return tuple(slot for slot in all_slots if slot.cluster == selector.cluster and slot.role == selector.role)

        upstream_matches = tuple(resolve(selector) for selector in relation.upstream)
        downstream_matches = resolve(relation.downstream)
        if any(not matches for matches in upstream_matches) or not downstream_matches:
            return "selector_missing"
        if any(len(matches) != 1 for matches in upstream_matches) or len(downstream_matches) != 1:
            return "selector_ambiguous"
        upstream = tuple(matches[0] for matches in upstream_matches)
        downstream = downstream_matches[0]
        relation_scopes = {int(slot.key.split("-", maxsplit=2)[1]) for slot in (*upstream, downstream)}
        if len(relation_scopes) != 1:
            return "cross_scope_relation"
        if (
            not 1 <= len(upstream) <= 2
            or any(slot.role not in {"person_given_name", "person_family_name"} for slot in upstream)
            or downstream.role != "email_address"
        ):
            return "relation_role_mismatch"
        scope_index = relation_scopes.pop()
        compiled_relations[scope_index].append((relation.version, tuple(slot.key for slot in upstream), downstream.key))
    manifests = [
        replace(manifest, relations=tuple(compiled_relations[index])) for index, manifest in enumerate(manifests)
    ]
    if not declaration.capability:
        return "missing_capability"
    return tuple(manifests), tuple(mention_slots)


def _validate_candidate(
    declaration: ReferenceDeclaration,
    manifest: ReferenceManifest,
    assignments: tuple[tuple[str, str], ...],
    mention_slots: tuple[str, ...],
) -> tuple[tuple[str, str], ...] | str:
    expected = {slot.key for slot in manifest.slots}
    keys = tuple(key for key, _value in assignments)
    if len(keys) != len(set(keys)):
        return "duplicate_slot"
    if set(keys) != expected:
        return "foreign_slot" if set(keys) - expected else "partial_bundle"
    assignment_by_slot = dict(assignments)
    canonical: dict[str, str] = {}
    originals = tuple(
        mention.source
        for mention_index, mention in enumerate(declaration.mentions)
        if mention_slots[mention_index] in expected
    )
    original_skeletons = {_canonical_value(original) for original in originals}
    for slot in manifest.slots:
        value = assignment_by_slot[slot.key]
        skeleton = _canonical_value(value)
        if not value or not skeleton or skeleton in original_skeletons:
            return "candidate_matches_original"
        if len(value.encode("utf-8")) > 256 or not _format_valid(slot.format, value):
            return "unsupported_role"
        if slot.mask == "digit_literal/v1" and any(
            not _digit_mask_valid(declaration.mentions[index].source, value) for index in slot.mention_indexes
        ):
            return "relation_failed"
        canonical[slot.key] = skeleton
    for left, right in manifest.required_pairs:
        if canonical[left] == canonical[right]:
            return "canonical_collision"
    for version, upstream, downstream in manifest.relations:
        if version != "email_from_name/v1":
            return "unsupported_constraint"
        local = assignment_by_slot[downstream].split("@", maxsplit=1)[0]
        local_skeleton = _canonical_value(local)
        if not any(canonical[slot_key] in local_skeleton for slot_key in upstream):
            return "relation_failed"
    return tuple(sorted(assignments))


def _canonical_value(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip().casefold()
    return "".join(character for character in normalized if unicodedata.category(character)[0] in {"L", "N"})


def _format_valid(format_name: str, value: str) -> bool:
    if format_name == "unicode_person_name/v1":
        normalized = unicodedata.normalize("NFKC", value)
        return (
            1 <= len(normalized.encode("utf-8")) <= 128
            and any(unicodedata.category(character).startswith("L") for character in normalized)
            and all(
                unicodedata.category(character)[:1] in {"L", "M"}
                or unicodedata.category(character) == "Zs"
                or character in "'.-"
                for character in normalized
            )
        )
    if format_name == "username_ascii/v1":
        return bool(re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,62}[A-Za-z0-9])?", value))
    if format_name == "telephone_ascii/v1":
        return (
            bool(re.fullmatch(r"[0-9 ()+.-]+", value))
            and 7 <= sum(character.isascii() and character.isdigit() for character in value) <= 15
            and value.count("+") <= 1
            and ("+" not in value or value.startswith("+"))
        )
    if format_name == "email_addr_spec_ascii/v1":
        if len(value.encode("utf-8")) > 254 or value.count("@") != 1 or not value.isascii():
            return False
        local, domain = value.split("@")
        labels = domain.split(".")
        return (
            1 <= len(local) <= 64
            and not local.startswith(".")
            and not local.endswith(".")
            and ".." not in local
            and bool(re.fullmatch(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+", local))
            and len(labels) >= 2
            and all(
                1 <= len(label) <= 63 and bool(re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?", label))
                for label in labels
            )
            and 2 <= len(labels[-1]) <= 63
            and labels[-1].isalpha()
        )
    return False


def _digit_mask_valid(source: str, candidate: str) -> bool:
    normalized_source = unicodedata.normalize("NFKC", source)
    normalized_candidate = unicodedata.normalize("NFKC", candidate)
    return len(normalized_source) == len(normalized_candidate) and all(
        (candidate_character.isascii() and candidate_character.isdigit())
        if source_character.isascii() and source_character.isdigit()
        else source_character == candidate_character
        for source_character, candidate_character in zip(normalized_source, normalized_candidate, strict=True)
    )


def _apply_anchored(
    declaration: ReferenceDeclaration,
    manifests: tuple[ReferenceManifest, ...],
    bundles: list[tuple[tuple[str, str], ...]],
    mention_slots: tuple[str, ...],
    datum_id: str,
) -> str | None:
    datum = next(datum for datum in declaration.datums if datum.id == datum_id)
    assignment_by_slot = {key: value for manifest_bundle in bundles for key, value in manifest_bundle}
    mentions = tuple(
        (mention, mention_slots[index])
        for index, mention in enumerate(declaration.mentions)
        if mention.datum == datum_id
    )
    if not mentions:
        return datum.text
    if any(slot_key not in assignment_by_slot for _mention, slot_key in mentions):
        return None
    ordered = sorted(mentions, key=lambda item: item[0].start)
    if any(left[0].end > right[0].start for left, right in zip(ordered, ordered[1:], strict=False)):
        return None
    output = datum.text
    for mention, slot_key in reversed(ordered):
        if output[mention.start : mention.end] != mention.source:
            return None
        output = output[: mention.start] + assignment_by_slot[slot_key] + output[mention.end :]
    return output


def _eligible_datums(
    declaration: ReferenceDeclaration,
    manifests: tuple[ReferenceManifest, ...],
    scope_outcomes: tuple[str, ...],
    transformed: dict[str, str],
    transform_failed: set[str],
) -> set[str]:
    scope_by_datum = {
        datum_id: scope_index for scope_index, manifest in enumerate(manifests) for datum_id in manifest.members
    }
    mentioned = {mention.datum for mention in declaration.mentions}
    return {
        datum.id
        for datum in declaration.datums
        if scope_outcomes[scope_by_datum[datum.id]] == "planned"
        and datum.id not in transform_failed
        and (datum.id not in mentioned or datum.id in transformed)
    }


def _phase4_scope_outcome(outcome: str) -> str:
    return {
        "planned": "succeeded",
        "blocked": "blocked",
        "failed": "failed",
        "cancelled": "cancelled",
        "lost": "lost",
        "inconsistent": "inconsistent",
    }[outcome]


def finite_reference_cases() -> tuple[ReferenceCase, ...]:
    """Return the frozen governed corpus, not an unbounded Cartesian product."""
    return (
        *_owner_contract_cases(),
        *_slot_envelope_cases(),
        *_admission_cases(),
        *_lifecycle_cases(),
    )


def case_by_name(name: str) -> ReferenceCase:
    return next(case for case in finite_reference_cases() if case.name == name)


def corpus_document() -> dict[str, object]:
    cases = tuple(sorted(finite_reference_cases(), key=lambda case: case.name))
    return {
        "cases": [_canonical_case_record(case) for case in cases],
        "generator_version": GENERATOR_VERSION,
        "reference_model_version": REFERENCE_MODEL_VERSION,
        "schema_version": "phase7-reference-corpus/v1",
    }


def _canonical_case_record(case: ReferenceCase) -> dict[str, object]:
    canonical = replace(case, events=canonical_events(case.events, case.declaration))
    return {"case": asdict(canonical), "result": asdict(reduce_reference(canonical))}


def canonical_corpus_bytes() -> bytes:
    """Serialize the complete corpus as compact sorted-key UTF-8 JSON."""
    return json.dumps(
        corpus_document(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def reference_manifest() -> dict[str, object]:
    cases = finite_reference_cases()
    graph_payloads = {
        json.dumps(asdict(case.declaration), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for case in cases
    }
    trace_payloads = {
        json.dumps(
            {
                "declaration": asdict(case.declaration),
                "events": [asdict(event) for event in canonical_events(case.events, case.declaration)],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for case in cases
    }
    slot_count_cases = {
        str(slot_count): sum(
            len(result.manifests) == 1 and sum(len(manifest.slots) for manifest in result.manifests) == slot_count
            for result in (reduce_reference(case) for case in cases)
        )
        for slot_count in range(5)
    }
    return {
        "actual_event_count": sum(len(case.events) for case in cases),
        "canonical_serialization": "UTF8_compact_sorted_key_JSON_complete_corpus_no_trailing_newline",
        "canonical_trace_count": len(trace_payloads),
        "case_count": len(cases),
        "digest": hashlib.sha256(canonical_corpus_bytes()).hexdigest(),
        "event_alphabet": list(EVENT_ALPHABET),
        "generator_version": GENERATOR_VERSION,
        "graph_count": len(graph_payloads),
        "independence_rules": list(INDEPENDENCE_RULES),
        "max_exogenous_observations": MAX_EXOGENOUS_OBSERVATIONS,
        "owner_case_count": sum(case.owner_case is not None for case in cases),
        "owner_corpus_version": OWNER_CORPUS_VERSION,
        "reference_model_version": REFERENCE_MODEL_VERSION,
        "slot_count_case_counts": slot_count_cases,
    }


def owner_case_outcome(case: ReferenceCase) -> str:
    """Project a derived result into the frozen owner's 30-case vocabulary."""
    result = reduce_reference(case)
    if result.admission != "admitted":
        return result.admission
    if any(
        reason in {"canonical_collision", "candidate_matches_original", "relation_failed"}
        for reason in result.reason_codes
    ):
        return next(reason for reason in result.reason_codes if reason is not None)
    if result.invocation == "inconsistent":
        return "inconsistent_global_embargo"
    if result.scope_outcomes and all(outcome == "blocked" for outcome in result.scope_outcomes):
        return "blocked_zero_effects"
    if any(outcome == "failed" for outcome in result.scope_outcomes):
        return "failed"
    if result.manifests and all(not manifest.slots for manifest in result.manifests):
        return "planned_empty"
    if result.released_groups:
        return "release_eligible" if case.owner_case == "cleanup_attestation_verified" else "planned"
    return "planned"


def _owner_contract_cases() -> tuple[ReferenceCase, ...]:
    empty = _declaration_for_labels(((),))
    one = _declaration_for_labels((("first_name",),))
    two_names = _declaration_for_labels((("first_name", "last_name"),))
    relation = _relation_declaration()
    phone = _declaration_for_labels((("phone_number",),))
    four = _declaration_for_labels((SLOT_LABELS,))
    five = _declaration_for_labels(((*SLOT_LABELS, "user_name"),))

    def owner(case_id: str, declaration: ReferenceDeclaration, events: tuple[ReferenceEvent, ...]) -> ReferenceCase:
        return ReferenceCase(f"owner-{case_id}", declaration, events, case_id)

    invalid_label = replace(
        one,
        mentions=(replace(one.mentions[0], label="account_number"),),
    )
    missing_selector = replace(
        relation,
        relations=(
            ReferenceRelation(
                "email_from_name/v1",
                (ReferenceSelector("c0", "user_name"),),
                ReferenceSelector("c0", "email_address"),
            ),
        ),
    )
    duplicate_selector = replace(
        relation,
        relations=(
            ReferenceRelation(
                "email_from_name/v1",
                (
                    ReferenceSelector("c0", "person_given_name"),
                    ReferenceSelector("c0", "person_given_name"),
                ),
                ReferenceSelector("c0", "email_address"),
            ),
        ),
    )
    cross_scope = _declaration_for_labels((("first_name",), ("email",)))
    cross_scope = replace(
        cross_scope,
        relations=(
            ReferenceRelation(
                "email_from_name/v1",
                (ReferenceSelector("c0", "person_given_name"),),
                ReferenceSelector("c1", "email_address"),
            ),
        ),
    )
    wrong_roles = replace(
        relation,
        relations=(
            ReferenceRelation(
                "email_from_name/v1",
                (ReferenceSelector("c0", "email_address"),),
                ReferenceSelector("c0", "person_given_name"),
            ),
        ),
    )
    exact_bytes = _single_source_declaration("A" * 256)
    over_bytes = _single_source_declaration("A" * 257)
    success_one = _success_events(one)
    return (
        owner("valid_empty_scope_zero_dispatch", empty, _success_events(empty)),
        owner("valid_single_given_name", one, success_one),
        owner("valid_given_family_email_relation", relation, _success_events(relation)),
        owner("valid_phone_source_mask", phone, _success_events(phone)),
        owner("unknown_contract_version", replace(one, contract_version="phase7/v2"), ()),
        owner("contract_digest_mismatch", replace(one, contract_digest_valid=False), ()),
        owner("missing_detector_disposition", replace(one, detector_universe_complete=False), ()),
        owner("unknown_role", replace(one, declared_role="unknown_role"), ()),
        owner(
            "unknown_relation",
            replace(
                relation,
                relations=(replace(relation.relations[0], version="unknown_relation/v1"),),
            ),
            (),
        ),
        owner("unknown_mask", replace(one, declared_mask="unknown_mask/v1"), ()),
        owner("unsupported_detector_label", invalid_label, ()),
        owner("selector_resolves_zero_slots", missing_selector, ()),
        owner("selector_resolves_multiple_slots", duplicate_selector, ()),
        owner("relation_crosses_scopes", cross_scope, ()),
        owner("email_relation_wrong_roles", wrong_roles, ()),
        owner(
            "distinct_slots_same_canonical_value",
            two_names,
            _success_events(two_names, {"slot-0-0": "Nova", "slot-0-1": "Ｎｏｖａ"}),
        ),
        owner(
            "candidate_matches_own_original",
            one,
            _success_events(one, {"slot-0-0": " Alice "}),
        ),
        owner(
            "candidate_matches_other_slot_original",
            two_names,
            _success_events(two_names, {"slot-0-0": "Adams", "slot-0-1": "Vale"}),
        ),
        owner(
            "email_local_part_omits_name",
            relation,
            _success_events(
                relation,
                {
                    "slot-0-0": "Nova",
                    "slot-0-1": "Vale",
                    "slot-0-2": "other@example.test",
                },
            ),
        ),
        owner("count_limits_exact", four, _success_events(four)),
        owner("count_limits_one_over", five, ()),
        owner("byte_limits_exact", exact_bytes, _success_events(exact_bytes)),
        owner("byte_limits_one_over", over_bytes, ()),
        owner("runtime_capability_missing", replace(one, capability=False), ()),
        owner(
            "trusted_task_failure",
            one,
            (
                ReferenceEvent(ReferenceEventKind.DISPATCH_ACCEPTED, "scope", 0),
                ReferenceEvent(ReferenceEventKind.FAILED_RECORD, "scope", 0, outcome="attributed"),
                *_failed_tail(),
            ),
        ),
        owner(
            "unattributable_failure",
            one,
            (
                ReferenceEvent(ReferenceEventKind.DISPATCH_ACCEPTED, "scope", 0),
                ReferenceEvent(ReferenceEventKind.FAILED_RECORD, "scope", 0, attempt=1, outcome="foreign"),
                *_failed_tail(),
            ),
        ),
        owner("cleanup_attestation_verified", one, success_one),
        owner(
            "cleanup_attestation_missing",
            one,
            tuple(event for event in success_one if event.kind is not ReferenceEventKind.CLEANUP),
        ),
        owner(
            "cleanup_attestation_contradictory",
            one,
            tuple(
                replace(event, outcome="contradictory") if event.kind is ReferenceEventKind.CLEANUP else event
                for event in success_one
            ),
        ),
        owner(
            "redact_policy_role_bearing_scope",
            replace(one, policy=ReferencePolicy.CURRENT_EMPTY),
            _failed_tail(),
        ),
    )


def _slot_envelope_cases() -> tuple[ReferenceCase, ...]:
    cases: list[ReferenceCase] = []
    for slot_count in range(5):
        declaration = _declaration_for_labels((SLOT_LABELS[:slot_count],))
        cases.append(
            ReferenceCase(
                f"future-slots-{slot_count}",
                declaration,
                _success_events(declaration),
            )
        )
        current = replace(declaration, policy=ReferencePolicy.CURRENT_EMPTY)
        cases.append(
            ReferenceCase(
                f"current-empty-policy-slots-{slot_count}",
                current,
                _success_events(current) if slot_count == 0 else _failed_tail(),
            )
        )
    for total_slots in range(5):
        for left_slots in range(total_slots + 1):
            right_slots = total_slots - left_slots
            declaration = _declaration_for_labels((SLOT_LABELS[:left_slots], SLOT_LABELS[:right_slots]))
            cases.append(
                ReferenceCase(
                    f"independent-scopes-{left_slots}-{right_slots}",
                    declaration,
                    _success_events(declaration),
                )
            )
    equal_text = _equal_text_distinct_clusters_declaration()
    cases.append(ReferenceCase("equal-text-distinct-clusters", equal_text, _success_events(equal_text)))
    shared = _shared_slot_declaration()
    cases.append(ReferenceCase("shared-slot-reuse", shared, _success_events(shared)))
    return tuple(cases)


def _admission_cases() -> tuple[ReferenceCase, ...]:
    datums = tuple(ReferenceDatum(f"d{index}", f"plain-{index}") for index in range(4))
    base = ReferenceDeclaration(
        datums,
        (ReferenceScope(("d0", "d1")), ReferenceScope(("d2", "d3"))),
        groups=tuple((datum.id,) for datum in datums),
    )
    first, second, third, fourth = (datum.id for datum in base.datums)
    cases = {
        "empty-scope": replace(
            base,
            scopes=(ReferenceScope(()), ReferenceScope((first, second, third, fourth))),
        ),
        "duplicate-scope": replace(
            base,
            scopes=(
                ReferenceScope((first, second, third, fourth)),
                ReferenceScope((fourth, third, second, first)),
            ),
        ),
        "duplicate-member": replace(
            base,
            scopes=(ReferenceScope((first, first, second)), ReferenceScope((third, fourth))),
        ),
        "unknown-datum": replace(
            base,
            scopes=(
                ReferenceScope((first, second, third)),
                ReferenceScope(("foreign",)),
            ),
        ),
        "coverage-gap": replace(base, scopes=(ReferenceScope((first, second, third)),)),
        "overlap": replace(
            base,
            scopes=(
                ReferenceScope((first, second)),
                ReferenceScope((second, third, fourth)),
            ),
        ),
        "nesting": replace(
            base,
            scopes=(
                ReferenceScope((first, second, third, fourth)),
                ReferenceScope((first, second)),
            ),
        ),
    }
    return tuple(ReferenceCase(f"admission-{name}", declaration) for name, declaration in cases.items())


def _lifecycle_cases() -> tuple[ReferenceCase, ...]:
    one = _declaration_for_labels((("first_name",),))
    success = _success_events(one)
    dispatch = ReferenceEvent(ReferenceEventKind.DISPATCH_ACCEPTED, "scope", 0)
    candidate = next(event for event in success if event.kind is ReferenceEventKind.CANDIDATE_ROWS)
    tail = tuple(
        event
        for event in success
        if event.kind not in {ReferenceEventKind.DISPATCH_ACCEPTED, ReferenceEventKind.CANDIDATE_ROWS}
    )
    two_datums = _declaration_for_labels((("first_name",), ("first_name",)))
    grouped = replace(two_datums, groups=(("d0", "d1"),))
    dependent = replace(two_datums, dependencies=(("d0", "d1"),))
    cascading = _cascading_declaration()
    cases = (
        ReferenceCase(
            "dispatch-rejected",
            one,
            (ReferenceEvent(ReferenceEventKind.DISPATCH_REJECTED, "scope", 0), *_failed_tail()),
        ),
        ReferenceCase(
            "backend-exception",
            one,
            (
                dispatch,
                ReferenceEvent(ReferenceEventKind.BACKEND_EXCEPTION, "scope", 0),
                *_failed_tail(),
            ),
        ),
        ReferenceCase(
            "contradictory-candidate-evidence",
            one,
            (
                dispatch,
                replace(candidate, assignments=(("slot-0-0", "Nova"), ("slot-0-0", "Vale"))),
                *_failed_tail(),
            ),
        ),
        ReferenceCase(
            "cancel-before-dispatch", one, (ReferenceEvent(ReferenceEventKind.CANCELLATION), *_failed_tail())
        ),
        ReferenceCase(
            "dispatch-cancel-without-stop",
            one,
            (dispatch, ReferenceEvent(ReferenceEventKind.CANCELLATION), *_failed_tail()),
        ),
        ReferenceCase(
            "dispatch-cancel-trusted-stop",
            one,
            (
                dispatch,
                ReferenceEvent(ReferenceEventKind.CANCELLATION),
                ReferenceEvent(ReferenceEventKind.TRUSTED_STOP, "scope", 0),
                *_failed_tail(),
            ),
        ),
        ReferenceCase(
            "late-candidate-after-stop",
            one,
            (
                dispatch,
                ReferenceEvent(ReferenceEventKind.CANCELLATION),
                ReferenceEvent(ReferenceEventKind.TRUSTED_STOP, "scope", 0),
                candidate,
                *_failed_tail(),
            ),
        ),
        ReferenceCase(
            "late-candidate-after-loss",
            one,
            (dispatch, ReferenceEvent(ReferenceEventKind.LOSS, "scope", 0), candidate, *_failed_tail()),
        ),
        ReferenceCase(
            "foreign-candidate-before-acceptance",
            one,
            (dispatch, replace(candidate, attempt=1), *_failed_tail()),
        ),
        ReferenceCase(
            "partial-candidate",
            _declaration_for_labels((("first_name", "last_name"),)),
            _partial_candidate_events(),
        ),
        ReferenceCase(
            "planned-then-foreign-is-absorbing",
            one,
            (dispatch, candidate, replace(candidate, attempt=1), *tail),
        ),
        ReferenceCase(
            "finalization-failure",
            one,
            tuple(
                replace(event, outcome="failed") if event.kind is ReferenceEventKind.FINALIZE else event
                for event in success
            ),
        ),
        ReferenceCase(
            "cleanup-failure",
            one,
            tuple(
                replace(event, outcome="failed") if event.kind is ReferenceEventKind.CLEANUP else event
                for event in success
            ),
        ),
        ReferenceCase(
            "teardown-failure-after-acceptance",
            one,
            tuple(
                replace(event, outcome="failed") if event.kind is ReferenceEventKind.TEARDOWN else event
                for event in success
            ),
        ),
        ReferenceCase(
            "atomic-group-member-failure",
            grouped,
            _success_events(grouped, transform_failures={"d1"}),
        ),
        ReferenceCase(
            "dependent-datum-withheld",
            dependent,
            _success_events(dependent, transform_failures={"d0"}),
        ),
        ReferenceCase(
            "independent-scope-local-failure",
            two_datums,
            _success_events(two_datums, transform_failures={"d0"}),
        ),
        ReferenceCase(
            "anchored-non-cascading-application",
            cascading,
            _success_events(cascading, {"slot-0-0": "Nova Blake", "slot-0-1": "Vale"}),
        ),
        ReferenceCase(
            "release-then-cancel-is-absorbing",
            one,
            (*success, ReferenceEvent(ReferenceEventKind.CANCELLATION)),
        ),
    )
    return cases


def _declaration_for_labels(
    labels_by_scope: tuple[tuple[str, ...], ...],
) -> ReferenceDeclaration:
    datums: list[ReferenceDatum] = []
    scopes: list[ReferenceScope] = []
    mentions: list[ReferenceMention] = []
    source_sets = (
        {
            "first_name": "Alice",
            "last_name": "Adams",
            "email": "alice@example.com",
            "phone_number": "555-0100",
            "user_name": "alice_1",
        },
        {
            "first_name": "Bob",
            "last_name": "Stone",
            "email": "bob@example.com",
            "phone_number": "555-0200",
            "user_name": "bob_2",
        },
    )
    for scope_index, labels in enumerate(labels_by_scope):
        datum_id = f"d{scope_index}"
        sources = tuple(source_sets[scope_index][label] for label in labels)
        text = " ".join(sources) if sources else f"plain-{scope_index}"
        datums.append(ReferenceDatum(datum_id, text))
        scopes.append(ReferenceScope((datum_id,)))
        cursor = 0
        for label, source in zip(labels, sources, strict=True):
            start = text.index(source, cursor)
            mentions.append(ReferenceMention(datum_id, start, start + len(source), source, label, f"c{scope_index}"))
            cursor = start + len(source)
    return ReferenceDeclaration(
        tuple(datums),
        tuple(scopes),
        tuple(mentions),
        groups=tuple((datum.id,) for datum in datums),
    )


def _relation_declaration() -> ReferenceDeclaration:
    declaration = _declaration_for_labels((("first_name", "last_name", "email"),))
    return replace(
        declaration,
        relations=(
            ReferenceRelation(
                "email_from_name/v1",
                (
                    ReferenceSelector("c0", "person_given_name"),
                    ReferenceSelector("c0", "person_family_name"),
                ),
                ReferenceSelector("c0", "email_address"),
            ),
        ),
    )


def _single_source_declaration(source: str) -> ReferenceDeclaration:
    datum = ReferenceDatum("d0", source)
    mention = ReferenceMention("d0", 0, len(source), source, "first_name", "c0")
    return ReferenceDeclaration((datum,), (ReferenceScope(("d0",)),), (mention,), groups=(("d0",),))


def _equal_text_distinct_clusters_declaration() -> ReferenceDeclaration:
    datum = ReferenceDatum("d0", "Alice Alice")
    return ReferenceDeclaration(
        (datum,),
        (ReferenceScope(("d0",)),),
        (
            ReferenceMention("d0", 0, 5, "Alice", "first_name", "c0"),
            ReferenceMention("d0", 6, 11, "Alice", "first_name", "c1"),
        ),
        groups=(("d0",),),
    )


def _shared_slot_declaration() -> ReferenceDeclaration:
    datum = ReferenceDatum("d0", "Alice and Alicia")
    return ReferenceDeclaration(
        (datum,),
        (ReferenceScope(("d0",)),),
        (
            ReferenceMention("d0", 0, 5, "Alice", "first_name", "c0"),
            ReferenceMention("d0", 10, 16, "Alicia", "first_name", "c0"),
        ),
        groups=(("d0",),),
    )


def _cascading_declaration() -> ReferenceDeclaration:
    datum = ReferenceDatum("d0", "Alice met Nova")
    return ReferenceDeclaration(
        (datum,),
        (ReferenceScope(("d0",)),),
        (
            ReferenceMention("d0", 0, 5, "Alice", "first_name", "c0"),
            ReferenceMention("d0", 10, 14, "Nova", "last_name", "c1"),
        ),
        groups=(("d0",),),
    )


def _success_events(
    declaration: ReferenceDeclaration,
    assignments: dict[str, str] | None = None,
    *,
    transform_failures: set[str] | None = None,
) -> tuple[ReferenceEvent, ...]:
    compiled = _compile_reference(declaration)
    if isinstance(compiled, str):
        return ()
    manifests, _mention_slots = compiled
    candidate_values = dict(assignments or {})
    events: list[ReferenceEvent] = []
    for scope_index, manifest in enumerate(manifests):
        if not manifest.slots or declaration.policy is ReferencePolicy.CURRENT_EMPTY:
            continue
        for slot in manifest.slots:
            candidate_values.setdefault(slot.key, _candidate_for_slot(slot, scope_index))
        events.append(ReferenceEvent(ReferenceEventKind.DISPATCH_ACCEPTED, "scope", scope_index))
        events.append(
            ReferenceEvent(
                ReferenceEventKind.CANDIDATE_ROWS,
                "scope",
                scope_index,
                assignments=tuple((slot.key, candidate_values[slot.key]) for slot in manifest.slots),
            )
        )
    failures = transform_failures or set()
    mentioned = {mention.datum for mention in declaration.mentions}
    for datum_index, datum in enumerate(declaration.datums):
        if datum.id in mentioned:
            events.append(
                ReferenceEvent(
                    ReferenceEventKind.TRANSFORM,
                    "datum",
                    datum_index,
                    outcome="failed" if datum.id in failures else "accepted",
                )
            )
    groups = declaration.groups or tuple((datum.id,) for datum in declaration.datums)
    events.extend(
        ReferenceEvent(ReferenceEventKind.VERIFY, "group", group_index) for group_index, _group in enumerate(groups)
    )
    events.extend(
        (
            ReferenceEvent(ReferenceEventKind.FINALIZE),
            ReferenceEvent(ReferenceEventKind.CLEANUP, outcome="verified"),
            ReferenceEvent(ReferenceEventKind.IMMUTABLE_ACCEPT),
            ReferenceEvent(ReferenceEventKind.TEARDOWN),
            ReferenceEvent(ReferenceEventKind.RELEASE),
        )
    )
    if len(events) > MAX_EXOGENOUS_OBSERVATIONS:
        raise AssertionError("generated trace exceeds frozen bound")
    return tuple(events)


def _candidate_for_slot(slot: ReferenceSlot, scope_index: int) -> str:
    candidates = {
        "person_given_name": ("Nova", "Orion"),
        "person_family_name": ("Vale", "Stonebridge"),
        "email_address": ("nova.vale@example.test", "orion.stonebridge@example.test"),
        "voice_phone_number": ("555-0199", "555-0299"),
        "fax_number": ("555-0198", "555-0298"),
        "user_name": ("nova_1", "orion_2"),
    }
    return candidates[slot.role][scope_index]


def _failed_tail() -> tuple[ReferenceEvent, ...]:
    return (
        ReferenceEvent(ReferenceEventKind.FINALIZE),
        ReferenceEvent(ReferenceEventKind.CLEANUP, outcome="verified"),
        ReferenceEvent(ReferenceEventKind.IMMUTABLE_ACCEPT),
        ReferenceEvent(ReferenceEventKind.TEARDOWN),
        ReferenceEvent(ReferenceEventKind.RELEASE),
    )


def _partial_candidate_events() -> tuple[ReferenceEvent, ...]:
    return (
        ReferenceEvent(ReferenceEventKind.DISPATCH_ACCEPTED, "scope", 0),
        ReferenceEvent(
            ReferenceEventKind.CANDIDATE_ROWS,
            "scope",
            0,
            assignments=(("slot-0-0", "Nova"),),
        ),
        *_failed_tail(),
    )
