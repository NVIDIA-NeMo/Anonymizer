# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure admission for the private Phase 7 stable-Substitute profile."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from anonymizer.engine.execution.accounting_plan import _AccountingPlan, _ScopeTaskSubject, _StageId, _TaskKey
from anonymizer.engine.execution.context_admission import _CompiledContextProjection
from anonymizer.engine.execution.graph import _CoherenceScope, _DatumId, _TextDatum
from anonymizer.engine.execution.mention_admission import _MentionId
from anonymizer.engine.execution.mention_resolution import _ClusterId
from anonymizer.engine.execution.phase6_plan import (
    _is_admitted_phase6_plan,
    _Phase6Plan,
    _Phase6ProfileVersion,
)
from anonymizer.engine.execution.phase6_runtime import (
    _is_admitted_substitute_handoff,
    _Phase6SubstituteHandoff,
)
from anonymizer.engine.execution.phase7_contract import (
    _is_admitted_phase7_contract,
    _Phase7StableSubstituteContract,
)
from anonymizer.engine.execution.role_policy import _ClassifiedRole


class _PrivatePhase7AdmissionValue:
    def __repr__(self) -> str:
        return f"<private {type(self).__name__.strip('_').replace('_', ' ').lower()}>"

    def __reduce__(self) -> str | tuple[object, ...]:
        raise TypeError("private Phase 7 admission values are not serializable")


class _Phase7AdmissionCode(str, Enum):
    INVALID_INPUT = "invalid_input"
    LIMIT_EXCEEDED = "limit_exceeded"
    EMPTY_SCOPE = "empty_scope"
    DUPLICATE_SCOPE = "duplicate_scope"
    DUPLICATE_SCOPE_MEMBER = "duplicate_scope_member"
    UNKNOWN_SCOPE_DATUM = "unknown_scope_datum"
    SCOPE_COVERAGE_GAP = "scope_coverage_gap"
    SCOPE_OVERLAP = "scope_overlap"
    UNSUPPORTED_SCOPE_NESTING = "unsupported_scope_nesting"
    PHASE6_HANDOFF_MISMATCH = "phase6_handoff_mismatch"
    UNSUPPORTED_SELECTOR = "unsupported_selector"
    SELECTOR_MISSING = "selector_missing"
    SELECTOR_AMBIGUOUS = "selector_ambiguous"
    UNSUPPORTED_RELATION = "unsupported_relation"
    CROSS_SCOPE_RELATION = "cross_scope_relation"
    RELATION_ROLE_MISMATCH = "relation_role_mismatch"


@dataclass(frozen=True, slots=True, repr=False)
class _Phase7Rejected(_PrivatePhase7AdmissionValue):
    code: _Phase7AdmissionCode


@dataclass(frozen=True, slots=True, repr=False)
class _Phase7Declarations(_PrivatePhase7AdmissionValue):
    coherence_scopes: tuple[_CoherenceScope, ...]
    relations: tuple[object, ...] = ()


@dataclass(frozen=True, slots=True, repr=False, eq=False)
class _Phase7ScopeId(_PrivatePhase7AdmissionValue):
    """Compiler-issued opaque capability for one flat coherence scope."""


@dataclass(frozen=True, slots=True, repr=False)
class _ClusterRoleSelector(_PrivatePhase7AdmissionValue):
    version: str
    cluster: _ClusterId
    role: str


@dataclass(frozen=True, slots=True, repr=False)
class _RelationDeclaration(_PrivatePhase7AdmissionValue):
    version: str
    upstream: tuple[_ClusterRoleSelector, ...]
    downstream: _ClusterRoleSelector


@dataclass(frozen=True, slots=True, repr=False)
class _PreSlot(_PrivatePhase7AdmissionValue):
    scope_id: _Phase7ScopeId
    cluster_id: _ClusterId
    role: str
    format: str
    mask: str
    mention_ids: tuple[_MentionId, ...]


@dataclass(frozen=True, slots=True, repr=False, eq=False)
class _ReplacementSlotId(_PrivatePhase7AdmissionValue):
    """Compiler-issued identity for one structural scope-cluster-role key."""


@dataclass(frozen=True, slots=True, repr=False)
class _ReplacementSlot(_PrivatePhase7AdmissionValue):
    id: _ReplacementSlotId
    cluster_id: _ClusterId
    role: str
    format: str
    mask: str
    mention_ids: tuple[_MentionId, ...]


@dataclass(frozen=True, slots=True, repr=False)
class _CompiledRelation(_PrivatePhase7AdmissionValue):
    version: str
    upstream: tuple[_ReplacementSlotId, ...]
    downstream: _ReplacementSlotId


@dataclass(frozen=True, slots=True, repr=False)
class _RequiredDistinctPair(_PrivatePhase7AdmissionValue):
    left: _ReplacementSlotId
    right: _ReplacementSlotId


@dataclass(frozen=True, slots=True, repr=False)
class _PreSlotRelation(_PrivatePhase7AdmissionValue):
    version: str
    upstream: tuple[_PreSlot, ...]
    downstream: _PreSlot


@dataclass(frozen=True, slots=True, repr=False)
class _ScopeManifestProof(_PrivatePhase7AdmissionValue):
    seal: object = field(compare=False)
    snapshot: tuple[object, ...]


@dataclass(frozen=True, slots=True, repr=False)
class _ScopeManifest(_PrivatePhase7AdmissionValue):
    id: _Phase7ScopeId
    members: tuple[_DatumId, ...]
    slots: tuple[_ReplacementSlot, ...] = ()
    required_pairs: tuple[_RequiredDistinctPair, ...] = ()
    relations: tuple[_CompiledRelation, ...] = ()
    _proof: _ScopeManifestProof | None = field(default=None, compare=False)


@dataclass(frozen=True, slots=True, repr=False)
class _Phase7PlanProof(_PrivatePhase7AdmissionValue):
    seal: object = field(compare=False)
    snapshot: tuple[object, ...]


@dataclass(frozen=True, slots=True, repr=False)
class _Phase7Plan(_PrivatePhase7AdmissionValue):
    manifests: tuple[_ScopeManifest, ...]
    # These are issued while compiling the Phase 7 declaration.  Runtime must
    # consume them verbatim; reconstructing scope tasks later loses Phase 4
    # conservation and turns an opaque owner capability into presentation data.
    accounting: _AccountingPlan
    scope_tasks: tuple[_TaskKey, ...]
    _proof: _Phase7PlanProof | None = field(default=None, compare=False)


_SCOPE_MANIFEST_SEAL = object()
_PHASE7_PLAN_SEAL = object()


def _compile_phase7_plan(
    phase6: object,
    handoffs: object,
    declarations: object,
    contract: object,
) -> _Phase7Plan | _Phase7Rejected:
    """Validate authored Phase 7 scope shape without creating runtime state."""
    if (
        not isinstance(phase6, _Phase6Plan)
        or not _is_admitted_phase6_plan(phase6)
        or not isinstance(handoffs, tuple)
        or not isinstance(declarations, _Phase7Declarations)
        or not isinstance(contract, _Phase7StableSubstituteContract)
        or not _is_admitted_phase7_contract(contract)
        or not isinstance(declarations.coherence_scopes, tuple)
        or not isinstance(declarations.relations, tuple)
    ):
        return _Phase7Rejected(_Phase7AdmissionCode.INVALID_INPUT)
    if _total_limits_exceeded(phase6, handoffs, declarations, contract):
        return _Phase7Rejected(_Phase7AdmissionCode.LIMIT_EXCEEDED)
    rejected = _validate_partition(declarations.coherence_scopes, phase6.accounting.datums)
    if rejected is not None:
        return rejected
    if not _valid_phase6_handoffs(phase6, handoffs, contract):
        return _Phase7Rejected(_Phase7AdmissionCode.PHASE6_HANDOFF_MISMATCH)
    materialized = _materialize_scope_manifests(
        declarations.coherence_scopes,
        phase6,
        handoffs,
        declarations.relations,
        contract,
    )
    if isinstance(materialized, _Phase7Rejected):
        return materialized
    manifests = materialized
    subjects = tuple(_ScopeTaskSubject() for _manifest in manifests)
    accounting = phase6.accounting.with_scope_tasks(_StageId("phase7-plan"), subjects)
    scope_tasks = tuple(_TaskKey(_StageId("phase7-plan"), subject) for subject in subjects)
    candidate = _Phase7Plan(manifests, accounting, scope_tasks)
    snapshot = _phase7_plan_snapshot(candidate)
    if snapshot is None:
        return _Phase7Rejected(_Phase7AdmissionCode.INVALID_INPUT)
    return _Phase7Plan(manifests, accounting, scope_tasks, _Phase7PlanProof(_PHASE7_PLAN_SEAL, snapshot))


def _is_admitted_phase7_plan(value: object) -> bool:
    return (
        isinstance(value, _Phase7Plan)
        and value._proof is not None
        and value._proof.seal is _PHASE7_PLAN_SEAL
        and all(_is_admitted_scope_manifest(manifest) for manifest in value.manifests)
        and len(value.scope_tasks) == len(value.manifests)
        and all(
            task in value.accounting.tasks and isinstance(task.subject, _ScopeTaskSubject) for task in value.scope_tasks
        )
        and value._proof.snapshot == _phase7_plan_snapshot(value)
    )


def _valid_phase6_handoffs(
    phase6: _Phase6Plan,
    handoffs: tuple[object, ...],
    contract: _Phase7StableSubstituteContract,
) -> bool:
    if phase6.profile_version is not _Phase6ProfileVersion.SUBSTITUTE_V1:
        return False
    if len(handoffs) != len(phase6.components) or not all(
        isinstance(handoff, _Phase6SubstituteHandoff) for handoff in handoffs
    ):
        return False
    typed_handoffs = tuple(handoff for handoff in handoffs if isinstance(handoff, _Phase6SubstituteHandoff))
    expected_components = {component.key for component in phase6.components}
    if (
        len({handoff.component for handoff in typed_handoffs}) != len(typed_handoffs)
        or {handoff.component for handoff in typed_handoffs} != expected_components
    ):
        return False
    contract_roles = {role.name for role in contract.roles}
    expected_datums = {datum.id for datum in phase6.accounting.datums}
    terminal_datums: list[_DatumId] = []
    for handoff in typed_handoffs:
        if (
            not _is_admitted_substitute_handoff(handoff, phase6)
            or handoff.result_version != contract.phase6_result_version
            or handoff.policy_version != contract.phase6_policy_version
            or handoff.policy_digest != contract.phase6_policy_digest
            or any(
                not isinstance(mention.role_result, _ClassifiedRole)
                or mention.role_result.role.value not in contract_roles
                for mention in handoff.resolved.mentions
            )
        ):
            return False
        terminal_datums.extend(handoff.terminal_evidence.datum_ids)
    return len(set(terminal_datums)) == len(terminal_datums) and set(terminal_datums) == expected_datums


def _materialize_scope_manifests(
    scopes: tuple[_CoherenceScope, ...],
    phase6: _Phase6Plan,
    handoffs: tuple[object, ...],
    relations: tuple[object, ...],
    contract: _Phase7StableSubstituteContract,
) -> tuple[_ScopeManifest, ...] | _Phase7Rejected:
    scope_members = _canonical_scope_members(scopes, phase6)
    typed_handoffs = tuple(handoff for handoff in handoffs if isinstance(handoff, _Phase6SubstituteHandoff))
    pre_slots = _compile_pre_slots(scope_members, phase6, typed_handoffs, contract)
    relation_result = _compile_pre_slot_relations(relations, pre_slots)
    if isinstance(relation_result, _Phase7Rejected):
        return relation_result
    slots = _materialize_slots(pre_slots)
    compiled_relations = _bind_relations(relation_result, pre_slots, slots)
    return _seal_scope_manifests(scope_members, pre_slots, slots, compiled_relations, contract)


def _canonical_scope_members(
    scopes: tuple[_CoherenceScope, ...],
    phase6: _Phase6Plan,
) -> tuple[tuple[_Phase7ScopeId, tuple[_DatumId, ...]], ...]:
    datum_by_value = {datum.id.value: datum.id for datum in phase6.accounting.datums}
    position = {datum.id.value: index for index, datum in enumerate(phase6.accounting.datums)}
    canonical = sorted(
        (frozenset(member.value for member in scope.members) for scope in scopes),
        key=lambda values: tuple(sorted(position[value] for value in values)),
    )
    scope_members: list[tuple[_Phase7ScopeId, tuple[_DatumId, ...]]] = []
    for values in canonical:
        members = tuple(datum_by_value[value] for value in sorted(values, key=position.__getitem__))
        scope_members.append((_Phase7ScopeId(), members))
    return tuple(scope_members)


def _materialize_slots(pre_slots: tuple[_PreSlot, ...]) -> tuple[_ReplacementSlot, ...]:
    return tuple(
        _ReplacementSlot(
            _ReplacementSlotId(),
            pre_slot.cluster_id,
            pre_slot.role,
            pre_slot.format,
            pre_slot.mask,
            pre_slot.mention_ids,
        )
        for pre_slot in pre_slots
    )


def _bind_relations(
    relations: tuple[_PreSlotRelation, ...],
    pre_slots: tuple[_PreSlot, ...],
    slots: tuple[_ReplacementSlot, ...],
) -> tuple[_CompiledRelation, ...]:
    slot_by_pre_slot = dict(zip(pre_slots, slots, strict=True))
    compiled_relations = tuple(
        _CompiledRelation(
            relation.version,
            tuple(slot_by_pre_slot[pre_slot].id for pre_slot in relation.upstream),
            slot_by_pre_slot[relation.downstream].id,
        )
        for relation in relations
    )
    slot_position = {slot.id: index for index, slot in enumerate(slots)}
    return tuple(
        sorted(
            compiled_relations,
            key=lambda relation: (
                slot_position[relation.downstream],
                tuple(slot_position[slot_id] for slot_id in relation.upstream),
            ),
        )
    )


def _seal_scope_manifests(
    scope_members: tuple[tuple[_Phase7ScopeId, tuple[_DatumId, ...]], ...],
    pre_slots: tuple[_PreSlot, ...],
    slots: tuple[_ReplacementSlot, ...],
    relations: tuple[_CompiledRelation, ...],
    contract: _Phase7StableSubstituteContract,
) -> tuple[_ScopeManifest, ...] | _Phase7Rejected:
    manifests: list[_ScopeManifest] = []
    for scope_id, members in scope_members:
        manifest = _seal_scope_manifest(scope_id, members, pre_slots, slots, relations, contract)
        if isinstance(manifest, _Phase7Rejected):
            return manifest
        manifests.append(manifest)
    return tuple(manifests)


def _seal_scope_manifest(
    scope_id: _Phase7ScopeId,
    members: tuple[_DatumId, ...],
    pre_slots: tuple[_PreSlot, ...],
    slots: tuple[_ReplacementSlot, ...],
    relations: tuple[_CompiledRelation, ...],
    contract: _Phase7StableSubstituteContract,
) -> _ScopeManifest | _Phase7Rejected:
    scope_slots = tuple(slot for pre, slot in zip(pre_slots, slots, strict=True) if pre.scope_id is scope_id)
    pairs = tuple(
        _RequiredDistinctPair(left.id, right.id)
        for index, left in enumerate(scope_slots)
        for right in scope_slots[index + 1 :]
    )
    slot_ids = {slot.id for slot in scope_slots}
    scope_relations = tuple(relation for relation in relations if relation.downstream in slot_ids)
    limits = dict(contract.count_limits)
    if (
        len(scope_slots) > limits["max_slots_per_scope"]
        or len(pairs) > limits["max_distinct_pairs_per_scope"]
        or len({slot.cluster_id for slot in scope_slots}) > limits["max_clusters_per_scope"]
        or sum(len(slot.mention_ids) for slot in scope_slots) > limits["max_mentions_per_scope"]
        or len(scope_relations) > limits["max_relations_per_scope"]
    ):
        return _Phase7Rejected(_Phase7AdmissionCode.LIMIT_EXCEEDED)
    candidate = _ScopeManifest(scope_id, members, scope_slots, pairs, scope_relations)
    snapshot = _scope_manifest_snapshot(candidate)
    if snapshot is None:
        return _Phase7Rejected(_Phase7AdmissionCode.INVALID_INPUT)
    proof = _ScopeManifestProof(_SCOPE_MANIFEST_SEAL, snapshot)
    return _ScopeManifest(scope_id, members, scope_slots, pairs, scope_relations, proof)


def _compile_pre_slots(
    scopes: tuple[tuple[_Phase7ScopeId, tuple[_DatumId, ...]], ...],
    phase6: _Phase6Plan,
    handoffs: tuple[_Phase6SubstituteHandoff, ...],
    contract: _Phase7StableSubstituteContract,
) -> tuple[_PreSlot, ...]:
    scope_by_datum = {member: scope_id for scope_id, members in scopes for member in members}
    role_by_name = {role.name: role for role in contract.roles}
    component_position = {component.key: index for index, component in enumerate(phase6.components)}
    ordered_handoffs = sorted(handoffs, key=lambda handoff: component_position[handoff.component])
    keys: list[tuple[_Phase7ScopeId, _ClusterId, str]] = []
    mention_ids: dict[tuple[_Phase7ScopeId, _ClusterId, str], list[_MentionId]] = {}
    for handoff in ordered_handoffs:
        for resolved in handoff.resolved.mentions:
            role_result = resolved.role_result
            if not isinstance(role_result, _ClassifiedRole):
                continue
            key = (
                scope_by_datum[resolved.mention.target_datum_id],
                resolved.cluster_id,
                role_result.role.value,
            )
            if key not in mention_ids:
                keys.append(key)
                mention_ids[key] = []
            mention_ids[key].append(resolved.mention.id)
    return tuple(
        _PreSlot(
            scope_id,
            cluster_id,
            role,
            role_by_name[role].format,
            role_by_name[role].mask,
            tuple(mention_ids[(scope_id, cluster_id, role)]),
        )
        for scope_id, cluster_id, role in keys
    )


def _total_limits_exceeded(
    phase6: _Phase6Plan,
    handoffs: tuple[object, ...],
    declarations: _Phase7Declarations,
    contract: _Phase7StableSubstituteContract,
) -> bool:
    counts = dict(contract.count_limits)
    bytes_limits = dict(contract.byte_limits)
    if (
        len(phase6.accounting.datums) > counts["max_datums_per_invocation"]
        or len(declarations.coherence_scopes) > counts["max_scopes_per_invocation"]
        or any(
            isinstance(scope, _CoherenceScope)
            and isinstance(scope.members, tuple)
            and len(scope.members) > counts["max_scope_members"]
            for scope in declarations.coherence_scopes
        )
    ):
        return True
    if _context_limits_exceeded(phase6, declarations.coherence_scopes, counts, bytes_limits):
        return True
    return _handoff_limits_exceeded(handoffs, counts, bytes_limits)


def _handoff_limits_exceeded(
    handoffs: tuple[object, ...],
    counts: dict[str, int],
    bytes_limits: dict[str, int],
) -> bool:
    clusters: set[_ClusterId] = set()
    slot_keys: set[tuple[_ClusterId, str]] = set()
    mention_count = 0
    original_bytes = 0
    for handoff in handoffs:
        if not isinstance(handoff, _Phase6SubstituteHandoff):
            continue
        clusters.update(cluster.id for cluster in handoff.resolved.clustered.clusters)
        for resolved in handoff.resolved.mentions:
            mention_count += 1
            source = resolved.mention.source_slice
            detector_label = resolved.mention.detector_label
            try:
                source_bytes = len(source.encode("utf-8"))
                detector_label_bytes = len(detector_label.encode("utf-8"))
            except (AttributeError, UnicodeEncodeError):
                continue
            original_bytes += source_bytes
            if (
                source_bytes > bytes_limits["max_original_value_bytes"]
                or detector_label_bytes > bytes_limits["max_detector_label_bytes"]
            ):
                return True
            if isinstance(resolved.role_result, _ClassifiedRole):
                slot_keys.add((resolved.cluster_id, resolved.role_result.role.value))
    return (
        len(clusters) > counts["max_clusters_per_invocation"]
        or len(slot_keys) > counts["max_slots_per_invocation"]
        or mention_count > counts["max_mentions_per_invocation"]
        or original_bytes > bytes_limits["max_all_original_value_bytes"]
    )


def _context_limits_exceeded(
    phase6: _Phase6Plan,
    scopes: tuple[_CoherenceScope, ...],
    counts: dict[str, int],
    bytes_limits: dict[str, int],
) -> bool:
    if not all(
        isinstance(scope, _CoherenceScope)
        and isinstance(scope.members, tuple)
        and all(isinstance(member, _DatumId) for member in scope.members)
        for scope in scopes
    ):
        return False
    projection_by_target = {projection.target_datum_id: projection for projection in phase6.context.projections}
    datum_by_id = {datum.id: datum for datum in (*phase6.accounting.datums, *phase6.context.context_only_datums)}
    target_by_value = {datum.id.value: datum.id for datum in phase6.accounting.datums}
    for scope in scopes:
        context_ids = _context_ids_for_scope(scope, target_by_value, projection_by_target)
        if context_ids is None:
            continue
        if len(context_ids) > counts["max_context_fragments_per_scope"]:
            return True
        sizes = _context_sizes(context_ids, datum_by_id)
        if (
            any(size > bytes_limits["max_context_fragment_bytes"] for size in sizes)
            or sum(sizes) > bytes_limits["max_context_bytes_per_scope"]
        ):
            return True
    return False


def _context_ids_for_scope(
    scope: _CoherenceScope,
    target_by_value: dict[str, _DatumId],
    projection_by_target: dict[_DatumId, _CompiledContextProjection],
) -> set[_DatumId] | None:
    target_ids = tuple(target_by_value.get(member.value) for member in scope.members)
    if any(target_id is None for target_id in target_ids):
        return None
    return {
        context_id
        for target_id in target_ids
        if target_id is not None and target_id in projection_by_target
        for context_id in projection_by_target[target_id].context_datum_ids
    }


def _context_sizes(context_ids: set[_DatumId], datum_by_id: dict[_DatumId, _TextDatum]) -> tuple[int, ...]:
    sizes: list[int] = []
    for context_id in context_ids:
        datum = datum_by_id.get(context_id)
        if datum is None:
            continue
        try:
            sizes.append(len(datum.text.encode("utf-8")))
        except UnicodeEncodeError:
            continue
    return tuple(sizes)


def _compile_pre_slot_relations(
    declarations: tuple[object, ...],
    candidates: tuple[_PreSlot, ...],
) -> tuple[_PreSlotRelation, ...] | _Phase7Rejected:
    compiled: list[_PreSlotRelation] = []
    for declaration in declarations:
        if (
            not isinstance(declaration, _RelationDeclaration)
            or declaration.version != "email_from_name/v1"
            or not isinstance(declaration.upstream, tuple)
            or not 1 <= len(declaration.upstream) <= 2
            or not isinstance(declaration.downstream, _ClusterRoleSelector)
        ):
            return _Phase7Rejected(_Phase7AdmissionCode.UNSUPPORTED_RELATION)
        if len(set(declaration.upstream)) != len(declaration.upstream):
            return _Phase7Rejected(_Phase7AdmissionCode.SELECTOR_AMBIGUOUS)
        resolved_upstream: list[_PreSlot] = []
        for selector in declaration.upstream:
            resolved = _resolve_selector(selector, candidates)
            if isinstance(resolved, _Phase7Rejected):
                return resolved
            resolved_upstream.append(resolved)
        downstream = _resolve_selector(declaration.downstream, candidates)
        if isinstance(downstream, _Phase7Rejected):
            return downstream
        selected = (*resolved_upstream, downstream)
        if len({item.scope_id for item in selected}) != 1:
            return _Phase7Rejected(_Phase7AdmissionCode.CROSS_SCOPE_RELATION)
        if (
            any(item.role not in {"person_family_name", "person_given_name"} for item in resolved_upstream)
            or downstream.role != "email_address"
        ):
            return _Phase7Rejected(_Phase7AdmissionCode.RELATION_ROLE_MISMATCH)
        compiled.append(_PreSlotRelation(declaration.version, tuple(resolved_upstream), downstream))
    return tuple(compiled)


def _resolve_selector(
    selector: object,
    candidates: tuple[_PreSlot, ...],
) -> _PreSlot | _Phase7Rejected:
    if not isinstance(selector, _ClusterRoleSelector) or selector.version != "cluster_role/v1":
        return _Phase7Rejected(_Phase7AdmissionCode.UNSUPPORTED_SELECTOR)
    if not isinstance(selector.cluster, _ClusterId) or not isinstance(selector.role, str):
        return _Phase7Rejected(_Phase7AdmissionCode.SELECTOR_MISSING)
    matches = tuple(
        candidate
        for candidate in candidates
        if candidate.cluster_id is selector.cluster and candidate.role == selector.role
    )
    if not matches:
        return _Phase7Rejected(_Phase7AdmissionCode.SELECTOR_MISSING)
    if len(matches) != 1:
        return _Phase7Rejected(_Phase7AdmissionCode.SELECTOR_AMBIGUOUS)
    return matches[0]


def _is_admitted_scope_manifest(value: object) -> bool:
    return (
        isinstance(value, _ScopeManifest)
        and value._proof is not None
        and value._proof.seal is _SCOPE_MANIFEST_SEAL
        and value._proof.snapshot == _scope_manifest_snapshot(value)
    )


def _scope_manifest_snapshot(manifest: _ScopeManifest) -> tuple[object, ...] | None:
    try:
        return (
            manifest.id,
            tuple(member.value for member in manifest.members),
            manifest.slots,
            manifest.required_pairs,
            manifest.relations,
        )
    except (AttributeError, TypeError):
        return None


def _phase7_plan_snapshot(plan: _Phase7Plan) -> tuple[object, ...] | None:
    try:
        return (
            tuple(manifest._proof for manifest in plan.manifests),
            plan.accounting._proof,
            # The accounting-plan proof alone authenticates its own source
            # compilation, not this Phase 7 expansion.  Seal the concrete
            # Phase 6 prefix and appended scope-task sequence as well.
            tuple(plan.accounting.tasks),
            tuple(plan.scope_tasks),
        )
    except (AttributeError, TypeError):
        return None


def _validate_partition(
    scopes: tuple[_CoherenceScope, ...],
    datums: tuple[_TextDatum, ...],
) -> _Phase7Rejected | None:
    compiled = _compile_scope_semantics(scopes)
    if isinstance(compiled, _Phase7Rejected):
        return compiled
    semantic_scopes = compiled
    known = {datum.id.value for datum in datums}
    if any(
        not isinstance(member, _DatumId) or not isinstance(member.value, str) or member.value not in known
        for scope in scopes
        for member in scope.members
    ):
        return _Phase7Rejected(_Phase7AdmissionCode.UNKNOWN_SCOPE_DATUM)
    covered = set().union(*semantic_scopes) if semantic_scopes else set()
    if covered != known:
        return _Phase7Rejected(_Phase7AdmissionCode.SCOPE_COVERAGE_GAP)
    overlap = _unsupported_scope_relationship(semantic_scopes)
    return overlap


def _compile_scope_semantics(
    scopes: tuple[_CoherenceScope, ...],
) -> tuple[frozenset[object], ...] | _Phase7Rejected:
    if not all(isinstance(scope, _CoherenceScope) and isinstance(scope.members, tuple) for scope in scopes):
        return _Phase7Rejected(_Phase7AdmissionCode.INVALID_INPUT)
    if any(not scope.members for scope in scopes):
        return _Phase7Rejected(_Phase7AdmissionCode.EMPTY_SCOPE)
    scope_values = tuple(tuple(getattr(member, "value", None) for member in scope.members) for scope in scopes)
    semantic_scopes = tuple(frozenset(values) for values in scope_values)
    if len(set(semantic_scopes)) != len(semantic_scopes):
        return _Phase7Rejected(_Phase7AdmissionCode.DUPLICATE_SCOPE)
    if any(len(set(values)) != len(values) for values in scope_values):
        return _Phase7Rejected(_Phase7AdmissionCode.DUPLICATE_SCOPE_MEMBER)
    return semantic_scopes


def _unsupported_scope_relationship(
    semantic_scopes: tuple[frozenset[object], ...],
) -> _Phase7Rejected | None:
    for index, left in enumerate(semantic_scopes):
        for right in semantic_scopes[index + 1 :]:
            if left & right and not (left < right or right < left):
                return _Phase7Rejected(_Phase7AdmissionCode.SCOPE_OVERLAP)
    if any(
        left < right or right < left
        for index, left in enumerate(semantic_scopes)
        for right in semantic_scopes[index + 1 :]
    ):
        return _Phase7Rejected(_Phase7AdmissionCode.UNSUPPORTED_SCOPE_NESTING)
    return None
