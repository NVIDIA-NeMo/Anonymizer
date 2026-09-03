# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import importlib.util
import pickle
from collections.abc import Callable
from dataclasses import FrozenInstanceError, dataclass, replace
from types import ModuleType
from typing import Any

import pytest

from anonymizer.engine.execution.accounting_plan import _AccountingLimits, _DatumTaskSubject, _TaskPredecessor
from anonymizer.engine.execution.context_contract import (
    _BackendArtifactClass,
    _ContextBackendCapability,
    _ContextExecutionContract,
    _ContextLimits,
    _ContextOrdering,
    _ContextProfile,
    _ContextSchemaVersion,
    _RetentionPosture,
)
from anonymizer.engine.execution.graph import (
    _AtomicGroup,
    _CoherenceScope,
    _ContextScope,
    _DatumId,
    _DatumPurpose,
    _ProtectionGraph,
    _TextDatum,
)
from anonymizer.engine.execution.mention_admission import (
    _MentionId,
    _MentionLimits,
    _ValidationDecision,
    _ValidationDecisionKind,
)
from anonymizer.engine.execution.mention_resolution import (
    _EvidenceVersion,
    _SameSubjectEvidence,
)
from anonymizer.engine.execution.phase6_plan import (
    _compile_phase6_plan,
    _Phase6Plan,
    _Phase6ProfileVersion,
)
from anonymizer.engine.execution.phase6_runtime import (
    _CandidateProposal,
    _Phase6AugmentationWork,
    _Phase6CandidateWork,
    _Phase6Execution,
    _Phase6ResolverWork,
    _Phase6Runtime,
    _Phase6ValidationWork,
)
from anonymizer.engine.execution.phase7_contract import _load_phase7_contract
from anonymizer.engine.execution.role_policy import _ClassifiedRole

_ACCOUNTING_LIMITS = _AccountingLimits(16, 16_384, 65_536, max_stages=8)
_MENTION_LIMITS = _MentionLimits(16, 16, 128, 512)


@dataclass(frozen=True)
class _Proposal:
    source: str
    label: str
    cluster: str


class _Backend:
    def __init__(self, proposals: dict[str, tuple[_Proposal, ...]]) -> None:
        self._proposals = proposals
        self.calls: list[str] = []
        self.planner_effect_count = 0

    def context_capability(self) -> _ContextBackendCapability:
        return _contract_and_capability()[1]

    def detect(self, work: _Phase6CandidateWork) -> tuple[_CandidateProposal, ...]:
        self.calls.append("detect")
        proposals = self._proposals.get(work.target.datum_id.value, ())
        return tuple(
            _CandidateProposal(
                work.target.text.index(proposal.source),
                work.target.text.index(proposal.source) + len(proposal.source),
                proposal.source,
                proposal.label,
            )
            for proposal in proposals
        )

    def augment(self, work: _Phase6AugmentationWork) -> tuple[_CandidateProposal, ...]:
        del work
        self.calls.append("augment")
        return ()

    def validate(self, work: _Phase6ValidationWork) -> tuple[_ValidationDecision, ...]:
        self.calls.append("validate")
        return tuple(_ValidationDecision(item.token, _ValidationDecisionKind.KEEP) for item in work.candidates)

    def resolve(self, work: _Phase6ResolverWork) -> tuple[_SameSubjectEvidence, ...]:
        self.calls.append("resolve")
        proposal_by_source = {
            proposal.source: proposal for proposals in self._proposals.values() for proposal in proposals
        }
        grouped: dict[str, list[_MentionId]] = {}
        for mention in work.eligible_mentions:
            proposal = proposal_by_source[mention.source_slice]
            grouped.setdefault(proposal.cluster, []).append(mention.id)
        evidence: list[_SameSubjectEvidence] = []
        for mention_ids in grouped.values():
            for left, right in zip(mention_ids, mention_ids[1:], strict=False):
                evidence.append(_SameSubjectEvidence(work.owner.token, left, right, _EvidenceVersion.V1))
        return tuple(evidence)

    def close_phase6(self) -> bool:
        return True

    def plan(self, _work: object) -> None:
        self.planner_effect_count += 1


def _contract_and_capability() -> tuple[_ContextExecutionContract, _ContextBackendCapability]:
    limits = _ContextLimits(16, 16_384, 16, 65_536)
    contract = _ContextExecutionContract(
        _ContextProfile.TARGET_CONTEXT_V1,
        _ContextSchemaVersion.V1,
        limits,
        True,
        _ContextOrdering.DECLARED,
        (_BackendArtifactClass.CONTEXT_REQUEST,),
    )
    capability = _ContextBackendCapability(
        contract.profile,
        contract.schema_version,
        limits,
        True,
        contract.ordering,
        contract.required_artifacts,
        _RetentionPosture.DISABLED,
    )
    return contract, capability


def _graph(
    texts: tuple[str, ...],
    scopes: tuple[tuple[str, ...], ...],
    *,
    connect_context: bool = False,
    context_texts: tuple[str, ...] = (),
) -> _ProtectionGraph:
    targets = tuple(
        _TextDatum(_DatumId(f"target-{index}"), text, _DatumPurpose.TARGET) for index, text in enumerate(texts)
    )
    context_datums = tuple(
        _TextDatum(_DatumId(f"context-{index}"), text, _DatumPurpose.CONTEXT_ONLY)
        for index, text in enumerate(context_texts)
    )
    datums = (*targets, *context_datums)
    by_value = {datum.id.value: datum.id for datum in targets}
    return _ProtectionGraph(
        datums=datums,
        links=(),
        context_scopes=tuple(
            _ContextScope(
                datum.id,
                (tuple(candidate.id for candidate in targets if candidate.id != datum.id) if connect_context else ())
                + tuple(context.id for context in context_datums),
            )
            for datum in targets
        ),
        coherence_scopes=tuple(_CoherenceScope(tuple(by_value[value] for value in members)) for members in scopes),
        atomic_groups=tuple(_AtomicGroup((datum.id,)) for datum in targets),
    )


def _qualified_phase6(
    texts: tuple[str, ...],
    scopes: tuple[tuple[str, ...], ...],
    proposals: dict[str, tuple[_Proposal, ...]],
    *,
    connect_context: bool = False,
    context_texts: tuple[str, ...] = (),
) -> tuple[_Phase6Plan, _Backend, _Phase6Execution]:
    contract, capability = _contract_and_capability()
    plan = _compile_phase6_plan(
        _graph(texts, scopes, connect_context=connect_context, context_texts=context_texts),
        accounting_limits=_ACCOUNTING_LIMITS,
        context_contract=contract,
        capability=capability,
        mention_limits=_MENTION_LIMITS,
        profile_version=_Phase6ProfileVersion.SUBSTITUTE_V1,
    )
    assert isinstance(plan, _Phase6Plan)
    backend = _Backend(proposals)
    execution = _Phase6Runtime(backend).run(plan)
    assert len(execution.handoffs) == len(plan.components)
    return plan, backend, execution


def _phase7_module() -> ModuleType:
    module_name = "anonymizer.engine.execution.phase7_admission"
    assert importlib.util.find_spec(module_name) is not None, "the private Phase 7 compiler module is missing"
    return importlib.import_module(module_name)


def _compile_phase7(
    plan: _Phase6Plan,
    execution: _Phase6Execution,
    scopes: tuple[_CoherenceScope, ...],
    relations: tuple[object, ...] = (),
) -> object:
    module = _phase7_module()
    compiler = getattr(module, "_compile_phase7_plan", None)
    declarations_type = getattr(module, "_Phase7Declarations", None)
    assert callable(compiler), "the private Phase 7 compiler is missing"
    assert callable(declarations_type), "the private Phase 7 declaration grammar is missing"
    declarations = declarations_type(scopes, relations)
    return compiler(plan, execution.handoffs, declarations, _load_phase7_contract())


def _compile_raw(
    plan: _Phase6Plan,
    handoffs: tuple[object, ...],
    scopes: tuple[_CoherenceScope, ...],
    relations: tuple[object, ...] = (),
) -> object:
    module = _phase7_module()
    declarations = module._Phase7Declarations(scopes, relations)
    return module._compile_phase7_plan(plan, handoffs, declarations, _load_phase7_contract())


def _rejection_code(result: object) -> str:
    code = getattr(result, "code", None)
    assert code is not None, "Phase 7 malformed input did not return a typed rejection"
    value = getattr(code, "value", None)
    assert isinstance(value, str)
    return value


def _ids(plan: _Phase6Plan) -> tuple[_DatumId, ...]:
    return tuple(datum.id for datum in plan.accounting.datums)


def _required_type(module: ModuleType, name: str) -> Callable[..., Any]:
    value = getattr(module, name, None)
    assert callable(value), f"the private Phase 7 {name} type is missing"
    return value


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        pytest.param("empty", "empty_scope", id="empty-scope"),
        pytest.param("duplicate-scope", "duplicate_scope", id="duplicate-semantic-scope"),
        pytest.param("duplicate-member", "duplicate_scope_member", id="duplicate-member"),
        pytest.param("gap", "scope_coverage_gap", id="coverage-gap"),
        pytest.param("unknown", "unknown_scope_datum", id="unknown-datum"),
        pytest.param("overlap", "scope_overlap", id="partial-overlap"),
        pytest.param("nesting", "unsupported_scope_nesting", id="scope-nesting"),
    ],
)
def test_phase7_scope_admission_rejects_every_non_exact_partition(case: str, expected: str) -> None:
    plan, _backend, execution = _qualified_phase6(
        ("one", "two", "three", "four"),
        (("target-0",), ("target-1",), ("target-2",), ("target-3",)),
        {},
    )
    first, second, third, fourth = _ids(plan)
    scopes_by_case = {
        "empty": (_CoherenceScope(()), _CoherenceScope((first, second, third, fourth))),
        "duplicate-scope": (
            _CoherenceScope((first, second, third, fourth)),
            _CoherenceScope((fourth, third, second, first)),
        ),
        "duplicate-member": (
            _CoherenceScope((first, first, second)),
            _CoherenceScope((third, fourth)),
        ),
        "gap": (_CoherenceScope((first, second, third)),),
        "unknown": (
            _CoherenceScope((first, second, third)),
            _CoherenceScope((_DatumId("foreign"),)),
        ),
        "overlap": (
            _CoherenceScope((first, second)),
            _CoherenceScope((second, third, fourth)),
        ),
        "nesting": (
            _CoherenceScope((first, second, third, fourth)),
            _CoherenceScope((first, second)),
        ),
    }

    result = _compile_phase7(plan, execution, scopes_by_case[case])

    assert _rejection_code(result) == expected


@pytest.mark.parametrize(
    ("scopes", "expected"),
    [
        pytest.param(
            lambda ids: (_CoherenceScope(()), _CoherenceScope(())),
            "empty_scope",
            id="empty-before-duplicate",
        ),
        pytest.param(
            lambda ids: (_CoherenceScope((ids[0], ids[1], ids[1])),) * 2,
            "duplicate_scope",
            id="duplicate-scope-before-duplicate-member",
        ),
        pytest.param(
            lambda ids: (
                _CoherenceScope((ids[0], ids[0], _DatumId("foreign"))),
                _CoherenceScope((ids[1], ids[2], ids[3])),
            ),
            "duplicate_scope_member",
            id="duplicate-member-before-unknown",
        ),
        pytest.param(
            lambda ids: (
                _CoherenceScope((ids[0], ids[1], _DatumId("foreign"))),
                _CoherenceScope((ids[2],)),
            ),
            "unknown_scope_datum",
            id="unknown-before-gap",
        ),
        pytest.param(
            lambda ids: (_CoherenceScope((ids[0], ids[1])), _CoherenceScope((ids[1], ids[2]))),
            "scope_coverage_gap",
            id="gap-before-overlap",
        ),
    ],
)
def test_phase7_scope_rejection_precedence_is_fixed_under_declaration_permutation(
    scopes: Callable[[tuple[_DatumId, ...]], tuple[_CoherenceScope, ...]],
    expected: str,
) -> None:
    plan, _backend, execution = _qualified_phase6(
        ("one", "two", "three", "four"),
        (("target-0",), ("target-1",), ("target-2",), ("target-3",)),
        {},
    )
    declared = scopes(_ids(plan))
    assert isinstance(declared, tuple)

    forward = _compile_phase7(plan, execution, declared)
    reverse = _compile_phase7(plan, execution, tuple(reversed(declared)))

    assert _rejection_code(forward) == expected
    assert _rejection_code(reverse) == expected


def test_phase7_compiler_requires_exact_phase6_scope_cluster_role_and_terminal_handoffs() -> None:
    plan, _backend, execution = _qualified_phase6(
        ("Alice", "555-0100"),
        (("target-0",), ("target-1",)),
        {
            "target-0": (_Proposal("Alice", "first_name", "person"),),
            "target-1": (_Proposal("555-0100", "phone_number", "phone"),),
        },
    )
    module = _phase7_module()
    declared = (_CoherenceScope(tuple(reversed(_ids(plan)))),)

    result = _compile_phase7(plan, execution, declared)

    assert isinstance(result, module._Phase7Plan)
    assert len(result.manifests) == 1
    assert tuple(member.value for member in result.manifests[0].members) == ("target-0", "target-1")
    assert tuple(task.stage.value for task in result.scope_tasks) == ("phase7-plan",)
    assert tuple(task.stage.value for task in result.application_tasks) == ("phase7-apply", "phase7-apply")
    assert tuple(
        task.subject.datum_id for task in result.application_tasks if isinstance(task.subject, _DatumTaskSubject)
    ) == (
        _DatumId("target-0"),
        _DatumId("target-1"),
    )
    expected_phase7_predecessors = {
        _TaskPredecessor(result.scope_tasks[0], result.application_tasks[0]),
        _TaskPredecessor(result.scope_tasks[0], result.application_tasks[1]),
    }
    phase7_tasks = {*result.scope_tasks, *result.application_tasks}
    assert {
        predecessor
        for predecessor in result.accounting.task_predecessors
        if predecessor.prerequisite in phase7_tasks or predecessor.dependent in phase7_tasks
    } == expected_phase7_predecessors
    assert module._is_admitted_phase7_plan(result)

    extra = _TaskPredecessor(result.application_tasks[0], result.application_tasks[1])
    expanded_accounting = result.accounting.with_task_predecessors((*result.accounting.task_predecessors, extra))
    assert not module._has_exact_application_predecessors(replace(result, accounting=expanded_accounting))


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param("missing", id="missing-handoff"),
        pytest.param("duplicate", id="duplicate-handoff"),
        pytest.param("foreign", id="foreign-handoff"),
        pytest.param("terminal", id="terminal-evidence-tampering"),
        pytest.param("role", id="role-result-tampering"),
    ],
)
def test_phase7_compiler_rejects_inexact_phase6_handoff_equality(mutation: str) -> None:
    plan, _backend, execution = _qualified_phase6(
        ("Alice",),
        (("target-0",),),
        {"target-0": (_Proposal("Alice", "first_name", "person"),)},
    )
    handoff = execution.handoffs[0]
    handoffs: tuple[object, ...]
    if mutation == "missing":
        handoffs = ()
    elif mutation == "duplicate":
        handoffs = (handoff, handoff)
    elif mutation == "foreign":
        _foreign_plan, _foreign_backend, foreign = _qualified_phase6(
            ("Mallory",),
            (("target-0",),),
            {"target-0": (_Proposal("Mallory", "first_name", "person"),)},
        )
        handoffs = foreign.handoffs
    elif mutation == "terminal":
        handoffs = (replace(handoff, terminal_evidence=replace(handoff.terminal_evidence, datum_ids=())),)
    else:
        resolved_mention = handoff.resolved.mentions[0]
        role_result = resolved_mention.role_result
        assert isinstance(role_result, _ClassifiedRole)
        changed_role = replace(role_result, role=replace(role_result.role, value="email_address"))
        changed_resolved = replace(handoff.resolved, mentions=(replace(resolved_mention, role_result=changed_role),))
        handoffs = (replace(handoff, resolved=changed_resolved),)

    result = _compile_raw(plan, handoffs, plan.coherence_scopes)

    assert _rejection_code(result) == "phase6_handoff_mismatch"


def _person_relation_fixture() -> tuple[_Phase6Plan, _Backend, _Phase6Execution, object]:
    plan, backend, execution = _qualified_phase6(
        ("Alice Adams alice@example.com",),
        (("target-0",),),
        {
            "target-0": (
                _Proposal("Alice", "first_name", "person"),
                _Proposal("Adams", "last_name", "person"),
                _Proposal("alice@example.com", "email", "person"),
            )
        },
    )
    clusters = execution.handoffs[0].resolved.clustered.clusters
    assert len(clusters) == 1
    return plan, backend, execution, clusters[0].id


def test_phase7_pre_slot_selectors_resolve_exact_cluster_role_targets_before_slot_identity() -> None:
    plan, _backend, execution, cluster = _person_relation_fixture()
    module = _phase7_module()
    selector = _required_type(module, "_ClusterRoleSelector")
    relation_type = _required_type(module, "_RelationDeclaration")
    relation = relation_type(
        "email_from_name/v1",
        (
            selector("cluster_role/v1", cluster, "person_given_name"),
            selector("cluster_role/v1", cluster, "person_family_name"),
        ),
        selector("cluster_role/v1", cluster, "email_address"),
    )

    result = _compile_phase7(plan, execution, plan.coherence_scopes, (relation,))

    assert isinstance(result, module._Phase7Plan)
    assert len(result.manifests[0].relations) == 1
    assert len(result.manifests[0].relations[0].upstream) == 2
    assert result.manifests[0].relations[0].downstream in {slot.id for slot in result.manifests[0].slots}


def test_phase7_selector_rejects_a_missing_cluster_role_target() -> None:
    plan, _backend, execution, cluster = _person_relation_fixture()
    module = _phase7_module()
    selector = _required_type(module, "_ClusterRoleSelector")
    relation_type = _required_type(module, "_RelationDeclaration")
    relation = relation_type(
        "email_from_name/v1",
        (selector("cluster_role/v1", cluster, "user_name"),),
        selector("cluster_role/v1", cluster, "email_address"),
    )

    result = _compile_phase7(plan, execution, plan.coherence_scopes, (relation,))

    assert _rejection_code(result) == "selector_missing"


def test_phase7_selector_rejects_ambiguous_pre_slot_targets() -> None:
    _plan, _backend, execution, cluster = _person_relation_fixture()
    module = _phase7_module()
    scope = module._Phase7ScopeId()
    mention_id = execution.handoffs[0].resolved.mentions[0].mention.id
    pre_slot = _required_type(module, "_PreSlot")
    selector_type = _required_type(module, "_ClusterRoleSelector")
    candidate = pre_slot(scope, cluster, "person_given_name", "format", "mask", (mention_id,))
    selector = selector_type("cluster_role/v1", cluster, "person_given_name")

    result = module._resolve_selector(selector, (candidate, candidate))

    assert _rejection_code(result) == "selector_ambiguous"


def test_phase7_selector_never_resolves_by_an_opaque_slot_id() -> None:
    plan, _backend, execution, cluster = _person_relation_fixture()
    module = _phase7_module()
    initial = _compile_phase7(plan, execution, plan.coherence_scopes)
    assert isinstance(initial, module._Phase7Plan)
    assert initial.manifests[0].slots, "qualified Phase 6 roles did not materialize private slots"
    opaque_slot_id = initial.manifests[0].slots[0].id
    selector_type = _required_type(module, "_ClusterRoleSelector")
    relation_type = _required_type(module, "_RelationDeclaration")
    selector = selector_type(
        "cluster_role/v1",
        opaque_slot_id,
        "person_given_name",
    )
    relation = relation_type(
        "email_from_name/v1",
        (selector,),
        selector_type("cluster_role/v1", cluster, "email_address"),
    )

    result = _compile_phase7(plan, execution, plan.coherence_scopes, (relation,))

    assert _rejection_code(result) == "selector_missing"


def test_phase7_slot_identity_is_one_opaque_capability_per_scope_cluster_role_key() -> None:
    plan, _backend, execution = _qualified_phase6(
        ("Alice Alicia",),
        (("target-0",),),
        {
            "target-0": (
                _Proposal("Alice", "first_name", "person"),
                _Proposal("Alicia", "first_name", "person"),
            )
        },
    )
    module = _phase7_module()

    result = _compile_phase7(plan, execution, plan.coherence_scopes)

    assert isinstance(result, module._Phase7Plan)
    assert len(result.manifests[0].slots) == 1
    slot = result.manifests[0].slots[0]
    assert slot.role == "person_given_name"
    assert len(slot.mention_ids) == 2
    assert type(slot.id).__name__ == "_ReplacementSlotId"
    assert not hasattr(slot.id, "value")


def test_phase7_equal_text_and_labels_in_distinct_clusters_create_distinct_slots() -> None:
    plan, _backend, execution = _qualified_phase6(
        ("Alice", "Alice"),
        (("target-0",), ("target-1",)),
        {
            "target-0": (_Proposal("Alice", "first_name", "first-person"),),
            "target-1": (_Proposal("Alice", "first_name", "second-person"),),
        },
    )
    module = _phase7_module()
    declared = (_CoherenceScope(_ids(plan)),)

    result = _compile_phase7(plan, execution, declared)

    assert isinstance(result, module._Phase7Plan)
    slots = result.manifests[0].slots
    assert len(slots) == 2
    assert slots[0].id is not slots[1].id
    assert slots[0].cluster_id is not slots[1].cluster_id
    assert slots[0].role == slots[1].role == "person_given_name"


def test_phase7_materializes_every_required_distinct_pair_once_in_deterministic_order() -> None:
    plan, _backend, execution = _qualified_phase6(
        ("Alice Adams alice@example.com 555-0100",),
        (("target-0",),),
        {
            "target-0": (
                _Proposal("Alice", "first_name", "person"),
                _Proposal("Adams", "last_name", "person"),
                _Proposal("alice@example.com", "email", "person"),
                _Proposal("555-0100", "phone_number", "phone"),
            )
        },
    )
    module = _phase7_module()

    forward = _compile_phase7(plan, execution, plan.coherence_scopes)
    reverse = _compile_phase7(plan, execution, tuple(reversed(plan.coherence_scopes)))

    assert isinstance(forward, module._Phase7Plan)
    assert isinstance(reverse, module._Phase7Plan)
    manifest = forward.manifests[0]
    assert len(manifest.slots) == 4
    assert len(manifest.required_pairs) == 6
    slot_position = {slot.id: index for index, slot in enumerate(manifest.slots)}
    assert tuple((slot_position[pair.left], slot_position[pair.right]) for pair in manifest.required_pairs) == (
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    )
    assert tuple(slot.role for slot in forward.manifests[0].slots) == tuple(
        slot.role for slot in reverse.manifests[0].slots
    )


def test_phase7_count_limits_accept_exact_and_reject_one_over() -> None:
    exact_plan, _backend, exact_execution = _qualified_phase6(
        ("one", "two", "three", "four"),
        (("target-0",), ("target-1",), ("target-2",), ("target-3",)),
        {},
    )
    over_plan, _backend, over_execution = _qualified_phase6(
        ("one", "two", "three", "four", "five"),
        (("target-0",), ("target-1",), ("target-2",), ("target-3",), ("target-4",)),
        {},
    )
    module = _phase7_module()

    exact = _compile_phase7(exact_plan, exact_execution, (_CoherenceScope(_ids(exact_plan)),))
    over = _compile_phase7(over_plan, over_execution, (_CoherenceScope(_ids(over_plan)),))

    assert isinstance(exact, module._Phase7Plan)
    assert _rejection_code(over) == "limit_exceeded"


def test_phase7_scope_count_limit_accepts_exact_and_precedes_partition_rejections_one_over() -> None:
    plan, _backend, execution = _qualified_phase6(
        ("one", "two", "three"),
        (("target-0",), ("target-1",), ("target-2",)),
        {},
    )
    first, second, third = _ids(plan)
    module = _phase7_module()

    exact = _compile_phase7(
        plan,
        execution,
        (_CoherenceScope((first, second)), _CoherenceScope((third,))),
    )
    one_over_and_empty = _compile_phase7(
        plan,
        execution,
        (_CoherenceScope((first,)), _CoherenceScope((second, third)), _CoherenceScope(())),
    )

    assert isinstance(exact, module._Phase7Plan)
    assert _rejection_code(one_over_and_empty) == "limit_exceeded"


@pytest.mark.parametrize(
    ("exact_count", "over_count", "cluster_mode"),
    [
        pytest.param(3, 4, "distinct", id="clusters"),
        pytest.param(6, 7, "shared", id="mentions"),
    ],
)
def test_phase7_cluster_and_mention_limits_accept_exact_and_reject_one_over(
    exact_count: int,
    over_count: int,
    cluster_mode: str,
) -> None:
    def compile_count(count: int) -> object:
        sources = tuple(chr(ord("A") + index) for index in range(count))
        proposals = tuple(
            _Proposal(source, "first_name", source if cluster_mode == "distinct" else "shared") for source in sources
        )
        plan, _backend, execution = _qualified_phase6(
            (" ".join(sources),),
            (("target-0",),),
            {"target-0": proposals},
        )
        return _compile_phase7(plan, execution, plan.coherence_scopes)

    module = _phase7_module()
    exact = compile_count(exact_count)
    over = compile_count(over_count)

    assert isinstance(exact, module._Phase7Plan)
    assert _rejection_code(over) == "limit_exceeded"


def test_phase7_slot_and_pair_limits_accept_exact_and_reject_one_over() -> None:
    exact_plan, _backend, exact_execution = _qualified_phase6(
        ("Alice Adams alice@example.com 555-0100",),
        (("target-0",),),
        {
            "target-0": (
                _Proposal("Alice", "first_name", "person"),
                _Proposal("Adams", "last_name", "person"),
                _Proposal("alice@example.com", "email", "person"),
                _Proposal("555-0100", "phone_number", "phone"),
            )
        },
    )
    over_plan, _backend, over_execution = _qualified_phase6(
        ("Alice Adams alice@example.com 555-0100 alice_user",),
        (("target-0",),),
        {
            "target-0": (
                _Proposal("Alice", "first_name", "person"),
                _Proposal("Adams", "last_name", "person"),
                _Proposal("alice@example.com", "email", "person"),
                _Proposal("555-0100", "phone_number", "phone"),
                _Proposal("alice_user", "user_name", "account"),
            )
        },
    )
    module = _phase7_module()

    exact = _compile_phase7(exact_plan, exact_execution, exact_plan.coherence_scopes)
    over = _compile_phase7(over_plan, over_execution, over_plan.coherence_scopes)

    assert isinstance(exact, module._Phase7Plan)
    assert len(exact.manifests[0].required_pairs) == 6
    assert _rejection_code(over) == "limit_exceeded"


def test_phase7_original_value_byte_limit_accepts_exact_and_rejects_one_over() -> None:
    def compile_size(size: int) -> object:
        source = "A" * size
        plan, _backend, execution = _qualified_phase6(
            (source,),
            (("target-0",),),
            {"target-0": (_Proposal(source, "first_name", "person"),)},
        )
        return _compile_phase7(plan, execution, plan.coherence_scopes)

    module = _phase7_module()
    exact = compile_size(256)
    over = compile_size(257)

    assert isinstance(exact, module._Phase7Plan)
    assert _rejection_code(over) == "limit_exceeded"


@pytest.mark.parametrize(
    ("exact_context", "over_context"),
    [
        pytest.param(("a", "b", "c", "d"), ("a", "b", "c", "d", "e"), id="fragment-count"),
        pytest.param(("A" * 4096,), ("A" * 4097,), id="fragment-bytes"),
        pytest.param(
            ("A" * 2731, "B" * 2731, "C" * 2730),
            ("A" * 2731, "B" * 2731, "C" * 2731),
            id="scope-context-bytes",
        ),
    ],
)
def test_phase7_context_limits_accept_exact_and_reject_one_over(
    exact_context: tuple[str, ...],
    over_context: tuple[str, ...],
) -> None:
    def compile_context(context: tuple[str, ...]) -> object:
        plan, _backend, execution = _qualified_phase6(
            ("target",),
            (("target-0",),),
            {},
            context_texts=context,
        )
        return _compile_phase7(plan, execution, plan.coherence_scopes)

    module = _phase7_module()
    exact = compile_context(exact_context)
    over = compile_context(over_context)

    assert isinstance(exact, module._Phase7Plan)
    assert _rejection_code(over) == "limit_exceeded"


def test_phase7_relation_limit_accepts_exact_and_rejects_one_over() -> None:
    plan, _backend, execution, cluster = _person_relation_fixture()
    module = _phase7_module()
    relation = module._RelationDeclaration(
        "email_from_name/v1",
        (module._ClusterRoleSelector("cluster_role/v1", cluster, "person_given_name"),),
        module._ClusterRoleSelector("cluster_role/v1", cluster, "email_address"),
    )

    exact = _compile_phase7(plan, execution, plan.coherence_scopes, (relation,) * 4)
    over = _compile_phase7(plan, execution, plan.coherence_scopes, (relation,) * 5)

    assert isinstance(exact, module._Phase7Plan)
    assert len(exact.manifests[0].relations) == 4
    assert _rejection_code(over) == "limit_exceeded"


@pytest.mark.parametrize("failure", ["cross-scope", "role-mismatch", "unknown-relation", "unknown-selector"])
def test_phase7_relation_declarations_fail_closed(failure: str) -> None:
    plan, _backend, execution = _qualified_phase6(
        ("Alice alice@example.com", "Bob bob@example.com"),
        (("target-0",), ("target-1",)),
        {
            "target-0": (
                _Proposal("Alice", "first_name", "alice"),
                _Proposal("alice@example.com", "email", "alice"),
            ),
            "target-1": (
                _Proposal("Bob", "first_name", "bob"),
                _Proposal("bob@example.com", "email", "bob"),
            ),
        },
    )
    module = _phase7_module()
    first_cluster = execution.handoffs[0].resolved.clustered.clusters[0].id
    second_cluster = execution.handoffs[1].resolved.clustered.clusters[0].id
    selector_version = "cluster_role/v2" if failure == "unknown-selector" else "cluster_role/v1"
    upstream_role = "email_address" if failure == "role-mismatch" else "person_given_name"
    downstream_cluster = second_cluster if failure == "cross-scope" else first_cluster
    downstream_role = "person_given_name" if failure == "role-mismatch" else "email_address"
    relation = module._RelationDeclaration(
        "email_from_name/v2" if failure == "unknown-relation" else "email_from_name/v1",
        (module._ClusterRoleSelector(selector_version, first_cluster, upstream_role),),
        module._ClusterRoleSelector("cluster_role/v1", downstream_cluster, downstream_role),
    )

    result = _compile_phase7(plan, execution, plan.coherence_scopes, (relation,))

    expected = {
        "cross-scope": "cross_scope_relation",
        "role-mismatch": "relation_role_mismatch",
        "unknown-relation": "unsupported_relation",
        "unknown-selector": "unsupported_selector",
    }
    assert _rejection_code(result) == expected[failure]


def test_phase7_plan_and_nested_manifests_reject_tampering_and_serialization() -> None:
    plan, _backend, execution, _cluster = _person_relation_fixture()
    module = _phase7_module()
    result = _compile_phase7(plan, execution, plan.coherence_scopes)
    assert isinstance(result, module._Phase7Plan)
    manifest = result.manifests[0]
    slot = manifest.slots[0]
    changed_slot = replace(slot, role="email_address")
    changed_manifest = replace(manifest, slots=(changed_slot, *manifest.slots[1:]))

    assert not module._is_admitted_phase7_plan(replace(result, manifests=(changed_manifest,)))
    assert not module._is_admitted_phase7_plan(replace(result, manifests=(replace(manifest, members=()),)))
    assert not module._is_admitted_phase7_plan(replace(result, application_tasks=()))
    with pytest.raises(FrozenInstanceError):
        manifest.members = ()
    with pytest.raises(TypeError, match="not serializable"):
        pickle.dumps(result)


def test_phase7_manifest_shape_is_invariant_to_scope_member_handoff_and_relation_permutations() -> None:
    plan, _backend, execution = _qualified_phase6(
        ("Alice alice@example.com", "Bob bob@example.com"),
        (("target-0",), ("target-1",)),
        {
            "target-0": (
                _Proposal("Alice", "first_name", "alice"),
                _Proposal("alice@example.com", "email", "alice"),
            ),
            "target-1": (
                _Proposal("Bob", "first_name", "bob"),
                _Proposal("bob@example.com", "email", "bob"),
            ),
        },
    )
    module = _phase7_module()
    selectors = []
    for handoff in execution.handoffs:
        cluster = handoff.resolved.clustered.clusters[0].id
        selectors.append(
            module._RelationDeclaration(
                "email_from_name/v1",
                (module._ClusterRoleSelector("cluster_role/v1", cluster, "person_given_name"),),
                module._ClusterRoleSelector("cluster_role/v1", cluster, "email_address"),
            )
        )
    forward_scope = (_CoherenceScope(_ids(plan)),)
    reverse_scope = (_CoherenceScope(tuple(reversed(_ids(plan)))),)

    forward = _compile_raw(plan, execution.handoffs, forward_scope, tuple(selectors))
    reverse = _compile_raw(plan, tuple(reversed(execution.handoffs)), reverse_scope, tuple(reversed(selectors)))

    assert isinstance(forward, module._Phase7Plan)
    assert isinstance(reverse, module._Phase7Plan)
    assert _normalized_manifest(forward.manifests[0]) == _normalized_manifest(reverse.manifests[0])


def _normalized_manifest(manifest: object) -> tuple[object, ...]:
    slots = getattr(manifest, "slots")
    slot_position = {slot.id: index for index, slot in enumerate(slots)}
    return (
        tuple(member.value for member in getattr(manifest, "members")),
        tuple((slot.role, len(slot.mention_ids)) for slot in slots),
        tuple((slot_position[pair.left], slot_position[pair.right]) for pair in getattr(manifest, "required_pairs")),
        tuple(
            (
                relation.version,
                tuple(slot_position[slot_id] for slot_id in relation.upstream),
                slot_position[relation.downstream],
            )
            for relation in getattr(manifest, "relations")
        ),
    )


def test_phase7_empty_manifest_and_compilation_have_zero_runtime_effects() -> None:
    plan, backend, execution = _qualified_phase6(
        ("no supported mention",),
        (("target-0",),),
        {},
    )
    module = _phase7_module()
    calls_before = tuple(backend.calls)

    result = _compile_phase7(plan, execution, plan.coherence_scopes)

    assert isinstance(result, module._Phase7Plan)
    assert result.manifests[0].slots == ()
    assert result.manifests[0].required_pairs == ()
    assert result.manifests[0].relations == ()
    assert tuple(backend.calls) == calls_before
    assert backend.planner_effect_count == 0
    assert not {
        "_AccountingLedger",
        "_observe_context_boundary",
        "_build_context_workframes",
        "_apply_redact_patches",
        "NddAdapter",
        "DataDesigner",
    } & set(vars(module))
