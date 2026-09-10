# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import pickle
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from anonymizer.engine.execution import phase6_plan as phase6_plan_module
from anonymizer.engine.execution import phase6_runtime as phase6_runtime_module
from anonymizer.engine.execution import role_policy as role_policy_module
from anonymizer.engine.execution.accounting_outcomes import _AccountingResult, _TaskFailed
from anonymizer.engine.execution.accounting_plan import _AccountingLimits
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
    _AnchoredMention,
    _DetectedGraph,
    _MentionId,
    _MentionLimits,
    _MentionProvenance,
    _MentionTarget,
    _MentionTargetToken,
    _ValidationDecision,
    _ValidationDecisionKind,
)
from anonymizer.engine.execution.mention_resolution import (
    _ClusteredGraph,
    _ClusterId,
    _EntityCluster,
    _EvidenceVersion,
    _SameSubjectEvidence,
)
from anonymizer.engine.execution.phase6_plan import _Phase6Plan, _Phase6ProfileVersion, _Phase6Rejected
from anonymizer.engine.execution.phase6_runtime import (
    _CandidateProposal,
    _Phase6AugmentationWork,
    _Phase6CandidateWork,
    _Phase6Execution,
    _Phase6ResolverWork,
    _Phase6Runtime,
    _Phase6ValidationWork,
)
from anonymizer.engine.execution.role_policy import (
    _ClassifiedRole,
    _ReplacementRole,
    _ResolvedGraph,
    _RolePolicy,
    _RolePolicyRejected,
    _RolePolicyVersion,
)

_SUBSTITUTE_DIGEST = "c27580bd2cc4051bdd11b63a91391f8995bdef1ed2052534623cdd3160318ef8"
_SUBSTITUTE_POLICY_VERSION = "phase6-substitute-role-policy/v1"
_POLICY_PATH = (
    Path(__file__).parents[3] / "src" / "anonymizer" / "engine" / "execution" / "phase6_substitute_role_policy.json"
)
_ACCOUNTING_LIMITS = _AccountingLimits(8, 128, 512, max_stages=8)
_MENTION_LIMITS = _MentionLimits(8, 8, 64, 128)


class _Resource:
    def __init__(self, payloads: dict[str, str], name: str | None = None) -> None:
        self._payloads = payloads
        self._name = name

    def joinpath(self, name: str) -> _Resource:
        return _Resource(self._payloads, name)

    def read_text(self, *, encoding: str) -> str:
        assert encoding == "utf-8"
        assert self._name is not None
        return self._payloads[self._name]


class _Backend:
    def __init__(self, label: str, *, join_targets: bool = False) -> None:
        self.label = label
        self.join_targets = join_targets
        self.calls: list[str] = []
        self.planner_effect_count = 0

    def context_capability(self) -> _ContextBackendCapability:
        return _contract_and_capability()[1]

    def detect(self, work: _Phase6CandidateWork) -> tuple[_CandidateProposal, ...]:
        self.calls.append("detect")
        return (_CandidateProposal(0, len(work.target.text), work.target.text, self.label),)

    def augment(self, work: _Phase6AugmentationWork) -> tuple[_CandidateProposal, ...]:
        self.calls.append("augment")
        return ()

    def validate(self, work: _Phase6ValidationWork) -> tuple[_ValidationDecision, ...]:
        self.calls.append("validate")
        return tuple(_ValidationDecision(item.token, _ValidationDecisionKind.KEEP) for item in work.candidates)

    def resolve(self, work: _Phase6ResolverWork) -> tuple[_SameSubjectEvidence, ...]:
        self.calls.append("resolve")
        if not self.join_targets or work.owner.datum_id.value != "target-a":
            return ()
        owned = next(item for item in work.eligible_mentions if item.target_datum_id == work.owner.datum_id)
        foreign = next(item for item in work.eligible_mentions if item.target_datum_id != work.owner.datum_id)
        return (_SameSubjectEvidence(work.owner.token, owned.id, foreign.id, _EvidenceVersion.V1),)

    def plan(self, _work: object) -> None:
        self.planner_effect_count += 1

    def close_phase6(self) -> bool:
        return True


def _contract_and_capability() -> tuple[_ContextExecutionContract, _ContextBackendCapability]:
    limits = _ContextLimits(4, 128, 8, 512)
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


def _graph(*, two_targets: bool = False) -> _ProtectionGraph:
    target_a = _TextDatum(_DatumId("target-a"), "Alice", _DatumPurpose.TARGET)
    targets = (target_a,)
    if two_targets:
        targets = (*targets, _TextDatum(_DatumId("target-b"), "Alicia", _DatumPurpose.TARGET))
    ids = tuple(target.id for target in targets)
    return _ProtectionGraph(
        datums=targets,
        links=(),
        context_scopes=tuple(
            _ContextScope(target.id, tuple(candidate.id for candidate in targets if candidate.id != target.id))
            for target in targets
        ),
        coherence_scopes=tuple(_CoherenceScope((datum_id,)) for datum_id in ids),
        atomic_groups=(_AtomicGroup(ids),),
    )


def _substitute_profile() -> _Phase6ProfileVersion:
    profile = getattr(_Phase6ProfileVersion, "SUBSTITUTE_V1", None)
    assert isinstance(profile, _Phase6ProfileVersion), "the private Phase 6 Substitute profile is missing"
    return profile


def _compile_substitute(graph: _ProtectionGraph) -> _Phase6Plan:
    contract, capability = _contract_and_capability()
    result = phase6_plan_module._compile_phase6_plan(
        graph,
        accounting_limits=_ACCOUNTING_LIMITS,
        context_contract=contract,
        capability=capability,
        mention_limits=_MENTION_LIMITS,
        profile_version=_substitute_profile(),
    )
    assert isinstance(result, _Phase6Plan)
    return result


def _valid_execution(*, two_targets: bool = False) -> tuple[_Phase6Plan, _Backend, _Phase6Execution]:
    plan = _compile_substitute(_graph(two_targets=two_targets))
    backend = _Backend("first_name")
    return plan, backend, _Phase6Runtime(backend).run(plan)


def _foreign_resolved_graph(policy: _RolePolicy) -> _ResolvedGraph:
    token = _MentionTargetToken()
    target = _MentionTarget(token, _DatumId("foreign"), "Mallory")
    mention = _AnchoredMention(
        _MentionId(),
        target.datum_id,
        0,
        len(target.text),
        target.text,
        "first_name",
        _MentionProvenance.SPAN_DETECTOR,
    )
    clustered = _ClusteredGraph(
        _DetectedGraph((target,), (mention,)),
        (_EntityCluster(_ClusterId(), (mention.id,), ()),),
        (),
    )
    result = role_policy_module._classify_roles(clustered, policy)
    assert isinstance(result, _ResolvedGraph)
    return result


def test_substitute_policy_loader_requires_the_exact_p0_result_version_policy_version_and_digest() -> None:
    loader = getattr(role_policy_module, "_load_substitute_role_policy", None)
    assert callable(loader), "the exact private Substitute role-policy loader is missing"

    result = loader()

    assert isinstance(result, _RolePolicy)
    assert result.result_version is _RolePolicyVersion.V1
    assert result.policy_version == _SUBSTITUTE_POLICY_VERSION
    assert result.digest == _SUBSTITUTE_DIGEST
    assert tuple((label, role.value) for label, role in result.mappings) == (
        ("email", "email_address"),
        ("fax_number", "fax_number"),
        ("first_name", "person_given_name"),
        ("last_name", "person_family_name"),
        ("phone_number", "voice_phone_number"),
        ("user_name", "user_name"),
    )


def test_substitute_policy_loader_rejects_a_wrong_policy_version(monkeypatch: pytest.MonkeyPatch) -> None:
    loader = getattr(role_policy_module, "_load_substitute_role_policy", None)
    assert callable(loader), "the exact private Substitute role-policy loader is missing"
    payload = json.loads(_POLICY_PATH.read_text(encoding="utf-8"))
    payload["version"] = "phase6-substitute-role-policy/v2"
    monkeypatch.setattr(
        role_policy_module,
        "files",
        lambda _package: _Resource({"phase6_substitute_role_policy.json": json.dumps(payload)}),
    )

    assert isinstance(loader(), _RolePolicyRejected)


def test_substitute_plan_rejects_a_wrong_policy_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    loader = getattr(role_policy_module, "_load_substitute_role_policy", None)
    assert callable(loader), "the exact private Substitute role-policy loader is missing"
    policy = loader()
    assert isinstance(policy, _RolePolicy)
    monkeypatch.setattr(
        phase6_plan_module,
        "_load_substitute_role_policy",
        lambda: replace(policy, digest="0" * 64),
        raising=False,
    )
    contract, capability = _contract_and_capability()

    result = phase6_plan_module._compile_phase6_plan(
        _graph(),
        accounting_limits=_ACCOUNTING_LIMITS,
        context_contract=contract,
        capability=capability,
        mention_limits=_MENTION_LIMITS,
        profile_version=_substitute_profile(),
    )

    assert isinstance(result, _Phase6Rejected)


def test_future_substitute_policy_releases_exact_classified_roles_with_zero_planner_effects() -> None:
    plan, backend, execution = _valid_execution()

    assert plan.profile_version is _substitute_profile()
    assert tuple(stage.value for stage in plan.accounting.stages) == (
        "detect",
        "augment",
        "validate",
        "finalize",
        "resolve",
        "classify",
    )
    assert plan.role_policy.result_version is _RolePolicyVersion.V1
    assert plan.role_policy.policy_version == _SUBSTITUTE_POLICY_VERSION
    assert plan.role_policy.digest == _SUBSTITUTE_DIGEST
    assert execution.released == ()
    assert len(execution.handoffs) == 1
    handoff = execution.handoffs[0]
    assert handoff.result_version == _RolePolicyVersion.V1.value
    assert handoff.policy_version == _SUBSTITUTE_POLICY_VERSION
    assert handoff.policy_digest == _SUBSTITUTE_DIGEST
    role_results = tuple(item.role_result for item in handoff.resolved.mentions)
    assert all(isinstance(item, _ClassifiedRole) for item in role_results)
    assert tuple(item.role.value for item in role_results if isinstance(item, _ClassifiedRole)) == (
        "person_given_name",
    )
    assert tuple(item.mention for item in handoff.resolved.mentions) == handoff.resolved.clustered.detected.mentions
    assert tuple(item.cluster_id for item in handoff.resolved.mentions) == tuple(
        cluster.id for cluster in handoff.resolved.clustered.clusters
    )
    assert tuple(datum_id.value for datum_id in handoff.terminal_evidence.datum_ids) == ("target-a",)
    assert tuple(task.stage.value for task in handoff.terminal_evidence.tasks) == (
        "detect",
        "augment",
        "validate",
        "finalize",
        "resolve",
        "classify",
    )
    assert backend.planner_effect_count == 0
    assert backend.calls == ["detect", "augment", "validate", "resolve"]
    assert phase6_runtime_module._is_admitted_substitute_handoff(handoff, plan)
    with pytest.raises((AttributeError, TypeError)):
        setattr(handoff, "policy_digest", "changed")
    with pytest.raises(TypeError, match="not serializable"):
        pickle.dumps(handoff)


def test_substitute_handoff_rejects_an_unsupported_role() -> None:
    plan = _compile_substitute(_graph())

    execution = _Phase6Runtime(_Backend("custom_secret")).run(plan)

    assert execution.handoffs == ()
    assert any(
        isinstance(outcome, _TaskFailed) and outcome.task.stage.value == "classify"
        for outcome in execution.accounting.tasks
    )


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param("wrong-result-version", id="wrong-result-version"),
        pytest.param("wrong-policy-version", id="wrong-policy-version"),
        pytest.param("wrong-policy-digest", id="wrong-policy-digest"),
        pytest.param("mismatched-role", id="mismatched-role"),
        pytest.param("missing-role-result", id="missing-role-result"),
        pytest.param("duplicate-role-result", id="duplicate-role-result"),
        pytest.param("stale-graph", id="stale-graph"),
        pytest.param("foreign-graph", id="foreign-graph"),
        pytest.param("unresolved-graph", id="unresolved-graph"),
    ],
)
def test_substitute_runtime_rejects_inexact_classified_graphs(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    plan = _compile_substitute(_graph())
    classify = role_policy_module._classify_roles

    def _mutated_classification(clustered: _ClusteredGraph, policy: _RolePolicy) -> object:
        resolved = classify(clustered, policy)
        assert isinstance(resolved, _ResolvedGraph)
        if mutation == "wrong-result-version":
            return replace(resolved, policy_version=cast(_RolePolicyVersion, "phase6-role-result/v2"))
        if mutation == "wrong-policy-version":
            return replace(resolved, source_policy_version="phase6-substitute-role-policy/v2")
        if mutation == "wrong-policy-digest":
            return replace(resolved, policy_digest="0" * 64)
        if mutation == "mismatched-role":
            role_result = resolved.mentions[0].role_result
            assert isinstance(role_result, _ClassifiedRole)
            wrong = replace(role_result, role=_ReplacementRole("email_address"))
            return replace(resolved, mentions=(replace(resolved.mentions[0], role_result=wrong),))
        if mutation == "missing-role-result":
            return replace(resolved, mentions=())
        if mutation == "duplicate-role-result":
            return replace(resolved, mentions=(resolved.mentions[0], resolved.mentions[0]))
        if mutation == "stale-graph":
            stale = replace(resolved.clustered, detected=replace(resolved.clustered.detected, mentions=()))
            return replace(resolved, clustered=stale)
        if mutation == "foreign-graph":
            return _foreign_resolved_graph(policy)
        return role_policy_module._RolePolicyRejected(role_policy_module._RolePolicyRejectionCode.UNSUPPORTED_ROLE)

    monkeypatch.setattr(phase6_runtime_module, "_classify_roles", _mutated_classification)

    execution = _Phase6Runtime(_Backend("first_name")).run(plan)

    assert execution.handoffs == ()
    assert any(
        isinstance(outcome, _TaskFailed) and outcome.task.stage.value == "classify"
        for outcome in execution.accounting.tasks
    )


@pytest.mark.parametrize(
    "mutation",
    ["missing-task", "duplicate-task", "missing-datum", "duplicate-datum", "missing-group", "duplicate-group"],
)
def test_substitute_handoff_rejects_incomplete_or_duplicate_terminal_evidence(mutation: str) -> None:
    plan, _backend, execution = _valid_execution()
    assert len(execution.handoffs) == 1
    builder = getattr(phase6_runtime_module, "_build_substitute_handoffs", None)
    assert callable(builder), "the terminal-evidence handoff builder is missing"
    accounting = cast(_AccountingResult[object], execution.accounting)
    corrupted = accounting
    if mutation == "missing-task":
        corrupted = replace(accounting, tasks=accounting.tasks[1:])
    elif mutation == "duplicate-task":
        corrupted = replace(accounting, tasks=(*accounting.tasks, accounting.tasks[0]))
    elif mutation == "missing-datum":
        corrupted = replace(accounting, datums=())
    elif mutation == "duplicate-datum":
        corrupted = replace(accounting, datums=(*accounting.datums, *accounting.datums))
    elif mutation == "missing-group":
        corrupted = replace(accounting, groups=())
    else:
        corrupted = replace(accounting, groups=(*accounting.groups, *accounting.groups))

    result = builder(plan, corrupted)

    assert type(result).__name__ == "_Phase6HandoffRejected"


def test_substitute_handoff_rejects_cross_scope_cluster_evidence() -> None:
    plan = _compile_substitute(_graph(two_targets=True))

    execution = _Phase6Runtime(_Backend("first_name", join_targets=True)).run(plan)

    assert execution.handoffs == ()
    assert any(
        isinstance(outcome, _TaskFailed) and outcome.task.stage.value == "classify"
        for outcome in execution.accounting.tasks
    )
