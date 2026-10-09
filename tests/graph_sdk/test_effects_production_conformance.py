# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Translate the frozen effects corpus through the production request reducer."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any, cast

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectRejected
from anonymizer.engine.graph_sdk.capabilities import ImplementationRef, PreparationRejected
from anonymizer.engine.graph_sdk.context import (
    ContextMaterialization,
    ContextSelector,
    ContextSourceRef,
    InitialContextDecl,
    RetrievalBounds,
    SourceFailure,
    SourceItem,
    admit_context_plan,
)
from anonymizer.engine.graph_sdk.executor import (
    ExecutionImplementation,
    ExecutionLimits,
    OperationExecutionPolicy,
    admit_execution_plan,
)
from anonymizer.engine.graph_sdk.requests import (
    AcceptResult,
    AssociationResult,
    BindingAssociation,
    BindingDeclarationId,
    BindingId,
    BindingRequestScope,
    Dispatch,
    ExternalSettlement,
    InvocationRequestScope,
    MarkLost,
    ObserveSettlement,
    PhysicalRequestId,
    PhysicalRequestPolicy,
    PortArtifact,
    RequestCancel,
    RequestPolicyBinding,
    Reserve,
    SemanticAssociation,
    StopAcknowledged,
    TextArtifactValue,
    TextCollectionItem,
    TextCollectionValue,
    UnknownUsage,
    advance_requests,
    bind_request_policies,
    initialize_requests,
)
from anonymizer.engine.graph_sdk.resources import ResourceLease, close_resource
from anonymizer.graph._values import ActivationKey, InvocationId, PlanId, TaskAttemptId
from anonymizer.graph.workflow import (
    ArtifactType,
    NodeId,
    WorkflowId,
)
from tests.graph_sdk.effects_adaptive_assertions import (
    _assert_adaptive_materialization_case as _assert_adaptive_materialization_case,
)
from tests.graph_sdk.effects_admission_fixtures import (
    ADAPTIVE_MATERIALIZATION_CASES,
    ADMISSION_CASES,
    BINDING_RESULT_SHAPE_CASES,
    CASES,
    LATE_ADAPTIVE_CASES,
    LATE_INITIAL_CASES,
    LOCAL_BRIDGE_CASES,
    REAL_BINDING_CASES,
    RESOURCE_CASES,
    _admit_case,
)
from tests.graph_sdk.effects_admission_fixtures import (
    _assert_decision_submission as _assert_decision_submission,
)
from tests.graph_sdk.effects_admission_fixtures import _assessment_limits as _assessment_limits
from tests.graph_sdk.effects_admission_fixtures import _valid_runtime_rows as _valid_runtime_rows
from tests.graph_sdk.effects_admission_fixtures import _ZeroClock as _ZeroClock
from tests.graph_sdk.effects_binding_assertions import (
    _assert_binding_corpus_case,
)
from tests.graph_sdk.effects_late_binding_assertions import (
    _assert_late_initial_provider_case,
    _reject_foreign_binding_target,
)
from tests.graph_sdk.effects_provider_fixtures import (
    _assert_local_bridge_case,
)
from tests.graph_sdk.effects_provider_fixtures import _CaseProvider as _CaseProvider
from tests.graph_sdk.effects_provider_fixtures import _ContextConsumer as _ContextConsumer
from tests.graph_sdk.effects_request_fixtures import _apply as _apply
from tests.graph_sdk.effects_request_fixtures import _Closable, _usage
from tests.graph_sdk.effects_request_fixtures import _normalize as _normalize
from tests.graph_sdk.effects_request_fixtures import _policy as _policy
from tests.graph_sdk.reference.corpora import load_cases
from tests.graph_sdk.test_preparation import _capability, _data, _limits, _prepare, _workflow


@pytest.mark.parametrize(
    ("case_id", "expected"),
    (
        ("decisions/stale_artifact", "foreign_owner"),
        ("decisions/foreign_workflow", "foreign_owner"),
        ("decisions/foreign_wait", "foreign_owner"),
        ("decisions/unknown_decision", "unsupported"),
        ("decisions/duplicate_response", "duplicate"),
    ),
)
def test_decision_submission_rejections_use_public_running_execution(case_id: str, expected: str) -> None:
    asyncio.run(_assert_decision_submission(case_id, expected))


@pytest.mark.parametrize(
    ("case_id", "items", "expected"),
    (
        (
            "map/duplicate_item",
            (
                TextCollectionItem(key=0, version=1, value=TextArtifactValue(text="a")),
                TextCollectionItem(key=0, version=1, value=TextArtifactValue(text="b")),
            ),
            "duplicate",
        ),
        (
            "map/noncanonical_items",
            (
                TextCollectionItem(key=1, version=1, value=TextArtifactValue(text="b")),
                TextCollectionItem(key=0, version=1, value=TextArtifactValue(text="a")),
            ),
            "invalid_value",
        ),
    ),
)
def test_map_collection_shape_rejects_at_typed_constructor(
    case_id: str,
    items: tuple[TextCollectionItem, ...],
    expected: str,
) -> None:
    case = next(item for item in load_cases("effects") if item["case_id"] == case_id)
    assert case["expected"] == {"status": "rejected", "code": expected}
    with pytest.raises(EffectRejected) as rejected:
        TextCollectionValue(items=items)
    assert rejected.value.code.value == case["expected"]["code"]
    canonical = (
        TextCollectionItem(key=0, version=1, value=TextArtifactValue(text="a")),
        TextCollectionItem(key=1, version=1, value=TextArtifactValue(text="b")),
    )
    assert TextCollectionValue(items=canonical).items == canonical


@pytest.mark.parametrize(
    ("case_id", "value"),
    (
        ("materialization/initial_wrong_item", 7),
        ("materialization/adaptive_wrong_item", 7),
        ("materialization/initial_nested_value", {"items": []}),
        ("materialization/adaptive_nested_value", {"items": []}),
    ),
)
def test_materialization_item_shape_rejects_at_typed_constructor(case_id: str, value: object) -> None:
    del case_id
    binding = BindingId.new()
    association = BindingAssociation(declaration=BindingDeclarationId.new(binding=binding, ordinal=0))
    with pytest.raises(EffectRejected) as rejected:
        SourceItem(association=association, key=0, version=1, text=cast(Any, value))
    assert rejected.value.code.value == "invalid_type"
    assert SourceItem(association=association, key=0, version=1, text="valid").text == "valid"


@pytest.mark.parametrize(
    ("case_id", "kind", "same_type", "max_items"),
    (
        ("materialization/single_output_mismatch", "single", False, 1),
        ("materialization/single_max_items", "single", True, 2),
        ("materialization/collection_scalar_output", "collection", True, 3),
        ("materialization/collection_zero_max", "collection", False, 0),
    ),
)
def test_initial_materialization_shape_rejects_at_typed_constructor(
    case_id: str,
    kind: str,
    same_type: bool,
    max_items: int,
) -> None:
    del case_id
    data = _data(1)
    target = next(iter(data.targets))
    workflow = WorkflowId.new()
    item_type = ArtifactType(name="text", revision=1)
    output_type = item_type if same_type else ArtifactType(name="text-collection", revision=1)
    common = {
        "target": target,
        "node": NodeId.new(workflow=workflow),
        "port": "context",
        "source": ContextSourceRef(name="source", revision=1),
        "selector": ContextSelector(fields=()),
        "requirement": "required",
    }
    with pytest.raises(EffectRejected) as rejected:
        InitialContextDecl(
            **cast(Any, common),
            artifact_type=output_type,
            bounds=RetrievalBounds(max_items=max_items, max_bytes=12, max_requests=1),
            materialization=ContextMaterialization(kind=cast(Any, kind), item_type=item_type),
        )
    assert rejected.value.code.value == "contradictory"
    valid_kind = cast(Any, kind)
    valid_output = item_type if kind == "single" else ArtifactType(name="valid-collection", revision=1)
    valid = InitialContextDecl(
        **cast(Any, common),
        artifact_type=valid_output,
        bounds=RetrievalBounds(max_items=1, max_bytes=12, max_requests=1),
        materialization=ContextMaterialization(kind=valid_kind, item_type=item_type),
    )
    assert valid.materialization.kind == kind


def test_map_collection_limit_rejects_boolean_at_typed_constructor() -> None:
    with pytest.raises(EffectRejected) as rejected:
        ExecutionLimits(
            max_local_in_flight=1,
            max_remote_outstanding=1,
            max_runtime_artifacts=8,
            max_runtime_artifact_bytes=32,
            max_collection_items=cast(Any, True),
        )
    assert rejected.value.code.value == "invalid_type"
    assert (
        ExecutionLimits(
            max_local_in_flight=1,
            max_remote_outstanding=1,
            max_runtime_artifacts=8,
            max_runtime_artifact_bytes=32,
            max_collection_items=4,
        ).max_collection_items
        == 4
    )


@pytest.mark.parametrize(
    "case_id",
    (
        "binding/source_failure_missing_failure",
        "binding/source_failure_missing_settlement",
        "binding/omission_failure_mismatch",
    ),
)
def test_binding_source_failure_constructor_cases(case_id: str) -> None:
    if case_id == "binding/source_failure_missing_failure":
        with pytest.raises(TypeError):
            cast(Any, SourceFailure)(source=ContextSourceRef(name="source", revision=1))
        return
    if case_id == "binding/source_failure_missing_settlement":
        with pytest.raises(TypeError):
            cast(Any, SourceFailure)(
                source=ContextSourceRef(name="source", revision=1),
                failure="permanent",
            )
        return
    with pytest.raises(EffectRejected) as rejected:
        SourceFailure(
            source=ContextSourceRef(name="source", revision=1),
            failure="retryable",
            settlement=None,
            disposition="omitted_optional",
        )
    assert rejected.value.code.value == "contradictory"


@pytest.mark.parametrize("case", ADMISSION_CASES, ids=lambda case: cast(str, case["case_id"]))
def test_admission_corpus_case_through_production(case: dict[str, Any]) -> None:
    rejected: str | None = None
    try:
        _admit_case(case)
    except (EffectRejected, PreparationRejected) as exc:
        rejected = exc.code.value
    expected = cast(dict[str, str], case["expected"])
    assert ("rejected" if rejected else "accepted") == expected["status"]
    assert rejected == expected.get("code")


@pytest.mark.parametrize(
    ("duplicate", "expected"),
    ((False, None), (True, "duplicate")),
)
def test_admission_capability_identity_survives_a_raised_limit(duplicate: bool, expected: str | None) -> None:
    workflow, node, _ = _workflow(requests=3)
    primary = replace(_capability(workflow, external=True), max_physical_requests_per_activation=2)
    alternate = (
        primary
        if duplicate
        else replace(
            primary,
            implementation=ImplementationRef(name="alternate", revision=1),
        )
    )
    prepared = _prepare(
        data=_data(2),
        workflow=workflow,
        capability=primary,
        limits=_limits(capabilities=2),
    )
    context = admit_context_plan(
        prepared=prepared,
        bound_context=None,
        adaptive_retrievals=(),
        context_capabilities=(),
    )
    request = PhysicalRequestPolicy(
        visibility="dispatch_and_settlement",
        pre_dispatch_control="executor",
        retry_owner="executor",
        replay="idempotent",
        max_attempts=2,
    )
    policy = OperationExecutionPolicy(
        node=node,
        kind="external",
        request=request,
        safe_detachment="forbidden",
        implementations=tuple(
            ExecutionImplementation(
                implementation=item.implementation,
                configuration=item.configuration,
                capability=item,
                request=request,
            )
            for item in (primary, alternate)
        ),
        result_outcomes=frozenset({"ok"}),
        runtime_outcomes=_valid_runtime_rows("external", frozenset({"ok"})),
    )
    rejected: str | None = None
    try:
        admit_execution_plan(
            context=context,
            capabilities=(primary, alternate),
            policies=(policy,),
            decisions=(),
            assessment_productions=(),
            assessment_limits=_assessment_limits(),
        )
    except EffectRejected as exc:
        rejected = exc.code.value
    assert rejected == expected


@pytest.mark.parametrize("case", LOCAL_BRIDGE_CASES, ids=lambda case: cast(str, case["case_id"]))
def test_local_bridge_case_through_real_execution(case: dict[str, Any]) -> None:
    asyncio.run(_assert_local_bridge_case(case))


@pytest.mark.parametrize("case", REAL_BINDING_CASES, ids=lambda case: cast(str, case["case_id"]))
def test_binding_corpus_case_through_real_provider(case: dict[str, Any]) -> None:
    asyncio.run(_assert_binding_corpus_case(case))


@pytest.mark.parametrize("case", LATE_INITIAL_CASES, ids=lambda case: cast(str, case["case_id"]))
def test_late_initial_provider_facts_through_running_binding(case: dict[str, Any]) -> None:
    asyncio.run(_assert_late_initial_provider_case(case))


def test_binding_foreign_target_admission_case_through_production() -> None:
    case = next(item for item in load_cases("effects") if item["case_id"] == "binding/invalid_local_zero_effects")
    rejected = asyncio.run(_reject_foreign_binding_target())
    assert rejected == cast(dict[str, str], case["expected"])["code"]


@pytest.mark.parametrize(
    "case",
    ADAPTIVE_MATERIALIZATION_CASES,
    ids=lambda case: cast(str, case["case_id"]),
)
def test_adaptive_materialization_corpus_through_real_execution(case: dict[str, Any]) -> None:
    asyncio.run(_assert_adaptive_materialization_case(case))


@pytest.mark.parametrize("case", LATE_ADAPTIVE_CASES, ids=lambda case: cast(str, case["case_id"]))
def test_late_adaptive_result_stays_request_terminal_at_authority_boundary(case: dict[str, Any]) -> None:
    raw = cast(dict[str, Any], case["declaration"])
    invocation = InvocationId.new(plan=PlanId.new())
    scope = InvocationRequestScope(invocation=invocation)
    association = SemanticAssociation(
        task=TaskAttemptId.new(
            activation=ActivationKey(invocation=invocation, occurrence=0, parent=None, iteration=None)
        )
    )
    policy = _policy(cast(dict[str, Any], raw["policies"])["P0"])
    state = initialize_requests(scope=scope, hard_limit=cast(int, raw["hard_limit"]), policies=frozenset({policy}))
    state = bind_request_policies(
        state=state,
        binding=RequestPolicyBinding.create(association=association, policies=frozenset({policy})),
    )
    request = PhysicalRequestId.new(scope=scope)
    state = advance_requests(
        state=state,
        event=Reserve(
            request=request,
            purpose="adaptive_retrieval",
            associations=frozenset({association}),
            policy=policy,
        ),
    )
    state = advance_requests(state=state, event=Dispatch(request=request))
    if case["case_id"] == "materialization/adaptive_late_lost":
        state = advance_requests(state=state, event=RequestCancel(request=request))
        state = advance_requests(state=state, event=MarkLost(request=request))
    else:
        state = advance_requests(state=state, event=RequestCancel(request=request))
        state = advance_requests(
            state=state,
            event=StopAcknowledged(request=request, usage=UnknownUsage()),
        )
    state = advance_requests(
        state=state,
        event=AcceptResult(
            request=request,
            results=(
                AssociationResult(
                    association=association,
                    outcome="ok",
                    outputs=(),
                    consumed_context_ports=frozenset(),
                ),
            ),
        ),
    )
    settlement_value = next(
        cast(dict[str, Any], event["settlement"])
        for event in cast(list[dict[str, Any]], case["events"])
        if event["kind"] == "materialize_result"
    )
    state = advance_requests(
        state=state,
        event=ObserveSettlement(
            settlement=ExternalSettlement(
                request=request,
                disposition=settlement_value["disposition"],
                usage=_usage(settlement_value["usage"]),
                remote_stopped=settlement_value["remote_stopped"],
            )
        ),
    )
    actual = _normalize(state, {"A0": association}, {"R0": request}, {"P0": policy})
    expected = cast(dict[str, Any], case["expected"])["state"]
    for key in (
        "bindings",
        "dispatched",
        "dispatched_count",
        "terminals",
        "settlements",
        "request_facts",
        "request_failures",
        "association_terminals",
        "request_associations",
        "request_policies",
        "cancel_requested",
        "local_in_flight",
        "remote_outstanding",
        "defects",
        "denials",
        "association_requests",
        "attempts",
    ):
        assert actual[key] == expected[key], (case["case_id"], key)
    assert not expected["materialization"]["ports"]
    assert all(not item["identity"].startswith("OperationOutputKey:") for item in expected["artifacts"])


@pytest.mark.parametrize("case", LATE_ADAPTIVE_CASES, ids=lambda case: cast(str, case["case_id"]))
def test_late_adaptive_provider_result_is_suppressed_by_real_executor(case: dict[str, Any]) -> None:
    asyncio.run(
        _assert_adaptive_materialization_case(
            case,
            late_mode="lost" if case["case_id"] == "materialization/adaptive_late_lost" else "cancelled",
        )
    )


@pytest.mark.parametrize("case", RESOURCE_CASES, ids=lambda case: cast(str, case["case_id"]))
def test_resource_corpus_case_through_production(case: dict[str, Any]) -> None:
    events = cast(list[dict[str, Any]], case["events"])
    resource = events[0]
    close_event = events[1]
    disposition = cast(str, close_event.get("disposition", "closed"))
    lease = ResourceLease.create(
        owner=resource["owner"],
        safe_detachment=resource["safe_detachment"],
        handle=object() if disposition == "close_unknown" else _Closable(disposition),
    )
    fact = asyncio.run(close_resource(lease))
    expected = cast(dict[str, Any], case["expected"])["state"]["cleanup"]["Q0"]
    assert fact.disposition == expected


@pytest.mark.parametrize("case", BINDING_RESULT_SHAPE_CASES, ids=lambda case: cast(str, case["case_id"]))
def test_binding_result_shape_case_through_production(case: dict[str, Any]) -> None:
    event = cast(list[dict[str, Any]], case["events"])[-1]
    binding = BindingId.new()
    association = BindingAssociation(declaration=BindingDeclarationId.new(binding=binding, ordinal=0))
    outputs = tuple(
        PortArtifact(
            port=cast(str, item["port"]),
            artifact_type=ArtifactType(name="text", revision=1),
            artifact=None,
            value=TextArtifactValue(text="unexpected"),
        )
        for item in cast(list[dict[str, object]], event.get("outputs", []))
    )
    rejected: str | None = None
    try:
        AssociationResult(
            association=association,
            outcome=cast(str, event.get("outcome", "retrieved")),
            outputs=outputs,
            consumed_context_ports=frozenset(cast(list[str], event.get("consumed_context_ports", []))),
        )
    except EffectRejected as exc:
        rejected = exc.code.value
    expected = cast(dict[str, str], case["expected"])
    assert ("rejected" if rejected else "accepted") == expected["status"]
    assert rejected == expected["code"]


@pytest.mark.parametrize("case", CASES, ids=lambda case: cast(str, case["case_id"]))
def test_request_corpus_case_through_production(case: dict[str, Any]) -> None:
    declaration = cast(dict[str, Any], case["declaration"])
    invocation = InvocationId.new(plan=PlanId.new())
    binding = BindingId.new()
    binding_labels = set(cast(dict[str, str], declaration.get("binding_declarations", {})))
    scope = BindingRequestScope(binding=binding) if binding_labels else InvocationRequestScope(invocation=invocation)
    policies = {
        name: _policy(value) for name, value in cast(dict[str, dict[str, Any]], declaration["policies"]).items()
    }
    association_names = (
        {
            item
            for event in cast(list[dict[str, Any]], case["events"])
            for field in ("associations", "returned")
            for item in cast(list[str], event.get(field, ()))
        }
        | {
            cast(str, event["association"])
            for event in cast(list[dict[str, Any]], case["events"])
            if "association" in event
        }
        | {"T0", "T1"}
    )
    foreign_invocation = InvocationId.new(plan=invocation.plan)
    tasks = {
        name: (
            BindingAssociation(declaration=BindingDeclarationId.new(binding=binding, ordinal=index))
            if name in binding_labels
            else SemanticAssociation(
                task=TaskAttemptId.new(
                    activation=ActivationKey(
                        invocation=foreign_invocation if name.startswith("X") else invocation,
                        occurrence=index,
                        parent=None,
                        iteration=None,
                    )
                )
            )
        )
        for index, name in enumerate(sorted(association_names))
    }
    requests = {name: PhysicalRequestId.new(scope=scope) for name in ("R0", "R1", "R2")}
    state = initialize_requests(
        scope=scope,
        hard_limit=cast(int | None, declaration.get("hard_limit")),
        policies=frozenset(policies.values()),
    )
    rejected: str | None = None
    try:
        for event in cast(list[dict[str, Any]], case["events"]):
            state = _apply(state, event, tasks, requests, policies)
    except EffectRejected as exc:
        rejected = exc.code.value
    expected = cast(dict[str, Any], case["expected"])
    assert ("rejected" if rejected else "accepted") == expected["status"]
    if rejected:
        assert rejected == expected["code"]
        return
    expected_state = cast(dict[str, Any], expected["state"])
    actual = _normalize(state, tasks, requests, policies)
    for key in (
        "bindings",
        "dispatched",
        "dispatched_count",
        "denials",
        "terminals",
        "defects",
        "local_in_flight",
        "remote_outstanding",
        "attempts",
        "request_failures",
        "association_terminals",
        "reservations",
        "request_associations",
        "request_policies",
        "reservation_policies",
        "settlements",
        "request_facts",
        "cancel_requested",
        "association_requests",
    ):
        assert actual[key] == expected_state[key], (case["case_id"], key)


@pytest.mark.parametrize(
    "case_id", ["decisions/cancel_condition", "decisions/implementation_failure", "decisions/deadline"]
)
def test_decision_runtime_terminal_cases(case_id: str) -> None:
    asyncio.run(_assert_decision_submission(case_id, ""))


@pytest.mark.parametrize("pending_limit", [1, 2])
def test_decision_capacity_includes_callbacks_before_wait_publication(pending_limit: int) -> None:
    asyncio.run(_assert_decision_submission("capacity", "", pending_limit=pending_limit))


@pytest.mark.parametrize(
    ("admission_id", "runtime_id"),
    [
        ("materialization/admit_initial_single", "materialization/initial_single_exact"),
        ("materialization/admit_initial_collection", "materialization/initial_collection_1"),
        ("materialization/admit_adaptive_single", "materialization/adaptive_single_exact"),
        ("materialization/admit_adaptive_collection", "materialization/adaptive_collection_1"),
    ],
)
def test_materialization_admission_matches_executed_declaration(admission_id: str, runtime_id: str) -> None:
    corpus = load_cases("effects")
    admission = next(case for case in corpus if case["case_id"] == admission_id)
    runtime = next(case for case in corpus if case["case_id"] == runtime_id)
    assert admission["expected"] == {"status": "accepted"}
    for key, value in admission["declaration"].items():
        assert runtime["declaration"][key] == value, key
    # The same declarations are admitted on the real path. Initial provider
    # binding supplies setup facts before pure context/execution admission.
    if "initial" in admission_id:
        asyncio.run(_assert_binding_corpus_case(runtime))
    else:
        asyncio.run(_assert_adaptive_materialization_case(runtime))
