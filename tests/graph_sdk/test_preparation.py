# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for pure graph preparation and capability binding."""

from __future__ import annotations

import copy
import pickle
from dataclasses import FrozenInstanceError, replace
from typing import Any, cast

import pytest

from anonymizer.engine.graph_sdk.capabilities import (
    ConfigBoolean,
    ConfigField,
    ConfigInteger,
    ConfigNull,
    ConfigNumber,
    ConfigSequence,
    ConfigText,
    FrozenConfig,
    ImplementationCapability,
    ImplementationRef,
    ImplementationSelection,
    PreparationCode,
    PreparationRejected,
)
from anonymizer.engine.graph_sdk.data import DataGraph, DataLimits, ValidatedDataGraph
from anonymizer.engine.graph_sdk.preparation import (
    BoundInput,
    PreparationConfiguration,
    PreparationLimits,
    PreparedPlan,
    StateRevision,
    StateRevisionView,
    prepare,
    recheck_capabilities,
)
from anonymizer.graph._values import ActivationKey, InvocationId, PlanId
from anonymizer.graph.activation import ActivationLimits, ActivationSeed, initialize_activation
from anonymizer.graph.workflow import (
    AdmittedActivationWorkflow,
    ArtifactType,
    DynamicLimits,
    DynamicScope,
    InputBinding,
    InputPort,
    NodeId,
    NodeInputRef,
    NodeOutcomeRef,
    OperationNode,
    OperationSpec,
    OutcomeBinding,
    OutcomeSpec,
    ResourceCeiling,
    SequenceEdge,
    StateEffect,
    SubgraphNode,
    WorkflowId,
    WorkflowInputRef,
    WorkflowLimits,
    WorkflowOutcomeRef,
    admit_activation_workflow,
    admit_static_workflow,
)


def _data(target_count: int) -> ValidatedDataGraph:
    graph = DataGraph.new()
    targets = []
    for index in range(target_count):
        graph, target = graph.add_text(f"target-{index}")
        targets.append(target)
    return graph.validate(
        targets=tuple(targets),
        source_relations=(),
        contexts=(),
        dependencies=(),
        coherence=(),
        atomic=(),
        output_regions=(),
        limits=DataLimits(
            max_datums=target_count,
            max_targets=target_count,
            max_text_bytes=100,
            max_declarations=0,
            max_group_members=0,
        ),
    )


def _workflow(*, requests: int = 0, read: StateEffect | None = None, with_input: bool = False):
    owner = WorkflowId.new()
    node = NodeId.new(workflow=owner)
    artifact = ArtifactType(name="text", revision=1)
    inputs = (InputPort(name="input", artifact_type=artifact),) if with_input else ()
    outcome = OutcomeSpec(
        name="ok",
        category="success",
        produced_ports=frozenset(),
        context=frozenset(),
        evidence=frozenset(),
        state_effects=frozenset({read}) if read is not None else frozenset(),
        model_requirements=frozenset(),
        ceiling=ResourceCeiling(
            max_activations=1,
            max_model_requests=requests,
            max_input_bytes=0,
            max_output_bytes=0,
        ),
    )
    operation = OperationSpec(
        name="operation",
        inputs=inputs,
        outputs=(),
        output_dependencies=(),
        outcomes=(outcome,),
    )
    bindings = (
        InputBinding(source=WorkflowInputRef(port="input"), destination=NodeInputRef(node=node, port="input")),
    ) if with_input else ()
    static = admit_static_workflow(
        workflow=owner,
        interface=operation,
        nodes=(OperationNode(id=node, operation=operation),),
        input_bindings=bindings,
        output_bindings=(),
        outcome_bindings=(
            OutcomeBinding(
                source=NodeOutcomeRef(node=node, outcome="ok"),
                destination=WorkflowOutcomeRef(outcome="ok"),
            ),
        ),
        sequence=(),
        choices=(),
        protection=(),
        limits=WorkflowLimits(
            max_nodes=1,
            max_bindings=len(bindings) + 1,
            max_sequence_edges=0,
            max_choices=0,
            max_branch_members=0,
            max_subgraph_depth=1,
            max_choice_states=1,
        ),
    )
    dynamic = admit_activation_workflow(
        workflow=static,
        scopes=(DynamicScope(workflow=static, maps=(), joins=(), loops=()),),
        limits=DynamicLimits(
            max_maps=0,
            max_joins=0,
            max_loops=0,
            max_children_per_map=0,
            max_iterations_per_loop=0,
            max_dynamic_depth=1,
            max_activation_occurrences=1,
        ),
    )
    return dynamic, node, artifact


def _config() -> FrozenConfig:
    return FrozenConfig(
        fields=(
            ConfigField(
                name="nested",
                value=ConfigSequence(
                    items=(ConfigBoolean(value=True), ConfigInteger(value=1), ConfigNumber(numerator=2, denominator=4))
                ),
            ),
        )
    )


def _capability(
    workflow: AdmittedActivationWorkflow,
    *,
    configuration: FrozenConfig | None = None,
    external: bool = False,
) -> ImplementationCapability:
    node = next(iter(workflow.workflow.nodes))
    return ImplementationCapability(
        implementation=ImplementationRef(name="implementation", revision=1),
        operation=node.operation,
        configuration=configuration or _config(),
        effect="external" if external else "local",
        attribution="per_task",
        request_visibility="dispatch_and_settlement" if external else "none",
        pre_dispatch_control="executor" if external else "none",
        retry_owner="executor" if external else "none",
        error_reporting="typed_terminal",
        cancellation="cooperative_ack" if external else "before_dispatch_only",
        settlement="explicit_ack" if external else "synchronous",
        usage="upper_bound" if external else "exact",
        resource_lifetime="executor_owned" if external else "stateless",
        max_physical_requests_per_activation=1 if external else 0,
    )


def _limits(*, capabilities: int = 1, slots: int = 10) -> PreparationLimits:
    return PreparationLimits(
        max_bound_inputs=10,
        max_state_revisions=10,
        max_selections=10,
        max_config_depth=10,
        max_config_atoms=100,
        max_total_activation_slots=slots,
        max_capabilities=capabilities,
    )


def _prepare(
    *,
    data: ValidatedDataGraph | None = None,
    workflow: AdmittedActivationWorkflow | None = None,
    capability: ImplementationCapability | None = None,
    activation_limits: ActivationLimits | None = None,
    configuration: PreparationConfiguration | None = None,
    state: StateRevisionView | None = None,
    bound_inputs: tuple[BoundInput, ...] = (),
    limits: PreparationLimits | None = None,
) -> PreparedPlan:
    actual_workflow, node, _ = _workflow() if workflow is None else (workflow, next(iter(workflow.workflow.nodes)).id, None)
    actual_capability = capability or _capability(actual_workflow)
    return prepare(
        data=data or _data(2),
        workflow=actual_workflow,
        activation_limits=activation_limits or ActivationLimits(max_events=3, max_entries=1, max_parent_depth=1),
        bound_inputs=bound_inputs,
        configuration=configuration
        or PreparationConfiguration(
            purpose="execution_only",
            required_protection_outcomes=frozenset(),
            hard_request_limit=None,
        ),
        state=state or StateRevisionView(revisions=frozenset()),
        selections=(
            ImplementationSelection(
                node=node,
                implementation=actual_capability.implementation,
                configuration=actual_capability.configuration,
            ),
        ),
        capabilities=(actual_capability,),
        limits=limits or _limits(),
    )


def test_prepare_two_targets_builds_real_p3_reservations() -> None:
    plan = _prepare()
    assert plan.activation_slots_per_target == 1
    assert plan.total_activation_slots == 2
    assert [item.occurrence_offset for item in plan.target_occurrences] == [0, 1]
    invocation = InvocationId.new(plan=plan.plan)
    for target in plan.target_occurrences:
        keys = {
            slot.index: ActivationKey(
                invocation=invocation,
                occurrence=target.occurrence_offset + slot.index,
                parent=None,
                iteration=slot.iteration,
            )
            for slot in plan.reservation_recipe
        }
        reservations = frozenset(
            ActivationSeed(template=slot.template, activation=keys[slot.index])
            for slot in plan.reservation_recipe
        )
        state = initialize_activation(
            workflow=plan.workflow,
            invocation=invocation,
            reservations=reservations,
            limits=plan.activation_limits,
        )
        assert not state.entries


def test_shared_subgraph_body_is_selected_once_and_reserved_per_parent() -> None:
    body_dynamic, child, _ = _workflow()
    body = body_dynamic.workflow
    owner = WorkflowId.new()
    first, second = (NodeId.new(workflow=owner) for _ in range(2))
    root_interface = replace(
        body.interface,
        name="shared-body-root",
        outcomes=tuple(
            replace(
                outcome,
                ceiling=replace(outcome.ceiling, max_activations=4),
            )
            for outcome in body.interface.outcomes
        ),
    )
    root = admit_static_workflow(
        workflow=owner,
        interface=root_interface,
        nodes=(
            SubgraphNode(id=first, operation=body.interface, body=body),
            SubgraphNode(id=second, operation=body.interface, body=body),
        ),
        input_bindings=(),
        output_bindings=(),
        outcome_bindings=tuple(
            OutcomeBinding(
                source=NodeOutcomeRef(node=second, outcome=outcome.name),
                destination=WorkflowOutcomeRef(outcome=outcome.name),
            )
            for outcome in body.interface.outcomes
        ),
        sequence=(SequenceEdge(before=first, after=second),),
        choices=(),
        protection=(),
        limits=WorkflowLimits(
            max_nodes=4,
            max_bindings=1,
            max_sequence_edges=1,
            max_choices=0,
            max_branch_members=0,
            max_subgraph_depth=2,
            max_choice_states=1,
        ),
    )
    workflow = admit_activation_workflow(
        workflow=root,
        scopes=(
            DynamicScope(workflow=root, maps=(), joins=(), loops=()),
            DynamicScope(workflow=body, maps=(), joins=(), loops=()),
        ),
        limits=DynamicLimits(
            max_maps=0,
            max_joins=0,
            max_loops=0,
            max_children_per_map=0,
            max_iterations_per_loop=0,
            max_dynamic_depth=2,
            max_activation_occurrences=4,
        ),
    )
    capability = _capability(body_dynamic)
    plan = prepare(
        data=_data(1),
        workflow=workflow,
        activation_limits=ActivationLimits(max_events=12, max_entries=4, max_parent_depth=2),
        bound_inputs=(),
        configuration=PreparationConfiguration(
            purpose="execution_only", required_protection_outcomes=frozenset(), hard_request_limit=None
        ),
        state=StateRevisionView(revisions=frozenset()),
        selections=(
            ImplementationSelection(
                node=child,
                implementation=capability.implementation,
                configuration=capability.configuration,
            ),
        ),
        capabilities=(capability,),
        limits=_limits(slots=4),
    )
    child_slots = [slot for slot in plan.reservation_recipe if slot.template == child]
    assert len(plan.implementations) == 1
    assert len(child_slots) == 2
    assert len({slot.parent_index for slot in child_slots}) == 2
    invocation = InvocationId.new(plan=plan.plan)
    keys: dict[int, ActivationKey] = {}
    for slot in plan.reservation_recipe:
        keys[slot.index] = ActivationKey(
            invocation=invocation,
            occurrence=slot.index,
            parent=keys.get(slot.parent_index),
            iteration=slot.iteration,
        )
    state = initialize_activation(
        workflow=workflow,
        invocation=invocation,
        reservations=frozenset(
            ActivationSeed(template=slot.template, activation=keys[slot.index])
            for slot in plan.reservation_recipe
        ),
        limits=plan.activation_limits,
    )
    assert not state.entries


def test_zero_target_plan_has_no_recipe_uses_or_runtime_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def forbidden(*, plan: PlanId) -> InvocationId:
        del plan
        nonlocal calls
        calls += 1
        raise AssertionError("preparation must not allocate an invocation")

    monkeypatch.setattr(InvocationId, "new", forbidden)
    plan = _prepare(data=_data(0))
    assert plan.activation_slots_per_target == 1
    assert plan.total_activation_slots == 0
    assert plan.target_occurrences == ()
    assert calls == 0


def test_tagged_configuration_is_type_sensitive_canonical_and_immutable() -> None:
    assert ConfigBoolean(value=True) != ConfigInteger(value=1)
    assert ConfigBoolean(value=False) != ConfigInteger(value=0)
    assert ConfigInteger(value=-1).value == -1
    assert ConfigNumber(numerator=2, denominator=4) == ConfigNumber(numerator=1, denominator=2)
    with pytest.raises(PreparationRejected) as raised:
        ConfigNumber(numerator=True, denominator=1)  # type: ignore[arg-type]
    assert raised.value.code is PreparationCode.INVALID_TYPE
    configuration = _config()
    assert copy.copy(configuration) is configuration
    assert copy.deepcopy(configuration) is configuration
    assert repr(configuration) == "<FrozenConfig>"
    with pytest.raises((TypeError, pickle.PicklingError)):
        pickle.dumps(configuration)
    with pytest.raises(FrozenInstanceError):
        mutable: Any = configuration
        mutable.fields = ()
    assert repr(PreparationRejected(PreparationCode.MISSING_INPUT)) == "<PreparationRejected>"


def test_prepare_binds_exact_root_input_and_read_revision() -> None:
    read = StateEffect(kind="read", name="state")
    workflow, _, artifact = _workflow(read=read, with_input=True)
    data = _data(1)
    target = next(iter(data.targets))
    bound = BoundInput(target=target, port="input", source=target, artifact_type=artifact)
    plan = _prepare(
        data=data,
        workflow=workflow,
        capability=_capability(workflow),
        bound_inputs=(bound,),
        state=StateRevisionView(revisions=frozenset({StateRevision(effect=read, revision=1)})),
    )
    assert plan.bound_inputs == frozenset({bound})
    assert plan.state.revisions == frozenset({StateRevision(effect=read, revision=1)})
    with pytest.raises(PreparationRejected) as raised:
        _prepare(data=data, workflow=workflow, capability=_capability(workflow), state=plan.state)
    assert raised.value.code is PreparationCode.MISSING_INPUT

    extra = StateEffect(kind="read", name="extra")
    with pytest.raises(PreparationRejected) as raised:
        _prepare(
            data=data,
            workflow=workflow,
            capability=_capability(workflow),
            bound_inputs=(bound,),
            state=StateRevisionView(
                revisions=frozenset(
                    {StateRevision(effect=read, revision=1), StateRevision(effect=extra, revision=1)}
                )
            ),
        )
    assert raised.value.code is PreparationCode.MISSING_STATE


def test_hard_budget_zero_is_allowed_only_for_controllable_external_work() -> None:
    workflow, _, _ = _workflow(requests=1)
    capability = _capability(workflow, external=True)
    configuration = PreparationConfiguration(
        purpose="execution_only",
        required_protection_outcomes=frozenset(),
        hard_request_limit=0,
    )
    plan = _prepare(workflow=workflow, capability=capability, configuration=configuration)
    assert plan.configuration.hard_request_limit == 0
    for budget in (1, plan.declared_request_upper_bound, plan.declared_request_upper_bound + 1):
        assert _prepare(
            workflow=workflow,
            capability=capability,
            configuration=replace(configuration, hard_request_limit=budget),
        ).configuration.hard_request_limit == budget
    incompatible = replace(capability, retry_owner="implementation")
    with pytest.raises(PreparationRejected) as raised:
        _prepare(workflow=workflow, capability=incompatible, configuration=configuration)
    assert raised.value.code is PreparationCode.HARD_BUDGET_INCOMPATIBLE


def test_capability_recheck_requires_exact_selected_declaration() -> None:
    plan = _prepare(limits=_limits(capabilities=2))
    selected = next(iter(plan.implementations)).capability
    recheck_capabilities(prepared=plan, capabilities=(selected,))
    changed = replace(selected, implementation=ImplementationRef(name="implementation", revision=2))
    with pytest.raises(PreparationRejected) as raised:
        recheck_capabilities(prepared=plan, capabilities=(changed,))
    assert raised.value.code is PreparationCode.CAPABILITY_CHANGED
    unrelated = replace(selected, implementation=ImplementationRef(name="unrelated", revision=1))
    recheck_capabilities(prepared=plan, capabilities=(selected, unrelated))


def test_catalog_limit_precedes_invalid_member_in_prepare_and_recheck() -> None:
    workflow, node, _ = _workflow()
    capability = _capability(workflow)
    data = _data(1)
    activation_limits = ActivationLimits(max_events=3, max_entries=1, max_parent_depth=1)
    configuration = PreparationConfiguration(
        purpose="execution_only",
        required_protection_outcomes=frozenset(),
        hard_request_limit=None,
    )
    state = StateRevisionView(revisions=frozenset())
    selections = (
        ImplementationSelection(
            node=node,
            implementation=capability.implementation,
            configuration=capability.configuration,
        ),
    )
    with pytest.raises(PreparationRejected) as raised:
        invalid_capabilities = cast(tuple[ImplementationCapability, ...], (capability, object()))
        prepare(
            data=data,
            workflow=workflow,
            activation_limits=activation_limits,
            bound_inputs=(),
            configuration=configuration,
            state=state,
            selections=selections,
            capabilities=invalid_capabilities,
            limits=_limits(capabilities=1),
        )
    assert raised.value.code is PreparationCode.LIMIT_EXCEEDED
    plan = _prepare(data=data, workflow=workflow, capability=capability)
    with pytest.raises(PreparationRejected) as raised:
        invalid_capabilities = cast(tuple[ImplementationCapability, ...], (capability, object()))
        recheck_capabilities(
            prepared=plan,
            capabilities=invalid_capabilities,
        )
    assert raised.value.code is PreparationCode.LIMIT_EXCEEDED


def test_activation_and_configuration_limits_reject_before_plan_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = PlanId.new

    def counted() -> PlanId:
        nonlocal calls
        calls += 1
        return original()

    monkeypatch.setattr(PlanId, "new", counted)
    with pytest.raises(PreparationRejected) as raised:
        _prepare(activation_limits=ActivationLimits(max_events=2, max_entries=1, max_parent_depth=1))
    assert (raised.value.code, calls) == (PreparationCode.LIMIT_EXCEEDED, 0)
    plan = _prepare()
    assert calls == 1 and isinstance(plan.plan, PlanId)


@pytest.mark.parametrize(
    ("activation_limits", "limits", "succeeds"),
    [
        (ActivationLimits(max_events=3, max_entries=1, max_parent_depth=1), _limits(slots=2), True),
        (ActivationLimits(max_events=3, max_entries=0, max_parent_depth=1), _limits(), False),
        (ActivationLimits(max_events=3, max_entries=1, max_parent_depth=0), _limits(), False),
        (ActivationLimits(max_events=2, max_entries=1, max_parent_depth=1), _limits(), False),
        (ActivationLimits(max_events=3, max_entries=1, max_parent_depth=1), _limits(slots=1), False),
    ],
)
def test_structural_limits_are_exact(
    activation_limits: ActivationLimits,
    limits: PreparationLimits,
    succeeds: bool,
) -> None:
    if succeeds:
        assert _prepare(activation_limits=activation_limits, limits=limits).total_activation_slots == 2
    else:
        with pytest.raises(PreparationRejected) as raised:
            _prepare(activation_limits=activation_limits, limits=limits)
        assert raised.value.code is PreparationCode.LIMIT_EXCEEDED


def test_configuration_limits_count_complete_submitted_values() -> None:
    # Each nested configuration is six atoms; selection and capability each
    # carry one exact copy in the preparation request.
    assert _prepare(limits=replace(_limits(), max_config_depth=3, max_config_atoms=12))
    for limits in (
        replace(_limits(), max_config_depth=2),
        replace(_limits(), max_config_atoms=11),
        replace(_limits(), max_selections=0),
        replace(_limits(), max_capabilities=0),
    ):
        with pytest.raises(PreparationRejected) as raised:
            _prepare(limits=limits)
        assert raised.value.code is PreparationCode.LIMIT_EXCEEDED


def test_eligibility_is_distinct_from_runtime_qualification() -> None:
    plan = _prepare()
    assert not plan.eligibility.eligible_for_requested_protection
    with pytest.raises(PreparationRejected) as raised:
        _prepare(
            configuration=PreparationConfiguration(
                purpose="protection",
                required_protection_outcomes=frozenset({"missing"}),
                hard_request_limit=None,
            )
        )
    assert raised.value.code is PreparationCode.PROTECTION_INELIGIBLE


def test_prepared_plan_constructor_is_factory_protected_and_identity_based() -> None:
    first = _prepare()
    second = _prepare()
    assert first != second and hash(first) != hash(second)
    assert repr(first) == "<PreparedPlan>"
    with pytest.raises(TypeError):
        PreparedPlan(  # type: ignore[call-arg]
            _key=object(),
            plan=first.plan,
            data=first.data,
            workflow=first.workflow,
            activation_limits=first.activation_limits,
            bound_inputs=first.bound_inputs,
            configuration=first.configuration,
            state=first.state,
            implementations=first.implementations,
            reservation_recipe=first.reservation_recipe,
            target_occurrences=first.target_occurrences,
            activation_slots_per_target=first.activation_slots_per_target,
            total_activation_slots=first.total_activation_slots,
            declared_request_upper_bound=first.declared_request_upper_bound,
            eligibility=first.eligibility,
            limits=first.limits,
        )
