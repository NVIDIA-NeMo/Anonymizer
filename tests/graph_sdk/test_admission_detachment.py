# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Caller-supplied admission inputs are detached once, so later caller changes never reach a run."""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Any

import pytest

from anonymizer.engine.graph_sdk.executor import (
    DelegatedRequestContract,
    IngressLimits,
    admit_execution_plan,
)
from anonymizer.engine.graph_sdk.requests import ChildRoute
from anonymizer.graph import EffectCode, EffectRejected
from tests.graph_sdk import test_delegated_execution as delegated
from tests.graph_sdk.typed_artifact_fixtures import span_graph


def _readmit(execution: Any, *, policies: Any = None, ingress_limits: Any = None) -> Any:
    return admit_execution_plan(
        context=execution.context,
        capabilities=execution.capabilities,
        policies=tuple(execution.policies) if policies is None else policies,
        assessment_productions=execution.assessment_productions,
        assessment_limits=execution.assessment_limits,
        ingress_limits=execution.ingress_limits if ingress_limits is None else ingress_limits,
    )


def _copy_as(kind: type[Any], value: Any) -> Any:
    return kind(**{item.name: getattr(value, item.name) for item in fields(value) if item.init})


def test_issued_ingress_limits_are_a_detached_copy() -> None:
    caller = IngressLimits(max_text_bytes=10)
    execution, *_ = span_graph(ingress=caller)
    assert execution.ingress_limits == caller and execution.ingress_limits is not caller
    object.__setattr__(caller, "max_text_bytes", None)  # the caller's own object, after admission
    assert execution.ingress_limits.max_text_bytes == 10


def test_ingress_limit_subclasses_are_not_admitted() -> None:
    class Widening(IngressLimits):
        pass

    execution, *_ = span_graph()
    _readmit(execution)
    with pytest.raises(EffectRejected) as error:
        _readmit(execution, ingress_limits=_copy_as(Widening, IngressLimits()))
    assert error.value.code is EffectCode.INVALID_TYPE


@pytest.mark.parametrize("member", ["contract", "route"])
def test_delegated_recipes_are_admitted_only_as_exact_immutable_types(member: str) -> None:
    class SubContract(DelegatedRequestContract):
        pass

    class SubRoute(ChildRoute):
        pass

    execution, *_ = delegated._fixture(simple=True)
    owner = next(policy for policy in execution.policies if policy.delegated is not None)
    contract = owner.delegated
    assert contract is not None
    if member == "contract":
        changed = _copy_as(SubContract, contract)
    else:
        changed = replace(contract, routes=tuple(_copy_as(SubRoute, route) for route in contract.routes))
    _readmit(execution)
    policies = tuple(replace(owner, delegated=changed) if item is owner else item for item in execution.policies)
    with pytest.raises(EffectRejected) as error:
        _readmit(execution, policies=policies)
    assert error.value.code is EffectCode.INVALID_TYPE


def test_captured_presentation_dtypes_are_detached_from_the_callers_mapping() -> None:
    import pandas as pd

    from anonymizer.engine.detection._frame_publication import FramePublication
    from anonymizer.engine.graph_sdk.requests import SemanticAssociation
    from anonymizer.graph._values import ActivationKey, InvocationId, PlanId, TaskAttemptId

    activation = ActivationKey(
        invocation=InvocationId.new(plan=PlanId.new()), occurrence=0, parent=None, iteration=None
    )
    frame = pd.DataFrame({"text": ["Alice"]})
    dtypes: dict[str, pd.api.extensions.ExtensionDtype] = {"text": pd.StringDtype()}
    captured = FramePublication.capture(
        frame,
        associations=(SemanticAssociation(task=TaskAttemptId.new(activation=activation)),),
        text_ports=(("text", "out"),),
        failed_records=(),
        opaque_source=frame,
        observation_dtypes=dtypes,
    )
    dtypes["text"] = pd.Int64Dtype()  # the caller's own mapping, after capture
    assert captured._observation_dtypes == (("text", pd.StringDtype()),)
