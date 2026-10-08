# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qualification inputs and context binding and recovery providers."""

from __future__ import annotations

from dataclasses import dataclass

from anonymizer.engine.graph_sdk.context import (
    ContextResource,
    ContextSelector,
    ContextSourceCapability,
    ContextSourceRef,
    RetrievalBounds,
    SourceItem,
    SourceResponse,
)
from anonymizer.engine.graph_sdk.evidence import AssessmentSubmission, admit_qualification, evidence_revision_view
from anonymizer.engine.graph_sdk.executor import AdmittedExecutionPlan, ExecutionResult
from anonymizer.engine.graph_sdk.requests import (
    AssociationResult,
    DispatchEnvelope,
    ExactUsage,
    ExternalSettlement,
    FailureClass,
    PhysicalRequestId,
    PhysicalRequestPolicy,
    PortArtifact,
    RequestAssociation,
    StopConfirmed,
    TransportFailure,
    TransportResult,
    TransportSuccess,
    UnknownUsage,
)
from anonymizer.graph.workflow import ArtifactType
from tests.graph_sdk.evidence_fixtures import _qualification_limits


def _inputs(execution: AdmittedExecutionPlan, result: ExecutionResult, *, decisions: int = 16, latest: bool = False):
    admitted = admit_qualification(
        execution=execution,
        productions=()
        if execution.context.prepared.configuration.purpose == "execution_only"
        else execution.assessment_productions,
        limits=_qualification_limits(max_required_decisions=decisions),
    )
    current = evidence_revision_view(
        admitted=admitted,
        result=result,
        artifacts=tuple(
            {ref.key: ref for ref, _ in sorted(result.artifacts, key=lambda item: item[0].version)}.values()
        )
        if latest
        else tuple(ref for ref, _ in result.artifacts),
        absences=tuple({absence for fact in result.assessments for absence in fact.environment.absences}),
        configurations=tuple({(fact.node, fact.environment.configuration) for fact in result.assessments}),
        state=execution.context.prepared.state,
    )
    submissions = tuple(AssessmentSubmission(fact=fact) for fact in result.assessments)
    return admitted, current, submissions


@dataclass
class _RecoveringTransport:
    failure: FailureClass | None = None
    unknown_usage: bool = False
    fail_close: bool = False
    calls: int = 0
    closes: int = 0

    async def dispatch(self, request: DispatchEnvelope) -> TransportResult:
        self.calls += 1
        settlement = ExternalSettlement(
            request=request.request,
            disposition="completed",
            remote_stopped=True,
            usage=UnknownUsage()
            if self.unknown_usage and self.calls == 1
            else ExactUsage(input_units=1, output_units=1),
        )
        if self.calls == 1 and self.failure is not None:
            return TransportFailure(failure=self.failure, settlement=settlement)
        item = request.associations[0]
        return TransportSuccess(
            results=(
                AssociationResult(
                    association=item.association,
                    outcome="ok",
                    consumed_context_ports=frozenset(),
                    outputs=(
                        PortArtifact(
                            port="context",
                            artifact_type=item.inputs[0].artifact_type,
                            artifact=None,
                            value=item.inputs[0].value,
                        ),
                    ),
                ),
            ),
            settlement=settlement,
        )

    async def cancel(self, request: object) -> StopConfirmed:
        return StopConfirmed(usage=ExactUsage(input_units=0, output_units=0))

    async def close(self) -> None:
        self.closes += 1
        if self.fail_close:
            raise RuntimeError("test close failure")


@dataclass
class _BindingSource:
    source: ContextSourceRef
    fail_close: bool = False
    calls: int = 0
    closes: int = 0
    versions: tuple[int, ...] | None = None

    async def retrieve(
        self,
        *,
        request: PhysicalRequestId,
        association: RequestAssociation,
        selector: ContextSelector,
        bounds: RetrievalBounds,
    ) -> SourceResponse:
        del selector, bounds
        self.calls += 1
        return SourceResponse(
            source=self.source,
            items=tuple(
                SourceItem(
                    association=association,
                    key=0,
                    version=version,
                    text="initial context" if self.versions is None else str(version),
                )
                for version in ((1,) if self.versions is None else self.versions)
            ),
            settlement=ExternalSettlement(
                request=request,
                disposition="completed",
                usage=ExactUsage(input_units=1, output_units=1),
                remote_stopped=True,
            ),
        )

    async def cancel(self, request: PhysicalRequestId) -> StopConfirmed:
        del request
        return StopConfirmed(usage=ExactUsage(input_units=0, output_units=0))


class _BindingProvider(_BindingSource):
    async def close(self) -> None:
        self.closes += 1
        if self.fail_close:
            raise RuntimeError("provider cleanup failed")


def _initial_resource(provider: _BindingSource) -> ContextResource:
    return ContextResource(
        source=provider.source,
        capability=ContextSourceCapability(
            source=provider.source,
            artifact_type=ArtifactType(name="text", revision=1),
            uses=frozenset({"initial_binding"}),
            execution="async",
            resource_owner="sdk",
            cancellation="cooperative_ack",
            settlement="explicit_ack",
            usage="exact",
            request=PhysicalRequestPolicy(
                visibility="dispatch_and_settlement",
                pre_dispatch_control="executor",
                retry_owner="executor",
                replay="idempotent",
                max_attempts=1,
            ),
            safe_detachment="forbidden",
        ),
        lease=None,
        factory=lambda: provider,
    )
