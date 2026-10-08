# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure request inspection through exact occurrence and binding ownership joins."""

from __future__ import annotations

from dataclasses import dataclass

from anonymizer.engine.graph_sdk._effect_values import PrivateValue
from anonymizer.engine.graph_sdk._execution_values import ExecutionResult, OperationOutputKey, PublicationFailureFact
from anonymizer.engine.graph_sdk.context import BindingArtifactRef, BindingReceipt, SourceTerminal
from anonymizer.engine.graph_sdk.records import TerminalFact
from anonymizer.engine.graph_sdk.requests import (
    BindingAssociation,
    BindingRequestScope,
    InvocationRequestScope,
    PhysicalRequestId,
    RequestAssociation,
    RequestAssociationSummary,
    RequestTerminalFact,
    SemanticAssociation,
)
from anonymizer.graph._values import ArtifactRef, DatumId


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class RequestInspection(PrivateValue):
    """First physical acceptance alongside current owner facts, without values.

    Multiple physical attempts may join the same semantic occurrence. Their first
    acceptance facts stay distinct; publication facts belong to the occurrence.
    Binding references retain their own type and never become graph references.
    """

    request: PhysicalRequestId
    association: RequestAssociation
    physical_terminal: RequestTerminalFact | None
    acceptance: RequestAssociationSummary | None
    ownership_resolved: bool
    target: DatumId | None
    operation_terminal: TerminalFact | None = None
    binding_terminal: SourceTerminal | None = None
    artifacts: tuple[ArtifactRef, ...] = ()
    binding_artifacts: tuple[BindingArtifactRef, ...] = ()
    publication_failure: PublicationFailureFact | None = None


def inspect_binding_requests(receipt: BindingReceipt) -> tuple[RequestInspection, ...]:
    """Join a binding receipt without inventing an invocation or loading values."""
    views: list[RequestInspection] = []
    for dispatch in receipt.requests.dispatches:
        terminal = next((fact for fact in receipt.requests.terminals if fact.request == dispatch.request), None)
        for association in dispatch.associations:
            acceptance = (
                None
                if terminal is None
                else next(
                    (row for row in terminal.results if row.association == association),
                    None,
                )
            )
            sources = tuple(
                source
                for source in receipt.sources
                if isinstance(association, BindingAssociation)
                and source.identity == association.declaration
                and source.identity.binding == receipt.binding
            )
            resolved = (
                isinstance(receipt.requests.scope, BindingRequestScope)
                and receipt.requests.scope.binding == receipt.binding
                and len(sources) == 1
            )
            source = sources[0] if resolved else None
            artifacts = tuple(
                item.reference
                for item in receipt.artifacts
                if source is not None
                and item.reference.declaration == source.identity
                and item.target == source.declaration.target
                and item.source == source.declaration.source
                and item.node == source.declaration.node
                and item.port == source.declaration.port
            )
            views.append(
                RequestInspection(
                    request=dispatch.request,
                    association=association,
                    physical_terminal=terminal,
                    acceptance=acceptance,
                    ownership_resolved=resolved,
                    target=None if source is None else source.declaration.target,
                    binding_terminal=None if source is None else source.terminal,
                    binding_artifacts=artifacts,
                )
            )
    return tuple(views)


def inspect_requests(result: ExecutionResult) -> tuple[RequestInspection, ...]:
    """Resolve first acceptance against exact final occurrence publication facts."""
    views: list[RequestInspection] = []
    for dispatch in result.requests.dispatches:
        terminal = next((fact for fact in result.requests.terminals if fact.request == dispatch.request), None)
        for association in dispatch.associations:
            acceptance = (
                None
                if terminal is None
                else next(
                    (row for row in terminal.results if row.association == association),
                    None,
                )
            )
            operations = tuple(
                fact
                for fact in result.record.terminals
                if isinstance(association, SemanticAssociation)
                and fact.attempt == association.task
                and fact.activation == association.task.activation
            )
            targets = tuple(
                target.target
                for target, state in zip(
                    result._execution.context.prepared.target_occurrences, result.states, strict=True
                )
                if isinstance(association, SemanticAssociation)
                and any(entry.activation == association.task.activation for entry in state.entries)
            )
            resolved = (
                len(operations) == len(targets) == 1
                and isinstance(result.requests.scope, InvocationRequestScope)
                and isinstance(association, SemanticAssociation)
                and association.task.activation.invocation == result.requests.scope.invocation
            )
            operation = operations[0] if resolved else None
            target = targets[0] if resolved else None
            entries = tuple(
                entry
                for state in result.states
                for entry in state.entries
                if operation is not None and entry.activation == operation.activation
            )
            expected_ports: frozenset[str] = frozenset()
            if len(entries) == 1 and entries[0].outcome is not None:
                node = result._execution.nodes[entries[0].template]
                outcomes = tuple(item for item in node.operation.outcomes if item.name == entries[0].outcome)
                if len(outcomes) == 1:
                    expected_ports = outcomes[0].produced_ports
                else:
                    resolved = False
            elif len(entries) != 1:
                resolved = False
            ports = tuple(
                port
                for port in result.ports
                if operation is not None
                and port.activation == operation.activation
                and port.target == target
                and port.port in expected_ports
            )
            resolved = (
                resolved and len(ports) == len(expected_ports) and {port.port for port in ports} == expected_ports
            )
            artifacts = tuple(
                dict.fromkeys(
                    port.artifact
                    for port in ports
                    if any(
                        fact.artifact == port.artifact
                        and isinstance(fact.key, OperationOutputKey)
                        and fact.key.activation == port.activation
                        and fact.key.target == target
                        and fact.key.port == port.port
                        for fact in result.provenance
                    )
                )
            )
            failures = tuple(
                fact
                for fact in result.publication_failures
                if fact.association == association and fact.target == target
            )
            # A partial provenance join is not evidence of complete publication.
            if any(port.artifact not in artifacts for port in ports) or len(failures) > 1:
                resolved = False
            views.append(
                RequestInspection(
                    request=dispatch.request,
                    association=association,
                    physical_terminal=terminal,
                    acceptance=acceptance,
                    ownership_resolved=resolved,
                    target=target,
                    operation_terminal=operation,
                    artifacts=artifacts if resolved else (),
                    publication_failure=failures[0] if len(failures) == 1 else None,
                )
            )
    bound = result._execution.context.bound_context
    if bound is not None:
        views.extend(inspect_binding_requests(bound.receipt))
    return tuple(views)
