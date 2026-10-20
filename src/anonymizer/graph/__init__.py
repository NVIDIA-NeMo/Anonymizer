# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public graph authoring, local execution and retained-result APIs."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anonymizer.engine.graph_sdk._activation_values import ActivationLimits as ActivationLimits
    from anonymizer.engine.graph_sdk._effect_values import EffectCode as EffectCode
    from anonymizer.engine.graph_sdk._effect_values import EffectRejected as EffectRejected
    from anonymizer.engine.graph_sdk._graph_values import ActivationKey as ActivationKey
    from anonymizer.engine.graph_sdk._graph_values import ArtifactRef as ArtifactRef
    from anonymizer.engine.graph_sdk._graph_values import ContractViolation as ContractViolation
    from anonymizer.engine.graph_sdk._graph_values import DatumId as DatumId
    from anonymizer.engine.graph_sdk._graph_values import InvocationId as InvocationId
    from anonymizer.engine.graph_sdk._graph_values import TaskAttemptId as TaskAttemptId
    from anonymizer.engine.graph_sdk._graph_values import ValidationCode as ValidationCode
    from anonymizer.engine.graph_sdk._workflow_admission import compile_graph as compile_graph
    from anonymizer.engine.graph_sdk._workflow_values import ArtifactType as ArtifactType
    from anonymizer.engine.graph_sdk._workflow_values import ChoiceBranch as ChoiceBranch
    from anonymizer.engine.graph_sdk._workflow_values import ChoiceDecl as ChoiceDecl
    from anonymizer.engine.graph_sdk._workflow_values import ContextInputRef as ContextInputRef
    from anonymizer.engine.graph_sdk._workflow_values import ContextUse as ContextUse
    from anonymizer.engine.graph_sdk._workflow_values import CoverageAtom as CoverageAtom
    from anonymizer.engine.graph_sdk._workflow_values import DynamicLimits as DynamicLimits
    from anonymizer.engine.graph_sdk._workflow_values import EvidencePort as EvidencePort
    from anonymizer.engine.graph_sdk._workflow_values import EvidencePromise as EvidencePromise
    from anonymizer.engine.graph_sdk._workflow_values import Graph as Graph
    from anonymizer.engine.graph_sdk._workflow_values import InputBinding as InputBinding
    from anonymizer.engine.graph_sdk._workflow_values import InputPort as InputPort
    from anonymizer.engine.graph_sdk._workflow_values import KeyedJoinDecl as KeyedJoinDecl
    from anonymizer.engine.graph_sdk._workflow_values import LoopCarriedBinding as LoopCarriedBinding
    from anonymizer.engine.graph_sdk._workflow_values import LoopDecl as LoopDecl
    from anonymizer.engine.graph_sdk._workflow_values import LoopInitialBinding as LoopInitialBinding
    from anonymizer.engine.graph_sdk._workflow_values import MapDecl as MapDecl
    from anonymizer.engine.graph_sdk._workflow_values import MapItemPort as MapItemPort
    from anonymizer.engine.graph_sdk._workflow_values import ModelRequirement as ModelRequirement
    from anonymizer.engine.graph_sdk._workflow_values import NodeId as NodeId
    from anonymizer.engine.graph_sdk._workflow_values import NodeInputRef as NodeInputRef
    from anonymizer.engine.graph_sdk._workflow_values import NodeOutcomeRef as NodeOutcomeRef
    from anonymizer.engine.graph_sdk._workflow_values import NodeOutputRef as NodeOutputRef
    from anonymizer.engine.graph_sdk._workflow_values import OperationNode as OperationNode
    from anonymizer.engine.graph_sdk._workflow_values import OperationSpec as OperationSpec
    from anonymizer.engine.graph_sdk._workflow_values import OutcomeBinding as OutcomeBinding
    from anonymizer.engine.graph_sdk._workflow_values import OutcomeSpec as OutcomeSpec
    from anonymizer.engine.graph_sdk._workflow_values import OutputBinding as OutputBinding
    from anonymizer.engine.graph_sdk._workflow_values import OutputDependency as OutputDependency
    from anonymizer.engine.graph_sdk._workflow_values import OutputPort as OutputPort
    from anonymizer.engine.graph_sdk._workflow_values import ProtectionRequirement as ProtectionRequirement
    from anonymizer.engine.graph_sdk._workflow_values import ResourceCeiling as ResourceCeiling
    from anonymizer.engine.graph_sdk._workflow_values import SequenceEdge as SequenceEdge
    from anonymizer.engine.graph_sdk._workflow_values import SubgraphNode as SubgraphNode
    from anonymizer.engine.graph_sdk._workflow_values import WorkflowId as WorkflowId
    from anonymizer.engine.graph_sdk._workflow_values import WorkflowInputRef as WorkflowInputRef
    from anonymizer.engine.graph_sdk._workflow_values import WorkflowLimits as WorkflowLimits
    from anonymizer.engine.graph_sdk._workflow_values import WorkflowOutcomeRef as WorkflowOutcomeRef
    from anonymizer.engine.graph_sdk._workflow_values import WorkflowOutputRef as WorkflowOutputRef
    from anonymizer.engine.graph_sdk.binding import BindingRejected as BindingRejected
    from anonymizer.engine.graph_sdk.binding import RunningBinding as RunningBinding
    from anonymizer.engine.graph_sdk.binding import start_initial_binding as start_initial_binding
    from anonymizer.engine.graph_sdk.blocking import BlockingAffinity as BlockingAffinity
    from anonymizer.engine.graph_sdk.blocking import BlockingContextProvider as BlockingContextProvider
    from anonymizer.engine.graph_sdk.blocking import BlockingPool as BlockingPool
    from anonymizer.engine.graph_sdk.blocking import BlockingTransport as BlockingTransport
    from anonymizer.engine.graph_sdk.capabilities import ConfigBoolean as ConfigBoolean
    from anonymizer.engine.graph_sdk.capabilities import ConfigField as ConfigField
    from anonymizer.engine.graph_sdk.capabilities import ConfigInteger as ConfigInteger
    from anonymizer.engine.graph_sdk.capabilities import ConfigNull as ConfigNull
    from anonymizer.engine.graph_sdk.capabilities import ConfigNumber as ConfigNumber
    from anonymizer.engine.graph_sdk.capabilities import ConfigSequence as ConfigSequence
    from anonymizer.engine.graph_sdk.capabilities import ConfigText as ConfigText
    from anonymizer.engine.graph_sdk.capabilities import FrozenConfig as FrozenConfig
    from anonymizer.engine.graph_sdk.capabilities import ImplementationCapability as ImplementationCapability
    from anonymizer.engine.graph_sdk.capabilities import ImplementationRef as ImplementationRef
    from anonymizer.engine.graph_sdk.capabilities import ImplementationSelection as ImplementationSelection
    from anonymizer.engine.graph_sdk.capabilities import PreparationCode as PreparationCode
    from anonymizer.engine.graph_sdk.capabilities import PreparationRejected as PreparationRejected
    from anonymizer.engine.graph_sdk.context import AdmittedContextPlan as AdmittedContextPlan
    from anonymizer.engine.graph_sdk.context import BindingLimits as BindingLimits
    from anonymizer.engine.graph_sdk.context import BindingReceipt as BindingReceipt
    from anonymizer.engine.graph_sdk.context import BindingResult as BindingResult
    from anonymizer.engine.graph_sdk.context import BoundContext as BoundContext
    from anonymizer.engine.graph_sdk.context import ContextMaterialization as ContextMaterialization
    from anonymizer.engine.graph_sdk.context import ContextProvider as ContextProvider
    from anonymizer.engine.graph_sdk.context import ContextResource as ContextResource
    from anonymizer.engine.graph_sdk.context import ContextSelector as ContextSelector
    from anonymizer.engine.graph_sdk.context import ContextSourceCapability as ContextSourceCapability
    from anonymizer.engine.graph_sdk.context import ContextSourceRef as ContextSourceRef
    from anonymizer.engine.graph_sdk.context import InitialContextDecl as InitialContextDecl
    from anonymizer.engine.graph_sdk.context import InitialVersionSelection as InitialVersionSelection
    from anonymizer.engine.graph_sdk.context import RetrievalBounds as RetrievalBounds
    from anonymizer.engine.graph_sdk.context import SdkContextFactory as SdkContextFactory
    from anonymizer.engine.graph_sdk.context import SelectorField as SelectorField
    from anonymizer.engine.graph_sdk.context import SourceFailure as SourceFailure
    from anonymizer.engine.graph_sdk.context import SourceItem as SourceItem
    from anonymizer.engine.graph_sdk.context import SourceLost as SourceLost
    from anonymizer.engine.graph_sdk.context import SourceResponse as SourceResponse
    from anonymizer.engine.graph_sdk.context import SourceResult as SourceResult
    from anonymizer.engine.graph_sdk.context import admit_context_plan as admit_context_plan
    from anonymizer.engine.graph_sdk.data import AtomicGroup as AtomicGroup
    from anonymizer.engine.graph_sdk.data import CoherenceScope as CoherenceScope
    from anonymizer.engine.graph_sdk.data import ContextView as ContextView
    from anonymizer.engine.graph_sdk.data import DataGraph as DataGraph
    from anonymizer.engine.graph_sdk.data import DataLimits as DataLimits
    from anonymizer.engine.graph_sdk.data import Datum as Datum
    from anonymizer.engine.graph_sdk.data import DatumDependency as DatumDependency
    from anonymizer.engine.graph_sdk.data import OutputRegion as OutputRegion
    from anonymizer.engine.graph_sdk.data import SourceRelation as SourceRelation
    from anonymizer.engine.graph_sdk.data import SourceView as SourceView
    from anonymizer.engine.graph_sdk.data import ValidatedDataGraph as ValidatedDataGraph
    from anonymizer.engine.graph_sdk.evidence import AdmittedQualification as AdmittedQualification
    from anonymizer.engine.graph_sdk.evidence import EvidenceRevisionView as EvidenceRevisionView
    from anonymizer.engine.graph_sdk.evidence import MapItemSubjectRef as MapItemSubjectRef
    from anonymizer.engine.graph_sdk.evidence import QualificationLimits as QualificationLimits
    from anonymizer.engine.graph_sdk.evidence import VerifiedEvidence as VerifiedEvidence
    from anonymizer.engine.graph_sdk.evidence import admit_qualification as admit_qualification
    from anonymizer.engine.graph_sdk.evidence import evidence_revision_view as evidence_revision_view
    from anonymizer.engine.graph_sdk.evidence import evidence_validity as evidence_validity
    from anonymizer.engine.graph_sdk.executor import AdmittedExecutionPlan as AdmittedExecutionPlan
    from anonymizer.engine.graph_sdk.executor import AssessmentEnvironment as AssessmentEnvironment
    from anonymizer.engine.graph_sdk.executor import AssessmentFinding as AssessmentFinding
    from anonymizer.engine.graph_sdk.executor import AssessmentLimits as AssessmentLimits
    from anonymizer.engine.graph_sdk.executor import EvidenceProductionDecl as EvidenceProductionDecl
    from anonymizer.engine.graph_sdk.executor import ExecutionAssessmentFact as ExecutionAssessmentFact
    from anonymizer.engine.graph_sdk.executor import ExecutionImplementation as ExecutionImplementation
    from anonymizer.engine.graph_sdk.executor import ExecutionLimits as ExecutionLimits
    from anonymizer.engine.graph_sdk.executor import ExecutionRejected as ExecutionRejected
    from anonymizer.engine.graph_sdk.executor import ExecutionResult as ExecutionResult
    from anonymizer.engine.graph_sdk.executor import ExecutionServices as ExecutionServices
    from anonymizer.engine.graph_sdk.executor import ImplementationHandle as ImplementationHandle
    from anonymizer.engine.graph_sdk.executor import IngressLimits as IngressLimits
    from anonymizer.engine.graph_sdk.executor import LocalAssessmentResult as LocalAssessmentResult
    from anonymizer.engine.graph_sdk.executor import LocalCallable as LocalCallable
    from anonymizer.engine.graph_sdk.executor import LocalCompleted as LocalCompleted
    from anonymizer.engine.graph_sdk.executor import LocalFailure as LocalFailure
    from anonymizer.engine.graph_sdk.executor import LocalResult as LocalResult
    from anonymizer.engine.graph_sdk.executor import MapExpansionDecl as MapExpansionDecl
    from anonymizer.engine.graph_sdk.executor import MapItemKey as MapItemKey
    from anonymizer.engine.graph_sdk.executor import MonotonicClock as MonotonicClock
    from anonymizer.engine.graph_sdk.executor import NestedEventLoopError as NestedEventLoopError
    from anonymizer.engine.graph_sdk.executor import OperationExecutionPolicy as OperationExecutionPolicy
    from anonymizer.engine.graph_sdk.executor import RequestTransport as RequestTransport
    from anonymizer.engine.graph_sdk.executor import RunningExecution as RunningExecution
    from anonymizer.engine.graph_sdk.executor import RuntimeOutcome as RuntimeOutcome
    from anonymizer.engine.graph_sdk.executor import admit_execution_plan as admit_execution_plan
    from anonymizer.engine.graph_sdk.executor import execute_sync as execute_sync
    from anonymizer.engine.graph_sdk.executor import start_execution as start_execution
    from anonymizer.engine.graph_sdk.extensions import ArtifactSelection as ArtifactSelection
    from anonymizer.engine.graph_sdk.extensions import ContentProjection as ContentProjection
    from anonymizer.engine.graph_sdk.extensions import ProjectionSource as ProjectionSource
    from anonymizer.engine.graph_sdk.extensions import ProtectedProjection as ProtectedProjection
    from anonymizer.engine.graph_sdk.extensions import RemoteContext as RemoteContext
    from anonymizer.engine.graph_sdk.extensions import bind_projection_source as bind_projection_source
    from anonymizer.engine.graph_sdk.extensions import project_content as project_content
    from anonymizer.engine.graph_sdk.extensions import project_protected as project_protected
    from anonymizer.engine.graph_sdk.inspection import BindingInspectionSource as BindingInspectionSource
    from anonymizer.engine.graph_sdk.inspection import InspectionLimits as InspectionLimits
    from anonymizer.engine.graph_sdk.inspection import InspectionSource as InspectionSource
    from anonymizer.engine.graph_sdk.inspection import bind_binding_inspection_source as bind_binding_inspection_source
    from anonymizer.engine.graph_sdk.inspection import bind_inspection_source as bind_inspection_source
    from anonymizer.engine.graph_sdk.inspection import diagnose as diagnose
    from anonymizer.engine.graph_sdk.inspection import diagnose_binding as diagnose_binding
    from anonymizer.engine.graph_sdk.inspection import diagnose_observations as diagnose_observations
    from anonymizer.engine.graph_sdk.inspection import encode_inspection as encode_inspection
    from anonymizer.engine.graph_sdk.inspection import explain as explain
    from anonymizer.engine.graph_sdk.inspection import inspect as inspect
    from anonymizer.engine.graph_sdk.inspection import inspect_binding as inspect_binding
    from anonymizer.engine.graph_sdk.observations import CorrelationRef as CorrelationRef
    from anonymizer.engine.graph_sdk.observations import LifecyclePair as LifecyclePair
    from anonymizer.engine.graph_sdk.observations import NodeMeasurement as NodeMeasurement
    from anonymizer.engine.graph_sdk.observations import ObservationFailure as ObservationFailure
    from anonymizer.engine.graph_sdk.observations import ObservationLimits as ObservationLimits
    from anonymizer.engine.graph_sdk.observations import ObservationReceipt as ObservationReceipt
    from anonymizer.engine.graph_sdk.observations import ObservationRequirements as ObservationRequirements
    from anonymizer.engine.graph_sdk.observations import ObservationSession as ObservationSession
    from anonymizer.engine.graph_sdk.observations import ObservationSink as ObservationSink
    from anonymizer.engine.graph_sdk.observations import ObservationStart as ObservationStart
    from anonymizer.engine.graph_sdk.observations import ObservationTerminal as ObservationTerminal
    from anonymizer.engine.graph_sdk.observations import WorkMeasurement as WorkMeasurement
    from anonymizer.engine.graph_sdk.observations import begin_observations as begin_observations
    from anonymizer.engine.graph_sdk.observations import (
        binding_observation_requirements as binding_observation_requirements,
    )
    from anonymizer.engine.graph_sdk.observations import (
        execution_observation_requirements as execution_observation_requirements,
    )
    from anonymizer.engine.graph_sdk.observations import observed_context_admission as observed_context_admission
    from anonymizer.engine.graph_sdk.observations import observed_execution_admission as observed_execution_admission
    from anonymizer.engine.graph_sdk.observations import observed_prepare as observed_prepare
    from anonymizer.engine.graph_sdk.observations import observed_project_content as observed_project_content
    from anonymizer.engine.graph_sdk.observations import observed_project_protected as observed_project_protected
    from anonymizer.engine.graph_sdk.observations import observed_qualify_retained as _observed_qualify
    from anonymizer.engine.graph_sdk.observations import (
        post_execution_observation_requirements as post_execution_observation_requirements,
    )
    from anonymizer.engine.graph_sdk.observations import (
        prepared_observation_requirements as prepared_observation_requirements,
    )
    from anonymizer.engine.graph_sdk.observations import reserve_observation_scope as reserve_observation_scope
    from anonymizer.engine.graph_sdk.preparation import BoundInput as BoundInput
    from anonymizer.engine.graph_sdk.preparation import PreparationConfiguration as PreparationConfiguration
    from anonymizer.engine.graph_sdk.preparation import PreparationLimits as PreparationLimits
    from anonymizer.engine.graph_sdk.preparation import PreparedPlan as PreparedPlan
    from anonymizer.engine.graph_sdk.preparation import prepare as prepare
    from anonymizer.engine.graph_sdk.preparation import recheck_capabilities as recheck_capabilities
    from anonymizer.engine.graph_sdk.qualification import QualificationResult as QualificationResult
    from anonymizer.engine.graph_sdk.qualification import QualifiedOutput as QualifiedOutput
    from anonymizer.engine.graph_sdk.qualification import TargetQualification as TargetQualification
    from anonymizer.engine.graph_sdk.qualification import qualify_retained as _qualify
    from anonymizer.engine.graph_sdk.records import CandidateRef as CandidateRef
    from anonymizer.engine.graph_sdk.records import CanonicalRecord as CanonicalRecord
    from anonymizer.engine.graph_sdk.records import EvidenceRef as EvidenceRef
    from anonymizer.engine.graph_sdk.records import TargetStatus as TargetStatus
    from anonymizer.engine.graph_sdk.requests import ArtifactValue as ArtifactValue
    from anonymizer.engine.graph_sdk.requests import AssociationInput as AssociationInput
    from anonymizer.engine.graph_sdk.requests import AssociationResult as AssociationResult
    from anonymizer.engine.graph_sdk.requests import BindingAssociation as BindingAssociation
    from anonymizer.engine.graph_sdk.requests import DispatchEnvelope as DispatchEnvelope
    from anonymizer.engine.graph_sdk.requests import ExactUsage as ExactUsage
    from anonymizer.engine.graph_sdk.requests import ExternalSettlement as ExternalSettlement
    from anonymizer.engine.graph_sdk.requests import FailureClass as FailureClass
    from anonymizer.engine.graph_sdk.requests import PhysicalRequestId as PhysicalRequestId
    from anonymizer.engine.graph_sdk.requests import PhysicalRequestPolicy as PhysicalRequestPolicy
    from anonymizer.engine.graph_sdk.requests import PortArtifact as PortArtifact
    from anonymizer.engine.graph_sdk.requests import RequestAssociation as RequestAssociation
    from anonymizer.engine.graph_sdk.requests import RequestReceipt as RequestReceipt
    from anonymizer.engine.graph_sdk.requests import RequestTerminalFact as RequestTerminalFact
    from anonymizer.engine.graph_sdk.requests import SemanticAssociation as SemanticAssociation
    from anonymizer.engine.graph_sdk.requests import StopConfirmed as StopConfirmed
    from anonymizer.engine.graph_sdk.requests import StopResult as StopResult
    from anonymizer.engine.graph_sdk.requests import StopUnknown as StopUnknown
    from anonymizer.engine.graph_sdk.requests import TextArtifactValue as TextArtifactValue
    from anonymizer.engine.graph_sdk.requests import TextCollectionItem as TextCollectionItem
    from anonymizer.engine.graph_sdk.requests import TextCollectionValue as TextCollectionValue
    from anonymizer.engine.graph_sdk.requests import TransportFailure as TransportFailure
    from anonymizer.engine.graph_sdk.requests import TransportLost as TransportLost
    from anonymizer.engine.graph_sdk.requests import TransportResult as TransportResult
    from anonymizer.engine.graph_sdk.requests import TransportSuccess as TransportSuccess
    from anonymizer.engine.graph_sdk.requests import UnknownUsage as UnknownUsage
    from anonymizer.engine.graph_sdk.resources import CleanupFact as CleanupFact
    from anonymizer.engine.graph_sdk.resources import CloseableResource as CloseableResource
    from anonymizer.engine.graph_sdk.resources import ResourceId as ResourceId
    from anonymizer.engine.graph_sdk.resources import ResourceLease as ResourceLease

    qualify = _qualify
    observed_qualify = _observed_qualify

_EXPORTS: dict[str, str] = {
    "diagnose_observations": "anonymizer.engine.graph_sdk.inspection",
    "BindingRejected": "anonymizer.engine.graph_sdk.binding",
    "MapItemKey": "anonymizer.engine.graph_sdk.executor",
    "MapExpansionDecl": "anonymizer.engine.graph_sdk.executor",
    "BlockingAffinity": "anonymizer.engine.graph_sdk.blocking",
    "qualify": "anonymizer.engine.graph_sdk.qualification",
    "observed_qualify": "anonymizer.engine.graph_sdk.observations",
    "ActivationKey": "anonymizer.engine.graph_sdk._graph_values",
    "ActivationLimits": "anonymizer.engine.graph_sdk._activation_values",
    "Graph": "anonymizer.engine.graph_sdk._workflow_values",
    "AdmittedContextPlan": "anonymizer.engine.graph_sdk.context",
    "AdmittedExecutionPlan": "anonymizer.engine.graph_sdk.executor",
    "AdmittedQualification": "anonymizer.engine.graph_sdk.evidence",
    "ArtifactRef": "anonymizer.engine.graph_sdk._graph_values",
    "ArtifactSelection": "anonymizer.engine.graph_sdk.extensions",
    "ArtifactType": "anonymizer.engine.graph_sdk._workflow_values",
    "ArtifactValue": "anonymizer.engine.graph_sdk.requests",
    "AssessmentEnvironment": "anonymizer.engine.graph_sdk.executor",
    "AssessmentFinding": "anonymizer.engine.graph_sdk.executor",
    "AssessmentLimits": "anonymizer.engine.graph_sdk.executor",
    "AssociationInput": "anonymizer.engine.graph_sdk.requests",
    "AssociationResult": "anonymizer.engine.graph_sdk.requests",
    "AtomicGroup": "anonymizer.engine.graph_sdk.data",
    "BindingAssociation": "anonymizer.engine.graph_sdk.requests",
    "BindingInspectionSource": "anonymizer.engine.graph_sdk.inspection",
    "BindingLimits": "anonymizer.engine.graph_sdk.context",
    "BindingReceipt": "anonymizer.engine.graph_sdk.context",
    "BindingResult": "anonymizer.engine.graph_sdk.context",
    "BlockingContextProvider": "anonymizer.engine.graph_sdk.blocking",
    "BlockingPool": "anonymizer.engine.graph_sdk.blocking",
    "BlockingTransport": "anonymizer.engine.graph_sdk.blocking",
    "BoundContext": "anonymizer.engine.graph_sdk.context",
    "BoundInput": "anonymizer.engine.graph_sdk.preparation",
    "CandidateRef": "anonymizer.engine.graph_sdk.records",
    "CanonicalRecord": "anonymizer.engine.graph_sdk.records",
    "ChoiceBranch": "anonymizer.engine.graph_sdk._workflow_values",
    "ChoiceDecl": "anonymizer.engine.graph_sdk._workflow_values",
    "CleanupFact": "anonymizer.engine.graph_sdk.resources",
    "CloseableResource": "anonymizer.engine.graph_sdk.resources",
    "CoherenceScope": "anonymizer.engine.graph_sdk.data",
    "ConfigBoolean": "anonymizer.engine.graph_sdk.capabilities",
    "ConfigField": "anonymizer.engine.graph_sdk.capabilities",
    "ConfigInteger": "anonymizer.engine.graph_sdk.capabilities",
    "ConfigNull": "anonymizer.engine.graph_sdk.capabilities",
    "ConfigNumber": "anonymizer.engine.graph_sdk.capabilities",
    "ConfigSequence": "anonymizer.engine.graph_sdk.capabilities",
    "ConfigText": "anonymizer.engine.graph_sdk.capabilities",
    "ContentProjection": "anonymizer.engine.graph_sdk.extensions",
    "ContextInputRef": "anonymizer.engine.graph_sdk._workflow_values",
    "ContextMaterialization": "anonymizer.engine.graph_sdk.context",
    "ContextProvider": "anonymizer.engine.graph_sdk.context",
    "ContextResource": "anonymizer.engine.graph_sdk.context",
    "ContextSelector": "anonymizer.engine.graph_sdk.context",
    "ContextSourceCapability": "anonymizer.engine.graph_sdk.context",
    "ContextSourceRef": "anonymizer.engine.graph_sdk.context",
    "ContextUse": "anonymizer.engine.graph_sdk._workflow_values",
    "ContextView": "anonymizer.engine.graph_sdk.data",
    "ContractViolation": "anonymizer.engine.graph_sdk._graph_values",
    "CorrelationRef": "anonymizer.engine.graph_sdk.observations",
    "CoverageAtom": "anonymizer.engine.graph_sdk._workflow_values",
    "DataGraph": "anonymizer.engine.graph_sdk.data",
    "DataLimits": "anonymizer.engine.graph_sdk.data",
    "Datum": "anonymizer.engine.graph_sdk.data",
    "DatumDependency": "anonymizer.engine.graph_sdk.data",
    "DatumId": "anonymizer.engine.graph_sdk._graph_values",
    "DispatchEnvelope": "anonymizer.engine.graph_sdk.requests",
    "DynamicLimits": "anonymizer.engine.graph_sdk._workflow_values",
    "EffectCode": "anonymizer.engine.graph_sdk._effect_values",
    "EffectRejected": "anonymizer.engine.graph_sdk._effect_values",
    "EvidencePort": "anonymizer.engine.graph_sdk._workflow_values",
    "EvidenceProductionDecl": "anonymizer.engine.graph_sdk.executor",
    "EvidencePromise": "anonymizer.engine.graph_sdk._workflow_values",
    "EvidenceRef": "anonymizer.engine.graph_sdk.records",
    "EvidenceRevisionView": "anonymizer.engine.graph_sdk.evidence",
    "ExactUsage": "anonymizer.engine.graph_sdk.requests",
    "ExecutionAssessmentFact": "anonymizer.engine.graph_sdk.executor",
    "ExecutionImplementation": "anonymizer.engine.graph_sdk.executor",
    "ExecutionLimits": "anonymizer.engine.graph_sdk.executor",
    "ExecutionRejected": "anonymizer.engine.graph_sdk.executor",
    "ExecutionResult": "anonymizer.engine.graph_sdk.executor",
    "ExecutionServices": "anonymizer.engine.graph_sdk.executor",
    "ExternalSettlement": "anonymizer.engine.graph_sdk.requests",
    "FailureClass": "anonymizer.engine.graph_sdk.requests",
    "FrozenConfig": "anonymizer.engine.graph_sdk.capabilities",
    "ImplementationCapability": "anonymizer.engine.graph_sdk.capabilities",
    "ImplementationHandle": "anonymizer.engine.graph_sdk.executor",
    "ImplementationRef": "anonymizer.engine.graph_sdk.capabilities",
    "ImplementationSelection": "anonymizer.engine.graph_sdk.capabilities",
    "IngressLimits": "anonymizer.engine.graph_sdk.executor",
    "InitialContextDecl": "anonymizer.engine.graph_sdk.context",
    "InitialVersionSelection": "anonymizer.engine.graph_sdk.context",
    "InputBinding": "anonymizer.engine.graph_sdk._workflow_values",
    "InputPort": "anonymizer.engine.graph_sdk._workflow_values",
    "InspectionLimits": "anonymizer.engine.graph_sdk.inspection",
    "InspectionSource": "anonymizer.engine.graph_sdk.inspection",
    "InvocationId": "anonymizer.engine.graph_sdk._graph_values",
    "KeyedJoinDecl": "anonymizer.engine.graph_sdk._workflow_values",
    "LifecyclePair": "anonymizer.engine.graph_sdk.observations",
    "LocalAssessmentResult": "anonymizer.engine.graph_sdk.executor",
    "LocalCallable": "anonymizer.engine.graph_sdk.executor",
    "LocalCompleted": "anonymizer.engine.graph_sdk.executor",
    "LocalFailure": "anonymizer.engine.graph_sdk.executor",
    "LocalResult": "anonymizer.engine.graph_sdk.executor",
    "LoopCarriedBinding": "anonymizer.engine.graph_sdk._workflow_values",
    "LoopDecl": "anonymizer.engine.graph_sdk._workflow_values",
    "LoopInitialBinding": "anonymizer.engine.graph_sdk._workflow_values",
    "MapDecl": "anonymizer.engine.graph_sdk._workflow_values",
    "MapItemPort": "anonymizer.engine.graph_sdk._workflow_values",
    "MapItemSubjectRef": "anonymizer.engine.graph_sdk.evidence",
    "ModelRequirement": "anonymizer.engine.graph_sdk._workflow_values",
    "MonotonicClock": "anonymizer.engine.graph_sdk.executor",
    "NestedEventLoopError": "anonymizer.engine.graph_sdk.executor",
    "NodeId": "anonymizer.engine.graph_sdk._workflow_values",
    "NodeInputRef": "anonymizer.engine.graph_sdk._workflow_values",
    "NodeMeasurement": "anonymizer.engine.graph_sdk.observations",
    "NodeOutcomeRef": "anonymizer.engine.graph_sdk._workflow_values",
    "NodeOutputRef": "anonymizer.engine.graph_sdk._workflow_values",
    "ObservationFailure": "anonymizer.engine.graph_sdk.observations",
    "ObservationLimits": "anonymizer.engine.graph_sdk.observations",
    "ObservationReceipt": "anonymizer.engine.graph_sdk.observations",
    "ObservationRequirements": "anonymizer.engine.graph_sdk.observations",
    "ObservationSession": "anonymizer.engine.graph_sdk.observations",
    "ObservationSink": "anonymizer.engine.graph_sdk.observations",
    "ObservationStart": "anonymizer.engine.graph_sdk.observations",
    "ObservationTerminal": "anonymizer.engine.graph_sdk.observations",
    "OperationExecutionPolicy": "anonymizer.engine.graph_sdk.executor",
    "OperationNode": "anonymizer.engine.graph_sdk._workflow_values",
    "OperationSpec": "anonymizer.engine.graph_sdk._workflow_values",
    "OutcomeBinding": "anonymizer.engine.graph_sdk._workflow_values",
    "OutcomeSpec": "anonymizer.engine.graph_sdk._workflow_values",
    "OutputBinding": "anonymizer.engine.graph_sdk._workflow_values",
    "OutputDependency": "anonymizer.engine.graph_sdk._workflow_values",
    "OutputPort": "anonymizer.engine.graph_sdk._workflow_values",
    "OutputRegion": "anonymizer.engine.graph_sdk.data",
    "PhysicalRequestId": "anonymizer.engine.graph_sdk.requests",
    "PhysicalRequestPolicy": "anonymizer.engine.graph_sdk.requests",
    "PortArtifact": "anonymizer.engine.graph_sdk.requests",
    "PreparationCode": "anonymizer.engine.graph_sdk.capabilities",
    "PreparationConfiguration": "anonymizer.engine.graph_sdk.preparation",
    "PreparationLimits": "anonymizer.engine.graph_sdk.preparation",
    "PreparationRejected": "anonymizer.engine.graph_sdk.capabilities",
    "PreparedPlan": "anonymizer.engine.graph_sdk.preparation",
    "ProjectionSource": "anonymizer.engine.graph_sdk.extensions",
    "ProtectedProjection": "anonymizer.engine.graph_sdk.extensions",
    "ProtectionRequirement": "anonymizer.engine.graph_sdk._workflow_values",
    "QualificationLimits": "anonymizer.engine.graph_sdk.evidence",
    "QualificationResult": "anonymizer.engine.graph_sdk.qualification",
    "QualifiedOutput": "anonymizer.engine.graph_sdk.qualification",
    "RemoteContext": "anonymizer.engine.graph_sdk.extensions",
    "RequestAssociation": "anonymizer.engine.graph_sdk.requests",
    "RequestReceipt": "anonymizer.engine.graph_sdk.requests",
    "RequestTerminalFact": "anonymizer.engine.graph_sdk.requests",
    "RequestTransport": "anonymizer.engine.graph_sdk.executor",
    "ResourceCeiling": "anonymizer.engine.graph_sdk._workflow_values",
    "ResourceId": "anonymizer.engine.graph_sdk.resources",
    "ResourceLease": "anonymizer.engine.graph_sdk.resources",
    "RetrievalBounds": "anonymizer.engine.graph_sdk.context",
    "RunningBinding": "anonymizer.engine.graph_sdk.binding",
    "RunningExecution": "anonymizer.engine.graph_sdk.executor",
    "RuntimeOutcome": "anonymizer.engine.graph_sdk.executor",
    "SdkContextFactory": "anonymizer.engine.graph_sdk.context",
    "SelectorField": "anonymizer.engine.graph_sdk.context",
    "SemanticAssociation": "anonymizer.engine.graph_sdk.requests",
    "SequenceEdge": "anonymizer.engine.graph_sdk._workflow_values",
    "SourceFailure": "anonymizer.engine.graph_sdk.context",
    "SourceItem": "anonymizer.engine.graph_sdk.context",
    "SourceLost": "anonymizer.engine.graph_sdk.context",
    "SourceRelation": "anonymizer.engine.graph_sdk.data",
    "SourceResponse": "anonymizer.engine.graph_sdk.context",
    "SourceResult": "anonymizer.engine.graph_sdk.context",
    "SourceView": "anonymizer.engine.graph_sdk.data",
    "StopConfirmed": "anonymizer.engine.graph_sdk.requests",
    "StopResult": "anonymizer.engine.graph_sdk.requests",
    "StopUnknown": "anonymizer.engine.graph_sdk.requests",
    "SubgraphNode": "anonymizer.engine.graph_sdk._workflow_values",
    "TargetQualification": "anonymizer.engine.graph_sdk.qualification",
    "TargetStatus": "anonymizer.engine.graph_sdk.records",
    "TaskAttemptId": "anonymizer.engine.graph_sdk._graph_values",
    "TextArtifactValue": "anonymizer.engine.graph_sdk.requests",
    "TextCollectionItem": "anonymizer.engine.graph_sdk.requests",
    "TextCollectionValue": "anonymizer.engine.graph_sdk.requests",
    "TransportFailure": "anonymizer.engine.graph_sdk.requests",
    "TransportLost": "anonymizer.engine.graph_sdk.requests",
    "TransportResult": "anonymizer.engine.graph_sdk.requests",
    "TransportSuccess": "anonymizer.engine.graph_sdk.requests",
    "UnknownUsage": "anonymizer.engine.graph_sdk.requests",
    "ValidatedDataGraph": "anonymizer.engine.graph_sdk.data",
    "ValidationCode": "anonymizer.engine.graph_sdk._graph_values",
    "VerifiedEvidence": "anonymizer.engine.graph_sdk.evidence",
    "WorkMeasurement": "anonymizer.engine.graph_sdk.observations",
    "WorkflowId": "anonymizer.engine.graph_sdk._workflow_values",
    "WorkflowInputRef": "anonymizer.engine.graph_sdk._workflow_values",
    "WorkflowLimits": "anonymizer.engine.graph_sdk._workflow_values",
    "WorkflowOutcomeRef": "anonymizer.engine.graph_sdk._workflow_values",
    "WorkflowOutputRef": "anonymizer.engine.graph_sdk._workflow_values",
    "admit_context_plan": "anonymizer.engine.graph_sdk.context",
    "admit_execution_plan": "anonymizer.engine.graph_sdk.executor",
    "admit_qualification": "anonymizer.engine.graph_sdk.evidence",
    "compile_graph": "anonymizer.engine.graph_sdk._workflow_admission",
    "begin_observations": "anonymizer.engine.graph_sdk.observations",
    "bind_binding_inspection_source": "anonymizer.engine.graph_sdk.inspection",
    "bind_inspection_source": "anonymizer.engine.graph_sdk.inspection",
    "bind_projection_source": "anonymizer.engine.graph_sdk.extensions",
    "binding_observation_requirements": "anonymizer.engine.graph_sdk.observations",
    "diagnose": "anonymizer.engine.graph_sdk.inspection",
    "diagnose_binding": "anonymizer.engine.graph_sdk.inspection",
    "encode_inspection": "anonymizer.engine.graph_sdk.inspection",
    "evidence_revision_view": "anonymizer.engine.graph_sdk.evidence",
    "evidence_validity": "anonymizer.engine.graph_sdk.evidence",
    "execute_sync": "anonymizer.engine.graph_sdk.executor",
    "execution_observation_requirements": "anonymizer.engine.graph_sdk.observations",
    "explain": "anonymizer.engine.graph_sdk.inspection",
    "inspect": "anonymizer.engine.graph_sdk.inspection",
    "inspect_binding": "anonymizer.engine.graph_sdk.inspection",
    "observed_context_admission": "anonymizer.engine.graph_sdk.observations",
    "observed_execution_admission": "anonymizer.engine.graph_sdk.observations",
    "observed_prepare": "anonymizer.engine.graph_sdk.observations",
    "observed_project_content": "anonymizer.engine.graph_sdk.observations",
    "observed_project_protected": "anonymizer.engine.graph_sdk.observations",
    "post_execution_observation_requirements": "anonymizer.engine.graph_sdk.observations",
    "prepare": "anonymizer.engine.graph_sdk.preparation",
    "prepared_observation_requirements": "anonymizer.engine.graph_sdk.observations",
    "project_content": "anonymizer.engine.graph_sdk.extensions",
    "project_protected": "anonymizer.engine.graph_sdk.extensions",
    "recheck_capabilities": "anonymizer.engine.graph_sdk.preparation",
    "reserve_observation_scope": "anonymizer.engine.graph_sdk.observations",
    "start_execution": "anonymizer.engine.graph_sdk.executor",
    "start_initial_binding": "anonymizer.engine.graph_sdk.binding",
}


def __getattr__(name: str) -> object:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module), name + "_retained" if name in {"qualify", "observed_qualify"} else name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


# The explicit map is the single runtime export inventory.
__all__ = list(_EXPORTS)
