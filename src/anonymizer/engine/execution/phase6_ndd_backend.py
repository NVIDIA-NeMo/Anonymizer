# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage-specific NDD effects for the private Phase 6 Redact profile."""

from __future__ import annotations

import json
from enum import Enum
from typing import TypeVar

import pandas as pd
from data_designer.config.column_configs import LLMStructuredColumnConfig, LLMTextColumnConfig
from pydantic import BaseModel, ConfigDict

from anonymizer.engine.constants import (
    COL_PHASE6_AUGMENTED,
    COL_PHASE6_CANDIDATES,
    COL_PHASE6_CONTEXT,
    COL_PHASE6_VALIDATION,
    COL_RAW_DETECTED,
    COL_TEXT,
    DEFAULT_ENTITY_LABELS,
    _jinja,
)
from anonymizer.engine.detection.detection_workflow import _inject_detector_params
from anonymizer.engine.execution.context_contract import (
    _BackendArtifactClass,
    _ContextBackendCapability,
    _ContextLimits,
    _ContextOrdering,
    _ContextProfile,
    _ContextSchemaVersion,
    _RetentionPosture,
)
from anonymizer.engine.execution.invocation import _CompiledInvocation
from anonymizer.engine.execution.mention_admission import (
    _ValidationDecision,
    _ValidationDecisionKind,
)
from anonymizer.engine.execution.mention_resolution import _SubjectEvidence
from anonymizer.engine.execution.phase6_runtime import (
    _CandidateProposal,
    _Phase6AugmentationWork,
    _Phase6CandidateWork,
    _Phase6ResolverWork,
    _Phase6ValidationWork,
)
from anonymizer.engine.ndd.adapter import NddAdapter
from anonymizer.engine.ndd.model_loader import resolve_model_alias, resolve_model_aliases

T = TypeVar("T", bound=BaseModel)


class _PrivatePhase6BackendValue:
    def __repr__(self) -> str:
        return f"<private {type(self).__name__.strip('_').replace('_', ' ').lower()}>"

    def __reduce__(self) -> str | tuple[object, ...]:
        raise TypeError("private Phase 6 backend values are not serializable")


class _Phase6NddStageError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("private Phase 6 provider stage failed")

    def __repr__(self) -> str:
        return "<private Phase 6 provider stage error>"


class _AugmentedSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: int
    end: int
    source_slice: str
    detector_label: str


class _AugmentedSpans(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entities: list[_AugmentedSpan]


class _DecisionKind(str, Enum):
    KEEP = "keep"
    RECLASS = "reclass"
    DROP = "drop"


class _CandidateDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ordinal: int
    decision: _DecisionKind
    proposed_label: str | None = None


class _CandidateDecisions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decisions: list[_CandidateDecision]


class _Phase6NddBackend(_PrivatePhase6BackendValue):
    """Execute separately accounted detector, augmenter, and validator stages."""

    def __init__(self, adapter: NddAdapter, invocation: _CompiledInvocation) -> None:
        self._adapter = adapter
        self._invocation = invocation

    def context_capability(self) -> _ContextBackendCapability:
        return _ContextBackendCapability(
            _ContextProfile.TARGET_CONTEXT_V1,
            _ContextSchemaVersion.V1,
            _ContextLimits(0, 0, 0, 65_536),
            False,
            _ContextOrdering.DECLARED,
            (_BackendArtifactClass.CONTEXT_REQUEST,),
            _RetentionPosture.DISABLED,
        )

    def detect(self, work: _Phase6CandidateWork) -> tuple[_CandidateProposal, ...]:
        labels = list(self._invocation.entity_labels or DEFAULT_ENTITY_LABELS)
        model_configs = _inject_detector_params(
            model_configs=list(self._invocation.model_configs),
            selected_models=self._invocation.selected_models.detection,
            labels=labels,
            gliner_detection_threshold=self._invocation.gliner_detection_threshold,
        )
        result = self._adapter.run_workflow(
            pd.DataFrame({COL_TEXT: [work.target.text]}),
            model_configs=model_configs,
            columns=[
                LLMTextColumnConfig(
                    name=COL_RAW_DETECTED,
                    prompt=_jinja(COL_TEXT),
                    model_alias=resolve_model_alias("entity_detector", self._invocation.selected_models.detection),
                )
            ],
            workflow_name="phase6-detect",
        )
        raw = _single_stage_value(result, COL_RAW_DETECTED)
        return _decode_detector(raw)

    def augment(self, work: _Phase6AugmentationWork) -> tuple[_CandidateProposal, ...]:
        context = json.dumps([datum.text for datum in work.context], ensure_ascii=False, separators=(",", ":"))
        result = self._adapter.run_workflow(
            pd.DataFrame({COL_TEXT: [work.target.text], COL_PHASE6_CONTEXT: [context]}),
            model_configs=list(self._invocation.model_configs),
            columns=[
                LLMStructuredColumnConfig(
                    name=COL_PHASE6_AUGMENTED,
                    prompt="""Find additional privacy-sensitive spans in the target only.
Target: """
                    + _jinja(COL_TEXT)
                    + """
Declared context (evidence only): """
                    + _jinja(COL_PHASE6_CONTEXT)
                    + """
Return Python-character start/end offsets, the exact target source_slice, and detector_label.
Return an empty entities list when no additional exact target span is justified.""",
                    model_alias=resolve_model_alias(
                        "entity_augmenter",
                        self._invocation.selected_models.detection,
                    ),
                    output_format=_AugmentedSpans,
                )
            ],
            workflow_name="phase6-augment",
        )
        parsed = _coerce_model(_single_stage_value(result, COL_PHASE6_AUGMENTED), _AugmentedSpans)
        return tuple(
            _CandidateProposal(item.start, item.end, item.source_slice, item.detector_label) for item in parsed.entities
        )

    def validate(self, work: _Phase6ValidationWork) -> tuple[_ValidationDecision, ...]:
        if not work.candidates:
            return ()
        candidate_payload = json.dumps(
            [
                {
                    "ordinal": ordinal,
                    "start": candidate.start,
                    "end": candidate.end,
                    "source_slice": candidate.source_slice,
                    "detector_label": candidate.detector_label,
                    "provenance": candidate.provenance.value,
                }
                for ordinal, candidate in enumerate(work.candidates)
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        aliases = resolve_model_aliases("entity_validator", self._invocation.selected_models.detection)
        if not aliases:
            raise _Phase6NddStageError
        result = self._adapter.run_workflow(
            pd.DataFrame({COL_TEXT: [work.target.text], COL_PHASE6_CANDIDATES: [candidate_payload]}),
            model_configs=list(self._invocation.model_configs),
            columns=[
                LLMStructuredColumnConfig(
                    name=COL_PHASE6_VALIDATION,
                    prompt="""Validate every candidate against the target. Return exactly one decision per ordinal.
Target: """
                    + _jinja(COL_TEXT)
                    + """
Candidates: """
                    + _jinja(COL_PHASE6_CANDIDATES)
                    + """
Allowed decisions are keep, reclass, and drop. proposed_label is required only for reclass.""",
                    model_alias=aliases[0],
                    output_format=_CandidateDecisions,
                )
            ],
            workflow_name="phase6-validate",
        )
        parsed = _coerce_model(_single_stage_value(result, COL_PHASE6_VALIDATION), _CandidateDecisions)
        ordinals = tuple(item.ordinal for item in parsed.decisions)
        expected = tuple(range(len(work.candidates)))
        if len(set(ordinals)) != len(ordinals) or set(ordinals) != set(expected):
            raise _Phase6NddStageError
        by_ordinal = {item.ordinal: item for item in parsed.decisions}
        decisions: list[_ValidationDecision] = []
        for ordinal, candidate in enumerate(work.candidates):
            item = by_ordinal[ordinal]
            if item.decision is _DecisionKind.RECLASS:
                if not isinstance(item.proposed_label, str) or not item.proposed_label:
                    raise _Phase6NddStageError
                decisions.append(
                    _ValidationDecision(candidate.token, _ValidationDecisionKind.RECLASS, item.proposed_label)
                )
            else:
                if item.proposed_label is not None:
                    raise _Phase6NddStageError
                kind = (
                    _ValidationDecisionKind.KEEP
                    if item.decision is _DecisionKind.KEEP
                    else _ValidationDecisionKind.DROP
                )
                decisions.append(_ValidationDecision(candidate.token, kind))
        return tuple(decisions)

    def resolve(self, work: _Phase6ResolverWork) -> tuple[_SubjectEvidence, ...]:
        del work
        return ()

    def close_phase6(self) -> bool:
        return True


def _single_stage_value(result: object, column: str) -> object:
    dataframe = getattr(result, "dataframe", None)
    failures = getattr(result, "failed_records", None)
    if (
        not isinstance(dataframe, pd.DataFrame)
        or failures != []
        or len(dataframe) != 1
        or column not in dataframe.columns
    ):
        raise _Phase6NddStageError
    return dataframe.iloc[0][column]


def _decode_detector(raw: object) -> tuple[_CandidateProposal, ...]:
    try:
        payload = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(payload, dict) or set(payload) != {"entities"}:
            raise TypeError
        entities = payload["entities"]
        if not isinstance(entities, list):
            raise TypeError
        proposals: list[_CandidateProposal] = []
        for entity in entities:
            if not isinstance(entity, dict):
                raise TypeError
            allowed = {"text", "label", "start", "end", "score"}
            if not {"text", "label", "start", "end"}.issubset(entity) or set(entity) - allowed:
                raise TypeError
            proposals.append(
                _CandidateProposal(
                    entity["start"],
                    entity["end"],
                    entity["text"],
                    entity["label"],
                )
            )
        return tuple(proposals)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        raise _Phase6NddStageError from None


def _coerce_model(raw: object, model: type[T]) -> T:
    try:
        if isinstance(raw, model):
            return raw
        if isinstance(raw, str):
            raw = json.loads(raw)
        return model.model_validate(raw)
    except Exception as error:
        del error
        raise _Phase6NddStageError from None
