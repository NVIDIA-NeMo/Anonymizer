# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The single private pandas runtime for normalized anonymizer invocations."""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol, TypeGuard

from anonymizer.engine.constants import (
    COL_CONTEXT_BINDING_ID,
    COL_CONTEXT_ORDINAL,
    COL_CONTEXT_OWNER_WORK_ID,
    COL_CONTEXT_TEXT,
    COL_DETECTED_ENTITIES,
    COL_TEXT,
    DEFAULT_ENTITY_LABELS,
)
from anonymizer.engine.execution.context_contract import (
    _BackendArtifactClass,
    _ContextBackendCapability,
    _ContextLimits,
    _ContextOrdering,
    _ContextProfile,
    _ContextSchemaVersion,
    _RetentionPosture,
)
from anonymizer.engine.execution.context_workframes import (
    _BackendArtifactId,
    _BackendClosureAttestation,
    _ContextBindingEvidence,
    _make_context_binding_evidence,
)
from anonymizer.engine.execution.invocation import _CompiledInvocation
from anonymizer.engine.ndd.adapter import FailedRecord, _FailedRowEvidence
from anonymizer.engine.private_row_verification import _InvocationRowVerifier, _TerminalOutcome

logger = logging.getLogger("anonymizer")


class _ListConvertible(Protocol):
    def tolist(self) -> object: ...


def _is_list_convertible(value: object) -> TypeGuard[_ListConvertible]:
    return callable(getattr(value, "tolist", None))


def _entity_counts(dataframe: pd.DataFrame) -> Counter[str]:
    counts: Counter[str] = Counter()
    for raw in dataframe.get(COL_DETECTED_ENTITIES, []):
        if isinstance(raw, dict):
            entities = raw.get("entities", [])
        elif isinstance(raw, list):
            entities = raw
        else:
            entities = getattr(raw, "entities", [])
        if _is_list_convertible(entities):
            entities = entities.tolist()
        if not isinstance(entities, list):
            continue
        for entity in entities:
            label = entity.get("label") if isinstance(entity, dict) else getattr(entity, "label", None)
            if isinstance(label, str):
                counts[label] += 1
    return counts


if TYPE_CHECKING:
    import pandas as pd

    from anonymizer.engine.detection.detection_workflow import EntityDetectionWorkflow
    from anonymizer.engine.replace.replace_runner import ReplacementWorkflow
    from anonymizer.engine.rewrite.combined_rewrite_workflow import CombinedRewriteWorkflow
    from anonymizer.engine.rewrite.rewrite_workflow import RewriteWorkflow


@dataclass(frozen=True, repr=False)
class _PandasExecutionResult:
    dataframe: pd.DataFrame
    failed_records: list[FailedRecord]
    terminal_outcomes: tuple[tuple[str, _TerminalOutcome], ...] = ()
    result_row_tokens: tuple[str, ...] = ()
    failed_row_evidence: tuple[_FailedRowEvidence, ...] = ()
    trusted_stop_tokens: tuple[str, ...] = ()
    context_binding_evidence: tuple[_ContextBindingEvidence, ...] = ()
    closure_attestations: tuple[_BackendClosureAttestation, ...] = ()

    def __repr__(self) -> str:
        return "<private pandas execution result>"

    def __reduce__(self) -> str | tuple[object, ...]:
        raise TypeError("private pandas execution results are not serializable")


class _PandasRuntime:
    """Coordinate existing workflows over one normalized pandas dataframe."""

    def __init__(
        self,
        *,
        detection_workflow: EntityDetectionWorkflow,
        replace_runner: ReplacementWorkflow,
        rewrite_runner: RewriteWorkflow,
        combined_rewrite_runner: CombinedRewriteWorkflow,
    ) -> None:
        self._detection_workflow = detection_workflow
        self._replace_runner = replace_runner
        self._rewrite_runner = rewrite_runner
        self._combined_rewrite_runner = combined_rewrite_runner

    def context_capability(self) -> _ContextBackendCapability:
        """Declare the bounded framing profile supported by this private runtime."""
        return _ContextBackendCapability(
            profile=_ContextProfile.TARGET_CONTEXT_V1,
            schema_version=_ContextSchemaVersion.V1,
            limits=_ContextLimits(
                max_context_members_per_target=128,
                max_context_bytes_per_target=1_048_576,
                max_total_context_references=16_384,
                max_expanded_frame_bytes=2_097_152,
            ),
            allow_target_as_context=True,
            ordering=_ContextOrdering.DECLARED,
            artifact_classes=(_BackendArtifactClass.CONTEXT_REQUEST,),
            retention=_RetentionPosture.DISABLED,
        )

    def run_context(
        self,
        dataframe: pd.DataFrame,
        *,
        context_dataframe: pd.DataFrame,
        artifact_id: _BackendArtifactId,
        invocation: _CompiledInvocation,
        data_summary: str | None,
        preview_num_records: int | None,
        verifier: _InvocationRowVerifier,
    ) -> _PandasExecutionResult:
        """Execute targets unchanged and attest consumption of the separate context frame.

        Phase 5 qualifies framing only. Context is therefore reconciled as typed
        input evidence but is not added to a prompt or used for entity decisions.
        """
        required = {COL_CONTEXT_BINDING_ID, COL_CONTEXT_OWNER_WORK_ID, COL_CONTEXT_ORDINAL}
        if set(context_dataframe.columns) != {*required, COL_CONTEXT_TEXT}:
            raise TypeError("private context frame is malformed")
        evidence = tuple(
            _make_context_binding_evidence(
                row[COL_CONTEXT_BINDING_ID],
                row[COL_CONTEXT_OWNER_WORK_ID],
                row[COL_CONTEXT_ORDINAL],
                row[COL_CONTEXT_TEXT],
            )
            for _index, row in context_dataframe.iterrows()
        )
        result = self.run(
            dataframe,
            invocation=invocation,
            data_summary=data_summary,
            preview_num_records=preview_num_records,
            verifier=verifier,
        )
        return replace(
            result,
            context_binding_evidence=evidence,
            closure_attestations=(
                _BackendClosureAttestation(artifact_id, _BackendArtifactClass.CONTEXT_REQUEST, True),
            ),
        )

    def run(
        self,
        dataframe: pd.DataFrame,
        *,
        invocation: _CompiledInvocation,
        data_summary: str | None,
        preview_num_records: int | None,
        verifier: _InvocationRowVerifier,
    ) -> _PandasExecutionResult:
        """Run detection then the compiled replacement or rewrite workflow."""
        num_records = len(dataframe)
        if preview_num_records is not None and preview_num_records != num_records:
            effective_records = min(preview_num_records, num_records)
            if effective_records < preview_num_records:
                logger.info(
                    "  |-- 🔍 Running entity detection on capped %d records (requested %d, available %d)",
                    effective_records,
                    preview_num_records,
                    num_records,
                )
            else:
                logger.info("  |-- 🔍 Running entity detection on %d of %d records", effective_records, num_records)
            preview_num_records = effective_records
        else:
            logger.info("🔍 Running entity detection on %d records", num_records)
        if logger.isEnabledFor(logging.DEBUG):
            text_lengths = dataframe[COL_TEXT].astype(str).str.len()
            logger.debug(
                "input text lengths: min=%d, max=%d, mean=%.0f chars (%d records)",
                text_lengths.min(),
                text_lengths.max(),
                text_lengths.mean(),
                num_records,
            )
            logger.debug(
                "detection config: threshold=%.2f, labels=%s",
                invocation.gliner_detection_threshold,
                invocation.entity_labels
                or f"(default: {len(DEFAULT_ENTITY_LABELS)} labels; see anonymizer.DEFAULT_ENTITY_LABELS for list)",
            )
        else:
            logger.info(
                "detection labels in scope: %s",
                invocation.entity_labels
                or f"(default: {len(DEFAULT_ENTITY_LABELS)} labels; see anonymizer.DEFAULT_ENTITY_LABELS for list)",
            )
        started = time.perf_counter()
        detection_result = self._detection_workflow.run(
            dataframe,
            model_configs=list(invocation.model_configs),
            selected_models=invocation.selected_models.detection,
            gliner_detection_threshold=invocation.gliner_detection_threshold,
            validation_max_entities_per_call=invocation.validation_max_entities_per_call,
            validation_excerpt_window_chars=invocation.validation_excerpt_window_chars,
            entity_labels=list(invocation.entity_labels) if invocation.entity_labels is not None else None,
            privacy_goal=invocation.rewrite.privacy_goal if invocation.rewrite is not None else None,
            data_summary=data_summary,
            tag_latent_entities=invocation.rewrite is not None,
            compute_grouped_entities=invocation.replace_method is not None or invocation.rewrite is not None,
            preview_num_records=preview_num_records,
        )
        logger.info(
            "  |-- 📋 Detection complete — %d entities found across %d records (%d failed) [%.1fs]",
            sum(_entity_counts(detection_result.dataframe).values()),
            len(detection_result.dataframe),
            len(detection_result.failed_records),
            time.perf_counter() - started,
        )
        label_counts = _entity_counts(detection_result.dataframe)
        if label_counts:
            logger.info(
                "  |-- labels: %s", ", ".join(f"{label}={count}" for label, count in label_counts.most_common())
            )
        detected = verifier.bind_complete_stage_output(detection_result.dataframe)
        verifier.freeze_accepted_detections(detected)
        if invocation.replace_method is not None:
            logger.info("🔄 Running %s replacement", type(invocation.replace_method).__name__)
            started = time.perf_counter()
            result = self._replace_runner.run(
                detected,
                replace_method=invocation.replace_method,
                model_configs=list(invocation.model_configs),
                selected_models=invocation.selected_models.replace,
                preview_num_records=preview_num_records,
            )
            logger.info(
                "  |-- 📋 Replacement complete (%d failed) [%.1fs]",
                len(result.failed_records),
                time.perf_counter() - started,
            )
        elif invocation.rewrite is not None:
            runner = self._combined_rewrite_runner if invocation.rewrite.use_combined_graph else self._rewrite_runner
            logger.info("✏️ Running rewrite pipeline")
            started = time.perf_counter()
            result = runner.run(
                detected,
                model_configs=list(invocation.model_configs),
                selected_models=invocation.selected_models.rewrite,
                replace_model_selection=invocation.selected_models.replace,
                privacy_goal=invocation.rewrite.privacy_goal,
                evaluation=invocation.rewrite.evaluation,
                data_summary=data_summary,
                preview_num_records=preview_num_records,
                strict_entity_protection=invocation.rewrite.strict_entity_protection,
            )
            logger.info(
                "  |-- 📋 Rewrite complete (%d failed) [%.1fs]",
                len(result.failed_records),
                time.perf_counter() - started,
            )
        else:
            final = verifier.finish(verifier.bind_complete_stage_output(detected))
            if detection_result.failed_records:
                logger.warning("%d record(s) failed during pipeline processing.", len(detection_result.failed_records))
            logger.info(
                "🎉 Pipeline complete — %d records processed, %d total failures",
                num_records,
                len(detection_result.failed_records),
            )
            return _PandasExecutionResult(
                dataframe=final,
                failed_records=detection_result.failed_records,
                terminal_outcomes=verifier.take_terminal_outcomes(),
                result_row_tokens=verifier.take_result_order(),
                failed_row_evidence=detection_result.failed_row_evidence,
            )
        failed_records = [*detection_result.failed_records, *result.failed_records]
        if failed_records:
            logger.warning("%d record(s) failed during pipeline processing.", len(failed_records))
        final = verifier.finish(verifier.bind_complete_stage_output(result.dataframe))
        logger.info(
            "🎉 Pipeline complete — %d records processed, %d total failures",
            num_records,
            len(detection_result.failed_records) + len(result.failed_records),
        )
        return _PandasExecutionResult(
            dataframe=final,
            failed_records=failed_records,
            terminal_outcomes=verifier.take_terminal_outcomes(),
            result_row_tokens=verifier.take_result_order(),
            failed_row_evidence=(
                *detection_result.failed_row_evidence,
                *result.failed_row_evidence,
            ),
        )
