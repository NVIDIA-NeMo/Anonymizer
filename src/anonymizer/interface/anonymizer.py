# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import re
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

from data_designer.config.models import ModelProvider
from data_designer.config.utils.io_helpers import load_config_file
from data_designer.interface.data_designer import DataDesigner

from anonymizer.config.anonymizer_config import (
    AnonymizerConfig,
    AnonymizerInput,
    EvaluateConfig,
)
from anonymizer.config.replace_strategies import Substitute
from anonymizer.engine.constants import (
    COL_ANY_HIGH_LEAKED,
    COL_ATTRIBUTE_FIDELITY_INVALID_ENTITIES,
    COL_ATTRIBUTE_FIDELITY_VALID,
    COL_DETECTED_ENTITIES,
    COL_DETECTION_INVALID_ENTITIES,
    COL_DETECTION_VALID,
    COL_FINAL_ENTITIES,
    COL_LEAKAGE_MASS,
    COL_NEEDS_HUMAN_REVIEW,
    COL_RELATIONAL_CONSISTENCY_INVALID_RELATIONS,
    COL_RELATIONAL_CONSISTENCY_VALID,
    COL_REPLACED_TEXT,
    COL_REWRITTEN_TEXT,
    COL_TAGGED_TEXT,
    COL_TEXT,
    COL_TYPE_FIDELITY_INVALID_REPLACEMENTS,
    COL_TYPE_FIDELITY_VALID,
    COL_UTILITY_SCORE,
    COL_WEIGHTED_LEAKAGE_RATE,
    DEFAULT_ENTITY_LABELS,
)
from anonymizer.engine.detection.detection_workflow import EntityDetectionWorkflow
from anonymizer.engine.evaluation.detection_judge import DetectionJudgeWorkflow
from anonymizer.engine.evaluation.replace.attribute_fidelity_judge import AttributeFidelityJudgeWorkflow
from anonymizer.engine.evaluation.replace.relational_consistency_judge import RelationalConsistencyJudgeWorkflow
from anonymizer.engine.evaluation.replace.type_fidelity_judge import TypeFidelityJudgeWorkflow
from anonymizer.engine.io.reader import read_input
from anonymizer.engine.ndd.adapter import FailedRecord, NddAdapter
from anonymizer.engine.ndd.model_loader import (
    load_default_model_providers,
    parse_model_configs,
    validate_model_alias_references,
    validate_model_configs_reference_providers,
)
from anonymizer.engine.replace.llm_replace_workflow import LlmReplaceWorkflow
from anonymizer.engine.replace.replace_runner import ReplacementWorkflow
from anonymizer.engine.resolved_input import ResolvedInput
from anonymizer.engine.rewrite.rewrite_workflow import RewriteWorkflow
from anonymizer.interface.errors import InvalidConfigError
from anonymizer.interface.results import AnonymizerResult, PreviewResult
from anonymizer.logging import LOG_INDENT, configure_logging, reapply_log_levels
from anonymizer.measurement import (
    record_record_metrics,
    record_run_metadata,
    stage_timer,
)
from anonymizer.telemetry import (
    NOT_APPLICABLE,
    AnonymizerEvent,
    TaskEnum,
    TaskStatusEnum,
    TelemetryHandler,
    _telemetry_enabled,
    avg_tokens_per_record,
    classify_model_host,
    collect_model_hosts,
    sort_join_aliases,
)

if TYPE_CHECKING:
    import pandas as pd
    from data_designer.config.config_builder import DataDesignerConfigBuilder

logger = logging.getLogger("anonymizer")


def _initialize_logging() -> None:
    """Run one-time logging setup if the user hasn't already called configure_logging()."""
    from anonymizer import logging as _logging_mod

    if _logging_mod._configured:
        return
    configure_logging()


class Anonymizer:
    """Public facade for anonymization workflows."""

    def __init__(
        self,
        *,
        model_configs: str | Path | None = None,
        model_providers: list[ModelProvider] | str | Path | None = None,
        artifact_path: str | Path | None = None,
        data_designer: DataDesigner | None = None,
        detection_workflow: EntityDetectionWorkflow | None = None,
        replace_runner: ReplacementWorkflow | None = None,
        rewrite_runner: RewriteWorkflow | None = None,
    ) -> None:
        """Create an Anonymizer instance.

        Args:
            model_configs: Unified YAML (string or file path) defining the model
                pool and optional ``selected_models`` overrides. ``None`` uses
                bundled defaults. See ``default_model_configs/README.md``.
            model_providers: Provider definitions (list, YAML string, or file path).
                Each provider maps a name to an endpoint and API key. ``None`` uses
                bundled defaults from ``default_model_configs/providers.yaml``.
            artifact_path: Directory for intermediate artifacts. Defaults to
                ``.anonymizer-artifacts``.
            data_designer: Pre-configured DataDesigner instance (advanced usage).
            detection_workflow: Custom detection workflow (advanced/testing).
            replace_runner: Custom replacement workflow (advanced/testing).
            rewrite_runner: Custom rewrite workflow (advanced/testing).
        """
        _initialize_logging()
        # Tag DataDesigner telemetry events so they're filterable as anonymizer traffic in
        # the shared NeMo dashboards. `setdefault` so users (or upstream hosts) can override.
        os.environ.setdefault("NEMO_SESSION_PREFIX", "anonymizer-")
        os.environ.setdefault("NEMO_DEPLOYMENT_TYPE", "sdk")
        resolved_artifact_path = Path(artifact_path or ".anonymizer-artifacts")
        try:
            parsed = parse_model_configs(model_configs)
            self._model_configs = parsed.model_configs
            self._selected_models = parsed.selected_models
            self._resolved_providers = _resolve_model_providers(model_providers)
            # When the caller supplies a preconfigured DataDesigner, provider
            # registration is owned by that instance — our resolved providers
            # (bundled defaults when model_providers is None) are never passed to it
            # and only feed telemetry host classification. Validating the user's
            # model_configs against our defaults would wrongly reject configs that
            # reference providers registered on their own DataDesigner.
            if data_designer is None:
                validate_model_configs_reference_providers(self._model_configs, self._resolved_providers)
        except ValueError as exc:
            raise InvalidConfigError(str(exc)) from exc
        logger.info("🔧 Anonymizer initialized with %d model configs", len(self._model_configs))
        det = self._selected_models.detection
        logger.info(LOG_INDENT + "🔎 detector:  %s", det.entity_detector)
        logger.info(LOG_INDENT + "✅ validator: %s", ", ".join(det.entity_validator))
        logger.info(LOG_INDENT + "🧩 augmenter: %s", det.entity_augmenter)

        if data_designer is not None:
            self._data_designer = data_designer
        else:
            self._data_designer = DataDesigner(
                artifact_path=resolved_artifact_path,
                model_providers=self._resolved_providers,
            )
            reapply_log_levels()
        self._adapter = NddAdapter(data_designer=self._data_designer)
        self._detection_workflow = detection_workflow or EntityDetectionWorkflow(adapter=self._adapter)
        self._replace_runner = replace_runner or ReplacementWorkflow(
            llm_workflow=LlmReplaceWorkflow(adapter=self._adapter),
            detection_judge=DetectionJudgeWorkflow(adapter=self._adapter),
            type_fidelity_judge=TypeFidelityJudgeWorkflow(adapter=self._adapter),
            relational_consistency_judge=RelationalConsistencyJudgeWorkflow(adapter=self._adapter),
            attribute_fidelity_judge=AttributeFidelityJudgeWorkflow(adapter=self._adapter),
            adapter=self._adapter,
        )
        self._rewrite_runner = rewrite_runner or RewriteWorkflow(adapter=self._adapter)

    def run(
        self,
        *,
        config: AnonymizerConfig,
        data: AnonymizerInput,
    ) -> AnonymizerResult:
        """Run the full anonymization pipeline (detection + replacement).

        No LLM evaluation judges run here — call :meth:`evaluate` on the
        result's ``trace_dataframe`` when you want the LLM alignment scores.

        Args:
            config: Workflow behavior — replace strategy, entity labels, thresholds.
            data: Input source with file path, text column, and optional data summary.
        """
        self._validate_preflight_config(config)
        context = read_input(data)
        input_df = context.dataframe
        t_start = time.perf_counter()
        status = TaskStatusEnum.COMPLETED
        result: AnonymizerResult | None = None
        try:
            result = self._run_internal(config=config, data=data, context=context, preview_num_records=None)
            return result
        except KeyboardInterrupt:
            status = TaskStatusEnum.CANCELED
            raise
        except Exception:
            status = TaskStatusEnum.ERROR
            raise
        finally:
            self._maybe_emit_telemetry(
                task=TaskEnum.BATCH,
                status=status,
                config=config,
                data=data,
                input_df=input_df,
                result=result,
                duration_sec=time.perf_counter() - t_start,
            )

    def export_detection_config(
        self,
        *,
        config: AnonymizerConfig,
        data: AnonymizerInput,
        seed_path: str | Path,
    ) -> DataDesignerConfigBuilder:
        """Build (without running) the core detection workflow as a DataDesigner config.

        For handing the detection pipeline to an *external* at-scale executor (e.g. an
        on-SLURM DataDesigner orchestrator) instead of running it in-process. Reads
        ``data``, resolves the same detection model configs/selections as :meth:`run`,
        writes the seed dataset to ``seed_path``, and returns the
        ``DataDesignerConfigBuilder`` for the GLiNER + LLM-validate/augment workflow.
        The external runtime supplies the model providers (e.g. served vLLM endpoints);
        its per-record output carries the finalized entity columns to score.

        Args:
            config: Same workflow config as :meth:`run` (entity labels, thresholds).
            data: Input source (the records to detect over).
            seed_path: Destination parquet path for the seed dataset.
        """
        self._validate_preflight_config(config)
        context = read_input(data)
        return self._detection_workflow.build_detection_config(
            context.dataframe,
            seed_path=seed_path,
            model_configs=self._model_configs,
            selected_models=self._selected_models.detection,
            gliner_detection_threshold=config.detect.gliner_threshold,
            validation_max_entities_per_call=config.detect.validation_max_entities_per_call,
            validation_excerpt_window_chars=config.detect.validation_excerpt_window_chars,
            entity_labels=config.detect.entity_labels,
            data_summary=data.data_summary,
        )

    def export_detection_builder_for_seed(
        self,
        *,
        config: AnonymizerConfig,
        seed_path: str | Path,
        job_index: int = 0,
        num_jobs: int = 1,
        data_summary: str | None = None,
    ) -> DataDesignerConfigBuilder:
        """Build the detection workflow reading an EXISTING seed parquet (no write).

        Like :meth:`export_detection_config`, but for a distributed executor that builds
        the workflow *in-process on each worker* (so the custom-column callables stay
        live — they cannot survive JSON serialization) and received the seed dataset from
        the orchestrator. The seed is read from ``seed_path`` (not rewritten), and
        ``num_jobs > 1`` selects this worker's ordered partition (``job_index`` of
        ``num_jobs``). See ``anonymizer.distributed`` for the worker factory entrypoint.
        """
        self._validate_preflight_config(config)
        return self._detection_workflow.build_detection_builder_for_seed(
            seed_path=seed_path,
            model_configs=self._model_configs,
            selected_models=self._selected_models.detection,
            gliner_detection_threshold=config.detect.gliner_threshold,
            validation_max_entities_per_call=config.detect.validation_max_entities_per_call,
            validation_excerpt_window_chars=config.detect.validation_excerpt_window_chars,
            entity_labels=config.detect.entity_labels,
            data_summary=data_summary,
            job_index=job_index,
            num_jobs=num_jobs,
        )

    def preview(
        self,
        *,
        config: AnonymizerConfig,
        data: AnonymizerInput,
        num_records: int = 10,
    ) -> PreviewResult:
        """Run the pipeline on a subset of records for quick inspection.

        No LLM evaluation judges run here — call :meth:`evaluate` on the
        result's ``trace_dataframe`` when you want the LLM alignment scores.

        Args:
            config: Workflow behavior — replace strategy, entity labels, thresholds.
            data: Input source with file path, text column, and optional data summary.
            num_records: Maximum records to process (default 10).
        """
        self._validate_preflight_config(config)
        context = read_input(data, nrows=num_records)
        input_df = context.dataframe
        t_start = time.perf_counter()
        status = TaskStatusEnum.COMPLETED
        result: AnonymizerResult | None = None
        try:
            result = self._run_internal(config=config, data=data, context=context, preview_num_records=num_records)
            return PreviewResult(
                dataframe=result.dataframe,
                trace_dataframe=result.trace_dataframe,
                resolved_text_column=result.resolved_text_column,
                failed_records=result.failed_records,
                preview_num_records=num_records,
                replace_method=config.replace,
            )
        except KeyboardInterrupt:
            status = TaskStatusEnum.CANCELED
            raise
        except Exception:
            status = TaskStatusEnum.ERROR
            raise
        finally:
            self._maybe_emit_telemetry(
                task=TaskEnum.PREVIEW,
                status=status,
                config=config,
                data=data,
                input_df=input_df,
                result=result,
                duration_sec=time.perf_counter() - t_start,
            )

    def evaluate(
        self,
        output: AnonymizerResult | PreviewResult,
        *,
        config: EvaluateConfig | None = None,
    ) -> AnonymizerResult:
        """Run LLM-as-judge evaluation on a prior ``preview()`` / ``run()`` output.

        The anonymization strategy is read from ``output.replace_method`` (set
        when ``run()`` / ``preview()`` produced the result), so users don't
        restate it and can't mis-state it.

        Typical flow::

            preview = anonymizer.preview(config=cfg, data=src, num_records=15)
            evaluated = anonymizer.evaluate(preview)
            evaluated.display_record(0)

        Save/reload across sessions::

            import pickle

            with open("/tmp/preview.pkl", "wb") as f:
                pickle.dump(preview, f)
            # … later …
            with open("/tmp/preview.pkl", "rb") as f:
                loaded = pickle.load(f)
            evaluated = anonymizer.evaluate(loaded)

        Args:
            output: An :class:`AnonymizerResult` or :class:`PreviewResult` from
                a prior ``preview()`` / ``run()``. Carries the trace dataframe,
                the resolved text-column name, and the replace strategy.
            config: Optional :class:`EvaluateConfig` for evaluation-specific
                knobs (placeholder today; reserved for metric selection,
                per-judge model/prompt overrides, etc.).
        """
        _ = config  # placeholder; no knobs to read yet
        replace_method = getattr(output, "replace_method", None)
        if replace_method is None:
            raise ValueError(
                "Cannot evaluate this output — it has no associated replace strategy. "
                "Pass an AnonymizerResult / PreviewResult produced by run() / preview() "
                "on this branch (the strategy is recorded then). Hand-built or legacy "
                "results need their `replace_method` attribute set before calling evaluate()."
            )
        try:
            validate_model_alias_references(
                self._model_configs,
                self._selected_models,
                check_substitute=isinstance(replace_method, Substitute),
                check_rewrite=False,
                check_evaluate=True,
            )
        except ValueError as exc:
            raise InvalidConfigError(str(exc)) from exc
        text_column = output.resolved_text_column
        # trace_dataframe is in user-facing form (e.g., 'biography' instead of
        # '__nemo_anonymizer_text_input__'). The judge prompts reference the
        # internal names, so reverse the rename before the DD call and re-apply
        # it on the result.
        internal_df = _unrename_output_columns(output.trace_dataframe, resolved_text_column=text_column)
        replace_result = self._replace_runner.evaluate(
            internal_df,
            replace_method=replace_method,
            model_configs=self._model_configs,
            selected_models=self._selected_models.evaluate,
        )
        renamed_trace = _rename_output_columns(replace_result.dataframe, resolved_text_column=text_column)
        return AnonymizerResult(
            dataframe=_build_user_dataframe(renamed_trace, resolved_text_column=text_column),
            trace_dataframe=renamed_trace,
            resolved_text_column=text_column,
            failed_records=replace_result.failed_records,
            replace_method=replace_method,
        )

    def validate_config(self, config: AnonymizerConfig) -> None:
        """Validate that the active workflow config is compatible with model selections."""
        self._validate_preflight_config(config)

    def _run_internal(
        self,
        *,
        config: AnonymizerConfig,
        data: AnonymizerInput,
        context: ResolvedInput,
        preview_num_records: int | None,
    ) -> AnonymizerResult:
        input_df = context.dataframe
        mode = "replace" if config.replace is not None else "rewrite"
        strategy = type(config.replace).__name__ if config.replace is not None else "Rewrite"
        with stage_timer(
            "Anonymizer._run_internal",
            mode=mode,
            strategy=strategy,
            input_row_count=len(input_df),
            preview_num_records=preview_num_records,
        ) as measurement:
            record_run_metadata(
                config=config,
                data=data,
                mode=mode,
                strategy=strategy,
                input_row_count=len(input_df),
                preview_num_records=preview_num_records,
                model_configs=self._model_configs,
            )
            result = self._run_internal_impl(
                config=config,
                data=data,
                context=context,
                preview_num_records=preview_num_records,
            )
            measurement.update(
                output_row_count=len(result.trace_dataframe),
                failed_record_count=len(result.failed_records),
            )
            return result

    def _run_internal_impl(
        self,
        *,
        config: AnonymizerConfig,
        data: AnonymizerInput,
        context: ResolvedInput,
        preview_num_records: int | None,
    ) -> AnonymizerResult:
        input_df = context.dataframe
        num_records = len(input_df)
        if preview_num_records is not None and preview_num_records != num_records:
            effective_records = min(preview_num_records, num_records)
            if effective_records < preview_num_records:
                logger.info(
                    LOG_INDENT + "🔍 Running entity detection on capped %d records (requested %d, available %d)",
                    effective_records,
                    preview_num_records,
                    num_records,
                )
            else:
                logger.info(
                    LOG_INDENT + "🔍 Running entity detection on %d of %d records", effective_records, num_records
                )
            preview_num_records = effective_records
        else:
            logger.info("🔍 Running entity detection on %d records", num_records)
        if logger.isEnabledFor(logging.DEBUG):
            text_lengths = input_df[COL_TEXT].astype(str).str.len()
            logger.debug(
                "input text lengths: min=%d, max=%d, mean=%.0f chars (%d records)",
                text_lengths.min(),
                text_lengths.max(),
                text_lengths.mean(),
                num_records,
            )
            logger.debug(
                "detection config: threshold=%.2f, labels=%s",
                config.detect.gliner_threshold,
                config.detect.entity_labels
                or f"(default: {len(DEFAULT_ENTITY_LABELS)} labels; see anonymizer.DEFAULT_ENTITY_LABELS for list)",
            )
        else:
            logger.info(
                "detection labels in scope: %s",
                config.detect.entity_labels
                or f"(default: {len(DEFAULT_ENTITY_LABELS)} labels; see anonymizer.DEFAULT_ENTITY_LABELS for list)",
            )

        t0 = time.perf_counter()
        detection_result = self._detection_workflow.run(
            input_df,
            model_configs=self._model_configs,
            selected_models=self._selected_models.detection,
            gliner_detection_threshold=config.detect.gliner_threshold,
            validation_max_entities_per_call=config.detect.validation_max_entities_per_call,
            validation_excerpt_window_chars=config.detect.validation_excerpt_window_chars,
            entity_labels=config.detect.entity_labels,
            privacy_goal=config.rewrite.privacy_goal if config.rewrite else None,
            data_summary=data.data_summary,
            tag_latent_entities=config.rewrite is not None,
            compute_grouped_entities=config.replace is not None or config.rewrite is not None,
            preview_num_records=preview_num_records,
        )
        detection_elapsed = time.perf_counter() - t0
        entity_count = _count_entities(detection_result.dataframe)
        detection_failed = len(detection_result.failed_records)
        logger.info(
            LOG_INDENT + "📋 Detection complete — %d entities found across %d records (%d failed) [%.1fs]",
            entity_count,
            len(detection_result.dataframe),
            detection_failed,
            detection_elapsed,
        )
        if COL_DETECTED_ENTITIES in detection_result.dataframe.columns:
            label_counts = _count_labels(detection_result.dataframe[COL_DETECTED_ENTITIES])
            if label_counts:
                summary = ", ".join(f"{label}={count}" for label, count in label_counts.most_common())
                logger.info(LOG_INDENT + "labels: %s", summary)
        if config.replace is not None:
            strategy_name = type(config.replace).__name__
            logger.info("🔄 Running %s replacement", strategy_name)
            t0 = time.perf_counter()
            replace_result = self._replace_runner.run(
                detection_result.dataframe,
                replace_method=config.replace,
                model_configs=self._model_configs,
                selected_models=self._selected_models.replace,
                preview_num_records=preview_num_records,
            )
            replace_elapsed = time.perf_counter() - t0
            final_df = replace_result.dataframe
            post_detection_failures = replace_result.failed_records
            logger.info(
                LOG_INDENT + "📋 Replacement complete (%d failed) [%.1fs]",
                len(post_detection_failures),
                replace_elapsed,
            )
        elif config.rewrite is not None:
            logger.info("✏️ Running rewrite pipeline")
            t0 = time.perf_counter()
            rewrite_result = self._rewrite_runner.run(
                detection_result.dataframe,
                model_configs=self._model_configs,
                selected_models=self._selected_models.rewrite,
                replace_model_selection=self._selected_models.replace,
                privacy_goal=config.rewrite.privacy_goal,
                evaluation=config.rewrite.evaluation,
                data_summary=data.data_summary,
                preview_num_records=preview_num_records,
                strict_entity_protection=config.rewrite.strict_entity_protection,
            )
            rewrite_elapsed = time.perf_counter() - t0
            final_df = rewrite_result.dataframe
            post_detection_failures = rewrite_result.failed_records
            logger.info(
                LOG_INDENT + "📋 Rewrite complete (%d failed) [%.1fs]",
                len(post_detection_failures),
                rewrite_elapsed,
            )
        else:
            final_df = detection_result.dataframe
            post_detection_failures = []

        all_failures = [*detection_result.failed_records, *post_detection_failures]
        if all_failures:
            logger.warning("%d record(s) failed during pipeline processing.", len(all_failures))
            for f in all_failures:
                logger.debug("  %s (%s: %s)", f.record_id, f.step, f.reason)
        text_col = context.resolved_text_column
        renamed_trace = _rename_output_columns(final_df, resolved_text_column=text_col)
        logger.info("🎉 Pipeline complete — %d records processed, %d total failures", num_records, len(all_failures))
        record_record_metrics(
            final_df,
            mode="replace" if config.replace is not None else "rewrite",
            strategy=type(config.replace).__name__ if config.replace is not None else "Rewrite",
            text_column=COL_TEXT,
            validation_max_entities_per_call=config.detect.validation_max_entities_per_call,
        )
        return AnonymizerResult(
            dataframe=_build_user_dataframe(renamed_trace, resolved_text_column=text_col),
            trace_dataframe=renamed_trace,
            resolved_text_column=text_col,
            failed_records=all_failures,
            replace_method=config.replace,
        )

    def _validate_preflight_config(self, config: AnonymizerConfig) -> None:
        """Run semantic preflight checks shared by validate, run, and preview."""
        try:
            validate_model_alias_references(
                self._model_configs,
                self._selected_models,
                check_substitute=isinstance(config.replace, Substitute) or config.rewrite is not None,
                check_rewrite=config.rewrite is not None,
            )
        except ValueError as exc:
            raise InvalidConfigError(str(exc)) from exc

    # ------------------------------------------------------------------ telemetry

    def _maybe_emit_telemetry(
        self,
        *,
        task: TaskEnum,
        status: TaskStatusEnum,
        config: AnonymizerConfig,
        data: AnonymizerInput,
        input_df: pd.DataFrame,
        result: AnonymizerResult | None,
        duration_sec: float,
    ) -> None:
        """Build and fire-and-flush a single telemetry event.

        Telemetry is best-effort: any exception here is swallowed so that a
        broken telemetry path can never disrupt the anonymization pipeline.
        """
        try:
            if not getattr(config, "emit_telemetry", True):
                return
            # Short-circuit before _build_telemetry_event so we don't pay the
            # tiktoken cost on every record when telemetry is globally disabled
            # via NEMO_TELEMETRY_ENABLED=false.
            if not _telemetry_enabled():
                return
            event = self._build_telemetry_event(
                task=task,
                status=status,
                config=config,
                data=data,
                input_df=input_df,
                result=result,
                duration_sec=duration_sec,
            )
            from anonymizer import __version__ as _anonymizer_version

            with TelemetryHandler(
                source_client_version=_anonymizer_version,
                session_id=uuid.uuid4().hex,
            ) as handler:
                handler.enqueue(event)
        except Exception:  # noqa: BLE001 - best-effort
            logger.debug("Failed to emit telemetry event", exc_info=True)

    def _build_telemetry_event(
        self,
        *,
        task: TaskEnum,
        status: TaskStatusEnum,
        config: AnonymizerConfig,
        data: AnonymizerInput,
        input_df: pd.DataFrame,
        result: AnonymizerResult | None,
        duration_sec: float,
    ) -> AnonymizerEvent:
        """Construct an AnonymizerEvent from the current pipeline state."""
        total_records = int(len(input_df))
        failed = list(result.failed_records) if result is not None else []
        failure_count = len(failed)
        success_count = max(total_records - failure_count, 0)

        avg_tokens = -1
        if total_records > 0 and COL_TEXT in input_df.columns:
            avg_tokens = avg_tokens_per_record(input_df[COL_TEXT].astype(str))

        transformation_type = _transformation_type_string(config)
        rewrite = config.rewrite
        substitute = config.replace if isinstance(config.replace, Substitute) else None

        models = _collect_step_models(
            selected=self._selected_models,
            has_substitute=substitute is not None,
            has_rewrite=rewrite is not None,
        )
        failure_counts = _collect_failure_counts(failed)
        hosts = _resolve_model_hosts(self._resolved_providers)

        return AnonymizerEvent(
            task=task,
            task_status=status,
            job_duration_sec=duration_sec,
            num_input_records=total_records,
            num_success_records=success_count,
            num_failure_records=failure_count,
            avg_tokens_per_record=avg_tokens,
            transformation_type=transformation_type,
            custom_data_summary_provided=bool(data.data_summary),
            custom_privacy_goal_provided=_custom_privacy_goal_provided(rewrite),
            custom_substitute_instructions_provided=bool(substitute is not None and substitute.instructions),
            max_repair_iterations=(rewrite.max_repair_iterations if rewrite is not None else -1),
            strict_entity_protection=(rewrite.strict_entity_protection if rewrite is not None else False),
            repair_iterations_triggered=_repair_iterations_triggered(failed, rewrite is not None),
            entity_detector_model=models["entity_detector"],
            entity_validator_model=models["entity_validator"],
            entity_augmenter_model=models["entity_augmenter"],
            latent_detector_model=models["latent_detector"],
            replacement_generator_model=models["replacement_generator"],
            domain_classifier_model=models["domain_classifier"],
            disposition_analyzer_model=models["disposition_analyzer"],
            meaning_extractor_model=models["meaning_extractor"],
            qa_generator_model=models["qa_generator"],
            rewriter_model=models["rewriter"],
            evaluator_model=models["evaluator"],
            repairer_model=models["repairer"],
            judge_model=models["judge"],
            model_hosts=hosts,
            entity_detection_failure_count=failure_counts["entity_detection"],
            latent_detection_failure_count=failure_counts["latent_detection"],
            replace_map_generation_failure_count=failure_counts["replace_map_generation"],
            rewrite_pipeline_failure_count=failure_counts["rewrite_pipeline"],
            rewrite_evaluate_failure_count=failure_counts["rewrite_evaluate"],
            rewrite_repair_failure_count=failure_counts["rewrite_repair"],
            rewrite_final_judge_failure_count=failure_counts["rewrite_final_judge"],
            unknown_step_failure_count=failure_counts["unknown"],
        )


def _unwrap_entities(raw: object) -> list:
    if isinstance(raw, dict):
        return raw.get("entities", [])
    if isinstance(raw, list):
        return raw
    return getattr(raw, "entities", [])


def _entity_label(entity: object) -> str | None:
    if isinstance(entity, dict):
        return entity.get("label")
    return getattr(entity, "label", None)


def _count_labels_for_row(raw: object) -> Counter[str]:
    counts: Counter[str] = Counter()
    for entity in _unwrap_entities(raw):
        label = _entity_label(entity)
        if label:
            counts[label] += 1
    return counts


def _count_labels(entities_column: pd.Series) -> Counter[str]:
    counts: Counter[str] = Counter()
    for raw in entities_column:
        counts += _count_labels_for_row(raw)
    return counts


def _count_entities(df: pd.DataFrame) -> int:
    if COL_DETECTED_ENTITIES not in df.columns:
        return 0
    return int(df[COL_DETECTED_ENTITIES].apply(lambda raw: len(_unwrap_entities(raw))).sum())


def _resolve_model_providers(
    model_providers: list[ModelProvider] | str | Path | None,
) -> list[ModelProvider]:
    if model_providers is None:
        return load_default_model_providers()
    if isinstance(model_providers, list):
        if not model_providers:
            raise ValueError("model_providers must contain at least one provider.")
        return model_providers
    if isinstance(model_providers, str) and "\n" not in model_providers:
        candidate = Path(model_providers.strip()).expanduser()
        if candidate.suffix in (".yaml", ".yml"):
            if not candidate.is_file():
                raise FileNotFoundError(f"Providers config file not found: {candidate}")
            model_providers = candidate
    config_dict = load_config_file(model_providers)
    raw_providers = config_dict.get("providers")
    if not isinstance(raw_providers, list):
        raise ValueError("model_providers YAML must contain a top-level 'providers' list.")
    if not raw_providers:
        raise ValueError("model_providers must contain at least one provider.")
    return [ModelProvider.model_validate(provider) for provider in raw_providers]


def _rename_output_columns(df: pd.DataFrame, *, resolved_text_column: str) -> pd.DataFrame:
    """Rename internal column names to user-facing names."""
    rename_map: dict[str, str] = {}
    if COL_TEXT in df.columns:
        rename_map[COL_TEXT] = resolved_text_column
    if COL_REPLACED_TEXT in df.columns:
        rename_map[COL_REPLACED_TEXT] = f"{resolved_text_column}_replaced"
    if COL_TAGGED_TEXT in df.columns:
        rename_map[COL_TAGGED_TEXT] = f"{resolved_text_column}_with_spans"
    if COL_REWRITTEN_TEXT in df.columns:
        rename_map[COL_REWRITTEN_TEXT] = f"{resolved_text_column}_rewritten"
    if not rename_map:
        return df
    return df.rename(columns=rename_map)


def _unrename_output_columns(df: pd.DataFrame, *, resolved_text_column: str) -> pd.DataFrame:
    """Reverse of :func:`_rename_output_columns`.

    Converts user-facing column names (``biography``, ``biography_replaced``, …)
    back to the internal names (``__nemo_anonymizer_text_input__``, …) that the
    judges' prompt templates reference. No-op if the dataframe is already in
    internal form (``COL_TEXT`` already present).
    """
    if COL_TEXT in df.columns:
        return df
    rename_map: dict[str, str] = {}
    if resolved_text_column in df.columns:
        rename_map[resolved_text_column] = COL_TEXT
    if f"{resolved_text_column}_replaced" in df.columns:
        rename_map[f"{resolved_text_column}_replaced"] = COL_REPLACED_TEXT
    if f"{resolved_text_column}_with_spans" in df.columns:
        rename_map[f"{resolved_text_column}_with_spans"] = COL_TAGGED_TEXT
    if f"{resolved_text_column}_rewritten" in df.columns:
        rename_map[f"{resolved_text_column}_rewritten"] = COL_REWRITTEN_TEXT
    if not rename_map:
        return df
    return df.rename(columns=rename_map)


def _build_user_dataframe(trace_dataframe: pd.DataFrame, *, resolved_text_column: str) -> pd.DataFrame:
    """Filter trace dataframe to the public column set for the active mode.

    Replace:     {text_col}, {text_col}_replaced, {text_col}_with_spans, final_entities,
                 optional judge verdict columns when available
    Rewrite:     {text_col}, {text_col}_rewritten, utility_score, leakage_mass, weighted_leakage_rate,
                 any_high_leaked, needs_human_review
    Detect-only: {text_col}, {text_col}_with_spans, final_entities
    """
    t = trace_dataframe
    text_col = resolved_text_column

    if f"{text_col}_rewritten" in t.columns:
        allowed = {
            text_col,
            f"{text_col}_rewritten",
            COL_UTILITY_SCORE,
            COL_LEAKAGE_MASS,
            COL_WEIGHTED_LEAKAGE_RATE,
            COL_ANY_HIGH_LEAKED,
            COL_NEEDS_HUMAN_REVIEW,
        }
    elif f"{text_col}_replaced" in t.columns:
        allowed = {
            text_col,
            f"{text_col}_replaced",
            f"{text_col}_with_spans",
            COL_FINAL_ENTITIES,
            COL_DETECTION_VALID,
            COL_DETECTION_INVALID_ENTITIES,
            COL_TYPE_FIDELITY_VALID,
            COL_TYPE_FIDELITY_INVALID_REPLACEMENTS,
            COL_RELATIONAL_CONSISTENCY_VALID,
            COL_RELATIONAL_CONSISTENCY_INVALID_RELATIONS,
            COL_ATTRIBUTE_FIDELITY_VALID,
            COL_ATTRIBUTE_FIDELITY_INVALID_ENTITIES,
        }
    else:
        allowed = {
            text_col,
            f"{text_col}_with_spans",
            COL_FINAL_ENTITIES,
        }

    return t[[col for col in t.columns if col in allowed]].copy()


# ----------------------------------------------------------------- telemetry helpers


_REWRITE_REPAIR_RE = re.compile(r"^rewrite-repair-(\d+)$")
_REWRITE_EVALUATE_RE = re.compile(r"^rewrite-evaluate-(\d+)$")


def _transformation_type_string(config: AnonymizerConfig) -> str:
    """Map AnonymizerConfig to the schema's transformationType value.

    Schema accepts exactly one of: ``annotate``, ``redact``, ``hash``,
    ``substitute``, ``rewrite``. AnonymizerConfig's validator enforces exactly one
    of replace/rewrite, so one of these branches always fires.
    """
    if config.rewrite is not None:
        return "rewrite"
    # The four ReplaceMethodBase subclasses (Annotate, Redact, Hash, Substitute)
    # lowercase directly to their schema values.
    return type(config.replace).__name__.lower()


def _custom_privacy_goal_provided(rewrite: object | None) -> bool:
    """Detect whether the user supplied a non-default privacy_goal.

    ``Rewrite.populate_default_privacy_goal`` always populates a default if the
    user passed None, so we treat the default protect/preserve text as "not custom".
    """
    if rewrite is None or rewrite.privacy_goal is None:  # type: ignore[union-attr]
        return False
    from anonymizer.config.rewrite import DEFAULT_PRESERVE_TEXT, DEFAULT_PROTECT_TEXT

    goal = rewrite.privacy_goal  # type: ignore[union-attr]
    return goal.protect != DEFAULT_PROTECT_TEXT or goal.preserve != DEFAULT_PRESERVE_TEXT


def _collect_step_models(
    *,
    selected,  # ModelSelection
    has_substitute: bool,
    has_rewrite: bool,
) -> dict[str, str]:
    """Project the user's model selection into the schema's step-keyed shape."""
    det = selected.detection
    rewrite = selected.rewrite
    replace = selected.replace
    return {
        "entity_detector": det.entity_detector or NOT_APPLICABLE,
        "entity_validator": sort_join_aliases(det.entity_validator or []),
        "entity_augmenter": det.entity_augmenter or NOT_APPLICABLE,
        # latent_detector only runs in rewrite mode
        "latent_detector": (det.latent_detector or NOT_APPLICABLE) if has_rewrite else NOT_APPLICABLE,
        # replacement_generator only runs in Substitute mode
        "replacement_generator": replace.replacement_generator if has_substitute else NOT_APPLICABLE,
        # All rewrite-only roles
        "domain_classifier": rewrite.domain_classifier if has_rewrite else NOT_APPLICABLE,
        "disposition_analyzer": rewrite.disposition_analyzer if has_rewrite else NOT_APPLICABLE,
        "meaning_extractor": rewrite.meaning_extractor if has_rewrite else NOT_APPLICABLE,
        "qa_generator": rewrite.qa_generator if has_rewrite else NOT_APPLICABLE,
        "rewriter": rewrite.rewriter if has_rewrite else NOT_APPLICABLE,
        "evaluator": rewrite.evaluator if has_rewrite else NOT_APPLICABLE,
        "repairer": rewrite.repairer if has_rewrite else NOT_APPLICABLE,
        "judge": rewrite.judge if has_rewrite else NOT_APPLICABLE,
    }


def _step_to_field(step: str) -> str:
    """Map a FailedRecord.step (workflow_name) to a schema failure-count field key."""
    match step:
        case "entity-detection":
            return "entity_detection"
        case "latent-entity-detection":
            return "latent_detection"
        case "replace-map-generation":
            return "replace_map_generation"
        case "rewrite-pipeline":
            return "rewrite_pipeline"
        case "rewrite-final-judge":
            return "rewrite_final_judge"
        case _ if _REWRITE_EVALUATE_RE.match(step):
            return "rewrite_evaluate"
        case _ if _REWRITE_REPAIR_RE.match(step):
            return "rewrite_repair"
        case _:
            return "unknown"


def _collect_failure_counts(failed: list[FailedRecord]) -> dict[str, int]:
    """Aggregate FailedRecord.step values into per-workflow failure counts."""
    counts = {
        "entity_detection": 0,
        "latent_detection": 0,
        "replace_map_generation": 0,
        "rewrite_pipeline": 0,
        "rewrite_evaluate": 0,
        "rewrite_repair": 0,
        "rewrite_final_judge": 0,
        "unknown": 0,
    }
    for fr in failed:
        counts[_step_to_field(fr.step)] += 1
    return counts


def _repair_iterations_triggered(failed: list[FailedRecord], is_rewrite: bool) -> int:
    """Count distinct repair iterations observed in FailedRecord step names.

    Falls back to -1 when the run wasn't a rewrite. Returns 0 when rewrite ran
    but no failures surfaced from repair iterations — note that this undercounts
    repair iterations that completed without producing FailedRecord entries. A
    follow-up could plumb a richer signal up from the rewrite workflow.
    """
    if not is_rewrite:
        return -1
    iterations: set[int] = set()
    for fr in failed:
        m = _REWRITE_REPAIR_RE.match(fr.step)
        if m:
            iterations.add(int(m.group(1)))
    return len(iterations)


def _resolve_model_hosts(providers: list[ModelProvider]) -> list[str]:
    """Sorted, deduplicated list of provider host classifications."""
    return collect_model_hosts([classify_model_host(p) for p in providers])
