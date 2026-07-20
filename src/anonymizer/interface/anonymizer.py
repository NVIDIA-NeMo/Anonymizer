# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Owns the `Anonymizer` facade: wiring config to engine workflows and returning results."""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

from data_designer.config.models import ModelProvider
from data_designer.config.run_config import RunConfig
from data_designer.config.utils.io_helpers import load_config_file
from data_designer.interface.data_designer import DataDesigner

from anonymizer.config.anonymizer_config import (
    AnonymizerConfig,
    AnonymizerInput,
    EvaluateConfig,
)
from anonymizer.config.replace_strategies import Substitute
from anonymizer.engine.constants import (
    COL_DETECTED_ENTITIES,
    COL_TEXT,
    DEFAULT_ENTITY_LABELS,
)
from anonymizer.engine.detection.detection_workflow import EntityDetectionWorkflow
from anonymizer.engine.evaluation.detection_judge import DetectionJudgeWorkflow
from anonymizer.engine.evaluation.replace.attribute_fidelity_judge import AttributeFidelityJudgeWorkflow
from anonymizer.engine.evaluation.replace.relational_consistency_judge import RelationalConsistencyJudgeWorkflow
from anonymizer.engine.evaluation.replace.type_fidelity_judge import TypeFidelityJudgeWorkflow
from anonymizer.engine.io.reader import read_input
from anonymizer.engine.ndd.adapter import NddAdapter
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
from anonymizer.interface.output_columns import (
    build_user_dataframe,
    rename_output_columns,
    unrename_output_columns,
)
from anonymizer.interface.results import AnonymizerResult, PreviewResult
from anonymizer.interface.run_telemetry import build_anonymizer_event
from anonymizer.logging import LOG_INDENT, configure_logging, reapply_log_levels
from anonymizer.measurement import (
    record_record_metrics,
    record_run_metadata,
    stage_timer,
)
from anonymizer.telemetry import (
    TaskEnum,
    TaskStatusEnum,
    TelemetryHandler,
    _telemetry_enabled,
)

if TYPE_CHECKING:
    import pandas as pd
    from data_designer.config.config_builder import DataDesignerConfigBuilder

__all__ = ["Anonymizer"]


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
        data_designer_run_config: RunConfig | None = None,
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
            data_designer_run_config: Optional DataDesigner run configuration
                applied to the Anonymizer-managed or caller-supplied DataDesigner
                instance. Use this for DataDesigner execution knobs such as
                ``buffer_size`` and ``max_in_flight_tasks``.
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
        if data_designer_run_config is not None:
            self._data_designer.set_run_config(data_designer_run_config)
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
                rewrite_config=config.rewrite.privacy_goal if config.rewrite is not None else None,
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

        The anonymization strategy is read from the result (set when ``run()`` /
        ``preview()`` produced it), so users don't restate it and can't mis-state it.

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
                the resolved text-column name, and the anonymization config.
            config: Optional :class:`EvaluateConfig` for evaluation-specific
                knobs (placeholder today; reserved for metric selection,
                per-judge model/prompt overrides, etc.).
        """
        _ = config  # placeholder; no knobs to read yet
        rewrite_config = getattr(output, "rewrite_config", None)
        replace_method = getattr(output, "replace_method", None)

        if rewrite_config is not None:
            try:
                validate_model_alias_references(
                    self._model_configs,
                    self._selected_models,
                    check_rewrite=False,
                    check_evaluate=True,
                    check_rewrite_judge=True,
                )
            except ValueError as exc:
                raise InvalidConfigError(str(exc)) from exc
            text_column = output.resolved_text_column
            internal_df = unrename_output_columns(output.trace_dataframe, resolved_text_column=text_column)
            rewrite_result = self._rewrite_runner.evaluate(
                internal_df,
                model_configs=self._model_configs,
                selected_models=self._selected_models.evaluate,
                privacy_goal=rewrite_config,
            )
            renamed_trace = rename_output_columns(rewrite_result.dataframe, resolved_text_column=text_column)
            return AnonymizerResult(
                dataframe=build_user_dataframe(renamed_trace, resolved_text_column=text_column),
                trace_dataframe=renamed_trace,
                resolved_text_column=text_column,
                failed_records=rewrite_result.failed_records,
                rewrite_config=rewrite_config,
            )

        if replace_method is None:
            raise ValueError(
                "Cannot evaluate this output — it has no associated anonymization config. "
                "Pass an AnonymizerResult / PreviewResult produced by run() / preview(). "
                "Hand-built or legacy results need their `replace_method` or `rewrite_config` "
                "attribute set before calling evaluate()."
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
        internal_df = unrename_output_columns(output.trace_dataframe, resolved_text_column=text_column)
        replace_result = self._replace_runner.evaluate(
            internal_df,
            replace_method=replace_method,
            model_configs=self._model_configs,
            selected_models=self._selected_models.evaluate,
        )
        renamed_trace = rename_output_columns(replace_result.dataframe, resolved_text_column=text_column)
        return AnonymizerResult(
            dataframe=build_user_dataframe(renamed_trace, resolved_text_column=text_column),
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
        renamed_trace = rename_output_columns(final_df, resolved_text_column=text_col)
        logger.info("🎉 Pipeline complete — %d records processed, %d total failures", num_records, len(all_failures))
        record_record_metrics(
            final_df,
            mode="replace" if config.replace is not None else "rewrite",
            strategy=type(config.replace).__name__ if config.replace is not None else "Rewrite",
            text_column=COL_TEXT,
            validation_max_entities_per_call=config.detect.validation_max_entities_per_call,
        )
        return AnonymizerResult(
            dataframe=build_user_dataframe(renamed_trace, resolved_text_column=text_col),
            trace_dataframe=renamed_trace,
            resolved_text_column=text_col,
            failed_records=all_failures,
            replace_method=config.replace,
            rewrite_config=config.rewrite.privacy_goal if config.rewrite is not None else None,
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
            # Short-circuit before build_anonymizer_event so we don't pay the
            # tiktoken cost on every record when telemetry is globally disabled
            # via NEMO_TELEMETRY_ENABLED=false.
            if not _telemetry_enabled():
                return
            event = build_anonymizer_event(
                selected_models=self._selected_models,
                resolved_providers=self._resolved_providers,
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
