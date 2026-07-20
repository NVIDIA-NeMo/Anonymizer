# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Owns construction of the per-run `AnonymizerEvent` telemetry payload from a
finished pipeline run.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from data_designer.config.models import ModelProvider

from anonymizer.config.anonymizer_config import (
    AnonymizerConfig,
    AnonymizerInput,
)
from anonymizer.config.models import ModelSelection
from anonymizer.config.replace_strategies import Substitute
from anonymizer.engine.constants import (
    COL_TEXT,
)
from anonymizer.engine.ndd.adapter import FailedRecord
from anonymizer.interface.results import AnonymizerResult
from anonymizer.telemetry import (
    NOT_APPLICABLE,
    AnonymizerEvent,
    TaskEnum,
    TaskStatusEnum,
    avg_tokens_per_record,
    classify_model_host,
    collect_model_hosts,
    sort_join_aliases,
)

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["build_anonymizer_event"]


def build_anonymizer_event(
    *,
    selected_models: ModelSelection,
    resolved_providers: list[ModelProvider],
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
        selected=selected_models,
        has_substitute=substitute is not None,
        has_rewrite=rewrite is not None,
    )
    failure_counts = _collect_failure_counts(failed)
    hosts = _resolve_model_hosts(resolved_providers)

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
    selected: ModelSelection,
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
