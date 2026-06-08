#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run benchmark-only DD-free staged entity detection probes.

Usage:
    uv run python tools/measurement/staged_detection_probe.py docs/data/NVIDIA_synthetic_biographies.csv \
      --text-column biography --labels age,city,first_name,last_name,occupation \
      --output /tmp/staged-probe --overwrite
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Protocol

import cyclopts
import httpx
import pandas as pd
from analyze_detection_artifacts import DetectionArtifactRow, build_detection_artifact_row_from_entities
from dd_parser_compat import _load_embedded_json
from direct_detection_probe import (
    CaseStatus,
    DirectCompletion,
    DirectDetectionClient,
    DirectDetectionRequest,
    DirectGenerationRequest,
    HttpxDirectDetectionClient,
    LogFormat,
    PromptMode,
    SignatureComparison,
    build_direct_prompt,
    compare_signature_sets,
    parse_labels,
)
from pydantic import BaseModel, Field, ValidationError, model_validator

from anonymizer.engine.constants import (
    COL_AUGMENTED_ENTITIES,
    COL_DETECTED_ENTITIES,
    COL_RAW_DETECTED,
    COL_SEED_ENTITIES,
    COL_SEED_ENTITIES_JSON,
    COL_SEED_VALIDATION_CANDIDATES,
    COL_TAG_NOTATION,
    COL_TAGGED_TEXT,
    COL_TEXT,
    COL_VALIDATED_ENTITIES,
    COL_VALIDATION_CANDIDATES,
    COL_VALIDATION_DECISIONS,
)
from anonymizer.engine.detection.chunked_validation import (
    build_chunk_excerpt,
    chunk_candidates,
    merge_chunk_decisions,
    order_candidates_by_position,
)
from anonymizer.engine.detection.custom_columns import (
    apply_validation_and_finalize,
    apply_validation_to_seed_entities,
    enrich_validation_decisions,
    merge_and_build_candidates,
    prepare_validation_inputs,
)
from anonymizer.engine.detection.postprocess import (
    VALIDATION_CONTEXT_WINDOW,
    EntitySpan,
    TagNotation,
    apply_augmented_entities,
    build_tagged_text,
    build_validation_candidates,
    get_tag_notation,
    parse_raw_entities,
    resolve_overlaps,
)
from anonymizer.engine.detection.rules import (
    STRUCTURED_RULE_FAST_LANE_LABELS,
    detect_high_confidence_entities,
)
from anonymizer.engine.schemas import (
    EntitiesSchema,
    EntitySchema,
    RawValidationDecisionsSchema,
    ValidatedDecisionSchema,
    ValidatedDecisionsSchema,
    ValidationCandidatesSchema,
)

app = cyclopts.App(help=__doc__)
logger = logging.getLogger("measurement.staged_detection_probe")
_log_format = LogFormat.plain
_DATE_OF_BIRTH_CONTEXT_RE = re.compile(r"\b(born|birth|date of birth|dob)\b", re.IGNORECASE)


class SeedSource(StrEnum):
    direct_llm = "direct_llm"
    gliner = "gliner"
    rules = "rules"
    rules_trusted = "rules_trusted"
    rules_plus_direct_llm = "rules_plus_direct_llm"
    rules_router = "rules_router"


class ValidationPromptMode(StrEnum):
    full_text = "full_text"
    chunked_excerpt = "chunked_excerpt"


class GlinerDetectionRequest(BaseModel):
    endpoint: str
    model: str
    text: str
    labels: list[str] = Field(min_length=1)
    threshold: float = 0.3
    max_tokens: int = Field(default=4096, gt=0)
    timeout_sec: float = Field(default=120.0, gt=0)
    api_key_env: str = "NVIDIA_API_KEY"


class GlinerSeedClient(Protocol):
    def detect(self, request: GlinerDetectionRequest) -> DirectCompletion: ...


class StagedExecutionConfig(BaseModel):
    endpoint: str = "http://gpu-dev-pod-serve-svc:8000/v1"
    model: str = "nvidia/nemotron-3-super"
    seed_source: SeedSource = SeedSource.direct_llm
    gliner_endpoint: str = "https://integrate.api.nvidia.com/v1"
    gliner_model: str = "nvidia/gliner-pii"
    gliner_api_key_env: str = "NVIDIA_API_KEY"
    gliner_threshold: float = 0.3
    max_tokens: int = Field(default=4096, gt=0)
    timeout_sec: float = Field(default=180.0, gt=0)
    skip_augmentation: bool = False
    skip_augmentation_when_rule_covered: bool = False
    validation_prompt_mode: ValidationPromptMode = ValidationPromptMode.full_text
    validation_max_entities_per_call: int = Field(default=10, gt=0)
    validation_excerpt_window_chars: int = Field(default=160, gt=0)


class HttpxGlinerSeedClient:
    def detect(self, request: GlinerDetectionRequest) -> DirectCompletion:
        api_key = os.environ.get(request.api_key_env)
        if not api_key:
            raise ValueError(f"{request.api_key_env} is not set")
        response = httpx.post(
            f"{request.endpoint.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=_gliner_payload(request),
            timeout=request.timeout_sec,
        )
        response.raise_for_status()
        data = response.json()
        return DirectCompletion(
            content=_completion_content(data),
            elapsed_sec=float(response.elapsed.total_seconds()),
            usage=data.get("usage") or {},
        )


def _gliner_payload(request: GlinerDetectionRequest) -> dict[str, Any]:
    return {
        "model": request.model,
        "messages": [{"role": "user", "content": request.text}],
        "temperature": 0,
        "max_tokens": request.max_tokens,
        "labels": request.labels,
        "threshold": request.threshold,
        "chunk_length": 384,
        "overlap": 128,
        "flat_ner": False,
    }


def _completion_content(data: dict[str, Any]) -> str:
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") if isinstance(choice, dict) else {}
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return str(choice.get("text") or "") if isinstance(choice, dict) else ""


class StagedDetectionRequest(BaseModel):
    case_id: str
    text: str
    labels: list[str] = Field(min_length=1)
    row_index: int = 0
    data_summary: str | None = None

    @model_validator(mode="after")
    def normalize_labels(self) -> StagedDetectionRequest:
        self.labels = list(dict.fromkeys(label.strip() for label in self.labels if label.strip()))
        if not self.labels:
            raise ValueError("labels must contain at least one non-empty label")
        return self


class PhaseUsage(BaseModel):
    seed: dict[str, Any] = Field(default_factory=dict)
    validation: dict[str, Any] = Field(default_factory=dict)
    augmentation: dict[str, Any] = Field(default_factory=dict)


class PhaseModelWork(BaseModel):
    seed: bool = False
    validation: bool = False
    augmentation: bool = False


class PhaseSkipReasons(BaseModel):
    seed: str | None = None
    validation: str | None = None
    augmentation: str | None = None


class PhaseModelRequests(BaseModel):
    seed: int = 0
    validation: int = 0
    augmentation: int = 0


class StagedDetectionCase(BaseModel):
    case_id: str
    row_index: int
    seed_source: SeedSource = SeedSource.direct_llm
    status: CaseStatus
    elapsed_sec: float | None = None
    model_elapsed_sec: float | None = None
    phase_usage: PhaseUsage = Field(default_factory=PhaseUsage)
    phase_model_work: PhaseModelWork = Field(default_factory=PhaseModelWork)
    phase_skip_reasons: PhaseSkipReasons = Field(default_factory=PhaseSkipReasons)
    phase_model_requests: PhaseModelRequests = Field(default_factory=PhaseModelRequests)
    total_usage: dict[str, int] = Field(default_factory=dict)
    model_phase_count: int = 0
    model_request_count: int = 0
    rule_covered_label_set: bool = False
    seed_suggestion_count: int = 0
    seed_entity_count: int = 0
    validation_candidate_count: int = 0
    validation_decision_count: int = 0
    augmented_suggestion_count: int = 0
    final_entity_count: int = 0
    final_entity_signature_count: int = 0
    final_entity_signature_hashes: list[str] = Field(default_factory=list)
    final_label_counts: dict[str, int] = Field(default_factory=dict)
    comparison: SignatureComparison | None = None
    artifact: DetectionArtifactRow | None = None
    error: str | None = None


class StagedDetectionRun(BaseModel):
    input_path: str
    text_column: str
    endpoint: str
    model: str
    labels: list[str]
    rows: list[StagedDetectionCase] = Field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for row in self.rows if row.status == CaseStatus.error)


@dataclass(frozen=True)
class StagedDetectionExecution:
    case: StagedDetectionCase
    row: dict[str, Any]


def configure_logging(log_format: LogFormat) -> None:
    global _log_format

    _log_format = log_format
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def log_bad_input(error: str) -> None:
    if _log_format == LogFormat.json:
        sys.stderr.write(json.dumps({"level": "error", "event": "bad_input", "error": error}) + "\n")
        return
    logger.error("bad_input error=%s", error)


def run_staged_detection_case(
    request: StagedDetectionRequest,
    *,
    client: DirectDetectionClient,
    seed_client: GlinerSeedClient | None = None,
    seed_source: SeedSource = SeedSource.direct_llm,
    endpoint: str = "http://gpu-dev-pod-serve-svc:8000/v1",
    model: str = "nvidia/nemotron-3-super",
    gliner_endpoint: str = "https://integrate.api.nvidia.com/v1",
    gliner_model: str = "nvidia/gliner-pii",
    gliner_api_key_env: str = "NVIDIA_API_KEY",
    gliner_threshold: float = 0.3,
    max_tokens: int = 4096,
    timeout_sec: float = 180.0,
    skip_augmentation: bool = False,
    skip_augmentation_when_rule_covered: bool = False,
    validation_prompt_mode: ValidationPromptMode = ValidationPromptMode.full_text,
    validation_max_entities_per_call: int = 10,
    validation_excerpt_window_chars: int = 160,
) -> StagedDetectionCase:
    return execute_staged_detection_case(
        request,
        client=client,
        seed_client=seed_client,
        seed_source=seed_source,
        endpoint=endpoint,
        model=model,
        gliner_endpoint=gliner_endpoint,
        gliner_model=gliner_model,
        gliner_api_key_env=gliner_api_key_env,
        gliner_threshold=gliner_threshold,
        max_tokens=max_tokens,
        timeout_sec=timeout_sec,
        skip_augmentation=skip_augmentation,
        skip_augmentation_when_rule_covered=skip_augmentation_when_rule_covered,
        validation_prompt_mode=validation_prompt_mode,
        validation_max_entities_per_call=validation_max_entities_per_call,
        validation_excerpt_window_chars=validation_excerpt_window_chars,
    ).case


def execute_staged_detection_case(
    request: StagedDetectionRequest,
    *,
    client: DirectDetectionClient,
    seed_client: GlinerSeedClient | None = None,
    seed_source: SeedSource = SeedSource.direct_llm,
    endpoint: str = "http://gpu-dev-pod-serve-svc:8000/v1",
    model: str = "nvidia/nemotron-3-super",
    gliner_endpoint: str = "https://integrate.api.nvidia.com/v1",
    gliner_model: str = "nvidia/gliner-pii",
    gliner_api_key_env: str = "NVIDIA_API_KEY",
    gliner_threshold: float = 0.3,
    max_tokens: int = 4096,
    timeout_sec: float = 180.0,
    skip_augmentation: bool = False,
    skip_augmentation_when_rule_covered: bool = False,
    validation_prompt_mode: ValidationPromptMode = ValidationPromptMode.full_text,
    validation_max_entities_per_call: int = 10,
    validation_excerpt_window_chars: int = 160,
) -> StagedDetectionExecution:
    config = _execution_config_from_params(locals())
    try:
        return _run_staged_detection_execution(
            request,
            client=client,
            seed_client=seed_client,
            config=config,
        )
    except Exception as exc:  # noqa: BLE001 - benchmark probe records per-case failures
        return StagedDetectionExecution(case=_errored_case(request, config.seed_source, exc), row={})


def _execution_config_from_params(params: dict[str, Any]) -> StagedExecutionConfig:
    return StagedExecutionConfig(
        **{key: value for key, value in params.items() if key in StagedExecutionConfig.model_fields}
    )


def _errored_case(request: StagedDetectionRequest, seed_source: SeedSource, exc: Exception) -> StagedDetectionCase:
    return StagedDetectionCase(
        case_id=request.case_id,
        row_index=request.row_index,
        seed_source=seed_source,
        status=CaseStatus.error,
        error=f"{type(exc).__name__}: {exc}",
    )


def _run_staged_detection_execution(
    request: StagedDetectionRequest,
    *,
    client: DirectDetectionClient,
    seed_client: GlinerSeedClient | None,
    config: StagedExecutionConfig,
) -> StagedDetectionExecution:
    started = time.perf_counter()
    row, seed_suggestion_count, seed_completion = _run_seed_phase(
        request,
        client=client,
        seed_client=seed_client,
        config=config,
    )
    validation_completion = _run_validation_phase(row, request, client, config)
    augmentation_completion = _run_augmentation_phase(row, request, client, config)
    artifact = _finalize_row(row, request)
    return StagedDetectionExecution(
        case=_completed_case(
            request,
            seed_completion,
            validation_completion,
            augmentation_completion,
            artifact,
            row,
            config=config,
            seed_suggestion_count=seed_suggestion_count,
            elapsed_sec=time.perf_counter() - started,
        ),
        row=row,
    )


def _run_seed_phase(
    request: StagedDetectionRequest,
    *,
    client: DirectDetectionClient,
    seed_client: GlinerSeedClient | None,
    config: StagedExecutionConfig,
) -> tuple[dict[str, Any], int, DirectCompletion]:
    if _uses_rule_short_circuit(request, config):
        return _run_rules_seed_phase(request)
    if config.seed_source == SeedSource.gliner:
        return _run_gliner_seed_phase(request, seed_client or HttpxGlinerSeedClient(), config)
    if config.seed_source in {SeedSource.rules, SeedSource.rules_trusted}:
        return _run_rules_seed_phase(request)
    if config.seed_source in {SeedSource.rules_plus_direct_llm, SeedSource.rules_router}:
        return _run_rules_plus_direct_llm_seed_phase(request, client, config)
    return _run_direct_llm_seed_phase(request, client, config)


def _run_gliner_seed_phase(
    request: StagedDetectionRequest,
    detector: GlinerSeedClient,
    config: StagedExecutionConfig,
) -> tuple[dict[str, Any], int, DirectCompletion]:
    completion = detector.detect(
        GlinerDetectionRequest(
            endpoint=config.gliner_endpoint,
            model=config.gliner_model,
            text=request.text,
            labels=request.labels,
            threshold=config.gliner_threshold,
            max_tokens=config.max_tokens,
            timeout_sec=config.timeout_sec,
            api_key_env=config.gliner_api_key_env,
        )
    )
    seed_spans = parse_raw_entities(raw_response=completion.content, text=request.text)
    return _seed_row_from_spans(request, seed_spans), _raw_detector_entity_count(completion.content), completion


def _run_direct_llm_seed_phase(
    request: StagedDetectionRequest,
    client: DirectDetectionClient,
    config: StagedExecutionConfig,
) -> tuple[dict[str, Any], int, DirectCompletion]:
    completion = _complete(
        client,
        prompt=_seed_prompt(request),
        config=config,
    )
    row, seed_suggestions = _seed_row_from_llm(request, completion.content)
    return row, len(seed_suggestions), completion


def _run_rules_plus_direct_llm_seed_phase(
    request: StagedDetectionRequest,
    client: DirectDetectionClient,
    config: StagedExecutionConfig,
) -> tuple[dict[str, Any], int, DirectCompletion]:
    completion = _complete(client, prompt=_seed_prompt(request), config=config)
    direct_spans, seed_suggestions = _direct_seed_spans(request, completion.content)
    rule_spans = detect_high_confidence_entities(request.text, labels=request.labels)
    row = _seed_row_from_spans(request, resolve_overlaps([*rule_spans, *direct_spans]))
    _limit_validation_candidates_to_sources(row, sources={"direct_seed"})
    return row, len(seed_suggestions) + len(rule_spans), completion


def _run_rules_seed_phase(request: StagedDetectionRequest) -> tuple[dict[str, Any], int, DirectCompletion]:
    seed_spans = detect_high_confidence_entities(request.text, labels=request.labels)
    completion = DirectCompletion(content="", elapsed_sec=0.0, usage={})
    return _seed_row_from_spans(request, seed_spans), len(seed_spans), completion


def _complete(
    client: DirectDetectionClient,
    *,
    prompt: str,
    config: StagedExecutionConfig,
) -> DirectCompletion:
    return client.complete(
        DirectGenerationRequest(
            endpoint=config.endpoint,
            model=config.model,
            prompt=prompt,
            max_tokens=config.max_tokens,
            timeout_sec=config.timeout_sec,
        )
    )


def _seed_prompt(request: StagedDetectionRequest) -> str:
    return build_direct_prompt(_direct_seed_request(request))


def _direct_seed_request(request: StagedDetectionRequest) -> DirectDetectionRequest:
    return DirectDetectionRequest(
        case_id=request.case_id,
        text=request.text,
        labels=request.labels,
        row_index=request.row_index,
        prompt_mode=PromptMode.compact,
        data_summary=request.data_summary,
    )


def _seed_row_from_llm(request: StagedDetectionRequest, content: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seed_spans, suggestions = _direct_seed_spans(request, content)
    return _seed_row_from_spans(request, seed_spans), suggestions


def _direct_seed_spans(request: StagedDetectionRequest, content: str) -> tuple[list[EntitySpan], list[dict[str, Any]]]:
    suggestions = _extract_entity_suggestions(content)
    seed_spans = _spans_from_suggestions(request.text, suggestions, labels=request.labels, source="direct_seed")
    return seed_spans, suggestions


def _seed_row_from_spans(request: StagedDetectionRequest, seed_spans: list[EntitySpan]) -> dict[str, Any]:
    allowed = set(request.labels)
    seed_spans = [_normalize_seed_span(request.text, span, allowed_labels=allowed) for span in seed_spans]
    seed_spans = [span for span in seed_spans if span.label in allowed]
    row = {COL_TEXT: request.text, COL_TAG_NOTATION: get_tag_notation(text=request.text)}
    row[COL_RAW_DETECTED] = ""
    row[COL_SEED_ENTITIES] = EntitiesSchema(entities=[_span_to_entity_schema(span) for span in seed_spans]).model_dump(
        mode="json"
    )
    prepare_validation_inputs(row)
    return row


def _normalize_seed_span(text: str, span: EntitySpan, *, allowed_labels: set[str]) -> EntitySpan:
    if _should_promote_birth_context_date_seed(text, span, allowed_labels=allowed_labels):
        return replace(span, entity_id=_seed_entity_id("date_of_birth", span), label="date_of_birth")
    return span


def _should_promote_birth_context_date_seed(
    text: str,
    span: EntitySpan,
    *,
    allowed_labels: set[str],
) -> bool:
    return span.label == "date" and "date_of_birth" in allowed_labels and _span_has_date_of_birth_context(text, span)


def _span_has_date_of_birth_context(text: str, span: EntitySpan) -> bool:
    before_start = max(0, span.start_position - VALIDATION_CONTEXT_WINDOW)
    after_end = min(len(text), span.end_position + VALIDATION_CONTEXT_WINDOW)
    return _contains_date_of_birth_context(
        text[before_start : span.start_position], span.value, text[span.end_position : after_end]
    )


def _seed_entity_id(label: str, span: EntitySpan) -> str:
    return f"{label}_{span.start_position}_{span.end_position}"


def _limit_validation_candidates_to_sources(row: dict[str, Any], *, sources: set[str]) -> None:
    text = str(row.get(COL_TEXT, ""))
    seed_spans = [span for span in _seed_entity_spans(row) if span.source in sources]
    row[COL_SEED_VALIDATION_CANDIDATES] = ValidationCandidatesSchema(
        candidates=build_validation_candidates(text=text, entities=seed_spans)
    ).model_dump(mode="json")


def _run_validation_phase(
    row: dict[str, Any],
    request: StagedDetectionRequest,
    client: DirectDetectionClient,
    config: StagedExecutionConfig,
) -> DirectCompletion:
    if config.seed_source == SeedSource.rules_trusted or _uses_rule_short_circuit(request, config):
        _trust_seed_entities(row)
        return DirectCompletion(content="", elapsed_sec=0.0, usage={})
    candidates = ValidationCandidatesSchema.from_raw(row.get(COL_SEED_VALIDATION_CANDIDATES, {}))
    if not candidates.candidates:
        row[COL_VALIDATION_DECISIONS] = {"decisions": []}
        row[COL_VALIDATED_ENTITIES] = {"decisions": []}
        apply_validation_to_seed_entities(row)
        return DirectCompletion(content='{"decisions": []}', elapsed_sec=0.0, usage={})
    if config.validation_prompt_mode == ValidationPromptMode.chunked_excerpt:
        completion = _run_chunked_validation_phase(row, request, candidates, client, config)
    else:
        completion = _run_full_text_validation_phase(row, request, candidates, client, config)
    row[COL_VALIDATION_DECISIONS] = _extract_validation_decisions(
        completion.content,
        candidates=candidates,
        labels=request.labels,
    )
    enrich_validation_decisions(row)
    apply_validation_to_seed_entities(row)
    return completion


def _run_full_text_validation_phase(
    row: dict[str, Any],
    request: StagedDetectionRequest,
    candidates: ValidationCandidatesSchema,
    client: DirectDetectionClient,
    config: StagedExecutionConfig,
) -> DirectCompletion:
    completion = _complete(
        client,
        prompt=_validation_prompt(request, candidates),
        config=config,
    )
    return completion


def _run_chunked_validation_phase(
    row: dict[str, Any],
    request: StagedDetectionRequest,
    candidates: ValidationCandidatesSchema,
    client: DirectDetectionClient,
    config: StagedExecutionConfig,
) -> DirectCompletion:
    completions: list[DirectCompletion] = []
    chunk_results: list[RawValidationDecisionsSchema] = []
    for chunk in _validation_chunks(row, candidates, config):
        completion = _complete(client, prompt=_chunk_validation_prompt(request, chunk), config=config)
        completions.append(completion)
        chunk_results.append(
            RawValidationDecisionsSchema.from_raw(
                _extract_validation_decisions(completion.content, candidates=chunk[1], labels=request.labels)
            )
        )
    decisions = merge_chunk_decisions(chunk_results, candidates)
    return _combine_completions(completions, content=json.dumps(decisions, ensure_ascii=True, sort_keys=True))


def _validation_chunks(
    row: dict[str, Any],
    candidates: ValidationCandidatesSchema,
    config: StagedExecutionConfig,
) -> list[tuple[str, ValidationCandidatesSchema]]:
    seed_spans = _seed_entity_spans(row)
    ordered = order_candidates_by_position(candidates, seed_spans)
    return [
        (_validation_excerpt(row, chunk, seed_spans, config), _chunk_candidate_schema(chunk))
        for chunk in chunk_candidates(ordered, config.validation_max_entities_per_call)
    ]


def _validation_excerpt(
    row: dict[str, Any],
    chunk: list[tuple[Any, EntitySpan]],
    seed_spans: list[EntitySpan],
    config: StagedExecutionConfig,
) -> str:
    notation = TagNotation(str(row.get(COL_TAG_NOTATION) or TagNotation.sentinel.value))
    return build_chunk_excerpt(
        text=str(row.get(COL_TEXT, "")),
        chunk_spans=[span for _candidate, span in chunk],
        all_spans=seed_spans,
        window_chars=config.validation_excerpt_window_chars,
        notation=notation,
    )


def _chunk_candidate_schema(chunk: list[tuple[Any, EntitySpan]]) -> ValidationCandidatesSchema:
    return ValidationCandidatesSchema(candidates=[candidate for candidate, _span in chunk])


def _chunk_validation_prompt(
    request: StagedDetectionRequest,
    chunk: tuple[str, ValidationCandidatesSchema],
) -> str:
    excerpt, candidates = chunk
    return _validation_prompt(request, candidates, input_text=excerpt)


def _seed_entity_spans(row: dict[str, Any]) -> list[EntitySpan]:
    return [
        EntitySpan(
            entity_id=entity.id,
            value=entity.value,
            label=entity.label,
            start_position=entity.start_position,
            end_position=entity.end_position,
            score=entity.score,
            source=entity.source,
        )
        for entity in EntitiesSchema.from_raw(row.get(COL_SEED_ENTITIES, {})).entities
    ]


def _combine_completions(completions: list[DirectCompletion], *, content: str) -> DirectCompletion:
    return DirectCompletion(
        content=content,
        elapsed_sec=sum(completion.elapsed_sec for completion in completions),
        usage=_sum_completion_usage(completions),
    )


def _sum_completion_usage(completions: list[DirectCompletion]) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for completion in completions:
        for key, value in completion.usage.items():
            if isinstance(value, int):
                totals[key] += value
    return dict(sorted(totals.items()))


def _trust_seed_entities(row: dict[str, Any]) -> None:
    candidates = ValidationCandidatesSchema.from_raw(row.get(COL_SEED_VALIDATION_CANDIDATES, {}))
    row[COL_VALIDATED_ENTITIES] = ValidatedDecisionsSchema(
        decisions=[
            ValidatedDecisionSchema(
                id=candidate.id,
                decision="keep",
                value=candidate.value,
                label=candidate.label,
                reason="trusted deterministic rule",
            )
            for candidate in candidates.candidates
        ]
    ).model_dump(mode="json")
    apply_validation_to_seed_entities(row)


def _validation_prompt(
    request: StagedDetectionRequest,
    candidates: ValidationCandidatesSchema,
    *,
    input_text: str | None = None,
) -> str:
    text_for_prompt = input_text if input_text is not None else request.text
    label_guidance = _validation_label_guidance(request.labels)
    return f"""Validate candidate privacy-sensitive entities.
Use only these decisions: keep, reclass, drop.
Use only these labels when reclassifying: {", ".join(request.labels)}.
Prefer keep when the candidate already has the right specific label. For example, reclass date_of_birth to date only when the surrounding context is not birth-related.
{label_guidance}
Return JSON only with this shape:
{{"decisions": [{{"id": "candidate id", "decision": "keep|reclass|drop", "proposed_label": "", "reason": "short reason"}}]}}

Context text:
---
{text_for_prompt}
---

Candidates:
{candidates.model_dump_json()}
"""


def _validation_label_guidance(labels: list[str]) -> str:
    allowed = set(labels)
    lines: list[str] = []
    if "degree" in allowed:
        lines.append(
            "Prefer degree for credential names and degree abbreviations such as Bachelor of Science, BSc, BA, MA, MSc, PhD, or JD; use education_level for broad levels such as undergraduate, graduate, or high school."
        )
    if not lines:
        return ""
    return "Label boundary guidance:\n" + "\n".join(f"- {line}" for line in lines)


def _run_augmentation_phase(
    row: dict[str, Any],
    request: StagedDetectionRequest,
    client: DirectDetectionClient,
    config: StagedExecutionConfig,
) -> DirectCompletion:
    if _should_skip_augmentation(request, config):
        row[COL_AUGMENTED_ENTITIES] = {"entities": []}
        return DirectCompletion(content="", elapsed_sec=0.0, usage={})
    completion = _complete(
        client,
        prompt=_augmentation_prompt(request, row),
        config=config,
    )
    row[COL_AUGMENTED_ENTITIES] = {
        "entities": _allowed_entity_suggestions(_extract_entity_suggestions(completion.content), labels=request.labels)
    }
    return completion


def _should_skip_augmentation(request: StagedDetectionRequest, config: StagedExecutionConfig) -> bool:
    if config.skip_augmentation:
        return True
    if _uses_rule_short_circuit(request, config):
        return True
    if not config.skip_augmentation_when_rule_covered:
        return False
    if config.seed_source not in {SeedSource.rules, SeedSource.rules_trusted, SeedSource.rules_plus_direct_llm}:
        return False
    return set(request.labels).issubset(STRUCTURED_RULE_FAST_LANE_LABELS)


def _uses_rule_short_circuit(request: StagedDetectionRequest, config: StagedExecutionConfig) -> bool:
    return config.seed_source == SeedSource.rules_router and _is_rule_covered_label_set(request)


def _augmentation_prompt(request: StagedDetectionRequest, row: dict[str, Any]) -> str:
    seed_entities = row.get(COL_SEED_ENTITIES_JSON, "[]")
    label_guidance = _augmentation_label_guidance(request.labels)
    return f"""Find additional sensitive entities not already covered by the validated seed entities.
Use only these labels: {", ".join(request.labels)}.
Return exact substrings only. Do not invent values.
{label_guidance}
Return JSON only with this shape:
{{"entities": [{{"value": "exact substring", "label": "one_allowed_label", "reason": "short reason"}}]}}

Input text:
---
{request.text}
---

Validated seed entities:
{seed_entities}
"""


def _augmentation_label_guidance(labels: list[str]) -> str:
    allowed = set(labels)
    lines: list[str] = []
    if "first_name" in allowed:
        lines.append(
            "For family/member lists, split personal names connected by 'and' into separate first_name or last_name entities. Do not label a list of people as organization_name."
        )
    if "last_name" in allowed and ({"place_name", "company_name", "organization_name"} & allowed):
        lines.append(
            "If a surname appears inside a street, place, organization, company, or possessive business phrase, also return the surname substring as last_name when it is not already covered."
        )
    if not lines:
        return ""
    return "Label boundary guidance:\n" + "\n".join(f"- {line}" for line in lines)


def _finalize_row(row: dict[str, Any], request: StagedDetectionRequest) -> DetectionArtifactRow:
    merge_and_build_candidates(row)
    apply_validation_and_finalize(row)
    normalize_birth_context_final_entities(row, allowed_labels=request.labels)
    seed_entities = EntitiesSchema.from_raw(row.get(COL_SEED_ENTITIES, {})).entities
    final_entities = EntitiesSchema.from_raw(row.get(COL_DETECTED_ENTITIES, {})).entities
    augmented = _augmented_entity_schemas(row, request)
    return build_detection_artifact_row_from_entities(
        workflow_name="staged-direct-detection",
        batch_file="staged-direct-detection",
        row_index=request.row_index,
        seed_entities=list(seed_entities),
        seed_validation_candidate_count=_candidate_count(row.get(COL_SEED_VALIDATION_CANDIDATES)),
        merged_validation_candidate_count=_candidate_count(row.get(COL_VALIDATION_CANDIDATES)),
        augmented_entities=augmented,
        final_entities=list(final_entities),
    )


def _completed_case(
    request: StagedDetectionRequest,
    seed: DirectCompletion,
    validation: DirectCompletion,
    augmentation: DirectCompletion,
    artifact: DetectionArtifactRow,
    row: dict[str, Any],
    config: StagedExecutionConfig,
    seed_suggestion_count: int,
    elapsed_sec: float,
) -> StagedDetectionCase:
    phase_usage = PhaseUsage(seed=seed.usage, validation=validation.usage, augmentation=augmentation.usage)
    phase_model_work = _phase_model_work(request, artifact, config)
    phase_model_requests = _phase_model_requests(artifact, phase_model_work, config)
    model_elapsed_sec = seed.elapsed_sec + validation.elapsed_sec + augmentation.elapsed_sec
    return StagedDetectionCase(
        case_id=request.case_id,
        row_index=request.row_index,
        seed_source=config.seed_source,
        status=CaseStatus.completed,
        elapsed_sec=elapsed_sec,
        model_elapsed_sec=model_elapsed_sec,
        phase_usage=phase_usage,
        phase_model_work=phase_model_work,
        phase_skip_reasons=_phase_skip_reasons(request, artifact, config),
        phase_model_requests=phase_model_requests,
        total_usage=_sum_usage(phase_usage),
        model_phase_count=_model_phase_count(phase_model_work),
        model_request_count=_model_request_count(phase_model_requests),
        rule_covered_label_set=_is_rule_covered_label_set(request),
        seed_suggestion_count=seed_suggestion_count,
        seed_entity_count=artifact.seed_entity_count,
        validation_candidate_count=artifact.seed_validation_candidate_count,
        validation_decision_count=_validated_decision_count(row.get(COL_VALIDATED_ENTITIES)),
        augmented_suggestion_count=artifact.augmented_entity_count,
        final_entity_count=artifact.final_entity_count,
        final_entity_signature_count=artifact.final_entity_signature_count,
        final_entity_signature_hashes=artifact.final_entity_signature_hashes,
        final_label_counts=artifact.final_label_counts,
        artifact=artifact,
    )


def _phase_model_work(
    request: StagedDetectionRequest, artifact: DetectionArtifactRow, config: StagedExecutionConfig
) -> PhaseModelWork:
    return PhaseModelWork(
        seed=_uses_seed_model(request, config),
        validation=_uses_validation_model(request, artifact, config),
        augmentation=not _should_skip_augmentation(request, config),
    )


def _uses_seed_model(request: StagedDetectionRequest, config: StagedExecutionConfig) -> bool:
    if _uses_rule_short_circuit(request, config):
        return False
    return config.seed_source in {
        SeedSource.direct_llm,
        SeedSource.gliner,
        SeedSource.rules_plus_direct_llm,
        SeedSource.rules_router,
    }


def _uses_validation_model(
    request: StagedDetectionRequest, artifact: DetectionArtifactRow, config: StagedExecutionConfig
) -> bool:
    if _uses_rule_short_circuit(request, config):
        return False
    if (
        config.seed_source in {SeedSource.rules_trusted, SeedSource.rules_router}
        and artifact.seed_validation_candidate_count == 0
    ):
        return False
    if config.seed_source == SeedSource.rules_trusted:
        return False
    return artifact.seed_validation_candidate_count > 0


def _phase_skip_reasons(
    request: StagedDetectionRequest, artifact: DetectionArtifactRow, config: StagedExecutionConfig
) -> PhaseSkipReasons:
    return PhaseSkipReasons(
        seed=_seed_skip_reason(request, config),
        validation=_validation_skip_reason(request, artifact, config),
        augmentation=_augmentation_skip_reason(request, config),
    )


def _seed_skip_reason(request: StagedDetectionRequest, config: StagedExecutionConfig) -> str | None:
    if config.seed_source in {SeedSource.rules, SeedSource.rules_trusted} or _uses_rule_short_circuit(request, config):
        return "deterministic_rules"
    return None


def _validation_skip_reason(
    request: StagedDetectionRequest, artifact: DetectionArtifactRow, config: StagedExecutionConfig
) -> str | None:
    if config.seed_source == SeedSource.rules_trusted or _uses_rule_short_circuit(request, config):
        return "trusted_rules"
    if artifact.seed_validation_candidate_count == 0:
        return "no_seed_candidates"
    return None


def _augmentation_skip_reason(request: StagedDetectionRequest, config: StagedExecutionConfig) -> str | None:
    if config.skip_augmentation:
        return "disabled"
    if _should_skip_augmentation(request, config):
        return "rule_covered_labels"
    return None


def _model_phase_count(phase_model_work: PhaseModelWork) -> int:
    return sum((phase_model_work.seed, phase_model_work.validation, phase_model_work.augmentation))


def _phase_model_requests(
    artifact: DetectionArtifactRow,
    phase_model_work: PhaseModelWork,
    config: StagedExecutionConfig,
) -> PhaseModelRequests:
    return PhaseModelRequests(
        seed=1 if phase_model_work.seed else 0,
        validation=_validation_model_request_count(artifact, phase_model_work, config),
        augmentation=1 if phase_model_work.augmentation else 0,
    )


def _validation_model_request_count(
    artifact: DetectionArtifactRow,
    phase_model_work: PhaseModelWork,
    config: StagedExecutionConfig,
) -> int:
    if not phase_model_work.validation:
        return 0
    if config.validation_prompt_mode == ValidationPromptMode.full_text:
        return 1
    return _ceil_div(artifact.seed_validation_candidate_count, config.validation_max_entities_per_call)


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _model_request_count(phase_model_requests: PhaseModelRequests) -> int:
    return phase_model_requests.seed + phase_model_requests.validation + phase_model_requests.augmentation


def _is_rule_covered_label_set(request: StagedDetectionRequest) -> bool:
    return set(request.labels).issubset(STRUCTURED_RULE_FAST_LANE_LABELS)


def _extract_entity_suggestions(content: str) -> list[dict[str, str]]:
    payload = _load_embedded_json(content)
    raw_entities = payload.get("entities", []) if isinstance(payload, dict) else []
    return [
        {"value": str(item.get("value", "")).strip(), "label": str(item.get("label", "")).strip()}
        for item in raw_entities
        if isinstance(item, dict) and item.get("value") and item.get("label")
    ]


def _allowed_entity_suggestions(suggestions: list[dict[str, str]], *, labels: list[str]) -> list[dict[str, str]]:
    allowed = set(labels)
    return [suggestion for suggestion in suggestions if suggestion["label"] in allowed]


def _extract_validation_decisions(
    content: str,
    *,
    candidates: ValidationCandidatesSchema | None = None,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    payload = _load_embedded_json(content)
    decisions = payload.get("decisions", []) if isinstance(payload, dict) else []
    if not isinstance(decisions, list):
        decisions = []
    if candidates is not None:
        decisions = _preserve_specific_seed_labels(decisions, candidates)
    if labels is not None:
        decisions = _keep_invalid_reclass_labels(decisions, allowed_labels=set(labels))
    return {"decisions": decisions}


def _preserve_specific_seed_labels(
    decisions: list[object],
    candidates: ValidationCandidatesSchema,
) -> list[object]:
    candidate_by_id = {candidate.id: candidate for candidate in candidates.candidates}
    normalized: list[object] = []
    for decision in decisions:
        if not isinstance(decision, dict):
            normalized.append(decision)
            continue
        candidate = candidate_by_id.get(str(decision.get("id") or ""))
        if candidate is None or not _should_preserve_specific_seed_label(decision, candidate):
            normalized.append(decision)
            continue
        normalized.append(
            {
                **decision,
                "decision": "keep",
                "proposed_label": "",
                "reason": _preserved_label_reason(decision.get("reason")),
            }
        )
    return normalized


def _should_preserve_specific_seed_label(decision: dict[str, object], candidate: Any) -> bool:
    if str(decision.get("decision") or "") != "reclass":
        return False
    proposed_label = str(decision.get("proposed_label") or "")
    if candidate.label == "date_of_birth" and proposed_label == "date":
        return _has_date_of_birth_context(candidate)
    return False


def _has_date_of_birth_context(candidate: Any) -> bool:
    return _contains_date_of_birth_context(candidate.context_before, candidate.value, candidate.context_after)


def _keep_invalid_reclass_labels(decisions: list[object], *, allowed_labels: set[str]) -> list[object]:
    normalized: list[object] = []
    for decision in decisions:
        if not isinstance(decision, dict):
            normalized.append(decision)
            continue
        if not _has_invalid_reclass_label(decision, allowed_labels=allowed_labels):
            normalized.append(decision)
            continue
        normalized.append(
            {
                **decision,
                "decision": "keep",
                "proposed_label": "",
                "reason": _invalid_reclass_label_reason(decision.get("reason"), decision.get("proposed_label")),
            }
        )
    return normalized


def _has_invalid_reclass_label(decision: dict[str, object], *, allowed_labels: set[str]) -> bool:
    if str(decision.get("decision") or "") != "reclass":
        return False
    proposed_label = str(decision.get("proposed_label") or "").strip()
    return proposed_label not in allowed_labels


def _invalid_reclass_label_reason(reason: object, proposed_label: object) -> str:
    text = str(reason or "").strip()
    suffix = f"ignored invalid reclass label {str(proposed_label or '').strip()!r}"
    return f"{text}; {suffix}" if text else suffix


def normalize_birth_context_final_entities(
    row: dict[str, Any], *, allowed_labels: list[str] | set[str]
) -> dict[str, Any]:
    """Demote native date_of_birth spans without birth context back to date."""
    allowed = set(allowed_labels)
    if not {"date", "date_of_birth"}.issubset(allowed):
        return row
    text = str(row.get(COL_TEXT, ""))
    entities = EntitiesSchema.from_raw(row.get(COL_DETECTED_ENTITIES, {})).entities
    normalized: list[EntitySchema] = []
    changed = False
    for entity in entities:
        if entity.label == "date_of_birth" and not _entity_has_date_of_birth_context(text, entity):
            normalized.append(entity.model_copy(update={"id": _entity_id("date", entity), "label": "date"}))
            changed = True
        else:
            normalized.append(entity)
    if changed:
        row[COL_DETECTED_ENTITIES] = EntitiesSchema(entities=normalized).model_dump(mode="json")
        row[COL_TAGGED_TEXT] = build_tagged_text(
            text=text,
            entities=[_entity_schema_to_span(entity) for entity in normalized],
        )
    return row


def _entity_has_date_of_birth_context(text: str, entity: EntitySchema) -> bool:
    before_start = max(0, entity.start_position - VALIDATION_CONTEXT_WINDOW)
    after_end = min(len(text), entity.end_position + VALIDATION_CONTEXT_WINDOW)
    return _contains_date_of_birth_context(
        text[before_start : entity.start_position],
        entity.value,
        text[entity.end_position : after_end],
    )


def _entity_id(label: str, entity: EntitySchema) -> str:
    return f"{label}_{entity.start_position}_{entity.end_position}"


def _entity_schema_to_span(entity: EntitySchema) -> EntitySpan:
    return EntitySpan(
        entity_id=entity.id,
        value=entity.value,
        label=entity.label,
        start_position=entity.start_position,
        end_position=entity.end_position,
        score=entity.score,
        source=entity.source,
    )


def _contains_date_of_birth_context(context_before: object, value: object, context_after: object) -> bool:
    context = f"{context_before} {value} {context_after}"
    return bool(_DATE_OF_BIRTH_CONTEXT_RE.search(context))


def _preserved_label_reason(reason: object) -> str:
    text = str(reason or "").strip()
    suffix = "preserved specific seed label from birth-related context"
    return f"{text}; {suffix}" if text else suffix


def _raw_detector_entity_count(content: str) -> int:
    payload = _load_embedded_json(content)
    entities = payload.get("entities", []) if isinstance(payload, dict) else []
    return len(entities) if isinstance(entities, list) else 0


def _spans_from_suggestions(
    text: str, suggestions: list[dict[str, str]], *, labels: list[str], source: str
) -> list[EntitySpan]:
    allowed = set(labels)
    cleaned = [item for item in suggestions if item["value"] and item["label"] in allowed]
    spans = apply_augmented_entities(text=text, entities=[], augmented_output={"entities": cleaned})
    return [
        EntitySpan(
            entity_id=span.entity_id,
            value=span.value,
            label=span.label,
            start_position=span.start_position,
            end_position=span.end_position,
            score=span.score,
            source=source,
        )
        for span in spans
    ]


def _span_to_entity_schema(span: EntitySpan) -> EntitySchema:
    return EntitySchema(
        id=span.entity_id,
        value=span.value,
        label=span.label,
        start_position=span.start_position,
        end_position=span.end_position,
        score=span.score,
        source=span.source,
    )


def _augmented_entity_schemas(row: dict[str, Any], request: StagedDetectionRequest) -> list[EntitySchema]:
    spans = _spans_from_suggestions(
        request.text,
        _extract_entities_from_payload(row.get(COL_AUGMENTED_ENTITIES)),
        labels=request.labels,
        source="augmenter",
    )
    return [_span_to_entity_schema(span) for span in spans]


def _extract_entities_from_payload(payload: object) -> list[dict[str, str]]:
    entities = payload.get("entities", []) if isinstance(payload, dict) else []
    return [
        {"value": str(item.get("value", "")).strip(), "label": str(item.get("label", "")).strip()}
        for item in entities
        if isinstance(item, dict)
    ]


def _candidate_count(raw: object) -> int:
    return len(ValidationCandidatesSchema.from_raw(raw).candidates)


def _validated_decision_count(raw: object) -> int:
    return len(ValidatedDecisionsSchema.from_raw(raw).decisions)


def _sum_usage(usage: PhaseUsage) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for phase in (usage.seed, usage.validation, usage.augmentation):
        for key, value in phase.items():
            if isinstance(value, int):
                totals[key] += value
    return dict(sorted(totals.items()))


def apply_baseline_comparisons(
    cases: list[StagedDetectionCase],
    baseline_artifacts: Path,
) -> list[StagedDetectionCase]:
    baseline = _read_baseline_artifacts(baseline_artifacts)
    return [_case_with_comparison(case, baseline.get(case.row_index)) for case in cases]


def _case_with_comparison(case: StagedDetectionCase, baseline_row: dict[str, Any] | None) -> StagedDetectionCase:
    if baseline_row is None or case.status != CaseStatus.completed or case.artifact is None:
        return case
    baseline_hashes = _baseline_signature_hashes(baseline_row)
    if baseline_hashes is None:
        return case
    comparison = compare_signature_sets(
        baseline_hashes=baseline_hashes,
        baseline_labels=_signature_labels(baseline_row),
        direct_hashes=set(case.artifact.final_entity_signature_hashes),
        direct_labels=case.artifact.final_entity_signature_labels,
    )
    return case.model_copy(update={"comparison": comparison})


def _baseline_signature_hashes(row: dict[str, Any]) -> set[str] | None:
    hashes = row.get("final_entity_signature_hashes")
    if not isinstance(hashes, list):
        return None
    return {str(item) for item in hashes}


def _read_baseline_artifacts(path: Path) -> dict[int, dict[str, Any]]:
    baseline: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            row_index = int(row.get("row_index", 0))
            if row_index in baseline:
                raise ValueError(f"baseline artifacts has multiple rows for row_index={row_index}")
            baseline[row_index] = row
    return baseline


def _signature_labels(row: dict[str, Any]) -> dict[str, str]:
    return {
        key.removeprefix("final_entity_signature_labels."): str(value)
        for key, value in row.items()
        if key.startswith("final_entity_signature_labels.") and value is not None
    }


def run_probe(
    input_path: Path,
    *,
    text_column: str,
    labels: list[str],
    output: Path | None = None,
    overwrite: bool = False,
    endpoint: str = "http://gpu-dev-pod-serve-svc:8000/v1",
    model: str = "nvidia/nemotron-3-super",
    seed_source: SeedSource = SeedSource.direct_llm,
    gliner_endpoint: str = "https://integrate.api.nvidia.com/v1",
    gliner_model: str = "nvidia/gliner-pii",
    gliner_api_key_env: str = "NVIDIA_API_KEY",
    gliner_threshold: float = 0.3,
    skip_augmentation: bool = False,
    skip_augmentation_when_rule_covered: bool = False,
    validation_prompt_mode: ValidationPromptMode = ValidationPromptMode.full_text,
    validation_max_entities_per_call: int = 10,
    validation_excerpt_window_chars: int = 160,
    limit: int = 1,
    offset: int = 0,
    baseline_artifacts: Path | None = None,
) -> StagedDetectionRun:
    requests = _load_requests(input_path, text_column=text_column, labels=labels, limit=limit, offset=offset)
    config = _execution_config_from_params(locals())
    cases = _run_probe_cases(requests, config)
    if baseline_artifacts is not None:
        cases = apply_baseline_comparisons(cases, baseline_artifacts)
    result = StagedDetectionRun(
        input_path=str(input_path), text_column=text_column, endpoint=endpoint, model=model, labels=labels, rows=cases
    )
    if output is not None:
        write_outputs(result, output, overwrite=overwrite)
    return result


def _run_probe_cases(
    requests: list[StagedDetectionRequest],
    config: StagedExecutionConfig,
) -> list[StagedDetectionCase]:
    client = HttpxDirectDetectionClient()
    seed_client = HttpxGlinerSeedClient() if config.seed_source == SeedSource.gliner else None
    return [
        run_staged_detection_case(
            request,
            client=client,
            seed_client=seed_client,
            seed_source=config.seed_source,
            endpoint=config.endpoint,
            model=config.model,
            gliner_endpoint=config.gliner_endpoint,
            gliner_model=config.gliner_model,
            gliner_api_key_env=config.gliner_api_key_env,
            gliner_threshold=config.gliner_threshold,
            skip_augmentation=config.skip_augmentation,
            skip_augmentation_when_rule_covered=config.skip_augmentation_when_rule_covered,
            validation_prompt_mode=config.validation_prompt_mode,
            validation_max_entities_per_call=config.validation_max_entities_per_call,
            validation_excerpt_window_chars=config.validation_excerpt_window_chars,
        )
        for request in requests
    ]


def _load_requests(
    input_path: Path, *, text_column: str, labels: list[str], limit: int, offset: int
) -> list[StagedDetectionRequest]:
    dataframe = pd.read_csv(input_path)
    if text_column not in dataframe.columns:
        raise ValueError(f"text column {text_column!r} not found in {input_path}")
    selected = dataframe.iloc[offset : offset + limit]
    return [
        StagedDetectionRequest(
            case_id=f"{input_path.stem}-row-{int(index)}",
            text=str(row[text_column]),
            labels=labels,
            row_index=int(index),
        )
        for index, row in selected.iterrows()
    ]


def write_outputs(result: StagedDetectionRun, output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise ValueError(f"output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    _write_jsonl(output_dir / "staged-detection-cases.jsonl", [_case_payload(case) for case in result.rows])
    _write_jsonl(output_dir / "staged-detection-artifacts.jsonl", [_artifact_payload(case) for case in result.rows])
    (output_dir / "summary.json").write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")


def _case_payload(case: StagedDetectionCase) -> dict[str, Any]:
    payload = case.model_dump(exclude={"artifact"})
    payload["record_type"] = "staged_detection_case"
    return payload


def _artifact_payload(case: StagedDetectionCase) -> dict[str, Any]:
    payload = case.artifact.model_dump() if case.artifact is not None else {}
    payload.update({"case_id": case.case_id, "row_index": case.row_index, "record_type": "staged_detection_artifact"})
    return payload


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def render_result(result: StagedDetectionRun, *, json_output: bool) -> str:
    if json_output:
        return result.model_dump_json(indent=2)
    completed = len(result.rows) - result.error_count
    return f"Ran {completed}/{len(result.rows)} staged detection case(s); errors={result.error_count}"


@app.default
def main(
    input_path: Path,
    *,
    text_column: Annotated[str, cyclopts.Parameter("--text-column")],
    labels: Annotated[str, cyclopts.Parameter("--labels")],
    output: Annotated[Path | None, cyclopts.Parameter(("--output", "-o"))] = None,
    overwrite: Annotated[bool, cyclopts.Parameter("--overwrite")] = False,
    endpoint: Annotated[str, cyclopts.Parameter("--endpoint")] = "http://gpu-dev-pod-serve-svc:8000/v1",
    model: Annotated[str, cyclopts.Parameter("--model")] = "nvidia/nemotron-3-super",
    seed_source: Annotated[SeedSource, cyclopts.Parameter("--seed-source")] = SeedSource.direct_llm,
    gliner_endpoint: Annotated[str, cyclopts.Parameter("--gliner-endpoint")] = "https://integrate.api.nvidia.com/v1",
    gliner_model: Annotated[str, cyclopts.Parameter("--gliner-model")] = "nvidia/gliner-pii",
    gliner_api_key_env: Annotated[str, cyclopts.Parameter("--gliner-api-key-env")] = "NVIDIA_API_KEY",
    gliner_threshold: Annotated[float, cyclopts.Parameter("--gliner-threshold")] = 0.3,
    skip_augmentation: Annotated[bool, cyclopts.Parameter("--skip-augmentation")] = False,
    skip_augmentation_when_rule_covered: Annotated[
        bool, cyclopts.Parameter("--skip-augmentation-when-rule-covered")
    ] = False,
    validation_prompt_mode: Annotated[
        ValidationPromptMode, cyclopts.Parameter("--validation-prompt-mode")
    ] = ValidationPromptMode.full_text,
    validation_max_entities_per_call: Annotated[int, cyclopts.Parameter("--validation-max-entities-per-call")] = 10,
    validation_excerpt_window_chars: Annotated[int, cyclopts.Parameter("--validation-excerpt-window-chars")] = 160,
    limit: Annotated[int, cyclopts.Parameter("--limit")] = 1,
    offset: Annotated[int, cyclopts.Parameter("--offset")] = 0,
    baseline_artifacts: Annotated[Path | None, cyclopts.Parameter("--baseline-artifacts")] = None,
    json_output: Annotated[bool, cyclopts.Parameter("--json")] = False,
    log_format: Annotated[LogFormat, cyclopts.Parameter("--log-format")] = LogFormat.plain,
) -> None:
    params = locals()
    json_output = bool(params.pop("json_output"))
    result = _run_main_probe(params)
    sys.stdout.write(render_result(result, json_output=json_output) + "\n")
    if result.error_count:
        raise SystemExit(1)


def _run_main_probe(params: dict[str, Any]) -> StagedDetectionRun:
    configure_logging(params.pop("log_format"))
    params["labels"] = parse_labels(params["labels"])
    try:
        return run_probe(**params)
    except (ValueError, ValidationError, httpx.HTTPError) as exc:
        log_bad_input(str(exc))
        raise SystemExit(125) from exc


if __name__ == "__main__":
    app()
