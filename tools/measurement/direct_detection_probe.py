#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run benchmark-only DD-free direct entity extraction probes.

Usage:
    uv run python tools/measurement/direct_detection_probe.py docs/data/NVIDIA_synthetic_biographies.csv \
      --text-column biography --labels age,city,first_name,last_name,occupation \
      --output /tmp/direct-probe --overwrite
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Protocol

import cyclopts
import httpx
import pandas as pd
from analyze_detection_artifacts import DetectionArtifactRow, build_detection_artifact_row_from_entities
from dd_parser_compat import _load_embedded_json
from pydantic import BaseModel, Field, ValidationError, model_validator

from anonymizer.engine.detection.detection_workflow import _format_label_examples
from anonymizer.engine.detection.postprocess import EntitySpan, apply_augmented_entities, expand_entity_occurrences
from anonymizer.engine.schemas import EntitySchema

app = cyclopts.App(help=__doc__)
logger = logging.getLogger("measurement.direct_detection_probe")


class CaseStatus(StrEnum):
    completed = "completed"
    error = "error"


class LogFormat(StrEnum):
    plain = "plain"
    json = "json"


class PromptMode(StrEnum):
    compact = "compact"
    recall = "recall"


class DirectCompletion(BaseModel):
    content: str
    elapsed_sec: float
    usage: dict[str, Any] = Field(default_factory=dict)


class DirectDetectionRequest(BaseModel):
    case_id: str
    text: str
    labels: list[str] = Field(min_length=1)
    row_index: int = 0
    prompt_mode: PromptMode = PromptMode.compact
    data_summary: str | None = None

    @model_validator(mode="after")
    def normalize_labels(self) -> DirectDetectionRequest:
        self.labels = list(dict.fromkeys(label.strip() for label in self.labels if label.strip()))
        if not self.labels:
            raise ValueError("labels must contain at least one non-empty label")
        return self


class DirectGenerationRequest(BaseModel):
    endpoint: str
    model: str
    prompt: str
    max_tokens: int = Field(gt=0)
    temperature: float = 0.0
    top_p: float = 1.0
    timeout_sec: float = Field(gt=0)
    json_mode: bool = True
    disable_thinking: bool = True


class SignatureComparison(BaseModel):
    baseline_final_entity_signature_count: int
    shared_final_entity_signature_count: int
    baseline_only_final_entity_signature_count: int
    direct_only_final_entity_signature_count: int
    baseline_only_label_counts: dict[str, int] = Field(default_factory=dict)
    direct_only_label_counts: dict[str, int] = Field(default_factory=dict)


class DirectDetectionCase(BaseModel):
    case_id: str
    row_index: int
    status: CaseStatus
    elapsed_sec: float | None = None
    usage: dict[str, Any] = Field(default_factory=dict)
    raw_suggestion_count: int = 0
    allowed_suggestion_count: int = 0
    final_entity_count: int = 0
    final_entity_signature_count: int = 0
    final_entity_signature_hashes: list[str] = Field(default_factory=list)
    final_label_counts: dict[str, int] = Field(default_factory=dict)
    comparison: SignatureComparison | None = None
    artifact: DetectionArtifactRow | None = None
    error: str | None = None


class DirectDetectionRun(BaseModel):
    input_path: str
    text_column: str
    endpoint: str
    model: str
    prompt_mode: PromptMode
    labels: list[str]
    rows: list[DirectDetectionCase] = Field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for row in self.rows if row.status == CaseStatus.error)


class DirectDetectionClient(Protocol):
    def complete(self, request: DirectGenerationRequest) -> DirectCompletion: ...


class HttpxDirectDetectionClient:
    def complete(self, request: DirectGenerationRequest) -> DirectCompletion:
        payload = build_chat_payload(request)
        response = httpx.post(
            f"{request.endpoint.rstrip('/')}/chat/completions",
            json=payload,
            timeout=request.timeout_sec,
        )
        response.raise_for_status()
        data = response.json()
        return DirectCompletion(
            content=str(data["choices"][0]["message"].get("content") or ""),
            elapsed_sec=float(response.elapsed.total_seconds()),
            usage=data.get("usage") or {},
        )


_log_format = LogFormat.plain


def configure_logging(log_format: LogFormat) -> None:
    global _log_format

    _log_format = log_format
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def log_bad_input(error: str) -> None:
    if _log_format == LogFormat.json:
        sys.stderr.write(json.dumps({"level": "error", "event": "bad_input", "error": error}) + "\n")
        return
    logger.error("bad_input error=%s", error)


def build_chat_payload(request: DirectGenerationRequest) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": request.model,
        "messages": [
            {"role": "system", "content": "You are a precise entity extraction engine. Return JSON only."},
            {"role": "user", "content": request.prompt},
        ],
        "temperature": request.temperature,
        "top_p": request.top_p,
        "max_tokens": request.max_tokens,
    }
    if request.json_mode:
        payload["response_format"] = {"type": "json_object"}
    if request.disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    return payload


def build_direct_prompt(request: DirectDetectionRequest) -> str:
    label_text = (
        _format_label_examples(request.labels)
        if request.prompt_mode == PromptMode.recall
        else ", ".join(request.labels)
    )
    recall_block = _recall_block() if request.prompt_mode == PromptMode.recall else ""
    summary = f"\nData context: {request.data_summary}\n" if request.data_summary else ""
    return f"""Extract privacy-sensitive entities from the input text.
{summary}
Use only these labels:
{label_text}

Rules:
- Return exact substrings from the input text.
- Do not invent values.
- Prefer specific labels from the allowed list.
- Skip generic nouns, syntax, and non-sensitive filler.
{recall_block}- Return only a JSON object with this shape:
  {{"entities": [{{"value": "exact substring", "label": "one_allowed_label", "reason": "short reason"}}]}}

Input text:
---
{request.text}
---"""


def _recall_block() -> str:
    return """- Bias toward high recall. Missing a sensitive value is worse than returning one extra plausible value.
- Include family members, colleagues, employers, schools, institutions, locations, dates, demographics, beliefs, and identifiers when allowed.
"""


def run_direct_detection_case(
    request: DirectDetectionRequest,
    *,
    client: DirectDetectionClient,
    endpoint: str = "http://gpu-dev-pod-serve-svc:8000/v1",
    model: str = "nvidia/nemotron-3-super",
    max_tokens: int = 4096,
    timeout_sec: float = 180.0,
) -> DirectDetectionCase:
    try:
        completion = client.complete(
            DirectGenerationRequest(
                endpoint=endpoint,
                model=model,
                prompt=build_direct_prompt(request),
                max_tokens=max_tokens,
                timeout_sec=timeout_sec,
            )
        )
        suggestions = _extract_suggestions(completion.content)
        allowed_suggestions = filter_direct_suggestions(suggestions, request.labels)
        artifact = finalize_direct_suggestions(
            text=request.text,
            suggestions=allowed_suggestions,
            labels=request.labels,
            row_index=request.row_index,
            workflow_name="direct-detection",
        )
        return _completed_case(request, completion, suggestions, allowed_suggestions, artifact)
    except Exception as exc:  # noqa: BLE001 - benchmark probe records per-case failures
        return DirectDetectionCase(
            case_id=request.case_id,
            row_index=request.row_index,
            status=CaseStatus.error,
            error=f"{type(exc).__name__}: {exc}",
        )


def _extract_suggestions(content: str) -> list[dict[str, Any]]:
    payload = _load_embedded_json(content)
    if not isinstance(payload, dict):
        return []
    suggestions = payload.get("entities")
    return suggestions if isinstance(suggestions, list) else []


def finalize_direct_suggestions(
    *,
    text: str,
    suggestions: list[dict[str, Any]],
    labels: list[str],
    row_index: int,
    workflow_name: str,
) -> DetectionArtifactRow:
    cleaned = filter_direct_suggestions(suggestions, labels)
    direct_spans = apply_augmented_entities(text=text, entities=[], augmented_output={"entities": cleaned})
    entities = [
        _span_to_entity_schema(span, source="direct_llm") for span in expand_entity_occurrences(text, direct_spans)
    ]
    return build_detection_artifact_row_from_entities(
        workflow_name=workflow_name,
        batch_file="direct-detection",
        row_index=row_index,
        seed_entities=[],
        seed_validation_candidate_count=0,
        merged_validation_candidate_count=0,
        augmented_entities=entities,
        final_entities=entities,
    )


def filter_direct_suggestions(suggestions: list[dict[str, Any]], labels: list[str]) -> list[dict[str, str]]:
    allowed = set(labels)
    cleaned = [
        {"value": str(item.get("value", "")).strip(), "label": str(item.get("label", "")).strip()}
        for item in suggestions
        if isinstance(item, dict)
    ]
    return [item for item in cleaned if item["value"] and item["label"] in allowed]


def _span_to_entity_schema(span: EntitySpan, *, source: str) -> EntitySchema:
    span_source = source if span.source == "augmenter" else span.source
    return EntitySchema(
        value=span.value,
        label=span.label,
        start_position=span.start_position,
        end_position=span.end_position,
        score=span.score,
        source=span_source,
    )


def _completed_case(
    request: DirectDetectionRequest,
    completion: DirectCompletion,
    suggestions: list[dict[str, Any]],
    allowed_suggestions: list[dict[str, str]],
    artifact: DetectionArtifactRow,
) -> DirectDetectionCase:
    return DirectDetectionCase(
        case_id=request.case_id,
        row_index=request.row_index,
        status=CaseStatus.completed,
        elapsed_sec=completion.elapsed_sec,
        usage=completion.usage,
        raw_suggestion_count=len(suggestions),
        allowed_suggestion_count=len(allowed_suggestions),
        final_entity_count=artifact.final_entity_count,
        final_entity_signature_count=artifact.final_entity_signature_count,
        final_entity_signature_hashes=artifact.final_entity_signature_hashes,
        final_label_counts=artifact.final_label_counts,
        artifact=artifact,
    )


def compare_signature_sets(
    *,
    baseline_hashes: set[str],
    baseline_labels: dict[str, str],
    direct_hashes: set[str],
    direct_labels: dict[str, str],
) -> SignatureComparison:
    baseline_only = baseline_hashes - direct_hashes
    direct_only = direct_hashes - baseline_hashes
    return SignatureComparison(
        baseline_final_entity_signature_count=len(baseline_hashes),
        shared_final_entity_signature_count=len(baseline_hashes & direct_hashes),
        baseline_only_final_entity_signature_count=len(baseline_only),
        direct_only_final_entity_signature_count=len(direct_only),
        baseline_only_label_counts=_label_counts(baseline_only, baseline_labels),
        direct_only_label_counts=_label_counts(direct_only, direct_labels),
    )


def _label_counts(hashes: set[str], labels: dict[str, str]) -> dict[str, int]:
    return dict(sorted(Counter(labels.get(item, "unknown") for item in hashes).items()))


def apply_baseline_comparisons(
    cases: list[DirectDetectionCase],
    baseline_artifacts: Path,
) -> list[DirectDetectionCase]:
    baseline = _read_baseline_artifacts(baseline_artifacts)
    compared: list[DirectDetectionCase] = []
    for case in cases:
        if case.status != CaseStatus.completed or case.artifact is None:
            compared.append(case)
            continue
        baseline_row = baseline.get(case.row_index)
        if baseline_row is None:
            compared.append(case)
            continue
        compared.append(_case_with_comparison(case, baseline_row))
    return compared


def _case_with_comparison(case: DirectDetectionCase, baseline_row: dict[str, Any]) -> DirectDetectionCase:
    baseline_hashes = _baseline_signature_hashes(baseline_row)
    if baseline_hashes is None:
        return case
    comparison = compare_signature_sets(
        baseline_hashes=baseline_hashes,
        baseline_labels=_signature_labels(baseline_row),
        direct_hashes=set(case.artifact.final_entity_signature_hashes if case.artifact else []),
        direct_labels=case.artifact.final_entity_signature_labels if case.artifact else {},
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
                raise ValueError(
                    f"baseline artifacts has multiple rows for row_index={row_index}; "
                    "pass a per-case sidecar or pre-filter the artifact file"
                )
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
    limit: int = 1,
    offset: int = 0,
    prompt_mode: PromptMode = PromptMode.compact,
    baseline_artifacts: Path | None = None,
) -> DirectDetectionRun:
    requests = _load_requests(
        input_path, text_column=text_column, labels=labels, limit=limit, offset=offset, prompt_mode=prompt_mode
    )
    client = HttpxDirectDetectionClient()
    cases = [run_direct_detection_case(request, client=client, endpoint=endpoint, model=model) for request in requests]
    if baseline_artifacts is not None:
        cases = apply_baseline_comparisons(cases, baseline_artifacts)
    result = DirectDetectionRun(
        input_path=str(input_path),
        text_column=text_column,
        endpoint=endpoint,
        model=model,
        prompt_mode=prompt_mode,
        labels=labels,
        rows=cases,
    )
    if output is not None:
        write_outputs(result, output, overwrite=overwrite)
    return result


def _load_requests(
    input_path: Path,
    *,
    text_column: str,
    labels: list[str],
    limit: int,
    offset: int,
    prompt_mode: PromptMode,
) -> list[DirectDetectionRequest]:
    dataframe = pd.read_csv(input_path)
    if text_column not in dataframe.columns:
        raise ValueError(f"text column {text_column!r} not found in {input_path}")
    selected = dataframe.iloc[offset : offset + limit]
    return [
        DirectDetectionRequest(
            case_id=f"{input_path.stem}-row-{int(index)}",
            text=str(row[text_column]),
            labels=labels,
            row_index=int(index),
            prompt_mode=prompt_mode,
        )
        for index, row in selected.iterrows()
    ]


def write_outputs(result: DirectDetectionRun, output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise ValueError(f"output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    _write_jsonl(output_dir / "direct-detection-cases.jsonl", [_case_payload(case) for case in result.rows])
    _write_jsonl(output_dir / "direct-detection-artifacts.jsonl", [_artifact_payload(case) for case in result.rows])
    (output_dir / "summary.json").write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")


def _case_payload(case: DirectDetectionCase) -> dict[str, Any]:
    payload = case.model_dump(exclude={"artifact"})
    payload["record_type"] = "direct_detection_case"
    return payload


def _artifact_payload(case: DirectDetectionCase) -> dict[str, Any]:
    payload = case.artifact.model_dump() if case.artifact is not None else {}
    payload.update({"case_id": case.case_id, "row_index": case.row_index, "record_type": "direct_detection_artifact"})
    return payload


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def render_result(result: DirectDetectionRun, *, json_output: bool) -> str:
    if json_output:
        return result.model_dump_json(indent=2)
    completed = len(result.rows) - result.error_count
    return f"Ran {completed}/{len(result.rows)} direct detection case(s); errors={result.error_count}"


def parse_labels(raw: str) -> list[str]:
    return [label.strip() for label in raw.split(",") if label.strip()]


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
    limit: Annotated[int, cyclopts.Parameter("--limit")] = 1,
    offset: Annotated[int, cyclopts.Parameter("--offset")] = 0,
    prompt_mode: Annotated[PromptMode, cyclopts.Parameter("--prompt-mode")] = PromptMode.compact,
    baseline_artifacts: Annotated[Path | None, cyclopts.Parameter("--baseline-artifacts")] = None,
    json_output: Annotated[bool, cyclopts.Parameter("--json")] = False,
    log_format: Annotated[LogFormat, cyclopts.Parameter("--log-format")] = LogFormat.plain,
) -> None:
    configure_logging(log_format)
    try:
        result = run_probe(
            input_path,
            text_column=text_column,
            labels=parse_labels(labels),
            output=output,
            overwrite=overwrite,
            endpoint=endpoint,
            model=model,
            limit=limit,
            offset=offset,
            prompt_mode=prompt_mode,
            baseline_artifacts=baseline_artifacts,
        )
    except (ValueError, ValidationError, httpx.HTTPError) as exc:
        log_bad_input(str(exc))
        raise SystemExit(125) from exc
    sys.stdout.write(render_result(result, json_output=json_output) + "\n")
    if result.error_count:
        raise SystemExit(1)


if __name__ == "__main__":
    app()
