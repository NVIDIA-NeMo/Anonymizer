# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A Rewrite record whose only finding is latent, through ``Anonymizer.run()`` and ``evaluate()``.

DataDesigner runs for real; only the HTTP model client is scripted, answering by the step its prompt belongs to.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
from data_designer.config.models import ModelConfig, ModelProvider
from data_designer.engine.models.clients import factory as client_factory
from data_designer.engine.models.clients.types import AssistantMessage, ChatCompletionRequest, ChatCompletionResponse

from anonymizer import Anonymizer
from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput, EvaluateConfig, Rewrite
from anonymizer.config.rewrite import PrivacyGoal
from anonymizer.engine.constants import (
    COL_DETECTION_INVALID_ENTITIES,
    COL_DETECTION_VALID,
    COL_PRIVACY_QA,
    COL_SENSITIVITY_DISPOSITION,
    RECORD_ID_COLUMN,
)
from anonymizer.engine.detection.detection_workflow import EntityDetectionWorkflow
from anonymizer.engine.detection.result import DetectorResult, DetectorRow
from anonymizer.engine.schemas import LatentEntitySchema

_LATENT_TEXT = "She works night shifts on the cardiac ward."
_PLAIN_TEXT = "The sky is blue today."
_REWRITTEN = "She works night shifts."
_GOAL = PrivacyGoal(protect="The person's occupation and employer", preserve="The pattern of the work shifts")
_NURSE = LatentEntitySchema(
    category="latent_identifier",
    label="occupation",
    value="nurse",
    confidence="high",
    evidence=["night shifts on the cardiac ward"],
    rationale="Night shifts on a cardiac ward imply a nursing job.",
)

# The step a request belongs to, by a phrase of its prompt; the first match wins.
_KINDS = (
    ("LATENT ENTITY & INFERENCE ANALYSIS", "latent"),
    ("Classify the text into ONE primary domain", "domain"),
    ("Create a sensitivity disposition", "disposition"),
    ("IMPORTANT MEANING UNITS", "meaning"),
    ("QUALITY-CHECK QUESTIONS", "quality_qa"),
    ("You are an expert writer", "rewriter"),
    ("privacy auditor", "privacy"),
    ("A student took", "compare"),
    ("reading comprehension exam", "quality"),
    ("helping to rewrite text for privacy", "repair"),
    ("expert judge evaluating", "judge"),
    ("Generate synthetic replacements", "replacement"),
    ("privacy-entity span extractor", "coverage"),
    ("Hello!", "health"),
)  # anything else is a step of the built-in detector, which finds nothing


def _disposition(sensitivity: str = "high", method: str = "suppress_inference") -> str:
    reason = "Not tied to the goal." if sensitivity == "low" else None
    entry = {"id": 1, "sensitivity": sensitivity, "protection_method_suggestion": method}
    return json.dumps({"sensitivity_disposition": [{**entry, "low_sensitivity_reason": reason}]})


def _privacy(answer: str) -> str:
    item = {"id": 1, "answer": answer, "confidence": 0.9, "reason": "Read the text.", "evidence": []}
    return json.dumps({"answers": [item]})


_JUDGE = json.dumps({name: {"reasoning": "Fine.", "score": "high"} for name in ("privacy", "quality", "style")})
_REPLIES = {
    "latent": json.dumps({"latent_entities": []}),
    "domain": '{"domain": "META_TEXT", "domain_confidence": 0.5}',
    "disposition": _disposition(),
    "meaning": '{"units": [{"id": 1, "aspect": "Work", "unit": "She works night shifts.", "importance": "important"}]}',
    "quality_qa": json.dumps(
        {
            "items": [
                {
                    "id": 1,
                    "aspect": "Work",
                    "importance": "important",
                    "question": "When does she work?",
                    "reference_answer": "At night.",
                }
            ]
        }
    ),
    "rewriter": json.dumps({"rewritten_text": _REWRITTEN}),
    "privacy": _privacy("no"),
    "compare": '{"per_item": [{"id": 1, "score": 0.9, "reason": "Close."}]}',
    "quality": '{"answers": [{"id": 1, "answer": "At night."}]}',
    "repair": json.dumps({"rewritten_text": _REWRITTEN}),
    "judge": _JUDGE,
    "replacement": '{"replacements": []}',
    "health": "{}",
    "coverage": '{"candidate_entities": []}',
    "detector": '{"entities": []}',
}


class _Client:
    """Answers each request from ``replies`` by its step; a request about the first record is counted, and is answered
    from ``"<step>@first"`` when the script has that key."""

    def __init__(self, replies: dict[str, str], calls: Counter[str], prompts: dict[str, list[str]]) -> None:
        self._replies, self._calls, self._prompts = replies, calls, prompts

    def completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        prompt = json.dumps(request.messages, default=str)
        kind = next((k for marker, k in _KINDS if marker in prompt), "detector")
        reply = self._replies[kind]
        if _LATENT_TEXT in prompt or "nurse" in prompt:
            self._calls[kind] += 1
            self._prompts.setdefault(kind, []).append(prompt)
            reply = self._replies.get(f"{kind}@first", reply)
        return ChatCompletionResponse(message=AssistantMessage(content=reply))

    async def acompletion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        return self.completion(request)

    def supports_chat_completion(self) -> bool:
        return True

    def close(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


@pytest.fixture
def models(monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, str], Counter[str], dict[str, list[str]]]:
    replies, calls, prompts = dict(_REPLIES), Counter[str](), {}

    def create_model_client(model_config: ModelConfig, *args: Any, **kwargs: Any) -> _Client:
        return _Client(replies, calls, prompts)

    monkeypatch.setattr(client_factory, "create_model_client", create_model_client)
    return replies, calls, prompts


class _Detector:
    """A custom detector: the first record has only a latent entity, the other records have nothing."""

    def __init__(self, latent: Callable[[int], tuple[LatentEntitySchema, ...]]) -> None:
        self._latent = latent

    def run(self, dataframe: pd.DataFrame, **_: object) -> DetectorResult:
        ids = dataframe[RECORD_ID_COLUMN].tolist()
        return DetectorResult(rows=tuple(DetectorRow(record_id=r, latent=self._latent(i)) for i, r in enumerate(ids)))


_PROVIDERS = [
    ModelProvider(name=n, endpoint="http://scripted.invalid/v1", api_key="k") for n in ("openrouter", "local-gliner2")
]


def _anonymizer(tmp_path: Path, detector: Any = None) -> Anonymizer:
    detector = detector or _Detector(lambda i: (_NURSE,) if i == 0 else ())
    detector = cast(EntityDetectionWorkflow, detector)  # a custom detector answers ``run`` with a DetectorResult
    return Anonymizer(model_providers=_PROVIDERS, artifact_path=tmp_path / "artifacts", detection_workflow=detector)


def _config(mode: str, **rewrite: Any) -> AnonymizerConfig:
    return AnonymizerConfig(
        rewrite=Rewrite(privacy_goal=_GOAL, use_combined_graph=mode == "combined", **rewrite), emit_telemetry=False
    )


def _run(anonymizer: Anonymizer, tmp_path: Path, mode: str, **rewrite: Any) -> Any:
    path = tmp_path / "input.csv"
    pd.DataFrame({"text": [_LATENT_TEXT, _PLAIN_TEXT]}).to_csv(path, index=False)
    return anonymizer.run(config=_config(mode, **rewrite), data=AnonymizerInput(source=str(path), text_column="text"))


_MODES = pytest.mark.parametrize("mode", ["staged", "combined"])


@_MODES
def test_a_latent_only_record_is_rewritten_under_the_privacy_goal(tmp_path: Path, models: Any, mode: str) -> None:
    _, calls, prompts = models

    result = _run(_anonymizer(tmp_path), tmp_path, mode)

    assert result.failed_records == []
    assert result.dataframe["text_rewritten"].tolist() == [_REWRITTEN, _PLAIN_TEXT]
    assert calls["disposition"] == 1
    assert "(none)" in prompts["disposition"][0]  # no explicit entity; the latent one is listed after it
    (entry,) = result.trace_dataframe.iloc[0][COL_SENSITIVITY_DISPOSITION]["sensitivity_disposition"]
    assert (entry["source"], entry["entity_value"]) == ("latent", "nurse")
    (question,) = result.trace_dataframe.iloc[0][COL_PRIVACY_QA]["items"]
    assert question["entity_value"] == "nurse"
    assert calls["rewriter"] == 1
    assert result.outcome(0).qualification == "met"
    assert result.outcome(0).available


@_MODES
def test_a_latent_only_record_that_still_leaks_is_withheld(tmp_path: Path, models: Any, mode: str) -> None:
    replies, calls, _ = models
    replies["privacy"] = _privacy("yes")

    result = _run(_anonymizer(tmp_path), tmp_path, mode, max_repair_iterations=0)

    assert result.failed_records == []
    row = result.dataframe.iloc[0]
    assert pd.isna(row["text_rewritten"])
    assert bool(row["any_high_leaked"])
    assert bool(row["needs_human_review"])
    outcome = result.outcome(0)
    assert outcome.qualification == "not_met"
    assert not outcome.available
    assert ("failed", "high_leak") in {(r.code, r.check) for r in outcome.reasons}
    assert calls["privacy"] >= 1


@_MODES
def test_a_record_with_no_finding_passes_through_without_a_model_request(
    tmp_path: Path, models: Any, mode: str
) -> None:
    _, calls, _ = models

    result = _run(_anonymizer(tmp_path, _Detector(lambda i: ())), tmp_path, mode)

    assert result.dataframe["text_rewritten"].tolist() == [_LATENT_TEXT, _PLAIN_TEXT]
    assert result.outcome(0).qualification == "met"
    assert sum(calls.values()) == 0


def test_combined_mode_asks_no_replacement_model_for_a_latent_only_record(tmp_path: Path, models: Any) -> None:
    _, calls, _ = models

    result = _run(_anonymizer(tmp_path), tmp_path, "combined")

    assert result.dataframe["text_rewritten"].tolist() == [_REWRITTEN, _PLAIN_TEXT]
    assert calls["rewriter"] == 1
    assert calls["replacement"] == 0


@_MODES
def test_a_record_whose_latent_detection_fails_is_neither_rewritten_nor_released(
    tmp_path: Path, models: Any, mode: str
) -> None:
    """The built-in detector: the latent step's reply never parses for the first record, so that record fails."""
    replies, calls, _ = models
    replies["latent@first"] = "I cannot answer that."
    anonymizer = Anonymizer(model_providers=_PROVIDERS, artifact_path=tmp_path / "artifacts")

    result = _run(anonymizer, tmp_path, mode)

    assert [f.step for f in result.failed_records] == ["latent-entity-detection"]
    assert result.dataframe["text"].tolist() == [_PLAIN_TEXT]
    assert result.dataframe["text_rewritten"].tolist() == [_PLAIN_TEXT]
    assert calls["latent"] > 1  # asked again, then given up
    assert calls["disposition"] == calls["rewriter"] == 0


@_MODES
def test_evaluate_judges_a_latent_only_record(tmp_path: Path, models: Any, mode: str) -> None:
    _, calls, _ = models
    anonymizer = _anonymizer(tmp_path)
    result = _run(anonymizer, tmp_path, mode)

    evaluated = anonymizer.evaluate(result, config=EvaluateConfig(compute_detection_validity=True))

    assert calls["judge"] == 1
    trace = evaluated.trace_dataframe.iloc[0]
    assert trace[COL_DETECTION_VALID] == 1.0  # the detection judge reads explicit entities only; none to judge
    assert trace[COL_DETECTION_INVALID_ENTITIES] == []
    advisories = {a.verifier: a.status for a in evaluated.outcome(0).advisories}
    assert advisories.get("rewrite-judge") == "scored"


@_MODES
def test_a_failed_judge_on_a_latent_only_record_reads_unavailable(tmp_path: Path, models: Any, mode: str) -> None:
    replies, _, _ = models
    anonymizer = _anonymizer(tmp_path)
    result = _run(anonymizer, tmp_path, mode)
    replies["judge@first"] = "I cannot answer that."

    evaluated = anonymizer.evaluate(result)

    advisories = {a.verifier: (a.status, a.cause) for a in evaluated.outcome(0).advisories}
    assert advisories.get("rewrite-judge") == ("could_not_assess", "judge_unavailable")
