# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A malformed model reply through the real DataDesigner run and model facade.

Only the HTTP client is scripted: DataDesigner builds the facades, runs the columns and records failed rows.
"""

from __future__ import annotations

import contextlib
import json
import logging
from collections import defaultdict
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pandas as pd
import pytest
from data_designer.config.column_configs import LLMStructuredColumnConfig
from data_designer.config.models import ModelConfig, ModelProvider
from data_designer.config.run_config import RunConfig
from data_designer.engine.models.clients import factory as client_factory
from data_designer.engine.models.clients.types import AssistantMessage, ChatCompletionRequest, ChatCompletionResponse
from data_designer.interface.data_designer import DataDesigner

from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput, Rewrite
from anonymizer.config.models import DetectionModelSelection, RewriteModelSelection
from anonymizer.config.rewrite import PrivacyGoal
from anonymizer.engine.constants import (
    COL_ANY_HIGH_LEAKED,
    COL_AUGMENTED_ENTITIES,
    COL_DETECTED_ENTITIES,
    COL_DOMAIN,
    COL_DOMAIN_SUPPLEMENT,
    COL_DOMAIN_UNAVAILABLE,
    COL_ENTITIES_BY_VALUE,
    COL_ENTITY_CLASSIFICATION,
    COL_FULL_REWRITE,
    COL_LATENT_ENTITIES,
    COL_LEAKAGE_MASS,
    COL_MEANING_UNITS,
    COL_MEANING_UNITS_SERIALIZED,
    COL_NEEDS_HUMAN_REVIEW,
    COL_NEEDS_REPAIR,
    COL_PRIVACY_QA,
    COL_PRIVACY_QA_REANSWER,
    COL_QUALITY_QA,
    COL_QUALITY_QA_COMPARE,
    COL_QUALITY_QA_REANSWER,
    COL_RAW_SENSITIVITY_DISPOSITION,
    COL_REPLACEMENT_MAP_FOR_PROMPT,
    COL_REWRITE_BASELINE_TEXT,
    COL_REWRITE_DISPOSITION_BLOCK,
    COL_REWRITE_REPLACEMENT_READY,
    COL_REWRITE_TAGGED_TEXT,
    COL_REWRITTEN_TEXT,
    COL_REWRITTEN_TEXT_INITIAL,
    COL_REWRITTEN_TEXT_NEXT,
    COL_SENSITIVITY_DISPOSITION,
    COL_TAG_NOTATION,
    COL_TAGGED_TEXT,
    COL_TEXT,
    COL_UTILITY_SCORE,
    COL_WEIGHTED_LEAKAGE_RATE,
)
from anonymizer.engine.detection.detection_workflow import EntityDetectionResult, EntityDetectionWorkflow
from anonymizer.engine.ndd.adapter import NddAdapter
from anonymizer.engine.ndd.model_loader import load_default_model_selection
from anonymizer.engine.rewrite import combined_rewrite_workflow as combined
from anonymizer.engine.rewrite.combined_rewrite_workflow import CombinedRewriteWorkflow
from anonymizer.engine.rewrite.domain_classification import _DOMAIN_BY_ENUM, DomainClassificationWorkflow
from anonymizer.engine.rewrite.evaluate import EvaluateWorkflow
from anonymizer.engine.rewrite.qa_generation import QAGenerationWorkflow
from anonymizer.engine.rewrite.rewrite_generation import RewriteGenerationWorkflow
from anonymizer.engine.rewrite.rewrite_workflow import RewriteWorkflow
from anonymizer.engine.rewrite.sensitivity_disposition import SensitivityDispositionWorkflow
from anonymizer.engine.schemas import Domain
from anonymizer.interface.anonymizer import Anonymizer

_PROVIDER = ModelProvider(name="scripted", endpoint="http://scripted.invalid/v1", api_key="unused")
_DETECTOR = "scripted-detector"
_VALIDATOR = "scripted-validator"
_AUGMENTER = "scripted-augmenter"
_TEXT = "Card ending in 7312 will be charged."
# A second, clean record keeps the run alive when the first record fails (DataDesigner raises when no record survives).
_OTHER_TEXT = "Nothing to see."
_OTHER_REPLIES = {
    "scripted-detector": '{"entities": []}',
    "scripted-augmenter": '```json\n{"entities": []}\n```',
    "scripted-domain": '{"domain": "META_TEXT", "domain_confidence": 0.5}',
    "scripted-rewriter": '{"rewritten_text": "Nothing to see."}',
    "scripted-meaning": '{"units": [{"id": 1, "aspect": "Topic", "unit": "Nothing to see.", "importance": "important"}]}',
}

# The 11-14 reply shape: malformed JSON at context_after, which the parser's repair turns into an item without label.
_MALFORMED_AUGMENTER = (
    '```json\n{"entities": [{"value": "7312", "context_before": "ending in ", '
    '"context_after". will be charged", "label": "card_last_four"}]}\n```'
)
_GOOD_AUGMENTER = (
    '```json\n{"entities": [{"value": "7312", "context_before": "ending in ", '
    '"context_after": " will be charged", "label": "card_last_four"}]}\n```'
)
_MALFORMED_QA = (
    '```json\n{"items": [{"id": 1, "aspect": "Charge", "importance": "important", "question": "What?"}]}\n```'
)
_GOOD_QA = (
    '```json\n{"items": [{"id": 1, "aspect": "Charge", "importance": "important", '
    '"question": "What is charged?", "reference_answer": "The card."}]}\n```'
)


_MESSAGE_COUNTS: dict[str, list[int]] = defaultdict(list)


class _ScriptedClient:
    """Replies per alias from a script for the record under test; the last reply repeats once the script runs out.

    Requests for the clean record get a fixed well-formed reply and are not counted.
    """

    def __init__(self, alias: str, script: dict[str, list[str]], calls: dict[str, int]) -> None:
        self._alias = alias
        self._script = script
        self._calls = calls

    def _response(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        if _OTHER_TEXT in json.dumps(request.messages, default=str):
            content = _OTHER_REPLIES.get(self._alias, _GOOD_QA)
            return ChatCompletionResponse(message=AssistantMessage(content=content))
        replies = self._script[self._alias]
        index = self._calls[self._alias]
        self._calls[self._alias] += 1
        _MESSAGE_COUNTS[self._alias].append(len(request.messages))
        return ChatCompletionResponse(message=AssistantMessage(content=replies[min(index, len(replies) - 1)]))

    def completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        return self._response(request)

    async def acompletion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        return self._response(request)

    def supports_chat_completion(self) -> bool:
        return True

    def close(self) -> None:
        return None

    async def aclose(self) -> None:
        return None

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, list[str]], dict[str, int]]:
    script: dict[str, list[str]] = {}
    calls: dict[str, int] = defaultdict(int)
    _MESSAGE_COUNTS.clear()

    def create_model_client(model_config: ModelConfig, *args: Any, **kwargs: Any) -> _ScriptedClient:
        return _ScriptedClient(model_config.alias, script, calls)

    monkeypatch.setattr(client_factory, "create_model_client", create_model_client)
    return script, calls


def _adapter(tmp_path: Path, run_config: RunConfig | None = None) -> NddAdapter:
    data_designer = DataDesigner(artifact_path=tmp_path, model_providers=[_PROVIDER])
    if run_config is not None:
        data_designer.set_run_config(run_config)
    return NddAdapter(data_designer=data_designer)


def _model(alias: str) -> ModelConfig:
    return ModelConfig(alias=alias, model=alias, provider=_PROVIDER.name, skip_health_check=True)


def _detect(tmp_path: Path, run_config: RunConfig | None = None):
    selection = DetectionModelSelection(
        entity_detector=_DETECTOR,
        entity_validator=[_VALIDATOR],
        entity_augmenter=_AUGMENTER,
        latent_detector=_AUGMENTER,
    )
    return EntityDetectionWorkflow(adapter=_adapter(tmp_path, run_config)).detect_and_validate_entities(
        pd.DataFrame({COL_TEXT: [_TEXT, _OTHER_TEXT]}),
        model_configs=[_model(_DETECTOR), _model(_VALIDATOR), _model(_AUGMENTER)],
        selected_models=selection,
        gliner_detection_threshold=0.3,
    )


def test_augmenter_retries_a_malformed_reply_instead_of_dropping_the_record(
    tmp_path: Path, scripted: tuple[dict[str, list[str]], dict[str, int]]
) -> None:
    script, calls = scripted
    script[_DETECTOR] = [json.dumps({"entities": []})]
    script[_AUGMENTER] = [_MALFORMED_AUGMENTER, _GOOD_AUGMENTER]

    result = _detect(tmp_path)

    assert result.failed_records == []
    assert calls[_AUGMENTER] == 2
    assert len(result.dataframe) == 2
    entities = result.dataframe[COL_DETECTED_ENTITIES].iloc[0]["entities"]
    assert [(e["value"], e["label"]) for e in entities] == [("7312", "card_last_four")]
    assert COL_AUGMENTED_ENTITIES in result.dataframe.columns


def test_augmenter_reply_malformed_on_every_attempt_still_fails_the_record(
    tmp_path: Path, scripted: tuple[dict[str, list[str]], dict[str, int]]
) -> None:
    script, calls = scripted
    script[_DETECTOR] = [json.dumps({"entities": []})]
    script[_AUGMENTER] = [_MALFORMED_AUGMENTER]

    result = _detect(tmp_path)

    assert len(result.dataframe) == 1
    assert [record.step for record in result.failed_records] == ["entity-detection"]
    # One grounding attempt, given the run config's restarts (DataDesigner default 5).
    assert calls[_AUGMENTER] == 6


def test_augmenter_uses_the_run_config_restarts_and_corrections(
    tmp_path: Path, scripted: tuple[dict[str, list[str]], dict[str, int]]
) -> None:
    script, calls = scripted
    script[_DETECTOR] = [json.dumps({"entities": []})]
    script[_AUGMENTER] = [_MALFORMED_AUGMENTER]

    result = _detect(tmp_path, RunConfig(max_conversation_restarts=1, max_conversation_correction_steps=2))

    assert [record.step for record in result.failed_records] == ["entity-detection"]
    # Two conversations (1 restart), each a first request plus 2 corrections that grow the conversation.
    first = _MESSAGE_COUNTS[_AUGMENTER][0]
    assert _MESSAGE_COUNTS[_AUGMENTER] == [first, first + 2, first + 4] * 2
    assert calls[_AUGMENTER] == 6


def _quality_qa(tmp_path: Path):
    alias = "scripted-qa"
    selection = RewriteModelSelection.model_validate(
        {**load_default_model_selection().rewrite.model_dump(), "qa_generator": alias}
    )
    (column,) = [
        c
        for c in QAGenerationWorkflow().columns(selected_models=selection)
        if isinstance(c, LLMStructuredColumnConfig) and c.name == COL_QUALITY_QA
    ]
    return _adapter(tmp_path).run_workflow(
        pd.DataFrame(
            {COL_TEXT: [_TEXT, _OTHER_TEXT], COL_MEANING_UNITS_SERIALIZED: ["1. A card is charged.", _OTHER_TEXT]}
        ),
        model_configs=[_model(alias)],
        columns=[column],
        workflow_name="rewrite-pipeline",
    ), alias


def test_quality_qa_already_retries_a_malformed_reply(
    tmp_path: Path, scripted: tuple[dict[str, list[str]], dict[str, int]]
) -> None:
    script, calls = scripted
    script["scripted-qa"] = [_MALFORMED_QA, _GOOD_QA]

    result, alias = _quality_qa(tmp_path)

    assert result.failed_records == []
    assert len(result.dataframe) == 2
    assert calls[alias] == 2


def test_quality_qa_response_malformed_on_every_attempt_keeps_the_row_without_quality_questions(
    tmp_path: Path, scripted: tuple[dict[str, list[str]], dict[str, int]]
) -> None:
    script, calls = scripted
    script["scripted-qa"] = [_MALFORMED_QA]

    result, alias = _quality_qa(tmp_path)

    assert result.failed_records == []
    assert len(result.dataframe) == 2
    assert pd.isna(result.dataframe.iloc[0][COL_QUALITY_QA])
    assert calls[alias] == 6


# --- Rewrite evaluation: a response that never parses keeps the row with its evaluation unavailable ---

_EVALUATOR = "scripted-evaluator"
_EVAL_TEXT = "A card will be charged."
_EVAL_QUALITY_QA = {
    "items": [
        {"id": 1, "aspect": "Charge", "importance": "important", "question": "What?", "reference_answer": "Card."}
    ]
}
_EVAL_PRIVACY_QA = {
    "items": [
        {
            "id": 1,
            "question": "Is the card number 7312?",
            "sensitivity": "high",
            "entity_label": "card_last_four",
            "entity_value": "7312",
            "category": "direct_identifier",
        }
    ]
}
_PRIVACY_GOAL = PrivacyGoal(protect="Card numbers and other direct identifiers", preserve="That a charge happens")
_QUALITY_RESPONSE = '{"answers": [{"id": 1, "answer": "A card."}]}'
_COMPARE_RESPONSE = '{"per_item": [{"id": 1, "score": 0.8, "reason": "Close."}]}'


def _privacy_response(reason: str) -> str:
    return json.dumps({"answers": [{"id": 1, "answer": "no", "confidence": 0.9, "reason": reason, "evidence": []}]})


def _ask_kind(request: ChatCompletionRequest) -> str:
    prompt = json.dumps(request.messages, default=str)
    if "privacy auditor" in prompt:
        return "privacy"
    return "compare" if "A student took" in prompt else "quality"


class _EvaluatorClient(_ScriptedClient):
    """Responses per evaluate call (``quality``, ``privacy``, ``compare``) for the record under test, else well formed."""

    def _response(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        kind = _ask_kind(request)
        if _OTHER_TEXT in json.dumps(request.messages, default=str):
            good = {"quality": _QUALITY_RESPONSE, "privacy": _privacy_response("Absent."), "compare": _COMPARE_RESPONSE}
            return ChatCompletionResponse(message=AssistantMessage(content=good[kind]))
        responses = self._script.get(kind) or [_QUALITY_RESPONSE if kind == "quality" else _COMPARE_RESPONSE]
        index = self._calls[kind]
        self._calls[kind] += 1
        _MESSAGE_COUNTS[kind].append(len(request.messages))
        return ChatCompletionResponse(message=AssistantMessage(content=responses[min(index, len(responses) - 1)]))


@pytest.fixture
def evaluator(monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, list[str]], dict[str, int]]:
    script: dict[str, list[str]] = {"privacy": [_privacy_response("Absent.")]}
    calls: dict[str, int] = defaultdict(int)
    _MESSAGE_COUNTS.clear()

    def create_model_client(model_config: ModelConfig, *args: Any, **kwargs: Any) -> _EvaluatorClient:
        return _EvaluatorClient(model_config.alias, script, calls)

    monkeypatch.setattr(client_factory, "create_model_client", create_model_client)
    return script, calls


def _evaluate(tmp_path: Path) -> Any:
    selection = RewriteModelSelection.model_validate(
        {**load_default_model_selection().rewrite.model_dump(), "evaluator": _EVALUATOR}
    )
    evaluation = Rewrite(privacy_goal=_PRIVACY_GOAL).evaluation
    other_quality = {"items": [{**_EVAL_QUALITY_QA["items"][0], "reference_answer": _OTHER_TEXT}]}
    return _adapter(tmp_path).run_workflow(
        pd.DataFrame(
            {
                COL_REWRITTEN_TEXT: [_EVAL_TEXT, _OTHER_TEXT],
                COL_QUALITY_QA: [_EVAL_QUALITY_QA, other_quality],
                COL_PRIVACY_QA: [_EVAL_PRIVACY_QA, _EVAL_PRIVACY_QA],
            }
        ),
        model_configs=[_model(_EVALUATOR)],
        columns=EvaluateWorkflow(adapter=Mock()).columns(selected_models=selection, evaluation=evaluation),
        workflow_name="rewrite-evaluate-0",
    )


def test_privacy_response_unparsed_on_every_attempt_keeps_the_row_without_a_privacy_verdict(
    tmp_path: Path, evaluator: tuple[dict[str, list[str]], dict[str, int]]
) -> None:
    script, calls = evaluator
    script["privacy"] = ["I cannot answer that."]

    result = _evaluate(tmp_path)

    assert result.failed_records == []
    row = result.dataframe.iloc[0]
    assert pd.isna(row[COL_PRIVACY_QA_REANSWER])
    assert all(pd.isna(row[c]) for c in (COL_LEAKAGE_MASS, COL_WEIGHTED_LEAKAGE_RATE, COL_ANY_HIGH_LEAKED))
    assert row[COL_UTILITY_SCORE] == pytest.approx(0.8)
    assert not row[COL_NEEDS_REPAIR]  # unavailable evidence goes to review, not repair


def test_evaluate_calls_restart_once_after_their_corrections(
    tmp_path: Path, evaluator: tuple[dict[str, list[str]], dict[str, int]]
) -> None:
    script, calls = evaluator
    script["privacy"] = ["I cannot answer that."]

    _evaluate(tmp_path)

    # Two conversations, each a first request plus 3 corrections that grow the conversation.
    first = _MESSAGE_COUNTS["privacy"][0]
    assert _MESSAGE_COUNTS["privacy"] == [first, first + 2, first + 4, first + 6] * 2
    assert calls["privacy"] == 8


def test_quality_response_unparsed_skips_the_comparison_and_leaves_utility_unavailable(
    tmp_path: Path, evaluator: tuple[dict[str, list[str]], dict[str, int]]
) -> None:
    script, calls = evaluator
    script["quality"] = ["I cannot answer that."]

    result = _evaluate(tmp_path)

    assert result.failed_records == []
    row = result.dataframe.iloc[0]
    assert pd.isna(row[COL_QUALITY_QA_REANSWER])
    assert pd.isna(row[COL_QUALITY_QA_COMPARE])
    assert pd.isna(row[COL_UTILITY_SCORE])
    assert row[COL_LEAKAGE_MASS] == 0.0
    assert calls["compare"] == 0


@pytest.mark.parametrize(
    ("reason", "kept"),
    [("x" * 201, "x" * 200), ("", "Model returned an empty reason.")],
    ids=["too-long", "empty"],
)
def test_privacy_reason_outside_its_limits_does_not_reject_the_response(
    tmp_path: Path, evaluator: tuple[dict[str, list[str]], dict[str, int]], reason: str, kept: str
) -> None:
    script, calls = evaluator
    script["privacy"] = [_privacy_response(reason)]

    result = _evaluate(tmp_path)

    assert result.failed_records == []
    assert calls["privacy"] == 1
    assert result.dataframe.iloc[0][COL_PRIVACY_QA_REANSWER]["answers"][0]["reason"] == kept


# --- Rewrite pipeline: a response that never parses keeps the row with the site's unavailable value ---

_DOMAIN = "scripted-domain"


def _domain(tmp_path: Path, run_config: RunConfig | None = None):
    selection = RewriteModelSelection.model_validate(
        {**load_default_model_selection().rewrite.model_dump(), "domain_classifier": _DOMAIN}
    )
    return _adapter(tmp_path, run_config).run_workflow(
        pd.DataFrame({COL_TEXT: [_TEXT, _OTHER_TEXT]}),
        model_configs=[_model(_DOMAIN)],
        columns=DomainClassificationWorkflow().columns(selected_models=selection),
        workflow_name="rewrite-pipeline",
    )


def test_domain_response_unparsed_on_every_attempt_keeps_the_row_as_other(
    tmp_path: Path, scripted: tuple[dict[str, list[str]], dict[str, int]]
) -> None:
    script, calls = scripted
    script[_DOMAIN] = ["I cannot answer that."]

    result = _domain(tmp_path)

    assert result.failed_records == []
    assert len(result.dataframe) == 2
    row = result.dataframe.iloc[0]
    assert row[COL_DOMAIN]["domain"] == "OTHER"
    assert row[COL_DOMAIN_UNAVAILABLE]
    assert row[COL_DOMAIN_SUPPLEMENT] == _DOMAIN_BY_ENUM[Domain.OTHER].quality_supplement
    assert calls[_DOMAIN] == 6  # the run config's budget (DataDesigner default: 5 restarts)


def test_domain_response_uses_the_run_config_restarts_and_corrections(
    tmp_path: Path, scripted: tuple[dict[str, list[str]], dict[str, int]]
) -> None:
    script, calls = scripted
    script[_DOMAIN] = ["I cannot answer that."]

    result = _domain(tmp_path, RunConfig(max_conversation_restarts=1, max_conversation_correction_steps=2))

    assert result.failed_records == []
    first = _MESSAGE_COUNTS[_DOMAIN][0]
    assert _MESSAGE_COUNTS[_DOMAIN] == [first, first + 2, first + 4] * 2
    assert calls[_DOMAIN] == 6


def test_domain_response_that_parses_is_kept_as_returned(
    tmp_path: Path, scripted: tuple[dict[str, list[str]], dict[str, int]]
) -> None:
    script, calls = scripted
    script[_DOMAIN] = ["I cannot answer that.", '{"domain": "FINANCIAL", "domain_confidence": 0.9}']

    result = _domain(tmp_path)

    assert result.failed_records == []
    assert result.dataframe.iloc[0][COL_DOMAIN] == {"domain": "FINANCIAL", "domain_confidence": 0.9}
    assert not result.dataframe.iloc[0][COL_DOMAIN_UNAVAILABLE]
    assert calls[_DOMAIN] == 2


def test_domain_response_of_other_at_zero_confidence_is_a_response_not_the_fallback(
    tmp_path: Path, scripted: tuple[dict[str, list[str]], dict[str, int]]
) -> None:
    script, calls = scripted
    script[_DOMAIN] = ['{"domain": "OTHER", "domain_confidence": 0.0}']

    result = _domain(tmp_path)

    row = result.dataframe.iloc[0]
    assert row[COL_DOMAIN] == {"domain": "OTHER", "domain_confidence": 0.0}
    assert not row[COL_DOMAIN_UNAVAILABLE]


_MEANING = "scripted-meaning"
_DISPOSITION = {
    "sensitivity_disposition": [
        {
            "id": 1,
            "source": "tagged",
            "category": "direct_identifier",
            "sensitivity": "high",
            "entity_label": "card_last_four",
            "entity_value": "7312",
            "protection_method_suggestion": "replace",
        }
    ]
}


def _qa_pipeline(tmp_path: Path):
    selection = RewriteModelSelection.model_validate(
        {
            **load_default_model_selection().rewrite.model_dump(),
            "meaning_extractor": _MEANING,
            "qa_generator": "scripted-qa",
        }
    )
    domain = {"domain": "OTHER", "domain_confidence": 0.5}
    return _adapter(tmp_path).run_workflow(
        pd.DataFrame(
            {
                COL_TEXT: [_TEXT, _OTHER_TEXT],
                COL_DOMAIN: [domain, domain],
                COL_DOMAIN_SUPPLEMENT: ["Keep the charge.", "Keep it."],
                COL_SENSITIVITY_DISPOSITION: [_DISPOSITION, _DISPOSITION],
            }
        ),
        model_configs=[_model(_MEANING), _model("scripted-qa")],
        columns=QAGenerationWorkflow().columns(selected_models=selection),
        workflow_name="rewrite-pipeline",
    )


def test_meaning_units_response_unparsed_keeps_the_row_and_skips_the_quality_questions(
    tmp_path: Path, scripted: tuple[dict[str, list[str]], dict[str, int]]
) -> None:
    script, calls = scripted
    script[_MEANING] = ["I cannot answer that."]
    script["scripted-qa"] = [_GOOD_QA]

    result = _qa_pipeline(tmp_path)

    assert result.failed_records == []
    row = result.dataframe.iloc[0]
    assert pd.isna(row[COL_MEANING_UNITS])
    assert pd.isna(row[COL_QUALITY_QA])
    assert calls[_MEANING] == 6
    assert calls["scripted-qa"] == 0  # not asked about missing meaning units
    assert row[COL_PRIVACY_QA]["items"][0]["entity_value"] == "7312"  # privacy questions do not depend on them
    assert result.dataframe.iloc[1][COL_QUALITY_QA]["items"]


def _evaluate_rows(tmp_path: Path, quality_qa: Any, rewritten_text: Any = _EVAL_TEXT) -> Any:
    selection = RewriteModelSelection.model_validate(
        {**load_default_model_selection().rewrite.model_dump(), "evaluator": _EVALUATOR}
    )
    evaluation = Rewrite(privacy_goal=_PRIVACY_GOAL).evaluation
    other_quality = {"items": [{**_EVAL_QUALITY_QA["items"][0], "reference_answer": _OTHER_TEXT}]}
    return _adapter(tmp_path).run_workflow(
        pd.DataFrame(
            {
                COL_REWRITTEN_TEXT: [rewritten_text, _OTHER_TEXT],
                COL_QUALITY_QA: [quality_qa, other_quality],
                COL_PRIVACY_QA: [_EVAL_PRIVACY_QA, _EVAL_PRIVACY_QA],
            }
        ),
        model_configs=[_model(_EVALUATOR)],
        columns=EvaluateWorkflow(adapter=Mock()).columns(selected_models=selection, evaluation=evaluation),
        workflow_name="rewrite-evaluate-0",
    )


def test_evaluation_without_quality_questions_leaves_utility_unavailable_and_still_scores_privacy(
    tmp_path: Path, evaluator: tuple[dict[str, list[str]], dict[str, int]]
) -> None:
    script, calls = evaluator

    result = _evaluate_rows(tmp_path, quality_qa=None)

    assert result.failed_records == []
    row = result.dataframe.iloc[0]
    assert pd.isna(row[COL_QUALITY_QA_REANSWER])
    assert pd.isna(row[COL_UTILITY_SCORE])
    assert row[COL_LEAKAGE_MASS] == 0.0
    assert calls["quality"] == 0
    assert calls["compare"] == 0
    assert calls["privacy"] == 1
    assert result.dataframe.iloc[1][COL_UTILITY_SCORE] == pytest.approx(0.8)


def _combined(tmp_path: Path, quality_qa: Any, rewritten_text: Any = _EVAL_TEXT) -> pd.DataFrame:
    """Combined mode's evaluation and finalize columns in one real DataDesigner graph, as CombinedRewriteWorkflow runs them."""
    selection = RewriteModelSelection.model_validate(
        {**load_default_model_selection().rewrite.model_dump(), "evaluator": _EVALUATOR, "repairer": _EVALUATOR}
    )
    evaluation = Rewrite(privacy_goal=_PRIVACY_GOAL, max_repair_iterations=0).evaluation
    state = combined._evaluation_state(0, COL_REWRITTEN_TEXT_INITIAL)
    columns = [
        *combined._evaluation_columns(Mock(), selected_models=selection, evaluation=evaluation, state=state),
        combined._finalization_column([state]),
    ]
    other_quality = {"items": [{**_EVAL_QUALITY_QA["items"][0], "reference_answer": _OTHER_TEXT}]}
    result = _adapter(tmp_path).run_workflow(
        pd.DataFrame(
            {
                COL_TEXT: [_TEXT, _OTHER_TEXT],
                COL_REWRITTEN_TEXT_INITIAL: [rewritten_text, _OTHER_TEXT],
                COL_REWRITE_BASELINE_TEXT: [_EVAL_TEXT, _OTHER_TEXT],
                COL_QUALITY_QA: [quality_qa, other_quality],
                COL_PRIVACY_QA: [_EVAL_PRIVACY_QA, _EVAL_PRIVACY_QA],
                COL_REWRITE_REPLACEMENT_READY: [True, True],
            }
        ),
        model_configs=[_model(_EVALUATOR)],
        columns=columns,
        workflow_name="rewrite-combined",
    )
    assert result.failed_records == []
    return result.dataframe


def test_combined_evaluation_without_quality_questions_keeps_the_privacy_metrics(
    tmp_path: Path, evaluator: tuple[dict[str, list[str]], dict[str, int]]
) -> None:
    script, calls = evaluator

    frame = _combined(tmp_path, quality_qa=None)

    row = frame.iloc[0]
    assert row[COL_REWRITTEN_TEXT] == _EVAL_TEXT
    assert pd.isna(row[COL_UTILITY_SCORE])
    assert row[COL_LEAKAGE_MASS] == 0.0
    assert calls["quality"] == 0


_REWRITER = "scripted-rewriter"


def _rewrite(tmp_path: Path, disposition_block: Any = "[]"):
    selection = RewriteModelSelection.model_validate(
        {**load_default_model_selection().rewrite.model_dump(), "rewriter": _REWRITER}
    )
    columns = [
        column
        for column in RewriteGenerationWorkflow().columns(selected_models=selection, privacy_goal=_PRIVACY_GOAL)
        if column.name in (COL_FULL_REWRITE, COL_REWRITTEN_TEXT)
    ]
    return _adapter(tmp_path).run_workflow(
        pd.DataFrame(
            {
                COL_TAG_NOTATION: ["xml", "xml"],
                COL_REWRITE_TAGGED_TEXT: [_TEXT, _OTHER_TEXT],
                COL_REPLACEMENT_MAP_FOR_PROMPT: ['{"replacements": []}', '{"replacements": []}'],
                COL_REWRITE_DISPOSITION_BLOCK: [disposition_block, "[]"],
                COL_REWRITE_REPLACEMENT_READY: [True, True],
            }
        ),
        model_configs=[_model(_REWRITER)],
        columns=columns,
        workflow_name="rewrite-pipeline",
    )


def test_rewrite_response_unparsed_on_every_attempt_keeps_the_row_without_a_rewrite(
    tmp_path: Path, scripted: tuple[dict[str, list[str]], dict[str, int]]
) -> None:
    script, calls = scripted
    script[_REWRITER] = ["Here is your rewrite: a card is charged."]

    result = _rewrite(tmp_path)

    assert result.failed_records == []
    row = result.dataframe.iloc[0]
    assert pd.isna(row[COL_FULL_REWRITE])
    assert pd.isna(row[COL_REWRITTEN_TEXT])
    assert calls[_REWRITER] == 6
    assert result.dataframe.iloc[1][COL_REWRITTEN_TEXT] == _OTHER_TEXT


def test_evaluation_of_a_missing_rewrite_asks_nothing_and_leaves_every_metric_unavailable(
    tmp_path: Path, evaluator: tuple[dict[str, list[str]], dict[str, int]]
) -> None:
    script, calls = evaluator

    result = _evaluate_rows(tmp_path, quality_qa=_EVAL_QUALITY_QA, rewritten_text=None)

    assert result.failed_records == []
    row = result.dataframe.iloc[0]
    assert all(pd.isna(row[c]) for c in (COL_QUALITY_QA_REANSWER, COL_PRIVACY_QA_REANSWER, COL_QUALITY_QA_COMPARE))
    assert all(pd.isna(row[c]) for c in (COL_UTILITY_SCORE, COL_LEAKAGE_MASS, COL_ANY_HIGH_LEAKED))
    assert not row[COL_NEEDS_REPAIR]
    assert dict(calls) == {}  # no request renders a missing text as "None"
    assert result.dataframe.iloc[1][COL_UTILITY_SCORE] == pytest.approx(0.8)


def test_combined_evaluation_of_a_missing_rewrite_keeps_the_row_withheld(
    tmp_path: Path, evaluator: tuple[dict[str, list[str]], dict[str, int]]
) -> None:
    script, calls = evaluator

    frame = _combined(tmp_path, quality_qa=_EVAL_QUALITY_QA, rewritten_text=None)

    row = frame.iloc[0]
    assert pd.isna(row[COL_REWRITTEN_TEXT])
    assert pd.isna(row[COL_LEAKAGE_MASS])
    assert dict(calls) == {}
    assert frame.iloc[1][COL_REWRITTEN_TEXT] == _OTHER_TEXT


# --- Repair: a response that never parses keeps the row, withholds its rewrite and flags it ---

_REPAIRER = "scripted-repairer"
_REPAIRED_TEXT = "A charge will happen."
_LEAK = json.dumps(
    {"answers": [{"id": 1, "answer": "yes", "confidence": 0.9, "reason": "Stated.", "evidence": ["7312"]}]}
)


class _RepairClient(_EvaluatorClient):
    """The repairer answers from ``script["repair"]``; every other alias answers as the evaluator does."""

    def _response(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        if self._alias != _REPAIRER:
            return super()._response(request)
        responses = self._script["repair"]
        index = self._calls["repair"]
        self._calls["repair"] += 1
        return ChatCompletionResponse(message=AssistantMessage(content=responses[min(index, len(responses) - 1)]))


@pytest.fixture
def repairer(monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, list[str]], dict[str, int]]:
    script: dict[str, list[str]] = {"privacy": [_LEAK]}
    calls: dict[str, int] = defaultdict(int)
    _MESSAGE_COUNTS.clear()

    def create_model_client(model_config: ModelConfig, *args: Any, **kwargs: Any) -> _RepairClient:
        return _RepairClient(model_config.alias, script, calls)

    monkeypatch.setattr(client_factory, "create_model_client", create_model_client)
    return script, calls


class _StagedLoop(RewriteWorkflow):
    """Staged mode with only its evaluate-repair loop, over rows that already hold a rewrite."""

    def _produce(self, rows: Any, **options: Any) -> Any:
        options = {**options, **_scripted_models()}
        stages = self._evaluate_repair_stages(
            rows,
            model_configs=options["model_configs"],
            selected_models=options["selected_models"],
            privacy_goal=options["privacy_goal"],
            evaluation=options["evaluation"],
            preview_num_records=None,
        )
        return lambda: [stage() for _, stage in stages]


class _CombinedLoop(CombinedRewriteWorkflow):
    """Combined mode's evaluation, repair and finalize columns in one graph, over rows that already hold a rewrite."""

    def _produce(self, rows: Any, **options: Any) -> Any:
        options = {**options, **_scripted_models()}
        selection, evaluation = options["selected_models"], options["evaluation"]
        states = [combined._evaluation_state(0, COL_REWRITTEN_TEXT_INITIAL)]
        columns = combined._evaluation_columns(
            Mock(), selected_models=selection, evaluation=evaluation, state=states[0]
        )
        internal = [COL_REWRITTEN_TEXT_INITIAL]
        for iteration in range(evaluation.max_repair_iterations):
            repair = combined._repair_state(iteration)
            columns += combined._repair_columns(
                Mock(),
                selected_models=selection,
                privacy_goal=options["privacy_goal"],
                evaluation=evaluation,
                previous=states[-1],
                state=repair,
            )
            states.append(combined._evaluation_state(iteration + 1, repair.rewritten_text))
            columns += combined._evaluation_columns(
                Mock(), selected_models=selection, evaluation=evaluation, state=states[-1]
            )
            internal += [c for c in repair.__dict__.values() if isinstance(c, str)]
        columns.append(combined._finalization_column(states))
        internal += [c for s in states for c in s.__dict__.values() if isinstance(c, str)]

        def generate() -> None:
            seed = rows.frame.rename(columns={COL_REWRITTEN_TEXT: COL_REWRITTEN_TEXT_INITIAL})
            result = self._adapter.run_workflow(
                seed, model_configs=options["model_configs"], columns=columns, workflow_name="rewrite-combined"
            )
            rows.failed.extend(result.failed_records)
            rows.frame = result.dataframe.drop(columns=list(dict.fromkeys(internal)), errors="ignore")

        return generate


def _scripted_models() -> dict[str, Any]:
    """The evaluator and repairer models the loops run with, whatever models their caller passes."""
    selection = RewriteModelSelection.model_validate(
        {**load_default_model_selection().rewrite.model_dump(), "evaluator": _EVALUATOR, "repairer": _REPAIRER}
    )
    return {"model_configs": [_model(_EVALUATOR), _model(_REPAIRER)], "selected_models": selection}


def _seeded_rows(rewritten_text: str | None = _EVAL_TEXT) -> pd.DataFrame:
    """The record under test and a clean record, each with a rewrite and the columns its evaluation reads."""
    other_quality = {"items": [{**_EVAL_QUALITY_QA["items"][0], "reference_answer": _OTHER_TEXT}]}
    entities = {"entities_by_value": [{"value": "7312", "labels": ["card_last_four"]}]}
    return pd.DataFrame(
        {
            COL_TEXT: [_TEXT, _OTHER_TEXT],
            COL_ENTITIES_BY_VALUE: [entities, entities],
            COL_REWRITTEN_TEXT: [rewritten_text, _OTHER_TEXT],
            COL_REWRITE_BASELINE_TEXT: [_EVAL_TEXT, _OTHER_TEXT],
            COL_QUALITY_QA: [_EVAL_QUALITY_QA, other_quality],
            COL_PRIVACY_QA: [_EVAL_PRIVACY_QA, _EVAL_PRIVACY_QA],
            COL_SENSITIVITY_DISPOSITION: [{"sensitivity_disposition": []}] * 2,
            COL_REWRITE_REPLACEMENT_READY: [True, True],
        }
    )


def _repair_run(tmp_path: Path, workflow: type[RewriteWorkflow], max_repair_iterations: int) -> Any:
    frame = _seeded_rows()
    detected = EntityDetectionResult(dataframe=frame, failed_records=[], source_positions=(0, 1))
    return workflow(adapter=_adapter(tmp_path)).run_with_detection(
        frame[[COL_TEXT]],
        detect=lambda _: detected,
        replace_model_selection=load_default_model_selection().replace,
        privacy_goal=_PRIVACY_GOAL,
        evaluation=Rewrite(privacy_goal=_PRIVACY_GOAL, max_repair_iterations=max_repair_iterations).evaluation,
        **_scripted_models(),
    )


def _anonymizer_run(
    tmp_path: Path, workflow: type[RewriteWorkflow], max_repair_iterations: int, rewritten_text: str | None = _EVAL_TEXT
) -> Any:
    """Run both records through ``Anonymizer.run()``: detection returns the seeded rows, and the loop evaluates."""
    seeded = _seeded_rows(rewritten_text).drop(columns=[COL_TEXT])
    detector = Mock(spec=EntityDetectionWorkflow)
    detector.run.side_effect = lambda frame, **_: EntityDetectionResult(
        dataframe=frame.assign(**{column: seeded[column].tolist() for column in seeded.columns}),
        failed_records=[],
        source_positions=tuple(range(len(frame))),
    )
    csv_path = tmp_path / "input.csv"
    pd.DataFrame({"text": [_TEXT, _OTHER_TEXT]}).to_csv(csv_path, index=False)
    runner = workflow(adapter=_adapter(tmp_path))
    combined_graph = isinstance(runner, CombinedRewriteWorkflow)
    anonymizer = (
        Anonymizer(detection_workflow=detector, combined_rewrite_runner=runner)
        if combined_graph
        else Anonymizer(detection_workflow=detector, rewrite_runner=runner)
    )
    rewrite = Rewrite(
        privacy_goal=_PRIVACY_GOAL, max_repair_iterations=max_repair_iterations, use_combined_graph=combined_graph
    )
    return anonymizer.run(
        config=AnonymizerConfig(rewrite=rewrite, emit_telemetry=False),
        data=AnonymizerInput(source=str(csv_path), text_column="text"),
    )


@pytest.mark.parametrize("workflow", [_StagedLoop, _CombinedLoop], ids=["staged", "combined"])
@pytest.mark.parametrize(
    ("responses", "repair_calls"),
    [
        (["I cannot rewrite that."], 4),  # the first repair never parses
        ([json.dumps({"rewritten_text": _REPAIRED_TEXT}), "I cannot rewrite that."], 5),  # the second one
    ],
    ids=["first-repair", "later-repair"],
)
def test_repair_response_unparsed_keeps_the_row_withheld_and_flagged(
    tmp_path: Path,
    repairer: tuple[dict[str, list[str]], dict[str, int]],
    caplog: pytest.LogCaptureFixture,
    workflow: type[RewriteWorkflow],
    responses: list[str],
    repair_calls: int,
) -> None:
    script, calls = repairer
    script["repair"] = responses

    with caplog.at_level(logging.WARNING, logger="anonymizer.rewrite.responses"):
        result = _repair_run(tmp_path, workflow, max_repair_iterations=3)

    assert result.failed_records == []
    assert len(result.dataframe) == 2
    row, clean = result.dataframe.iloc[0], result.dataframe.iloc[1]
    assert pd.isna(row[COL_REWRITTEN_TEXT])  # neither the failing candidate nor the repaired one is released
    assert pd.isna(row[COL_LEAKAGE_MASS])
    assert row[COL_NEEDS_HUMAN_REVIEW]
    outcome = result.outcomes[0]
    assert outcome.qualification == "not_met"
    assert not outcome.available
    assert not outcome.protected
    assert calls["repair"] == repair_calls  # a first request and 3 corrections, then no further round
    assert clean[COL_REWRITTEN_TEXT] == _OTHER_TEXT
    assert not clean[COL_NEEDS_HUMAN_REVIEW]
    assert result.outcomes[1].qualification == "met"
    # One record, which caplog can list twice once DataDesigner has reconfigured logging in an earlier test.
    summaries = dict.fromkeys(r for r in caplog.records if r.name == "anonymizer.rewrite.responses")
    assert [record.getMessage() for record in summaries] == [
        "Rows kept without a usable model response after their retries, by step: repair=1. "
        "If most rows are listed, check that step's model."
    ]


def test_rewrite_of_a_row_without_a_disposition_asks_nothing_and_keeps_the_row(
    tmp_path: Path, scripted: tuple[dict[str, list[str]], dict[str, int]]
) -> None:
    script, calls = scripted
    script[_REWRITER] = ['{"rewritten_text": "A card is charged."}']

    result = _rewrite(tmp_path, disposition_block=None)

    assert result.failed_records == []
    assert pd.isna(result.dataframe.iloc[0][COL_REWRITTEN_TEXT])
    assert calls[_REWRITER] == 0


_DISPOSER = "scripted-disposition"


def _disposition_response(count: int) -> str:
    entry = {"sensitivity": "high", "protection_method_suggestion": "replace", "low_sensitivity_reason": None}
    return json.dumps({"sensitivity_disposition": [{"id": i, **entry} for i in range(1, count + 1)]})


def _disposition(tmp_path: Path):
    """Two supplied entities for the record under test; the clean record has none and the latent list only."""
    selection = RewriteModelSelection.model_validate(
        {**load_default_model_selection().rewrite.model_dump(), "disposition_analyzer": _DISPOSER}
    )
    _OTHER_REPLIES[_DISPOSER] = _disposition_response(1)
    rows = [
        (_TEXT, [{"value": "7312", "labels": ["card_last_four"]}, {"value": "charged", "labels": ["event"]}]),
        (_OTHER_TEXT, [{"value": "Nothing", "labels": ["event"]}]),
    ]
    return _adapter(tmp_path).run_workflow(
        pd.DataFrame(
            {
                COL_TAGGED_TEXT: [text for text, _ in rows],
                COL_ENTITIES_BY_VALUE: [{"entities_by_value": entities} for _, entities in rows],
                COL_LATENT_ENTITIES: [{"latent_entities": []}] * 2,
                COL_ENTITY_CLASSIFICATION: [{"card_last_four": "direct_identifier", "event": "quasi_identifier"}] * 2,
            }
        ),
        model_configs=[_model(_DISPOSER)],
        columns=SensitivityDispositionWorkflow().columns(selected_models=selection, privacy_goal=_PRIVACY_GOAL),
        workflow_name="rewrite-pipeline",
    )


def test_disposition_response_with_surplus_entries_is_asked_again(
    tmp_path: Path, scripted: tuple[dict[str, list[str]], dict[str, int]]
) -> None:
    script, calls = scripted
    script[_DISPOSER] = [
        _disposition_response(4),
        _disposition_response(2),
    ]  # one entry per occurrence, then per entity

    result = _disposition(tmp_path)

    assert result.failed_records == []
    entries = result.dataframe.iloc[0][COL_SENSITIVITY_DISPOSITION]["sensitivity_disposition"]
    assert [(e["id"], e["entity_value"]) for e in entries] == [(1, "7312"), (2, "charged")]
    assert calls[_DISPOSER] == 2


def test_disposition_response_with_surplus_entries_on_every_attempt_keeps_the_row_without_a_disposition(
    tmp_path: Path, scripted: tuple[dict[str, list[str]], dict[str, int]]
) -> None:
    script, calls = scripted
    script[_DISPOSER] = [_disposition_response(4)]

    result = _disposition(tmp_path)

    assert result.failed_records == []
    row = result.dataframe.iloc[0]
    assert pd.isna(row[COL_RAW_SENSITIVITY_DISPOSITION])
    assert pd.isna(row[COL_SENSITIVITY_DISPOSITION])
    assert calls[_DISPOSER] == 6
    assert len(result.dataframe.iloc[1][COL_SENSITIVITY_DISPOSITION]["sensitivity_disposition"]) == 1


# --- Evaluation: a step that never responds in a round the repair loop replaces is still counted ---

_UNPARSED = "I cannot answer that."
_RETRIES = 8  # a first request and its corrections and restarts, as the default run config allows


def _per_row_warnings(caplog: pytest.LogCaptureFixture, *columns: str) -> int:
    """The per-row "unavailable after" warnings logged for ``columns`` (in either mode's column names)."""
    messages = [r.getMessage() for r in dict.fromkeys(caplog.records)]
    return sum(1 for m in messages if m.startswith(columns) and " unavailable after " in m)


@contextlib.contextmanager
def _captured(caplog: pytest.LogCaptureFixture) -> Generator[None]:
    """Capture warnings during ``Anonymizer.run()``, which stops the package's records from propagating."""
    package = logging.getLogger("anonymizer")
    package.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING):
            yield
    finally:
        package.removeHandler(caplog.handler)


def _summaries(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage() for r in dict.fromkeys(r for r in caplog.records if r.name == "anonymizer.rewrite.responses")
    ]


@pytest.mark.parametrize("workflow", [_StagedLoop, _CombinedLoop], ids=["staged", "combined"])
@pytest.mark.parametrize("middle", [False, True], ids=["round-0", "middle-round"])
@pytest.mark.parametrize("step", ["quality", "compare"])
def test_quality_evaluation_without_a_response_in_an_earlier_round_is_counted(
    tmp_path: Path,
    repairer: tuple[dict[str, list[str]], dict[str, int]],
    caplog: pytest.LogCaptureFixture,
    workflow: type[RewriteWorkflow],
    step: str,
    middle: bool,
) -> None:
    """The round without a response leaks, so the row is repaired and a later round passes: not the last round.

    In the middle-round case the first evaluation is complete and leaks too.
    """
    script, calls = repairer
    good = _QUALITY_RESPONSE if step == "quality" else _COMPARE_RESPONSE
    script[step] = [*[good] * middle, *[_UNPARSED] * _RETRIES, good]
    script["privacy"] = [*[_LEAK] * (1 + middle), _privacy_response("Absent.")]
    script["repair"] = [json.dumps({"rewritten_text": _REPAIRED_TEXT})]

    with _captured(caplog):
        result = _anonymizer_run(tmp_path, workflow, max_repair_iterations=3)

    assert calls[step] >= _RETRIES + 1  # every attempt of that round was spent
    assert result.outcome(0).qualification == "met"  # the last round passed, so the summary must not rely on it
    per_row = _per_row_warnings(caplog, COL_QUALITY_QA_REANSWER, COL_QUALITY_QA_COMPARE)
    assert per_row == 1
    assert _summaries(caplog) == [
        f"Rows kept without a usable model response after their retries, by step: quality evaluation={per_row}. "
        "If most rows are listed, check that step's model."
    ]


@pytest.mark.parametrize("workflow", [_StagedLoop, _CombinedLoop], ids=["staged", "combined"])
def test_trace_of_a_repair_withheld_for_its_privacy_response(
    tmp_path: Path, repairer: tuple[dict[str, list[str]], dict[str, int]], workflow: type[RewriteWorkflow]
) -> None:
    """Pins what the docs say each mode's trace keeps: staged keeps the repair candidate, combined does not."""
    script, _ = repairer
    script["privacy"] = [_LEAK, *[_UNPARSED] * _RETRIES]
    script["repair"] = [json.dumps({"rewritten_text": _REPAIRED_TEXT})]

    frame = _repair_run(tmp_path, workflow, max_repair_iterations=3).dataframe

    assert pd.isna(frame.iloc[0][COL_REWRITTEN_TEXT])
    if workflow is _StagedLoop:
        assert frame.iloc[0][COL_REWRITTEN_TEXT_NEXT] == _REPAIRED_TEXT
    else:
        assert COL_REWRITTEN_TEXT_NEXT not in frame.columns
        # The production graph drops every round's repair candidate with its internal columns, as the harness does.
        graph = CombinedRewriteWorkflow(adapter=Mock()).build_graph(
            selected_models=load_default_model_selection().rewrite,
            replace_model_selection=load_default_model_selection().replace,
            privacy_goal=_PRIVACY_GOAL,
            evaluation=Rewrite(privacy_goal=_PRIVACY_GOAL, max_repair_iterations=3).evaluation,
        )
        produced = {column.name for column in graph.columns}
        candidates = {c for c in produced if c.startswith(COL_REWRITTEN_TEXT_NEXT)}
        assert len(candidates) == 3
        assert candidates <= set(graph.internal_columns)


# --- Repair: a repair that still fails privacy after its last round is withheld, as an unparsed one is ---


@pytest.mark.parametrize("workflow", [_StagedLoop, _CombinedLoop], ids=["staged", "combined"])
@pytest.mark.parametrize("rounds", [1, 3])
def test_repair_that_still_leaks_after_its_last_round_is_withheld_and_flagged(
    tmp_path: Path,
    repairer: tuple[dict[str, list[str]], dict[str, int]],
    caplog: pytest.LogCaptureFixture,
    workflow: type[RewriteWorkflow],
    rounds: int,
) -> None:
    script, calls = repairer
    script["privacy"] = [_LEAK]  # every evaluation of the record leaks
    script["repair"] = [json.dumps({"rewritten_text": _REPAIRED_TEXT})]

    with _captured(caplog):
        result = _anonymizer_run(tmp_path, workflow, max_repair_iterations=rounds)

    assert calls["repair"] == rounds  # every repair parsed; the loop ran out of rounds
    assert result.failed_records == []
    row, clean = result.dataframe.iloc[0], result.dataframe.iloc[1]
    assert pd.isna(row["text_rewritten"])  # the failing repair candidate is not released
    assert row[COL_NEEDS_HUMAN_REVIEW]
    outcome = result.outcome(0)
    assert outcome.qualification == "not_met"
    assert ("failed", "high_leak") in [(r.code, r.check) for r in outcome.reasons]
    assert clean["text_rewritten"] == _OTHER_TEXT
    assert result.outcome(1).qualification == "met"
    assert _summaries(caplog) == []  # every response parsed, so no step is counted


@pytest.mark.parametrize("workflow", [_StagedLoop, _CombinedLoop], ids=["staged", "combined"])
def test_missing_rewrite_is_withheld_and_flagged_through_run(
    tmp_path: Path, repairer: tuple[dict[str, list[str]], dict[str, int]], workflow: type[RewriteWorkflow]
) -> None:
    """The review flag is assigned after the final checks, so it is asserted on the run's output."""
    result = _anonymizer_run(tmp_path, workflow, max_repair_iterations=0, rewritten_text=None)

    row = result.dataframe.iloc[0]
    assert pd.isna(row["text_rewritten"])
    assert row[COL_NEEDS_HUMAN_REVIEW]
    assert result.outcome(0).qualification == "not_met"
    assert not result.dataframe.iloc[1][COL_NEEDS_HUMAN_REVIEW]
