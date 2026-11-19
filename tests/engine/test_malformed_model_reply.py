# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A malformed model reply through the real DataDesigner run and model facade.

Only the HTTP client is scripted: DataDesigner builds the facades, runs the columns and records failed rows.
"""

from __future__ import annotations

import json
from collections import defaultdict
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

from anonymizer.config.anonymizer_config import Rewrite
from anonymizer.config.models import DetectionModelSelection, RewriteModelSelection
from anonymizer.config.rewrite import PrivacyGoal
from anonymizer.engine.constants import (
    COL_ANY_HIGH_LEAKED,
    COL_AUGMENTED_ENTITIES,
    COL_DETECTED_ENTITIES,
    COL_DOMAIN,
    COL_DOMAIN_SUPPLEMENT,
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
    COL_REWRITE_BASELINE_TEXT,
    COL_REWRITE_REPLACEMENT_READY,
    COL_REWRITTEN_TEXT,
    COL_REWRITTEN_TEXT_INITIAL,
    COL_SENSITIVITY_DISPOSITION,
    COL_TEXT,
    COL_UTILITY_SCORE,
    COL_WEIGHTED_LEAKAGE_RATE,
)
from anonymizer.engine.detection.detection_workflow import EntityDetectionWorkflow
from anonymizer.engine.ndd.adapter import NddAdapter
from anonymizer.engine.ndd.model_loader import load_default_model_selection
from anonymizer.engine.rewrite import combined_rewrite_workflow as combined
from anonymizer.engine.rewrite.domain_classification import _DOMAIN_BY_ENUM, DomainClassificationWorkflow
from anonymizer.engine.rewrite.evaluate import EvaluateWorkflow
from anonymizer.engine.rewrite.qa_generation import QAGenerationWorkflow
from anonymizer.engine.schemas import Domain

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

    def _reply(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        if _OTHER_TEXT in json.dumps(request.messages, default=str):
            content = _OTHER_REPLIES.get(self._alias, _GOOD_QA)
            return ChatCompletionResponse(message=AssistantMessage(content=content))
        replies = self._script[self._alias]
        index = self._calls[self._alias]
        self._calls[self._alias] += 1
        _MESSAGE_COUNTS[self._alias].append(len(request.messages))
        return ChatCompletionResponse(message=AssistantMessage(content=replies[min(index, len(replies) - 1)]))

    def completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        return self._reply(request)

    async def acompletion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        return self._reply(request)

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


def test_quality_qa_reply_malformed_on_every_attempt_keeps_the_row_without_quality_questions(
    tmp_path: Path, scripted: tuple[dict[str, list[str]], dict[str, int]]
) -> None:
    script, calls = scripted
    script["scripted-qa"] = [_MALFORMED_QA]

    result, alias = _quality_qa(tmp_path)

    assert result.failed_records == []
    assert len(result.dataframe) == 2
    assert pd.isna(result.dataframe.iloc[0][COL_QUALITY_QA])
    assert calls[alias] == 6


# --- Rewrite evaluation: a reply that never parses keeps the row with its evaluation unavailable ---

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
_QUALITY_REPLY = '{"answers": [{"id": 1, "answer": "A card."}]}'
_COMPARE_REPLY = '{"per_item": [{"id": 1, "score": 0.8, "reason": "Close."}]}'


def _privacy_reply(reason: str) -> str:
    return json.dumps({"answers": [{"id": 1, "answer": "no", "confidence": 0.9, "reason": reason, "evidence": []}]})


def _ask_kind(request: ChatCompletionRequest) -> str:
    prompt = json.dumps(request.messages, default=str)
    if "privacy auditor" in prompt:
        return "privacy"
    return "compare" if "A student took" in prompt else "quality"


class _EvaluatorClient(_ScriptedClient):
    """Replies per evaluate call (``quality``, ``privacy``, ``compare``) for the record under test, else well formed."""

    def _reply(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        kind = _ask_kind(request)
        if _OTHER_TEXT in json.dumps(request.messages, default=str):
            good = {"quality": _QUALITY_REPLY, "privacy": _privacy_reply("Absent."), "compare": _COMPARE_REPLY}
            return ChatCompletionResponse(message=AssistantMessage(content=good[kind]))
        replies = self._script.get(kind) or [_QUALITY_REPLY if kind == "quality" else _COMPARE_REPLY]
        index = self._calls[kind]
        self._calls[kind] += 1
        _MESSAGE_COUNTS[kind].append(len(request.messages))
        return ChatCompletionResponse(message=AssistantMessage(content=replies[min(index, len(replies) - 1)]))


@pytest.fixture
def evaluator(monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, list[str]], dict[str, int]]:
    script: dict[str, list[str]] = {"privacy": [_privacy_reply("Absent.")]}
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


def test_privacy_reply_unparsed_on_every_attempt_keeps_the_row_without_a_privacy_verdict(
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


def test_quality_reply_unparsed_skips_the_comparison_and_leaves_utility_unavailable(
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
def test_privacy_reason_outside_its_limits_does_not_reject_the_reply(
    tmp_path: Path, evaluator: tuple[dict[str, list[str]], dict[str, int]], reason: str, kept: str
) -> None:
    script, calls = evaluator
    script["privacy"] = [_privacy_reply(reason)]

    result = _evaluate(tmp_path)

    assert result.failed_records == []
    assert calls["privacy"] == 1
    assert result.dataframe.iloc[0][COL_PRIVACY_QA_REANSWER]["answers"][0]["reason"] == kept


# --- Rewrite pipeline: a reply that never parses keeps the row with the site's unavailable value ---

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


def test_domain_reply_unparsed_on_every_attempt_keeps_the_row_as_other(
    tmp_path: Path, scripted: tuple[dict[str, list[str]], dict[str, int]]
) -> None:
    script, calls = scripted
    script[_DOMAIN] = ["I cannot answer that."]

    result = _domain(tmp_path)

    assert result.failed_records == []
    assert len(result.dataframe) == 2
    row = result.dataframe.iloc[0]
    assert row[COL_DOMAIN]["domain"] == "OTHER"
    assert row[COL_DOMAIN_SUPPLEMENT] == _DOMAIN_BY_ENUM[Domain.OTHER].quality_supplement
    assert calls[_DOMAIN] == 6  # the run config's budget (DataDesigner default: 5 restarts)


def test_domain_reply_uses_the_run_config_restarts_and_corrections(
    tmp_path: Path, scripted: tuple[dict[str, list[str]], dict[str, int]]
) -> None:
    script, calls = scripted
    script[_DOMAIN] = ["I cannot answer that."]

    result = _domain(tmp_path, RunConfig(max_conversation_restarts=1, max_conversation_correction_steps=2))

    assert result.failed_records == []
    first = _MESSAGE_COUNTS[_DOMAIN][0]
    assert _MESSAGE_COUNTS[_DOMAIN] == [first, first + 2, first + 4] * 2
    assert calls[_DOMAIN] == 6


def test_domain_reply_that_parses_is_kept_as_returned(
    tmp_path: Path, scripted: tuple[dict[str, list[str]], dict[str, int]]
) -> None:
    script, calls = scripted
    script[_DOMAIN] = ["I cannot answer that.", '{"domain": "FINANCIAL", "domain_confidence": 0.9}']

    result = _domain(tmp_path)

    assert result.failed_records == []
    assert result.dataframe.iloc[0][COL_DOMAIN] == {"domain": "FINANCIAL", "domain_confidence": 0.9}
    assert calls[_DOMAIN] == 2


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


def test_meaning_units_reply_unparsed_keeps_the_row_and_skips_the_quality_questions(
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
        combined._finalization_column([state], evaluation),
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
    assert row[COL_NEEDS_HUMAN_REVIEW]  # unavailable utility is flagged
    assert calls["quality"] == 0
