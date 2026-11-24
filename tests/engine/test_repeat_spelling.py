# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Built-in Replace adds a repeat span only where the detected spelling recurs exactly: a common word that folds to
a detected name ("may" for ``May``) is not edited. Runs ``Anonymizer.run()`` on the built-in route offline; only
DataDesigner's model client is scripted."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from data_designer.config.models import ModelConfig
from data_designer.engine.models.clients import factory as client_factory
from data_designer.engine.models.clients.types import AssistantMessage, ChatCompletionRequest, ChatCompletionResponse

from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput
from anonymizer.config.replace_strategies import Redact, Substitute
from anonymizer.interface.anonymizer import Anonymizer

# text -> the detector's (value, label, start) spans
_DETECTED: dict[str, list[tuple[str, str, int]]] = {
    "May said she may come back in May.": [("May", "first_name", 0), ("May", "first_name", 30)],
    "Will said he will call tomorrow.": [("Will", "first_name", 0)],
    "Mark wants to mark the page.": [("Mark", "first_name", 0)],
    "Bill paid the bill.": [("Bill", "first_name", 0)],
    "We moved to Nice; it is nice there.": [("Nice", "city", 12)],
    "Ông Hàn nói thẻ đã hết hạn.": [("Hàn", "last_name", 4)],
    "Alice met Alice.": [("Alice", "first_name", 0)],
    "Alice met alice.": [("Alice", "first_name", 0)],
}
# Each value's substitute, and one for its folded variant: the map offers both, so only spans decide what is edited.
_SYNTHETIC = {
    "May": ("first_name", "Ruth"),
    "Will": ("first_name", "Owen"),
    "Mark": ("first_name", "Paul"),
    "Bill": ("first_name", "Hugo"),
    "Nice": ("city", "Lyon"),
    "Hàn": ("last_name", "Trần"),
    "Alice": ("first_name", "Carol"),
}


def _text_of(messages: str) -> str | None:
    return next((text for text in _DETECTED if json.dumps(text)[1:-1] in messages or text in messages), None)


def _reply(alias: str, request: ChatCompletionRequest) -> str:
    messages = json.dumps(request.messages, default=str, ensure_ascii=False)
    text = _text_of(messages)
    if alias == "gliner-pii-detector":
        spans = _DETECTED.get(text or "", [])
        return json.dumps(
            {
                "entities": [
                    {"text": v, "label": lab, "start": s, "end": s + len(v), "score": 0.9} for v, lab, s in spans
                ]
            }
        )
    if "decision" in messages and "keep" in messages:
        ids = list(dict.fromkeys(re.findall(r'\\"id\\":\s*\\"([^\\"]+)\\"', messages)))
        decisions = [{"id": i, "decision": "keep", "proposed_label": "", "reason": ""} for i in ids]
        return json.dumps({"decisions": decisions})
    if "context_before" in messages:
        return '{"entities": []}'
    replacements = []
    for value, (label, synthetic) in _SYNTHETIC.items():
        replacements.append({"original": value, "label": label, "synthetic": synthetic})
        replacements.append({"original": value.lower(), "label": label, "synthetic": synthetic.lower()})
    return json.dumps({"replacements": replacements}, ensure_ascii=False)


class _Client:
    def __init__(self, alias: str) -> None:
        self._alias = alias

    def _response(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        return ChatCompletionResponse(message=AssistantMessage(content=_reply(self._alias, request)))

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


@pytest.fixture(autouse=True)
def _scripted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "unused")

    def create_model_client(model_config: ModelConfig, *args: Any, **kwargs: Any) -> _Client:
        return _Client(model_config.alias)

    monkeypatch.setattr(client_factory, "create_model_client", create_model_client)


def _run(tmp_path: Path, text: str, strategy: str) -> Any:
    source = tmp_path / "in.csv"
    pd.DataFrame({"text": [text]}).to_csv(source, index=False)
    replace = Redact() if strategy == "redact" else Substitute()
    result = Anonymizer(artifact_path=tmp_path / "artifacts").run(
        config=AnonymizerConfig(replace=replace), data=AnonymizerInput(source=str(source), text_column="text")
    )
    assert result.failed_records == []
    return result


@pytest.mark.parametrize(
    ("text", "strategy", "expected"),
    [
        (
            "May said she may come back in May.",
            "redact",
            "[REDACTED_FIRST_NAME] said she may come back in [REDACTED_FIRST_NAME].",
        ),
        ("May said she may come back in May.", "substitute", "Ruth said she may come back in Ruth."),
        ("Will said he will call tomorrow.", "redact", "[REDACTED_FIRST_NAME] said he will call tomorrow."),
        ("Mark wants to mark the page.", "redact", "[REDACTED_FIRST_NAME] wants to mark the page."),
        ("Bill paid the bill.", "substitute", "Hugo paid the bill."),
        ("We moved to Nice; it is nice there.", "redact", "We moved to [REDACTED_CITY]; it is nice there."),
        ("Ông Hàn nói thẻ đã hết hạn.", "redact", "Ông [REDACTED_LAST_NAME] nói thẻ đã hết hạn."),
        ("Alice met Alice.", "redact", "[REDACTED_FIRST_NAME] met [REDACTED_FIRST_NAME]."),
        ("Alice met Alice.", "substitute", "Carol met Carol."),
    ],
)
def test_a_repeat_span_needs_the_detected_spelling(tmp_path: Path, text: str, strategy: str, expected: str) -> None:
    """A folded variant of a detected value (a common word, or a word that differs only by accent) gains no repeat
    span and keeps its text; an exact repeat still does."""
    result = _run(tmp_path, text, strategy)

    assert result.dataframe["text_replaced"].iloc[0] == expected
    assert all(
        e["value"] in {v for v, _, _ in _DETECTED[text]} for e in result.dataframe["final_entities"].iloc[0]["entities"]
    )
    assert result.outcome(0).qualification == "met"


@pytest.mark.parametrize("strategy", ["redact", "substitute"])
def test_a_case_variant_is_kept_and_flagged_by_the_folded_audit(tmp_path: Path, strategy: str) -> None:
    """ "alice" after a detected "Alice" is not edited; the folded leftover count still sees it, as an advisory."""
    result = _run(tmp_path, "Alice met alice.", strategy)

    replaced = result.dataframe["text_replaced"].iloc[0]
    assert replaced.endswith(" met alice.")
    assert "Alice" not in replaced
    outcome = result.outcome(0)
    assert outcome.qualification == "met"
    assert [(a.check, a.status) for a in outcome.advisories] == [("no_detected_value_added", "passed")]
    [leftover] = result.leftovers(0)
    assert (leftover.value, leftover.in_source, leftover.detected, leftover.in_output) == ("Alice", 2, 1, 1)
    assert leftover.undetected_kept
