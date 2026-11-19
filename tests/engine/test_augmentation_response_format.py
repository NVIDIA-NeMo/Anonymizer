# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema-constrained augmenter requests: when ``response_format`` is sent, and the fallback when it is rejected."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pandas as pd
import pytest
import yaml
from data_designer.config.column_types import ColumnConfigT
from data_designer.config.models import ChatCompletionInferenceParams, ModelConfig
from data_designer.config.run_config import RunConfig
from data_designer.engine.models.clients.errors import ProviderError, map_http_status_to_provider_error_kind
from data_designer.engine.models.errors import (
    ModelBadRequestError,
    ModelContextWindowExceededError,
    ModelUnprocessableEntityError,
    handle_llm_exceptions,
)
from data_designer.engine.models.recipes.response_recipes import PydanticResponseRecipe
from data_designer.engine.models.utils import ChatMessage
from data_designer.interface.data_designer import DataDesigner

from anonymizer.config.anonymizer_config import AnonymizerConfig, Detect, TextRecord, TextRecordsInput
from anonymizer.config.regex import RegexRule
from anonymizer.config.replace_strategies import Redact
from anonymizer.engine.constants import (
    COL_ADMITTED_CANDIDATES,
    COL_AUGMENTATION_RESPONSE_FORMAT,
    COL_AUGMENTED_ENTITIES,
    COL_TEXT,
    RECORD_ID_COLUMN,
)
from anonymizer.engine.detection.augmentation import augment_row, augment_row_async, build_response_format_switch
from anonymizer.engine.ndd.adapter import NddAdapter, WorkflowRunResult
from anonymizer.engine.ndd.model_loader import load_models_config
from anonymizer.engine.schemas import AugmentedEntitiesSchema
from anonymizer.engine.workflow_columns.detection.config import AugmentationConfig
from anonymizer.interface.anonymizer import Anonymizer
from tests.engine.ndd.stub_provider import Reply, stub_provider

_ALIAS = "augmenter-alias"
_TEXT = "Ask Alice now"
_ALICE = {"value": "Alice", "label": "first_name", "context_before": "Ask ", "context_after": " now"}
_GOOD = json.dumps({"entities": [_ALICE]})
_UNGROUNDED = json.dumps({"entities": [{**_ALICE, "value": "ALICE"}]})
_FENCED = f"```json\n{_GOOD}\n```"
_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "AugmentedEntitiesSchema", "schema": AugmentedEntitiesSchema.model_json_schema()},
}
_FENCE_LINE = "* Your JSON response must be returned within a Markdown, ```json code fence.\n"
_REJECTION = "response_format json_schema is not supported"


def _dd_error(status: int, message: str) -> Exception:
    """The DataDesigner error a facade raises for this HTTP status and provider message."""
    provider_error = ProviderError(
        kind=map_http_status_to_provider_error_kind(status, message.lower()), message=message, status_code=status
    )
    try:
        try:
            raise provider_error
        except ProviderError as exc:
            handle_llm_exceptions(exc, model_name="stub-model", model_provider_name="stub")
    except Exception as converted:
        return converted
    raise AssertionError("handle_llm_exceptions did not raise")


class _ScriptedFacade:
    """Replies with scripted text (parsed by the caller's parser) or raises scripted errors."""

    def __init__(self, script: list[str | Exception]) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> tuple[AugmentedEntitiesSchema, list[ChatMessage]]:
        self.calls.append(kwargs)
        reply = self.script.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return kwargs["parser"](reply), [ChatMessage.as_assistant(content=reply)]

    async def agenerate(self, **kwargs: Any) -> tuple[AugmentedEntitiesSchema, list[ChatMessage]]:
        return self.generate(**kwargs)


def _augment(facade: _ScriptedFacade, switch: Any, *, use_async: bool = False, text: str = _TEXT) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "row": {RECORD_ID_COLUMN: "record-1", COL_TEXT: text},
        "prompt_template": "Source: {{ " + COL_TEXT + " }}",
        "model_alias": _ALIAS,
        "facade": facade,
        "allowed_entity_labels": ["first_name"],
        "excluded_entity_labels": [],
        "system_prompt": "Augmentation rules",
        "response_format": switch,
    }
    return asyncio.run(augment_row_async(**kwargs)) if use_async else augment_row(**kwargs)


def _logged(caplog: pytest.LogCaptureFixture, level: int) -> list[logging.LogRecord]:
    """Each record once: the suite's logging setup can hand caplog the same record twice."""
    return list({id(record): record for record in caplog.records if record.levelno == level}.values())


def _openai() -> Any:
    return build_response_format_switch(provider_type="openai", extra_bodies=(None, None))


@pytest.mark.parametrize("use_async", [False, True])
def test_openai_compatible_requests_carry_the_schema_on_both_grounding_attempts(use_async: bool) -> None:
    facade = _ScriptedFacade([_UNGROUNDED, _GOOD])

    row = _augment(facade, _openai(), use_async=use_async)

    assert len(facade.calls) == 2
    assert [call["response_format"] for call in facade.calls] == [_RESPONSE_FORMAT, _RESPONSE_FORMAT]
    assert row[COL_AUGMENTATION_RESPONSE_FORMAT] == "sent"
    assert row[COL_AUGMENTED_ENTITIES]["entities"][0]["value"] == "Alice"


@pytest.mark.parametrize("use_async", [False, True])
def test_anthropic_requests_omit_response_format(use_async: bool) -> None:
    facade = _ScriptedFacade([_GOOD])

    row = _augment(
        facade, build_response_format_switch(provider_type="anthropic", extra_bodies=()), use_async=use_async
    )

    assert "response_format" not in facade.calls[0]
    assert row[COL_AUGMENTATION_RESPONSE_FORMAT] == "unconstrained"


@pytest.mark.parametrize(
    "extra_bodies",
    [
        ({"response_format": {"type": "text"}}, None),
        (None, {"response_format": {"type": "json_object"}}),
        ({"response_format": None}, None),
    ],
    ids=["model-config", "provider", "null"],
)
def test_user_extra_body_response_format_turns_ours_off(
    extra_bodies: tuple[dict[str, Any] | None, ...], caplog: pytest.LogCaptureFixture
) -> None:
    facade = _ScriptedFacade([_GOOD])
    with caplog.at_level(logging.INFO, logger="anonymizer.detection.augmentation"):
        switch = build_response_format_switch(provider_type="openai", extra_bodies=extra_bodies)
        row = _augment(facade, switch)

    assert "response_format" not in facade.calls[0]
    assert row[COL_AUGMENTATION_RESPONSE_FORMAT] == "unconstrained"
    infos = _logged(caplog, logging.INFO)
    assert len(infos) == 1
    assert "extra_body" in infos[0].getMessage()
    assert _ALIAS not in caplog.text and "Alice" not in caplog.text


@pytest.mark.parametrize("use_async", [False, True])
def test_fence_instruction_is_dropped_only_when_the_schema_is_sent(use_async: bool) -> None:
    plain_recipe = PydanticResponseRecipe(data_type=AugmentedEntitiesSchema)
    today = plain_recipe.apply_recipe_to_user_prompt(f"Source: {_TEXT}")
    assert _FENCE_LINE in today
    sent = _ScriptedFacade([_GOOD])
    not_sent = _ScriptedFacade([_GOOD])

    _augment(sent, _openai(), use_async=use_async)
    _augment(not_sent, build_response_format_switch(provider_type="anthropic", extra_bodies=()), use_async=use_async)

    assert sent.calls[0]["prompt"] == today.replace(_FENCE_LINE, "")
    assert not_sent.calls[0]["prompt"] == today
    for call in (sent.calls[0], not_sent.calls[0]):
        assert call["system_prompt"] == plain_recipe.apply_recipe_to_system_prompt("Augmentation rules")


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize(
    ("status", "message"),
    [(400, _REJECTION), (422, "guided decoding backend failed to compile")],
    ids=["400-response-format", "422-guided"],
)
def test_matching_rejection_retries_once_without_the_field_and_switches_off(
    use_async: bool, status: int, message: str, caplog: pytest.LogCaptureFixture
) -> None:
    error = _dd_error(status, message)
    if status == 422:
        assert isinstance(error, ModelUnprocessableEntityError)
        assert "guided" not in str(error)  # Only the chained provider error names it.
    switch = _openai()
    facade = _ScriptedFacade([error, _GOOD])

    with caplog.at_level(logging.INFO, logger="anonymizer.detection.augmentation"):
        first = _augment(facade, switch, use_async=use_async)
        later_facade = _ScriptedFacade([_GOOD])
        later = _augment(later_facade, switch, use_async=use_async)

    assert [("response_format" in call) for call in facade.calls] == [True, False]
    assert facade.calls[1]["prompt"].endswith(
        PydanticResponseRecipe(data_type=AugmentedEntitiesSchema).task_instructions
    )
    assert first[COL_AUGMENTATION_RESPONSE_FORMAT] == "fallback"
    assert len(later_facade.calls) == 1 and "response_format" not in later_facade.calls[0]
    assert later[COL_AUGMENTATION_RESPONSE_FORMAT] == "fallback"
    warnings = _logged(caplog, logging.WARNING)
    assert len(warnings) == 1
    for private in (_ALIAS, message, "Alice", "stub-model"):
        assert private not in caplog.text


@pytest.mark.parametrize("use_async", [False, True])
def test_non_matching_bad_request_raises_unchanged_without_retry(use_async: bool) -> None:
    error = _dd_error(400, "temperature out of range")
    assert isinstance(error, ModelBadRequestError)
    switch = _openai()
    facade = _ScriptedFacade([error, _GOOD])

    with pytest.raises(ModelBadRequestError) as raised:
        _augment(facade, switch, use_async=use_async)

    assert raised.value is error
    assert len(facade.calls) == 1
    assert facade.calls[0]["response_format"] == _RESPONSE_FORMAT
    assert switch.send


@pytest.mark.parametrize("use_async", [False, True])
def test_rejection_that_persists_on_the_retry_raises_the_original_error(use_async: bool) -> None:
    original = _dd_error(400, _REJECTION)
    switch = _openai()
    facade = _ScriptedFacade([original, _dd_error(400, "still a bad request")])

    with pytest.raises(type(original)) as raised:
        _augment(facade, switch, use_async=use_async)

    assert raised.value is original
    assert isinstance(raised.value.__context__, ProviderError)
    assert len(facade.calls) == 2
    assert switch.send


@pytest.mark.parametrize("use_async", [False, True])
def test_context_window_error_naming_response_format_is_not_retried(use_async: bool) -> None:
    error = _dd_error(400, "This model's maximum context length is 8192 tokens (response_format included)")
    assert isinstance(error, ModelContextWindowExceededError)
    facade = _ScriptedFacade([error, _GOOD])

    with pytest.raises(ModelContextWindowExceededError):
        _augment(facade, _openai(), use_async=use_async)

    assert len(facade.calls) == 1
    assert facade.calls[0]["response_format"] == _RESPONSE_FORMAT


@pytest.mark.parametrize("use_async", [False, True])
def test_fenced_reply_to_a_constrained_request_is_recorded_as_ignored(
    use_async: bool, caplog: pytest.LogCaptureFixture
) -> None:
    switch = _openai()
    with caplog.at_level(logging.INFO, logger="anonymizer.detection.augmentation"):
        rows = [_augment(_ScriptedFacade([_FENCED]), switch, use_async=use_async) for _ in range(2)]

    assert [row[COL_AUGMENTATION_RESPONSE_FORMAT] for row in rows] == ["ignored", "ignored"]
    assert switch.send
    assert len(_logged(caplog, logging.WARNING)) == 1
    assert _ALIAS not in caplog.text and "Alice" not in caplog.text


# --- Real DataDesigner runs against a loopback provider -------------------------------------------------------------


def _complete(content: str) -> Reply:
    return 200, content, "stop"


def _run_column(
    tmp_path: Path,
    texts: list[str],
    *,
    body_script: Any,
    bodies: list[dict[str, Any]],
    model_extra_body: dict[str, Any] | None = None,
    provider_extra_body: dict[str, Any] | None = None,
    max_parallel_requests: int = 4,
    run_config: RunConfig | None = None,
) -> WorkflowRunResult:
    df = pd.DataFrame({COL_TEXT: texts, COL_ADMITTED_CANDIDATES: [{"candidates": []} for _ in texts]})
    with stub_provider(lambda *_: _complete(_GOOD), bodies=bodies, body_script=body_script) as provider:
        designer = DataDesigner(
            artifact_path=tmp_path / "artifacts",
            model_providers=[provider.model_copy(update={"extra_body": provider_extra_body})],
            auto_configure_logging=False,
        )
        if run_config is not None:
            designer.set_run_config(run_config)
        return NddAdapter(designer).run_workflow(
            df,
            model_configs=[
                ModelConfig(
                    alias=_ALIAS,
                    model="stub-model",
                    provider="stub",
                    skip_health_check=True,
                    inference_parameters=ChatCompletionInferenceParams(
                        max_parallel_requests=max_parallel_requests, extra_body=model_extra_body
                    ),
                )
            ],
            columns=[
                cast(
                    ColumnConfigT,
                    AugmentationConfig(
                        name=COL_AUGMENTED_ENTITIES,
                        model_alias=_ALIAS,
                        prompt_template="Source: {{ " + COL_TEXT + " }}",
                        allowed_entity_labels=["first_name"],
                    ),
                )
            ],
            workflow_name="augmentation-probe",
        )


def _reject_schema(body: dict[str, Any]) -> Reply | None:
    return (400, _REJECTION, "stop") if "response_format" in body else None


@pytest.mark.parametrize(
    ("body_script", "expected_mode", "expected_requests"),
    [
        (lambda _body: None, "sent", 1),
        (lambda _body: _complete(_FENCED), "ignored", 1),
        (_reject_schema, "fallback", 2),
    ],
    ids=["sent", "ignored", "fallback"],
)
def test_side_column_records_the_mode_through_a_real_run(
    tmp_path: Path, body_script: Any, expected_mode: str, expected_requests: int
) -> None:
    bodies: list[dict[str, Any]] = []

    result = _run_column(tmp_path, [_TEXT], body_script=body_script, bodies=bodies)

    assert result.failed_records == []
    assert result.dataframe[COL_AUGMENTATION_RESPONSE_FORMAT].tolist() == [expected_mode]
    assert result.dataframe[COL_AUGMENTED_ENTITIES].iloc[0]["entities"][0]["value"] == "Alice"
    assert len(bodies) == expected_requests
    assert bodies[0]["response_format"] == _RESPONSE_FORMAT
    assert all("response_format" not in body for body in bodies[1:])


@pytest.mark.parametrize("where", ["model-config", "provider"])
def test_user_extra_body_reaches_the_wire_instead_of_ours(tmp_path: Path, where: str) -> None:
    bodies: list[dict[str, Any]] = []
    users = {"response_format": None}

    result = _run_column(
        tmp_path,
        [_TEXT],
        body_script=lambda _body: None,
        bodies=bodies,
        model_extra_body=users if where == "model-config" else None,
        provider_extra_body=users if where == "provider" else None,
    )

    assert result.dataframe[COL_AUGMENTATION_RESPONSE_FORMAT].tolist() == ["unconstrained"]
    assert [body["response_format"] for body in bodies] == [None]
    assert _FENCE_LINE.strip() in str(bodies[0]["messages"][-1]["content"])


def test_restarts_and_corrections_keep_the_schema(tmp_path: Path) -> None:
    bodies: list[dict[str, Any]] = []
    replies = iter(['{"entities": [{"value": "Alice",', '{"entities": [{"label": "first_name"}]}'])

    result = _run_column(
        tmp_path,
        [_TEXT],
        body_script=lambda _body: _complete(next(replies, _GOOD)),
        bodies=bodies,
        run_config=RunConfig(max_conversation_restarts=1, max_conversation_correction_steps=1),
    )

    assert result.failed_records == []
    assert len(bodies) == 3
    assert [len(body["messages"]) for body in bodies] == [1, 3, 1]
    assert all(body["response_format"] == _RESPONSE_FORMAT for body in bodies)
    assert result.dataframe[COL_AUGMENTATION_RESPONSE_FORMAT].tolist() == ["sent"]


def test_backend_that_always_rejects_the_schema_completes_without_early_shutdown(tmp_path: Path) -> None:
    bodies: list[dict[str, Any]] = []
    texts = [f"Ask Alice now, case {index}" for index in range(20)]
    reply = json.dumps({"entities": [{**_ALICE, "context_after": " now,"}]})

    result = _run_column(
        tmp_path,
        texts,
        body_script=lambda body: _reject_schema(body) or _complete(reply),
        bodies=bodies,
        max_parallel_requests=1,
    )

    assert len(result.dataframe) == 20
    assert result.failed_records == []
    assert result.dataframe[COL_AUGMENTATION_RESPONSE_FORMAT].tolist() == ["fallback"] * 20
    assert sum("response_format" in body for body in bodies) == 1
    assert len(bodies) == 21


def test_side_column_is_kept_in_the_trace_and_left_out_of_user_output(tmp_path: Path) -> None:
    rule = RegexRule(
        label="ticket", pattern=r"TKT-[0-9]+", validate_matches_with_llm=False, detect_additional_matches=False
    )
    config = AnonymizerConfig(
        detect=Detect(entity_labels=["ticket"], builtin_regexes=False, regex_rules=[rule]), replace=Redact()
    )
    models_yaml = tmp_path / "models.yaml"
    models = load_models_config()["model_configs"]
    models_yaml.write_text(yaml.safe_dump({"model_configs": [{**m, "skip_health_check": True} for m in models]}))
    anonymizer = Anonymizer(model_configs=models_yaml, artifact_path=tmp_path / "artifacts")

    with (
        patch("httpx.Client.send", side_effect=AssertionError("unexpected synchronous model request")),
        patch("httpx.AsyncClient.send", side_effect=AssertionError("unexpected asynchronous model request")),
    ):
        result = anonymizer.run(
            config=config, data=TextRecordsInput(records=[TextRecord(id="a", text="Escalate TKT-4821 today.")])
        )

    assert COL_AUGMENTATION_RESPONSE_FORMAT not in result.dataframe.columns
    assert result.trace_dataframe[COL_AUGMENTATION_RESPONSE_FORMAT].isna().tolist() == [True]
