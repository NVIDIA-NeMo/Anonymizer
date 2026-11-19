# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The structured reply column: DataDesigner's structured column, except that a reply that never parses keeps the row."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Any, cast

import pytest
from data_designer.config.column_configs import LLMStructuredColumnConfig
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.models import ModelConfig, ModelProvider
from data_designer.config.run_config import RunConfig
from data_designer.config.utils.trace_type import TraceType
from data_designer.engine.column_generators.utils.prompt_renderer import create_response_recipe
from data_designer.engine.model_provider import ModelProviderRegistry
from data_designer.engine.models.clients.base import ModelClient
from data_designer.engine.models.clients.types import AssistantMessage, ChatCompletionRequest, ChatCompletionResponse
from data_designer.engine.models.facade import ModelFacade
from data_designer.engine.resources.resource_provider import ResourceProvider
from data_designer.engine.testing.utils import assert_valid_plugin
from pydantic import BaseModel

from anonymizer.engine.workflow_columns.reply.config import ReplyColumnConfig
from anonymizer.engine.workflow_columns.reply.impl import ReplyColumnGenerator
from anonymizer.engine.workflow_columns.reply.plugins import reply_column_plugin

_ALIAS = "reply-model"
_MODEL = ModelConfig(alias=_ALIAS, model="test-model", provider="test-provider")
_PROVIDER = ModelProvider(name="test-provider", endpoint="https://example.invalid/v1")


class _Answer(BaseModel):
    answer: str


class _Client:
    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.requests: list[ChatCompletionRequest] = []

    async def acompletion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        self.requests.append(request)
        reply = self.replies[min(len(self.requests) - 1, len(self.replies) - 1)]
        return ChatCompletionResponse(message=AssistantMessage(content=reply))


class _Registry:
    def __init__(self, facade: ModelFacade) -> None:
        self._facade = facade

    def get_model(self, *, model_alias: str) -> ModelFacade:
        return self._facade

    def get_model_config(self, *, model_alias: str) -> ModelConfig:
        return _MODEL

    def get_model_provider(self, *, model_alias: str) -> ModelProvider:
        return _PROVIDER


def _config(**overrides: Any) -> ReplyColumnConfig:
    return ReplyColumnConfig(
        name="_answer",
        prompt="Answer about {{ text }}.",
        model_alias=_ALIAS,
        output_format=_Answer,
        **overrides,
    )


def _generator(config: ReplyColumnConfig, replies: list[str]) -> tuple[ReplyColumnGenerator, _Client]:
    client = _Client(replies)
    facade = ModelFacade(_MODEL, ModelProviderRegistry(providers=[_PROVIDER]), client=cast(ModelClient, client))
    resource_provider = SimpleNamespace(
        model_registry=_Registry(facade),
        run_config=RunConfig(max_conversation_correction_steps=1, max_conversation_restarts=1),
    )
    return ReplyColumnGenerator(config, cast(ResourceProvider, resource_provider)), client


def test_reply_plugin_satisfies_data_designer_contract() -> None:
    assert_valid_plugin(reply_column_plugin)


def test_reply_config_declares_what_the_stock_structured_column_declares() -> None:
    reply = _config()
    stock = LLMStructuredColumnConfig(
        name="_answer", prompt="Answer about {{ text }}.", model_alias=_ALIAS, output_format=_Answer
    )

    assert reply.required_columns == stock.required_columns
    assert reply.side_effect_columns == stock.side_effect_columns
    assert reply.output_format == stock.output_format
    generator, _ = _generator(reply, ["{}"])
    stock_recipe = create_response_recipe(stock, _MODEL)
    assert generator.response_recipe.apply_recipe_to_user_prompt("Q") == stock_recipe.apply_recipe_to_user_prompt("Q")
    assert generator.response_recipe.apply_recipe_to_system_prompt(None) == stock_recipe.apply_recipe_to_system_prompt(
        None
    )


def test_reply_config_round_trips_through_the_builder(tmp_path: Any) -> None:
    builder = DataDesignerConfigBuilder(model_configs=[_MODEL]).add_column(
        _config(unavailable={"answer": "none"}, ask_only_if=["text"])
    )
    path = tmp_path / "config.json"
    builder.get_builder_config().to_json(path)

    (restored,) = DataDesignerConfigBuilder.from_config(path).get_column_configs()

    assert isinstance(restored, ReplyColumnConfig)
    assert restored.unavailable == {"answer": "none"}
    assert restored.ask_only_if == ["text"]


def test_reply_that_parses_is_stored_as_the_stock_column_stores_it() -> None:
    generator, client = _generator(_config(), ["no json", '{"answer": "yes"}'])

    row = asyncio.run(generator.agenerate({"text": "a"}))

    assert row["_answer"] == {"answer": "yes"}
    assert len(client.requests) == 2


def test_reply_that_never_parses_writes_the_unavailable_value_and_an_empty_trace(
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = _config(unavailable={"answer": "unknown"}, with_trace=TraceType.ALL_MESSAGES)
    generator, client = _generator(config, ["no json"])

    with caplog.at_level(logging.WARNING, logger="anonymizer.ndd.reply"):
        row = asyncio.run(generator.agenerate({"text": "a"}))

    assert row["_answer"] == {"answer": "unknown"}
    assert row["_answer__trace"] == []
    assert len(client.requests) == 4  # the run config's budget: (1 + 1 correction) * (1 + 1 restart)
    assert "_answer unavailable after 4 replies (unparseable_reply)" in caplog.text


@pytest.mark.parametrize("missing", [None, float("nan")])
def test_reply_is_not_asked_when_an_ask_only_if_column_has_no_value(missing: object) -> None:
    generator, client = _generator(_config(ask_only_if=["text"]), ['{"answer": "yes"}'])

    row = asyncio.run(generator.agenerate({"text": missing}))

    assert row["_answer"] is None
    assert client.requests == []
