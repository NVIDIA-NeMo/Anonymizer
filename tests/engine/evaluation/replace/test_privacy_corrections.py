# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from data_designer.config.models import ModelConfig, ModelProvider
from data_designer.engine.model_provider import ModelProviderRegistry
from data_designer.engine.models.clients.base import ModelClient
from data_designer.engine.models.clients.types import AssistantMessage, ChatCompletionResponse
from data_designer.engine.models.errors import ModelGenerationValidationFailureError
from data_designer.engine.models.facade import ModelFacade
from data_designer.engine.models.recipes.response_recipes import PydanticResponseRecipe

from anonymizer.engine.evaluation.replace.entity_sensitivity import (
    IndexedEntitySensitivitiesSchema,
    _make_sensitivity_parser,
)
from anonymizer.engine.evaluation.replace.privacy_answers import (
    CompletePrivacyAnswersSchema,
    _make_complete_answer_parser,
)


def _parser_and_responses(kind: str) -> tuple[Any, str, str]:
    if kind == "sensitivity":
        recipe = PydanticResponseRecipe(data_type=IndexedEntitySensitivitiesSchema)
        parser = _make_sensitivity_parser(recipe, [("USA", "country")])
        missing = '{"entities": []}'
        complete = json.dumps(
            {
                "entities": [
                    {
                        "id": 1,
                        "sensitivity": "low",
                        "reason": "Broad country designation.",
                    }
                ]
            }
        )
    else:
        recipe = PydanticResponseRecipe(data_type=CompletePrivacyAnswersSchema)
        parser = _make_complete_answer_parser(recipe, [1])
        missing = '{"answers": []}'
        complete = json.dumps(
            {
                "answers": [
                    {
                        "id": 1,
                        "answer": "no",
                        "confidence": 0.9,
                        "reason": "Country is not recoverable.",
                        "evidence": [],
                    }
                ]
            }
        )
    return parser, missing, complete


@pytest.mark.parametrize("kind", ["sensitivity", "privacy_answers"])
@pytest.mark.parametrize("recovers", [True, False])
def test_coverage_errors_use_real_data_designer_correction_loop(kind: str, recovers: bool) -> None:
    facade = ModelFacade(
        ModelConfig(alias="synthetic", model="synthetic", provider="synthetic"),
        ModelProviderRegistry(providers=[ModelProvider(name="synthetic", endpoint="https://example.invalid/v1")]),
        client=Mock(spec=ModelClient),
    )
    parser, missing, complete = _parser_and_responses(kind)
    responses = [missing, complete] if recovers else [missing] * 4
    completion = AsyncMock(
        side_effect=[ChatCompletionResponse(message=AssistantMessage(content=response)) for response in responses]
    )
    # Stub only the completion boundary: agenerate's retry/correction policy is real.
    with patch.object(facade, "acompletion", completion):
        if recovers:
            result, _ = asyncio.run(facade.agenerate("Assess supplied targets.", parser=parser, max_correction_steps=3))
            assert result is not None
            assert completion.call_count == 2
        else:
            with pytest.raises(ModelGenerationValidationFailureError):
                asyncio.run(facade.agenerate("Assess supplied targets.", parser=parser, max_correction_steps=3))
            assert completion.call_count == 4
    correction_messages = completion.call_args_list[1].args[0]
    assert any("must cover exactly" in str(message.content) for message in correction_messages)
