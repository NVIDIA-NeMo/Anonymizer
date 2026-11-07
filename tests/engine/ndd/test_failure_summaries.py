# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from data_designer.engine.errors import DataDesignerError
from data_designer.engine.models.clients.errors import ProviderError, ProviderErrorKind
from data_designer.engine.models.errors import GenerationValidationFailureError, handle_llm_exceptions
from data_designer.engine.models.utils import GenerationTruncationReason

from anonymizer.engine.ndd.adapter import _failure_reason

_SENSITIVE_TEXT = "Hollingsworth-Featherstonehaugh lives at Marlborough Crescent"


@pytest.mark.parametrize(
    ("kind", "diagnostic"),
    [
        (ProviderErrorKind.API_ERROR, "unexpected api error"),
        (ProviderErrorKind.API_CONNECTION, "connection to the model failed"),
        (ProviderErrorKind.AUTHENTICATION, "api key is invalid or expired"),
        (ProviderErrorKind.CLIENT_ERROR, "provider rejected the request"),
        (ProviderErrorKind.CONTEXT_WINDOW_EXCEEDED, "context width"),
        (ProviderErrorKind.QUOTA_EXCEEDED, "insufficient credits or quota"),
        (ProviderErrorKind.UNSUPPORTED_PARAMS, "does not support the request parameters"),
        (ProviderErrorKind.BAD_REQUEST, "malformed or missing required parameters"),
        (ProviderErrorKind.INTERNAL_SERVER, "internal server issues"),
        (ProviderErrorKind.NOT_FOUND, "could not be found"),
        (ProviderErrorKind.PERMISSION_DENIED, "lacks model permissions"),
        (ProviderErrorKind.RATE_LIMIT, "rate limit exceeded"),
        (ProviderErrorKind.REQUEST_ADMISSION_TIMEOUT, "provider request was not sent"),
        (ProviderErrorKind.TIMEOUT, "inference_parameters.timeout"),
        (ProviderErrorKind.UNPROCESSABLE_ENTITY, "could not process the request"),
        (ProviderErrorKind.UNSUPPORTED_CAPABILITY, "generation failed"),
    ],
)
def test_provider_failures_keep_actionable_categories_without_echoing_dynamic_text(
    kind: ProviderErrorKind, diagnostic: str
) -> None:
    """Preserve PR #349's actionable reasons without copying provider, model or purpose text."""
    exception = ProviderError(kind=kind, message=_SENSITIVE_TEXT, status_code=400)
    with pytest.raises(DataDesignerError) as caught:
        handle_llm_exceptions(
            exception, model_name=_SENSITIVE_TEXT, model_provider_name=_SENSITIVE_TEXT, purpose=_SENSITIVE_TEXT
        )

    reason = _failure_reason({"column": "checked", "error": str(caught.value)})

    assert reason.startswith("Record missing from workflow output: checked: ")
    assert diagnostic in reason.lower()
    assert _SENSITIVE_TEXT not in reason
    assert len(reason) <= 500


@pytest.mark.parametrize(
    ("truncation", "remedy"),
    [
        (None, "Simplify or modify the output schema"),
        (GenerationTruncationReason.MAX_TOKENS, "Increase inference_parameters.max_tokens"),
        (GenerationTruncationReason.MODEL_CONTEXT_WINDOW_EXCEEDED, "use a model with a larger context window"),
    ],
)
def test_parse_failures_preserve_the_remedy_without_echoing_validation_details(
    truncation: GenerationTruncationReason | None, remedy: str
) -> None:
    """Distinguish output budget from exhausted context, which a larger max_tokens cannot fix."""
    exception = GenerationValidationFailureError("parse failed", detail=_SENSITIVE_TEXT, truncation_reason=truncation)
    with pytest.raises(DataDesignerError) as caught:
        handle_llm_exceptions(exception, model_name="stub", model_provider_name="stub")

    reason = _failure_reason({"column": "checked", "error": str(caught.value)})

    assert "could not be parsed" in reason
    assert remedy in reason
    assert _SENSITIVE_TEXT not in reason
    if truncation == GenerationTruncationReason.MODEL_CONTEXT_WINDOW_EXCEEDED:
        assert "Increase inference_parameters.max_tokens" not in reason
