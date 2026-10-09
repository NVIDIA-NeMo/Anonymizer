# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any, cast

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectCode, EffectRejected
from anonymizer.engine.graph_sdk.capabilities import (
    PreparationCode,
    PreparationRejected,
)
from anonymizer.engine.graph_sdk.context import ContextSourceRef, SourceFailure
from anonymizer.engine.graph_sdk.preparation import recheck_capabilities
from tests.graph_sdk.test_preparation import _limits, _prepare


def test_recheck_missing_selected_capability_does_not_mutate_admitted_plan() -> None:
    prepared = _prepare(limits=_limits(capabilities=2))
    selected = next(iter(prepared.implementations)).capability
    retained = prepared.implementations
    with pytest.raises(PreparationRejected) as error:
        recheck_capabilities(prepared=prepared, capabilities=())
    assert error.value.code is PreparationCode.CAPABILITY_CHANGED
    assert prepared.implementations is retained
    recheck_capabilities(prepared=prepared, capabilities=(selected,))


def test_source_failure_construction_does_not_publish_or_replace_an_admitted_plan() -> None:
    prepared = _prepare()
    retained = prepared.implementations
    source = ContextSourceRef(name="S0", revision=1)
    for kwargs in ({"source": source}, {"source": source, "failure": "permanent"}):
        with pytest.raises(TypeError):
            cast(Any, SourceFailure)(**kwargs)
        assert prepared.implementations is retained
    with pytest.raises(EffectRejected) as error:
        SourceFailure(
            source=source,
            failure="retryable",
            settlement=None,
            disposition="omitted_optional",
        )
    assert error.value.code is EffectCode.CONTRADICTORY
    valid = SourceFailure(
        source=source,
        failure="permanent",
        settlement=None,
        disposition="omitted_optional",
    )
    assert valid.failure == "permanent"
    assert prepared.implementations is retained
