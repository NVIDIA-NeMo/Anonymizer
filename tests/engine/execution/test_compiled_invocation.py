# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pickle
from dataclasses import FrozenInstanceError

import pytest

from anonymizer.config.anonymizer_config import AnonymizerConfig, Rewrite
from anonymizer.config.models import ModelSelection
from anonymizer.config.replace_strategies import Annotate, Hash, Redact, Substitute
from anonymizer.engine.execution.invocation import _CompiledInvocation


@pytest.mark.parametrize("replace", [Redact(), Annotate(), Hash(), Substitute(instructions="keep tone")])
def test_compiled_replace_invocation_is_immutable_and_preserves_strategy(
    replace: Redact | Annotate | Hash | Substitute, stub_slim_model_selection: ModelSelection
) -> None:
    compiled = _CompiledInvocation.compile(AnonymizerConfig(replace=replace), stub_slim_model_selection)

    assert compiled.replace_method == replace
    assert compiled.rewrite is None
    with pytest.raises(FrozenInstanceError):
        setattr(compiled, "replace_method", Redact())
    with pytest.raises(TypeError, match="not serializable"):
        pickle.dumps(compiled)


def test_compiled_rewrite_uses_rewrite_evaluation_property(
    stub_slim_model_selection: ModelSelection, monkeypatch: pytest.MonkeyPatch
) -> None:
    rewrite = Rewrite(max_repair_iterations=2, use_combined_graph=True, strict_entity_protection=True)
    expected = rewrite.evaluation
    calls = 0
    original = Rewrite.evaluation.fget
    assert original is not None

    def evaluation(self: Rewrite):
        nonlocal calls
        calls += 1
        return original(self)

    monkeypatch.setattr(Rewrite, "evaluation", property(evaluation))
    compiled = _CompiledInvocation.compile(AnonymizerConfig(rewrite=rewrite), stub_slim_model_selection)

    assert calls == 1
    assert compiled.rewrite is not None
    assert compiled.rewrite.evaluation == expected
    assert compiled.rewrite.use_combined_graph is True
    assert compiled.rewrite.strict_entity_protection is True
