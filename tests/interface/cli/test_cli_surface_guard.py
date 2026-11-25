# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every config model field is a CLI flag, or is named here as not one. The walk recurses into every field typed as
a model, so a new field or a new nested model fails this test until it is a flag or listed."""

from __future__ import annotations

import types
from typing import Any, Union, get_args, get_origin

import cyclopts
import pytest
from pydantic import BaseModel

from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput, Detect, Rewrite
from anonymizer.interface.cli.main import MODES, app

# Fields with no flag of their own, and why. See CHANGELOG "CLI and config surface".
NOT_A_FLAG: dict[tuple[str, str], str] = {
    **{(cls.__name__, "kind"): "the subcommand picks the strategy" for cls in MODES.values() if cls is not Rewrite},
    ("Hash", "key"): "read from ANONYMIZER_HASH_KEY only, to keep the key out of argv",
    ("Rewrite", "instructions"): "never read by the rewrite (audit F10); removal is pending (#407)",
    ("AnonymizerConfig", "replace"): "the subcommand; its model is walked as the mode",
    ("AnonymizerConfig", "rewrite"): "the subcommand; its model is walked as the mode",
    ("AnonymizerConfig", "detect"): "walked as its own root",
}
# SDK-only surfaces with no field here: RegexRule.validator as a Python callable (a registered name works from the
# CLI), the Anonymizer arguments that take live objects, and TextRecordsInput (in-memory records).


def _flags(verb: str, mode: str) -> set[str]:
    collection = app[verb][mode].assemble_argument_collection()
    return {name for argument in collection if argument.parse for name in argument.names}


def _nested_model(annotation: Any) -> type[BaseModel] | None:
    """The model class a field holds, through ``X | None``; ``None`` for a list, dict or plain value."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    if get_origin(annotation) in (Union, types.UnionType):
        models = [arg for arg in get_args(annotation) if isinstance(arg, type) and issubclass(arg, BaseModel)]
        others = [arg for arg in get_args(annotation) if arg is not type(None) and arg not in models]
        if len(models) == 1 and not others:
            return models[0]
    return None


def _expected_flags(model: type[BaseModel], prefix: str = "") -> list[tuple[str, str, str]]:
    """(model, field, flag) for every field, recursing into nested models. A nested model's flags take the field's
    name as a prefix (``--regex-limits.``) unless the field is flattened with ``Parameter(name="*")``."""
    expected: list[tuple[str, str, str]] = []
    for field, info in model.model_fields.items():
        if (model.__name__, field) in NOT_A_FLAG:
            continue
        nested = _nested_model(info.annotation)
        if nested is None:
            expected.append((model.__name__, field, f"--{prefix}{field.replace('_', '-')}"))
            continue
        flattened = any(isinstance(m, cyclopts.Parameter) and m.name == ("*",) for m in info.metadata)
        expected.extend(_expected_flags(nested, prefix if flattened else f"{prefix}{field.replace('_', '-')}."))
    return expected


@pytest.mark.parametrize("verb", ["run", "preview", "validate"])
@pytest.mark.parametrize("mode", list(MODES))
def test_every_config_field_is_a_flag_or_listed(verb: str, mode: str) -> None:
    flags = _flags(verb, mode)
    expected = [
        *_expected_flags(AnonymizerInput),
        *_expected_flags(Detect),
        *_expected_flags(AnonymizerConfig),
        *_expected_flags(MODES[mode]),
    ]
    assert [entry for entry in expected if entry[2] not in flags] == []


def test_the_walk_reaches_nested_models() -> None:
    flags = {flag for _, _, flag in _expected_flags(Detect)} | {flag for _, _, flag in _expected_flags(Rewrite)}
    assert "--regex-limits.max-matches-per-rule" in flags
    assert {"--protect", "--preserve"} <= flags


def test_listed_fields_have_no_flag() -> None:
    for mode, cls in MODES.items():
        flags = _flags("run", mode)
        for model, field in NOT_A_FLAG:
            if model == cls.__name__:
                assert f"--{field.replace('_', '-')}" not in flags, (mode, field)
