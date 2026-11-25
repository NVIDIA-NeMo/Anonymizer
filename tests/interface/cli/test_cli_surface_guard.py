# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every config model field is a CLI flag, or is named here as SDK-only. A new field fails this test until it is
one or the other."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput, Detect, Rewrite
from anonymizer.config.regex import RegexLimits
from anonymizer.config.rewrite import PrivacyGoal
from anonymizer.interface.cli.main import MODES, app

# Fields with no flag, and why. The SDK-only list is short on purpose; see CHANGELOG "CLI and config surface".
NOT_A_FLAG: dict[tuple[str, str], str] = {
    **{(cls.__name__, "kind"): "the subcommand picks the strategy" for cls in MODES.values() if cls is not Rewrite},
    ("Hash", "key"): "read from ANONYMIZER_HASH_KEY only, to keep the key out of argv",
    ("Rewrite", "instructions"): "never read by the rewrite (audit F10); removal is pending (#407)",
    ("AnonymizerConfig", "replace"): "the subcommand",
    ("AnonymizerConfig", "rewrite"): "the subcommand",
    ("AnonymizerConfig", "detect"): "its fields are flags",
    ("Detect", "regex_limits"): "its fields are flags under --regex-limits.",
    ("Rewrite", "privacy_goal"): "its fields are the flags --protect and --preserve",
}
# SDK-only surfaces with no field here: RegexRule.validator as a Python callable (a registered name works from the
# CLI), the Anonymizer arguments that take live objects, and TextRecordsInput (in-memory records).


def _flags(verb: str, mode: str) -> set[str]:
    collection = app[verb][mode].assemble_argument_collection()
    return {name for argument in collection if argument.parse for name in argument.names}


def _expected(prefix: str, model: type[BaseModel]) -> list[tuple[str, str, str]]:
    return [(model.__name__, field, f"--{prefix}{field.replace('_', '-')}") for field in model.model_fields]


@pytest.mark.parametrize("verb", ["run", "preview", "validate"])
@pytest.mark.parametrize("mode", list(MODES))
def test_every_config_field_is_a_flag_or_named_sdk_only(verb: str, mode: str) -> None:
    flags = _flags(verb, mode)
    models = [
        ("", AnonymizerInput),
        ("", Detect),
        ("regex-limits.", RegexLimits),
        ("", AnonymizerConfig),
        ("", MODES[mode]),
    ]
    if mode == "rewrite":
        models.append(("", PrivacyGoal))
    missing = [
        (model, field, flag)
        for prefix, cls in models
        for model, field, flag in _expected(prefix, cls)
        if flag not in flags and (model, field) not in NOT_A_FLAG
    ]
    assert missing == []


def test_sdk_only_fields_have_no_flag() -> None:
    for mode, cls in MODES.items():
        flags = _flags("run", mode)
        for model, field in NOT_A_FLAG:
            if model == cls.__name__ and field in ("kind", "key", "instructions"):
                assert f"--{field}" not in flags, (mode, field)
