# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The CLI's parameters: the modes, the help panels, the flag models and the run options, and the exit codes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, get_args

import cyclopts
from pydantic import BaseModel, Field, create_model, model_validator

from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput, Detect, Rewrite
from anonymizer.config.replace_strategies import ReplaceMethod

# One subcommand per mode, from the members of ReplaceMethod (by their ``kind``) plus rewrite.
MODES: dict[str, type[BaseModel]] = {
    cls.model_fields["kind"].default: cls for cls in get_args(get_args(ReplaceMethod)[0])
} | {"rewrite": Rewrite}


# Help panels in reading order: what to read and write, the mode's own settings, detection, then run options.
INPUT_GROUP = cyclopts.Group("Input and output", sort_key=0)
DETECT_GROUP = cyclopts.Group(
    "Detect",
    sort_key=2,
    help=(
        "List flags take one value each; repeat the flag for more (--entity-labels first_name --entity-labels "
        "city). --entity-label-examples takes a JSON object. --regex-rules takes one JSON rule or a JSON list of "
        "rules per flag; a rule's validator is a registered validator name, since Python callables cannot be "
        "given on the command line."
    ),
)
RUN_GROUP = cyclopts.Group("Run", sort_key=3)


# The input flags: ``AnonymizerInput``'s fields, each optional, since a ``--config`` file's ``input`` section can
# supply them. Generated from the model, with its descriptions; ``_input`` validates the result as an
# ``AnonymizerInput``.
_INPUT_FIELDS: dict[str, Any] = {
    # A required field (source) defaults to None here; the others keep their defaults, which help shows. A default
    # never replaces a value from the file: only the fields the user gave are laid over it.
    name: (
        info.annotation | None,
        Field(default=None if info.is_required() else info.default, description=info.description),
    )
    for name, info in AnonymizerInput.model_fields.items()
}
_InputFlags: type[BaseModel] = create_model("_InputFlags", **_INPUT_FIELDS)


class _DetectFlags(Detect):
    """``Detect`` as the flags fill it: each field is checked as it is parsed, but the checks across fields wait for
    the merge with the ``--config`` file. ``_config`` runs them once, on the merged settings. A flag such as
    ``--entity-label-examples`` may need ``entity_labels`` from the file."""

    @model_validator(mode="after")
    def validate_entity_label_overlap(self) -> _DetectFlags:
        return self

    @model_validator(mode="after")
    def validate_regex_rule_scope(self) -> _DetectFlags:
        return self


@dataclass(frozen=True)
class RunOptions:
    """Options that are not config: how the ``Anonymizer`` is built, telemetry and logging."""

    config: Annotated[
        Path | None,
        cyclopts.Parameter(
            group=INPUT_GROUP,
            help=(
                "YAML or JSON file in the same shape as the SDK's AnonymizerConfig (detect; replace or rewrite; "
                "emit_telemetry), plus an optional input section (source, text_column, id_column, data_summary). "
                "Flags given on the command line replace the file's values. The file must match the subcommand's "
                "mode and must not hold the hash key; unknown keys are refused."
            ),
        ),
    ] = None
    model_configs: Annotated[
        str | None,
        cyclopts.Parameter(
            help=(
                "Path to a YAML file (.yaml/.yml) or inline YAML string defining the model pool "
                "and optional selected_models overrides. Omit to use bundled defaults."
            )
        ),
    ] = None
    model_providers: Annotated[
        str | None,
        cyclopts.Parameter(
            help=(
                "Path to a YAML file (.yaml/.yml) or inline YAML string defining custom provider "
                "endpoints. Each entry maps a provider name to its API base URL and optional key."
            )
        ),
    ] = None
    artifact_path: Annotated[
        str | None,
        cyclopts.Parameter(help="Directory for intermediate pipeline artifacts. Defaults to .anonymizer-artifacts."),
    ] = None
    # AnonymizerConfig.emit_telemetry; the other AnonymizerConfig fields are the subcommand and its flags.
    emit_telemetry: Annotated[
        bool,
        cyclopts.Parameter(
            help=(
                "Whether to emit anonymous Anonymizer telemetry events. "
                "Use --no-emit-telemetry to opt out for this invocation. "
                "Also overridable globally via NEMO_TELEMETRY_ENABLED=false."
            )
        ),
    ] = AnonymizerConfig.model_fields["emit_telemetry"].default
    verbose: Annotated[bool, cyclopts.Parameter(help="Verbose logging.", negative="")] = False
    debug: Annotated[bool, cyclopts.Parameter(help="Debug logging.", negative="")] = False


# Exit codes. A usage or config error (an unknown flag, a value a model refuses, a bad config) exits 2, as cyclopts
# does for parse errors; an input or I/O error (a bad input file or id column, an unreadable path) exits 1. Any other
# error is a runtime fault and keeps its traceback.
EXIT_USAGE = 2
EXIT_INPUT = 1


def env_only(model: type[BaseModel]) -> dict[str, tuple[str, ...]]:
    """The fields of ``model`` read only from the environment, with their variables."""
    return {
        name: tuple(meta.env_var)
        for name, info in model.model_fields.items()
        for meta in info.metadata
        if isinstance(meta, cyclopts.Parameter) and meta.parse is False and meta.env_var
    }
