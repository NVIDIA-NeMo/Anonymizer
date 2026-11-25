# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``--config FILE``: the file as the base, the flags the user gave laid over it, and one validation."""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any, get_args

import yaml
from pydantic import BaseModel, ValidationError

from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput, Detect
from anonymizer.config.replace_strategies import HASH_KEY_ENV
from anonymizer.interface.cli.errors import Arguments, _describe, _flag_after_parse
from anonymizer.interface.cli.params import MODES, RunOptions, env_only
from anonymizer.interface.errors import InvalidConfigError


def _given(model: BaseModel | None) -> dict[str, Any]:
    """The fields the user set on the command line (or through the environment), nested models included. cyclopts
    builds each model from only the tokens it was given, so ``model_fields_set`` holds exactly those fields and a
    flag's default never stands in for a value from the config file."""
    # An after-validator that fills a field also adds it to ``model_fields_set``: Rewrite fills ``privacy_goal``
    # with ``PrivacyGoal()``. That nested model has no fields set, so it contributes an empty mapping, which the
    # merge lays over the file's ``privacy_goal`` without changing it. A validator that filled a plain value would
    # replace the file's value; none does today (pinned by a test).
    if model is None:
        return {}
    return {
        name: _given(value) if isinstance(value, BaseModel) else value
        for name in model.model_fields_set
        for value in (getattr(model, name),)
    }


def _model_of(annotation: Any) -> type[BaseModel] | None:
    """The model a field holds (through ``X | None``), or ``None`` for a plain value, list or mapping."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    models = [arg for arg in get_args(annotation) if isinstance(arg, type) and issubclass(arg, BaseModel)]
    return models[0] if len(models) == 1 else None


def _merge(base: dict[str, Any], top: dict[str, Any], model: type[BaseModel]) -> dict[str, Any]:
    """``base`` (the file) with ``top`` (the given flags) laid over it. Nested settings merge field by field; any
    other value from a flag, a list or a mapping included, replaces the file's whole value."""
    merged = dict(base)
    for key, value in top.items():
        info = model.model_fields.get(key)
        nested = _model_of(info.annotation) if info is not None else None
        if nested is not None and isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value, nested)
        else:
            merged[key] = value
    return merged


def _unknown_fields(path: Path, data: dict[str, Any], model: type[BaseModel], where: str) -> None:
    """Refuse a key the target model does not have, recursing into model-typed fields. The SDK models still ignore
    unknown keys; a file is checked here so that a misspelt setting is not silently dropped."""
    for key, value in data.items():
        info = model.model_fields.get(str(key))
        if info is None:
            raise InvalidConfigError(f"{path}: {where}.{key}: unknown field")
        nested = _model_of(info.annotation)
        if nested is not None and isinstance(value, dict):
            _unknown_fields(path, value, nested, f"{where}.{key}")


_CONFIG_SECTIONS = ("input", "detect", "replace", "rewrite", "emit_telemetry")


def _read_config_file(path: Path, mode: str) -> dict[str, Any]:
    """The ``--config`` file: an ``AnonymizerConfig`` in YAML or JSON (JSON is YAML) plus an optional ``input``
    section. The file may set only this subcommand's mode, and never the hash key."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InvalidConfigError(f"{path}: cannot read the config file ({exc.strerror or exc})") from None
    except UnicodeDecodeError:
        raise InvalidConfigError(f"{path}: cannot read the config file (not UTF-8 text)") from None
    except yaml.YAMLError as exc:
        # The problem and its position only: yaml's own message quotes the offending line.
        mark = getattr(exc, "problem_mark", None)
        where = f" at line {mark.line + 1}, column {mark.column + 1}" if mark is not None else ""
        raise InvalidConfigError(
            f"{path}: not valid YAML or JSON{where}: {getattr(exc, 'problem', None) or 'parse error'}"
        ) from None
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise InvalidConfigError(f"{path}: expected a mapping with {', '.join(_CONFIG_SECTIONS)}")
    unknown = sorted(str(key) for key in data if key not in _CONFIG_SECTIONS)
    if unknown:
        raise InvalidConfigError(
            f"{path}: unknown section(s) {', '.join(unknown)}; expected {', '.join(_CONFIG_SECTIONS)}"
        )
    section, other = ("rewrite", "replace") if mode == "rewrite" else ("replace", "rewrite")
    if data.get(other) is not None:
        raise InvalidConfigError(
            f"{path}: the file sets {other}, but this command is {mode}; use the matching subcommand"
        )
    body = data.get(section)
    if body is not None and not isinstance(body, dict):
        raise InvalidConfigError(f"{path}: {section}: expected a mapping")
    if section == "replace" and body:
        if any(str(key).lower() == "key" for key in body):
            raise InvalidConfigError(
                f"{path}: replace.key is not read from a file; set {HASH_KEY_ENV} instead and remove the key from the "
                "file"
            )
        kind = str(body.get("kind", mode)).lower()
        if kind not in MODES or kind == "rewrite":
            # The value is not echoed: it is not a known kind, so it may be anything, a secret included.
            kinds = ", ".join(name for name in MODES if name != "rewrite")
            raise InvalidConfigError(f"{path}: replace.kind is not one of {kinds}")
        if kind != mode:
            raise InvalidConfigError(
                f"{path}: replace.kind is {kind}, but this command is {mode}; use the {kind} subcommand"
            )
    targets = {"input": AnonymizerInput, "detect": Detect, section: MODES[mode]}
    for name, model in targets.items():
        if data.get(name) is None:
            continue
        if not isinstance(data[name], dict):
            raise InvalidConfigError(f"{path}: {name}: expected a mapping")
        _unknown_fields(
            path, {k: v for k, v in data[name].items() if not (name == "replace" and k == "kind")}, model, name
        )
    return data


def _missing_env(mode: str) -> None:
    """A field read only from the environment that is unset is named by its variable."""
    missing = [env[0] for env in env_only(MODES[mode]).values() if not any(os.environ.get(var) for var in env)]
    if missing:
        raise InvalidConfigError(
            f"{' and '.join(missing)} is not set. run {mode} reads its secret key from it; there is no flag."
        )


def _input(
    arguments: Arguments, mode: str, data: BaseModel | None, file: dict[str, Any], path: Path | None
) -> AnonymizerInput:
    """The input: the file's ``input`` section with the input flags over it."""
    given = _given(data)
    merged = _merge(file.get("input") or {}, given, AnonymizerInput)
    if merged.get("source") is None:
        raise InvalidConfigError("--source is required" + (f" (or input.source in {path})" if path else ""))
    try:
        return AnonymizerInput.model_validate(merged)
    except ValidationError as exc:
        name_for = functools.partial(_flag_after_parse, arguments, mode=mode, given={"data": given}, path=path)
        raise InvalidConfigError(_describe(exc, lambda loc: name_for(("input", *loc)))) from None


def _config(
    arguments: Arguments,
    mode: str,
    data: BaseModel | None,
    strategy: BaseModel | None,
    detect: Detect | None,
    options: RunOptions,
) -> tuple[AnonymizerConfig, AnonymizerInput]:
    """The config and the input: the ``--config`` file as the base, the flags the user gave laid over it field by
    field, validated once. A model the user gave no flag for comes from the file, else from its defaults. Telemetry
    is off if either the file or ``--no-emit-telemetry`` turns it off."""
    path = options.config
    file = _read_config_file(path, mode) if path is not None else {}
    section = "rewrite" if mode == "rewrite" else "replace"
    if strategy is None:
        _missing_env(mode)
        strategy = MODES[mode]()
    flags = {"strategy": _given(strategy), "detect": _given(detect)}
    merged = {key: value for key, value in file.items() if key != "input"}
    merged[section] = _merge(file.get(section) or {}, flags["strategy"], MODES[mode])
    merged["detect"] = _merge(file.get("detect") or {}, flags["detect"], Detect)
    if section == "replace":
        merged["replace"]["kind"] = mode
    name_for = functools.partial(_flag_after_parse, arguments, mode=mode, given=flags, path=path)
    try:
        # Detect's cross-field checks run here, once, on the merged settings (the flags were parsed without them).
        # A check of the whole model has no flag to name: without a file it reads "value", as it did at parse time;
        # with a file it reads "detect".
        merged["detect"] = Detect.model_validate(merged["detect"])
    except ValidationError as exc:
        whole = "detect" if path is not None else None  # file and flags together: no single flag to name
        raise InvalidConfigError(_describe(exc, lambda loc: name_for(("detect", *loc)) if loc else whole)) from None
    try:
        config = AnonymizerConfig.model_validate(merged)
    except ValidationError as exc:
        raise InvalidConfigError(_describe(exc, name_for)) from None
    if not options.emit_telemetry and config.emit_telemetry:
        config = config.model_copy(update={"emit_telemetry": False})
    return config, _input(arguments, mode, data, file, path)
