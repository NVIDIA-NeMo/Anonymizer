# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``--config FILE``: the file as the base, the flags the user gave laid over it, and one validation.

pydantic-settings does the layering: a settings model in the SDK's shape whose sources are the given flags (init
keyword arguments) over the YAML file, deep-merged and validated once. It has no environment source; the hash key
still comes only from ``ANONYMIZER_HASH_KEY`` through the flag model. What pydantic-settings does not do is kept here:
the refusals (mode, kind, key, unknown keys at any depth, a file that is not a mapping), one-line read errors that
quote nothing from the file, and replacing a mapping a flag gives instead of merging into it.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any, ClassVar, get_args

import yaml
from pydantic import BaseModel, ValidationError, create_model
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, YamlConfigSettingsSource

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


def _drop_replaced(file: dict[str, Any], given: dict[str, Any], model: type[BaseModel]) -> None:
    """Remove from the file's values each one a flag replaces whole. pydantic-settings deep-merges every mapping;
    here only nested settings merge field by field, and a mapping a flag gives (``--entity-label-examples``)
    replaces the file's whole value, as a list or a plain value does."""
    for key, value in given.items():
        info = model.model_fields.get(key)
        nested = _model_of(info.annotation) if info is not None else None
        if nested is not None and isinstance(value, dict) and isinstance(file.get(key), dict):
            _drop_replaced(file[key], value, nested)
        elif isinstance(value, dict):
            file.pop(key, None)


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


class _Layers(BaseSettings):
    """The settings model's base: the given flags over the ``--config`` file, and no other source."""

    _file: ClassVar[PydanticBaseSettingsSource]

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (init_settings, cls._file)


def _settings_class(mode: str) -> type[_Layers]:
    """The settings model for ``mode``: the SDK's ``AnonymizerConfig`` shape with this mode's section, and the
    CLI's ``input`` section, which ``_input`` validates on its own."""
    section = "rewrite" if mode == "rewrite" else "replace"
    fields: dict[str, Any] = {
        "input": (dict[str, Any] | None, None),
        "detect": (Detect, Detect()),
        section: (MODES[mode] | None, None),
        "emit_telemetry": (bool, AnonymizerConfig.model_fields["emit_telemetry"].default),
    }
    return create_model(f"_{mode.title()}Settings", __base__=_Layers, **fields)


def _read_config_file(path: Path, mode: str, settings_class: type[_Layers]) -> PydanticBaseSettingsSource:
    """The ``--config`` file as a settings source: an ``AnonymizerConfig`` in YAML or JSON (JSON is YAML) plus an
    optional ``input`` section. The file may set only this subcommand's mode, and never the hash key."""
    try:
        path.read_bytes()  # pydantic-settings reads a missing file as empty; a missing file is an error here
        source = YamlConfigSettingsSource(settings_class, yaml_file=path, yaml_file_encoding="utf-8")
        data = source()
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
    except (TypeError, ValueError):  # the file is a list or scalar, which the source cannot read as settings
        raise InvalidConfigError(f"{path}: expected a mapping with {', '.join(_CONFIG_SECTIONS)}") from None
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
    return source


def _missing_env(mode: str) -> None:
    """A field read only from the environment that is unset is named by its variable."""
    missing = [env[0] for env in env_only(MODES[mode]).values() if not any(os.environ.get(var) for var in env)]
    if missing:
        raise InvalidConfigError(
            f"{' and '.join(missing)} is not set. run {mode} reads its secret key from it; there is no flag."
        )


def _input(
    arguments: Arguments, mode: str, given: dict[str, Any], merged: dict[str, Any], path: Path | None
) -> AnonymizerInput:
    """The input: the file's ``input`` section with the input flags over it, as the settings merged them."""
    if merged.get("source") is None:
        raise InvalidConfigError("--source is required" + (f" (or input.source in {path})" if path else ""))
    try:
        return AnonymizerInput.model_validate(merged)
    except ValidationError as exc:
        name_for = functools.partial(_flag_after_parse, arguments, mode=mode, given={"data": given}, path=path)
        raise InvalidConfigError(_describe(exc, lambda loc: name_for(("input", *loc)))) from None


class _NoFile(PydanticBaseSettingsSource):
    """No ``--config`` file: the flags alone."""

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        return {}


def _settings_error(exc: ValidationError, name_for: Any, path: Path | None) -> str:
    """The settings' validation errors, as the CLI names them. Errors in ``detect`` come first and alone, as they
    did when Detect was validated before the rest. A check of the whole Detect has no flag to name: without a file
    it reads "value", as it did at parse time; with a file it reads "detect"."""
    detect = [item for item in exc.errors() if item["loc"][:1] == ("detect",)]
    if detect:
        exc = ValidationError.from_exception_data(exc.title, detect)  # ty: ignore[invalid-argument-type]
    whole = "value" if path is None else "detect"
    return _describe(exc, lambda loc: whole if tuple(loc) == ("detect",) else name_for(loc))


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
    section = "rewrite" if mode == "rewrite" else "replace"
    if strategy is None:
        _missing_env(mode)
        strategy = MODES[mode]()
    flags = {"strategy": _given(strategy), "detect": _given(detect), "data": _given(data)}
    settings_class = _settings_class(mode)
    file = _read_config_file(path, mode, settings_class) if path is not None else None
    if file is not None:
        values = file()
        _drop_replaced(values.get(section) or {}, flags["strategy"], MODES[mode])
        _drop_replaced(values.get("detect") or {}, flags["detect"], Detect)
    settings_class._file = file if file is not None else _NoFile(settings_class)
    name_for = functools.partial(_flag_after_parse, arguments, mode=mode, given=flags, path=path)
    try:
        # A replace strategy's kind is set, as AnonymizerConfig's discriminator needs it in a dump.
        strategy_values = {**flags["strategy"], "kind": mode} if section == "replace" else flags["strategy"]
        given: dict[str, Any] = {"input": flags["data"], section: strategy_values, "detect": flags["detect"]}
        settings = settings_class(**given)
    except ValidationError as exc:
        raise InvalidConfigError(_settings_error(exc, name_for, path)) from None
    layered: dict[str, Any] = dict(settings)  # the validated fields, models kept as models
    try:
        config = AnonymizerConfig.model_validate(
            {section: layered[section], "detect": layered["detect"], "emit_telemetry": layered["emit_telemetry"]}
        )
    except ValidationError as exc:
        raise InvalidConfigError(_describe(exc, name_for)) from None
    if not options.emit_telemetry and config.emit_telemetry:
        config = config.model_copy(update={"emit_telemetry": False})
    return config, _input(arguments, mode, flags["data"], layered["input"] or {}, path)
