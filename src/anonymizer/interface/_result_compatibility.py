# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private legacy pandas result-compatibility materialization."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from importlib.resources import files
from math import isfinite
from typing import cast

import pandas as pd

from anonymizer.config.anonymizer_config import AnonymizerConfig
from anonymizer.config.replace_strategies import ReplaceMethod
from anonymizer.config.rewrite import PrivacyGoal
from anonymizer.engine.constants import (
    COL_ANY_HIGH_LEAKED,
    COL_ATTRIBUTE_FIDELITY_INVALID_ENTITIES,
    COL_ATTRIBUTE_FIDELITY_VALID,
    COL_DETECTION_INVALID_ENTITIES,
    COL_DETECTION_VALID,
    COL_ENTITY_COVERAGE,
    COL_FINAL_ENTITIES,
    COL_JUDGE_EVALUATION,
    COL_LEAKAGE_MASS,
    COL_MISSED_ENTITIES,
    COL_NEEDS_HUMAN_REVIEW,
    COL_RELATIONAL_CONSISTENCY_INVALID_RELATIONS,
    COL_RELATIONAL_CONSISTENCY_VALID,
    COL_REPLACED_TEXT,
    COL_REWRITTEN_TEXT,
    COL_TAGGED_TEXT,
    COL_TEXT,
    COL_TYPE_FIDELITY_INVALID_REPLACEMENTS,
    COL_TYPE_FIDELITY_VALID,
    COL_UTILITY_SCORE,
    COL_WEIGHTED_LEAKAGE_RATE,
)
from anonymizer.engine.ndd.adapter import FailedRecord
from anonymizer.interface.results import AnonymizerResult, PreviewResult

_DIGEST = "c91a410289c3549f608cc0b088da3ce9db56ac10aeabe430a8254b637ef4b12d"
_RESOURCE = "result_compatibility_contract.json"
_SEAL = object()
_ENVELOPE_KEYS = {"schema_version", "digest_algorithm", "digest", "contract"}
_SCHEMA_VERSION = "anonymizer-phase9-result-compatibility-owner-contract-envelope/v1"
_DIGEST_ALGORITHM = "sha256_of_UTF8_compact_sorted_key_JSON_of_contract_member_with_no_trailing_newline"
_CONTRACT_VERSION = "result-compatibility-v1"


class _PrivateResultCompatibilityContractValue:
    def __repr__(self) -> str:
        return f"<private {type(self).__name__.strip('_').replace('_', ' ').lower()}>"

    def __reduce__(self) -> str | tuple[object, ...]:
        raise TypeError("private result compatibility contract values are not serializable")


@dataclass(frozen=True, slots=True, repr=False)
class _ResultCompatibilityContract(_PrivateResultCompatibilityContractValue):
    digest: str
    version: str
    _contract: tuple[tuple[str, object], ...] = field(compare=False)
    _proof: object | None = field(default=None, compare=False)


@dataclass(frozen=True, slots=True, repr=False)
class _ResultCompatibilityContractRejected(_PrivateResultCompatibilityContractValue):
    code: str = "contract_invalid"


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _same_json_value(value: object, expected: object) -> bool:
    if type(value) is not type(expected):
        return False
    if type(value) is dict:
        actual_items = cast(dict[str, object], value)
        expected_items = cast(dict[str, object], expected)
        return set(actual_items) == set(expected_items) and all(
            _same_json_value(actual_items[key], expected_items[key]) for key in expected_items
        )
    if type(value) is list:
        actual_items = cast(list[object], value)
        expected_items = cast(list[object], expected)
        return len(actual_items) == len(expected_items) and all(
            _same_json_value(item, expected_item)
            for item, expected_item in zip(actual_items, expected_items, strict=True)
        )
    return value == expected


def _read_contract_envelope() -> object:
    return json.loads(files("anonymizer.interface").joinpath(_RESOURCE).read_text(encoding="utf-8"))


def _frozen_contract() -> dict[str, object] | None:
    try:
        envelope = _read_contract_envelope()
        if type(envelope) is not dict or type(envelope.get("contract")) is not dict:
            return None
        data = cast(dict[str, object], envelope)
        return cast(dict[str, object], data["contract"])
    except (OSError, TypeError, ValueError):
        return None


_FROZEN_CONTRACT = _frozen_contract()


def _freeze(value: object) -> object:
    if type(value) is dict:
        return tuple((key, _freeze(item)) for key, item in sorted(cast(dict[str, object], value).items()))
    if type(value) is list:
        return tuple(_freeze(item) for item in cast(list[object], value))
    if type(value) is float:
        if not isfinite(value):
            raise TypeError
        return value
    if type(value) in {str, int, bool} or value is None:
        return value
    raise TypeError


def _compile_result_compatibility_contract(
    envelope: object,
) -> _ResultCompatibilityContract | _ResultCompatibilityContractRejected:
    try:
        if type(envelope) is not dict or set(envelope) != _ENVELOPE_KEYS:
            raise TypeError
        data = cast(dict[str, object], envelope)
        contract = data["contract"]
        if (
            data["schema_version"] != _SCHEMA_VERSION
            or data["digest_algorithm"] != _DIGEST_ALGORITHM
            or type(contract) is not dict
            or data["digest"] != _DIGEST
            or _canonical_digest(contract) != _DIGEST
            or _FROZEN_CONTRACT is None
            or not _same_json_value(contract, _FROZEN_CONTRACT)
        ):
            raise ValueError
        body = cast(dict[str, object], contract)
        if body.get("version") != _CONTRACT_VERSION:
            raise ValueError
        return _ResultCompatibilityContract(
            digest=_DIGEST,
            version=_CONTRACT_VERSION,
            _contract=cast(tuple[tuple[str, object], ...], _freeze(body)),
            _proof=_SEAL,
        )
    except (KeyError, TypeError, ValueError, UnicodeError):
        return _ResultCompatibilityContractRejected()


def _load_result_compatibility_contract() -> _ResultCompatibilityContract | _ResultCompatibilityContractRejected:
    try:
        return _compile_result_compatibility_contract(_read_contract_envelope())
    except (OSError, TypeError, ValueError):
        return _ResultCompatibilityContractRejected()


def _is_admitted_result_compatibility_contract(value: object) -> bool:
    return (
        isinstance(value, _ResultCompatibilityContract)
        and value._proof is _SEAL
        and value.digest == _DIGEST
        and value.version == _CONTRACT_VERSION
    )


def _require_result_compatibility_contract() -> None:
    if not _is_admitted_result_compatibility_contract(_load_result_compatibility_contract()):
        raise RuntimeError("result compatibility contract is unavailable")


def _require_dataframe(value: object) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise TypeError("result materialization requires a pandas DataFrame")
    return value


def _rename_output_columns(df: pd.DataFrame, *, resolved_text_column: str) -> pd.DataFrame:
    """Rename internal column names to user-facing names."""
    dataframe = _require_dataframe(df)
    rename_map: dict[str, str] = {}
    if COL_TEXT in dataframe.columns:
        rename_map[COL_TEXT] = resolved_text_column
    if COL_REPLACED_TEXT in dataframe.columns:
        rename_map[COL_REPLACED_TEXT] = f"{resolved_text_column}_replaced"
    if COL_TAGGED_TEXT in dataframe.columns:
        rename_map[COL_TAGGED_TEXT] = f"{resolved_text_column}_with_spans"
    if COL_REWRITTEN_TEXT in dataframe.columns:
        rename_map[COL_REWRITTEN_TEXT] = f"{resolved_text_column}_rewritten"
    if not rename_map:
        return dataframe
    return dataframe.rename(columns=rename_map)


def _unrename_output_columns(df: pd.DataFrame, *, resolved_text_column: str) -> pd.DataFrame:
    """Reverse the four known public output names for evaluation."""
    dataframe = _require_dataframe(df)
    if COL_TEXT in dataframe.columns:
        return dataframe
    rename_map: dict[str, str] = {}
    if resolved_text_column in dataframe.columns:
        rename_map[resolved_text_column] = COL_TEXT
    if f"{resolved_text_column}_replaced" in dataframe.columns:
        rename_map[f"{resolved_text_column}_replaced"] = COL_REPLACED_TEXT
    if f"{resolved_text_column}_with_spans" in dataframe.columns:
        rename_map[f"{resolved_text_column}_with_spans"] = COL_TAGGED_TEXT
    if f"{resolved_text_column}_rewritten" in dataframe.columns:
        rename_map[f"{resolved_text_column}_rewritten"] = COL_REWRITTEN_TEXT
    if not rename_map:
        return dataframe
    return dataframe.rename(columns=rename_map)


def _build_user_dataframe(
    trace_dataframe: pd.DataFrame,
    *,
    resolved_text_column: str,
    compute_detection_validity: bool = False,
) -> pd.DataFrame:
    """Copy the active mode's public columns in trace-column order."""
    trace = _require_dataframe(trace_dataframe)
    text_column = resolved_text_column

    if f"{text_column}_rewritten" in trace.columns:
        allowed = {
            text_column,
            f"{text_column}_rewritten",
            COL_UTILITY_SCORE,
            COL_LEAKAGE_MASS,
            COL_WEIGHTED_LEAKAGE_RATE,
            COL_ANY_HIGH_LEAKED,
            COL_NEEDS_HUMAN_REVIEW,
            COL_JUDGE_EVALUATION,
            COL_ENTITY_COVERAGE,
            COL_MISSED_ENTITIES,
        }
        if compute_detection_validity:
            allowed |= {COL_DETECTION_VALID, COL_DETECTION_INVALID_ENTITIES}
    elif f"{text_column}_replaced" in trace.columns:
        allowed = {
            text_column,
            f"{text_column}_replaced",
            f"{text_column}_with_spans",
            COL_FINAL_ENTITIES,
            COL_ENTITY_COVERAGE,
            COL_MISSED_ENTITIES,
            COL_TYPE_FIDELITY_VALID,
            COL_TYPE_FIDELITY_INVALID_REPLACEMENTS,
            COL_RELATIONAL_CONSISTENCY_VALID,
            COL_RELATIONAL_CONSISTENCY_INVALID_RELATIONS,
            COL_ATTRIBUTE_FIDELITY_VALID,
            COL_ATTRIBUTE_FIDELITY_INVALID_ENTITIES,
        }
        if compute_detection_validity:
            allowed |= {COL_DETECTION_VALID, COL_DETECTION_INVALID_ENTITIES}
    else:
        allowed = {
            text_column,
            f"{text_column}_with_spans",
            COL_FINAL_ENTITIES,
        }

    return trace[[column for column in trace.columns if column in allowed]].copy()


def _materialize_run_result(
    dataframe: pd.DataFrame,
    *,
    config: AnonymizerConfig,
    resolved_text_column: str,
    failed_records: list[FailedRecord],
    data_summary: str | None,
) -> AnonymizerResult:
    """Materialize one verified legacy run result."""
    _require_result_compatibility_contract()
    trace = _rename_output_columns(dataframe, resolved_text_column=resolved_text_column)
    return AnonymizerResult(
        dataframe=_build_user_dataframe(trace, resolved_text_column=resolved_text_column),
        trace_dataframe=trace,
        resolved_text_column=resolved_text_column,
        failed_records=failed_records,
        replace_method=config.replace,
        rewrite_config=config.rewrite.privacy_goal if config.rewrite is not None else None,
        entity_labels=config.detect.entity_labels,
        data_summary=data_summary,
    )


def _materialize_preview_result(
    result: AnonymizerResult,
    *,
    config: AnonymizerConfig,
    preview_num_records: int,
) -> PreviewResult:
    """Wrap an already-materialized run result as a preview result."""
    _require_result_compatibility_contract()
    if not isinstance(result, AnonymizerResult):
        raise TypeError("preview materialization requires an AnonymizerResult")
    return PreviewResult(
        dataframe=result.dataframe,
        trace_dataframe=result.trace_dataframe,
        resolved_text_column=result.resolved_text_column,
        failed_records=result.failed_records,
        preview_num_records=preview_num_records,
        replace_method=config.replace,
        rewrite_config=config.rewrite.privacy_goal if config.rewrite is not None else None,
        entity_labels=config.detect.entity_labels,
        data_summary=result.data_summary,
    )


def _materialize_evaluation_result(
    dataframe: pd.DataFrame,
    *,
    resolved_text_column: str,
    failed_records: list[FailedRecord],
    compute_detection_validity: bool,
    replace_method: ReplaceMethod | None = None,
    rewrite_config: PrivacyGoal | None = None,
    entity_labels: list[str] | None = None,
    data_summary: str | None = None,
) -> AnonymizerResult:
    """Materialize one already-judged legacy evaluation dataframe."""
    _require_result_compatibility_contract()
    trace = _rename_output_columns(dataframe, resolved_text_column=resolved_text_column)
    return AnonymizerResult(
        dataframe=_build_user_dataframe(
            trace,
            resolved_text_column=resolved_text_column,
            compute_detection_validity=compute_detection_validity,
        ),
        trace_dataframe=trace,
        resolved_text_column=resolved_text_column,
        failed_records=failed_records,
        replace_method=replace_method,
        rewrite_config=rewrite_config,
        entity_labels=entity_labels,
        data_summary=data_summary,
    )
