# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed loader for the private Phase 10 bounded-inspection contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from importlib.resources import files
from typing import TypeAlias, cast

_FrozenJson: TypeAlias = None | bool | int | str | tuple["_FrozenJson", ...] | tuple[tuple[str, "_FrozenJson"], ...]
_ContractValues: TypeAlias = tuple[
    str,
    str,
    tuple[tuple[str, int], ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    _FrozenJson,
]

_DIGEST = "0d6e189bf3d89472a6880a76367ed99b5462b6c5d460818282e403e4a285eb95"
_RESOURCE = "phase10_bounded_inspection_contract.json"
_ENVELOPE_KEYS = {"schema_version", "digest_algorithm", "digest", "contract"}
_SCHEMA_VERSION = "anonymizer-phase10-bounded-inspection-owner-contract-envelope/v1"
_DIGEST_ALGORITHM = "sha256_of_UTF8_compact_sorted_key_JSON_of_contract_member_with_no_trailing_newline"
_CONTRACT_VERSION = "anonymizer-phase10-bounded-inspection/v1"
_SEAL = object()


class _PrivatePhase10ContractValue:
    def __repr__(self) -> str:
        return f"<private {type(self).__name__.strip('_').replace('_', ' ').lower()}>"

    def __reduce__(self) -> str | tuple[object, ...]:
        raise TypeError("private Phase 10 contract values are not serializable")


@dataclass(frozen=True, slots=True, repr=False)
class _Phase10ContractProof(_PrivatePhase10ContractValue):
    seal: object = field(compare=False)
    snapshot: tuple[object, ...]


@dataclass(frozen=True, slots=True, repr=False)
class _Phase10BoundedInspectionContract(_PrivatePhase10ContractValue):
    digest: str
    version: str
    limits: tuple[tuple[str, int], ...]
    count_buckets: tuple[str, ...]
    byte_buckets: tuple[str, ...]
    capture_boundaries: tuple[str, ...]
    reason_categories: tuple[str, ...]
    provenance_fields: tuple[str, ...]
    _contract: _FrozenJson = field(compare=False)
    _proof: _Phase10ContractProof | None = field(default=None, compare=False)


@dataclass(frozen=True, slots=True, repr=False)
class _Phase10ContractRejected(_PrivatePhase10ContractValue):
    code: str = "contract_invalid"


def _parse_json(text: object) -> object:
    if type(text) is not str:
        raise TypeError
    return json.loads(text, object_pairs_hook=_object_without_duplicates)


def _object_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _read_contract_envelope() -> object:
    text = files("anonymizer.engine.execution").joinpath(_RESOURCE).read_text(encoding="utf-8")
    return _parse_json(text)


def _frozen_contract() -> dict[str, object] | None:
    try:
        envelope = _read_contract_envelope()
        if type(envelope) is not dict:
            return None
        contract = cast(dict[str, object], envelope).get("contract")
        return cast(dict[str, object], contract) if type(contract) is dict else None
    except (OSError, TypeError, ValueError):
        return None


_FROZEN_CONTRACT = _frozen_contract()


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _same_json_value(value: object, expected: object) -> bool:
    if type(value) is not type(expected):
        return False
    if type(value) is dict:
        actual = cast(dict[str, object], value)
        frozen = cast(dict[str, object], expected)
        return set(actual) == set(frozen) and all(_same_json_value(actual[key], frozen[key]) for key in frozen)
    if type(value) is list:
        actual_items = cast(list[object], value)
        frozen_items = cast(list[object], expected)
        return len(actual_items) == len(frozen_items) and all(
            _same_json_value(item, frozen_item) for item, frozen_item in zip(actual_items, frozen_items, strict=True)
        )
    return value == expected


def _freeze_json(value: object) -> _FrozenJson:
    if value is None or type(value) in {bool, int, str}:
        return cast(None | bool | int | str, value)
    if type(value) is list:
        return tuple(_freeze_json(item) for item in cast(list[object], value))
    if type(value) is dict:
        mapping = cast(dict[str, object], value)
        return tuple((key, _freeze_json(mapping[key])) for key in sorted(mapping))
    raise TypeError


def _require_dict(container: dict[str, object], key: str) -> dict[str, object]:
    value = container[key]
    if type(value) is not dict:
        raise TypeError
    return cast(dict[str, object], value)


def _require_strings(container: dict[str, object], key: str) -> tuple[str, ...]:
    value = container[key]
    if type(value) is not list or any(type(item) is not str for item in value):
        raise TypeError
    return tuple(cast(list[str], value))


def _integer_items(container: dict[str, object]) -> tuple[tuple[str, int], ...]:
    items = tuple((key, value) for key, value in sorted(container.items()) if type(value) is int)
    if not items or any(value < 0 for _key, value in items):
        raise TypeError
    return cast(tuple[tuple[str, int], ...], items)


def _contract_values(body: dict[str, object]) -> _ContractValues:
    privacy = _require_dict(body, "privacy_and_redaction")
    lifecycle = _require_dict(body, "lifecycle_and_cancellation")
    diagnostics = _require_dict(body, "failure_diagnostics")
    provenance = _require_dict(body, "provenance")
    return (
        _DIGEST,
        _CONTRACT_VERSION,
        _integer_items(_require_dict(body, "limits")),
        _require_strings(privacy, "count_buckets"),
        _require_strings(privacy, "byte_buckets"),
        _require_strings(lifecycle, "allowed_capture_points"),
        _require_strings(diagnostics, "reason_categories"),
        _require_strings(provenance, "required_fields"),
        _freeze_json(body),
    )


def _compile_phase10_contract(
    envelope: object,
) -> _Phase10BoundedInspectionContract | _Phase10ContractRejected:
    try:
        if type(envelope) is not dict or set(envelope) != _ENVELOPE_KEYS:
            raise TypeError
        data = cast(dict[str, object], envelope)
        contract = data["contract"]
        if not _valid_envelope(data, contract):
            raise ValueError
        values = _contract_values(cast(dict[str, object], contract))
        candidate = _contract_from_values(values)
        snapshot = _contract_snapshot(candidate)
        if snapshot is None:
            raise TypeError
        return _contract_from_values(values, _Phase10ContractProof(_SEAL, snapshot))
    except (KeyError, TypeError, UnicodeError, ValueError):
        return _Phase10ContractRejected()


def _contract_from_values(
    values: _ContractValues,
    proof: _Phase10ContractProof | None = None,
) -> _Phase10BoundedInspectionContract:
    digest, version, limits, counts, bytes_, boundaries, reasons, provenance, frozen = values
    return _Phase10BoundedInspectionContract(
        digest=digest,
        version=version,
        limits=limits,
        count_buckets=counts,
        byte_buckets=bytes_,
        capture_boundaries=boundaries,
        reason_categories=reasons,
        provenance_fields=provenance,
        _contract=frozen,
        _proof=proof,
    )


def _valid_envelope(data: dict[str, object], contract: object) -> bool:
    return (
        data["schema_version"] == _SCHEMA_VERSION
        and data["digest_algorithm"] == _DIGEST_ALGORITHM
        and type(contract) is dict
        and data["digest"] == _DIGEST
        and _canonical_digest(contract) == _DIGEST
        and _FROZEN_CONTRACT is not None
        and _same_json_value(contract, _FROZEN_CONTRACT)
        and cast(dict[str, object], contract).get("version") == _CONTRACT_VERSION
        and cast(dict[str, object], contract).get("sdk_phase") == 10
    )


def _contract_snapshot(contract: _Phase10BoundedInspectionContract) -> tuple[object, ...] | None:
    try:
        return (
            contract.digest,
            contract.version,
            contract.limits,
            contract.count_buckets,
            contract.byte_buckets,
            contract.capture_boundaries,
            contract.reason_categories,
            contract.provenance_fields,
            contract._contract,
        )
    except (AttributeError, TypeError):
        return None


def _load_phase10_contract() -> _Phase10BoundedInspectionContract | _Phase10ContractRejected:
    try:
        return _compile_phase10_contract(_read_contract_envelope())
    except (OSError, TypeError, ValueError):
        return _Phase10ContractRejected()


def _is_admitted_phase10_contract(value: object) -> bool:
    if not isinstance(value, _Phase10BoundedInspectionContract) or value._proof is None:
        return False
    return (
        value._proof.seal is _SEAL
        and value.digest == _DIGEST
        and value.version == _CONTRACT_VERSION
        and value._proof.snapshot == _contract_snapshot(value)
    )
