# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frozen owner contract for the private Phase 7 stable-Substitute profile."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from importlib.resources import files
from typing import TypeAlias, cast

from anonymizer.engine.constants import DEFAULT_ENTITY_LABELS

_FrozenJson: TypeAlias = None | bool | int | str | tuple["_FrozenJson", ...] | tuple[tuple[str, "_FrozenJson"], ...]


class _PrivatePhase7ContractValue:
    def __repr__(self) -> str:
        return f"<private {type(self).__name__.strip('_').replace('_', ' ').lower()}>"

    def __reduce__(self) -> str | tuple[object, ...]:
        raise TypeError("private Phase 7 contract values are not serializable")


class _Phase7ContractVersion(str, Enum):
    V1 = "anonymizer-phase7-stable-substitute/v1"


class _Phase7ContractRejectionCode(str, Enum):
    INVALID_CONTRACT = "contract_invalid"


@dataclass(frozen=True, slots=True, repr=False)
class _Phase7Role(_PrivatePhase7ContractValue):
    name: str
    format: str
    mask: str


@dataclass(frozen=True, slots=True, repr=False)
class _Phase7ContractProof(_PrivatePhase7ContractValue):
    seal: object = field(compare=False)
    snapshot: tuple[object, ...]


@dataclass(frozen=True, slots=True, repr=False)
class _Phase7StableSubstituteContract(_PrivatePhase7ContractValue):
    version: _Phase7ContractVersion
    digest: str
    phase6_result_version: str
    phase6_policy_version: str
    phase6_policy_digest: str
    roles: tuple[_Phase7Role, ...]
    selectors: tuple[str, ...]
    relations: tuple[str, ...]
    formats: tuple[str, ...]
    masks: tuple[str, ...]
    count_limits: tuple[tuple[str, int], ...]
    byte_limits: tuple[tuple[str, int], ...]
    corpus_version: str
    corpus_case_count: int
    corpus_digest: str
    _source_snapshot: _FrozenJson = field(compare=False)
    _proof: _Phase7ContractProof | None = field(default=None, compare=False)


@dataclass(frozen=True, slots=True, repr=False)
class _Phase7ContractRejected(_PrivatePhase7ContractValue):
    code: _Phase7ContractRejectionCode = _Phase7ContractRejectionCode.INVALID_CONTRACT


_PHASE7_CONTRACT_SEAL = object()
_PHASE7_CONTRACT_RESOURCE = "phase7_stable_substitute_contract.json"
_PHASE7_POLICY_RESOURCE = "phase6_substitute_role_policy.json"
_PHASE7_CORPUS_RESOURCE = "phase7_owner_contract_corpus.json"
_PHASE7_CONTRACT_DIGEST = "3755832ecc64fe6e9dbeccc136c40020e5158446e5c74d6dca2392efdbb006bb"
_PHASE7_POLICY_DIGEST = "c27580bd2cc4051bdd11b63a91391f8995bdef1ed2052534623cdd3160318ef8"
_PHASE7_CORPUS_DIGEST = "5ba1e69c7836a428b16e9d2ac7fc0cf3fbb445fd0ca72c1d0a8a049194115123"
_ENVELOPE_SCHEMA_VERSION = "anonymizer-phase7-owner-contract-envelope/v1"
_DIGEST_ALGORITHM = "sha256_of_UTF8_compact_sorted_key_JSON_of_contract_member_with_no_trailing_newline"


def _load_phase7_contract() -> _Phase7StableSubstituteContract | _Phase7ContractRejected:
    """Load the exact owner-frozen contract and its companion resources."""
    try:
        package = files("anonymizer.engine.execution")
        envelope = _parse_json(package.joinpath(_PHASE7_CONTRACT_RESOURCE).read_text(encoding="utf-8"))
        policy = _parse_json(package.joinpath(_PHASE7_POLICY_RESOURCE).read_text(encoding="utf-8"))
        corpus = _parse_json(package.joinpath(_PHASE7_CORPUS_RESOURCE).read_text(encoding="utf-8"))
        return _compile_phase7_contract(envelope, policy, corpus)
    except (OSError, TypeError, UnicodeEncodeError, ValueError):
        return _Phase7ContractRejected()


def _compile_phase7_contract(
    envelope: object,
    policy: object,
    corpus: object,
) -> _Phase7StableSubstituteContract | _Phase7ContractRejected:
    """Validate and seal only the exact contract approved by the owners."""
    try:
        if type(envelope) is not dict or set(envelope) != {
            "contract",
            "digest",
            "digest_algorithm",
            "schema_version",
        }:
            raise TypeError
        envelope_dict = cast(dict[str, object], envelope)
        contract = envelope_dict["contract"]
        digest = envelope_dict["digest"]
        if (
            type(contract) is not dict
            or type(digest) is not str
            or envelope_dict["digest_algorithm"] != _DIGEST_ALGORITHM
            or envelope_dict["schema_version"] != _ENVELOPE_SCHEMA_VERSION
            or _canonical_digest(contract) != digest
            or digest != _PHASE7_CONTRACT_DIGEST
        ):
            raise ValueError
        contract_dict = cast(dict[str, object], contract)
        if contract_dict["status"] != "frozen_owner_contract":
            raise ValueError
        version = _Phase7ContractVersion(contract_dict["version"])

        roles_payload = _require_dict(contract_dict, "roles")
        roles = tuple(
            _Phase7Role(
                name,
                _require_string(_require_dict(roles_payload, name), "format"),
                _require_string(_require_dict(roles_payload, name), "mask"),
            )
            for name in sorted(roles_payload)
        )
        formats = _closed_keys(contract_dict, "formats", excluded=())
        masks = _closed_keys(contract_dict, "masks", excluded=("unknown",))
        selectors = _closed_keys(contract_dict, "selectors", excluded=("unknown",))
        relations = _closed_keys(contract_dict, "relations", excluded=("unknown", "wildcard_constraints"))
        count_limits = _positive_integer_items(_require_dict(_require_dict(contract_dict, "limits"), "counts"))
        byte_limits = _positive_integer_items(_require_dict(_require_dict(contract_dict, "limits"), "bytes"))

        handoff = _require_dict(contract_dict, "phase6_handoff")
        policy_contract = _require_dict(handoff, "substitute_policy")
        _validate_policy(policy, policy_contract, roles)
        corpus_contract = _require_dict(contract_dict, "oracle_contract_corpus")
        _validate_corpus(corpus, corpus_contract)

        values = (
            version,
            digest,
            _require_string(handoff, "required_result_version"),
            _require_string(policy_contract, "version"),
            _require_string(policy_contract, "digest"),
            roles,
            selectors,
            relations,
            formats,
            masks,
            count_limits,
            byte_limits,
            _require_string(corpus_contract, "version"),
            _require_integer(corpus_contract, "case_count"),
            _require_string(corpus_contract, "digest"),
            (_freeze_json(contract_dict), _freeze_json(policy), _freeze_json(corpus)),
        )
        candidate = _Phase7StableSubstituteContract(*values)
        snapshot = _phase7_contract_snapshot(candidate)
        if snapshot is None:
            raise TypeError
        return _Phase7StableSubstituteContract(
            *values,
            _Phase7ContractProof(_PHASE7_CONTRACT_SEAL, snapshot),
        )
    except (KeyError, TypeError, UnicodeEncodeError, ValueError):
        return _Phase7ContractRejected()


def _is_admitted_phase7_contract(value: object) -> bool:
    if not isinstance(value, _Phase7StableSubstituteContract) or value._proof is None:
        return False
    return (
        value._proof.seal is _PHASE7_CONTRACT_SEAL
        and value.digest == _PHASE7_CONTRACT_DIGEST
        and value._proof.snapshot == _phase7_contract_snapshot(value)
    )


def _validate_policy(
    policy: object,
    expected: dict[str, object],
    roles: tuple[_Phase7Role, ...],
) -> None:
    if type(policy) is not dict:
        raise TypeError
    policy_dict = cast(dict[str, object], policy)
    if (
        set(policy_dict) != {"dispositions", "result_version", "version"}
        or _canonical_digest(policy_dict) != _PHASE7_POLICY_DIGEST
        or _require_string(expected, "digest") != _PHASE7_POLICY_DIGEST
        or policy_dict["version"] != expected["version"]
        or policy_dict["result_version"] != "phase6-role-result/v1"
    ):
        raise ValueError
    dispositions = _require_dict(policy_dict, "dispositions")
    if set(dispositions) != set(DEFAULT_ENTITY_LABELS) or len(dispositions) != len(DEFAULT_ENTITY_LABELS):
        raise ValueError
    role_names = {role.name for role in roles}
    classified = {label: role for label, role in dispositions.items() if role is not None}
    if (
        any(type(role) is not str or role not in role_names for role in classified.values())
        or set(classified.values()) != role_names
        or sorted(classified) != expected["supported_detector_labels"]
        or _require_integer(expected, "detector_label_count") != len(dispositions)
    ):
        raise ValueError


def _validate_corpus(corpus: object, expected: dict[str, object]) -> None:
    if type(corpus) is not dict:
        raise TypeError
    corpus_dict = cast(dict[str, object], corpus)
    if (
        set(corpus_dict) != {"cases", "version"}
        or _canonical_digest(corpus_dict) != _PHASE7_CORPUS_DIGEST
        or _require_string(expected, "digest") != _PHASE7_CORPUS_DIGEST
        or corpus_dict["version"] != expected["version"]
    ):
        raise ValueError
    cases = corpus_dict["cases"]
    if type(cases) is not list or len(cases) != _require_integer(expected, "case_count"):
        raise ValueError
    case_list = cast(list[object], cases)
    case_ids: set[str] = set()
    for case in case_list:
        if type(case) is not dict or set(case) != {"expected", "id"}:
            raise TypeError
        case_dict = cast(dict[str, object], case)
        case_id = _require_string(case_dict, "id")
        _require_string(case_dict, "expected")
        if case_id in case_ids:
            raise ValueError
        case_ids.add(case_id)


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


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _closed_keys(container: dict[str, object], key: str, *, excluded: tuple[str, ...]) -> tuple[str, ...]:
    values = _require_dict(container, key)
    return tuple(sorted(name for name in values if name not in excluded))


def _positive_integer_items(values: dict[str, object]) -> tuple[tuple[str, int], ...]:
    result: list[tuple[str, int]] = []
    for key in sorted(values):
        value = values[key]
        if type(value) is not int or value <= 0:
            raise TypeError
        result.append((key, value))
    return tuple(result)


def _require_dict(container: dict[str, object], key: str) -> dict[str, object]:
    value = container[key]
    if type(value) is not dict:
        raise TypeError
    return cast(dict[str, object], value)


def _require_string(container: dict[str, object], key: str) -> str:
    value = container[key]
    if type(value) is not str:
        raise TypeError
    return value


def _require_integer(container: dict[str, object], key: str) -> int:
    value = container[key]
    if type(value) is not int:
        raise TypeError
    return value


def _freeze_json(value: object) -> _FrozenJson:
    if value is None or type(value) in {bool, int, str}:
        return cast(None | bool | int | str, value)
    if type(value) is list:
        return tuple(_freeze_json(item) for item in cast(list[object], value))
    if type(value) is dict:
        mapping = cast(dict[str, object], value)
        if any(type(key) is not str for key in mapping):
            raise TypeError
        return tuple((key, _freeze_json(mapping[key])) for key in sorted(mapping))
    raise TypeError


def _phase7_contract_snapshot(contract: _Phase7StableSubstituteContract) -> tuple[object, ...] | None:
    try:
        return (
            contract.version.value,
            contract.digest,
            contract.phase6_result_version,
            contract.phase6_policy_version,
            contract.phase6_policy_digest,
            tuple((role.name, role.format, role.mask) for role in contract.roles),
            contract.selectors,
            contract.relations,
            contract.formats,
            contract.masks,
            contract.count_limits,
            contract.byte_limits,
            contract.corpus_version,
            contract.corpus_case_count,
            contract.corpus_digest,
            contract._source_snapshot,
        )
    except (AttributeError, TypeError):
        return None
