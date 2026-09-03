# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frozen owner contract loader for the private Phase 8 profile."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from importlib.resources import files
from math import isfinite
from typing import cast

_DIGEST = "597a410aee8cb8ca428e82737f385213ce9ce47eae216caea68ebc2f9907d227"
_RESOURCE = "phase8_grouped_rewrite_contract.json"
_SEAL = object()


class _PrivatePhase8ContractValue:
    def __repr__(self) -> str:
        return f"<private {type(self).__name__.strip('_').replace('_', ' ').lower()}>"

    def __reduce__(self) -> str | tuple[object, ...]:
        raise TypeError("private Phase 8 contract values are not serializable")


@dataclass(frozen=True, slots=True, repr=False)
class _Phase8GroupedRewriteContract(_PrivatePhase8ContractValue):
    digest: str
    version: str
    limits: tuple[tuple[str, int], ...]
    _contract: tuple[tuple[str, object], ...] = field(compare=False)
    _proof: object | None = field(default=None, compare=False)


@dataclass(frozen=True, slots=True, repr=False)
class _Phase8ContractRejected(_PrivatePhase8ContractValue):
    code: str = "contract_invalid"


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


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


def _compile_phase8_contract(envelope: object) -> _Phase8GroupedRewriteContract | _Phase8ContractRejected:
    try:
        if type(envelope) is not dict or set(envelope) != {"schema_version", "digest_algorithm", "digest", "contract"}:
            raise TypeError
        data = cast(dict[str, object], envelope)
        contract = data["contract"]
        if type(contract) is not dict or data["digest"] != _DIGEST or _canonical_digest(contract) != _DIGEST:
            raise ValueError
        body = cast(dict[str, object], contract)
        if body.get("version") != "anonymizer-phase8-grouped-rewrite/v1":
            raise ValueError
        limits = body.get("scheduling_and_limits")
        if type(limits) is not dict:
            raise TypeError
        integer_limits = tuple(sorted((key, value) for key, value in cast(dict[str, object], limits).items() if type(value) is int))
        if dict(integer_limits).get("max_repair_iterations") != 3 or dict(integer_limits).get("max_members_per_rewrite_group") != 4:
            raise ValueError
        return _Phase8GroupedRewriteContract(_DIGEST, cast(str, body["version"]), integer_limits, cast(tuple[tuple[str, object], ...], _freeze(body)), _SEAL)
    except (KeyError, TypeError, ValueError, UnicodeError):
        return _Phase8ContractRejected()


def _load_phase8_contract() -> _Phase8GroupedRewriteContract | _Phase8ContractRejected:
    try:
        return _compile_phase8_contract(json.loads(files("anonymizer.engine.execution").joinpath(_RESOURCE).read_text(encoding="utf-8")))
    except (OSError, TypeError, ValueError):
        return _Phase8ContractRejected()


def _is_admitted_phase8_contract(value: object) -> bool:
    return isinstance(value, _Phase8GroupedRewriteContract) and value._proof is _SEAL and value.digest == _DIGEST
