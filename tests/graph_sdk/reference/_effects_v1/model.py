# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent effects_v1 reference: model."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from typing import TypeAlias, cast

Json: TypeAlias = str | int | bool | None | list["Json"] | dict[str, "Json"]


Object: TypeAlias = dict[str, Json]


MappingKey: TypeAlias = tuple[str, str | None, str | None]


CONTRACT_SHA256 = "9b0ab07b8c0212ffd954dc37eb37540141da6899753fc778ad26238494aeaeca"


REQUEST_BOUNDARY_ADDENDUM_SHA256 = "6344f7f1bdbb14a4c9e08f546928d31c89f26d3ed26e0c50362d9537d891c3a5"


GENERATOR_VERSION = "effects-v1-generator-18-owner-complete-latest"


SELF_TEST_VERSION = "effects-v1-self-test-18-owner-complete-latest"


MATERIALIZATION_ADDENDUM_SHA256 = "b1a5651ee2649b01c89e80bd1442e3f03209292b48ce846d27714698f57eb07c"


BASE_CORPUS_SHA256 = "c62f2cc7e7237ea030451ac8d35a3c30b39f71766b4275348597949d560a7a6a"


PREDECESSOR_CASE_COUNT = 214


PREDECESSOR_CORPUS_SHA256 = "b509a2e1dc697ae5e16c8f2f9652aac74d862e4b449f64958fbba99abda87e93"


OPTIONAL_OMISSION_ADDENDUM_SHA256 = "543d46e66988e11e6386ecdc686a77207b74e42211b8b164bc1decbb96e5d93e"


MAP_EXECUTION_ADDENDUM_SHA256 = "9563fd44c040d546352853de03e57bcc94a6e34b7772ca2d9239f84074bef41d"


BINDING_SUCCESS_ADDENDUM_SHA256 = "c6689c78f712837072235ad8343de8bd8240ea2c7e8580d1594e055de2e017cd"


MATERIALIZED_VERSION_CONTRACT_SHA256 = "165c7c95bce31a7c5808860f28d012bbe1986bf0712cebdc86fb08ad07afcb21"


ACCEPTED_PREDECESSOR_CORPUS_SHA256 = "d56c9c64367ca9f4aa211aeb0e1bc8c5c213c977af07f906eced572764bc7726"


ACCEPTED_PREDECESSOR_CASE_COUNT = 296


CORPUS_PATH = "future-contracts/r2-version-selection-v9/effects_v1_cases.json"


FAMILIES = (
    "budgets",
    "keyed",
    "retry",
    "races",
    "inflight",
    "binding",
    "resources",
    "bridges",
    "decisions",
    "admission",
    "materialization",
    "map",
)


FAILURE_CLASSES = (
    "rejected_before_acceptance",
    "retryable",
    "malformed_response",
    "permanent",
    "transport_unknown",
    "implementation_exception",
)


RUNTIME_CONDITIONS = (
    "result",
    "failure",
    "cancel_before_start",
    "cancel_after_start",
    "cancel_after_dispatch",
    "lost",
    "request_inconsistent",
    "budget_exhausted",
    "request_limit_exhausted",
    "artifact_limit_exhausted",
    "deadline_exhausted",
)


POLICY_CONDITIONS = {
    "local": (
        "cancel_before_start",
        "cancel_after_start",
        "artifact_limit_exhausted",
        "deadline_exhausted",
    ),
    "external": (
        "cancel_before_start",
        "cancel_after_start",
        "cancel_after_dispatch",
        "lost",
        "request_inconsistent",
        "budget_exhausted",
        "request_limit_exhausted",
        "artifact_limit_exhausted",
        "deadline_exhausted",
    ),
    "decision": (
        "cancel_before_start",
        "cancel_after_start",
        "artifact_limit_exhausted",
        "deadline_exhausted",
    ),
}


def _mapping_key(mapping: Object) -> MappingKey:
    return (
        cast(str, mapping.get("condition")),
        cast(str | None, mapping.get("reported_outcome")),
        cast(str | None, mapping.get("failure")),
    )


def _expected_mapping_keys(kind: str, outcomes: Sequence[str]) -> set[MappingKey]:
    keys: set[MappingKey] = {("result", outcome, None) for outcome in outcomes}
    failures = ("permanent", "implementation_exception") if kind == "decision" else FAILURE_CLASSES
    keys.update(("failure", None, failure) for failure in failures)
    keys.update((condition, None, None) for condition in POLICY_CONDITIONS[kind])
    return keys


def _object(value: Json) -> Object:
    if not isinstance(value, dict):
        raise TypeError(f"Expected object, got {type(value)!r}")
    return value


def _array(value: Json) -> list[Json]:
    if not isinstance(value, list):
        raise TypeError(f"Expected array, got {type(value)!r}")
    return value


def _strings(value: Json) -> list[str]:
    return [cast(str, item) for item in _array(value)]


def canonical_bytes(cases: Iterable[Object]) -> bytes:
    return (json.dumps(tuple(cases), indent=2, sort_keys=True) + "\n").encode()


def load_cases(value: object) -> tuple[Object, ...]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise TypeError("effects corpus must be a list of objects")
    return tuple(cast(Object, item) for item in value)
