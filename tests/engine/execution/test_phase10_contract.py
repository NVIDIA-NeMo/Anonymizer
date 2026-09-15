# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import pickle
from dataclasses import FrozenInstanceError, replace
from importlib.resources import files
from pathlib import Path
from typing import Any, Callable, cast

import pytest

_CONTRACT_DIGEST = "0d6e189bf3d89472a6880a76367ed99b5462b6c5d460818282e403e4a285eb95"
_CONTRACT_RAW_DIGEST = "1b2bf397cfaed7d74b4ec2e0bb5db428d59a9ea24679faa7e2056705f00528bc"
_P9_DIGEST = "c91a410289c3549f608cc0b088da3ce9db56ac10aeabe430a8254b637ef4b12d"


def _module() -> Any:
    return importlib.import_module("anonymizer.engine.execution.phase10_contract")


def _resource_bytes() -> bytes:
    return files("anonymizer.engine.execution").joinpath("phase10_bounded_inspection_contract.json").read_bytes()


def _envelope() -> dict[str, object]:
    return cast(dict[str, object], json.loads(_resource_bytes()))


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_phase10_contract_resource_has_exact_approved_raw_and_member_digests() -> None:
    resource = _resource_bytes()
    envelope = _envelope()

    assert hashlib.sha256(resource).hexdigest() == _CONTRACT_RAW_DIGEST
    assert set(envelope) == {"schema_version", "digest_algorithm", "digest", "contract"}
    assert envelope["digest"] == _canonical_digest(envelope["contract"]) == _CONTRACT_DIGEST


def test_phase10_contract_loader_admits_only_the_frozen_member() -> None:
    module = _module()
    contract = module._load_phase10_contract()

    assert module._is_admitted_phase10_contract(contract)
    assert contract.digest == _CONTRACT_DIGEST
    assert contract.version == "anonymizer-phase10-bounded-inspection/v1"
    assert dict(contract.limits)["max_canonical_json_utf8_bytes"] == 16_384
    assert contract.count_buckets == ("0", "1", "2-4", "5-16", "17-64", "65+")
    assert contract.capture_boundaries[-1] == "invocation_closed"
    assert contract.reason_categories[-1] == "unexpected_failure"


def test_phase10_contract_freezes_fixed_arity_payload_limits_and_distinct_publication_points() -> None:
    body = cast(dict[str, object], _envelope()["contract"])
    scope = cast(dict[str, object], body["scope"])
    semantics = cast(dict[str, object], body["view_semantics"])
    diagnose = cast(dict[str, object], semantics["diagnose"])
    limits = cast(dict[str, object], body["limits"])
    measurement = cast(dict[str, object], limits["payload_limit_measurement"])
    lifecycle = cast(dict[str, object], body["lifecycle_and_cancellation"])
    publication = cast(dict[str, object], lifecycle["inspection_request_cancellation"])

    assert scope["admitted_subjects"] == [
        "an admitted immutable compiled private plan or a content-free admission rejection receipt",
        "an owner-issued immutable invocation snapshot at an allowed lifecycle capture point",
        "a content-free terminal accounting or cleanup receipt",
    ]
    assert diagnose["subjects"] == [
        "content-free admission rejection receipt",
        "owner-issued immutable non-success snapshot",
        "content-free terminal accounting or cleanup receipt",
    ]
    assert limits["fixed_arity_enforcement"] == (
        "The private explain inspect and diagnose functions accept exactly one scalar subject and return exactly one "
        "scalar view or rejection; no batch request or multi-view shape exists."
    )
    assert measurement == {
        "domain": "detached payloads built only by schema-specific Phase 10 builders from validated private values",
        "top_level_fields": "count exact string keys in the root exact built-in dictionary",
        "provenance_fields": "count exact string keys in the exact built-in provenance dictionary",
        "json_nesting_depth": (
            "root dictionary has container depth 1; entering an exact built-in dictionary or list adds 1; scalar "
            "values do not add depth"
        ),
        "allowlisted_string_utf8_bytes": (
            "measure each exact string key and each exact string value separately as the length of its UTF-8 "
            "encoding; do not aggregate strings"
        ),
        "static_schema_ceiling_evidence": (
            "when the current closed schema cannot naturally reach a ceiling, require direct exact and one-over "
            "tests of the authoritative measurement primitive, full-encoder tests with the ceiling lowered to the "
            "actual value and one less, and a mutation proving every encoder variant invokes the validator"
        ),
    }
    assert publication == {
        "availability": "no cancellable asynchronous or multi-view inspection request exists in Phase 10 v1",
        "effect_on_protection": "none",
        "view_publication_linearization_point": "successful return from the synchronous private inspection function",
        "view_construction_failure": (
            "an internal denial limit redaction or state failure before operation return discards builder state and "
            "returns no view"
        ),
        "byte_publication_linearization_point": "successful return from the private canonical encoder",
        "encoding_failure": (
            "a detached-payload limit redaction or encoding failure returns no encoded bytes and cannot revoke an "
            "existing immutable view"
        ),
        "after_view_publication": "the returned immutable view cannot be rewritten or revoked",
        "trusted_stop_claim": "none",
    }


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda value: value.update(extra=True), id="unknown-envelope-field"),
        pytest.param(lambda value: value.update(schema_version="unknown"), id="schema"),
        pytest.param(lambda value: value.update(digest="0" * 64), id="digest"),
        pytest.param(lambda value: cast(dict[str, object], value["contract"]).update(sdk_phase="10"), id="type"),
        pytest.param(
            lambda value: cast(dict[str, object], value["contract"]).update(status="frozen_owner_contract"),
            id="alternate-member",
        ),
    ],
)
def test_phase10_contract_loader_rejects_unknown_type_and_member_mutations(
    mutate: Callable[[dict[str, object]], None],
) -> None:
    module = _module()
    envelope = copy.deepcopy(_envelope())
    mutate(envelope)
    if isinstance(envelope.get("contract"), dict) and envelope.get("digest") != "0" * 64:
        envelope["digest"] = _canonical_digest(envelope["contract"])

    rejected = module._compile_phase10_contract(envelope)

    assert not module._is_admitted_phase10_contract(rejected)
    assert rejected.code == "contract_invalid"


@pytest.mark.parametrize("text", ["{", '{"schema_version":"one","schema_version":"two"}'])
def test_phase10_contract_loader_fails_closed_on_invalid_or_duplicate_json(
    monkeypatch: pytest.MonkeyPatch,
    text: str,
) -> None:
    module = _module()
    monkeypatch.setattr(module, "files", lambda _package: _Resource(text))

    rejected = module._load_phase10_contract()

    assert not module._is_admitted_phase10_contract(rejected)


def test_phase10_contract_values_are_immutable_masked_and_not_picklable() -> None:
    module = _module()
    contract = module._load_phase10_contract()
    assert module._is_admitted_phase10_contract(contract)

    assert repr(contract).startswith("<private ")
    assert _CONTRACT_DIGEST not in repr(contract)
    with pytest.raises(TypeError, match="not serializable"):
        pickle.dumps(contract)
    with pytest.raises(FrozenInstanceError):
        contract.version = "changed"
    assert not module._is_admitted_phase10_contract(replace(contract, digest="0" * 64))


def test_phase10_tier1_preserves_the_p9_contract_digest() -> None:
    path = Path("src/anonymizer/interface/result_compatibility_contract.json")
    envelope = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))

    assert envelope["digest"] == _canonical_digest(envelope["contract"]) == _P9_DIGEST


class _Resource:
    def __init__(self, text: str) -> None:
        self._text = text

    def joinpath(self, _name: str) -> _Resource:
        return self

    def read_text(self, *, encoding: str) -> str:
        assert encoding == "utf-8"
        return self._text
