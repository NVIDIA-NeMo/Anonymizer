# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
import pickle
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, replace
from importlib.resources import files
from types import ModuleType
from typing import cast

import pytest

from anonymizer.engine.execution import role_policy as role_policy_module
from anonymizer.engine.execution.role_policy import _load_redact_role_policy, _RolePolicyRejected

_CONTRACT_DIGEST = "3755832ecc64fe6e9dbeccc136c40020e5158446e5c74d6dca2392efdbb006bb"
_POLICY_DIGEST = "c27580bd2cc4051bdd11b63a91391f8995bdef1ed2052534623cdd3160318ef8"
_CORPUS_DIGEST = "5ba1e69c7836a428b16e9d2ac7fc0cf3fbb445fd0ca72c1d0a8a049194115123"


def test_phase7_contract_test_infrastructure() -> None:
    assert importlib.util.find_spec("anonymizer.engine.execution.role_policy") is not None


def _phase7_contract_module() -> ModuleType:
    module_name = "anonymizer.engine.execution.phase7_contract"
    assert importlib.util.find_spec(module_name) is not None, "Phase 7 contract module is missing"
    return importlib.import_module(module_name)


def test_phase7_contract_loader_returns_the_exact_frozen_private_contract() -> None:
    module = _phase7_contract_module()

    result = module._load_phase7_contract()

    assert type(result).__name__ == "_Phase7StableSubstituteContract"
    assert result.version.value == "anonymizer-phase7-stable-substitute/v1"
    assert result.digest == _CONTRACT_DIGEST
    assert result.phase6_result_version == "phase6-role-result/v1"
    assert result.phase6_policy_version == "phase6-substitute-role-policy/v1"
    assert result.phase6_policy_digest == _POLICY_DIGEST
    assert result.selectors == ("cluster_role/v1",)
    assert result.relations == ("email_from_name/v1",)
    assert result.formats == (
        "email_addr_spec_ascii/v1",
        "telephone_ascii/v1",
        "unicode_person_name/v1",
        "username_ascii/v1",
    )
    assert result.masks == ("digit_literal/v1", "none/v1")
    assert result.corpus_version == "anonymizer-phase7-owner-contract-corpus/v1"
    assert result.corpus_case_count == 30
    assert result.corpus_digest == _CORPUS_DIGEST
    assert tuple(role.name for role in result.roles) == (
        "email_address",
        "fax_number",
        "person_family_name",
        "person_given_name",
        "user_name",
        "voice_phone_number",
    )
    assert dict(result.count_limits) == {
        "max_clusters_per_invocation": 3,
        "max_clusters_per_scope": 3,
        "max_context_fragments_per_scope": 4,
        "max_datums_per_invocation": 4,
        "max_distinct_pairs_per_scope": 6,
        "max_mentions_per_invocation": 6,
        "max_mentions_per_scope": 6,
        "max_relations_per_scope": 4,
        "max_scope_members": 4,
        "max_scopes_per_invocation": 2,
        "max_slots_per_invocation": 4,
        "max_slots_per_scope": 4,
    }
    assert module._is_admitted_phase7_contract(result)
    assert repr(result).startswith("<private ")
    assert _CONTRACT_DIGEST not in repr(result)
    with pytest.raises(TypeError, match="not serializable"):
        pickle.dumps(result)


def _resource_payloads() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    package = files("anonymizer.engine.execution")
    names = (
        "phase7_stable_substitute_contract.json",
        "phase6_substitute_role_policy.json",
        "phase7_owner_contract_corpus.json",
    )
    payloads = tuple(json.loads(package.joinpath(name).read_text(encoding="utf-8")) for name in names)
    return cast(tuple[dict[str, object], dict[str, object], dict[str, object]], payloads)


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _set_nested(mapping: dict[str, object], path: tuple[str, ...], value: object) -> None:
    current = mapping
    for key in path[:-1]:
        current = cast(dict[str, object], current[key])
    current[path[-1]] = value


@pytest.mark.parametrize(
    ("path", "value"),
    [
        pytest.param(("version",), "anonymizer-phase7-stable-substitute/v2", id="unknown-version"),
        pytest.param(("selectors", "unknown_selector/v1"), {}, id="unknown-selector"),
        pytest.param(("relations", "unknown_relation/v1"), {}, id="unknown-relation"),
        pytest.param(("relations", "wildcard_constraints"), "wildcard/v1", id="unsupported-wildcard"),
        pytest.param(("roles", "email_address", "format"), "unknown-format/v1", id="unknown-format"),
        pytest.param(("roles", "email_address", "mask"), "unknown-mask/v1", id="unknown-mask"),
        pytest.param(
            ("relations", "email_from_name/v1", "downstream_role"),
            "voice_phone_number",
            id="unsupported-relation",
        ),
        pytest.param(("execution", "capability", "profile"), "unknown-profile/v1", id="invalid-capability"),
        pytest.param(("limits", "counts", "max_slots_per_scope"), 5, id="count-one-over"),
        pytest.param(("limits", "bytes", "max_candidate_value_bytes"), 257, id="bytes-one-over"),
    ],
)
def test_phase7_contract_rejects_every_self_consistent_semantic_mutation(
    path: tuple[str, ...],
    value: object,
) -> None:
    module = _phase7_contract_module()
    envelope, policy, corpus = copy.deepcopy(_resource_payloads())
    contract = cast(dict[str, object], envelope["contract"])
    _set_nested(contract, path, value)
    envelope["digest"] = _canonical_digest(contract)

    result = module._compile_phase7_contract(envelope, policy, corpus)

    assert type(result).__name__ == "_Phase7ContractRejected"


@pytest.mark.parametrize("mutation", ["missing-label", "unknown-role"])
def test_phase7_contract_rejects_an_incomplete_or_unknown_phase6_disposition(mutation: str) -> None:
    module = _phase7_contract_module()
    envelope, policy, corpus = copy.deepcopy(_resource_payloads())
    contract = cast(dict[str, object], envelope["contract"])
    handoff = cast(dict[str, object], contract["phase6_handoff"])
    expected_policy = cast(dict[str, object], handoff["substitute_policy"])
    dispositions = cast(dict[str, object], policy["dispositions"])
    if mutation == "missing-label":
        dispositions.pop("account_number")
        expected_policy["detector_label_count"] = 64
    else:
        dispositions["email"] = "unknown_role"
    expected_policy["digest"] = _canonical_digest(policy)
    envelope["digest"] = _canonical_digest(contract)

    result = module._compile_phase7_contract(envelope, policy, corpus)

    assert type(result).__name__ == "_Phase7ContractRejected"


def test_phase7_contract_rejects_noncanonical_digest_input() -> None:
    module = _phase7_contract_module()
    envelope, policy, corpus = copy.deepcopy(_resource_payloads())
    contract = cast(dict[str, object], envelope["contract"])
    envelope["digest"] = hashlib.sha256(json.dumps(contract, indent=2).encode("utf-8")).hexdigest()

    result = module._compile_phase7_contract(envelope, policy, corpus)

    assert type(result).__name__ == "_Phase7ContractRejected"


def test_phase7_contract_rejects_a_self_consistent_alternate_corpus() -> None:
    module = _phase7_contract_module()
    envelope, policy, corpus = copy.deepcopy(_resource_payloads())
    contract = cast(dict[str, object], envelope["contract"])
    expected_corpus = cast(dict[str, object], contract["oracle_contract_corpus"])
    cases = cast(list[object], corpus["cases"])
    cases.pop()
    expected_corpus["case_count"] = 29
    expected_corpus["digest"] = _canonical_digest(corpus)
    envelope["digest"] = _canonical_digest(contract)

    result = module._compile_phase7_contract(envelope, policy, corpus)

    assert type(result).__name__ == "_Phase7ContractRejected"


def test_phase7_contract_accepts_json_whitespace_and_object_key_order_variation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _phase7_contract_module()
    envelope, policy, corpus = _resource_payloads()
    texts = {
        "phase7_stable_substitute_contract.json": json.dumps(envelope, indent=7),
        "phase6_substitute_role_policy.json": json.dumps(policy, separators=(", ", ": ")),
        "phase7_owner_contract_corpus.json": json.dumps(corpus, indent=1),
    }
    monkeypatch.setattr(module, "files", lambda _package: _ResourceDirectory(texts))

    result = module._load_phase7_contract()

    assert type(result).__name__ == "_Phase7StableSubstituteContract"
    assert result.digest == _CONTRACT_DIGEST


@pytest.mark.parametrize("failure", ["invalid-json", "duplicate-key", "read-error"])
def test_phase7_contract_loader_fails_closed_on_malformed_or_unreadable_resources(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    module = _phase7_contract_module()
    envelope, policy, corpus = _resource_payloads()
    contract_text = json.dumps(envelope)
    if failure == "invalid-json":
        contract_text = "{"
    elif failure == "duplicate-key":
        contract_text = contract_text[:-1] + ',"schema_version":"anonymizer-phase7-owner-contract-envelope/v1"}'
    texts: dict[str, str | OSError] = {
        "phase7_stable_substitute_contract.json": OSError("unreadable") if failure == "read-error" else contract_text,
        "phase6_substitute_role_policy.json": json.dumps(policy),
        "phase7_owner_contract_corpus.json": json.dumps(corpus),
    }
    monkeypatch.setattr(module, "files", lambda _package: _ResourceDirectory(texts))

    result = module._load_phase7_contract()

    assert type(result).__name__ == "_Phase7ContractRejected"


def test_phase7_contract_proof_rejects_forgery_and_nested_values_are_immutable() -> None:
    module = _phase7_contract_module()
    result = module._load_phase7_contract()
    assert type(result).__name__ == "_Phase7StableSubstituteContract"

    forged = replace(result, digest="0" * 64)

    assert not module._is_admitted_phase7_contract(forged)
    with pytest.raises(FrozenInstanceError):
        result.roles[0].name = "changed"
    assert _contains_only_immutable_json(result._source_snapshot)


def _contains_only_immutable_json(value: object) -> bool:
    if value is None or type(value) in {bool, int, str}:
        return True
    return type(value) is tuple and all(_contains_only_immutable_json(item) for item in value)


def test_phase7_substitute_policy_cannot_load_through_the_redact_policy_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = files("anonymizer.engine.execution")
    substitute_text = package.joinpath("phase6_substitute_role_policy.json").read_text(encoding="utf-8")
    monkeypatch.setattr(
        role_policy_module,
        "files",
        lambda _package: _ResourceDirectory({"phase6_redact_role_policy.json": substitute_text}),
    )

    assert isinstance(_load_redact_role_policy(), _RolePolicyRejected)


class _ResourceDirectory:
    def __init__(self, texts: Mapping[str, str | OSError]) -> None:
        self._texts = dict(texts)
        self._name: str | None = None

    def joinpath(self, name: str) -> _ResourceDirectory:
        resource = _ResourceDirectory(self._texts)
        resource._name = name
        return resource

    def read_text(self, *, encoding: str) -> str:
        assert encoding == "utf-8"
        assert self._name is not None
        value = self._texts[self._name]
        if isinstance(value, OSError):
            raise value
        return value
