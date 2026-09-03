# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
from typing import cast

import pytest

from anonymizer.engine.execution import role_policy as role_policy_module
from anonymizer.engine.execution.graph import _DatumId
from anonymizer.engine.execution.mention_admission import (
    _AnchoredMention,
    _DetectedGraph,
    _MentionId,
    _MentionProvenance,
    _MentionTarget,
    _MentionTargetToken,
)
from anonymizer.engine.execution.mention_resolution import _ClusteredGraph, _ClusterId, _EntityCluster
from anonymizer.engine.execution.role_policy import (
    _ClassifiedRole,
    _classify_roles,
    _compile_role_policy,
    _load_redact_role_policy,
    _ResolvedGraph,
    _RolePolicy,
    _RolePolicyRejected,
    _RolePolicyVersion,
    _UnsupportedRole,
)


def test_phase6_role_test_infrastructure() -> None:
    assert _ClusteredGraph.__name__ == "_ClusteredGraph"


def test_phase6_role_policy_module_exposes_closed_classification_boundary() -> None:
    module_name = "anonymizer.engine.execution.role_policy"
    assert importlib.util.find_spec(module_name) is not None, "Phase 6 role policy module is missing"
    module = importlib.import_module(module_name)

    assert callable(getattr(module, "_classify_roles", None))


def _clustered_graph(*labels: str) -> _ClusteredGraph:
    target_token = _MentionTargetToken()
    target = _MentionTarget(target_token, _DatumId("target"), "Alice Bob")
    mentions = tuple(
        _AnchoredMention(
            _MentionId(),
            target.datum_id,
            index * 6,
            index * 6 + 5 if index == 0 else index * 6 + 3,
            "Alice" if index == 0 else "Bob",
            label,
            _MentionProvenance.SPAN_DETECTOR,
        )
        for index, label in enumerate(labels)
    )
    clusters = tuple(_EntityCluster(_ClusterId(), (mention.id,), ()) for mention in mentions)
    return _ClusteredGraph(_DetectedGraph((target,), mentions), clusters, ())


def test_role_policy_classifies_only_frozen_mappings_and_marks_unknown_labels_unsupported() -> None:
    policy = _compile_role_policy(_RolePolicyVersion.V1, (("name", "person_name"),))
    assert isinstance(policy, _RolePolicy)

    result = _classify_roles(_clustered_graph("name", "custom_secret"), policy)

    assert isinstance(result, _ResolvedGraph)
    assert isinstance(result.mentions[0].role_result, _ClassifiedRole)
    assert result.mentions[0].role_result.role.value == "person_name"
    assert isinstance(result.mentions[1].role_result, _UnsupportedRole)
    assert result.policy_version is _RolePolicyVersion.V1
    assert len(result.policy_digest) == 64


def test_role_policy_digest_and_results_are_declaration_order_invariant() -> None:
    forward = _compile_role_policy(_RolePolicyVersion.V1, (("name", "person_name"), ("email", "contact")))
    reverse = _compile_role_policy(_RolePolicyVersion.V1, (("email", "contact"), ("name", "person_name")))

    assert isinstance(forward, _RolePolicy)
    assert isinstance(reverse, _RolePolicy)
    assert forward.digest == reverse.digest
    assert tuple(label for label, _role in forward.mappings) == ("email", "name")
    assert tuple(label for label, _role in reverse.mappings) == ("email", "name")


def test_empty_redact_policy_is_fail_closed_without_blocking_structural_resolution() -> None:
    policy = _compile_role_policy(_RolePolicyVersion.V1, ())
    assert isinstance(policy, _RolePolicy)

    result = _classify_roles(_clustered_graph("name"), policy)

    assert isinstance(result, _ResolvedGraph)
    assert isinstance(result.mentions[0].role_result, _UnsupportedRole)


def test_role_policy_rejects_unknown_version_duplicate_labels_and_unsealed_policy() -> None:
    assert isinstance(_compile_role_policy(cast(_RolePolicyVersion, "unknown"), ()), _RolePolicyRejected)
    assert isinstance(
        _compile_role_policy(_RolePolicyVersion.V1, (("name", "one"), ("name", "two"))),
        _RolePolicyRejected,
    )
    direct = _RolePolicy(_RolePolicyVersion.V1, (), "0" * 64)

    assert isinstance(_classify_roles(_clustered_graph("name"), direct), _RolePolicyRejected)


def test_redact_role_policy_manifest_freezes_fail_closed_structural_version() -> None:
    manifest_path = (
        Path(__file__).parents[3] / "src" / "anonymizer" / "engine" / "execution" / "phase6_redact_role_policy.json"
    )
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    policy = _compile_role_policy(_RolePolicyVersion(manifest["version"]), tuple(map(tuple, manifest["mappings"])))

    assert isinstance(policy, _RolePolicy)
    assert manifest == {
        "digest": policy.digest,
        "mappings": [],
        "version": _RolePolicyVersion.V1.value,
    }


@pytest.mark.parametrize(
    "manifest_text",
    [
        pytest.param("{", id="invalid-json"),
        pytest.param(
            json.dumps(
                {
                    "digest": "e11a29db1af26c9572e1b4dec9e0a91e80966c6de1d813a378b64210a3bdfc40",
                    "mappings": [],
                    "unexpected": True,
                    "version": _RolePolicyVersion.V1.value,
                }
            ),
            id="extra-key",
        ),
        pytest.param(
            json.dumps(
                {
                    "digest": "e11a29db1af26c9572e1b4dec9e0a91e80966c6de1d813a378b64210a3bdfc40",
                    "mappings": {},
                    "version": _RolePolicyVersion.V1.value,
                }
            ),
            id="non-list-mappings",
        ),
        pytest.param(
            json.dumps(
                {
                    "digest": "e11a29db1af26c9572e1b4dec9e0a91e80966c6de1d813a378b64210a3bdfc40",
                    "mappings": [["name"]],
                    "version": _RolePolicyVersion.V1.value,
                }
            ),
            id="malformed-mapping",
        ),
        pytest.param(
            json.dumps(
                {
                    "digest": "e11a29db1af26c9572e1b4dec9e0a91e80966c6de1d813a378b64210a3bdfc40",
                    "mappings": [],
                    "version": 1,
                }
            ),
            id="non-string-version",
        ),
    ],
)
def test_redact_role_policy_manifest_loader_rejects_noncanonical_content(
    monkeypatch: pytest.MonkeyPatch,
    manifest_text: str,
) -> None:
    class _ManifestResource:
        def joinpath(self, _name: str) -> _ManifestResource:
            return self

        def read_text(self, *, encoding: str) -> str:
            assert encoding == "utf-8"
            return manifest_text

    monkeypatch.setattr(role_policy_module, "files", lambda _package: _ManifestResource())

    assert isinstance(_load_redact_role_policy(), _RolePolicyRejected)


def test_redact_role_policy_manifest_loader_rejects_a_self_consistent_nonempty_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nonempty = _compile_role_policy(_RolePolicyVersion.V1, (("name", "person_name"),))
    assert isinstance(nonempty, _RolePolicy)
    manifest_text = json.dumps(
        {
            "digest": nonempty.digest,
            "mappings": [["name", "person_name"]],
            "version": _RolePolicyVersion.V1.value,
        }
    )

    class _ManifestResource:
        def joinpath(self, _name: str) -> _ManifestResource:
            return self

        def read_text(self, *, encoding: str) -> str:
            assert encoding == "utf-8"
            return manifest_text

    monkeypatch.setattr(role_policy_module, "files", lambda _package: _ManifestResource())

    assert isinstance(_load_redact_role_policy(), _RolePolicyRejected)
