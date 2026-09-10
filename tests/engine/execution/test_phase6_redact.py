# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import importlib.util
import pickle
from dataclasses import replace

import pytest

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
from anonymizer.engine.execution.redact_patches import (
    _apply_redact_patches,
    _bind_patch_manifest,
    _BoundPatchManifest,
    _build_patch_manifest,
    _materialize_redact_patches,
    _PatchManifest,
    _PatchRejected,
    _PatchRejectionCode,
    _PatchToken,
    _RedactPatch,
    _ReturnedRedact,
    _VerifiedGraph,
    _verify_redact_patches,
)
from anonymizer.engine.execution.role_policy import (
    _classify_roles,
    _compile_role_policy,
    _ResolvedGraph,
    _RolePolicy,
    _RolePolicyVersion,
)


def test_phase6_redact_test_infrastructure() -> None:
    assert _ResolvedGraph.__name__ == "_ResolvedGraph"


def test_phase6_redact_module_exposes_mention_keyed_verification_boundary() -> None:
    module_name = "anonymizer.engine.execution.redact_patches"
    assert importlib.util.find_spec(module_name) is not None, "Phase 6 Redact patch module is missing"
    module = importlib.import_module(module_name)

    assert callable(getattr(module, "_verify_redact_patches", None))


def _resolved_graph(
    text: str,
    spans: tuple[tuple[int, int, str], ...],
) -> tuple[_ResolvedGraph, _MentionTargetToken]:
    target_token = _MentionTargetToken()
    target = _MentionTarget(target_token, _DatumId("target"), text)
    mentions = tuple(
        _AnchoredMention(
            _MentionId(),
            target.datum_id,
            start,
            end,
            text[start:end],
            label,
            _MentionProvenance.SPAN_DETECTOR,
        )
        for start, end, label in spans
    )
    clustered = _ClusteredGraph(
        _DetectedGraph((target,), mentions),
        tuple(_EntityCluster(_ClusterId(), (mention.id,), ()) for mention in mentions),
        (),
    )
    policy = _compile_role_policy(_RolePolicyVersion.V1, ())
    assert isinstance(policy, _RolePolicy)
    resolved = _classify_roles(clustered, policy)
    assert isinstance(resolved, _ResolvedGraph)
    return resolved, target_token


def _bound_and_patches(resolved: _ResolvedGraph) -> tuple[_BoundPatchManifest, tuple[_RedactPatch, ...]]:
    manifest = _build_patch_manifest(resolved)
    assert isinstance(manifest, _PatchManifest)
    bound = _bind_patch_manifest(manifest)
    assert isinstance(bound, _BoundPatchManifest)
    patches = _materialize_redact_patches(bound)
    assert isinstance(patches, tuple)
    return bound, patches


def test_exact_reconstruction_protects_only_anchored_repeated_occurrence() -> None:
    resolved, target_token = _resolved_graph("Alice and Alice", ((0, 5, "name"),))
    bound, patches = _bound_and_patches(resolved)

    result = _verify_redact_patches(
        bound,
        patches,
        (_ReturnedRedact(target_token, "[REDACTED] and Alice"),),
    )

    assert isinstance(result, _VerifiedGraph)
    assert result.datums[0].output == "[REDACTED] and Alice"
    assert result.datums[0].applied is True
    assert "Alice" not in repr(result)
    with pytest.raises(TypeError, match="not serializable"):
        pickle.dumps(result)


def test_patch_application_is_anchored_and_does_not_reconsider_inserted_text() -> None:
    resolved, target_token = _resolved_graph("Alice and Alice", ((0, 5, "name"),))
    _bound, patches = _bound_and_patches(resolved)

    returned = _apply_redact_patches(resolved, patches)

    assert returned == (_ReturnedRedact(target_token, "[REDACTED] and Alice"),)


def test_verified_no_work_path_requires_exact_unchanged_output_and_zero_patches() -> None:
    resolved, target_token = _resolved_graph("plain text", ())
    bound, patches = _bound_and_patches(resolved)

    accepted = _verify_redact_patches(
        bound,
        patches,
        (_ReturnedRedact(target_token, "plain text"),),
    )
    rejected = _verify_redact_patches(
        bound,
        patches,
        (_ReturnedRedact(target_token, "changed"),),
    )

    assert isinstance(accepted, _VerifiedGraph)
    assert accepted.datums[0].applied is False
    assert rejected == _PatchRejected(_PatchRejectionCode.RELEASE_PREDICATE_FAILED, target_token)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "foreign", "offset", "replacement", "target"])
def test_patch_verification_rejects_every_non_bijective_or_inexact_patch(fault: str) -> None:
    resolved, target_token = _resolved_graph("Alice", ((0, 5, "name"),))
    bound, patches = _bound_and_patches(resolved)
    patch = patches[0]
    match fault:
        case "missing":
            corrupted = ()
        case "duplicate":
            corrupted = (patch, patch)
        case "foreign":
            corrupted = (replace(patch, token=_PatchToken()),)
        case "offset":
            corrupted = (replace(patch, start=1),)
        case "replacement":
            corrupted = (replace(patch, replacement="Alice"),)
        case "target":
            corrupted = (replace(patch, target=_MentionTargetToken()),)
        case unreachable:
            raise AssertionError(unreachable)

    result = _verify_redact_patches(
        bound,
        corrupted,
        (_ReturnedRedact(target_token, "[REDACTED]"),),
    )

    assert isinstance(result, _PatchRejected)
    assert result.code in {_PatchRejectionCode.INVALID_PATCH, _PatchRejectionCode.FOREIGN_TOKEN}


@pytest.mark.parametrize("fault", ["missing", "duplicate", "foreign", "mismatch"])
def test_output_verification_requires_one_exact_result_per_target(fault: str) -> None:
    resolved, target_token = _resolved_graph("Alice", ((0, 5, "name"),))
    bound, patches = _bound_and_patches(resolved)
    valid = _ReturnedRedact(target_token, "[REDACTED]")
    match fault:
        case "missing":
            returned = ()
        case "duplicate":
            returned = (valid, valid)
        case "foreign":
            returned = (_ReturnedRedact(_MentionTargetToken(), "[REDACTED]"),)
        case "mismatch":
            returned = (_ReturnedRedact(target_token, "Alice"),)
        case unreachable:
            raise AssertionError(unreachable)

    result = _verify_redact_patches(bound, patches, returned)

    assert isinstance(result, _PatchRejected)
