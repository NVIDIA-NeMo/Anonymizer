# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import importlib.util
from typing import cast

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
from anonymizer.engine.execution.mention_resolution import (
    _ClusteredGraph,
    _DistinctSubjectEvidence,
    _EvidenceVersion,
    _ResolutionRejected,
    _ResolutionRejectionCode,
    _resolve_mentions,
    _ResolverScope,
    _SameSubjectEvidence,
)


def test_phase6_resolution_test_infrastructure() -> None:
    assert _DetectedGraph.__name__ == "_DetectedGraph"


def test_phase6_resolution_module_exposes_explicit_evidence_boundary() -> None:
    module_name = "anonymizer.engine.execution.mention_resolution"
    assert importlib.util.find_spec(module_name) is not None, "Phase 6 mention resolution module is missing"
    module = importlib.import_module(module_name)

    assert callable(getattr(module, "_resolve_mentions", None))


def _detected_graph() -> tuple[_DetectedGraph, tuple[_MentionTargetToken, ...], tuple[_AnchoredMention, ...]]:
    target_a = _MentionTargetToken()
    target_b = _MentionTargetToken()
    targets = (
        _MentionTarget(target_a, _DatumId("a"), "Alice met Bob"),
        _MentionTarget(target_b, _DatumId("b"), "A. Example"),
    )
    mentions = (
        _AnchoredMention(_MentionId(), targets[0].datum_id, 0, 5, "Alice", "name", _MentionProvenance.SPAN_DETECTOR),
        _AnchoredMention(_MentionId(), targets[0].datum_id, 10, 13, "Bob", "name", _MentionProvenance.SPAN_DETECTOR),
        _AnchoredMention(
            _MentionId(), targets[1].datum_id, 0, 10, "A. Example", "name", _MentionProvenance.EXACT_AUGMENTER
        ),
    )
    return _DetectedGraph(targets, mentions), (target_a, target_b), mentions


def _members(result: _ClusteredGraph) -> frozenset[frozenset[_MentionId]]:
    return frozenset(frozenset(cluster.ordered_mention_ids) for cluster in result.clusters)


def test_resolution_starts_singleton_and_never_merges_by_label_or_content() -> None:
    detected, (target_a, target_b), mentions = _detected_graph()

    result = _resolve_mentions(
        detected,
        (_ResolverScope(target_a, (target_a, target_b)), _ResolverScope(target_b, (target_b,))),
        (),
    )

    assert isinstance(result, _ClusteredGraph)
    assert _members(result) == frozenset(frozenset((mention.id,)) for mention in mentions)


def test_resolution_merges_only_explicit_same_subject_components_independent_of_order() -> None:
    detected, (target_a, target_b), mentions = _detected_graph()
    scopes = (_ResolverScope(target_a, (target_a, target_b)), _ResolverScope(target_b, (target_a, target_b)))
    first = _SameSubjectEvidence(target_a, mentions[0].id, mentions[2].id, _EvidenceVersion.V1)
    duplicate_other_owner = _SameSubjectEvidence(target_b, mentions[2].id, mentions[0].id, _EvidenceVersion.V1)

    forward = _resolve_mentions(detected, scopes, (first, duplicate_other_owner))
    reverse = _resolve_mentions(detected, tuple(reversed(scopes)), (duplicate_other_owner, first))

    assert isinstance(forward, _ClusteredGraph)
    assert isinstance(reverse, _ClusteredGraph)
    expected = frozenset((frozenset((mentions[0].id, mentions[2].id)), frozenset((mentions[1].id,))))
    assert _members(forward) == expected
    assert _members(reverse) == expected
    assert len(forward.accepted_evidence) == len(reverse.accepted_evidence) == 1


def test_resolution_rejects_transitive_distinct_subject_contradiction() -> None:
    detected, (target_a, target_b), mentions = _detected_graph()
    scopes = (_ResolverScope(target_a, (target_a, target_b)), _ResolverScope(target_b, (target_a, target_b)))
    evidence = (
        _SameSubjectEvidence(target_a, mentions[0].id, mentions[1].id, _EvidenceVersion.V1),
        _SameSubjectEvidence(target_b, mentions[1].id, mentions[2].id, _EvidenceVersion.V1),
        _DistinctSubjectEvidence(target_a, mentions[0].id, mentions[2].id, _EvidenceVersion.V1),
    )

    assert _resolve_mentions(detected, scopes, evidence) == _ResolutionRejected(
        _ResolutionRejectionCode.EVIDENCE_CONTRADICTION
    )


@pytest.mark.parametrize("fault", ["self", "duplicate", "foreign", "ownerless"])
def test_resolution_rejects_unattributable_or_malformed_evidence(fault: str) -> None:
    detected, (target_a, target_b), mentions = _detected_graph()
    scopes = (_ResolverScope(target_a, (target_a, target_b)), _ResolverScope(target_b, (target_b,)))
    valid = _SameSubjectEvidence(target_a, mentions[0].id, mentions[2].id, _EvidenceVersion.V1)
    match fault:
        case "self":
            evidence = (_SameSubjectEvidence(target_a, mentions[0].id, mentions[0].id, _EvidenceVersion.V1),)
        case "duplicate":
            evidence = (valid, _SameSubjectEvidence(target_a, mentions[2].id, mentions[0].id, _EvidenceVersion.V1))
        case "foreign":
            evidence = (_SameSubjectEvidence(target_a, mentions[0].id, _MentionId(), _EvidenceVersion.V1),)
        case "ownerless":
            evidence = (_SameSubjectEvidence(target_a, mentions[2].id, mentions[2].id, _EvidenceVersion.V1),)
        case unreachable:
            raise AssertionError(unreachable)

    result = _resolve_mentions(detected, scopes, evidence)

    assert isinstance(result, _ResolutionRejected)
    assert result.code in {_ResolutionRejectionCode.INVALID_EVIDENCE, _ResolutionRejectionCode.FOREIGN_TOKEN}


def test_resolver_rejects_a_malformed_detected_graph_without_inspecting_foreign_values() -> None:
    detected, (target_a, target_b), _mentions = _detected_graph()
    malformed = _DetectedGraph(detected.targets, cast(tuple[_AnchoredMention, ...], (object(),)))
    scopes = (_ResolverScope(target_a, (target_a, target_b)), _ResolverScope(target_b, (target_b,)))

    result = _resolve_mentions(malformed, scopes, ())

    assert result == _ResolutionRejected(_ResolutionRejectionCode.STALE_TOKEN)
