# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the test-only OpenShell OCSF JSONL replay adapter."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from anonymizer.engine.constants import COL_TEXT
from tests.streaming.openshell_ocsf_process_adapter import (
    FIDELITY_CLASS,
    MAPPING_VERSION,
    project_process_activity_item,
    protect_process_activity_item,
    replay_process_activity_corpus,
)
from tests.streaming.structured_trace_prototype import (
    SEGMENT_KEY_COLUMN,
    CodecBounds,
    FailureCode,
    StructuredItemError,
    build_synthetic_anonymizer,
)

CORPUS = Path(__file__).parents[1] / "fixtures" / "streaming" / "openshell_process_activity.jsonl"
SENSITIVE_ENTITIES = {
    "alice@example.test": "email",
    "alice-host.example.test": "hostname",
    "alice-workspace": "workspace",
    "alice-agent": "process_name",
    "registry.example.test/alice/agent:latest": "container_image",
}


@pytest.fixture
def bounds() -> CodecBounds:
    return CodecBounds(
        max_bytes=8_192,
        max_depth=12,
        max_targets=16,
        max_scalars=64,
        max_scalar_bytes=1_024,
        max_events=256,
    )


def test_openshell_process_activity_replay_is_keyed_and_complete_item_buffered(bounds: CodecBounds) -> None:
    source = CORPUS.read_bytes()
    emitted: list[bytes] = []

    protected = replay_process_activity_corpus(
        source,
        max_corpus_bytes=32_768,
        max_records=8,
        bounds=bounds,
        anonymizer=build_synthetic_anonymizer(SENSITIVE_ENTITIES),
        source_ref=CORPUS,
        emit=emitted.append,
    )

    assert tuple(emitted) == protected
    assert len(protected) == 2
    assert all(item.endswith(b"\n") and item.count(b"\n") == 1 for item in protected)
    assert all(value.encode() not in b"".join(protected) for value in SENSITIVE_ENTITIES)

    source_items = [json.loads(line) for line in source.splitlines()]
    protected_items = [json.loads(line) for line in protected]
    source_order = [(item["metadata"]["uid"], item["time"], item["type_uid"]) for item in source_items]
    protected_order = [(item["metadata"]["uid"], item["time"], item["type_uid"]) for item in protected_items]
    assert protected_order == source_order
    assert [item["activity_name"] for item in protected_items] == ["Launch", "Terminate"]


def test_mapping_and_manifest_are_deterministic_and_source_specific(bounds: CodecBounds) -> None:
    line = CORPUS.read_bytes().splitlines(keepends=True)[0]
    first = project_process_activity_item(line, bounds=bounds)
    second = project_process_activity_item(line, bounds=bounds)

    assert first.manifest.mapping_version == MAPPING_VERSION
    assert first.manifest.fidelity == FIDELITY_CLASS
    assert first.manifest.source_identity == "sandbox-alpha"
    assert first.manifest.source_order == ("Process Activity: Launch",)
    assert [segment.segment_key for segment in first.manifest.segments] == [
        segment.segment_key for segment in second.manifest.segments
    ]
    assert first.dataframe[SEGMENT_KEY_COLUMN].is_unique
    target_text = "\n".join(first.dataframe[COL_TEXT])
    assert all(value in target_text for value in SENSITIVE_ENTITIES)
    assert all(segment.segment_key.startswith(f"{MAPPING_VERSION}:/") for segment in first.manifest.segments)


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        (lambda item: item.__setitem__("unmapped", {"private": "alice@example.test"}), FailureCode.MAPPING_MISMATCH),
        (lambda item: item.__setitem__("type_uid", 100799), FailureCode.MAPPING_MISMATCH),
        (
            lambda item: cast(dict[str, object], item["process"]).__setitem__(
                "environment", "TOKEN=alice@example.test"
            ),
            FailureCode.MAPPING_MISMATCH,
        ),
    ],
)
def test_unknown_or_schema_invalid_fields_fail_closed_before_emission(
    mutation: Callable[[dict[str, object]], None],
    expected_code: FailureCode,
    bounds: CodecBounds,
) -> None:
    item = cast(dict[str, object], json.loads(CORPUS.read_bytes().splitlines()[0]))
    mutation(item)
    source = json.dumps(item, separators=(",", ":")).encode() + b"\n"
    emitted: list[bytes] = []

    with pytest.raises(StructuredItemError) as exc_info:
        protect_process_activity_item(
            source,
            bounds=bounds,
            anonymizer=build_synthetic_anonymizer(SENSITIVE_ENTITIES),
            source_ref=CORPUS,
            emit=emitted.append,
        )

    assert exc_info.value.code is expected_code
    assert emitted == []
    assert "alice@example.test" not in str(exc_info.value)


def test_corpus_limits_and_incomplete_final_record_fail_before_emission(bounds: CodecBounds) -> None:
    corpus = CORPUS.read_bytes()
    emitted: list[bytes] = []

    with pytest.raises(StructuredItemError) as count_error:
        replay_process_activity_corpus(
            corpus,
            max_corpus_bytes=32_768,
            max_records=1,
            bounds=bounds,
            anonymizer=build_synthetic_anonymizer(SENSITIVE_ENTITIES),
            source_ref=CORPUS,
            emit=emitted.append,
        )
    assert count_error.value.code is FailureCode.ITEM_TOO_LARGE
    assert emitted == []

    with pytest.raises(StructuredItemError) as partial_error:
        replay_process_activity_corpus(
            corpus.rstrip(b"\n"),
            max_corpus_bytes=32_768,
            max_records=8,
            bounds=bounds,
            anonymizer=build_synthetic_anonymizer(SENSITIVE_ENTITIES),
            source_ref=CORPUS,
            emit=emitted.append,
        )
    assert partial_error.value.code is FailureCode.INVALID_SOURCE
    assert emitted == []
