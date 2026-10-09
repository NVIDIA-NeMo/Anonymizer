# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import json
import pickle
from pathlib import Path
from unittest.mock import patch

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectCode, EffectRejected
from anonymizer.engine.graph_sdk.continuity import (
    ContinuitySnapshot,
    StateNamespace,
    accept_handoff,
    capture_snapshot,
    create_handoff,
    encode_snapshot,
    fork_continuity,
    fresh_continuity,
    load_snapshot,
    restore_snapshot,
    write_snapshot,
)
from anonymizer.engine.graph_sdk.requests import TextArtifactValue, TextCollectionValue
from tests.graph_sdk.continuity_fixtures import limits


def wire() -> dict:
    return {
        "format": "nemo-anonymizer-continuity",
        "version": 2,
        "namespace": "private-history",
        "revision": 4,
        "decisions": [
            {
                "group": "private-group",
                "subject": {"kind": "person", "name": "private-subject", "version": 1},
                "slot": "name",
                "revision": 3,
                "value": {"tag": "text", "text": ""},
                "supports": [{"kind": "assessment", "name": "private-support", "version": 1}],
            }
        ],
    }


def load(path: Path, value: dict) -> ContinuitySnapshot:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return load_snapshot(path=path, limits=limits())


@pytest.mark.parametrize(
    "value",
    [
        {"tag": "text", "text": ""},
        {"tag": "text", "text": "π🙂"},
        {"tag": "collection", "items": []},
        {"tag": "collection", "items": [{"key": 0, "version": 1, "text": ""}, {"key": 0, "version": 2, "text": "二"}]},
    ],
)
def test_roundtrip_restore_fork_fresh_handoff(tmp_path: Path, value: dict) -> None:
    source = wire()
    source["decisions"][0]["value"] = value
    snapshot = load(tmp_path / "input.json", source)
    encoded = encode_snapshot(snapshot=snapshot, limits=limits())
    assert encoded == json.dumps(source, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    write_snapshot(path=tmp_path / "output.json", snapshot=snapshot, limits=limits())
    reloaded = load_snapshot(path=tmp_path / "output.json", limits=limits())
    assert reloaded == snapshot
    one = restore_snapshot(snapshot=snapshot, limits=limits())
    two = restore_snapshot(snapshot=snapshot, limits=limits())
    assert one.identity != two.identity
    assert one.namespace == two.namespace and one.view(limits=limits()).decisions == two.view(limits=limits()).decisions
    fork = fork_continuity(state=one, expected_revision=4, namespace=StateNamespace(name="fork"), limits=limits())
    assert fork.view(limits=limits()).revision == 1
    assert fork.view(limits=limits()).decisions[0].reference.revision == 3
    fork_snapshot = capture_snapshot(state=fork, expected_revision=1, limits=limits())
    write_snapshot(path=tmp_path / "fork.json", snapshot=fork_snapshot, limits=limits())
    assert load_snapshot(path=tmp_path / "fork.json", limits=limits()) == fork_snapshot
    handoff = create_handoff(state=one, expected_revision=4, limits=limits())
    accepted = accept_handoff(handoff=handoff, limits=limits())
    assert accepted.identity not in {one.identity, two.identity}
    assert accepted.view(limits=limits()).decisions == snapshot.decisions
    fresh = fresh_continuity(namespace=one.namespace, limits=limits())
    assert fresh.view(limits=limits()).revision == 0 and not fresh.view(limits=limits()).decisions
    assert type(snapshot.decisions[0].value) is (TextArtifactValue if value["tag"] == "text" else TextCollectionValue)
    for private in [snapshot, one, one.view(limits=limits()), handoff]:
        assert "private" not in repr(private)
        assert copy.copy(private) is private and copy.deepcopy(private) is private
        with pytest.raises(TypeError):
            pickle.dumps(private)
    with pytest.raises(EffectRejected) as failure:
        capture_snapshot(state=one, expected_revision=3, limits=limits())
    assert failure.value.code is EffectCode.CONTRADICTORY


@pytest.mark.parametrize(
    "mutation",
    [
        "v1",
        "bool_version",
        "zero_decision",
        "bool_history",
        "duplicate_decision",
        "duplicate_support",
        "support_versions",
        "noncanonical_decisions",
        "noncanonical_supports",
        "extra_root",
        "extra_value",
        "null_text",
        "float_revision",
        "surrogate",
        "unknown_tag",
        "bool_key",
        "zero_item_version",
        "duplicate_item",
        "unordered_items",
    ],
)
def test_malformed_history_is_rejected_whole(tmp_path: Path, mutation: str) -> None:
    source = wire()
    row = source["decisions"][0]
    if mutation == "v1":
        source["version"] = 1
    elif mutation == "bool_version":
        source["version"] = True
    elif mutation == "zero_decision":
        row["revision"] = 0
    elif mutation == "bool_history":
        source["revision"] = False
    elif mutation == "duplicate_decision":
        source["decisions"].append(copy.deepcopy(row))
    elif mutation == "duplicate_support":
        row["supports"].append(copy.deepcopy(row["supports"][0]))
    elif mutation == "support_versions":
        row["supports"].append({**row["supports"][0], "version": 2})
    elif mutation == "noncanonical_decisions":
        source["decisions"].append({**row, "slot": "a"})
    elif mutation == "noncanonical_supports":
        row["supports"].append({**row["supports"][0], "name": "a"})
    elif mutation == "extra_root":
        source["credential"] = "excluded"
    elif mutation == "extra_value":
        row["value"]["credential"] = "excluded"
    elif mutation == "null_text":
        row["value"]["text"] = None
    elif mutation == "float_revision":
        source["revision"] = 1.0
    elif mutation == "surrogate":
        row["value"]["text"] = "\ud800"
    elif mutation == "unknown_tag":
        row["value"]["tag"] = "mapping"
    else:
        item = {"key": 0, "version": 1, "text": "value"}
        row["value"] = {"tag": "collection", "items": [item]}
        if mutation == "bool_key":
            item["key"] = True
        elif mutation == "zero_item_version":
            item["version"] = 0
        elif mutation == "duplicate_item":
            row["value"]["items"].append(copy.deepcopy(item))
        elif mutation == "unordered_items":
            row["value"]["items"].insert(0, {**item, "key": 2})
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(EffectRejected):
        load_snapshot(path=path, limits=limits())


@pytest.mark.parametrize("raw", [b'{"format":1,"format":2}', b"{}{}", b'{"revision":NaN}', b"\xff", b"[" * 10000])
def test_bad_bytes_do_not_publish_partial_snapshot(tmp_path: Path, raw: bytes) -> None:
    path = tmp_path / "bad"
    path.write_bytes(raw)
    with pytest.raises(EffectRejected):
        load_snapshot(path=path, limits=limits())


def test_snapshot_byte_and_depth_bounds(tmp_path: Path) -> None:
    snapshot = load(tmp_path / "input", wire())
    raw = encode_snapshot(snapshot=snapshot, limits=limits())
    assert encode_snapshot(snapshot=snapshot, limits=limits(max_snapshot_bytes=len(raw), max_snapshot_depth=6)) == raw
    for bound in [
        limits(max_snapshot_bytes=len(raw) - 1),
        limits(max_snapshot_depth=5),
        limits(max_value_atoms=2),
        limits(max_value_depth=1),
        limits(max_decisions=0),
        limits(max_groups=0),
        limits(max_total_supports=0),
        limits(max_decisions_per_group=0),
        limits(max_name_bytes=2),
        limits(max_slot_bytes=2),
    ]:
        with pytest.raises(EffectRejected) as failure:
            encode_snapshot(snapshot=snapshot, limits=bound)
        assert failure.value.code is EffectCode.LIMIT_EXCEEDED
    path = tmp_path / "bounded"
    path.write_bytes(raw)
    with patch(
        "anonymizer.engine.graph_sdk._continuity_codec.json.loads", side_effect=AssertionError("must not parse")
    ):
        with pytest.raises(EffectRejected) as failure:
            load_snapshot(path=path, limits=limits(max_snapshot_bytes=len(raw) - 1))
    assert failure.value.code is EffectCode.LIMIT_EXCEEDED


def test_pre_replace_failure_preserves_destination(tmp_path: Path) -> None:
    snapshot = load(tmp_path / "input", wire())
    destination = tmp_path / "saved"
    destination.write_bytes(b"previous")
    with patch("anonymizer.engine.graph_sdk._continuity_codec.os.replace", side_effect=OSError("injected")):
        with pytest.raises(OSError):
            write_snapshot(path=destination, snapshot=snapshot, limits=limits())
    assert destination.read_bytes() == b"previous"
    assert not tuple(tmp_path.glob(".continuity-*"))
    with pytest.raises(EffectRejected):
        write_snapshot(path=destination, snapshot=snapshot, limits=limits(max_snapshot_bytes=0))
    assert destination.read_bytes() == b"previous"


def test_unicode_payload_and_collection_atom_bounds(tmp_path: Path) -> None:
    source = wire()
    source["decisions"][0]["value"] = {"tag": "collection", "items": [{"key": 0, "version": 1, "text": "🙂"}]}
    snapshot = load(tmp_path / "unicode", source)
    encoded = encode_snapshot(snapshot=snapshot, limits=limits(max_value_bytes=4, max_value_atoms=7, max_value_depth=4))
    assert "🙂".encode() in encoded
    for bounds in [limits(max_value_bytes=3), limits(max_value_atoms=6), limits(max_value_depth=3)]:
        with pytest.raises(EffectRejected) as failure:
            encode_snapshot(snapshot=snapshot, limits=bounds)
        assert failure.value.code is EffectCode.LIMIT_EXCEEDED


def test_bounded_large_integer_revision_does_not_depend_on_python_digit_limit(tmp_path: Path) -> None:
    path = tmp_path / "large-revision"
    raw = (
        b'{"decisions":[],"format":"nemo-anonymizer-continuity","namespace":"history","revision":'
        + b"1" * 5000
        + b',"version":2}'
    )
    path.write_bytes(raw)
    snapshot = load_snapshot(path=path, limits=limits())
    assert encode_snapshot(snapshot=snapshot, limits=limits()) == raw
