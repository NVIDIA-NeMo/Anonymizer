# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded version-two continuity encoding and explicit file publication."""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import TypeAlias, cast

from anonymizer.engine.graph_sdk._continuity_values import (
    _CONTINUITY_KEY,
    SNAPSHOT_FORMAT,
    SNAPSHOT_VERSION,
    ContinuityLimits,
    ContinuitySnapshot,
    RetainedDecision,
    StateDecisionKey,
    StateDecisionRef,
    StateGroupRef,
    StateNamespace,
    StateSubjectRef,
    StateSupportRef,
    _key_order,
    _support_order,
)
from anonymizer.engine.graph_sdk._effect_values import EffectCode, reject, require_count, require_instance
from anonymizer.engine.graph_sdk.requests import (
    ArtifactValue,
    TextArtifactValue,
    TextCollectionItem,
    TextCollectionValue,
)

_JSON: TypeAlias = str | int | list["_JSON"] | dict[str, "_JSON"]


def _text(value: object, bound: int, *, empty: bool = False) -> str:
    if type(value) is not str:
        reject(EffectCode.INVALID_TYPE)
    if len(value) > bound:
        reject(EffectCode.LIMIT_EXCEEDED)
    if not value and not empty:
        reject(EffectCode.INVALID_VALUE)
    used = 0
    for offset in range(0, len(value), 1024):
        try:
            used += len(value[offset : offset + 1024].encode("utf-8", "strict"))
        except UnicodeEncodeError:
            reject(EffectCode.INVALID_VALUE)
        if used > bound:
            reject(EffectCode.LIMIT_EXCEEDED)
    return value


def _stable_ref(value: StateSubjectRef | StateSupportRef, limits: ContinuityLimits) -> None:
    _text(value.kind, limits.max_name_bytes)
    _text(value.name, limits.max_name_bytes)
    require_count(value.version, positive=True)


def _value_wire(value: ArtifactValue, limits: ContinuityLimits) -> dict[str, _JSON]:
    if type(value) is TextArtifactValue:
        if limits.max_value_depth < 2 or limits.max_value_atoms < 3:
            reject(EffectCode.LIMIT_EXCEEDED)
        return {"tag": "text", "text": _text(value.text, limits.max_value_bytes, empty=True)}
    if type(value) is not TextCollectionValue:
        reject(EffectCode.INVALID_TYPE)
    if type(value.items) is not tuple:
        reject(EffectCode.INVALID_TYPE)
    if 3 + 4 * len(value.items) > limits.max_value_atoms or (4 if value.items else 2) > limits.max_value_depth:
        reject(EffectCode.LIMIT_EXCEEDED)
    items: list[_JSON] = []
    used = 0
    order = []
    for item in value.items:
        if type(item) is not TextCollectionItem or type(item.value) is not TextArtifactValue:
            reject(EffectCode.INVALID_TYPE)
        require_count(item.key)
        require_count(item.version, positive=True)
        text = _text(item.value.text, limits.max_value_bytes - used, empty=True)
        used += len(text.encode("utf-8"))
        items.append({"key": item.key, "version": item.version, "text": text})
        order.append((item.key, item.version))
    if len(set(order)) != len(order):
        reject(EffectCode.DUPLICATE)
    if order != sorted(order):
        reject(EffectCode.CONTRADICTORY)
    return {"tag": "collection", "items": items}


def _reference_wire(value: StateSubjectRef | StateSupportRef) -> dict[str, _JSON]:
    return {"kind": value.kind, "name": value.name, "version": value.version}


def _state_wire(
    namespace: StateNamespace, revision: int, decisions: tuple[RetainedDecision, ...], limits: ContinuityLimits
) -> dict[str, _JSON]:
    require_instance(limits, ContinuityLimits)
    require_instance(namespace, StateNamespace)
    require_instance(decisions, tuple)
    if len(decisions) > limits.max_decisions:
        reject(EffectCode.LIMIT_EXCEEDED)
    _text(namespace.name, limits.max_name_bytes)
    require_count(revision)
    total = 0
    groups: Counter[StateGroupRef] = Counter()
    keys = []
    rows: list[_JSON] = []
    for decision in decisions:
        require_instance(decision, RetainedDecision)
        supports = decision.supports
        require_instance(supports, frozenset)
        total += len(supports)
        if len(supports) > limits.max_supports_per_decision or total > limits.max_total_supports:
            reject(EffectCode.LIMIT_EXCEEDED)
        require_instance(decision.reference, StateDecisionRef)
        key = decision.reference.key
        require_instance(key, StateDecisionKey)
        require_instance(key.group, StateGroupRef)
        require_instance(key.subject, StateSubjectRef)
        _text(key.group.name, limits.max_name_bytes)
        _text(key.slot, limits.max_slot_bytes)
        _stable_ref(key.subject, limits)
        require_count(decision.reference.revision, positive=True)
        for support in supports:
            require_instance(support, StateSupportRef)
            _stable_ref(support, limits)
        if len({(s.kind, s.name) for s in supports}) != len(supports):
            reject(EffectCode.DUPLICATE)
        if not supports:
            reject(EffectCode.MISSING)
        groups[key.group] += 1
        keys.append(_key_order(key))
        rows.append(
            {
                "group": key.group.name,
                "subject": _reference_wire(key.subject),
                "slot": key.slot,
                "revision": decision.reference.revision,
                "value": _value_wire(decision.value, limits),
                "supports": [_reference_wire(s) for s in sorted(supports, key=_support_order)],
            }
        )
    if len(groups) > limits.max_groups or any(n > limits.max_decisions_per_group for n in groups.values()):
        reject(EffectCode.LIMIT_EXCEEDED)
    if len(set(keys)) != len(keys):
        reject(EffectCode.DUPLICATE)
    if keys != sorted(keys):
        reject(EffectCode.CONTRADICTORY)
    return {
        "format": SNAPSHOT_FORMAT,
        "version": SNAPSHOT_VERSION,
        "namespace": namespace.name,
        "revision": revision,
        "decisions": rows,
    }


def _check_json_depth(raw: bytes, maximum: int) -> None:
    # No tree allocation or recursive parser before bounding nesting. The closed
    # grammar has at most seven containers; deeper input cannot be a snapshot.
    depth = 0
    quoted = False
    escaped = False
    for byte in raw:
        if quoted:
            if escaped:
                escaped = False
            elif byte == 92:
                escaped = True
            elif byte == 34:
                quoted = False
            continue
        if byte == 34:
            quoted = True
            if depth + 1 > maximum:
                reject(EffectCode.LIMIT_EXCEEDED)
        elif byte in (123, 91):
            depth += 1
            if depth > maximum:
                reject(EffectCode.LIMIT_EXCEEDED)
            if depth > 7:
                reject(EffectCode.INVALID_VALUE)
        elif byte in (125, 93):
            depth -= 1
            if depth < 0:
                reject(EffectCode.INVALID_VALUE)
        elif byte not in (9, 10, 13, 32, 44, 58) and depth + 1 > maximum:
            reject(EffectCode.LIMIT_EXCEEDED)
    if quoted or depth:
        reject(EffectCode.INVALID_VALUE)


def _decimal(value: int, bound: int) -> bytes:
    # Decimal conversion must not inherit Python's process-global digit limit.
    # Each intermediate chunk has at most nine digits and the byte bound applies
    # before assembling an arbitrary-size revision or collection coordinate.
    if value.bit_length() > bound * 4:
        reject(EffectCode.LIMIT_EXCEEDED)
    chunks = []
    while value >= 1_000_000_000:
        value, tail = divmod(value, 1_000_000_000)
        chunks.append(f"{tail:09d}")
        if len(chunks) * 9 > bound:
            reject(EffectCode.LIMIT_EXCEEDED)
    return (str(value) + "".join(reversed(chunks))).encode("ascii")


def _wire_chunks(wire: _JSON, bound: int) -> Iterator[bytes]:
    # This serializes only the already validated closed snapshot wire tree.
    pending: list[_JSON | bytes] = [wire]
    while pending:
        value = pending.pop()
        if isinstance(value, bytes):
            yield value
        elif isinstance(value, str):
            yield json.dumps(value, ensure_ascii=False).encode("utf-8", "strict")
        elif isinstance(value, int):
            yield _decimal(value, bound)
        else:
            entries: list[_JSON | bytes] = []
            if isinstance(value, list):
                for item in value:
                    if entries:
                        entries.append(b",")
                    entries.append(item)
                entries = [b"[", *entries, b"]"]
            else:
                for key in sorted(value):
                    if entries:
                        entries.append(b",")
                    entries.extend((key, b":", value[key]))
                entries = [b"{", *entries, b"}"]
            pending.extend(reversed(entries))


def _encode_state(
    namespace: StateNamespace, revision: int, decisions: tuple[RetainedDecision, ...], limits: ContinuityLimits
) -> bytes:
    wire = _state_wire(namespace, revision, decisions, limits)
    chunks: list[bytes] = []
    total = 0
    for raw in _wire_chunks(wire, limits.max_snapshot_bytes):
        total += len(raw)
        if total > limits.max_snapshot_bytes:
            reject(EffectCode.LIMIT_EXCEEDED)
        chunks.append(raw)
    result = b"".join(chunks)
    _check_json_depth(result, limits.max_snapshot_depth)
    return result


def encode_snapshot(*, snapshot: ContinuitySnapshot, limits: ContinuityLimits) -> bytes:
    require_instance(snapshot, ContinuitySnapshot)
    require_instance(limits, ContinuityLimits)
    if snapshot.format != SNAPSHOT_FORMAT or type(snapshot.version) is not int or snapshot.version != SNAPSHOT_VERSION:
        reject(EffectCode.UNSUPPORTED)
    return _encode_state(snapshot.namespace, snapshot.revision, snapshot.decisions, limits)


def _fields(value: object, names: set[str]) -> dict[str, _JSON]:
    if type(value) is not dict:
        reject(EffectCode.INVALID_TYPE)
    if value.keys() != names:
        reject(EffectCode.INVALID_VALUE)
    return cast(dict[str, _JSON], value)


def _array(value: object, bound: int) -> list[_JSON]:
    if type(value) is not list:
        reject(EffectCode.INVALID_TYPE)
    if len(value) > bound:
        reject(EffectCode.LIMIT_EXCEEDED)
    return cast(list[_JSON], value)


def _read_ref(value: object, limits: ContinuityLimits) -> tuple[str, str, int]:
    row = _fields(value, {"kind", "name", "version"})
    return (
        _text(row["kind"], limits.max_name_bytes),
        _text(row["name"], limits.max_name_bytes),
        require_count(row["version"], positive=True),
    )


def _read_value(value: object, limits: ContinuityLimits) -> ArtifactValue:
    if type(value) is not dict:
        reject(EffectCode.INVALID_TYPE)
    if type(value.get("tag")) is not str:
        reject(EffectCode.INVALID_TYPE)
    if value.get("tag") == "text":
        row = _fields(value, {"tag", "text"})
        result: ArtifactValue = TextArtifactValue(text=_text(row["text"], limits.max_value_bytes, empty=True))
    elif value.get("tag") == "collection":
        row = _fields(value, {"tag", "items"})
        members = _array(row["items"], max(0, (limits.max_value_atoms - 3) // 4))
        items = []
        for member in members:
            item = _fields(member, {"key", "version", "text"})
            items.append(
                TextCollectionItem(
                    key=require_count(item["key"]),
                    version=require_count(item["version"], positive=True),
                    value=TextArtifactValue(text=_text(item["text"], limits.max_value_bytes, empty=True)),
                )
            )
        result = TextCollectionValue(items=tuple(items))
    else:
        reject(EffectCode.UNSUPPORTED)
    _value_wire(result, limits)
    return result


def _pairs(pairs: list[tuple[str, _JSON]]) -> dict[str, _JSON]:
    if len({key for key, _ in pairs}) != len(pairs):
        reject(EffectCode.DUPLICATE)
    return dict(pairs)


def _no_number(value: str) -> None:
    del value
    reject(EffectCode.INVALID_TYPE)


def _parse_integer(raw: str) -> int:
    negative = raw.startswith("-")
    digits = raw[1:] if negative else raw
    value = 0
    for offset in range(0, len(digits), 9):
        chunk = digits[offset : offset + 9]
        value = value * (10 ** len(chunk)) + int(chunk)
    return -value if negative else value


def _decode(raw: bytes, limits: ContinuityLimits) -> ContinuitySnapshot:
    if len(raw) > limits.max_snapshot_bytes:
        reject(EffectCode.LIMIT_EXCEEDED)
    _check_json_depth(raw, limits.max_snapshot_depth)
    try:
        parsed = json.loads(
            raw.decode("utf-8", "strict"),
            object_pairs_hook=_pairs,
            parse_int=_parse_integer,
            parse_float=_no_number,
            parse_constant=_no_number,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        reject(EffectCode.INVALID_VALUE)
    root = _fields(parsed, {"format", "version", "namespace", "revision", "decisions"})
    rows = _array(root["decisions"], limits.max_decisions)
    namespace = StateNamespace(name=_text(root["namespace"], limits.max_name_bytes))
    revision = require_count(root["revision"])
    version = require_count(root["version"])
    if type(root["format"]) is not str:
        reject(EffectCode.INVALID_TYPE)
    if root["format"] != SNAPSHOT_FORMAT or version != SNAPSHOT_VERSION:
        reject(EffectCode.UNSUPPORTED)
    decisions = []
    total_supports = 0
    for member in rows:
        row = _fields(member, {"group", "subject", "slot", "revision", "value", "supports"})
        supports_raw = _array(row["supports"], limits.max_supports_per_decision)
        total_supports += len(supports_raw)
        if total_supports > limits.max_total_supports:
            reject(EffectCode.LIMIT_EXCEEDED)
        subject_kind, subject_name, subject_version = _read_ref(row["subject"], limits)
        supports = [
            StateSupportRef(kind=k, name=n, version=v) for k, n, v in (_read_ref(item, limits) for item in supports_raw)
        ]
        support_keys = [_support_order(s) for s in supports]
        if len(set(support_keys)) != len(support_keys):
            reject(EffectCode.DUPLICATE)
        if support_keys != sorted(support_keys):
            reject(EffectCode.CONTRADICTORY)
        key = StateDecisionKey(
            group=StateGroupRef(name=_text(row["group"], limits.max_name_bytes)),
            subject=StateSubjectRef(kind=subject_kind, name=subject_name, version=subject_version),
            slot=_text(row["slot"], limits.max_slot_bytes),
        )
        decisions.append(
            RetainedDecision(
                reference=StateDecisionRef(key=key, revision=require_count(row["revision"], positive=True)),
                value=_read_value(row["value"], limits),
                supports=frozenset(supports),
            )
        )
    snapshot = ContinuitySnapshot(
        _key=_CONTINUITY_KEY,
        format=SNAPSHOT_FORMAT,
        version=SNAPSHOT_VERSION,
        namespace=namespace,
        revision=revision,
        decisions=tuple(decisions),
    )
    encode_snapshot(snapshot=snapshot, limits=limits)
    return snapshot


def load_snapshot(*, path: os.PathLike[str], limits: ContinuityLimits) -> ContinuitySnapshot:
    require_instance(limits, ContinuityLimits)
    with open(path, "rb") as stream:
        raw = stream.read(limits.max_snapshot_bytes + 1)
    return _decode(raw, limits)


def write_snapshot(*, path: os.PathLike[str], snapshot: ContinuitySnapshot, limits: ContinuityLimits) -> None:
    raw = encode_snapshot(snapshot=snapshot, limits=limits)
    destination = Path(path)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, prefix=".continuity-", delete=False
        ) as stream:
            temporary = stream.name
            stream.write(raw)
            stream.flush()
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass
