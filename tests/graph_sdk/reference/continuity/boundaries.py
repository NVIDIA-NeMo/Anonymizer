# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TypeAlias

JSON: TypeAlias = str | int | bool | None | list["JSON"] | dict[str, "JSON"]


def fields(value: JSON, expected: set[str]) -> dict[str, JSON]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("closed shape")
    return value


def integer(value: JSON, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError("integer")
    return value


def text(value: JSON, *, name: bool = False) -> str:
    if type(value) is not str or (name and not value):
        raise ValueError("text")
    value.encode("utf-8", errors="strict")
    return value


def semantic_ref(value: JSON) -> tuple[str, str, int]:
    member = fields(value, {"kind", "name", "version"})
    return (
        text(member["kind"], name=True),
        text(member["name"], name=True),
        integer(member["version"], 1),
    )


def artifact(value: JSON) -> None:
    if not isinstance(value, dict):
        raise ValueError("tagged artifact")
    if value.get("tag") == "text":
        text(fields(value, {"tag", "text"})["text"])
        return
    items = fields(value, {"tag", "items"})
    if items["tag"] != "collection" or not isinstance(items["items"], list):
        raise ValueError("closed artifact tag")
    order = []
    for item in items["items"]:
        row = fields(item, {"key", "version", "text"})
        order.append((integer(row["key"], 0), integer(row["version"], 1)))
        text(row["text"])
    if order != sorted(set(order)):
        raise ValueError("item order/uniqueness")


def snapshot(value: JSON) -> None:
    root = fields(value, {"format", "version", "namespace", "revision", "decisions"})
    if root["format"] != "nemo-anonymizer-continuity" or integer(root["version"], 0) != 2:
        raise ValueError("unsupported format/version")
    text(root["namespace"], name=True)
    integer(root["revision"], 0)
    if not isinstance(root["decisions"], list):
        raise ValueError("decisions")
    keys = []
    for decision in root["decisions"]:
        row = fields(decision, {"group", "subject", "slot", "revision", "value", "supports"})
        keys.append(
            (
                text(row["group"], name=True),
                semantic_ref(row["subject"]),
                text(row["slot"], name=True),
            )
        )
        integer(row["revision"], 1)
        artifact(row["value"])
        if not isinstance(row["supports"], list) or not row["supports"]:
            raise ValueError("supports")
        supports = [semantic_ref(support) for support in row["supports"]]
        if supports != sorted(set(supports)):
            raise ValueError("support order/uniqueness")
    if keys != sorted(set(keys)):
        raise ValueError("decision order/uniqueness")


def canonical(value: JSON) -> bytes:
    snapshot(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def no_duplicates(pairs: list[tuple[str, JSON]]) -> dict[str, JSON]:
    if len({key for key, _ in pairs}) != len(pairs):
        raise ValueError("duplicate JSON key")
    return dict(pairs)


def decode_small(raw: bytes, bound: int) -> JSON:
    """Tiny fixture decoder, NOT a production bounded parser/depth reference."""
    if len(raw) > bound:
        raise ValueError("bytes")
    value: JSON = json.loads(raw.decode("utf-8", errors="strict"), object_pairs_hook=no_duplicates)
    snapshot(value)
    return value


@dataclass(frozen=True)
class State:
    identity: str
    namespace: str
    revision: int
    decision_revisions: tuple[int, ...]


def copy_state(source: State, mode: str, fresh_identity: str, namespace: str) -> State:
    if fresh_identity == source.identity:
        raise ValueError("fresh identity")
    if mode == "fresh":
        return State(fresh_identity, namespace, 0, ())
    if mode == "fork":
        if namespace == source.namespace:
            raise ValueError("fork namespace")
        return State(fresh_identity, namespace, 1, source.decision_revisions)
    if mode not in {"restore", "handoff"} or namespace != source.namespace:
        raise ValueError("mode/namespace")
    return State(fresh_identity, namespace, source.revision, source.decision_revisions)


def lifetime(mode: str, supplied_identity: str | None) -> bool:
    return mode == "per_call" and supplied_identity is None or mode == "across_calls" and supplied_identity is not None


def protocol(events: tuple[str, ...], *, caller_owned: bool) -> bool:
    """Single finite physical request: first terminal and late settlement are separate."""
    alphabet = {
        "reserve",
        "dispatch",
        "success",
        "failure",
        "lost",
        "cancelled",
        "settle",
        "close",
        "trusted_stop",
    }
    if not set(events) <= alphabet or len(events) != len(set(events)):
        return False
    if events.count("reserve") != 1 or events.count("dispatch") > 1:
        return False
    if "dispatch" in events and events.index("dispatch") < events.index("reserve"):
        return False
    terminals = [e for e in events if e in {"success", "failure", "lost", "cancelled"}]
    if events.count("settle") > 1 or len(terminals) != 1:
        return False
    terminal = terminals[0]
    if events.index(terminal) < events.index("reserve"):
        return False
    if terminal != "cancelled" and ("dispatch" not in events or events.index(terminal) < events.index("dispatch")):
        return False
    if "settle" in events and ("dispatch" not in events or events.index("settle") < events.index("dispatch")):
        return False
    if "trusted_stop" in events and (
        "dispatch" not in events or events.index("trusted_stop") < events.index("dispatch")
    ):
        return False
    if "cancelled" in events and "dispatch" in events:
        if "trusted_stop" not in events or events.index("trusted_stop") > events.index("cancelled"):
            return False
    if "close" in events:
        if caller_owned or "settle" not in events or events.index("close") < events.index("settle"):
            return False
    return True
