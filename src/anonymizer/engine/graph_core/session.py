# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session: caller-held state that lets assignments outlive one :func:`protect` call.

A Session stores assignments under a slot of three strings, ``(subject, label, normalized source value)``, and knows
nothing about entities. Operations read it through a :class:`SessionView` seed and change it only by emitting
:class:`Decision` values on an output port of kind ``decisions``; ``protect`` accepts them once per invocation.
"""

from __future__ import annotations

import contextlib
import json
import threading
from collections import Counter
from collections.abc import Generator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from anonymizer.engine.graph_core.kernel import GraphRejection, register_kind

NORMALIZATIONS = frozenset({"casefold-whitespace-v1"})
_FORMAT, _VERSION = "anonymizer.session", 1


def normalize(value: str) -> str:
    """The ``casefold-whitespace-v1`` normalization: casefolded, with each run of whitespace one space."""
    return " ".join(value.casefold().split())


@dataclass(frozen=True)
class SubjectKey:
    """A subject declared by the caller and scoped to a Session; the SDK never derives one from text."""

    value: str = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or not self.value:
            raise GraphRejection("invalid_subject_key")
        if self.value in _RESERVED_SUBJECTS:
            raise GraphRejection("reserved_subject_key")

    @classmethod
    def _reserved(cls, value: str) -> SubjectKey:
        key = object.__new__(cls)
        object.__setattr__(key, "value", value)
        return key


def _subject(value: str) -> SubjectKey:
    """The subject a snapshot names: a reserved one, or one a caller may construct."""
    return _RESERVED_SUBJECTS.get(value) or SubjectKey(value)


Slot = tuple[SubjectKey, str, str]  # (subject, label, normalized source value)
# Subjects only the SDK constructs; ``SubjectKey(...)`` rejects their values. A by-value slot holds an assignment
# for a label and value. A fallback slot is bookkeeping, not an assignment: it is keyed by a substitute used once in
# place of an assignment, so no lookup for a value returns it, and it only marks that substitute as taken.
BY_VALUE = SubjectKey._reserved("by-value")
FALLBACK = SubjectKey._reserved("substitute-fallback")
_RESERVED_SUBJECTS = {key.value: key for key in (BY_VALUE, FALLBACK)}


def _check_value(value: object) -> None:
    if not isinstance(value, str):
        raise GraphRejection("invalid_value")


def _check_slot(slot: object) -> Slot:
    if not (
        isinstance(slot, tuple)
        and len(slot) == 3
        and isinstance(slot[0], SubjectKey)
        and isinstance(slot[1], str)
        and isinstance(slot[2], str)
    ):
        raise GraphRejection("invalid_slot")
    return slot


@dataclass(frozen=True)
class SessionMode:
    """What a Session retains: ``stable_replacements`` keeps assignments across calls; ``identity_matching`` is not
    supported yet. ``normalization`` names the fold's policy and is part of snapshot compatibility."""

    stable_replacements: bool = False
    identity_matching: bool = False
    normalization: str = "casefold-whitespace-v1"


def _mode_valid(stable: object, matching: object) -> bool:
    return type(stable) is bool and type(matching) is bool and (stable or not matching)


@dataclass(frozen=True)
class Decision:
    """An operation's assignment for one slot; ``supersedes`` replaces a retained assignment as a recorded revision.
    Its repr shows the label only."""

    slot: Slot
    value: str = field(repr=False)
    supersedes: bool = False

    def __post_init__(self) -> None:
        _check_slot(self.slot)
        if not isinstance(self.value, str) or type(self.supersedes) is not bool:
            raise GraphRejection("invalid_decision")

    def __repr__(self) -> str:
        return f"Decision(label={self.slot[1]!r}, supersedes={self.supersedes})"


@dataclass(frozen=True)
class SessionChanges:
    """Content-free counts of one invocation's acceptance: new slots, equal re-emissions, revisions, and decisions
    that differ from a retained assignment without superseding it (kept, not applied); ``fallbacks`` of ``accepted``."""

    accepted: int = 0
    reused: int = 0
    revised: int = 0
    ignored: int = 0
    fallbacks: int = 0


@dataclass(frozen=True)
class _Entry:
    assignment: str
    revision: int
    by: str
    superseded: tuple[tuple[str, int, str], ...] = ()


class SessionView:
    """What an operation reads: the assignments at the revision captured when the invocation started. It answers
    lookups and membership questions only, offers no iteration or listing, and cannot be pickled."""

    __slots__ = ("_assigned", "_assignments", "_mode", "_originals", "_revision")

    def __init__(self, mode: SessionMode, revision: int, assignments: dict[Slot, str]) -> None:
        self._mode = mode
        self._revision = revision
        self._assignments: Mapping[Slot, str] = MappingProxyType(assignments)
        self._assigned: dict[str, set[Slot]] = {}
        for slot, value in assignments.items():
            self._assigned.setdefault(normalize(value), set()).add(slot)
        # A slot's value is normalized here too: nothing makes an operation's Decision carry a normalized one.
        self._originals = frozenset(normalize(slot[2]) for slot in assignments if slot[0] != FALLBACK)

    @property
    def mode(self) -> SessionMode:
        return self._mode

    def lookup(self, slot: Slot) -> str | None:
        return self._assignments.get(_check_slot(slot))

    def assigned_elsewhere(self, value: str, slot: Slot) -> bool:
        """Whether ``value``, normalized, is the assignment of a slot other than ``slot``."""
        _check_slot(slot)
        _check_value(value)
        return any(other != slot for other in self._assigned.get(normalize(value), ()))

    def is_original(self, value: str) -> bool:
        """Whether ``value``, normalized, is the source value of some slot; fallback slots hold none."""
        _check_value(value)
        return normalize(value) in self._originals

    def __repr__(self) -> str:
        return f"SessionView(mode={self._mode!r}, revision={self._revision})"

    def __reduce__(self) -> Any:
        raise TypeError("SessionView is process-local and not serializable")


@dataclass(frozen=True)
class SessionSnapshot:
    """An immutable copy of a Session. ``to_bytes`` holds original values and subject keys in plain text."""

    mode: SessionMode
    revision: int
    _slots: tuple[tuple[Slot, _Entry], ...] = field(default=(), repr=False)

    def __repr__(self) -> str:
        return f"SessionSnapshot(mode={self.mode!r}, revision={self.revision}, slots={len(self._slots)})"

    def to_bytes(self) -> bytes:
        slots = [
            {
                "subject": slot[0].value,
                "label": slot[1],
                "value": slot[2],
                "assignment": e.assignment,
                "revision": e.revision,
                "by": e.by,
                "superseded": [{"assignment": a, "revision": r, "by": b} for a, r, b in e.superseded],
            }
            for slot, e in self._slots
        ]
        mode = {
            "stable_replacements": self.mode.stable_replacements,
            "identity_matching": self.mode.identity_matching,
            "normalization": self.mode.normalization,
        }
        doc = {"format": _FORMAT, "version": _VERSION, "mode": mode, "revision": self.revision, "slots": slots}
        return json.dumps(doc, ensure_ascii=False, sort_keys=True).encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> SessionSnapshot:
        """Parse ``to_bytes`` output; anything else is ``GraphRejection("snapshot_incompatible")``."""
        if not isinstance(data, (bytes, bytearray)):
            raise GraphRejection("snapshot_incompatible", "not bytes")
        try:
            doc = json.loads(bytes(data).decode("utf-8"))
        except (ValueError, RecursionError):
            raise GraphRejection("snapshot_incompatible", "not json") from None
        if not isinstance(doc, dict) or doc.get("format") != _FORMAT:
            raise GraphRejection("snapshot_incompatible", "format")
        if doc.get("version") != _VERSION or type(doc.get("version")) is not int:
            raise GraphRejection("snapshot_incompatible", "version")
        mode = doc.get("mode")
        if not isinstance(mode, dict) or set(mode) != {"stable_replacements", "identity_matching", "normalization"}:
            raise GraphRejection("snapshot_incompatible", "mode")
        if not isinstance(mode["normalization"], str) or mode["normalization"] not in NORMALIZATIONS:
            raise GraphRejection("snapshot_incompatible", "normalization")
        if not _mode_valid(mode["stable_replacements"], mode["identity_matching"]):
            raise GraphRejection("snapshot_incompatible", "mode")
        revision, raw = doc.get("revision"), doc.get("slots")
        if not _count(revision) or not isinstance(raw, list):
            raise GraphRejection("snapshot_incompatible", "revision")
        try:
            slots = tuple(_parse_slot(s) for s in raw)
        except (GraphRejection, KeyError, TypeError, ValueError):
            raise GraphRejection("snapshot_incompatible", "slot") from None
        if len({slot for slot, _ in slots}) != len(slots) or any(
            e.revision > revision or any(r > revision for _, r, _ in e.superseded) for _, e in slots
        ):
            raise GraphRejection("snapshot_incompatible", "slot")
        return cls(SessionMode(**mode), revision, slots)


def _count(v: object) -> bool:
    return type(v) is int and v >= 0


def _parse_slot(s: dict[str, Any]) -> tuple[Slot, _Entry]:
    texts = [s["subject"], s["label"], s["value"], s["assignment"], s["by"]]
    if not all(isinstance(t, str) for t in texts) or not _count(s["revision"]):
        raise ValueError
    superseded = tuple((p["assignment"], p["revision"], p["by"]) for p in s["superseded"])
    if not all(isinstance(a, str) and _count(r) and isinstance(b, str) for a, r, b in superseded):
        raise ValueError
    slot = (_subject(s["subject"]), s["label"], s["value"])
    return slot, _Entry(s["assignment"], s["revision"], s["by"], superseded)


class Session:
    """Caller-held state passed to :func:`prepare` and :func:`protect`; the SDK persists nothing.

    Matching without stability is rejected here (``session_mode_invalid``); matching with stability constructs but
    is rejected at ``prepare`` (``session_mode_unsupported``). One execution uses a Session at a time: ``protect``
    and ``snapshot`` raise ``session_busy`` instead of waiting. A Session cannot be pickled or copied with
    :mod:`copy`; use :meth:`snapshot` and :meth:`restore`.
    """

    def __init__(self, *, stable_replacements: bool = False, identity_matching: bool = False) -> None:
        if not _mode_valid(stable_replacements, identity_matching):
            raise GraphRejection("session_mode_invalid")
        self._mode = SessionMode(stable_replacements, identity_matching)
        self._revision = 0
        self._slots: dict[Slot, _Entry] = {}
        self._lock = threading.Lock()

    @property
    def mode(self) -> SessionMode:
        return self._mode

    @property
    def size(self) -> int:
        """Assignments held; fallback slots are left out."""
        return sum(slot[0] is not FALLBACK for slot in list(self._slots))  # one-step copy: safe during protect

    def label_counts(self) -> dict[str, int]:
        """Assignments held per label, sorted by label; fallback slots are left out."""
        with self._hold():
            counts = Counter(slot[1] for slot in self._slots if slot[0] is not FALLBACK)
        return dict(sorted(counts.items()))

    @property
    def revision(self) -> int:
        """A high-water mark, not one step per call: a call that supersedes a slot it fixed advances it by more than
        one, and entries accepted in one call can carry different revisions."""
        return self._revision

    def snapshot(self) -> SessionSnapshot:
        with self._hold():
            return SessionSnapshot(self._mode, self._revision, tuple(self._slots.items()))

    @classmethod
    def restore(cls, snapshot: SessionSnapshot) -> Session:
        """An independent live Session holding the snapshot's state."""
        if not isinstance(snapshot, SessionSnapshot):
            raise GraphRejection("snapshot_incompatible", "not a snapshot")
        session = cls()
        session._mode, session._revision, session._slots = snapshot.mode, snapshot.revision, dict(snapshot._slots)
        return session

    def __repr__(self) -> str:
        return f"Session(mode={self._mode!r}, revision={self._revision}, slots={len(self._slots)})"

    def __reduce__(self) -> Any:
        raise TypeError("Session is process-local and not serializable; use snapshot()")

    @contextlib.contextmanager
    def _hold(self) -> Generator[None]:
        if not self._lock.acquire(blocking=False):
            raise GraphRejection("session_busy")
        try:
            yield
        finally:
            self._lock.release()

    def _view(self) -> SessionView:
        return SessionView(self._mode, self._revision, {s: e.assignment for s, e in self._slots.items()})

    def _accept(self, decisions: list[tuple[Decision, str]]) -> SessionChanges:
        """Apply one invocation's decisions in order; the first fixes a new slot and an explicit supersede revises it,
        in this invocation or a later one, at a later revision than the one it replaces. Off/off retains nothing."""
        if not self._mode.stable_replacements:
            return SessionChanges()
        accepted = reused = revised = ignored = fallbacks = 0
        revision = self._revision + 1
        for d, by in decisions:
            current = self._slots.get(d.slot)
            if current is None:
                self._slots[d.slot] = _Entry(d.value, revision, by)
                accepted += 1
                fallbacks += d.slot[0] is FALLBACK
            elif current.assignment == d.value:
                reused += 1
            elif d.supersedes:
                revision = max(revision, current.revision + 1)
                old = (current.assignment, current.revision, current.by)
                self._slots[d.slot] = _Entry(d.value, revision, by, (*current.superseded, old))
                revised += 1
            else:
                ignored += 1
        if accepted or revised:
            self._revision = revision
        return SessionChanges(accepted, reused, revised, ignored, fallbacks)


SESSION_VIEW = register_kind("session-view", lambda v: isinstance(v, SessionView))
DECISIONS = register_kind(
    "decisions", lambda v: isinstance(v, tuple) and all(isinstance(d, Decision) for d in v), copy=tuple
)
