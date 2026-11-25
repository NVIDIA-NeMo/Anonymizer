# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reconciling detector observations into accepted mentions, with a declared outcome for a conflict nobody resolved.

Internal: no public name, and no caller in the run path yet. A detector that offers
observations instead of final spans hands each target's observations here. Observations of one occurrence (the same
span) that disagree on the label form an ``ObservationGroup``; a resolver decides each group. The result is either
``Accepted``, with every accepted mention, or ``Withheld``, naming the groups no decision covered. A caller that needs
accepted mentions cannot read a partial success from a withheld target: ``Withheld`` holds no mentions.

A resolver's answers:

- ``Accept(label)``: one of the group's readings stands;
- ``RejectAll(reason)``: a valid decision that none stands (for example, "May" is a month and the goal covers people
  only); the occurrence gets no mention;
- ``Undecided()``: the resolver could not decide. The target is withheld with ``UnresolvedCause.UNDECIDED_GROUP``.

A required view that never answered withholds the target with ``UnresolvedCause.MISSING_VIEW`` before any group is
decided, since a decision among the observations present would still miss evidence.

Current limits, to settle before a caller uses it:

- Only observations with identical spans form a group: nested or crossing observations are separate groups, so both
  are ``Accepted``.
- Labels are compared as given, without the shared label fold (``normalize_entity_label``).
- No value or grounding check: an observation's value is not compared with the target's text.
- ``ObservationGroup`` names no target; the caller reconciles one target at a time.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum

from anonymizer.engine.spans import BY_START_END, Span


@dataclass(frozen=True, slots=True)
class Observation:
    """One view's reading of one occurrence: its span on the target, its value and label."""

    view: str
    span: Span
    value: str
    label: str


@dataclass(frozen=True, slots=True)
class ObservationGroup:
    """The observations of one occurrence; a conflict when their labels disagree."""

    span: Span
    observations: tuple[Observation, ...]

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(o.label for o in self.observations))

    @property
    def conflicting(self) -> bool:
        return len(self.labels) > 1


@dataclass(frozen=True, slots=True)
class Accept:
    label: str


@dataclass(frozen=True, slots=True)
class RejectAll:
    reason: str


@dataclass(frozen=True, slots=True)
class Undecided:
    pass


Decision = Accept | RejectAll | Undecided
Resolver = Callable[[ObservationGroup], Decision]


@dataclass(frozen=True, slots=True)
class Mention:
    span: Span
    value: str
    label: str


class UnresolvedCause(StrEnum):
    UNDECIDED_GROUP = "undecided_group"
    MISSING_VIEW = "missing_view"


@dataclass(frozen=True, slots=True)
class Accepted:
    """Every group is accounted for: the accepted mentions, and the groups rejected with their reasons."""

    mentions: tuple[Mention, ...]
    rejected: tuple[tuple[ObservationGroup, str], ...] = ()


@dataclass(frozen=True, slots=True)
class Withheld:
    """The target is not released: ``cause``, and the groups (or views) it names. It holds no mentions."""

    cause: UnresolvedCause
    groups: tuple[ObservationGroup, ...] = ()
    missing_views: tuple[str, ...] = ()


Reconciliation = Accepted | Withheld


def group_observations(observations: Iterable[Observation]) -> tuple[ObservationGroup, ...]:
    """Observations grouped by span, in ``BY_START_END`` order; two equal values at different spans are two
    occurrences."""
    by_span: dict[Span, list[Observation]] = {}
    for observation in observations:
        by_span.setdefault(observation.span, []).append(observation)
    spans = BY_START_END.sorted(by_span, span=lambda span: span)
    return tuple(ObservationGroup(span, tuple(by_span[span])) for span in spans)


def reconcile(
    observations: Iterable[Observation],
    resolver: Resolver,
    *,
    required_views: Iterable[str] = (),
    answered_views: Iterable[str] | None = None,
) -> Reconciliation:
    """One target's observations as accepted mentions, or withheld with a typed cause.

    ``answered_views`` names the views that answered, findings or not, so a view that found nothing is not missing.
    When it is not given, a view counts as answered only if it made an observation.

    A group whose observations agree is accepted without asking the resolver; a conflicting group goes to it.
    """
    items = tuple(observations)
    answered = {o.view for o in items} | set(answered_views or ())
    if missing := tuple(view for view in dict.fromkeys(required_views) if view not in answered):
        return Withheld(UnresolvedCause.MISSING_VIEW, missing_views=missing)
    mentions: list[Mention] = []
    rejected: list[tuple[ObservationGroup, str]] = []
    undecided: list[ObservationGroup] = []
    for group in group_observations(items):
        decision = resolver(group) if group.conflicting else Accept(group.labels[0])
        match decision:
            case Accept(label) if label in group.labels:
                mentions.append(Mention(group.span, group.observations[0].value, label))
            case RejectAll(reason):
                rejected.append((group, reason))
            case _:  # undecided, or a label no view observed
                undecided.append(group)
    if undecided:
        return Withheld(UnresolvedCause.UNDECIDED_GROUP, groups=tuple(undecided))
    return Accepted(tuple(mentions), tuple(rejected))
