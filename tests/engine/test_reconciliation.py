# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reconciling detector observations: "May met Lee", where two views read May as a name and as a month."""

from __future__ import annotations

from anonymizer.engine.detection.reconciliation import (
    Accept,
    Accepted,
    Mention,
    Observation,
    ObservationGroup,
    RejectAll,
    Undecided,
    UnresolvedCause,
    Withheld,
    reconcile,
)
from anonymizer.engine.spans import Span

TEXT = "May met Lee"
MAY_PERSON = Observation("people", Span(0, 3), "May", "first_name")
MAY_MONTH = Observation("dates", Span(0, 3), "May", "date")
LEE = Observation("people", Span(8, 11), "Lee", "last_name")
LEE_AGAIN = Observation("dates", Span(8, 11), "Lee", "last_name")


def _never(group: ObservationGroup) -> Accept:
    raise AssertionError(f"the resolver was asked about {group}")


def test_no_observations_accept_nothing() -> None:
    assert reconcile((), _never) == Accepted(())


def test_an_unresolved_group_withholds_the_target_and_hands_out_no_mentions() -> None:
    """Before, a caller reading only the kept observations saw a plausible partial success (Lee alone)."""
    for observations in ((MAY_PERSON, MAY_MONTH), (MAY_PERSON, MAY_MONTH, LEE, LEE_AGAIN)):
        result = reconcile(observations, lambda group: Undecided())
        assert isinstance(result, Withheld)
        assert result.cause is UnresolvedCause.UNDECIDED_GROUP
        assert [group.labels for group in result.groups] == [("first_name", "date")]
        assert not hasattr(result, "mentions")


def test_a_resolver_that_accepts_one_reading() -> None:
    result = reconcile((MAY_PERSON, MAY_MONTH, LEE, LEE_AGAIN), lambda group: Accept("first_name"))
    assert result == Accepted((Mention(Span(0, 3), "May", "first_name"), Mention(Span(8, 11), "Lee", "last_name")))


def test_a_resolver_that_legitimately_rejects_every_reading() -> None:
    result = reconcile((MAY_PERSON, MAY_MONTH, LEE), lambda group: RejectAll("a month; the goal covers people"))
    assert isinstance(result, Accepted)
    assert result.mentions == (Mention(Span(8, 11), "Lee", "last_name"),)
    assert [(group.span, reason) for group, reason in result.rejected] == [
        (Span(0, 3), "a month; the goal covers people")
    ]


def test_an_accepted_label_no_view_observed_is_undecided() -> None:
    result = reconcile((MAY_PERSON, MAY_MONTH), lambda group: Accept("city"))
    assert isinstance(result, Withheld) and result.cause is UnresolvedCause.UNDECIDED_GROUP


def test_an_independent_target_passes() -> None:
    assert reconcile((LEE, LEE_AGAIN), _never) == Accepted((Mention(Span(8, 11), "Lee", "last_name"),))


def test_a_missing_required_view_withholds_before_any_decision() -> None:
    result = reconcile((LEE,), _never, required_views=("people", "dates"))
    assert result == Withheld(UnresolvedCause.MISSING_VIEW, missing_views=("dates",))


def test_a_required_view_that_answered_with_no_findings_is_not_missing() -> None:
    result = reconcile((LEE,), _never, required_views=("people", "dates"), answered_views=("people", "dates"))
    assert result == Accepted((Mention(Span(8, 11), "Lee", "last_name"),))
    still_missing = reconcile((LEE,), _never, required_views=("people", "dates"), answered_views=("people",))
    assert still_missing == Withheld(UnresolvedCause.MISSING_VIEW, missing_views=("dates",))
