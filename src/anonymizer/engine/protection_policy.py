# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One place that decides what happens to a record in each protection situation.

Every decision site asks ``assess(situation, mode, detector)`` for a ``Verdict`` instead of writing its own: whether a
missed repeat fails, is protected or is released; whether a kept value or a placeholder flags the row for review;
whether a failing candidate is withheld. ``POLICY`` holds one row for every situation, mode and detector kind, so
Replace and Rewrite cannot drift apart without a visible table change. The graph kernel stays mode-agnostic: its
verifiers report facts, and the sites that turn those facts into release decisions read this table.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import product
from typing import Literal, get_args

from anonymizer.engine.replace.llm_replace_workflow import is_placeholder
from anonymizer.engine.rewrite.parsers import normalize_payload

Mode = Literal["replace", "rewrite"]
Detector = Literal["builtin", "custom"]
Action = Literal["fail", "protect", "release", "flag", "withhold", "pass"]
Situation = Literal[
    # A reported value occurs in the text outside every kept span.
    "missed_repeat",
    # The output keeps a replaced value that the source also holds outside the detected spans (within the bound).
    "undetected_value_kept",
    # The output keeps a placeholder substitute of the row's own replacement map.
    "placeholder_in_text",
    # The final candidate fails a privacy check (high_leak or leakage_within_threshold).
    "privacy_check_failed",
    # The privacy evaluation of the final candidate had no usable response.
    "privacy_unassessed",
    # A detector neither answered nor failed an input record.
    "dropped_record",
    # An override runner's output row cannot be placed against its input (no id, an unknown or a repeated id).
    "unplaced_output",
    # A step stopped early after too many rows without a usable model response.
    "stopped_early",
]


@dataclass(frozen=True)
class Verdict:
    """What happens to the record: ``action``, and for ``fail`` the failure ``reason`` it is listed with."""

    action: Action
    reason: str = ""


def _rows(situation: Situation, verdicts: dict[tuple[Mode, Detector], Verdict]) -> dict[tuple[str, str, str], Verdict]:
    return {(situation, mode, detector): verdict for (mode, detector), verdict in verdicts.items()}


def _all(verdict: Verdict) -> dict[tuple[Mode, Detector], Verdict]:
    return {(mode, detector): verdict for mode, detector in product(get_args(Mode), get_args(Detector))}


POLICY: dict[tuple[str, str, str], Verdict] = {
    # Replace replaces spans only: a custom detector must span every occurrence (fail); the built-in detector's
    # missed repeat gets a span (protect, ruled 432/repeat-variant V2). Rewrite passes it to the rewriter in its
    # replacement view, as before (ruled contract/open-rows).
    **_rows(
        "missed_repeat",
        {
            ("replace", "custom"): Verdict("fail", "unspanned_value"),
            ("replace", "builtin"): Verdict("protect"),
            ("rewrite", "custom"): Verdict("release"),
            ("rewrite", "builtin"): Verdict("release"),
        },
    ),
    # Rewrite flags a kept value for review and keeps the row met (#426). Replace's own check
    # (no_detected_value_added) fails over the bound and so never reaches this situation.
    **_rows(
        "undetected_value_kept",
        {
            ("replace", "custom"): Verdict("pass"),
            ("replace", "builtin"): Verdict("pass"),
            ("rewrite", "custom"): Verdict("flag"),
            ("rewrite", "builtin"): Verdict("flag"),
        },
    ),
    # A placeholder leaks nothing but needs a reader: both modes flag the row, and it stays met.
    **_rows("placeholder_in_text", _all(Verdict("flag"))),
    # A failing final candidate is withheld, repaired or not (#431 R1). Replace has no such candidate.
    **_rows(
        "privacy_check_failed",
        {
            ("replace", "custom"): Verdict("pass"),
            ("replace", "builtin"): Verdict("pass"),
            ("rewrite", "custom"): Verdict("withhold"),
            ("rewrite", "builtin"): Verdict("withhold"),
        },
    ),
    **_rows(
        "privacy_unassessed",
        {
            ("replace", "custom"): Verdict("pass"),
            ("replace", "builtin"): Verdict("pass"),
            ("rewrite", "custom"): Verdict("withhold"),
            ("rewrite", "builtin"): Verdict("withhold"),
        },
    ),
    **_rows("dropped_record", _all(Verdict("fail", "dropped_output"))),
    **_rows("unplaced_output", _all(Verdict("fail", "unplaced_output"))),
    # #431's early stop: each row not run is listed with ``stopped_early (<step>)``; this is its reason code.
    **_rows("stopped_early", _all(Verdict("fail", "stopped_early"))),
}


def assess(situation: Situation, mode: Mode, detector: Detector) -> Verdict:
    """The table's verdict for ``situation`` in ``mode`` with a ``detector`` of that kind."""
    return POLICY[(situation, mode, detector)]


def repeat_handling(mode: Mode, detector: Detector) -> Literal["fail", "protect", "release"]:
    """The ``missed_repeat`` action, as the detect step's ``repeats`` takes it."""
    action = assess("missed_repeat", mode, detector).action
    if action not in ("fail", "protect", "release"):
        raise ValueError(f"missed_repeat has no detect-step handling for {action!r}")
    return action


def fail_reasons() -> frozenset[str]:
    """Every failure reason the table can give a record."""
    return frozenset(v.reason for v in POLICY.values() if v.action == "fail")


def placeholder_in_text(text: str, replacement_map: object) -> bool:
    """Whether ``text`` holds a placeholder substitute of the row's replacement map; ``False`` without a map.

    A placeholder stands in for a substitute that collided with a value in the record. It leaks nothing, so the row
    stays met, but text that carries one needs review. The match is loose: any case, brackets optional, ``_``, ``-``
    or one space between words, and the separator before the number optional. The number is required.
    """
    payload = normalize_payload(replacement_map)
    entries = payload.get("replacements") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return False
    folded = text.casefold()
    return any(
        _placeholder_pattern(entry["synthetic"]).search(folded)
        for entry in entries
        if isinstance(entry, dict)
        and isinstance(entry.get("label"), str)
        and isinstance(entry.get("synthetic"), str)
        and is_placeholder(entry["label"], entry["synthetic"])
    )


def _placeholder_pattern(placeholder: str) -> re.Pattern[str]:
    """The loose casefolded pattern of ``placeholder``, an ``is_placeholder`` substitute. Its words are the minted
    token's own, split at each ``_`` with empty items kept, so a label's punctuation runs match as minted."""
    token, number = placeholder.casefold()[1:-1].rsplit("_", 1)
    number = re.escape(number)
    separator = r"[_\- ]"
    body = separator.join(re.escape(word) for word in token.split("_"))
    return re.compile(rf"(?<![^\W_]){body}{separator}?{number}(?!\d)")
