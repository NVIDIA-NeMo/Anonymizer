# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest
import regex

from anonymizer import RegexRule
from anonymizer.engine.detection import regex_detection
from anonymizer.engine.detection.regex_detection import (
    calculate_regex_timeout,
    detect_regex_entities,
    resolve_regex_rules,
)


def test_regex_timeout_uses_code_points_and_minimum() -> None:
    assert calculate_regex_timeout(50_000) == 0.05
    assert calculate_regex_timeout(50_001) == pytest.approx(0.050001)
    assert calculate_regex_timeout(len("漢" * 1_000_000)) == 1
    assert calculate_regex_timeout(10, min_timeout_seconds=0.2) == 0.2


@pytest.mark.parametrize(
    "label,text,expected",
    [
        ("ipv6", "\n".join(f"10:{i // 60 % 60:02d}:{i % 60:02d} INFO step {i} ok" for i in range(1200)), 0),
        ("ipv4", "\n".join(f"GET /api from 10.0.{i // 250}.{i % 250} status=200" for i in range(1200)), 1200),
        # Cross-card rotations can also pass Luhn; final overlap resolution is separate.
        ("credit_debit_card", " ".join(["4111 1111 1111 1111"] * 251), 1001),
        ("email", "a@example.org " * 600, 600),
    ],
    ids=["timestamps", "ips", "cards", "emails"],
)
def test_default_regex_limits_handle_ordinary_records(label: str, text: str, expected: int) -> None:
    rules = resolve_regex_rules(labels=[label], builtin_regexes=True, rules=[])
    result = detect_regex_entities(text, rules=rules)
    assert len(result.llm_entities) == expected


def test_rejected_candidates_do_not_spend_match_cap() -> None:
    rules = resolve_regex_rules(labels=["ipv4"], builtin_regexes=True, rules=[])
    text = "999.999.999.999 " * 1100 + "192.0.2.1"
    result = detect_regex_entities(text, rules=rules, max_matches_per_rule=1)
    assert [entity.value for entity in result.llm_entities] == ["192.0.2.1"]
    with pytest.raises(RuntimeError) as exc:
        detect_regex_entities(text + " 192.0.2.2", rules=rules, max_matches_per_rule=1)
    message = str(exc.value)
    assert "detect.regex_limits.max_matches_per_rule" in message
    assert "nemo-anonymizer.ipv4.v1" in message
    assert "1 locally validated matches" in message
    assert "no partial regex result" in message
    assert "192.0.2" not in message


def test_rule_timeout_is_shared_across_scan_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [0.0]

    def validate(candidate: regex_detection.RegexCandidate) -> bool:
        clock[0] += 1  # Python callbacks must not consume the search budget.
        return False

    monkeypatch.setattr(regex_detection, "_rule_cpu_time", lambda: clock[0])
    rules = resolve_regex_rules(
        labels=["ticket"],
        builtin_regexes=False,
        rules=[RegexRule(label="ticket", pattern="TKT-123", validator=validate)],
    )
    compiled = regex_detection._compile_pattern(rules[0].pattern)
    budgets = []

    class Pattern:
        def finditer(self, text: str, *, pos: int, timeout: float, concurrent: bool, overlapped: bool) -> Iterator[Any]:
            budgets.append(timeout)
            clock[0] += 0.03
            return compiled.finditer(text, pos=pos)

    monkeypatch.setattr(regex_detection, "_compile_pattern", lambda _: Pattern())
    with pytest.raises(RuntimeError) as exc:
        detect_regex_entities("TKT-123 " * 600, rules=rules, timeout_seconds=0.05)
    assert budgets == pytest.approx([0.05, 0.02])
    message = str(exc.value)
    assert "0.05 CPU seconds" in message and "4800-character" in message
    assert "detect.regex_limits.min_timeout_seconds" in message
    assert "detect.regex_limits.timeout_seconds_per_million_characters" in message
    assert "TKT-123" not in message


def _ticket_rules() -> list[regex_detection.ResolvedRegexRule]:
    return resolve_regex_rules(
        labels=["ticket"], builtin_regexes=False, rules=[RegexRule(label="ticket", pattern="TKT-123")]
    )


def test_rule_is_not_charged_for_waiting_on_the_gil(monkeypatch: pytest.MonkeyPatch) -> None:
    """A thread holding the GIL for about 0.2 s delays the rule; only the rule's own CPU time counts."""
    rules = _ticket_rules()
    compiled = regex_detection._compile_pattern(rules[0].pattern)
    release_hog, hog_started = threading.Event(), threading.Event()
    hog_timed_out: list[bool] = []

    def hog() -> None:
        release_hog.wait()
        hog_started.set()
        try:
            # Holds the GIL for its whole timeout, which regex measures in process CPU time.
            regex.compile(r"\w+\d").search("a" * 200_000, timeout=0.2, concurrent=False)
        except TimeoutError:
            hog_timed_out.append(True)

    class Pattern:
        def finditer(self, text: str, *, pos: int, timeout: float, concurrent: bool, overlapped: bool) -> Iterator[Any]:
            if not release_hog.is_set():
                release_hog.set()
                hog_started.wait()  # Waking up needs the GIL, which the hog holds.
            return compiled.finditer(text, pos=pos)

    monkeypatch.setattr(regex_detection, "_compile_pattern", lambda _: Pattern())
    thread = threading.Thread(target=hog)
    thread.start()
    try:
        started = time.perf_counter()
        result = detect_regex_entities("TKT-123", rules=rules, timeout_seconds=0.05)
        waited = time.perf_counter() - started
    finally:
        release_hog.set()
        thread.join()
    assert hog_timed_out
    assert [entity.value for entity in result.llm_entities] == ["TKT-123"]
    assert waited >= 0.15


def test_early_regex_timeout_is_rerun_with_the_budget_left(monkeypatch: pytest.MonkeyPatch) -> None:
    """regex's timeout counts other threads' CPU too, so it can fire while this rule still has budget."""
    clock = [0.0]
    monkeypatch.setattr(regex_detection, "_rule_cpu_time", lambda: clock[0])
    rules = _ticket_rules()
    compiled = regex_detection._compile_pattern(rules[0].pattern)
    budgets: list[float] = []

    class Pattern:
        def finditer(self, text: str, *, pos: int, timeout: float, concurrent: bool, overlapped: bool) -> Iterator[Any]:
            budgets.append(timeout)
            clock[0] += 0.01
            if len(budgets) == 1:
                raise TimeoutError("regular expression timed out")
            return compiled.finditer(text, pos=pos)

    monkeypatch.setattr(regex_detection, "_compile_pattern", lambda _: Pattern())
    result = detect_regex_entities("TKT-123", rules=rules, timeout_seconds=0.05)
    assert [entity.value for entity in result.llm_entities] == ["TKT-123"]
    assert budgets[:2] == pytest.approx([0.05, 0.04])


def test_early_timeout_reruns_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """With other threads burning CPU, a rule that keeps being interrupted fails as starved, not as too slow."""
    monkeypatch.setattr(regex_detection, "_rule_cpu_time", lambda: 0.0)
    monkeypatch.setattr(regex_detection, "monotonic", lambda: 0.0)
    rules = _ticket_rules()
    calls = [0]

    class Pattern:
        def finditer(self, text: str, *, pos: int, timeout: float, concurrent: bool, overlapped: bool) -> Iterator[Any]:
            calls[0] += 1
            # Without a rerun limit nothing else would stop this rule, so fail rather than loop forever.
            assert calls[0] <= 100, "early timeouts were rerun without a limit"
            raise TimeoutError("regular expression timed out")

    monkeypatch.setattr(regex_detection, "_compile_pattern", lambda _: Pattern())
    with pytest.raises(RuntimeError) as exc:
        detect_regex_entities("TKT-123", rules=rules, timeout_seconds=0.05)
    assert calls[0] == regex_detection._MAX_EARLY_TIMEOUT_RERUNS + 1
    message = str(exc.value)
    assert "was starved" in message and "interrupted 9 times by CPU use elsewhere" in message
    assert "0 of its 0.05 CPU seconds" in message
    assert "TKT-123" not in message


def test_wall_clock_ceiling_fails_a_starved_rule_with_its_own_message(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [0.0]
    monkeypatch.setattr(regex_detection, "_rule_cpu_time", lambda: 0.0)
    monkeypatch.setattr(regex_detection, "monotonic", lambda: clock[0])
    rules = _ticket_rules()
    compiled = regex_detection._compile_pattern(rules[0].pattern)

    class Pattern:
        def finditer(self, text: str, *, pos: int, timeout: float, concurrent: bool, overlapped: bool) -> Iterator[Any]:
            clock[0] += 0.999  # Under the 1 s ceiling (20 x 0.05 CPU seconds) after one batch, over it after two.
            return compiled.finditer(text, pos=pos)

    monkeypatch.setattr(regex_detection, "_compile_pattern", lambda _: Pattern())
    with pytest.raises(RuntimeError) as starved:
        detect_regex_entities("TKT-123 " * 300, rules=rules, timeout_seconds=0.05, max_matches_per_rule=1000)
    message = str(starved.value)
    assert "was starved: it did not finish within its wall-clock ceiling of 1 second" in message
    assert "0 of its 0.05 CPU seconds" in message and "2400-character" in message
    assert "timed out" not in message and "TKT-123" not in message

    monkeypatch.setattr(regex_detection, "_rule_cpu_time", lambda: clock[0])
    with pytest.raises(RuntimeError) as slow:
        detect_regex_entities("TKT-123 " * 300, rules=rules, timeout_seconds=0.05, max_matches_per_rule=1000)
    assert "timed out after 0.05 CPU seconds" in str(slow.value)
    assert "starved" not in str(slow.value)


def test_garbage_collection_inside_a_rule_is_not_charged_to_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """A collection triggered inside the rule's span costs CPU time, but it is not the rule's work."""
    rules = _ticket_rules()
    compiled = regex_detection._compile_pattern(rules[0].pattern)
    tracked = [[] for _ in range(1_000_000)]  # Enough tracked objects to make a full collection take a while.
    collected: list[float] = []

    class Pattern:
        def finditer(self, text: str, *, pos: int, timeout: float, concurrent: bool, overlapped: bool) -> Iterator[Any]:
            if not collected:
                started = time.thread_time()
                gc.collect()
                collected.append(time.thread_time() - started)
            return compiled.finditer(text, pos=pos)

    monkeypatch.setattr(regex_detection, "_compile_pattern", lambda _: Pattern())
    result = detect_regex_entities("TKT-123", rules=rules, timeout_seconds=0.01)
    assert len(tracked) == 1_000_000
    assert collected[0] > 0.01, "the collection must cost more than the budget for the test to mean anything"
    assert [entity.value for entity in result.llm_entities] == ["TKT-123"]


def test_budget_is_checked_after_each_match_within_a_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """regex restarts its timeout at each match a scanner yields, so the batch's total is checked as it grows."""
    clock = [0.0]
    monkeypatch.setattr(regex_detection, "_rule_cpu_time", lambda: clock[0])
    rules = _ticket_rules()
    compiled = regex_detection._compile_pattern(rules[0].pattern)
    pulled = [0]

    class Pattern:
        def finditer(self, text: str, *, pos: int, timeout: float, concurrent: bool, overlapped: bool) -> Iterator[Any]:
            for match in compiled.finditer(text, pos=pos):
                clock[0] += 0.03  # Each match costs just under the remaining budget on its own.
                pulled[0] += 1
                yield match

    monkeypatch.setattr(regex_detection, "_compile_pattern", lambda _: Pattern())
    with pytest.raises(RuntimeError, match="timed out after 0.05 CPU seconds"):
        detect_regex_entities("TKT-123 " * 300, rules=rules, timeout_seconds=0.05, max_matches_per_rule=1000)
    assert pulled[0] == 2


def test_garbage_collection_inside_a_rule_does_not_count_toward_its_wall_clock_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The thread's own collection holds the GIL and runs, so a long one is not starvation."""
    clock = [0.0]
    monkeypatch.setattr(regex_detection, "monotonic", lambda: clock[0])
    monkeypatch.setattr(regex_detection, "_rule_cpu_time", lambda: 0.0)
    rules = _ticket_rules()
    compiled = regex_detection._compile_pattern(rules[0].pattern)

    class Pattern:
        def finditer(self, text: str, *, pos: int, timeout: float, concurrent: bool, overlapped: bool) -> Iterator[Any]:
            clock[0] += 5.0  # A 5 s collection in this thread, far past the 1 s ceiling.
            monkeypatch.setattr(regex_detection._THREAD_GC_TIME, "total", regex_detection._THREAD_GC_TIME.total + 5.0)
            return compiled.finditer(text, pos=pos)

    monkeypatch.setattr(regex_detection, "_compile_pattern", lambda _: Pattern())
    result = detect_regex_entities("TKT-123", rules=rules, timeout_seconds=0.05)
    assert [entity.value for entity in result.llm_entities] == ["TKT-123"]


def test_wall_clock_ceiling_is_checked_after_each_match_within_a_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Between matches the thread can lose the GIL, so a long batch does not run on past the ceiling."""
    clock = [0.0]
    monkeypatch.setattr(regex_detection, "monotonic", lambda: clock[0])
    monkeypatch.setattr(regex_detection, "_rule_cpu_time", lambda: 0.0)
    rules = _ticket_rules()
    compiled = regex_detection._compile_pattern(rules[0].pattern)
    pulled = [0]

    class Pattern:
        def finditer(self, text: str, *, pos: int, timeout: float, concurrent: bool, overlapped: bool) -> Iterator[Any]:
            for match in compiled.finditer(text, pos=pos):
                clock[0] += 0.4  # Waiting for the GIL between matches: no CPU time, 0.4 s of wall time.
                pulled[0] += 1
                yield match

    monkeypatch.setattr(regex_detection, "_compile_pattern", lambda _: Pattern())
    with pytest.raises(RuntimeError, match="wall-clock ceiling of 1 second"):
        detect_regex_entities("TKT-123 " * 300, rules=rules, timeout_seconds=0.05, max_matches_per_rule=1000)
    assert pulled[0] == 3


def test_validator_time_between_batches_does_not_count_toward_the_wall_clock_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validators run while the scan waits between batches; slow ones are not a starved scan."""
    clock = [0.0]
    monkeypatch.setattr(regex_detection, "monotonic", lambda: clock[0])
    monkeypatch.setattr(regex_detection, "_rule_cpu_time", lambda: 0.0)

    def slow(candidate: regex_detection.RegexCandidate) -> bool:
        clock[0] += 1.0  # Each validation takes as long as the whole 1 s ceiling.
        return True

    rules = resolve_regex_rules(
        labels=["ticket"],
        builtin_regexes=False,
        rules=[RegexRule(label="ticket", pattern="TKT-123", validator=slow)],
    )
    result = detect_regex_entities("TKT-123 " * 5, rules=rules, timeout_seconds=0.05)
    assert len(result.llm_entities) == 5
    assert clock[0] == 5.0
