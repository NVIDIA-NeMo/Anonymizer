# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

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

    monkeypatch.setattr(regex_detection, "monotonic", lambda: clock[0])
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
    assert "0.05 seconds" in message and "4800-character" in message
    assert "detect.regex_limits.min_timeout_seconds" in message
    assert "detect.regex_limits.timeout_seconds_per_million_characters" in message
    assert "TKT-123" not in message
