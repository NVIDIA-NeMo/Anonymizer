# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from anonymizer import BuiltinRegex, RegexCandidate, RegexRule, RegexValidationResult
from anonymizer.engine.detection.postprocess import expand_entity_occurrences
from anonymizer.engine.detection.regex_detection import (
    detect_regex_entities,
    resolve_regex_rules,
    validate_exportable_regex_rules,
)


@pytest.mark.parametrize(
    ("label", "text", "expected"),
    [
        ("credit_debit_card", "Card 4111 1111 1111 1111.", "4111 1111 1111 1111"),
        ("email", "联系：用户@example.公司，处理", "用户@example.公司"),
        ("ipv4", "Address 192.168.1.10.", "192.168.1.10"),
        ("ipv6", "地址2001:db8::1结束", "2001:db8::1"),
        ("mac_address", "MAC 00:1A:2B:3C:4D:5E", "00:1A:2B:3C:4D:5E"),
        ("url", "请访问https://例子.公司/路径?值=一。", "https://例子.公司/路径?值=一"),
    ],
)
def test_builtin_regexes_detect_valid_values(label: str, text: str, expected: str) -> None:
    rules = resolve_regex_rules(
        labels=[label],
        builtin_regexes=True,
        rules=[],
    )

    result = detect_regex_entities(text, rules=rules)

    assert [entity.value for entity in result.llm_entities] == [expected]
    assert result.accepted_entities == []


@pytest.mark.parametrize(
    ("label", "text"),
    [
        ("credit_debit_card", "Card 4111 1111 1111 1112."),
        ("ipv4", "Address 999.168.1.10."),
        ("url", "Visit http://localhost/path."),
    ],
)
def test_builtin_validators_reject_invalid_values(label: str, text: str) -> None:
    rules = resolve_regex_rules(
        labels=[label],
        builtin_regexes=True,
        rules=[],
    )

    result = detect_regex_entities(text, rules=rules)

    assert result.llm_entities == []
    assert result.accepted_entities == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("See https://en.wikipedia.org/wiki/Foo_(bar).", "https://en.wikipedia.org/wiki/Foo_(bar)"),
        ("See https://example.com/Foo_(bar)).", "https://example.com/Foo_(bar)"),
        ("See https://example.com/a_(b_(c)).", "https://example.com/a_(b_(c))"),
        ("请访问https://例子.公司/路径（内部）。", "https://例子.公司/路径（内部）"),
        ("请访问https://例子.公司/路径【内部】】。", "https://例子.公司/路径【内部】"),
        ("See http://[2001:db8::1]/docs.", "http://[2001:db8::1]/docs"),
    ],
)
def test_url_trimming_preserves_balanced_delimiters(text: str, expected: str) -> None:
    rules = resolve_regex_rules(labels=["url"], builtin_regexes=True, rules=[])

    result = detect_regex_entities(text, rules=rules)

    assert [(entity.value, text[entity.start_position : entity.end_position]) for entity in result.llm_entities] == [
        (expected, expected)
    ]


def test_builtin_rules_only_activate_for_requested_labels() -> None:
    rules = resolve_regex_rules(
        labels=["email"],
        builtin_regexes=True,
        rules=[],
    )

    assert [rule.label for rule in rules] == ["email"]


def test_builtin_regexes_can_be_disabled() -> None:
    rules = resolve_regex_rules(
        labels=["email"],
        builtin_regexes=False,
        rules=[],
    )

    assert rules == []


def test_one_builtin_can_be_disabled_without_disabling_the_registry() -> None:
    rules = resolve_regex_rules(
        labels=["email", "ipv4"],
        builtin_regexes=True,
        rules=[BuiltinRegex(label="email", enabled=False)],
    )

    assert [rule.label for rule in rules] == ["ipv4"]


def test_disabled_builtin_can_be_replaced_by_custom_rule_with_same_label() -> None:
    rules = resolve_regex_rules(
        labels=["email"],
        builtin_regexes=True,
        rules=[
            BuiltinRegex(label="email", enabled=False),
            RegexRule(label="email", pattern=r"internal:[a-z]+"),
        ],
    )

    result = detect_regex_entities("internal:alice alice@example.com", rules=rules)

    assert [(entity.value, entity.source.split(":", 1)[0]) for entity in result.llm_entities] == [
        ("internal:alice", "regex_user")
    ]


def test_llm_validation_can_be_disabled_per_builtin() -> None:
    rules = resolve_regex_rules(
        labels=["email"],
        builtin_regexes=True,
        rules=[BuiltinRegex(label="email", validate_with_llm=False)],
    )

    result = detect_regex_entities("Email alice@example.com", rules=rules)

    assert result.llm_entities == []
    assert [entity.value for entity in result.accepted_entities] == ["alice@example.com"]


def test_custom_rule_uses_callable_validator_and_defaults_to_llm_validation() -> None:
    def is_valid(candidate: RegexCandidate) -> RegexValidationResult:
        return RegexValidationResult(valid=candidate.groups["number"] == "42")

    rules = resolve_regex_rules(
        labels=["support_case"],
        builtin_regexes=True,
        rules=[
            RegexRule(
                label="support_case",
                pattern=r"CASE-(?P<number>\d+)",
                validator=is_valid,
            )
        ],
    )

    result = detect_regex_entities("CASE-41 then CASE-42", rules=rules)

    assert [entity.value for entity in result.llm_entities] == ["CASE-42"]
    assert result.accepted_entities == []


def test_custom_rule_can_bypass_llm_validation() -> None:
    rules = resolve_regex_rules(
        labels=["ticket"],
        builtin_regexes=False,
        rules=[RegexRule(label="ticket", pattern=r"TKT-\d+", validate_with_llm=False)],
    )

    result = detect_regex_entities("TKT-123", rules=rules)

    assert result.llm_entities == []
    assert [entity.value for entity in result.accepted_entities] == ["TKT-123"]


def test_custom_rule_ids_and_results_are_stable_when_rules_are_reordered() -> None:
    first = RegexRule(label="ticket", pattern=r"TKT-\d+")
    second = RegexRule(label="case", pattern=r"CASE-\d+")

    forward_rules = resolve_regex_rules(
        labels=["ticket", "case"],
        builtin_regexes=False,
        rules=[first, second],
    )
    reverse_rules = resolve_regex_rules(
        labels=["ticket", "case"],
        builtin_regexes=False,
        rules=[second, first],
    )

    forward_ids = {rule.label: rule.rule_id for rule in forward_rules}
    reverse_ids = {rule.label: rule.rule_id for rule in reverse_rules}
    assert forward_ids == reverse_ids

    text = "CASE-20 TKT-10"
    forward = detect_regex_entities(text, rules=forward_rules)
    reverse = detect_regex_entities(text, rules=reverse_rules)
    assert [entity.as_dict() for entity in forward.llm_entities] == [
        entity.as_dict() for entity in reverse.llm_entities
    ]


def test_user_rule_wins_an_identical_span_conflict_with_builtin() -> None:
    rules = resolve_regex_rules(
        labels=["email", "company_contact"],
        builtin_regexes=True,
        rules=[RegexRule(label="company_contact", pattern=r"alice@example\.com")],
    )

    result = detect_regex_entities("alice@example.com", rules=rules)

    assert [(entity.value, entity.label) for entity in result.llm_entities] == [
        ("alice@example.com", "company_contact")
    ]


def test_match_cap_fails_without_including_matched_values_in_error() -> None:
    rules = resolve_regex_rules(
        labels=["token"],
        builtin_regexes=False,
        rules=[RegexRule(label="token", pattern=r"secret\d")],
    )

    with pytest.raises(RuntimeError, match="exceeded the maximum") as exc_info:
        detect_regex_entities("secret1 secret2", rules=rules, max_matches_per_rule=1)

    assert "secret1" not in str(exc_info.value)


def test_regex_entities_are_not_propagated_to_unvalidated_occurrences() -> None:
    rules = resolve_regex_rules(
        labels=["token"],
        builtin_regexes=False,
        rules=[RegexRule(label="token", pattern=r"(?<=allow:)ABC", validate_with_llm=False)],
    )
    result = detect_regex_entities("allow:ABC deny:ABC", rules=rules)

    expanded = expand_entity_occurrences(
        text="allow:ABC deny:ABC",
        entities=result.accepted_entities,
    )

    assert [(entity.start_position, entity.end_position) for entity in expanded] == [(6, 9)]


def test_export_requires_registered_name_instead_of_direct_callable() -> None:
    def validator(candidate: RegexCandidate) -> bool:
        return bool(candidate.value)

    with pytest.raises(ValueError, match="registered validator names"):
        validate_exportable_regex_rules([RegexRule(label="ticket", pattern=r"TKT-\d+", validator=validator)])

    validate_exportable_regex_rules([RegexRule(label="ticket", pattern=r"TKT-\d+", validator="installed.ticket.v1")])
