# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
from collections.abc import Callable
from weakref import ref

import pytest

from anonymizer import BuiltinRegex, RegexCandidate, RegexRule, RegexValidationResult
from anonymizer.engine.detection import regex_detection
from anonymizer.engine.detection.postprocess import expand_entity_occurrences
from anonymizer.engine.detection.regex_detection import (
    ResolvedRegexRule,
    detect_regex_entities,
    resolve_regex_rules,
    validate_exportable_regex_rules,
)


def test_runtime_rejects_zero_width_match_that_bypasses_config_validation() -> None:
    rule = ResolvedRegexRule(
        rule_id="test:zero-width",
        label="case_id",
        pattern=r"CASE-[0-9]+|(?=Z{20})",
        source="regex_user",
    )

    with pytest.raises(RuntimeError, match="produced a zero-width match"):
        detect_regex_entities("Z" * 20, rules=[rule])


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


def test_builtin_rule_and_validator_ids_use_product_namespace() -> None:
    rules = resolve_regex_rules(
        labels=["credit_debit_card", "email", "ipv4", "ipv6", "mac_address", "url"],
        builtin_regexes=True,
        rules=[],
    )

    assert all(rule.rule_id.startswith("nemo-anonymizer.") for rule in rules)
    assert all(rule.validator_id is not None and rule.validator_id.startswith("nemo-anonymizer.") for rule in rules)


@pytest.mark.parametrize(
    "address",
    [
        "x@उदाहरण.भारत",
        "x@e\u0301xample.com",
    ],
)
def test_email_regex_supports_international_domains(address: str) -> None:
    text = f"Email {address} now"
    rules = resolve_regex_rules(labels=["email"], builtin_regexes=True, rules=[])

    result = detect_regex_entities(text, rules=rules)

    assert [(entity.value, text[entity.start_position : entity.end_position]) for entity in result.llm_entities] == [
        (address, address)
    ]


def test_ipv6_regex_consumes_an_embedded_ipv4_tail() -> None:
    address = "::ffff:192.0.2.128"
    text = f"Mapped address {address} is reserved"
    rules = resolve_regex_rules(labels=["ipv6", "ipv4"], builtin_regexes=True, rules=[])

    result = detect_regex_entities(text, rules=rules)

    ipv6 = next(entity for entity in result.llm_entities if entity.label == "ipv6")
    assert (ipv6.value, ipv6.start_position, ipv6.end_position) == (
        address,
        len("Mapped address "),
        len("Mapped address ") + len(address),
    )
    assert text[ipv6.start_position : ipv6.end_position] == address
    assert [(entity.value, entity.label) for entity in result.llm_entities] == [
        (address, "ipv6"),
        ("192.0.2.128", "ipv4"),
    ]


def test_url_validator_accepts_uppercase_www_prefix() -> None:
    address = "WWW.example.com/path"
    text = f"Visit {address} now"
    rules = resolve_regex_rules(labels=["url"], builtin_regexes=True, rules=[])

    result = detect_regex_entities(text, rules=rules)

    assert [entity.value for entity in result.llm_entities] == [address]


@pytest.mark.parametrize(
    "address",
    [
        "http://-bad.com/path",
        "http://bad-.com/path",
        "http://bad_host.com/path",
        "http://999.999.999.999/path",
        "http://example..com/path",
    ],
)
def test_url_validator_rejects_malformed_hosts_without_llm_validation(address: str) -> None:
    rules = resolve_regex_rules(
        labels=["url"],
        builtin_regexes=True,
        rules=[BuiltinRegex(label="url", validate_with_llm=False)],
    )

    result = detect_regex_entities(address, rules=rules)

    assert result.llm_entities == []
    assert result.accepted_entities == []


@pytest.mark.parametrize(
    "address",
    [
        "https://example.com/path",
        "https://例子.公司/路径",
        "http://192.0.2.1/path",
        "http://[2001:db8::1]/docs",
    ],
)
def test_url_validator_accepts_valid_hosts_without_llm_validation(address: str) -> None:
    rules = resolve_regex_rules(
        labels=["url"],
        builtin_regexes=True,
        rules=[BuiltinRegex(label="url", validate_with_llm=False)],
    )

    result = detect_regex_entities(address, rules=rules)

    assert result.llm_entities == []
    assert [entity.value for entity in result.accepted_entities] == [address]


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


def test_custom_rules_only_activate_for_requested_labels() -> None:
    rules = resolve_regex_rules(
        labels=["email"],
        builtin_regexes=False,
        rules=[
            RegexRule(
                label="support_case",
                pattern=r"CASE-[0-9]+",
                validator="not-installed",
            )
        ],
    )

    assert rules == []


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


def test_distinct_validator_closures_from_same_factory_do_not_collide() -> None:
    def validator_for(expected: str) -> Callable[[RegexCandidate], bool]:
        def validate(candidate: RegexCandidate) -> bool:
            return candidate.value == expected

        return validate

    case_validator = validator_for("CASE-42")
    ticket_validator = validator_for("TKT-7")
    rules = resolve_regex_rules(
        labels=["case_id", "ticket_id"],
        builtin_regexes=False,
        rules=[
            RegexRule(label="case_id", pattern=r"CASE-[0-9]+", validator=case_validator),
            RegexRule(label="ticket_id", pattern=r"TKT-[0-9]+", validator=ticket_validator),
        ],
    )

    assert rules[0].validator_id != rules[1].validator_id
    result = detect_regex_entities("CASE-41 CASE-42 TKT-7 TKT-8", rules=rules)
    assert [(entity.value, entity.label) for entity in result.llm_entities] == [
        ("CASE-42", "case_id"),
        ("TKT-7", "ticket_id"),
    ]


def test_reusing_same_validator_callable_reuses_local_identity() -> None:
    def validate(candidate: RegexCandidate) -> bool:
        return bool(candidate.value)

    rules = resolve_regex_rules(
        labels=["case_id", "ticket_id"],
        builtin_regexes=False,
        rules=[
            RegexRule(label="case_id", pattern=r"CASE-[0-9]+", validator=validate),
            RegexRule(label="ticket_id", pattern=r"TKT-[0-9]+", validator=validate),
        ],
    )

    assert rules[0].validator_id == rules[1].validator_id


def test_non_weak_referenceable_callable_is_owned_by_resolved_rule() -> None:
    class Validator:
        __slots__ = ()

        def __call__(self, candidate: RegexCandidate) -> bool:
            return candidate.value == "CASE-42"

    rules = resolve_regex_rules(
        labels=["case_id"],
        builtin_regexes=False,
        rules=[RegexRule(label="case_id", pattern=r"CASE-[0-9]+", validator=Validator())],
    )

    result = detect_regex_entities("CASE-41 CASE-42", rules=rules)
    assert [entity.value for entity in result.llm_entities] == ["CASE-42"]


def test_resolved_rule_owns_validator_until_deferred_execution_finishes() -> None:
    class CapturedState:
        expected = "CASE-42"

    def resolve_temporary_validator() -> tuple[
        ref[CapturedState],
        ref[Callable[..., bool]],
        str,
        list[ResolvedRegexRule],
    ]:
        state = CapturedState()

        def validate(candidate: RegexCandidate) -> bool:
            return candidate.value == state.expected

        rules = resolve_regex_rules(
            labels=["case_id"],
            builtin_regexes=False,
            rules=[RegexRule(label="case_id", pattern=r"CASE-[0-9]+", validator=validate)],
        )
        validator_id = rules[0].validator_id
        assert validator_id is not None
        assert validator_id in regex_detection._LOCAL_VALIDATORS
        return ref(state), ref(validate), validator_id, rules

    state_ref, validator_ref, validator_id, rules = resolve_temporary_validator()
    gc.collect()

    assert state_ref() is not None
    assert validator_ref() is not None
    assert validator_id in regex_detection._LOCAL_VALIDATORS
    assert "local_validator" not in rules[0].model_dump(mode="json")
    result = detect_regex_entities("CASE-42", rules=rules)
    assert [entity.value for entity in result.llm_entities] == ["CASE-42"]

    del rules
    gc.collect()

    assert state_ref() is None
    assert validator_ref() is None
    assert validator_id not in regex_detection._LOCAL_VALIDATORS


def test_repeated_temporary_validator_closures_do_not_grow_local_registry() -> None:
    gc.collect()
    initial_validator_ids = set(regex_detection._LOCAL_VALIDATORS)

    for expected in range(50):

        def validate(candidate: RegexCandidate, expected: int = expected) -> bool:
            return candidate.value == str(expected)

        resolve_regex_rules(
            labels=["number"],
            builtin_regexes=False,
            rules=[RegexRule(label="number", pattern=r"[0-9]+", validator=validate)],
        )

    del validate
    gc.collect()

    assert set(regex_detection._LOCAL_VALIDATORS) == initial_validator_ids


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


def test_same_span_label_conflict_survives_until_contextual_validation() -> None:
    rules = resolve_regex_rules(
        labels=["email", "company_contact"],
        builtin_regexes=True,
        rules=[RegexRule(label="company_contact", pattern=r"alice@example\.com")],
    )

    result = detect_regex_entities("alice@example.com", rules=rules)

    assert [(entity.value, entity.label) for entity in result.llm_entities] == [
        ("alice@example.com", "company_contact"),
        ("alice@example.com", "email"),
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


def test_export_ignores_callable_validator_on_disabled_custom_rule() -> None:
    def validator(candidate: RegexCandidate) -> bool:
        return bool(candidate.value)

    validate_exportable_regex_rules([RegexRule(label="ticket", pattern=r"TKT-\d+", validator=validator, enabled=False)])
