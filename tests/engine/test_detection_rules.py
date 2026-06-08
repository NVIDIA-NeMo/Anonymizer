# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import Counter
from unittest.mock import Mock

import pandas as pd
import pytest

from anonymizer.engine.constants import COL_DETECTED_ENTITIES, COL_TAGGED_TEXT, COL_TEXT
from anonymizer.engine.detection.detection_workflow import EntityDetectionWorkflow
from anonymizer.engine.detection.rules import (
    STRUCTURED_RULE_FAST_LANE_LABELS,
    SUPPORTED_RULE_LABELS,
    detect_high_confidence_entities,
)
from anonymizer.engine.schemas import EntitiesSchema

SHELL_TEXT = """$ curl -H 'Authorization: Bearer sk-test-AAAAAAAAAAAAAAAAAAAAAAAA' https://internal.example.test/api
$ export AWS_ACCESS_KEY_ID=AKIATEST1234567890FAKE
$ export AWS_SECRET_ACCESS_KEY=fakeSecretValue1234567890!
$ docker run -e DATABASE_URL='postgres://app_user:fakeDbPass123!@db.example.test:5432/app' -e API_KEY=ghp_FAKEtoken1234567890abcdef myapp:latest
$ ssh jane.doe@example.test@host-01.example.test
Password: fakeSshPass123!
"""


def test_detect_high_confidence_entities_extracts_shell_secret_values() -> None:
    entities = detect_high_confidence_entities(
        SHELL_TEXT,
        labels=["api_key", "password", "email", "url"],
    )

    assert Counter(entity.label for entity in entities) == {
        "api_key": 3,
        "password": 2,
        "email": 1,
        "url": 2,
    }
    values_by_label = {(entity.label, entity.value) for entity in entities}
    assert ("api_key", "sk-test-AAAAAAAAAAAAAAAAAAAAAAAA") in values_by_label
    assert ("api_key", "AKIATEST1234567890FAKE") in values_by_label
    assert ("api_key", "ghp_FAKEtoken1234567890abcdef") in values_by_label
    assert ("password", "fakeSecretValue1234567890!") in values_by_label
    assert ("password", "fakeSshPass123!") in values_by_label
    assert ("email", "jane.doe@example.test") in values_by_label
    assert ("url", "https://internal.example.test/api") in values_by_label
    assert ("url", "postgres://app_user:fakeDbPass123!@db.example.test:5432/app") in values_by_label

    values = [entity.value for entity in entities]
    assert all(not value.startswith(("Authorization", "Bearer", "API_KEY=", "Password:")) for value in values)


def test_detect_high_confidence_entities_extracts_email_before_sentence_punctuation() -> None:
    entities = detect_high_confidence_entities(
        "Email alice@example.com. Then contact bob@example.co.uk, if needed.",
        labels=["email"],
    )

    assert [entity.value for entity in entities] == ["alice@example.com", "bob@example.co.uk"]


def test_detect_high_confidence_entities_excludes_config_url_separators() -> None:
    text = (
        "DATABASE_URL=postgres://svc_user:DbSecretPass2026!@db.example.test:5432/app; "
        "endpoint: https://internal.example.test/admin;"
    )

    entities = detect_high_confidence_entities(text, labels=["url"])

    assert [entity.value for entity in entities] == [
        "postgres://svc_user:DbSecretPass2026!@db.example.test:5432/app",
        "https://internal.example.test/admin",
    ]


def test_supported_rule_labels_match_detected_label_families() -> None:
    assert SUPPORTED_RULE_LABELS == {
        "api_key",
        "date_of_birth",
        "email",
        "http_cookie",
        "organization_name",
        "password",
        "pin",
        "religious_belief",
        "street_address",
        "unique_id",
        "url",
        "user_name",
    }


def test_structured_rule_fast_lane_excludes_narrow_prose_labels() -> None:
    assert STRUCTURED_RULE_FAST_LANE_LABELS == {
        "api_key",
        "email",
        "http_cookie",
        "password",
        "pin",
        "unique_id",
        "url",
        "user_name",
    }
    assert {"date_of_birth", "organization_name", "religious_belief", "street_address"}.isdisjoint(
        STRUCTURED_RULE_FAST_LANE_LABELS
    )


def test_detect_high_confidence_entities_respects_label_filter() -> None:
    entities = detect_high_confidence_entities(SHELL_TEXT, labels=["password"])

    assert Counter(entity.label for entity in entities) == {"password": 3}
    assert {entity.value for entity in entities} == {
        "fakeSecretValue1234567890!",
        "fakeDbPass123!",
        "fakeSshPass123!",
    }


def test_detect_high_confidence_entities_extracts_sudo_stdin_password() -> None:
    text = '$ echo "P@ssw0rd-local-2026!" | sudo -S systemctl restart nginx'

    entities = detect_high_confidence_entities(text, labels=["password"])

    assert [(entity.label, entity.value) for entity in entities] == [("password", "P@ssw0rd-local-2026!")]


def test_detect_high_confidence_entities_does_not_treat_generic_echo_as_password() -> None:
    text = '$ echo "P@ssw0rd-local-2026!" | grep local'

    assert detect_high_confidence_entities(text, labels=["password"]) == []


def test_detect_high_confidence_entities_does_not_emit_secret_false_positives_for_prose() -> None:
    prose = (
        "Alice Johnson filed Case No. 2025-CV-12345 in Superior Court. "
        "The opinion cites Section 10(b), Exhibit A-17, and docket trace order_390974. "
        "A biography says Jordan Patel joined NVIDIA in 2021 and later moved to Seattle."
    )

    entities = detect_high_confidence_entities(prose, labels=["api_key", "password", "email", "url"])

    assert entities == []


def test_detect_high_confidence_entities_extracts_contextual_date_of_birth() -> None:
    text = "The applicant was born in 1978 and later moved to Berlin. Another report cites 2024."

    entities = detect_high_confidence_entities(text, labels=["date_of_birth"])

    assert [(entity.label, entity.value) for entity in entities] == [("date_of_birth", "1978")]


def test_detect_high_confidence_entities_ignores_standalone_year_for_date_of_birth() -> None:
    text = "The report cites filings from 1978, 2021, and 2024."

    assert detect_high_confidence_entities(text, labels=["date_of_birth"]) == []


def test_detect_high_confidence_entities_extracts_narrow_prose_patterns() -> None:
    text = (
        "After graduation he spent three years at NASA's Goddard Space Flight Center before joining a lab. "
        "Idilio describes himself as secular and leans progressive on most political issues. "
        "Outside the lab, Idilio shares a modest house on West Roberts Drive with his wife."
    )

    entities = detect_high_confidence_entities(
        text,
        labels=["organization_name", "religious_belief", "street_address"],
    )

    assert [(entity.label, entity.value) for entity in entities] == [
        ("organization_name", "NASA's Goddard Space Flight Center"),
        ("religious_belief", "secular"),
        ("street_address", "West Roberts Drive"),
    ]


def test_detect_high_confidence_entities_avoids_generic_prose_belief_false_positive() -> None:
    text = "Jordan describes himself as careful and later worked at a local lab near Roberts Drive."

    assert (
        detect_high_confidence_entities(
            text,
            labels=["organization_name", "religious_belief", "street_address"],
        )
        == []
    )


def test_detect_high_confidence_entities_returns_sorted_non_overlapping_spans() -> None:
    entities = detect_high_confidence_entities(
        "token=sk-test-BBBBBBBBBBBBBBBBBBBBBBBB and Auth: ignored\nPassword: fakePass123!",
        labels=["api_key", "password"],
    )

    assert [(entity.label, entity.value) for entity in entities] == [
        ("api_key", "sk-test-BBBBBBBBBBBBBBBBBBBBBBBB"),
        ("password", "fakePass123!"),
    ]
    assert entities[0].end_position < entities[1].start_position


def test_detect_high_confidence_entities_extracts_session_id_assignments() -> None:
    text = "Cookie: session_id=abc123xyz; auth_token=xoxb-STRUCTURED-Slack-Token-000000"

    entities = detect_high_confidence_entities(text, labels=["api_key"])

    assert [(entity.label, entity.value) for entity in entities] == [
        ("api_key", "abc123xyz"),
        ("api_key", "xoxb-STRUCTURED-Slack-Token-000000"),
    ]


def test_detect_high_confidence_entities_extracts_structured_identifier_labels() -> None:
    text = (
        "POST /audit HTTP/1.1\n"
        "Cookie: session_id=abc123xyz; user_id=26762; auth_token=token-abcdef\n"
        "trace-id: req_KA5k78XNwT0yUNZkPpwq\n"
        "pin=97294\n"
        "user_name=sloanenguy217\n"
    )

    entities = detect_high_confidence_entities(
        text,
        labels=["http_cookie", "pin", "unique_id", "user_name"],
    )

    assert [(entity.label, entity.value) for entity in entities] == [
        ("http_cookie", "session_id=abc123xyz; user_id=26762; auth_token=token-abcdef"),
        ("unique_id", "req_KA5k78XNwT0yUNZkPpwq"),
        ("pin", "97294"),
        ("user_name", "sloanenguy217"),
    ]


def test_detect_high_confidence_entities_extracts_quoted_structured_identifier_keys() -> None:
    text = '{"user": "avery_khan", "pin": "4921", "callback": "https://internal.example.test/admin"}'

    entities = detect_high_confidence_entities(text, labels=["pin", "url", "user_name"])

    assert [(entity.label, entity.value) for entity in entities] == [
        ("user_name", "avery_khan"),
        ("pin", "4921"),
        ("url", "https://internal.example.test/admin"),
    ]


def test_detect_high_confidence_entities_excludes_cookie_sentence_punctuation() -> None:
    text = "Cookie: session_id=abc123xyz; auth_token=token-abcdef. Recovery flow starts."

    entities = detect_high_confidence_entities(text, labels=["http_cookie"])

    assert [(entity.label, entity.value) for entity in entities] == [
        ("http_cookie", "session_id=abc123xyz; auth_token=token-abcdef"),
    ]


def test_detect_high_confidence_entities_extracts_service_principal_user_and_tenant_id() -> None:
    text = "$ az login --service-principal -u skylerlee985 -p fakePass123! --tenant trace-1b7278d77a73ef4e"

    entities = detect_high_confidence_entities(text, labels=["user_name", "unique_id"])

    assert [(entity.label, entity.value) for entity in entities] == [
        ("user_name", "skylerlee985"),
        ("unique_id", "trace-1b7278d77a73ef4e"),
    ]


def test_detect_high_confidence_entities_extracts_audit_user_and_trace_id() -> None:
    text = "Audit record: user skylerlee985 opened session with trace-id req_KA5k78XNwT0yUNZkPpwq."

    entities = detect_high_confidence_entities(text, labels=["user_name", "unique_id"])

    assert [(entity.label, entity.value) for entity in entities] == [
        ("user_name", "skylerlee985"),
        ("unique_id", "req_KA5k78XNwT0yUNZkPpwq"),
    ]


def test_detect_high_confidence_entities_does_not_extract_structured_identifiers_from_generic_prose() -> None:
    text = "The order_390974 filing mentions user research, cookie recipes, and a five digit docket page."

    assert detect_high_confidence_entities(text, labels=["http_cookie", "pin", "unique_id", "user_name"]) == []


def test_workflow_can_detect_with_high_confidence_rules_without_adapter_calls() -> None:
    adapter = Mock()
    workflow = EntityDetectionWorkflow(adapter=adapter)

    result = workflow.detect_with_high_confidence_rules(
        pd.DataFrame({COL_TEXT: ["token=sk-test-AAAAAAAAAAAAAAAAAAAAAAAA\nPassword: fakePass123!"]}),
        entity_labels=["api_key", "password"],
    )

    adapter.run_workflow.assert_not_called()
    entities = EntitiesSchema.from_raw(result.dataframe[COL_DETECTED_ENTITIES].iloc[0]).entities
    assert [(entity.label, entity.value) for entity in entities] == [
        ("api_key", "sk-test-AAAAAAAAAAAAAAAAAAAAAAAA"),
        ("password", "fakePass123!"),
    ]
    tagged_text = result.dataframe[COL_TAGGED_TEXT].iloc[0]
    assert "<api_key>sk-test-AAAAAAAAAAAAAAAAAAAAAAAA</api_key>" in tagged_text
    assert "<password>fakePass123!</password>" in tagged_text
    assert result.failed_records == []


def test_workflow_rule_detection_rejects_unsupported_labels() -> None:
    workflow = EntityDetectionWorkflow(adapter=Mock())

    with pytest.raises(ValueError, match="unsupported high-confidence rule labels.*person"):
        workflow.detect_with_high_confidence_rules(
            pd.DataFrame({COL_TEXT: ["Alice has token=sk-test-AAAAAAAAAAAAAAAAAAAAAAAA"]}),
            entity_labels=["api_key", "person"],
        )
