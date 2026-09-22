# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy

import pytest

from nemo_anonymizer_relay.projection import (
    ProjectionLimitError,
    collect_export_text_leaves,
    omitted_event,
    prepare_event_for_export,
    replace_export_text_leaves,
)


def test_projection_budget_is_fail_closed_signal() -> None:
    with pytest.raises(ProjectionLimitError):
        collect_export_text_leaves(["a", "b"], max_leaves=1, max_bytes=100)


def test_llm_export_preserves_envelope_and_omits_exact_provider_body() -> None:
    event = {
        "atof_version": "1.0",
        "uuid": "event-1",
        "parent_uuid": "turn-1",
        "propagation_root_uuid": "session-1",
        "timestamp": "2026-09-21T10:00:00Z",
        "kind": "SPAN",
        "name": "openai.responses",
        "category": "llm",
        "attributes": ["streaming"],
        "data": {"input": "Marisol at marisol@example.com"},
        "category_profile": {
            "annotated_request": {"messages": [{"role": "user", "content": "Marisol at marisol@example.com"}]}
        },
        "metadata": {"turn": 7},
    }

    projected = prepare_event_for_export(event)

    envelope = (
        "atof_version",
        "uuid",
        "parent_uuid",
        "propagation_root_uuid",
        "timestamp",
        "kind",
        "name",
        "category",
        "attributes",
    )
    assert all(projected[key] == event[key] for key in envelope)
    assert projected["data"] is None
    assert projected["category_profile"] == event["category_profile"]
    assert projected["metadata"] == {
        "turn": 7,
        "nemo_anonymizer.coverage": "relay_annotation",
        "nemo_anonymizer.provider_body_omitted": True,
    }
    # Projection must not add coverage markers to Relay's source event.
    assert event["metadata"] == {"turn": 7}


def test_only_annotated_protocol_vocabulary_is_excluded_from_inspection() -> None:
    cases = [
        (
            {
                "data": {
                    "role": "Marisol Vega",
                    "model": "marisol@example.com",
                    "type": "Call +1 415 555 0137",
                },
                "category_profile": {
                    "annotated_request": {"messages": [{"role": "user", "content": "marisol@example.com"}]}
                },
            },
            ["Marisol Vega", "marisol@example.com", "Call +1 415 555 0137", "marisol@example.com"],
        ),
        (
            [
                {
                    "data": {"role": "Marisol Vega"},
                    "category_profile": {
                        "annotated_request": {
                            "messages": [{"role": "user", "type": "message", "content": "Marisol Vega"}]
                        }
                    },
                    "metadata": {"model": "marisol@example.com"},
                }
            ],
            ["Marisol Vega", "Marisol Vega", "marisol@example.com"],
        ),
    ]

    for value, expected in cases:
        leaves = collect_export_text_leaves(value, max_leaves=10, max_bytes=1000)
        assert [leaf.text for leaf in leaves] == expected


def test_export_projection_inspects_and_rewrites_application_mapping_keys() -> None:
    value = {
        "data": {
            "marisol@example.com": "case owner",
            "078-05-1120": "tax record",
            "203.0.113.44": "client address",
            "999.0.0.1": "not an IPv4 address",
            "ordinary_field": "safe",
        }
    }
    original = deepcopy(value)
    leaves = collect_export_text_leaves(value, max_leaves=10, max_bytes=1000)
    unusual = next(leaf for leaf in leaves if leaf.key == "marisol@example.com")
    assert {leaf.key for leaf in leaves if leaf.key is not None} == set(value["data"])

    rewritten = replace_export_text_leaves(
        value,
        {(unusual.path, unusual.key): "[REDACTED]"},
    )

    expected = dict(original["data"])
    expected["[REDACTED]"] = expected.pop("marisol@example.com")
    assert rewritten["data"] == expected
    assert value == original


def test_provider_and_tool_identifiers_are_inspected() -> None:
    value = {
        "category_profile": {
            "annotated_request": {
                "model": "marisol@example.com",
                "tools": [{"type": "function", "function": {"name": "Marisol Vega"}}],
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_marisol@example.com",
                                "type": "function",
                                "function": {"name": "lookup_Marisol_Vega", "arguments": "{}"},
                            }
                        ],
                    }
                ],
            }
        }
    }

    leaves = collect_export_text_leaves(value, max_leaves=20, max_bytes=2000)

    assert {leaf.text for leaf in leaves} >= {
        "marisol@example.com",
        "Marisol Vega",
        "call_marisol@example.com",
        "lookup_Marisol_Vega",
    }
    assert "assistant" not in {leaf.text for leaf in leaves}
    assert "function" not in {leaf.text for leaf in leaves}


def test_projection_replaces_spoofed_exporter_metadata() -> None:
    source = {
        "uuid": "event-1",
        "kind": "mark",
        "name": "custom",
        "category": "custom",
        "metadata": {
            "nemo_anonymizer.coverage": "complete",
            "nemo_anonymizer.failure": "none",
            "owner": "Marisol Vega",
        },
    }

    projected = prepare_event_for_export(source)

    assert projected["metadata"] == {
        "owner": "Marisol Vega",
        "nemo_anonymizer.coverage": "generic_json_text",
        "nemo_anonymizer.provider_body_omitted": False,
    }
    assert source["metadata"]["nemo_anonymizer.failure"] == "none"


def test_omitted_event_retains_only_safe_envelope() -> None:
    source = {
        "uuid": "event-1",
        "parent_uuid": "turn-1",
        "propagation_root_uuid": "session-1",
        "kind": "SPAN",
        "attributes": ["streaming", "marisol@example.com"],
        "name": "tool",
        "category": "tool",
        "data": {"email": "marisol@example.com"},
        "category_profile": {"result": "marisol@example.com"},
        "metadata": {"owner": "Marisol Vega"},
    }

    result = omitted_event(source, "DetectorFailure")

    assert result == {
        "uuid": "event-1",
        "parent_uuid": "turn-1",
        "propagation_root_uuid": "session-1",
        "kind": "SPAN",
        "attributes": ["streaming"],
        "name": "nemo_anonymizer.omitted",
        "category": "unknown",
        "data_schema": None,
        "data": None,
        "category_profile": None,
        "metadata": {
            "nemo_anonymizer.coverage": "omitted",
            "nemo_anonymizer.failure": "DetectorFailure",
        },
    }


def test_projection_omits_unknown_scope_attributes() -> None:
    source = {
        "uuid": "event-1",
        "kind": "scope",
        "name": "tool",
        "category": "tool",
        "attributes": ["parallel", "owner-marisol@example.com", 7],
        "metadata": {},
    }

    projected = prepare_event_for_export(source)

    assert projected["attributes"] == ["parallel"]
    assert projected["metadata"]["nemo_anonymizer.coverage"] == "generic_json_text_partial"
    assert projected["metadata"]["nemo_anonymizer.unknown_attributes_omitted"] == 2


def test_export_projection_omits_media_before_detection() -> None:
    source = {
        "uuid": "event-1",
        "kind": "scope",
        "name": "openai.responses",
        "category": "llm",
        "data": {"duplicate": "raw body"},
        "category_profile": {
            "annotated_request": {
                "messages": [
                    {"role": "user", "content": "inspect this"},
                    {"type": "input_image", "image_url": "data:image/png;base64,SECRET"},
                ]
            }
        },
        "metadata": {},
    }

    projected = prepare_event_for_export(source)

    media = projected["category_profile"]["annotated_request"]["messages"][1]
    assert media == {"type": "input_image", "content": "[UNSUPPORTED MEDIA OMITTED]"}
    assert projected["metadata"]["nemo_anonymizer.coverage"] == "relay_annotation_partial"
    assert projected["metadata"]["nemo_anonymizer.unsupported_media_omitted"] == 1
    assert source["category_profile"]["annotated_request"]["messages"][1]["image_url"].startswith("data:")


def test_export_projection_distinguishes_media_sized_and_short_encoded_text() -> None:
    long_blob = {
        "uuid": "event-1",
        "kind": "mark",
        "name": "tool.result",
        "category": "tool",
        "data": {"attachment": "A_-0" * 1024},
        "metadata": {},
    }
    encoded_ssn = "MDc4LTA1LTExMjA="
    short_text = {**long_blob, "data": {"opaque": encoded_ssn}}

    omitted = prepare_event_for_export(long_blob)
    visible = prepare_event_for_export(short_text)
    leaves = collect_export_text_leaves(visible, max_leaves=20, max_bytes=2000)

    assert omitted["data"] == {"attachment": "[UNSUPPORTED MEDIA OMITTED]"}
    assert omitted["metadata"]["nemo_anonymizer.unsupported_media_omitted"] == 1
    assert encoded_ssn in {leaf.text for leaf in leaves}
    assert visible["metadata"]["nemo_anonymizer.coverage"] == "generic_json_text"
    assert "nemo_anonymizer.unsupported_media_omitted" not in visible["metadata"]


def test_export_projection_does_not_treat_profile_type_as_media() -> None:
    source = {
        "uuid": "event-1",
        "kind": "mark",
        "name": "profile.update",
        "category": "custom",
        "data": {"type": "profile", "content": "Marisol Vega"},
        "metadata": {},
    }

    projected = prepare_event_for_export(source)

    assert projected["data"] == {"type": "profile", "content": "Marisol Vega"}
    assert "nemo_anonymizer.unsupported_media_omitted" not in projected["metadata"]
