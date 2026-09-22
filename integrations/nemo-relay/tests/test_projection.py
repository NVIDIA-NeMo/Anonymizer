# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy

import pytest

from nemo_anonymizer_relay.projection import (
    ProjectionLimitError,
    collect_text_leaves,
    omit_unsupported_media,
    replace_text_leaves,
)


def test_projection_budget_is_fail_closed_signal() -> None:
    with pytest.raises(ProjectionLimitError):
        collect_text_leaves(["a", "b"], max_leaves=1, max_bytes=100)


def test_protocol_mode_preserves_control_values_but_inspects_identifiers() -> None:
    value = {
        "messages": [{"role": "assistant", "content": "Marisol at marisol@example.com"}],
        "tools": [{"type": "function", "function": {"name": "lookup_Marisol_Vega"}}],
        "model": "marisol@example.com",
    }

    leaves = collect_text_leaves(
        value,
        max_leaves=20,
        max_bytes=2000,
        preserve_protocol_values=True,
    )
    selected = {leaf.text for leaf in leaves}

    assert "assistant" not in selected
    assert "function" not in selected
    assert selected >= {
        "Marisol at marisol@example.com",
        "lookup_Marisol_Vega",
        "marisol@example.com",
    }


def test_generic_mode_does_not_exempt_protocol_shaped_application_data() -> None:
    leaves = collect_text_leaves(
        {"role": "user", "type": "message"},
        max_leaves=10,
        max_bytes=100,
    )

    assert [leaf.text for leaf in leaves] == ["user", "message"]


def test_projection_inspects_and_rewrites_application_mapping_keys() -> None:
    value = {"data": {"marisol@example.com": "case owner", "ordinary_field": "safe"}}
    original = deepcopy(value)
    leaves = collect_text_leaves(value, max_leaves=10, max_bytes=1000)
    email = next(leaf for leaf in leaves if leaf.key == "marisol@example.com")

    rewritten = replace_text_leaves(value, {(email.path, email.key): "[REDACTED]"})

    assert rewritten == {"data": {"[REDACTED]": "case owner", "ordinary_field": "safe"}}
    assert value == original


def test_projection_omits_media_without_mutating_source() -> None:
    source = {
        "messages": [
            {"role": "user", "content": "inspect this"},
            {"type": "input_image", "image_url": "data:image/png;base64,SECRET"},
        ]
    }

    projected = omit_unsupported_media(source)

    assert projected["messages"][1] == {
        "type": "input_image",
        "content": "[UNSUPPORTED MEDIA OMITTED]",
    }
    assert source["messages"][1]["image_url"].startswith("data:")


def test_projection_distinguishes_large_binary_from_short_encoded_text() -> None:
    encoded_ssn = "MDc4LTA1LTExMjA="
    projected = omit_unsupported_media({"attachment": "A_-0" * 1024, "opaque": encoded_ssn})

    assert projected == {
        "attachment": "[UNSUPPORTED MEDIA OMITTED]",
        "opaque": encoded_ssn,
    }
