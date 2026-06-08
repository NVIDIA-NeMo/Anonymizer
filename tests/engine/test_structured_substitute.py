# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pandas as pd
import pytest

from anonymizer.engine.constants import COL_ENTITIES_BY_VALUE, COL_REPLACEMENT_MAP
from anonymizer.engine.replace import structured_substitute as structured_substitute_module
from anonymizer.engine.replace.structured_substitute import (
    apply_structured_substitution_maps,
    build_structured_substitution_map,
    structured_substitute_value,
)


@pytest.mark.parametrize(
    ("label", "value"),
    [
        ("api_key", "sk-test-AAAAAAAAAAAAAAAAAAAAAAAA"),
        ("date_of_birth", "1978-02-03"),
        ("email", "alice@example.com"),
        ("http_cookie", "session_id=abc123xyz; user_id=12345; auth_token=secret-token"),
        ("organization_name", "Acme Research Center"),
        ("password", "CorrectHorse!123"),
        ("pin", "97294"),
        ("religious_belief", "secular"),
        ("street_address", "123 Maple Street"),
        ("unique_id", "req_KA5k78XNwT0yUNZkPpwq"),
        ("url", "https://staging.example.com/admin"),
        ("user_name", "sloanenguy217"),
    ],
)
def test_structured_substitute_value_does_not_preserve_original(label: str, value: str) -> None:
    synthetic = structured_substitute_value(value, label)

    assert synthetic != value
    assert value not in synthetic


def test_build_structured_substitution_map_for_entities_by_value() -> None:
    raw_entities = {
        "entities_by_value": [
            {"value": "alice@example.com", "labels": ["email"]},
            {"value": "sk-test-AAAAAAAAAAAAAAAAAAAAAAAA", "labels": ["api_key"]},
        ]
    }

    replacement_map = build_structured_substitution_map(raw_entities)

    replacements = replacement_map["replacements"]
    assert [(item["original"], item["label"]) for item in replacements] == [
        ("alice@example.com", "email"),
        ("sk-test-AAAAAAAAAAAAAAAAAAAAAAAA", "api_key"),
    ]
    serialized = str(replacement_map)
    assert "alice@example.com" in serialized
    assert "sk-test-AAAAAAAAAAAAAAAAAAAAAAAA" in serialized
    assert all(item["original"] != item["synthetic"] for item in replacements)


def test_build_structured_substitution_map_avoids_other_original_values(monkeypatch: pytest.MonkeyPatch) -> None:
    first_original = "alice@example.com"
    second_original = "bob@example.com"
    first_default_digest = structured_substitute_module._digest(value=first_original, label="email")

    def synthetic_email(value: str, digest: str) -> str:
        if value == first_original and digest == first_default_digest:
            return second_original
        return f"user-{digest[:12]}@example.invalid"

    monkeypatch.setitem(structured_substitute_module._GENERATORS, "email", synthetic_email)
    raw_entities = {
        "entities_by_value": [
            {"value": first_original, "labels": ["email"]},
            {"value": second_original, "labels": ["email"]},
        ]
    }

    replacement_map = build_structured_substitution_map(raw_entities)

    replacements = {(item["original"], item["label"]): item["synthetic"] for item in replacement_map["replacements"]}
    assert replacements[(first_original, "email")] != second_original
    assert set(replacements.values()).isdisjoint({first_original, second_original})


def test_build_structured_substitution_map_avoids_duplicate_synthetic_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_original = "Acme Research Center"
    second_original = "Globex Research Center"
    default_digests = {
        structured_substitute_module._digest(value=first_original, label="organization_name"),
        structured_substitute_module._digest(value=second_original, label="organization_name"),
    }

    def synthetic_organization(_value: str, digest: str) -> str:
        if digest in default_digests:
            return "Northbridge Center"
        return f"Helios {digest[:8]} Center"

    monkeypatch.setitem(structured_substitute_module._GENERATORS, "organization_name", synthetic_organization)
    raw_entities = {
        "entities_by_value": [
            {"value": first_original, "labels": ["organization_name"]},
            {"value": second_original, "labels": ["organization_name"]},
        ]
    }

    replacement_map = build_structured_substitution_map(raw_entities)

    synthetics = [item["synthetic"] for item in replacement_map["replacements"]]
    assert len(synthetics) == 2
    assert len(set(synthetics)) == 2
    assert "Northbridge Center" in synthetics


def test_structured_cookie_substitute_preserves_cookie_shape() -> None:
    synthetic = structured_substitute_value(
        "session_id=abc123xyz; user_id=12345; auth_token=secret-token",
        "http_cookie",
    )

    assert "session_id=" in synthetic
    assert "user_id=" in synthetic
    assert "auth_token=" in synthetic
    session_value = synthetic.split("session_id=", 1)[1].split(";", 1)[0]
    assert len(session_value) >= 16
    assert not session_value.isdigit()
    assert "abc123xyz" not in synthetic
    assert "secret-token" not in synthetic


def test_structured_unique_id_substitute_preserves_uuid_shape() -> None:
    synthetic = structured_substitute_value("1ce21179-998b-447b-2dee-3e8adb6afa35", "unique_id")

    assert synthetic.count("-") == 4
    assert synthetic != "1ce21179-998b-447b-2dee-3e8adb6afa35"


def test_build_structured_substitution_map_rejects_unsupported_labels() -> None:
    raw_entities = {"entities_by_value": [{"value": "Alice", "labels": ["person"]}]}

    with pytest.raises(ValueError, match="unsupported labels: person"):
        build_structured_substitution_map(raw_entities)


def test_apply_structured_substitution_maps_adds_replacement_map_column() -> None:
    dataframe = pd.DataFrame(
        {COL_ENTITIES_BY_VALUE: [{"entities_by_value": [{"value": "alice@example.com", "labels": ["email"]}]}]}
    )

    output = apply_structured_substitution_maps(dataframe)

    assert COL_REPLACEMENT_MAP in output.columns
    replacement = output[COL_REPLACEMENT_MAP].iloc[0]["replacements"][0]
    assert replacement["original"] == "alice@example.com"
    assert replacement["label"] == "email"
    assert replacement["synthetic"].endswith("@example.invalid")
