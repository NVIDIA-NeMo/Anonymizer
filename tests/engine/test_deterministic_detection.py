# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from anonymizer.engine.detection.deterministic import detect_deterministic_entities


def _by_label(text: str) -> dict[str, list[str]]:
    labels: dict[str, list[str]] = {}
    for entity in detect_deterministic_entities(text):
        labels.setdefault(entity.label, []).append(entity.value)
    return labels


def test_deterministic_detector_finds_structured_labels() -> None:
    text = (
        "Email alice@example.com, visit https://example.com/a?x=1, card 4111 1111 1111 1111, "
        "SSN 123-45-6789, IPv4 192.168.1.1, IPv6 2001:db8::1, MAC 49:FD:EE:1A:3B:7C."
    )

    labels = _by_label(text)

    assert labels["email"] == ["alice@example.com"]
    assert labels["url"] == ["https://example.com/a?x=1"]
    assert labels["credit_debit_card"] == ["4111 1111 1111 1111"]
    assert "ssn" not in labels
    assert labels["ipv4"] == ["192.168.1.1"]
    assert labels["ipv6"] == ["2001:db8::1"]
    assert labels["mac_address"] == ["49:FD:EE:1A:3B:7C"]


def test_card_detection_requires_luhn() -> None:
    labels = _by_label("Valid 4111-1111-1111-1111 invalid 4111-1111-1111-1112")

    assert labels["credit_debit_card"] == ["4111-1111-1111-1111"]


def test_deterministic_detector_does_not_claim_us_ssn() -> None:
    text = "Good SSN 123-45-6789, reserved 000-45-6789 and 666-45-6789, ZIP 123456789."

    labels = _by_label(text)

    assert "ssn" not in labels


def test_deterministic_detector_respects_requested_labels() -> None:
    entities = detect_deterministic_entities(
        "alice@example.com paid with 4111 1111 1111 1111",
        labels=["email"],
    )

    assert [(entity.label, entity.value) for entity in entities] == [("email", "alice@example.com")]
