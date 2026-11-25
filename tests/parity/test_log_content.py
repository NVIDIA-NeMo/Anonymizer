# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""No log record from ``run``, ``preview`` or ``evaluate`` names an entity value, a substitute or a record's text."""

from __future__ import annotations

import logging
import re
from collections.abc import Generator

import pytest

from anonymizer.logging import LoggingConfig, configure_logging
from tests.parity.harness import CASES, RECORDS, SYNTHETIC, Case, record_case


def _probes(case: Case) -> set[str]:
    """Every value a log line must not carry: detected values, substitutes, the coverage miss and whole texts."""
    values = {*case.lexicon, *SYNTHETIC.values(), case.missed_entity["value"], *(text for _, text in RECORDS)}
    return {value for value in values if len(value) >= 3}


def _leaks(text: str, probes: set[str]) -> dict[str, int]:
    # Word-bounded, so a value that starts or ends with punctuation still matches where ``\b`` would not.
    found = {value: len(re.findall(rf"(?<!\w){re.escape(value)}(?!\w)", text)) for value in probes}
    return {value: count for value, count in found.items() if count}


@pytest.fixture
def debug_logs(caplog: pytest.LogCaptureFixture) -> Generator[pytest.LogCaptureFixture]:
    # caplog restores the level each logger had at its first set_level, so record those before configure_logging
    # changes them, then lower both to DEBUG.
    caplog.set_level(logging.DEBUG, logger="anonymizer")
    caplog.set_level(logging.DEBUG, logger="data_designer")
    configure_logging(LoggingConfig.debug())
    caplog.set_level(logging.DEBUG, logger="data_designer")
    data_designer = logging.getLogger("data_designer")
    data_designer.addHandler(caplog.handler)  # the anonymizer logger's handler comes from tests/conftest.py
    yield caplog
    data_designer.removeHandler(caplog.handler)
    configure_logging(LoggingConfig.default())


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_logs_name_no_entity_value(case: Case, debug_logs: pytest.LogCaptureFixture) -> None:
    record_case(case)
    records = debug_logs.records
    assert len(records) >= 10, "too few log records captured for the check to mean anything"
    text = "\n".join(r.getMessage() + ("\n" + r.exc_text if r.exc_text else "") for r in records)
    assert not _leaks(text, _probes(case))


def test_leak_check_finds_a_logged_value() -> None:
    assert _leaks("row: Alice works at Acme.", {"Alice", "Acme"}) == {"Alice": 1, "Acme": 1}
    assert _leaks("Alicette", {"Alice"}) == {}
