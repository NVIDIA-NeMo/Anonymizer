# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from unittest.mock import Mock

import pytest

from anonymizer.logging import LoggingConfig, configure_logging


@pytest.mark.parametrize(
    "preset, expected_anonymizer, expected_dd",
    [
        ("default", "INFO", "WARNING"),
        ("verbose", "INFO", "INFO"),
        ("debug", "DEBUG", "INFO"),
    ],
)
def test_preset_levels(preset: str, expected_anonymizer: str, expected_dd: str) -> None:
    config = getattr(LoggingConfig, preset)()
    assert config.anonymizer_level == expected_anonymizer
    assert config.data_designer_level == expected_dd


@pytest.mark.parametrize(
    "preset, expected_anonymizer, expected_dd",
    [
        ("default", logging.INFO, logging.WARNING),
        ("verbose", logging.INFO, logging.INFO),
        ("debug", logging.DEBUG, logging.INFO),
    ],
)
def test_configure_logging_sets_levels(preset: str, expected_anonymizer: int, expected_dd: int) -> None:
    configure_logging(getattr(LoggingConfig, preset)())
    assert logging.getLogger("anonymizer").level == expected_anonymizer
    assert logging.getLogger("data_designer").level == expected_dd


def test_no_args_uses_default() -> None:
    configure_logging()
    assert logging.getLogger("anonymizer").level == logging.INFO
    assert logging.getLogger("data_designer").level == logging.WARNING


def test_verbose_bool_backward_compat() -> None:
    configure_logging(verbose=True)
    assert logging.getLogger("data_designer").level == logging.INFO


def test_user_config_survives_anonymizer_init() -> None:
    """configure_logging() before Anonymizer() should not be overwritten."""
    from anonymizer.interface.anonymizer import Anonymizer

    configure_logging(LoggingConfig.debug())
    Anonymizer(detection_workflow=Mock(), replace_runner=Mock())
    assert logging.getLogger("anonymizer").level == logging.DEBUG
