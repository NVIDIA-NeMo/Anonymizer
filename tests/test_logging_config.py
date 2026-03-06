# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

from anonymizer.logging import LoggingConfig, configure_logging


class TestLoggingConfig:
    def test_default_preset(self) -> None:
        config = LoggingConfig.default()
        assert config.anonymizer_level == "INFO"
        assert config.data_designer_level == "WARNING"

    def test_verbose_preset(self) -> None:
        config = LoggingConfig.verbose()
        assert config.anonymizer_level == "INFO"
        assert config.data_designer_level == "INFO"

    def test_debug_preset(self) -> None:
        config = LoggingConfig.debug()
        assert config.anonymizer_level == "DEBUG"
        assert config.data_designer_level == "DEBUG"


class TestConfigureLogging:
    def test_default_config_sets_levels(self) -> None:
        configure_logging(LoggingConfig.default())
        assert logging.getLogger("anonymizer").level == logging.INFO
        assert logging.getLogger("data_designer").level == logging.WARNING

    def test_verbose_config_sets_levels(self) -> None:
        configure_logging(LoggingConfig.verbose())
        assert logging.getLogger("anonymizer").level == logging.INFO
        assert logging.getLogger("data_designer").level == logging.INFO

    def test_debug_config_sets_levels(self) -> None:
        configure_logging(LoggingConfig.debug())
        assert logging.getLogger("anonymizer").level == logging.DEBUG
        assert logging.getLogger("data_designer").level == logging.DEBUG

    def test_no_args_uses_default(self) -> None:
        configure_logging()
        assert logging.getLogger("anonymizer").level == logging.INFO
        assert logging.getLogger("data_designer").level == logging.WARNING

    def test_verbose_bool_backward_compat(self) -> None:
        configure_logging(verbose=True)
        assert logging.getLogger("data_designer").level == logging.INFO
