# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entry point for the ``anonymizer`` CLI (``anonymizer.interface.cli.main:main``). The CLI is generated from the
config models: ``params`` (modes, flag models, help panels), ``commands`` (run, preview, validate), ``config_file``
(``--config``), ``errors`` (one-line errors named by flag) and ``app`` (the cyclopts App).
"""

from __future__ import annotations

import os
import sys

# When invoked via the CLI entry point, telemetry deployment type defaults to "cli".
# Anonymizer's SDK path sets "sdk" via os.environ.setdefault in Anonymizer.__init__,
# but the CLI is loaded first so its setdefault wins for CLI-driven runs.
os.environ.setdefault("NEMO_DEPLOYMENT_TYPE", "cli")

from anonymizer.interface.cli.app import app


def main() -> None:
    """Entry point for the anonymizer CLI. Errors are printed by ``errors._format_error`` and
    ``errors._cli_error_handler``."""
    app(sys.argv[1:])
