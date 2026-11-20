# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the native GLiNER2 server from its dependency-isolated environment.

``python -m anonymizer.engine.local_inference.serve --host H --port P`` builds or reuses the
isolated environment and replaces this process with the server, keeping its PID and environment.
``--prepare`` only builds or reuses the environment.
"""

from __future__ import annotations

import sys

from anonymizer.engine.local_inference.runtime import prepare_development_server, run_development_server


def main() -> None:
    """Prepare the isolated environment, then serve unless only ``--prepare`` was requested."""
    if sys.argv[1:] == ["--prepare"]:
        prepare_development_server()
        return
    run_development_server()


if __name__ == "__main__":
    main()
