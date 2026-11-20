# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every vLLM import path the serving child uses, pinned to vLLM 0.33.

These mirror ``vllm.entrypoints.launchers.api_server.entry.main``. Imports are function-local
so the controller and tests can import this package without vLLM installed.
"""

from __future__ import annotations

import argparse


def make_serve_parser() -> argparse.ArgumentParser:
    """Return vLLM's ``serve`` parser with exact option matching (no abbreviations)."""
    from vllm.entrypoints.common.cli_args import make_arg_parser
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    return make_arg_parser(FlexibleArgumentParser(allow_abbrev=False))


def validate_serve_arguments(arguments: argparse.Namespace) -> None:
    """Run vLLM's pre-load checks on parsed serve arguments."""
    from vllm.entrypoints.common.cli_args import validate_parsed_serve_args

    validate_parsed_serve_args(arguments)


def run_api_server(arguments: argparse.Namespace) -> None:
    """Run a single-process vLLM OpenAI-compatible server until it exits."""
    import uvloop
    from vllm.entrypoints.common.startup import cli_env_setup
    from vllm.entrypoints.launchers.api_server.entry import run_server
    from vllm.logger import configure_logging_from_args

    cli_env_setup()
    configure_logging_from_args(arguments)
    uvloop.run(run_server(arguments))
