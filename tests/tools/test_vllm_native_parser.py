# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The serving child turns a launch spec into vLLM argv and delegates parsing to vLLM."""

from __future__ import annotations

import argparse
from unittest import mock

import pytest

from inference_service_compiler import compiler, launch_ownership, models, vllm_runtime


def spec(**backend: object) -> models.ServiceSpec:
    return models.ServiceSpec.model_validate(
        {"backend": {"kind": "vllm", "model": "org/model", "revision": "pin"} | backend, "port": 8123}
    )


def test_option_values_render_as_argv_tokens() -> None:
    tokens = vllm_runtime.option_tokens(
        {
            "enable-prefix-caching": True,
            "async-scheduling": False,
            "gpu-memory-utilization": 0.9,
            "max-model-len": 8192,
            "reasoning-parser": "openai_gptoss",
            "cuda-graph-sizes": [1, 2, 4],
            "hf-overrides": {"rope_scaling": {"factor": 2}},
        }
    )
    assert tokens == [
        "--enable-prefix-caching",
        "--no-async-scheduling",
        "--gpu-memory-utilization=0.9",
        "--max-model-len=8192",
        "--reasoning-parser=openai_gptoss",
        "--cuda-graph-sizes",
        "1",
        "2",
        "4",
        '--hf-overrides={"rope_scaling": {"factor": 2}}',
    ]


def test_serve_arguments_put_tool_owned_options_before_profile_options() -> None:
    arguments = vllm_runtime.serve_arguments(
        spec(served_model_name="alias", options={"max-model-len": 2048}, probe_extra_body={"reasoning_effort": "low"})
    )
    assert arguments == [
        "--model=org/model",
        "--host=127.0.0.1",
        "--port=8123",
        "--revision=pin",
        "--tokenizer-revision=pin",
        "--served-model-name",
        "alias",
        f"--middleware={launch_ownership.LAUNCH_OWNERSHIP_MIDDLEWARE}",
        "--max-model-len=2048",
    ]


def test_lora_adapter_is_served_under_its_name() -> None:
    arguments = vllm_runtime.serve_arguments(spec(model="base", adapter={"path": "/a", "name": "p"}))
    assert arguments[arguments.index("--enable-lora") + 1 :][:2] == ["--lora-modules", "p=/a"]


def test_tokens_parse_exactly_under_an_argparse_parser_without_abbreviations() -> None:
    """The token forms are the ones argparse accepts for store, boolean-optional, append and nargs actions."""
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for flag in ("--model", "--host", "--revision", "--tokenizer-revision", "--max-model-len"):
        parser.add_argument(flag)
    parser.add_argument("--port", type=int)
    parser.add_argument("--served-model-name", nargs="+")
    parser.add_argument("--middleware", action="append", default=[])
    parser.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction)
    parsed = parser.parse_args(
        vllm_runtime.serve_arguments(
            spec(served_model_name="alias", options={"max-model-len": 2048, "enable-prefix-caching": False})
        )
    )
    assert parsed.served_model_name == ["alias"]
    assert parsed.middleware == [launch_ownership.LAUNCH_OWNERSHIP_MIDDLEWARE]
    assert parsed.enable_prefix_caching is False
    assert parsed.port == 8123
    with pytest.raises(SystemExit):
        parser.parse_args(["--max-model", "1"])


def test_child_reads_the_spec_from_the_compiled_plan() -> None:
    plan = compiler.compile_spec(spec(options={"max-model-len": 2048}))
    parsed, validate_only = vllm_runtime.parse_launch_spec([*plan.argv[2:], "--validate"])
    assert parsed == plan.spec
    assert validate_only is True


def test_child_refuses_a_non_vllm_backend() -> None:
    gliner = models.ServiceSpec.model_validate({"backend": {"kind": "gliner2"}, "port": 8001})
    with pytest.raises(ValueError, match="cannot serve backend 'gliner2'"):
        vllm_runtime.parse_launch_spec(["--launch-spec-json", gliner.model_dump_json()])


def test_validate_mode_parses_and_validates_without_serving() -> None:
    plan = compiler.compile_spec(spec(options={"mamba-backend": "flashinfer"}))
    parser = mock.Mock()
    with (
        mock.patch.object(vllm_runtime.vllm_compat, "make_serve_parser", return_value=parser),
        mock.patch.object(vllm_runtime.vllm_compat, "validate_serve_arguments") as validate,
        mock.patch.object(vllm_runtime.vllm_compat, "run_api_server") as serve,
        mock.patch.object(vllm_runtime, "configure_flashinfer_toolchain") as flashinfer,
        mock.patch.object(vllm_runtime, "expose_interpreter_tools"),
    ):
        vllm_runtime.run_server([*plan.argv[2:], "--validate"])
        parser.parse_args.assert_called_once_with(vllm_runtime.serve_arguments(plan.spec))
        validate.assert_called_once_with(parser.parse_args.return_value)
        serve.assert_not_called()
        flashinfer.assert_not_called()

        vllm_runtime.run_server(list(plan.argv[2:]))
        flashinfer.assert_called_once_with(required=True)
        serve.assert_called_once_with(parser.parse_args.return_value)
