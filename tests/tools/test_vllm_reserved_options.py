# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Owned vLLM actions cannot enter plans through alternate option spellings."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_service_compiler import cli, compiler, models

SYNTHETIC = "SYNTHETIC_NOT_A_SECRET"


def spec(config: dict[str, object]) -> models.LocalInferenceServiceSpec:
    return models.LocalInferenceServiceSpec.model_validate(
        {
            "task": {"kind": "generation"},
            "model": {"model_id": "example/model", "revision": "abc"},
            "vllm": {"config": config, "api_key_env": "LOCAL_KEY"},
            "local": {},
        }
    )


@pytest.mark.parametrize(
    "key",
    [
        "api-key",
        "api-ke",
        "api_ke",
        "hf-token",
        "hf-toke",
        "hf_toke",
        "mode",
        "hos",
        "por",
        "revisio",
        "tokenizer-revisio",
        "served-model-nam",
    ],
)
def test_owned_options_rejected_before_serialization(key: str) -> None:
    with pytest.raises(compiler.CompilationError) as error:
        compiler.compile_profile(spec({key: SYNTHETIC}))
    assert error.value.diagnostic.code == "reserved-vllm-option"
    assert SYNTHETIC not in error.value.diagnostic.model_dump_json()


@pytest.mark.parametrize("key", ["api-ke", "hf_toke"])
def test_cli_rejects_credentials_without_writing_plan(
    key: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profile = tmp_path / "profile.toml"
    output = tmp_path / "plan.json"
    profile.write_text(
        f'schema_version = "inference-service.local-spec/v2"\n[task]\nkind = "generation"\n[model]\nmodel_id = "example/model"\n[vllm.config]\n{key} = "{SYNTHETIC}"\n[local]\n'
    )
    with pytest.raises(SystemExit) as error:
        cli.compile_plan(profile=profile, output=output)
    assert error.value.code == 125
    captured = capsys.readouterr()
    assert not output.exists()
    assert "reserved-vllm-option" in captured.err
    assert SYNTHETIC not in captured.out + captured.err


def test_unrelated_native_options_and_environment_route_preserved() -> None:
    plan = compiler.compile_profile(spec({"api-server-count": 2, "enable-prefix-caching": False}))
    config = json.loads(plan.command.render_argv()[3])
    assert config["api-server-count"] == 2
    assert config["enable-prefix-caching"] is False
    assert "hf-token" not in config
    assert plan.readiness.bearer_token_environment_variable == "LOCAL_KEY"
