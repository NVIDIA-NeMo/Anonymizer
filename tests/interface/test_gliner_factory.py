# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the public GLiNER runtime factory."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from data_designer.config.models import ModelProvider

import anonymizer
from anonymizer.engine.ndd.model_loader import parse_model_configs


def test_external_endpoint_preserves_bundled_generation_providers() -> None:
    with patch("anonymizer.interface.factory._new_anonymizer", return_value=Mock()) as constructor:
        anonymizer.create_anonymizer(
            gliner=anonymizer.GlinerEndpoint("http://127.0.0.1:43210/v1", "org/gliner-served", None)
        )

    arguments = constructor.call_args.kwargs
    parsed = parse_model_configs(arguments["model_configs"])
    assert parsed.selected_models.detection.entity_detector == "anonymizer-gliner-detector"
    assert any(provider.name == "openrouter" for provider in arguments["model_providers"])
    injected = next(provider for provider in arguments["model_providers"] if provider.name == "anonymizer-gliner-endpoint")
    assert injected.endpoint == "http://127.0.0.1:43210/v1"


def test_external_endpoint_injects_detector_and_preserves_other_configuration() -> None:
    assert hasattr(anonymizer, "GlinerEndpoint")
    assert hasattr(anonymizer, "create_anonymizer")
    GlinerEndpoint = anonymizer.GlinerEndpoint
    create_anonymizer = anonymizer.create_anonymizer
    source = json.dumps(
        {
            "model_configs": [
                {"alias": "old-detector", "model": "test/old", "provider": "custom"},
                {"alias": "custom-llm", "model": "test/llm", "provider": "custom"},
            ],
            "selected_models": {
                "detection": {
                    "entity_detector": "old-detector",
                    "entity_validator": "custom-llm",
                    "entity_augmenter": "custom-llm",
                },
                "replace": {"replacement_generator": "custom-llm"},
            },
        }
    )
    provider = ModelProvider(name="custom", endpoint="https://example.test/v1", api_key="CUSTOM_KEY")
    constructed = Mock()

    with patch("anonymizer.interface.factory._new_anonymizer", return_value=constructed) as constructor:
        result = create_anonymizer(
            gliner=GlinerEndpoint(
                url="https://gliner.example.test/v1",
                model="org/gliner-served",
                api_key_env="GLINER_TOKEN",
            ),
            model_configs=source,
            model_providers=[provider],
        )

    assert result is constructed
    arguments = constructor.call_args.kwargs
    parsed = parse_model_configs(arguments["model_configs"])
    injected = next(config for config in parsed.model_configs if config.alias == "anonymizer-gliner-detector")
    assert injected.model == "org/gliner-served"
    assert injected.provider == "anonymizer-gliner-endpoint"
    assert injected.skip_health_check is True
    assert next(config for config in parsed.model_configs if config.alias == "custom-llm").model == "test/llm"
    assert parsed.selected_models.detection.entity_detector == "anonymizer-gliner-detector"
    assert parsed.selected_models.detection.entity_validator == ["custom-llm"]
    assert parsed.selected_models.detection.entity_augmenter == "custom-llm"
    assert parsed.selected_models.replace.replacement_generator == "custom-llm"
    assert arguments["model_providers"][0] is provider
    endpoint_provider = arguments["model_providers"][-1]
    assert endpoint_provider.name == "anonymizer-gliner-endpoint"
    assert endpoint_provider.endpoint == "https://gliner.example.test/v1"
    assert endpoint_provider.api_key == "GLINER_TOKEN"


@pytest.mark.parametrize("collision", ["anonymizer-gliner-detector", "anonymizer-gliner-endpoint"])
def test_external_endpoint_rejects_reserved_name_collisions(collision: str) -> None:
    GlinerEndpoint = anonymizer.GlinerEndpoint
    create_anonymizer = anonymizer.create_anonymizer
    model_configs: str | None = None
    providers: list[ModelProvider] | None = None
    if collision == "anonymizer-gliner-detector":
        model_configs = json.dumps(
            {"model_configs": [{"alias": collision, "model": "test/model", "provider": "nvidia"}]}
        )
    else:
        providers = [ModelProvider(name=collision, endpoint="https://example.test/v1")]

    with pytest.raises(ValueError, match="reserved"):
        create_anonymizer(
            gliner=GlinerEndpoint("https://gliner.example.test/v1", "org/model", "GLINER_TOKEN"),
            model_configs=model_configs,
            model_providers=providers,
        )


def test_native_request_delegates_to_owned_runtime() -> None:
    NativeGliner = anonymizer.NativeGliner
    create_anonymizer = anonymizer.create_anonymizer
    expected = Mock()
    request = NativeGliner(device="cpu")
    with patch("anonymizer.notebooks._runtime._create_native_anonymizer", return_value=expected) as native:
        result = create_anonymizer(gliner=request)

    assert result is expected
    native.assert_called_once_with(
        request=request,
        model_configs=None,
        model_providers=None,
        artifact_path=None,
        data_designer_run_config=None,
    )


def test_external_endpoint_does_not_import_or_control_native_runtime() -> None:
    GlinerEndpoint = anonymizer.GlinerEndpoint
    create_anonymizer = anonymizer.create_anonymizer
    with (
        patch("anonymizer.interface.factory._new_anonymizer", return_value=Mock()),
        patch("subprocess.Popen") as popen,
    ):
        create_anonymizer(
            gliner=GlinerEndpoint("https://gliner.example.test/v1", "org/model", "GLINER_TOKEN")
        )

    popen.assert_not_called()


def test_external_factory_import_path_excludes_heavy_runtime_dependencies() -> None:
    script = """
import sys
from anonymizer import GlinerEndpoint, create_anonymizer
instance = create_anonymizer(gliner=GlinerEndpoint('https://example.test/v1', 'org/model', 'TOKEN'))
assert type(instance).__name__ == 'Anonymizer'
forbidden = ('torch', 'gliner2', 'fastapi', 'vllm', 'tools.inference_service_compiler')
loaded = sorted(name for name in sys.modules if name == forbidden or name.startswith(tuple(item + '.' for item in forbidden)))
assert not loaded, loaded
assert 'anonymizer.notebooks._runtime' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=Path(__file__).parents[2],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
